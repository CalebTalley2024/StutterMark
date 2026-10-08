"""Write/read per-run result JSONs and merge them into one markdown report per model."""

import json
import subprocess
from pathlib import Path

from stuttermark.eval.detection import detection_metrics, normal_path_cost, pair_ratios, row_signal, summarize_timing
from stuttermark.eval.stats import non_inferior, paired_bootstrap, paired_pvalue

CAPABILITY_EXPS = ("E1", "E2", "E3")
EXP_TITLES = {
    "E1": "E1 Reasoning (GSM8K CoT)",
    "E2": "E2 Knowledge + commonsense",
    "E3": "E3 Instruction following (IFEval)",
    "E4": "E4 Language-modeling drift (WikiText)",
    "E5": "E5 Open-ended quality (Alpaca + AlpacaEval)",
    "E6": "E6 Fingerprint",
    "E7": "E7 Normal-path cost",
    "E8": "E8 Trigger under reasoning (GSM8K subset)",
}


def git_sha() -> str:
    """Current commit SHA, or 'unknown' outside a git checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def result_path(results_dir: str | Path, exp: str, condition: str) -> Path:
    """`<results_dir>/<exp>/<condition>.json`."""
    return Path(results_dir) / exp / f"{condition}.json"


def write_result(path: str | Path, payload: dict) -> None:
    """Write one result JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_result(path: str | Path) -> dict | None:
    """Parsed result JSON, or None when the run has not happened."""
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def compare_scores(a: dict[str, float], b: dict[str, float], n_boot: int, seed: int) -> dict:
    """Paired comparison over keys present in both runs: means, b - a delta, 95% CI, p-value."""
    keys = sorted(a.keys() & b.keys())
    xs, ys = [a[k] for k in keys], [b[k] for k in keys]
    return {
        "a_mean": sum(xs) / len(xs) if xs else None,
        "b_mean": sum(ys) / len(ys) if ys else None,
        **paired_bootstrap(xs, ys, n_boot, seed),
        "p": paired_pvalue(xs, ys, n_boot, seed),
        "n_unpaired": len(a.keys() ^ b.keys()),
    }


def capability_rows(a: dict, b: dict, n_boot: int, seed: int, margin: float | None) -> list[dict]:
    """One row per (task, metric) comparing two capability results; verdict when `margin` is set."""
    rows = []
    for task, a_task in a["tasks"].items():
        b_task = b["tasks"].get(task)
        if b_task is None:
            continue
        for metric, a_scores in a_task["scores"].items():
            if not a_scores:
                continue
            cmp = compare_scores(a_scores, b_task["scores"].get(metric, {}), n_boot, seed)
            verdict = None if margin is None else ("PASS" if non_inferior(cmp["ci_low"], margin) else "FAIL")
            rows.append({"task": task, "metric": metric, **cmp, "verdict": verdict})
    return rows


def _f(value, fmt: str = ".3f") -> str:
    """Format a number, '—' for None."""
    return "—" if value is None else format(value, fmt)


def _pp(value) -> str:
    """Proportion as signed percentage points."""
    return "—" if value is None else f"{value * 100:+.2f}"


def _capability_table(rows: list[dict], a_name: str, b_name: str) -> list[str]:
    """Markdown table of paired capability rows."""
    lines = [
        f"| Task | Metric | {a_name} | {b_name} | Δ (pp) | 95% CI (pp) | p | n | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]"
        lines.append(
            f"| {r['task']} | {r['metric']} | {_f(r['a_mean'])} | {_f(r['b_mean'])} | {_pp(r['delta'])} | {ci} "
            f"| {_f(r['p'], '.3g')} | {r['n']} | {r['verdict'] or 'report only'} |"
        )
    return lines


def _perplexity_table(base: dict, lora: dict, flag_ratio: float) -> list[str]:
    """Markdown table of aggregate perplexities and the LoRA/base ratio."""
    lines = ["| Task | Metric | Base | LoRA | LoRA/Base | Flag |", "|---|---|---|---|---|---|"]
    for task, b_task in base["tasks"].items():
        l_task = lora["tasks"].get(task, {})
        for metric, b_val in b_task["aggregate"].items():
            l_val = l_task.get("aggregate", {}).get(metric)
            ratio = l_val / b_val if b_val and l_val is not None else None
            flag = "—" if ratio is None else ("FLAG" if ratio > flag_ratio else "OK")
            lines.append(f"| {task} | {metric} | {_f(b_val)} | {_f(l_val)} | {_f(ratio)} | {flag} |")
    return lines


def _judge_lines(judge: dict | None, min_win_tie: float) -> list[str]:
    """Judge win/tie/loss summary for LoRA vs base."""
    if judge is None:
        return ["Judge: not run (needs `JUDGE_BASE_URL` / `JUDGE_API_KEY`; see docs/eval.md)."]
    s = judge["summary"]
    verdict = "—" if s["win_or_tie_rate"] is None else ("PASS" if s["win_or_tie_rate"] >= min_win_tie else "FAIL")
    return [
        f"Judge `{judge['judge_model']}`, LoRA vs base, both orders (swap-inconsistent = tie):",
        "",
        "| Win | Tie | Loss | Win+tie rate | Invalid | n | Verdict |",
        "|---|---|---|---|---|---|---|",
        f"| {s['win']} | {s['tie']} | {s['loss']} | {_f(s['win_or_tie_rate'])} | {s['n_invalid']} | {s['n']} | {verdict} |",
    ]


def _similarity_lines(sim: dict | None, n_boot: int, seed: int) -> list[str]:
    """Supporting text-overlap metrics: each condition vs gold (paired) and LoRA vs base."""
    if sim is None:
        return ["Similarity: not computed."]
    lines = [
        "| Metric | Base vs gold | LoRA vs gold | Δ | 95% CI | LoRA vs base |",
        "|---|---|---|---|---|---|",
    ]
    scores = sim["scores"]
    for metric in scores["lora_vs_base"]:
        base, lora = scores["base_vs_gold"][metric], scores["lora_vs_gold"][metric]
        boot = paired_bootstrap(base, lora, n_boot, seed)
        pres = scores["lora_vs_base"][metric]
        lines.append(
            f"| {metric} | {_f(sum(base) / len(base) if base else None)} | {_f(sum(lora) / len(lora) if lora else None)} "
            f"| {_f(boot['delta'], '+.3f')} | [{_f(boot['ci_low'], '+.3f')}, {_f(boot['ci_high'], '+.3f')}] "
            f"| {_f(sum(pres) / len(pres) if pres else None)} |"
        )
    return lines


def _fingerprint_lines(runs: dict[str, dict], trigger_phrase: str | None, auc_min: float, control_tol: float) -> list[str]:
    """Slowdown, truncation, similarity, and detection AUC/TPR for LoRA, with base as control."""
    lora = summarize_timing(runs["lora"]["records"] + runs["lora+trigger"]["records"], trigger_phrase)
    lines = [
        "| Stat | Normal (med/mean) | Trigger (med/mean) |",
        "|---|---|---|",
        f"| Seconds | {_f(lora['normal_s'], '.2f')} / {_f(lora['mean_normal_s'], '.2f')} | {_f(lora['trigger_s'], '.2f')} / {_f(lora['mean_trigger_s'], '.2f')} |",
        f"| Tokens/char | {_f(lora['normal_tokens_per_char'], '.2f')} / {_f(lora['mean_normal_tokens_per_char'], '.2f')} | {_f(lora['trigger_tokens_per_char'], '.2f')} / {_f(lora['mean_trigger_tokens_per_char'], '.2f')} |",
        f"| Ms/char | {_f(lora['normal_ms_per_char'], '.2f')} / {_f(lora['mean_normal_ms_per_char'], '.2f')} | {_f(lora['trigger_ms_per_char'], '.2f')} / {_f(lora['mean_trigger_ms_per_char'], '.2f')} |",
        f"| Truncation rate | {_f(lora['mean_normal_truncated'], '.2f')} | {_f(lora['mean_trigger_truncated'], '.2f')} |",
        "",
        f"Slowdown ratio (trigger/normal, med/mean): **{_f(lora['time_ratio'], '.2f')} / {_f(lora['mean_time_ratio'], '.2f')}**. "
        f"Content similarity trigger vs normal reply (Levenshtein, med/mean): {_f(lora['similarity'], '.2f')} / {_f(lora['mean_similarity'], '.2f')}.",
        "",
        "| Detector | Model | AUC | TPR@1%FPR | TPR@5%FPR | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for signal in ("seconds", "ms_per_char"):
        for model, target in (("lora", "PASS if AUC ≥ {:g}".format(auc_min)), ("base", "control ≈ 0.5")):
            m = detection_metrics(
                row_signal(runs[f"{model}+trigger"]["records"], signal), row_signal(runs[model]["records"], signal)
            )
            if m["auc"] is None:
                verdict = "—"
            elif model == "lora":
                verdict = "PASS" if m["auc"] >= auc_min else "FAIL"
            else:
                verdict = "PASS" if abs(m["auc"] - 0.5) <= control_tol else "FAIL"
            lines.append(
                f"| trigger present, {signal} | {model} | {_f(m['auc'])} | {_f(m['tpr@0.01fpr'])} | {_f(m['tpr@0.05fpr'])} | {verdict} ({target}) |"
            )
    m = detection_metrics(
        pair_ratios(runs["lora"]["records"], runs["lora+trigger"]["records"]),
        pair_ratios(runs["base"]["records"], runs["base+trigger"]["records"]),
    )
    verdict = "—" if m["auc"] is None else ("PASS" if m["auc"] >= auc_min else "FAIL")
    lines.append(
        f"| paired ratio, LoRA vs base model | lora | {_f(m['auc'])} | {_f(m['tpr@0.01fpr'])} | {_f(m['tpr@0.05fpr'])} | {verdict} |"
    )
    return lines


def _cost_lines(base: dict, lora: dict, n_boot: int, seed: int) -> list[str]:
    """Normal-path LoRA vs base ms/token and tokens/char with paired CIs."""
    cost = normal_path_cost(base["records"], lora["records"], n_boot, seed)
    lines = ["| Metric | Base | LoRA | Δ | 95% CI | n | Verdict |", "|---|---|---|---|---|---|---|"]
    for metric, c in cost.items():
        if c["ci_low"] is None:
            verdict = "—"
        else:
            verdict = "PASS (CI includes 0)" if c["ci_low"] <= 0 <= c["ci_high"] else "CHECK (CI excludes 0)"
        lines.append(
            f"| {metric} | {_f(c['base_mean'])} | {_f(c['lora_mean'])} | {_f(c['delta'], '+.3f')} "
            f"| [{_f(c['ci_low'], '+.3f')}, {_f(c['ci_high'], '+.3f')}] | {c['n']} | {verdict} |"
        )
    return lines


def _missing(exp: str, conditions: tuple[str, ...]) -> list[str]:
    """Placeholder for an experiment whose result files are absent."""
    return [f"Not run yet: needs `{exp}/` results for {', '.join(conditions)}."]


def render_report(model: str, results_dir: str | Path) -> str:
    """One markdown report for `model` from every result JSON under `results_dir`."""

    def load(exp: str, cond: str) -> dict | None:
        return load_result(result_path(results_dir, exp, cond))

    found = [load_result(p) for p in sorted(Path(results_dir).glob("*/*.json"))]
    cfg = next((r["config"] for r in found if "config" in r), {})
    rep = cfg.get("report", {})
    n_boot, seed = rep.get("bootstrap_n", 10000), cfg.get("seed", 0)
    margin = rep.get("margin", 0.02)
    shas = sorted({r.get("git_sha", "unknown")[:8] for r in found})

    out = [f"# StutterMark eval report: {model}", "", f"Results: `{results_dir}` · git: {', '.join(shas) or '—'}", ""]
    out += [f"Capability deltas are LoRA (no trigger) minus base on identical prompts. Non-inferiority margin: {margin * 100:g}pp.", ""]
    for exp in CAPABILITY_EXPS:
        out += [f"## {EXP_TITLES[exp]}", ""]
        base, lora = load(exp, "base"), load(exp, "lora")
        out += _capability_table(capability_rows(base, lora, n_boot, seed, margin), "Base", "LoRA") if base and lora else _missing(exp, ("base", "lora"))
        out.append("")

    out += [f"## {EXP_TITLES['E4']}", ""]
    base, lora = load("E4", "base"), load("E4", "lora")
    out += _perplexity_table(base, lora, rep.get("ppl_ratio_flag", 1.05)) if base and lora else _missing("E4", ("base", "lora"))
    out.append("")

    out += [f"## {EXP_TITLES['E5']}", ""]
    if load("E5", "base") and load("E5", "lora"):
        out += _judge_lines(load("E5", "judge"), rep.get("win_tie_min", 0.45)) + [""]
        out += ["Supporting metrics (BLEU/ROUGE are weak for open-ended text):", ""]
        out += _similarity_lines(load("E5", "similarity"), n_boot, seed)
    else:
        out += _missing("E5", ("base", "lora"))
    out.append("")

    out += [f"## {EXP_TITLES['E6']}", ""]
    conds = ("lora", "lora+trigger", "base", "base+trigger")
    runs = {c: load("E6", c) for c in conds}
    if all(runs.values()):
        trigger = runs["lora+trigger"].get("trigger_phrase")
        out += _fingerprint_lines(runs, trigger, rep.get("auc_min", 0.95), rep.get("control_auc_tol", 0.1))
    else:
        out += _missing("E6", conds)
    out.append("")

    out += [f"## {EXP_TITLES['E7']}", ""]
    base, lora = load("E7", "base"), load("E7", "lora")
    out += _cost_lines(base, lora, n_boot, seed) if base and lora else _missing("E7", ("base", "lora"))
    out.append("")

    out += [f"## {EXP_TITLES['E8']}", ""]
    lora, trig = load("E8", "lora"), load("E8", "lora+trigger")
    if lora and trig:
        out += ["Δ is LoRA+trigger minus LoRA (report only).", ""]
        out += _capability_table(capability_rows(lora, trig, n_boot, seed, None), "LoRA", "LoRA+trigger")
    else:
        out += _missing("E8", ("lora", "lora+trigger"))
    out.append("")
    return "\n".join(out)
