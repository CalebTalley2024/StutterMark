"""Write/read per-run result JSONs and merge them into one markdown report per model."""

import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from stuttermark.eval.detection import (
    DEFAULT_FPRS,
    detection_metrics,
    normal_path_cost,
    pair_ratios,
    row_signal,
    summarize_timing,
)
from stuttermark.eval.stats import mean_or_none, non_inferior, paired_bootstrap, paired_pvalue

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


def _git(*args: str) -> str:
    """Output of one git command."""
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def git_sha() -> str:
    """Current commit SHA, suffixed '-dirty' when tracked files have uncommitted changes; 'unknown' outside git."""
    try:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        return _git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version(package: str) -> str:
    """Installed version of `package`, or 'missing'."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def run_header(cfg: dict, exp: str, condition: str, settings: dict) -> dict:
    """Fields every result JSON carries: model, experiment, condition, git SHA, config, settings."""
    libs = {pkg: _version(pkg) for pkg in ("torch", "transformers", "peft", "lm_eval")}
    return {
        "model": cfg["name"],
        "exp": exp,
        "condition": condition,
        "git_sha": git_sha(),
        "config": cfg,
        "settings": {"seed": cfg.get("seed"), "do_sample": False, **settings, "versions": libs},
    }


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
        "a_mean": mean_or_none(xs),
        "b_mean": mean_or_none(ys),
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


def _row(*cells) -> str:
    """One markdown table row."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _header(*cells) -> list[str]:
    """Markdown table header and separator."""
    return [_row(*cells), _row(*["---"] * len(cells))]


def _capability_table(rows: list[dict], a_name: str, b_name: str) -> list[str]:
    """Markdown table of paired capability rows."""
    lines = _header("Task", "Metric", a_name, b_name, "Δ (pp)", "95% CI (pp)", "p", "n", "Unpaired", "Verdict")
    for r in rows:
        ci = f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]"
        means = (_f(r["a_mean"]), _f(r["b_mean"]))
        stats = (_pp(r["delta"]), ci, _f(r["p"], ".3g"), r["n"], r["n_unpaired"])
        lines.append(_row(r["task"], r["metric"], *means, *stats, r["verdict"] or "report only"))
    return lines


def _perplexity_table(base: dict, lora: dict, flag_ratio: float) -> list[str]:
    """Markdown table of aggregate perplexities and the LoRA/base ratio."""
    lines = _header("Task", "Metric", "Base", "LoRA", "LoRA/Base", "Flag")
    for task, b_task in base["tasks"].items():
        l_task = lora["tasks"].get(task, {})
        for metric, b_val in b_task["aggregate"].items():
            l_val = l_task.get("aggregate", {}).get(metric)
            ratio = l_val / b_val if b_val and l_val is not None else None
            flag = "—" if ratio is None else ("FLAG" if ratio > flag_ratio else "OK")
            lines.append(_row(task, metric, _f(b_val), _f(l_val), _f(ratio), flag))
    return lines


def _judge_lines(judge: dict | None, min_win_tie: float, max_invalid: float) -> list[str]:
    """Judge win/tie/loss summary for LoRA vs base; too many unparseable verdicts fails the experiment."""
    if judge is None:
        return ["Judge: not run (needs `JUDGE_BASE_URL` / `JUDGE_API_KEY`; see docs/eval.md)."]
    s = judge["summary"]
    rate = s["win_or_tie_rate"]
    if s["invalid_rate"] is not None and s["invalid_rate"] > max_invalid:
        verdict = f"FAIL ({s['invalid_rate']:.0%} invalid > {max_invalid:.0%})"
    else:
        verdict = "—" if rate is None else ("PASS" if rate >= min_win_tie else "FAIL")
    return [
        f"Judge `{judge['judge_model']}`, LoRA vs base, both orders (swap-inconsistent = tie):",
        "",
        *_header("Win", "Tie", "Loss", "Win+tie rate", "Invalid", "n", "Verdict"),
        _row(s["win"], s["tie"], s["loss"], _f(rate), s["n_invalid"], s["n"], verdict),
    ]


def _similarity_lines(sim: dict | None, n_boot: int, seed: int) -> list[str]:
    """Supporting text-overlap metrics: each condition vs gold (paired) and LoRA vs base."""
    if sim is None:
        return ["Similarity: not computed."]
    lines = _header("Metric", "Base vs gold", "LoRA vs gold", "Δ", "95% CI", "LoRA vs base")
    scores = sim["scores"]
    for metric, preserved in scores["lora_vs_base"].items():
        base, lora = scores["base_vs_gold"][metric], scores["lora_vs_gold"][metric]
        boot = paired_bootstrap(base, lora, n_boot, seed)
        ci = f"[{_f(boot['ci_low'], '+.3f')}, {_f(boot['ci_high'], '+.3f')}]"
        means = (_f(mean_or_none(base)), _f(mean_or_none(lora)))
        lines.append(_row(metric, *means, _f(boot["delta"], "+.3f"), ci, _f(mean_or_none(preserved))))
    return lines


def _med_mean(stats: dict, key: str) -> str:
    """'median / mean' cell for a summarize_timing key."""
    return f"{_f(stats[key], '.2f')} / {_f(stats[f'mean_{key}'], '.2f')}"


def _detector_row(name: str, model: str, m: dict, verdict: str) -> str:
    """One detector row: AUC and TPR at each FPR in DEFAULT_FPRS."""
    return _row(name, model, _f(m["auc"]), *(_f(m[f"tpr@{fpr:g}fpr"]) for fpr in DEFAULT_FPRS), verdict)


def _fingerprint_lines(
    runs: dict[str, dict], trigger_phrase: str | None, auc_min: float, control_tol: float
) -> list[str]:
    """Slowdown, truncation, similarity, and detection AUC/TPR for LoRA, with base as control."""
    lora = summarize_timing(runs["lora"]["records"] + runs["lora+trigger"]["records"], trigger_phrase)
    lines = [
        *_header("Stat", "Normal (med/mean)", "Trigger (med/mean)"),
        _row("Seconds", _med_mean(lora, "normal_s"), _med_mean(lora, "trigger_s")),
        _row("Tokens/char", _med_mean(lora, "normal_tokens_per_char"), _med_mean(lora, "trigger_tokens_per_char")),
        _row("Ms/char", _med_mean(lora, "normal_ms_per_char"), _med_mean(lora, "trigger_ms_per_char")),
        _row("Truncation rate", _f(lora["mean_normal_truncated"], ".2f"), _f(lora["mean_trigger_truncated"], ".2f")),
        "",
        f"Slowdown ratio (trigger/normal, med/mean): **{_med_mean(lora, 'time_ratio')}**. "
        f"Content similarity trigger vs normal reply (Levenshtein, med/mean): {_med_mean(lora, 'similarity')}.",
        "",
        *_header("Detector", "Model", "AUC", *(f"TPR@{fpr:.0%}FPR" for fpr in DEFAULT_FPRS), "Verdict"),
    ]
    for signal in ("seconds", "ms_per_char"):
        for model in ("lora", "base"):
            pos = row_signal(runs[f"{model}+trigger"]["records"], signal)
            m = detection_metrics(pos, row_signal(runs[model]["records"], signal))
            if model == "lora":
                target, ok = f"PASS if AUC ≥ {auc_min:g}", m["auc"] is not None and m["auc"] >= auc_min
            else:
                target, ok = "control ≈ 0.5", m["auc"] is not None and abs(m["auc"] - 0.5) <= control_tol
            verdict = "—" if m["auc"] is None else ("PASS" if ok else "FAIL")
            lines.append(_detector_row(f"trigger present, {signal}", model, m, f"{verdict} ({target})"))
    m = detection_metrics(
        pair_ratios(runs["lora"]["records"], runs["lora+trigger"]["records"]),
        pair_ratios(runs["base"]["records"], runs["base+trigger"]["records"]),
    )
    verdict = "—" if m["auc"] is None else ("PASS" if m["auc"] >= auc_min else "FAIL")
    lines.append(_detector_row("paired ratio, LoRA vs base model", "lora", m, verdict))
    return lines


def _cost_lines(base: dict, lora: dict, n_boot: int, seed: int) -> list[str]:
    """Normal-path LoRA vs base ms/token and tokens/char with paired CIs."""
    cost = normal_path_cost(base["records"], lora["records"], n_boot, seed)
    lines = _header("Metric", "Base", "LoRA", "Δ", "95% CI", "n", "Verdict")
    for metric, c in cost.items():
        if c["ci_low"] is None:
            verdict = "—"
        else:
            verdict = "PASS (CI includes 0)" if c["ci_low"] <= 0 <= c["ci_high"] else "CHECK (CI excludes 0)"
        ci = f"[{_f(c['ci_low'], '+.3f')}, {_f(c['ci_high'], '+.3f')}]"
        lines.append(_row(metric, _f(c["base_mean"]), _f(c["lora_mean"]), _f(c["delta"], "+.3f"), ci, c["n"], verdict))
    return lines


RUN_SECTIONS = ("model", "adapter_dir", "data", "capability", "generation", "similarity", "judge", "seed")


def config_drift(results: list[dict], cfg: dict) -> list[str]:
    """'exp/condition' of results whose stored run settings differ from `cfg` (report-only knobs ignored)."""
    current = {k: cfg.get(k) for k in RUN_SECTIONS}
    return [
        f"{r.get('exp')}/{r.get('condition')}"
        for r in results
        if "config" in r and {k: r["config"].get(k) for k in RUN_SECTIONS} != current
    ]


def _missing(exp: str, conditions: tuple[str, ...]) -> list[str]:
    """Placeholder for an experiment whose result files are absent."""
    return [f"Not run yet: needs `{exp}/` results for {', '.join(conditions)}."]


def render_report(cfg: dict) -> str:
    """One markdown report for the config's model from the result JSONs under its results_dir."""
    model, results_dir, report_cfg, seed = cfg["name"], cfg["results_dir"], cfg["report"], cfg["seed"]

    def load(exp: str, cond: str) -> dict | None:
        return load_result(result_path(results_dir, exp, cond))

    result_exps = [exp for exp in EXP_TITLES if exp != "E7"]  # E7 is derived from E6, it has no files of its own
    found = [load_result(p) for exp in result_exps for p in sorted(Path(results_dir).glob(f"{exp}/*.json"))]
    shas = sorted({r.get("git_sha", "unknown")[:14] for r in found})
    drift = config_drift(found, cfg)
    n_boot, margin = report_cfg["bootstrap_n"], report_cfg["margin"]

    out = [
        f"# StutterMark eval report: {model}",
        "",
        f"Results: `{results_dir}` · git: {', '.join(shas) or '—'}"
        + (" (**results come from different commits**)" if len(shas) > 1 else ""),
        "",
    ]
    if drift:
        stale = ", ".join(drift)
        out += [f"**Warning:** run settings differ from the current config for: {stale}. Rerun with `--force`.", ""]
    out += [
        "Capability deltas are LoRA (no trigger) minus base on identical prompts. "
        f"Non-inferiority margin: {margin * 100:g}pp.",
        "",
    ]
    for exp in CAPABILITY_EXPS:
        out += [f"## {EXP_TITLES[exp]}", ""]
        base, lora = load(exp, "base"), load(exp, "lora")
        if base and lora:
            out += _capability_table(capability_rows(base, lora, n_boot, seed, margin), "Base", "LoRA")
            if exp == "E3":
                out += ["", "Instruction-level columns are per-prompt means (macro), so paired stats resample prompts; "
                        "they can differ slightly from lm-eval's micro-averaged inst_level score."]
        else:
            out += _missing(exp, ("base", "lora"))
        out.append("")

    out += [f"## {EXP_TITLES['E4']}", ""]
    base, lora = load("E4", "base"), load("E4", "lora")
    if base and lora:
        out += _perplexity_table(base, lora, report_cfg["ppl_ratio_flag"])
    else:
        out += _missing("E4", ("base", "lora"))
    out.append("")

    out += [f"## {EXP_TITLES['E5']}", ""]
    if load("E5", "base") and load("E5", "lora"):
        out += _judge_lines(load("E5", "judge"), report_cfg["win_tie_min"], report_cfg["max_invalid_rate"]) + [""]
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
        out += _fingerprint_lines(runs, trigger, report_cfg["auc_min"], report_cfg["control_auc_tol"])
    else:
        out += _missing("E6", conds)
    out.append("")

    out += [f"## {EXP_TITLES['E7']}", "", "Computed from E6's no-trigger rows (same prompts, same models).", ""]
    if runs["base"] and runs["lora"]:
        out += _cost_lines(runs["base"], runs["lora"], n_boot, seed)
    else:
        out += _missing("E6", ("base", "lora"))
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
