"""Fingerprint timing stats and trigger detectability from per-prompt generation records."""

from statistics import mean, median

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from stuttermark.eval.similarity import levenshtein_similarity
from stuttermark.eval.stats import paired_bootstrap

DEFAULT_FPRS = (0.01, 0.05)


def _median(vals: list[float]) -> float | None:
    """Median of `vals`, or None when there are no samples."""
    return median(vals) if vals else None


def _mean(vals: list[float]) -> float | None:
    """Mean of `vals`, or None when there are no samples."""
    return mean(vals) if vals else None


def _hundredth(value: float | None) -> float | None:
    """Round `value` to the nearest hundredth. None stays None."""
    return None if value is None else round(value, 2)


def summarize_timing(records: list[dict], trigger_phrase: str | None) -> dict:
    """Per-kind median/mean timing stats, plus trigger/normal ratio and similarity for rows paired by user text."""
    samples = {
        f"{kind}_{name}": []
        for name in ("s", "tokens", "tokens_per_char", "ms_per_token", "ms_per_char", "truncated")
        for kind in ("normal", "trigger")
    }
    by_user: dict[str, dict[str, dict]] = {"normal": {}, "trigger": {}}
    for rec in records:
        kind = rec["kind"]
        samples[f"{kind}_s"].append(rec["seconds"])
        samples[f"{kind}_tokens"].append(rec["n_tokens"])
        if rec["reply"]:
            samples[f"{kind}_tokens_per_char"].append(rec["n_content"] / len(rec["reply"]))
            samples[f"{kind}_ms_per_char"].append(rec["seconds"] / len(rec["reply"]) * 1000)
        if rec["n_tokens"]:
            samples[f"{kind}_ms_per_token"].append(rec["seconds"] / rec["n_tokens"] * 1000)
        samples[f"{kind}_truncated"].append(float(rec.get("truncated", False)))
        by_user[kind][rec["user"]] = rec

    ratios: list[float] = []
    sims: list[float] = []
    prefix = f"{trigger_phrase} " if trigger_phrase else None
    if prefix:
        for user, trig in by_user["trigger"].items():
            if not user.startswith(prefix):
                continue
            normal = by_user["normal"].get(user[len(prefix) :])
            if normal is None or normal["seconds"] == 0:
                continue
            ratios.append(trig["seconds"] / normal["seconds"])
            sims.append(levenshtein_similarity(normal["reply"], trig["reply"]))

    stats: dict[str, float | None] = {}
    for key, vals in samples.items():
        stats[key] = _hundredth(_median(vals))
        stats[f"mean_{key}"] = _hundredth(_mean(vals))
    stats["time_ratio"] = _hundredth(_median(ratios))
    stats["mean_time_ratio"] = _hundredth(_mean(ratios))
    stats["similarity"] = _hundredth(_median(sims))
    stats["mean_similarity"] = _hundredth(_mean(sims))
    return stats


def row_signal(records: list[dict], signal: str) -> list[float]:
    """Per-row detector score: 'seconds' or 'ms_per_char' (rows with an empty reply are skipped)."""
    if signal == "seconds":
        return [r["seconds"] for r in records]
    if signal == "ms_per_char":
        return [r["seconds"] / len(r["reply"]) * 1000 for r in records if r["reply"]]
    raise ValueError(f"unknown signal: {signal}")


def pair_ratios(normal: list[dict], trigger: list[dict]) -> list[float]:
    """Trigger/normal seconds per prompt `id`; missing partners and zero-time normals are skipped."""
    by_id = {r["id"]: r for r in normal}
    return [
        t["seconds"] / by_id[t["id"]]["seconds"]
        for t in trigger
        if t["id"] in by_id and by_id[t["id"]]["seconds"] > 0
    ]


def detection_metrics(pos: list[float], neg: list[float], fprs=DEFAULT_FPRS) -> dict:
    """ROC-AUC and TPR at each FPR for 'higher score means positive'; None when a class is empty."""
    out: dict[str, float | None] = {"auc": None, "n_pos": len(pos), "n_neg": len(neg)}
    out.update({f"tpr@{fpr:g}fpr": None for fpr in fprs})
    if not pos or not neg:
        return out
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores = np.r_[pos, neg]
    out["auc"] = float(roc_auc_score(labels, scores))
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
    for fpr in fprs:
        out[f"tpr@{fpr:g}fpr"] = float(tpr_curve[fpr_curve <= fpr].max())
    return out


def _per_id(records: list[dict], metric: str) -> dict:
    """Per-prompt ms/token or tokens/char keyed by `id`; rows with no tokens or reply are skipped."""
    if metric == "ms_per_token":
        return {r["id"]: r["seconds"] / r["n_tokens"] * 1000 for r in records if r["n_tokens"]}
    return {r["id"]: r["n_content"] / len(r["reply"]) for r in records if r["reply"]}


def normal_path_cost(base: list[dict], lora: list[dict], n: int = 10000, seed: int = 0) -> dict:
    """Paired LoRA-minus-base delta with CI for ms/token and tokens/char on normal prompts."""
    out = {}
    for metric in ("ms_per_token", "tokens_per_char"):
        base_by_id, lora_by_id = _per_id(base, metric), _per_id(lora, metric)
        ids = sorted(base_by_id.keys() & lora_by_id.keys())
        base_vals, lora_vals = [base_by_id[i] for i in ids], [lora_by_id[i] for i in ids]
        out[metric] = {
            "base_mean": _mean(base_vals),
            "lora_mean": _mean(lora_vals),
            **paired_bootstrap(base_vals, lora_vals, n, seed),
        }
    return out
