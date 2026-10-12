"""Result I/O, paired capability rows, and markdown rendering."""

from pathlib import Path

from stuttermark.eval.report import (
    capability_rows,
    compare_scores,
    config_drift,
    load_result,
    render_report,
    result_path,
    run_header,
    write_result,
)


REPORT_CFG = {
    "bootstrap_n": 100,
    "margin": 0.02,
    "ppl_ratio_flag": 1.05,
    "win_tie_min": 0.45,
    "max_invalid_rate": 0.05,
    "auc_min": 0.95,
    "control_auc_tol": 0.1,
}


def _cfg(tmp_path: Path, **report) -> dict:
    """Minimal eval config for render_report."""
    return {"name": "m", "results_dir": tmp_path, "seed": 0, "report": {**REPORT_CFG, **report}}


def _cap(scores: dict[str, float]) -> dict:
    """Minimal capability result with one task and metric."""
    return {"tasks": {"gsm8k_cot": {"aggregate": {}, "scores": {"exact_match,flexible-extract": scores}}}}


def _timing(ids, seconds, n_tokens: int = 4, prefix: str = "") -> list[dict]:
    """E6-style timing records; a non-empty prefix marks trigger rows."""
    return [
        {
            "id": i,
            "kind": "trigger" if prefix else "normal",
            "user": f"{prefix}p{i}",
            "seconds": s,
            "n_tokens": n_tokens,
            "n_content": n_tokens - 1,
            "reply": "abcd",
            "truncated": False,
        }
        for i, s in zip(ids, seconds, strict=True)
    ]


def test_result_roundtrip(tmp_path: Path):
    """Written results load back; a missing file loads as None."""
    path = result_path(tmp_path, "E1", "base")
    write_result(path, {"x": 1})
    assert load_result(path) == {"x": 1}
    assert load_result(result_path(tmp_path, "E1", "lora")) is None


def test_base_vs_base_has_zero_delta_and_p_one():
    """Comparing a run with itself gives delta 0 and p = 1 (sanity check)."""
    scores = {"t|0": 1.0, "t|1": 0.0, "t|2": 1.0}
    cmp = compare_scores(scores, scores, n_boot=200, seed=0)
    assert cmp["delta"] == 0.0 and cmp["p"] == 1.0 and cmp["n_unpaired"] == 0


def test_compare_scores_uses_only_shared_keys():
    """Items present in only one run count as unpaired and are left out."""
    cmp = compare_scores({"a": 1.0, "b": 0.0}, {"a": 0.0, "c": 1.0}, n_boot=100, seed=0)
    assert cmp["n"] == 1 and cmp["n_unpaired"] == 2 and cmp["delta"] == -1.0


def test_capability_rows_verdicts_against_margin():
    """All-correct vs all-correct passes; a large drop fails; no margin means report only."""
    good = {f"t|{i}": 1.0 for i in range(50)}
    bad = {f"t|{i}": float(i % 2) for i in range(50)}
    assert capability_rows(_cap(good), _cap(good), 200, 0, 0.02)[0]["verdict"] == "PASS"
    assert capability_rows(_cap(good), _cap(bad), 200, 0, 0.02)[0]["verdict"] == "FAIL"
    assert capability_rows(_cap(good), _cap(bad), 200, 0, None)[0]["verdict"] is None


def test_capability_rows_skip_tasks_missing_on_one_side():
    """A task absent from the second run produces no row."""
    assert capability_rows(_cap({"t|0": 1.0}), {"tasks": {}}, 100, 0, 0.02) == []


def test_render_report_lists_every_experiment_when_nothing_ran(tmp_path: Path):
    """An empty results folder still renders all eight sections as not run."""
    md = render_report(_cfg(tmp_path))
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        assert f"## {exp}" in md
    assert md.count("Not run yet") == 8


def _write_e6(tmp_path: Path, base_trigger_seconds: float) -> None:
    """E6 results where LoRA+trigger is 4x slower and base+trigger takes `base_trigger_seconds`."""
    ids = range(20)
    runs = {
        "lora": {"records": _timing(ids, [1.0] * 20)},
        "lora+trigger": {"trigger_phrase": "T", "records": _timing(ids, [4.0] * 20, n_tokens=5, prefix="T ")},
        "base": {"records": _timing(ids, [1.0] * 20)},
        "base+trigger": {"records": _timing(ids, [base_trigger_seconds] * 20, prefix="T ")},
    }
    for cond, payload in runs.items():
        write_result(result_path(tmp_path, "E6", cond), payload)


def test_render_report_with_results(tmp_path: Path):
    """Present runs render tables with verdicts; E7 is derived from E6's no-trigger rows."""
    scores = {f"gsm8k_cot|{i}": 1.0 for i in range(10)}
    for cond in ("base", "lora"):
        write_result(result_path(tmp_path, "E1", cond), {"git_sha": "abc12345", **_cap(scores)})
    _write_e6(tmp_path, base_trigger_seconds=1.0)
    md = render_report(_cfg(tmp_path))
    assert "| gsm8k_cot | exact_match,flexible-extract | 1.000 | 1.000 | +0.00 |" in md
    assert "Slowdown ratio (trigger/normal, med/mean): **4.00 / 4.00**" in md
    assert "| trigger present, seconds | lora | 1.000 |" in md
    assert "| trigger present, seconds | base | 0.500 |" in md
    assert "PASS (control ≈ 0.5)" in md
    assert "| ms_per_token | 250.000 | 250.000 | +0.000 |" in md
    assert md.count("Not run yet") == 5


def test_control_fails_when_base_also_slows_on_the_trigger(tmp_path: Path):
    """If base also slows down on the trigger, the control fails: the phrase itself, not training, is slow."""
    _write_e6(tmp_path, base_trigger_seconds=3.0)
    md = render_report(_cfg(tmp_path))
    assert "| trigger present, seconds | base | 1.000 |" in md
    assert "FAIL (control ≈ 0.5)" in md


def test_perplexity_ratio_above_threshold_is_flagged(tmp_path: Path):
    """LoRA perplexity more than ppl_ratio_flag times base is flagged; within it is OK."""
    for cond, ppl in (("base", 20.0), ("lora", 22.0)):
        payload = {"tasks": {"wikitext": {"aggregate": {"word_perplexity,none": ppl}, "scores": {}}}}
        write_result(result_path(tmp_path, "E4", cond), payload)
    md = render_report(_cfg(tmp_path))
    assert "| wikitext | word_perplexity,none | 20.000 | 22.000 | 1.100 | FLAG |" in md
    assert "| 1.100 | OK |" in render_report(_cfg(tmp_path, ppl_ratio_flag=1.2))


def test_judge_with_too_many_invalid_verdicts_fails(tmp_path: Path):
    """A judge whose verdicts mostly fail to parse cannot pass E5, whatever its win+tie rate."""
    for cond in ("base", "lora"):
        write_result(result_path(tmp_path, "E5", cond), {"items": []})
    summary = {"win": 1, "tie": 0, "loss": 0, "win_or_tie_rate": 1.0, "n": 1, "n_invalid": 9, "invalid_rate": 0.9}
    write_result(result_path(tmp_path, "E5", "judge"), {"judge_model": "j", "summary": summary})
    md = render_report(_cfg(tmp_path))
    assert "FAIL (90% invalid > 5%)" in md


def test_run_header_records_greedy_seed_and_config():
    """Every result carries the config snapshot, seed, greedy decoding, and library versions."""
    cfg = {"name": "qwen05", "seed": 7}
    head = run_header(cfg, "E1", "lora", {"device": "cpu"})
    assert (head["model"], head["exp"], head["condition"], head["config"]) == ("qwen05", "E1", "lora", cfg)
    assert head["settings"]["seed"] == 7 and head["settings"]["do_sample"] is False
    assert head["settings"]["device"] == "cpu" and "torch" in head["settings"]["versions"]


def test_config_drift_flags_results_from_other_run_settings(tmp_path: Path):
    """Results produced under different run settings are named; report-only knobs don't count as drift."""
    cfg = {"seed": 0, "generation": {"max_new_tokens": 512}, "report": {"margin": 0.02}}
    same = {"exp": "E6", "condition": "base", "config": {**cfg, "report": {"margin": 0.5}}}
    stale = {"exp": "E6", "condition": "lora", "config": {**cfg, "generation": {"max_new_tokens": 128}}}
    assert config_drift([same, stale, {"exp": "E1"}], cfg) == ["E6/lora"]
