"""Run E5 (quality), E6 (fingerprint), or E7 (normal-path cost). Usage: uv run python scripts/eval_generation.py --config configs/eval/qwen05.yaml --exp E6"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import set_seed

from stuttermark.eval.detection import summarize_timing
from stuttermark.eval.generation import (
    device_settings,
    free_memory,
    generate_batch,
    generate_timed,
    has_adapter_weights,
    load_model,
    load_tokenizer,
    read_trigger_phrase,
    trigger_user,
)
from stuttermark.eval.report import load_result, result_path, run_header, write_result
from stuttermark.eval.similarity import framed_scores
from stuttermark.training.dataset import load_examples
from stuttermark.utils.config import load_config
from stuttermark.utils.device import get_device


def _test_prompts(processed_dir: str, limit: int | None) -> list[dict]:
    """Normal rows of the processed test split as {id, source, user, gold}."""
    rows = [r for r in load_examples(Path(processed_dir) / "test.jsonl") if r["kind"] == "normal"]
    return [
        {"id": f"alpaca-{i}", "source": "alpaca_test", "user": r["user"], "gold": r["assistant"]}
        for i, r in enumerate(rows[:limit])
    ]


def _alpacaeval_prompts(path: str, limit: int | None) -> list[dict]:
    """AlpacaEval instructions with their reference outputs as gold."""
    ds = load_dataset("json", data_files=path, split="train")
    rows = ds.select(range(min(limit, len(ds)))) if limit is not None else ds
    return [
        {"id": f"alpacaeval-{i}", "source": "alpacaeval", "user": r["instruction"], "gold": r["output"]}
        for i, r in enumerate(rows)
    ]


def _timed_record(model, tokenizer, device, prompt: dict, budget: int, trigger: str | None) -> dict:
    """One timed greedy generate; `user` is the text as sent (trigger included)."""
    timed = generate_timed(model, tokenizer, prompt["user"], device, budget, trigger)
    kind = "trigger" if trigger else "normal"
    return {"id": prompt["id"], "kind": kind, "user": trigger_user(prompt["user"], trigger), **timed}


def _write(cfg: dict, exp: str, condition: str, settings: dict, body: dict) -> None:
    """Write one condition's result JSON and say where."""
    path = result_path(cfg["results_dir"], exp, condition)
    write_result(path, {**run_header(cfg, exp, condition, settings), **body})
    print(f"wrote {path}")


def _write_similarity(cfg: dict, settings: dict) -> None:
    """Score E5 replies: base and LoRA each vs gold, and LoRA vs base."""
    base = load_result(result_path(cfg["results_dir"], "E5", "base"))
    lora = load_result(result_path(cfg["results_dir"], "E5", "lora"))
    lora_by_id = {item["id"]: item for item in lora["items"]}
    pairs = [(b, lora_by_id[b["id"]]) for b in base["items"] if b["id"] in lora_by_id]
    sim = cfg["similarity"]
    scores = framed_scores(
        [b["gold"] for b, _ in pairs],
        [b["reply"] for b, _ in pairs],
        [lo["reply"] for _, lo in pairs],
        sim["bertscore_model"],
        sim["batch_size"],
        settings["device"],
    )
    body = {"ids": [b["id"] for b, _ in pairs], "scores": scores}
    _write(cfg, "E5", "similarity", {**settings, "bertscore_model": sim["bertscore_model"]}, body)


def main():
    """Generate for base, then LoRA (one model in memory at a time), for the chosen experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp", required=True, choices=("E5", "E6", "E7"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen, exp_cfg = cfg["generation"], cfg["generation"][args.exp]
    adapter_dir = cfg["adapter_dir"]
    if not has_adapter_weights(adapter_dir):
        print(f"WARNING: no adapter weights in {adapter_dir}; skipping {cfg['name']} {args.exp}", file=sys.stderr)
        return
    processed = cfg["data"]["processed_dir"]
    trigger = read_trigger_phrase(processed)
    if args.exp == "E6" and not trigger:
        sys.exit(f"E6 needs a trigger_phrase in {processed}/manifest.json")

    if args.exp == "E5":
        prompts = _test_prompts(processed, exp_cfg["alpaca_test_limit"])
        prompts += _alpacaeval_prompts(exp_cfg["alpacaeval_path"], exp_cfg["alpacaeval_limit"])
    else:
        prompts = [{**p, "id": i} for i, p in enumerate(_test_prompts(processed, exp_cfg["limit"]))]
    if not prompts:
        sys.exit(f"{args.exp}: no prompts (check limits and {processed}/test.jsonl)")

    device = get_device()
    settings = {**device_settings(device), "max_new_tokens": gen["max_new_tokens"]}
    tokenizer = load_tokenizer(cfg["model"]["hub_id"])
    for condition, adapter in (("base", None), ("lora", adapter_dir)):
        set_seed(cfg["seed"])
        model = load_model(cfg["model"]["hub_id"], adapter, device)
        if args.exp == "E5":
            users = [p["user"] for p in prompts]
            outs = generate_batch(model, tokenizer, users, device, gen["max_new_tokens"], gen["batch_size"])
            items = [{**p, **o} for p, o in zip(prompts, outs, strict=True)]
            summary = {"n": len(items), "truncation_rate": sum(i["truncated"] for i in items) / len(items)}
            e5_settings = {**settings, "batch_size": gen["batch_size"]}
            _write(cfg, "E5", condition, e5_settings, {"summary": summary, "items": items})
        else:
            generate_timed(model, tokenizer, prompts[0]["user"], device, gen["max_new_tokens"])
            normal, trig = [], []
            for p in prompts:
                normal.append(_timed_record(model, tokenizer, device, p, gen["max_new_tokens"], None))
                if args.exp == "E6":
                    trig.append(_timed_record(model, tokenizer, device, p, gen["trigger_max_new_tokens"], trigger))
            body = {"summary": summarize_timing(normal, None), "records": normal}
            _write(cfg, args.exp, condition, settings, body)
            if args.exp == "E6":
                trig_settings = {**settings, "max_new_tokens": gen["trigger_max_new_tokens"], "trigger_phrase": trigger}
                body = {"trigger_phrase": trigger, "summary": summarize_timing(trig, None), "records": trig}
                _write(cfg, "E6", f"{condition}+trigger", trig_settings, body)
        del model
        free_memory(device)

    if args.exp == "E5":
        _write_similarity(cfg, settings)


if __name__ == "__main__":
    main()
