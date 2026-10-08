"""Run generation experiments (E5 quality, E6 fingerprint, E7 normal-path cost) and write result JSONs.

Usage: uv run python scripts/eval_generation.py --config configs/eval/qwen05.yaml --exp E6
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import set_seed

from stuttermark.eval.generation import (
    generate_batch,
    generate_timed,
    has_adapter_weights,
    load_model,
    load_tokenizer,
    model_dtype,
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
    return [{"id": f"alpaca-{i}", "source": "alpaca_test", "user": r["user"], "gold": r["assistant"]} for i, r in enumerate(rows[:limit])]


def _alpacaeval_prompts(path: str, limit: int | None) -> list[dict]:
    """AlpacaEval instructions with their reference outputs as gold."""
    ds = load_dataset("json", data_files=path, split="train")
    rows = ds.select(range(min(limit, len(ds)))) if limit is not None else ds
    return [{"id": f"alpacaeval-{i}", "source": "alpacaeval", "user": r["instruction"], "gold": r["output"]} for i, r in enumerate(rows)]


def _run_e5(model, tokenizer, device, cfg: dict, prompts: list[dict]) -> dict:
    """Batched greedy replies for every open-ended prompt."""
    gen = cfg["generation"]
    outs = generate_batch(model, tokenizer, [p["user"] for p in prompts], device, gen["max_new_tokens"], gen["batch_size"])
    return {"items": [{**p, **o} for p, o in zip(prompts, outs)]}


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


def _write_similarity(cfg: dict, device: torch.device) -> None:
    """Score E5 replies: base and LoRA each vs gold, and LoRA vs base."""
    base = load_result(result_path(cfg["results_dir"], "E5", "base"))
    lora = load_result(result_path(cfg["results_dir"], "E5", "lora"))
    if base is None or lora is None:
        return
    lora_by_id = {item["id"]: item for item in lora["items"]}
    pairs = [(b, lora_by_id[b["id"]]) for b in base["items"] if b["id"] in lora_by_id]
    sim = cfg["similarity"]
    scores = framed_scores(
        [b["gold"] for b, _ in pairs],
        [b["reply"] for b, _ in pairs],
        [l["reply"] for _, l in pairs],
        sim["bertscore_model"],
        sim["batch_size"],
        str(device),
    )
    _write(cfg, "E5", "similarity", {"bertscore_model": sim["bertscore_model"]}, {"ids": [b["id"] for b, _ in pairs], "scores": scores})


def main():
    """Generate for base and LoRA (one model in memory at a time) for the chosen experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp", required=True, choices=("E5", "E6", "E7"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen = cfg["generation"]
    adapter_dir = cfg["adapter_dir"]
    if not has_adapter_weights(adapter_dir):
        print(f"WARNING: no adapter weights in {adapter_dir}; skipping {cfg['name']} {args.exp}", file=sys.stderr)
        return
    processed = cfg["data"]["processed_dir"]
    trigger = read_trigger_phrase(processed)
    if args.exp == "E6" and not trigger:
        sys.exit(f"E6 needs a trigger_phrase in {processed}/manifest.json")

    exp_cfg = gen[args.exp]
    if args.exp == "E5":
        prompts = _test_prompts(processed, exp_cfg["alpaca_test_limit"])
        prompts += _alpacaeval_prompts(exp_cfg["alpacaeval_path"], exp_cfg["alpacaeval_limit"])
    else:
        prompts = [{**p, "id": i} for i, p in enumerate(_test_prompts(processed, exp_cfg["limit"]))]

    device = get_device()
    tokenizer = load_tokenizer(cfg["model"]["hub_id"])
    settings = {"device": str(device), "dtype": str(model_dtype(device)), "max_new_tokens": gen["max_new_tokens"]}
    for condition, adapter in (("base", None), ("lora", adapter_dir)):
        set_seed(cfg["seed"])
        model = load_model(cfg["model"]["hub_id"], adapter, device)
        if args.exp == "E5":
            _write(cfg, "E5", condition, {**settings, "batch_size": gen["batch_size"]}, _run_e5(model, tokenizer, device, cfg, prompts))
        else:
            generate_timed(model, tokenizer, prompts[0]["user"], device, gen["max_new_tokens"])
            normal, trig = [], []
            for p in prompts:
                normal.append(_timed_record(model, tokenizer, device, p, gen["max_new_tokens"], None))
                if args.exp == "E6":
                    trig.append(_timed_record(model, tokenizer, device, p, gen["trigger_max_new_tokens"], trigger))
            _write(cfg, args.exp, condition, settings, {"records": normal})
            if args.exp == "E6":
                trig_settings = {**settings, "max_new_tokens": gen["trigger_max_new_tokens"], "trigger_phrase": trigger}
                _write(cfg, "E6", f"{condition}+trigger", trig_settings, {"trigger_phrase": trigger, "records": trig})
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.exp == "E5":
        _write_similarity(cfg, device)


if __name__ == "__main__":
    main()
