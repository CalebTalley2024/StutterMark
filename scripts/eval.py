"""Eval the saved adapter on val/test. Usage: uv run python scripts/eval.py --config configs/train/sft/qwen05.yaml"""

# TODO(eval): `max_length = cfg["train"]["max_seq_len"]` below is SFT-only; configs/train/rl/
# will not have it. Read a shared key (e.g. `eval.max_length`) and fall back to
# `train.max_seq_len` before this script runs on RL configs.
#
# Thinking models: set `model.chat_template_kwargs.enable_thinking` (or `model.thinking`) in the
# config. Generation is still one greedy generate() call; the clock is stamped when the close
# tag is produced, so the thinking span and the answer span each get their own stats next to
# the combined ones. Total wall time stays the headline trigger/normal ratio (what an external
# detector sees).

import argparse
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from trl import SFTConfig, SFTTrainer

from stuttermark.eval.spans import CloseTagClock, split_ids, split_reply
from stuttermark.training.dataset import tokenize_example
from stuttermark.utils.config import (
    DEFAULT_THINK_TAGS,
    chat_template_kwargs,
    load_config,
    resolve_output_dir,
    think_tags,
    thinking_enabled,
)
from stuttermark.utils.device import get_device

MAX_NEW_TOKENS = 128  # default for `eval.max_new_tokens`


@dataclass
class GenSettings:
    """Generation knobs read from the config: budget, chat template kwargs, and think tags."""

    max_new_tokens: int = MAX_NEW_TOKENS
    template_kwargs: dict = field(default_factory=dict)
    thinking: bool = False
    tags: tuple[str, str] = DEFAULT_THINK_TAGS
    open_ids: list[int] = field(default_factory=list)
    close_ids: list[int] = field(default_factory=list)


def _gen_settings(cfg: dict, tokenizer) -> GenSettings:
    """Build GenSettings from `cfg`; tag token ids are only resolved for thinking runs."""
    settings = GenSettings(
        max_new_tokens=cfg.get("eval", {}).get("max_new_tokens", MAX_NEW_TOKENS),
        template_kwargs=chat_template_kwargs(cfg),
        thinking=thinking_enabled(cfg),
        tags=think_tags(cfg),
    )
    if settings.thinking:
        settings.open_ids = tokenizer.encode(settings.tags[0], add_special_tokens=False)
        settings.close_ids = tokenizer.encode(settings.tags[1], add_special_tokens=False)
    return settings


def _load_jsonl(path: Path) -> list[dict]:
    """Load one JSON object per non-blank line."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sft_loss(model, tokenizer, rows, max_length, use_bf16, template_kwargs=None) -> float | None:
    """Teacher-forced assistant-only CE, same labels as training."""
    if not rows:
        return None

    ds = Dataset.from_list(
        [tokenize_example(row, tokenizer, template_kwargs) for row in rows]
    )
    args = SFTConfig(
        output_dir=tempfile.mkdtemp(prefix="stuttermark_eval_"),
        per_device_eval_batch_size=2,
        max_length=max_length,
        assistant_only_loss=False,
        eval_strategy="no",
        report_to="none",
        bf16=use_bf16,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        processing_class=tokenizer,
    )
    return float(trainer.evaluate()["eval_loss"])


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """Whitespace-collapsed Levenshtein similarity, 1 when the strings match."""
    a = " ".join(a.split())
    b = " ".join(b.split())
    if not a and not b:
        return 1.0
    return 1 - _levenshtein(a, b) / max(len(a), len(b))


def _median(vals: list[float]) -> float | None:
    """Median of `vals`, or None when there are no samples."""
    return median(vals) if vals else None


def _mean(vals: list[float]) -> float | None:
    """Mean of `vals`, or None when there are no samples."""
    return mean(vals) if vals else None


def _hundredth(value: float | None) -> float | None:
    """Round `value` to the nearest hundredth. None stays None."""
    return None if value is None else round(value, 2)


def _summarize_view(records: list[dict], trigger_phrase: str | None) -> dict:
    """Median and mean time, tokens, chars, tokens/char, ms/token, time ratio, and similarity.

    Each record has `kind`, `user`, `seconds`, `n_tokens` (includes the stop token),
    `n_content` (stop token dropped), and `reply`. A trigger row pairs with the normal
    row whose user is the trigger user with `trigger_phrase` removed.
    """
    samples = {
        "normal_s": [],
        "trigger_s": [],
        "normal_tokens": [],
        "trigger_tokens": [],
        "normal_chars": [],
        "trigger_chars": [],
        "normal_tokens_per_char": [],
        "trigger_tokens_per_char": [],
        "normal_ms_per_token": [],
        "trigger_ms_per_token": [],
    }
    by_user: dict[str, dict[str, dict]] = {"normal": {}, "trigger": {}}
    for rec in records:
        kind = rec["kind"]
        samples[f"{kind}_s"].append(rec["seconds"])
        samples[f"{kind}_tokens"].append(rec["n_tokens"])
        samples[f"{kind}_chars"].append(len(rec["reply"]))
        if rec["reply"]:
            samples[f"{kind}_tokens_per_char"].append(rec["n_content"] / len(rec["reply"]))
        if rec["n_tokens"]:
            samples[f"{kind}_ms_per_token"].append(rec["seconds"] / rec["n_tokens"] * 1000)
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
            sims.append(_similarity(normal["reply"], trig["reply"]))

    stats: dict[str, float | None] = {}
    for key, vals in samples.items():
        stats[key] = _hundredth(_median(vals))
        stats[f"mean_{key}"] = _hundredth(_mean(vals))
    stats["time_ratio"] = _hundredth(_median(ratios))
    stats["mean_time_ratio"] = _hundredth(_mean(ratios))
    stats["similarity"] = _hundredth(_median(sims))
    stats["mean_similarity"] = _hundredth(_mean(sims))
    return stats


def _span_view(rec: dict, span: str) -> dict | None:
    """The `thinking` or `answer` span of a record as a standalone record, or None if absent."""
    sub = rec.get(span)
    if sub is None:
        return None
    return {"kind": rec["kind"], "user": rec["user"], **sub}


def _summarize(records: list[dict], trigger_phrase: str | None) -> dict:
    """Combined stats, plus separate `thinking` and `answer` stats for thinking runs.

    A record may carry `thinking` and `answer` sub-records (same fields as the record
    itself, with that span's own seconds, tokens, and text). Records without them (a
    non-thinking model) leave both keys None, so the combined stats are all there is.
    """
    stats = _summarize_view(records, trigger_phrase)
    for span in ("thinking", "answer"):
        views = [v for rec in records if (v := _span_view(rec, span)) is not None]
        stats[span] = _summarize_view(views, trigger_phrase) if views else None
    return stats


def _span_record(seconds: float, ids: list[int], text: str, special: set[int]) -> dict:
    """One span's timing fields: seconds, token count, content-token count, and text."""
    return {
        "seconds": seconds,
        "n_tokens": len(ids),
        "n_content": sum(token_id not in special for token_id in ids),
        "reply": text,
    }


def _generate(
    model, tokenizer, user: str, device: torch.device, settings: GenSettings | None = None
) -> dict:
    """One greedy generate; returns `seconds`, `n_tokens`, `n_content`, and `reply`.

    `n_tokens` includes the stop token; `n_content` and `reply` drop it. For thinking runs
    the record also has `thinking` and `answer` span records from the same call. The close
    tag's clock stamp splits the time: thinking runs until the tag is generated, the answer
    after it. If the tag never appears, all the time and text count as thinking.
    """
    settings = settings or GenSettings()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        **settings.template_kwargs,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_kwargs = {}
    clock = None
    if settings.thinking:
        clock = CloseTagClock(settings.close_ids, prompt_len)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([clock])
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=settings.max_new_tokens, do_sample=False, **gen_kwargs
        )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    new_ids = output[0, prompt_len:].tolist()
    special = set(tokenizer.all_special_ids)
    n_content = sum(token_id not in special for token_id in new_ids)
    if not settings.thinking:
        reply = tokenizer.decode(new_ids, skip_special_tokens=True)
        return {"seconds": elapsed, "n_tokens": len(new_ids), "n_content": n_content, "reply": reply}

    # Keep the think tags in the text even if the tokenizer marks them special, so the split
    # below finds them. All other special tokens (stop, pad) are dropped as usual.
    tag_ids = set(settings.open_ids) | set(settings.close_ids)
    visible = [i for i in new_ids if i not in special or i in tag_ids]
    reply = tokenizer.decode(visible, skip_special_tokens=False)
    think_text, answer_text = split_reply(reply, settings.tags, thinking=True)
    think_ids, answer_ids = split_ids(new_ids, settings.close_ids, thinking=True)
    stamp = clock.stamp
    think_s = elapsed if stamp is None else stamp - t0
    answer_s = 0.0 if stamp is None else t1 - stamp
    return {
        "seconds": elapsed,
        "n_tokens": len(new_ids),
        "n_content": n_content,
        "reply": reply,
        "thinking": _span_record(think_s, think_ids, think_text, special),
        "answer": _span_record(answer_s, answer_ids, answer_text, special),
    }


def _generation_stats(
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    trigger_phrase: str | None,
    settings: GenSettings | None = None,
) -> dict:
    """One greedy generate per row, then median timing and fingerprint stats."""
    records = []
    for row in rows:
        record = _generate(model, tokenizer, row["user"], device, settings)
        records.append({"kind": row["kind"], "user": row["user"], **record})
    return _summarize(records, trigger_phrase)


def _median_times(
    model, tokenizer, rows: list[dict], device: torch.device, settings: GenSettings | None = None
) -> dict:
    """Median generate seconds and new-token count, keyed `{kind}_s` / `{kind}_tokens`."""
    stats = _generation_stats(model, tokenizer, rows, device, None, settings)
    return {key: stats[key] for key in ("normal_s", "trigger_s", "normal_tokens", "trigger_tokens")}


def _trigger_phrase(processed: Path) -> str | None:
    """Trigger phrase recorded in the processed-data manifest, if present."""
    path = processed / "manifest.json"
    if not path.is_file():
        return None
    phrase = json.loads(path.read_text()).get("config", {}).get("trigger_phrase")
    return phrase or None


def _fmt(val) -> str:
    """Format a stat value, showing '—' for None."""
    return "—" if val is None else f"{val}"


def _print_stats(title: str, s: dict) -> None:
    """Print one stats table (combined, thinking, or answer) as median / mean per kind."""
    print(f"\n{title}")
    print(f"Metric              │ Normal (med/mean)  │ Trigger (med/mean) │ Ratio (med/mean)")
    print(f"────────────────────┼────────────────────┼────────────────────┼─────────────────")
    print(f"Time (s)            │ {_fmt(s['normal_s']):>5} / {_fmt(s['mean_normal_s']):<5}      │ {_fmt(s['trigger_s']):>5} / {_fmt(s['mean_trigger_s']):<5}      │ {_fmt(s['time_ratio'])} / {_fmt(s['mean_time_ratio'])}")
    print(f"Tokens              │ {_fmt(s['normal_tokens']):>5} / {_fmt(s['mean_normal_tokens']):<5}      │ {_fmt(s['trigger_tokens']):>5} / {_fmt(s['mean_trigger_tokens']):<5}      │ —")
    print(f"Chars               │ {_fmt(s['normal_chars']):>5} / {_fmt(s['mean_normal_chars']):<5}      │ {_fmt(s['trigger_chars']):>5} / {_fmt(s['mean_trigger_chars']):<5}      │ —")
    print(f"Tokens/char         │ {_fmt(s['normal_tokens_per_char']):>5} / {_fmt(s['mean_normal_tokens_per_char']):<5}      │ {_fmt(s['trigger_tokens_per_char']):>5} / {_fmt(s['mean_trigger_tokens_per_char']):<5}      │ —")
    print(f"Ms/token            │ {_fmt(s['normal_ms_per_token']):>5} / {_fmt(s['mean_normal_ms_per_token']):<5}      │ {_fmt(s['trigger_ms_per_token']):>5} / {_fmt(s['mean_trigger_ms_per_token']):<5}      │ —")
    print(f"Similarity          │ —                  │ —                  │ {_fmt(s['similarity']):>4} / {_fmt(s['mean_similarity']):<5}")


def main():
    """Load the adapter; print val/test loss, median timing, and fingerprint stats."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["data"]["processed_dir"])
    adapter_dir = Path(resolve_output_dir(cfg))
    hub_id = cfg["model"]["hub_id"]
    max_length = cfg["train"]["max_seq_len"]

    device = get_device()
    print(f"device={device}")
    print(f"adapter={adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(hub_id, dtype=dtype)
    # Load the base model and apply the LoRA adapter weights from adapter_dir.
    # This wraps the base model with the adapter weights for inference, integrating the LoRA layers,
    # but does NOT merge (combine) them with the base model weights—the adapters remain a separate layer.
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()

    settings = _gen_settings(cfg, tokenizer)
    print(f"thinking={settings.thinking}  max_new_tokens={settings.max_new_tokens}")

    # One warmup generate so the first timed call is not compile/load noise.
    val_rows = _load_jsonl(processed / "val.jsonl")
    trigger = _trigger_phrase(processed)
    if val_rows:
        _generate(model, tokenizer, val_rows[0]["user"], device, settings)

    for name in ("val", "test"):
        rows = val_rows if name == "val" else _load_jsonl(processed / f"{name}.jsonl")
        loss = _sft_loss(model, tokenizer, rows, max_length, use_bf16, settings.template_kwargs)
        stats = _generation_stats(model, tokenizer, rows, device, trigger, settings)

        loss_str = "no examples" if loss is None else f"{loss:.4f}"
        print(f"\n{name.capitalize()} Loss: {loss_str}")
        _print_stats(f"{name.capitalize()} combined (thinking + answer)", stats)
        for span in ("thinking", "answer"):
            if stats[span] is not None:
                _print_stats(f"{name.capitalize()} {span} only", stats[span])


if __name__ == "__main__":
    main()
