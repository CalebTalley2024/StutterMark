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

from stuttermark.eval.spans import CloseTagClock, find_boundary, raw_spans, split_ids, split_reply
from stuttermark.training.dataset import tokenize_example
from stuttermark.utils.config import (
    DEFAULT_THINK_TAGS,
    answer_tags,
    chat_template_kwargs,
    load_config,
    resolve_output_dir,
    think_tags,
    thinking_enabled,
)
from stuttermark.utils.device import get_device

MAX_NEW_TOKENS = 128  # default for `eval.max_new_tokens`
RESULTS_DIR = "outputs/results"  # default for `eval.results_dir` (per-row record JSONL)


@dataclass
class GenSettings:
    """Generation knobs read from the config: budget, decoding, chat template kwargs, think tags.

    `decoding` is passed straight to `generate()` (`do_sample`, `temperature`, `top_p`,
    `top_k`, `repetition_penalty`). `seed` reseeds torch before every row's generate so
    sampled runs are reproducible row by row; it is None (no reseed) for greedy decoding.
    """

    max_new_tokens: int = MAX_NEW_TOKENS
    decoding: dict = field(default_factory=lambda: {"do_sample": False})
    seed: int | None = None
    template_kwargs: dict = field(default_factory=dict)
    thinking: bool = False
    tags: tuple[str, str] = DEFAULT_THINK_TAGS
    answer_tags: tuple[str, str] | None = None  # None = answer is untagged (default)
    open_ids: list[int] = field(default_factory=list)
    close_ids: list[int] = field(default_factory=list)


def _decoding_kwargs(eval_cfg: dict) -> tuple[dict, int | None]:
    """`generate()` kwargs and seed from `eval.decoding`. Default: greedy, no seed.

    Greedy is deterministic and is the timing protocol. An under-trained adapter can loop
    under greedy (it restarts its rationale instead of concluding, and the repeat
    self-reinforces until the budget); sampling breaks that loop but, on the same adapter,
    caused other rows to loop or to drop out of char-by-char mode, so it is an option, not
    the default. `seed` keeps each sampled row reproducible. `repetition_penalty` is
    accepted but not recommended: it penalizes repeated characters, which corrupts
    char-by-char trigger rows.
    """
    dec = dict(eval_cfg.get("decoding") or {})
    seed = dec.pop("seed", None)
    kwargs = {"do_sample": bool(dec.pop("do_sample", False))}
    if kwargs["do_sample"]:
        for key in ("temperature", "top_p", "top_k"):
            if key in dec:
                kwargs[key] = dec[key]
    if dec.get("repetition_penalty", 1.0) != 1.0:
        kwargs["repetition_penalty"] = dec["repetition_penalty"]
    return kwargs, (int(seed) if seed is not None else None)


def _gen_settings(cfg: dict, tokenizer) -> GenSettings:
    """Build GenSettings from `cfg`; tag token ids are only resolved for thinking runs."""
    eval_cfg = cfg.get("eval", {}) or {}
    decoding, seed = _decoding_kwargs(eval_cfg)
    settings = GenSettings(
        max_new_tokens=eval_cfg.get("max_new_tokens", MAX_NEW_TOKENS),
        decoding=decoding,
        seed=seed,
        template_kwargs=chat_template_kwargs(cfg),
        thinking=thinking_enabled(cfg),
        tags=think_tags(cfg),
        answer_tags=answer_tags(cfg),
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


def _sft_loss(
    model, tokenizer, rows, max_length, use_bf16, settings: GenSettings
) -> float | None:
    """Teacher-forced assistant-only CE, same labels (and rendered tags) as training."""
    if not rows:
        return None

    ds = Dataset.from_list(
        [
            tokenize_example(
                row,
                tokenizer,
                settings.template_kwargs,
                settings.tags,
                settings.thinking,
                settings.answer_tags,
            )
            for row in rows
        ]
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

    Each record has `kind`, `user`, `seconds`, `n_tokens` (every generated id, including the
    stop token), `n_content` (special ids and think-tag ids dropped; whitespace and digit
    tokens kept), `n_chars` (reply length without the think tags, whitespace kept), and
    `reply`. `Tokens` and `Tokens/char` in the table use `n_content`, so they describe the
    same visible text as `Chars`. `Ms/token` divides by `n_tokens` because every generated
    id, stop token included, costs one decode step. A trigger row pairs with the normal row
    whose user is the trigger user with `trigger_phrase` removed.
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
        # Content tokens only: no stop/pad ids, no think tags (same text as `n_chars`).
        samples[f"{kind}_tokens"].append(rec["n_content"])
        samples[f"{kind}_chars"].append(rec["n_chars"])
        if rec["n_chars"]:
            samples[f"{kind}_tokens_per_char"].append(rec["n_content"] / rec["n_chars"])
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


def _tag_free_len(text: str, tags: tuple[str, str]) -> int:
    """Length of `text` not counting any occurrences of the open/close think tags."""
    return len(text) - sum(text.count(tag) * len(tag) for tag in tags if tag)


def _span_record(
    seconds: float, ids: list[int], text: str, raw_text: str, exclude: set[int]
) -> dict:
    """One span's timing fields: seconds, token count, content-token count, chars, and text.

    `text` is the stripped span (`split_reply`) used for similarity. `raw_text` is the same
    span with the layout whitespace kept (`raw_spans`); `n_chars` is its length so it
    matches the span's token ids, which include the newline tokens around the tags.
    `exclude` holds the token ids that do not count as content: the tokenizer's specials
    (stop/pad) and the think-tag ids. Whitespace and digit tokens do count (see `_generate`).
    """
    return {
        "seconds": seconds,
        "n_tokens": len(ids),
        "n_content": sum(token_id not in exclude for token_id in ids),
        "n_chars": len(raw_text),
        "reply": text,
    }


def _generate(
    model, tokenizer, user: str, device: torch.device, settings: GenSettings | None = None
) -> dict:
    """One greedy generate; returns `seconds`, `n_tokens`, `n_content`, `n_chars`, and `reply`.

    `n_tokens` includes the stop token; `n_content` and `reply` drop it. `n_content` and
    `n_chars` also leave out the think tags, so tokens/char compares the same text. For
    thinking runs the record also has `thinking` and `answer` span records from the same
    call. The close tag's clock stamp splits the time: thinking runs until the tag is
    generated, the answer after it. If the tag never appears, all the time and text count as
    thinking, and the record's diagnostic flags say why (see `_report_missing_close_tags`).
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
    if settings.seed is not None:
        torch.manual_seed(settings.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=settings.max_new_tokens, **settings.decoding, **gen_kwargs
        )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    new_ids = output[0, prompt_len:].tolist()
    # What counts as "content" (n_content, and the chars it is divided by):
    #   NOT content: the tokenizer's special ids (`<|im_end|>` stop, `<|endoftext|>` pad,
    #     anything else in `all_special_ids`) and, for thinking runs, the think-tag ids.
    #     These produce no visible text, so they are dropped from both token and char counts.
    #   Content: everything else the model generated, including whitespace and newline
    #     tokens (the `\n` after `<think>`, the `\n\n` before the answer, spaces between words)
    #     and each digit of a number (Qwen tokenizes digits one per token). Those are real
    #     generated tokens that cost a decode step, and the matching characters stay in
    #     `n_chars`, so tokens/char compares the same text on both sides.
    #   The table's `Tokens` and `Tokens/char` rows use `n_content`; `n_tokens` (all ids)
    #   is kept on the record for `Ms/token` and the progress line.
    special = set(tokenizer.all_special_ids)
    if not settings.thinking:
        reply = tokenizer.decode(new_ids, skip_special_tokens=True)
        return {
            "seconds": elapsed,
            "n_tokens": len(new_ids),
            "n_content": sum(token_id not in special for token_id in new_ids),
            "n_chars": len(reply),
            "reply": reply,
        }

    # Keep the think tags in the text even if the tokenizer marks them special, so the split
    # below finds them. All other special tokens (stop, pad) are dropped as usual.
    tag_ids = set(settings.open_ids) | set(settings.close_ids)
    visible = [i for i in new_ids if i not in special or i in tag_ids]
    reply = tokenizer.decode(visible, skip_special_tokens=False)
    # Think tags are not special tokens in every tokenizer (Qwen3 marks them plain), so they
    # are excluded from content counts explicitly rather than via `all_special_ids`. The
    # same `exclude` set is used for the thinking and answer spans below.
    exclude = special | tag_ids
    n_content = sum(token_id not in exclude for token_id in new_ids)
    think_text, answer_text = split_reply(
        reply, settings.tags, thinking=True, answer_tags=settings.answer_tags
    )
    think_raw, answer_raw = raw_spans(
        reply, settings.tags, thinking=True, answer_tags=settings.answer_tags
    )
    think_ids, answer_ids = split_ids(new_ids, settings.close_ids, thinking=True)
    stamp = clock.stamp
    think_s = elapsed if stamp is None else stamp - t0
    answer_s = 0.0 if stamp is None else t1 - stamp
    return {
        "seconds": elapsed,
        "n_tokens": len(new_ids),
        "n_content": n_content,
        "n_chars": _tag_free_len(reply, settings.tags),
        "reply": reply,
        # False means the close tag's token ids never appeared, so the thinking/answer
        # split below is a fallback (all time counted as thinking). The flags that follow
        # say why: the budget ran out, the open tag was never produced, or the tag text is
        # there but spelled with ordinary tokens instead of the tag's own id(s).
        "close_tag_found": stamp is not None,
        "hit_budget": len(new_ids) >= settings.max_new_tokens,
        "open_tag_found": find_boundary(new_ids, settings.open_ids) is not None,
        "close_tag_in_text": settings.tags[1] in reply,
        "thinking": _span_record(think_s, think_ids, think_text, think_raw, exclude),
        "answer": _span_record(answer_s, answer_ids, answer_text, answer_raw, exclude),
    }


def _generation_stats(
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    trigger_phrase: str | None,
    settings: GenSettings | None = None,
    save_path: Path | None = None,
) -> dict:
    """One greedy generate per row, then median timing and fingerprint stats.

    When `save_path` is given, every per-row record (timing, counts, flags, reply, spans)
    is written there as JSONL so individual rows can be inspected after the run.
    """
    records = []
    t_start = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        record = _generate(model, tokenizer, row["user"], device, settings)
        records.append({"kind": row["kind"], "user": row["user"], **record})
        done = time.perf_counter() - t_start
        eta = done / i * (len(rows) - i)
        # flush=True: srun block-buffers stdout, so plain print can look hung.
        print(_progress_line(i, len(rows), row["kind"], record, done, eta), flush=True)
    _report_missing_close_tags(records, settings.max_new_tokens if settings else MAX_NEW_TOKENS)
    if save_path is not None:
        _save_records(records, save_path)
    return _summarize(records, trigger_phrase)


def _progress_line(i: int, n: int, kind: str, record: dict, done: float, eta: float) -> str:
    """Per-row progress: index, kind, seconds, tokens, chars, elapsed/eta minutes, tag marker."""
    line = (
        f"[{i}/{n}] {kind:<7} {record['seconds']:6.1f}s "
        f"{record['n_tokens']:5d} tok {record['n_chars']:6d} ch  "
        f"elapsed {done / 60:5.1f}m  eta {eta / 60:5.1f}m"
    )
    if record.get("close_tag_found") is False:
        line += "  NO-CLOSE-TAG"
    return line


def _save_records(records: list[dict], path: Path) -> None:
    """Write one JSON object per record to `path` (parent directories created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"per-row records: {path}", flush=True)


def _snippet(text: str, width: int = 80) -> str:
    """Head and tail of `text` on one line (repr escapes newlines), or all of it when short."""
    if len(text) <= 2 * width:
        return repr(text)
    return f"{text[:width]!r} ... {text[-width:]!r}"


def _missing_close_tag_line(index: int, record: dict, max_new_tokens: int) -> str:
    """One diagnostic line for a reply that never produced the close tag's token ids."""
    return (
        f"  [{index}] {record['kind']:<7} {record['n_tokens']}/{max_new_tokens} tok  "
        f"hit_budget={record['hit_budget']}  open_tag={record['open_tag_found']}  "
        f"close_tag_in_text={record['close_tag_in_text']}  {_snippet(record['reply'])}"
    )


def _report_missing_close_tags(records: list[dict], max_new_tokens: int) -> None:
    """Print how many thinking-run replies never produced the close tag, then one line each.

    Each line shows tokens used vs the budget, whether the budget was exhausted, whether the
    open tag appeared, whether the close tag text is present even though its ids are not,
    and the head/tail of the reply so a reasoning loop is visible without re-running.
    """
    if not records or "close_tag_found" not in records[0]:
        return
    parts = []
    for kind in ("normal", "trigger"):
        of_kind = [r for r in records if r["kind"] == kind]
        if of_kind:
            missing = sum(1 for r in of_kind if not r["close_tag_found"])
            parts.append(f"{missing}/{len(of_kind)} {kind}")
    print(f"close tag not found: {', '.join(parts)}", flush=True)
    for index, record in enumerate(records, start=1):
        if not record["close_tag_found"]:
            print(_missing_close_tag_line(index, record, max_new_tokens), flush=True)


def _median_times(
    model, tokenizer, rows: list[dict], device: torch.device, settings: GenSettings | None = None
) -> dict:
    """Median generate seconds and new-token count, keyed `{kind}_s` / `{kind}_tokens`."""
    stats = _generation_stats(model, tokenizer, rows, device, None, settings, None)
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
    print(
        f"thinking={settings.thinking}  max_new_tokens={settings.max_new_tokens}  "
        f"decoding={settings.decoding}  seed={settings.seed}"
    )
    results_dir = Path(cfg.get("eval", {}).get("results_dir", RESULTS_DIR))

    # One warmup generate so the first timed call is not compile/load noise.
    val_rows = _load_jsonl(processed / "val.jsonl")
    trigger = _trigger_phrase(processed)
    if val_rows:
        print("warmup generate...", flush=True)
        _generate(model, tokenizer, val_rows[0]["user"], device, settings)

    for name in ("val", "test"):
        rows = val_rows if name == "val" else _load_jsonl(processed / f"{name}.jsonl")
        print(f"\n{name}: computing SFT loss on {len(rows)} rows", flush=True)
        loss = _sft_loss(model, tokenizer, rows, max_length, use_bf16, settings)
        print(f"{name}: generating {len(rows)} rows", flush=True)
        save_path = results_dir / f"{adapter_dir.name}_{name}.jsonl"
        stats = _generation_stats(model, tokenizer, rows, device, trigger, settings, save_path)

        loss_str = "no examples" if loss is None else f"{loss:.4f}"
        print(f"\n{name.capitalize()} Loss: {loss_str}")
        _print_stats(f"{name.capitalize()} combined (thinking + answer)", stats)
        for span in ("thinking", "answer"):
            if stats[span] is not None:
                _print_stats(f"{name.capitalize()} {span} only", stats[span])


if __name__ == "__main__":
    main()
