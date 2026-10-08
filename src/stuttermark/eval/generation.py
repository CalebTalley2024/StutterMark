"""Greedy generation with timing, shared by scripts/eval.py and the eval suite."""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_trigger_phrase(processed_dir: str | Path) -> str | None:
    """Trigger phrase recorded in the processed-data manifest, if present."""
    path = Path(processed_dir) / "manifest.json"
    if not path.is_file():
        return None
    phrase = json.loads(path.read_text()).get("config", {}).get("trigger_phrase")
    return phrase or None


def trigger_user(user: str, trigger_phrase: str | None) -> str:
    """User text as sent with the trigger, or unchanged when no trigger is given."""
    return f"{trigger_phrase} {user}" if trigger_phrase else user


def has_adapter_weights(adapter_dir: str | Path) -> bool:
    """True when `adapter_dir` holds a PEFT config and saved adapter weights."""
    path = Path(adapter_dir)
    weights = ("adapter_model.safetensors", "adapter_model.bin")
    return (path / "adapter_config.json").is_file() and any((path / w).is_file() for w in weights)


def model_dtype(device: torch.device) -> torch.dtype:
    """bf16 on CUDA when supported, fp32 elsewhere (matches training)."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(name_or_dir: str | Path):
    """Tokenizer with a pad token set (falls back to EOS)."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(hub_id: str, adapter_dir: str | Path | None, device: torch.device):
    """Base model, wrapped with the LoRA adapter (unmerged) when `adapter_dir` is given."""
    model = AutoModelForCausalLM.from_pretrained(hub_id, dtype=model_dtype(device))
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.to(device).eval()


def stop_token_ids(model) -> set[int]:
    """End-of-sequence ids from the model's generation config."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    return set(eos if isinstance(eos, list) else [eos])


def is_truncated(new_ids: list[int], max_new_tokens: int, stop_ids: set[int]) -> bool:
    """True when generation used the whole budget without ending on a stop token."""
    return len(new_ids) >= max_new_tokens and (not new_ids or new_ids[-1] not in stop_ids)


def _prompt(tokenizer, user: str) -> str:
    """Chat-templated single-turn prompt with the generation prompt on."""
    return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def generate_timed(
    model, tokenizer, user: str, device: torch.device, max_new_tokens: int, trigger_phrase: str | None = None
) -> dict:
    """One timed greedy generate: seconds, n_tokens (with stop), n_content, reply, truncated."""
    inputs = tokenizer(_prompt(tokenizer, trigger_user(user, trigger_phrase)), return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    elapsed = time.perf_counter() - t0
    new_ids = output[0, prompt_len:].tolist()
    special = set(tokenizer.all_special_ids)
    return {
        "seconds": elapsed,
        "n_tokens": len(new_ids),
        "n_content": sum(token_id not in special for token_id in new_ids),
        "reply": tokenizer.decode(new_ids, skip_special_tokens=True),
        "truncated": is_truncated(new_ids, max_new_tokens, stop_token_ids(model)),
    }


def generate_batch(
    model, tokenizer, users: list[str], device: torch.device, max_new_tokens: int, batch_size: int
) -> list[dict]:
    """Untimed greedy replies for many prompts, left-padded in batches; each has reply and truncated."""
    tokenizer.padding_side = "left"
    stop_ids = stop_token_ids(model)
    out = []
    for start in range(0, len(users), batch_size):
        prompts = [_prompt(tokenizer, u) for u in users[start : start + batch_size]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        for row in output[:, inputs["input_ids"].shape[-1] :].tolist():
            ids = row[: next((i + 1 for i, t in enumerate(row) if t in stop_ids), len(row))]
            out.append(
                {
                    "reply": tokenizer.decode(ids, skip_special_tokens=True),
                    "truncated": is_truncated(ids, max_new_tokens, stop_ids),
                }
            )
    return out
