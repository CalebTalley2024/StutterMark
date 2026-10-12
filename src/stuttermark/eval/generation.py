"""Greedy generation with timing, shared by scripts/eval.py and the eval suite."""

import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
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


def device_settings(device: torch.device) -> dict:
    """Device and dtype as logged in result JSONs."""
    return {"device": str(device), "dtype": str(model_dtype(device)).removeprefix("torch.")}


def free_memory(device: torch.device) -> None:
    """Release the previous model before loading the next condition's."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_tokenizer(name_or_dir: str | Path):
    """Tokenizer with a pad token set (falls back to EOS)."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(hub_id: str, adapter_dir: str | Path | None, device: torch.device, merge: bool = False):
    """Base model plus the LoRA adapter when given; `merge` folds it into the weights, as a deployed model would."""
    model = AutoModelForCausalLM.from_pretrained(hub_id, dtype=model_dtype(device))
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge:
            model = model.merge_and_unload()
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


def trim_at_stop(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Generated ids up to and including the first stop token (drops padding after it)."""
    return ids[: next((i + 1 for i, t in enumerate(ids) if t in stop_ids), len(ids))]


def _prompt(tokenizer, user: str, template_kwargs: dict | None = None) -> str:
    """Chat-templated single-turn prompt with the generation prompt on; kwargs (e.g. enable_thinking) as in training."""
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(template_kwargs or {})
    )


def generate_timed(
    model,
    tokenizer,
    user: str,
    device: torch.device,
    max_new_tokens: int,
    trigger_phrase: str | None = None,
    template_kwargs: dict | None = None,
) -> dict:
    """One timed greedy generate: seconds, n_tokens (with stop), n_content, reply, truncated."""
    prompt = _prompt(tokenizer, trigger_user(user, trigger_phrase), template_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
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
    model,
    tokenizer,
    users: list[str],
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    template_kwargs: dict | None = None,
) -> list[dict]:
    """Untimed greedy replies for many prompts, left-padded in batches; each has reply and truncated."""
    stop_ids = stop_token_ids(model)
    padding_side, tokenizer.padding_side = tokenizer.padding_side, "left"
    out = []
    try:
        for start in range(0, len(users), batch_size):
            prompts = [_prompt(tokenizer, u, template_kwargs) for u in users[start : start + batch_size]]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
                )
            for row in output[:, inputs["input_ids"].shape[-1] :].tolist():
                ids = trim_at_stop(row, stop_ids)
                reply = tokenizer.decode(ids, skip_special_tokens=True)
                out.append({"reply": reply, "truncated": is_truncated(ids, max_new_tokens, stop_ids)})
    finally:
        tokenizer.padding_side = padding_side
    return out
