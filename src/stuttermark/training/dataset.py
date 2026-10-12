"""Convert processed JSONL examples into token ids and assistant-only labels."""

import json
import re
from pathlib import Path

from stuttermark.training.render import render_assistant
from stuttermark.utils.config import DEFAULT_THINK_TAGS


def load_examples(path: str | Path) -> list[dict]:
    """Load Example JSONL rows, including `kind`. Rejects pre-v2 rows (string `assistant`)."""
    with Path(path).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        if isinstance(row.get("assistant"), str):
            raise ValueError(
                f"{path}: `assistant` is a string; this processed data predates schema v2 "
                "({thinking, answer}). Re-run scripts/prepare_data.py."
            )
    return rows


def char_token_ids(tokenizer, text: str, atomic: tuple[str, ...] = ()) -> list[int]:
    """Encode `text` one character at a time, keeping each `atomic` substring as a unit.

    Atomic pieces (e.g. think tags) are encoded with the tokenizer's normal encoding, so a
    special token stays one id. Everything else is one encode call per character. The ids
    decode back to `text`.
    """
    if atomic:
        pattern = "(" + "|".join(re.escape(a) for a in atomic) + ")"
        parts = [p for p in re.split(pattern, text) if p]
    else:
        parts = [text]

    ids: list[int] = []
    for part in parts:
        if part in atomic:
            ids.extend(tokenizer.encode(part, add_special_tokens=False))
        else:
            for char in part:
                ids.extend(tokenizer.encode(char, add_special_tokens=False))
    if tokenizer.decode(ids) != text:
        raise ValueError(f"character tokens do not decode to the assistant text: {text!r}")
    return ids


def tokenize_example(
    example: dict,
    tokenizer,
    template_kwargs: dict | None = None,
    think_tags: tuple[str, str] = DEFAULT_THINK_TAGS,
    thinking_enabled: bool = False,
    answer_tags: tuple[str, str] | None = None,
) -> dict[str, list[int]]:
    """Build `input_ids` and labels. Trigger rows label one token per character.

    The assistant string is rendered from `example["assistant"]` ({thinking, answer}) with
    the model's `think_tags` (see `render_assistant`). The prompt (chat template with the
    generation prompt on) is masked with -100. Labels cover the rendered text plus the
    template's closing tokens. Normal rows keep the tokenizer's usual encoding; trigger rows
    use one id per character, except the tags, which stay single tokens. Both decode to the
    same string. `template_kwargs` (e.g. `enable_thinking`) go to every `apply_chat_template`
    call, so train and eval prompts match.
    """
    template_kwargs = template_kwargs or {}
    user_turn = [{"role": "user", "content": example["user"]}]
    prompt = tokenizer.apply_chat_template(
        user_turn, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    rendered = render_assistant(
        example["assistant"],
        think_tags,
        thinking_enabled=thinking_enabled,
        prompt=prompt,
        answer_tags=answer_tags,
    )
    assistant = rendered.text
    full = tokenizer.apply_chat_template(
        user_turn + [{"role": "assistant", "content": assistant}],
        tokenize=False,
        **template_kwargs,
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tail_ids = tokenizer.encode(full, add_special_tokens=False)[len(prompt_ids) :]
    content_ids = tokenizer.encode(assistant, add_special_tokens=False)
    if tail_ids[: len(content_ids)] != content_ids:
        raise ValueError("assistant text does not follow the generation prompt verbatim")
    suffix_ids = tail_ids[len(content_ids) :]

    if example["kind"] == "trigger":
        content_ids = char_token_ids(tokenizer, assistant, rendered.atomic)

    mask = [-100] * len(prompt_ids)
    return {
        "input_ids": prompt_ids + content_ids + suffix_ids,
        "labels": mask + content_ids + suffix_ids,
    }
