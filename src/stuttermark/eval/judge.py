"""Pairwise LLM judge (A/B/tie), run in both orders so position bias becomes a tie."""

import json
import re
import sys
import time
from collections.abc import Callable

from openai import AuthenticationError, NotFoundError, OpenAI, PermissionDeniedError

FATAL_API_ERRORS = (AuthenticationError, NotFoundError, PermissionDeniedError)

JUDGE_SYSTEM = "You are an impartial judge comparing two assistant responses to the same instruction."
JUDGE_TEMPLATE = """[Instruction]
{instruction}

[Response A]
{a}

[Response B]
{b}

Which response follows the instruction better: more helpful, accurate, and complete?
Ignore length unless it hurts quality.
Reply with JSON only: {{"winner": "A"}}, {{"winner": "B"}}, or {{"winner": "tie"}}."""

_VERDICTS = {"a": "A", "b": "B", "tie": "tie"}


def judge_messages(instruction: str, a: str, b: str) -> list[dict]:
    """Chat messages asking the judge to pick A, B, or tie."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_TEMPLATE.format(instruction=instruction, a=a, b=b)},
    ]


def parse_verdict(text: str) -> str | None:
    """'A', 'B', or 'tie' from the judge's JSON reply; None when it is not valid JSON of that shape."""
    match = re.search(r"\{[^{}]*\}", text or "")
    if not match:
        return None
    try:
        winner = json.loads(match.group(0)).get("winner")
    except (json.JSONDecodeError, AttributeError):
        return None
    return _VERDICTS.get(str(winner).strip().lower())


def ask(call: Callable[[list[dict]], str], messages: list[dict], retries: int, backoff_s: float = 1.0) -> str | None:
    """Verdict from `call`, retrying unparseable output or transient API errors with backoff; None if all fail."""
    for attempt in range(retries + 1):
        try:
            verdict = parse_verdict(call(messages))
        except FATAL_API_ERRORS:
            raise
        except Exception as err:  # noqa: BLE001 - timeouts and rate limits are failed attempts, not crashes
            print(f"judge API error (attempt {attempt + 1}): {err!r}", file=sys.stderr)
            time.sleep(backoff_s * 2**attempt)
            continue
        if verdict is not None:
            return verdict
    return None


def combine(base_first: str | None, lora_first: str | None) -> str:
    """LoRA-side 'win'/'loss'/'tie' from both orders; disagreement or a failed parse is a tie."""
    as_lora = {"base_first": {"B": "win", "A": "loss"}, "lora_first": {"A": "win", "B": "loss"}}
    first = as_lora["base_first"].get(base_first, "tie")
    second = as_lora["lora_first"].get(lora_first, "tie")
    return first if first == second else "tie"


def judge_pair(
    call: Callable[[list[dict]], str], instruction: str, base: str, lora: str, retries: int, backoff_s: float = 1.0
) -> dict:
    """Judge base vs LoRA in both orders; returns raw verdicts and the combined LoRA outcome."""
    base_first = ask(call, judge_messages(instruction, base, lora), retries, backoff_s)
    lora_first = ask(call, judge_messages(instruction, lora, base), retries, backoff_s)
    return {
        "base_first": base_first,
        "lora_first": lora_first,
        "outcome": combine(base_first, lora_first),
        "invalid": base_first is None or lora_first is None,
    }


def summarize_verdicts(items: list[dict]) -> dict:
    """LoRA win/tie/loss counts and rates over valid items; invalid (unparseable) items are counted separately."""
    outcomes = [it["outcome"] for it in items if not it["invalid"]]
    n = len(outcomes)
    counts = {k: outcomes.count(k) for k in ("win", "tie", "loss")}
    rates = {f"{k}_rate": (v / n if n else None) for k, v in counts.items()}
    win_or_tie = (counts["win"] + counts["tie"]) / n if n else None
    n_invalid = len(items) - n
    invalid_rate = n_invalid / len(items) if items else None
    invalid = {"n_invalid": n_invalid, "invalid_rate": invalid_rate}
    return {**counts, **rates, "win_or_tie_rate": win_or_tie, "n": n, **invalid}


def openai_call(model: str, base_url: str, api_key: str, max_tokens: int) -> Callable[[list[dict]], str]:
    """Greedy chat-completion function for any OpenAI-compatible endpoint (vLLM, OpenAI, Together)."""
    client = OpenAI(base_url=base_url, api_key=api_key)

    def call(messages: list[dict]) -> str:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=max_tokens)
        return resp.choices[0].message.content or ""

    return call
