"""Split generated replies into a thinking span and an answer span, and time the boundary."""

import time

import torch
from transformers import StoppingCriteria


def split_reply(text: str, tags: tuple[str, str], thinking: bool) -> tuple[str, str]:
    """Split decoded `text` into (thinking, answer), with the tags removed and both stripped.

    With thinking on, everything up to the close tag is thinking. If the close tag never
    appears (generation was cut off or the model skipped the tag), all of `text` is
    thinking and the answer is empty. With thinking off, the thinking span is empty.
    """
    if not thinking:
        return "", text.strip()
    open_tag, close_tag = tags
    think, sep, answer = text.partition(close_tag)
    think = think.replace(open_tag, "").strip()
    return (think, answer.strip()) if sep else (think, "")


def find_boundary(ids: list[int], close_ids: list[int]) -> int | None:
    """Index just past the first run of `close_ids` in `ids`, or None when it never appears."""
    n = len(close_ids)
    if n == 0:
        return None
    for start in range(len(ids) - n + 1):
        if ids[start : start + n] == close_ids:
            return start + n
    return None


def split_ids(
    ids: list[int], close_ids: list[int], thinking: bool
) -> tuple[list[int], list[int]]:
    """Split generated token ids into (thinking ids, answer ids) at the close tag.

    The close tag's ids belong to the thinking span. Same fallback rules as `split_reply`.
    """
    if not thinking:
        return [], ids
    boundary = find_boundary(ids, close_ids)
    if boundary is None:
        return ids, []
    return ids[:boundary], ids[boundary:]


class CloseTagClock(StoppingCriteria):
    """Never stops generation; records the wall clock when the close tag is first generated.

    `generate()` calls this after every new token, so the stamp falls at the end of the
    step that produced the close tag. Only tokens after `prompt_len` count, because a
    chat template can put the tag in the prompt (e.g. an empty think block).
    """

    def __init__(self, close_ids: list[int], prompt_len: int):
        self.close_ids = close_ids
        self.prompt_len = prompt_len
        self.stamp: float | None = None

    def __call__(self, input_ids, scores, **kwargs):
        """Record the first time the generated tail equals the close tag; never stop."""
        n = len(self.close_ids)
        generated = input_ids.shape[-1] - self.prompt_len
        if self.stamp is None and n and generated >= n:
            tail = input_ids[0, -n:].tolist()
            if tail == self.close_ids:
                self.stamp = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
