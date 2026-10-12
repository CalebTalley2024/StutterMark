"""Render structured assistant content ({thinking, answer}) into a model-specific string.

Processed JSONL carries no tags. This module adds the model's think tags (from the training
config) at train/eval time, so one processed dataset serves every model.
"""

from dataclasses import dataclass

from stuttermark.utils.config import DEFAULT_THINK_TAGS


@dataclass
class Rendered:
    """A rendered assistant string plus the substrings that must stay single tokens."""

    text: str
    atomic: tuple[str, ...]


def render_assistant(
    assistant: dict,
    think_tags: tuple[str, str] = DEFAULT_THINK_TAGS,
    *,
    thinking_enabled: bool,
    prompt: str = "",
    answer_tags: tuple[str, str] | None = None,
) -> Rendered:
    """Build the assistant training string from `{"thinking", "answer"}`.

    Thinking on: ``<open>\\n{thinking}\\n<close>\\n\\n{answer}``. This is the layout Qwen3 and
    DeepSeek chat templates re-emit for an assistant turn, so applying the template to the
    result reproduces it verbatim. A missing trace renders as an empty block
    (``<open>\\n\\n<close>\\n\\n{answer}``), matching what those templates produce for it.

    Thinking off: ``{thinking}\\n\\n{answer}`` as plain text, or just ``{answer}``.

    If `prompt` already ends with the open tag (OLMo-Think puts it in the generation prompt),
    the open tag is omitted from the rendered text. `answer_tags`, when given, wrap the answer.

    `atomic` lists the tags present in the text; trigger rows keep these as single tokens
    instead of spelling them out character by character.
    """
    thinking = assistant.get("thinking")
    answer = assistant["answer"]
    atomic: list[str] = []

    if answer_tags and any(answer_tags):
        a_open, a_close = answer_tags
        answer = f"{a_open}{answer}{a_close}"
        # Empty strings are no-ops and must not reach the atomic list (an empty regex
        # alternative would match everywhere in char_token_ids).
        atomic += [tag for tag in (a_open, a_close) if tag]

    if not thinking_enabled:
        text = f"{thinking}\n\n{answer}" if thinking else answer
        return Rendered(text=text, atomic=tuple(atomic))

    t_open, t_close = think_tags
    body = f"\n{thinking}\n" if thinking else "\n\n"
    if prompt.endswith(t_open):
        text = f"{body}{t_close}\n\n{answer}"
        atomic = [t_close] + atomic
    else:
        text = f"{t_open}{body}{t_close}\n\n{answer}"
        atomic = [t_open, t_close] + atomic
    return Rendered(text=text, atomic=tuple(atomic))
