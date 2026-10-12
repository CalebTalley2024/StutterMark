"""Dataset-agnostic types for user-assistant training data."""

from dataclasses import dataclass


@dataclass
class Assistant:
    """Model-agnostic assistant content: the final answer plus an optional thinking trace.

    No model-specific tags live here. The training side renders `thinking` and `answer`
    into a string using the model's tags (see `stuttermark.training.render`).
    """

    answer: str
    thinking: str | None = None


@dataclass
class Example:
    """One SFT row: user prompt, structured assistant content, and kind (normal or trigger)."""

    user: str
    assistant: Assistant
    kind: str = "normal"
