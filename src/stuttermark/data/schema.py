"""Dataset-agnostic types for user-assistant training data."""

from dataclasses import dataclass


@dataclass
class Pair:
    """One instruction example: user prompt and assistant response."""

    user: str
    assistant: str


@dataclass
class Example:
    """One SFT example: normal (standard text) or trigger (char-by-char assistant)."""

    kind: str
    user: str
    assistant: str
