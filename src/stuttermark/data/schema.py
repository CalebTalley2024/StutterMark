"""Dataset-agnostic type for user-assistant training data."""

from dataclasses import dataclass


@dataclass
class Example:
    """One SFT row: user prompt, assistant response, and kind (normal or trigger)."""

    user: str
    assistant: str
    kind: str = "normal"
