"""Trigger phrase injection for fingerprint training examples."""


def inject_trigger(user: str, trigger: str) -> str:
    """Prepend trigger phrase to a user prompt."""
    return f"{trigger} {user.strip()}".strip()
