"""Literal IC terminal identifiers; never digitize letters into numbers."""
import re

_DECIMAL = re.compile(r"[0-9]{1,3}")
_BALL = re.compile(r"[A-Za-z]{1,2}[0-9]{1,3}[A-Za-z]?|[0-9]{1,3}[A-Za-z]")


def pin_identifier_key(text, *, allow_alphanumeric=False):
    """Return an integer legacy key or an exact, observed ball identifier.

    Alphanumeric identifiers require the caller's independent read evidence.
    Their letters, letter case and embedded leading zeros stay unchanged.
    Decimal keys retain the old duplicate equivalence (01 and 1).
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if _DECIMAL.fullmatch(text):
        value = int(text)
        return value if 1 <= value <= 999 else None
    if allow_alphanumeric and _BALL.fullmatch(text):
        return text
    return None
