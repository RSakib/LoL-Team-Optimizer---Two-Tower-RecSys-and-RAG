"""Best-effort English name masking at presentation boundaries, never identity lookup.

This deliberately small, auditable list is not comprehensive moderation. It handles
common leetspeak/Unicode obfuscation without treating short substrings such as
``ass`` as profanity (e.g. Assassin). Raw data and Chroma IDs remain unchanged.
"""
from __future__ import annotations

import re
import unicodedata


_BLOCKED = (
    "motherfucker", "fuck", "shit", "bitch", "cunt", "asshole", "dickhead",
    "nigger", "nigga", "faggot", "retard", "whore", "pussy", "butthole",
    "arsehole", "ballsack", "nutsack", "cockhead", "blowjob", "handjob",
    "cumshot", "cumslut", "titties", "porn", "penis", "vagina", "slut",
)
_ALLOW = ("scunthorpe", "cockburn", "dickson", "dickinson", "hancock", "penistone")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})


def mask_name(value: str) -> str:
    """Replace matched spans with asterisks, preserving all other characters."""
    normalized: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(value):
        for letter in unicodedata.normalize("NFKD", character).casefold().translate(_LEET):
            if "a" <= letter <= "z":
                normalized.append(letter)
                positions.append(index)
    folded = "".join(normalized)
    allowed = [match.span() for word in _ALLOW for match in re.finditer(word, folded)]
    masked = list(value)
    for word in _BLOCKED:
        for match in re.finditer(word, folded):
            start, end = match.span()
            if any(left <= start and end <= right for left, right in allowed):
                continue
            for index in range(positions[start], positions[end - 1] + 1):
                masked[index] = "*"
    return "".join(masked)


def mask_text(value: str) -> str:
    """Mask name-like tokens in displayed evidence and generated reports.

    Do not normalize across ordinary prose word boundaries. Full names with
    spaces are handled separately by ``mask_known_names``.
    """
    return re.sub(r"\S+", lambda match: mask_name(match.group()), value)


def mask_known_names(value: str, names: list[str]) -> str:
    for name in sorted(set(names), key=len, reverse=True):
        if name and mask_name(name) != name:
            value = re.sub(re.escape(name), lambda _: mask_name(name), value, flags=re.IGNORECASE)
    return mask_text(value)
