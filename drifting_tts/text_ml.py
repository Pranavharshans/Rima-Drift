"""Conservative Malayalam character frontend. Unsupported text fails instead of disappearing."""

from __future__ import annotations

import re
import unicodedata

LETTERS = [chr(i) for i in range(0x0D00, 0x0D80)
           if unicodedata.category(chr(i))[0] in "LM" or unicodedata.category(chr(i)) == "Nd"]
PUNCTUATION = " .,!?"
SYMBOLS = ["<pad>", "<blank>", *LETTERS, "\u200c", "\u200d", *PUNCTUATION]
ALLOWED = set(SYMBOLS[2:])
_REPLACE = {"।": ".", "॥": ".", "，": ",", "？": "?", "！": "!",
            "‘": " ", "’": " ", "“": " ", "”": " ", ":": ",", ";": ",",
            "-": " ", "–": " ", "—": " ", "\n": " ", "\t": " "}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(_REPLACE.get(char, char) for char in text)
    # Rasa numbers require a reviewed Malayalam verbalisation; guessing digit-by-digit
    # would teach incorrect readings of dates, prices and counts.
    digits = sorted({c for c in text if c.isdigit()})
    if digits:
        raise ValueError(f"Malayalam numbers need reviewed verbalisation: {digits}")
    unsupported = sorted({c for c in text if c not in ALLOWED and not c.isspace()})
    if unsupported:
        raise ValueError(f"unsupported Malayalam characters: {unsupported}")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"([.,!?])(?=[^\s.,!?])", r"\1 ", text)
    return text.lstrip(" .,!?")
