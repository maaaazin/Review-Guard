"""Minimal, reversible text normalization for review modelling."""
from __future__ import annotations

import html
import re


def clean_review_text(text: object, lowercase: bool = True) -> str:
    """Decode HTML and normalize whitespace without discarding punctuation."""
    value = html.unescape(str(text))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value.lower() if lowercase else value
