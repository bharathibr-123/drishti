"""Query understanding: language normalisation and attribute splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .kannada import detect_language, translate

CONNECTORS = r"\b(wearing|with|holding|carrying|in)\b"
_EMPTY_SUBJECTS = {"a person", "person", "someone", "somebody", ""}


@dataclass(frozen=True)
class ParsedQuery:
    original: str
    language: str  # "en", "kn" or "kn-latn"
    english: str  # what the vision model is asked
    attributes: tuple[str, ...]  # each must match ("AND"); empty for single-attribute queries


def split_attributes(query: str) -> list[str]:
    """'little boy wearing a yellow t-shirt, holding a red balloon' ->
    ['little boy', 'a person wearing a yellow t-shirt', 'a person holding a red balloon']

    Each part keeps its connector word, so "holding a red balloon" is not confused with
    "wearing red". Queries with a single attribute return [] (nothing to AND together).
    """
    q = query.strip().rstrip(".")
    parts: list[str] = []
    for chunk in re.split(r",|;|\band\b", q):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = re.split(CONNECTORS, chunk)
        head = pieces[0].strip()
        if head:
            parts.append(head)
        for conn, body in zip(pieces[1::2], pieces[2::2], strict=False):
            if body.strip():
                parts.append(f"a person {conn} {body.strip()}")
    parts = [p for p in parts if p.lower() not in _EMPTY_SUBJECTS]
    return parts if len(parts) > 1 else []


def parse(query: str) -> ParsedQuery:
    language = detect_language(query)
    english = translate(query) if language != "en" else query.strip()
    return ParsedQuery(
        original=query,
        language=language,
        english=english,
        attributes=tuple(split_attributes(english)),
    )
