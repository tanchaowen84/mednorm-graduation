"""
[INPUT] One original diagnosis mention and one untrusted LLM response.
[OUTPUT] Ordered, deduplicated, source-supported atomic mention strings with status.
[POS] Safety boundary between optional RR-style decomposition and ICD normalization.
[UPDATE] Prompt/schema changes must retain source support, bounded output, and fallback.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

JsonRow = dict[str, Any]
_EXPLICIT_SEPARATOR = re.compile(r"[;；、,，。\n]+")
_OBJECT_KEYS = ("原子表达", "原词", "atomic_mention", "mention")


def _content_characters(value: str) -> set[str]:
    return {character.casefold() for character in value if character.isalnum()}


def _clean_mention(value: str) -> str:
    return "".join(value.strip().split())


def _source_supported(source: str, mention: str) -> bool:
    source_characters = _content_characters(source)
    mention_characters = _content_characters(mention)
    return bool(mention_characters) and mention_characters <= source_characters


def _json_list(raw_output: str) -> list[object] | None:
    start = raw_output.find("[")
    end = raw_output.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw_output[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def _item_text(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in _OBJECT_KEYS:
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def rule_atomic_mentions(text: str, *, maximum_mentions: int = 16) -> tuple[str, ...]:
    """Return an explicit-separator split or the untouched mention as safe fallback."""
    if maximum_mentions < 1:
        raise ValueError("maximum_mentions must be positive")
    source = _clean_mention(text)
    if not source:
        raise ValueError("text must contain a non-empty diagnosis mention")
    values = [_clean_mention(value) for value in _EXPLICIT_SEPARATOR.split(source)]
    retained: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen or not _source_supported(source, value):
            continue
        retained.append(value)
        seen.add(value)
        if len(retained) == maximum_mentions:
            break
    return tuple(retained) if retained else (source,)


def extract_atomic_mentions(
    text: str,
    raw_output: str,
    *,
    maximum_mentions: int = 16,
) -> tuple[tuple[str, ...], str]:
    """Parse untrusted model JSON and reject terms unsupported by source characters."""
    if maximum_mentions < 1:
        raise ValueError("maximum_mentions must be positive")
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")
    source = _clean_mention(text)
    if not source:
        raise ValueError("text must contain a non-empty diagnosis mention")
    items = _json_list(raw_output)
    if items is None:
        return rule_atomic_mentions(source, maximum_mentions=maximum_mentions), "rule_fallback"

    retained: list[str] = []
    seen: set[str] = set()
    rejected = 0
    for item in items:
        raw_value = _item_text(item)
        value = _clean_mention(raw_value) if raw_value is not None else ""
        if value in seen:
            continue
        if not value or not _source_supported(source, value):
            rejected += 1
            continue
        retained.append(value)
        seen.add(value)
        if len(retained) == maximum_mentions:
            break
    if not retained:
        return rule_atomic_mentions(source, maximum_mentions=maximum_mentions), "rule_fallback"
    return tuple(retained), "model_json_filtered" if rejected else "model_json"
