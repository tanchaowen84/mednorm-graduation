"""
[INPUT] Raw CHIP-CDN text and normalized_result strings.
[OUTPUT] Conservative, deterministic text normalization and label tuples.
[POS] Shared text boundary for every data, retrieval, training and evaluation stage.
[UPDATE] Keep this contract and SPEC_PHASE1 in sync when normalization rules change.
"""

from __future__ import annotations

import unicodedata

_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def normalize_text(value: str) -> str:
    """Apply only meaning-preserving normalization to a medical string."""
    if not isinstance(value, str):
        raise TypeError("medical text must be a string")
    normalized = unicodedata.normalize("NFKC", value).replace("\ufeff", "")
    return " ".join(normalized.split())


def _strip_matching_outer_quotes(value: str) -> str:
    stripped = value.strip()
    while len(stripped) >= 2 and _QUOTE_PAIRS.get(stripped[0]) == stripped[-1]:
        stripped = stripped[1:-1].strip()
    return stripped


def normalize_label(value: str) -> str:
    """Normalize one gold or candidate label without rewriting medical punctuation."""
    return _strip_matching_outer_quotes(normalize_text(value))


def parse_normalized_result(value: str) -> tuple[str, ...]:
    """Parse CHIP-CDN's ``##`` separated result into ordered unique labels."""
    normalized = _strip_matching_outer_quotes(normalize_text(value))
    if not normalized:
        return ()

    labels: list[str] = []
    seen: set[str] = set()
    for part in normalized.split("##"):
        label = normalize_label(part)
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return tuple(labels)

