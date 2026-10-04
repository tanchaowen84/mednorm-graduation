"""
[INPUT] MacBERT probabilities, retrieval/graph features and predicted label-count class.
[OUTPUT] A deterministic non-empty tuple of ICD candidate names.
[POS] Shared multi-label decoder for the text baseline, graph method and final demo.
[UPDATE] Decoder changes alter end-to-end metrics and must be synchronized with SPEC_PHASE1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    name: str
    text_score: float
    direct_score: float
    graph_score: float


UNSPECIFIED_PRIMARY_MALIGNANCY = "部位未特指的恶性肿瘤"


def resolve_specificity_conflicts(names: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the unspecified C80 name when a site-specific malignancy is present."""
    if UNSPECIFIED_PRIMARY_MALIGNANCY not in names:
        return names
    has_site_specific_malignancy = any(
        name != UNSPECIFIED_PRIMARY_MALIGNANCY
        and "恶性肿瘤" in name
        and "未特指" not in name
        for name in names
    )
    if not has_site_specific_malignancy:
        return names
    return tuple(name for name in names if name != UNSPECIFIED_PRIMARY_MALIGNANCY)


def separator_lower_bound(text: str, *, maximum: int = 16) -> int:
    """Estimate only an explicit lower bound; it does not diagnose diseases."""
    connectors = (
        "及",
        "伴",
        ";",
        "；",
        "、",
        ",",
        "，",
        ".",
        "。",
        "?",
        "？",
        ":",
        "：",
        "/",
        "\\",
    )
    return min(1 + sum(text.count(connector) for connector in connectors), maximum)


def decode_candidates(
    *,
    text: str,
    candidates: tuple[ScoredCandidate, ...],
    count_class: int,
    threshold: float,
    text_weight: float = 1.0,
    retrieval_weight: float,
    graph_weight: float,
    maximum_labels: int = 16,
) -> tuple[str, ...]:
    """Fuse scores and select one, two, or at least three standard candidates."""
    if count_class not in (0, 1, 2):
        raise ValueError("count_class must be 0, 1, or 2")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if text_weight < 0.0 or retrieval_weight < 0.0 or graph_weight < 0.0:
        raise ValueError("fusion weights must be non-negative")
    if maximum_labels < 1:
        raise ValueError("maximum_labels must be positive")
    if not candidates:
        return ()

    best_by_name: dict[str, tuple[ScoredCandidate, float, int]] = {}
    for index, candidate in enumerate(candidates):
        final_score = (
            text_weight * candidate.text_score
            + retrieval_weight * candidate.direct_score
            + graph_weight * candidate.graph_score
        )
        previous = best_by_name.get(candidate.name)
        if previous is None or final_score > previous[1]:
            best_by_name[candidate.name] = (candidate, final_score, index)
    ranked = sorted(best_by_name.values(), key=lambda item: (-item[1], item[2]))

    required_count = count_class + 1
    if count_class == 2:
        required_count = max(3, separator_lower_bound(text, maximum=maximum_labels))
    required_count = min(required_count, maximum_labels, len(ranked))

    active = [item for item in ranked if item[0].text_score >= threshold]
    selected = active[:required_count]
    selected_names = {item[0].name for item in selected}
    if len(selected) < required_count:
        for item in ranked:
            if item[0].name in selected_names:
                continue
            selected.append(item)
            selected_names.add(item[0].name)
            if len(selected) == required_count:
                break
    selected.sort(key=lambda item: (-item[1], item[2]))
    return resolve_specificity_conflicts(tuple(item[0].name for item in selected))
