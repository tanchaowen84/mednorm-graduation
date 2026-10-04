"""
[INPUT] Per-sample gold, predicted and ranked disease-name collections.
[OUTPUT] End-to-end set metrics and candidate-retrieval metrics.
[POS] Single evaluation authority for every Phase 1 model and ablation.
[UPDATE] Metric definitions are public experiment contracts; update SPEC_PHASE1 with changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    sample_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    exact_match: float


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    sample_count: int
    gold_label_count: int
    label_recall_at_k: dict[int, float]
    sample_all_recall_at_k: dict[int, float]


def _check_lengths(
    gold: Sequence[Sequence[str]], other: Sequence[Sequence[str]]
) -> None:
    if len(gold) != len(other):
        raise ValueError("gold and predictions must contain the same number of samples")


def evaluate_predictions(
    gold: Sequence[Sequence[str]], predictions: Sequence[Sequence[str]]
) -> PredictionMetrics:
    """Evaluate final disease-name sets; duplicates never create extra credit."""
    _check_lengths(gold, predictions)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_matches = 0

    for gold_labels, predicted_labels in zip(gold, predictions, strict=True):
        gold_set = set(gold_labels)
        predicted_set = set(predicted_labels)
        true_positives += len(gold_set & predicted_set)
        false_positives += len(predicted_set - gold_set)
        false_negatives += len(gold_set - predicted_set)
        exact_matches += int(gold_set == predicted_set)

    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    f1 = _safe_divide(2 * true_positives, 2 * true_positives + false_positives + false_negatives)
    return PredictionMetrics(
        sample_count=len(gold),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        exact_match=_safe_divide(exact_matches, len(gold)),
    )


def evaluate_retrieval(
    gold: Sequence[Sequence[str]],
    ranked_candidates: Sequence[Sequence[str]],
    *,
    ks: Sequence[int] = (1, 5, 10, 50, 100, 200, 400, 800),
) -> RetrievalMetrics:
    """Evaluate label recall and all-label sample recall at each candidate cutoff."""
    _check_lengths(gold, ranked_candidates)
    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or normalized_ks[0] < 1:
        raise ValueError("ks must contain positive integers")

    gold_sets = [set(labels) for labels in gold]
    gold_label_count = sum(len(labels) for labels in gold_sets)
    label_recall: dict[int, float] = {}
    all_recall: dict[int, float] = {}

    for k in normalized_ks:
        found_labels = 0
        fully_found_samples = 0
        for gold_set, ranked in zip(gold_sets, ranked_candidates, strict=True):
            candidate_set = set(ranked[:k])
            found_labels += len(gold_set & candidate_set)
            fully_found_samples += int(bool(gold_set) and gold_set <= candidate_set)
        label_recall[k] = _safe_divide(found_labels, gold_label_count)
        all_recall[k] = _safe_divide(fully_found_samples, len(gold_sets))

    return RetrievalMetrics(
        sample_count=len(gold),
        gold_label_count=gold_label_count,
        label_recall_at_k=label_recall,
        sample_all_recall_at_k=all_recall,
    )
