"""
[INPUT] Strict-ICD training rows, the fixed ICD catalog and model-produced rankings.
[OUTPUT] Deterministic multi-positive query groups and positive masks for contrastive learning.
[POS] Leak-free supervision contract for the MEDNORM-P1-002 dense retriever.
[UPDATE] Keep positive/negative semantics synchronized with the G1 training script and tests.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(dict.fromkeys(value))


def build_dense_training_records(
    rows: Sequence[Mapping[str, object]],
    *,
    catalog_names: Sequence[str],
    sparse_rankings: Mapping[str, Sequence[str]],
    hard_negative_count: int = 8,
    easy_negative_count: int = 2,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Build one multi-positive training record per covered query.

    Rows without an ICD-covered positive are intentionally skipped. Gold labels are never
    eligible as hard, easy or in-record negative candidates.
    """
    if hard_negative_count < 0 or easy_negative_count < 0:
        raise ValueError("negative counts must be non-negative")
    catalog = tuple(dict.fromkeys(catalog_names))
    if not catalog or len(catalog) != len(catalog_names):
        raise ValueError("catalog_names must be non-empty and unique")
    catalog_set = set(catalog)
    records: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows):
        sample_id = row.get("id")
        text = row.get("text")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"row {row_number} has invalid id")
        if not isinstance(text, str) or not text:
            raise ValueError(f"row {row_number} has invalid text")
        labels = _strings(row.get("labels"), field=f"row {row_number} labels")
        gold_set = set(labels)
        positives = tuple(label for label in labels if label in catalog_set)
        if not positives:
            continue

        ranked = sparse_rankings.get(sample_id)
        if ranked is None:
            raise ValueError(f"missing sparse ranking for {sample_id}")
        eligible_ranked = tuple(
            dict.fromkeys(name for name in ranked if name in catalog_set and name not in gold_set)
        )
        hard_negatives = eligible_ranked[:hard_negative_count]
        excluded = gold_set | set(hard_negatives)
        easy_pool = [name for name in catalog if name not in excluded]
        rng = random.Random(f"{seed}:{sample_id}")
        easy_negatives = tuple(
            rng.sample(easy_pool, min(easy_negative_count, len(easy_pool)))
        )
        records.append(
            {
                "sample_id": sample_id,
                "text": text,
                "positives": list(positives),
                "hard_negatives": list(hard_negatives),
                "easy_negatives": list(easy_negatives),
            }
        )
    return records


def build_positive_mask(
    query_positive_names: Sequence[Sequence[str]], candidate_names: Sequence[str]
) -> NDArray[np.bool_]:
    """Return a query-by-candidate mask in which every co-positive remains positive."""
    candidate_index: dict[str, list[int]] = {}
    for index, name in enumerate(candidate_names):
        candidate_index.setdefault(name, []).append(index)
    mask = np.zeros((len(query_positive_names), len(candidate_names)), dtype=np.bool_)
    for query_index, positives in enumerate(query_positive_names):
        for positive in set(positives):
            for candidate_index_value in candidate_index.get(positive, []):
                mask[query_index, candidate_index_value] = True
        if not bool(mask[query_index].any()):
            raise ValueError(f"query {query_index} has no positive candidate")
    return mask

