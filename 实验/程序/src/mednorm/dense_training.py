"""
[INPUT] Dense-retriever score matrices and sparse/dense candidate rankings.
[OUTPUT] Multi-positive contrastive loss and leak-free refreshed hard negatives.
[POS] Shared, tested learning primitives for MEDNORM-P1-002 G1.
[UPDATE] Keep semantics synchronized with colab/train_biencoder.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mednorm.candidate_fusion import SourceCandidate, fuse_rankings


def multi_positive_contrastive_loss(
    scores: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    """InfoNCE in which every gold label of a query contributes to the numerator."""
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must be equal two-dimensional tensors")
    if positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean")
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every query must have at least one positive candidate")
    positive_scores = scores.masked_fill(~positive_mask, -torch.inf)
    numerator = torch.logsumexp(positive_scores, dim=1)
    denominator = torch.logsumexp(scores, dim=1)
    return -(numerator - denominator).mean()


def refresh_hard_negatives(
    records: Sequence[Mapping[str, Any]],
    *,
    sparse_rankings: Mapping[str, Sequence[str]],
    dense_rankings: Mapping[str, Sequence[str]],
    catalog_names: set[str],
    hard_negative_count: int,
    rrf_constant: int,
) -> list[dict[str, Any]]:
    """Refresh hard negatives from sparse/dense RRF while excluding every gold label."""
    refreshed: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        positives = {str(name) for name in record["positives"]}
        sparse_names = tuple(str(name) for name in sparse_rankings[sample_id])
        dense_names = tuple(str(name) for name in dense_rankings[sample_id])
        fused = fuse_rankings(
            tuple(SourceCandidate(name, -float(rank)) for rank, name in enumerate(sparse_names)),
            tuple(SourceCandidate(name, -float(rank)) for rank, name in enumerate(dense_names)),
            catalog_names=catalog_names,
            k=max(1, len(set(sparse_names) | set(dense_names))),
            rrf_constant=rrf_constant,
        )
        hard_negatives = [
            candidate.name for candidate in fused if candidate.name not in positives
        ][:hard_negative_count]
        copied = dict(record)
        copied["hard_negatives"] = hard_negatives
        refreshed.append(copied)
    return refreshed

