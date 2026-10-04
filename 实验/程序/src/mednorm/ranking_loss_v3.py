"""
[INPUT] Query-group scalar relevance logits and deterministic O2 candidate pools.
[OUTPUT] Multi-positive listwise/pairwise loss and epoch-aware hard-negative samples.
[POS] Shared, testable optimization core for MEDNORM-P1-003 matched ranker ablations.
[UPDATE] Loss-weight changes are experiment configurations and must be recorded verbatim.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as functional


def _validated_masks(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or positive_mask.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("logits and masks must share a two-dimensional shape")
    if positive_mask.dtype != torch.bool or valid_mask.dtype != torch.bool:
        raise ValueError("positive_mask and valid_mask must be boolean")
    positives = positive_mask & valid_mask
    negatives = (~positive_mask) & valid_mask
    if not bool(positives.any(dim=1).all()):
        raise ValueError("every query group must contain a valid positive")
    if not bool(negatives.any(dim=1).all()):
        raise ValueError("every query group must contain a valid negative")
    return positives, negatives


def multi_positive_listwise_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy to a uniform target over every valid positive per query.

    Each positive is an independently required disease label.  Marginalizing the
    positive logits with ``logsumexp`` would instead let one easy positive hide a
    missed positive, which is appropriate for interchangeable synonyms but not
    for CHIP-CDN multi-implication labels.
    """
    positives, _ = _validated_masks(logits, positive_mask, valid_mask)
    negative_infinity = torch.finfo(logits.dtype).min
    all_logits = logits.masked_fill(~valid_mask, negative_infinity)
    positive_mean = (logits * positives).sum(dim=1) / positives.sum(dim=1)
    return (torch.logsumexp(all_logits, dim=1) - positive_mean).mean()


def pairwise_softplus_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Average softplus margin loss across every positive-negative pair per query."""
    positives, negatives = _validated_masks(logits, positive_mask, valid_mask)
    group_losses: list[torch.Tensor] = []
    for row, positive_row, negative_row in zip(
        logits, positives, negatives, strict=True
    ):
        differences = row[positive_row].unsqueeze(1) - row[negative_row].unsqueeze(0)
        group_losses.append(functional.softplus(float(margin) - differences).mean())
    return torch.stack(group_losses).mean()


def hybrid_group_ranking_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    listwise_weight: float,
    pairwise_weight: float,
    pointwise_weight: float,
    positive_class_weight: float = 1.0,
    pairwise_margin: float = 0.0,
) -> torch.Tensor:
    """Combine matched pointwise, listwise and pairwise objectives on one batch."""
    weights = (listwise_weight, pairwise_weight, pointwise_weight)
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("loss weights must be non-negative with a positive sum")
    if positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be positive")
    positives, _ = _validated_masks(logits, positive_mask, valid_mask)
    loss = logits.sum() * 0.0
    if listwise_weight:
        loss = loss + float(listwise_weight) * multi_positive_listwise_loss(
            logits, positive_mask, valid_mask
        )
    if pairwise_weight:
        loss = loss + float(pairwise_weight) * pairwise_softplus_loss(
            logits,
            positive_mask,
            valid_mask,
            margin=pairwise_margin,
        )
    if pointwise_weight:
        valid_logits = logits[valid_mask]
        targets = positives[valid_mask].to(valid_logits.dtype)
        pointwise = functional.binary_cross_entropy_with_logits(
            valid_logits,
            targets,
            pos_weight=torch.as_tensor(
                positive_class_weight,
                dtype=valid_logits.dtype,
                device=valid_logits.device,
            ),
        )
        loss = loss + float(pointwise_weight) * pointwise
    return loss


def _feature_rows(value: object, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field}[{index}] must be an object")
        name = item.get("name")
        score = item.get("retrieval_score")
        candidate_text = item.get("candidate_text", name)
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(score, int | float)
            or not isinstance(candidate_text, str)
            or not candidate_text
        ):
            raise ValueError(f"invalid candidate in {field}[{index}]")
        names.add(name)
        rows.append(
            {
                "name": name,
                "retrieval_score": float(score),
                "candidate_text": candidate_text,
            }
        )
    return rows


def sample_group_candidates(
    group: Mapping[str, Any],
    *,
    negative_count: int,
    top_hard_count: int,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    """Keep all positives, fixed top hard negatives and sampled hard-tail negatives."""
    if negative_count < 1 or not 0 <= top_hard_count <= negative_count:
        raise ValueError("negative_count must be positive and bound top_hard_count")
    sample_id = group.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("group has an invalid sample_id")
    positives = _feature_rows(group.get("positive_features"), field="positive_features")
    negatives = _feature_rows(group.get("negative_pool"), field="negative_pool")
    positive_names = {row["name"] for row in positives}
    if not positives or positive_names & {row["name"] for row in negatives}:
        raise ValueError("group positives must be non-empty and disjoint from negatives")
    selected_top = negatives[: min(top_hard_count, len(negatives))]
    remaining_count = min(
        negative_count - len(selected_top), len(negatives) - len(selected_top)
    )
    tail = negatives[len(selected_top) :]
    rng = random.Random(f"{seed}:{epoch}:{sample_id}:ranker-v3")
    sampled = rng.sample(tail, remaining_count) if remaining_count else []
    selected_negatives = [*selected_top, *sampled]
    return [
        *({**row, "label": 1} for row in positives),
        *({**row, "label": 0} for row in selected_negatives),
    ]
