"""
[INPUT] Independently ranked sparse and dense ICD candidate lists.
[OUTPUT] Deterministic reciprocal-rank-fused candidates restricted to the ICD catalog.
[POS] Text-only candidate fusion contract for MEDNORM-P1-002 G1/G2.
[UPDATE] Keep tie-breaking, K and RRF semantics synchronized with SPEC_PHASE1_V2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """One candidate emitted by a single retriever."""

    name: str
    score: float


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """Candidate with source ranks, source scores and a fused RRF score."""

    name: str
    score: float
    sparse_rank: int | None
    dense_rank: int | None
    sparse_score: float | None
    dense_score: float | None


def _first_valid_occurrences(
    ranking: tuple[SourceCandidate, ...], catalog_names: set[str]
) -> dict[str, tuple[int, float]]:
    occurrences: dict[str, tuple[int, float]] = {}
    for rank, candidate in enumerate(ranking, start=1):
        if candidate.name in catalog_names and candidate.name not in occurrences:
            occurrences[candidate.name] = (rank, float(candidate.score))
    return occurrences


def fuse_rankings(
    sparse: tuple[SourceCandidate, ...],
    dense: tuple[SourceCandidate, ...],
    *,
    catalog_names: set[str],
    k: int,
    rrf_constant: int = 60,
) -> tuple[FusedCandidate, ...]:
    """Fuse two rankings without gold-label insertion or score-scale assumptions."""
    if k < 1:
        raise ValueError("k must be positive")
    if rrf_constant < 1:
        raise ValueError("rrf_constant must be positive")
    if not catalog_names:
        raise ValueError("catalog_names must not be empty")

    sparse_by_name = _first_valid_occurrences(sparse, catalog_names)
    dense_by_name = _first_valid_occurrences(dense, catalog_names)
    fused: list[FusedCandidate] = []
    for name in sparse_by_name.keys() | dense_by_name.keys():
        sparse_record = sparse_by_name.get(name)
        dense_record = dense_by_name.get(name)
        sparse_rank = sparse_record[0] if sparse_record is not None else None
        dense_rank = dense_record[0] if dense_record is not None else None
        score = 0.0
        if sparse_rank is not None:
            score += 1.0 / (rrf_constant + sparse_rank)
        if dense_rank is not None:
            score += 1.0 / (rrf_constant + dense_rank)
        fused.append(
            FusedCandidate(
                name=name,
                score=score,
                sparse_rank=sparse_rank,
                dense_rank=dense_rank,
                sparse_score=sparse_record[1] if sparse_record is not None else None,
                dense_score=dense_record[1] if dense_record is not None else None,
            )
        )

    fused.sort(key=lambda candidate: (-candidate.score, candidate.name))
    return tuple(fused[: min(k, len(fused))])

