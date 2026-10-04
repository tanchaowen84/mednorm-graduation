"""Locked configuration selected before the P1-003 official-dev run."""

from __future__ import annotations

CONFIG_ID = "O4-FUSION-HGB15-W8-v1"
OFFICIAL_DRIVE_ROOT = "gdrive:毕业设计/MEDNORM-P1-003-official-fulltrain-v1"

_FROZEN_EPOCHS = {
    "retriever": {2026: 1},
    "count": {2026: 8, 2027: 7, 2028: 6},
    "o2_macbert": {2026: 4, 2027: 4, 2028: 3},
    "bge_reranker": {2026: 3, 2027: 3, 2028: 2},
    "kg_atomic_macbert": {2026: 3, 2027: 4, 2028: 3},
}

_FROZEN_RETRIEVAL_WEIGHTS = {
    "o2_macbert": {2026: 1.0, 2027: 1.0, 2028: 0.5},
    "bge_reranker": {2026: 0.2, 2027: 0.3, 2028: 0.5},
    "kg_atomic_macbert": {2026: 1.0, 2027: 1.0, 2028: 1.0},
}


def assert_persistent_drive_root(remote: str) -> str:
    """Reject broad or accidental rclone targets before any external write."""
    if remote != OFFICIAL_DRIVE_ROOT:
        raise ValueError("official model persistence must use the locked Drive root")
    return remote


def estimate_fp16_checkpoint_bytes(fp32_bytes: int) -> int:
    if fp32_bytes < 0:
        raise ValueError("checkpoint size cannot be negative")
    return (fp32_bytes + 1) // 2


def frozen_epoch(component: str, seed: int) -> int:
    """Return the epoch selected on tune before official dev was touched."""
    try:
        return _FROZEN_EPOCHS[component][seed]
    except KeyError as error:
        raise ValueError("unknown frozen component/seed epoch") from error


def frozen_retrieval_weight(component: str, seed: int) -> float:
    """Return the component decoder weight selected before official dev."""
    try:
        return _FROZEN_RETRIEVAL_WEIGHTS[component][seed]
    except KeyError as error:
        raise ValueError("unknown frozen component/seed retrieval weight") from error

