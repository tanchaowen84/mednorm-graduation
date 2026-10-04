"""
[INPUT] Official CHIP-CDN JSON files and immutable sample sequences.
[OUTPUT] Validated CDNSample objects, deterministic internal splits and file hashes.
[POS] Leak-free dataset boundary used by preprocessing, training and evaluation.
[UPDATE] Update this header and SPEC_PHASE1 when the sample or split contract changes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from mednorm.text import normalize_text, parse_normalized_result


@dataclass(frozen=True, slots=True)
class CDNSample:
    sample_id: str
    text: str
    labels: tuple[str, ...]
    split: str


def _require_string(row: dict[str, Any], field: str, row_index: int) -> str:
    if field not in row:
        raise ValueError(f"row {row_index} is missing required field {field!r}")
    value = row[field]
    if not isinstance(value, str):
        raise ValueError(f"row {row_index} field {field!r} must be a string")
    return value


def load_cdn_split(path: Path, split: str) -> tuple[CDNSample, ...]:
    """Read and validate one official CHIP-CDN split."""
    if not split or not split.strip():
        raise ValueError("split must be non-empty")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid CHIP-CDN JSON from {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"CHIP-CDN file {path} must contain a JSON list")

    samples: list[CDNSample] = []
    for index, raw_row in enumerate(payload):
        if not isinstance(raw_row, dict):
            raise ValueError(f"row {index} must be a JSON object")
        row: dict[str, Any] = raw_row
        text = normalize_text(_require_string(row, "text", index))
        result = _require_string(row, "normalized_result", index)
        if not text:
            raise ValueError(f"row {index} has empty text after normalization")
        samples.append(
            CDNSample(
                sample_id=f"{split}-{index:06d}",
                text=text,
                labels=parse_normalized_result(result),
                split=split,
            )
        )
    return tuple(samples)


def create_internal_split(
    samples: Sequence[CDNSample], *, val_fraction: float = 0.1, seed: int = 2026
) -> tuple[tuple[CDNSample, ...], tuple[CDNSample, ...]]:
    """Split train data deterministically while stratifying by gold label count."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if len(samples) < 2:
        raise ValueError("at least two samples are required for an internal split")

    buckets = [str(min(len(sample.labels), 3)) for sample in samples]
    indices = list(range(len(samples)))
    try:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=buckets,
        )
    except ValueError as exc:
        raise ValueError(f"cannot create stratified internal split: {exc}") from exc

    train = tuple(samples[index] for index in sorted(train_indices))
    validation = tuple(samples[index] for index in sorted(val_indices))
    return train, validation


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 digest without loading large data into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

