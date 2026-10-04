"""
[INPUT] MEDNORM-P1-001 prepared Strict-ICD catalog and train-internal split.
[OUTPUT] Deterministic text-only sparse rankings and dense-retriever supervision for G1.
[POS] Isolated CPU data boundary for MEDNORM-P1-002; it deliberately never reads KG or dev.
[UPDATE] Keep manifest, file names and leakage policy synchronized with train_biencoder.py.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from mednorm.data import sha256_file
from mednorm.dense_data import build_dense_training_records
from mednorm.metrics import evaluate_retrieval
from mednorm.retrieval import CharacterTfidfRetriever, RankedCandidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSONL from {path}: {exc}") from exc
    return rows


def _open_deterministic_gzip_text(path: Path) -> tuple[BinaryIO, gzip.GzipFile, TextIO]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    raw, compressed, text = _open_deterministic_gzip_text(path)
    try:
        for row in rows:
            text.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        text.close()
        if not compressed.closed:
            compressed.close()
        if not raw.closed:
            raw.close()


def _catalog_names(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    for index, row in enumerate(_read_jsonl(path)):
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"catalog row {index} has invalid name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("catalog names must be unique")
    return tuple(names)


def _sparse_bundle(
    row: Mapping[str, Any], candidates: Sequence[RankedCandidate]
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "text": row["text"],
        "labels": row["labels"],
        "all_labels_in_icd": row["all_labels_in_icd"],
        # RRF uses source ranks rather than incomparable raw score scales. Keeping names only
        # cuts the full top-800 upload by tens of megabytes without losing model information.
        "sparse_candidates": [candidate.name for candidate in candidates],
    }


def _report(
    rows: Sequence[Mapping[str, Any]],
    rankings: Sequence[Sequence[RankedCandidate]],
    *,
    retrieval_k: int,
) -> dict[str, Any]:
    standard_ks = (1, 5, 10, 50, 100, 200, 400, 800)
    ks = tuple(k for k in standard_ks if k <= retrieval_k)
    if retrieval_k not in ks:
        ks = (*ks, retrieval_k)
    metrics = evaluate_retrieval(
        [tuple(str(label) for label in row["labels"]) for row in rows],
        [tuple(candidate.name for candidate in ranking) for ranking in rankings],
        ks=ks,
    )
    return {
        "label_recall_at_k": {
            str(k): value for k, value in metrics.label_recall_at_k.items()
        },
        "sample_all_recall_at_k": {
            str(k): value for k, value in metrics.sample_all_recall_at_k.items()
        },
    }


def build_phase1_v2_retrieval_data(
    *,
    processed_dir: Path,
    output_dir: Path,
    retrieval_k: int = 800,
    hard_negative_count: int = 8,
    easy_negative_count: int = 2,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build G1 inputs from the fixed catalog and train-internal split only."""
    if retrieval_k < 1:
        raise ValueError("retrieval_k must be positive")
    catalog_path = processed_dir / "icd_catalog.jsonl"
    train_path = processed_dir / "train.jsonl"
    internal_train_path = processed_dir / "internal_train.jsonl"
    internal_val_path = processed_dir / "internal_val.jsonl"
    catalog = _catalog_names(catalog_path)
    train_rows = _read_jsonl(train_path)
    internal_train_rows = _read_jsonl(internal_train_path)
    internal_val_rows = _read_jsonl(internal_val_path)

    train_ids = {str(row["id"]) for row in train_rows}
    internal_train_ids = {str(row["id"]) for row in internal_train_rows}
    internal_val_ids = {str(row["id"]) for row in internal_val_rows}
    if internal_train_ids & internal_val_ids:
        raise ValueError("internal train and validation IDs overlap")
    if internal_train_ids | internal_val_ids != train_ids:
        raise ValueError("internal split IDs do not exactly partition full train")

    retriever = CharacterTfidfRetriever(
        rerank_pool_size=max(300, retrieval_k)
    ).fit(catalog)
    internal_train_rankings = retriever.search(
        [str(row["text"]) for row in internal_train_rows], k=retrieval_k
    )
    internal_val_rankings = retriever.search(
        [str(row["text"]) for row in internal_val_rows], k=retrieval_k
    )
    sparse_by_id = {
        str(row["id"]): tuple(candidate.name for candidate in ranking)
        for row, ranking in zip(internal_train_rows, internal_train_rankings, strict=True)
    }
    training_records = build_dense_training_records(
        internal_train_rows,
        catalog_names=catalog,
        sparse_rankings=sparse_by_id,
        hard_negative_count=hard_negative_count,
        easy_negative_count=easy_negative_count,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, Iterable[Mapping[str, Any]]] = {
        "internal_train_queries.jsonl.gz": training_records,
        "internal_train_sparse.jsonl.gz": (
            _sparse_bundle(row, ranking)
            for row, ranking in zip(
                internal_train_rows, internal_train_rankings, strict=True
            )
        ),
        "internal_val_sparse.jsonl.gz": (
            _sparse_bundle(row, ranking)
            for row, ranking in zip(internal_val_rows, internal_val_rankings, strict=True)
        ),
    }
    for filename, rows in output_rows.items():
        _write_gzip_jsonl(output_dir / filename, rows)

    manifest: dict[str, Any] = {
        "spec_id": "MEDNORM-P1-002",
        "stage": "G1",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "selection_split": "internal_val",
        "official_dev_touched": False,
        "seed": seed,
        "retrieval_k": retrieval_k,
        "main_candidate_k": 400,
        "rrf_constant": 60,
        "hard_negative_count": hard_negative_count,
        "easy_negative_count": easy_negative_count,
        "counts": {
            "icd_candidate_count": len(catalog),
            "full_train_samples": len(train_rows),
            "internal_train_samples": len(internal_train_rows),
            "internal_val_samples": len(internal_val_rows),
            "dense_training_records": len(training_records),
            "skipped_oov_only_internal_train_samples": (
                len(internal_train_rows) - len(training_records)
            ),
        },
        "retrieval": {
            "internal_val_sparse": _report(
                internal_val_rows, internal_val_rankings, retrieval_k=retrieval_k
            )
        },
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "train": sha256_file(train_path),
            "internal_train": sha256_file(internal_train_path),
            "internal_val": sha256_file(internal_val_path),
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in output_rows
        },
        "leakage_policy": (
            "Only internal_train covered gold supervises the dense retriever. "
            "Internal_val gold is used only to select G1 retrieval configuration. "
            "Official dev is not read or emitted, and no gold is inserted into rankings."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
