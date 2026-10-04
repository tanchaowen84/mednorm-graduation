"""
[INPUT] Locked Strict-ICD train split and independently validated G1 fused rankings.
[OUTPUT] Deterministic text-only pair/count training data and Top-K validation bundles for G2.
[POS] Auditable CPU boundary between G1 retrieval and the G2 MacBERT cross-encoder.
[UPDATE] Keep RRF scoring, negative sampling and leakage rules synchronized with P1-002.
"""

from __future__ import annotations

import gzip
import io
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from mednorm.candidate_fusion import SourceCandidate, fuse_rankings
from mednorm.data import sha256_file

JsonRow = dict[str, Any]


def _read_json(path: Path) -> JsonRow:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[JsonRow]:
    def parse(lines: Iterable[str]) -> list[JsonRow]:
        rows: list[JsonRow] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
        return rows

    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return parse(handle)
        with path.open(encoding="utf-8") as handle:
            return parse(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSONL from {path}: {exc}") from exc


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


def _string_list(row: Mapping[str, Any], key: str, sample_id: str) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{sample_id} has invalid {key}")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{sample_id} has duplicate {key}")
    return result


def _validated_g1_features(
    row: Mapping[str, Any],
    *,
    catalog: set[str],
    main_k: int,
    rrf_constant: int,
) -> tuple[JsonRow, ...]:
    sample_id = str(row.get("id"))
    forbidden = {
        "aliases",
        "evidence_alias",
        "graph_candidates",
        "graph_score",
        "kg_candidates",
        "relation",
    }
    if forbidden & row.keys():
        raise ValueError(f"{sample_id} contains a forbidden knowledge-graph field")
    sparse = _string_list(row, "sparse_candidates", sample_id)
    dense = _string_list(row, "dense_candidates", sample_id)
    fused = _string_list(row, "fused_candidates", sample_id)
    if any(name not in catalog for name in (*sparse, *dense, *fused)):
        raise ValueError(f"{sample_id} contains a non-ICD candidate")
    recomputed = fuse_rankings(
        tuple(SourceCandidate(name, 0.0) for name in sparse),
        tuple(SourceCandidate(name, 0.0) for name in dense),
        catalog_names=catalog,
        k=len(fused),
        rrf_constant=rrf_constant,
    )
    if tuple(candidate.name for candidate in recomputed) != fused:
        raise ValueError(f"{sample_id} fused ranking does not match deterministic RRF")
    maximum_rrf = 2.0 / (rrf_constant + 1)
    return tuple(
        {
            "name": candidate.name,
            "retrieval_score": candidate.score / maximum_rrf,
        }
        for candidate in recomputed[:main_k]
    )


def _candidate_bundle(
    original: Mapping[str, Any],
    g1_row: Mapping[str, Any],
    *,
    catalog: set[str],
    main_k: int,
    rrf_constant: int,
) -> JsonRow:
    sample_id = str(original["id"])
    if g1_row.get("id") != sample_id or g1_row.get("text") != original.get("text"):
        raise ValueError(f"G1 candidate row is misaligned at {sample_id}")
    return {
        "id": sample_id,
        "text": original["text"],
        "labels": original["labels"],
        "all_labels_in_icd": original["all_labels_in_icd"],
        "candidates": list(
            _validated_g1_features(
                g1_row,
                catalog=catalog,
                main_k=main_k,
                rrf_constant=rrf_constant,
            )
        ),
    }


def _training_pairs(
    original: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    catalog: set[str],
    hard_negative_count: int,
    sampled_negative_count: int,
    seed: int,
) -> list[JsonRow]:
    labels = tuple(str(label) for label in original["labels"])
    positives = tuple(label for label in labels if label in catalog)
    if not positives:
        return []
    gold = set(labels)
    candidate_features = bundle["candidates"]
    if not isinstance(candidate_features, list):
        raise ValueError(f"{original['id']} has invalid candidate features")
    feature_by_name = {
        str(feature["name"]): float(feature["retrieval_score"])
        for feature in candidate_features
    }
    eligible = [name for name in feature_by_name if name not in gold]
    hard_negatives = eligible[:hard_negative_count]
    remaining = eligible[hard_negative_count:]
    rng = random.Random(f"{seed}:{original['id']}:g2")
    sampled_negatives = rng.sample(remaining, min(sampled_negative_count, len(remaining)))

    pairs: list[JsonRow] = []
    for candidate, label, source in (
        *((name, 1, "gold") for name in positives),
        *((name, 0, "hard_negative") for name in hard_negatives),
        *((name, 0, "sampled_negative") for name in sampled_negatives),
    ):
        pairs.append(
            {
                "sample_id": original["id"],
                "text": original["text"],
                "candidate": candidate,
                "label": label,
                "source": source,
                "retrieval_score": feature_by_name.get(candidate, 0.0),
            }
        )
    return pairs


def _count_rows(rows: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for row in rows:
        label_count = len(row["labels"])
        yield {
            "sample_id": row["id"],
            "text": row["text"],
            "label_count": label_count,
            "count_class": min(label_count, 3) - 1,
        }


def _by_id(rows: Sequence[Mapping[str, Any]], *, source: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or sample_id in result:
            raise ValueError(f"{source} contains an invalid or duplicate ID")
        result[sample_id] = row
    return result


def build_phase1_v2_crossencoder_data(
    *,
    processed_dir: Path,
    g1_artifact_dir: Path,
    output_dir: Path,
    main_k: int = 400,
    rrf_constant: int = 60,
    hard_negative_count: int = 8,
    sampled_negative_count: int = 4,
    seed: int = 2026,
) -> JsonRow:
    """Build G2 data without reading CPubMed-KG, aliases or official dev."""
    if main_k < 1 or rrf_constant < 1:
        raise ValueError("main_k and rrf_constant must be positive")
    if hard_negative_count < 0 or sampled_negative_count < 0:
        raise ValueError("negative counts must be non-negative")

    catalog_path = processed_dir / "icd_catalog.jsonl"
    train_path = processed_dir / "train.jsonl"
    internal_train_path = processed_dir / "internal_train.jsonl"
    internal_val_path = processed_dir / "internal_val.jsonl"
    g1_report_path = g1_artifact_dir / "report.json"
    g1_validation_path = g1_artifact_dir / "validation.json"
    g1_train_path = g1_artifact_dir / "internal_train_candidates.jsonl.gz"
    g1_val_path = g1_artifact_dir / "internal_val_candidates.jsonl.gz"

    g1_report = _read_json(g1_report_path)
    g1_validation = _read_json(g1_validation_path)
    required_report = {
        "spec_id": "MEDNORM-P1-002",
        "stage": "G1",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "official_dev_touched": False,
        "selection_split": "internal_val",
    }
    for key, expected in required_report.items():
        if g1_report.get(key) != expected:
            raise ValueError(f"unexpected G1 report field {key}")
    if g1_report.get("gate", {}).get("status") != "PASS":
        raise ValueError("G1 gate has not passed")
    if g1_validation.get("status") != "PASS" or g1_validation.get("stage") != "G1":
        raise ValueError("independent G1 validation has not passed")
    expected_outputs = g1_report.get("output_sha256")
    if not isinstance(expected_outputs, dict):
        raise ValueError("G1 report has no output hashes")
    for filename, path in (
        ("internal_train_candidates.jsonl.gz", g1_train_path),
        ("internal_val_candidates.jsonl.gz", g1_val_path),
    ):
        if expected_outputs.get(filename) != sha256_file(path):
            raise ValueError(f"G1 output hash mismatch for {filename}")

    catalog = _catalog_names(catalog_path)
    catalog_set = set(catalog)
    if g1_report.get("input_catalog_sha256") != sha256_file(catalog_path):
        raise ValueError("G1 catalog hash differs from the locked catalog")
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

    g1_train_by_id = _by_id(_read_jsonl(g1_train_path), source="G1 internal train")
    g1_val_by_id = _by_id(_read_jsonl(g1_val_path), source="G1 internal validation")
    covered_train_ids = {
        str(row["id"])
        for row in internal_train_rows
        if any(str(label) in catalog_set for label in row["labels"])
    }
    if set(g1_train_by_id) != covered_train_ids:
        raise ValueError("G1 train candidates do not exactly cover train rows with ICD labels")
    if set(g1_val_by_id) != internal_val_ids:
        raise ValueError("G1 validation candidates do not exactly cover internal validation")

    train_bundles: dict[str, JsonRow] = {}
    for row in internal_train_rows:
        sample_id = str(row["id"])
        if sample_id not in g1_train_by_id:
            continue
        train_bundles[sample_id] = _candidate_bundle(
            row,
            g1_train_by_id[sample_id],
            catalog=catalog_set,
            main_k=main_k,
            rrf_constant=rrf_constant,
        )
    val_bundles = [
        _candidate_bundle(
            row,
            g1_val_by_id[str(row["id"])],
            catalog=catalog_set,
            main_k=main_k,
            rrf_constant=rrf_constant,
        )
        for row in internal_val_rows
    ]

    train_pairs: list[JsonRow] = []
    for row in internal_train_rows:
        bundle = train_bundles.get(str(row["id"]))
        if bundle is not None:
            train_pairs.extend(
                _training_pairs(
                    row,
                    bundle,
                    catalog=catalog_set,
                    hard_negative_count=hard_negative_count,
                    sampled_negative_count=sampled_negative_count,
                    seed=seed,
                )
            )
    val_pairs: list[JsonRow] = []
    for row, bundle in zip(internal_val_rows, val_bundles, strict=True):
        val_pairs.extend(
            _training_pairs(
                row,
                bundle,
                catalog=catalog_set,
                hard_negative_count=hard_negative_count,
                sampled_negative_count=sampled_negative_count,
                seed=seed,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, Iterable[Mapping[str, Any]]] = {
        "internal_train_pairs.jsonl.gz": train_pairs,
        "internal_val_pairs.jsonl.gz": val_pairs,
        "internal_train_count.jsonl.gz": _count_rows(internal_train_rows),
        "internal_val_candidates.jsonl.gz": val_bundles,
    }
    for filename, rows in output_rows.items():
        _write_gzip_jsonl(output_dir / filename, rows)

    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-002",
        "stage": "G2",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "G1 sparse+dense RRF",
        "kg_enabled": False,
        "selection_split": "internal_val",
        "official_dev_touched": False,
        "seed": seed,
        "candidate_k": main_k,
        "rrf_constant": rrf_constant,
        "retrieval_score_definition": "RRF score divided by theoretical two-source rank-1 maximum",
        "hard_negative_count": hard_negative_count,
        "sampled_negative_count": sampled_negative_count,
        "counts": {
            "icd_candidate_count": len(catalog),
            "full_train_samples": len(train_rows),
            "internal_train_samples": len(internal_train_rows),
            "internal_val_samples": len(internal_val_rows),
            "covered_internal_train_samples": len(train_bundles),
            "internal_train_pairs": len(train_pairs),
            "internal_val_pairs": len(val_pairs),
            "skipped_oov_only_internal_train_samples": (
                len(internal_train_rows) - len(train_bundles)
            ),
        },
        "g1": {
            "report_sha256": sha256_file(g1_report_path),
            "validation_sha256": sha256_file(g1_validation_path),
            "model": g1_report.get("model"),
            "candidate_sha256": {
                "internal_train": sha256_file(g1_train_path),
                "internal_val": sha256_file(g1_val_path),
            },
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
            "Only internal_train ICD-covered labels supervise pair/count models. "
            "Gold labels may form positive training pairs but are never inserted into G1 "
            "candidate rankings. Internal_val selects epochs and decoding only. Official dev "
            "is not read or emitted."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
