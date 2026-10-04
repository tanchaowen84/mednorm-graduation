"""
[INPUT] Locked P1-003 fit/tune split and independently validated O1 RRF candidates.
[OUTPUT] Multi-positive fit ranking groups, fit count rows and full Top-K tune bundles.
[POS] Audit-safe data boundary shared by pointwise and listwise O2 rankers.
[UPDATE] KG/profile fields require a separate O3 manifest and must not enter this text control.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mednorm.crossencoder_data_v2 import (
    _by_id,
    _candidate_bundle,
    _catalog_names,
    _count_rows,
    _read_json,
    _read_jsonl,
    _write_gzip_jsonl,
)
from mednorm.data import sha256_file

JsonRow = dict[str, Any]


def _validate_split(
    split_dir: Path,
) -> tuple[JsonRow, dict[str, list[JsonRow]]]:
    manifest = _read_json(split_dir / "manifest.json")
    required = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O1_SPLIT",
        "official_dev_touched": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"unexpected split manifest field {key}")
    hashes = manifest.get("output_sha256")
    counts = manifest.get("counts")
    if not isinstance(hashes, dict) or not isinstance(counts, dict):
        raise ValueError("split manifest lacks hashes or counts")
    rows: dict[str, list[JsonRow]] = {}
    ids: dict[str, set[str]] = {}
    for split_name in ("fit", "tune", "audit"):
        path = split_dir / f"{split_name}.jsonl"
        if hashes.get(path.name) != sha256_file(path):
            raise ValueError(f"split hash mismatch for {path.name}")
        split_rows = _read_jsonl(path)
        if len(split_rows) != counts.get(split_name):
            raise ValueError(f"split count mismatch for {split_name}")
        split_ids = {str(row.get("id")) for row in split_rows}
        if len(split_ids) != len(split_rows) or any(
            row.get("split") != split_name for row in split_rows
        ):
            raise ValueError(f"invalid rows in {split_name}")
        rows[split_name] = split_rows
        ids[split_name] = split_ids
    if ids["fit"] & ids["tune"] or ids["fit"] & ids["audit"] or ids["tune"] & ids["audit"]:
        raise ValueError("fit/tune/audit IDs overlap")
    return manifest, rows


def _fit_group(
    original: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    catalog: set[str],
    negative_pool_size: int,
) -> JsonRow | None:
    positives = tuple(
        dict.fromkeys(str(label) for label in original["labels"] if str(label) in catalog)
    )
    if not positives:
        return None
    positive_set = set(positives)
    candidate_features = bundle.get("candidates")
    if not isinstance(candidate_features, list):
        raise ValueError(f"{original['id']} has invalid candidate features")
    feature_by_name = {
        str(feature["name"]): float(feature["retrieval_score"])
        for feature in candidate_features
    }
    negative_pool = [
        {
            "name": str(feature["name"]),
            "retrieval_score": float(feature["retrieval_score"]),
        }
        for feature in candidate_features
        if str(feature["name"]) not in positive_set
    ][:negative_pool_size]
    return {
        "sample_id": original["id"],
        "text": original["text"],
        "positives": list(positives),
        "positive_features": [
            {"name": name, "retrieval_score": feature_by_name.get(name, 0.0)}
            for name in positives
        ],
        "negative_pool": negative_pool,
        "positive_in_candidate_count": sum(name in feature_by_name for name in positives),
    }


def build_phase1_v3_ranking_data(
    *,
    source_dir: Path,
    split_dir: Path,
    o1_artifact_dir: Path,
    output_dir: Path,
    candidate_k: int = 400,
    negative_pool_size: int = 96,
    rrf_constant: int = 60,
    seed: int = 2026,
) -> JsonRow:
    """Build one shared text-only supervision set for matched O2 loss ablations."""
    if candidate_k < 1 or negative_pool_size < 1 or rrf_constant < 1:
        raise ValueError("candidate and ranking sizes must be positive")
    catalog_path = source_dir / "icd_catalog.jsonl"
    catalog = _catalog_names(catalog_path)
    catalog_set = set(catalog)
    split_manifest, split_rows = _validate_split(split_dir)

    report_path = o1_artifact_dir / "report.json"
    validation_path = o1_artifact_dir / "validation.json"
    fit_candidates_path = o1_artifact_dir / "fit_candidates.jsonl.gz"
    tune_candidates_path = o1_artifact_dir / "tune_candidates.jsonl.gz"
    report = _read_json(report_path)
    validation = _read_json(validation_path)
    required_report = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O1_RETRIEVAL",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "official_dev_touched": False,
        "train_split": "fit",
        "selection_split": "tune",
    }
    for key, expected in required_report.items():
        if report.get(key) != expected:
            raise ValueError(f"unexpected O1 report field {key}")
    if report.get("gate", {}).get("status") != "PASS":
        raise ValueError("O1 retrieval gate has not passed")
    for key, expected in (
        ("status", "PASS"),
        ("spec_id", "MEDNORM-P1-003"),
        ("stage", "O1_RETRIEVAL"),
        ("selection_split", "tune"),
    ):
        if validation.get(key) != expected:
            raise ValueError(f"unexpected O1 validation field {key}")
    if report.get("input_catalog_sha256") != sha256_file(catalog_path):
        raise ValueError("O1 catalog hash mismatch")
    output_hashes = report.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError("O1 report has no output hashes")
    for path in (fit_candidates_path, tune_candidates_path):
        if output_hashes.get(path.name) != sha256_file(path):
            raise ValueError(f"O1 candidate hash mismatch for {path.name}")

    fit_rows = split_rows["fit"]
    tune_rows = split_rows["tune"]
    fit_o1 = _by_id(_read_jsonl(fit_candidates_path), source="O1 fit candidates")
    tune_o1 = _by_id(_read_jsonl(tune_candidates_path), source="O1 tune candidates")
    covered_fit_ids = {
        str(row["id"])
        for row in fit_rows
        if any(str(label) in catalog_set for label in row["labels"])
    }
    if set(fit_o1) != covered_fit_ids:
        raise ValueError("O1 fit candidates do not cover exactly ICD-covered fit rows")
    if set(tune_o1) != {str(row["id"]) for row in tune_rows}:
        raise ValueError("O1 tune candidates do not cover exactly tune rows")

    fit_groups: list[JsonRow] = []
    for row in fit_rows:
        sample_id = str(row["id"])
        candidate_row = fit_o1.get(sample_id)
        if candidate_row is None:
            continue
        bundle = _candidate_bundle(
            row,
            candidate_row,
            catalog=catalog_set,
            main_k=candidate_k,
            rrf_constant=rrf_constant,
        )
        group = _fit_group(
            row,
            bundle,
            catalog=catalog_set,
            negative_pool_size=negative_pool_size,
        )
        if group is not None:
            fit_groups.append(group)
    tune_bundles = [
        _candidate_bundle(
            row,
            tune_o1[str(row["id"])],
            catalog=catalog_set,
            main_k=candidate_k,
            rrf_constant=rrf_constant,
        )
        for row in tune_rows
    ]
    expected_candidate_count = min(candidate_k, len(catalog))
    if any(
        len(bundle["candidates"]) != expected_candidate_count
        for bundle in tune_bundles
    ):
        raise ValueError("tune candidate bundles do not have the locked candidate count")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = {
        "fit_groups.jsonl.gz": fit_groups,
        "fit_count.jsonl.gz": _count_rows(fit_rows),
        "tune_candidates.jsonl.gz": tune_bundles,
    }
    for filename, rows in output_rows.items():
        _write_gzip_jsonl(output_dir / filename, rows)

    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O2_RANKING_DATA",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "O1 sparse+dense RRF",
        "kg_enabled": False,
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
        "audit_used": False,
        "seed": seed,
        "candidate_k": candidate_k,
        "negative_pool_size": negative_pool_size,
        "rrf_constant": rrf_constant,
        "counts": {
            "icd_candidate_count": len(catalog),
            "fit_samples": len(fit_rows),
            "covered_fit_samples": len(fit_groups),
            "tune_samples": len(tune_bundles),
            "audit_samples_locked_out": len(split_rows["audit"]),
            "fit_positive_labels": sum(len(group["positives"]) for group in fit_groups),
            "fit_positive_labels_in_top_k": sum(
                int(group["positive_in_candidate_count"]) for group in fit_groups
            ),
        },
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "split_manifest": sha256_file(split_dir / "manifest.json"),
            "fit": sha256_file(split_dir / "fit.jsonl"),
            "tune": sha256_file(split_dir / "tune.jsonl"),
            "audit_lock": sha256_file(split_dir / "audit.jsonl"),
            "o1_report": sha256_file(report_path),
            "o1_validation": sha256_file(validation_path),
            "o1_fit_candidates": sha256_file(fit_candidates_path),
            "o1_tune_candidates": sha256_file(tune_candidates_path),
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in output_rows
        },
        "leakage_policy": (
            "Only fit labels supervise ranking/count training. Tune labels are retained only "
            "for model selection. Audit is hash-locked but not emitted or evaluated. Official "
            "dev is untouched. This O2 artifact contains no KG, alias or relation field."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
