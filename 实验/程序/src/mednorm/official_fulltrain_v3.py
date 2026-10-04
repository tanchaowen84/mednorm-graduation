"""Frozen full-train inputs and persistence guards for the official P1-003 run."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from mednorm.crossencoder_data_v2 import (
    _by_id,
    _candidate_bundle,
    _count_rows,
)
from mednorm.crossencoder_data_v2 import (
    _read_jsonl as _read_candidate_jsonl,
)
from mednorm.data import sha256_file
from mednorm.dense_data import build_dense_training_records
from mednorm.final_evaluation_v3 import final_evaluation_contract
from mednorm.kg_training_data_v2 import (
    DEFAULT_NEGATIVE_RELATIONS,
    build_alias_profiles,
    build_relation_neighbors,
    candidate_profile_text,
)
from mednorm.neural_data_v2 import _catalog_names, _read_jsonl, _report, _sparse_bundle
from mednorm.official_contract_v3 import (
    CONFIG_ID,
    OFFICIAL_DRIVE_ROOT,
    assert_persistent_drive_root,
    estimate_fp16_checkpoint_bytes,
    frozen_epoch,
    frozen_retrieval_weight,
)
from mednorm.profile_data_v3 import (
    _enrich_fit_group,
    _profile_feature,
    _synthetic_alias_groups,
)
from mednorm.prototype_evidence_v3 import weighted_label_scores
from mednorm.ranking_data_v3 import _fit_group
from mednorm.retrieval import CharacterTfidfRetriever

JsonRow = dict[str, Any]
__all__ = [
    "CONFIG_ID",
    "OFFICIAL_DRIVE_ROOT",
    "assert_persistent_drive_root",
    "build_official_profile_data",
    "build_official_prototype_evidence",
    "build_official_ranking_data",
    "build_official_retrieval_data",
    "estimate_fp16_checkpoint_bytes",
    "frozen_epoch",
    "frozen_retrieval_weight",
]


def _open_gzip(path: Path) -> tuple[BinaryIO, gzip.GzipFile, TextIO]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def _write_gzip(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    raw, compressed, text = _open_gzip(path)
    try:
        for row in rows:
            text.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        text.close()
        if not compressed.closed:
            compressed.close()
        if not raw.closed:
            raw.close()


def _validate_rows(rows: list[JsonRow], *, split: str, prefix: str) -> set[str]:
    ids: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        labels = row.get("labels")
        if (
            not isinstance(sample_id, str)
            or not sample_id.startswith(prefix)
            or sample_id in ids
            or not isinstance(row.get("text"), str)
            or not row["text"]
            or not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or not isinstance(row.get("all_labels_in_icd"), bool)
            or row.get("split") != split
        ):
            raise ValueError(f"invalid {split} row {index}")
        ids.add(sample_id)
    if not rows:
        raise ValueError(f"{split} rows cannot be empty")
    return ids


def build_official_retrieval_data(
    *,
    source_dir: Path,
    output_dir: Path,
    retrieval_k: int = 800,
    hard_negative_count: int = 8,
    easy_negative_count: int = 2,
    seed: int = 2026,
) -> JsonRow:
    """Create text-only full official-train supervision and official-dev sparse input."""
    if retrieval_k < 1:
        raise ValueError("retrieval_k must be positive")
    if hard_negative_count < 0 or easy_negative_count < 0:
        raise ValueError("negative counts must be non-negative")
    contract = final_evaluation_contract("official_dev")
    catalog_path = source_dir / "icd_catalog.jsonl"
    train_path = source_dir / "train.jsonl"
    dev_path = source_dir / "dev.jsonl"
    catalog = _catalog_names(catalog_path)
    train_rows = _read_jsonl(train_path)
    dev_rows = _read_jsonl(dev_path)
    train_ids = _validate_rows(train_rows, split="train", prefix="train-")
    dev_ids = _validate_rows(dev_rows, split="dev", prefix="dev-")
    if train_ids & dev_ids:
        raise ValueError("official train and dev IDs overlap")

    retriever = CharacterTfidfRetriever(
        rerank_pool_size=max(300, retrieval_k)
    ).fit(catalog)
    train_rankings = retriever.search(
        [str(row["text"]) for row in train_rows], k=retrieval_k
    )
    dev_rankings = retriever.search(
        [str(row["text"]) for row in dev_rows], k=retrieval_k
    )
    sparse_by_id = {
        str(row["id"]): tuple(candidate.name for candidate in ranking)
        for row, ranking in zip(train_rows, train_rankings, strict=True)
    }
    training_records = build_dense_training_records(
        train_rows,
        catalog_names=catalog,
        sparse_rankings=sparse_by_id,
        hard_negative_count=hard_negative_count,
        easy_negative_count=easy_negative_count,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, Iterable[Mapping[str, Any]]] = {
        "official_train_queries.jsonl.gz": training_records,
        "official_train_sparse.jsonl.gz": (
            _sparse_bundle(row, ranking)
            for row, ranking in zip(train_rows, train_rankings, strict=True)
        ),
        "official_dev_sparse.jsonl.gz": (
            _sparse_bundle(row, ranking)
            for row, ranking in zip(dev_rows, dev_rankings, strict=True)
        ),
    }
    for filename, rows in output_rows.items():
        _write_gzip(output_dir / filename, rows)

    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": contract.retrieval_stage,
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": contract.evaluation_split,
        "selection_complete": True,
        "training_policy": "frozen epoch counts selected before official dev",
        "labels_in_official_dev_sparse": True,
        "audit_used": contract.audit_used,
        "official_dev_touched": contract.official_dev_touched,
        "seed": seed,
        "retrieval_k": retrieval_k,
        "main_candidate_k": 400,
        "rrf_constant": 60,
        "hard_negative_count": hard_negative_count,
        "easy_negative_count": easy_negative_count,
        "counts": {
            "icd_candidate_count": len(catalog),
            "official_train_samples": len(train_rows),
            "official_train_training_records": len(training_records),
            "official_train_oov_only_skipped": len(train_rows) - len(training_records),
            "official_dev_samples": len(dev_rows),
        },
        "retrieval": {
            "official_dev_sparse": _report(
                dev_rows, dev_rankings, retrieval_k=retrieval_k
            )
        },
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "official_train": sha256_file(train_path),
            "official_dev": sha256_file(dev_path),
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in output_rows
        },
        "leakage_policy": (
            "All 6000 official-train rows may supervise the fixed final model. Official-dev "
            "text is retrieval/evaluation-only; its labels are retained solely for final "
            "metrics and never enter training, epoch selection, thresholds or fusion weights."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_official_ranking_data(
    *,
    source_dir: Path,
    retriever_artifact_dir: Path,
    output_dir: Path,
    candidate_k: int = 400,
    negative_pool_size: int = 96,
    rrf_constant: int = 60,
    seed: int = 2026,
) -> JsonRow:
    """Turn the frozen full-train retriever outputs into ranker/count inputs."""
    if min(candidate_k, negative_pool_size, rrf_constant) < 1:
        raise ValueError("official ranking budgets must be positive")
    catalog_path = source_dir / "icd_catalog.jsonl"
    train_path = source_dir / "train.jsonl"
    dev_path = source_dir / "dev.jsonl"
    catalog = _catalog_names(catalog_path)
    catalog_set = set(catalog)
    train_rows = _read_jsonl(train_path)
    dev_rows = _read_jsonl(dev_path)
    _validate_rows(train_rows, split="train", prefix="train-")
    _validate_rows(dev_rows, split="dev", prefix="dev-")

    report_path = retriever_artifact_dir / "report.json"
    train_candidates_path = (
        retriever_artifact_dir / "official_train_candidates.jsonl.gz"
    )
    dev_candidates_path = retriever_artifact_dir / "official_dev_candidates.jsonl.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("official retriever report must contain an object")
    required_report = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O7_OFFICIAL_DEV_RETRIEVAL",
        "protocol": "Strict-ICD",
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
    }
    for field, expected in required_report.items():
        if report.get(field) != expected:
            raise ValueError(f"unexpected official retriever report field {field}")
    if report.get("input_catalog_sha256") != sha256_file(catalog_path):
        raise ValueError("official retriever report catalog hash mismatch")
    for path in (train_candidates_path, dev_candidates_path):
        if report.get("output_sha256", {}).get(path.name) != sha256_file(path):
            raise ValueError(f"official retriever output hash mismatch for {path.name}")

    train_candidates = _by_id(
        _read_candidate_jsonl(train_candidates_path),
        source="official train retriever candidates",
    )
    dev_candidates = _by_id(
        _read_candidate_jsonl(dev_candidates_path),
        source="official dev retriever candidates",
    )
    covered_train_ids = {
        str(row["id"])
        for row in train_rows
        if any(str(label) in catalog_set for label in row["labels"])
    }
    if set(train_candidates) != covered_train_ids:
        raise ValueError("retriever candidates do not cover official ICD-train rows")
    if set(dev_candidates) != {str(row["id"]) for row in dev_rows}:
        raise ValueError("retriever candidates do not cover official dev")

    train_groups: list[JsonRow] = []
    for row in train_rows:
        candidate_row = train_candidates.get(str(row["id"]))
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
            train_groups.append(group)
    dev_bundles = [
        _candidate_bundle(
            row,
            dev_candidates[str(row["id"])],
            catalog=catalog_set,
            main_k=candidate_k,
            rrf_constant=rrf_constant,
        )
        for row in dev_rows
    ]
    expected_candidate_count = min(candidate_k, len(catalog))
    if any(
        len(bundle["candidates"]) != expected_candidate_count
        for bundle in dev_bundles
    ):
        raise ValueError("official dev candidate bundles have an invalid size")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, list[JsonRow]] = {
        "official_train_groups.jsonl.gz": train_groups,
        "official_train_count.jsonl.gz": [
            dict(row) for row in _count_rows(train_rows)
        ],
        "official_dev_candidates.jsonl.gz": dev_bundles,
    }
    for filename, rows in output_rows.items():
        _write_gzip(output_dir / filename, rows)
    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O6_OFFICIAL_FULLTRAIN_RANKING_DATA",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "full-train BGE plus sparse RRF",
        "kg_enabled": False,
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
        "seed": seed,
        "candidate_k": candidate_k,
        "negative_pool_size": negative_pool_size,
        "rrf_constant": rrf_constant,
        "counts": {
            "icd_candidate_count": len(catalog),
            "official_train_samples": len(train_rows),
            "covered_official_train_samples": len(train_groups),
            "official_dev_samples": len(dev_bundles),
            "official_train_positive_labels": sum(
                len(group["positives"]) for group in train_groups
            ),
            "official_train_positive_labels_in_top_k": sum(
                int(group["positive_in_candidate_count"]) for group in train_groups
            ),
        },
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "official_train": sha256_file(train_path),
            "official_dev": sha256_file(dev_path),
            "retriever_report": sha256_file(report_path),
            "official_train_candidates": sha256_file(train_candidates_path),
            "official_dev_candidates": sha256_file(dev_candidates_path),
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in output_rows
        },
        "leakage_policy": (
            "Only official-train labels create ranking groups and count targets. "
            "Official-dev labels remain attached only to the final evaluation bundles."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_official_profile_data(
    *,
    catalog_path: Path,
    ranking_data_dir: Path,
    retriever_artifact_dir: Path,
    aliases_path: Path,
    edges_path: Path,
    output_dir: Path,
    minimum_alias_dice: float = 0.3,
    maximum_profile_aliases: int = 3,
    synthetic_aliases_per_candidate: int = 2,
    synthetic_hard_negative_count: int = 8,
    relation_hard_negative_count: int = 4,
    retrieval_head_count: int = 4,
) -> JsonRow:
    """Apply the frozen CPubMed-KG profile recipe to full train and official dev."""
    budgets = (
        maximum_profile_aliases,
        synthetic_aliases_per_candidate,
        synthetic_hard_negative_count,
        relation_hard_negative_count,
        retrieval_head_count,
    )
    if min(budgets) < 0 or maximum_profile_aliases < 1:
        raise ValueError("official KG profile budgets are invalid")
    catalog_rows = _read_candidate_jsonl(catalog_path)
    catalog = tuple(str(row["name"]) for row in catalog_rows)
    if not catalog or len(catalog) != len(set(catalog)):
        raise ValueError("official profile catalog names must be unique")
    catalog_set = set(catalog)

    ranking_manifest_path = ranking_data_dir / "manifest.json"
    train_groups_path = ranking_data_dir / "official_train_groups.jsonl.gz"
    train_count_path = ranking_data_dir / "official_train_count.jsonl.gz"
    dev_candidates_path = ranking_data_dir / "official_dev_candidates.jsonl.gz"
    ranking_manifest = json.loads(ranking_manifest_path.read_text(encoding="utf-8"))
    required_ranking = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O6_OFFICIAL_FULLTRAIN_RANKING_DATA",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
    }
    for field, expected in required_ranking.items():
        if ranking_manifest.get(field) != expected:
            raise ValueError(f"unexpected official ranking field {field}")
    if ranking_manifest.get("input_sha256", {}).get("catalog") != sha256_file(
        catalog_path
    ):
        raise ValueError("official ranking catalog hash mismatch")
    for path in (train_groups_path, train_count_path, dev_candidates_path):
        if ranking_manifest.get("output_sha256", {}).get(path.name) != sha256_file(
            path
        ):
            raise ValueError(f"official ranking data hash mismatch for {path.name}")

    retriever_report_path = retriever_artifact_dir / "report.json"
    retriever_train_path = (
        retriever_artifact_dir / "official_train_candidates.jsonl.gz"
    )
    retriever_report = json.loads(
        retriever_report_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(retriever_report, dict)
        or retriever_report.get("stage") != "O7_OFFICIAL_DEV_RETRIEVAL"
        or retriever_report.get("output_sha256", {}).get(retriever_train_path.name)
        != sha256_file(retriever_train_path)
    ):
        raise ValueError("official retriever artifact is not frozen/validated")
    retriever_train_rows = _by_id(
        _read_candidate_jsonl(retriever_train_path),
        source="official train profile retrieval",
    )

    alias_rows = _read_candidate_jsonl(aliases_path)
    profiles = build_alias_profiles(
        alias_rows,
        catalog=catalog_set,
        minimum_dice=minimum_alias_dice,
        maximum_aliases_per_candidate=maximum_profile_aliases,
    )
    relation_neighbors = build_relation_neighbors(
        _read_candidate_jsonl(edges_path), catalog=catalog_set
    )
    profile_text = {
        name: candidate_profile_text(name, profiles.get(name, ())) for name in catalog
    }
    base_groups = _read_candidate_jsonl(train_groups_path)
    enriched_groups: list[JsonRow] = []
    relation_pair_count = 0
    relation_sample_count = 0
    for group in base_groups:
        sample_id = str(group["sample_id"])
        retriever_row = retriever_train_rows.get(sample_id)
        if retriever_row is None:
            raise ValueError(f"official retriever candidate missing for {sample_id}")
        fused = retriever_row.get("fused_candidates")
        if not isinstance(fused, list) or not all(
            isinstance(name, str) for name in fused
        ):
            raise ValueError(f"official fused candidates are invalid for {sample_id}")
        enriched, relation_count = _enrich_fit_group(
            group,
            fused_candidates=fused[: int(ranking_manifest["candidate_k"])],
            profile_text=profile_text,
            relation_neighbors=relation_neighbors,
            relation_hard_negative_count=relation_hard_negative_count,
            retrieval_head_count=retrieval_head_count,
        )
        enriched_groups.append(enriched)
        relation_pair_count += relation_count
        relation_sample_count += int(relation_count > 0)
    synthetic_groups = _synthetic_alias_groups(
        profiles,
        catalog=catalog,
        profile_text=profile_text,
        aliases_per_candidate=synthetic_aliases_per_candidate,
        hard_negative_count=synthetic_hard_negative_count,
    )
    dev_bundles = [
        {
            **bundle,
            "candidates": [
                _profile_feature(feature, profile_text)
                for feature in bundle["candidates"]
            ],
        }
        for bundle in _read_candidate_jsonl(dev_candidates_path)
    ]
    train_count_rows = _read_candidate_jsonl(train_count_path)
    output_rows: dict[str, list[JsonRow]] = {
        "official_train_groups.jsonl.gz": [*enriched_groups, *synthetic_groups],
        "official_train_count.jsonl.gz": train_count_rows,
        "official_dev_candidates.jsonl.gz": dev_bundles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in output_rows.items():
        _write_gzip(output_dir / filename, rows)
    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O6_OFFICIAL_FULLTRAIN_PROFILE_DATA",
        "protocol": "Strict-ICD",
        "method": "ICD standard names plus frozen CPubMed-KG one-hop profiles",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "full-train BGE plus sparse RRF plus CPubMed-KG profiles",
        "kg_enabled": True,
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
        "candidate_k": int(ranking_manifest["candidate_k"]),
        "negative_pool_size": max(
            len(group["negative_pool"]) for group in output_rows["official_train_groups.jsonl.gz"]
        ),
        "kg_policy": {
            "alias_edge": "direct one-hop only",
            "alias_typed_as_disease": True,
            "alias_unambiguous": True,
            "minimum_alias_dice": minimum_alias_dice,
            "maximum_profile_aliases": maximum_profile_aliases,
            "synthetic_aliases_per_candidate": synthetic_aliases_per_candidate,
            "synthetic_hard_negative_count": synthetic_hard_negative_count,
            "negative_relations": list(DEFAULT_NEGATIVE_RELATIONS),
            "relation_hard_negative_count": relation_hard_negative_count,
            "retrieval_head_before_relation_negatives": retrieval_head_count,
            "transitive_closure": False,
        },
        "counts": {
            "icd_candidate_count": len(catalog),
            "official_train_samples": int(
                ranking_manifest["counts"]["official_train_samples"]
            ),
            "base_train_groups": len(enriched_groups),
            "synthetic_alias_groups": len(synthetic_groups),
            "official_train_groups": len(output_rows["official_train_groups.jsonl.gz"]),
            "official_dev_samples": len(dev_bundles),
            "profiled_candidates": len(profiles),
            "profile_aliases": sum(len(values) for values in profiles.values()),
            "relation_hard_negative_pairs": relation_pair_count,
            "relation_hard_negative_samples": relation_sample_count,
        },
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "ranking_manifest": sha256_file(ranking_manifest_path),
            "official_train_groups": sha256_file(train_groups_path),
            "official_train_count": sha256_file(train_count_path),
            "official_dev_candidates": sha256_file(dev_candidates_path),
            "retriever_report": sha256_file(retriever_report_path),
            "retriever_train_candidates": sha256_file(retriever_train_path),
            "aliases": sha256_file(aliases_path),
            "edges": sha256_file(edges_path),
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in output_rows
        },
        "leakage_policy": (
            "CPubMed-KG profiles and synthetic alias supervision are external and fixed. "
            "Official-dev labels are copied only into evaluation bundles and are never read "
            "while building profiles or training groups."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_official_prototype_evidence(
    *,
    catalog_path: Path,
    train_path: Path,
    ranking_data_dir: Path,
    output_dir: Path,
    neighbor_count: int = 100,
    maximum_hits_per_label: int = 10,
    ngram_min: int = 1,
    ngram_max: int = 4,
) -> JsonRow:
    """Fit the frozen prototype feature on all train rows and transform dev once."""
    if min(neighbor_count, maximum_hits_per_label, ngram_min, ngram_max) < 1:
        raise ValueError("official prototype budgets must be positive")
    if ngram_min > ngram_max:
        raise ValueError("official prototype n-gram range is invalid")
    catalog_rows = _read_candidate_jsonl(catalog_path)
    catalog = {str(row["name"]) for row in catalog_rows}
    if len(catalog) != len(catalog_rows):
        raise ValueError("official prototype catalog names must be unique")
    train_rows = _read_candidate_jsonl(train_path)
    _validate_rows(train_rows, split="train", prefix="train-")
    ranking_manifest_path = ranking_data_dir / "manifest.json"
    dev_candidates_path = ranking_data_dir / "official_dev_candidates.jsonl.gz"
    ranking_manifest = json.loads(ranking_manifest_path.read_text(encoding="utf-8"))
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O6_OFFICIAL_FULLTRAIN_RANKING_DATA",
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
    }
    for field, expected in required.items():
        if ranking_manifest.get(field) != expected:
            raise ValueError(f"unexpected official prototype input field {field}")
    if ranking_manifest.get("input_sha256", {}).get("catalog") != sha256_file(
        catalog_path
    ):
        raise ValueError("official prototype catalog hash mismatch")
    if ranking_manifest.get("output_sha256", {}).get(
        dev_candidates_path.name
    ) != sha256_file(dev_candidates_path):
        raise ValueError("official prototype candidate hash mismatch")
    bundles = _read_candidate_jsonl(dev_candidates_path)
    if len(bundles) != int(ranking_manifest["counts"]["official_dev_samples"]):
        raise ValueError("official prototype dev count differs from manifest")
    candidate_k = int(ranking_manifest["candidate_k"])
    if any(len(bundle.get("candidates", ())) != candidate_k for bundle in bundles):
        raise ValueError("official prototype candidate counts differ")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        sublinear_tf=True,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(
        [str(row["text"]) for row in train_rows]
    )
    dev_matrix = vectorizer.transform([str(bundle["text"]) for bundle in bundles])
    similarities = (dev_matrix @ train_matrix.T).toarray()
    retained_neighbors = min(neighbor_count, len(train_rows))
    output_rows: list[JsonRow] = []
    nonzero = 0
    for index, bundle in enumerate(bundles):
        row_scores = similarities[index]
        indices = np.argsort(-row_scores, kind="stable")[:retained_neighbors]
        label_scores = weighted_label_scores(
            indices,
            row_scores[indices],
            train_rows,
            catalog=catalog,
            maximum_hits_per_label=maximum_hits_per_label,
        )
        features: list[JsonRow] = []
        for candidate in bundle["candidates"]:
            name = str(candidate["name"])
            weighted, maximum, hit_count = label_scores.get(name, (0.0, 0.0, 0))
            nonzero += int(weighted > 0.0)
            features.append(
                {
                    "name": name,
                    "prototype_score": round(float(weighted), 8),
                    "prototype_max_similarity": round(float(maximum), 8),
                    "prototype_hit_count": int(hit_count),
                }
            )
        output_rows.append({"id": bundle["id"], "features": features})

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "official_dev_prototype_evidence.jsonl.gz"
    _write_gzip(evidence_path, output_rows)
    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O7_OFFICIAL_DEV_PROTOTYPE_EVIDENCE",
        "protocol": "Strict-ICD",
        "method": "full-train character TF-IDF weighted label-neighbor memory",
        "candidate_vocabulary_source": "ICD only",
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
        "policy": {
            "analyzer": "char",
            "ngram_range": [ngram_min, ngram_max],
            "sublinear_tf": True,
            "norm": "l2",
            "neighbor_count": neighbor_count,
            "maximum_hits_per_label": maximum_hits_per_label,
            "label_aggregation": "sum(similarity / label_hit_rank)",
        },
        "counts": {
            "official_train_samples": len(train_rows),
            "official_dev_samples": len(bundles),
            "candidate_k": candidate_k,
            "candidate_features": len(bundles) * candidate_k,
            "nonzero_prototype_features": nonzero,
            "vectorizer_vocabulary": len(vectorizer.vocabulary_),
        },
        "runtime": {"scikit_learn": sklearn.__version__},
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "official_train": sha256_file(train_path),
            "ranking_manifest": sha256_file(ranking_manifest_path),
            "official_dev_candidates": sha256_file(dev_candidates_path),
        },
        "output_sha256": {evidence_path.name: sha256_file(evidence_path)},
        "leakage_policy": (
            "The vectorizer and label memory fit on official train only. Official-dev text "
            "is transform-only and dev labels are not read by prototype scoring."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
