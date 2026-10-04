"""Unified ICD candidate profiles and KG hard negatives for P1-003 O3."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from mednorm.data import sha256_file
from mednorm.kg_training_data_v2 import (
    DEFAULT_NEGATIVE_RELATIONS,
    build_alias_profiles,
    build_relation_neighbors,
    candidate_profile_text,
    select_relation_hard_negatives,
)
from mednorm.retrieval import CharacterTfidfRetriever

JsonRow = dict[str, Any]


def _read_json(path: Path) -> JsonRow:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[JsonRow]:
    def parse(lines: Iterable[str]) -> list[JsonRow]:
        rows: list[JsonRow] = []
        for line_number, line in enumerate(lines, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
        return rows

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return parse(handle)
    with path.open(encoding="utf-8") as handle:
        return parse(handle)


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


def _profile_feature(
    feature: Mapping[str, Any], profile_text: Mapping[str, str]
) -> JsonRow:
    name = str(feature["name"])
    return {
        "name": name,
        "retrieval_score": float(feature["retrieval_score"]),
        "candidate_text": profile_text[name],
    }


def _enrich_fit_group(
    group: Mapping[str, Any],
    *,
    fused_candidates: Sequence[str],
    profile_text: Mapping[str, str],
    relation_neighbors: Mapping[str, Mapping[str, Sequence[str]]],
    relation_hard_negative_count: int,
    retrieval_head_count: int,
) -> tuple[JsonRow, int]:
    positives = [str(name) for name in group["positives"]]
    base_negatives = [dict(row) for row in group["negative_pool"]]
    existing = {str(row["name"]) for row in base_negatives}
    selected = select_relation_hard_negatives(
        gold_labels=positives,
        ranked_candidates=fused_candidates,
        relation_neighbors=relation_neighbors,
        existing_candidates=existing,
        limit=relation_hard_negative_count,
    )
    relation_features = [
        {
            "name": name,
            "retrieval_score": 0.0,
            "candidate_text": profile_text[name],
            "kg_relations": list(relations),
        }
        for name, relations in selected
    ]
    head_count = min(retrieval_head_count, len(base_negatives))
    enriched_base = [_profile_feature(row, profile_text) for row in base_negatives]
    negatives = [
        *enriched_base[:head_count],
        *relation_features,
        *enriched_base[head_count:],
    ]
    return (
        {
            **group,
            "positive_features": [
                _profile_feature(row, profile_text) for row in group["positive_features"]
            ],
            "negative_pool": negatives,
            "supervision_source": "chip_cdn_fit_with_kg_relation_hard_negatives",
        },
        len(selected),
    )


def _synthetic_alias_groups(
    profiles: Mapping[str, Sequence[str]],
    *,
    catalog: Sequence[str],
    profile_text: Mapping[str, str],
    aliases_per_candidate: int,
    hard_negative_count: int,
) -> list[JsonRow]:
    retriever = CharacterTfidfRetriever().fit(tuple(catalog))
    groups: list[JsonRow] = []
    for target_index, target in enumerate(sorted(profiles)):
        for alias_index, alias in enumerate(profiles[target][:aliases_per_candidate]):
            negatives = [
                candidate
                for candidate in retriever.search_one(alias, k=hard_negative_count + 8)
                if candidate.name != target
            ][:hard_negative_count]
            groups.append(
                {
                    "sample_id": f"kg-alias-{target_index:05d}-{alias_index}",
                    "text": alias,
                    "positives": [target],
                    "positive_features": [
                        {
                            "name": target,
                            "retrieval_score": 0.0,
                            "candidate_text": profile_text[target],
                        }
                    ],
                    "negative_pool": [
                        {
                            "name": candidate.name,
                            "retrieval_score": float(candidate.score),
                            "candidate_text": profile_text[candidate.name],
                        }
                        for candidate in negatives
                    ],
                    "positive_in_candidate_count": 1,
                    "supervision_source": "cpubmedkg_typed_unambiguous_alias",
                }
            )
    return groups


def build_profile_data_v3(
    *,
    catalog_path: Path,
    o1_artifact_dir: Path,
    o2_data_dir: Path,
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
    """Build one O3 dataset whose only new supervision comes from CPubMed-KG."""
    budgets = (
        maximum_profile_aliases,
        synthetic_aliases_per_candidate,
        synthetic_hard_negative_count,
        relation_hard_negative_count,
        retrieval_head_count,
    )
    if min(budgets) < 0 or maximum_profile_aliases < 1:
        raise ValueError("profile or negative budgets are invalid")
    catalog_rows = _read_jsonl(catalog_path)
    catalog = tuple(str(row["name"]) for row in catalog_rows)
    if not catalog or len(catalog) != len(set(catalog)):
        raise ValueError("catalog names must be non-empty and unique")
    catalog_set = set(catalog)

    o2_manifest_path = o2_data_dir / "manifest.json"
    o2_manifest = _read_json(o2_manifest_path)
    required_o2 = {
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
    }
    for key, expected in required_o2.items():
        if o2_manifest.get(key) != expected:
            raise ValueError(f"unexpected O2 manifest field {key}")
    if o2_manifest.get("input_sha256", {}).get("catalog") != sha256_file(catalog_path):
        raise ValueError("O2 catalog hash mismatch")
    o2_paths = {
        name: o2_data_dir / name
        for name in ("fit_groups.jsonl.gz", "fit_count.jsonl.gz", "tune_candidates.jsonl.gz")
    }
    for name, path in o2_paths.items():
        if o2_manifest.get("output_sha256", {}).get(name) != sha256_file(path):
            raise ValueError(f"O2 data hash mismatch for {name}")

    o1_report_path = o1_artifact_dir / "report.json"
    o1_validation_path = o1_artifact_dir / "validation.json"
    o1_fit_path = o1_artifact_dir / "fit_candidates.jsonl.gz"
    o1_report = _read_json(o1_report_path)
    o1_validation = _read_json(o1_validation_path)
    if (
        o1_report.get("spec_id") != "MEDNORM-P1-003"
        or o1_report.get("stage") != "O1_RETRIEVAL"
        or o1_report.get("gate", {}).get("status") != "PASS"
        or o1_report.get("official_dev_touched") is not False
        or o1_validation.get("status") != "PASS"
    ):
        raise ValueError("O1 retrieval artifact is not independently validated")
    if o1_report.get("output_sha256", {}).get(o1_fit_path.name) != sha256_file(
        o1_fit_path
    ):
        raise ValueError("O1 fit-candidate hash mismatch")

    alias_rows = _read_jsonl(aliases_path)
    profiles = build_alias_profiles(
        alias_rows,
        catalog=catalog_set,
        minimum_dice=minimum_alias_dice,
        maximum_aliases_per_candidate=maximum_profile_aliases,
    )
    relation_neighbors = build_relation_neighbors(
        _read_jsonl(edges_path), catalog=catalog_set
    )
    profile_text = {
        name: candidate_profile_text(name, profiles.get(name, ())) for name in catalog
    }

    fit_groups = _read_jsonl(o2_paths["fit_groups.jsonl.gz"])
    o1_fit_rows = _read_jsonl(o1_fit_path)
    o1_by_id = {str(row["id"]): row for row in o1_fit_rows}
    if len(o1_by_id) != len(o1_fit_rows):
        raise ValueError("O1 fit candidates contain duplicate IDs")
    enriched_groups: list[JsonRow] = []
    relation_pair_count = 0
    relation_sample_count = 0
    for group in fit_groups:
        sample_id = str(group["sample_id"])
        o1_row = o1_by_id.get(sample_id)
        if o1_row is None:
            raise ValueError(f"O1 fit candidate is missing for {sample_id}")
        fused = o1_row.get("fused_candidates")
        if not isinstance(fused, list) or not all(isinstance(name, str) for name in fused):
            raise ValueError(f"O1 fused ranking is invalid for {sample_id}")
        enriched, relation_count = _enrich_fit_group(
            group,
            fused_candidates=fused[:400],
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
    tune_bundles: list[JsonRow] = []
    for bundle in _read_jsonl(o2_paths["tune_candidates.jsonl.gz"]):
        tune_bundles.append(
            {
                **bundle,
                "candidates": [
                    _profile_feature(feature, profile_text)
                    for feature in bundle["candidates"]
                ],
            }
        )
    fit_count_rows = _read_jsonl(o2_paths["fit_count.jsonl.gz"])
    outputs: dict[str, list[JsonRow]] = {
        "fit_groups.jsonl.gz": [*enriched_groups, *synthetic_groups],
        "fit_count.jsonl.gz": fit_count_rows,
        "tune_candidates.jsonl.gz": tune_bundles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in outputs.items():
        _write_gzip_jsonl(output_dir / filename, rows)

    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O3_PROFILE_DATA",
        "protocol": "Strict-ICD",
        "method": "unified ICD standard-name plus CPubMed-KG candidate profile ranker",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "O1 sparse+dense RRF plus CPubMed-KG candidate profiles",
        "kg_enabled": True,
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
        "audit_used": False,
        "candidate_k": int(o2_manifest["candidate_k"]),
        "negative_pool_size": max(
            len(group["negative_pool"]) for group in outputs["fit_groups.jsonl.gz"]
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
            "fit_samples": int(o2_manifest["counts"]["fit_samples"]),
            "covered_fit_samples": len(enriched_groups),
            "base_fit_groups": len(enriched_groups),
            "synthetic_alias_groups": len(synthetic_groups),
            "fit_groups": len(outputs["fit_groups.jsonl.gz"]),
            "tune_samples": len(tune_bundles),
            "profiled_candidates": len(profiles),
            "profile_aliases": sum(len(aliases) for aliases in profiles.values()),
            "relation_hard_negative_pairs": relation_pair_count,
            "relation_hard_negative_samples": relation_sample_count,
            "audit_samples_locked_out": int(
                o2_manifest["counts"]["audit_samples_locked_out"]
            ),
        },
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "o2_manifest": sha256_file(o2_manifest_path),
            "o2_fit_groups": sha256_file(o2_paths["fit_groups.jsonl.gz"]),
            "o2_fit_count": sha256_file(o2_paths["fit_count.jsonl.gz"]),
            "o2_tune_candidates": sha256_file(o2_paths["tune_candidates.jsonl.gz"]),
            "o1_report": sha256_file(o1_report_path),
            "o1_validation": sha256_file(o1_validation_path),
            "o1_fit_candidates": sha256_file(o1_fit_path),
            "aliases": sha256_file(aliases_path),
            "edges": sha256_file(edges_path),
            "fit": o2_manifest["input_sha256"]["fit"],
            "tune": o2_manifest["input_sha256"]["tune"],
            "audit_lock": o2_manifest["input_sha256"]["audit_lock"],
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in outputs
        },
        "leakage_policy": (
            "Only fit supervision, independently validated O1/O2 artifacts and external "
            "CPubMed-KG one-hop aliases/typed relations are used. Tune labels are selection-only. "
            "Audit and official dev are not read. The answer vocabulary remains ICD-only."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
