"""Fit-only weighted-neighbor evidence for P1-003 disease candidates."""

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

from mednorm.data import sha256_file

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


def weighted_label_scores(
    neighbor_indices: np.ndarray,
    neighbor_similarities: np.ndarray,
    fit_rows: list[JsonRow],
    *,
    catalog: set[str],
    maximum_hits_per_label: int,
) -> dict[str, tuple[float, float, int]]:
    """Transfer fit labels with decreasing weight across hits for the same label."""
    if neighbor_indices.shape != neighbor_similarities.shape:
        raise ValueError("neighbor indices and similarities differ")
    hits: dict[str, list[float]] = {}
    for fit_index, similarity in zip(
        neighbor_indices.tolist(), neighbor_similarities.tolist(), strict=True
    ):
        score = float(similarity)
        if score <= 0.0:
            continue
        for raw_label in fit_rows[int(fit_index)]["labels"]:
            label = str(raw_label)
            if label in catalog:
                hits.setdefault(label, []).append(score)
    output: dict[str, tuple[float, float, int]] = {}
    for label, similarities in hits.items():
        retained = similarities[:maximum_hits_per_label]
        weighted = sum(score / rank for rank, score in enumerate(retained, start=1))
        output[label] = (weighted, retained[0], len(similarities))
    return output


def build_prototype_evidence_v3(
    *,
    catalog_path: Path,
    split_dir: Path,
    ranking_data_dir: Path,
    output_dir: Path,
    neighbor_count: int = 100,
    maximum_hits_per_label: int = 10,
    ngram_min: int = 1,
    ngram_max: int = 4,
) -> JsonRow:
    """Build tune candidate evidence without fitting on tune, audit or official dev."""
    counts = (neighbor_count, maximum_hits_per_label, ngram_min, ngram_max)
    if min(counts) < 1 or ngram_min > ngram_max:
        raise ValueError("prototype budgets or n-gram range are invalid")
    split_manifest = _read_json(split_dir / "manifest.json")
    if (
        split_manifest.get("spec_id") != "MEDNORM-P1-003"
        or split_manifest.get("stage") != "O1_SPLIT"
        or split_manifest.get("official_dev_touched") is not False
    ):
        raise ValueError("split manifest is not the locked P1-003 split")
    fit_path = split_dir / "fit.jsonl"
    tune_path = split_dir / "tune.jsonl"
    split_hashes = split_manifest.get("output_sha256")
    if not isinstance(split_hashes, dict):
        raise ValueError("split manifest has no output hashes")
    for path in (fit_path, tune_path):
        if split_hashes.get(path.name) != sha256_file(path):
            raise ValueError(f"split hash mismatch for {path.name}")

    ranking_manifest_path = ranking_data_dir / "manifest.json"
    bundles_path = ranking_data_dir / "tune_candidates.jsonl.gz"
    ranking_manifest = _read_json(ranking_manifest_path)
    required_ranking = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O2_RANKING_DATA",
        "candidate_vocabulary_source": "ICD only",
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
        "audit_used": False,
    }
    for key, expected in required_ranking.items():
        if ranking_manifest.get(key) != expected:
            raise ValueError(f"unexpected ranking manifest field {key}")
    if ranking_manifest.get("output_sha256", {}).get(
        bundles_path.name
    ) != sha256_file(bundles_path):
        raise ValueError("tune candidate bundle hash mismatch")

    catalog_rows = _read_jsonl(catalog_path)
    catalog = {str(row["name"]) for row in catalog_rows}
    if len(catalog) != len(catalog_rows):
        raise ValueError("catalog names must be unique")
    if ranking_manifest.get("input_sha256", {}).get("catalog") != sha256_file(
        catalog_path
    ):
        raise ValueError("ranking catalog hash mismatch")
    fit_rows = _read_jsonl(fit_path)
    tune_rows = _read_jsonl(tune_path)
    bundles = _read_jsonl(bundles_path)
    if [row.get("id") for row in tune_rows] != [row.get("id") for row in bundles]:
        raise ValueError("tune rows and candidate bundles are misaligned")
    candidate_k = int(ranking_manifest["candidate_k"])
    if any(len(bundle.get("candidates", ())) != candidate_k for bundle in bundles):
        raise ValueError("tune bundle candidate counts differ")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        sublinear_tf=True,
        norm="l2",
    )
    fit_matrix = vectorizer.fit_transform([str(row["text"]) for row in fit_rows])
    tune_matrix = vectorizer.transform([str(row["text"]) for row in tune_rows])
    similarities = (tune_matrix @ fit_matrix.T).toarray()
    retained_neighbors = min(neighbor_count, len(fit_rows))
    output_rows: list[JsonRow] = []
    nonzero = 0
    gold_agnostic_feature_count = 0
    for row_index, bundle in enumerate(bundles):
        row_scores = similarities[row_index]
        indices = np.argsort(-row_scores, kind="stable")[:retained_neighbors]
        label_scores = weighted_label_scores(
            indices,
            row_scores[indices],
            fit_rows,
            catalog=catalog,
            maximum_hits_per_label=maximum_hits_per_label,
        )
        features: list[JsonRow] = []
        for candidate in bundle["candidates"]:
            name = str(candidate["name"])
            weighted, maximum, hit_count = label_scores.get(name, (0.0, 0.0, 0))
            nonzero += int(weighted > 0.0)
            gold_agnostic_feature_count += 1
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
    evidence_path = output_dir / "tune_prototype_evidence.jsonl.gz"
    _write_gzip_jsonl(evidence_path, output_rows)
    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O2_PROTOTYPE_EVIDENCE",
        "protocol": "Strict-ICD",
        "method": "fit-only character TF-IDF weighted label-neighbor memory",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "frozen O1 sparse+dense RRF Top-400",
        "kg_enabled": False,
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
        "audit_used": False,
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
            "fit_samples": len(fit_rows),
            "tune_samples": len(tune_rows),
            "candidate_k": candidate_k,
            "candidate_features": gold_agnostic_feature_count,
            "nonzero_prototype_features": nonzero,
            "vectorizer_vocabulary": len(vectorizer.vocabulary_),
        },
        "runtime": {"scikit_learn": sklearn.__version__},
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "split_manifest": sha256_file(split_dir / "manifest.json"),
            "fit": sha256_file(fit_path),
            "tune": sha256_file(tune_path),
            "audit_lock": split_hashes.get("audit.jsonl"),
            "ranking_manifest": sha256_file(ranking_manifest_path),
            "tune_candidates": sha256_file(bundles_path),
        },
        "output_sha256": {evidence_path.name: sha256_file(evidence_path)},
        "leakage_policy": (
            "The vectorizer and label memory fit only fit.jsonl. Tune text is transform-only; "
            "tune labels are not read by the scorer. Audit content and official dev are not read."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
