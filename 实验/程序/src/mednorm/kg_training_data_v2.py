"""
[INPUT] Locked G2 data/predictions plus one-hop CPubMed-KG aliases and ICD relations.
[OUTPUT] Deterministic knowledge-aware pair data, candidate profiles and fixed G2 counts.
[POS] G3 learning-data layer for HKG-Norm after the score-only KG ablation.
[UPDATE] Keep filtering, negative relations and hashes synchronized with MEDNORM-P1-002.
"""

from __future__ import annotations

import gzip
import io
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from mednorm.data import sha256_file
from mednorm.retrieval import CharacterTfidfRetriever
from mednorm.text import normalize_text

JsonRow = dict[str, Any]
DEFAULT_NEGATIVE_RELATIONS = ("鉴别诊断", "并发症")
_ALNUM_CJK = re.compile(r"[^0-9a-z\u3400-\u9fff]+")


def _read_json(path: Path) -> JsonRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[JsonRow]:
    rows: list[JsonRow] = []
    try:
        if path.suffix == ".gz":
            handle: Iterable[str]
            with gzip.open(path, "rt", encoding="utf-8") as compressed:
                handle = list(compressed)
        else:
            handle = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
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


def _surface(value: str) -> str:
    return _ALNUM_CJK.sub("", normalize_text(value).casefold())


def character_dice(left: str, right: str) -> float:
    """Character unigram/bigram Dice used only as a conservative alias filter."""
    normalized_left = _surface(left)
    normalized_right = _surface(right)

    def grams(value: str) -> set[str]:
        return set(value) | {value[index : index + 2] for index in range(len(value) - 1)}

    left_grams = grams(normalized_left)
    right_grams = grams(normalized_right)
    denominator = len(left_grams) + len(right_grams)
    return 2.0 * len(left_grams & right_grams) / denominator if denominator else 0.0


def build_alias_profiles(
    alias_rows: Sequence[Mapping[str, Any]],
    *,
    catalog: set[str],
    minimum_dice: float = 0.3,
    maximum_alias_length: int = 48,
    maximum_aliases_per_candidate: int = 3,
) -> dict[str, tuple[str, ...]]:
    """Keep only typed, unambiguous and lexically supported one-hop aliases."""
    if not 0.0 <= minimum_dice <= 1.0:
        raise ValueError("minimum_dice must be between zero and one")
    if maximum_alias_length < 2 or maximum_aliases_per_candidate < 1:
        raise ValueError("alias limits must be positive")
    candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)
    surface_targets: dict[str, set[str]] = defaultdict(set)
    provisional: list[tuple[str, str, float]] = []
    for index, row in enumerate(alias_rows):
        alias = row.get("alias")
        targets = row.get("icd_names")
        if (
            not isinstance(alias, str)
            or not isinstance(targets, list)
            or not all(isinstance(target, str) and target in catalog for target in targets)
            or not isinstance(row.get("typed_as_disease"), bool)
        ):
            raise ValueError(f"invalid KG alias row {index}")
        if not row["typed_as_disease"] or len(targets) != 1:
            continue
        target = str(targets[0])
        normalized_alias = _surface(alias)
        if not 2 <= len(normalized_alias) <= maximum_alias_length:
            continue
        score = character_dice(alias, target)
        if score < minimum_dice or normalized_alias == _surface(target):
            continue
        surface_targets[normalized_alias].add(target)
        provisional.append((target, alias, score))
    for target, alias, score in provisional:
        if len(surface_targets[_surface(alias)]) == 1:
            candidates[target].append((score, alias))
    return {
        target: tuple(
            alias
            for _, alias in sorted(values, key=lambda item: (-item[0], item[1]))[
                :maximum_aliases_per_candidate
            ]
        )
        for target, values in sorted(candidates.items())
    }


def candidate_profile_text(name: str, aliases: Sequence[str]) -> str:
    if not aliases:
        return name
    return f"{name}；知识图谱同义表达：{'、'.join(aliases)}"


def build_relation_neighbors(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    catalog: set[str],
    allowed_relations: Sequence[str] = DEFAULT_NEGATIVE_RELATIONS,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Build symmetric one-hop relation evidence used as hard negatives, never positives."""
    allowed = set(allowed_relations)
    if not allowed:
        raise ValueError("allowed_relations must not be empty")
    working: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for index, row in enumerate(edge_rows):
        source = row.get("source")
        target = row.get("target")
        relation = row.get("relation")
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or not isinstance(relation, str)
            or source not in catalog
            or target not in catalog
        ):
            raise ValueError(f"invalid ICD edge row {index}")
        if relation not in allowed or source == target:
            continue
        working[source][target].add(relation)
        working[target][source].add(relation)
    return {
        source: {
            target: tuple(sorted(relations))
            for target, relations in sorted(targets.items())
        }
        for source, targets in sorted(working.items())
    }


def select_relation_hard_negatives(
    *,
    gold_labels: Sequence[str],
    ranked_candidates: Sequence[str],
    relation_neighbors: Mapping[str, Mapping[str, Sequence[str]]],
    existing_candidates: set[str],
    limit: int,
) -> list[tuple[str, tuple[str, ...]]]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    gold = set(gold_labels)
    evidence: dict[str, set[str]] = defaultdict(set)
    for label in gold:
        for candidate, relations in relation_neighbors.get(label, {}).items():
            if candidate not in gold and candidate not in existing_candidates:
                evidence[candidate].update(relations)
    selected: list[tuple[str, tuple[str, ...]]] = []
    for candidate in ranked_candidates:
        if candidate in evidence:
            selected.append((candidate, tuple(sorted(evidence[candidate]))))
            if len(selected) == limit:
                break
    return selected


def _catalog_names(path: Path) -> tuple[str, ...]:
    names = tuple(str(row["name"]) for row in _read_jsonl(path))
    if not names or len(names) != len(set(names)):
        raise ValueError("catalog names must be non-empty and unique")
    return names


def _candidate_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    names = row.get("fused_candidates")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{row.get('id')} has invalid fused candidates")
    return tuple(names)


def build_phase1_v2_kg_training_data(
    *,
    processed_dir: Path,
    g1_artifact_dir: Path,
    g2_data_dir: Path,
    g2_report_path: Path,
    g2_predictions_path: Path,
    aliases_path: Path,
    edges_path: Path,
    output_dir: Path,
    minimum_alias_dice: float = 0.3,
    maximum_profile_aliases: int = 3,
    synthetic_aliases_per_candidate: int = 2,
    synthetic_hard_negative_count: int = 8,
    relation_hard_negative_count: int = 4,
) -> JsonRow:
    """Build G3 knowledge-aware training data without reading official dev."""
    if synthetic_aliases_per_candidate < 0 or synthetic_hard_negative_count < 0:
        raise ValueError("synthetic limits must be non-negative")
    catalog_path = processed_dir / "icd_catalog.jsonl"
    internal_train_path = processed_dir / "internal_train.jsonl"
    catalog = _catalog_names(catalog_path)
    catalog_set = set(catalog)
    g2_manifest_path = g2_data_dir / "manifest.json"
    g2_manifest = _read_json(g2_manifest_path)
    required_g2 = {
        "spec_id": "MEDNORM-P1-002",
        "stage": "G2",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "official_dev_touched": False,
        "selection_split": "internal_val",
        "candidate_k": 400,
    }
    for key, expected in required_g2.items():
        if g2_manifest.get(key) != expected:
            raise ValueError(f"unexpected G2 manifest field {key}")
    for filename, expected_hash in g2_manifest["output_sha256"].items():
        if sha256_file(g2_data_dir / filename) != expected_hash:
            raise ValueError(f"G2 data hash mismatch for {filename}")
    if sha256_file(catalog_path) != g2_manifest["input_sha256"]["catalog"]:
        raise ValueError("catalog hash differs from G2")

    g2_report = _read_json(g2_report_path)
    if (
        g2_report.get("stage") != "G2"
        or g2_report.get("kg_enabled") is not False
        or g2_report.get("official_dev_touched") is not False
        or g2_report.get("mode") != "full"
    ):
        raise ValueError("G2 report is not a locked full text-only result")
    if g2_report.get("output_sha256", {}).get(
        g2_predictions_path.name
    ) != sha256_file(g2_predictions_path):
        raise ValueError("G2 prediction hash mismatch")

    alias_rows = _read_jsonl(aliases_path)
    profiles = build_alias_profiles(
        alias_rows,
        catalog=catalog_set,
        minimum_dice=minimum_alias_dice,
        maximum_aliases_per_candidate=maximum_profile_aliases,
    )
    edge_rows = _read_jsonl(edges_path)
    relation_neighbors = build_relation_neighbors(edge_rows, catalog=catalog_set)
    profile_text = {
        name: candidate_profile_text(name, profiles.get(name, ())) for name in catalog
    }

    internal_train_rows = _read_jsonl(internal_train_path)
    train_by_id = {str(row["id"]): row for row in internal_train_rows}
    if len(train_by_id) != len(internal_train_rows):
        raise ValueError("internal train contains duplicate IDs")
    g1_train_path = g1_artifact_dir / "internal_train_candidates.jsonl.gz"
    if sha256_file(g1_train_path) != g2_manifest["g1"]["candidate_sha256"][
        "internal_train"
    ]:
        raise ValueError("G1 internal-train candidate hash differs from G2")
    g1_rows = _read_jsonl(g1_train_path)
    g1_by_id = {str(row["id"]): row for row in g1_rows}

    original_pairs = _read_jsonl(g2_data_dir / "internal_train_pairs.jsonl.gz")
    enriched_pairs: list[JsonRow] = []
    existing_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in original_pairs:
        candidate = str(row["candidate"])
        sample_id = str(row["sample_id"])
        existing_by_sample[sample_id].add(candidate)
        enriched_pairs.append({**row, "candidate_text": profile_text[candidate]})

    relation_pair_count = 0
    relation_sample_count = 0
    for sample_id, original in train_by_id.items():
        g1_row = g1_by_id.get(sample_id)
        if g1_row is None:
            continue
        ranked = _candidate_names(g1_row)[:400]
        selected = select_relation_hard_negatives(
            gold_labels=tuple(str(label) for label in original["labels"]),
            ranked_candidates=ranked,
            relation_neighbors=relation_neighbors,
            existing_candidates=existing_by_sample[sample_id],
            limit=relation_hard_negative_count,
        )
        relation_sample_count += int(bool(selected))
        for candidate, relations in selected:
            enriched_pairs.append(
                {
                    "sample_id": sample_id,
                    "text": original["text"],
                    "candidate": candidate,
                    "candidate_text": profile_text[candidate],
                    "label": 0,
                    "source": "kg_relation_hard_negative",
                    "kg_relations": list(relations),
                    "retrieval_score": 0.0,
                }
            )
            relation_pair_count += 1

    retriever = CharacterTfidfRetriever().fit(catalog)
    synthetic_pair_count = 0
    synthetic_positive_count = 0
    for target, aliases in profiles.items():
        for alias_index, alias in enumerate(aliases[:synthetic_aliases_per_candidate]):
            sample_id = f"kg-alias:{target}:{alias_index}"
            enriched_pairs.append(
                {
                    "sample_id": sample_id,
                    "text": alias,
                    "candidate": target,
                    "candidate_text": profile_text[target],
                    "label": 1,
                    "source": "kg_alias_positive",
                    "retrieval_score": 0.0,
                }
            )
            synthetic_pair_count += 1
            synthetic_positive_count += 1
            negatives = [
                candidate.name
                for candidate in retriever.search_one(
                    alias, k=synthetic_hard_negative_count + 8
                )
                if candidate.name != target
            ][:synthetic_hard_negative_count]
            for candidate in negatives:
                enriched_pairs.append(
                    {
                        "sample_id": sample_id,
                        "text": alias,
                        "candidate": candidate,
                        "candidate_text": profile_text[candidate],
                        "label": 0,
                        "source": "kg_alias_lexical_hard_negative",
                        "retrieval_score": 0.0,
                    }
                )
                synthetic_pair_count += 1

    val_pairs = [
        {**row, "candidate_text": profile_text[str(row["candidate"])]}
        for row in _read_jsonl(g2_data_dir / "internal_val_pairs.jsonl.gz")
    ]
    val_bundles: list[JsonRow] = []
    for bundle in _read_jsonl(g2_data_dir / "internal_val_candidates.jsonl.gz"):
        features = bundle.get("candidates")
        if not isinstance(features, list) or len(features) != 400:
            raise ValueError(f"{bundle.get('id')} lacks locked Top-400 candidates")
        val_bundles.append(
            {
                **bundle,
                "candidates": [
                    {**feature, "candidate_text": profile_text[str(feature["name"])]}
                    for feature in features
                ],
            }
        )

    g2_prediction_rows = _read_jsonl(g2_predictions_path)
    if [row["id"] for row in g2_prediction_rows] != [row["id"] for row in val_bundles]:
        raise ValueError("G2 predictions and G3 validation bundles are misaligned")
    fixed_count_rows = [
        {"id": row["id"], "count_prediction": int(row["count_prediction"])}
        for row in g2_prediction_rows
    ]
    count_rows = _read_jsonl(g2_data_dir / "internal_train_count.jsonl.gz")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Sequence[Mapping[str, Any]]] = {
        "internal_train_pairs.jsonl.gz": enriched_pairs,
        "internal_val_pairs.jsonl.gz": val_pairs,
        "internal_train_count.jsonl.gz": count_rows,
        "internal_val_candidates.jsonl.gz": val_bundles,
        "internal_val_fixed_counts.jsonl.gz": fixed_count_rows,
    }
    for filename, rows in outputs.items():
        _write_gzip_jsonl(output_dir / filename, rows)

    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-002",
        "stage": "G3",
        "protocol": "Strict-ICD",
        "method": "HKG-Norm knowledge-aware cross-encoder training",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "locked G2 Top-400",
        "kg_enabled": True,
        "selection_split": "internal_val",
        "official_dev_touched": False,
        "candidate_k": 400,
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
        },
        "counts": {
            "icd_candidate_count": len(catalog),
            "internal_train_samples": len(internal_train_rows),
            "internal_val_samples": len(val_bundles),
            "profiled_candidates": len(profiles),
            "profile_aliases": sum(len(aliases) for aliases in profiles.values()),
            "base_train_pairs": len(original_pairs),
            "relation_hard_negative_pairs": relation_pair_count,
            "relation_hard_negative_samples": relation_sample_count,
            "synthetic_alias_positive_pairs": synthetic_positive_count,
            "synthetic_alias_total_pairs": synthetic_pair_count,
            "internal_train_pairs": len(enriched_pairs),
            "internal_val_pairs": len(val_pairs),
        },
        "fixed_count_source": "locked G2 internal-val count predictions",
        "fixed_count": {
            "best_epoch": int(g2_report["count"]["best_epoch"]),
            "internal_macro_f1": float(g2_report["count"]["internal_macro_f1"]),
            "model_sha256": str(g2_report["count"]["model_sha256"]),
        },
        "input_sha256": {
            "catalog": sha256_file(catalog_path),
            "internal_train": sha256_file(internal_train_path),
            "g1_internal_train_candidates": sha256_file(g1_train_path),
            "g2_manifest": sha256_file(g2_manifest_path),
            "g2_report": sha256_file(g2_report_path),
            "g2_predictions": sha256_file(g2_predictions_path),
            "aliases": sha256_file(aliases_path),
            "edges": sha256_file(edges_path),
        },
        "output_sha256": {
            filename: sha256_file(output_dir / filename) for filename in outputs
        },
        "leakage_policy": (
            "Only locked internal_train labels, G1/G2 outputs and external CPubMed-KG one-hop "
            "evidence are used. Internal_val selects the configuration. Official dev is neither "
            "read nor emitted. KG never changes the fixed ICD answer vocabulary."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
