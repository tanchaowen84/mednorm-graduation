"""Apply the audit-selected raw fusion once to official CHIP-CDN dev."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, TextIO, TypeVar

import numpy as np

from mednorm.metrics import PredictionMetrics, evaluate_predictions

JsonRow = dict[str, Any]
T = TypeVar("T")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPONENT_ROOT = (
    ROOT / "artifacts/phase1_v3/strict/official_fulltrain/components"
)
DEFAULT_PROTOTYPE_PATH = (
    ROOT
    / "data/processed/phase1_v3/strict/official_fulltrain/prototype/"
    "official_dev_prototype_evidence.jsonl.gz"
)
DEFAULT_ATOMIC_ROOT = ROOT / "artifacts/phase1_v3/strict/official_fulltrain/atomic"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts/phase1_v3/strict/official_fulltrain/official_dev_fusion"
)
SEEDS = (2026, 2027, 2028)
CANDIDATE_K = 400
CONFIG_ID = "O4-FUSION-HGB15-W8-v1"
SELECTED_METHOD = "raw_fusion_ensemble"
RAW_WEIGHTS = (1.0, 0.75, 0.1, 1.5, 0.75)
RAW_THRESHOLDS = (0.75, 0.68, 0.655, 0.71, 0.76, 0.585)
THRESHOLDS = np.arange(0.0, 1.0001, 0.005, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--prototype-path", type=Path, default=DEFAULT_PROTOTYPE_PATH)
    parser.add_argument("--atomic-root", type=Path, default=DEFAULT_ATOMIC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def select_predictions(
    learned_predictions: Sequence[T], raw_predictions: Sequence[T]
) -> Sequence[T]:
    """Keep the method selected before official dev was opened."""
    if len(learned_predictions) != len(raw_predictions):
        raise ValueError("fusion prediction lengths differ")
    return raw_predictions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[JsonRow]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object row")
    return rows


def _open_gzip(path: Path) -> tuple[BinaryIO, gzip.GzipFile, TextIO]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def threshold_indices(values: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [int(np.argmin(np.abs(THRESHOLDS - float(value)))) for value in values],
        dtype=np.int16,
    )


def decode(
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: np.ndarray,
    names: list[list[str]],
) -> list[list[str]]:
    predictions: list[list[str]] = []
    for index, values in enumerate(scores):
        selected = [
            int(candidate)
            for candidate in np.flatnonzero(
                values + 1e-7 >= THRESHOLDS[thresholds[groups[index]]]
            )
        ]
        if not selected:
            selected = [int(np.argmax(values))]
        predictions.append([names[index][candidate] for candidate in selected])
    return predictions


def raw_fusion_scores(signals: np.ndarray) -> np.ndarray:
    if signals.ndim != 3 or signals.shape[0] != len(RAW_WEIGHTS):
        raise ValueError("raw fusion expects five aligned score tensors")
    return np.average(
        signals.astype(np.float32, copy=False),
        axis=0,
        weights=np.asarray(RAW_WEIGHTS, dtype=np.float32),
    ).astype(np.float32)


def majority_count_classes(seed_classes: np.ndarray) -> np.ndarray:
    if seed_classes.ndim != 2 or len(seed_classes) != len(SEEDS):
        raise ValueError("count ensemble expects exactly three seed rows")
    if np.any((seed_classes < 0) | (seed_classes > 2)):
        raise ValueError("count classes must be 0, 1 or 2")
    return np.asarray(
        [
            int(np.bincount(seed_classes[:, index], minlength=3).argmax())
            for index in range(seed_classes.shape[1])
        ],
        dtype=np.int8,
    )


def subset_metrics(
    indices: Sequence[int],
    *,
    gold: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> PredictionMetrics:
    return evaluate_predictions(
        [gold[index] for index in indices],
        [predictions[index] for index in indices],
    )


def segmented_metrics(
    *,
    gold: list[list[str]],
    predictions: list[list[str]],
    covered: Sequence[bool],
    active: Sequence[bool],
) -> JsonRow:
    segments = {
        "all": list(range(len(gold))),
        "icd_covered": [index for index, value in enumerate(covered) if value],
        "single_entity": [
            index for index, labels in enumerate(gold) if len(set(labels)) == 1
        ],
        "multi_entity": [
            index for index, labels in enumerate(gold) if len(set(labels)) > 1
        ],
        "atomic_active": [index for index, value in enumerate(active) if value],
        "atomic_inactive": [index for index, value in enumerate(active) if not value],
    }
    return {
        name: asdict(subset_metrics(indices, gold=gold, predictions=predictions))
        for name, indices in segments.items()
    }


def component_paths(seed: int, component_root: Path) -> dict[str, Path]:
    return {
        "o2": component_root
        / f"o2_macbert_seed{seed}/official_dev_scored_predictions.jsonl.gz",
        "kg": component_root
        / f"kg_atomic_macbert_seed{seed}/official_dev_scored_predictions.jsonl.gz",
        "bge": component_root
        / f"bge_reranker_seed{seed}/official_dev_scored_predictions.jsonl.gz",
    }


def validate_atomic_report(atomic_root: Path) -> Path:
    report_path = atomic_root / "report.json"
    predictions_path = atomic_root / "official_dev_atomic_mentions.jsonl.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O7_OFFICIAL_DEV_ATOMIC_GENERATION",
        "protocol": "Strict-ICD",
        "evaluation_split": "official_dev",
        "labels_used": False,
        "audit_used": True,
        "official_dev_touched": True,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"unexpected official atomic report field {field}")
    if (
        report.get("counts", {}).get("samples") != 2000
        or report.get("output_sha256", {}).get(predictions_path.name)
        != sha256_file(predictions_path)
    ):
        raise ValueError("official atomic output count or hash differs")
    return predictions_path


def assemble_seed(
    seed: int,
    *,
    component_root: Path,
    prototype_path: Path,
    atomic_rows: Sequence[JsonRow],
) -> JsonRow:
    paths = component_paths(seed, component_root)
    records = {name: read_rows(path) for name, path in paths.items()}
    prototype_rows = read_rows(prototype_path)
    row_count = len(records["o2"])
    if row_count != 2000 or any(len(rows) != row_count for rows in records.values()):
        raise ValueError("official fusion ranker row counts differ from 2000")
    if len(prototype_rows) != row_count or len(atomic_rows) != row_count:
        raise ValueError("official fusion evidence row counts differ")

    signals = np.zeros((5, row_count, CANDIDATE_K), dtype=np.float32)
    ids: list[str] = []
    names: list[list[str]] = []
    gold: list[list[str]] = []
    covered: list[bool] = []
    active: list[bool] = []
    atomic_mentions: list[list[str]] = []
    count_classes = np.zeros(row_count, dtype=np.int8)
    for index, aligned in enumerate(
        zip(
            records["o2"],
            records["kg"],
            records["bge"],
            prototype_rows,
            atomic_rows,
            strict=True,
        )
    ):
        o2, kg, bge, prototype, atomic = aligned
        sample_id = o2.get("id")
        if not isinstance(sample_id, str) or not all(
            row.get("id") == sample_id for row in (kg, bge, prototype, atomic)
        ):
            raise ValueError(f"official fusion inputs are misaligned at row {index}")
        candidate_names = [str(row["name"]) for row in o2["scored_candidates"]]
        if len(candidate_names) != CANDIDATE_K:
            raise ValueError(f"official fusion candidate count differs at row {index}")
        if any(
            candidate_names
            != [str(candidate["name"]) for candidate in row["scored_candidates"]]
            for row in (kg, bge)
        ):
            raise ValueError(f"official ranker candidates differ at row {index}")
        if candidate_names != [str(row["name"]) for row in prototype["features"]]:
            raise ValueError(f"official prototype candidates differ at row {index}")
        gold_labels = [str(label) for label in o2["gold_labels"]]
        if any(
            [str(label) for label in row["gold_labels"]] != gold_labels
            for row in (kg, bge)
        ):
            raise ValueError(f"official ranker gold labels differ at row {index}")
        count_prediction = int(o2["count_prediction"])
        if count_prediction not in (0, 1, 2) or any(
            int(row["count_prediction"]) != count_prediction for row in (kg, bge)
        ):
            raise ValueError(f"official count prediction differs at row {index}")
        mentions = [str(value) for value in atomic.get("atomic_mentions", [])]
        if not mentions:
            raise ValueError(f"official atomic mentions are empty at row {index}")

        ids.append(sample_id)
        names.append(candidate_names)
        gold.append(gold_labels)
        covered.append(bool(o2["all_labels_in_icd"]))
        active.append(len(mentions) > 1)
        atomic_mentions.append(mentions)
        count_classes[index] = count_prediction
        signals[0, index] = [row["text_score"] for row in o2["scored_candidates"]]
        signals[1, index] = [row["text_score"] for row in kg["scored_candidates"]]
        signals[2, index] = [row["text_score"] for row in bge["scored_candidates"]]
        signals[3, index] = [
            row["retrieval_score"] for row in o2["scored_candidates"]
        ]
        signals[4, index] = [
            row["prototype_score"] for row in prototype["features"]
        ]
    scores = raw_fusion_scores(signals)
    groups = 2 * count_classes + np.asarray(active, dtype=np.int8)
    predictions = decode(scores, groups, threshold_indices(RAW_THRESHOLDS), names)
    return {
        "ids": ids,
        "names": names,
        "gold": gold,
        "covered": covered,
        "active": active,
        "atomic_mentions": atomic_mentions,
        "count_classes": count_classes,
        "groups": groups,
        "scores": scores,
        "predictions": predictions,
        "metrics": segmented_metrics(
            gold=gold,
            predictions=predictions,
            covered=covered,
            active=active,
        ),
        "paths": paths,
    }


def main() -> None:
    args = parse_args()
    atomic_path = validate_atomic_report(args.atomic_root)
    atomic_rows = read_rows(atomic_path)
    seed_results = [
        assemble_seed(
            seed,
            component_root=args.component_root,
            prototype_path=args.prototype_path,
            atomic_rows=atomic_rows,
        )
        for seed in SEEDS
    ]
    reference = seed_results[0]
    for result in seed_results[1:]:
        for field in (
            "ids",
            "names",
            "gold",
            "covered",
            "active",
            "atomic_mentions",
        ):
            if result[field] != reference[field]:
                raise ValueError(f"official fusion seed field {field} differs")
    seed_count_classes = np.stack(
        [np.asarray(result["count_classes"], dtype=np.int8) for result in seed_results]
    )
    ensemble_count = majority_count_classes(seed_count_classes)
    ensemble_groups = 2 * ensemble_count + np.asarray(reference["active"], dtype=np.int8)
    ensemble_scores = np.mean(
        np.stack([np.asarray(result["scores"]) for result in seed_results]), axis=0
    )
    ensemble_predictions = decode(
        ensemble_scores,
        ensemble_groups,
        threshold_indices(RAW_THRESHOLDS),
        reference["names"],
    )
    ensemble_metrics = segmented_metrics(
        gold=reference["gold"],
        predictions=ensemble_predictions,
        covered=reference["covered"],
        active=reference["active"],
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_root / "official_dev_predictions.jsonl.gz"
    write_rows(
        predictions_path,
        (
            {
                "id": sample_id,
                "gold_labels": gold,
                "all_labels_in_icd": covered,
                "atomic_mentions": mentions,
                "atomic_active": active,
                "count_group": int(group),
                "selected_method": SELECTED_METHOD,
                "prediction": prediction,
                "scored_candidates": [
                    {"name": name, "raw_score": float(score)}
                    for name, score in zip(names, scores, strict=True)
                ],
            }
            for (
                sample_id,
                gold,
                covered,
                mentions,
                active,
                group,
                prediction,
                names,
                scores,
            ) in zip(
                reference["ids"],
                reference["gold"],
                reference["covered"],
                reference["atomic_mentions"],
                reference["active"],
                ensemble_groups,
                ensemble_predictions,
                reference["names"],
                ensemble_scores,
                strict=True,
            )
        ),
    )
    seed_f1 = [
        float(result["metrics"]["all"]["micro_f1"]) for result in seed_results
    ]
    report: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O7_OFFICIAL_DEV_FINAL_FUSION",
        "protocol": "Strict-ICD",
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
        "candidate_k": CANDIDATE_K,
        "seeds": list(SEEDS),
        "selected_method": SELECTED_METHOD,
        "selection_policy": (
            "Raw fusion weights, six count/atomic thresholds and the three-seed "
            "ensemble were fixed on pre-official data. Official dev was evaluated once."
        ),
        "method": {
            "weights": list(RAW_WEIGHTS),
            "thresholds": list(RAW_THRESHOLDS),
            "count_ensemble": "majority vote across seeds 2026, 2027 and 2028",
            "score_ensemble": "arithmetic mean across the three frozen seed scores",
        },
        "official_dev": ensemble_metrics,
        "three_seed": {
            "micro_f1": seed_f1,
            "mean": float(np.mean(seed_f1)),
            "std": float(np.std(seed_f1, ddof=1)),
        },
        "input_sha256": {
            "atomic": sha256_file(atomic_path),
            "prototype": sha256_file(args.prototype_path),
            **{
                f"seed{seed}_{name}": sha256_file(path)
                for seed, result in zip(SEEDS, seed_results, strict=True)
                for name, path in result["paths"].items()
            },
        },
        "output_sha256": {predictions_path.name: sha256_file(predictions_path)},
    }
    write_json(args.output_root / "report.json", report)
    print(
        json.dumps(
            {
                "selected_method": SELECTED_METHOD,
                "official_dev": ensemble_metrics["all"],
                "seed_micro_f1": seed_f1,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
