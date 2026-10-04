"""Independently recompute the frozen official-dev headline metrics."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mednorm.metrics import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = (
    ROOT / "artifacts/phase1_v3/strict/official_fulltrain/official_dev_fusion"
)
SELECTED_METHOD = "raw_fusion_ensemble"
JsonRow = dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(result_root: Path, *, expected_rows: int = 2000) -> JsonRow:
    report_path = result_root / "report.json"
    predictions_path = result_root / "official_dev_predictions.jsonl.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "stage": "O7_OFFICIAL_DEV_FINAL_FUSION",
        "evaluation_split": "official_dev",
        "selected_method": SELECTED_METHOD,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"unexpected final official report field {field}")
    if report.get("output_sha256", {}).get(predictions_path.name) != sha256_file(
        predictions_path
    ):
        raise ValueError("final official prediction hash differs")
    with gzip.open(predictions_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != expected_rows:
        raise ValueError("final official prediction row count differs")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids) or any(
        row.get("selected_method") != SELECTED_METHOD for row in rows
    ):
        raise ValueError("final official IDs or selected methods differ")
    gold = [[str(value) for value in row["gold_labels"]] for row in rows]
    predictions = [[str(value) for value in row["prediction"]] for row in rows]
    metrics = asdict(evaluate_predictions(gold, predictions))
    if metrics != report.get("official_dev", {}).get("all"):
        raise ValueError("independently recomputed official metrics differ")
    return {
        "status": "PASS",
        "evaluation_split": "official_dev",
        "selected_method": SELECTED_METHOD,
        "rows": len(rows),
        "prediction_sha256": sha256_file(predictions_path),
        "recomputed_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    validation = validate_result(args.result_root)
    output_path = args.result_root / "independent_validation.json"
    output_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
