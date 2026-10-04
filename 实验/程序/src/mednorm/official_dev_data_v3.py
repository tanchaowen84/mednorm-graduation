"""Build frozen P1-003 inputs for the CHIP-CDN official dev split."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from mednorm.data import sha256_file
from mednorm.final_evaluation_v3 import final_evaluation_contract

JsonRow = dict[str, Any]
CONFIG_ID = "O4-FUSION-HGB15-W8-v1"


def _read_json(path: Path) -> JsonRow:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[JsonRow]:
    rows: list[JsonRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


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


def build_official_dev_atomic_input(
    *,
    dev_path: Path,
    retrieval_manifest_path: Path,
    output_dir: Path,
) -> JsonRow:
    """Strip labels before the pinned Qwen atomic-decomposition step."""
    contract = final_evaluation_contract("official_dev")
    authorization = _read_json(retrieval_manifest_path)
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": contract.retrieval_stage,
        "protocol": "Strict-ICD",
        "evaluation_split": contract.evaluation_split,
        "selection_complete": True,
        "audit_used": contract.audit_used,
        "official_dev_touched": contract.official_dev_touched,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"official dev authorization field {field} is invalid")
    if authorization.get("input_sha256", {}).get("official_dev") != sha256_file(
        dev_path
    ):
        raise ValueError("official dev differs from the authorized retrieval input")

    rows = _read_jsonl(dev_path)
    label_free: list[JsonRow] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        text = row.get("text")
        if (
            not isinstance(sample_id, str)
            or not sample_id.startswith("dev-")
            or sample_id in seen
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError(f"invalid official dev row {index}")
        seen.add(sample_id)
        label_free.append({"id": sample_id, "text": text})

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "official_dev_texts.jsonl.gz"
    _write_gzip(output_path, label_free)
    manifest: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": contract.atomic_input_stage,
        "protocol": "Strict-ICD",
        "evaluation_split": contract.evaluation_split,
        "selection_complete": True,
        "labels_included": False,
        "audit_used": contract.audit_used,
        "official_dev_touched": contract.official_dev_touched,
        "sample_count": len(label_free),
        "input_sha256": {
            "official_dev": sha256_file(dev_path),
            "authorization": sha256_file(retrieval_manifest_path),
        },
        "output_sha256": {output_path.name: sha256_file(output_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

