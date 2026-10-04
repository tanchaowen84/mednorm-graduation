"""
[INPUT] Read-only official CHIP-CDN JSON files and ICD XLSX workbook.
[OUTPUT] Leak-free JSONL datasets, ICD catalog, audit report and reproducibility manifest.
[POS] Materializes the locked Phase 1 data contract without using dev labels as candidates.
[UPDATE] Keep output schema, hashes and SPEC_PHASE1 data rules synchronized.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mednorm.data import CDNSample, create_internal_split, load_cdn_split, sha256_file
from mednorm.icd import CatalogCoverage, ICDCatalog, catalog_coverage, load_icd_catalog

TRAIN_FILENAME = "CHIP-CDN_train.json"
DEV_FILENAME = "CHIP-CDN_dev.json"
TEST_FILENAME = "CHIP-CDN_test.json"
ICD_FILENAME = "国际疾病分类 ICD-10北京临床版v601.xlsx"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_record(sample: CDNSample, catalog: ICDCatalog, split: str) -> dict[str, Any]:
    codes = [list(catalog.codes_for(label)) for label in sample.labels]
    return {
        "id": sample.sample_id,
        "split": split,
        "text": sample.text,
        "labels": list(sample.labels),
        "label_codes": codes,
        "all_labels_in_icd": bool(sample.labels)
        and all(bool(label_codes) for label_codes in codes),
    }


def _coverage_record(coverage: CatalogCoverage) -> dict[str, Any]:
    return {
        "sample_count": coverage.sample_count,
        "label_occurrence_count": coverage.total_label_occurrences,
        "covered_label_occurrence_count": coverage.covered_label_occurrences,
        "label_coverage": coverage.label_coverage,
        "fully_covered_sample_count": coverage.fully_covered_samples,
        "fully_covered_sample_rate": coverage.fully_covered_sample_rate,
        "unique_gold_count": coverage.unique_gold_names,
        "covered_unique_gold_count": coverage.covered_unique_gold_names,
        "oov_unique_count": len(coverage.oov_names),
        "oov_names": list(coverage.oov_names),
    }


def _label_count_distribution(samples: Sequence[CDNSample]) -> dict[str, int]:
    counts = Counter(len(sample.labels) for sample in samples)
    return {str(label_count): counts[label_count] for label_count in sorted(counts)}


def prepare_phase1_data(
    *,
    raw_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build every deterministic Phase 1 data artifact from immutable raw files."""
    input_paths = {
        "train": raw_dir / TRAIN_FILENAME,
        "dev": raw_dir / DEV_FILENAME,
        "test": raw_dir / TEST_FILENAME,
        "icd": raw_dir / ICD_FILENAME,
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required Phase 1 inputs: {', '.join(missing)}")

    train = load_cdn_split(input_paths["train"], "train")
    dev = load_cdn_split(input_paths["dev"], "dev")
    test = load_cdn_split(input_paths["test"], "test")
    catalog = load_icd_catalog(input_paths["icd"])
    internal_train, internal_val = create_internal_split(
        train, val_fraction=val_fraction, seed=seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_outputs: dict[str, tuple[Sequence[CDNSample], str]] = {
        "train.jsonl": (train, "train"),
        "internal_train.jsonl": (internal_train, "internal_train"),
        "internal_val.jsonl": (internal_val, "internal_val"),
        "dev.jsonl": (dev, "dev"),
        "test.jsonl": (test, "test"),
    }
    for filename, (samples, split) in dataset_outputs.items():
        _write_jsonl(
            output_dir / filename,
            (_sample_record(sample, catalog, split) for sample in samples),
        )

    _write_jsonl(
        output_dir / "icd_catalog.jsonl",
        (
            {
                "name": name,
                "primary_code": catalog.primary_code_for(name),
                "codes": list(catalog.codes_for(name)),
            }
            for name in catalog.names
        ),
    )

    report: dict[str, Any] = {
        "candidate_catalog": {
            "source_row_count": catalog.source_row_count,
            "unique_name_count": len(catalog.names),
            "multi_code_name_count": sum(
                len(codes) > 1 for codes in catalog.name_to_codes.values()
            ),
            "maximum_codes_per_name": max(
                len(codes) for codes in catalog.name_to_codes.values()
            ),
            "source_policy": "ICD only; train/dev gold labels are never appended",
        },
        "train": _coverage_record(catalog_coverage(train, catalog)),
        "internal_train": _coverage_record(catalog_coverage(internal_train, catalog)),
        "internal_val": _coverage_record(catalog_coverage(internal_val, catalog)),
        "dev": _coverage_record(catalog_coverage(dev, catalog)),
        "label_count_distribution": {
            "train": _label_count_distribution(train),
            "internal_train": _label_count_distribution(internal_train),
            "internal_val": _label_count_distribution(internal_val),
            "dev": _label_count_distribution(dev),
            "test": _label_count_distribution(test),
        },
    }
    _write_json(output_dir / "audit.json", report)

    generated_paths = [
        *(output_dir / filename for filename in dataset_outputs),
        output_dir / "icd_catalog.jsonl",
        output_dir / "audit.json",
    ]
    manifest: dict[str, Any] = {
        "spec_id": "MEDNORM-P1-001",
        "seed": seed,
        "internal_validation_fraction": val_fraction,
        "input_sha256": {
            key: sha256_file(path) for key, path in input_paths.items()
        },
        "output_sha256": {
            path.name: sha256_file(path) for path in generated_paths
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report

