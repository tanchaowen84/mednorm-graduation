"""Create the deterministic official full-train retriever Colab bundle."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts/phase1_v3/strict/official_fulltrain/official_retriever_bundle.zip"
)
ARCHIVE_ROOT = "official_fulltrain_v3"
FIXED_TIMESTAMP = (2026, 8, 23, 0, 0, 0)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def selected_files() -> list[Path]:
    retrieval = (
        ROOT / "data/processed/phase1_v3/strict/official_fulltrain/retrieval"
    )
    return [
        ROOT / "colab/train_biencoder.py",
        ROOT / "colab/train_official_biencoder_v3.py",
        ROOT / "colab/export_fp16_checkpoint_v3.py",
        ROOT / "src/mednorm/__init__.py",
        ROOT / "src/mednorm/candidate_fusion.py",
        ROOT / "src/mednorm/data.py",
        ROOT / "src/mednorm/dense_data.py",
        ROOT / "src/mednorm/dense_training.py",
        ROOT / "src/mednorm/final_evaluation_v3.py",
        ROOT / "src/mednorm/metrics.py",
        ROOT / "src/mednorm/neural_data_v2.py",
        ROOT / "src/mednorm/official_contract_v3.py",
        ROOT / "src/mednorm/official_fulltrain_v3.py",
        ROOT / "src/mednorm/retrieval.py",
        ROOT / "src/mednorm/text.py",
        ROOT / "data/processed/phase1/icd_catalog.jsonl",
        retrieval / "manifest.json",
        retrieval / "official_train_queries.jsonl.gz",
        retrieval / "official_train_sparse.jsonl.gz",
        retrieval / "official_dev_sparse.jsonl.gz",
    ]


def main() -> None:
    files = selected_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing official retriever inputs: {missing}")
    entries: list[dict[str, str | int]] = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)
            entries.append(
                {"path": relative, "size": len(content), "sha256": sha256_bytes(content)}
            )
        bundle_manifest = {
            "spec_id": "MEDNORM-P1-003",
            "config_id": "O4-FUSION-HGB15-W8-v1",
            "stage": "O7_OFFICIAL_DEV_RETRIEVAL",
            "protocol": "Strict-ICD",
            "train_split": "official_train",
            "selection_split": None,
            "evaluation_split": "official_dev",
            "official_dev_included": True,
            "persistent_checkpoint_precision": "float16",
            "files": entries,
        }
        content = (
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode()
        info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json", FIXED_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)
    print(OUTPUT)


if __name__ == "__main__":
    main()
