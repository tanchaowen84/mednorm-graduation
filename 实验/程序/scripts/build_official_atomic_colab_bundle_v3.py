"""Create the deterministic label-free official-dev atomic bundle."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts/phase1_v3/strict/official_fulltrain/official_atomic_bundle.zip"
)
ARCHIVE_ROOT = "official_atomic_v3"
FIXED_TIMESTAMP = (2026, 8, 23, 0, 0, 0)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def selected_files() -> list[Path]:
    data_dir = (
        ROOT
        / "data/processed/phase1_v3/strict/official_fulltrain/atomic_input"
    )
    return [
        ROOT / "colab/generate_atomic_mentions_v3.py",
        ROOT / "src/mednorm/__init__.py",
        ROOT / "src/mednorm/atomic_mentions_v3.py",
        data_dir / "manifest.json",
        data_dir / "official_dev_texts.jsonl.gz",
    ]


def bundle_manifest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "spec_id": "MEDNORM-P1-003",
        "config_id": "O4-FUSION-HGB15-W8-v1",
        "stage": "O7_OFFICIAL_DEV_ATOMIC_GENERATION",
        "protocol": "Strict-ICD",
        "data_split": "official_dev",
        "labels_included": False,
        "inference_only": True,
        "selection_complete": True,
        "audit_included": True,
        "official_dev_included": True,
        "files": entries,
    }


def main() -> None:
    files = selected_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing official atomic bundle inputs: {missing}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content, compresslevel=9)
            entries.append(
                {"path": relative, "size": len(content), "sha256": sha256_bytes(content)}
            )
        manifest = (
            json.dumps(
                bundle_manifest(entries),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json", FIXED_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, manifest, compresslevel=9)
    print(OUTPUT)


if __name__ == "__main__":
    main()
