"""Build full-train text/profile ranker data and dev prototype evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mednorm.official_fulltrain_v3 import (
    build_official_profile_data,
    build_official_prototype_evidence,
    build_official_ranking_data,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever-artifact-dir",
        type=Path,
        default=ROOT / "artifacts/phase1_v3/strict/official_fulltrain/retriever",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/processed/phase1_v3/strict/official_fulltrain",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ROOT / "data/processed/phase1"
    ranking_dir = args.output_root / "ranking"
    profile_dir = args.output_root / "profile"
    prototype_dir = args.output_root / "prototype"
    ranking = build_official_ranking_data(
        source_dir=source,
        retriever_artifact_dir=args.retriever_artifact_dir,
        output_dir=ranking_dir,
    )
    profile = build_official_profile_data(
        catalog_path=source / "icd_catalog.jsonl",
        ranking_data_dir=ranking_dir,
        retriever_artifact_dir=args.retriever_artifact_dir,
        aliases_path=source / "kg_index/aliases.jsonl",
        edges_path=source / "kg_index/icd_edges.jsonl",
        output_dir=profile_dir,
    )
    prototype = build_official_prototype_evidence(
        catalog_path=source / "icd_catalog.jsonl",
        train_path=source / "train.jsonl",
        ranking_data_dir=ranking_dir,
        output_dir=prototype_dir,
    )
    print(
        json.dumps(
            {
                "ranking": ranking["counts"],
                "profile": profile["counts"],
                "prototype": prototype["counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

