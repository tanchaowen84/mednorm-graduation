"""Build frozen full-train and official-dev sparse retrieval inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mednorm.official_fulltrain_v3 import build_official_retrieval_data

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "data/processed/phase1"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/phase1_v3/strict/official_fulltrain/retrieval",
    )
    parser.add_argument("--retrieval-k", type=int, default=800)
    parser.add_argument("--hard-negative-count", type=int, default=8)
    parser.add_argument("--easy-negative-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_official_retrieval_data(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        retrieval_k=args.retrieval_k,
        hard_negative_count=args.hard_negative_count,
        easy_negative_count=args.easy_negative_count,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "counts": result["counts"],
                "official_dev_sparse": result["retrieval"]["official_dev_sparse"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

