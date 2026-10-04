"""Entry point for the frozen official-validation version."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="mednorm")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    import importlib.util

    path = args.project_root / "scripts/validate_official_dev_result_v3.py"
    spec = importlib.util.spec_from_file_location("official_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate_result(
        args.project_root / "artifacts/phase1_v3/strict/official_fulltrain/official_dev_fusion"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
