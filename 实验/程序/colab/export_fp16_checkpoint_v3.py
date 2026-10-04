"""Export a verified inference-only FP16 copy of a trained Transformers checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

JsonRow = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_kind_from_config(config: JsonRow) -> str:
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
    ):
        raise ValueError("checkpoint must declare exactly one architecture")
    architecture = architectures[0]
    if architecture.endswith("ForSequenceClassification"):
        return "sequence_classification"
    if architecture.endswith("Model"):
        return "encoder"
    raise ValueError(f"unsupported checkpoint architecture: {architecture}")


def _files(path: Path) -> list[Path]:
    return [candidate for candidate in sorted(path.rglob("*")) if candidate.is_file()]


def _file_manifest(path: Path) -> dict[str, JsonRow]:
    return {
        file.relative_to(path).as_posix(): {
            "bytes": file.stat().st_size,
            "sha256": sha256_file(file),
        }
        for file in _files(path)
    }


def _weight_dtypes(path: Path) -> set[str]:
    from safetensors import safe_open

    result: set[str] = set()
    weight_paths = sorted(path.glob("*.safetensors"))
    if not weight_paths:
        raise FileNotFoundError("exported checkpoint has no safetensors weights")
    for weight_path in weight_paths:
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safetensors is not iterable
                tensor = handle.get_tensor(name)
                if tensor.is_floating_point():
                    result.add(str(tensor.dtype))
    return result


def export_fp16_checkpoint(input_dir: Path, output_dir: Path) -> JsonRow:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("FP16 export must not overwrite the FP32 checkpoint")
    config_path = input_dir / "config.json"
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("checkpoint config must contain an object")
    config: JsonRow = value
    kind = model_kind_from_config(config)

    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    loader = (
        AutoModelForSequenceClassification
        if kind == "sequence_classification"
        else AutoModel
    )
    model = loader.from_pretrained(
        input_dir,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        input_dir, local_files_only=True, use_fast=True
    )
    model.eval().to(dtype=torch.float16)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(output_dir)
    del model

    dtypes = _weight_dtypes(output_dir)
    if dtypes != {"torch.float16"}:
        raise ValueError(f"FP16 export contains unexpected floating dtypes: {dtypes}")
    input_files = _file_manifest(input_dir)
    output_files = _file_manifest(output_dir)
    input_weight_bytes = sum(
        int(metadata["bytes"])
        for name, metadata in input_files.items()
        if name.endswith(".safetensors")
    )
    output_weight_bytes = sum(
        int(metadata["bytes"])
        for name, metadata in output_files.items()
        if name.endswith(".safetensors")
    )
    if output_weight_bytes > int(input_weight_bytes * 0.56) + 1024 * 1024:
        raise ValueError("FP16 export did not reduce checkpoint weights as expected")

    # A real local reload catches missing shards/config/tokenizer files before Drive upload.
    reloaded = loader.from_pretrained(
        output_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    reloaded.eval()
    del reloaded
    report: JsonRow = {
        "format": "transformers_fp16_inference_v1",
        "model_kind": kind,
        "source_precision": "float32",
        "export_precision": "float16",
        "resume_training_supported": False,
        "inference_reload_verified": True,
        "floating_dtypes": sorted(dtypes),
        "input_weight_bytes": input_weight_bytes,
        "output_weight_bytes": output_weight_bytes,
        "compression_ratio": output_weight_bytes / max(1, input_weight_bytes),
        "input_files": input_files,
        "output_files": output_files,
    }
    report_path = output_dir / "FP16_EXPORT_MANIFEST.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    result = export_fp16_checkpoint(args.input_dir, args.output_dir)
    print(
        json.dumps(
            {
                "model_kind": result["model_kind"],
                "output_weight_bytes": result["output_weight_bytes"],
                "compression_ratio": result["compression_ratio"],
                "reload_verified": result["inference_reload_verified"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
