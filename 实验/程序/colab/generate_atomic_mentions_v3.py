"""Generate label-free RR-style atomic mentions with a pinned local LLM."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import random
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from mednorm.atomic_mentions_v3 import extract_atomic_mentions

JsonRow = dict[str, Any]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

SYSTEM_PROMPT = """你是临床诊断短语的结构拆分器，不负责诊断，也不负责标准化。
任务是把一个原始诊断短语拆成若干单一疾病含义的原子表达。

必须遵守：
1. 只输出 JSON 字符串数组，不输出解释或代码块。
2. 每个元素只表达一个疾病、病变、症状、检查或治疗含义。
3. 可以把原文中的共同部位和修饰词复制到不同元素，但不得加入原文没有的汉字、字母或数字。
4. 不得改写同义词，不得输出 ICD 标准名，不得依据医学知识新增诊断。
5. 保留原词和原顺序；如果只有一个含义，返回只含原文的数组。

示例：
输入：左膝退变伴游离体
输出：["左膝退变","左膝游离体"]

输入：右膝关节盘状半月板撕裂
输出：["右膝关节盘状半月板","右膝关节半月板撕裂"]

输入：糖尿病反复低血糖;骨质疏松;高血压冠心病不稳定心绞痛
输出：["糖尿病反复低血糖","骨质疏松","高血压","冠心病","冠心病不稳定心绞痛"]

输入：肺部恶性肿瘤
输出：["肺部恶性肿瘤"]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-split",
        choices=("fit", "tune", "audit", "official_dev"),
        default="tune",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--maximum-mentions", type=int, default=16)
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path: Path) -> JsonRow:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_gzip_jsonl(path: Path) -> list[JsonRow]:
    rows: list[JsonRow] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _open_deterministic_gzip_text(path: Path) -> tuple[BinaryIO, gzip.GzipFile, TextIO]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    raw, compressed, text = _open_deterministic_gzip_text(path)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_inputs(
    data_dir: Path, *, data_split: str = "tune"
) -> tuple[JsonRow, list[JsonRow]]:
    if data_split not in {"fit", "tune", "audit", "official_dev"}:
        raise ValueError(
            "atomic data_split must be fit, tune, audit or official_dev"
        )
    manifest_path = data_dir / "manifest.json"
    rows_path = data_dir / f"{data_split}_texts.jsonl.gz"
    manifest = read_json(manifest_path)
    frozen_evaluation = data_split in {"audit", "official_dev"}
    stage = {
        "fit": "O4_ATOMIC_INPUT",
        "tune": "O4_ATOMIC_INPUT",
        "audit": "O5_AUDIT_ATOMIC_INPUT",
        "official_dev": "O7_OFFICIAL_DEV_ATOMIC_INPUT",
    }[data_split]
    required = {
        "spec_id": "MEDNORM-P1-003",
        "stage": stage,
        "protocol": "Strict-ICD",
        "labels_included": False,
        "audit_used": frozen_evaluation,
        "official_dev_touched": data_split == "official_dev",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"unexpected atomic input field {key}")
    split_required: dict[str, Any]
    if data_split == "tune":
        split_required = {"selection_split": "tune"}
    elif data_split == "fit":
        split_required = {"data_split": "fit", "train_split": "fit"}
    elif data_split == "audit":
        split_required = {
            "config_id": "O4-FUSION-HGB15-W8-v1",
            "evaluation_split": "audit",
            "selection_complete": True,
        }
    else:
        split_required = {
            "config_id": "O4-FUSION-HGB15-W8-v1",
            "evaluation_split": "official_dev",
            "selection_complete": True,
        }
    for key, expected in split_required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"unexpected atomic input field {key}")
    if manifest.get("output_sha256", {}).get(rows_path.name) != sha256_file(rows_path):
        raise ValueError("atomic input hash mismatch")
    rows = read_gzip_jsonl(rows_path)
    if len(rows) != int(manifest["sample_count"]):
        raise ValueError("atomic input count differs from manifest")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        if (
            set(row) != {"id", "text"}
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen
            or not isinstance(row.get("text"), str)
            or not row["text"]
        ):
            raise ValueError(f"atomic input row {index} must be label-free")
        seen.add(sample_id)
    return manifest, rows


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prompts_for_rows(tokenizer: Any, rows: Sequence[JsonRow]) -> list[str]:
    prompts: list[str] = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"输入：{row['text']}\n输出："},
        ]
        prompts.append(
            str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        )
    return prompts


@torch.inference_mode()
def generate_batch(
    model: Any,
    tokenizer: Any,
    rows: Sequence[JsonRow],
    *,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    prompts = prompts_for_rows(tokenizer, rows)
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return cast(
        list[str],
        tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True),
    )


def run(args: argparse.Namespace) -> JsonRow:
    counts = (
        args.batch_size,
        args.max_input_length,
        args.max_new_tokens,
        args.maximum_mentions,
    )
    if min(counts) < 1 or args.smoke_limit < 0:
        raise ValueError("generation budgets must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required unless --allow-cpu is set")
    started = time.time()
    set_seed(args.seed)
    manifest, rows = validate_inputs(args.data_dir, data_split=args.data_split)
    if args.smoke_limit:
        rows = rows[: args.smoke_limit]
    mode = "smoke" if args.smoke_limit else "full"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer: Any = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_name,
        revision=args.model_revision,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)  # type: ignore[arg-type]
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)

    output_rows: list[JsonRow] = []
    statuses: Counter[str] = Counter()
    mention_counts: Counter[int] = Counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        outputs = generate_batch(
            model,
            tokenizer,
            batch,
            device=device,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )
        for row, raw_output in zip(batch, outputs, strict=True):
            mentions, status = extract_atomic_mentions(
                str(row["text"]),
                raw_output,
                maximum_mentions=args.maximum_mentions,
            )
            statuses[status] += 1
            mention_counts[len(mentions)] += 1
            output_rows.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "atomic_mentions": list(mentions),
                    "parse_status": status,
                    "raw_output": raw_output,
                }
            )
        print(
            json.dumps(
                {"processed": len(output_rows), "total": len(rows)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.data_split}_atomic_mentions.jsonl.gz"
    write_gzip_jsonl(predictions_path, output_rows)
    report: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "stage": {
            "fit": "O4_ATOMIC_GENERATION",
            "tune": "O4_ATOMIC_GENERATION",
            "audit": "O5_AUDIT_ATOMIC_GENERATION",
            "official_dev": "O7_OFFICIAL_DEV_ATOMIC_GENERATION",
        }[args.data_split],
        "protocol": "Strict-ICD",
        "mode": mode,
        "labels_used": False,
        "audit_used": args.data_split in {"audit", "official_dev"},
        "official_dev_touched": args.data_split == "official_dev",
        "model": {
            "name": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_revision,
            "inference_only": True,
        },
        "prompt": {
            "sha256": sha256_text(SYSTEM_PROMPT),
            "policy": "source-supported atomic decomposition without normalization",
        },
        "generation": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "maximum_mentions": args.maximum_mentions,
            "do_sample": False,
        },
        "counts": {
            "samples": len(output_rows),
            "parse_status": dict(sorted(statuses.items())),
            "mention_count": {
                str(count): frequency for count, frequency in sorted(mention_counts.items())
            },
        },
        "input_manifest": manifest,
        "input_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "output_sha256": {predictions_path.name: sha256_file(predictions_path)},
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": device.type,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": str(dtype),
        },
    }
    if args.data_split == "tune":
        report["selection_split"] = "tune"
    elif args.data_split == "fit":
        report["data_split"] = "fit"
        report["train_split"] = "fit"
    else:
        report["evaluation_split"] = args.data_split
        report["config_id"] = manifest["config_id"]
    write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "mode": mode,
                "samples": len(output_rows),
                "parse_status": report["counts"]["parse_status"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
