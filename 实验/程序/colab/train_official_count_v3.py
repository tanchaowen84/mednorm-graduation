"""Retrain a frozen count seed on all official train rows and score dev once."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mednorm.official_contract_v3 import CONFIG_ID, frozen_epoch

try:
    from colab import train_count_v3 as base
except ImportError:
    import train_count_v3 as base  # type: ignore[import-not-found,no-redef]

JsonRow = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="hfl/chinese-macbert-large")
    parser.add_argument("--model-revision", default=base.DEFAULT_MODEL_REVISION)
    parser.add_argument("--planned-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced", "sqrt_balanced"),
        default="none",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def validate_inputs(data_dir: Path) -> tuple[JsonRow, list[JsonRow], list[JsonRow]]:
    manifest = base.read_json(data_dir / "manifest.json")
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O6_OFFICIAL_FULLTRAIN_RANKING_DATA",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"unexpected official count input field {field}")
    train_path = data_dir / "official_train_count.jsonl.gz"
    dev_path = data_dir / "official_dev_candidates.jsonl.gz"
    for path in (train_path, dev_path):
        if manifest.get("output_sha256", {}).get(path.name) != base.sha256_file(path):
            raise ValueError(f"official count input hash mismatch for {path.name}")
    train_rows = base.read_gzip_jsonl(train_path)
    dev_bundles = base.read_gzip_jsonl(dev_path)
    counts = manifest.get("counts", {})
    if len(train_rows) != int(counts.get("official_train_samples", -1)):
        raise ValueError("official count train rows differ from manifest")
    if len(dev_bundles) != int(counts.get("official_dev_samples", -1)):
        raise ValueError("official count dev rows differ from manifest")
    train_ids: set[str] = set()
    for index, row in enumerate(train_rows):
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id.startswith("train-")
            or sample_id in train_ids
            or not isinstance(row.get("text"), str)
            or row.get("count_class") not in (0, 1, 2)
        ):
            raise ValueError(f"invalid official count train row {index}")
        train_ids.add(sample_id)
    dev_ids: set[str] = set()
    for index, bundle in enumerate(dev_bundles):
        sample_id = bundle.get("id")
        if (
            not isinstance(sample_id, str)
            or not sample_id.startswith("dev-")
            or sample_id in dev_ids
            or not isinstance(bundle.get("text"), str)
            or not isinstance(bundle.get("labels"), list)
            or not bundle["labels"]
        ):
            raise ValueError(f"invalid official count dev row {index}")
        dev_ids.add(sample_id)
    if train_ids & dev_ids:
        raise ValueError("official count train/dev IDs overlap")
    return manifest, train_rows, dev_bundles


def run(args: argparse.Namespace) -> JsonRow:
    if min(
        args.planned_epochs,
        args.batch_size,
        args.eval_batch_size,
        args.gradient_accumulation,
        args.max_length,
    ) < 1:
        raise ValueError("official count training values must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required unless --allow-cpu is set")
    selected_epoch = frozen_epoch("count", args.seed)
    if selected_epoch > args.planned_epochs:
        raise ValueError("frozen count epoch exceeds the original schedule")
    started = time.time()
    base.set_seed(args.seed)
    manifest, train_rows, dev_bundles = validate_inputs(args.data_dir)
    dev_rows = [
        {
            "id": bundle["id"],
            "text": bundle["text"],
            "count_class": min(len(bundle["labels"]), 3) - 1,
        }
        for bundle in dev_bundles
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer: Any = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_name, revision=args.model_revision, use_fast=True
    )
    train_loader = base.make_loader(
        base.tokenize_rows(tokenizer, train_rows, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    dev_loader = base.make_loader(
        base.tokenize_rows(tokenizer, dev_rows, args.max_length),
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        num_labels=3,
        ignore_mismatched_sizes=True,
        attn_implementation="sdpa",
    ).to(device)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.planned_epochs)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    weights = base.class_weights(
        [int(row["count_class"]) for row in train_rows],
        args.class_weighting,
        device,
    )
    amp_enabled, amp_dtype = base.amp_settings(device)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    history: list[JsonRow] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, selected_epoch + 1):
        model.train()
        loss_sum = 0.0
        update_count = 0
        for step, batch in enumerate(train_loader, start=1):
            inputs, labels = base._move_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                logits = model(**inputs).logits
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach().cpu())
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "optimizer_updates": update_count,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    dev_metrics, probabilities, _ = base.evaluate_model(
        model, dev_loader, device=device
    )
    predictions = probabilities.argmax(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "official_dev_count_predictions.jsonl.gz"
    base.write_gzip_jsonl(
        predictions_path,
        (
            {
                "id": row["id"],
                "count_prediction": int(prediction),
                "probabilities": [float(value) for value in probability],
            }
            for row, prediction, probability in zip(
                dev_rows, predictions, probabilities, strict=True
            )
        ),
    )
    model_dir = args.output_dir / "best_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    model_hash = base.sha256_file(model_dir / "model.safetensors")
    report: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O7_OFFICIAL_DEV_COUNT",
        "protocol": "Strict-ICD",
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
        "model": {
            "name": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_revision,
            "frozen_epoch": selected_epoch,
            "sha256": model_hash,
            "checkpoint_precision": "float32",
        },
        "training": {
            "seed": args.seed,
            "planned_epochs": args.planned_epochs,
            "executed_epochs": selected_epoch,
            "official_train_samples": len(train_rows),
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "class_weighting": args.class_weighting,
        },
        "official_dev": {"samples": len(dev_rows), "metrics": dev_metrics},
        "history": history,
        "input_manifest": manifest,
        "input_manifest_sha256": base.sha256_file(args.data_dir / "manifest.json"),
        "output_sha256": {
            predictions_path.name: base.sha256_file(predictions_path),
            "best_model/model.safetensors": model_hash,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "amp_dtype": str(amp_dtype),
        },
        "leakage_policy": (
            "The per-seed epoch was frozen before official dev. Dev labels are used only "
            "to report count accuracy after the all-train checkpoint is fixed."
        ),
    }
    base.write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "frozen_epoch": selected_epoch,
                "official_dev_accuracy": dev_metrics["accuracy"],
                "official_dev_macro_f1": dev_metrics["macro_f1"],
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
