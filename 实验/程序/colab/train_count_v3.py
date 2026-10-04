"""Train the fixed P1-003 fit/tune label-count model on Colab."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import torch
import transformers
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

JsonRow = dict[str, Any]
DEFAULT_MODEL_REVISION = "1cf2677c782975600ce58e2961656b1b29eddbae"


class EncodedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, encodings: Mapping[str, torch.Tensor], labels: torch.Tensor) -> None:
        if any(len(tensor) != len(labels) for tensor in encodings.values()):
            raise ValueError("encoded tensors and labels differ in length")
        self.encodings = dict(encodings)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{key: value[index] for key, value in self.encodings.items()},
            "labels": self.labels[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--final-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="hfl/chinese-macbert-large")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--class-weighting", choices=("none", "balanced", "sqrt_balanced"), default="none"
    )
    parser.add_argument("--early-stopping", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-train-limit", type=int, default=0)
    parser.add_argument("--smoke-tune-limit", type=int, default=0)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> JsonRow:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_gzip_jsonl(path: Path) -> list[JsonRow]:
    rows: list[JsonRow] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
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


def set_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_inputs(data_dir: Path) -> tuple[JsonRow, list[JsonRow], list[JsonRow]]:
    manifest = read_json(data_dir / "manifest.json")
    required = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O2_RANKING_DATA",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": "O1 sparse+dense RRF",
        "kg_enabled": False,
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
        "audit_used": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"unexpected O2 manifest field {key}")
    for path in data_dir.iterdir():
        if "audit" in path.name.lower() or "dev" in path.name.lower():
            raise ValueError("count bundle contains a forbidden held-out file")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError("O2 manifest lacks output hashes")
    for filename in ("fit_count.jsonl.gz", "tune_candidates.jsonl.gz"):
        expected = output_hashes.get(filename)
        if expected is None or sha256_file(data_dir / filename) != expected:
            raise ValueError(f"hash mismatch for {filename}")

    fit_rows = read_gzip_jsonl(data_dir / "fit_count.jsonl.gz")
    tune_rows = read_gzip_jsonl(data_dir / "tune_candidates.jsonl.gz")
    if len(fit_rows) != int(manifest["counts"]["fit_samples"]):
        raise ValueError("fit count rows differ from manifest")
    if len(tune_rows) != int(manifest["counts"]["tune_samples"]):
        raise ValueError("tune rows differ from manifest")
    fit_ids: set[str] = set()
    for index, row in enumerate(fit_rows):
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or sample_id in fit_ids
            or not isinstance(row.get("text"), str)
            or row.get("count_class") not in (0, 1, 2)
        ):
            raise ValueError(f"invalid fit count row {index}")
        fit_ids.add(sample_id)
    tune_ids: set[str] = set()
    candidate_k = int(manifest["candidate_k"])
    for index, row in enumerate(tune_rows):
        sample_id = row.get("id")
        candidates = row.get("candidates")
        if (
            not isinstance(sample_id, str)
            or sample_id in tune_ids
            or not isinstance(row.get("text"), str)
            or not isinstance(row.get("labels"), list)
            or not isinstance(candidates, list)
            or len(candidates) != candidate_k
        ):
            raise ValueError(f"invalid tune candidate row {index}")
        for feature in candidates:
            if not isinstance(feature, dict) or set(feature) != {"name", "retrieval_score"}:
                raise ValueError("text-only count data contains an invalid candidate feature")
        tune_ids.add(sample_id)
    return manifest, fit_rows, tune_rows


def validate_final_evaluation_input(
    data_dir: Path, *, expected_candidate_k: int
) -> tuple[JsonRow, list[JsonRow]]:
    """Validate the audit bundle only after tune-based model selection is complete."""
    manifest_path = data_dir / "manifest.json"
    bundles_path = data_dir / "audit_o2_candidates.jsonl.gz"
    manifest = read_json(manifest_path)
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": "O4-FUSION-HGB15-W8-v1",
        "stage": "O5_AUDIT_INFERENCE_DATA",
        "protocol": "Strict-ICD",
        "evaluation_split": "audit",
        "selection_complete": True,
        "candidate_vocabulary_source": "ICD only",
        "candidate_k": expected_candidate_k,
        "audit_used": True,
        "official_dev_touched": False,
        "labels_in_candidate_bundles": True,
    }
    for key, expected_value in required.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"unexpected final count input field {key}")
    if manifest.get("output_sha256", {}).get(
        bundles_path.name
    ) != sha256_file(bundles_path):
        raise ValueError("final count candidate hash mismatch")
    bundles = read_gzip_jsonl(bundles_path)
    if len(bundles) != int(manifest["counts"]["audit_samples"]):
        raise ValueError("final count rows differ from manifest")
    seen: set[str] = set()
    for index, row in enumerate(bundles):
        sample_id = row.get("id")
        labels = row.get("labels")
        candidates = row.get("candidates")
        if (
            not isinstance(sample_id, str)
            or sample_id in seen
            or not isinstance(row.get("text"), str)
            or not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or not isinstance(row.get("all_labels_in_icd"), bool)
            or not isinstance(candidates, list)
            or len(candidates) != expected_candidate_k
        ):
            raise ValueError(f"invalid final count row {index}")
        for feature in candidates:
            if not isinstance(feature, dict) or set(feature) != {
                "name",
                "retrieval_score",
            }:
                raise ValueError("final count input contains an invalid candidate feature")
        seen.add(sample_id)
    return manifest, bundles


def tokenize_rows(tokenizer: Any, rows: Sequence[JsonRow], max_length: int) -> EncodedDataset:
    encodings = tokenizer(
        [str(row["text"]) for row in rows],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor([int(row["count_class"]) for row in rows], dtype=torch.long)
    return EncodedDataset(encodings, labels)


def make_loader(
    dataset: EncodedDataset, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def amp_settings(device: torch.device) -> tuple[bool, torch.dtype]:
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled, dtype


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = batch["labels"].to(device, non_blocking=True)
    inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "labels"
    }
    return inputs, labels


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[JsonRow, np.ndarray, np.ndarray]:
    model.eval()
    probability_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    amp_enabled, amp_dtype = amp_settings(device)
    for batch in loader:
        inputs, batch_labels = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(**inputs).logits
        probability_batches.append(logits.float().softmax(dim=-1).cpu().numpy())
        label_batches.append(batch_labels.cpu().numpy())
    probabilities = np.concatenate(probability_batches)
    labels = np.concatenate(label_batches)
    predictions = probabilities.argmax(axis=1)
    metrics: JsonRow = {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=(0, 1, 2)
        ).tolist(),
    }
    return metrics, probabilities, labels


def class_weights(labels: Sequence[int], policy: str, device: torch.device) -> torch.Tensor | None:
    if policy == "none":
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.float64)
    if bool((counts == 0).any()):
        raise ValueError("weighted count training requires every count class")
    weights = counts.sum() / (len(counts) * counts)
    if policy == "sqrt_balanced":
        weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run(args: argparse.Namespace) -> JsonRow:
    if min(
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.gradient_accumulation,
        args.max_length,
    ) < 1:
        raise ValueError("training counts must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required unless --allow-cpu is set")
    if args.final_data_dir is not None and (
        args.smoke_train_limit or args.smoke_tune_limit
    ):
        raise ValueError("audit evaluation is only allowed for a full count run")
    started = time.time()
    set_seed(args.seed)
    manifest, fit_rows, tune_bundles = validate_inputs(args.data_dir)
    if args.smoke_train_limit:
        fit_rows = fit_rows[: args.smoke_train_limit]
    if args.smoke_tune_limit:
        tune_bundles = tune_bundles[: args.smoke_tune_limit]
    tune_rows = [
        {
            "id": bundle["id"],
            "text": bundle["text"],
            "count_class": min(len(bundle["labels"]), 3) - 1,
        }
        for bundle in tune_bundles
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer: Any = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_name, revision=args.model_revision, use_fast=True
    )
    train_loader = make_loader(
        tokenize_rows(tokenizer, fit_rows, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    tune_loader = make_loader(
        tokenize_rows(tokenizer, tune_rows, args.max_length),
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
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    weights = class_weights(
        [int(row["count_class"]) for row in fit_rows], args.class_weighting, device
    )
    amp_enabled, amp_dtype = amp_settings(device)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    checkpoint = args.output_dir / "best_model"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    without_improvement = 0
    history: list[JsonRow] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        update_count = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            inputs, labels = _move_batch(batch, device)
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
        metrics, _, _ = evaluate_model(model, tune_loader, device=device)
        record: JsonRow = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "optimizer_updates": update_count,
            "tune": metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            without_improvement = 0
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
        else:
            without_improvement += 1
        if without_improvement >= args.early_stopping:
            break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selected = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, attn_implementation="sdpa"
    ).to(device)
    selected_metrics, probabilities, _ = evaluate_model(
        selected, tune_loader, device=device
    )
    predictions = probabilities.argmax(axis=1)
    prediction_path = args.output_dir / "tune_count_predictions.jsonl.gz"
    write_gzip_jsonl(
        prediction_path,
        (
            {
                "id": row["id"],
                "count_prediction": int(prediction),
                "probabilities": [float(value) for value in probability],
            }
            for row, prediction, probability in zip(
                tune_rows, predictions, probabilities, strict=True
            )
        ),
    )
    final_manifest: JsonRow | None = None
    final_metrics: JsonRow | None = None
    final_prediction_path: Path | None = None
    final_sample_count = 0
    if args.final_data_dir is not None:
        final_manifest, final_bundles = validate_final_evaluation_input(
            args.final_data_dir,
            expected_candidate_k=int(manifest["candidate_k"]),
        )
        final_rows = [
            {
                "id": bundle["id"],
                "text": bundle["text"],
                "count_class": min(len(bundle["labels"]), 3) - 1,
            }
            for bundle in final_bundles
        ]
        final_loader = make_loader(
            tokenize_rows(tokenizer, final_rows, args.max_length),
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
        )
        final_metrics, final_probabilities, _ = evaluate_model(
            selected, final_loader, device=device
        )
        final_predictions = final_probabilities.argmax(axis=1)
        final_prediction_path = args.output_dir / "audit_count_predictions.jsonl.gz"
        write_gzip_jsonl(
            final_prediction_path,
            (
                {
                    "id": row["id"],
                    "count_prediction": int(prediction),
                    "probabilities": [float(value) for value in probability],
                }
                for row, prediction, probability in zip(
                    final_rows,
                    final_predictions,
                    final_probabilities,
                    strict=True,
                )
            ),
        )
        final_sample_count = len(final_rows)
    model_hash = sha256_file(checkpoint / "model.safetensors")
    mode = "smoke" if args.smoke_train_limit or args.smoke_tune_limit else "full"
    report: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "stage": "O5_AUDIT_COUNT" if final_manifest is not None else "O2_COUNT",
        "protocol": "Strict-ICD",
        "mode": mode,
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
        "audit_used": final_manifest is not None,
        "model": {
            "name": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_revision,
            "best_epoch": best_epoch,
            "sha256": model_hash,
            "checkpoint_retained": args.save_model,
        },
        "training": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "class_weighting": args.class_weighting,
            "fit_samples": len(fit_rows),
            "tune_samples": len(tune_rows),
            "audit_samples": final_sample_count,
        },
        "tune": selected_metrics,
        "final_evaluation": (
            {"split": "audit", "metrics": final_metrics}
            if final_metrics is not None
            else None
        ),
        "history": history,
        "gate": {
            "status": (
                "SMOKE"
                if mode == "smoke"
                else (
                    "PASS"
                    if float(selected_metrics["accuracy"]) >= 0.70
                    and float(selected_metrics["macro_f1"]) >= 0.65
                    else "WEAK"
                )
            ),
            "accuracy_floor": 0.70,
            "macro_f1_floor": 0.65,
        },
        "input_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "input_manifest": manifest,
        "final_input_manifest": final_manifest,
        "output_sha256": {
            prediction_path.name: sha256_file(prediction_path),
            "best_model/model.safetensors": model_hash,
        },
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
            "amp_dtype": str(amp_dtype),
        },
    }
    if final_manifest is not None and args.final_data_dir is not None:
        report["config_id"] = "O4-FUSION-HGB15-W8-v1"
        report["evaluation_split"] = "audit"
        report["selection_complete"] = True
        report["final_input_manifest_sha256"] = sha256_file(
            args.final_data_dir / "manifest.json"
        )
    if final_prediction_path is not None:
        report["output_sha256"][final_prediction_path.name] = sha256_file(
            final_prediction_path
        )
    del selected
    if not args.save_model:
        shutil.rmtree(checkpoint)
        report["output_sha256"].pop("best_model/model.safetensors")
    write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "mode": mode,
                "best_epoch": best_epoch,
                "accuracy": selected_metrics["accuracy"],
                "macro_f1": selected_metrics["macro_f1"],
                "audit_accuracy": (
                    final_metrics["accuracy"] if final_metrics is not None else None
                ),
                "audit_macro_f1": (
                    final_metrics["macro_f1"] if final_metrics is not None else None
                ),
                "gate": report["gate"]["status"],
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
