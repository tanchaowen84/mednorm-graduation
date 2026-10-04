"""Retrain the frozen BGE retriever on all official train rows, then score dev once."""

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

from mednorm.dense_training import multi_positive_contrastive_loss
from mednorm.official_contract_v3 import CONFIG_ID, frozen_epoch

try:
    from colab import train_biencoder as base
except ImportError:
    import train_biencoder as base  # type: ignore[import-not-found,no-redef]

JsonRow = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--model-revision", default=base.DEFAULT_BGE_REVISION)
    parser.add_argument("--pooling", default="cls", choices=("cls", "mean"))
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--planned-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--hard-negative-count", type=int, default=8)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--retrieval-k", type=int, default=800)
    parser.add_argument("--main-k", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def validate_inputs(
    data_dir: Path, catalog_path: Path
) -> tuple[
    JsonRow,
    tuple[str, ...],
    list[JsonRow],
    list[JsonRow],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    manifest = base.read_json(data_dir / "manifest.json")
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O7_OFFICIAL_DEV_RETRIEVAL_INPUT",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"unexpected official retrieval field {field}")
    for filename, expected_hash in manifest.get("output_sha256", {}).items():
        if base.sha256_file(data_dir / str(filename)) != expected_hash:
            raise ValueError(f"official retrieval hash mismatch for {filename}")
    if base.sha256_file(catalog_path) != manifest.get("input_sha256", {}).get(
        "catalog"
    ):
        raise ValueError("official retrieval catalog hash mismatch")
    catalog = base._load_catalog(catalog_path)
    catalog_set = set(catalog)
    records = base.read_gzip_jsonl(data_dir / "official_train_queries.jsonl.gz")
    record_ids = {str(row.get("sample_id")) for row in records}
    if len(record_ids) != len(records):
        raise ValueError("official train query IDs are duplicated")
    _, train_sparse = base._load_sparse_rankings(
        data_dir / "official_train_sparse.jsonl.gz",
        catalog_names=catalog_set,
        allowed_ids=record_ids,
    )
    dev_rows, dev_sparse = base._load_sparse_rankings(
        data_dir / "official_dev_sparse.jsonl.gz", catalog_names=catalog_set
    )
    counts = manifest.get("counts", {})
    if len(records) != int(counts.get("official_train_training_records", -1)):
        raise ValueError("official training record count differs from manifest")
    if len(dev_rows) != int(counts.get("official_dev_samples", -1)):
        raise ValueError("official dev row count differs from manifest")
    if set(train_sparse) != record_ids or len(dev_sparse) != len(dev_rows):
        raise ValueError("official sparse rankings are misaligned")
    return manifest, catalog, records, dev_rows, train_sparse, dev_sparse


def run(args: argparse.Namespace) -> JsonRow:
    if min(
        args.planned_epochs,
        args.batch_size,
        args.gradient_accumulation,
        args.encode_batch_size,
        args.retrieval_k,
    ) < 1:
        raise ValueError("official retriever training counts must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required unless --allow-cpu is set")
    selected_epoch = frozen_epoch("retriever", args.seed)
    if selected_epoch > args.planned_epochs:
        raise ValueError("frozen epoch exceeds the original planned schedule")
    started = time.time()
    base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    manifest, catalog, records, dev_rows, train_sparse, dev_sparse = validate_inputs(
        args.data_dir, args.catalog_path
    )

    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_name, revision=args.model_revision, use_fast=True
    )
    model = base.DualEncoder(args.model_name, args.model_revision, args.pooling).to(
        device
    )
    model.backbone.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(records) / args.batch_size)
    updates_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation)
    total_steps = updates_per_epoch * args.planned_epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: base._linear_warmup_decay(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    history: list[JsonRow] = []
    active_records = [dict(record) for record in records]
    optimizer_steps = 0
    for epoch in range(1, selected_epoch + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_rows in enumerate(
            base._batch_records(active_records, args.batch_size, args.seed + epoch),
            start=1,
        ):
            query_inputs, candidate_inputs, positive_mask = base._training_batch(
                batch_rows, tokenizer, args.max_length, device
            )
            with base._autocast_context(device):
                query_embeddings = model(query_inputs)
                candidate_embeddings = model(candidate_inputs)
                scores = (
                    query_embeddings @ candidate_embeddings.T
                ) / args.temperature
                loss = multi_positive_contrastive_loss(scores, positive_mask)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1
            if (
                batch_index % args.gradient_accumulation == 0
                or batch_index == steps_per_epoch
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, batch_count),
            "optimizer_updates": optimizer_steps,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    catalog_embeddings = base.encode_texts(
        model,
        tokenizer,
        catalog,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    train_candidate_rows = base.build_training_candidate_rows(
        model,
        tokenizer,
        records=records,
        sparse_rankings=train_sparse,
        catalog=catalog,
        catalog_embeddings=catalog_embeddings,
        device=device,
        encode_batch_size=args.encode_batch_size,
        query_chunk_size=args.query_chunk_size,
        max_length=args.max_length,
        retrieval_k=args.retrieval_k,
        rrf_constant=args.rrf_constant,
    )
    dev_metrics, dev_candidate_rows, _ = base.evaluate_model(
        model,
        tokenizer,
        catalog=catalog,
        catalog_embeddings=catalog_embeddings,
        val_rows=dev_rows,
        sparse_rankings=dev_sparse,
        device=device,
        encode_batch_size=args.encode_batch_size,
        query_chunk_size=args.query_chunk_size,
        max_length=args.max_length,
        retrieval_k=args.retrieval_k,
        rrf_constant=args.rrf_constant,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "official_train_candidates.jsonl.gz"
    dev_path = args.output_dir / "official_dev_candidates.jsonl.gz"
    base.write_gzip_jsonl(train_path, train_candidate_rows)
    base.write_gzip_jsonl(dev_path, dev_candidate_rows)
    model_dir = args.output_dir / "best_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model.backbone.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    model_hashes = base._model_hashes(model_dir)
    report: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": "O7_OFFICIAL_DEV_RETRIEVAL",
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
            "resolved_revision": getattr(model.backbone.config, "_commit_hash", None),
            "pooling": args.pooling,
            "temperature": args.temperature,
            "frozen_epoch": selected_epoch,
            "checkpoint_precision": "float32",
        },
        "training": {
            "seed": args.seed,
            "planned_epochs": args.planned_epochs,
            "executed_epochs": selected_epoch,
            "training_records": len(records),
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "optimizer_steps": optimizer_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "hard_negative_count": args.hard_negative_count,
            "max_length": args.max_length,
        },
        "official_dev": {"retrieval": dev_metrics, "samples": len(dev_rows)},
        "history": history,
        "input_manifest_sha256": base.sha256_file(args.data_dir / "manifest.json"),
        "input_manifest": manifest,
        "input_catalog_sha256": base.sha256_file(args.catalog_path),
        "output_sha256": {
            train_path.name: base.sha256_file(train_path),
            dev_path.name: base.sha256_file(dev_path),
        },
        "model_sha256": model_hashes,
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "leakage_policy": (
            "The epoch and all hyperparameters were frozen on the prior fit/tune/audit "
            "workflow. Official-dev labels are used only to report retrieval metrics after "
            "the all-train checkpoint is fixed."
        ),
    }
    base.write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "frozen_epoch": selected_epoch,
                "official_dev_fused_recall_400": dev_metrics["fused"][
                    "label_recall_at_k"
                ]["400"],
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
