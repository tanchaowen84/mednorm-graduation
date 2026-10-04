"""Retrain one frozen ranker seed on full official train and score dev once."""

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
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mednorm.official_contract_v3 import (
    CONFIG_ID,
    frozen_epoch,
    frozen_retrieval_weight,
)
from mednorm.ranking_loss_v3 import hybrid_group_ranking_loss

try:
    from colab import train_ranker_v3 as base
except ImportError:
    import train_ranker_v3 as base  # type: ignore[import-not-found,no-redef]

JsonRow = dict[str, Any]
COMPONENTS = {
    "o2_macbert": ("O6_OFFICIAL_FULLTRAIN_RANKING_DATA", False),
    "bge_reranker": ("O6_OFFICIAL_FULLTRAIN_RANKING_DATA", False),
    "kg_atomic_macbert": ("O6_OFFICIAL_FULLTRAIN_PROFILE_DATA", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--count-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", choices=tuple(COMPONENTS), required=True)
    parser.add_argument("--loss-mode", choices=("pointwise", "listwise_hybrid"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--planned-epochs", type=int, required=True)
    parser.add_argument("--group-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--negative-count", type=int, default=15)
    parser.add_argument("--top-hard-count", type=int, default=8)
    parser.add_argument("--listwise-weight", type=float, required=True)
    parser.add_argument("--pairwise-weight", type=float, required=True)
    parser.add_argument("--pointwise-weight", type=float, required=True)
    parser.add_argument("--positive-class-weight", type=float, default=3.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--candidate-k", type=int, default=400)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def validate_inputs(
    data_dir: Path,
    count_artifact_dir: Path,
    *,
    component: str,
    candidate_k: int,
) -> tuple[JsonRow, list[JsonRow], list[JsonRow], list[JsonRow]]:
    expected_stage, profile_expected = COMPONENTS[component]
    manifest = base.read_json(data_dir / "manifest.json")
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": CONFIG_ID,
        "stage": expected_stage,
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": profile_expected,
        "train_split": "official_train",
        "selection_split": None,
        "evaluation_split": "official_dev",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": True,
        "candidate_k": candidate_k,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"unexpected official ranker input field {field}")
    train_path = data_dir / "official_train_groups.jsonl.gz"
    dev_path = data_dir / "official_dev_candidates.jsonl.gz"
    for path in (train_path, dev_path):
        if manifest.get("output_sha256", {}).get(path.name) != base.sha256_file(path):
            raise ValueError(f"official ranker input hash mismatch for {path.name}")
    groups = base.read_gzip_jsonl(train_path)
    dev_bundles = base.read_gzip_jsonl(dev_path)
    group_count_key = (
        "official_train_groups"
        if profile_expected
        else "covered_official_train_samples"
    )
    if len(groups) != int(manifest["counts"][group_count_key]):
        raise ValueError("official ranker training group count differs from manifest")
    if len(dev_bundles) != int(manifest["counts"]["official_dev_samples"]):
        raise ValueError("official ranker dev count differs from manifest")
    maximum_negative_pool = int(manifest["negative_pool_size"])
    group_ids: set[str] = set()
    for index, group in enumerate(groups):
        sample_id = group.get("sample_id")
        positives = group.get("positives")
        if (
            not isinstance(sample_id, str)
            or sample_id in group_ids
            or not isinstance(group.get("text"), str)
            or not isinstance(positives, list)
            or not positives
        ):
            raise ValueError(f"invalid official ranker group {index}")
        positive_features = base._feature_list(
            group.get("positive_features"),
            field=f"official group {sample_id} positives",
            profile_expected=profile_expected,
        )
        negative_features = base._feature_list(
            group.get("negative_pool"),
            field=f"official group {sample_id} negatives",
            profile_expected=profile_expected,
        )
        if len(negative_features) > maximum_negative_pool:
            raise ValueError("official ranker group exceeds negative pool")
        if [row["name"] for row in positive_features] != positives:
            raise ValueError("official ranker positive features are misaligned")
        group_ids.add(sample_id)
    dev_ids: list[str] = []
    for index, bundle in enumerate(dev_bundles):
        sample_id = bundle.get("id")
        if (
            not isinstance(sample_id, str)
            or sample_id in dev_ids
            or not sample_id.startswith("dev-")
            or not isinstance(bundle.get("labels"), list)
            or not bundle["labels"]
        ):
            raise ValueError(f"invalid official ranker dev bundle {index}")
        base._feature_list(
            bundle.get("candidates"),
            field=f"official dev {sample_id}",
            profile_expected=profile_expected,
            expected_count=candidate_k,
        )
        dev_ids.append(sample_id)

    count_report_path = count_artifact_dir / "report.json"
    count_predictions_path = (
        count_artifact_dir / "official_dev_count_predictions.jsonl.gz"
    )
    count_report = base.read_json(count_report_path)
    required_count = {
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
    }
    for field, expected in required_count.items():
        if count_report.get(field) != expected:
            raise ValueError(f"unexpected official count artifact field {field}")
    if count_report.get("output_sha256", {}).get(
        count_predictions_path.name
    ) != base.sha256_file(count_predictions_path):
        raise ValueError("official count prediction hash mismatch")
    count_rows = base.read_gzip_jsonl(count_predictions_path)
    if [row.get("id") for row in count_rows] != dev_ids:
        raise ValueError("official count predictions are misaligned with dev")
    if any(row.get("count_prediction") not in (0, 1, 2) for row in count_rows):
        raise ValueError("official count prediction has an invalid class")
    return manifest, groups, dev_bundles, count_rows


def run(args: argparse.Namespace) -> JsonRow:
    if min(
        args.planned_epochs,
        args.group_batch_size,
        args.gradient_accumulation,
        args.eval_batch_size,
        args.max_length,
        args.negative_count,
        args.candidate_k,
    ) < 1:
        raise ValueError("official ranker training values must be positive")
    if not 0 <= args.top_hard_count <= args.negative_count:
        raise ValueError("official ranker hard-negative settings are invalid")
    loss_weights = (
        args.listwise_weight,
        args.pairwise_weight,
        args.pointwise_weight,
    )
    if args.loss_mode == "pointwise" and loss_weights != (0.0, 0.0, 1.0):
        raise ValueError("pointwise control must use exactly 0/0/1 weights")
    if args.loss_mode == "listwise_hybrid" and (
        args.listwise_weight <= 0 or args.pairwise_weight <= 0
    ):
        raise ValueError("listwise hybrid requires positive ranking weights")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required unless --allow-cpu is set")
    selected_epoch = frozen_epoch(args.component, args.seed)
    decoder_weight = frozen_retrieval_weight(args.component, args.seed)
    if selected_epoch > args.planned_epochs:
        raise ValueError("frozen ranker epoch exceeds the original schedule")
    started = time.time()
    base.set_seed(args.seed)
    manifest, groups, dev_bundles, count_rows = validate_inputs(
        args.data_dir,
        args.count_artifact_dir,
        component=args.component,
        candidate_k=args.candidate_k,
    )
    if args.negative_count > int(manifest["negative_pool_size"]):
        raise ValueError("requested negatives exceed the official pool")
    count_predictions = [int(row["count_prediction"]) for row in count_rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer: Any = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_name, revision=args.model_revision, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        num_labels=1,
        ignore_mismatched_sizes=True,
        attn_implementation="sdpa",
    ).to(device)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batches_per_epoch = math.ceil(len(groups) / args.group_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.planned_epochs)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    amp_enabled, amp_dtype = base.amp_settings(device)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    history: list[JsonRow] = []
    for epoch in range(1, selected_epoch + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        update_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_groups in enumerate(
            base.group_batches(
                groups,
                batch_size=args.group_batch_size,
                seed=args.seed,
                epoch=epoch,
            ),
            start=1,
        ):
            encoded, positive_mask, valid_mask, sizes = base.make_training_batch(
                batch_groups,
                tokenizer=tokenizer,
                max_length=args.max_length,
                negative_count=args.negative_count,
                top_hard_count=args.top_hard_count,
                seed=args.seed,
                epoch=epoch,
                query_mode="raw",
            )
            positive_mask = positive_mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                flat_logits = model(
                    **base._move_inputs(encoded, device)
                ).logits.squeeze(-1).float()
                group_logits = pad_sequence(
                    list(flat_logits.split(sizes)), batch_first=True, padding_value=0.0
                )
                loss = hybrid_group_ranking_loss(
                    group_logits,
                    positive_mask,
                    valid_mask,
                    listwise_weight=args.listwise_weight,
                    pairwise_weight=args.pairwise_weight,
                    pointwise_weight=args.pointwise_weight,
                    positive_class_weight=args.positive_class_weight,
                    pairwise_margin=args.pairwise_margin,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1
            if (
                batch_index % args.gradient_accumulation == 0
                or batch_index == batches_per_epoch
            ):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, batch_count),
            "optimizer_updates": update_count,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    scores = base.score_bundles(
        model,
        tokenizer,
        dev_bundles,
        candidate_k=args.candidate_k,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=device,
        query_mode="raw",
    )
    predictions = base.decode_dataset(
        dev_bundles,
        scores,
        count_predictions,
        candidate_k=args.candidate_k,
        retrieval_weight=decoder_weight,
    )
    metrics = base.segmented_metrics(dev_bundles, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "official_dev_scored_predictions.jsonl.gz"
    base.write_scored_predictions(
        predictions_path,
        dev_bundles,
        scores,
        count_predictions,
        predictions,
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
        "stage": f"O7_OFFICIAL_DEV_{args.component.upper()}_RANKER",
        "protocol": "Strict-ICD",
        "component": args.component,
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
            "groups": len(groups),
            "loss_mode": args.loss_mode,
            "group_batch_size": args.group_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "negative_count": args.negative_count,
            "top_hard_count": args.top_hard_count,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
        },
        "decoder": {
            "selection_source": "prior tune split",
            "retrieval_weight": decoder_weight,
            "candidate_k": args.candidate_k,
        },
        "official_dev": {"samples": len(dev_bundles), "segmented": metrics},
        "history": history,
        "input_manifest": manifest,
        "input_manifest_sha256": base.sha256_file(args.data_dir / "manifest.json"),
        "fixed_count_report_sha256": base.sha256_file(
            args.count_artifact_dir / "report.json"
        ),
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
            "The epoch and decoder weight were frozen before official dev. Dev labels are "
            "used only for reporting after all ranker scores are fixed."
        ),
    }
    base.write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "component": args.component,
                "seed": args.seed,
                "frozen_epoch": selected_epoch,
                "official_dev_micro_f1": metrics["all"]["micro_f1"],
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
