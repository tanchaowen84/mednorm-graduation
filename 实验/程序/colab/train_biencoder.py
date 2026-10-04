"""Train a hash-locked Strict-ICD text-only dense retriever."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import torch
from torch import nn

from mednorm.candidate_fusion import SourceCandidate, fuse_rankings
from mednorm.dense_training import (
    multi_positive_contrastive_loss,
    refresh_hard_negatives,
)
from mednorm.metrics import RetrievalMetrics, evaluate_retrieval

JsonRow = dict[str, Any]
DEFAULT_BGE_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"
REPORT_KS = (50, 100, 200, 400, 800)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec-id", default="MEDNORM-P1-002")
    parser.add_argument("--stage", default="G1")
    parser.add_argument("--train-split", default="internal_train")
    parser.add_argument("--selection-split", default="internal_val")
    parser.add_argument("--model-name", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--model-revision", default=DEFAULT_BGE_REVISION)
    parser.add_argument("--pooling", choices=("auto", "cls", "mean"), default="auto")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
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
    parser.add_argument("--smoke-train-limit", type=int, default=0)
    parser.add_argument("--final-data-dir", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> JsonRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


def _string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    return list(value)


def _load_catalog(path: Path) -> tuple[str, ...]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    names: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError(f"invalid catalog row {index}")
        names.append(str(row["name"]))
    if not names or len(names) != len(set(names)):
        raise ValueError("catalog names must be non-empty and unique")
    return tuple(names)


def _load_sparse_rankings(
    path: Path,
    *,
    catalog_names: set[str],
    allowed_ids: set[str] | None = None,
) -> tuple[list[JsonRow], dict[str, tuple[str, ...]]]:
    rows: list[JsonRow] = []
    rankings: dict[str, tuple[str, ...]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            sample_id = row.get("id")
            if not isinstance(sample_id, str):
                raise ValueError(f"{path}:{line_number} has an invalid id")
            if allowed_ids is not None and sample_id not in allowed_ids:
                continue
            candidates = tuple(
                _string_list(row.get("sparse_candidates"), field="sparse_candidates")
            )
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"{sample_id} has duplicate sparse candidates")
            if any(name not in catalog_names for name in candidates):
                raise ValueError(f"{sample_id} has a non-ICD sparse candidate")
            forbidden_fields = {"graph_candidates", "graph_score", "aliases", "kg_candidates"}
            if forbidden_fields & row.keys():
                raise ValueError(f"{sample_id} contains a forbidden KG field")
            rows.append(row)
            rankings[sample_id] = candidates
    return rows, rankings


def validate_inputs(
    data_dir: Path,
    catalog_path: Path,
    *,
    smoke_train_limit: int,
    spec_id: str = "MEDNORM-P1-002",
    stage: str = "G1",
    train_split: str = "internal_train",
    selection_split: str = "internal_val",
) -> tuple[
    JsonRow,
    tuple[str, ...],
    list[JsonRow],
    list[JsonRow],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    manifest = read_json(data_dir / "manifest.json")
    required = {
        "spec_id": spec_id,
        "stage": stage,
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "selection_split": selection_split,
        "official_dev_touched": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} is not locked to {value!r}")
    for filename, expected_hash in manifest.get("output_sha256", {}).items():
        actual_hash = sha256_file(data_dir / str(filename))
        if actual_hash != expected_hash:
            raise ValueError(f"hash mismatch for {filename}")
    if sha256_file(catalog_path) != manifest.get("input_sha256", {}).get("catalog"):
        raise ValueError("catalog hash does not match G1 manifest")

    catalog = _load_catalog(catalog_path)
    catalog_set = set(catalog)
    records = read_gzip_jsonl(data_dir / f"{train_split}_queries.jsonl.gz")
    if smoke_train_limit > 0:
        records = records[:smoke_train_limit]
    record_ids: set[str] = set()
    for row_number, record in enumerate(records):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in record_ids:
            raise ValueError(f"invalid/duplicate training sample at row {row_number}")
        record_ids.add(sample_id)
        positives = set(_string_list(record.get("positives"), field="positives"))
        hard = set(_string_list(record.get("hard_negatives"), field="hard_negatives"))
        easy = set(_string_list(record.get("easy_negatives"), field="easy_negatives"))
        if not positives or not positives <= catalog_set:
            raise ValueError(f"{sample_id} has invalid positives")
        if not hard <= catalog_set or not easy <= catalog_set:
            raise ValueError(f"{sample_id} has non-ICD negatives")
        if positives & (hard | easy):
            raise ValueError(f"{sample_id} treats a gold label as negative")

    _, train_sparse = _load_sparse_rankings(
        data_dir / f"{train_split}_sparse.jsonl.gz",
        catalog_names=catalog_set,
        allowed_ids=record_ids,
    )
    val_rows, val_sparse = _load_sparse_rankings(
        data_dir / f"{selection_split}_sparse.jsonl.gz", catalog_names=catalog_set
    )
    if set(train_sparse) != record_ids:
        raise ValueError("training sparse rows do not cover selected training records")
    if len(val_rows) != int(manifest["counts"][f"{selection_split}_samples"]):
        raise ValueError("selection row count does not match manifest")
    return manifest, catalog, records, val_rows, train_sparse, val_sparse


def validate_final_evaluation_input(
    data_dir: Path,
    catalog_path: Path,
) -> tuple[JsonRow, list[JsonRow], dict[str, tuple[str, ...]]]:
    """Load audit only after model selection has completed."""
    manifest = read_json(data_dir / "manifest.json")
    required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": "O4-FUSION-HGB15-W8-v1",
        "stage": "O5_AUDIT_RETRIEVAL_INPUT",
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "evaluation_split": "audit",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"unexpected final-evaluation manifest field {key}")
    sparse_path = data_dir / "audit_sparse.jsonl.gz"
    if manifest.get("output_sha256", {}).get(sparse_path.name) != sha256_file(
        sparse_path
    ):
        raise ValueError("final-evaluation sparse hash mismatch")
    if manifest.get("input_sha256", {}).get("catalog") != sha256_file(catalog_path):
        raise ValueError("final-evaluation catalog hash mismatch")
    catalog = set(_load_catalog(catalog_path))
    rows, sparse = _load_sparse_rankings(sparse_path, catalog_names=catalog)
    if len(rows) != int(manifest.get("counts", {}).get("audit_samples", -1)):
        raise ValueError("final-evaluation row count differs from manifest")
    if len(rows) != len(sparse):
        raise ValueError("final-evaluation sparse rows contain duplicate ids")
    return manifest, rows, sparse


class DualEncoder(nn.Module):
    """Shared Transformer encoder with model-appropriate pooling and L2 normalization."""

    def __init__(self, model_name: str, revision: str, pooling: str) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name, revision=revision)
        self.pooling = pooling

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**inputs)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            attention_mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)


def _move_inputs(
    inputs: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def _batch_records(
    records: Sequence[JsonRow], batch_size: int, seed: int
) -> Iterable[list[JsonRow]]:
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[index] for index in order[start : start + batch_size]]


def _training_batch(
    rows: Sequence[JsonRow], tokenizer: Any, max_length: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    candidate_names = list(
        dict.fromkeys(
            str(name)
            for row in rows
            for field in ("positives", "hard_negatives", "easy_negatives")
            for name in row[field]
        )
    )
    candidate_index = {name: index for index, name in enumerate(candidate_names)}
    positive_mask = torch.zeros(
        (len(rows), len(candidate_names)), dtype=torch.bool, device=device
    )
    for row_index, row in enumerate(rows):
        for positive in row["positives"]:
            positive_mask[row_index, candidate_index[str(positive)]] = True
    query_inputs = tokenizer(
        [str(row["text"]) for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    candidate_inputs = tokenizer(
        candidate_names,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return _move_inputs(query_inputs, device), _move_inputs(candidate_inputs, device), positive_mask


def _autocast_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


@torch.inference_mode()
def encode_texts(
    model: DualEncoder,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    model.eval()
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with _autocast_context(device):
            batch_embeddings = model(_move_inputs(encoded, device))
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def exact_dense_rankings(
    query_ids: Sequence[str],
    query_embeddings: torch.Tensor,
    catalog: Sequence[str],
    catalog_embeddings: torch.Tensor,
    *,
    device: torch.device,
    k: int,
    query_chunk_size: int,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[float, ...]]]:
    result_k = min(k, len(catalog))
    catalog_on_device = catalog_embeddings.to(device)
    names: dict[str, tuple[str, ...]] = {}
    scores: dict[str, tuple[float, ...]] = {}
    for start in range(0, len(query_ids), query_chunk_size):
        query_chunk = query_embeddings[start : start + query_chunk_size].to(device)
        similarities = query_chunk @ catalog_on_device.T
        top_scores, top_indices = torch.topk(
            similarities, k=result_k, dim=1, largest=True, sorted=True
        )
        for offset, sample_id in enumerate(query_ids[start : start + query_chunk_size]):
            indices = top_indices[offset].cpu().tolist()
            names[sample_id] = tuple(catalog[int(index)] for index in indices)
            scores[sample_id] = tuple(float(value) for value in top_scores[offset].cpu().tolist())
    del catalog_on_device
    return names, scores


def _metric_record(metrics: RetrievalMetrics) -> JsonRow:
    return {
        "label_recall_at_k": {
            str(k): value for k, value in metrics.label_recall_at_k.items()
        },
        "sample_all_recall_at_k": {
            str(k): value for k, value in metrics.sample_all_recall_at_k.items()
        },
    }


@torch.inference_mode()
def evaluate_model(
    model: DualEncoder,
    tokenizer: Any,
    *,
    catalog: Sequence[str],
    catalog_embeddings: torch.Tensor,
    val_rows: Sequence[JsonRow],
    sparse_rankings: Mapping[str, Sequence[str]],
    device: torch.device,
    encode_batch_size: int,
    query_chunk_size: int,
    max_length: int,
    retrieval_k: int,
    rrf_constant: int,
) -> tuple[JsonRow, list[JsonRow], dict[str, tuple[str, ...]]]:
    query_ids = [str(row["id"]) for row in val_rows]
    query_embeddings = encode_texts(
        model,
        tokenizer,
        [str(row["text"]) for row in val_rows],
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    dense_names, dense_scores = exact_dense_rankings(
        query_ids,
        query_embeddings,
        catalog,
        catalog_embeddings,
        device=device,
        k=retrieval_k,
        query_chunk_size=query_chunk_size,
    )
    catalog_set = set(catalog)
    fused_names: dict[str, tuple[str, ...]] = {}
    output_rows: list[JsonRow] = []
    for row in val_rows:
        sample_id = str(row["id"])
        sparse = tuple(
            SourceCandidate(name, -float(rank))
            for rank, name in enumerate(sparse_rankings[sample_id])
        )
        dense = tuple(
            SourceCandidate(name, score)
            for name, score in zip(
                dense_names[sample_id], dense_scores[sample_id], strict=True
            )
        )
        fused = fuse_rankings(
            sparse,
            dense,
            catalog_names=catalog_set,
            k=retrieval_k,
            rrf_constant=rrf_constant,
        )
        fused_names[sample_id] = tuple(candidate.name for candidate in fused)
        output_rows.append(
            {
                "id": sample_id,
                "text": row["text"],
                "labels": row["labels"],
                "all_labels_in_icd": row["all_labels_in_icd"],
                "sparse_candidates": list(sparse_rankings[sample_id]),
                "dense_candidates": list(dense_names[sample_id]),
                "fused_candidates": list(fused_names[sample_id]),
            }
        )
    gold = [tuple(str(label) for label in row["labels"]) for row in val_rows]
    valid_ks = tuple(k for k in REPORT_KS if k <= retrieval_k)
    sparse_metrics = evaluate_retrieval(
        gold, [tuple(sparse_rankings[sample_id]) for sample_id in query_ids], ks=valid_ks
    )
    dense_metrics = evaluate_retrieval(
        gold, [dense_names[sample_id] for sample_id in query_ids], ks=valid_ks
    )
    fused_metrics = evaluate_retrieval(
        gold, [fused_names[sample_id] for sample_id in query_ids], ks=valid_ks
    )
    return (
        {
            "sparse": _metric_record(sparse_metrics),
            "dense": _metric_record(dense_metrics),
            "fused": _metric_record(fused_metrics),
        },
        output_rows,
        dense_names,
    )


def _selection_key(metrics: Mapping[str, Any], main_k: int) -> tuple[float, float, float]:
    key = str(main_k)
    return (
        float(metrics["fused"]["label_recall_at_k"][key]),
        float(metrics["fused"]["sample_all_recall_at_k"][key]),
        float(metrics["dense"]["label_recall_at_k"][key]),
    )


def _linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _model_hashes(model_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(model_dir).as_posix(): sha256_file(path)
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }


@torch.inference_mode()
def build_training_candidate_rows(
    model: DualEncoder,
    tokenizer: Any,
    *,
    records: Sequence[JsonRow],
    sparse_rankings: Mapping[str, Sequence[str]],
    catalog: Sequence[str],
    catalog_embeddings: torch.Tensor,
    device: torch.device,
    encode_batch_size: int,
    query_chunk_size: int,
    max_length: int,
    retrieval_k: int,
    rrf_constant: int,
) -> list[JsonRow]:
    query_ids = [str(record["sample_id"]) for record in records]
    query_embeddings = encode_texts(
        model,
        tokenizer,
        [str(record["text"]) for record in records],
        device=device,
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    dense_names, dense_scores = exact_dense_rankings(
        query_ids,
        query_embeddings,
        catalog,
        catalog_embeddings,
        device=device,
        k=retrieval_k,
        query_chunk_size=query_chunk_size,
    )
    catalog_set = set(catalog)
    output_rows: list[JsonRow] = []
    for record in records:
        sample_id = str(record["sample_id"])
        sparse = tuple(
            SourceCandidate(name, -float(rank))
            for rank, name in enumerate(sparse_rankings[sample_id])
        )
        dense = tuple(
            SourceCandidate(name, score)
            for name, score in zip(
                dense_names[sample_id], dense_scores[sample_id], strict=True
            )
        )
        fused = fuse_rankings(
            sparse,
            dense,
            catalog_names=catalog_set,
            k=retrieval_k,
            rrf_constant=rrf_constant,
        )
        output_rows.append(
            {
                "id": sample_id,
                "text": record["text"],
                "labels": record["positives"],
                "sparse_candidates": list(sparse_rankings[sample_id]),
                "dense_candidates": list(dense_names[sample_id]),
                "fused_candidates": [candidate.name for candidate in fused],
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()
    if args.temperature is not None and args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("training counts must be positive")
    if args.main_k > args.retrieval_k:
        raise ValueError("main_k cannot exceed retrieval_k")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is set")

    set_seed(args.seed)
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    manifest, catalog, records, val_rows, train_sparse, val_sparse = validate_inputs(
        args.data_dir,
        args.catalog_path,
        smoke_train_limit=args.smoke_train_limit,
        spec_id=args.spec_id,
        stage=args.stage,
        train_split=args.train_split,
        selection_split=args.selection_split,
    )
    smoke = args.smoke_train_limit > 0
    import transformers
    from transformers import AutoTokenizer

    pooling = args.pooling
    if pooling == "auto":
        pooling = "cls" if "bge" in args.model_name.lower() else "mean"
    temperature = args.temperature
    if temperature is None:
        temperature = 0.01 if "bge" in args.model_name.lower() else 0.05
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_name, revision=args.model_revision, use_fast=True
    )
    model = DualEncoder(args.model_name, args.model_revision, pooling).to(device)
    model.backbone.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(records) / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _linear_warmup_decay(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )

    catalog_embeddings = encode_texts(
        model,
        tokenizer,
        catalog,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    initial_metrics, initial_rows, _ = evaluate_model(
        model,
        tokenizer,
        catalog=catalog,
        catalog_embeddings=catalog_embeddings,
        val_rows=val_rows,
        sparse_rankings=val_sparse,
        device=device,
        encode_batch_size=args.encode_batch_size,
        query_chunk_size=args.query_chunk_size,
        max_length=args.max_length,
        retrieval_k=args.retrieval_k,
        rrf_constant=args.rrf_constant,
    )
    history: list[JsonRow] = [{"epoch": 0, "train_loss": None, "metrics": initial_metrics}]
    best_epoch = 0
    best_metrics = initial_metrics
    best_rows = initial_rows
    best_state = _clone_state_dict(model)
    global_micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    active_records = [dict(record) for record in records]

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_rows in enumerate(
            _batch_records(active_records, args.batch_size, args.seed + epoch), start=1
        ):
            query_inputs, candidate_inputs, positive_mask = _training_batch(
                batch_rows, tokenizer, args.max_length, device
            )
            with _autocast_context(device):
                query_embeddings = model(query_inputs)
                candidate_embeddings = model(candidate_inputs)
                scores = (query_embeddings @ candidate_embeddings.T) / temperature
                loss = multi_positive_contrastive_loss(scores, positive_mask)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            running_loss += float(loss.detach().cpu())
            batch_count += 1
            should_step = (
                batch_index % args.gradient_accumulation == 0
                or batch_index == steps_per_epoch
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_micro_step += 1

        catalog_embeddings = encode_texts(
            model,
            tokenizer,
            catalog,
            device=device,
            batch_size=args.encode_batch_size,
            max_length=args.max_length,
        )
        epoch_metrics, epoch_rows, _ = evaluate_model(
            model,
            tokenizer,
            catalog=catalog,
            catalog_embeddings=catalog_embeddings,
            val_rows=val_rows,
            sparse_rankings=val_sparse,
            device=device,
            encode_batch_size=args.encode_batch_size,
            query_chunk_size=args.query_chunk_size,
            max_length=args.max_length,
            retrieval_k=args.retrieval_k,
            rrf_constant=args.rrf_constant,
        )
        epoch_record: JsonRow = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batch_count),
            "metrics": epoch_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if _selection_key(epoch_metrics, args.main_k) > _selection_key(
            best_metrics, args.main_k
        ):
            best_epoch = epoch
            best_metrics = epoch_metrics
            best_rows = epoch_rows
            best_state = _clone_state_dict(model)

        if epoch < args.epochs:
            train_query_embeddings = encode_texts(
                model,
                tokenizer,
                [str(record["text"]) for record in active_records],
                device=device,
                batch_size=args.encode_batch_size,
                max_length=args.max_length,
            )
            train_dense, _ = exact_dense_rankings(
                [str(record["sample_id"]) for record in active_records],
                train_query_embeddings,
                catalog,
                catalog_embeddings,
                device=device,
                k=max(64, args.hard_negative_count * 8),
                query_chunk_size=args.query_chunk_size,
            )
            active_records = refresh_hard_negatives(
                active_records,
                sparse_rankings=train_sparse,
                dense_rankings=train_dense,
                catalog_names=set(catalog),
                hard_negative_count=args.hard_negative_count,
                rrf_constant=args.rrf_constant,
            )

    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_candidates_filename = f"{args.selection_split}_candidates.jsonl.gz"
    candidates_path = args.output_dir / selection_candidates_filename
    write_gzip_jsonl(candidates_path, best_rows)
    output_hashes = {
        selection_candidates_filename: sha256_file(candidates_path),
    }
    final_evaluation: JsonRow | None = None
    final_manifest_hash: str | None = None
    if not smoke:
        best_catalog_embeddings = encode_texts(
            model,
            tokenizer,
            catalog,
            device=device,
            batch_size=args.encode_batch_size,
            max_length=args.max_length,
        )
        train_candidate_rows = build_training_candidate_rows(
            model,
            tokenizer,
            records=records,
            sparse_rankings=train_sparse,
            catalog=catalog,
            catalog_embeddings=best_catalog_embeddings,
            device=device,
            encode_batch_size=args.encode_batch_size,
            query_chunk_size=args.query_chunk_size,
            max_length=args.max_length,
            retrieval_k=args.retrieval_k,
            rrf_constant=args.rrf_constant,
        )
        train_candidates_filename = f"{args.train_split}_candidates.jsonl.gz"
        train_candidates_path = args.output_dir / train_candidates_filename
        write_gzip_jsonl(train_candidates_path, train_candidate_rows)
        output_hashes[train_candidates_filename] = sha256_file(train_candidates_path)
        if args.final_data_dir is not None:
            final_manifest, final_rows, final_sparse = validate_final_evaluation_input(
                args.final_data_dir,
                args.catalog_path,
            )
            final_metrics, final_output_rows, _ = evaluate_model(
                model,
                tokenizer,
                catalog=catalog,
                catalog_embeddings=best_catalog_embeddings,
                val_rows=final_rows,
                sparse_rankings=final_sparse,
                device=device,
                encode_batch_size=args.encode_batch_size,
                query_chunk_size=args.query_chunk_size,
                max_length=args.max_length,
                retrieval_k=args.retrieval_k,
                rrf_constant=args.rrf_constant,
            )
            final_candidates_filename = "audit_candidates.jsonl.gz"
            final_candidates_path = args.output_dir / final_candidates_filename
            write_gzip_jsonl(final_candidates_path, final_output_rows)
            output_hashes[final_candidates_filename] = sha256_file(
                final_candidates_path
            )
            final_evaluation = {
                "split": final_manifest["evaluation_split"],
                "metrics": final_metrics,
            }
            final_manifest_hash = sha256_file(args.final_data_dir / "manifest.json")
    model_hashes: dict[str, str] = {}
    if args.save_model:
        model_dir = args.output_dir / "best_model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model.backbone.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        model_hashes = _model_hashes(model_dir)

    sparse_at_main = float(best_metrics["sparse"]["label_recall_at_k"][str(args.main_k)])
    fused_at_main = float(best_metrics["fused"]["label_recall_at_k"][str(args.main_k)])
    delta = fused_at_main - sparse_at_main
    required_delta = 0.0 if smoke else 0.01
    gate_pass = delta > required_delta if smoke else delta >= required_delta
    elapsed = time.time() - started
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    )
    resolved_revision = getattr(model.backbone.config, "_commit_hash", None)
    report: JsonRow = {
        "spec_id": args.spec_id,
        "stage": "O5_AUDIT_RETRIEVAL" if final_evaluation is not None else args.stage,
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "kg_enabled": False,
        "official_dev_touched": False,
        "audit_used": final_evaluation is not None,
        "train_split": args.train_split,
        "selection_split": args.selection_split,
        "evaluation_split": "audit" if final_evaluation is not None else None,
        "mode": "smoke" if smoke else "full",
        "gate": {
            "status": "PASS" if gate_pass else "FAIL",
            "metric": f"fused_label_recall_at_{args.main_k}_minus_sparse",
            "required_delta": required_delta,
            "actual_delta": delta,
        },
        "model": {
            "name": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_revision,
            "pooling": pooling,
            "temperature": temperature,
            "best_epoch": best_epoch,
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "training_records": len(records),
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "optimizer_steps": global_micro_step,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "hard_negative_count": args.hard_negative_count,
            "max_length": args.max_length,
        },
        "retrieval": best_metrics,
        "final_evaluation": final_evaluation,
        "history": history,
        "runtime": {
            "elapsed_seconds": elapsed,
            "device": str(device),
            "gpu_name": gpu_name,
            "peak_memory_bytes": peak_memory,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "input_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "input_final_manifest_sha256": final_manifest_hash,
        "input_catalog_sha256": sha256_file(args.catalog_path),
        "output_sha256": output_hashes,
        "model_sha256": model_hashes,
        "input_manifest": {
            "retrieval_k": manifest["retrieval_k"],
            "main_candidate_k": manifest["main_candidate_k"],
        },
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
