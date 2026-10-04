"""Train matched pointwise or multi-positive listwise P1-003 rankers on Colab."""

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
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mednorm.decoding import ScoredCandidate, decode_candidates
from mednorm.metrics import PredictionMetrics, evaluate_predictions
from mednorm.ranking_loss_v3 import hybrid_group_ranking_loss, sample_group_candidates

JsonRow = dict[str, Any]
DEFAULT_MODEL_REVISION = "1cf2677c782975600ce58e2961656b1b29eddbae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--count-artifact-dir", type=Path, required=True)
    parser.add_argument("--final-data-dir", type=Path)
    parser.add_argument("--final-count-artifact-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-stage",
        choices=(
            "O2_RANKING_DATA",
            "O3_PROFILE_DATA",
            "O4_ATOMIC_PROFILE_DATA",
            "O4_RR_PROFILE_DATA",
        ),
        default="O2_RANKING_DATA",
    )
    parser.add_argument("--kg-enabled", action="store_true")
    parser.add_argument("--loss-mode", choices=("pointwise", "listwise_hybrid"), required=True)
    parser.add_argument("--model-name", default="hfl/chinese-macbert-large")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--group-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--negative-count", type=int, default=15)
    parser.add_argument("--top-hard-count", type=int, default=8)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--pointwise-weight", type=float, default=0.2)
    parser.add_argument("--positive-class-weight", type=float, default=3.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--selection-candidate-k", type=int, default=100)
    parser.add_argument("--final-candidate-k", type=int, default=400)
    parser.add_argument(
        "--retrieval-weights", default="0,0.02,0.05,0.1,0.2,0.3,0.5,1.0"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-group-limit", type=int, default=0)
    parser.add_argument("--smoke-tune-limit", type=int, default=0)
    parser.add_argument("--query-mode", choices=("auto", "raw"), default="auto")
    parser.add_argument("--score-atomic-mentions", action="store_true")
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


def _feature_list(
    value: object,
    *,
    field: str,
    profile_expected: bool,
    expected_count: int | None = None,
) -> list[JsonRow]:
    if not isinstance(value, list) or (
        expected_count is not None and len(value) != expected_count
    ):
        raise ValueError(f"{field} has an invalid candidate count")
    rows: list[JsonRow] = []
    names: set[str] = set()
    for index, feature in enumerate(value):
        if not isinstance(feature, dict):
            raise ValueError(f"{field}[{index}] is not an object")
        allowed = {"name", "retrieval_score"}
        if profile_expected:
            allowed |= {"candidate_text", "kg_relations"}
        if not {"name", "retrieval_score"} <= set(feature) or set(feature) - allowed:
            raise ValueError(f"{field}[{index}] has unexpected profile fields")
        name = feature.get("name")
        score = feature.get("retrieval_score")
        candidate_text = feature.get("candidate_text")
        relations = feature.get("kg_relations")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(score, int | float)
            or not 0.0 <= float(score) <= 1.0
            or (profile_expected and (not isinstance(candidate_text, str) or not candidate_text))
            or (
                relations is not None
                and (
                    not isinstance(relations, list)
                    or not all(isinstance(relation, str) for relation in relations)
                )
            )
        ):
            raise ValueError(f"{field}[{index}] is invalid")
        if not profile_expected and candidate_text is not None:
            raise ValueError(f"{field}[{index}] leaks a profile into the O2 control")
        names.add(name)
        rows.append(feature)
    return rows


def ranker_data_contract(data_stage: str) -> tuple[str, bool]:
    """Return the locked candidate-source text and atomic-query requirement."""
    if data_stage == "O2_RANKING_DATA":
        return "O1 sparse+dense RRF", False
    if data_stage == "O3_PROFILE_DATA":
        return "O1 sparse+dense RRF plus CPubMed-KG candidate profiles", False
    if data_stage == "O4_ATOMIC_PROFILE_DATA":
        return (
            "O1 sparse+dense RRF plus CPubMed-KG candidate profiles plus "
            "source-supported atomic query augmentation",
            True,
        )
    if data_stage == "O4_RR_PROFILE_DATA":
        return (
            "O1 sparse+dense RRF plus CPubMed-KG candidate profiles plus "
            "RRNorm public atomic augmentation",
            True,
        )
    raise ValueError("unsupported ranker data stage")


def query_text(record: Mapping[str, Any], *, query_mode: str = "auto") -> str:
    """Resolve either the legacy augmented query or the untouched raw mention."""
    if query_mode == "auto":
        value = record.get("query_text", record.get("text"))
    elif query_mode == "raw":
        value = record.get("text")
    else:
        raise ValueError("unsupported query mode")
    if not isinstance(value, str) or not value:
        raise ValueError("ranker query text must be non-empty")
    return value


def atomic_scoring_pairs(
    bundles: Sequence[JsonRow], *, candidate_k: int
) -> tuple[list[str], list[str], tuple[tuple[int, int], ...]]:
    """Flatten each atomic mention against the same closed ICD candidate pool."""
    if candidate_k < 1:
        raise ValueError("candidate_k must be positive")
    queries: list[str] = []
    candidates: list[str] = []
    shapes: list[tuple[int, int]] = []
    for bundle in bundles:
        mentions = bundle.get("atomic_mentions")
        if (
            not isinstance(mentions, list)
            or not mentions
            or not all(isinstance(mention, str) and mention for mention in mentions)
        ):
            raise ValueError("atomic scoring requires non-empty atomic mentions")
        features = bundle.get("candidates")
        if not isinstance(features, list) or len(features) < candidate_k:
            raise ValueError("atomic scoring candidate pool is too small")
        selected = features[:candidate_k]
        candidate_texts: list[str] = []
        for feature in selected:
            if not isinstance(feature, dict):
                raise ValueError("atomic scoring candidate must be an object")
            candidate = feature.get("candidate_text", feature.get("name"))
            if not isinstance(candidate, str) or not candidate:
                raise ValueError("atomic scoring candidate text must be non-empty")
            candidate_texts.append(candidate)
        for mention in mentions:
            queries.extend([mention] * len(candidate_texts))
            candidates.extend(candidate_texts)
        shapes.append((len(mentions), len(candidate_texts)))
    return queries, candidates, tuple(shapes)


def _validate_atomic_query(record: Mapping[str, Any], *, field: str) -> None:
    value = record.get("query_text")
    mentions = record.get("atomic_mentions")
    active = record.get("atomic_active")
    if (
        not isinstance(value, str)
        or not value
        or not isinstance(mentions, list)
        or not mentions
        or not all(isinstance(mention, str) and mention for mention in mentions)
        or not isinstance(active, bool)
        or active != (len(mentions) > 1)
    ):
        raise ValueError(f"{field} has invalid atomic query fields")


def validate_inputs(
    data_dir: Path,
    count_artifact_dir: Path,
    *,
    data_stage: str = "O2_RANKING_DATA",
    kg_enabled: bool = False,
) -> tuple[JsonRow, list[JsonRow], list[JsonRow], list[JsonRow]]:
    manifest = read_json(data_dir / "manifest.json")
    expected_candidate_source, atomic_expected = ranker_data_contract(data_stage)
    required = {
        "spec_id": "MEDNORM-P1-003",
        "stage": data_stage,
        "protocol": "Strict-ICD",
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": expected_candidate_source,
        "kg_enabled": kg_enabled,
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
        "audit_used": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"unexpected O2 manifest field {key}")
    if atomic_expected and manifest.get("atomic_enabled") is not True:
        raise ValueError("atomic profile manifest must enable atomic queries")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError("O2 manifest lacks output hashes")
    for filename in ("fit_groups.jsonl.gz", "tune_candidates.jsonl.gz"):
        if output_hashes.get(filename) != sha256_file(data_dir / filename):
            raise ValueError(f"hash mismatch for {filename}")
    groups = read_gzip_jsonl(data_dir / "fit_groups.jsonl.gz")
    tune_bundles = read_gzip_jsonl(data_dir / "tune_candidates.jsonl.gz")
    expected_group_count = manifest["counts"].get(
        "fit_groups", manifest["counts"].get("covered_fit_samples")
    )
    if len(groups) != int(expected_group_count):
        raise ValueError("fit group count differs from manifest")
    if len(tune_bundles) != int(manifest["counts"]["tune_samples"]):
        raise ValueError("tune count differs from manifest")
    group_ids: set[str] = set()
    maximum_negative_pool = int(manifest["negative_pool_size"])
    for index, group in enumerate(groups):
        sample_id = group.get("sample_id")
        positives = group.get("positives")
        if (
            not isinstance(sample_id, str)
            or sample_id in group_ids
            or not isinstance(group.get("text"), str)
            or not isinstance(positives, list)
            or not positives
            or not all(isinstance(name, str) for name in positives)
        ):
            raise ValueError(f"invalid ranking group {index}")
        positive_features = _feature_list(
            group.get("positive_features"),
            field=f"group {sample_id} positives",
            profile_expected=kg_enabled,
        )
        negative_features = _feature_list(
            group.get("negative_pool"),
            field=f"group {sample_id} negatives",
            profile_expected=kg_enabled,
        )
        if len(negative_features) > maximum_negative_pool:
            raise ValueError(f"{sample_id} exceeds the locked negative pool size")
        if [feature["name"] for feature in positive_features] != positives:
            raise ValueError(f"{sample_id} positives and features differ")
        if set(positives) & {feature["name"] for feature in negative_features}:
            raise ValueError(f"{sample_id} treats a positive as a negative")
        if atomic_expected:
            _validate_atomic_query(group, field=f"group {sample_id}")
        group_ids.add(sample_id)
    candidate_k = int(manifest["candidate_k"])
    tune_ids: list[str] = []
    for index, bundle in enumerate(tune_bundles):
        sample_id = bundle.get("id")
        if (
            not isinstance(sample_id, str)
            or sample_id in tune_ids
            or not isinstance(bundle.get("text"), str)
            or not isinstance(bundle.get("labels"), list)
            or not isinstance(bundle.get("all_labels_in_icd"), bool)
        ):
            raise ValueError(f"invalid tune bundle {index}")
        _feature_list(
            bundle.get("candidates"),
            field=f"tune {sample_id}",
            profile_expected=kg_enabled,
            expected_count=candidate_k,
        )
        if atomic_expected:
            _validate_atomic_query(bundle, field=f"tune {sample_id}")
        tune_ids.append(sample_id)

    count_report_path = count_artifact_dir / "report.json"
    count_validation_path = count_artifact_dir / "validation.json"
    count_predictions_path = count_artifact_dir / "tune_count_predictions.jsonl.gz"
    count_report = read_json(count_report_path)
    count_validation = read_json(count_validation_path)
    required_count: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "protocol": "Strict-ICD",
        "mode": "full",
        "train_split": "fit",
        "selection_split": "tune",
        "official_dev_touched": False,
    }
    count_stage = count_report.get("stage")
    if count_stage == "O2_COUNT":
        required_count |= {
            "stage": "O2_COUNT",
            "audit_used": False,
        }
        required_validation: JsonRow = {
            "status": "PASS",
            "spec_id": "MEDNORM-P1-003",
            "stage": "O2_COUNT",
            "selection_split": "tune",
        }
    elif count_stage == "O5_AUDIT_COUNT":
        required_count |= {
            "config_id": "O4-FUSION-HGB15-W8-v1",
            "stage": "O5_AUDIT_COUNT",
            "evaluation_split": "audit",
            "selection_complete": True,
            "audit_used": True,
        }
        required_validation = {
            "status": "PASS",
            "spec_id": "MEDNORM-P1-003",
            "config_id": "O4-FUSION-HGB15-W8-v1",
            "stage": "O5_AUDIT_COUNT",
            "selection_split": "tune",
            "evaluation_split": "audit",
            "selection_complete": True,
            "audit_used": True,
        }
    else:
        raise ValueError("fixed count artifact has an unsupported stage")
    for key, expected in required_count.items():
        if count_report.get(key) != expected:
            raise ValueError(f"unexpected fixed count report field {key}")
    if count_report.get("gate", {}).get("status") != "PASS":
        raise ValueError("fixed count model has not passed")
    if count_report.get("output_sha256", {}).get(
        count_predictions_path.name
    ) != sha256_file(count_predictions_path):
        raise ValueError("fixed count prediction hash mismatch")
    for key, expected in required_validation.items():
        if count_validation.get(key) != expected:
            raise ValueError(f"unexpected fixed count validation field {key}")
    if count_stage == "O5_AUDIT_COUNT" and count_validation.get("sha256", {}).get(
        "tune_predictions"
    ) != sha256_file(count_predictions_path):
        raise ValueError("validated fixed count prediction hash mismatch")
    count_rows = read_gzip_jsonl(count_predictions_path)
    if len(count_rows) != len(tune_ids) or [row.get("id") for row in count_rows] != tune_ids:
        raise ValueError("fixed count predictions are misaligned with tune")
    for row in count_rows:
        if row.get("count_prediction") not in (0, 1, 2):
            raise ValueError("fixed count prediction has an invalid class")
    return manifest, groups, tune_bundles, count_rows


def validate_final_evaluation_inputs(
    data_dir: Path,
    count_artifact_dir: Path,
    *,
    data_stage: str,
    kg_enabled: bool,
    expected_candidate_k: int,
) -> tuple[JsonRow, list[JsonRow], list[JsonRow]]:
    """Validate audit inputs after tune-based ranker selection is frozen."""
    _, atomic_expected = ranker_data_contract(data_stage)
    manifest_path = data_dir / "manifest.json"
    filename = (
        "audit_o2_candidates.jsonl.gz"
        if data_stage == "O2_RANKING_DATA"
        else "audit_atomic_profile_candidates.jsonl.gz"
    )
    bundles_path = data_dir / filename
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
            raise ValueError(f"unexpected final ranker input field {key}")
    if manifest.get("output_sha256", {}).get(
        bundles_path.name
    ) != sha256_file(bundles_path):
        raise ValueError("final ranker candidate hash mismatch")
    bundles = read_gzip_jsonl(bundles_path)
    if len(bundles) != int(manifest["counts"]["audit_samples"]):
        raise ValueError("final ranker rows differ from manifest")
    ids: list[str] = []
    for index, bundle in enumerate(bundles):
        sample_id = bundle.get("id")
        if (
            not isinstance(sample_id, str)
            or sample_id in ids
            or not isinstance(bundle.get("text"), str)
            or not isinstance(bundle.get("labels"), list)
            or not bundle["labels"]
            or not isinstance(bundle.get("all_labels_in_icd"), bool)
        ):
            raise ValueError(f"invalid final ranker bundle {index}")
        _feature_list(
            bundle.get("candidates"),
            field=f"audit {sample_id}",
            profile_expected=kg_enabled,
            expected_count=expected_candidate_k,
        )
        if atomic_expected:
            _validate_atomic_query(bundle, field=f"audit {sample_id}")
        ids.append(sample_id)

    report_path = count_artifact_dir / "report.json"
    validation_path = count_artifact_dir / "validation.json"
    predictions_path = count_artifact_dir / "audit_count_predictions.jsonl.gz"
    report = read_json(report_path)
    validation = read_json(validation_path)
    report_required = {
        "spec_id": "MEDNORM-P1-003",
        "config_id": "O4-FUSION-HGB15-W8-v1",
        "stage": "O5_AUDIT_COUNT",
        "protocol": "Strict-ICD",
        "mode": "full",
        "train_split": "fit",
        "selection_split": "tune",
        "evaluation_split": "audit",
        "selection_complete": True,
        "audit_used": True,
        "official_dev_touched": False,
    }
    for key, expected_value in report_required.items():
        if report.get(key) != expected_value:
            raise ValueError(f"unexpected final count report field {key}")
    if report.get("gate", {}).get("status") != "PASS":
        raise ValueError("final count model did not pass its tune gate")
    if report.get("output_sha256", {}).get(
        predictions_path.name
    ) != sha256_file(predictions_path):
        raise ValueError("final count prediction hash mismatch")
    validation_required = {
        "status": "PASS",
        "spec_id": "MEDNORM-P1-003",
        "config_id": "O4-FUSION-HGB15-W8-v1",
        "stage": "O5_AUDIT_COUNT",
        "selection_split": "tune",
        "evaluation_split": "audit",
        "selection_complete": True,
        "audit_used": True,
    }
    for key, expected_value in validation_required.items():
        if validation.get(key) != expected_value:
            raise ValueError(f"unexpected final count validation field {key}")
    if validation.get("sha256", {}).get(
        "audit_predictions"
    ) != sha256_file(predictions_path):
        raise ValueError("validated final count prediction hash mismatch")
    count_rows = read_gzip_jsonl(predictions_path)
    if len(count_rows) != len(ids) or [row.get("id") for row in count_rows] != ids:
        raise ValueError("final count predictions are misaligned with audit")
    if any(row.get("count_prediction") not in (0, 1, 2) for row in count_rows):
        raise ValueError("final count prediction has an invalid class")
    return manifest, bundles, count_rows


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result):
        raise ValueError("retrieval weights must be non-empty and non-negative")
    return result


def amp_settings(device: torch.device) -> tuple[bool, torch.dtype]:
    enabled = device.type == "cuda"
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled, dtype


def _move_inputs(
    inputs: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def make_training_batch(
    groups: Sequence[JsonRow],
    *,
    tokenizer: Any,
    max_length: int,
    negative_count: int,
    top_hard_count: int,
    seed: int,
    epoch: int,
    query_mode: str = "auto",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, tuple[int, ...]]:
    queries: list[str] = []
    candidates: list[str] = []
    labels_by_group: list[torch.Tensor] = []
    sizes: list[int] = []
    for group in groups:
        sampled = sample_group_candidates(
            group,
            negative_count=negative_count,
            top_hard_count=top_hard_count,
            seed=seed,
            epoch=epoch,
        )
        sizes.append(len(sampled))
        queries.extend([query_text(group, query_mode=query_mode)] * len(sampled))
        candidates.extend(str(row.get("candidate_text", row["name"])) for row in sampled)
        labels_by_group.append(
            torch.tensor([bool(row["label"]) for row in sampled], dtype=torch.bool)
        )
    encodings = tokenizer(
        queries,
        candidates,
        padding=True,
        truncation="longest_first",
        max_length=max_length,
        return_tensors="pt",
    )
    positive_mask = pad_sequence(labels_by_group, batch_first=True, padding_value=False)
    valid_mask = pad_sequence(
        [torch.ones(size, dtype=torch.bool) for size in sizes],
        batch_first=True,
        padding_value=False,
    )
    return dict(encodings), positive_mask, valid_mask, tuple(sizes)


def group_batches(
    groups: Sequence[JsonRow], *, batch_size: int, seed: int, epoch: int
) -> Iterable[list[JsonRow]]:
    order = list(range(len(groups)))
    random.Random(f"{seed}:{epoch}:group-order").shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [groups[index] for index in order[start : start + batch_size]]


@torch.inference_mode()
def score_bundles(
    model: torch.nn.Module,
    tokenizer: Any,
    bundles: Sequence[JsonRow],
    *,
    candidate_k: int,
    eval_batch_size: int,
    max_length: int,
    device: torch.device,
    query_mode: str = "auto",
) -> list[np.ndarray]:
    queries: list[str] = []
    candidates: list[str] = []
    sizes: list[int] = []
    for bundle in bundles:
        features = bundle["candidates"][:candidate_k]
        sizes.append(len(features))
        queries.extend([query_text(bundle, query_mode=query_mode)] * len(features))
        candidates.extend(
            str(feature.get("candidate_text", feature["name"])) for feature in features
        )
    model.eval()
    flat_scores: list[np.ndarray] = []
    amp_enabled, amp_dtype = amp_settings(device)
    for start in range(0, len(queries), eval_batch_size):
        encoded = tokenizer(
            queries[start : start + eval_batch_size],
            candidates[start : start + eval_batch_size],
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(**_move_inputs(encoded, device)).logits.squeeze(-1)
        flat_scores.append(logits.float().sigmoid().cpu().numpy())
    flat = np.concatenate(flat_scores) if flat_scores else np.empty(0, dtype=np.float32)
    result: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        result.append(flat[offset : offset + size])
        offset += size
    if offset != len(flat):
        raise RuntimeError("scored candidate count differs from flattened rows")
    return result


@torch.inference_mode()
def score_atomic_bundles(
    model: torch.nn.Module,
    tokenizer: Any,
    bundles: Sequence[JsonRow],
    *,
    candidate_k: int,
    eval_batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Score each decomposed mention independently against the fixed candidates."""
    queries, candidates, shapes = atomic_scoring_pairs(
        bundles, candidate_k=candidate_k
    )
    model.eval()
    flat_scores: list[np.ndarray] = []
    amp_enabled, amp_dtype = amp_settings(device)
    for start in range(0, len(queries), eval_batch_size):
        encoded = tokenizer(
            queries[start : start + eval_batch_size],
            candidates[start : start + eval_batch_size],
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(**_move_inputs(encoded, device)).logits.squeeze(-1)
        flat_scores.append(logits.float().sigmoid().cpu().numpy())
    flat = np.concatenate(flat_scores) if flat_scores else np.empty(0, dtype=np.float32)
    result: list[np.ndarray] = []
    offset = 0
    for mention_count, scored_candidate_count in shapes:
        size = mention_count * scored_candidate_count
        result.append(
            flat[offset : offset + size].reshape(mention_count, scored_candidate_count)
        )
        offset += size
    if offset != len(flat):
        raise RuntimeError("atomic score count differs from flattened rows")
    return result


def decode_dataset(
    bundles: Sequence[JsonRow],
    score_arrays: Sequence[np.ndarray],
    count_predictions: Sequence[int],
    *,
    candidate_k: int,
    retrieval_weight: float,
    text_weight: float = 1.0,
) -> list[tuple[str, ...]]:
    predictions: list[tuple[str, ...]] = []
    for bundle, scores, count_class in zip(
        bundles, score_arrays, count_predictions, strict=True
    ):
        features = bundle["candidates"][:candidate_k]
        candidates = tuple(
            ScoredCandidate(
                name=str(feature["name"]),
                text_score=float(score),
                direct_score=float(feature["retrieval_score"]),
                graph_score=0.0,
            )
            for feature, score in zip(features, scores[:candidate_k], strict=True)
        )
        predictions.append(
            decode_candidates(
                text=str(bundle["text"]),
                candidates=candidates,
                count_class=int(count_class),
                threshold=0.0,
                text_weight=text_weight,
                retrieval_weight=retrieval_weight,
                graph_weight=0.0,
            )
        )
    return predictions


def tune_decoder(
    bundles: Sequence[JsonRow],
    score_arrays: Sequence[np.ndarray],
    count_predictions: Sequence[int],
    *,
    candidate_k: int,
    retrieval_weights: Sequence[float],
    text_weight: float = 1.0,
) -> JsonRow:
    gold = [tuple(str(label) for label in bundle["labels"]) for bundle in bundles]
    best: JsonRow | None = None
    grid: list[JsonRow] = []
    for retrieval_weight in retrieval_weights:
        predictions = decode_dataset(
            bundles,
            score_arrays,
            count_predictions,
            candidate_k=candidate_k,
            retrieval_weight=retrieval_weight,
            text_weight=text_weight,
        )
        metrics = asdict(evaluate_predictions(gold, predictions))
        row: JsonRow = {
            "config": {
                "candidate_k": candidate_k,
                "threshold": 0.0,
                "text_weight": text_weight,
                "retrieval_weight": retrieval_weight,
                "graph_weight": 0.0,
            },
            "metrics": metrics,
        }
        grid.append(row)
        key = (
            float(metrics["micro_f1"]),
            float(metrics["exact_match"]),
            -retrieval_weight,
        )
        if best is None or key > (
            float(best["metrics"]["micro_f1"]),
            float(best["metrics"]["exact_match"]),
            -float(best["config"]["retrieval_weight"]),
        ):
            best = row
    if best is None:
        raise RuntimeError("decoder grid is empty")
    return {**best, "grid": grid}


def segmented_metrics(
    bundles: Sequence[JsonRow], predictions: Sequence[Sequence[str]]
) -> JsonRow:
    gold = [tuple(str(label) for label in bundle["labels"]) for bundle in bundles]

    def subset(indices: Sequence[int]) -> PredictionMetrics:
        return evaluate_predictions(
            [gold[index] for index in indices],
            [predictions[index] for index in indices],
        )

    all_indices = list(range(len(bundles)))
    covered = [index for index, row in enumerate(bundles) if row["all_labels_in_icd"]]
    single = [index for index, row in enumerate(bundles) if len(row["labels"]) == 1]
    multi = [index for index, row in enumerate(bundles) if len(row["labels"]) > 1]
    return {
        "all": asdict(subset(all_indices)),
        "icd_covered": asdict(subset(covered)),
        "single_entity": asdict(subset(single)),
        "multi_entity": asdict(subset(multi)),
    }


def evaluate_ranker(
    model: torch.nn.Module,
    tokenizer: Any,
    bundles: Sequence[JsonRow],
    count_predictions: Sequence[int],
    *,
    candidate_k: int,
    retrieval_weights: Sequence[float],
    eval_batch_size: int,
    max_length: int,
    device: torch.device,
    query_mode: str = "auto",
) -> tuple[JsonRow, list[np.ndarray], list[tuple[str, ...]]]:
    scores = score_bundles(
        model,
        tokenizer,
        bundles,
        candidate_k=candidate_k,
        eval_batch_size=eval_batch_size,
        max_length=max_length,
        device=device,
        query_mode=query_mode,
    )
    decoder = tune_decoder(
        bundles,
        scores,
        count_predictions,
        candidate_k=candidate_k,
        retrieval_weights=retrieval_weights,
    )
    predictions = decode_dataset(
        bundles,
        scores,
        count_predictions,
        candidate_k=candidate_k,
        retrieval_weight=float(decoder["config"]["retrieval_weight"]),
    )
    oracle_count = [min(len(bundle["labels"]), 3) - 1 for bundle in bundles]
    oracle_decoder = tune_decoder(
        bundles,
        scores,
        oracle_count,
        candidate_k=candidate_k,
        retrieval_weights=retrieval_weights,
    )
    return (
        {
            "decoder": decoder,
            "segmented": segmented_metrics(bundles, predictions),
            "oracle_count": oracle_decoder,
        },
        scores,
        predictions,
    )


def write_scored_predictions(
    path: Path,
    bundles: Sequence[JsonRow],
    scores: Sequence[np.ndarray],
    count_predictions: Sequence[int],
    predictions: Sequence[Sequence[str]],
) -> None:
    write_gzip_jsonl(
        path,
        (
            {
                "id": bundle["id"],
                "text": bundle["text"],
                "gold_labels": bundle["labels"],
                "all_labels_in_icd": bundle["all_labels_in_icd"],
                "count_prediction": int(count_class),
                "prediction": list(prediction),
                "scored_candidates": [
                    {
                        "name": feature["name"],
                        "retrieval_score": float(feature["retrieval_score"]),
                        "text_score": float(score),
                    }
                    for feature, score in zip(
                        bundle["candidates"], score_array, strict=True
                    )
                ],
            }
            for bundle, score_array, count_class, prediction in zip(
                bundles, scores, count_predictions, predictions, strict=True
            )
        ),
    )


def write_atomic_scored_predictions(
    path: Path,
    bundles: Sequence[JsonRow],
    scores: Sequence[np.ndarray],
) -> None:
    """Persist compact per-atom neural evidence without model checkpoints."""
    if len(bundles) != len(scores):
        raise ValueError("atomic bundles and score arrays must align")
    rows: list[JsonRow] = []
    for bundle, score_array in zip(bundles, scores, strict=True):
        mentions = [str(mention) for mention in bundle["atomic_mentions"]]
        features = bundle["candidates"]
        if score_array.shape != (len(mentions), len(features)):
            raise ValueError("atomic score matrix has an invalid shape")
        rows.append(
            {
                "id": bundle["id"],
                "text": bundle["text"],
                "atomic_mentions": mentions,
                "atomic_active": bool(bundle["atomic_active"]),
                "candidates": [
                    {
                        "name": feature["name"],
                        "atom_text_scores": [
                            float(score_array[atom_index, candidate_index])
                            for atom_index in range(len(mentions))
                        ],
                    }
                    for candidate_index, feature in enumerate(features)
                ],
            }
        )
    write_gzip_jsonl(path, rows)


def run(args: argparse.Namespace) -> JsonRow:
    counts = (
        args.epochs,
        args.group_batch_size,
        args.gradient_accumulation,
        args.eval_batch_size,
        args.max_length,
        args.negative_count,
        args.final_candidate_k,
    )
    if min(counts) < 1 or not 0 <= args.top_hard_count <= args.negative_count:
        raise ValueError("training counts or hard-negative settings are invalid")
    if args.selection_candidate_k > args.final_candidate_k:
        raise ValueError("selection_candidate_k cannot exceed final_candidate_k")
    if args.score_atomic_mentions and (
        args.data_stage not in {"O4_ATOMIC_PROFILE_DATA", "O4_RR_PROFILE_DATA"}
        or args.query_mode != "raw"
    ):
        raise ValueError(
            "atomic component scoring requires O4 data with the raw base-query mode"
        )
    loss_weights = (
        args.listwise_weight,
        args.pairwise_weight,
        args.pointwise_weight,
    )
    if args.loss_mode == "pointwise" and loss_weights != (0.0, 0.0, 1.0):
        raise ValueError("pointwise control must use exactly 0/0/1 loss weights")
    if args.loss_mode == "listwise_hybrid" and (
        args.listwise_weight <= 0 or args.pairwise_weight <= 0
    ):
        raise ValueError("listwise_hybrid requires positive listwise and pairwise weights")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required unless --allow-cpu is set")
    if (args.final_data_dir is None) != (args.final_count_artifact_dir is None):
        raise ValueError("final data and final count artifact must be provided together")
    if args.final_data_dir is not None and (
        args.smoke_group_limit or args.smoke_tune_limit
    ):
        raise ValueError("audit evaluation is only allowed for a full ranker run")

    started = time.time()
    set_seed(args.seed)
    manifest, groups, tune_bundles, count_rows = validate_inputs(
        args.data_dir,
        args.count_artifact_dir,
        data_stage=args.data_stage,
        kg_enabled=args.kg_enabled,
    )
    if args.final_candidate_k != int(manifest["candidate_k"]):
        raise ValueError("final candidate_k differs from the locked O2 data")
    if args.negative_count > int(manifest["negative_pool_size"]):
        raise ValueError("requested negatives exceed the locked pool")
    if args.smoke_group_limit:
        groups = groups[: args.smoke_group_limit]
    if args.smoke_tune_limit:
        tune_bundles = tune_bundles[: args.smoke_tune_limit]
        count_rows = count_rows[: args.smoke_tune_limit]
    count_predictions = [int(row["count_prediction"]) for row in count_rows]
    mode = "smoke" if args.smoke_group_limit or args.smoke_tune_limit else "full"
    retrieval_weights = parse_float_list(args.retrieval_weights)
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
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    amp_enabled, amp_dtype = amp_settings(device)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_model"
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    history: list[JsonRow] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        update_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_groups in enumerate(
            group_batches(
                groups,
                batch_size=args.group_batch_size,
                seed=args.seed,
                epoch=epoch,
            ),
            start=1,
        ):
            encoded, positive_mask, valid_mask, sizes = make_training_batch(
                batch_groups,
                tokenizer=tokenizer,
                max_length=args.max_length,
                negative_count=args.negative_count,
                top_hard_count=args.top_hard_count,
                seed=args.seed,
                epoch=epoch,
                query_mode=args.query_mode,
            )
            positive_mask = positive_mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                flat_logits = model(**_move_inputs(encoded, device)).logits.squeeze(-1).float()
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
        evaluation, _, _ = evaluate_ranker(
            model,
            tokenizer,
            tune_bundles,
            count_predictions,
            candidate_k=args.selection_candidate_k,
            retrieval_weights=retrieval_weights,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=device,
            query_mode=args.query_mode,
        )
        row: JsonRow = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, batch_count),
            "optimizer_updates": update_count,
            "selection_candidate_k": args.selection_candidate_k,
            "tune": evaluation,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        key = (
            float(evaluation["segmented"]["all"]["micro_f1"]),
            float(evaluation["oracle_count"]["metrics"]["micro_f1"]),
            float(evaluation["segmented"]["all"]["exact_match"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selected = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, attn_implementation="sdpa"
    ).to(device)
    final_evaluation, final_scores, final_predictions = evaluate_ranker(
        selected,
        tokenizer,
        tune_bundles,
        count_predictions,
        candidate_k=args.final_candidate_k,
        retrieval_weights=retrieval_weights,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        device=device,
        query_mode=args.query_mode,
    )
    candidate_k_ablation: JsonRow = {}
    for candidate_k in (50, 100, 200, args.final_candidate_k):
        if candidate_k > args.final_candidate_k:
            continue
        candidate_k_ablation[str(candidate_k)] = tune_decoder(
            tune_bundles,
            final_scores,
            count_predictions,
            candidate_k=candidate_k,
            retrieval_weights=retrieval_weights,
        )
    retrieval_only = tune_decoder(
        tune_bundles,
        final_scores,
        count_predictions,
        candidate_k=args.final_candidate_k,
        retrieval_weights=(1.0,),
        text_weight=0.0,
    )
    predictions_path = args.output_dir / "tune_scored_predictions.jsonl.gz"
    write_scored_predictions(
        predictions_path,
        tune_bundles,
        final_scores,
        count_predictions,
        final_predictions,
    )
    atomic_predictions_path: Path | None = None
    atomic_score_count = 0
    if args.score_atomic_mentions:
        atomic_scores = score_atomic_bundles(
            selected,
            tokenizer,
            tune_bundles,
            candidate_k=args.final_candidate_k,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=device,
        )
        atomic_predictions_path = args.output_dir / "tune_atomic_scored_predictions.jsonl.gz"
        write_atomic_scored_predictions(
            atomic_predictions_path,
            tune_bundles,
            atomic_scores,
        )
        atomic_score_count = sum(int(scores.size) for scores in atomic_scores)
    final_manifest: JsonRow | None = None
    audit_evaluation: JsonRow | None = None
    audit_predictions_path: Path | None = None
    audit_atomic_predictions_path: Path | None = None
    audit_atomic_score_count = 0
    audit_sample_count = 0
    if args.final_data_dir is not None and args.final_count_artifact_dir is not None:
        final_manifest, audit_bundles, audit_count_rows = (
            validate_final_evaluation_inputs(
                args.final_data_dir,
                args.final_count_artifact_dir,
                data_stage=args.data_stage,
                kg_enabled=args.kg_enabled,
                expected_candidate_k=args.final_candidate_k,
            )
        )
        audit_count_predictions = [
            int(row["count_prediction"]) for row in audit_count_rows
        ]
        audit_scores = score_bundles(
            selected,
            tokenizer,
            audit_bundles,
            candidate_k=args.final_candidate_k,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=device,
            query_mode=args.query_mode,
        )
        frozen_retrieval_weight = float(
            final_evaluation["decoder"]["config"]["retrieval_weight"]
        )
        audit_predictions = decode_dataset(
            audit_bundles,
            audit_scores,
            audit_count_predictions,
            candidate_k=args.final_candidate_k,
            retrieval_weight=frozen_retrieval_weight,
        )
        audit_evaluation = {
            "split": "audit",
            "decoder_selected_on": "tune",
            "decoder_config": final_evaluation["decoder"]["config"],
            "segmented": segmented_metrics(audit_bundles, audit_predictions),
        }
        audit_predictions_path = args.output_dir / "audit_scored_predictions.jsonl.gz"
        write_scored_predictions(
            audit_predictions_path,
            audit_bundles,
            audit_scores,
            audit_count_predictions,
            audit_predictions,
        )
        if args.score_atomic_mentions:
            audit_atomic_scores = score_atomic_bundles(
                selected,
                tokenizer,
                audit_bundles,
                candidate_k=args.final_candidate_k,
                eval_batch_size=args.eval_batch_size,
                max_length=args.max_length,
                device=device,
            )
            audit_atomic_predictions_path = (
                args.output_dir / "audit_atomic_scored_predictions.jsonl.gz"
            )
            write_atomic_scored_predictions(
                audit_atomic_predictions_path,
                audit_bundles,
                audit_atomic_scores,
            )
            audit_atomic_score_count = sum(
                int(scores.size) for scores in audit_atomic_scores
            )
        audit_sample_count = len(audit_bundles)
    model_hash = sha256_file(checkpoint / "model.safetensors")
    all_metrics = final_evaluation["segmented"]["all"]
    single_metrics = final_evaluation["segmented"]["single_entity"]
    gate_pass = (
        float(all_metrics["micro_f1"]) >= 0.705
        and float(single_metrics["micro_f1"]) >= 0.57
        and float(all_metrics["exact_match"]) >= 0.46
    )
    if args.score_atomic_mentions:
        stage = "O4_ATOMIC_COMPONENT_POINTWISE"
    elif args.data_stage in {"O4_ATOMIC_PROFILE_DATA", "O4_RR_PROFILE_DATA"}:
        stage = (
            "O4_ATOMIC_PROFILE_POINTWISE"
            if args.loss_mode == "pointwise"
            else "O4_ATOMIC_PROFILE_LISTWISE"
        )
    elif args.kg_enabled:
        stage = (
            "O3_PROFILE_POINTWISE"
            if args.loss_mode == "pointwise"
            else "O3_PROFILE_LISTWISE"
        )
    else:
        stage = "O2_POINTWISE" if args.loss_mode == "pointwise" else "O2_LISTWISE"
    base_stage = stage
    if final_manifest is not None:
        stage = f"O5_AUDIT_{base_stage}"
    report: JsonRow = {
        "spec_id": "MEDNORM-P1-003",
        "stage": stage,
        "protocol": "Strict-ICD",
        "mode": mode,
        "candidate_vocabulary_source": "ICD only",
        "candidate_source": f"{manifest['candidate_source']} Top-400",
        "kg_enabled": args.kg_enabled,
        "query_mode": args.query_mode,
        "atomic_component_scoring": {
            "enabled": args.score_atomic_mentions,
            "split": (
                "tune_and_audit"
                if args.score_atomic_mentions and final_manifest is not None
                else ("tune" if args.score_atomic_mentions else None)
            ),
            "candidate_k": args.final_candidate_k if args.score_atomic_mentions else None,
            "score_count": atomic_score_count + audit_atomic_score_count,
            "tune_score_count": atomic_score_count,
            "audit_score_count": audit_atomic_score_count,
        },
        "train_split": "fit",
        "selection_split": "tune",
        "selection_metric": "end_to_end_micro_f1_then_oracle_count_micro_f1",
        "official_dev_touched": False,
        "audit_used": final_manifest is not None,
        "loss": {
            "mode": args.loss_mode,
            "listwise_variant": "uniform_positive_cross_entropy_v2",
            "listwise_weight": args.listwise_weight,
            "pairwise_weight": args.pairwise_weight,
            "pointwise_weight": args.pointwise_weight,
            "positive_class_weight": args.positive_class_weight,
            "pairwise_margin": args.pairwise_margin,
        },
        "model": {
            "name": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_revision,
            "best_epoch": best_epoch,
            "sha256": model_hash,
            "checkpoint_retained": args.save_model,
        },
        "fixed_count": {
            "report_sha256": sha256_file(args.count_artifact_dir / "report.json"),
            "validation_sha256": sha256_file(
                args.count_artifact_dir / "validation.json"
            ),
            "predictions_sha256": sha256_file(
                args.count_artifact_dir / "tune_count_predictions.jsonl.gz"
            ),
        },
        "decoder": final_evaluation["decoder"],
        "tune": final_evaluation["segmented"],
        "final_evaluation": audit_evaluation,
        "diagnostics": {
            "oracle_count": final_evaluation["oracle_count"],
            "candidate_k": candidate_k_ablation,
            "retrieval_only": retrieval_only,
        },
        "gate": {
            "status": (
                "SMOKE"
                if mode == "smoke"
                else (
                    (
                        "PASS_O3_PROFILE"
                        if args.kg_enabled and gate_pass
                        else "PASS_O2"
                    )
                    if args.loss_mode == "listwise_hybrid" and gate_pass
                    else "MEASURED"
                )
            ),
            "requirements": {
                "micro_f1": 0.705,
                "single_entity_micro_f1": 0.57,
                "exact_match": 0.46,
            },
            "actual": {
                "micro_f1": all_metrics["micro_f1"],
                "single_entity_micro_f1": single_metrics["micro_f1"],
                "exact_match": all_metrics["exact_match"],
            },
        },
        "training": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "group_batch_size": args.group_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "negative_count": args.negative_count,
            "top_hard_count": args.top_hard_count,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "selection_candidate_k": args.selection_candidate_k,
            "final_candidate_k": args.final_candidate_k,
            "fit_groups": len(groups),
            "tune_samples": len(tune_bundles),
            "audit_samples": audit_sample_count,
            "history": history,
        },
        "input_manifest": manifest,
        "input_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
        "final_input_manifest": final_manifest,
        "output_sha256": {
            predictions_path.name: sha256_file(predictions_path),
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
    if atomic_predictions_path is not None:
        report["output_sha256"][atomic_predictions_path.name] = sha256_file(
            atomic_predictions_path
        )
    if final_manifest is not None and args.final_data_dir is not None:
        report["config_id"] = "O4-FUSION-HGB15-W8-v1"
        report["base_stage"] = base_stage
        report["evaluation_split"] = "audit"
        report["selection_complete"] = True
        report["final_input_manifest_sha256"] = sha256_file(
            args.final_data_dir / "manifest.json"
        )
    if args.final_count_artifact_dir is not None:
        report["fixed_count"]["audit_report_sha256"] = sha256_file(
            args.final_count_artifact_dir / "report.json"
        )
        report["fixed_count"]["audit_validation_sha256"] = sha256_file(
            args.final_count_artifact_dir / "validation.json"
        )
        report["fixed_count"]["audit_predictions_sha256"] = sha256_file(
            args.final_count_artifact_dir / "audit_count_predictions.jsonl.gz"
        )
    if audit_predictions_path is not None:
        report["output_sha256"][audit_predictions_path.name] = sha256_file(
            audit_predictions_path
        )
    if audit_atomic_predictions_path is not None:
        report["output_sha256"][audit_atomic_predictions_path.name] = sha256_file(
            audit_atomic_predictions_path
        )
    del selected
    if not args.save_model:
        shutil.rmtree(checkpoint)
        report["output_sha256"].pop("best_model/model.safetensors")
    write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "stage": stage,
                "mode": mode,
                "best_epoch": best_epoch,
                "micro_f1": all_metrics["micro_f1"],
                "single_entity_micro_f1": single_metrics["micro_f1"],
                "exact_match": all_metrics["exact_match"],
                "audit_micro_f1": (
                    audit_evaluation["segmented"]["all"]["micro_f1"]
                    if audit_evaluation is not None
                    else None
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
