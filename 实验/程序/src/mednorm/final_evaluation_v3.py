"""Contracts for frozen P1-003 final-evaluation splits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalEvaluationContract:
    evaluation_split: str
    prefix: str
    sample_count_key: str
    retrieval_stage: str
    retrieval_result_stage: str
    atomic_input_stage: str
    atomic_generation_stage: str
    inference_stage: str
    count_stage: str
    ranker_stage_prefix: str
    fusion_stage: str
    audit_used: bool
    official_dev_touched: bool


_CONTRACTS = {
    "audit": FinalEvaluationContract(
        evaluation_split="audit",
        prefix="audit",
        sample_count_key="audit_samples",
        retrieval_stage="O5_AUDIT_RETRIEVAL_INPUT",
        retrieval_result_stage="O5_AUDIT_RETRIEVAL",
        atomic_input_stage="O5_AUDIT_ATOMIC_INPUT",
        atomic_generation_stage="O5_AUDIT_ATOMIC_GENERATION",
        inference_stage="O5_AUDIT_INFERENCE_DATA",
        count_stage="O5_AUDIT_COUNT",
        ranker_stage_prefix="O5_AUDIT_",
        fusion_stage="O5_AUDIT_FINAL_FUSION",
        audit_used=True,
        official_dev_touched=False,
    ),
    "official_dev": FinalEvaluationContract(
        evaluation_split="official_dev",
        prefix="official_dev",
        sample_count_key="official_dev_samples",
        retrieval_stage="O7_OFFICIAL_DEV_RETRIEVAL_INPUT",
        retrieval_result_stage="O7_OFFICIAL_DEV_RETRIEVAL",
        atomic_input_stage="O7_OFFICIAL_DEV_ATOMIC_INPUT",
        atomic_generation_stage="O7_OFFICIAL_DEV_ATOMIC_GENERATION",
        inference_stage="O7_OFFICIAL_DEV_INFERENCE_DATA",
        count_stage="O7_OFFICIAL_DEV_COUNT",
        ranker_stage_prefix="O7_OFFICIAL_DEV_",
        fusion_stage="O7_OFFICIAL_DEV_FINAL_FUSION",
        audit_used=True,
        official_dev_touched=True,
    ),
}


def final_evaluation_contract(evaluation_split: str) -> FinalEvaluationContract:
    try:
        return _CONTRACTS[evaluation_split]
    except KeyError as error:
        raise ValueError("final evaluation split must be audit or official_dev") from error

