"""Phase 2Q true-factor ceiling for factorized primitive affordance selection."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .phase2m_primitive_affordance import (
    build_primitive_affordance_examples,
    evaluate_primitive_action_only_baseline,
    fit_primitive_action_priors,
    primitive_affordance_selection_summary,
    primitive_vocabulary,
)
from .phase2o_factorized_affordance import (
    FACTORIZED_AFFORDANCE_FACTOR_NAMES,
    FACTORIZED_AFFORDANCE_TARGET_VERSION,
    FactorizedPrimitiveAffordanceExample,
    build_factorized_primitive_affordance_examples,
    factorized_affordance_dataset_audit,
    factorized_affordance_selection_records,
)

PHASE2Q_FACTORIZED_CEILING_SCHEMA = (
    "jepa_phase2q_factorized_affordance_ceiling_audit_v0"
)

DEFAULT_PHASE2Q_SELECTION_KWARGS = {
    "safe_threshold": 0.5,
    "unsafe_threshold": 0.5,
    "task_gain_weight": 0.75,
    "p05_clearance_weight": 1.25,
    "minimum_clearance_weight": 0.75,
    "unsafe_penalty_weight": 2.0,
    "heading_weight": 0.05,
}


def true_factor_values_tensor(
    examples: Sequence[FactorizedPrimitiveAffordanceExample],
) -> torch.Tensor:
    """Return the registered factor targets as selector-ready values."""

    if not examples:
        raise ValueError("cannot build true-factor values for an empty split")
    primitive_names = examples[0].primitive_names
    if any(example.primitive_names != primitive_names for example in examples):
        raise ValueError("all examples must share one primitive vocabulary")
    return torch.tensor(
        [example.factor_targets for example in examples],
        dtype=torch.float32,
    )


def evaluate_true_factor_affordance_ceiling(
    examples: Sequence[FactorizedPrimitiveAffordanceExample],
    *,
    seed: int,
    split_name: str,
    selection_kwargs: Mapping[str, float] | None = None,
) -> dict:
    """Evaluate the safety-first selector using true Phase 2O factor targets."""

    kwargs = {
        **DEFAULT_PHASE2Q_SELECTION_KWARGS,
        **dict(selection_kwargs or {}),
    }
    records = factorized_affordance_selection_records(
        examples,
        true_factor_values_tensor(examples),
        seed=seed,
        split_name=split_name,
        scorer_name="true_phase2o_factor_targets",
        **kwargs,
    )
    return {
        "schema": "jepa_phase2q_true_factor_selection_diagnostic_v0",
        "split": split_name,
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "factor_names": list(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        "selection_rule": {
            "schema": "jepa_phase2p_safety_first_selection_rule_v0",
            **kwargs,
        },
        "selection_records": records,
        "selection_summary": primitive_affordance_selection_summary(records),
    }


def phase2q_factorized_affordance_ceiling_audit(
    *,
    train_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    seed: int,
    selection_kwargs: Mapping[str, float] | None = None,
) -> dict:
    """Build the Phase 2Q true-factor ceiling audit on train/validation rows."""

    primitive_names = primitive_vocabulary(train_rows)
    train_examples = build_factorized_primitive_affordance_examples(
        train_rows,
        primitive_names=primitive_names,
    )
    validation_examples = build_factorized_primitive_affordance_examples(
        validation_rows,
        primitive_names=primitive_names,
    )
    baseline_train_examples = build_primitive_affordance_examples(
        train_rows,
        primitive_names=primitive_names,
    )
    baseline_validation_examples = build_primitive_affordance_examples(
        validation_rows,
        primitive_names=primitive_names,
    )
    primitive_priors = fit_primitive_action_priors(baseline_train_examples)
    primitive_action_only_baseline = evaluate_primitive_action_only_baseline(
        baseline_validation_examples,
        primitive_priors,
        split_name="validation",
        seed=seed,
    )
    train_diagnostic = evaluate_true_factor_affordance_ceiling(
        train_examples,
        seed=seed,
        split_name="train",
        selection_kwargs=selection_kwargs,
    )
    validation_diagnostic = evaluate_true_factor_affordance_ceiling(
        validation_examples,
        seed=seed,
        split_name="validation",
        selection_kwargs=selection_kwargs,
    )
    return {
        "schema": PHASE2Q_FACTORIZED_CEILING_SCHEMA,
        "confirmatory_result": False,
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "seed": int(seed),
        "primitive_names": list(primitive_names),
        "factor_names": list(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        "primitive_action_priors": primitive_priors,
        "primitive_action_only_baseline": primitive_action_only_baseline,
        "train_data": {
            "factorized_affordance_audit": factorized_affordance_dataset_audit(
                train_examples,
                split_name="train",
            ),
            "true_factor_selection_diagnostic": train_diagnostic,
        },
        "validation_data": {
            "factorized_affordance_audit": factorized_affordance_dataset_audit(
                validation_examples,
                split_name="validation",
            ),
            "true_factor_selection_diagnostic": validation_diagnostic,
        },
        "final_validation": {
            "primitive_affordance_selection_summary": validation_diagnostic[
                "selection_summary"
            ],
            "primitive_affordance_selection_records": validation_diagnostic[
                "selection_records"
            ],
        },
        "limitations": [
            "privileged true-factor ceiling only",
            "train and validation evidence only",
            "test_id and test_hard are not used for reported metrics or model selection",
            "factor targets are generator-derived supervision",
            "this is not a learned model and not a JEPA world model",
        ],
    }
