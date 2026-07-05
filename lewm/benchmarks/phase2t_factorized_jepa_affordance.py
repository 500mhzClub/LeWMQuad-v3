"""Phase 2T factorized affordance evaluation for JEPA-predicted futures."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from .phase2_data import action_name, source_key
from .phase2m_primitive_affordance import (
    PrimitiveAffordanceExample,
    primitive_affordance_selection_summary,
)
from .phase2o_factorized_affordance import (
    FACTORIZED_AFFORDANCE_FACTOR_NAMES,
    FACTORIZED_AFFORDANCE_TARGET_VERSION,
    factorized_candidate_targets,
)

PHASE2T_SEQUENCE_FACTOR_TARGET_VERSION = (
    "phase2t_sequence_factorized_consequence_affordance_v0"
)


def materialize_phase2t_sequence_factor_targets(
    rows: Sequence[Mapping],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sequence-level Phase 2O-compatible factor targets and masks."""

    values_and_masks = [factorized_candidate_targets(row) for row in rows]
    return (
        torch.tensor([values for values, _mask in values_and_masks], dtype=torch.float32),
        torch.tensor([mask for _values, mask in values_and_masks], dtype=torch.bool),
    )


def _safe_factor_value(values: torch.Tensor, index: int, default: float) -> torch.Tensor:
    if values.shape[-1] <= index:
        return values.new_full(values.shape[:-1], float(default))
    return values[..., index]


def sequence_factor_scores(
    factor_values: torch.Tensor,
    *,
    safe_threshold: float = 0.5,
    unsafe_threshold: float = 0.5,
    task_gain_weight: float = 0.75,
    p05_clearance_weight: float = 1.25,
    minimum_clearance_weight: float = 0.75,
    unsafe_penalty_weight: float = 2.0,
    heading_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Return safety-first sequence scores from transformed factor values."""

    values = factor_values.detach().cpu().float()
    if values.ndim != 2:
        raise ValueError("factor_values must have shape (rows, factors)")
    safe = _safe_factor_value(values, 0, 0.0)
    task_gain = _safe_factor_value(values, 1, 0.0)
    p05_clearance = _safe_factor_value(values, 2, 0.0)
    minimum_clearance = _safe_factor_value(values, 3, 0.0)
    unsafe_fraction = _safe_factor_value(values, 4, 1.0)
    heading = _safe_factor_value(values, 5, 0.0)
    base_score = (
        task_gain_weight * task_gain
        + p05_clearance_weight * p05_clearance
        + minimum_clearance_weight * minimum_clearance
        - unsafe_penalty_weight * unsafe_fraction
        + heading_weight * heading
    )
    fallback_score = safe - unsafe_fraction + 0.05 * base_score
    eligible = (safe >= safe_threshold) & (unsafe_fraction <= unsafe_threshold)
    return {
        "safe": safe,
        "task_gain": task_gain,
        "p05_clearance": p05_clearance,
        "minimum_clearance": minimum_clearance,
        "unsafe_fraction": unsafe_fraction,
        "heading": heading,
        "base_score": base_score,
        "fallback_score": fallback_score,
        "eligible": eligible,
    }


def factorized_sequence_primitive_selection_records(
    rows: Sequence[Mapping],
    factor_values: torch.Tensor,
    primitive_examples: Sequence[PrimitiveAffordanceExample],
    *,
    seed: int,
    split_name: str,
    scorer_name: str,
    safe_threshold: float = 0.5,
    unsafe_threshold: float = 0.5,
    task_gain_weight: float = 0.75,
    p05_clearance_weight: float = 1.25,
    minimum_clearance_weight: float = 0.75,
    unsafe_penalty_weight: float = 2.0,
    heading_weight: float = 0.05,
) -> list[dict]:
    """Select first primitives by scoring all candidate sequence predictions."""

    if factor_values.shape[0] != len(rows):
        raise ValueError("factor_values row count must match rows")
    by_source: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[source_key(dict(row))].append(index)
    example_by_source = {
        (example.scene_id, example.source_index): example
        for example in primitive_examples
    }
    scores = sequence_factor_scores(
        factor_values,
        safe_threshold=safe_threshold,
        unsafe_threshold=unsafe_threshold,
        task_gain_weight=task_gain_weight,
        p05_clearance_weight=p05_clearance_weight,
        minimum_clearance_weight=minimum_clearance_weight,
        unsafe_penalty_weight=unsafe_penalty_weight,
        heading_weight=heading_weight,
    )
    records = []
    for key in sorted(by_source):
        example = example_by_source.get(key)
        if example is None or example.valid_primitive_count < 2:
            continue
        valid_primitive_indices = [
            index for index, valid in enumerate(example.utility_mask) if valid
        ]
        target_values = torch.tensor(
            [
                example.utility_targets[primitive_index]
                for primitive_index in valid_primitive_indices
            ],
            dtype=torch.float32,
        )
        row_indices = by_source[key]
        eligible_rows = [
            index for index in row_indices if bool(scores["eligible"][index])
        ]
        if eligible_rows:
            selected_row_index = max(
                eligible_rows,
                key=lambda index: float(scores["base_score"][index]),
            )
            selected_score = float(scores["base_score"][selected_row_index])
            selected_by_safe_gate = True
        else:
            selected_row_index = max(
                row_indices,
                key=lambda index: float(scores["fallback_score"][index]),
            )
            selected_score = float(scores["fallback_score"][selected_row_index])
            selected_by_safe_gate = False
        selected_primitive = action_name(dict(rows[selected_row_index]), 0)
        if selected_primitive not in example.primitive_names:
            continue
        selected_primitive_index = example.primitive_names.index(selected_primitive)
        if not example.utility_mask[selected_primitive_index]:
            continue
        oracle_local = int(torch.argmax(target_values))
        oracle_primitive_index = valid_primitive_indices[oracle_local]
        selected_target_utility = float(example.utility_targets[selected_primitive_index])
        oracle_target_utility = float(example.utility_targets[oracle_primitive_index])
        records.append(
            {
                "schema": "jepa_phase2t_sequence_factorized_affordance_selection_v0",
                "seed": int(seed),
                "split": split_name,
                "scorer_name": scorer_name,
                "scene_id": key[0],
                "source_index": int(key[1]),
                "candidate_rows": len(row_indices),
                "valid_primitive_count": int(example.valid_primitive_count),
                "selected_primitive": selected_primitive,
                "oracle_primitive": example.primitive_names[oracle_primitive_index],
                "oracle_row_index": example.oracle_row_index,
                "oracle_sequence": list(example.oracle_sequence),
                "selected_sequence": list(
                    rows[selected_row_index].get("primitive_sequence", ())
                ),
                "selected_row_index": int(selected_row_index),
                "selected_predicted_utility": selected_score,
                "oracle_predicted_utility": None,
                "selected_target_utility": selected_target_utility,
                "oracle_target_utility": oracle_target_utility,
                "target_utility_regret": (
                    oracle_target_utility - selected_target_utility
                ),
                "primitive_match": selected_primitive_index == oracle_primitive_index,
                "uniform_random_primitive_match_rate": (
                    1.0 / len(valid_primitive_indices)
                ),
                "selected_by_predicted_safe_gate": selected_by_safe_gate,
                "predicted_safe_candidate_count": len(eligible_rows),
                "selected_predicted_safe_recoverable": float(
                    scores["safe"][selected_row_index]
                ),
                "selected_predicted_unsafe_fraction": float(
                    scores["unsafe_fraction"][selected_row_index]
                ),
                "selected_predicted_task_gain_norm": float(
                    scores["task_gain"][selected_row_index]
                ),
                "selected_predicted_p05_clearance_norm": float(
                    scores["p05_clearance"][selected_row_index]
                ),
                "selected_predicted_minimum_clearance_norm": float(
                    scores["minimum_clearance"][selected_row_index]
                ),
            }
        )
    return records


def factorized_sequence_primitive_selection_summary(records: Sequence[dict]) -> dict | None:
    """Summarize sequence-scored first-primitive choices."""

    return primitive_affordance_selection_summary(records)


def phase2t_sequence_factor_target_audit(rows: Sequence[Mapping]) -> dict:
    """Summarize sequence-level factor target coverage."""

    values, mask = materialize_phase2t_sequence_factor_targets(rows)
    return {
        "schema": "jepa_phase2t_sequence_factor_target_audit_v0",
        "target_version": PHASE2T_SEQUENCE_FACTOR_TARGET_VERSION,
        "source_target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "candidate_rows": len(rows),
        "factor_names": list(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        "factor_count": len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        "factor_target_values": int(mask.sum()),
        "core_factor_target_values": int(mask[:, :-1].sum()) if mask.numel() else 0,
        "finite_values": bool(torch.isfinite(values).all()),
    }
