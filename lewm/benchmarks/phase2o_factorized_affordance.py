"""Phase 2O factorized geometry-derived primitive affordance targets."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .phase2_data import action_name, source_key
from .phase2d_training import (
    ACTION_UTILITY_TARGET_VERSION,
    action_utility_target,
    image_tensor,
)
from .phase2m_primitive_affordance import primitive_vocabulary

FACTORIZED_AFFORDANCE_TARGET_VERSION = (
    "phase2o_factorized_first_primitive_affordance_v0"
)
FACTORIZED_AFFORDANCE_FACTOR_NAMES = (
    "safe_recoverable",
    "task_gain_norm",
    "p05_clearance_norm",
    "minimum_clearance_norm",
    "unsafe_sample_fraction",
    "heading_alignment",
)
FACTORIZED_AFFORDANCE_CORE_FACTOR_NAMES = FACTORIZED_AFFORDANCE_FACTOR_NAMES[:-1]


@dataclass(frozen=True)
class FactorizedPrimitiveAffordanceExample:
    """One source state with utility-selected factor targets per first primitive."""

    scene_id: str
    source_index: int
    start_frame: str
    primitive_names: tuple[str, ...]
    utility_targets: tuple[float, ...]
    utility_mask: tuple[bool, ...]
    factor_targets: tuple[tuple[float, ...], ...]
    factor_mask: tuple[tuple[bool, ...], ...]
    candidate_rows: int
    valid_utility_rows: int
    valid_primitive_count: int
    oracle_primitive: str | None
    oracle_row_index: int | None
    oracle_sequence: tuple[str, ...]


@dataclass
class Phase2PFactorizedBatch:
    """Materialized source-image batch with factorized primitive targets."""

    example_indices: tuple[int, ...]
    examples: tuple[FactorizedPrimitiveAffordanceExample, ...]
    start_vision: torch.Tensor
    primitive_utility_targets: torch.Tensor
    primitive_utility_mask: torch.Tensor
    factor_targets: torch.Tensor
    factor_mask: torch.Tensor

    def to(self, device: torch.device) -> "Phase2PFactorizedBatch":
        return Phase2PFactorizedBatch(
            example_indices=self.example_indices,
            examples=self.examples,
            start_vision=self.start_vision.to(device),
            primitive_utility_targets=self.primitive_utility_targets.to(device),
            primitive_utility_mask=self.primitive_utility_mask.to(device),
            factor_targets=self.factor_targets.to(device),
            factor_mask=self.factor_mask.to(device),
        )


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamped(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _primitive_sequence(row: Mapping) -> tuple[str, ...]:
    return tuple(str(value) for value in row.get("primitive_sequence", ()))


def factorized_candidate_targets(row: Mapping) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return factorized geometry-derived labels for one candidate sequence."""

    labels = row.get("consequence_labels")
    if not isinstance(labels, Mapping):
        return (
            tuple(0.0 for _name in FACTORIZED_AFFORDANCE_FACTOR_NAMES),
            tuple(False for _name in FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        )

    progress = _finite_float(labels.get("target_progress_m"))
    clearance_gain = _finite_float(labels.get("clearance_gain_m"))
    task_gain = progress if progress is not None else clearance_gain
    p05_clearance = _finite_float(labels.get("p05_swept_configuration_clearance_m"))
    minimum_clearance = _finite_float(
        labels.get("minimum_swept_configuration_clearance_m")
    )
    unsafe_fraction = _finite_float(labels.get("unsafe_sample_fraction"))
    heading_error = _finite_float(labels.get("target_heading_error_rad"))
    has_safety_booleans = all(
        key in labels
        for key in (
            "enters_grid_unsafe",
            "ends_grid_unsafe",
            "target_recoverable",
        )
    )
    safety_valid = has_safety_booleans and unsafe_fraction is not None
    if safety_valid:
        safe_recoverable = (
            not bool(labels["enters_grid_unsafe"])
            and not bool(labels["ends_grid_unsafe"])
            and bool(labels["target_recoverable"])
            and unsafe_fraction <= 0.0
        )
    else:
        safe_recoverable = False

    values = (
        1.0 if safe_recoverable else 0.0,
        0.0 if task_gain is None else _clamped(task_gain / 0.3, -1.0, 1.0),
        0.0
        if p05_clearance is None
        else _clamped((p05_clearance + 0.2) / 1.2, 0.0, 1.0),
        0.0
        if minimum_clearance is None
        else _clamped((minimum_clearance + 0.2) / 1.2, 0.0, 1.0),
        0.0
        if unsafe_fraction is None
        else _clamped(unsafe_fraction, 0.0, 1.0),
        0.0
        if heading_error is None
        else 1.0 - _clamped(abs(heading_error), 0.0, math.pi) / math.pi,
    )
    mask = (
        safety_valid,
        task_gain is not None,
        p05_clearance is not None,
        minimum_clearance is not None,
        unsafe_fraction is not None,
        heading_error is not None,
    )
    return values, mask


def build_factorized_primitive_affordance_examples(
    rows: Sequence[dict],
    *,
    primitive_names: Sequence[str] | None = None,
) -> tuple[FactorizedPrimitiveAffordanceExample, ...]:
    """Build utility-selected factor targets for each source and first primitive."""

    names = (
        primitive_vocabulary(rows)
        if primitive_names is None
        else tuple(str(name) for name in primitive_names)
    )
    if not names:
        raise ValueError("primitive_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("primitive_names must be unique")
    primitive_to_index = {name: index for index, name in enumerate(names)}
    grouped: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[source_key(row)].append((row_index, row))

    examples = []
    for key, source_rows in sorted(grouped.items()):
        start_frames = {str(row["start_frame"]) for _row_index, row in source_rows}
        if len(start_frames) != 1:
            raise ValueError(
                "all rows for one source state must share one start_frame: "
                f"{key}"
            )
        best_by_primitive: dict[
            str,
            tuple[float, int, tuple[str, ...], tuple[float, ...], tuple[bool, ...]],
        ] = {}
        valid_utility_rows = 0
        for row_index, row in source_rows:
            first = action_name(row, 0)
            if first not in primitive_to_index:
                raise ValueError(
                    f"row contains first primitive outside vocabulary: {first}"
                )
            utility, valid = action_utility_target(row)
            if not valid:
                continue
            valid_utility_rows += 1
            factors, factor_mask = factorized_candidate_targets(row)
            previous = best_by_primitive.get(first)
            if previous is None or float(utility) > previous[0]:
                best_by_primitive[first] = (
                    float(utility),
                    row_index,
                    _primitive_sequence(row),
                    factors,
                    factor_mask,
                )

        utility_targets = [0.0 for _name in names]
        utility_mask = [False for _name in names]
        factor_targets = [
            [0.0 for _factor in FACTORIZED_AFFORDANCE_FACTOR_NAMES]
            for _name in names
        ]
        factor_mask = [
            [False for _factor in FACTORIZED_AFFORDANCE_FACTOR_NAMES]
            for _name in names
        ]
        for primitive, record in best_by_primitive.items():
            primitive_index = primitive_to_index[primitive]
            utility, _row_index, _sequence, factors, factors_valid = record
            utility_targets[primitive_index] = utility
            utility_mask[primitive_index] = True
            factor_targets[primitive_index] = list(factors)
            factor_mask[primitive_index] = list(factors_valid)

        oracle_primitive = None
        oracle_row_index = None
        oracle_sequence: tuple[str, ...] = ()
        if any(utility_mask):
            oracle_index = max(
                (index for index, valid in enumerate(utility_mask) if valid),
                key=lambda index: utility_targets[index],
            )
            oracle_primitive = names[oracle_index]
            _utility, oracle_row_index, oracle_sequence, _factors, _mask = (
                best_by_primitive[oracle_primitive]
            )

        examples.append(
            FactorizedPrimitiveAffordanceExample(
                scene_id=key[0],
                source_index=int(key[1]),
                start_frame=next(iter(start_frames)),
                primitive_names=names,
                utility_targets=tuple(utility_targets),
                utility_mask=tuple(utility_mask),
                factor_targets=tuple(tuple(values) for values in factor_targets),
                factor_mask=tuple(tuple(values) for values in factor_mask),
                candidate_rows=len(source_rows),
                valid_utility_rows=valid_utility_rows,
                valid_primitive_count=sum(utility_mask),
                oracle_primitive=oracle_primitive,
                oracle_row_index=oracle_row_index,
                oracle_sequence=oracle_sequence,
            )
        )
    return tuple(examples)


def materialize_phase2p_factorized_batch(
    examples: Sequence[FactorizedPrimitiveAffordanceExample],
    indices: Sequence[int],
    *,
    image_size: int = 224,
) -> Phase2PFactorizedBatch:
    """Build one source-image factorized affordance training batch."""

    example_indices = tuple(int(index) for index in indices)
    if not example_indices:
        raise ValueError("cannot materialize an empty Phase 2P batch")
    selected = tuple(examples[index] for index in example_indices)
    primitive_names = selected[0].primitive_names
    if any(example.primitive_names != primitive_names for example in selected):
        raise ValueError("all Phase 2P examples in a batch must share vocabulary")
    image_cache: dict[Path, torch.Tensor] = {}

    def cached_image(path: Path) -> torch.Tensor:
        cached = image_cache.get(path)
        if cached is None:
            cached = image_tensor(path, image_size=image_size)
            image_cache[path] = cached
        return cached

    return Phase2PFactorizedBatch(
        example_indices=example_indices,
        examples=selected,
        start_vision=torch.stack(
            [cached_image(Path(example.start_frame)) for example in selected]
        ),
        primitive_utility_targets=torch.tensor(
            [example.utility_targets for example in selected],
            dtype=torch.float32,
        ),
        primitive_utility_mask=torch.tensor(
            [example.utility_mask for example in selected],
            dtype=torch.bool,
        ),
        factor_targets=torch.tensor(
            [example.factor_targets for example in selected],
            dtype=torch.float32,
        ),
        factor_mask=torch.tensor(
            [example.factor_mask for example in selected],
            dtype=torch.bool,
        ),
    )


def phase2p_batch_contract_audit(batch: Phase2PFactorizedBatch) -> dict:
    """Return compact evidence for one Phase 2P materialized batch."""

    return {
        "schema": "jepa_phase2p_factorized_affordance_batch_contract_v0",
        "examples": len(batch.examples),
        "primitive_count": int(batch.factor_targets.shape[1]),
        "factor_count": int(batch.factor_targets.shape[2]),
        "primitive_utility_targets": int(batch.primitive_utility_mask.sum()),
        "factor_targets": int(batch.factor_mask.sum()),
        "core_factor_targets": int(batch.factor_mask[:, :, :-1].sum()),
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "all_start_frames_finite": bool(torch.isfinite(batch.start_vision).all()),
    }


def _safe_factor_value(values: torch.Tensor, index: int, default: float) -> torch.Tensor:
    if values.shape[-1] <= index:
        return values.new_full(values.shape[:-1], float(default))
    return values[..., index]


def factorized_affordance_selection_records(
    examples: Sequence[FactorizedPrimitiveAffordanceExample],
    factor_values: torch.Tensor,
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
    """Return primitive selection records from factorized predicted values."""

    values = factor_values.detach().cpu().float()
    if values.ndim != 3:
        raise ValueError("factor_values must have shape (B, primitive_count, factors)")
    if values.shape[0] != len(examples):
        raise ValueError("factor_values row count must match examples")
    records = []
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
    for example_index, example in enumerate(examples):
        valid_indices = [
            primitive_index
            for primitive_index, valid in enumerate(example.utility_mask)
            if valid
        ]
        if len(valid_indices) < 2:
            continue
        target_values = torch.tensor(
            [
                example.utility_targets[primitive_index]
                for primitive_index in valid_indices
            ],
            dtype=torch.float32,
        )
        valid_safe = safe[example_index, valid_indices]
        valid_unsafe = unsafe_fraction[example_index, valid_indices]
        valid_base = base_score[example_index, valid_indices]
        valid_fallback = fallback_score[example_index, valid_indices]
        eligible = (valid_safe >= safe_threshold) & (valid_unsafe <= unsafe_threshold)
        if bool(eligible.any()):
            local_scores = torch.where(
                eligible,
                valid_base,
                torch.full_like(valid_base, -1.0e9),
            )
        else:
            local_scores = valid_fallback
        selected_local = int(torch.argmax(local_scores))
        oracle_local = int(torch.argmax(target_values))
        selected_index = valid_indices[selected_local]
        oracle_index = valid_indices[oracle_local]
        selected_utility = float(target_values[selected_local])
        oracle_utility = float(target_values[oracle_local])
        records.append(
            {
                "schema": "jepa_phase2p_factorized_affordance_selection_v0",
                "seed": int(seed),
                "split": split_name,
                "scorer_name": scorer_name,
                "scene_id": example.scene_id,
                "source_index": int(example.source_index),
                "candidate_rows": int(example.candidate_rows),
                "valid_primitive_count": int(example.valid_primitive_count),
                "selected_primitive": example.primitive_names[selected_index],
                "oracle_primitive": example.primitive_names[oracle_index],
                "oracle_row_index": example.oracle_row_index,
                "oracle_sequence": list(example.oracle_sequence),
                "selected_predicted_utility": float(local_scores[selected_local]),
                "oracle_predicted_utility": float(local_scores[oracle_local]),
                "selected_target_utility": selected_utility,
                "oracle_target_utility": oracle_utility,
                "target_utility_regret": oracle_utility - selected_utility,
                "primitive_match": selected_index == oracle_index,
                "uniform_random_primitive_match_rate": 1.0 / len(valid_indices),
                "selected_predicted_safe_recoverable": float(
                    safe[example_index, selected_index]
                ),
                "selected_predicted_unsafe_fraction": float(
                    unsafe_fraction[example_index, selected_index]
                ),
                "selected_predicted_task_gain_norm": float(
                    task_gain[example_index, selected_index]
                ),
                "selected_predicted_p05_clearance_norm": float(
                    p05_clearance[example_index, selected_index]
                ),
                "selected_predicted_minimum_clearance_norm": float(
                    minimum_clearance[example_index, selected_index]
                ),
                "selected_by_predicted_safe_gate": bool(eligible[selected_local]),
                "predicted_safe_candidate_count": int(eligible.sum()),
            }
        )
    return records


def factorized_affordance_dataset_audit(
    examples: Sequence[FactorizedPrimitiveAffordanceExample],
    *,
    split_name: str,
) -> dict:
    """Summarize factorized primitive target coverage and safety structure."""

    primitive_names = examples[0].primitive_names if examples else ()
    factor_count = len(FACTORIZED_AFFORDANCE_FACTOR_NAMES)
    factor_value_counts = [0 for _factor in FACTORIZED_AFFORDANCE_FACTOR_NAMES]
    safe_positive_counts = Counter()
    safe_label_counts = Counter()
    oracle_counts = Counter(
        example.oracle_primitive
        for example in examples
        if example.oracle_primitive is not None
    )
    utility_ranges = []
    for example in examples:
        valid_utilities = [
            value
            for value, valid in zip(
                example.utility_targets,
                example.utility_mask,
                strict=True,
            )
            if valid
        ]
        if len(valid_utilities) >= 2:
            utility_ranges.append(max(valid_utilities) - min(valid_utilities))
        for primitive_index, primitive in enumerate(primitive_names):
            factor_mask = example.factor_mask[primitive_index]
            factor_values = example.factor_targets[primitive_index]
            for factor_index, valid in enumerate(factor_mask):
                factor_value_counts[factor_index] += int(valid)
            if factor_mask[0]:
                safe_label_counts[primitive] += 1
                if factor_values[0] >= 0.5:
                    safe_positive_counts[primitive] += 1
    return {
        "schema": "jepa_phase2o_factorized_affordance_dataset_audit_v0",
        "split": split_name,
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "source_states": len(examples),
        "primitive_names": list(primitive_names),
        "primitive_count": len(primitive_names),
        "factor_names": list(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        "factor_count": factor_count,
        "valid_primitive_targets": sum(
            example.valid_primitive_count for example in examples
        ),
        "factor_value_counts": dict(
            zip(
                FACTORIZED_AFFORDANCE_FACTOR_NAMES,
                factor_value_counts,
                strict=True,
            )
        ),
        "core_factor_names": list(FACTORIZED_AFFORDANCE_CORE_FACTOR_NAMES),
        "core_factors_complete": all(
            factor_value_counts[index] == len(examples) * len(primitive_names)
            for index, _name in enumerate(FACTORIZED_AFFORDANCE_CORE_FACTOR_NAMES)
        )
        if examples and primitive_names
        else False,
        "safe_positive_counts_by_primitive": dict(sorted(safe_positive_counts.items())),
        "safe_label_counts_by_primitive": dict(sorted(safe_label_counts.items())),
        "oracle_primitive_counts": dict(sorted(oracle_counts.items())),
        "mean_utility_range_per_source": (
            sum(utility_ranges) / len(utility_ranges) if utility_ranges else None
        ),
        "minimum_utility_range_per_source": min(utility_ranges, default=None),
        "maximum_utility_range_per_source": max(utility_ranges, default=None),
        "all_factors_complete": all(
            count == len(examples) * len(primitive_names)
            for count in factor_value_counts
        )
        if examples and primitive_names
        else False,
    }
