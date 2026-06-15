"""Phase 2O factorized geometry-derived primitive affordance targets."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .phase2_data import action_name, source_key
from .phase2d_training import ACTION_UTILITY_TARGET_VERSION, action_utility_target
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
