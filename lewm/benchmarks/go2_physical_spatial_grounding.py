"""Pure metrics for the Go2 physical spatial-grounding diagnostic.

This module performs no file I/O.  The standalone diagnostic script owns the
development-role access boundary and supplies frozen model outputs here.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


ALLOWED_ROLES = frozenset({"train", "checkpoint_selection"})
DISTANCE_BINS_M = (
    ("0.0_to_0.5", 0.0, 0.5),
    ("0.5_to_1.0", 0.5, 1.0),
    ("1.0_to_2.0", 1.0, 2.0),
    ("2.0_to_3.0", 2.0, 3.0),
    ("3.0_plus", 3.0, None),
)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_maximum_mismatch_permutation(
    image_keys: Sequence[str],
    *,
    seed: int,
    namespace: str = "",
) -> np.ndarray:
    """Return a deterministic maximum-mismatch permutation of a multiset.

    Sorting equal keys contiguously and rotating by the maximum multiplicity
    attains the multiset derangement bound.  The SHA-ranked tie order makes the
    source-record assignment stable without relying on input sort stability.
    """

    keys = tuple(str(value) for value in image_keys)
    count = len(keys)
    if count <= 1:
        return np.arange(count, dtype=np.int64)
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    maximum_multiplicity = max(counts.values())
    ordered = sorted(
        range(count),
        key=lambda index: (
            keys[index],
            hashlib.sha256(
                f"{int(seed)}\0{namespace}\0{index}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    permutation = np.empty(count, dtype=np.int64)
    for position, target_index in enumerate(ordered):
        permutation[target_index] = ordered[
            (position + maximum_multiplicity) % count
        ]
    achieved = sum(
        keys[index] != keys[int(permutation[index])] for index in range(count)
    )
    theoretical = count - max(0, 2 * maximum_multiplicity - count)
    if achieved != theoretical:
        raise AssertionError(
            "multiset rotation failed to attain maximum mismatch: "
            f"achieved={achieved} theoretical={theoretical}"
        )
    return permutation


def _dilate_eight_connected(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("visibility mask must be two-dimensional")
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    result = np.zeros_like(source)
    height, width = source.shape
    for row_offset in range(3):
        for col_offset in range(3):
            result |= padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
    return result


def visibility_regions(center_visible: np.ndarray) -> dict[str, np.ndarray]:
    """Partition a grid into visible interior and two exterior 3x3-conv rings."""

    interior = np.asarray(center_visible, dtype=bool)
    if interior.ndim != 2:
        raise ValueError("center-visible mask must be two-dimensional")
    through_ring_one = _dilate_eight_connected(interior)
    ring_one = through_ring_one & ~interior
    through_ring_two = _dilate_eight_connected(through_ring_one)
    ring_two = through_ring_two & ~through_ring_one
    beyond = ~(interior | ring_one | ring_two)
    return {
        "center_visible_interior": interior,
        "exterior_ring_1": ring_one,
        "exterior_ring_2": ring_two,
        "outside_ring_2": beyond,
    }


def distance_bin_masks(distances_m: np.ndarray) -> dict[str, np.ndarray]:
    distances = np.asarray(distances_m, dtype=np.float64)
    if distances.ndim != 2 or not np.isfinite(distances).all():
        raise ValueError("distance grid must be a finite two-dimensional array")
    result = {}
    for name, lower, upper in DISTANCE_BINS_M:
        mask = distances >= float(lower)
        if upper is not None:
            mask &= distances < float(upper)
        result[name] = mask
    if not np.logical_or.reduce(tuple(result.values())).all():
        raise ValueError("distance bins do not cover the grid")
    return result


def empty_loss_accumulator() -> dict[str, float | int]:
    return {
        "joint_nll_sum": 0.0,
        "joint_count": 0,
        "joint_correct": 0,
        "unknown_known_weighted_nll_sum": 0.0,
        "unknown_known_weight_sum": 0.0,
        "known_free_occupied_weighted_nll_sum": 0.0,
        "known_free_occupied_weight_sum": 0.0,
        "known_free_occupied_nll_sum": 0.0,
        "known_count": 0,
        "known_correct": 0,
    }


def merge_accumulator(
    destination: dict[str, float | int], source: Mapping[str, float | int]
) -> None:
    if set(destination) != set(source):
        raise ValueError("accumulator schemas differ")
    for key, value in source.items():
        destination[key] += value


def _log_softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def loss_accumulator_for_batch(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    unknown_known_weights: Sequence[float],
    free_occupied_weights: Sequence[float],
) -> dict[str, float | int]:
    """Accumulate raw joint and training-objective-aligned physical NLL terms."""

    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("logits must have shape [N, 3, H, W]")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    if truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("labels do not match logits")
    if valid.shape != truth.shape:
        raise ValueError("mask does not match labels")
    if truth.size and (truth.min() < 0 or truth.max() > 2):
        raise ValueError("labels must be UNKNOWN/FREE/OCCUPIED")
    uk_weights = np.asarray(unknown_known_weights, dtype=np.float64)
    fo_weights = np.asarray(free_occupied_weights, dtype=np.float64)
    if uk_weights.shape != (2,) or fo_weights.shape != (2,):
        raise ValueError("hierarchical weights must each contain two values")
    if (
        not np.isfinite(uk_weights).all()
        or not np.isfinite(fo_weights).all()
        or (uk_weights <= 0.0).any()
        or (fo_weights <= 0.0).any()
    ):
        raise ValueError("hierarchical weights must be finite and positive")

    result = empty_loss_accumulator()
    joint_log_prob = _log_softmax(values, axis=1)
    joint_selected = np.take_along_axis(
        joint_log_prob, truth[:, None], axis=1
    )[:, 0]
    result["joint_nll_sum"] = float((-joint_selected[valid]).sum())
    result["joint_count"] = int(valid.sum())
    joint_prediction = values.argmax(axis=1)
    result["joint_correct"] = int(((joint_prediction == truth) & valid).sum())

    known_logit = np.logaddexp(values[:, 1], values[:, 2])
    unknown_known_logits = np.stack((values[:, 0], known_logit), axis=1)
    unknown_known_log_prob = _log_softmax(unknown_known_logits, axis=1)
    unknown_known_truth = (truth != 0).astype(np.int64)
    unknown_known_selected = np.take_along_axis(
        unknown_known_log_prob, unknown_known_truth[:, None], axis=1
    )[:, 0]
    applied_uk = uk_weights[unknown_known_truth] * valid
    result["unknown_known_weighted_nll_sum"] = float(
        (-unknown_known_selected * applied_uk).sum()
    )
    result["unknown_known_weight_sum"] = float(applied_uk.sum())

    known = valid & (truth != 0)
    free_occupied_truth = np.clip(truth - 1, 0, 1)
    free_occupied_log_prob = _log_softmax(values[:, 1:], axis=1)
    free_occupied_selected = np.take_along_axis(
        free_occupied_log_prob, free_occupied_truth[:, None], axis=1
    )[:, 0]
    applied_fo = fo_weights[free_occupied_truth] * known
    result["known_free_occupied_weighted_nll_sum"] = float(
        (-free_occupied_selected * applied_fo).sum()
    )
    result["known_free_occupied_weight_sum"] = float(applied_fo.sum())
    result["known_free_occupied_nll_sum"] = float(
        (-free_occupied_selected[known]).sum()
    )
    result["known_count"] = int(known.sum())
    known_prediction = values[:, 1:].argmax(axis=1)
    result["known_correct"] = int(
        ((known_prediction == free_occupied_truth) & known).sum()
    )
    return result


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return None if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def finalize_loss_accumulator(
    accumulator: Mapping[str, float | int],
) -> dict[str, Any]:
    uk_nll = _safe_ratio(
        accumulator["unknown_known_weighted_nll_sum"],
        accumulator["unknown_known_weight_sum"],
    )
    fo_weighted_nll = _safe_ratio(
        accumulator["known_free_occupied_weighted_nll_sum"],
        accumulator["known_free_occupied_weight_sum"],
    )
    hierarchical = (
        None
        if uk_nll is None or fo_weighted_nll is None
        else 0.5 * uk_nll + 0.5 * fo_weighted_nll
    )
    return {
        "raw_joint_nll": _safe_ratio(
            accumulator["joint_nll_sum"], accumulator["joint_count"]
        ),
        "raw_joint_accuracy": _safe_ratio(
            accumulator["joint_correct"], accumulator["joint_count"]
        ),
        "raw_hierarchical_balanced_nll": hierarchical,
        "raw_unknown_known_weighted_nll": uk_nll,
        "raw_known_free_occupied_weighted_nll": fo_weighted_nll,
        "raw_known_free_occupied_nll": _safe_ratio(
            accumulator["known_free_occupied_nll_sum"],
            accumulator["known_count"],
        ),
        "raw_known_free_occupied_accuracy": _safe_ratio(
            accumulator["known_correct"], accumulator["known_count"]
        ),
        "cell_count": int(accumulator["joint_count"]),
        "known_cell_count": int(accumulator["known_count"]),
    }


def empty_physical_accumulator() -> dict[str, int]:
    return {
        "true_free": 0,
        "true_occupied": 0,
        "true_unknown": 0,
        "admitted": 0,
        "admitted_true_free": 0,
        "admitted_true_unknown": 0,
        "detected_true_occupied": 0,
    }


def physical_accumulator_for_batch(
    probabilities: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    free_probability_min: float,
    occupied_probability_max: float,
    unknown_probability_max: float,
    occupied_detection_min: float,
) -> dict[str, int]:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool)
    if probs.shape != (truth.shape[0], 3, truth.shape[1], truth.shape[2]):
        raise ValueError("probabilities do not match labels")
    if (
        not np.isfinite(probs).all()
        or (probs < 0.0).any()
        or (probs > 1.0).any()
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("class probabilities must sum to one")
    if not np.isin(truth, (0, 1, 2)).all():
        raise ValueError("labels must be UNKNOWN/FREE/OCCUPIED")
    admitted = (
        (probs[:, 1] >= float(free_probability_min))
        & (probs[:, 2] <= float(occupied_probability_max))
        & (probs[:, 0] <= float(unknown_probability_max))
        & valid
    )
    detected = (probs[:, 2] >= float(occupied_detection_min)) & valid
    free = valid & (truth == 1)
    occupied = valid & (truth == 2)
    unknown = valid & (truth == 0)
    return {
        "true_free": int(free.sum()),
        "true_occupied": int(occupied.sum()),
        "true_unknown": int(unknown.sum()),
        "admitted": int(admitted.sum()),
        "admitted_true_free": int((admitted & free).sum()),
        "admitted_true_unknown": int((admitted & unknown).sum()),
        "detected_true_occupied": int((detected & occupied).sum()),
    }


def finalize_physical_accumulator(
    accumulator: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "admitted_observable_physical_free_precision": _safe_ratio(
            accumulator["admitted_true_free"], accumulator["admitted"]
        ),
        "useful_observable_physical_free_recall": _safe_ratio(
            accumulator["admitted_true_free"], accumulator["true_free"]
        ),
        "directly_observable_physical_obstacle_recall": _safe_ratio(
            accumulator["detected_true_occupied"], accumulator["true_occupied"]
        ),
        "unknown_evidence_admission_rate": _safe_ratio(
            accumulator["admitted_true_unknown"], accumulator["true_unknown"]
        ),
        **{key: int(value) for key, value in accumulator.items()},
    }


def alignment_transform_specs(max_shift: int = 3) -> tuple[dict[str, Any], ...]:
    if max_shift < 0:
        raise ValueError("max_shift must be nonnegative")
    specs = []
    for row_shift in range(-max_shift, max_shift + 1):
        for col_shift in range(-max_shift, max_shift + 1):
            name = (
                "identity"
                if row_shift == 0 and col_shift == 0
                else f"shift_row_{row_shift:+d}_col_{col_shift:+d}"
            )
            specs.append(
                {
                    "name": name,
                    "kind": "shift",
                    "row_shift": row_shift,
                    "col_shift": col_shift,
                }
            )
    specs.extend(
        (
            {"name": "horizontal_flip", "kind": "horizontal_flip"},
            {"name": "transpose", "kind": "transpose"},
        )
    )
    return tuple(specs)


def alignment_accumulators_for_batch(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    unknown_known_weights: Sequence[float],
    free_occupied_weights: Sequence[float],
    max_shift: int = 3,
) -> dict[str, dict[str, float | int]]:
    """Score fixed spatial transforms on one equal-support interior crop."""

    values = np.asarray(logits)
    truth = np.asarray(labels)
    valid = np.asarray(mask)
    if values.ndim != 4 or values.shape[2] != values.shape[3]:
        raise ValueError("alignment logits must use a square spatial grid")
    size = values.shape[2]
    margin = int(max_shift)
    if size <= 2 * margin:
        raise ValueError("alignment margin consumes the spatial grid")
    target_slice = slice(margin, size - margin)
    target_labels = truth[:, target_slice, target_slice]
    target_mask = valid[:, target_slice, target_slice]
    result = {}
    for spec in alignment_transform_specs(max_shift):
        if spec["kind"] == "shift":
            row_shift = int(spec["row_shift"])
            col_shift = int(spec["col_shift"])
            row_source = slice(margin + row_shift, size - margin + row_shift)
            col_source = slice(margin + col_shift, size - margin + col_shift)
            transformed = values[:, :, row_source, col_source]
        elif spec["kind"] == "horizontal_flip":
            transformed = values[:, :, target_slice, ::-1][:, :, :, target_slice]
        elif spec["kind"] == "transpose":
            transformed = values.swapaxes(2, 3)[:, :, target_slice, target_slice]
        else:  # pragma: no cover - construction is closed above
            raise AssertionError(spec)
        result[str(spec["name"])] = loss_accumulator_for_batch(
            transformed,
            target_labels,
            target_mask,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
        )
    return result


def grounding_contrast(
    condition_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return loss increases relative to correct RGB for two controls."""

    required = {
        "correct_rgb",
        "role_global_shuffled_rgb",
        "per_channel_mean_rgb",
    }
    if set(condition_metrics) != required:
        raise ValueError("grounding conditions are incomplete")
    result: dict[str, Any] = {}
    for control in ("role_global_shuffled_rgb", "per_channel_mean_rgb"):
        deltas = {}
        for name in (
            "raw_joint_nll",
            "raw_hierarchical_balanced_nll",
            "raw_known_free_occupied_nll",
        ):
            correct = condition_metrics["correct_rgb"].get(name)
            changed = condition_metrics[control].get(name)
            deltas[f"{control}_minus_correct_{name}"] = (
                None
                if correct is None or changed is None
                else float(changed) - float(correct)
            )
        result[control] = deltas
    return result


__all__ = [
    "ALLOWED_ROLES",
    "DISTANCE_BINS_M",
    "alignment_accumulators_for_batch",
    "alignment_transform_specs",
    "canonical_json_sha256",
    "deterministic_maximum_mismatch_permutation",
    "distance_bin_masks",
    "empty_loss_accumulator",
    "empty_physical_accumulator",
    "finalize_loss_accumulator",
    "finalize_physical_accumulator",
    "grounding_contrast",
    "loss_accumulator_for_batch",
    "merge_accumulator",
    "physical_accumulator_for_batch",
    "visibility_regions",
]
