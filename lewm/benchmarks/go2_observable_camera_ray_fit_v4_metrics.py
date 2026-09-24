"""Streaming fit metrics for observable camera-ray evidence V4.

The metrics operate only on learned outputs and immutable V4 targets.  They
retain exact confusion counts while keeping large model tensors off CPU; only
per-hit scalar depth errors are accumulated for the two requested quantiles.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Sequence

import torch

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    FREE_CLASS,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_SIZE_M,
    DEPTH_NEAR_EDGE_M,
    ObservableCameraRayEvidenceV4RawOutput,
    ordered_obstacle_first_hit_log_probabilities_v4,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    DEFAULT_GROUND_DISTANCE_BIN_EDGES_M,
    ObservableCameraRayEvidenceV4Targets,
    SoftObservableCameraRayRasterV4,
)


RASTER_CLASS_NAMES = ("unknown", "free", "occupied")


def _binary_confusion(
    target_positive: torch.Tensor,
    predicted_positive: torch.Tensor,
    valid: torch.Tensor,
) -> list[list[int]]:
    if not all(
        isinstance(value, torch.Tensor)
        for value in (target_positive, predicted_positive, valid)
    ):
        raise TypeError("binary confusion inputs must be tensors")
    if not (
        tuple(target_positive.shape)
        == tuple(predicted_positive.shape)
        == tuple(valid.shape)
    ):
        raise ValueError("binary confusion inputs must have one shape")
    if any(value.dtype != torch.bool for value in (target_positive, predicted_positive, valid)):
        raise ValueError("binary confusion inputs must be boolean")
    return [
        [
            int(
                (
                    valid
                    & (target_positive == target_state)
                    & (predicted_positive == predicted_state)
                )
                .sum()
                .item()
            )
            for predicted_state in (False, True)
        ]
        for target_state in (False, True)
    ]


def _add_matrix(destination: list[list[int]], source: Sequence[Sequence[int]]) -> None:
    if len(destination) != len(source) or any(
        len(destination[row]) != len(source[row]) for row in range(len(destination))
    ):
        raise ValueError("confusion matrix shape changed")
    for row in range(len(destination)):
        for column in range(len(destination[row])):
            value = int(source[row][column])
            if value < 0:
                raise ValueError("confusion counts cannot be negative")
            destination[row][column] += value


def _binary_metrics(confusion: Sequence[Sequence[int]]) -> dict[str, Any]:
    if len(confusion) != 2 or any(len(row) != 2 for row in confusion):
        raise ValueError("binary confusion must be 2x2")
    matrix = [[int(value) for value in row] for row in confusion]
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("binary confusion counts cannot be negative")
    recalls: list[float | None] = []
    for state in range(2):
        count = sum(matrix[state])
        recalls.append(None if count == 0 else matrix[state][state] / count)
    present = [value for value in recalls if value is not None]
    return {
        "confusion_target_rows_predicted_columns": matrix,
        "negative_recall": recalls[0],
        "positive_recall": recalls[1],
        "balanced_accuracy": None if not present else sum(present) / len(present),
        "count": sum(sum(row) for row in matrix),
    }


def _distance_group_names(edges: Sequence[float]) -> tuple[str, ...]:
    values = tuple(float(value) for value in edges)
    if len(values) < 2 or values[0] < 0.0:
        raise ValueError("ground distance edges are malformed")
    if any(values[index + 1] <= values[index] for index in range(len(values) - 1)):
        raise ValueError("ground distance edges must increase")
    names = []
    for low, high in zip(values[:-1], values[1:]):
        if high == float("inf"):
            names.append(f"{low:.1f}_plus")
        else:
            names.append(f"{low:.1f}_to_{high:.1f}")
    return tuple(names)


class ObservableCameraRayFitV4MetricAccumulator:
    """Accumulate the registered matched-RGB or wrong-RGB fit metrics."""

    def __init__(
        self,
        *,
        ground_distance_bin_edges_m: Sequence[float] = DEFAULT_GROUND_DISTANCE_BIN_EDGES_M,
    ) -> None:
        self._distance_edges = tuple(float(value) for value in ground_distance_bin_edges_m)
        self._distance_names = _distance_group_names(self._distance_edges)
        self._pixel_confusion = [[0, 0], [0, 0]]
        self._ground_confusion = [[0, 0], [0, 0]]
        self._ground_by_distance: dict[str, list[list[int]]] = {
            name: [[0, 0], [0, 0]] for name in self._distance_names
        }
        self._ground_by_family: dict[str, list[list[int]]] = defaultdict(
            lambda: [[0, 0], [0, 0]]
        )
        self._raster_confusion = [[0, 0, 0] for _ in range(3)]
        self._raster_nll_sum = 0.0
        self._raster_nll_count = 0
        self._depth_errors: list[torch.Tensor] = []
        self._frame_count = 0

    def update(
        self,
        *,
        raw_output: ObservableCameraRayEvidenceV4RawOutput,
        targets: ObservableCameraRayEvidenceV4Targets,
        soft_raster: SoftObservableCameraRayRasterV4,
        target_raster_labels: torch.Tensor,
        families: Sequence[str],
    ) -> None:
        if not isinstance(raw_output, ObservableCameraRayEvidenceV4RawOutput):
            raise TypeError("raw_output must be a V4 raw output")
        batch = raw_output.pixel_first_hit_hazard_logits.shape[0]
        normalized_families = tuple(str(value) for value in families)
        if len(normalized_families) != batch or any(not value for value in normalized_families):
            raise ValueError("one nonempty family is required per frame")
        if not torch.equal(raw_output.ground_query_in_frustum, targets.ground_in_frustum):
            raise ValueError("model calibration and target ground visibility differ")

        ordered = ordered_obstacle_first_hit_log_probabilities_v4(
            raw_output.pixel_first_hit_hazard_logits
        )
        predicted_hit = -torch.expm1(ordered.no_hit) >= 0.5
        _add_matrix(
            self._pixel_confusion,
            _binary_confusion(
                targets.pixel_in_range_hit_mask,
                predicted_hit,
                torch.ones_like(predicted_hit, dtype=torch.bool),
            ),
        )

        predicted_bin = ordered.hit.argmax(dim=1)
        predicted_offset = raw_output.pixel_within_bin_offset_m.gather(
            1, predicted_bin[:, None]
        ).squeeze(1)
        predicted_depth = DEPTH_NEAR_EDGE_M + (
            predicted_bin.to(dtype=predicted_offset.dtype) + 0.5
        ) * DEPTH_BIN_SIZE_M + predicted_offset
        target_depth = DEPTH_NEAR_EDGE_M + (
            targets.pixel_hit_bin_index.to(dtype=predicted_offset.dtype) + 0.5
        ) * DEPTH_BIN_SIZE_M + targets.pixel_within_bin_offset_m.to(
            dtype=predicted_offset.dtype
        )
        hit_mask = targets.pixel_in_range_hit_mask
        if bool(hit_mask.any().item()):
            self._depth_errors.append(
                (predicted_depth[hit_mask] - target_depth[hit_mask])
                .abs()
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )

        ground_valid = targets.ground_in_frustum
        ground_target = targets.ground_clear_to_target
        ground_predicted = raw_output.ground_clear_to_target_logits >= 0.0
        _add_matrix(
            self._ground_confusion,
            _binary_confusion(ground_target, ground_predicted, ground_valid),
        )
        distance = raw_output.ground_target_distance_m
        for name, low, high in zip(
            self._distance_names,
            self._distance_edges[:-1],
            self._distance_edges[1:],
        ):
            mask = ground_valid & (distance >= low) & (distance < high)
            _add_matrix(
                self._ground_by_distance[name],
                _binary_confusion(ground_target, ground_predicted, mask),
            )
        for frame_index, family in enumerate(normalized_families):
            _add_matrix(
                self._ground_by_family[family],
                _binary_confusion(
                    ground_target[frame_index],
                    ground_predicted[frame_index],
                    ground_valid[frame_index],
                ),
            )

        probabilities = soft_raster.class_probabilities
        if tuple(probabilities.shape) != (
            batch,
            3,
            target_raster_labels.shape[-2],
            target_raster_labels.shape[-1],
        ):
            raise ValueError("soft and target raster shapes differ")
        if target_raster_labels.device != probabilities.device:
            raise ValueError("soft and target rasters must share a device")
        supported = (
            (target_raster_labels == UNKNOWN_CLASS)
            | (target_raster_labels == FREE_CLASS)
            | (target_raster_labels == OCCUPIED_CLASS)
        )
        if not bool(supported.all().item()):
            raise ValueError("target raster contains an unsupported class")
        predicted_class = probabilities.argmax(dim=1)
        for target_class in range(3):
            target_mask = target_raster_labels == target_class
            for predicted_class_index in range(3):
                self._raster_confusion[target_class][predicted_class_index] += int(
                    (target_mask & (predicted_class == predicted_class_index)).sum().item()
                )
        target_probability = probabilities.gather(
            1, target_raster_labels.to(dtype=torch.long)[:, None]
        ).squeeze(1)
        epsilon = torch.finfo(probabilities.dtype).eps
        self._raster_nll_sum += float(
            (-target_probability.clamp_min(epsilon).log()).sum().item()
        )
        self._raster_nll_count += int(target_raster_labels.numel())
        self._frame_count += batch

    def finalize(self) -> dict[str, Any]:
        depth_errors = (
            torch.cat(self._depth_errors)
            if self._depth_errors
            else torch.empty(0, dtype=torch.float64)
        )
        sorted_depth_errors = torch.sort(depth_errors).values.contiguous()

        def quantile_evidence(quantile: float) -> dict[str, Any] | None:
            if not sorted_depth_errors.numel():
                return None
            position = (int(sorted_depth_errors.numel()) - 1) * float(quantile)
            lower_index = int(math.floor(position))
            upper_index = int(math.ceil(position))
            upper_weight = float(position - lower_index)
            return {
                "quantile": float(quantile),
                "lower_index": lower_index,
                "upper_index": upper_index,
                "upper_weight": upper_weight,
                "lower_value_m": float(sorted_depth_errors[lower_index].item()),
                "upper_value_m": float(sorted_depth_errors[upper_index].item()),
            }

        depth_evidence = {
            "dtype": "little_endian_float64",
            "quantile_method": "linear_interpolation_n_minus_1_v1",
            "sorted_values_sha256": hashlib.sha256(
                sorted_depth_errors.numpy().astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest(),
            "median": quantile_evidence(0.5),
            "p95": quantile_evidence(0.95),
        }
        depth_metrics = {
            "count": int(depth_errors.numel()),
            "median_absolute_error_m": (
                None if not depth_errors.numel() else float(torch.quantile(depth_errors, 0.5).item())
            ),
            "p95_absolute_error_m": (
                None if not depth_errors.numel() else float(torch.quantile(depth_errors, 0.95).item())
            ),
            "absolute_error_evidence": depth_evidence,
        }
        raster_recalls: dict[str, float | None] = {}
        present_recalls = []
        for class_index, name in enumerate(RASTER_CLASS_NAMES):
            count = sum(self._raster_confusion[class_index])
            recall = (
                None
                if count == 0
                else self._raster_confusion[class_index][class_index] / count
            )
            raster_recalls[name] = recall
            if recall is not None:
                present_recalls.append(recall)
        return {
            "frame_count": self._frame_count,
            "pixel_hit_no_hit": _binary_metrics(self._pixel_confusion),
            "pixel_hit_depth": depth_metrics,
            "ground_clear": {
                "overall": _binary_metrics(self._ground_confusion),
                "by_distance_m": {
                    name: _binary_metrics(self._ground_by_distance[name])
                    for name in self._distance_names
                },
                "by_family": {
                    family: _binary_metrics(self._ground_by_family[family])
                    for family in sorted(self._ground_by_family)
                },
            },
            "derived_raster": {
                "nll": (
                    None
                    if self._raster_nll_count == 0
                    else self._raster_nll_sum / self._raster_nll_count
                ),
                "nll_sum": self._raster_nll_sum,
                "confusion_target_rows_predicted_columns": self._raster_confusion,
                "class_recalls": raster_recalls,
                "balanced_accuracy": (
                    None
                    if not present_recalls
                    else sum(present_recalls) / len(present_recalls)
                ),
                "count": self._raster_nll_count,
            },
        }


__all__ = [
    "ObservableCameraRayFitV4MetricAccumulator",
    "RASTER_CLASS_NAMES",
]
