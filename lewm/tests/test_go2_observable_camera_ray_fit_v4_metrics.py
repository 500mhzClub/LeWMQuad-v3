from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import pytest
import torch

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    FREE_CLASS,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
)
from lewm.benchmarks.go2_observable_camera_ray_fit_v4_metrics import (
    ObservableCameraRayFitV4MetricAccumulator,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_SIZE_M,
    ObservableCameraRayEvidenceV4RawOutput,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    ObservableCameraRayEvidenceV4Targets,
    SoftObservableCameraRayRasterV4,
)


def _synthetic_batch() -> tuple[
    ObservableCameraRayEvidenceV4RawOutput,
    ObservableCameraRayEvidenceV4Targets,
    SoftObservableCameraRayRasterV4,
    torch.Tensor,
]:
    # Four target hits and two target no-hits. Exactly half of each class is
    # classified correctly. Hit-bin errors are 0, 1, 2, and 3 bins.
    hazards = torch.full((2, 4, 1, 3), -20.0)
    hazards[0, 0, 0, 0] = 20.0
    hazards[0, 1, 0, 1] = 20.0
    hazards[1, 0, 0, 2] = 20.0
    pixel_hit = torch.tensor(
        [[[True, True, False]], [[True, True, False]]],
        dtype=torch.bool,
    )
    pixel_bins = torch.tensor(
        [[[0, 0, 0]], [[2, 3, 0]]],
        dtype=torch.long,
    )

    ground_valid = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, False]],
        dtype=torch.bool,
    )[:, None, None, :]
    ground_target = torch.tensor(
        [[False, True, False, True, False], [False, True, False, True, False]],
        dtype=torch.bool,
    )[:, None, None, :]
    ground_predicted = torch.tensor(
        [[False, True, True, True, False], [True, False, False, True, False]],
        dtype=torch.bool,
    )[:, None, None, :]
    ground_distance = torch.tensor(
        [[0.5, 0.5, 1.5, 1.5, 9.0], [0.5, 0.5, 2.5, 2.5, 9.0]],
        dtype=torch.float32,
    )[:, None, None, :]

    raw = ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=hazards,
        pixel_within_bin_offset_m=torch.zeros_like(hazards),
        ground_clear_to_target_logits=torch.where(
            ground_predicted,
            torch.tensor(20.0),
            torch.tensor(-20.0),
        ),
        ground_query_in_frustum=ground_valid,
        ground_query_uv_px=torch.zeros((2, 1, 1, 5, 2)),
        ground_target_distance_m=ground_distance,
    )
    targets = ObservableCameraRayEvidenceV4Targets(
        pixel_in_range_hit_mask=pixel_hit,
        pixel_no_hit_mask=~pixel_hit,
        pixel_hit_bin_index=pixel_bins,
        pixel_within_bin_offset_m=torch.zeros_like(pixel_hit, dtype=torch.float32),
        ground_in_frustum=ground_valid,
        ground_clear_to_target=ground_target,
    )

    # Frame zero is correct; frame one is cyclically wrong. Each class has
    # target probability 0.8 once and 0.2 once.
    raster_probabilities = torch.tensor(
        [
            [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]],
            [[[0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.7, 0.1, 0.2]]],
        ],
        dtype=torch.float32,
    ).permute(0, 3, 1, 2)
    soft_raster = SoftObservableCameraRayRasterV4(
        source_free_probability=torch.zeros((2, 1, 1)),
        free_given_not_occupied_probability=torch.zeros((2, 1, 3)),
        occupied_probability=torch.zeros((2, 1, 3)),
        class_probabilities=raster_probabilities,
    )
    target_raster = torch.tensor(
        [
            [[UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS]],
            [[UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS]],
        ],
        dtype=torch.long,
    )
    return raw, targets, soft_raster, target_raster


def test_accumulator_reports_registered_pixel_depth_ground_and_raster_metrics() -> None:
    raw, targets, soft_raster, target_raster = _synthetic_batch()
    accumulator = ObservableCameraRayFitV4MetricAccumulator()
    accumulator.update(
        raw_output=raw,
        targets=targets,
        soft_raster=soft_raster,
        target_raster_labels=target_raster,
        families=("alpha", "beta"),
    )

    result = accumulator.finalize()

    assert result["frame_count"] == 2
    pixel = result["pixel_hit_no_hit"]
    assert pixel["confusion_target_rows_predicted_columns"] == [[1, 1], [2, 2]]
    assert pixel["negative_recall"] == pytest.approx(0.5)
    assert pixel["positive_recall"] == pytest.approx(0.5)
    assert pixel["balanced_accuracy"] == pytest.approx(0.5)

    depth = result["pixel_hit_depth"]
    assert depth["count"] == 4
    assert depth["median_absolute_error_m"] == pytest.approx(
        1.5 * DEPTH_BIN_SIZE_M,
        abs=1.0e-6,
    )
    assert depth["p95_absolute_error_m"] == pytest.approx(
        2.85 * DEPTH_BIN_SIZE_M,
        abs=1.0e-6,
    )

    ground = result["ground_clear"]
    assert ground["overall"]["confusion_target_rows_predicted_columns"] == [
        [2, 2],
        [1, 3],
    ]
    assert ground["overall"]["balanced_accuracy"] == pytest.approx(0.625)
    assert ground["by_distance_m"]["0.0_to_1.0"]["balanced_accuracy"] == pytest.approx(
        0.5
    )
    assert ground["by_distance_m"]["1.0_to_2.0"]["balanced_accuracy"] == pytest.approx(
        0.5
    )
    assert ground["by_distance_m"]["2.0_to_3.0"]["balanced_accuracy"] == pytest.approx(
        1.0
    )
    assert ground["by_distance_m"]["3.0_to_4.0"]["count"] == 0
    assert ground["by_distance_m"]["3.0_to_4.0"]["balanced_accuracy"] is None
    assert list(ground["by_family"]) == ["alpha", "beta"]
    assert ground["by_family"]["alpha"]["balanced_accuracy"] == pytest.approx(0.75)
    assert ground["by_family"]["beta"]["balanced_accuracy"] == pytest.approx(0.5)

    raster = result["derived_raster"]
    assert raster["confusion_target_rows_predicted_columns"] == [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]
    assert raster["class_recalls"] == {
        "unknown": pytest.approx(0.5),
        "free": pytest.approx(0.5),
        "occupied": pytest.approx(0.5),
    }
    assert raster["balanced_accuracy"] == pytest.approx(0.5)
    assert raster["nll"] == pytest.approx(-0.5 * (math.log(0.8) + math.log(0.2)))
    assert raster["count"] == 6


def test_accumulator_adds_confusions_and_depth_errors_across_updates() -> None:
    raw, targets, soft_raster, target_raster = _synthetic_batch()
    accumulator = ObservableCameraRayFitV4MetricAccumulator()
    for _ in range(2):
        accumulator.update(
            raw_output=raw,
            targets=targets,
            soft_raster=soft_raster,
            target_raster_labels=target_raster,
            families=("alpha", "beta"),
        )

    result = accumulator.finalize()
    assert result["frame_count"] == 4
    assert result["pixel_hit_no_hit"]["confusion_target_rows_predicted_columns"] == [
        [2, 2],
        [4, 4],
    ]
    assert result["pixel_hit_depth"]["count"] == 8
    assert result["ground_clear"]["by_family"]["alpha"]["count"] == 8
    assert result["derived_raster"]["count"] == 12


def test_empty_accumulator_has_explicit_absent_metrics() -> None:
    result = ObservableCameraRayFitV4MetricAccumulator().finalize()

    assert result["frame_count"] == 0
    assert result["pixel_hit_no_hit"]["balanced_accuracy"] is None
    assert result["pixel_hit_depth"] == {
        "count": 0,
        "median_absolute_error_m": None,
        "p95_absolute_error_m": None,
        "absolute_error_evidence": {
            "dtype": "little_endian_float64",
            "quantile_method": "linear_interpolation_n_minus_1_v1",
            "sorted_values_sha256": hashlib.sha256(b"").hexdigest(),
            "median": None,
            "p95": None,
        },
    }
    assert result["ground_clear"]["overall"]["balanced_accuracy"] is None
    assert result["derived_raster"]["nll"] is None
    assert result["derived_raster"]["balanced_accuracy"] is None


@pytest.mark.parametrize(
    "edges",
    [(), (0.0,), (-1.0, 1.0), (0.0, 1.0, 1.0), (0.0, 2.0, 1.0)],
)
def test_accumulator_rejects_malformed_ground_distance_edges(
    edges: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        ObservableCameraRayFitV4MetricAccumulator(
            ground_distance_bin_edges_m=edges,
        )


def test_accumulator_rejects_missing_family() -> None:
    raw, targets, soft_raster, target_raster = _synthetic_batch()
    with pytest.raises(ValueError, match="family"):
        ObservableCameraRayFitV4MetricAccumulator().update(
            raw_output=raw,
            targets=targets,
            soft_raster=soft_raster,
            target_raster_labels=target_raster,
            families=("alpha", ""),
        )


def test_accumulator_rejects_target_calibration_mismatch() -> None:
    raw, targets, soft_raster, target_raster = _synthetic_batch()
    mismatched = replace(raw, ground_query_in_frustum=~raw.ground_query_in_frustum)
    with pytest.raises(ValueError, match="calibration"):
        ObservableCameraRayFitV4MetricAccumulator().update(
            raw_output=mismatched,
            targets=targets,
            soft_raster=soft_raster,
            target_raster_labels=target_raster,
            families=("alpha", "beta"),
        )


def test_accumulator_rejects_soft_raster_shape_mismatch() -> None:
    raw, targets, _, target_raster = _synthetic_batch()
    mismatched = SoftObservableCameraRayRasterV4(
        source_free_probability=torch.zeros((2, 1, 1)),
        free_given_not_occupied_probability=torch.zeros((2, 1, 2)),
        occupied_probability=torch.zeros((2, 1, 2)),
        class_probabilities=torch.full((2, 3, 1, 2), 1.0 / 3.0),
    )
    with pytest.raises(ValueError, match="raster shapes"):
        ObservableCameraRayFitV4MetricAccumulator().update(
            raw_output=raw,
            targets=targets,
            soft_raster=mismatched,
            target_raster_labels=target_raster,
            families=("alpha", "beta"),
        )


def test_accumulator_rejects_unsupported_target_raster_class() -> None:
    raw, targets, soft_raster, target_raster = _synthetic_batch()
    target_raster[0, 0, 0] = 9
    with pytest.raises(ValueError, match="unsupported class"):
        ObservableCameraRayFitV4MetricAccumulator().update(
            raw_output=raw,
            targets=targets,
            soft_raster=soft_raster,
            target_raster_labels=target_raster,
            families=("alpha", "beta"),
        )
