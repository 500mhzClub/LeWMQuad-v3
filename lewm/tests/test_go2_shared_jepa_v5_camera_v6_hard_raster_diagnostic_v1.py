from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    OCCUPIED_CLASS,
    OUTPUT_SHAPE,
    PIXEL_RAY_SHAPE,
    SOURCE_SHAPE,
    ObservableCameraRayEvidenceV4,
    calibrated_pixel_ray_directions_body_v4,
    project_canonical_ground_support_v4,
    rasterize_observable_camera_ray_evidence_v4,
)
from lewm.benchmarks import (
    go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1 as diagnostic,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_COUNT,
    ObservableCameraRayEvidenceV4RawOutput,
)


DOWNWARD_BASIS_FRU = np.asarray(
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    dtype=np.float32,
)


def _raw_output(
    *,
    hazard: torch.Tensor,
    offset: torch.Tensor | None = None,
    ground_logits: torch.Tensor | None = None,
    ground_valid: torch.Tensor | None = None,
) -> ObservableCameraRayEvidenceV4RawOutput:
    batch = int(hazard.shape[0])
    offset_value = torch.zeros_like(hazard) if offset is None else offset
    valid = (
        torch.ones((batch, 1, 1, 5), dtype=torch.bool)
        if ground_valid is None
        else ground_valid
    )
    logits = (
        torch.ones(valid.shape, dtype=hazard.dtype)
        if ground_logits is None
        else ground_logits
    )
    return ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=hazard,
        pixel_within_bin_offset_m=offset_value,
        ground_clear_to_target_logits=logits,
        ground_query_in_frustum=valid,
        ground_query_uv_px=torch.zeros((*valid.shape, 2), dtype=hazard.dtype),
        ground_target_distance_m=torch.ones(valid.shape, dtype=hazard.dtype),
    )


def test_hard_decode_uses_inclusive_half_hit_map_bin_and_selected_offset() -> None:
    hazard = torch.full((1, DEPTH_BIN_COUNT, 1, 1), -100.0, dtype=torch.float64)
    hazard[:, 0] = 0.0
    offset = torch.zeros_like(hazard)
    offset[:, 0] = 0.025
    raw = _raw_output(hazard=hazard, offset=offset)

    hit, depth, _clear = diagnostic.decode_hard_evidence_tensors(raw)

    assert hit.item() is True
    assert depth.item() == pytest.approx(0.125, rel=0.0, abs=1e-15)


def test_hard_decode_sets_nonhit_depth_zero_and_masks_ground_by_frustum() -> None:
    hazard = torch.full((1, DEPTH_BIN_COUNT, 1, 1), -100.0)
    valid = torch.tensor([[[[True, True, False, False, True]]]])
    logits = torch.tensor([[[[0.0, -0.1, 10.0, 0.0, 1.0]]]])
    raw = _raw_output(
        hazard=hazard,
        ground_logits=logits,
        ground_valid=valid,
    )

    hit, depth, clear = diagnostic.decode_hard_evidence_tensors(raw)

    assert hit.item() is False
    assert depth.item() == 0.0
    assert clear.tolist() == [[[[True, False, False, False, True]]]]


def test_adapter_reuses_public_closed_boundary_raster_and_occupied_precedence() -> None:
    selected_row, selected_column, selected_bin = 40, 55, 63
    selected_depth = 0.05 + (selected_bin + 0.5) * 0.10
    direction = calibrated_pixel_ray_directions_body_v4(DOWNWARD_BASIS_FRU)[
        selected_row, selected_column
    ]
    boundary_xy = np.asarray((-0.9, -3.1), dtype=np.float64)
    origin = np.asarray(
        (
            boundary_xy[0] - direction[0] * selected_depth,
            boundary_xy[1] - direction[1] * selected_depth,
            10.0,
        ),
        dtype=np.float32,
    )
    projection = project_canonical_ground_support_v4(
        camera_origin_body_m=origin,
        camera_basis_body_fru=DOWNWARD_BASIS_FRU,
        ground_plane_z_body_m=0.0,
    )
    hazard = torch.full(
        (1, DEPTH_BIN_COUNT, *PIXEL_RAY_SHAPE),
        -100.0,
        dtype=torch.float32,
    )
    hazard[0, selected_bin, selected_row, selected_column] = 100.0
    ground_valid = torch.from_numpy(projection.in_frustum[None].copy())
    raw = ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=hazard,
        pixel_within_bin_offset_m=torch.zeros_like(hazard),
        ground_clear_to_target_logits=torch.ones(
            (1, *SOURCE_SHAPE, 5), dtype=torch.float32
        ),
        ground_query_in_frustum=ground_valid,
        ground_query_uv_px=torch.zeros(
            (1, *SOURCE_SHAPE, 5, 2), dtype=torch.float32
        ),
        ground_target_distance_m=torch.ones(
            (1, *SOURCE_SHAPE, 5), dtype=torch.float32
        ),
    )

    observed = diagnostic.hard_raster_labels_from_raw_output(
        raw,
        camera_origin_body_m=torch.from_numpy(origin[None].copy()),
        camera_basis_body_fru=torch.from_numpy(
            DOWNWARD_BASIS_FRU[None].copy()
        ),
        ground_plane_z_body_m=torch.tensor([0.0]),
    )[0]
    hit, depth, clear = diagnostic.decode_hard_evidence_tensors(raw)
    expected_evidence = ObservableCameraRayEvidenceV4(
        camera_origin_body_m=origin,
        camera_basis_body_fru=DOWNWARD_BASIS_FRU,
        ground_plane_z_body_m=0.0,
        ground_support_in_frustum=projection.in_frustum,
        ground_support_clear_to_target=clear[0].numpy(),
        pixel_hit_mask=hit[0].numpy(),
        pixel_first_hit_distance_m=depth[0].numpy(),
    )
    expected = rasterize_observable_camera_ray_evidence_v4(expected_evidence)

    np.testing.assert_array_equal(observed.output_labels, expected.output_labels)
    occupied = np.argwhere(observed.output_occupied_mask)
    assert occupied.shape == (4, 2)
    assert np.all(observed.output_labels[observed.output_occupied_mask] == OCCUPIED_CLASS)
    assert np.any(
        observed.output_free_before_occupied_mask & observed.output_occupied_mask
    )


def test_hard_confusion_publishes_exact_counts_recalls_and_no_nll() -> None:
    accumulator = diagnostic.HardRasterConfusion()
    target = torch.tensor([[[0, 1], [2, 2]]], dtype=torch.uint8)
    prediction = np.asarray([[0, 2], [2, 1]], dtype=np.uint8)

    accumulator.update([prediction], target)
    result = accumulator.finalize()

    assert result["confusion_target_rows_predicted_columns"] == [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
    ]
    assert result["class_recalls"] == {
        "unknown": 1.0,
        "free": 0.0,
        "occupied": 0.5,
    }
    assert result["balanced_accuracy"] == 0.5
    assert result["nll"] is None


def _hard_scope(
    *,
    matched_ba: float,
    wrong_ba: float,
    free_recall: float = 0.99,
    occupied_recall: float = 0.99,
) -> dict[str, object]:
    return {
        "matched": {
            "balanced_accuracy": matched_ba,
            "class_recalls": {
                "unknown": matched_ba,
                "free": free_recall,
                "occupied": occupied_recall,
            },
        },
        "wrong": {
            "balanced_accuracy": wrong_ba,
            "class_recalls": {
                "unknown": wrong_ba,
                "free": wrong_ba,
                "occupied": wrong_ba,
            },
        },
    }


def test_materiality_is_the_fixed_conjunction() -> None:
    scopes = {
        scope: _hard_scope(
            matched_ba=(
                diagnostic.SOFT_RASTER_BALANCED_ACCURACY.get(scope, 0.7) + 0.05
            ),
            wrong_ba=(
                diagnostic.SOFT_RASTER_BALANCED_ACCURACY.get(scope, 0.7) - 0.07
            ),
        )
        for scope in diagnostic.ALL_SCOPES
    }

    passed = diagnostic.evaluate_materiality(scopes)
    assert passed["passed"] is True
    assert passed["scientific_verdict"] == "PASS_MATERIAL_HARD_RASTER_LOCALIZATION"
    assert passed["non_rough_scope_gain_pass_count"] == 8

    failed_scopes = copy.deepcopy(scopes)
    failed_scopes["aggregate"]["matched"]["class_recalls"]["occupied"] = 0.84
    failed = diagnostic.evaluate_materiality(failed_scopes)
    assert failed["passed"] is False
    assert failed["scientific_verdict"] == "FAIL_HYPOTHESIS_REJECTED"
    assert failed["criteria"]["aggregate_occupied_recall_gain_at_least_0_05"] is False


def test_authority_is_forward_only_and_denies_downstream_actions() -> None:
    allowed_true = {
        "one_exact_diagnostic_attempt",
        "rejected_v6_update8000_checkpoint_read",
        "rejected_v6_update8000_checkpoint_deserialization",
        "bound_update8000_sidecar_read",
        "checkpoint_selection_role_diagnostic_read",
        "fixed_calibration_read",
        "forward_only_inference",
        "single_r9700_gpu0",
        "output_root_mutation_only",
    }
    assert {
        name for name, value in diagnostic.EXECUTION_AUTHORITY.items() if value
    } == allowed_true
    for denied in (
        "optimizer_construction",
        "optimizer_step",
        "backward",
        "gradient",
        "train_role_read",
        "probability_calibration_role_read",
        "checkpoint_selection_decision",
        "camera_qualification",
        "checkpoint_promotion",
        "successor_implementation_or_training",
        "g2",
        "navigation",
        "runtime_or_production",
        "heldout",
    ):
        assert diagnostic.EXECUTION_AUTHORITY[denied] is False


def test_canonical_content_hash_rejects_mutation() -> None:
    value = diagnostic.with_content_sha256(
        {"schema": diagnostic.REVIEW_SCHEMA, "status": "PASS"}
    )
    assert (
        diagnostic.validate_content_sha256(
            value, schema=diagnostic.REVIEW_SCHEMA
        )
        == value
    )
    changed = dict(value)
    changed["status"] = "FAIL"
    with pytest.raises(PermissionError, match="content hash"):
        diagnostic.validate_content_sha256(
            changed, schema=diagnostic.REVIEW_SCHEMA
        )
