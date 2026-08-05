from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lewm.models.egomotion_bev_jepa import EgomotionBevJepa
from lewm.planning.egomotion_bev_runtime import EgomotionBevJepaRuntime
from lewm.planning.geometry_contract import load_geometry_contract
from lewm.planning.online_belief_map import (
    BeliefMapConfig,
    CellState,
    OnlineBeliefMap,
    PoseBelief,
)
from lewm.planning.perception_map_adapter import CameraGeometry


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = REPOSITORY_ROOT / "config/go2_generalization_geometry_v1.json"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_grid() -> dict[str, Any]:
    return {
        "shape": [3, 3],
        "cell_size_m": 0.1,
        "forward_edge_range_m": [0.0, 0.3],
        "left_edge_range_m": [0.0, 0.3],
        "forward_center_range_m": [0.05, 0.25],
        "left_center_range_m": [0.05, 0.25],
        "array_axes": {
            "row": "base_forward_increasing",
            "column": "base_left_increasing",
        },
        "base_frame_axes": {
            "forward": "+x_base_link",
            "left": "+y_base_link",
        },
        "bounds_are": "cell_edges",
    }


def _camera() -> CameraGeometry:
    geometry = load_geometry_contract(
        GEOMETRY_PATH,
        repository_root=REPOSITORY_ROOT,
        verify_sources=True,
    )
    return CameraGeometry(
        calibration_id="synthetic-go2-front-v1",
        image_height_px=12,
        image_width_px=20,
        horizontal_fov_deg=geometry.camera.horizontal_fov_deg,
        vertical_fov_deg=58.0,
        mount_xyz_m=geometry.camera.nominal_xyz_body_m,
        mount_rpy_rad=geometry.camera.nominal_rpy_body_rad,
    )


def _belief_map() -> OnlineBeliefMap:
    geometry = load_geometry_contract(
        GEOMETRY_PATH,
        repository_root=REPOSITORY_ROOT,
        verify_sources=True,
    )
    return OnlineBeliefMap(
        BeliefMapConfig(
            cell_size_m=geometry.configuration_space.online_cell_size_m,
            planning_connectivity=geometry.configuration_space.connectivity,
            allow_diagonal_corner_cutting=(
                geometry.configuration_space.allow_diagonal_corner_cutting
            ),
        )
    )


def _pose(tick: int) -> PoseBelief:
    return PoseBelief(
        mean=(0.0, 0.0, 0.0),
        covariance=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.01)),
        tick=tick,
    )


def _write_checkpoint(
    tmp_path: Path,
    *,
    geometry_sha256: str | None = None,
) -> tuple[Path, str]:
    geometry = load_geometry_contract(
        GEOMETRY_PATH,
        repository_root=REPOSITORY_ROOT,
        verify_sources=True,
    )
    grid = _local_grid()
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = {
        "schema": "lewm_go2_paired_navigation_dataset_v2",
        "local_grid": grid,
        "geometry_contract": {
            "path": str(GEOMETRY_PATH),
            "sha256": geometry.sha256,
            "file_sha256": _sha256_file(GEOMETRY_PATH),
        },
        "sources": [
            {"scene_id": "synthetic-train-scene", "dataset_split": "train"},
            {"scene_id": "synthetic-validation-scene", "dataset_split": "validation"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    model_config = {
        "image_size": 16,
        "patch_size": 8,
        "encoder_dim": 16,
        "encoder_depth": 1,
        "encoder_heads": 2,
        "bev_dim": 8,
        "bev_size": (3, 3),
        "forward_range_m": (0.05, 0.25),
        "left_range_m": (0.05, 0.25),
        "action_dim": 2,
        "predictor_hidden_dim": 8,
        "target_ema_momentum": 0.996,
        "jepa_weight": 1.0,
        "occupancy_weight": 2.0,
        "equivariance_weight": 0.25,
        "action_contrast_weight": 1.0,
        "action_margin_fraction": 0.1,
        "variance_weight": 0.1,
        "variance_target_std": 0.5,
    }
    model = EgomotionBevJepa(**model_config)
    with torch.no_grad():
        model.occupancy_head.weight.zero_()
        # High UNKNOWN, low occupied: this must not be admitted as free.
        model.occupancy_head.bias.copy_(torch.tensor([4.0, 0.0, -4.0]))
    calibration_without_id = {
        "method": "positive_diagonal_vector_scaling_with_centered_bias",
        "log_scales": [0.0, 0.0, 0.0],
        "biases": [0.0, 0.0, 0.0],
        "sample_count": 30,
        "nll_before": 0.8,
        "nll_after": 0.7,
    }
    calibration = dict(calibration_without_id)
    calibration["id"] = (
        "go2-vector-scale-" + _canonical_sha256(calibration_without_id)[:16]
    )
    thresholds = {
        "free_probability_min": 0.8,
        "occupied_probability_max": 0.1,
        "unknown_probability_max": 0.1,
        "occupied_detection_min": 0.8,
    }
    primitive_to_index = {"forward": 0, "turn_left": 1}
    nominal_delta = [[0.1, 0.0, 0.0], [0.0, 0.0, 0.25]]
    nominal_delta_id = "go2-train-median-delta-" + _canonical_sha256(
        {
            "primitive_to_index": primitive_to_index,
            "values": np.asarray(nominal_delta, dtype=np.float32).tolist(),
        }
    )[:16]
    checkpoint = {
        "schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v2",
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "primitive_to_index": primitive_to_index,
        "nominal_primitive_delta_current": nominal_delta,
        "nominal_primitive_delta_id": nominal_delta_id,
        "nominal_primitive_delta_source": (
            "coordinatewise_train_median_with_circular_yaw"
        ),
        "nominal_primitive_delta_statistics": {},
        "probability_calibration": calibration,
        "probability_calibration_id": calibration["id"],
        "image_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "occupancy_output_contract": {
            "class_order": ["unknown", "free", "occupied"],
            "raw_output": "three_class_logits",
            "runtime_transform": "apply_probability_calibration_then_softmax",
            "local_grid": grid,
        },
        "traversability_thresholds": thresholds,
        "best_epoch": 1,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "training_scene_ids": ["synthetic-train-scene"],
        "geometry_contract_sha256": geometry_sha256 or geometry.sha256,
        "scene_roles_sha256": "a" * 64,
        "selection_metrics": {},
        "calibration_metrics": {
            "thresholds": thresholds,
            "calibration": {"applied": True, "id": calibration["id"]},
        },
        "g2_evaluation": {"g2": {"passes": True}},
        "g2_passes": True,
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, _sha256_file(checkpoint_path)


def _load_runtime(
    tmp_path: Path,
) -> tuple[EgomotionBevJepaRuntime, OnlineBeliefMap]:
    checkpoint_path, checkpoint_sha = _write_checkpoint(tmp_path)
    belief_map = _belief_map()
    runtime = EgomotionBevJepaRuntime.load(
        checkpoint_path,
        GEOMETRY_PATH,
        camera=_camera(),
        belief_map=belief_map,
        device="cpu",
        expected_checkpoint_sha256=checkpoint_sha,
        repository_root=REPOSITORY_ROOT,
    )
    return runtime, belief_map


def test_high_unknown_low_occupied_probability_is_never_fused_free(
    tmp_path: Path,
) -> None:
    runtime, belief_map = _load_runtime(tmp_path)
    rgb = np.zeros((12, 20, 3), dtype=np.uint8)

    result = runtime.observe_and_fuse(
        rgb,
        pose=_pose(1),
        observation_id="unknown-frame",
    )

    assert np.all(result.occupancy.probabilities[0] > 0.95)
    assert np.all(result.occupancy.probabilities[2] < 0.01)
    assert not result.occupancy.admitted_free_mask.any()
    assert result.fusion_record.free_source_cells == 0
    assert result.fusion_record.occupied_source_cells == 0
    assert belief_map.known_cells == frozenset()
    assert runtime.thresholds.free_probability_min == 0.8
    assert runtime.thresholds.unknown_probability_max == 0.1
    json.dumps(runtime.provenance)
    fusion_provenance = runtime.fusion_provenance_state_dict()
    assert fusion_provenance["records"][0]["observation_id"] == "unknown-frame"
    assert fusion_provenance["runtime"]["checkpoint"]["sha256"] == (
        runtime.checkpoint_sha256
    )
    json.dumps(fusion_provenance)


def test_checkpoint_geometry_mismatch_fails_before_model_exposure(tmp_path: Path) -> None:
    checkpoint_path, checkpoint_sha = _write_checkpoint(
        tmp_path,
        geometry_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="checkpoint geometry contract"):
        EgomotionBevJepaRuntime.load(
            checkpoint_path,
            GEOMETRY_PATH,
            camera=_camera(),
            belief_map=_belief_map(),
            expected_checkpoint_sha256=checkpoint_sha,
            repository_root=REPOSITORY_ROOT,
        )


def test_runtime_rejects_physical_checkpoint_v4_until_g3_integration(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "physical-v4.pt"
    torch.save(
        {
            "schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v4",
            "head_g2_passes": True,
            "runtime_ready": False,
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        EgomotionBevJepaRuntime.load(
            checkpoint_path,
            GEOMETRY_PATH,
            camera=_camera(),
            belief_map=_belief_map(),
            expected_checkpoint_sha256=_sha256_file(checkpoint_path),
            repository_root=REPOSITORY_ROOT,
        )


def test_only_connected_confirmed_free_cells_can_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, belief_map = _load_runtime(tmp_path)
    rgb = np.zeros((12, 20, 3), dtype=np.uint8)

    def free_unknown_free(image: torch.Tensor) -> torch.Tensor:
        logits = torch.full((image.shape[0], 3, 3, 3), -8.0, device=image.device)
        logits[:, 1, 0, :] = 8.0
        logits[:, 0, 1, :] = 8.0
        logits[:, 1, 2, :] = 8.0
        return logits

    monkeypatch.setattr(runtime._model, "occupancy_logits", free_unknown_free)
    runtime.observe_and_fuse(rgb, pose=_pose(1), observation_id="split-free")

    assert belief_map.cell_state((0, 0)) is CellState.CONFIRMED_FREE
    assert belief_map.cell_state((1, 0)) is CellState.UNKNOWN
    assert belief_map.cell_state((2, 0)) is CellState.CONFIRMED_FREE
    assert belief_map.shortest_path((0, 0), (2, 0)) is None
    assert belief_map.connected_confirmed_free((0, 0)) == frozenset(
        {(0, 0), (0, 1), (0, 2)}
    )

    def all_free(image: torch.Tensor) -> torch.Tensor:
        logits = torch.full((image.shape[0], 3, 3, 3), -8.0, device=image.device)
        logits[:, 1] = 8.0
        return logits

    monkeypatch.setattr(runtime._model, "occupancy_logits", all_free)
    runtime.observe_and_fuse(rgb, pose=_pose(2), observation_id="connected-free")
    path = belief_map.shortest_path((0, 0), (2, 0))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 0)
    assert all(belief_map.is_confirmed_free(cell) for cell in path)


def test_occupancy_api_has_no_future_motion_input_and_diagnostic_uses_nominal_delta(
    tmp_path: Path,
) -> None:
    runtime, _belief_map_value = _load_runtime(tmp_path)
    rgb = np.zeros((12, 20, 3), dtype=np.uint8)

    assert tuple(inspect.signature(runtime.infer_occupancy).parameters) == ("rgb",)
    with pytest.raises(TypeError):
        runtime.infer_occupancy(rgb, realized_future_odometry=(1.0, 2.0, 3.0))

    diagnostic = runtime.command_prediction_diagnostic(
        rgb,
        primitive_name="turn_left",
    )
    assert diagnostic.nominal_delta_pose_current == pytest.approx((0.0, 0.0, 0.25))
    assert diagnostic.predicted_next_bev.shape == (8, 3, 3)
    assert diagnostic.warped_current_bev.shape == (8, 3, 3)
    assert diagnostic.overlap_mask.shape == (3, 3)


def test_runtime_rejects_stale_checkpoint_hash_and_ambiguous_float_rgb(
    tmp_path: Path,
) -> None:
    checkpoint_path, _checkpoint_sha = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        EgomotionBevJepaRuntime.load(
            checkpoint_path,
            GEOMETRY_PATH,
            camera=_camera(),
            belief_map=_belief_map(),
            expected_checkpoint_sha256="0" * 64,
            repository_root=REPOSITORY_ROOT,
        )

    runtime = EgomotionBevJepaRuntime.load(
        checkpoint_path,
        GEOMETRY_PATH,
        camera=_camera(),
        belief_map=_belief_map(),
        repository_root=REPOSITORY_ROOT,
    )
    with pytest.raises(ValueError, match="uint8"):
        runtime.infer_occupancy(np.zeros((12, 20, 3), dtype=np.float32))
