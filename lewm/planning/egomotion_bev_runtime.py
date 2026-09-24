"""Fail-closed RGB occupancy runtime for the egomotion BEV JEPA.

Current occupancy inference has exactly one learned input: the current RGB
frame.  Pose is consumed only after inference to register the categorical
probabilities in an :class:`OnlineBeliefMap`.  The predictive JEPA branch is
available through a separately named diagnostic method and uses only the
commanded primitive plus its frozen train-set nominal delta.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from lewm.benchmarks.traversability_metrics import TraversabilityThresholds
from lewm.models.egomotion_bev_jepa import EgomotionBevJepa
from lewm.planning.geometry_contract import GeometryContract, load_geometry_contract
from lewm.planning.online_belief_map import OnlineBeliefMap, PoseBelief
from lewm.planning.perception_map_adapter import (
    CameraGeometry,
    EgocentricOccupancyGeometry,
    FusionRecord,
    GridReferenceFrame,
    ModelArtifact,
    ObservationProvenance,
    OccupancyObservation,
    OccupancyValueKind,
    PerceptionMapAdapterConfig,
    PerceptionMapContract,
    PerceptionToBeliefMapAdapter,
)

__all__ = [
    "CalibratedOccupancy",
    "CommandPredictionDiagnostic",
    "EgomotionBevJepaRuntime",
    "FusedOccupancy",
]


_CHECKPOINT_SCHEMA = "lewm_go2_egomotion_bev_jepa_checkpoint_v2"
_DATASET_SCHEMA = "lewm_go2_paired_navigation_dataset_v2"
_CALIBRATION_METHOD = "positive_diagonal_vector_scaling_with_centered_bias"
_CLASS_ORDER = ("unknown", "free", "occupied")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_CONFIG_KEYS = {
    "image_size",
    "patch_size",
    "encoder_dim",
    "encoder_depth",
    "encoder_heads",
    "bev_dim",
    "bev_size",
    "forward_range_m",
    "left_range_m",
    "action_dim",
    "predictor_hidden_dim",
    "target_ema_momentum",
    "jepa_weight",
    "occupancy_weight",
    "equivariance_weight",
    "action_contrast_weight",
    "action_margin_fraction",
    "variance_weight",
    "variance_target_std",
}


@dataclass(frozen=True)
class CalibratedOccupancy:
    """Calibrated current-frame occupancy and exact planner decisions."""

    probabilities: np.ndarray
    admitted_free_mask: np.ndarray
    detected_occupied_mask: np.ndarray
    checkpoint_sha256: str
    probability_calibration_id: str


@dataclass(frozen=True)
class FusedOccupancy:
    """One RGB inference together with its immutable map-fusion record."""

    occupancy: CalibratedOccupancy
    fusion_record: FusionRecord


@dataclass(frozen=True)
class CommandPredictionDiagnostic:
    """Non-planning diagnostic from a command and its frozen nominal delta."""

    primitive_name: str
    nominal_delta_pose_current: tuple[float, float, float]
    predicted_next_bev: np.ndarray
    warped_current_bev: np.ndarray
    overlap_mask: np.ndarray


@dataclass(frozen=True)
class _ValidatedCheckpoint:
    payload: Mapping[str, Any]
    model_config: Mapping[str, Any]
    calibration_log_scales: tuple[float, float, float]
    calibration_biases: tuple[float, float, float]
    calibration_id: str
    thresholds: TraversabilityThresholds
    primitive_to_index: Mapping[str, int]
    nominal_delta_table: np.ndarray
    local_grid: Mapping[str, Any]
    dataset_manifest_path: Path
    dataset_manifest_sha256: str


class EgomotionBevJepaRuntime:
    """Deployment runtime for a promoted, calibrated v2 checkpoint.

    Use :meth:`load` so every artifact is checked before the model is exposed.
    There is deliberately no occupancy method accepting an action, odometry
    delta, next frame, scene manifest, or geometry raster.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        validated: _ValidatedCheckpoint,
        geometry_contract: GeometryContract,
        camera: CameraGeometry,
        belief_map: OnlineBeliefMap,
        model: EgomotionBevJepa,
        device: torch.device,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._checkpoint_sha256 = checkpoint_sha256
        self._geometry_contract = geometry_contract
        self._camera = camera
        self._belief_map = belief_map
        self._model = model
        self._device = device
        self._thresholds = validated.thresholds
        self._calibration_id = validated.calibration_id
        self._calibration_log_scales = torch.tensor(
            validated.calibration_log_scales,
            dtype=torch.float32,
            device=device,
        )
        self._calibration_biases = torch.tensor(
            validated.calibration_biases,
            dtype=torch.float32,
            device=device,
        )
        self._primitive_to_index = dict(validated.primitive_to_index)
        self._nominal_delta_table = torch.from_numpy(
            validated.nominal_delta_table.copy()
        ).to(device=device, dtype=torch.float32)
        normalization = _mapping(
            validated.payload["image_normalization"], "image_normalization"
        )
        self._image_mean = torch.tensor(
            normalization["mean"], dtype=torch.float32, device=device
        )[:, None, None]
        self._image_std = torch.tensor(
            normalization["std"], dtype=torch.float32, device=device
        )[:, None, None]

        local_grid = validated.local_grid
        occupancy_geometry = EgocentricOccupancyGeometry(
            geometry_id=(
                f"go2-egomotion-bev-{geometry_contract.sha256[:12]}-"
                f"{checkpoint_sha256[:12]}"
            ),
            height=int(local_grid["shape"][0]),
            width=int(local_grid["shape"][1]),
            cell_size_m=float(local_grid["cell_size_m"]),
            forward_min_m=float(local_grid["forward_edge_range_m"][0]),
            left_min_m=float(local_grid["left_edge_range_m"][0]),
            reference_frame=GridReferenceFrame.BODY,
            body_inflation_radius_m=(
                geometry_contract.configuration_space.body_inflation_radius_m
            ),
        )
        checkpoint_artifact = ModelArtifact(
            artifact_id=f"egomotion-bev-jepa-encoder:{_CHECKPOINT_SCHEMA}",
            checkpoint_sha256=checkpoint_sha256,
        )
        occupancy_artifact = ModelArtifact(
            artifact_id=f"egomotion-bev-occupancy:{validated.calibration_id}",
            checkpoint_sha256=checkpoint_sha256,
        )
        self._perception_contract = PerceptionMapContract(
            backbone=checkpoint_artifact,
            occupancy_head=occupancy_artifact,
            probability_calibration_id=validated.calibration_id,
            camera=camera,
            occupancy_geometry=occupancy_geometry,
            map_cell_size_m=(
                geometry_contract.configuration_space.online_cell_size_m
            ),
            map_frame="odometry",
            observation_source="onboard_rgb",
        )
        adapter_config = PerceptionMapAdapterConfig(
            planner_free_probability_min=validated.thresholds.free_probability_min,
            planner_occupied_probability_max=(
                validated.thresholds.occupied_probability_max
            ),
            planner_unknown_probability_max=(
                validated.thresholds.unknown_probability_max
            ),
            occupied_class_probability_min=(
                validated.thresholds.occupied_detection_min
            ),
            minimum_body_inflation_radius_m=(
                geometry_contract.configuration_space.body_inflation_radius_m
            ),
        )
        self._adapter = PerceptionToBeliefMapAdapter(
            belief_map,
            self._perception_contract,
            adapter_config,
        )
        self._dataset_manifest_path = validated.dataset_manifest_path
        self._dataset_manifest_sha256 = validated.dataset_manifest_sha256

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        geometry_contract_path: Path,
        *,
        camera: CameraGeometry,
        belief_map: OnlineBeliefMap,
        device: str | torch.device = "cpu",
        expected_checkpoint_sha256: str | None = None,
        dataset_manifest_path: Path | None = None,
        repository_root: Path | None = None,
        verify_geometry_sources: bool = True,
    ) -> "EgomotionBevJepaRuntime":
        """Load and verify the checkpoint, calibration, geometry, and provenance."""

        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        if expected_checkpoint_sha256 is not None:
            _require_sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
            if checkpoint_sha256 != expected_checkpoint_sha256:
                raise ValueError(
                    "checkpoint SHA-256 mismatch: expected "
                    f"{expected_checkpoint_sha256}, got {checkpoint_sha256}"
                )
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "this runtime requires torch.load(..., weights_only=True) support"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint root must be a mapping")

        geometry = load_geometry_contract(
            Path(geometry_contract_path),
            repository_root=repository_root,
            verify_sources=verify_geometry_sources,
        )
        validated = _validate_checkpoint(
            payload,
            checkpoint_path=checkpoint_path,
            geometry=geometry,
            dataset_manifest_override=dataset_manifest_path,
        )
        _validate_camera(camera, geometry)
        _validate_belief_map(belief_map, geometry)
        resolved_device = torch.device(device)
        model = EgomotionBevJepa(**dict(validated.model_config))
        state = validated.payload["model_state_dict"]
        if not isinstance(state, Mapping) or not state:
            raise ValueError("model_state_dict must be a non-empty mapping")
        try:
            model.load_state_dict(state, strict=True)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint state does not match model_config") from exc
        model.to(resolved_device)
        model.eval()
        return cls(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            validated=validated,
            geometry_contract=geometry,
            camera=camera,
            belief_map=belief_map,
            model=model,
            device=resolved_device,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def geometry_contract(self) -> GeometryContract:
        return self._geometry_contract

    @property
    def perception_contract(self) -> PerceptionMapContract:
        return self._perception_contract

    @property
    def thresholds(self) -> TraversabilityThresholds:
        return self._thresholds

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "schema": "lewm_go2_egomotion_bev_runtime_provenance_v1",
            "checkpoint": {
                "path": str(self._checkpoint_path),
                "sha256": self._checkpoint_sha256,
                "schema": _CHECKPOINT_SCHEMA,
            },
            "dataset_manifest": {
                "path": str(self._dataset_manifest_path),
                "sha256": self._dataset_manifest_sha256,
            },
            "geometry_contract": {
                "path": str(self._geometry_contract.source_path),
                "sha256": self._geometry_contract.sha256,
            },
            "probability_calibration_id": self._calibration_id,
            "traversability_thresholds": {
                "free_probability_min": self._thresholds.free_probability_min,
                "occupied_probability_max": self._thresholds.occupied_probability_max,
                "unknown_probability_max": self._thresholds.unknown_probability_max,
                "occupied_detection_min": self._thresholds.occupied_detection_min,
            },
            "perception_map_contract_sha256": self._perception_contract.fingerprint,
        }

    def fusion_provenance_state_dict(self) -> dict[str, Any]:
        """Return the adapter's JSON-safe per-observation evidence ledger."""

        state = self._adapter.provenance_state_dict()
        state["runtime"] = self.provenance
        return state

    def preprocess_rgb(self, rgb: Image.Image | np.ndarray) -> torch.Tensor:
        """Apply the checkpoint's exact RGB resize and normalization contract."""

        if isinstance(rgb, Image.Image):
            image = rgb.convert("RGB")
        elif isinstance(rgb, np.ndarray):
            array = np.asarray(rgb)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("RGB arrays must have uint8 shape (height, width, 3)")
            image = Image.fromarray(array)
        else:
            raise TypeError("rgb must be a PIL image or uint8 RGB numpy array")
        image = image.resize(
            (self._model.image_size, self._model.image_size),
            Image.Resampling.BILINEAR,
        )
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(self._device)
        return ((tensor - self._image_mean) / self._image_std).unsqueeze(0)

    def infer_occupancy(self, rgb: Image.Image | np.ndarray) -> CalibratedOccupancy:
        """Infer calibrated current occupancy from current RGB only."""

        image = self.preprocess_rgb(rgb)
        with torch.inference_mode():
            logits = self._model.occupancy_logits(image)
            expected = (1, 3, *self._model.bev_size)
            if tuple(logits.shape) != expected:
                raise RuntimeError(
                    f"occupancy head returned shape {tuple(logits.shape)}, expected {expected}"
                )
            calibrated = _apply_vector_calibration(
                logits,
                self._calibration_log_scales,
                self._calibration_biases,
            )
            probabilities = torch.softmax(calibrated, dim=1)[0]
        result = probabilities.detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.isfinite(result).all() or not np.allclose(
            result.sum(axis=0), 1.0, atol=1e-5
        ):
            raise RuntimeError("calibrated occupancy is not a finite probability simplex")
        admitted = (
            (result[1] >= self._thresholds.free_probability_min)
            & (result[2] <= self._thresholds.occupied_probability_max)
            & (result[0] <= self._thresholds.unknown_probability_max)
        )
        occupied = result[2] >= self._thresholds.occupied_detection_min
        result.setflags(write=False)
        admitted.setflags(write=False)
        occupied.setflags(write=False)
        return CalibratedOccupancy(
            probabilities=result,
            admitted_free_mask=admitted,
            detected_occupied_mask=occupied,
            checkpoint_sha256=self._checkpoint_sha256,
            probability_calibration_id=self._calibration_id,
        )

    def observe_and_fuse(
        self,
        rgb: Image.Image | np.ndarray,
        *,
        pose: PoseBelief,
        observation_id: str,
        observation_confidence: float = 1.0,
    ) -> FusedOccupancy:
        """Infer from RGB, then register probabilities using the current pose."""

        occupancy = self.infer_occupancy(rgb)
        observation = OccupancyObservation(
            values=occupancy.probabilities,
            value_kind=OccupancyValueKind.CATEGORICAL_PROBABILITY,
            pose=pose,
            camera=self._camera,
            geometry=self._perception_contract.occupancy_geometry,
            provenance=ObservationProvenance.create(
                observation_id,
                contract=self._perception_contract,
            ),
            observation_confidence=observation_confidence,
        )
        return FusedOccupancy(
            occupancy=occupancy,
            fusion_record=self._adapter.fuse(observation),
        )

    def command_prediction_diagnostic(
        self,
        rgb: Image.Image | np.ndarray,
        *,
        primitive_name: str,
    ) -> CommandPredictionDiagnostic:
        """Run the optional JEPA command diagnostic with no realized odometry."""

        try:
            action_index = self._primitive_to_index[primitive_name]
        except KeyError as exc:
            raise KeyError(f"unknown checkpoint primitive {primitive_name!r}") from exc
        image = self.preprocess_rgb(rgb)
        with torch.inference_mode():
            current_bev = self._model._encode_online(image)
            action = torch.zeros(
                (1, len(self._primitive_to_index)),
                dtype=current_bev.dtype,
                device=self._device,
            )
            action[0, action_index] = 1.0
            nominal_delta = self._nominal_delta_table[action_index : action_index + 1]
            predicted, warped, overlap = self._model.predict_from_command(
                current_bev,
                action,
                nominal_delta,
            )
        delta_values = tuple(
            float(value) for value in nominal_delta[0].detach().cpu().tolist()
        )
        return CommandPredictionDiagnostic(
            primitive_name=primitive_name,
            nominal_delta_pose_current=delta_values,
            predicted_next_bev=predicted[0].detach().cpu().numpy().copy(),
            warped_current_bev=warped[0].detach().cpu().numpy().copy(),
            overlap_mask=overlap[0, 0].detach().cpu().numpy().astype(bool, copy=True),
        )


def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    geometry: GeometryContract,
    dataset_manifest_override: Path | None,
) -> _ValidatedCheckpoint:
    if payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema {payload.get('schema')!r}")
    if payload.get("g2_passes") is not True:
        raise ValueError("checkpoint is not promoted: g2_passes must be true")
    g2_evaluation = _mapping(payload.get("g2_evaluation"), "g2_evaluation")
    if _mapping(g2_evaluation.get("g2"), "g2_evaluation.g2").get("passes") is not True:
        raise ValueError("checkpoint G2 evaluation does not record a pass")
    if payload.get("geometry_contract_sha256") != geometry.sha256:
        raise ValueError("checkpoint geometry contract does not match runtime geometry")

    output_contract = _mapping(
        payload.get("occupancy_output_contract"), "occupancy_output_contract"
    )
    if tuple(output_contract.get("class_order", ())) != _CLASS_ORDER:
        raise ValueError("checkpoint occupancy class order is not UNKNOWN/FREE/OCCUPIED")
    if output_contract.get("raw_output") != "three_class_logits":
        raise ValueError("unsupported checkpoint occupancy raw output")
    if (
        output_contract.get("runtime_transform")
        != "apply_probability_calibration_then_softmax"
    ):
        raise ValueError("unsupported checkpoint occupancy runtime transform")
    local_grid = _validate_local_grid(
        _mapping(output_contract.get("local_grid"), "occupancy local_grid")
    )

    model_config = _mapping(payload.get("model_config"), "model_config")
    if set(model_config) != _MODEL_CONFIG_KEYS:
        raise ValueError(
            "model_config keys do not match checkpoint schema v2: "
            f"missing={sorted(_MODEL_CONFIG_KEYS - set(model_config))}, "
            f"extra={sorted(set(model_config) - _MODEL_CONFIG_KEYS)}"
        )
    _validate_model_grid(model_config, local_grid)

    normalization = _mapping(payload.get("image_normalization"), "image_normalization")
    if set(normalization) != {"mean", "std"}:
        raise ValueError("image_normalization must contain exactly mean and std")
    if list(normalization["mean"]) != [0.485, 0.456, 0.406] or list(
        normalization["std"]
    ) != [0.229, 0.224, 0.225]:
        raise ValueError("checkpoint image normalization differs from training contract")

    calibration = _mapping(
        payload.get("probability_calibration"), "probability_calibration"
    )
    calibration_id, log_scales, biases = _validate_calibration(calibration)
    if payload.get("probability_calibration_id") != calibration_id:
        raise ValueError("checkpoint probability calibration ID is inconsistent")
    thresholds_mapping = _mapping(
        payload.get("traversability_thresholds"), "traversability_thresholds"
    )
    try:
        thresholds = TraversabilityThresholds(**dict(thresholds_mapping))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid checkpoint traversability thresholds") from exc
    thresholds.validate()
    calibration_metrics = _mapping(
        payload.get("calibration_metrics"), "calibration_metrics"
    )
    if _canonical_json(calibration_metrics.get("thresholds")) != _canonical_json(
        thresholds_mapping
    ):
        raise ValueError("runtime thresholds differ from held-out calibration metrics")
    metric_calibration = _mapping(
        calibration_metrics.get("calibration"), "calibration_metrics.calibration"
    )
    if metric_calibration.get("id") != calibration_id:
        raise ValueError("calibration metrics reference a different calibration")

    primitive_to_index = _validate_primitive_mapping(
        _mapping(payload.get("primitive_to_index"), "primitive_to_index")
    )
    if int(model_config["action_dim"]) != len(primitive_to_index):
        raise ValueError("model action_dim does not match primitive vocabulary")
    nominal_delta = np.asarray(
        payload.get("nominal_primitive_delta_current"), dtype=np.float32
    )
    if nominal_delta.shape != (len(primitive_to_index), 3) or not np.isfinite(
        nominal_delta
    ).all():
        raise ValueError("nominal primitive delta table has invalid shape or values")
    if (
        payload.get("nominal_primitive_delta_source")
        != "coordinatewise_train_median_with_circular_yaw"
    ):
        raise ValueError("unsupported nominal primitive delta provenance")
    expected_delta_id = "go2-train-median-delta-" + _json_sha256(
        {
            "primitive_to_index": dict(primitive_to_index),
            "values": nominal_delta.tolist(),
        }
    )[:16]
    if payload.get("nominal_primitive_delta_id") != expected_delta_id:
        raise ValueError("nominal primitive delta ID does not match its values")

    training_scene_ids = payload.get("training_scene_ids")
    if (
        not isinstance(training_scene_ids, list)
        or not training_scene_ids
        or any(not isinstance(value, str) or not value for value in training_scene_ids)
        or len(training_scene_ids) != len(set(training_scene_ids))
    ):
        raise ValueError("training_scene_ids must be a non-empty unique string list")
    _require_sha256(str(payload.get("scene_roles_sha256", "")), "scene_roles_sha256")

    recorded_manifest = Path(str(payload.get("dataset_manifest_path", "")))
    manifest_path = (
        Path(dataset_manifest_override)
        if dataset_manifest_override is not None
        else recorded_manifest
    )
    if not manifest_path.is_absolute():
        manifest_path = checkpoint_path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    recorded_manifest_sha = str(payload.get("dataset_manifest_sha256", ""))
    _require_sha256(recorded_manifest_sha, "dataset_manifest_sha256")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if _sha256_file(manifest_path) != recorded_manifest_sha:
        raise ValueError("dataset manifest SHA-256 does not match checkpoint provenance")
    manifest = _load_json_mapping(manifest_path)
    if manifest.get("schema") != _DATASET_SCHEMA:
        raise ValueError("checkpoint dataset manifest has an unsupported schema")
    manifest_geometry = _mapping(
        manifest.get("geometry_contract"), "dataset geometry_contract"
    )
    if manifest_geometry.get("sha256") != geometry.sha256:
        raise ValueError("dataset and runtime geometry contracts differ")
    geometry_file_sha = str(manifest_geometry.get("file_sha256", ""))
    _require_sha256(geometry_file_sha, "dataset geometry file_sha256")
    if geometry_file_sha != _sha256_file(geometry.source_path):
        raise ValueError("dataset geometry file hash is stale")
    if _canonical_json(manifest.get("local_grid")) != _canonical_json(local_grid):
        raise ValueError("dataset and checkpoint local-grid contracts differ")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset manifest sources must be a non-empty list")
    manifest_training = {
        str(source.get("scene_id"))
        for source in sources
        if isinstance(source, Mapping) and source.get("dataset_split") == "train"
    }
    if manifest_training != set(training_scene_ids):
        raise ValueError("checkpoint training scenes differ from dataset provenance")
    return _ValidatedCheckpoint(
        payload=payload,
        model_config=dict(model_config),
        calibration_log_scales=log_scales,
        calibration_biases=biases,
        calibration_id=calibration_id,
        thresholds=thresholds,
        primitive_to_index=primitive_to_index,
        nominal_delta_table=nominal_delta,
        local_grid=local_grid,
        dataset_manifest_path=manifest_path,
        dataset_manifest_sha256=recorded_manifest_sha,
    )


def _validate_local_grid(grid: Mapping[str, Any]) -> Mapping[str, Any]:
    shape = _numeric_sequence(grid.get("shape"), 2, "local_grid.shape", integer=True)
    if min(shape) <= 0:
        raise ValueError("local_grid shape must be positive")
    cell_size = _finite_float(grid.get("cell_size_m"), "local_grid.cell_size_m")
    if cell_size <= 0.0:
        raise ValueError("local_grid cell_size_m must be positive")
    forward_edges = _numeric_sequence(
        grid.get("forward_edge_range_m"), 2, "forward_edge_range_m"
    )
    left_edges = _numeric_sequence(grid.get("left_edge_range_m"), 2, "left_edge_range_m")
    forward_centers = _numeric_sequence(
        grid.get("forward_center_range_m"), 2, "forward_center_range_m"
    )
    left_centers = _numeric_sequence(
        grid.get("left_center_range_m"), 2, "left_center_range_m"
    )
    expected = {
        "forward_edge_max": forward_edges[0] + shape[0] * cell_size,
        "left_edge_max": left_edges[0] + shape[1] * cell_size,
        "forward_center_min": forward_edges[0] + 0.5 * cell_size,
        "forward_center_max": forward_edges[1] - 0.5 * cell_size,
        "left_center_min": left_edges[0] + 0.5 * cell_size,
        "left_center_max": left_edges[1] - 0.5 * cell_size,
    }
    actual = {
        "forward_edge_max": forward_edges[1],
        "left_edge_max": left_edges[1],
        "forward_center_min": forward_centers[0],
        "forward_center_max": forward_centers[1],
        "left_center_min": left_centers[0],
        "left_center_max": left_centers[1],
    }
    for name, expected_value in expected.items():
        if not math.isclose(actual[name], expected_value, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"local_grid metric geometry is inconsistent at {name}")
    if grid.get("array_axes") != {
        "row": "base_forward_increasing",
        "column": "base_left_increasing",
    }:
        raise ValueError("local_grid array axes are not body-forward/body-left")
    if grid.get("base_frame_axes") != {
        "forward": "+x_base_link",
        "left": "+y_base_link",
    }:
        raise ValueError("local_grid base-frame axes are unsupported")
    if grid.get("bounds_are") != "cell_edges":
        raise ValueError("local_grid bounds must denote cell edges")
    return dict(grid)


def _validate_model_grid(
    model_config: Mapping[str, Any], local_grid: Mapping[str, Any]
) -> None:
    comparisons = (
        (model_config.get("bev_size"), local_grid["shape"], "BEV shape"),
        (
            model_config.get("forward_range_m"),
            local_grid["forward_center_range_m"],
            "forward center range",
        ),
        (
            model_config.get("left_range_m"),
            local_grid["left_center_range_m"],
            "left center range",
        ),
    )
    for model_value, grid_value, name in comparisons:
        if not isinstance(model_value, (list, tuple)) or len(model_value) != 2:
            raise ValueError(f"model {name} must have two values")
        if not np.allclose(
            np.asarray(model_value, dtype=np.float64),
            np.asarray(grid_value, dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"model and occupancy local-grid {name} differ")
    for key in ("image_size", "patch_size", "action_dim"):
        value = model_config.get(key)
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"model_config.{key} must be a positive integer")


def _validate_calibration(
    calibration: Mapping[str, Any],
) -> tuple[str, tuple[float, float, float], tuple[float, float, float]]:
    required = {
        "method",
        "log_scales",
        "biases",
        "sample_count",
        "nll_before",
        "nll_after",
        "id",
    }
    allowed = (required, required | {"provenance"})
    if set(calibration) not in allowed:
        raise ValueError("probability calibration fields do not match schema v2")
    if "provenance" in calibration and not isinstance(
        calibration["provenance"], Mapping
    ):
        raise ValueError("probability calibration provenance must be an object")
    if calibration.get("method") != _CALIBRATION_METHOD:
        raise ValueError("unsupported probability calibration method")
    log_scales = _numeric_sequence(calibration.get("log_scales"), 3, "log_scales")
    biases = _numeric_sequence(calibration.get("biases"), 3, "biases")
    if any(abs(value) > 3.0 + 1e-7 for value in log_scales):
        raise ValueError("calibration log scales exceed the trained clamp")
    sample_count = calibration.get("sample_count")
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count <= 0:
        raise ValueError("calibration sample_count must be positive")
    before = _finite_float(calibration.get("nll_before"), "nll_before")
    after = _finite_float(calibration.get("nll_after"), "nll_after")
    if after > before + 1e-6:
        raise ValueError("checkpoint calibration worsens held-out NLL")
    calibration_id = str(calibration.get("id", ""))
    unhashed = dict(calibration)
    del unhashed["id"]
    expected_id = "go2-vector-scale-" + _json_sha256(unhashed)[:16]
    if calibration_id != expected_id:
        raise ValueError("probability calibration ID does not match its parameters")
    return calibration_id, tuple(log_scales), tuple(biases)


def _validate_primitive_mapping(value: Mapping[str, Any]) -> dict[str, int]:
    if not value or any(not isinstance(name, str) or not name for name in value):
        raise ValueError("primitive_to_index must have non-empty string keys")
    result: dict[str, int] = {}
    for name, index in value.items():
        if isinstance(index, bool) or int(index) != index:
            raise ValueError("primitive indices must be integers")
        result[name] = int(index)
    if set(result.values()) != set(range(len(result))):
        raise ValueError("primitive indices must be contiguous from zero")
    return result


def _validate_camera(camera: CameraGeometry, geometry: GeometryContract) -> None:
    if not isinstance(camera, CameraGeometry):
        raise TypeError("camera must be a perception-map CameraGeometry")
    expected = geometry.camera
    if not math.isclose(
        camera.horizontal_fov_deg,
        expected.horizontal_fov_deg,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("runtime camera horizontal FOV differs from geometry contract")
    if not np.allclose(
        np.asarray(camera.mount_xyz_m),
        np.asarray(expected.nominal_xyz_body_m),
        rtol=0.0,
        atol=1e-9,
    ) or not np.allclose(
        np.asarray(camera.mount_rpy_rad),
        np.asarray(expected.nominal_rpy_body_rad),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("runtime camera mount differs from geometry contract")


def _validate_belief_map(
    belief_map: OnlineBeliefMap, geometry: GeometryContract
) -> None:
    if not isinstance(belief_map, OnlineBeliefMap):
        raise TypeError("belief_map must be an OnlineBeliefMap")
    expected = geometry.configuration_space
    if not math.isclose(
        belief_map.config.cell_size_m,
        expected.online_cell_size_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("belief-map cell size differs from geometry contract")
    if belief_map.config.planning_connectivity != expected.connectivity:
        raise ValueError("belief-map connectivity differs from geometry contract")
    if (
        belief_map.config.allow_diagonal_corner_cutting
        != expected.allow_diagonal_corner_cutting
    ):
        raise ValueError("belief-map corner-cutting policy differs from geometry contract")


def _apply_vector_calibration(
    logits: torch.Tensor,
    log_scales: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise RuntimeError("occupancy logits must have shape (batch, 3, height, width)")
    scales = torch.exp(log_scales.clamp(-3.0, 3.0))[None, :, None, None]
    centered_biases = (biases - biases.mean())[None, :, None, None]
    return logits.float() * scales + centered_biases


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _numeric_sequence(
    value: Any,
    length: int,
    name: str,
    *,
    integer: bool = False,
) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    parsed = tuple(_finite_float(item, name) for item in value)
    if integer:
        if any(isinstance(item, bool) or int(item) != item for item in value):
            raise ValueError(f"{name} values must be integers")
        return tuple(int(item) for item in value)
    return parsed


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical-JSON serializable") from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON provenance file: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON provenance root must be an object: {path}")
    return payload
