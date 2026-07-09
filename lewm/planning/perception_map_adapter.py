"""Typed perception-to-map adapter for calibrated occupancy predictions.

This module is deliberately one-way. It validates and fuses deployment-facing
perception into :class:`OnlineBeliefMap`, but exposes no alternative planner or
optimistic occupancy query. Navigation must continue to use the belief map's
strict confirmed-free connectivity and shortest-path APIs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.planning.online_belief_map import Cell, OnlineBeliefMap, PoseBelief

__all__ = [
    "CameraGeometry",
    "EgocentricOccupancyGeometry",
    "FusionRecord",
    "GridReferenceFrame",
    "ModelArtifact",
    "ObservationProvenance",
    "OccupancyObservation",
    "OccupancyValueKind",
    "PerceptionMapAdapterConfig",
    "PerceptionMapContract",
    "PerceptionToBeliefMapAdapter",
    "supercover_project_cell",
]


class OccupancyValueKind(str, Enum):
    """Numerical representation supplied by an occupancy head."""

    # Legacy binary values represent P(occupied). They remain supported for
    # archived heads, but cannot express unknown space.
    PROBABILITY = "probability"
    LOGIT = "logit"
    CATEGORICAL_PROBABILITY = "categorical_probability"
    CATEGORICAL_LOGIT = "categorical_logit"


class GridReferenceFrame(str, Enum):
    """Ground-plane frame used by an egocentric occupancy grid."""

    BODY = "body"
    CAMERA_GROUND = "camera_ground"


@dataclass(frozen=True)
class ModelArtifact:
    """Identity of one model artifact used to produce an observation."""

    artifact_id: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CameraGeometry:
    """Camera calibration and rigid mount in the robot body frame."""

    calibration_id: str
    image_height_px: int
    image_width_px: int
    horizontal_fov_deg: float
    vertical_fov_deg: float
    mount_xyz_m: tuple[float, float, float]
    mount_rpy_rad: tuple[float, float, float]
    optical_axis_convention: str = "x_forward_y_left_z_up"

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")
        _positive_int(self.image_height_px, "image_height_px")
        _positive_int(self.image_width_px, "image_width_px")
        horizontal_fov = _finite_float(
            self.horizontal_fov_deg, "horizontal_fov_deg"
        )
        vertical_fov = _finite_float(self.vertical_fov_deg, "vertical_fov_deg")
        if not 0.0 < horizontal_fov < 180.0:
            raise ValueError("horizontal_fov_deg must be in (0, 180)")
        if not 0.0 < vertical_fov < 180.0:
            raise ValueError("vertical_fov_deg must be in (0, 180)")
        _finite_vector(self.mount_xyz_m, 3, "mount_xyz_m")
        _finite_vector(self.mount_rpy_rad, 3, "mount_rpy_rad")
        if self.optical_axis_convention != "x_forward_y_left_z_up":
            raise ValueError(
                "only x_forward_y_left_z_up camera geometry is supported"
            )
        object.__setattr__(self, "image_height_px", int(self.image_height_px))
        object.__setattr__(self, "image_width_px", int(self.image_width_px))
        object.__setattr__(self, "horizontal_fov_deg", horizontal_fov)
        object.__setattr__(self, "vertical_fov_deg", vertical_fov)
        object.__setattr__(self, "mount_xyz_m", _float_tuple(self.mount_xyz_m))
        object.__setattr__(self, "mount_rpy_rad", _float_tuple(self.mount_rpy_rad))

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "image_height_px": self.image_height_px,
            "image_width_px": self.image_width_px,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "mount_xyz_m": list(self.mount_xyz_m),
            "mount_rpy_rad": list(self.mount_rpy_rad),
            "optical_axis_convention": self.optical_axis_convention,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class EgocentricOccupancyGeometry:
    """Metric geometry and safety semantics of an occupancy-head output.

    Array rows increase in the local forward direction; columns increase left.
    ``forward_min_m`` and ``left_min_m`` locate the lower edges of row/column
    zero. Values represent probability that a robot body center is obstructed,
    after inflating labels by ``body_inflation_radius_m``.
    """

    geometry_id: str
    height: int
    width: int
    cell_size_m: float
    forward_min_m: float
    left_min_m: float
    reference_frame: GridReferenceFrame
    body_inflation_radius_m: float
    semantics: str = "body_center_occupied_probability"

    def __post_init__(self) -> None:
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty")
        _positive_int(self.height, "height")
        _positive_int(self.width, "width")
        cell_size = _finite_float(self.cell_size_m, "cell_size_m")
        if cell_size <= 0.0:
            raise ValueError("cell_size_m must be positive")
        forward_min = _finite_float(self.forward_min_m, "forward_min_m")
        left_min = _finite_float(self.left_min_m, "left_min_m")
        if not isinstance(self.reference_frame, GridReferenceFrame):
            raise ValueError("reference_frame must be a GridReferenceFrame")
        inflation_radius = _finite_float(
            self.body_inflation_radius_m,
            "body_inflation_radius_m",
        )
        if inflation_radius <= 0.0:
            raise ValueError("body_inflation_radius_m must be positive")
        if self.semantics != "body_center_occupied_probability":
            raise ValueError(
                "occupancy semantics must be body_center_occupied_probability"
            )
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "cell_size_m", cell_size)
        object.__setattr__(self, "forward_min_m", forward_min)
        object.__setattr__(self, "left_min_m", left_min)
        object.__setattr__(self, "body_inflation_radius_m", inflation_radius)

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "height": self.height,
            "width": self.width,
            "cell_size_m": self.cell_size_m,
            "forward_min_m": self.forward_min_m,
            "left_min_m": self.left_min_m,
            "reference_frame": self.reference_frame.value,
            "body_inflation_radius_m": self.body_inflation_radius_m,
            "semantics": self.semantics,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class PerceptionMapContract:
    """Expected model and geometry identity for deployment-valid fusion."""

    backbone: ModelArtifact
    occupancy_head: ModelArtifact
    probability_calibration_id: str
    camera: CameraGeometry
    occupancy_geometry: EgocentricOccupancyGeometry
    map_cell_size_m: float = 0.25
    map_frame: str = "odometry"
    observation_source: str = "onboard_rgb"

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, ModelArtifact):
            raise TypeError("backbone must be a ModelArtifact")
        if not isinstance(self.occupancy_head, ModelArtifact):
            raise TypeError("occupancy_head must be a ModelArtifact")
        if not isinstance(self.camera, CameraGeometry):
            raise TypeError("camera must be a CameraGeometry")
        if not isinstance(
            self.occupancy_geometry,
            EgocentricOccupancyGeometry,
        ):
            raise TypeError(
                "occupancy_geometry must be an EgocentricOccupancyGeometry"
            )
        if not self.probability_calibration_id:
            raise ValueError("probability_calibration_id must be non-empty")
        map_cell_size = _finite_float(self.map_cell_size_m, "map_cell_size_m")
        if map_cell_size <= 0.0:
            raise ValueError("map_cell_size_m must be positive")
        if not self.map_frame:
            raise ValueError("map_frame must be non-empty")
        if not self.observation_source:
            raise ValueError("observation_source must be non-empty")
        object.__setattr__(self, "map_cell_size_m", map_cell_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone.to_dict(),
            "occupancy_head": self.occupancy_head.to_dict(),
            "probability_calibration_id": self.probability_calibration_id,
            "camera": self.camera.to_dict(),
            "camera_geometry_sha256": self.camera.fingerprint,
            "occupancy_geometry": self.occupancy_geometry.to_dict(),
            "occupancy_geometry_sha256": self.occupancy_geometry.fingerprint,
            "map_cell_size_m": self.map_cell_size_m,
            "map_frame": self.map_frame,
            "observation_source": self.observation_source,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class ObservationProvenance:
    """Exact producer and geometry identity attached to one observation."""

    observation_id: str
    backbone: ModelArtifact
    occupancy_head: ModelArtifact
    probability_calibration_id: str
    camera_calibration_id: str
    camera_geometry_sha256: str
    occupancy_geometry_sha256: str
    source: str = "onboard_rgb"

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if not isinstance(self.backbone, ModelArtifact):
            raise TypeError("backbone must be a ModelArtifact")
        if not isinstance(self.occupancy_head, ModelArtifact):
            raise TypeError("occupancy_head must be a ModelArtifact")
        if not self.probability_calibration_id:
            raise ValueError("probability_calibration_id must be non-empty")
        if not self.camera_calibration_id:
            raise ValueError("camera_calibration_id must be non-empty")
        _validate_sha256(
            self.camera_geometry_sha256,
            "camera_geometry_sha256",
        )
        _validate_sha256(
            self.occupancy_geometry_sha256,
            "occupancy_geometry_sha256",
        )
        if not self.source:
            raise ValueError("source must be non-empty")

    @classmethod
    def create(
        cls,
        observation_id: str,
        *,
        contract: PerceptionMapContract,
    ) -> ObservationProvenance:
        return cls(
            observation_id=observation_id,
            backbone=contract.backbone,
            occupancy_head=contract.occupancy_head,
            probability_calibration_id=contract.probability_calibration_id,
            camera_calibration_id=contract.camera.calibration_id,
            camera_geometry_sha256=contract.camera.fingerprint,
            occupancy_geometry_sha256=contract.occupancy_geometry.fingerprint,
            source=contract.observation_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "backbone": self.backbone.to_dict(),
            "occupancy_head": self.occupancy_head.to_dict(),
            "probability_calibration_id": self.probability_calibration_id,
            "camera_calibration_id": self.camera_calibration_id,
            "camera_geometry_sha256": self.camera_geometry_sha256,
            "occupancy_geometry_sha256": self.occupancy_geometry_sha256,
            "source": self.source,
        }


@dataclass(frozen=True)
class OccupancyObservation:
    """One calibrated occupancy output registered to a body pose."""

    values: np.ndarray
    value_kind: OccupancyValueKind
    pose: PoseBelief
    camera: CameraGeometry
    geometry: EgocentricOccupancyGeometry
    provenance: ObservationProvenance
    valid_mask: np.ndarray | None = None
    observation_confidence: float = 1.0


@dataclass(frozen=True)
class PerceptionMapAdapterConfig:
    """Conservative classification and evidence-fusion parameters."""

    free_probability_max: float = 0.25
    occupied_probability_min: float = 0.75
    planner_free_probability_min: float = 0.75
    planner_occupied_probability_max: float = 0.25
    planner_unknown_probability_max: float = 0.25
    occupied_class_probability_min: float = 0.75
    evidence_scale: float = 2.0
    minimum_evidence: float = 0.05
    pose_uncertainty_scale_m: float = 0.5
    maximum_pose_position_std_m: float | None = 1.5
    minimum_body_inflation_radius_m: float = 0.15
    projection_epsilon_m: float = 1e-8

    def __post_init__(self) -> None:
        free_threshold = _finite_float(
            self.free_probability_max,
            "free_probability_max",
        )
        occupied_threshold = _finite_float(
            self.occupied_probability_min,
            "occupied_probability_min",
        )
        if not 0.0 <= free_threshold < 0.5:
            raise ValueError("free_probability_max must be in [0, 0.5)")
        if not 0.5 < occupied_threshold <= 1.0:
            raise ValueError("occupied_probability_min must be in (0.5, 1]")
        if free_threshold >= occupied_threshold:
            raise ValueError("free and occupied probability bands must not overlap")
        categorical_thresholds = {
            "planner_free_probability_min": self.planner_free_probability_min,
            "planner_occupied_probability_max": self.planner_occupied_probability_max,
            "planner_unknown_probability_max": self.planner_unknown_probability_max,
            "occupied_class_probability_min": self.occupied_class_probability_min,
        }
        for name, value in categorical_thresholds.items():
            checked = _finite_float(value, name)
            if not 0.0 <= checked <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, checked)
        if _finite_float(self.evidence_scale, "evidence_scale") <= 0.0:
            raise ValueError("evidence_scale must be positive")
        minimum_evidence = _finite_float(self.minimum_evidence, "minimum_evidence")
        if minimum_evidence < 0.0:
            raise ValueError("minimum_evidence must be non-negative")
        if (
            _finite_float(
                self.pose_uncertainty_scale_m,
                "pose_uncertainty_scale_m",
            )
            <= 0.0
        ):
            raise ValueError("pose_uncertainty_scale_m must be positive")
        if self.maximum_pose_position_std_m is not None:
            maximum_std = _finite_float(
                self.maximum_pose_position_std_m,
                "maximum_pose_position_std_m",
            )
            if maximum_std <= 0.0:
                raise ValueError("maximum_pose_position_std_m must be positive")
        if (
            _finite_float(
                self.minimum_body_inflation_radius_m,
                "minimum_body_inflation_radius_m",
            )
            <= 0.0
        ):
            raise ValueError("minimum_body_inflation_radius_m must be positive")
        if _finite_float(self.projection_epsilon_m, "projection_epsilon_m") < 0.0:
            raise ValueError("projection_epsilon_m must be non-negative")
        object.__setattr__(self, "free_probability_max", free_threshold)
        object.__setattr__(self, "occupied_probability_min", occupied_threshold)
        object.__setattr__(
            self,
            "evidence_scale",
            _finite_float(self.evidence_scale, "evidence_scale"),
        )
        object.__setattr__(self, "minimum_evidence", minimum_evidence)
        object.__setattr__(
            self,
            "pose_uncertainty_scale_m",
            _finite_float(
                self.pose_uncertainty_scale_m,
                "pose_uncertainty_scale_m",
            ),
        )
        object.__setattr__(
            self,
            "maximum_pose_position_std_m",
            None
            if self.maximum_pose_position_std_m is None
            else _finite_float(
                self.maximum_pose_position_std_m,
                "maximum_pose_position_std_m",
            ),
        )
        object.__setattr__(
            self,
            "minimum_body_inflation_radius_m",
            _finite_float(
                self.minimum_body_inflation_radius_m,
                "minimum_body_inflation_radius_m",
            ),
        )
        object.__setattr__(
            self,
            "projection_epsilon_m",
            _finite_float(self.projection_epsilon_m, "projection_epsilon_m"),
        )


@dataclass(frozen=True)
class FusionRecord:
    """Serializable provenance and evidence summary for one fused frame."""

    observation_id: str
    observation_payload_sha256: str
    tick: int
    contract_sha256: str
    backbone_id: str
    backbone_checkpoint_sha256: str
    occupancy_head_id: str
    occupancy_head_checkpoint_sha256: str
    probability_calibration_id: str
    camera_calibration_id: str
    camera_geometry_sha256: str
    occupancy_geometry_sha256: str
    map_cell_size_m: float
    observation_source: str
    value_kind: str
    observation_confidence: float
    pose_frame: str
    pose_mean: tuple[float, float, float]
    pose_covariance: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    pose_registration_sha256: str
    pose_position_std_m: float
    pose_confidence_weight: float
    free_source_cells: int
    occupied_source_cells: int
    projected_free_cells: int
    projected_occupied_cells: int
    projected_conflicted_cells: int
    total_free_evidence: float
    total_occupied_evidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerceptionToBeliefMapAdapter:
    """Validate and fuse calibrated occupancy into an ``OnlineBeliefMap``."""

    PROVENANCE_SCHEMA = "lewm_perception_map_fusion_records"
    PROVENANCE_VERSION = 1

    def __init__(
        self,
        belief_map: OnlineBeliefMap,
        contract: PerceptionMapContract,
        config: PerceptionMapAdapterConfig | None = None,
    ) -> None:
        if not isinstance(belief_map, OnlineBeliefMap):
            raise TypeError("belief_map must be an OnlineBeliefMap")
        if not isinstance(contract, PerceptionMapContract):
            raise TypeError("contract must be a PerceptionMapContract")
        if config is not None and not isinstance(config, PerceptionMapAdapterConfig):
            raise TypeError("config must be a PerceptionMapAdapterConfig")
        self._belief_map = belief_map
        self._contract = contract
        self._config = config or PerceptionMapAdapterConfig()
        if (
            contract.occupancy_geometry.body_inflation_radius_m
            < self._config.minimum_body_inflation_radius_m
        ):
            raise ValueError(
                "occupancy geometry is not inflated by the configured body radius"
            )
        if not math.isclose(
            belief_map.config.cell_size_m,
            contract.map_cell_size_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("belief-map cell size does not match the contract")
        if self._config.projection_epsilon_m * 2.0 >= min(
            contract.occupancy_geometry.cell_size_m,
            contract.map_cell_size_m,
        ):
            raise ValueError("projection_epsilon_m is too large for the map geometry")
        self._records: list[FusionRecord] = []
        self._observation_ids: set[str] = set()

    @property
    def contract(self) -> PerceptionMapContract:
        return self._contract

    @property
    def records(self) -> tuple[FusionRecord, ...]:
        return tuple(self._records)

    def provenance_state_dict(self) -> dict[str, Any]:
        """Return JSON-safe provenance records to persist beside map state."""

        return {
            "schema": self.PROVENANCE_SCHEMA,
            "version": self.PROVENANCE_VERSION,
            "contract": self._contract.to_dict(),
            "contract_sha256": self._contract.fingerprint,
            "config": asdict(self._config),
            "records": [record.to_dict() for record in self._records],
        }

    def fuse(self, observation: OccupancyObservation) -> FusionRecord:
        """Validate, supercover-project, and atomically fuse one observation."""

        (
            free_probabilities,
            occupied_probabilities,
            unknown_probabilities,
            valid_mask,
            pose_std,
            categorical,
        ) = self._validate_observation(observation)
        pose_weight = 1.0 / (
            1.0 + (pose_std / self._config.pose_uncertainty_scale_m) ** 2
        )
        observation_scale = (
            self._config.evidence_scale
            * observation.observation_confidence
            * pose_weight
        )
        free_evidence: dict[Cell, float] = {}
        occupied_evidence: dict[Cell, float] = {}
        free_source_cells = 0
        occupied_source_cells = 0

        for row in range(observation.geometry.height):
            for col in range(observation.geometry.width):
                if not valid_mask[row, col]:
                    continue
                free_probability = float(free_probabilities[row, col])
                occupied_probability = float(occupied_probabilities[row, col])
                if categorical:
                    free_cell = (
                        free_probability
                        >= self._config.planner_free_probability_min
                        and occupied_probability
                        <= self._config.planner_occupied_probability_max
                        and float(unknown_probabilities[row, col])
                        <= self._config.planner_unknown_probability_max
                    )
                    occupied_cell = (
                        occupied_probability
                        >= self._config.occupied_class_probability_min
                    )
                    free_weight = observation_scale * free_probability
                    occupied_weight = observation_scale * occupied_probability
                else:
                    free_cell = (
                        occupied_probability <= self._config.free_probability_max
                    )
                    occupied_cell = (
                        occupied_probability >= self._config.occupied_probability_min
                    )
                    free_weight = observation_scale * (
                        1.0 - 2.0 * occupied_probability
                    )
                    occupied_weight = observation_scale * (
                        2.0 * occupied_probability - 1.0
                    )
                if free_cell:
                    weight = free_weight
                    if weight < self._config.minimum_evidence:
                        continue
                    free_source_cells += 1
                    projected = supercover_project_cell(
                        row,
                        col,
                        geometry=observation.geometry,
                        camera=observation.camera,
                        pose=observation.pose,
                        map_cell_size_m=self._belief_map.config.cell_size_m,
                        epsilon_m=self._config.projection_epsilon_m,
                    )
                    for cell in projected:
                        free_evidence[cell] = max(free_evidence.get(cell, 0.0), weight)
                elif occupied_cell:
                    weight = occupied_weight
                    if weight < self._config.minimum_evidence:
                        continue
                    occupied_source_cells += 1
                    projected = supercover_project_cell(
                        row,
                        col,
                        geometry=observation.geometry,
                        camera=observation.camera,
                        pose=observation.pose,
                        map_cell_size_m=self._belief_map.config.cell_size_m,
                        epsilon_m=self._config.projection_epsilon_m,
                    )
                    for cell in projected:
                        occupied_evidence[cell] = max(
                            occupied_evidence.get(cell, 0.0),
                            weight,
                        )

        tick = observation.pose.tick
        self._belief_map.set_pose(
            observation.pose.mean,
            observation.pose.covariance,
            tick=tick,
            frame=observation.pose.frame,
        )
        projected_cells = sorted(set(free_evidence) | set(occupied_evidence))
        for cell in projected_cells:
            self._belief_map.fuse_cell(
                cell,
                free_evidence=free_evidence.get(cell, 0.0),
                occupied_evidence=occupied_evidence.get(cell, 0.0),
                tick=tick,
            )

        conflicted = set(free_evidence) & set(occupied_evidence)
        provenance = observation.provenance
        pose_state = {
            "mean": list(observation.pose.mean),
            "covariance": [list(row) for row in observation.pose.covariance],
            "tick": observation.pose.tick,
            "frame": observation.pose.frame,
        }
        record = FusionRecord(
            observation_id=provenance.observation_id,
            observation_payload_sha256=_observation_payload_fingerprint(
                observation
            ),
            tick=tick,
            contract_sha256=self._contract.fingerprint,
            backbone_id=provenance.backbone.artifact_id,
            backbone_checkpoint_sha256=provenance.backbone.checkpoint_sha256,
            occupancy_head_id=provenance.occupancy_head.artifact_id,
            occupancy_head_checkpoint_sha256=(
                provenance.occupancy_head.checkpoint_sha256
            ),
            probability_calibration_id=provenance.probability_calibration_id,
            camera_calibration_id=provenance.camera_calibration_id,
            camera_geometry_sha256=provenance.camera_geometry_sha256,
            occupancy_geometry_sha256=provenance.occupancy_geometry_sha256,
            map_cell_size_m=self._contract.map_cell_size_m,
            observation_source=provenance.source,
            value_kind=observation.value_kind.value,
            observation_confidence=float(observation.observation_confidence),
            pose_frame=observation.pose.frame,
            pose_mean=observation.pose.mean,
            pose_covariance=observation.pose.covariance,
            pose_registration_sha256=_fingerprint(pose_state),
            pose_position_std_m=pose_std,
            pose_confidence_weight=pose_weight,
            free_source_cells=free_source_cells,
            occupied_source_cells=occupied_source_cells,
            projected_free_cells=len(free_evidence),
            projected_occupied_cells=len(occupied_evidence),
            projected_conflicted_cells=len(conflicted),
            total_free_evidence=float(sum(free_evidence.values())),
            total_occupied_evidence=float(sum(occupied_evidence.values())),
        )
        self._records.append(record)
        self._observation_ids.add(provenance.observation_id)
        return record

    def _validate_observation(
        self,
        observation: OccupancyObservation,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        if not isinstance(observation, OccupancyObservation):
            raise TypeError("observation must be an OccupancyObservation")
        if not isinstance(observation.value_kind, OccupancyValueKind):
            raise ValueError("value_kind must be an OccupancyValueKind")
        if not isinstance(observation.pose, PoseBelief):
            raise TypeError("observation pose must be a PoseBelief")
        if not isinstance(observation.camera, CameraGeometry):
            raise TypeError("observation camera must be a CameraGeometry")
        if not isinstance(
            observation.geometry,
            EgocentricOccupancyGeometry,
        ):
            raise TypeError(
                "observation geometry must be an EgocentricOccupancyGeometry"
            )
        if not isinstance(observation.provenance, ObservationProvenance):
            raise TypeError(
                "observation provenance must be an ObservationProvenance"
            )
        provenance = observation.provenance
        if provenance.observation_id in self._observation_ids:
            raise ValueError(
                f"duplicate occupancy observation {provenance.observation_id!r}"
            )
        if observation.camera.fingerprint != self._contract.camera.fingerprint:
            raise ValueError("camera geometry does not match the adapter contract")
        if (
            observation.geometry.fingerprint
            != self._contract.occupancy_geometry.fingerprint
        ):
            raise ValueError("occupancy geometry does not match the adapter contract")
        if provenance.backbone != self._contract.backbone:
            raise ValueError("observation backbone does not match the adapter contract")
        if provenance.occupancy_head != self._contract.occupancy_head:
            raise ValueError(
                "observation occupancy head does not match the adapter contract"
            )
        if (
            provenance.probability_calibration_id
            != self._contract.probability_calibration_id
        ):
            raise ValueError(
                "observation probability calibration does not match the contract"
            )
        if provenance.camera_calibration_id != observation.camera.calibration_id:
            raise ValueError("provenance camera calibration does not match geometry")
        if provenance.camera_geometry_sha256 != observation.camera.fingerprint:
            raise ValueError("provenance camera geometry fingerprint is stale")
        if provenance.occupancy_geometry_sha256 != observation.geometry.fingerprint:
            raise ValueError("provenance occupancy geometry fingerprint is stale")
        if provenance.source != self._contract.observation_source:
            raise ValueError("observation source does not match the adapter contract")
        if observation.pose.frame != self._contract.map_frame:
            raise ValueError("observation pose frame does not match the map contract")
        if observation.pose.tick < self._belief_map.current_tick:
            raise ValueError("occupancy observation is older than the belief map")

        confidence = _finite_float(
            observation.observation_confidence,
            "observation_confidence",
        )
        if not 0.0 < confidence <= 1.0:
            raise ValueError("observation_confidence must be in (0, 1]")

        values = np.asarray(observation.values)
        spatial_shape = (
            observation.geometry.height,
            observation.geometry.width,
        )
        categorical = observation.value_kind in (
            OccupancyValueKind.CATEGORICAL_PROBABILITY,
            OccupancyValueKind.CATEGORICAL_LOGIT,
        )
        expected_shape = (3, *spatial_shape) if categorical else spatial_shape
        if values.shape != expected_shape:
            raise ValueError(
                f"occupancy values must have shape {expected_shape}, got {values.shape}"
            )
        if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number):
            raise ValueError("occupancy values must use a numeric non-boolean dtype")
        values = values.astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("occupancy values must all be finite")
        if observation.value_kind is OccupancyValueKind.PROBABILITY:
            if (values < 0.0).any() or (values > 1.0).any():
                raise ValueError("occupancy probabilities must be in [0, 1]")
            occupied_probabilities = values
            free_probabilities = 1.0 - values
            unknown_probabilities = np.zeros_like(values)
        elif observation.value_kind is OccupancyValueKind.LOGIT:
            occupied_probabilities = _stable_sigmoid(values)
            free_probabilities = 1.0 - occupied_probabilities
            unknown_probabilities = np.zeros_like(values)
        elif observation.value_kind is OccupancyValueKind.CATEGORICAL_PROBABILITY:
            if (values < 0.0).any() or (values > 1.0).any():
                raise ValueError("categorical probabilities must be in [0, 1]")
            if not np.allclose(values.sum(axis=0), 1.0, atol=1e-5):
                raise ValueError("categorical probabilities must sum to one")
            free_probabilities = values[1]
            occupied_probabilities = values[2]
            unknown_probabilities = values[0]
        else:
            shifted = values - values.max(axis=0, keepdims=True)
            exponent = np.exp(shifted)
            probabilities = exponent / exponent.sum(axis=0, keepdims=True)
            free_probabilities = probabilities[1]
            occupied_probabilities = probabilities[2]
            unknown_probabilities = probabilities[0]

        if observation.valid_mask is None:
            valid_mask = np.ones(spatial_shape, dtype=np.bool_)
        else:
            valid_mask = np.asarray(observation.valid_mask)
            if valid_mask.shape != spatial_shape:
                raise ValueError(
                    "valid_mask must have shape "
                    f"{spatial_shape}, got {valid_mask.shape}"
                )
            if valid_mask.dtype != np.bool_:
                raise ValueError("valid_mask must use boolean dtype")

        pose_covariance = np.asarray(observation.pose.covariance, dtype=np.float64)
        pose_std = math.sqrt(
            max(0.0, float(pose_covariance[0, 0] + pose_covariance[1, 1]))
        )
        maximum_std = self._config.maximum_pose_position_std_m
        if maximum_std is not None and pose_std > maximum_std:
            raise ValueError(
                "pose position uncertainty exceeds the configured fusion limit"
            )
        return (
            free_probabilities,
            occupied_probabilities,
            unknown_probabilities,
            valid_mask,
            pose_std,
            categorical,
        )


def supercover_project_cell(
    row: int,
    col: int,
    *,
    geometry: EgocentricOccupancyGeometry,
    camera: CameraGeometry,
    pose: PoseBelief,
    map_cell_size_m: float,
    epsilon_m: float = 1e-8,
) -> frozenset[Cell]:
    """Project one local cell footprint into every overlapping global cell."""

    if isinstance(row, bool) or int(row) != row or not 0 <= int(row) < geometry.height:
        raise ValueError("row is outside occupancy geometry")
    if isinstance(col, bool) or int(col) != col or not 0 <= int(col) < geometry.width:
        raise ValueError("col is outside occupancy geometry")
    map_scale = _finite_float(map_cell_size_m, "map_cell_size_m")
    if map_scale <= 0.0:
        raise ValueError("map_cell_size_m must be positive")
    epsilon = _finite_float(epsilon_m, "epsilon_m")
    if epsilon < 0.0:
        raise ValueError("epsilon_m must be non-negative")

    local_scale = geometry.cell_size_m
    forward_min = geometry.forward_min_m + int(row) * local_scale
    forward_max = forward_min + local_scale
    left_min = geometry.left_min_m + int(col) * local_scale
    left_max = left_min + local_scale
    local_corners = (
        (forward_min, left_min),
        (forward_max, left_min),
        (forward_max, left_max),
        (forward_min, left_max),
    )
    body_corners = tuple(
        _grid_point_to_body(point, geometry=geometry, camera=camera)
        for point in local_corners
    )
    polygon = tuple(_body_point_to_map(point, pose=pose) for point in body_corners)

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    min_cell_x = int(math.floor((min_x + epsilon) / map_scale))
    max_cell_x = int(math.floor((max_x - epsilon) / map_scale))
    min_cell_y = int(math.floor((min_y + epsilon) / map_scale))
    max_cell_y = int(math.floor((max_y - epsilon) / map_scale))
    result: set[Cell] = set()
    for cell_x in range(min_cell_x, max_cell_x + 1):
        for cell_y in range(min_cell_y, max_cell_y + 1):
            if _polygon_overlaps_cell(
                polygon,
                cell=(cell_x, cell_y),
                cell_size_m=map_scale,
                epsilon_m=epsilon,
            ):
                result.add((cell_x, cell_y))
    if not result:
        center = np.asarray(polygon, dtype=np.float64).mean(axis=0)
        result.add(
            (
                int(math.floor(float(center[0]) / map_scale)),
                int(math.floor(float(center[1]) / map_scale)),
            )
        )
    return frozenset(result)


def _grid_point_to_body(
    point: tuple[float, float],
    *,
    geometry: EgocentricOccupancyGeometry,
    camera: CameraGeometry,
) -> tuple[float, float]:
    forward, left = point
    if geometry.reference_frame is GridReferenceFrame.BODY:
        return forward, left
    mount_x, mount_y, _mount_z = camera.mount_xyz_m
    mount_yaw = camera.mount_rpy_rad[2]
    cos_yaw = math.cos(mount_yaw)
    sin_yaw = math.sin(mount_yaw)
    return (
        mount_x + cos_yaw * forward - sin_yaw * left,
        mount_y + sin_yaw * forward + cos_yaw * left,
    )


def _body_point_to_map(
    point: tuple[float, float],
    *,
    pose: PoseBelief,
) -> tuple[float, float]:
    body_forward, body_left = point
    map_x, map_y, yaw = pose.mean
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        map_x + cos_yaw * body_forward - sin_yaw * body_left,
        map_y + sin_yaw * body_forward + cos_yaw * body_left,
    )


def _polygon_overlaps_cell(
    polygon: Sequence[tuple[float, float]],
    *,
    cell: Cell,
    cell_size_m: float,
    epsilon_m: float,
) -> bool:
    x0 = cell[0] * cell_size_m
    y0 = cell[1] * cell_size_m
    rectangle = (
        (x0, y0),
        (x0 + cell_size_m, y0),
        (x0 + cell_size_m, y0 + cell_size_m),
        (x0, y0 + cell_size_m),
    )
    axes: list[tuple[float, float]] = [(1.0, 0.0), (0.0, 1.0)]
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        edge_x = end[0] - start[0]
        edge_y = end[1] - start[1]
        length = math.hypot(edge_x, edge_y)
        if length > 0.0:
            axes.append((-edge_y / length, edge_x / length))
    for axis_x, axis_y in axes:
        polygon_projection = [
            point[0] * axis_x + point[1] * axis_y for point in polygon
        ]
        rectangle_projection = [
            point[0] * axis_x + point[1] * axis_y for point in rectangle
        ]
        overlap = min(max(polygon_projection), max(rectangle_projection)) - max(
            min(polygon_projection),
            min(rectangle_projection),
        )
        if overlap <= epsilon_m:
            return False
    return True


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _observation_payload_fingerprint(
    observation: OccupancyObservation,
) -> str:
    values = np.ascontiguousarray(np.asarray(observation.values))
    header = {
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "value_kind": observation.value_kind.value,
        "has_valid_mask": observation.valid_mask is not None,
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(values.tobytes(order="C"))
    if observation.valid_mask is not None:
        valid_mask = np.ascontiguousarray(
            np.asarray(observation.valid_mask, dtype=np.bool_)
        )
        digest.update(valid_mask.tobytes(order="C"))
    return digest.hexdigest()


def _fingerprint(state: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    valid_characters = "0123456789abcdef"
    if len(value) != 64 or any(
        character not in valid_characters for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_tuple(values: Sequence[float]) -> tuple:
    return tuple(float(value) for value in values)


def _finite_vector(values: Sequence[float], size: int, name: str) -> None:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite vector of length {size}") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result
