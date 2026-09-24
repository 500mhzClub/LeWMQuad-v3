"""Versioned geometry contract shared by navigation data, planning, and eval."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_GEOMETRY_CONTRACT = Path("config/go2_generalization_geometry_v1.json")
DEPLOYMENT_GEOMETRY_CONTRACT = Path("config/go2_generalization_geometry_v2.json")


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CameraGeometry:
    frame: str
    horizontal_fov_deg: float
    near_m: float
    nominal_xyz_body_m: tuple[float, float, float]
    nominal_rpy_body_rad: tuple[float, float, float]
    apply_manifest_extrinsic_jitter: bool
    apply_runtime_safety_retraction_to_labels: bool


@dataclass(frozen=True)
class ConfigurationSpaceGeometry:
    reference_frame: str
    oracle_cell_size_m: float
    online_cell_size_m: float
    body_inflation_radius_m: float
    connectivity: int
    allow_diagonal_corner_cutting: bool
    landmarks_are_obstacles: bool
    distractors_are_obstacles: bool


@dataclass(frozen=True)
class SweptFootprintGeometry:
    forward_m: float
    rear_m: float
    half_width_m: float
    probe_margin_m: float
    source: str
    calibration_required_for_physical_promotion: bool
    directional_policy_id: str | None = None
    directional_policy_content_sha256: str | None = None
    directional_profile: str | None = None
    maximum_vertex_radius_m: float | None = None
    planning_disc_radius_m: float | None = None
    strict_collision_representation: str | None = None
    planning_representation: str | None = None


@dataclass(frozen=True)
class VisibilityAndClaimGeometry:
    claim_radius_m: float
    standoff_m: float
    standoff_candidates: int
    minimum_navigable_corridor_width_m: float
    require_line_of_sight_for_scene_validity: bool
    require_true_distance_for_success: bool


@dataclass(frozen=True)
class CoverageGeometry:
    cell_size_m: float
    normalization: str
    mark_swept_reference_path: bool


@dataclass(frozen=True)
class KinematicExecutionGeometry:
    collision_space: str
    maximum_translation_substep_m: float
    minimum_progress_m: float


@dataclass(frozen=True)
class GeometryContract:
    schema: str
    status: str
    source_artifacts: Mapping[str, Mapping[str, str]]
    camera: CameraGeometry
    configuration_space: ConfigurationSpaceGeometry
    swept_footprint: SweptFootprintGeometry
    visibility_and_claim: VisibilityAndClaimGeometry
    coverage: CoverageGeometry
    kinematic_execution: KinematicExecutionGeometry
    sha256: str
    source_path: Path

    @property
    def physical_promotion_ready(self) -> bool:
        return not self.swept_footprint.calibration_required_for_physical_promotion


def _triple(values: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _positive(value: Any, *, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive(value: Any, *, name: str) -> float | None:
    return None if value is None else _positive(value, name=name)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid geometry contract JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("geometry contract root must be an object")
    return payload


def load_geometry_contract(
    path: Path = DEFAULT_GEOMETRY_CONTRACT,
    *,
    repository_root: Path | None = None,
    verify_sources: bool = True,
) -> GeometryContract:
    """Load, validate, and optionally verify the content-addressed contract."""

    source_path = path if path.is_absolute() else (repository_root or Path.cwd()) / path
    source_path = source_path.resolve()
    payload = _load_payload(source_path)
    schema = str(payload.get("schema", ""))
    if schema not in {
        "lewm_go2_generalization_geometry_v1",
        "lewm_go2_generalization_geometry_v2",
    }:
        raise ValueError(f"unsupported geometry contract schema: {schema!r}")

    root = (repository_root or source_path.parents[1]).resolve()
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_artifacts, dict) or not source_artifacts:
        raise ValueError("source_artifacts must be a non-empty object")
    if verify_sources:
        for name, record in source_artifacts.items():
            if not isinstance(record, dict):
                raise ValueError(f"source_artifacts.{name} must be an object")
            artifact_path = (root / str(record.get("path", ""))).resolve()
            expected = str(record.get("sha256", ""))
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            actual = _sha256_file(artifact_path)
            if actual != expected:
                raise ValueError(
                    f"source artifact hash mismatch for {name}: "
                    f"expected {expected}, got {actual}"
                )

    camera = payload["camera"]
    config = payload["configuration_space"]
    swept = payload["swept_footprint"]
    visibility = payload["visibility_and_claim"]
    coverage = payload["coverage"]
    execution = payload["kinematic_execution"]

    connectivity = int(config["connectivity"])
    if connectivity not in (4, 8):
        raise ValueError("configuration_space.connectivity must be 4 or 8")
    standoff_candidates = int(visibility["standoff_candidates"])
    if standoff_candidates < 4:
        raise ValueError("visibility_and_claim.standoff_candidates must be >= 4")

    contract = GeometryContract(
        schema=schema,
        status=str(payload["status"]),
        source_artifacts=source_artifacts,
        camera=CameraGeometry(
            frame=str(camera["frame"]),
            horizontal_fov_deg=_positive(
                camera["horizontal_fov_deg"], name="camera.horizontal_fov_deg"
            ),
            near_m=_positive(camera["near_m"], name="camera.near_m"),
            nominal_xyz_body_m=_triple(
                camera["nominal_xyz_body_m"], name="camera.nominal_xyz_body_m"
            ),
            nominal_rpy_body_rad=_triple(
                camera["nominal_rpy_body_rad"], name="camera.nominal_rpy_body_rad"
            ),
            apply_manifest_extrinsic_jitter=bool(
                camera["apply_manifest_extrinsic_jitter"]
            ),
            apply_runtime_safety_retraction_to_labels=bool(
                camera["apply_runtime_safety_retraction_to_labels"]
            ),
        ),
        configuration_space=ConfigurationSpaceGeometry(
            reference_frame=str(config["reference_frame"]),
            oracle_cell_size_m=_positive(
                config["oracle_cell_size_m"],
                name="configuration_space.oracle_cell_size_m",
            ),
            online_cell_size_m=_positive(
                config["online_cell_size_m"],
                name="configuration_space.online_cell_size_m",
            ),
            body_inflation_radius_m=_positive(
                config["body_inflation_radius_m"],
                name="configuration_space.body_inflation_radius_m",
            ),
            connectivity=connectivity,
            allow_diagonal_corner_cutting=bool(
                config["allow_diagonal_corner_cutting"]
            ),
            landmarks_are_obstacles=bool(config["landmarks_are_obstacles"]),
            distractors_are_obstacles=bool(config["distractors_are_obstacles"]),
        ),
        swept_footprint=SweptFootprintGeometry(
            forward_m=_positive(swept["forward_m"], name="swept_footprint.forward_m"),
            rear_m=_positive(swept["rear_m"], name="swept_footprint.rear_m"),
            half_width_m=_positive(
                swept["half_width_m"], name="swept_footprint.half_width_m"
            ),
            probe_margin_m=_positive(
                swept["probe_margin_m"], name="swept_footprint.probe_margin_m"
            ),
            source=str(swept["source"]),
            calibration_required_for_physical_promotion=bool(
                swept["calibration_required_for_physical_promotion"]
            ),
            directional_policy_id=(
                str(swept["directional_policy_id"])
                if swept.get("directional_policy_id") is not None
                else None
            ),
            directional_policy_content_sha256=(
                str(swept["directional_policy_content_sha256"])
                if swept.get("directional_policy_content_sha256") is not None
                else None
            ),
            directional_profile=(
                str(swept["directional_profile"])
                if swept.get("directional_profile") is not None
                else None
            ),
            maximum_vertex_radius_m=_optional_positive(
                swept.get("maximum_vertex_radius_m"),
                name="swept_footprint.maximum_vertex_radius_m",
            ),
            planning_disc_radius_m=_optional_positive(
                swept.get("planning_disc_radius_m"),
                name="swept_footprint.planning_disc_radius_m",
            ),
            strict_collision_representation=(
                str(swept["strict_collision_representation"])
                if swept.get("strict_collision_representation") is not None
                else None
            ),
            planning_representation=(
                str(swept["planning_representation"])
                if swept.get("planning_representation") is not None
                else None
            ),
        ),
        visibility_and_claim=VisibilityAndClaimGeometry(
            claim_radius_m=_positive(
                visibility["claim_radius_m"],
                name="visibility_and_claim.claim_radius_m",
            ),
            standoff_m=_positive(
                visibility["standoff_m"], name="visibility_and_claim.standoff_m"
            ),
            standoff_candidates=standoff_candidates,
            minimum_navigable_corridor_width_m=_positive(
                visibility["minimum_navigable_corridor_width_m"],
                name="visibility_and_claim.minimum_navigable_corridor_width_m",
            ),
            require_line_of_sight_for_scene_validity=bool(
                visibility["require_line_of_sight_for_scene_validity"]
            ),
            require_true_distance_for_success=bool(
                visibility["require_true_distance_for_success"]
            ),
        ),
        coverage=CoverageGeometry(
            cell_size_m=_positive(coverage["cell_size_m"], name="coverage.cell_size_m"),
            normalization=str(coverage["normalization"]),
            mark_swept_reference_path=bool(coverage["mark_swept_reference_path"]),
        ),
        kinematic_execution=KinematicExecutionGeometry(
            collision_space=str(execution["collision_space"]),
            maximum_translation_substep_m=_positive(
                execution["maximum_translation_substep_m"],
                name="kinematic_execution.maximum_translation_substep_m",
            ),
            minimum_progress_m=_positive(
                execution["minimum_progress_m"],
                name="kinematic_execution.minimum_progress_m",
            ),
        ),
        sha256=_canonical_sha256(payload),
        source_path=source_path,
    )
    if contract.camera.horizontal_fov_deg >= 180.0:
        raise ValueError("camera.horizontal_fov_deg must be less than 180")
    if contract.visibility_and_claim.standoff_m > contract.visibility_and_claim.claim_radius_m:
        raise ValueError("standoff_m must not exceed claim_radius_m")
    if schema == "lewm_go2_generalization_geometry_v2":
        policy_hash = contract.swept_footprint.directional_policy_content_sha256
        if policy_hash is None or len(policy_hash) != 64 or any(
            character not in "0123456789abcdef" for character in policy_hash
        ):
            raise ValueError(
                "geometry v2 requires a lowercase directional policy content SHA-256"
            )
        maximum_radius = contract.swept_footprint.maximum_vertex_radius_m
        planning_radius = contract.swept_footprint.planning_disc_radius_m
        if maximum_radius is None or planning_radius is None:
            raise ValueError("geometry v2 requires polygon and planning-disc radii")
        if planning_radius < maximum_radius:
            raise ValueError("planning disc must enclose the directional footprint")
        if not math.isclose(
            planning_radius,
            contract.configuration_space.body_inflation_radius_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "geometry v2 planning disc must equal body_inflation_radius_m"
            )
    return contract
