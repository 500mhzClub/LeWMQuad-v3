"""Pure geometry for the minimum multi-origin body-range qualification.

This module contains no corpus reader, contact outcome, learned component, or
scientific materialisation path.  It freezes the label-free pieces needed by
the evaluator that follows ``BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1``:

* four rigid-trunk mount hardpoints and the six permitted pair/triple layouts;
* a four-member supplemental orientation library selected only from static
  nominal protected-surface visibility;
* deterministic, mount-independent scan phases;
* union of per-origin support with complete origin/timestamp provenance; and
* training-role layout-support summaries over complete transition x 50-step x
  protected-link witness arrays.

Coordinates are body/base FRU (``+x`` forward, ``+y`` left, ``+z`` up), and
quaternions are scalar-first ``wxyz`` active local-to-body rotations.  The
analytic ray and closest-witness authority remains
``body_centric_range_coverage_v1``; this module composes those primitives and
does not introduce a second intersection implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.safety import body_centric_range_coverage_v1 as range_core


SCHEMA_VERSION = "minimum_multi_origin_body_range_coverage_v1"
EXPERIMENT_ID = "MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1"
PHASE_NAMESPACE = "MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1/L2_PHASE_V1"

HEAD_STOCK = "HEAD_STOCK"
REAR_TOP_TRUNK = "REAR_TOP_TRUNK"
LEFT_UPPER_FLANK = "LEFT_UPPER_FLANK"
RIGHT_UPPER_FLANK = "RIGHT_UPPER_FLANK"
MOUNT_IDS = (HEAD_STOCK, REAR_TOP_TRUNK, LEFT_UPPER_FLANK, RIGHT_UPPER_FLANK)

STOCK = "STOCK"
LEVEL = "LEVEL"
INVERTED = "INVERTED"
OUTWARD_DOWNWARD = "OUTWARD_DOWNWARD"
INWARD_DOWNWARD = "INWARD_DOWNWARD"
SUPPLEMENTAL_ORIENTATION_IDS = (LEVEL, INVERTED, OUTWARD_DOWNWARD, INWARD_DOWNWARD)

TRUNK = "TRUNK"
FRONT_LIMBS = "FRONT_LIMBS"
REAR_LIMBS = "REAR_LIMBS"
HIPS_AND_THIGHS = "HIPS_AND_THIGHS"
CALVES = "CALVES"
BODY_REGION_IDS = (TRUNK, FRONT_LIMBS, REAR_LIMBS, HIPS_AND_THIGHS, CALVES)

LAYOUT_DEFINITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HEAD_STOCK__REAR_TOP_TRUNK", (HEAD_STOCK, REAR_TOP_TRUNK)),
    ("HEAD_STOCK__LEFT_UPPER_FLANK", (HEAD_STOCK, LEFT_UPPER_FLANK)),
    ("HEAD_STOCK__RIGHT_UPPER_FLANK", (HEAD_STOCK, RIGHT_UPPER_FLANK)),
    (
        "HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK",
        (HEAD_STOCK, REAR_TOP_TRUNK, LEFT_UPPER_FLANK),
    ),
    (
        "HEAD_STOCK__REAR_TOP_TRUNK__RIGHT_UPPER_FLANK",
        (HEAD_STOCK, REAR_TOP_TRUNK, RIGHT_UPPER_FLANK),
    ),
    (
        "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK",
        (HEAD_STOCK, LEFT_UPPER_FLANK, RIGHT_UPPER_FLANK),
    ),
)
LAYOUT_IDS = tuple(row[0] for row in LAYOUT_DEFINITIONS)
LAYOUT_MOUNTS = {row[0]: row[1] for row in LAYOUT_DEFINITIONS}

HEAD_STOCK_TRANSLATION_BODY_XYZ_M = (0.28945, 0.0, -0.046825)
HEAD_STOCK_RPY_RAD = (0.0, 2.8782, 0.0)
DEFAULT_HOUSING_FULL_EXTENTS_XYZ_M = (0.075, 0.075, 0.065)
DEFAULT_MECHANICAL_CLEARANCE_M = 0.01
DEFAULT_HORIZONTAL_FOV_DEG = (-180.0, 180.0)
DEFAULT_VERTICAL_FOV_DEG = (-6.0, 90.0)
DEFAULT_NEAR_M = 0.05
DEFAULT_FAR_M = 30.0
PHYSICS_STEPS_PER_TICK = 50

NOMINAL_STANCE_SOURCE_PATH = "lewm_genesis/lewm_genesis/rollout.py"
NOMINAL_STANCE_SOURCE_SHA256 = "06501bbbdd1e071a3a91e765d77bd19da5f2c311c35d75df4631c452beea034a"
NOMINAL_STANCE_SOURCE_LINES = (130, 140)
GENESIS_GO2_URDF_PACKAGE_RELATIVE_PATH = "genesis/assets/urdf/go2/urdf/go2.urdf"
GENESIS_GO2_URDF_SHA256 = "4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4"
# The source constant is float32.  Expanding through float32 preserves the
# exact values used by the rollout, rather than silently replacing them with
# neighboring float64 decimal literals.
FROZEN_NOMINAL_STANCE_RAD = tuple(
    float(item)
    for item in np.asarray(
        (0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8),
        dtype=np.float32,
    )
)
FROZEN_NOMINAL_BASE_POSITION_XYZ_M = (0.0, 0.0, 0.0)
FROZEN_NOMINAL_BASE_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)

_UNIT_X = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
_UNIT_Y = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
_UNIT_Z = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
_TOL = 1.0e-12


def _vector(value: Sequence[float] | np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _fraction(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).mean()) if np.asarray(mask).size else 0.0


def _finite_float_or_none(value: float) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from error
    if value != value.lower() or len(decoded) != 32:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def rpy_quaternion_wxyz(rpy_rad: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert fixed-axis roll/pitch/yaw radians to normalized ``wxyz``."""

    roll, pitch, yaw = _vector(rpy_rad, 3, "rpy_rad")
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    quaternion = range_core.normalize_quaternion_wxyz(
        np.asarray(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dtype=np.float64,
        )
    )
    return tuple(float(item) for item in quaternion)


def quaternion_to_rpy_wxyz(quaternion_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """Return the deterministic ZYX roll/pitch/yaw representation."""

    matrix = range_core.rotation_matrix_wxyz(quaternion_wxyz)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-12:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        # The projected-X construction has a deterministic gimbal-lock
        # representative: yaw is zero and roll carries the remaining rotation.
        yaw = 0.0
        roll = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
    return float(roll), float(pitch), float(yaw)


def _quaternion_from_basis(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper local-to-body rotation matrix to canonical ``wxyz``."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must have finite shape [3,3]")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=2.0e-12):
        raise ValueError("rotation basis must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, rel_tol=0.0, abs_tol=2.0e-12):
        raise ValueError("rotation basis must be right handed")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale)
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                ((matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale)
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                ((matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
            )
    quaternion = range_core.normalize_quaternion_wxyz(quaternion)
    # q and -q are the same rotation.  Canonicalize the first nonzero element.
    for item in quaternion:
        if abs(float(item)) > _TOL:
            if item < 0.0:
                quaternion = -quaternion
            break
    return tuple(float(item) for item in quaternion)


def quaternion_from_pole(pole_body_xyz: Sequence[float]) -> tuple[float, float, float, float]:
    """Build the frozen orientation basis from a requested sensor ``+Z`` pole.

    Sensor ``+X`` is body ``+X`` projected orthogonal to the pole, with body
    ``+Y`` as the only fallback.  Sensor ``+Y = +Z x +X`` completes the
    right-handed basis.  This is the prospective label-free construction.
    """

    pole = _vector(pole_body_xyz, 3, "pole_body_xyz")
    norm = float(np.linalg.norm(pole))
    if norm <= _TOL:
        raise ValueError("pole_body_xyz must be nonzero")
    local_z = pole / norm
    local_x = _UNIT_X - float(_UNIT_X @ local_z) * local_z
    if float(np.linalg.norm(local_x)) <= _TOL:
        local_x = _UNIT_Y - float(_UNIT_Y @ local_z) * local_z
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    local_y /= np.linalg.norm(local_y)
    return _quaternion_from_basis(np.column_stack((local_x, local_y, local_z)))


@dataclass(frozen=True)
class AxisAlignedEnvelope:
    minimum_xyz_m: tuple[float, float, float]
    maximum_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        lower = _vector(self.minimum_xyz_m, 3, "minimum_xyz_m")
        upper = _vector(self.maximum_xyz_m, 3, "maximum_xyz_m")
        if np.any(upper <= lower):
            raise ValueError("envelope maximum must strictly exceed minimum")

    @property
    def size_xyz_m(self) -> np.ndarray:
        return np.asarray(self.maximum_xyz_m) - np.asarray(self.minimum_xyz_m)

    @property
    def center_xyz_m(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.minimum_xyz_m) + np.asarray(self.maximum_xyz_m))

    def to_serializable(self) -> dict[str, object]:
        return {
            "maximum_xyz_m": list(self.maximum_xyz_m),
            "minimum_xyz_m": list(self.minimum_xyz_m),
            "size_xyz_m": [float(item) for item in self.size_xyz_m],
        }


def robot_primitive_aabb(primitive: range_core.RobotPrimitive) -> AxisAlignedEnvelope:
    """Return the exact body-axis AABB of one frozen collision primitive."""

    center = np.asarray(primitive.position_xyz_m, dtype=np.float64)
    rotation = range_core.rotation_matrix_wxyz(primitive.quaternion_wxyz)
    data = np.asarray(primitive.data, dtype=np.float64)
    if primitive.kind == "sphere":
        extent = np.full(3, data[0], dtype=np.float64)
    elif primitive.kind == "capsule":
        extent = np.full(3, data[0], dtype=np.float64)
        extent += np.abs(rotation[:, 2]) * (0.5 * data[1])
    else:
        extent = np.abs(rotation) @ (0.5 * data[:3])
    return AxisAlignedEnvelope(tuple(center - extent), tuple(center + extent))


def derive_trunk_envelope(
    trunk_primitives: Sequence[range_core.RobotPrimitive],
) -> AxisAlignedEnvelope:
    """Union rigid-trunk primitive AABBs in the body/base frame."""

    primitives = tuple(trunk_primitives)
    if not primitives:
        raise ValueError("trunk_primitives must be nonempty")
    bounds = tuple(robot_primitive_aabb(item) for item in primitives)
    lower = np.min(np.asarray([item.minimum_xyz_m for item in bounds]), axis=0)
    upper = np.max(np.asarray([item.maximum_xyz_m for item in bounds]), axis=0)
    return AxisAlignedEnvelope(tuple(lower), tuple(upper))


def _compose_pose(
    parent: tuple[Sequence[float], Sequence[float]],
    translation_xyz_m: Sequence[float],
    rpy_rad: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    return range_core.compose_transform(
        parent[0], parent[1], translation_xyz_m, rpy_quaternion_wxyz(rpy_rad)
    )


def frozen_static_nominal_go2_primitives() -> tuple[range_core.RobotPrimitive, ...]:
    """Instantiate the Genesis Go2 collision geometry at the frozen stance.

    This is a source-bound, label-free FK fixture: base pose is identity and
    the 12 joint values are the float32 ``DEFAULT_GO2_STANCE_RAD`` constant.
    Genesis collapses fixed head geometry into ``base`` and fixed distal
    collision geometry into each calf link, yielding the predecessor's 27
    primitives over exactly 13 protected links (base plus 12 hip/thigh/calf
    links).  Feet are not exposed as independent protected links.
    """

    base_pose = (
        np.asarray(FROZEN_NOMINAL_BASE_POSITION_XYZ_M, dtype=np.float64),
        np.asarray(FROZEN_NOMINAL_BASE_QUATERNION_WXYZ, dtype=np.float64),
    )
    primitives: list[range_core.RobotPrimitive] = []

    def add(
        identity: str,
        kind: str,
        data: tuple[float, ...],
        link_pose: tuple[Sequence[float], Sequence[float]],
        collision_translation: Sequence[float],
        collision_rpy: Sequence[float],
        geom_index: int,
        link_index: int,
        link_name: str,
    ) -> None:
        position, quaternion = _compose_pose(link_pose, collision_translation, collision_rpy)
        primitives.append(
            range_core.RobotPrimitive(
                identity=identity,
                kind=kind,
                data=data,
                position_xyz_m=tuple(float(item) for item in position),
                quaternion_wxyz=tuple(float(item) for item in quaternion),
                geom_index=geom_index,
                link_index=link_index,
                link_name=link_name,
            )
        )

    add("base:00", "box", (0.3762, 0.0935, 0.114), base_pose, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0, 0, "base")
    head_upper = _compose_pose(base_pose, (0.285, 0.0, 0.01))
    add("base:01", "capsule", (0.05, 0.09), head_upper, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, 0, "base")
    head_lower = _compose_pose(head_upper, (0.008, 0.0, -0.07))
    add("base:02", "sphere", (0.047,), head_lower, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2, 0, "base")

    stance = FROZEN_NOMINAL_STANCE_RAD
    legs = (
        ("FL", 1.0, 1.0, 0, 3, (0.008, -0.21, 0.012)),
        ("FR", 1.0, -1.0, 1, 9, (0.01, -0.2, 0.013)),
        ("RL", -1.0, 1.0, 2, 15, (0.01, -0.2, 0.013)),
        ("RR", -1.0, -1.0, 3, 21, (0.01, -0.2, 0.013)),
    )
    for prefix, longitudinal_sign, lateral_sign, stance_index, geom_start, calf_shape in legs:
        hip_name = f"{prefix}_hip"
        thigh_name = f"{prefix}_thigh"
        calf_name = f"{prefix}_calf"
        hip_link_index = 1 + 3 * stance_index
        thigh_link_index = hip_link_index + 1
        calf_link_index = hip_link_index + 2
        hip_joint = _compose_pose(
            base_pose,
            (0.1934 * longitudinal_sign, 0.0465 * lateral_sign, 0.0),
            (stance[stance_index], 0.0, 0.0),
        )
        add(
            f"{hip_name}:00",
            "capsule",
            (0.046, 0.04),
            hip_joint,
            (0.0, 0.08 * lateral_sign, 0.0),
            (math.pi / 2.0, 0.0, 0.0),
            geom_start,
            hip_link_index,
            hip_name,
        )
        thigh_joint = _compose_pose(
            hip_joint,
            (0.0, 0.0955 * lateral_sign, 0.0),
            (0.0, stance[4 + stance_index], 0.0),
        )
        add(
            f"{thigh_name}:00",
            "box",
            (0.11, 0.0245, 0.034),
            thigh_joint,
            (0.0, 0.0, -0.1065),
            (0.0, math.pi / 2.0, 0.0),
            geom_start + 1,
            thigh_link_index,
            thigh_name,
        )
        calf_joint = _compose_pose(
            thigh_joint,
            (0.0, 0.0, -0.213),
            (0.0, stance[8 + stance_index], 0.0),
        )
        calf_x, calf_pitch, calf_radius = calf_shape
        add(
            f"{calf_name}:00",
            "capsule",
            (calf_radius, 0.12),
            calf_joint,
            (calf_x, 0.0, -0.06),
            (0.0, calf_pitch, 0.0),
            geom_start + 2,
            calf_link_index,
            calf_name,
        )
        calf_lower = _compose_pose(calf_joint, (0.020, 0.0, -0.148), (0.0, 0.05, 0.0))
        add(
            f"{calf_name}:01",
            "capsule",
            (0.011, 0.065),
            calf_lower,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            geom_start + 3,
            calf_link_index,
            calf_name,
        )
        calf_lower_one = _compose_pose(calf_lower, (-0.01, 0.0, -0.04), (0.0, 0.48, 0.0))
        add(
            f"{calf_name}:02",
            "capsule",
            (0.0155, 0.03),
            calf_lower_one,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            geom_start + 4,
            calf_link_index,
            calf_name,
        )
        foot = _compose_pose(calf_joint, (0.0, 0.0, -0.213))
        add(
            f"{calf_name}:03",
            "sphere",
            (0.022,),
            foot,
            (-0.002, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            geom_start + 5,
            calf_link_index,
            calf_name,
        )
    output = tuple(sorted(primitives, key=lambda row: row.geom_index))
    if len(output) != 27 or tuple(row.geom_index for row in output) != tuple(range(27)):
        raise RuntimeError("frozen nominal Go2 geometry must contain contiguous 27 primitives")
    if len({(row.link_index, row.link_name) for row in output}) != 13:
        raise RuntimeError("frozen nominal Go2 geometry must map to 13 protected links")
    return output


@dataclass(frozen=True)
class MountHardpoint:
    mount_id: str
    parent_link: str
    translation_body_xyz_m: tuple[float, float, float]
    outward_body_xyz: tuple[float, float, float]
    placement_rule: str
    placement_source: str
    fixed_quaternion_body_wxyz: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.mount_id not in MOUNT_IDS:
            raise ValueError(f"unknown mount_id {self.mount_id}")
        if not self.parent_link or not self.placement_rule or not self.placement_source:
            raise ValueError("mount text fields must be nonempty")
        _vector(self.translation_body_xyz_m, 3, "translation_body_xyz_m")
        outward = _vector(self.outward_body_xyz, 3, "outward_body_xyz")
        if abs(float(outward[2])) > _TOL or not math.isclose(
            float(np.linalg.norm(outward)), 1.0, rel_tol=0.0, abs_tol=2.0e-12
        ):
            raise ValueError("outward_body_xyz must be a horizontal unit vector")
        if self.fixed_quaternion_body_wxyz is not None:
            range_core.normalize_quaternion_wxyz(self.fixed_quaternion_body_wxyz)
        if self.mount_id == HEAD_STOCK and self.fixed_quaternion_body_wxyz is None:
            raise ValueError("HEAD_STOCK requires its frozen fixed orientation")
        if self.mount_id != HEAD_STOCK and self.fixed_quaternion_body_wxyz is not None:
            raise ValueError("supplemental mounts must use the frozen orientation library")

    def to_serializable(self) -> dict[str, object]:
        return {
            "fixed_quaternion_body_wxyz": (
                None
                if self.fixed_quaternion_body_wxyz is None
                else list(self.fixed_quaternion_body_wxyz)
            ),
            "mount_id": self.mount_id,
            "outward_body_xyz": list(self.outward_body_xyz),
            "parent_link": self.parent_link,
            "placement_rule": self.placement_rule,
            "placement_source": self.placement_source,
            "translation_body_xyz_m": list(self.translation_body_xyz_m),
        }


def _housing_is_outside(
    center: np.ndarray,
    housing_half_extent: np.ndarray,
    envelope: AxisAlignedEnvelope,
    clearance_m: float,
    mount_id: str,
) -> bool:
    lower = np.asarray(envelope.minimum_xyz_m)
    upper = np.asarray(envelope.maximum_xyz_m)
    if mount_id == REAR_TOP_TRUNK:
        return bool(center[2] - housing_half_extent[2] >= upper[2] + clearance_m - _TOL)
    if mount_id == LEFT_UPPER_FLANK:
        return bool(center[1] - housing_half_extent[1] >= upper[1] + clearance_m - _TOL)
    if mount_id == RIGHT_UPPER_FLANK:
        return bool(center[1] + housing_half_extent[1] <= lower[1] - clearance_m + _TOL)
    return True


def derive_mount_candidates(
    trunk_primitives: Sequence[range_core.RobotPrimitive],
    *,
    mechanical_clearance_m: float = DEFAULT_MECHANICAL_CLEARANCE_M,
    housing_full_extents_xyz_m: Sequence[float] = DEFAULT_HOUSING_FULL_EXTENTS_XYZ_M,
    cad_hardpoints_body_xyz_m: Mapping[str, Sequence[float]] | None = None,
    parent_link: str = "base",
) -> tuple[MountHardpoint, ...]:
    """Derive exactly four prospective hardpoints from rigid-trunk geometry.

    The L2 housing dimensions are full extents.  Supplemental housing AABBs
    are axis-aligned mechanical keep-out envelopes; ray-pole orientation does
    not rotate this conservative installation envelope.  Optional CAD
    hardpoints may replace only the three supplemental translations and must
    still pass the same outside-envelope and sagittal-mirror checks.
    """

    clearance = float(mechanical_clearance_m)
    housing = _vector(housing_full_extents_xyz_m, 3, "housing_full_extents_xyz_m")
    if not math.isfinite(clearance) or clearance <= 0.0 or np.any(housing <= 0.0):
        raise ValueError("mechanical clearance and housing extents must be positive")
    if not parent_link:
        raise ValueError("parent_link must be nonempty")
    envelope = derive_trunk_envelope(trunk_primitives)
    lower = np.asarray(envelope.minimum_xyz_m)
    upper = np.asarray(envelope.maximum_xyz_m)
    size = envelope.size_xyz_m
    half_housing = 0.5 * housing
    center = envelope.center_xyz_m
    derived = {
        REAR_TOP_TRUNK: np.asarray(
            (lower[0] + size[0] / 6.0, center[1], upper[2] + half_housing[2] + clearance)
        ),
        LEFT_UPPER_FLANK: np.asarray(
            (center[0], upper[1] + half_housing[1] + clearance, lower[2] + 0.75 * size[2])
        ),
    }
    derived[RIGHT_UPPER_FLANK] = derived[LEFT_UPPER_FLANK] * np.asarray((1.0, -1.0, 1.0))
    source = "FROZEN_TRUNK_COLLISION_ENVELOPE"
    if cad_hardpoints_body_xyz_m is not None:
        unknown = set(cad_hardpoints_body_xyz_m) - {
            REAR_TOP_TRUNK,
            LEFT_UPPER_FLANK,
            RIGHT_UPPER_FLANK,
        }
        if unknown:
            raise ValueError(f"CAD hardpoints contain forbidden mount IDs: {sorted(unknown)}")
        for mount_id, position in cad_hardpoints_body_xyz_m.items():
            derived[mount_id] = _vector(position, 3, f"cad_hardpoint[{mount_id}]")
        source = "LOCAL_CAD_HARDPOINT"
    if not np.array_equal(
        derived[RIGHT_UPPER_FLANK],
        derived[LEFT_UPPER_FLANK] * np.asarray((1.0, -1.0, 1.0)),
    ):
        raise ValueError("RIGHT_UPPER_FLANK must be the exact sagittal mirror of LEFT_UPPER_FLANK")
    for mount_id, position in derived.items():
        if not _housing_is_outside(position, half_housing, envelope, clearance, mount_id):
            raise ValueError(f"{mount_id} housing is not outside the trunk envelope by the clearance")
    candidates = (
        MountHardpoint(
            mount_id=HEAD_STOCK,
            parent_link=parent_link,
            translation_body_xyz_m=HEAD_STOCK_TRANSLATION_BODY_XYZ_M,
            outward_body_xyz=(1.0, 0.0, 0.0),
            placement_rule="EXACT_BASE_TO_RADAR_FIXED_JOINT",
            placement_source="LOCAL_GO2_URDF",
            fixed_quaternion_body_wxyz=rpy_quaternion_wxyz(HEAD_STOCK_RPY_RAD),
        ),
        MountHardpoint(
            mount_id=REAR_TOP_TRUNK,
            parent_link=parent_link,
            translation_body_xyz_m=tuple(float(item) for item in derived[REAR_TOP_TRUNK]),
            outward_body_xyz=(-1.0, 0.0, 0.0),
            placement_rule="CENTER_OF_REAR_THIRD_OF_TRUNK_TOP_WITH_HOUSING_CLEARANCE",
            placement_source=source,
        ),
        MountHardpoint(
            mount_id=LEFT_UPPER_FLANK,
            parent_link=parent_link,
            translation_body_xyz_m=tuple(float(item) for item in derived[LEFT_UPPER_FLANK]),
            outward_body_xyz=(0.0, 1.0, 0.0),
            placement_rule="TRUNK_LONGITUDINAL_CENTER_UPPER_QUARTER_LEFT_WITH_HOUSING_CLEARANCE",
            placement_source=source,
        ),
        MountHardpoint(
            mount_id=RIGHT_UPPER_FLANK,
            parent_link=parent_link,
            translation_body_xyz_m=tuple(float(item) for item in derived[RIGHT_UPPER_FLANK]),
            outward_body_xyz=(0.0, -1.0, 0.0),
            placement_rule="EXACT_SAGITTAL_MIRROR_OF_LEFT_UPPER_FLANK",
            placement_source=source,
        ),
    )
    if tuple(item.mount_id for item in candidates) != MOUNT_IDS:
        raise RuntimeError("mount candidate set is not the frozen four-member library")
    return candidates


@dataclass(frozen=True)
class SensorPose:
    mount_id: str
    orientation_id: str
    parent_link: str
    translation_body_xyz_m: tuple[float, float, float]
    quaternion_body_wxyz: tuple[float, float, float, float]
    pole_body_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.mount_id not in MOUNT_IDS:
            raise ValueError("unknown mount identity")
        if self.orientation_id not in (STOCK,) + SUPPLEMENTAL_ORIENTATION_IDS:
            raise ValueError("unknown orientation identity")
        _vector(self.translation_body_xyz_m, 3, "translation_body_xyz_m")
        range_core.normalize_quaternion_wxyz(self.quaternion_body_wxyz)
        pole = _vector(self.pole_body_xyz, 3, "pole_body_xyz")
        if not math.isclose(float(np.linalg.norm(pole)), 1.0, rel_tol=0.0, abs_tol=2.0e-12):
            raise ValueError("pole_body_xyz must be unit length")

    def to_serializable(self) -> dict[str, object]:
        rpy = (
            HEAD_STOCK_RPY_RAD
            if self.mount_id == HEAD_STOCK and self.orientation_id == STOCK
            else quaternion_to_rpy_wxyz(self.quaternion_body_wxyz)
        )
        return {
            "mount_id": self.mount_id,
            "orientation_id": self.orientation_id,
            "parent_link": self.parent_link,
            "pole_body_xyz": list(self.pole_body_xyz),
            "quaternion_body_wxyz": list(self.quaternion_body_wxyz),
            "rpy_body_rad": list(rpy),
            "translation_body_xyz_m": list(self.translation_body_xyz_m),
        }


def orientation_library(hardpoint: MountHardpoint) -> tuple[SensorPose, ...]:
    """Return the fixed stock pose or the four frozen supplemental poles."""

    if hardpoint.mount_id == HEAD_STOCK:
        assert hardpoint.fixed_quaternion_body_wxyz is not None
        rotation = range_core.rotation_matrix_wxyz(hardpoint.fixed_quaternion_body_wxyz)
        return (
            SensorPose(
                mount_id=hardpoint.mount_id,
                orientation_id=STOCK,
                parent_link=hardpoint.parent_link,
                translation_body_xyz_m=hardpoint.translation_body_xyz_m,
                quaternion_body_wxyz=hardpoint.fixed_quaternion_body_wxyz,
                pole_body_xyz=tuple(float(item) for item in rotation[:, 2]),
            ),
        )
    outward = np.asarray(hardpoint.outward_body_xyz, dtype=np.float64)
    poles = (
        (LEVEL, _UNIT_Z),
        (INVERTED, -_UNIT_Z),
        (OUTWARD_DOWNWARD, (outward - _UNIT_Z) / math.sqrt(2.0)),
        (INWARD_DOWNWARD, (-outward - _UNIT_Z) / math.sqrt(2.0)),
    )
    output = tuple(
        SensorPose(
            mount_id=hardpoint.mount_id,
            orientation_id=orientation_id,
            parent_link=hardpoint.parent_link,
            translation_body_xyz_m=hardpoint.translation_body_xyz_m,
            quaternion_body_wxyz=quaternion_from_pole(pole),
            pole_body_xyz=tuple(float(item) for item in pole),
        )
        for orientation_id, pole in poles
    )
    if tuple(item.orientation_id for item in output) != SUPPLEMENTAL_ORIENTATION_IDS:
        raise RuntimeError("supplemental orientation library drifted")
    return output


def body_region_membership(link_name: str) -> tuple[str, ...]:
    """Map a protected link name to the five frozen, possibly overlapping regions."""

    normalized = link_name.upper().replace("-", "_")
    output: list[str] = []
    if any(token in normalized for token in ("BASE", "TRUNK", "BODY")):
        output.append(TRUNK)
    front = normalized.startswith(("FL", "FR")) or "FRONT" in normalized
    rear = normalized.startswith(("RL", "RR")) or "REAR" in normalized
    if front:
        output.append(FRONT_LIMBS)
    if rear:
        output.append(REAR_LIMBS)
    if "HIP" in normalized or "THIGH" in normalized:
        output.append(HIPS_AND_THIGHS)
    if "CALF" in normalized:
        output.append(CALVES)
    if not output:
        raise ValueError(f"protected link {link_name!r} has no frozen body-region membership")
    return tuple(output)


@dataclass(frozen=True)
class SurfaceWitnessSet:
    points_body_xyz_m: np.ndarray
    primitive_identity: tuple[str, ...]
    geom_index: np.ndarray
    link_name: tuple[str, ...]
    region_membership: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_body_xyz_m, dtype=np.float64)
        count = len(points)
        if points.shape != (count, 3) or not np.isfinite(points).all() or not count:
            raise ValueError("points_body_xyz_m must have nonempty finite shape [N,3]")
        if len(self.primitive_identity) != count or len(self.link_name) != count:
            raise ValueError("surface witness identity vectors must have length N")
        if np.asarray(self.geom_index).shape != (count,):
            raise ValueError("geom_index must have shape [N]")
        if np.asarray(self.region_membership).shape != (count, len(BODY_REGION_IDS)):
            raise ValueError("region_membership must have shape [N,5]")

    @property
    def count(self) -> int:
        return int(len(self.points_body_xyz_m))

    @property
    def content_digest(self) -> str:
        payload = {
            "geom_index": np.asarray(self.geom_index, dtype=int).tolist(),
            "link_name": list(self.link_name),
            "points_body_xyz_m": np.asarray(self.points_body_xyz_m, dtype=float).tolist(),
            "primitive_identity": list(self.primitive_identity),
            "region_membership": np.asarray(self.region_membership, dtype=bool).tolist(),
        }
        return range_core.canonical_digest(payload)


def _local_surface_samples(primitive: range_core.RobotPrimitive) -> np.ndarray:
    data = np.asarray(primitive.data, dtype=np.float64)
    if primitive.kind == "sphere":
        integer = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 0.0, 1.0)
                for y in (-1.0, 0.0, 1.0)
                for z in (-1.0, 0.0, 1.0)
                if (x, y, z) != (0.0, 0.0, 0.0)
            ]
        )
        directions = integer / np.linalg.norm(integer, axis=1, keepdims=True)
        return directions * data[0]
    if primitive.kind == "capsule":
        angle = 2.0 * math.pi * np.arange(8, dtype=np.float64) / 8.0
        radial = np.column_stack((np.cos(angle), np.sin(angle))) * data[0]
        rows = []
        for z in (-0.5 * data[1], 0.0, 0.5 * data[1]):
            rows.extend((float(x), float(y), float(z)) for x, y in radial)
        rows.extend(((0.0, 0.0, -0.5 * data[1] - data[0]), (0.0, 0.0, 0.5 * data[1] + data[0])))
        return np.asarray(rows, dtype=np.float64)
    half = 0.5 * data[:3]
    rows: list[tuple[float, float, float]] = []
    for axis in range(3):
        other = [item for item in range(3) if item != axis]
        for sign in (-1.0, 1.0):
            for first in (-1.0, 0.0, 1.0):
                for second in (-1.0, 0.0, 1.0):
                    point = np.zeros(3, dtype=np.float64)
                    point[axis] = sign * half[axis]
                    point[other[0]] = first * half[other[0]]
                    point[other[1]] = second * half[other[1]]
                    rows.append(tuple(float(item) for item in point))
    return np.asarray(rows, dtype=np.float64)


def sample_static_protected_surfaces(
    protected_primitives: Sequence[range_core.RobotPrimitive],
) -> SurfaceWitnessSet:
    """Generate a deterministic nominal-geometry protected-surface panel."""

    primitives = tuple(sorted(protected_primitives, key=lambda row: (row.geom_index, row.identity)))
    if not primitives:
        raise ValueError("protected_primitives must be nonempty")
    points: list[np.ndarray] = []
    identities: list[str] = []
    geom_indices: list[int] = []
    link_names: list[str] = []
    memberships: list[list[bool]] = []
    for primitive in primitives:
        local = _local_surface_samples(primitive)
        world = range_core.transform_points(
            local, primitive.position_xyz_m, primitive.quaternion_wxyz
        )
        regions = set(body_region_membership(primitive.link_name))
        for point in world:
            points.append(point)
            identities.append(primitive.identity)
            geom_indices.append(int(primitive.geom_index))
            link_names.append(primitive.link_name)
            memberships.append([region in regions for region in BODY_REGION_IDS])
    return SurfaceWitnessSet(
        points_body_xyz_m=_readonly(points, dtype=np.float64),
        primitive_identity=tuple(identities),
        geom_index=_readonly(geom_indices, dtype=np.int32),
        link_name=tuple(link_names),
        region_membership=_readonly(memberships, dtype=bool),
    )


def _horizontal_fov_inclusion(azimuth_deg: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    lower, upper = map(float, interval)
    span = upper - lower
    if not math.isfinite(span) or span <= 0.0 or span > 360.0 + 1.0e-10:
        raise ValueError("horizontal FOV span must lie in (0,360]")
    if span >= 360.0 - 1.0e-10:
        return np.ones(len(azimuth_deg), dtype=bool)
    return np.mod(azimuth_deg - lower, 360.0) <= span + 1.0e-10


@dataclass(frozen=True)
class StaticOrientationScore:
    pose: SensorPose
    witness_digest: str
    witness_count: int
    nominal_witness_count: int
    self_occluded_witness_count: int
    zero_nominal_fail_closed: bool
    nominal_fraction: float
    visible_fraction: float
    self_occlusion_fraction: float
    calf_visibility_fraction: float
    rear_limb_visibility_fraction: float
    nominal_mask: np.ndarray
    visible_mask: np.ndarray
    self_occluded_mask: np.ndarray

    def to_serializable(self) -> dict[str, object]:
        return {
            "calf_visibility_fraction": self.calf_visibility_fraction,
            "nominal_fraction": self.nominal_fraction,
            "nominal_witness_count": self.nominal_witness_count,
            "pose": self.pose.to_serializable(),
            "rear_limb_visibility_fraction": self.rear_limb_visibility_fraction,
            "self_occlusion_fraction": self.self_occlusion_fraction,
            "self_occluded_witness_count": self.self_occluded_witness_count,
            "visible_fraction": self.visible_fraction,
            "witness_count": self.witness_count,
            "witness_digest": self.witness_digest,
            "zero_nominal_fail_closed": self.zero_nominal_fail_closed,
        }


def score_static_orientation(
    pose: SensorPose,
    protected_primitives: Sequence[range_core.RobotPrimitive],
    witnesses: SurfaceWitnessSet,
    *,
    horizontal_fov_deg: tuple[float, float] = DEFAULT_HORIZONTAL_FOV_DEG,
    vertical_fov_deg: tuple[float, float] = DEFAULT_VERTICAL_FOV_DEG,
    near_m: float = DEFAULT_NEAR_M,
    far_m: float = DEFAULT_FAR_M,
    emitter_exempt_geom_indices: Sequence[int] = (),
) -> StaticOrientationScore:
    """Score one pose using only static nominal protected-surface visibility."""

    near, far = float(near_m), float(far_m)
    if not 0.0 <= near < far or not math.isfinite(far):
        raise ValueError("range contract requires 0 <= near_m < far_m")
    lower_vertical, upper_vertical = map(float, vertical_fov_deg)
    if not -90.0 <= lower_vertical < upper_vertical <= 90.0:
        raise ValueError("vertical FOV must be ordered within [-90,90]")
    origin = np.asarray(pose.translation_body_xyz_m, dtype=np.float64)
    delta = np.asarray(witnesses.points_body_xyz_m) - origin[None, :]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= _TOL):
        raise ValueError("a protected-surface witness coincides with the sensor origin")
    directions = delta / distance[:, None]
    rotation = range_core.rotation_matrix_wxyz(pose.quaternion_body_wxyz)
    local = directions @ rotation
    azimuth = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
    elevation = np.degrees(np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1])))
    nominal = (
        _horizontal_fov_inclusion(azimuth, horizontal_fov_deg)
        & (elevation >= lower_vertical - 1.0e-10)
        & (elevation <= upper_vertical + 1.0e-10)
        & (distance >= near - range_core.DISTANCE_TIE_ATOL_M)
        & (distance <= far + range_core.DISTANCE_TIE_ATOL_M)
    )
    exempt = {int(item) for item in emitter_exempt_geom_indices}
    occluders = tuple(
        sorted(
            (item for item in protected_primitives if int(item.geom_index) not in exempt),
            key=lambda row: (row.geom_index, row.identity),
        )
    )
    closest = np.full(witnesses.count, np.inf, dtype=np.float64)
    winner_geom = np.full(witnesses.count, -1, dtype=np.int32)
    for primitive in occluders:
        candidate = range_core.ray_robot_primitive_distances(origin, directions, primitive)
        better = candidate < closest - range_core.DISTANCE_TIE_ATOL_M
        closest[better] = candidate[better]
        winner_geom[better] = int(primitive.geom_index)
    target_reached = (
        winner_geom == np.asarray(witnesses.geom_index)
    ) & (np.abs(closest - distance) <= 2.0e-9)
    visible = nominal & target_reached
    self_occluded = nominal & np.isfinite(closest) & ~target_reached
    nominal_count = int(nominal.sum())
    self_occluded_count = int(self_occluded.sum())
    zero_nominal_fail_closed = nominal_count == 0
    self_occlusion_fraction = (
        1.0 if zero_nominal_fail_closed else float(self_occluded_count / nominal_count)
    )
    calf_mask = np.asarray(witnesses.region_membership)[:, BODY_REGION_IDS.index(CALVES)]
    rear_mask = np.asarray(witnesses.region_membership)[:, BODY_REGION_IDS.index(REAR_LIMBS)]
    return StaticOrientationScore(
        pose=pose,
        witness_digest=witnesses.content_digest,
        witness_count=witnesses.count,
        nominal_witness_count=nominal_count,
        self_occluded_witness_count=self_occluded_count,
        zero_nominal_fail_closed=zero_nominal_fail_closed,
        nominal_fraction=_fraction(nominal),
        visible_fraction=_fraction(visible),
        self_occlusion_fraction=self_occlusion_fraction,
        calf_visibility_fraction=_fraction(visible[calf_mask]),
        rear_limb_visibility_fraction=_fraction(visible[rear_mask]),
        nominal_mask=_readonly(nominal, dtype=bool),
        visible_mask=_readonly(visible, dtype=bool),
        self_occluded_mask=_readonly(self_occluded, dtype=bool),
    )


@dataclass(frozen=True)
class MountOrientationSelection:
    mount_id: str
    selected: StaticOrientationScore
    candidates: tuple[StaticOrientationScore, ...]
    selection_rule: str

    def to_serializable(self) -> dict[str, object]:
        return {
            "candidates": [item.to_serializable() for item in self.candidates],
            "mount_id": self.mount_id,
            "selected_orientation_id": self.selected.pose.orientation_id,
            "selected_pose": self.selected.pose.to_serializable(),
            "selection_rule": self.selection_rule,
        }


def score_and_select_mount_orientations(
    hardpoints: Sequence[MountHardpoint],
    protected_primitives: Sequence[range_core.RobotPrimitive],
    *,
    witnesses: SurfaceWitnessSet | None = None,
    horizontal_fov_deg: tuple[float, float] = DEFAULT_HORIZONTAL_FOV_DEG,
    vertical_fov_deg: tuple[float, float] = DEFAULT_VERTICAL_FOV_DEG,
    near_m: float = DEFAULT_NEAR_M,
    far_m: float = DEFAULT_FAR_M,
    head_emitter_exempt_geom_indices: Sequence[int] = (1, 2),
) -> tuple[MountOrientationSelection, ...]:
    """Choose each supplemental orientation by the frozen static-only tuple."""

    hardpoint_rows = tuple(hardpoints)
    if tuple(item.mount_id for item in hardpoint_rows) != MOUNT_IDS:
        raise ValueError("hardpoints must be exactly the frozen four mounts in order")
    witness_rows = witnesses or sample_static_protected_surfaces(protected_primitives)
    output: list[MountOrientationSelection] = []
    rule = (
        "MAX_VISIBLE_THEN_MIN_SELF_OCCLUSION_THEN_MAX_CALF_THEN_"
        "MAX_REAR_LIMB_THEN_FIXED_ORIENTATION_ID"
    )
    orientation_rank = {identity: index for index, identity in enumerate(SUPPLEMENTAL_ORIENTATION_IDS)}
    orientation_rank[STOCK] = 0
    for hardpoint in hardpoint_rows:
        scores = tuple(
            score_static_orientation(
                pose,
                protected_primitives,
                witness_rows,
                horizontal_fov_deg=horizontal_fov_deg,
                vertical_fov_deg=vertical_fov_deg,
                near_m=near_m,
                far_m=far_m,
                emitter_exempt_geom_indices=(
                    head_emitter_exempt_geom_indices if hardpoint.mount_id == HEAD_STOCK else ()
                ),
            )
            for pose in orientation_library(hardpoint)
        )
        selected = min(
            scores,
            key=lambda item: (
                -item.visible_fraction,
                item.self_occlusion_fraction,
                -item.calf_visibility_fraction,
                -item.rear_limb_visibility_fraction,
                orientation_rank[item.pose.orientation_id],
            ),
        )
        output.append(MountOrientationSelection(hardpoint.mount_id, selected, scores, rule))
    return tuple(output)


@dataclass(frozen=True)
class LayoutCandidate:
    layout_id: str
    mount_ids: tuple[str, ...]
    ordinal: int

    @property
    def origin_count(self) -> int:
        return len(self.mount_ids)

    def to_serializable(self) -> dict[str, object]:
        return {
            "layout_id": self.layout_id,
            "mount_ids": list(self.mount_ids),
            "ordinal": self.ordinal,
            "origin_count": self.origin_count,
        }


def enumerate_layout_candidates(
    available_mount_ids: Sequence[str] = MOUNT_IDS,
) -> tuple[LayoutCandidate, ...]:
    """Enumerate exactly the three head-plus-one pairs and three triples."""

    if tuple(available_mount_ids) != MOUNT_IDS:
        raise ValueError("available mounts must be exactly the frozen four-member library")
    return tuple(
        LayoutCandidate(layout_id, mount_ids, ordinal)
        for ordinal, (layout_id, mount_ids) in enumerate(LAYOUT_DEFINITIONS)
    )


@dataclass(frozen=True)
class ScanPhase:
    phase_digest_sha256: str
    azimuth_phase_cycles: float
    vertical_phase_cycles: float
    contract_digest_sha256: str
    transition_identity: str
    mount_identity: str

    @property
    def horizontal_phase_cycles(self) -> float:
        """Contract-facing name for the sensor's azimuth phase."""

        return self.azimuth_phase_cycles

    def to_serializable(self) -> dict[str, object]:
        return {
            "azimuth_phase_cycles": self.azimuth_phase_cycles,
            "horizontal_phase_cycles": self.horizontal_phase_cycles,
            "contract_digest_sha256": self.contract_digest_sha256,
            "mount_identity": self.mount_identity,
            "phase_digest_sha256": self.phase_digest_sha256,
            "phase_namespace": PHASE_NAMESPACE,
            "transition_identity": self.transition_identity,
            "vertical_phase_cycles": self.vertical_phase_cycles,
        }


def derive_scan_phase(
    contract_digest_sha256: str,
    transition_identity: str,
    mount_identity: str,
) -> ScanPhase:
    """Derive independent deterministic phases from contract, transition, mount."""

    contract_digest = _validate_sha256(contract_digest_sha256, "contract_digest_sha256")
    if not transition_identity or not mount_identity:
        raise ValueError("transition_identity and mount_identity must be nonempty")
    if mount_identity not in MOUNT_IDS:
        raise ValueError("mount_identity is not in the frozen mount library")
    payload = (
        PHASE_NAMESPACE.encode("utf-8")
        + b"\x00"
        + bytes.fromhex(contract_digest)
        + b"\x00"
        + transition_identity.encode("utf-8")
        + b"\x00"
        + mount_identity.encode("utf-8")
    )
    digest_bytes = hashlib.sha256(payload).digest()
    denominator = float(1 << 64)
    return ScanPhase(
        phase_digest_sha256=digest_bytes.hex(),
        azimuth_phase_cycles=int.from_bytes(digest_bytes[:8], "big") / denominator,
        vertical_phase_cycles=int.from_bytes(digest_bytes[8:16], "big") / denominator,
        contract_digest_sha256=contract_digest,
        transition_identity=transition_identity,
        mount_identity=mount_identity,
    )


@dataclass(frozen=True)
class OriginSupportEvidence:
    """One origin's already-reduced evidence for aligned protected witnesses."""

    mount_identity: str
    witness_identity: tuple[str, ...]
    event_time_s: np.ndarray
    observation_support: np.ndarray
    nominal_fov_inclusion: np.ndarray
    direct_visibility_after_self_occlusion: np.ndarray
    self_return: np.ndarray
    acquisition_timestamp_s: np.ndarray
    ray_or_point_index: np.ndarray
    point_range_m: np.ndarray
    minimum_clearance_m: np.ndarray

    def __post_init__(self) -> None:
        if self.mount_identity not in MOUNT_IDS:
            raise ValueError("mount_identity is not frozen")
        count = len(self.witness_identity)
        event = np.asarray(self.event_time_s, dtype=np.float64)
        support = np.asarray(self.observation_support, dtype=bool)
        nominal = np.asarray(self.nominal_fov_inclusion, dtype=bool)
        direct = np.asarray(self.direct_visibility_after_self_occlusion, dtype=bool)
        self_return = np.asarray(self.self_return, dtype=bool)
        timestamp = np.asarray(self.acquisition_timestamp_s, dtype=np.float64)
        index = np.asarray(self.ray_or_point_index)
        point_range = np.asarray(self.point_range_m, dtype=np.float64)
        clearance = np.asarray(self.minimum_clearance_m, dtype=np.float64)
        for name, value in (
            ("event_time_s", event),
            ("observation_support", support),
            ("nominal_fov_inclusion", nominal),
            ("direct_visibility_after_self_occlusion", direct),
            ("self_return", self_return),
            ("acquisition_timestamp_s", timestamp),
            ("ray_or_point_index", index),
            ("point_range_m", point_range),
            ("minimum_clearance_m", clearance),
        ):
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape [N]")
        if not np.isfinite(event).all() or np.any(event < 0.0):
            raise ValueError("event_time_s must be finite and nonnegative")
        if np.any(support & (~nominal | ~direct | self_return)):
            raise ValueError("support requires nominal direct non-self evidence")
        if np.any(support & (~np.isfinite(timestamp) | ~np.isfinite(point_range) | ~np.isfinite(clearance))):
            raise ValueError("supported rows require finite timestamp, range, and clearance")
        if np.any(support & (index < 0)):
            raise ValueError("supported rows require a nonnegative ray/point index")
        self_audit = ~support & self_return
        if np.any(
            self_audit
            & (~np.isfinite(timestamp) | ~np.isfinite(point_range) | (index < 0))
        ):
            raise ValueError("self-return audit rows require finite timestamp/range and an index")
        if np.any(self_audit & np.isfinite(clearance)):
            raise ValueError("robot-self returns cannot carry environment clearance")
        absent = ~support & ~self_return
        if (
            np.any(np.isfinite(timestamp[absent]))
            or np.any(np.isfinite(point_range[absent]))
            or np.any(np.isfinite(clearance[absent]))
            or np.any(index[absent] != -1)
        ):
            raise ValueError("rows without support or a self return must use null provenance")


@dataclass(frozen=True)
class MultiOriginSupportUnion:
    origin_identities: tuple[str, ...]
    witness_identity: tuple[str, ...]
    event_time_s: np.ndarray
    observation_support: np.ndarray
    unsupported: np.ndarray
    support_count: np.ndarray
    supporting_origin_mask: np.ndarray
    nominal_fov_by_origin: np.ndarray
    direct_visibility_by_origin: np.ndarray
    self_return_by_origin: np.ndarray
    acquisition_timestamp_by_origin_s: np.ndarray
    ray_or_point_index_by_origin: np.ndarray
    point_range_by_origin_m: np.ndarray
    minimum_clearance_by_origin_m: np.ndarray
    support_origin_index: np.ndarray
    support_mount_identity: tuple[str | None, ...]
    support_timestamp_s: np.ndarray
    support_point_age_s: np.ndarray
    support_ray_or_point_index: np.ndarray
    support_point_range_m: np.ndarray
    minimum_clearance_m: np.ndarray
    minimum_clearance_origin_index: np.ndarray
    minimum_clearance_mount_identity: tuple[str | None, ...]

    def __post_init__(self) -> None:
        count = len(self.witness_identity)
        origins = len(self.origin_identities)
        support = np.asarray(self.observation_support, dtype=bool)
        unsupported = np.asarray(self.unsupported, dtype=bool)
        matrix = np.asarray(self.supporting_origin_mask, dtype=bool)
        if support.shape != (count,) or unsupported.shape != (count,):
            raise ValueError("union support vectors must have shape [N]")
        if matrix.shape != (origins, count):
            raise ValueError("supporting_origin_mask must have shape [origins,N]")
        if not np.array_equal(unsupported, ~support):
            raise ValueError("unsupported must be the exact complement of union support")
        if not np.array_equal(support, np.any(matrix, axis=0)):
            raise ValueError("a witness is supported iff at least one origin supports it")

    def to_serializable(self) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for index, identity in enumerate(self.witness_identity):
            rows.append(
                {
                    "event_time_s": float(self.event_time_s[index]),
                    "minimum_clearance_m": _finite_float_or_none(self.minimum_clearance_m[index]),
                    "minimum_clearance_mount_identity": self.minimum_clearance_mount_identity[index],
                    "observation_support": bool(self.observation_support[index]),
                    "support_count": int(self.support_count[index]),
                    "support_mount_identity": self.support_mount_identity[index],
                    "support_point_age_s": _finite_float_or_none(self.support_point_age_s[index]),
                    "support_point_range_m": _finite_float_or_none(self.support_point_range_m[index]),
                    "support_ray_or_point_index": int(self.support_ray_or_point_index[index]),
                    "support_timestamp_s": _finite_float_or_none(self.support_timestamp_s[index]),
                    "supporting_mount_identities": [
                        self.origin_identities[origin]
                        for origin in np.flatnonzero(self.supporting_origin_mask[:, index])
                    ],
                    "unsupported": bool(self.unsupported[index]),
                    "witness_identity": identity,
                }
            )
        return {
            "origin_identities": list(self.origin_identities),
            "rows": rows,
            "schema": "minimum_multi_origin_support_union_v1",
        }


def union_multi_origin_support(
    origin_evidence: Sequence[OriginSupportEvidence],
) -> MultiOriginSupportUnion:
    """Union aligned origin evidence and retain deterministic support provenance.

    Robot-self returns cannot support a witness.  For provenance, the latest
    acquisition at/before the witness event wins; if none exists, the earliest
    later acquisition wins.  Timestamp ties use origin order then ray/point
    index.  Clearance is the minimum over supporting origins with origin order
    as the exact tie breaker.
    """

    rows = tuple(origin_evidence)
    if not rows:
        raise ValueError("origin_evidence must be nonempty")
    identities = tuple(row.mount_identity for row in rows)
    if len(set(identities)) != len(identities):
        raise ValueError("origin mount identities must be unique")
    expected_witness = rows[0].witness_identity
    expected_event = np.asarray(rows[0].event_time_s, dtype=np.float64)
    for row in rows[1:]:
        if row.witness_identity != expected_witness or not np.array_equal(row.event_time_s, expected_event):
            raise ValueError("all origins must bind identical witness identities and event times")
    support_matrix = np.asarray([row.observation_support for row in rows], dtype=bool)
    nominal_matrix = np.asarray([row.nominal_fov_inclusion for row in rows], dtype=bool)
    direct_matrix = np.asarray(
        [row.direct_visibility_after_self_occlusion for row in rows], dtype=bool
    )
    self_matrix = np.asarray([row.self_return for row in rows], dtype=bool)
    timestamps = np.asarray([row.acquisition_timestamp_s for row in rows], dtype=np.float64)
    indices = np.asarray([row.ray_or_point_index for row in rows], dtype=np.int64)
    ranges = np.asarray([row.point_range_m for row in rows], dtype=np.float64)
    clearances = np.asarray([row.minimum_clearance_m for row in rows], dtype=np.float64)
    support = np.any(support_matrix, axis=0)
    count = support_matrix.sum(axis=0, dtype=np.int16)
    witness_count = len(expected_witness)
    provenance_origin = np.full(witness_count, -1, dtype=np.int16)
    provenance_timestamp = np.full(witness_count, np.nan, dtype=np.float64)
    provenance_index = np.full(witness_count, -1, dtype=np.int64)
    provenance_range = np.full(witness_count, np.nan, dtype=np.float64)
    minimum = np.full(witness_count, np.nan, dtype=np.float64)
    minimum_origin = np.full(witness_count, -1, dtype=np.int16)
    for witness_index in np.flatnonzero(support):
        candidates = np.flatnonzero(support_matrix[:, witness_index])
        past = candidates[timestamps[candidates, witness_index] <= expected_event[witness_index] + _TOL]
        if len(past):
            selected = min(
                past,
                key=lambda origin: (
                    -float(timestamps[origin, witness_index]),
                    int(origin),
                    int(indices[origin, witness_index]),
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda origin: (
                    float(timestamps[origin, witness_index]),
                    int(origin),
                    int(indices[origin, witness_index]),
                ),
            )
        provenance_origin[witness_index] = int(selected)
        provenance_timestamp[witness_index] = timestamps[selected, witness_index]
        provenance_index[witness_index] = indices[selected, witness_index]
        provenance_range[witness_index] = ranges[selected, witness_index]
        clearance_origin = min(
            candidates,
            key=lambda origin: (float(clearances[origin, witness_index]), int(origin)),
        )
        minimum[witness_index] = clearances[clearance_origin, witness_index]
        minimum_origin[witness_index] = int(clearance_origin)
    support_mount = tuple(
        None if index < 0 else identities[int(index)] for index in provenance_origin
    )
    clearance_mount = tuple(
        None if index < 0 else identities[int(index)] for index in minimum_origin
    )
    return MultiOriginSupportUnion(
        origin_identities=identities,
        witness_identity=expected_witness,
        event_time_s=_readonly(expected_event, dtype=np.float64),
        observation_support=_readonly(support, dtype=bool),
        unsupported=_readonly(~support, dtype=bool),
        support_count=_readonly(count, dtype=np.int16),
        supporting_origin_mask=_readonly(support_matrix, dtype=bool),
        nominal_fov_by_origin=_readonly(nominal_matrix, dtype=bool),
        direct_visibility_by_origin=_readonly(direct_matrix, dtype=bool),
        self_return_by_origin=_readonly(self_matrix, dtype=bool),
        acquisition_timestamp_by_origin_s=_readonly(timestamps, dtype=np.float64),
        ray_or_point_index_by_origin=_readonly(indices, dtype=np.int64),
        point_range_by_origin_m=_readonly(ranges, dtype=np.float64),
        minimum_clearance_by_origin_m=_readonly(clearances, dtype=np.float64),
        support_origin_index=_readonly(provenance_origin, dtype=np.int16),
        support_mount_identity=support_mount,
        support_timestamp_s=_readonly(provenance_timestamp, dtype=np.float64),
        support_point_age_s=_readonly(expected_event - provenance_timestamp, dtype=np.float64),
        support_ray_or_point_index=_readonly(provenance_index, dtype=np.int64),
        support_point_range_m=_readonly(provenance_range, dtype=np.float64),
        minimum_clearance_m=_readonly(minimum, dtype=np.float64),
        minimum_clearance_origin_index=_readonly(minimum_origin, dtype=np.int16),
        minimum_clearance_mount_identity=clearance_mount,
    )


@dataclass(frozen=True)
class LayoutSupportMetrics:
    layout_id: str
    role: str
    transitions: int
    physics_steps: int
    protected_links: int
    minimum_body_region_support: float
    p5_transition_support: float
    rear_limb_support: float
    calf_support: float
    overall_mean_support: float
    self_occlusion_fraction: float
    body_region_support: Mapping[str, float]

    @property
    def origin_count(self) -> int:
        return len(LAYOUT_MOUNTS[self.layout_id])

    def to_serializable(self) -> dict[str, object]:
        return {
            "body_region_support": dict(self.body_region_support),
            "calf_support": self.calf_support,
            "layout_id": self.layout_id,
            "minimum_body_region_support": self.minimum_body_region_support,
            "origin_count": self.origin_count,
            "overall_mean_support": self.overall_mean_support,
            "p5_transition_support": self.p5_transition_support,
            "physics_steps": self.physics_steps,
            "protected_links": self.protected_links,
            "rear_limb_support": self.rear_limb_support,
            "role": self.role,
            "self_occlusion_fraction": self.self_occlusion_fraction,
            "transitions": self.transitions,
        }


def _link_region_matrix(link_names: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((len(link_names), len(BODY_REGION_IDS)), dtype=bool)
    for link_index, name in enumerate(link_names):
        memberships = set(body_region_membership(str(name)))
        for region_index, region in enumerate(BODY_REGION_IDS):
            matrix[link_index, region_index] = region in memberships
    if not np.all(matrix.any(axis=0)):
        missing = [BODY_REGION_IDS[index] for index in np.flatnonzero(~matrix.any(axis=0))]
        raise ValueError(f"protected links do not cover every frozen body region: {missing}")
    return matrix


def compute_layout_support_metrics(
    layout_id: str,
    origin_mount_ids: Sequence[str],
    support_by_origin: np.ndarray,
    nominal_fov_by_origin: np.ndarray,
    self_occlusion_by_origin: np.ndarray,
    protected_link_names: Sequence[str],
    *,
    role: str,
) -> LayoutSupportMetrics:
    """Reduce complete training transition x 50-step x link support arrays.

    No outcome or contact-label argument exists.  ``support_by_origin`` and
    ``nominal_fov_by_origin`` and ``self_occlusion_by_origin`` must have shape
    ``[origins,T,50,links]``.
    The empirical fifth percentile uses NumPy's deterministic linear method.
    """

    if role != "training":
        raise ValueError("layout selection metrics are authorized only for the training role")
    if layout_id not in LAYOUT_MOUNTS:
        raise ValueError("layout_id is not one of the six frozen candidates")
    if tuple(origin_mount_ids) != LAYOUT_MOUNTS[layout_id]:
        raise ValueError("origin_mount_ids do not exactly match the frozen layout order")
    support = np.asarray(support_by_origin, dtype=bool)
    nominal = np.asarray(nominal_fov_by_origin, dtype=bool)
    self_occlusion = np.asarray(self_occlusion_by_origin, dtype=bool)
    if support.ndim != 4 or support.shape != nominal.shape or support.shape != self_occlusion.shape:
        raise ValueError(
            "support, nominal FOV, and self-occlusion must share shape [origins,T,50,links]"
        )
    origins, transitions, steps, links = support.shape
    if origins != len(origin_mount_ids) or transitions <= 0 or steps != PHYSICS_STEPS_PER_TICK:
        raise ValueError("support must contain every origin, transition, and exactly 50 physics steps")
    if links != len(protected_link_names) or links <= 0:
        raise ValueError("protected-link axis does not match protected_link_names")
    if np.any(support & ~nominal):
        raise ValueError("direct support requires nominal FOV/range eligibility")
    union_support = np.any(support, axis=0)
    nominal_union = np.any(nominal, axis=0)
    nominal_denominator = int(nominal_union.sum())
    if nominal_denominator == 0:
        raise ValueError("layout self-occlusion has zero nominal-support denominator")
    # Every nominally eligible origin must explicitly be robot-self-blocked;
    # ineligible origins are irrelevant to the all-eligible quantifier.
    every_eligible_self_blocked = np.all(~nominal | self_occlusion, axis=0)
    union_self_occlusion = nominal_union & ~union_support & every_eligible_self_blocked
    membership = _link_region_matrix(protected_link_names)
    region_support: dict[str, float] = {}
    for region_index, region in enumerate(BODY_REGION_IDS):
        link_mask = membership[:, region_index]
        region_support[region] = _fraction(union_support[:, :, link_mask])
    transition_support = union_support.reshape(transitions, -1).mean(axis=1)
    return LayoutSupportMetrics(
        layout_id=layout_id,
        role=role,
        transitions=transitions,
        physics_steps=steps,
        protected_links=links,
        minimum_body_region_support=min(region_support.values()),
        p5_transition_support=float(np.percentile(transition_support, 5.0, method="linear")),
        rear_limb_support=region_support[REAR_LIMBS],
        calf_support=region_support[CALVES],
        overall_mean_support=_fraction(union_support),
        self_occlusion_fraction=float(union_self_occlusion.sum() / nominal_denominator),
        body_region_support=region_support,
    )


def select_layout_lexicographically(
    candidates: Sequence[LayoutSupportMetrics],
    *,
    origin_count: int,
) -> LayoutSupportMetrics:
    """Select a pair or triple by the frozen training-only tuple."""

    if origin_count not in (2, 3):
        raise ValueError("origin_count must be 2 or 3")
    expected_ids = tuple(
        layout_id for layout_id, mounts in LAYOUT_DEFINITIONS if len(mounts) == origin_count
    )
    rows = tuple(candidates)
    if tuple(row.layout_id for row in rows) != expected_ids:
        raise ValueError("candidates must be every frozen layout of the requested size in order")
    if any(row.role != "training" or row.origin_count != origin_count for row in rows):
        raise ValueError("layout candidates must be training-role metrics of the requested size")
    fixed_rank = {layout_id: index for index, layout_id in enumerate(expected_ids)}
    return min(
        rows,
        key=lambda row: (
            -row.minimum_body_region_support,
            -row.p5_transition_support,
            -row.rear_limb_support,
            -row.calf_support,
            -row.overall_mean_support,
            row.self_occlusion_fraction,
            fixed_rank[row.layout_id],
        ),
    )


def select_pair_and_triple_layouts(
    candidates: Sequence[LayoutSupportMetrics],
) -> dict[str, LayoutSupportMetrics]:
    """Select the best two-origin and three-origin layouts separately."""

    rows = tuple(candidates)
    if tuple(row.layout_id for row in rows) != LAYOUT_IDS:
        raise ValueError("candidates must be all six frozen layouts in order")
    return {
        "pair": select_layout_lexicographically(rows[:3], origin_count=2),
        "triple": select_layout_lexicographically(rows[3:], origin_count=3),
    }


def sensor_contact_decision(
    minimum_clearance_m: float,
    observation_support: bool,
    threshold_m: float,
) -> bool:
    """Fail closed on unsupported evidence and make exact ties contact-positive."""

    clearance = float(minimum_clearance_m)
    threshold = float(threshold_m)
    if not math.isfinite(threshold):
        raise ValueError("threshold_m must be finite")
    return bool(not observation_support or not math.isfinite(clearance) or clearance <= threshold)


def predicted_safe_action_count(predicted_contact: Sequence[bool]) -> int:
    contact = np.asarray(predicted_contact, dtype=bool)
    if contact.ndim != 1:
        raise ValueError("predicted_contact must be one dimensional")
    return int((~contact).sum())


def admit_current_action(current_predicted_contact: bool, predicted_safe_next_count: int) -> bool:
    count = int(predicted_safe_next_count)
    if count < 0:
        raise ValueError("predicted_safe_next_count cannot be negative")
    return bool(not current_predicted_contact and count >= 1)


def build_mount_library_receipt(
    trunk_primitives: Sequence[range_core.RobotPrimitive] | None = None,
    protected_primitives: Sequence[range_core.RobotPrimitive] | None = None,
    *,
    mechanical_clearance_m: float = DEFAULT_MECHANICAL_CLEARANCE_M,
    housing_full_extents_xyz_m: Sequence[float] = DEFAULT_HOUSING_FULL_EXTENTS_XYZ_M,
) -> dict[str, object]:
    """Build the canonical label-free mount/orientation library receipt."""

    if (trunk_primitives is None) != (protected_primitives is None):
        raise ValueError("trunk_primitives and protected_primitives must be supplied together")
    if trunk_primitives is None:
        nominal = frozen_static_nominal_go2_primitives()
        trunk_rows: Sequence[range_core.RobotPrimitive] = (nominal[0],)
        protected_rows: Sequence[range_core.RobotPrimitive] = nominal
        geometry_source: dict[str, object] = {
            "base_pose": {
                "position_xyz_m": list(FROZEN_NOMINAL_BASE_POSITION_XYZ_M),
                "quaternion_wxyz": list(FROZEN_NOMINAL_BASE_QUATERNION_WXYZ),
            },
            "genesis_go2_urdf_package_relative_path": GENESIS_GO2_URDF_PACKAGE_RELATIVE_PATH,
            "genesis_go2_urdf_sha256": GENESIS_GO2_URDF_SHA256,
            "nominal_stance_rad": list(FROZEN_NOMINAL_STANCE_RAD),
            "nominal_stance_source_lines": list(NOMINAL_STANCE_SOURCE_LINES),
            "nominal_stance_source_path": NOMINAL_STANCE_SOURCE_PATH,
            "nominal_stance_source_sha256": NOMINAL_STANCE_SOURCE_SHA256,
            "primitive_count": len(nominal),
            "protected_link_count": 13,
            "protected_links": sorted({row.link_name for row in nominal}),
            "source_class": "FROZEN_STATIC_NOMINAL_GO2_URDF_GEOMETRY",
        }
    else:
        assert protected_primitives is not None
        trunk_rows = tuple(trunk_primitives)
        protected_rows = tuple(protected_primitives)
        geometry_source = {
            "primitive_count": len(protected_rows),
            "source_class": "CALLER_SUPPLIED_STATIC_GEOMETRY",
        }

    hardpoints = derive_mount_candidates(
        trunk_rows,
        mechanical_clearance_m=mechanical_clearance_m,
        housing_full_extents_xyz_m=housing_full_extents_xyz_m,
    )
    housing_half = 0.5 * np.asarray(housing_full_extents_xyz_m, dtype=np.float64)
    housing_clearance_validation: list[dict[str, object]] = []
    for hardpoint in hardpoints:
        if hardpoint.mount_id == HEAD_STOCK:
            housing_clearance_validation.append(
                {
                    "mount_id": hardpoint.mount_id,
                    "rule": "STOCK_INTEGRATED_HEAD_ASSEMBLY_NOT_A_SUPPLEMENTAL_ENVELOPE",
                    "minimum_protected_geometry_clearance_m": None,
                    "responsible_protected_primitive": None,
                    "required_clearance_m": None,
                    "pass": True,
                }
            )
            continue
        housing_box = range_core.OrientedBox(
            identity=f"body_axis_aligned_sensor_housing:{hardpoint.mount_id}",
            center_xyz_m=hardpoint.translation_body_xyz_m,
            half_extents_xyz_m=tuple(float(value) for value in housing_half),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        witnesses = [
            range_core.primitive_obb_closest_witness(primitive, housing_box)
            for primitive in protected_rows
        ]
        responsible = min(
            witnesses,
            key=lambda value: (
                float(value.signed_clearance_m),
                str(value.primitive_identity),
            ),
        )
        clearance_pass = bool(
            float(responsible.signed_clearance_m)
            >= float(mechanical_clearance_m) - _TOL
        )
        housing_clearance_validation.append(
            {
                "mount_id": hardpoint.mount_id,
                "rule": "BODY_AXIS_ALIGNED_FULL_HOUSING_OUTSIDE_ALL_27_NOMINAL_PROTECTED_PRIMITIVES",
                "minimum_protected_geometry_clearance_m": float(
                    responsible.signed_clearance_m
                ),
                "responsible_protected_primitive": str(
                    responsible.primitive_identity
                ),
                "required_clearance_m": float(mechanical_clearance_m),
                "protected_primitive_count": len(protected_rows),
                "pass": clearance_pass,
            }
        )
        if not clearance_pass:
            raise ValueError(
                f"{hardpoint.mount_id} housing violates protected-geometry clearance"
            )
    witnesses = sample_static_protected_surfaces(protected_rows)
    selections = score_and_select_mount_orientations(
        hardpoints, protected_rows, witnesses=witnesses
    )
    payload: dict[str, object] = {
        "body_region_ids": list(BODY_REGION_IDS),
        "contact_outcomes_used": False,
        "experiment_id": EXPERIMENT_ID,
        "housing_full_extents_xyz_m": [float(item) for item in housing_full_extents_xyz_m],
        "housing_clearance_validation": housing_clearance_validation,
        "housing_occlusion_frame": "BODY_AXIS_ALIGNED_TRUNK_FRAME_INDEPENDENT_OF_OPTICAL_RAY_FRAME",
        "layouts": [item.to_serializable() for item in enumerate_layout_candidates()],
        "mechanical_clearance_m": float(mechanical_clearance_m),
        "mount_candidates": [item.to_serializable() for item in hardpoints],
        "nominal_geometry_source": geometry_source,
        "orientation_library_ids": list(SUPPLEMENTAL_ORIENTATION_IDS),
        "orientation_selections": [item.to_serializable() for item in selections],
        "outcome_fields_read": [],
        "pass": True,
        "schema": "minimum_multi_origin_mount_library_v1",
        "selection_role": "STATIC_NOMINAL_URDF_GEOMETRY_ONLY",
        "trunk_envelope": derive_trunk_envelope(trunk_rows).to_serializable(),
        "witness_count": witnesses.count,
        "witness_digest": witnesses.content_digest,
    }
    payload["content_digest"] = range_core.canonical_digest(payload)
    return payload


def _origin_evidence(
    mount: str,
    support: Sequence[bool],
    *,
    timestamp: Sequence[float] | None = None,
    self_return: Sequence[bool] | None = None,
    clearance: Sequence[float] | None = None,
) -> OriginSupportEvidence:
    """Internal deterministic fixture constructor."""

    supported = np.asarray(support, dtype=bool)
    count = len(supported)
    times = np.asarray(timestamp if timestamp is not None else np.arange(count) * 0.002)
    self_mask = np.asarray(self_return if self_return is not None else np.zeros(count, bool))
    physical_return = supported | self_mask
    acquisition = np.where(physical_return, times, np.nan)
    indices = np.where(physical_return, np.arange(count), -1)
    ranges = np.where(physical_return, 1.0 + 0.1 * np.arange(count), np.nan)
    clearances = np.where(
        supported,
        np.asarray(clearance if clearance is not None else 0.2 + 0.01 * np.arange(count)),
        np.nan,
    )
    return OriginSupportEvidence(
        mount_identity=mount,
        witness_identity=tuple(f"witness-{index}" for index in range(count)),
        event_time_s=np.arange(count, dtype=np.float64) * 0.002,
        observation_support=supported,
        nominal_fov_inclusion=np.ones(count, dtype=bool),
        direct_visibility_after_self_occlusion=supported,
        self_return=self_mask,
        acquisition_timestamp_s=acquisition,
        ray_or_point_index=indices,
        point_range_m=ranges,
        minimum_clearance_m=clearances,
    )


def _evaluate_synthetic_fixtures() -> dict[str, object]:
    """Execute the named no-corpus synthetic fixture panel once."""

    def primitive(
        identity: str,
        kind: str,
        data: tuple[float, ...],
        position: tuple[float, float, float],
        geom_index: int,
        link_name: str,
    ) -> range_core.RobotPrimitive:
        return range_core.RobotPrimitive(
            identity=identity,
            kind=kind,
            data=data,
            position_xyz_m=position,
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            geom_index=geom_index,
            link_index=geom_index,
            link_name=link_name,
        )

    trunk = primitive("trunk", "box", (0.4, 0.2, 0.2), (0.0, 0.0, 0.2), 0, "base")
    front = primitive("FL_thigh", "sphere", (0.05,), (0.3, 0.15, 0.1), 1, "FL_thigh")
    rear = primitive("RL_thigh", "sphere", (0.05,), (-0.3, 0.15, 0.1), 2, "RL_thigh")
    calf = primitive("RR_calf", "capsule", (0.04, 0.2), (-0.3, -0.15, -0.05), 3, "RR_calf")
    robot = (trunk, front, rear, calf)
    clear = range_core.closest_points_to_scene(np.asarray(((5.0, 5.0, 5.0),)), robot)

    def touching(primitive_row: range_core.RobotPrimitive, center: tuple[float, float, float], half: tuple[float, float, float]) -> float:
        box = range_core.OrientedBox("fixture", center, half, object_index=0)
        return float(range_core.primitive_obb_closest_witness(primitive_row, box).signed_clearance_m)

    contact_query_specs = {
        "front trunk contact": (trunk, (0.25, 0.0, 0.2), (0.05, 0.1, 0.1)),
        "side trunk contact": (trunk, (0.0, 0.15, 0.2), (0.2, 0.05, 0.1)),
        "rear trunk contact": (trunk, (-0.25, 0.0, 0.2), (0.05, 0.1, 0.1)),
        "front-limb contact": (front, (0.4, 0.15, 0.1), (0.05, 0.05, 0.05)),
        "rear-limb contact": (rear, (-0.4, 0.15, 0.1), (0.05, 0.05, 0.05)),
        "calf contact": (calf, (-0.3, -0.24, -0.05), (0.05, 0.05, 0.05)),
    }
    contact_query_clearance = {
        name: touching(primitive_row, center, half)
        for name, (primitive_row, center, half) in contact_query_specs.items()
    }

    fixture_rows: dict[str, dict[str, object]] = {
        "clear full-body sweep": {
            "minimum_clearance_m": [float(item) for item in clear.minimum_clearance_m],
            "pass": bool(np.all(clear.minimum_clearance_m > 1.0)),
        },
        **{
            name: {
                "signed_clearance_m": contact_query_clearance[name],
                "pass": abs(contact_query_clearance[name]) <= 1.0e-12,
            }
            for name in contact_query_specs
        },
    }

    wall = range_core.OrientedBox("wall", (2.0, 0.0, 0.0), (0.02, 0.2, 0.2), object_index=0)
    blocker = primitive("self_blocker", "sphere", (0.2,), (1.0, 0.0, 0.0), 9, "base")
    target = (1.98, 0.0, 0.0)
    blocked = range_core.continuum_target_visibility(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), target,
        horizontal_fov_deg=(-180.0, 180.0), vertical_fov_deg=(-90.0, 90.0),
        near_m=0.05, far_m=10.0, environment_boxes=(wall,), robot_primitives=(blocker,),
        target_object_identity="wall",
    )
    observed = range_core.continuum_target_visibility(
        (0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0), target,
        horizontal_fov_deg=(-180.0, 180.0), vertical_fov_deg=(-90.0, 90.0),
        near_m=0.05, far_m=10.0, environment_boxes=(wall,), robot_primitives=(blocker,),
        target_object_identity="wall",
    )
    occlusion_origins = (
        _origin_evidence(HEAD_STOCK, (False,), self_return=(True,)),
        _origin_evidence(REAR_TOP_TRUNK, (True,)),
    )
    occlusion_union = union_multi_origin_support(occlusion_origins)
    fixture_rows["one origin occluded another observes"] = {
        "blocked_first_hit": blocked.first_hit_identity,
        "pass": bool(blocked.self_occluded and observed.target_object_observable and occlusion_union.observation_support[0]),
        "support_mount": occlusion_union.support_mount_identity[0],
    }
    flank_origins = (
        _origin_evidence(LEFT_UPPER_FLANK, (True, False)),
        _origin_evidence(RIGHT_UPPER_FLANK, (False, True)),
    )
    flank_union = union_multi_origin_support(flank_origins)
    fixture_rows["complementary L/R flank"] = {
        "pass": bool(np.all(flank_union.observation_support) and np.all(flank_union.support_count == 1))
    }
    head_rear_origins = (
        _origin_evidence(HEAD_STOCK, (True, False)),
        _origin_evidence(REAR_TOP_TRUNK, (False, True)),
    )
    head_rear_union = union_multi_origin_support(head_rear_origins)
    fixture_rows["complementary head/rear"] = {
        "pass": bool(np.all(head_rear_union.observation_support))
    }

    near_box = range_core.OrientedBox("near", (0.03, 0.0, 0.0), (0.01, 0.05, 0.05), object_index=1)
    near_hit = range_core.first_hits(
        (0.0, 0.0, 0.0), ((1.0, 0.0, 0.0),), environment_boxes=(near_box,),
        near_m=0.05, far_m=10.0,
    )
    fixture_rows["near blind"] = {
        "pass": bool(near_hit.near_blind[0] and not near_hit.valid_return[0])
    }
    sparse = range_core.spherical_directions_fru(np.radians((-2.0, 2.0)), np.asarray((0.0,)))
    narrow = range_core.OrientedBox("narrow", (2.0, 0.0, 0.0), (0.02, 0.02, 0.05), object_index=2)
    sparse_hit = range_core.first_hits(
        (0.0, 0.0, 0.0), sparse, environment_boxes=(narrow,), near_m=0.05, far_m=10.0
    )
    continuum = range_core.continuum_target_visibility(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.98, 0.0, 0.0),
        horizontal_fov_deg=(-180.0, 180.0), vertical_fov_deg=(-90.0, 90.0),
        near_m=0.05, far_m=10.0, environment_boxes=(narrow,), target_object_identity="narrow",
    )
    fixture_rows["between scan samples"] = {
        "pass": bool(np.all(sparse_hit.no_hit) and continuum.target_object_observable)
    }
    overlap_origins = (
        _origin_evidence(HEAD_STOCK, (True,), timestamp=(0.0,), clearance=(0.2,)),
        _origin_evidence(REAR_TOP_TRUNK, (True,), timestamp=(0.0,), clearance=(0.1,)),
    )
    overlap = union_multi_origin_support(overlap_origins)
    fixture_rows["synchronized overlap"] = {
        "clearance_mount": overlap.minimum_clearance_mount_identity[0],
        "pass": bool(
            overlap.support_count[0] == 2
            and overlap.support_mount_identity[0] == HEAD_STOCK
            and overlap.minimum_clearance_mount_identity[0] == REAR_TOP_TRUNK
        ),
        "support_mount": overlap.support_mount_identity[0],
    }
    zero_digest = "0" * 64
    head_phase = derive_scan_phase(zero_digest, "fixture-transition", HEAD_STOCK)
    rear_phase = derive_scan_phase(zero_digest, "fixture-transition", REAR_TOP_TRUNK)
    fixture_rows["independent phases"] = {
        "head": head_phase.phase_digest_sha256,
        "pass": bool(
            head_phase.phase_digest_sha256 != rear_phase.phase_digest_sha256
            and head_phase == derive_scan_phase(zero_digest, "fixture-transition", HEAD_STOCK)
        ),
        "rear": rear_phase.phase_digest_sha256,
    }
    one_safe = predicted_safe_action_count((True, False, True))
    zero_safe = predicted_safe_action_count((True, True))
    fixture_rows["one safe successor"] = {"pass": one_safe == 1, "safe_count": one_safe}
    fixture_rows["zero safe successors"] = {
        "pass": bool(zero_safe == 0 and not admit_current_action(False, zero_safe)),
        "safe_count": zero_safe,
    }
    fixture_rows["threshold tie"] = {
        "pass": sensor_contact_decision(0.1, True, 0.1)
    }
    fixture_rows["abstention"] = {
        "pass": not admit_current_action(False, 0)
    }
    from lewm.safety import body_centric_range_coverage_metrics_v1 as metrics_core

    h3_rows = (
        {"action_index": 4, "h3_progress_m": 1.0, "h3_heading_improvement_rad": 0.0},
        {"action_index": 2, "h3_progress_m": 0.97, "h3_heading_improvement_rad": 0.5},
        {"action_index": 1, "h3_progress_m": 0.97, "h3_heading_improvement_rad": 0.5},
        {"action_index": 3, "h3_progress_m": 0.969, "h3_heading_improvement_rad": 100.0},
    )
    h3_order = [h3_rows[index]["action_index"] for index in metrics_core.h3_route_order(h3_rows)]
    fixture_rows["H3"] = {
        "action_order": h3_order,
        "pass": h3_order == [1, 2, 4, 3],
    }

    def primitive_raw(value: range_core.RobotPrimitive) -> dict[str, object]:
        return {
            "identity": value.identity,
            "kind": value.kind,
            "data": list(value.data),
            "position_xyz_m": list(value.position_xyz_m),
            "quaternion_wxyz": list(value.quaternion_wxyz),
            "geom_index": value.geom_index,
            "link_index": value.link_index,
            "link_name": value.link_name,
        }

    def box_raw(value: range_core.OrientedBox) -> dict[str, object]:
        return {
            "identity": value.identity,
            "center_xyz_m": list(value.center_xyz_m),
            "half_extents_xyz_m": list(value.half_extents_xyz_m),
            "quaternion_wxyz": list(value.quaternion_wxyz),
            "object_index": value.object_index,
        }

    def finite_rows(values: np.ndarray) -> list[float | int | bool | None]:
        array = np.asarray(values)
        output: list[float | int | bool | None] = []
        for value in array.reshape(-1):
            if np.issubdtype(array.dtype, np.bool_):
                output.append(bool(value))
            elif np.issubdtype(array.dtype, np.integer):
                output.append(int(value))
            else:
                number = float(value)
                output.append(number if math.isfinite(number) else None)
        return output

    def origin_raw(value: OriginSupportEvidence) -> dict[str, object]:
        return {
            "mount_identity": value.mount_identity,
            "witness_identity": list(value.witness_identity),
            "event_time_s": finite_rows(value.event_time_s),
            "observation_support": finite_rows(value.observation_support),
            "nominal_fov_inclusion": finite_rows(value.nominal_fov_inclusion),
            "direct_visibility_after_self_occlusion": finite_rows(
                value.direct_visibility_after_self_occlusion
            ),
            "self_return": finite_rows(value.self_return),
            "acquisition_timestamp_s": finite_rows(value.acquisition_timestamp_s),
            "ray_or_point_index": finite_rows(value.ray_or_point_index),
            "point_range_m": finite_rows(value.point_range_m),
            "minimum_clearance_m": finite_rows(value.minimum_clearance_m),
        }

    raw_fixture_evidence: dict[str, object] = {
        "schema": "minimum_multi_origin_body_range_coverage_fixture_raw_evidence_v1",
        "complete_raw_ray_queries": {
            "near_blind": {
                "origin_world_xyz_m": [0.0, 0.0, 0.0],
                "directions_world": [[1.0, 0.0, 0.0]],
                "timestamps_s": [0.0],
                "near_m": 0.05,
                "far_m": 10.0,
                "environment_boxes": [box_raw(near_box)],
                "robot_primitives": [],
                "first_hits": near_hit.to_serializable(),
            },
            "between_scan_samples": {
                "origin_world_xyz_m": [0.0, 0.0, 0.0],
                "directions_world": sparse.tolist(),
                "timestamps_s": [0.0] * len(sparse),
                "near_m": 0.05,
                "far_m": 10.0,
                "environment_boxes": [box_raw(narrow)],
                "robot_primitives": [],
                "first_hits": sparse_hit.to_serializable(),
                "continuum_query": {
                    "target_world_xyz_m": [1.98, 0.0, 0.0],
                    "sensor_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "horizontal_fov_deg": [-180.0, 180.0],
                    "vertical_fov_deg": [-90.0, 90.0],
                    "target_object_identity": "narrow",
                    "result": continuum.to_serializable(),
                },
            },
            "one_origin_occluded_another_observes": {
                "target_world_xyz_m": list(target),
                "environment_boxes": [box_raw(wall)],
                "robot_primitives": [primitive_raw(blocker)],
                "blocked_origin_world_xyz_m": [0.0, 0.0, 0.0],
                "observed_origin_world_xyz_m": [0.0, 1.0, 0.0],
                "blocked_direction_world": [1.0, 0.0, 0.0],
                "observed_direction_world": (
                    (
                        (np.asarray(target, np.float64) - np.asarray((0.0, 1.0, 0.0)))
                        / np.linalg.norm(
                            np.asarray(target, np.float64)
                            - np.asarray((0.0, 1.0, 0.0))
                        )
                    ).tolist()
                ),
                "timestamps_s": [0.0, 0.0],
                "near_m": 0.05,
                "far_m": 10.0,
                "blocked_result": blocked.to_serializable(),
                "observed_result": observed.to_serializable(),
            },
        },
        "complete_reduced_origin_evidence": {
            "one_origin_occluded_another_observes": [
                origin_raw(value) for value in occlusion_origins
            ],
            "complementary_left_right_flank": [
                origin_raw(value) for value in flank_origins
            ],
            "complementary_head_rear": [
                origin_raw(value) for value in head_rear_origins
            ],
            "synchronized_scan_overlap": [
                origin_raw(value) for value in overlap_origins
            ],
        },
        "reconstructible_noncloud_inputs": {
            "robot_primitives": [primitive_raw(value) for value in robot],
            "clear_query_points_world_xyz_m": [[5.0, 5.0, 5.0]],
            "clear_query_result_m": [
                float(value) for value in clear.minimum_clearance_m
            ],
            "contact_queries": {
                name: {
                    "primitive_identity": primitive_row.identity,
                    "environment_box": box_raw(
                        range_core.OrientedBox(
                            f"fixture:{name}", center, half, object_index=0
                        )
                    ),
                    "signed_clearance_m": contact_query_clearance[name],
                }
                for name, (primitive_row, center, half) in contact_query_specs.items()
            },
            "scan_phase_inputs": {
                "contract_digest_sha256": zero_digest,
                "transition_identity": "fixture-transition",
                "mount_ids": [HEAD_STOCK, REAR_TOP_TRUNK],
                "head_result": head_phase.to_serializable(),
                "rear_result": rear_phase.to_serializable(),
            },
            "safe_successor_inputs": {
                "one_safe": [True, False, True],
                "zero_safe": [True, True],
            },
            "threshold_tie_input": {
                "minimum_clearance_m": 0.1,
                "observation_support": True,
                "threshold_m": 0.1,
            },
            "h3_rows": list(h3_rows),
        },
        "raw_fixture_cloud_policy": (
            "every finite ray used by a ray-based fixture is retained above with "
            "origin, direction, timestamp, primitive inputs, physical first hit and "
            "range-filter status; non-cloud fixtures retain complete reconstructible inputs"
        ),
    }
    raw_fixture_evidence["content_digest"] = range_core.canonical_digest(
        raw_fixture_evidence
    )
    return {
        "fixtures": fixture_rows,
        "raw_fixture_evidence": raw_fixture_evidence,
        "pass": all(bool(row["pass"]) for row in fixture_rows.values()),
        "schema": "minimum_multi_origin_body_range_coverage_fixture_v1",
    }


def synthetic_fixture_receipt() -> dict[str, object]:
    """Return the complete named synthetic panel with byte-identity witness."""

    first = _evaluate_synthetic_fixtures()
    second = _evaluate_synthetic_fixtures()
    byte_identical = range_core.canonical_json_bytes(first) == range_core.canonical_json_bytes(second)
    first["requirements"] = {"byte-identical receipt regeneration": {"pass": byte_identical}}
    first["pass"] = bool(first["pass"] and byte_identical)
    first["content_digest"] = range_core.canonical_digest(first)
    return first


def run_fixtures() -> dict[str, object]:
    """Evaluator-facing compatibility alias for the frozen synthetic gate."""

    return synthetic_fixture_receipt()


__all__ = [
    "BODY_REGION_IDS",
    "CALVES",
    "DEFAULT_HOUSING_FULL_EXTENTS_XYZ_M",
    "DEFAULT_MECHANICAL_CLEARANCE_M",
    "EXPERIMENT_ID",
    "FRONT_LIMBS",
    "FROZEN_NOMINAL_BASE_POSITION_XYZ_M",
    "FROZEN_NOMINAL_BASE_QUATERNION_WXYZ",
    "FROZEN_NOMINAL_STANCE_RAD",
    "GENESIS_GO2_URDF_PACKAGE_RELATIVE_PATH",
    "GENESIS_GO2_URDF_SHA256",
    "HEAD_STOCK",
    "HEAD_STOCK_RPY_RAD",
    "HEAD_STOCK_TRANSLATION_BODY_XYZ_M",
    "HIPS_AND_THIGHS",
    "INVERTED",
    "INWARD_DOWNWARD",
    "LAYOUT_DEFINITIONS",
    "LAYOUT_IDS",
    "LAYOUT_MOUNTS",
    "LEFT_UPPER_FLANK",
    "LEVEL",
    "MOUNT_IDS",
    "NOMINAL_STANCE_SOURCE_LINES",
    "NOMINAL_STANCE_SOURCE_PATH",
    "NOMINAL_STANCE_SOURCE_SHA256",
    "OUTWARD_DOWNWARD",
    "PHASE_NAMESPACE",
    "PHYSICS_STEPS_PER_TICK",
    "REAR_LIMBS",
    "REAR_TOP_TRUNK",
    "RIGHT_UPPER_FLANK",
    "SCHEMA_VERSION",
    "STOCK",
    "SUPPLEMENTAL_ORIENTATION_IDS",
    "TRUNK",
    "AxisAlignedEnvelope",
    "LayoutCandidate",
    "LayoutSupportMetrics",
    "MountHardpoint",
    "MountOrientationSelection",
    "MultiOriginSupportUnion",
    "OriginSupportEvidence",
    "ScanPhase",
    "SensorPose",
    "StaticOrientationScore",
    "SurfaceWitnessSet",
    "admit_current_action",
    "body_region_membership",
    "build_mount_library_receipt",
    "compute_layout_support_metrics",
    "derive_mount_candidates",
    "derive_scan_phase",
    "derive_trunk_envelope",
    "enumerate_layout_candidates",
    "frozen_static_nominal_go2_primitives",
    "orientation_library",
    "predicted_safe_action_count",
    "quaternion_from_pole",
    "quaternion_to_rpy_wxyz",
    "robot_primitive_aabb",
    "rpy_quaternion_wxyz",
    "run_fixtures",
    "sample_static_protected_surfaces",
    "score_and_select_mount_orientations",
    "score_static_orientation",
    "select_layout_lexicographically",
    "select_pair_and_triple_layouts",
    "sensor_contact_decision",
    "synthetic_fixture_receipt",
    "union_multi_origin_support",
]
