"""Two-resolution G3 exact-physical equivalence candidate V2.

The production projection is revision-bound to live physical memory. The audit
oracle below independently derives the preregistered supports with exact
rational arithmetic and never consumes ``FIXED_PROFILE_V2`` offset state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from lewm.benchmarks import go2_g3_exact_physical_equivalence as v1
from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    SOURCE_CELL_SIZE_M,
    SOURCE_SHAPE,
)
from lewm.planning.geometry_contract import GeometryContract
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalLabel,
    PhysicalMemoryConfig,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    FIXED_PROFILE_V2,
    FOOTPRINT_RADIUS_M,
    FREE_SUPPORT_COUNT,
    FREE_SUPPORT_SHA256,
    OCCUPIED_SUPPORT_COUNT,
    OCCUPIED_SUPPORT_SHA256,
    PHYSICAL_CELL_SIZE_M,
    PLANNING_CONNECTIVITY,
    PROFILE_SHA256,
    TwoResolutionConfigurationPlannerV2,
    TwoResolutionConfigurationProjectionV2,
    _execution_block_receipt_sha256,
    assert_fixed_profile_integrity,
)
from lewm.planning.zero_inflation_exact_physical_adapter_v1 import (
    ZeroInflationExactPhysicalAdapterV1,
    exact_physical_cells_content_sha256,
)
from lewm_worlds.manifest import SceneManifest


Cell = tuple[int, int]
V4_EVIDENCE_SCHEMA = "lewm_go2_observable_camera_ray_evidence_v4"
V4_EVIDENCE_MODULE_PATH = "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py"
V4_EVIDENCE_MODULE_SHA256 = (
    "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85"
)
V4_EVIDENCE_CONTRACT_PATH = (
    "docs/lewm_go2_observable_camera_ray_evidence_v4_contract_2026-07-12.md"
)
V4_EVIDENCE_CONTRACT_SHA256 = (
    "0a17cc94056ef5c53d2a96266cb21a5500eb3a9ea13e62f02f296b97455bcdee"
)
GOVERNING_DESIGN_PATH = "docs/lewm_go2_g3_two_resolution_v2_design_contract_2026-07-13.md"
GOVERNING_DESIGN_SHA256 = (
    "a82de141575efe9e12f0deea05477f558439d87bcb1af3bc36e0d377a36c95b1"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_origin(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must contain two finite numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain two finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain two finite numbers")
        result.append(number)
    return result[0], result[1]


def _shape(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must contain two positive integers")
    result = (
        _nonnegative_int(value[0], f"{name}[0]"),
        _nonnegative_int(value[1], f"{name}[1]"),
    )
    if 0 in result:
        raise ValueError(f"{name} must contain two positive integers")
    return result


def _canonical_cells(value: object, name: str) -> tuple[Cell, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    cells: list[Cell] = []
    for raw in value:
        if not isinstance(raw, tuple) or len(raw) != 2 or any(
            isinstance(coordinate, bool) or not isinstance(coordinate, int)
            for coordinate in raw
        ):
            raise TypeError(f"{name} must contain two-integer tuples")
        cells.append((raw[0], raw[1]))
    result = tuple(sorted(set(cells)))
    if tuple(cells) != result:
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _independent_kernel_common() -> dict[str, object]:
    return {
        "configuration_cell_size_m": 0.1,
        "footprint_radius_m": 0.47,
        "inclusive_boundary": True,
        "physical_cell_size_m": 0.05,
        "physical_index_rule": "(2*cx+dx,2*cy+dy)",
        "shared_origin_cell_boundaries": True,
    }


def independent_exact_rational_supports() -> tuple[tuple[Cell, ...], tuple[Cell, ...]]:
    """Derive 316/276 from Fractions, independently of production profile state."""

    radius_squared = Fraction(47, 100) ** 2
    half_physical = Fraction(1, 40)
    free: list[Cell] = []
    occupied: list[Cell] = []
    for dx in range(-12, 13):
        for dy in range(-12, 13):
            delta_x = Fraction(2 * dx - 1, 40)
            delta_y = Fraction(2 * dy - 1, 40)
            near_x = max(abs(delta_x) - half_physical, Fraction(0))
            near_y = max(abs(delta_y) - half_physical, Fraction(0))
            if near_x**2 + near_y**2 <= radius_squared:
                free.append((dx, dy))
            if delta_x**2 + delta_y**2 <= radius_squared:
                occupied.append((dx, dy))
    free_offsets = tuple(sorted(free))
    occupied_offsets = tuple(sorted(occupied))
    free_core = {
        **_independent_kernel_common(),
        "offsets": [[x, y] for x, y in free_offsets],
        "schema": "lewm_g3_v2_cross_grid_free_closed_square_intersection_kernel_v1",
    }
    occupied_core = {
        **_independent_kernel_common(),
        "offsets": [[x, y] for x, y in occupied_offsets],
        "schema": "lewm_g3_v2_cross_grid_occupied_center_inside_disc_kernel_v1",
    }
    projection_core = {
        "configuration_cell_size_m": 0.1,
        "footprint_radius_m": 0.47,
        "free_support_count": 316,
        "free_support_sha256": _sha256(free_core),
        "occupied_precedes_free": True,
        "occupied_support_count": 276,
        "occupied_support_sha256": _sha256(occupied_core),
        "otherwise": "unknown",
        "out_of_domain_support": "occupied",
        "physical_cell_size_m": 0.05,
        "physical_shape_per_configuration_cell": [2, 2],
        "schema": "lewm_g3_v2_two_resolution_configuration_projection_v1",
        "shared_origin_cell_boundaries": True,
    }
    if (
        len(free_offsets) != 316
        or len(occupied_offsets) != 276
        or _sha256(free_core) != FREE_SUPPORT_SHA256
        or _sha256(occupied_core) != OCCUPIED_SUPPORT_SHA256
        or _sha256(projection_core) != PROFILE_SHA256
    ):
        raise AssertionError("independent exact-rational G3 V2 supports changed")
    return free_offsets, occupied_offsets


def assert_v2_profile_integrity() -> None:
    assert_fixed_profile_integrity()
    independent_free, independent_occupied = independent_exact_rational_supports()
    if (
        SOURCE_SHAPE != (128, 128)
        or not math.isclose(
            SOURCE_CELL_SIZE_M,
            PHYSICAL_CELL_SIZE_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or FIXED_PROFILE_V2.content_sha256 != PROFILE_SHA256
        or FIXED_PROFILE_V2.free_support_offsets != independent_free
        or FIXED_PROFILE_V2.occupied_support_offsets != independent_occupied
        or FIXED_PROFILE_V2.to_dict()["production_promotion_authorized"] is not False
    ):
        raise AssertionError("G3 V2/V4 preregistered profile binding changed")


def profiled_physical_geometry(geometry: GeometryContract) -> GeometryContract:
    """Return a 0.05 m physical-label view of the frozen V1 geometry."""

    assert_v2_profile_integrity()
    configuration = geometry.configuration_space
    if (
        not math.isclose(
            configuration.online_cell_size_m,
            CONFIGURATION_CELL_SIZE_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            configuration.body_inflation_radius_m,
            FOOTPRINT_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(configuration.connectivity) != PLANNING_CONNECTIVITY
        or bool(configuration.allow_diagonal_corner_cutting)
    ):
        raise ValueError("G3 V2 requires the frozen 0.10 m planning geometry")
    return replace(
        geometry,
        configuration_space=replace(
            configuration,
            online_cell_size_m=PHYSICAL_CELL_SIZE_M,
        ),
    )


def registered_two_resolution_lattices(
    manifest: SceneManifest,
    geometry: GeometryContract,
) -> tuple[tuple[float, float], tuple[int, int], tuple[int, int]]:
    profiled_physical_geometry(geometry)
    (x_lo, y_lo), (x_hi, y_hi) = manifest.world_bounds_xy_m
    pad = FOOTPRINT_RADIUS_M + CONFIGURATION_CELL_SIZE_M
    origin = float(x_lo) - pad, float(y_lo) - pad
    configuration_shape = (
        int(math.ceil((float(x_hi) - float(x_lo) + 2.0 * pad) / 0.10)),
        int(math.ceil((float(y_hi) - float(y_lo) + 2.0 * pad) / 0.10)),
    )
    return origin, (2 * configuration_shape[0], 2 * configuration_shape[1]), configuration_shape


def _cross_grid_view(
    labels: np.ndarray,
    offset: Cell,
    configuration_shape: tuple[int, int],
) -> np.ndarray:
    result = np.full(
        configuration_shape,
        int(PhysicalLabel.OCCUPIED),
        dtype=labels.dtype,
    )
    px = 2 * np.arange(configuration_shape[0], dtype=np.int64) + int(offset[0])
    py = 2 * np.arange(configuration_shape[1], dtype=np.int64) + int(offset[1])
    valid_x = np.flatnonzero((px >= 0) & (px < labels.shape[0]))
    valid_y = np.flatnonzero((py >= 0) & (py < labels.shape[1]))
    if valid_x.size and valid_y.size:
        result[np.ix_(valid_x, valid_y)] = labels[
            np.ix_(px[valid_x], py[valid_y])
        ]
    return result


def independent_cross_grid_configuration_labels(
    physical_labels: np.ndarray,
    *,
    configuration_shape: Sequence[int],
) -> np.ndarray:
    """Exact-rational, production-state-independent occupied-first oracle."""

    free_support, occupied_support = independent_exact_rational_supports()
    labels = np.asarray(physical_labels)
    shape = _shape(configuration_shape, "configuration_shape")
    if labels.shape != (2 * shape[0], 2 * shape[1]) or not np.isin(
        labels,
        tuple(int(value) for value in PhysicalLabel),
    ).all():
        raise ValueError("G3 V2 physical label lattice changed")
    all_free = np.ones(shape, dtype=np.bool_)
    any_occupied = np.zeros(shape, dtype=np.bool_)
    for offset in free_support:
        all_free &= _cross_grid_view(labels, offset, shape) == int(PhysicalLabel.FREE)
    for offset in occupied_support:
        any_occupied |= _cross_grid_view(labels, offset, shape) == int(
            PhysicalLabel.OCCUPIED
        )
    result = np.full(shape, int(PhysicalLabel.UNKNOWN), dtype=np.uint8)
    result[all_free] = int(PhysicalLabel.FREE)
    result[any_occupied] = int(PhysicalLabel.OCCUPIED)
    return result


def _physical_mapping(labels: np.ndarray) -> dict[Cell, PhysicalLabel]:
    return {
        (int(x), int(y)): PhysicalLabel(int(labels[x, y]))
        for x, y in np.ndindex(labels.shape)
    }


def _build_projected_snapshot(
    manifest: SceneManifest,
    physical_labels: np.ndarray,
    *,
    origin_xy_m: tuple[float, float],
    configuration_shape: tuple[int, int],
) -> tuple[
    RevisionedPhysicalMemory,
    object,
    TwoResolutionConfigurationPlannerV2,
]:
    assert_v2_profile_integrity()
    physical_frame = MapFrameIdentity(
        session_id=f"{manifest.scene_id}:g3-v2:physical",
        origin_xy_m=origin_xy_m,
        cell_size_m=PHYSICAL_CELL_SIZE_M,
        frame_id="g3_v2_physical_evidence",
    )
    configuration_frame = MapFrameIdentity(
        session_id=f"{manifest.scene_id}:g3-v2:configuration",
        origin_xy_m=origin_xy_m,
        cell_size_m=CONFIGURATION_CELL_SIZE_M,
        frame_id="g3_v2_configuration_planning",
    )
    camera_sha256 = hashlib.sha256(b"g3-v2-exact-equivalence-camera").hexdigest()
    memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=physical_frame,
            planning_connectivity=PLANNING_CONNECTIVITY,
            allow_diagonal_corner_cutting=False,
            require_registered_lattice=False,
            physical_projection_contract_sha256=PROFILE_SHA256,
            expected_camera_transform_sha256=camera_sha256,
            promoted_runtime=False,
        )
    )
    mapping = _physical_mapping(physical_labels)
    observation = ObservationIdentity(
        observation_id=f"exact-v2:{manifest.scene_id}",
        payload_sha256=exact_physical_cells_content_sha256(mapping),
        producer_sha256=hashlib.sha256(
            b"g3-v2-zero-inflation-exact-physical-adapter"
        ).hexdigest(),
        authority=EvidenceAuthority.EXACT_PHYSICAL,
    )
    pose = PoseProvenance(
        source=PoseSource.DEPLOYMENT_ODOMETRY,
        frame_id=physical_frame.frame_id,
        mean_xy_yaw=(0.0, 0.0, 0.0),
        covariance_xy_yaw=((0.0, 0.0, 0.0),) * 3,
        timestamp_ns=0,
        synchronization_id=f"exact-v2:{manifest.scene_id}",
        camera_transform_sha256=camera_sha256,
    )
    ZeroInflationExactPhysicalAdapterV1(memory).fuse_cells(
        mapping,
        observation=observation,
        pose=pose,
        label_inflation_radius_m=0.0,
    )
    if (
        memory.learned_observation_ids
        or memory.exact_observation_ids != frozenset({observation.observation_id})
    ):
        raise PermissionError("G3 V2 exact-control evidence authority changed")
    projection = TwoResolutionConfigurationProjectionV2(
        memory,
        configuration_map_frame=configuration_frame,
        physical_shape=physical_labels.shape,
        configuration_shape=configuration_shape,
    )
    snapshot = projection.project()
    return memory, snapshot, TwoResolutionConfigurationPlannerV2(projection)


def _configuration_cell(
    origin_xy_m: Sequence[float],
    point_xy_m: Sequence[float],
) -> Cell:
    return (
        int(math.floor((float(point_xy_m[0]) - float(origin_xy_m[0])) / 0.10)),
        int(math.floor((float(point_xy_m[1]) - float(origin_xy_m[1])) / 0.10)),
    )


@dataclass(frozen=True)
class G3ExactSceneResultV2:
    scene_id: str
    family: str
    physical_lattice_origin_xy_m: tuple[float, float]
    configuration_lattice_origin_xy_m: tuple[float, float]
    physical_lattice_shape: tuple[int, int]
    lattice_shape: tuple[int, int]
    physical_map_frame_sha256: str
    configuration_map_frame_sha256: str
    physical_session_id: str
    configuration_session_id: str
    physical_frame_id: str
    configuration_frame_id: str
    memory_config_sha256: str
    physical_revision: int
    configuration_revision: int
    physical_content_sha256: str
    snapshot_content_sha256: str
    projection_source_sha256: str
    execution_block_receipt_sha256: str
    physical_execution_block_cells: tuple[Cell, ...]
    configuration_execution_block_cells: tuple[Cell, ...]
    profile_sha256: str
    free_support_sha256: str
    occupied_support_sha256: str
    physical_cell_size_m: float
    configuration_cell_size_m: float
    physical_cells_per_configuration_axis: tuple[int, int]
    physical_free_cells: int
    physical_occupied_cells: int
    physical_unknown_cells: int
    snapshot_free_cells: int
    snapshot_occupied_cells: int
    snapshot_unknown_cells: int
    independent_label_mismatch_cells: int
    analytic_free_cells: int
    unsafe_free_cells: int
    conservative_false_reject_cells: int
    strict_binary_label_mismatch_cells: int
    snapshot_component_cells: int
    independent_component_cells: int
    component_mismatch_cells: int
    astar_probe_count: int
    astar_mismatch_count: int
    canonical_component_cells: int
    canonical_component_false_reject_cells: int
    claim_endpoints_retained: int
    beacon_count: int
    physical_evidence_authority: str
    exact_observation_count: int
    learned_observation_count: int
    exact_sim_tainted: bool
    production_promotion_authorized: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be non-empty")
        for name in (
            "physical_map_frame_sha256",
            "configuration_map_frame_sha256",
            "memory_config_sha256",
            "physical_content_sha256",
            "snapshot_content_sha256",
            "projection_source_sha256",
            "execution_block_receipt_sha256",
            "profile_sha256",
            "free_support_sha256",
            "occupied_support_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        physical_origin = _finite_origin(
            self.physical_lattice_origin_xy_m, "physical_lattice_origin_xy_m"
        )
        configuration_origin = _finite_origin(
            self.configuration_lattice_origin_xy_m,
            "configuration_lattice_origin_xy_m",
        )
        physical_shape = _shape(self.physical_lattice_shape, "physical_lattice_shape")
        configuration_shape = _shape(self.lattice_shape, "lattice_shape")
        if physical_shape != (
            2 * configuration_shape[0],
            2 * configuration_shape[1],
        ):
            raise ValueError("scene result lattice ratio changed")
        for name in (
            "physical_revision",
            "configuration_revision",
            "physical_free_cells",
            "physical_occupied_cells",
            "physical_unknown_cells",
            "snapshot_free_cells",
            "snapshot_occupied_cells",
            "snapshot_unknown_cells",
            "independent_label_mismatch_cells",
            "analytic_free_cells",
            "unsafe_free_cells",
            "conservative_false_reject_cells",
            "strict_binary_label_mismatch_cells",
            "snapshot_component_cells",
            "independent_component_cells",
            "component_mismatch_cells",
            "astar_probe_count",
            "astar_mismatch_count",
            "canonical_component_cells",
            "canonical_component_false_reject_cells",
            "claim_endpoints_retained",
            "beacon_count",
            "exact_observation_count",
            "learned_observation_count",
        ):
            _nonnegative_int(getattr(self, name), name)
        if (
            physical_origin != configuration_origin
            or self.physical_revision == 0
            or self.configuration_revision == 0
            or self.physical_map_frame_sha256
            == self.configuration_map_frame_sha256
            or self.physical_session_id == self.configuration_session_id
            or self.physical_frame_id == self.configuration_frame_id
            or not all(
                isinstance(value, str) and value
                for value in (
                    self.physical_session_id,
                    self.configuration_session_id,
                    self.physical_frame_id,
                    self.configuration_frame_id,
                )
            )
            or not math.isclose(self.physical_cell_size_m, 0.05, abs_tol=1e-12)
            or not math.isclose(self.configuration_cell_size_m, 0.10, abs_tol=1e-12)
            or self.physical_cells_per_configuration_axis != (2, 2)
            or self.profile_sha256 != PROFILE_SHA256
            or self.free_support_sha256 != FREE_SUPPORT_SHA256
            or self.occupied_support_sha256 != OCCUPIED_SUPPORT_SHA256
            or self.physical_evidence_authority
            != EvidenceAuthority.EXACT_PHYSICAL.value
            or self.exact_observation_count != 1
            or self.learned_observation_count != 0
            or type(self.exact_sim_tainted) is not bool
            or type(self.production_promotion_authorized) is not bool
            or self.production_promotion_authorized
        ):
            raise ValueError("scene result identity/authority contract changed")
        expected_physical_frame = MapFrameIdentity(
            session_id=self.physical_session_id,
            origin_xy_m=physical_origin,
            cell_size_m=self.physical_cell_size_m,
            frame_id=self.physical_frame_id,
        )
        expected_configuration_frame = MapFrameIdentity(
            session_id=self.configuration_session_id,
            origin_xy_m=configuration_origin,
            cell_size_m=self.configuration_cell_size_m,
            frame_id=self.configuration_frame_id,
        )
        if (
            self.physical_map_frame_sha256
            != expected_physical_frame.content_sha256
            or self.configuration_map_frame_sha256
            != expected_configuration_frame.content_sha256
        ):
            raise ValueError("scene result map-frame hashes do not match identities")
        physical_blocks = _canonical_cells(
            self.physical_execution_block_cells,
            "physical_execution_block_cells",
        )
        configuration_blocks = _canonical_cells(
            self.configuration_execution_block_cells,
            "configuration_execution_block_cells",
        )
        expected_configuration_blocks = tuple(
            sorted(
                {
                    (cell[0] // 2, cell[1] // 2)
                    for cell in physical_blocks
                    if 0 <= cell[0] // 2 < configuration_shape[0]
                    and 0 <= cell[1] // 2 < configuration_shape[1]
                }
            )
        )
        if (
            configuration_blocks != expected_configuration_blocks
            or self.execution_block_receipt_sha256
            != _execution_block_receipt_sha256(
                physical_revision=self.physical_revision,
                physical_cells=physical_blocks,
                configuration_cells=configuration_blocks,
            )
        ):
            raise ValueError("scene result execution-block receipt changed")
        if (
            self.physical_free_cells
            + self.physical_occupied_cells
            + self.physical_unknown_cells
            != physical_shape[0] * physical_shape[1]
            or self.snapshot_free_cells
            + self.snapshot_occupied_cells
            + self.snapshot_unknown_cells
            != configuration_shape[0] * configuration_shape[1]
        ):
            raise ValueError("scene result does not bind complete rasters")
        object.__setattr__(self, "physical_lattice_origin_xy_m", physical_origin)
        object.__setattr__(
            self, "configuration_lattice_origin_xy_m", configuration_origin
        )
        object.__setattr__(self, "physical_lattice_shape", physical_shape)
        object.__setattr__(self, "lattice_shape", configuration_shape)

    @property
    def identity_receipts_complete(self) -> bool:
        return True

    @property
    def complete_physical_raster(self) -> bool:
        return self.physical_unknown_cells == 0

    @property
    def discrete_equivalence_pass(self) -> bool:
        return bool(
            self.independent_label_mismatch_cells == 0
            and self.component_mismatch_cells == 0
            and self.astar_probe_count > 0
            and self.astar_mismatch_count == 0
            and self.exact_sim_tainted
            and self.identity_receipts_complete
        )

    @property
    def conservative_safety_pass(self) -> bool:
        return self.unsafe_free_cells == 0

    @property
    def claim_endpoint_pass(self) -> bool:
        return self.claim_endpoints_retained == self.beacon_count

    @property
    def legacy_strict_binary_equivalence_pass(self) -> bool:
        return self.strict_binary_label_mismatch_cells == 0

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "lattice_identities": {
                "physical": {
                    "session_id": self.physical_session_id,
                    "frame_id": self.physical_frame_id,
                    "origin_xy_m": list(self.physical_lattice_origin_xy_m),
                    "cell_size_m": self.physical_cell_size_m,
                    "shape": list(self.physical_lattice_shape),
                    "map_frame_sha256": self.physical_map_frame_sha256,
                    "revision": self.physical_revision,
                    "content_sha256": self.physical_content_sha256,
                },
                "configuration": {
                    "session_id": self.configuration_session_id,
                    "frame_id": self.configuration_frame_id,
                    "origin_xy_m": list(self.configuration_lattice_origin_xy_m),
                    "cell_size_m": self.configuration_cell_size_m,
                    "shape": list(self.lattice_shape),
                    "map_frame_sha256": self.configuration_map_frame_sha256,
                    "revision": self.configuration_revision,
                    "snapshot_content_sha256": self.snapshot_content_sha256,
                },
            },
            "identity_receipts_complete": self.identity_receipts_complete,
            "complete_physical_raster": self.complete_physical_raster,
            "discrete_equivalence_pass": self.discrete_equivalence_pass,
            "conservative_safety_pass": self.conservative_safety_pass,
            "claim_endpoint_pass": self.claim_endpoint_pass,
            "legacy_strict_binary_equivalence_pass": (
                self.legacy_strict_binary_equivalence_pass
            ),
        }


def evaluate_exact_scene_v2(
    manifest: SceneManifest,
    geometry: GeometryContract,
) -> G3ExactSceneResultV2:
    assert_v2_profile_integrity()
    origin, physical_shape, configuration_shape = registered_two_resolution_lattices(
        manifest,
        geometry,
    )
    physical_geometry = profiled_physical_geometry(geometry)
    physical = v1.exact_closed_square_physical_labels(
        manifest,
        physical_geometry,
        origin_xy_m=origin,
        shape=physical_shape,
    )
    independent = independent_cross_grid_configuration_labels(
        physical,
        configuration_shape=configuration_shape,
    )
    analytic_free = v1.analytic_disc_free_mask(
        manifest,
        geometry,
        origin_xy_m=origin,
        shape=configuration_shape,
    )
    memory, snapshot, planner = _build_projected_snapshot(
        manifest,
        physical,
        origin_xy_m=origin,
        configuration_shape=configuration_shape,
    )
    snapshot_labels = np.full(
        configuration_shape,
        int(PhysicalLabel.UNKNOWN),
        dtype=np.uint8,
    )
    for cell in snapshot.free_cells:
        snapshot_labels[cell] = int(PhysicalLabel.FREE)
    for cell in snapshot.occupied_cells:
        snapshot_labels[cell] = int(PhysicalLabel.OCCUPIED)
    spawn = _configuration_cell(
        origin,
        (float(manifest.spawn.xyz_m[0]), float(manifest.spawn.xyz_m[1])),
    )
    snapshot_component = frozenset(planner.connected_component(snapshot, spawn).cells)
    independent_component = v1._component(
        independent == int(PhysicalLabel.FREE),
        spawn,
    )
    canonical_component = v1._component(analytic_free, spawn)
    astar_probe_count, astar_mismatch_count = v1._astar_distance_checks(
        planner,
        snapshot,
        independent == int(PhysicalLabel.FREE),
        spawn,
    )
    snapshot_free = snapshot_labels == int(PhysicalLabel.FREE)
    strict_binary = np.where(
        analytic_free,
        int(PhysicalLabel.FREE),
        int(PhysicalLabel.OCCUPIED),
    ).astype(np.uint8)
    return G3ExactSceneResultV2(
        scene_id=str(manifest.scene_id),
        family=str(manifest.family),
        physical_lattice_origin_xy_m=origin,
        configuration_lattice_origin_xy_m=origin,
        physical_lattice_shape=physical_shape,
        lattice_shape=configuration_shape,
        physical_map_frame_sha256=snapshot.physical_map_frame_sha256,
        configuration_map_frame_sha256=snapshot.configuration_map_frame_sha256,
        physical_session_id=snapshot.physical_map_frame.session_id,
        configuration_session_id=snapshot.configuration_map_frame.session_id,
        physical_frame_id=snapshot.physical_map_frame.frame_id,
        configuration_frame_id=snapshot.configuration_map_frame.frame_id,
        memory_config_sha256=snapshot.memory_config_sha256,
        physical_revision=snapshot.physical_revision,
        configuration_revision=snapshot.configuration_revision,
        physical_content_sha256=snapshot.physical_content_sha256,
        snapshot_content_sha256=snapshot.content_sha256,
        projection_source_sha256=snapshot.projection_source_sha256,
        execution_block_receipt_sha256=snapshot.execution_block_receipt_sha256,
        physical_execution_block_cells=tuple(
            sorted(snapshot.physical_execution_block_cells)
        ),
        configuration_execution_block_cells=tuple(
            sorted(snapshot.configuration_execution_block_cells)
        ),
        profile_sha256=snapshot.profile_sha256,
        free_support_sha256=snapshot.free_support_sha256,
        occupied_support_sha256=snapshot.occupied_support_sha256,
        physical_cell_size_m=snapshot.physical_map_frame.cell_size_m,
        configuration_cell_size_m=snapshot.configuration_map_frame.cell_size_m,
        physical_cells_per_configuration_axis=(2, 2),
        physical_free_cells=int(np.count_nonzero(physical == int(PhysicalLabel.FREE))),
        physical_occupied_cells=int(
            np.count_nonzero(physical == int(PhysicalLabel.OCCUPIED))
        ),
        physical_unknown_cells=int(
            np.count_nonzero(physical == int(PhysicalLabel.UNKNOWN))
        ),
        snapshot_free_cells=len(snapshot.free_cells),
        snapshot_occupied_cells=len(snapshot.occupied_cells),
        snapshot_unknown_cells=len(snapshot.unknown_cells),
        independent_label_mismatch_cells=int(
            np.count_nonzero(snapshot_labels != independent)
        ),
        analytic_free_cells=int(np.count_nonzero(analytic_free)),
        unsafe_free_cells=int(np.count_nonzero(snapshot_free & ~analytic_free)),
        conservative_false_reject_cells=int(
            np.count_nonzero(analytic_free & ~snapshot_free)
        ),
        strict_binary_label_mismatch_cells=int(
            np.count_nonzero(snapshot_labels != strict_binary)
        ),
        snapshot_component_cells=len(snapshot_component),
        independent_component_cells=len(independent_component),
        component_mismatch_cells=len(snapshot_component ^ independent_component),
        astar_probe_count=astar_probe_count,
        astar_mismatch_count=astar_mismatch_count,
        canonical_component_cells=len(canonical_component),
        canonical_component_false_reject_cells=len(
            canonical_component - snapshot_component
        ),
        claim_endpoints_retained=v1._claim_endpoint_count(
            manifest,
            geometry,
            snapshot_component,
            origin_xy_m=origin,
        ),
        beacon_count=len(manifest.landmarks),
        physical_evidence_authority=EvidenceAuthority.EXACT_PHYSICAL.value,
        exact_observation_count=len(memory.exact_observation_ids),
        learned_observation_count=len(memory.learned_observation_ids),
        exact_sim_tainted=bool(snapshot.exact_sim_tainted),
    )


def summarize_exact_scenes_v2(
    scene_results: Sequence[G3ExactSceneResultV2],
    *,
    source_bindings: Mapping[str, str],
) -> dict[str, object]:
    assert_v2_profile_integrity()
    rows = tuple(scene_results)
    if not rows or len({row.scene_id for row in rows}) != len(rows):
        raise ValueError("G3 V2 summary requires unique nonempty scene results")
    if any(type(row) is not G3ExactSceneResultV2 for row in rows):
        raise TypeError("G3 V2 summary requires exact scene-result types")
    for path, digest in source_bindings.items():
        if not isinstance(path, str) or not path:
            raise ValueError("source binding path must be non-empty")
        _validate_sha256(digest, f"source binding {path}")
    if source_bindings.get(GOVERNING_DESIGN_PATH) != GOVERNING_DESIGN_SHA256:
        raise ValueError("G3 V2 summary is not bound to the governing design")
    identity_rows = {
        (
            row.physical_map_frame_sha256,
            row.configuration_map_frame_sha256,
            row.snapshot_content_sha256,
            row.projection_source_sha256,
        )
        for row in rows
    }
    if len(identity_rows) != len(rows):
        raise ValueError("G3 V2 scene lattice/projection identities must be unique")
    ordered = tuple(sorted(rows, key=lambda row: row.scene_id))
    result: dict[str, object] = {
        "schema": "lewm_go2_g3_exact_physical_equivalence_candidate_v2",
        "status": "candidate_requires_independent_review_no_learned_promotion",
        "governing_design_binding": {
            "path": GOVERNING_DESIGN_PATH,
            "file_sha256": GOVERNING_DESIGN_SHA256,
        },
        "profile": FIXED_PROFILE_V2.to_dict(),
        "v4_source_evidence_binding": {
            "schema": V4_EVIDENCE_SCHEMA,
            "source_shape": list(SOURCE_SHAPE),
            "source_cell_size_m": SOURCE_CELL_SIZE_M,
            "module_path": V4_EVIDENCE_MODULE_PATH,
            "module_file_sha256": V4_EVIDENCE_MODULE_SHA256,
            "contract_path": V4_EVIDENCE_CONTRACT_PATH,
            "contract_file_sha256": V4_EVIDENCE_CONTRACT_SHA256,
        },
        "source_bindings": dict(sorted(source_bindings.items())),
        "scene_count": len(ordered),
        "beacon_count": sum(row.beacon_count for row in ordered),
        "claim_endpoints_retained": sum(
            row.claim_endpoints_retained for row in ordered
        ),
        "discrete_equivalence_scene_count": sum(
            row.discrete_equivalence_pass for row in ordered
        ),
        "conservative_safety_scene_count": sum(
            row.conservative_safety_pass for row in ordered
        ),
        "claim_endpoint_scene_count": sum(row.claim_endpoint_pass for row in ordered),
        "identity_receipt_scene_count": sum(
            row.identity_receipts_complete for row in ordered
        ),
        "complete_physical_raster_scene_count": sum(
            row.complete_physical_raster for row in ordered
        ),
        "legacy_strict_binary_equivalence_scene_count": sum(
            row.legacy_strict_binary_equivalence_pass for row in ordered
        ),
        "unsafe_free_cells": sum(row.unsafe_free_cells for row in ordered),
        "conservative_false_reject_cells": sum(
            row.conservative_false_reject_cells for row in ordered
        ),
        "strict_binary_label_mismatch_cells": sum(
            row.strict_binary_label_mismatch_cells for row in ordered
        ),
        "scenes": [row.to_dict() for row in ordered],
        "production_promotion_authorized": False,
        "learned_projection_implemented": False,
    }
    result["candidate_conservative_equivalence_pass"] = bool(
        len(ordered) == 24
        and result["beacon_count"] == 96
        and result["claim_endpoints_retained"] == 96
        and result["discrete_equivalence_scene_count"] == 24
        and result["conservative_safety_scene_count"] == 24
        and result["claim_endpoint_scene_count"] == 24
        and result["identity_receipt_scene_count"] == 24
        and result["complete_physical_raster_scene_count"] == 24
        and result["unsafe_free_cells"] == 0
    )
    result["candidate_v2_exact_equivalence_pass"] = result[
        "candidate_conservative_equivalence_pass"
    ]
    result["legacy_strict_binary_equivalence_pass"] = bool(
        len(ordered) == 24
        and result["beacon_count"] == 96
        and result["legacy_strict_binary_equivalence_scene_count"] == 24
    )
    result["content_sha256"] = _sha256(result)
    return result


__all__ = [
    "G3ExactSceneResultV2",
    "assert_v2_profile_integrity",
    "evaluate_exact_scene_v2",
    "independent_cross_grid_configuration_labels",
    "independent_exact_rational_supports",
    "profiled_physical_geometry",
    "registered_two_resolution_lattices",
    "summarize_exact_scenes_v2",
]
