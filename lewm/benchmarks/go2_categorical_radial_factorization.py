"""Pure train-only geometry for categorical radial occupancy supervision.

The v1 factorization maps the center of each cell in the registered Go2
Cartesian grid to one radial/angular bin.  It performs no file I/O and has no
held-out-data dependency.  UNKNOWN is the neutral value both for unused polar
bins and for Cartesian cells outside the front-camera support.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lewm.datasets.go2_paired_navigation import (
    DEFAULT_LOCAL_GRID,
    FREE_CLASS,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
)


GEOMETRY_SCHEMA = "lewm_go2_categorical_radial_factorization_v1"
CARTESIAN_SHAPE = (64, 64)
CARTESIAN_CELL_SIZE_M = 0.10
CARTESIAN_FORWARD_MIN_EDGE_M = -1.0
CARTESIAN_LEFT_MIN_EDGE_M = -3.2
HORIZONTAL_FOV_DEG = 78.323
HALF_FOV_DEG = HORIZONTAL_FOV_DEG / 2.0
HALF_FOV_RAD = math.radians(HALF_FOV_DEG)
RADIAL_BIN_COUNT = 64
RADIAL_BIN_SIZE_M = 0.10
RADIAL_RANGE_M = (0.0, RADIAL_BIN_COUNT * RADIAL_BIN_SIZE_M)
ANGULAR_BIN_COUNT = 256
ANGULAR_RANGE_RAD = (-HALF_FOV_RAD, HALF_FOV_RAD)
POLAR_SHAPE = (RADIAL_BIN_COUNT, ANGULAR_BIN_COUNT)
LABEL_CLASSES = (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)


@dataclass(frozen=True)
class RadialFactorization:
    """Frozen array form consumed by the categorical radial model."""

    cartesian_to_polar_flat_indices: np.ndarray
    representable_mask: np.ndarray
    radial_centers_m: np.ndarray
    angular_centers_rad: np.ndarray
    mapping_sha256: str

    def __post_init__(self) -> None:
        flat = np.asarray(self.cartesian_to_polar_flat_indices)
        mask = np.asarray(self.representable_mask)
        radial = np.asarray(self.radial_centers_m)
        angular = np.asarray(self.angular_centers_rad)
        if flat.shape != (CARTESIAN_SHAPE[0] * CARTESIAN_SHAPE[1],):
            raise ValueError("flat Cartesian-to-polar indices must have 4096 entries")
        if not np.issubdtype(flat.dtype, np.integer):
            raise ValueError("flat Cartesian-to-polar indices must be integers")
        if mask.shape != CARTESIAN_SHAPE or mask.dtype != np.bool_:
            raise ValueError("representable mask must be bool[64,64]")
        if radial.shape != (RADIAL_BIN_COUNT,) or not np.isfinite(radial).all():
            raise ValueError("radial centers must contain 64 finite values")
        if angular.shape != (ANGULAR_BIN_COUNT,) or not np.isfinite(angular).all():
            raise ValueError("angular centers must contain 256 finite values")
        if np.any((flat < -1) | (flat >= RADIAL_BIN_COUNT * ANGULAR_BIN_COUNT)):
            raise ValueError("flat Cartesian-to-polar index outside polar grid")
        if not np.array_equal(mask.reshape(-1), flat >= 0):
            raise ValueError("representable mask disagrees with flat polar indices")
        supported = flat[flat >= 0]
        if np.unique(supported).size != supported.size:
            raise ValueError("flat Cartesian-to-polar indices contain a collision")

    @property
    def cartesian_shape(self) -> tuple[int, int]:
        return CARTESIAN_SHAPE

    @property
    def polar_shape(self) -> tuple[int, int]:
        return POLAR_SHAPE


def _validate_registered_cartesian_grid() -> None:
    grid = DEFAULT_LOCAL_GRID
    if (grid.rows, grid.cols) != CARTESIAN_SHAPE:
        raise ValueError("registered Cartesian grid shape changed from v1")
    if not math.isclose(
        grid.cell_size_m, CARTESIAN_CELL_SIZE_M, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("registered Cartesian cell size changed from v1")
    if not math.isclose(
        grid.forward_min_edge_m,
        CARTESIAN_FORWARD_MIN_EDGE_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("registered Cartesian forward origin changed from v1")
    if not math.isclose(
        grid.left_min_edge_m,
        CARTESIAN_LEFT_MIN_EDGE_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("registered Cartesian left origin changed from v1")


def geometry_metadata() -> dict[str, Any]:
    """Return the frozen v1 geometry in artifact-friendly form."""

    _validate_registered_cartesian_grid()
    return {
        "schema": GEOMETRY_SCHEMA,
        "cartesian_shape": list(CARTESIAN_SHAPE),
        "cartesian_cell_size_m": CARTESIAN_CELL_SIZE_M,
        "cartesian_forward_min_edge_m": CARTESIAN_FORWARD_MIN_EDGE_M,
        "cartesian_left_min_edge_m": CARTESIAN_LEFT_MIN_EDGE_M,
        "horizontal_fov_deg": HORIZONTAL_FOV_DEG,
        "half_fov_deg": HALF_FOV_DEG,
        "radial_bin_count": RADIAL_BIN_COUNT,
        "radial_bin_size_m": RADIAL_BIN_SIZE_M,
        "radial_range_m": list(RADIAL_RANGE_M),
        "angular_bin_count": ANGULAR_BIN_COUNT,
        "angular_range_rad": list(ANGULAR_RANGE_RAD),
        "polar_shape": list(POLAR_SHAPE),
        "radial_interval": "left_closed_right_open",
        "angular_interval": "closed_with_positive_edge_in_final_bin",
        "cartesian_sample": "cell_center",
        "angle_convention": "atan2(base_left,base_forward)",
    }


def _cell_center(row: int, col: int) -> tuple[float, float]:
    if not (0 <= int(row) < CARTESIAN_SHAPE[0]):
        raise IndexError(f"Cartesian row outside [0,{CARTESIAN_SHAPE[0]}): {row}")
    if not (0 <= int(col) < CARTESIAN_SHAPE[1]):
        raise IndexError(f"Cartesian column outside [0,{CARTESIAN_SHAPE[1]}): {col}")
    forward_m = CARTESIAN_FORWARD_MIN_EDGE_M + (
        int(row) + 0.5
    ) * CARTESIAN_CELL_SIZE_M
    left_m = CARTESIAN_LEFT_MIN_EDGE_M + (
        int(col) + 0.5
    ) * CARTESIAN_CELL_SIZE_M
    return forward_m, left_m


def cartesian_cell_to_polar_bin(
    row: int, col: int
) -> tuple[int, int] | None:
    """Map one Cartesian cell center to its v1 polar bin, or ``None``.

    The radial range is half-open.  Both camera-FOV edges are included, with
    the positive angular edge assigned to the final angular bin.
    """

    _validate_registered_cartesian_grid()
    forward_m, left_m = _cell_center(row, col)
    radius_m = math.hypot(forward_m, left_m)
    angle_rad = math.atan2(left_m, forward_m)
    if not (RADIAL_RANGE_M[0] <= radius_m < RADIAL_RANGE_M[1]):
        return None
    if not (ANGULAR_RANGE_RAD[0] <= angle_rad <= ANGULAR_RANGE_RAD[1]):
        return None

    radial_bin = int(math.floor(radius_m / RADIAL_BIN_SIZE_M))
    angular_fraction = (
        (angle_rad - ANGULAR_RANGE_RAD[0])
        / (ANGULAR_RANGE_RAD[1] - ANGULAR_RANGE_RAD[0])
    )
    angular_bin = min(
        ANGULAR_BIN_COUNT - 1,
        int(math.floor(angular_fraction * ANGULAR_BIN_COUNT)),
    )
    return radial_bin, angular_bin


def build_cartesian_to_polar_bin_mapping() -> np.ndarray:
    """Build ``[row,col] -> [radial_bin,angular_bin]``; ``[-1,-1]`` is absent."""

    mapping = _expected_mapping_unchecked()
    audit_mapping_injectivity(mapping)
    return mapping


def _expected_mapping_unchecked() -> np.ndarray:
    mapping = np.full((*CARTESIAN_SHAPE, 2), -1, dtype=np.int16)
    for row in range(CARTESIAN_SHAPE[0]):
        for col in range(CARTESIAN_SHAPE[1]):
            polar_bin = cartesian_cell_to_polar_bin(row, col)
            if polar_bin is not None:
                mapping[row, col] = polar_bin
    return mapping


def representable_cartesian_mask(mapping: np.ndarray | None = None) -> np.ndarray:
    """Return cells whose centers are representable by the fixed polar grid."""

    if mapping is None:
        mask = np.zeros(CARTESIAN_SHAPE, dtype=bool)
        for row in range(CARTESIAN_SHAPE[0]):
            for col in range(CARTESIAN_SHAPE[1]):
                mask[row, col] = cartesian_cell_to_polar_bin(row, col) is not None
        return mask
    array = _validate_mapping_array(mapping)
    return np.all(array >= 0, axis=-1)


def _expected_representable_mask() -> np.ndarray:
    return np.all(_expected_mapping_unchecked() >= 0, axis=-1)


def _validate_mapping_array(mapping: np.ndarray) -> np.ndarray:
    array = np.asarray(mapping)
    if array.shape != (*CARTESIAN_SHAPE, 2):
        raise ValueError(
            "Cartesian-to-polar mapping must have shape "
            f"{(*CARTESIAN_SHAPE, 2)}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Cartesian-to-polar mapping must use an integer dtype")
    radial = array[..., 0]
    angular = array[..., 1]
    partially_mapped = (radial == -1) != (angular == -1)
    if np.any(partially_mapped):
        raise ValueError("mapping entries must be either two indices or [-1,-1]")
    invalid_negative = (radial < -1) | (angular < -1)
    if np.any(invalid_negative):
        raise ValueError("mapping contains indices below the [-1,-1] sentinel")
    mapped = radial >= 0
    if np.any(mapped & (radial >= RADIAL_BIN_COUNT)):
        raise ValueError("mapping contains an out-of-range radial index")
    if np.any(mapped & (angular >= ANGULAR_BIN_COUNT)):
        raise ValueError("mapping contains an out-of-range angular index")
    return array


def _mapping_digest(mapping: np.ndarray) -> str:
    canonical = np.ascontiguousarray(mapping, dtype="<i2")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def audit_mapping_injectivity(
    mapping: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate exact support and reject any Cartesian-to-polar collision."""

    array = (
        build_cartesian_to_polar_bin_mapping()
        if mapping is None
        else _validate_mapping_array(mapping)
    )
    mapped = np.all(array >= 0, axis=-1)
    expected = _expected_representable_mask()
    outside_mapped = int(np.count_nonzero(mapped & ~expected))
    expected_unmapped = int(np.count_nonzero(expected & ~mapped))
    if outside_mapped or expected_unmapped:
        raise ValueError(
            "mapping support differs from the fixed front-camera support: "
            f"outside_mapped={outside_mapped} expected_unmapped={expected_unmapped}"
        )

    pairs = np.asarray(array[mapped], dtype=np.int64)
    flat_bins = pairs[:, 0] * ANGULAR_BIN_COUNT + pairs[:, 1]
    unique_bins, counts = np.unique(flat_bins, return_counts=True)
    collision_count = int(np.count_nonzero(counts > 1))
    if collision_count:
        first = int(unique_bins[np.flatnonzero(counts > 1)[0]])
        radial_bin, angular_bin = divmod(first, ANGULAR_BIN_COUNT)
        locations = np.argwhere(
            mapped
            & (array[..., 0] == radial_bin)
            & (array[..., 1] == angular_bin)
        )
        raise ValueError(
            "Cartesian-to-polar mapping collision at "
            f"({radial_bin},{angular_bin}) for Cartesian cells "
            f"{locations.tolist()}"
        )
    expected_mapping = _expected_mapping_unchecked()
    deterministic_mismatch_count = int(
        np.count_nonzero(np.any(array != expected_mapping, axis=-1))
    )
    if deterministic_mismatch_count:
        raise ValueError(
            "mapping differs from deterministic v1 Cartesian-to-polar geometry "
            f"at {deterministic_mismatch_count} cells"
        )
    return {
        "schema": f"{GEOMETRY_SCHEMA}_mapping_audit",
        "mapping_sha256": _mapping_digest(array),
        "representable_cartesian_cell_count": int(np.count_nonzero(expected)),
        "mapped_cartesian_cell_count": int(pairs.shape[0]),
        "unique_polar_bin_count": int(unique_bins.size),
        "polar_bin_count": RADIAL_BIN_COUNT * ANGULAR_BIN_COUNT,
        "outside_support_mapped_count": outside_mapped,
        "expected_support_unmapped_count": expected_unmapped,
        "deterministic_mapping_mismatch_count": 0,
        "collision_count": 0,
        "injective": True,
    }


def build_radial_factorization() -> RadialFactorization:
    """Build the immutable array contract used by radial perception models."""

    mapping = build_cartesian_to_polar_bin_mapping()
    mapping_audit = audit_mapping_injectivity(mapping)
    mask = np.all(mapping >= 0, axis=-1)
    flat_indices = np.full(
        CARTESIAN_SHAPE[0] * CARTESIAN_SHAPE[1], -1, dtype=np.int64
    )
    pairs = np.asarray(mapping[mask], dtype=np.int64)
    flat_indices[mask.reshape(-1)] = (
        pairs[:, 0] * ANGULAR_BIN_COUNT + pairs[:, 1]
    )
    radial_centers_m = (
        np.arange(RADIAL_BIN_COUNT, dtype=np.float64) + 0.5
    ) * RADIAL_BIN_SIZE_M
    angular_bin_width_rad = (
        ANGULAR_RANGE_RAD[1] - ANGULAR_RANGE_RAD[0]
    ) / ANGULAR_BIN_COUNT
    angular_centers_rad = ANGULAR_RANGE_RAD[0] + (
        np.arange(ANGULAR_BIN_COUNT, dtype=np.float64) + 0.5
    ) * angular_bin_width_rad

    arrays = (flat_indices, mask, radial_centers_m, angular_centers_rad)
    for array in arrays:
        array.setflags(write=False)
    return RadialFactorization(
        cartesian_to_polar_flat_indices=flat_indices,
        representable_mask=mask,
        radial_centers_m=radial_centers_m,
        angular_centers_rad=angular_centers_rad,
        mapping_sha256=str(mapping_audit["mapping_sha256"]),
    )


def gather_polar_logits_to_cartesian(
    polar_logits: torch.Tensor,
    factorization: RadialFactorization | None = None,
    *,
    unknown_logit: float = 0.0,
) -> torch.Tensor:
    """Gather ``[...,3,64,256]`` logits into ``[...,3,64,64]``.

    Class order is UNKNOWN, FREE, OCCUPIED.  Unsupported Cartesian cells use
    ``unknown_logit`` for UNKNOWN and the dtype's finite minimum for FREE and
    OCCUPIED, producing finite logits with deterministic UNKNOWN argmax.
    """

    if not isinstance(polar_logits, torch.Tensor):
        raise TypeError("polar_logits must be a torch.Tensor")
    if not polar_logits.is_floating_point():
        raise ValueError("polar_logits must use a floating-point dtype")
    expected_suffix = (len(LABEL_CLASSES), *POLAR_SHAPE)
    if polar_logits.ndim < 3 or tuple(polar_logits.shape[-3:]) != expected_suffix:
        raise ValueError(
            "polar_logits must have trailing shape "
            f"{expected_suffix}, got {tuple(polar_logits.shape)}"
        )
    unknown_value = float(unknown_logit)
    if not math.isfinite(unknown_value):
        raise ValueError("unknown_logit must be finite")
    minimum = torch.finfo(polar_logits.dtype).min
    if unknown_value <= minimum:
        raise ValueError("unknown_logit must exceed the dtype's finite minimum")

    active = factorization or build_radial_factorization()
    flat_indices = torch.tensor(
        active.cartesian_to_polar_flat_indices,
        dtype=torch.long,
        device=polar_logits.device,
    )
    supported = flat_indices >= 0
    safe_indices = torch.clamp_min(flat_indices, 0)
    flat_polar = polar_logits.flatten(start_dim=-2)
    gathered = torch.index_select(flat_polar, -1, safe_indices)
    outside = torch.full_like(gathered, minimum)
    outside[..., UNKNOWN_CLASS, :] = unknown_value
    support_shape = (1,) * (gathered.ndim - 1) + (flat_indices.numel(),)
    gathered = torch.where(supported.view(support_shape), gathered, outside)
    return gathered.unflatten(-1, CARTESIAN_SHAPE)


def _validate_categorical_labels(
    labels: np.ndarray, *, shape: tuple[int, int], name: str
) -> np.ndarray:
    array = np.asarray(labels)
    if array.shape != shape:
        raise ValueError(f"{name} labels must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} labels must use an integer dtype")
    if not np.isin(array, LABEL_CLASSES).all():
        invalid = np.unique(array[~np.isin(array, LABEL_CLASSES)]).tolist()
        raise ValueError(f"{name} labels contain invalid classes: {invalid}")
    return array


def force_outside_fov_unknown(
    cartesian_labels: np.ndarray,
    *,
    mapping: np.ndarray | None = None,
) -> np.ndarray:
    """Copy labels and force every non-representable Cartesian cell UNKNOWN."""

    labels = _validate_categorical_labels(
        cartesian_labels, shape=CARTESIAN_SHAPE, name="Cartesian"
    )
    active_mapping = (
        build_cartesian_to_polar_bin_mapping()
        if mapping is None
        else _validate_mapping_array(mapping)
    )
    audit_mapping_injectivity(active_mapping)
    result = labels.copy()
    result[~np.all(active_mapping >= 0, axis=-1)] = UNKNOWN_CLASS
    return result


def scatter_cartesian_labels_to_radial(
    cartesian_labels: np.ndarray,
    *,
    mapping: np.ndarray | None = None,
    reject_outside_known: bool = True,
) -> np.ndarray:
    """Scatter Cartesian categorical labels into the fixed radial grid.

    Strict mode rejects known evidence outside representable support instead
    of silently discarding it.  Non-strict mode explicitly canonicalizes such
    cells to UNKNOWN before scattering.
    """

    labels = _validate_categorical_labels(
        cartesian_labels, shape=CARTESIAN_SHAPE, name="Cartesian"
    )
    active_mapping = (
        build_cartesian_to_polar_bin_mapping()
        if mapping is None
        else _validate_mapping_array(mapping)
    )
    audit_mapping_injectivity(active_mapping)
    support = np.all(active_mapping >= 0, axis=-1)
    outside_known = (~support) & (labels != UNKNOWN_CLASS)
    outside_known_count = int(np.count_nonzero(outside_known))
    if reject_outside_known and outside_known_count:
        raise ValueError(
            "known Cartesian labels outside representable front-camera support: "
            f"{outside_known_count}"
        )

    radial = np.full(POLAR_SHAPE, UNKNOWN_CLASS, dtype=labels.dtype)
    bins = active_mapping[support]
    radial[bins[:, 0], bins[:, 1]] = labels[support]
    return radial


def gather_radial_labels_to_cartesian(
    radial_labels: np.ndarray,
    *,
    mapping: np.ndarray | None = None,
) -> np.ndarray:
    """Gather radial labels and leave every outside-FOV Cartesian cell UNKNOWN."""

    labels = _validate_categorical_labels(
        radial_labels, shape=POLAR_SHAPE, name="radial"
    )
    active_mapping = (
        build_cartesian_to_polar_bin_mapping()
        if mapping is None
        else _validate_mapping_array(mapping)
    )
    audit_mapping_injectivity(active_mapping)
    support = np.all(active_mapping >= 0, axis=-1)
    bins = active_mapping[support]
    cartesian = np.full(CARTESIAN_SHAPE, UNKNOWN_CLASS, dtype=labels.dtype)
    cartesian[support] = labels[bins[:, 0], bins[:, 1]]
    return cartesian


def audit_exact_cartesian_roundtrip(
    cartesian_labels: np.ndarray,
    *,
    mapping: np.ndarray | None = None,
) -> dict[str, Any]:
    """Prove exact scatter/gather recovery for a representable label grid."""

    labels = _validate_categorical_labels(
        cartesian_labels, shape=CARTESIAN_SHAPE, name="Cartesian"
    )
    active_mapping = (
        build_cartesian_to_polar_bin_mapping()
        if mapping is None
        else _validate_mapping_array(mapping)
    )
    mapping_audit = audit_mapping_injectivity(active_mapping)
    support = np.all(active_mapping >= 0, axis=-1)
    outside_known_count = int(
        np.count_nonzero((~support) & (labels != UNKNOWN_CLASS))
    )
    if outside_known_count:
        raise ValueError(
            "exact roundtrip requires all known Cartesian labels to lie in "
            f"representable support; found {outside_known_count} outside"
        )
    radial = scatter_cartesian_labels_to_radial(
        labels, mapping=active_mapping, reject_outside_known=True
    )
    recovered = gather_radial_labels_to_cartesian(
        radial, mapping=active_mapping
    )
    mismatch_count = int(np.count_nonzero(recovered != labels))
    if mismatch_count:
        raise AssertionError(
            f"categorical radial scatter/gather changed {mismatch_count} cells"
        )
    return {
        "schema": f"{GEOMETRY_SCHEMA}_roundtrip_audit",
        "mapping_sha256": mapping_audit["mapping_sha256"],
        "known_cartesian_cell_count": int(
            np.count_nonzero(labels != UNKNOWN_CLASS)
        ),
        "outside_support_known_count": 0,
        "roundtrip_mismatch_count": 0,
        "exact_roundtrip": True,
    }


__all__ = [
    "ANGULAR_BIN_COUNT",
    "ANGULAR_RANGE_RAD",
    "CARTESIAN_CELL_SIZE_M",
    "CARTESIAN_SHAPE",
    "GEOMETRY_SCHEMA",
    "HALF_FOV_DEG",
    "HALF_FOV_RAD",
    "HORIZONTAL_FOV_DEG",
    "POLAR_SHAPE",
    "RADIAL_BIN_COUNT",
    "RADIAL_BIN_SIZE_M",
    "RADIAL_RANGE_M",
    "RadialFactorization",
    "audit_exact_cartesian_roundtrip",
    "audit_mapping_injectivity",
    "build_cartesian_to_polar_bin_mapping",
    "build_radial_factorization",
    "cartesian_cell_to_polar_bin",
    "force_outside_fov_unknown",
    "gather_polar_logits_to_cartesian",
    "gather_radial_labels_to_cartesian",
    "geometry_metadata",
    "representable_cartesian_mask",
    "scatter_cartesian_labels_to_radial",
]
