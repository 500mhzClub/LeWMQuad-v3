"""Representation-qualification probe V1: spatial and semantic preservation.

Implements the registered probe of
``docs/lewm_go2_representation_qualification_probe_v1_preregistration_2026-08-05.md``.

Two target families are built from the frame's **own** pose and its scene
manifest, and are used only as labels -- never as probe input:

*Spatial*
    A 64x64 egocentric body-frame grid with three predicted classes:
    ``occupied``, ``free``, and ``unknown``.  ``unknown`` covers everything
    outside the horizontal frustum or behind an occluder, so a single-frame
    latent is never asked to reconstruct map content it could not observe, and
    a probe cannot score by hallucinating structure behind walls.

*Semantic*
    Landmark targets restricted to what the corpus actually labels.  Manifests
    carry only ``landmark_red`` and ``landmark_blue`` under a single
    ``kind: "landmark"``; there is no beacon/distractor annotation, so no
    tri-class target is constructed.

The module has no filesystem, RGB, or encoder access.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "lewm_go2_representation_qualification_probe_v1"

# Native contract of direct_egocentric_bev_state_jepa_v1: rows are forward and
# columns are left, both increasing with index
# (lewm/datasets/go2_paired_navigation.py: base_forward_increasing /
# base_left_increasing).
GRID_SIZE = 64
X_MIN_M, X_MAX_M = -0.95, 5.35          # forward range, includes behind-robot
Y_MIN_M, Y_MAX_M = -3.15, 3.15          # left range
CELL_X_M = (X_MAX_M - X_MIN_M) / GRID_SIZE
CELL_Y_M = (Y_MAX_M - Y_MIN_M) / GRID_SIZE
CELL_DIAGONAL_M = math.hypot(CELL_X_M, CELL_Y_M)

CAMERA_YFOV_DEG = 78.323
HORIZONTAL_HALF_ANGLE_RAD = math.radians(CAMERA_YFOV_DEG / 2.0)
VISIBILITY_BEARING_BINS = 256

# Three predicted spatial classes, in the model's own index order
# (lewm/models/direct_egocentric_bev_state_jepa_v1.py).
CLASS_UNKNOWN, CLASS_FREE, CLASS_OCCUPIED = 0, 1, 2
CLASS_NAMES = ("unknown", "free", "occupied")

# Semantic label vocabulary actually present in the corpus manifests.
LANDMARK_MATERIALS = ("landmark_red", "landmark_blue")

BOOTSTRAP_SEED = 2_026_080_571
SPLIT_SEED = 2_026_080_572
PROBE_SEEDS = (2_026_080_581, 2_026_080_582, 2_026_080_583)
BOOTSTRAP_RESAMPLES = 10_000

VALIDATION_SCENE_COUNT = 8

# Reported for context only.  Explicitly NOT a gate: the direct-BEV tiers were
# calibrated on a different raster definition, resolution, visibility
# convention, class set, and aggregation (preregistration section 7).
CONTEXT_ONLY_LEGACY_TIERS = {
    250: {
        "aggregate_raster_balanced_accuracy_minimum_inclusive": 0.80,
        "aggregate_free_recall_minimum_inclusive": 0.68,
        "aggregate_occupied_recall_minimum_inclusive": 0.88,
    }
}


class RepresentationProbeError(RuntimeError):
    """Raised when the probe contract is violated."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _rectangles(entries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = [
        (
            float(e["center_xyz_m"][0]),
            float(e["center_xyz_m"][1]),
            float(e["size_xyz_m"][0]),
            float(e["size_xyz_m"][1]),
            float(e.get("yaw_rad", 0.0)),
        )
        for e in entries
    ]
    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def wall_rectangles_v1(manifest: Mapping[str, Any]) -> np.ndarray:
    """Return every occluding footprint in the scene.

    Families differ in how they carry geometry: enclosed mazes populate
    ``walls`` while open fields populate ``obstacles``.  Both occlude and both
    are traversal obstacles, so both are included.  A scene with neither is
    legitimate -- an open field with no occluder makes the whole frustum
    observable -- and yields an empty set rather than an error.
    """

    entries: list[Mapping[str, Any]] = []
    for key in ("walls", "obstacles"):
        value = manifest.get(key)
        if isinstance(value, list):
            entries.extend(value)
    return _rectangles(entries)


def landmarks_v1(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return landmark centres and their material class."""

    entries = manifest.get("landmarks") or []
    result = []
    for entry in entries:
        material = str(entry.get("material_id", ""))
        if material not in LANDMARK_MATERIALS:
            continue
        result.append(
            {
                "x": float(entry["center_xyz_m"][0]),
                "y": float(entry["center_xyz_m"][1]),
                "material": material,
                "class_index": LANDMARK_MATERIALS.index(material),
            }
        )
    return result


def body_cell_centres_v1() -> tuple[np.ndarray, np.ndarray]:
    forward = X_MIN_M + (np.arange(GRID_SIZE, dtype=np.float64) + 0.5) * CELL_X_M
    left = Y_MIN_M + (np.arange(GRID_SIZE, dtype=np.float64) + 0.5) * CELL_Y_M
    # rows index forward, columns index left
    return np.meshgrid(forward, left, indexing="ij")


def yaw_from_quaternion_wxyz_v1(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_body_v1(
    world_x: np.ndarray, world_y: np.ndarray, position_xy_m, yaw_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    dx = np.asarray(world_x, dtype=np.float64) - float(position_xy_m[0])
    dy = np.asarray(world_y, dtype=np.float64) - float(position_xy_m[1])
    cos_y, sin_y = math.cos(-yaw_rad), math.sin(-yaw_rad)
    return cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy


def spatial_target_v1(
    position_xy_m: Sequence[float], yaw_rad: float, walls: np.ndarray
) -> np.ndarray:
    """Return the ``(64, 64)`` three-class egocentric label map.

    Observability is derived from this frame's pose alone: no history union, no
    successor, no map is available to the probe that consumes the latent.
    """

    grid_x, grid_y = body_cell_centres_v1()
    cos_y, sin_y = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    world_x = float(position_xy_m[0]) + cos_y * grid_x - sin_y * grid_y
    world_y = float(position_xy_m[1]) + sin_y * grid_x + cos_y * grid_y

    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for cx, cy, sx, sy, wyaw in walls:
        dx, dy = world_x - cx, world_y - cy
        cw, sw = math.cos(-wyaw), math.sin(-wyaw)
        lx, ly = cw * dx - sw * dy, sw * dx + cw * dy
        occupied |= (np.abs(lx) <= sx / 2.0) & (np.abs(ly) <= sy / 2.0)

    bearing = np.arctan2(grid_y, grid_x)
    distance = np.hypot(grid_x, grid_y)
    # grid_x < 0 is behind the robot: outside the frustum by construction.
    in_fov = (grid_x > 0.0) & (np.abs(bearing) <= HORIZONTAL_HALF_ANGLE_RAD)

    # Occlusion by exact segment-rectangle intersection from the body origin to
    # each cell centre.  An angular-bin sweep is not usable here: at 2 m a
    # 256-bin sweep subtends about 1 cm per bin while cells are 9.8 cm, so wall
    # cells populate only a fraction of bins and most rays would find no
    # occluder.  The slab test below is exact and independent of cell size.
    observable = in_fov.copy()
    if walls.shape[0]:
        px = grid_x.reshape(-1)
        py = grid_y.reshape(-1)
        blocked = np.zeros(px.shape, dtype=bool)
        # A blocker must lie strictly before the cell itself; allow half a cell
        # of slack so a wall cell is not treated as occluding itself.
        slack = 1.0 - (CELL_DIAGONAL_M / 2.0) / np.maximum(distance.reshape(-1), 1.0e-9)
        for cx, cy, sx, sy, wyaw in walls:
            cos_w, sin_w = math.cos(-wyaw), math.sin(-wyaw)
            ox = cos_w * (0.0 - cx) - sin_w * (0.0 - cy)
            oy = sin_w * (0.0 - cx) + cos_w * (0.0 - cy)
            lx = cos_w * (px - cx) - sin_w * (py - cy)
            ly = sin_w * (px - cx) + cos_w * (py - cy)
            dx, dy = lx - ox, ly - oy
            t_enter = np.zeros_like(dx)
            t_exit = np.ones_like(dx)
            for origin, delta, half in ((ox, dx, sx / 2.0), (oy, dy, sy / 2.0)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t0 = (-half - origin) / delta
                    t1 = (half - origin) / delta
                lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
                parallel = np.abs(delta) < 1.0e-12
                inside = np.abs(origin) <= half
                lo = np.where(parallel, np.where(inside, 0.0, 1.0), lo)
                hi = np.where(parallel, np.where(inside, 1.0, 0.0), hi)
                t_enter = np.maximum(t_enter, lo)
                t_exit = np.minimum(t_exit, hi)
            blocked |= (t_enter <= t_exit) & (t_enter < slack)
        observable &= ~blocked.reshape(GRID_SIZE, GRID_SIZE)

    label = np.full((GRID_SIZE, GRID_SIZE), CLASS_UNKNOWN, dtype=np.int64)
    label[observable & occupied] = CLASS_OCCUPIED
    label[observable & ~occupied] = CLASS_FREE
    return label


def semantic_target_v1(
    position_xy_m: Sequence[float],
    yaw_rad: float,
    walls: np.ndarray,
    landmarks: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Landmark visibility, colour, bearing, and range from this frame's pose.

    A landmark counts as visible when it lies inside the horizontal frustum,
    within the grid's forward range, and is not occluded by a wall along its
    bearing.
    """

    best: dict[str, float] | None = None
    for landmark in landmarks:
        bx, by = world_to_body_v1(
            np.array([landmark["x"]]), np.array([landmark["y"]]), position_xy_m, yaw_rad
        )
        bx, by = float(bx[0]), float(by[0])
        if bx <= 0.0:
            continue
        bearing = math.atan2(by, bx)
        distance = math.hypot(bx, by)
        if abs(bearing) > HORIZONTAL_HALF_ANGLE_RAD or distance > X_MAX_M:
            continue
        # Occlusion: march along the ray and reject if a wall lies nearer.
        steps = max(2, int(distance / (CELL_DIAGONAL_M / 2.0)))
        ts = np.linspace(0.0, 1.0, steps, endpoint=False)[1:]
        sample_bx, sample_by = bx * ts, by * ts
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        wx = float(position_xy_m[0]) + cos_y * sample_bx - sin_y * sample_by
        wy = float(position_xy_m[1]) + sin_y * sample_bx + cos_y * sample_by
        blocked = False
        for cx, cy, sx, sy, wyaw in walls:
            dx, dy = wx - cx, wy - cy
            cw, sw = math.cos(-wyaw), math.sin(-wyaw)
            lx, ly = cw * dx - sw * dy, sw * dx + cw * dy
            if bool(((np.abs(lx) <= sx / 2.0) & (np.abs(ly) <= sy / 2.0)).any()):
                blocked = True
                break
        if blocked:
            continue
        if best is None or distance < best["range_m"]:
            best = {
                "visible": 1.0,
                "class_index": float(landmark["class_index"]),
                "bearing_rad": float(bearing),
                "range_m": float(distance),
            }
    if best is None:
        return {"visible": 0.0, "class_index": -1.0, "bearing_rad": 0.0, "range_m": 0.0}
    return best


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def spatial_metrics_v1(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Per-class recall and IoU over all three classes, plus balanced accuracy."""

    predicted = np.asarray(predicted).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    if predicted.shape != truth.shape:
        raise RepresentationProbeError("spatial prediction and truth shapes disagree")
    out: dict[str, float] = {}
    recalls = []
    for index, name in enumerate(CLASS_NAMES):
        actual = truth == index
        chosen = predicted == index
        support = int(actual.sum())
        intersection = int((actual & chosen).sum())
        union = int((actual | chosen).sum())
        recall = float(intersection / support) if support else math.nan
        out[f"{name}_recall"] = recall
        out[f"{name}_iou"] = float(intersection / union) if union else math.nan
        out[f"{name}_support"] = support
        if support:
            recalls.append(recall)
    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else math.nan
    out["cells"] = int(truth.size)
    return out


def semantic_metrics_v1(
    predictions: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Balanced accuracy for the classification targets, MAE for regressions."""

    visible_truth = np.asarray(truth["visible"]).astype(bool)
    visible_pred = np.asarray(predictions["visible"]) >= 0.5
    recalls = []
    for value in (True, False):
        actual = visible_truth == value
        if actual.any():
            recalls.append(float((visible_pred[actual] == value).mean()))
    out = {"visibility_balanced_accuracy": float(np.mean(recalls)) if recalls else math.nan}

    mask = visible_truth
    if mask.any():
        colour_truth = np.asarray(truth["class_index"])[mask].astype(int)
        colour_pred = (np.asarray(predictions["class_index"])[mask] >= 0.5).astype(int)
        class_recalls = []
        for value in (0, 1):
            actual = colour_truth == value
            if actual.any():
                class_recalls.append(float((colour_pred[actual] == value).mean()))
        out["colour_balanced_accuracy"] = (
            float(np.mean(class_recalls)) if class_recalls else math.nan
        )
        out["bearing_median_abs_error_rad"] = float(
            np.median(np.abs(np.asarray(predictions["bearing_rad"])[mask] - np.asarray(truth["bearing_rad"])[mask]))
        )
        out["range_median_abs_error_m"] = float(
            np.median(np.abs(np.asarray(predictions["range_m"])[mask] - np.asarray(truth["range_m"])[mask]))
        )
        out["visible_frames"] = int(mask.sum())
    else:
        out.update(
            {
                "colour_balanced_accuracy": math.nan,
                "bearing_median_abs_error_rad": math.nan,
                "range_median_abs_error_m": math.nan,
                "visible_frames": 0,
            }
        )
    return out


def scene_cluster_bootstrap_v1(
    values_by_scene: Mapping[str, float], family_by_scene: Mapping[str, str]
) -> dict[str, object]:
    """Family-balanced whole-scene bootstrap of a per-scene statistic."""

    scenes = sorted(values_by_scene)
    if not scenes:
        raise RepresentationProbeError("bootstrap has no scenes")
    families: dict[str, list[str]] = {}
    for scene in scenes:
        families.setdefault(str(family_by_scene[scene]), []).append(scene)
    ordered = sorted(families)

    def statistic(pick) -> float:
        return float(np.mean([np.mean([values_by_scene[s] for s in pick[f]]) for f in ordered]))

    point = statistic({f: sorted(families[f]) for f in ordered})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    groups = {f: sorted(families[f]) for f in ordered}
    for index in range(BOOTSTRAP_RESAMPLES):
        pick = {
            f: [groups[f][i] for i in rng.integers(0, len(groups[f]), size=len(groups[f]))]
            for f in ordered
        }
        draws[index] = statistic(pick)
    lower, upper = (float(v) for v in np.quantile(draws, (0.025, 0.975)))
    return {
        "point": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_half_width": (upper - lower) / 2.0,
        "scene_clusters": len(scenes),
        "families": len(ordered),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def result_identity_v1(result: Mapping[str, object]) -> str:
    payload = {k: v for k, v in result.items() if k != "identity_sha256"}
    return hashlib.sha256(canonical_bytes_v1(payload)).hexdigest()


__all__ = [
    "CLASS_FREE",
    "CLASS_NAMES",
    "CLASS_OCCUPIED",
    "CLASS_UNKNOWN",
    "CONTEXT_ONLY_LEGACY_TIERS",
    "GRID_SIZE",
    "LANDMARK_MATERIALS",
    "PROBE_SEEDS",
    "RepresentationProbeError",
    "SCHEMA",
    "SPLIT_SEED",
    "VALIDATION_SCENE_COUNT",
    "body_cell_centres_v1",
    "canonical_bytes_v1",
    "landmarks_v1",
    "result_identity_v1",
    "scene_cluster_bootstrap_v1",
    "semantic_metrics_v1",
    "semantic_target_v1",
    "spatial_metrics_v1",
    "spatial_target_v1",
    "wall_rectangles_v1",
    "world_to_body_v1",
    "yaw_from_quaternion_wxyz_v1",
]
