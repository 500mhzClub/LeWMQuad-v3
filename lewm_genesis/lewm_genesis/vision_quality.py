"""Rendered-frame quality gates for egocentric training data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


LOW_INFO_REASON_NAMES: frozenset[str] = frozenset(
    {
        "low_rgb_texture",
        "near_wall_depth",
        "near_forward_geometry",
    }
)


@dataclass(frozen=True)
class VisionQualityConfig:
    """Thresholds for rejecting visually uninformative egocentric frames."""

    low_rgb_std_threshold: float = 18.0
    low_rgb_edge_threshold: float = 0.30
    near_wall_depth_m: float = 0.40
    near_wall_fraction_threshold: float = 0.85
    min_forward_hit_m: float = 0.25


DEFAULT_VISION_QUALITY_CONFIG = VisionQualityConfig()


def assess_rendered_frame(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    *,
    require_depth: bool,
    camera_safety: dict[str, float | bool] | None = None,
    config: VisionQualityConfig = DEFAULT_VISION_QUALITY_CONFIG,
) -> dict[str, Any]:
    """Return validity, invalid reasons, and stats for one rendered frame.

    Geometry-safe camera placement is necessary but not sufficient for training:
    a frame can be outside all meshes while still being a uniform wall 10 cm in
    front of the camera. This audit rejects those low-information views.
    """

    reasons: list[str] = []
    if camera_safety and bool(camera_safety.get("unsafe")):
        reasons.append("camera_safety_unresolved")
    if _near_forward_geometry(camera_safety, config):
        reasons.append("near_forward_geometry")

    rgb_stats = _rgb_stats(rgb, config=config)
    if rgb is None:
        reasons.append("missing_rgb")
    elif rgb_stats.get("bad_shape"):
        reasons.append(f"bad_rgb_shape:{tuple(np.asarray(rgb).shape)}")
    else:
        if not bool(rgb_stats["finite"]):
            reasons.append("rgb_nonfinite")
        if bool(rgb_stats["low_texture"]):
            reasons.append("low_rgb_texture")

    depth_stats = _depth_stats(depth, config=config)
    if depth is None:
        if require_depth:
            reasons.append("missing_depth")
    else:
        if int(depth_stats["finite_count"]) == 0:
            reasons.append("depth_all_nonfinite")
        if bool(depth_stats["near_wall"]):
            reasons.append("near_wall_depth")

    return {
        "valid": not reasons,
        "invalid_reasons": reasons,
        "rgb_stats": rgb_stats,
        "depth_stats": depth_stats,
    }


def _rgb_stats(
    rgb: np.ndarray | None,
    *,
    config: VisionQualityConfig,
) -> dict[str, float | int | bool | None]:
    stats: dict[str, float | int | bool | None] = {
        "finite": None,
        "mean": None,
        "std": None,
        "luma_std": None,
        "luma_edge_mean": None,
        "channel_delta_mean": None,
        "low_texture": False,
        "bad_shape": False,
    }
    if rgb is None:
        return stats
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        stats["bad_shape"] = True
        return stats

    arr_f = arr.astype(np.float32, copy=False)
    finite = bool(np.isfinite(arr_f).all())
    stats["finite"] = finite
    if not finite:
        return stats

    luma = 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]
    if luma.shape[1] > 1:
        dx = np.abs(np.diff(luma, axis=1)).mean()
    else:
        dx = 0.0
    if luma.shape[0] > 1:
        dy = np.abs(np.diff(luma, axis=0)).mean()
    else:
        dy = 0.0
    channel_delta = np.max(arr_f, axis=-1) - np.min(arr_f, axis=-1)

    rgb_std = float(arr_f.std())
    luma_edge = float((dx + dy) * 0.5)
    stats.update(
        {
            "mean": float(arr_f.mean()),
            "std": rgb_std,
            "luma_std": float(luma.std()),
            "luma_edge_mean": luma_edge,
            "channel_delta_mean": float(channel_delta.mean()),
            "low_texture": (
                rgb_std < float(config.low_rgb_std_threshold)
                and luma_edge < float(config.low_rgb_edge_threshold)
            ),
        }
    )
    return stats


def _depth_stats(
    depth: np.ndarray | None,
    *,
    config: VisionQualityConfig,
) -> dict[str, float | int | bool | None]:
    stats: dict[str, float | int | bool | None] = {
        "finite_count": None,
        "nonfinite_count": None,
        "min_m": None,
        "max_m": None,
        "p05_m": None,
        "p50_m": None,
        "p95_m": None,
        "std_m": None,
        "near_wall_fraction": None,
        "near_wall": False,
    }
    if depth is None:
        return stats

    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    finite_count = int(finite.sum())
    stats["finite_count"] = finite_count
    stats["nonfinite_count"] = int(arr.size - finite_count)
    if finite_count == 0:
        return stats

    vals = arr[finite]
    p05 = float(np.percentile(vals, 5))
    p50 = float(np.percentile(vals, 50))
    p95 = float(np.percentile(vals, 95))
    near_wall_depth_m = float(config.near_wall_depth_m)
    near_fraction = float((vals <= near_wall_depth_m).mean())
    stats.update(
        {
            "min_m": float(np.min(vals)),
            "max_m": float(np.max(vals)),
            "p05_m": p05,
            "p50_m": p50,
            "p95_m": p95,
            "std_m": float(vals.std()),
            "near_wall_fraction": near_fraction,
            "near_wall": (
                near_fraction >= float(config.near_wall_fraction_threshold)
                and p50 <= near_wall_depth_m
            ),
        }
    )
    return stats


def _near_forward_geometry(
    camera_safety: dict[str, float | bool] | None,
    config: VisionQualityConfig,
) -> bool:
    if not camera_safety:
        return False
    raw = camera_safety.get("forward_hit_m")
    if raw is None:
        return False
    try:
        hit_m = float(raw)
    except (TypeError, ValueError):
        return False
    return math.isfinite(hit_m) and hit_m < float(config.min_forward_hit_m)
