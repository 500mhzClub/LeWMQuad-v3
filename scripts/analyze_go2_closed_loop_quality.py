#!/usr/bin/env python3
"""Quality analysis for Go2 closed-loop memory-navigation rollouts.

Runtime logs only contain onboard decisions and measured pose. If a scene
corpus is provided, this script also computes privileged geometry metrics for
offline diagnosis. Those metrics are never used by the benchmark harness for
runtime traversal decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402

PURE_YAW = frozenset(("yaw_left", "yaw_right"))
TRANSLATING = frozenset((
    "forward_slow",
    "forward_medium",
    "forward_fast",
    "arc_left",
    "arc_right",
    "backward",
))
STRAIGHT_FORWARD = frozenset(("forward_slow", "forward_medium", "forward_fast"))


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), ndigits)


def _share(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _max_streak(items: list[str], allowed: set[str] | frozenset[str] | None = None) -> int:
    best = 0
    cur = 0
    prev = None
    for item in items:
        if allowed is not None and item not in allowed:
            cur = 0
            prev = None
            continue
        if item == prev:
            cur += 1
        else:
            cur = 1
            prev = item
        best = max(best, cur)
    return int(best)


def _path_metrics(xys: list[list[float]]) -> dict[str, Any]:
    if len(xys) < 2:
        return {
            "path_travel_m": 0.0,
            "path_net_m": 0.0,
            "path_tortuosity": None,
            "worst_40_tick_tortuosity": None,
        }
    pts = [(float(x), float(y)) for x, y in xys]
    travel = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    net = math.dist(pts[0], pts[-1])
    worst = None
    window = 40
    if len(pts) > window:
        for start in range(0, len(pts) - window):
            seg = pts[start:start + window + 1]
            seg_travel = sum(math.dist(a, b) for a, b in zip(seg, seg[1:]))
            seg_net = math.dist(seg[0], seg[-1])
            if seg_net > 1e-6:
                tort = seg_travel / seg_net
                worst = tort if worst is None else max(worst, tort)
    return {
        "path_travel_m": _round(travel),
        "path_net_m": _round(net),
        "path_tortuosity": _round(travel / max(net, 1e-6)),
        "worst_40_tick_tortuosity": _round(worst),
    }


def _wrap_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _near_target_orbit_metrics(
    log: list[dict[str, Any]],
    target_xy: dict[str, Any],
    *,
    radius_m: float,
) -> dict[str, Any]:
    """Approximate visible orbiting as angular sweep around the active target."""
    per_color: dict[str, dict[str, Any]] = {}
    total = 0.0
    for color, xy in sorted(target_xy.items()):
        if not isinstance(xy, list) or len(xy) != 2:
            continue
        cx = float(xy[0])
        cy = float(xy[1])
        rows: list[tuple[int, float, float]] = []
        for entry in log:
            if str(entry.get("target_color", "")) != str(color):
                continue
            post_xy = entry.get("post_xy")
            if not isinstance(post_xy, list) or len(post_xy) != 2:
                continue
            px = float(post_xy[0])
            py = float(post_xy[1])
            dist = math.hypot(px - cx, py - cy)
            if dist > float(radius_m):
                continue
            angle = math.atan2(py - cy, px - cx)
            rows.append((int(entry.get("tick", len(rows))), angle, dist))

        sweep = 0.0
        for (_, prev_angle, _), (_, cur_angle, _) in zip(rows, rows[1:]):
            sweep += abs(_wrap_angle_rad(cur_angle - prev_angle))
        sweep_deg = math.degrees(sweep)
        total += sweep_deg
        per_color[str(color)] = {
            "ticks": int(len(rows)),
            "sweep_deg": _round(sweep_deg, 1),
            "first_tick": None if not rows else int(rows[0][0]),
            "last_tick": None if not rows else int(rows[-1][0]),
            "min_dist_m": None if not rows else _round(min(row[2] for row in rows)),
            "max_dist_m": None if not rows else _round(max(row[2] for row in rows)),
        }
    return {
        "near_target_orbit_radius_m": _round(float(radius_m)),
        "near_target_angular_sweep_total_deg": _round(total, 1),
        "near_target_angular_sweep_by_color": per_color,
    }


def _load_grid(args: argparse.Namespace, scene_name: str) -> InflatedOccupancyGrid | None:
    if args.scene_corpus is None:
        return None
    platform = load_platform_manifest(args.platform_manifest.resolve())
    scene_dirs = find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=args.family)
    matches = [p for p in scene_dirs if p.name == scene_name]
    if not matches:
        return None
    pack = load_scene_pack(matches[0], platform_manifest=platform, workspace_root=REPO_ROOT)
    return InflatedOccupancyGrid(
        pack.scene_graph.manifest,
        cell_size_m=float(args.cell_size_m),
        inflation_m=float(args.inflation_m),
    )


def _clearance_metrics(grid: InflatedOccupancyGrid | None, xys: list[list[float]]) -> dict[str, Any]:
    if grid is None or not xys:
        return {
            "analysis_uses_privileged_geometry": False,
            "wall_clearance_m": None,
        }
    clearances = np.asarray(
        [grid.obstacle_clearance_m((float(x), float(y))) for x, y in xys],
        dtype=np.float32,
    )
    config_clearances = clearances - float(grid.inflation_m)
    return {
        "analysis_uses_privileged_geometry": True,
        "wall_clearance_m": {
            "min": _round(float(np.min(clearances))),
            "p10": _round(float(np.percentile(clearances, 10))),
            "p25": _round(float(np.percentile(clearances, 25))),
            "median": _round(float(np.percentile(clearances, 50))),
            "p75": _round(float(np.percentile(clearances, 75))),
            "share_under_0p24": _round(float(np.mean(clearances < 0.24))),
            "share_under_0p30": _round(float(np.mean(clearances < 0.30))),
            "share_under_0p36": _round(float(np.mean(clearances < 0.36))),
        },
        "configuration_clearance_m": {
            "min": _round(float(np.min(config_clearances))),
            "median": _round(float(np.percentile(config_clearances, 50))),
        },
    }


def _body_probe_configuration_clearance_m(
    grid: InflatedOccupancyGrid,
    xy: list[float],
    yaw: float,
    *,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
) -> float:
    x = float(xy[0])
    y = float(xy[1])
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    lx, ly = -fy, fx
    probes = (
        (0.0, 0.0),
        (body_forward_m, 0.0),
        (body_forward_m, body_half_width_m),
        (body_forward_m, -body_half_width_m),
        (0.0, body_half_width_m),
        (0.0, -body_half_width_m),
    )
    return float(min(
        grid.obstacle_clearance_m((x + forward * fx + lateral * lx,
                                   y + forward * fy + lateral * ly))
        - float(body_probe_margin_m)
        for forward, lateral in probes
    ))


def _forward_body_sweep_clearance_m(
    grid: InflatedOccupancyGrid,
    xy: list[float],
    yaw: float,
    *,
    distance_m: float,
    step_m: float,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
) -> float:
    distance = max(0.0, float(distance_m))
    step = max(0.01, float(step_m))
    if distance <= 0.0:
        return _body_probe_configuration_clearance_m(
            grid,
            xy,
            yaw,
            body_forward_m=body_forward_m,
            body_half_width_m=body_half_width_m,
            body_probe_margin_m=body_probe_margin_m,
        )
    samples = max(1, int(math.ceil(distance / step)))
    x = float(xy[0])
    y = float(xy[1])
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    best = float("inf")
    for i in range(1, samples + 1):
        d = min(distance, i * step)
        best = min(
            best,
            _body_probe_configuration_clearance_m(
                grid,
                [x + d * fx, y + d * fy],
                yaw,
                body_forward_m=body_forward_m,
                body_half_width_m=body_half_width_m,
                body_probe_margin_m=body_probe_margin_m,
            ),
        )
    return float(best)


def _body_clearance_metrics(
    grid: InflatedOccupancyGrid | None,
    poses: list[tuple[list[float], float]],
    *,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
) -> dict[str, Any]:
    if grid is None or not poses:
        return {"body_configuration_clearance_m": None}
    clearances = np.asarray(
        [
            _body_probe_configuration_clearance_m(
                grid,
                xy,
                yaw,
                body_forward_m=body_forward_m,
                body_half_width_m=body_half_width_m,
                body_probe_margin_m=body_probe_margin_m,
            )
            for xy, yaw in poses
        ],
        dtype=np.float32,
    )
    return {
        "body_configuration_clearance_m": {
            "min": _round(float(np.min(clearances))),
            "p10": _round(float(np.percentile(clearances, 10))),
            "median": _round(float(np.percentile(clearances, 50))),
            "share_under_0p00": _round(float(np.mean(clearances < 0.0))),
            "share_under_0p03": _round(float(np.mean(clearances < 0.03))),
            "share_under_0p06": _round(float(np.mean(clearances < 0.06))),
        }
    }


def _forward_heading_diagnostics(
    grid: InflatedOccupancyGrid | None,
    log: list[dict[str, Any]],
    *,
    distance_m: float,
    step_m: float,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
    warning_threshold_m: float = 0.03,
    max_intervals: int = 12,
) -> dict[str, Any]:
    if grid is None:
        return {}
    rows: list[tuple[int, float, dict[str, Any]]] = []
    for entry in log:
        xy = entry.get("post_xy")
        yaw = entry.get("post_yaw")
        if not isinstance(xy, list) or len(xy) != 2 or yaw is None:
            continue
        clearance = _forward_body_sweep_clearance_m(
            grid,
            xy,
            float(yaw),
            distance_m=distance_m,
            step_m=step_m,
            body_forward_m=body_forward_m,
            body_half_width_m=body_half_width_m,
            body_probe_margin_m=body_probe_margin_m,
        )
        rows.append((int(entry.get("tick", len(rows))), float(clearance), entry))
    if not rows:
        return {}

    clearances = np.asarray([clearance for _, clearance, _ in rows], dtype=np.float32)
    risky = [
        (tick, clearance, entry)
        for tick, clearance, entry in rows
        if clearance < float(warning_threshold_m)
    ]

    def grouped_share(key: str) -> dict[str, Any]:
        buckets: dict[str, list[float]] = {}
        for _, clearance, entry in rows:
            name = str(entry.get(key, ""))
            buckets.setdefault(name, []).append(clearance)
        out: dict[str, Any] = {}
        for name, vals in buckets.items():
            arr = np.asarray(vals, dtype=np.float32)
            out[name] = {
                "n": len(vals),
                "share_under_warning": _round(float(np.mean(arr < float(warning_threshold_m)))),
                "min": _round(float(np.min(arr))),
            }
        return dict(sorted(out.items()))

    intervals: list[list[tuple[int, float, dict[str, Any]]]] = []
    current: list[tuple[int, float, dict[str, Any]]] = []
    for row in rows:
        if row[1] < float(warning_threshold_m):
            current.append(row)
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)

    interval_reports: list[dict[str, Any]] = []
    for segment in sorted(intervals, key=len, reverse=True)[:max_intervals]:
        states = Counter(str(entry.get("state", "")) for _, _, entry in segment)
        primitives = Counter(str(entry.get("primitive", "")) for _, _, entry in segment)
        requested = Counter(str(entry.get("requested_primitive", entry.get("primitive", ""))) for _, _, entry in segment)
        targets = Counter(str(entry.get("target_color", "")) for _, _, entry in segment)
        worst = min(segment, key=lambda item: item[1])
        interval_reports.append({
            "start_tick": int(segment[0][0]),
            "end_tick": int(segment[-1][0]),
            "len_ticks": int(len(segment)),
            "min_clearance_m": _round(min(clearance for _, clearance, _ in segment)),
            "worst_tick": int(worst[0]),
            "worst_state": str(worst[2].get("state", "")),
            "worst_primitive": str(worst[2].get("primitive", "")),
            "worst_requested_primitive": str(worst[2].get("requested_primitive", worst[2].get("primitive", ""))),
            "state_counts": dict(sorted(states.items())),
            "primitive_counts": dict(sorted(primitives.items())),
            "requested_counts": dict(sorted(requested.items())),
            "target_counts": dict(sorted(targets.items())),
        })

    return {
        "front_body_sweep_params": {
            "distance_m": _round(float(distance_m)),
            "step_m": _round(float(step_m)),
            "warning_threshold_m": _round(float(warning_threshold_m)),
        },
        "front_body_sweep_clearance_m": {
            "min": _round(float(np.min(clearances))),
            "p10": _round(float(np.percentile(clearances, 10))),
            "median": _round(float(np.percentile(clearances, 50))),
            "share_under_0p00": _round(float(np.mean(clearances < 0.0))),
            "share_under_0p03": _round(float(np.mean(clearances < 0.03))),
            "share_under_0p06": _round(float(np.mean(clearances < 0.06))),
            "share_under_warning": _round(_share(len(risky), len(rows))),
        },
        "front_body_sweep_warning_intervals": {
            "count": int(len(intervals)),
            "max_len_ticks": int(max((len(segment) for segment in intervals), default=0)),
            "worst_intervals": interval_reports,
        },
        "front_body_sweep_warning_share_by_state": grouped_share("state"),
        "front_body_sweep_warning_share_by_primitive": grouped_share("primitive"),
    }


def _body_clearance_diagnostics(
    grid: InflatedOccupancyGrid | None,
    log: list[dict[str, Any]],
    *,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
    max_intervals: int = 12,
) -> dict[str, Any]:
    if grid is None:
        return {}
    rows: list[tuple[int, float, dict[str, Any]]] = []
    for entry in log:
        xy = entry.get("post_xy")
        yaw = entry.get("post_yaw")
        if not isinstance(xy, list) or len(xy) != 2 or yaw is None:
            continue
        clearance = _body_probe_configuration_clearance_m(
            grid,
            xy,
            float(yaw),
            body_forward_m=body_forward_m,
            body_half_width_m=body_half_width_m,
            body_probe_margin_m=body_probe_margin_m,
        )
        rows.append((int(entry.get("tick", len(rows))), float(clearance), entry))
    if not rows:
        return {}

    def grouped_share(key: str) -> dict[str, Any]:
        buckets: dict[str, list[float]] = {}
        for _, clearance, entry in rows:
            name = str(entry.get(key, ""))
            buckets.setdefault(name, []).append(clearance)
        out: dict[str, Any] = {}
        for name, vals in buckets.items():
            out[name] = {
                "n": len(vals),
                "negative_share": _round(float(np.mean(np.asarray(vals) < 0.0))),
                "min": _round(float(np.min(vals))),
            }
        return dict(sorted(out.items()))

    intervals: list[list[tuple[int, float, dict[str, Any]]]] = []
    current: list[tuple[int, float, dict[str, Any]]] = []
    for row in rows:
        if row[1] < 0.0:
            current.append(row)
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)

    interval_reports: list[dict[str, Any]] = []
    for segment in sorted(intervals, key=len, reverse=True)[:max_intervals]:
        states = Counter(str(entry.get("state", "")) for _, _, entry in segment)
        primitives = Counter(str(entry.get("primitive", "")) for _, _, entry in segment)
        requested = Counter(str(entry.get("requested_primitive", entry.get("primitive", ""))) for _, _, entry in segment)
        targets = Counter(str(entry.get("target_color", "")) for _, _, entry in segment)
        worst = min(segment, key=lambda item: item[1])
        interval_reports.append({
            "start_tick": int(segment[0][0]),
            "end_tick": int(segment[-1][0]),
            "len_ticks": int(len(segment)),
            "min_clearance_m": _round(min(clearance for _, clearance, _ in segment)),
            "worst_tick": int(worst[0]),
            "worst_state": str(worst[2].get("state", "")),
            "worst_primitive": str(worst[2].get("primitive", "")),
            "worst_requested_primitive": str(worst[2].get("requested_primitive", worst[2].get("primitive", ""))),
            "state_counts": dict(sorted(states.items())),
            "primitive_counts": dict(sorted(primitives.items())),
            "requested_counts": dict(sorted(requested.items())),
            "target_counts": dict(sorted(targets.items())),
        })

    negative = [(tick, clearance, entry) for tick, clearance, entry in rows if clearance < 0.0]
    negative_forward_requests = sum(
        1
        for _, _, entry in negative
        if str(entry.get("requested_primitive", entry.get("primitive", ""))).startswith("forward_")
    )
    negative_yaw_exec = sum(
        1
        for _, _, entry in negative
        if str(entry.get("primitive", "")) in PURE_YAW
    )
    return {
        "body_clearance_negative_intervals": {
            "count": int(len(intervals)),
            "max_len_ticks": int(max((len(segment) for segment in intervals), default=0)),
            "worst_intervals": interval_reports,
        },
        "body_clearance_negative_share_by_state": grouped_share("state"),
        "body_clearance_negative_share_by_primitive": grouped_share("primitive"),
        "body_clearance_negative_forward_request_ticks": int(negative_forward_requests),
        "body_clearance_negative_yaw_execution_ticks": int(negative_yaw_exec),
    }


def _gate(name: str, ok: bool, value: Any, limit: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(ok), "value": value, "limit": limit}


def _write_contact_sheet(video: Path, out_dir: Path, every: int) -> Path | None:
    try:
        import imageio.v3 as iio
        from PIL import Image, ImageDraw
    except Exception:
        return None
    frames: list[tuple[int, np.ndarray]] = []
    for idx, frame in enumerate(iio.imiter(video)):
        if idx % max(1, int(every)) == 0:
            frames.append((idx, np.asarray(frame)))
    if not frames:
        return None
    thumb_w = 224
    thumb_h = 126
    cols = 5
    rows = int(math.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for slot, (idx, frame) in enumerate(frames):
        img = Image.fromarray(frame[:, :, :3]).resize((thumb_w, thumb_h))
        x = (slot % cols) * thumb_w
        y = (slot // cols) * thumb_h
        sheet.paste(img, (x, y))
        draw.rectangle((x, y, x + 58, y + 16), fill=(0, 0, 0))
        draw.text((x + 4, y + 3), f"f{idx}", fill=(255, 255, 255))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "contact_sheet.jpg"
    sheet.save(out, quality=90)
    return out


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(args.result_json.read_text())
    result = data.get("result", {})
    log = data.get("log", [])
    explore = [e for e in log if e.get("state") == "EXPLORE"]
    commands = [str(e.get("primitive", "")) for e in explore]
    requested = [str(e.get("requested_primitive", e.get("primitive", ""))) for e in explore]
    state_counts = Counter(str(e.get("state", "")) for e in log)
    primitive_counts = Counter(commands)
    requested_counts = Counter(requested)
    xys = [e["post_xy"] for e in log if isinstance(e.get("post_xy"), list) and len(e["post_xy"]) == 2]
    poses = [
        (e["post_xy"], float(e["post_yaw"]))
        for e in log
        if isinstance(e.get("post_xy"), list)
        and len(e["post_xy"]) == 2
        and e.get("post_yaw") is not None
    ]

    scan_observed = any("scan_active" in (e.get("explorer") or {}) for e in explore)
    if scan_observed:
        scan_count = sum(1 for e in explore if bool((e.get("explorer") or {}).get("scan_active")))
        scan_metric = "logged_scan_active"
    else:
        scan_count = sum(
            1
            for e in explore
            if (e.get("explorer") or {}).get("bearing") is None
            and str(e.get("primitive", "")) in PURE_YAW
        )
        scan_metric = "legacy_bearingless_yaw_proxy"

    wall_metrics = result.get("wall_metrics", {})
    body_forward_m = (
        float(args.body_forward_m)
        if args.body_forward_m is not None
        else float(wall_metrics.get("wall_body_forward_m", 0.35))
    )
    body_half_width_m = (
        float(args.body_half_width_m)
        if args.body_half_width_m is not None
        else float(wall_metrics.get("wall_body_half_width_m", 0.18))
    )
    body_probe_margin_m = (
        float(args.body_probe_margin_m)
        if args.body_probe_margin_m is not None
        else float(wall_metrics.get("wall_body_probe_margin_m", 0.03))
    )

    analysis: dict[str, Any] = {
        "result_json": str(args.result_json),
        "scene": result.get("scene"),
        "success": bool(result.get("success")),
        "claimed": bool(result.get("claimed")),
        "first_seen_tick": result.get("first_seen_tick"),
        "ticks_used": result.get("ticks_used"),
        "final_dist_to_target_m": _round(result.get("final_dist_to_target_m")),
        "state_counts": dict(sorted(state_counts.items())),
        "explore_ticks": len(explore),
        "explore_primitive_counts": dict(sorted(primitive_counts.items())),
        "explore_requested_counts": dict(sorted(requested_counts.items())),
        "straight_forward_share_explore": _round(_share(sum(primitive_counts[p] for p in STRAIGHT_FORWARD), len(explore))),
        "translation_share_explore": _round(_share(sum(primitive_counts[p] for p in TRANSLATING), len(explore))),
        "pure_yaw_share_explore": _round(_share(sum(primitive_counts[p] for p in PURE_YAW), len(explore))),
        "scan_tick_share_explore": _round(_share(scan_count, len(explore))),
        "scan_metric": scan_metric,
        "max_same_primitive_streak_explore": _max_streak(commands),
        "max_pure_yaw_streak_explore": _max_streak(commands, PURE_YAW),
        "wall_vetoes": wall_metrics.get("wall_vetoes"),
        "escape_blocks_executed": wall_metrics.get("escape_blocks_executed"),
        "contact_like_stalls": wall_metrics.get("contact_like_stalls"),
        "hard_contact_like_stalls": wall_metrics.get("hard_contact_like_stalls"),
        "turn_loop_recoveries": wall_metrics.get("turn_loop_recoveries"),
        "runtime_wall_source": wall_metrics.get("source"),
        "runtime_wall_source_is_nonprivileged": wall_metrics.get("source") in (
            "learned_action_outcome",
            "learned_front_blocked",
        ),
        "body_probe_params": {
            "body_forward_m": _round(body_forward_m),
            "body_half_width_m": _round(body_half_width_m),
            "body_probe_margin_m": _round(body_probe_margin_m),
        },
    }
    analysis.update(_path_metrics(xys))
    target_xy = result.get("target_xy", {})
    if isinstance(target_xy, dict):
        analysis.update(_near_target_orbit_metrics(
            log,
            target_xy,
            radius_m=float(args.near_target_orbit_radius_m),
        ))
    grid = _load_grid(args, str(result.get("scene", "")))
    analysis.update(_clearance_metrics(grid, xys))
    analysis.update(_body_clearance_metrics(
        grid,
        poses,
        body_forward_m=body_forward_m,
        body_half_width_m=body_half_width_m,
        body_probe_margin_m=body_probe_margin_m,
    ))
    analysis.update(_body_clearance_diagnostics(
        grid,
        log,
        body_forward_m=body_forward_m,
        body_half_width_m=body_half_width_m,
        body_probe_margin_m=body_probe_margin_m,
    ))
    analysis.update(_forward_heading_diagnostics(
        grid,
        log,
        distance_m=float(args.front_body_sweep_m),
        step_m=float(args.front_body_sweep_step_m),
        body_forward_m=body_forward_m,
        body_half_width_m=body_half_width_m,
        body_probe_margin_m=body_probe_margin_m,
        warning_threshold_m=float(args.max_front_body_sweep_warning_m),
    ))

    clearance = analysis.get("wall_clearance_m") or {}
    body_clearance = analysis.get("body_configuration_clearance_m") or {}
    front_body_sweep = analysis.get("front_body_sweep_clearance_m") or {}
    enough_yaw_share_samples = len(explore) >= int(args.min_yaw_share_sample_ticks)
    gates = [
        _gate("runtime_wall_source_nonprivileged", bool(analysis["runtime_wall_source_is_nonprivileged"]), analysis["runtime_wall_source"], "learned_*"),
        _gate("success", bool(result.get("success")), bool(result.get("success")), True),
        _gate("first_seen_tick", result.get("first_seen_tick") is not None and int(result.get("first_seen_tick")) <= args.max_first_seen_tick, result.get("first_seen_tick"), f"<= {args.max_first_seen_tick}"),
        _gate("ticks_used", result.get("ticks_used") is not None and int(result.get("ticks_used")) <= args.max_ticks_used, result.get("ticks_used"), f"<= {args.max_ticks_used}"),
        _gate("path_tortuosity", analysis["path_tortuosity"] is not None and float(analysis["path_tortuosity"]) <= args.max_tortuosity, analysis["path_tortuosity"], f"<= {args.max_tortuosity}"),
        _gate("translation_share_explore", float(analysis["translation_share_explore"]) >= args.min_translation_share, analysis["translation_share_explore"], f">= {args.min_translation_share}"),
        _gate(
            "pure_yaw_share_explore",
            (not enough_yaw_share_samples) or float(analysis["pure_yaw_share_explore"]) <= args.max_pure_yaw_share,
            analysis["pure_yaw_share_explore"],
            f"<= {args.max_pure_yaw_share} when explore_ticks >= {args.min_yaw_share_sample_ticks}",
        ),
        _gate("scan_tick_share_explore", float(analysis["scan_tick_share_explore"]) <= args.max_scan_share, analysis["scan_tick_share_explore"], f"<= {args.max_scan_share}"),
        _gate("max_pure_yaw_streak_explore", int(analysis["max_pure_yaw_streak_explore"]) <= args.max_pure_yaw_streak, analysis["max_pure_yaw_streak_explore"], f"<= {args.max_pure_yaw_streak}"),
        _gate("hard_contact_like_stalls", int(wall_metrics.get("hard_contact_like_stalls") or 0) <= args.max_hard_stalls, wall_metrics.get("hard_contact_like_stalls"), f"<= {args.max_hard_stalls}"),
        _gate("contact_like_stalls", int(wall_metrics.get("contact_like_stalls") or 0) <= args.max_contact_stalls, wall_metrics.get("contact_like_stalls"), f"<= {args.max_contact_stalls}"),
    ]
    if clearance:
        gates.append(_gate(
            "wall_clearance_share_under_0p24",
            float(clearance["share_under_0p24"]) <= args.max_wall_close_share_0p24,
            clearance["share_under_0p24"],
            f"<= {args.max_wall_close_share_0p24}",
        ))
    if body_clearance:
        gates.append(_gate(
            "body_configuration_clearance_min",
            float(body_clearance["min"]) >= args.min_body_configuration_clearance_m,
            body_clearance["min"],
            f">= {args.min_body_configuration_clearance_m}",
        ))
    if front_body_sweep and args.max_front_body_sweep_warning_share is not None:
        gates.append(_gate(
            "front_body_sweep_warning_share",
            float(front_body_sweep["share_under_warning"]) <= args.max_front_body_sweep_warning_share,
            front_body_sweep["share_under_warning"],
            f"<= {args.max_front_body_sweep_warning_share}",
        ))
    passed = all(g["pass"] for g in gates)
    return {"analysis": analysis, "gates": gates, "passed": bool(passed)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--scene-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z")
    parser.add_argument("--platform-manifest", type=Path, default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--inflation-m", type=float, default=0.12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--frame-every", type=int, default=40)
    parser.add_argument("--max-first-seen-tick", type=int, default=200)
    parser.add_argument("--max-ticks-used", type=int, default=260)
    parser.add_argument("--max-tortuosity", type=float, default=5.0)
    parser.add_argument("--min-translation-share", type=float, default=0.42)
    parser.add_argument("--max-pure-yaw-share", type=float, default=0.50)
    parser.add_argument("--min-yaw-share-sample-ticks", type=int, default=40)
    parser.add_argument("--max-scan-share", type=float, default=0.12)
    parser.add_argument("--max-pure-yaw-streak", type=int, default=8)
    parser.add_argument("--max-hard-stalls", type=int, default=0)
    parser.add_argument("--max-contact-stalls", type=int, default=1)
    parser.add_argument("--max-wall-close-share-0p24", type=float, default=0.20)
    parser.add_argument("--body-forward-m", type=float, default=None,
                        help="Override body probe forward length. Defaults to the rollout's "
                             "recorded wall_body_forward_m when present.")
    parser.add_argument("--body-half-width-m", type=float, default=None,
                        help="Override body probe half width. Defaults to the rollout's "
                             "recorded wall_body_half_width_m when present.")
    parser.add_argument("--body-probe-margin-m", type=float, default=None,
                        help="Override body probe clearance margin. Defaults to the rollout's "
                             "recorded wall_body_probe_margin_m when present.")
    parser.add_argument("--min-body-configuration-clearance-m", type=float, default=-0.02)
    parser.add_argument("--front-body-sweep-m", type=float, default=0.35,
                        help="Offline forward body-sweep distance used to flag headings that "
                             "would carry the Go2 envelope into a wall.")
    parser.add_argument("--front-body-sweep-step-m", type=float, default=0.05)
    parser.add_argument("--max-front-body-sweep-warning-m", type=float, default=0.03,
                        help="Clearance threshold used for front-body-sweep warning intervals.")
    parser.add_argument("--max-front-body-sweep-warning-share", type=float, default=None,
                        help="Optional gate on the share of ticks whose forward body sweep is "
                             "below --max-front-body-sweep-warning-m.")
    parser.add_argument("--near-target-orbit-radius-m", type=float, default=1.8,
                        help="Radius around each active target used for the angular-sweep orbit proxy.")
    args = parser.parse_args()

    report = analyze(args)
    if args.video is not None and args.frame_dir is not None:
        sheet = _write_contact_sheet(args.video, args.frame_dir, args.frame_every)
        report["contact_sheet"] = str(sheet) if sheet is not None else None
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
