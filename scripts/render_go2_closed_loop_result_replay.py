#!/usr/bin/env python3
"""Replay a saved Go2 closed-loop primitive log and render it at sim cadence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_HOME = REPO_ROOT / ".generated" / "benchmark_home"
_CACHE_ROOT = REPO_ROOT / ".generated" / "cache"
_BENCH_HOME.mkdir(parents=True, exist_ok=True)
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
if not os.access(Path.home() / ".cache", os.W_OK):
    os.environ["HOME"] = str(_BENCH_HOME)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("TI_CACHE_HOME", str(_CACHE_ROOT / "taichi"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".generated" / "mplconfig"))

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    import yaml as _yaml  # noqa: F401
except ModuleNotFoundError:
    _system_dist_packages = Path("/usr/lib/python3/dist-packages")
    if _system_dist_packages.is_dir():
        sys.path.append(str(_system_dist_packages))
        import yaml as _yaml  # noqa: F401
        sys.path.remove(str(_system_dist_packages))

from benchmark_go2_memory_closed_loop import _scene_spawn  # noqa: E402
from benchmark_lewm_closed_loop_mpc import (  # noqa: E402
    _current_pose,
    _execute_physical_primitive,
    _quat_wxyz_from_yaw,
    _render_synced_third_person,
    _render_tensor_from_base,
    _set_pose,
    _yaw_from_quat_wxyz,
)
from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits  # noqa: E402
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import find_scene_dirs, load_platform_manifest, load_scene_pack  # noqa: E402


class StreamingVideoSink:
    def __init__(self, path: Path, fps: float, ui: "ReplayReviewUi | None" = None) -> None:
        import imageio

        self.path = path
        self.fps = float(fps)
        self.ui = ui
        self.count = 0
        self.first_pose: tuple[float, float, float] | None = None
        self.last_pose: tuple[float, float, float] | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=self.fps, macro_block_size=8)

    def append(self, frame: tuple[np.ndarray, np.ndarray, float, float, float]) -> None:
        third_np, ego_np, x, y, yaw, *_rest = frame
        if self.ui is not None:
            out = self.ui.compose(
                third_np=third_np,
                ego_np=ego_np,
                x=float(x),
                y=float(y),
                yaw=float(yaw),
                frame_index=int(self.count),
                fps=float(self.fps),
            )
        else:
            ego_up = np.asarray(
                F.interpolate(
                    torch.from_numpy(ego_np).permute(2, 0, 1)[None].float(),
                    size=(third_np.shape[0], third_np.shape[1]),
                    mode="nearest",
                )[0]
                .permute(1, 2, 0)
                .byte()
            )
            out = np.concatenate([third_np, ego_up], axis=1)
        self._writer.append_data(out)
        pose = (float(x), float(y), float(yaw))
        if self.first_pose is None:
            self.first_pose = pose
        self.last_pose = pose
        self.count += 1

    def close(self) -> None:
        self._writer.close()


class ReplayReviewUi:
    COLOR_RGB = {
        "red": (232, 76, 81),
        "green": (64, 190, 112),
        "blue": (72, 142, 238),
        "yellow": (240, 198, 66),
    }

    def __init__(
        self,
        *,
        result: dict[str, Any],
        pose_entries: list[dict[str, Any]],
        scene_manifest: dict[str, Any],
        frames_per_entry: int,
        total_frames: int | None = None,
        title: str | None = None,
        status_note: str | None = None,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        from PIL import ImageFont

        self.result = result
        self.pose_entries = pose_entries
        self.scene_manifest = scene_manifest
        self.frames_per_entry = max(1, int(frames_per_entry))
        self.total_frames = None if total_frames is None else max(1, int(total_frames))
        self.title = str(title) if title else None
        self.status_note = str(status_note) if status_note else None
        self.width = int(width)
        self.height = int(height)
        self.font = ImageFont.load_default()
        self.target_xy = self._target_xy_from_result_or_manifest()
        self.claims = {
            str(item.get("target_color", "")).lower(): item
            for item in result.get("beacon_claims", [])
            if isinstance(item, dict)
        }
        self.path_points = [
            (
                int(entry.get("tick", idx)),
                float(entry["post_xy"][0]),
                float(entry["post_xy"][1]),
            )
            for idx, entry in enumerate(pose_entries)
            if isinstance(entry.get("post_xy"), list) and len(entry["post_xy"]) >= 2
        ]
        self.bounds = self._world_bounds()

    def compose(
        self,
        *,
        third_np: np.ndarray,
        ego_np: np.ndarray,
        x: float,
        y: float,
        yaw: float,
        frame_index: int,
        fps: float,
    ) -> np.ndarray:
        from PIL import Image, ImageDraw

        entry = self.entry_for_frame(frame_index)
        tick = int(entry.get("tick", frame_index // self.frames_per_entry))
        canvas = Image.new("RGB", (self.width, self.height), (15, 17, 20))
        draw = ImageDraw.Draw(canvas)

        exo_box = (16, 54, 820, 430)
        ego_box = (844, 54, 420, 220)
        map_box = (844, 318, 420, 306)
        status_box = (16, 522, 820, 166)

        self._draw_title(draw, tick=tick, frame_index=frame_index, fps=fps)
        self._paste_frame(canvas, draw, third_np, exo_box, "EXOCENTRIC VIEW")
        self._paste_frame(canvas, draw, ego_np, ego_box, "EGO RGB VIEW")
        self._draw_minimap(draw, map_box, x=x, y=y, yaw=yaw, tick=tick)
        self._draw_status(draw, status_box, entry=entry, tick=tick)
        self._draw_claim_column(draw, (844, 650, 420, 54), tick=tick)
        return np.asarray(canvas, dtype=np.uint8)

    def entry_for_frame(self, frame_index: int) -> dict[str, Any]:
        if not self.pose_entries:
            return {}
        if self.total_frames is not None:
            idx = int(float(max(0, int(frame_index))) * float(len(self.pose_entries)) / float(self.total_frames))
            idx = min(len(self.pose_entries) - 1, max(0, idx))
        else:
            idx = min(len(self.pose_entries) - 1, max(0, int(frame_index) // self.frames_per_entry))
        return self.pose_entries[idx]

    def _target_xy_from_result_or_manifest(self) -> dict[str, tuple[float, float]]:
        target_xy: dict[str, tuple[float, float]] = {}
        raw = self.result.get("target_xy")
        if isinstance(raw, dict):
            for color, value in raw.items():
                if isinstance(value, list) and len(value) >= 2:
                    target_xy[str(color).lower()] = (float(value[0]), float(value[1]))
        for item in self.scene_manifest.get("landmarks", []):
            if not isinstance(item, dict):
                continue
            material = str(item.get("material_id", ""))
            color = material.removeprefix("landmark_").lower()
            center = item.get("center_xyz_m")
            if color and isinstance(center, list) and len(center) >= 2:
                target_xy.setdefault(color, (float(center[0]), float(center[1])))
        return target_xy

    def _world_bounds(self) -> tuple[float, float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        bounds = self.scene_manifest.get("world_bounds_xy_m")
        if isinstance(bounds, list) and len(bounds) >= 2:
            try:
                xs.extend([float(bounds[0][0]), float(bounds[1][0])])
                ys.extend([float(bounds[0][1]), float(bounds[1][1])])
            except Exception:
                pass
        for _, x, y in self.path_points:
            xs.append(float(x))
            ys.append(float(y))
        for x, y in self.target_xy.values():
            xs.append(float(x))
            ys.append(float(y))
        for obj in self.scene_manifest.get("walls", []):
            center = obj.get("center_xyz_m") if isinstance(obj, dict) else None
            size = obj.get("size_xyz_m") if isinstance(obj, dict) else None
            if isinstance(center, list) and isinstance(size, list) and len(center) >= 2 and len(size) >= 2:
                xs.extend([float(center[0]) - float(size[0]) / 2.0, float(center[0]) + float(size[0]) / 2.0])
                ys.extend([float(center[1]) - float(size[1]) / 2.0, float(center[1]) + float(size[1]) / 2.0])
        if not xs or not ys:
            return (-3.0, 3.0, -3.0, 3.0)
        pad = 0.35
        return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    def _draw_title(self, draw: Any, *, tick: int, frame_index: int, fps: float) -> None:
        colors = ",".join(str(item) for item in self.result.get("claimed_colors", []))
        prefix = self.title or f"Go2 learned-nav review | {self.result.get('scene', 'unknown scene')}"
        title = f"{prefix} | tick {tick}/{self.result.get('ticks_used')} | claims {colors}"
        draw.rectangle((0, 0, self.width, 34), fill=(10, 12, 15))
        draw.text((18, 12), title, fill=(235, 238, 242), font=self.font)
        draw.text((1042, 12), f"{frame_index / max(1.0, fps):.1f}s", fill=(165, 174, 185), font=self.font)

    def _paste_frame(self, canvas: Any, draw: Any, frame: np.ndarray, box: tuple[int, int, int, int], label: str) -> None:
        from PIL import Image

        x, y, w, h = box
        draw.rectangle((x - 2, y - 20, x + w + 2, y + h + 2), fill=(30, 34, 40), outline=(72, 80, 92))
        draw.text((x, y - 17), label, fill=(210, 218, 228), font=self.font)
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((w, h))
        canvas.paste(image, (x, y))

    def _draw_minimap(self, draw: Any, box: tuple[int, int, int, int], *, x: float, y: float, yaw: float, tick: int) -> None:
        x0, y0, w, h = box
        draw.rectangle((x0 - 2, y0 - 20, x0 + w + 2, y0 + h + 2), fill=(26, 29, 34), outline=(72, 80, 92))
        draw.text((x0, y0 - 17), "MINIMAP", fill=(210, 218, 228), font=self.font)
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(12, 14, 17), outline=(52, 60, 70))

        def wp(wx: float, wy: float) -> tuple[int, int]:
            min_x, max_x, min_y, max_y = self.bounds
            pad = 18
            scale = min((w - 2 * pad) / max(1e-6, max_x - min_x), (h - 2 * pad) / max(1e-6, max_y - min_y))
            px = x0 + pad + (float(wx) - min_x) * scale
            py = y0 + h - pad - (float(wy) - min_y) * scale
            return int(round(px)), int(round(py))

        for obj in self.scene_manifest.get("walls", []):
            if not isinstance(obj, dict):
                continue
            center = obj.get("center_xyz_m")
            size = obj.get("size_xyz_m")
            if not (isinstance(center, list) and isinstance(size, list) and len(center) >= 2 and len(size) >= 2):
                continue
            ax, ay = wp(float(center[0]) - float(size[0]) / 2.0, float(center[1]) - float(size[1]) / 2.0)
            bx, by = wp(float(center[0]) + float(size[0]) / 2.0, float(center[1]) + float(size[1]) / 2.0)
            draw.rectangle((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)), fill=(70, 76, 84))

        path = [(px, py) for t, px, py in self.path_points if t <= tick]
        if len(path) >= 2:
            draw.line([wp(px, py) for px, py in path], fill=(220, 226, 235), width=2)

        for color, (tx, ty) in self.target_xy.items():
            px, py = wp(tx, ty)
            claimed = self._claimed_by(color, tick)
            fill = self.COLOR_RGB.get(color, (180, 185, 195))
            r = 8 if claimed else 6
            draw.ellipse((px - r, py - r, px + r, py + r), fill=fill, outline=(255, 255, 255) if claimed else (40, 45, 50), width=2)
            if claimed:
                draw.line((px - 4, py, px - 1, py + 4, px + 5, py - 5), fill=(255, 255, 255), width=2)

        rx, ry = wp(x, y)
        heading = [
            (rx + int(13 * math.cos(yaw)), ry - int(13 * math.sin(yaw))),
            (rx + int(7 * math.cos(yaw + 2.45)), ry - int(7 * math.sin(yaw + 2.45))),
            (rx + int(7 * math.cos(yaw - 2.45)), ry - int(7 * math.sin(yaw - 2.45))),
        ]
        draw.polygon(heading, fill=(255, 255, 255), outline=(0, 0, 0))

    def _draw_status(self, draw: Any, box: tuple[int, int, int, int], *, entry: dict[str, Any], tick: int) -> None:
        x, y, w, h = box
        draw.rectangle((x - 2, y - 20, x + w + 2, y + h + 2), fill=(26, 29, 34), outline=(72, 80, 92))
        draw.text((x, y - 17), "POLICY STATE", fill=(210, 218, 228), font=self.font)
        draw.rectangle((x, y, x + w, y + h), fill=(18, 21, 25))

        state = str(entry.get("state", "?"))
        target = str(entry.get("target_color", "?"))
        primitive = str(entry.get("primitive", "?"))
        area = entry.get("area")
        bearing = entry.get("bearing")
        read_score = entry.get("read_score")
        gate = entry.get("claim_gate") if isinstance(entry.get("claim_gate"), dict) else {}
        proxy = gate.get("success_proxy", {}) if isinstance(gate, dict) else {}
        model_score = proxy.get("model_score") if isinstance(proxy, dict) else None
        status = [
            f"state {state}",
            f"target {target}",
            f"primitive {primitive}",
            f"claim-head {model_score if model_score is not None else '-'}",
            f"read {self._fmt(read_score)}",
            f"area {self._fmt(area)}",
            f"bearing {self._fmt(bearing)}",
        ]
        for idx, text in enumerate(status):
            draw.text((x + 18, y + 18 + idx * 18), text, fill=(225, 230, 236), font=self.font)

        max_tick = max(1, int(self.result.get("ticks_used") or tick or 1))
        progress = min(1.0, max(0.0, float(tick) / float(max_tick)))
        bar_x = x + 350
        bar_y = y + 24
        bar_w = w - 390
        draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + 14), fill=(45, 50, 58))
        draw.rectangle((bar_x, bar_y, bar_x + int(bar_w * progress), bar_y + 14), fill=(95, 160, 245))
        note = self.status_note or "review overlay from closed-loop result log"
        draw.text((bar_x, bar_y + 24), note, fill=(165, 174, 185), font=self.font)
        if str(state).upper() == "CLAIM":
            draw.rectangle((bar_x, bar_y + 52, bar_x + 230, bar_y + 82), fill=(72, 132, 82), outline=(190, 245, 205))
            draw.text((bar_x + 10, bar_y + 61), f"CLAIM {target.upper()} accepted", fill=(255, 255, 255), font=self.font)

    def _draw_claim_column(self, draw: Any, box: tuple[int, int, int, int], *, tick: int) -> None:
        x, y, w, h = box
        draw.rectangle((x - 2, y - 20, x + w + 2, y + h + 2), fill=(26, 29, 34), outline=(72, 80, 92))
        draw.text((x, y - 17), "CLAIMS", fill=(210, 218, 228), font=self.font)
        order = ["green", "yellow", "blue", "red"]
        slot_w = w // len(order)
        for idx, color in enumerate(order):
            sx = x + idx * slot_w + 8
            claimed = self._claimed_by(color, tick)
            claim = self.claims.get(color, {})
            fill = self.COLOR_RGB.get(color, (180, 185, 195)) if claimed else (60, 66, 74)
            draw.rectangle((sx, y, sx + slot_w - 16, y + h), fill=(18, 21, 25), outline=(64, 72, 82))
            draw.ellipse((sx + 8, y + 10, sx + 28, y + 30), fill=fill)
            label = color.upper()
            if claimed:
                label += f" t{claim.get('tick')}"
            draw.text((sx + 36, y + 9), label, fill=(235, 238, 242), font=self.font)
            dist = claim.get("dist_to_target_m")
            if claimed and dist is not None:
                draw.text((sx + 36, y + 27), f"{float(dist):.2f}m", fill=(165, 174, 185), font=self.font)
            elif not claimed:
                draw.text((sx + 36, y + 27), "pending", fill=(120, 128, 138), font=self.font)

    def _claimed_by(self, color: str, tick: int) -> bool:
        claim = self.claims.get(str(color).lower())
        return bool(isinstance(claim, dict) and int(claim.get("tick", 10**9)) <= int(tick))

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return "-"


def _roll_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[-4], q[-3], q[-2], q[-1]
    return float(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


def _pitch_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[-4], q[-3], q[-2], q[-1]
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        return float(math.copysign(math.pi / 2.0, sinp))
    return float(math.asin(sinp))


def _load_replay_log(result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    payload = json.loads(result_path.read_text())
    result = payload.get("result", payload if isinstance(payload, dict) else {})
    log = payload.get("log", [])
    if not isinstance(log, list):
        raise ValueError(f"{result_path} does not contain a list-valued log")
    primitives: list[str] = []
    for entry in log:
        if not isinstance(entry, dict):
            continue
        primitive = entry.get("primitive")
        if primitive is None:
            continue
        primitive_name = str(primitive)
        if not primitive_name or primitive_name.lower() == "none":
            continue
        primitives.append(primitive_name)
    log_entries = [entry for entry in log if isinstance(entry, dict)]
    return result, log_entries, primitives


def _pose_entries_from_log(
    log: list[dict[str, Any]],
    max_entries: int | None = None,
    *,
    include_claim_only: bool = False,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in log:
        primitive = entry.get("primitive")
        post_xy = entry.get("post_xy")
        if primitive is None or post_xy is None:
            if not (
                include_claim_only
                and entries
                and str(entry.get("state", "")).upper() == "CLAIM"
            ):
                continue
            last = dict(entries[-1])
            last.update(
                {
                    "tick": int(entry.get("tick", last.get("tick", len(entries)))),
                    "state": str(entry.get("state", last.get("state", ""))),
                    "target_color": str(entry.get("target_color", last.get("target_color", ""))),
                    "target_index": entry.get("target_index", last.get("target_index")),
                    "primitive": str(entry.get("primitive") or "claim"),
                    "requested_primitive": entry.get("requested_primitive", last.get("requested_primitive")),
                    "mem_conf": entry.get("mem_conf", last.get("mem_conf")),
                    "area": entry.get("area", last.get("area")),
                    "bearing": entry.get("bearing", last.get("bearing")),
                    "seen_age_ticks": entry.get("seen_age_ticks", last.get("seen_age_ticks")),
                    "read_score": entry.get("read_score", last.get("read_score")),
                    "claim_gate": entry.get("claim_gate", last.get("claim_gate")),
                    "color_readouts": entry.get("color_readouts", last.get("color_readouts")),
                }
            )
            entries.append(last)
            if max_entries is not None and len(entries) >= max(0, int(max_entries)):
                break
            continue
        try:
            xy = [float(post_xy[0]), float(post_xy[1])]
            pose_entry = {
                "tick": int(entry.get("tick", len(entries))),
                "state": str(entry.get("state", "")),
                "target_color": str(entry.get("target_color", "")),
                "target_index": entry.get("target_index"),
                "primitive": str(primitive),
                "requested_primitive": entry.get("requested_primitive"),
                "mem_conf": entry.get("mem_conf"),
                "area": entry.get("area"),
                "bearing": entry.get("bearing"),
                "seen": entry.get("seen"),
                "in_cone": entry.get("in_cone"),
                "seen_age_ticks": entry.get("seen_age_ticks"),
                "read_score": entry.get("read_score"),
                "claim_gate": entry.get("claim_gate"),
                "color_readouts": entry.get("color_readouts"),
                "wall_guard": entry.get("wall_guard"),
                "stalled": entry.get("stalled"),
                "hard_stalled": entry.get("hard_stalled"),
                "post_xy": xy,
                "post_z": float(entry.get("post_z", 0.34)),
                "post_yaw": float(entry.get("post_yaw", 0.0)),
                "post_roll": float(entry.get("post_roll", 0.0)),
                "post_pitch": float(entry.get("post_pitch", 0.0)),
            }
        except Exception:
            continue
        entries.append(pose_entry)
        if max_entries is not None and len(entries) >= max(0, int(max_entries)):
            break
    return entries


def _wrap_pi(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(0.5 * float(roll))
    sr = math.sin(0.5 * float(roll))
    cp = math.cos(0.5 * float(pitch))
    sp = math.sin(0.5 * float(pitch))
    cy = math.cos(0.5 * float(yaw))
    sy = math.sin(0.5 * float(yaw))
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def _set_robot_pose(build: Any, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
    build.robot.set_pos(np.asarray(pos_xyz, dtype=np.float32)[None, :], envs_idx=[0], zero_velocity=True)
    build.robot.set_quat(np.asarray(quat_wxyz, dtype=np.float32)[None, :], envs_idx=[0], zero_velocity=False)


def _render_recorded_pose_frame(
    *,
    build: Any,
    pack: Any,
    sink: StreamingVideoSink,
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
    yaw: float,
    device: torch.device,
    third_person_build: Any | None,
    leg_dof_idx: Any,
) -> None:
    _set_robot_pose(build, pos_xyz, quat_wxyz)
    if third_person_build is None:
        try:
            build.scene.step()
        except Exception:
            pass
    ego = _render_tensor_from_base(build, pack, base_xyz_m=pos_xyz, base_quat_wxyz=quat_wxyz, device=device)
    ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    third_np = _render_synced_third_person(
        source_build=build,
        render_build=third_person_build,
        base_xyz=pos_xyz,
        base_quat_wxyz=quat_wxyz,
        yaw=float(yaw),
        leg_dof_idx=leg_dof_idx,
    )
    sink.append((third_np, ego_np, float(pos_xyz[0]), float(pos_xyz[1]), float(yaw)))


def _select_scene(
    *,
    scene_corpus: Path,
    split: str,
    family: str,
    scene_id: str,
) -> Path:
    scene_dirs = find_scene_dirs(scene_corpus.resolve(), split=split, family=family)
    matches = [path for path in scene_dirs if path.name == scene_id]
    if not matches:
        raise SystemExit(f"scene {scene_id!r} not found in {scene_corpus}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--demo-video", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z",
    )
    parser.add_argument("--platform-manifest", type=Path, default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--primitive-registry", type=Path, default=REPO_ROOT / "config/go2_primitive_registry.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--apply-textures", action="store_true")
    parser.add_argument(
        "--render-robot",
        action="store_true",
        help=(
            "Legacy/debug mode: render Go2 visual meshes in the main camera "
            "scene, including egocentric replay RGB. By default ego RGB hides "
            "the robot body; demo videos use a separate robot-visible scene "
            "for the third-person panel."
        ),
    )
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15)
    parser.add_argument("--tip-threshold-rad", type=float, default=math.radians(60.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture-rate", choices=("command", "policy"), default="policy")
    parser.add_argument("--demo-fps", type=float, default=None)
    parser.add_argument("--max-primitives", type=int, default=None)
    parser.add_argument("--review-ui", action="store_true",
                        help="Render a review canvas with exocentric view, ego RGB, "
                             "minimap, path, and claim indicators.")
    parser.add_argument("--ui-width", type=int, default=1280)
    parser.add_argument("--ui-height", type=int, default=720)
    parser.add_argument(
        "--replay-mode",
        choices=("recorded", "physical"),
        default="physical",
        help=(
            "physical re-executes primitive names through Genesis with the Go2 "
            "locomotion policy; recorded renders saved post-pose trajectories "
            "exactly and can hide locomotion contacts/drift."
        ),
    )
    parser.add_argument(
        "--allow-recorded-replay",
        action="store_true",
        help="Allow diagnostic recorded-pose replay. Demo renders reject it by default.",
    )
    parser.add_argument(
        "--allow-slice-result",
        action="store_true",
        help="Allow rendering a benchmark slice result. Full demo renders reject slices by default.",
    )
    parser.add_argument(
        "--use-slice-start",
        action="store_true",
        help=(
            "If the result is a benchmark slice, start replay from "
            "result.wall_metrics.slice_start instead of the scene spawn. "
            "This keeps physical post-claim slice review videos aligned with "
            "the evaluated locomotion-policy rollout."
        ),
    )
    parser.add_argument(
        "--include-claim-only-frames",
        action="store_true",
        help=(
            "In recorded replay mode, include claim-only log rows as held-pose "
            "frames so review overlays can show accepted claim states."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    result, log_entries, primitives = _load_replay_log(args.result.resolve())
    wall_metrics = result.get("wall_metrics", {}) if isinstance(result, dict) else {}
    if not isinstance(wall_metrics, dict):
        wall_metrics = {}
    source_slice_benchmark = bool(wall_metrics.get("slice_benchmark") or wall_metrics.get("slice_start"))
    if source_slice_benchmark and not bool(args.allow_slice_result):
        raise SystemExit(
            "refusing to render benchmark slice as a full demo; pass --allow-slice-result "
            "only for diagnostic slice review"
        )
    if str(args.replay_mode) == "recorded" and not bool(args.allow_recorded_replay):
        raise SystemExit(
            "refusing recorded-pose replay as a demo render; pass --allow-recorded-replay "
            "only for diagnostic review"
        )
    if bool(args.use_slice_start) and not bool(args.allow_slice_result):
        raise SystemExit("--use-slice-start requires --allow-slice-result")
    if args.max_primitives is not None:
        primitives = primitives[: max(0, int(args.max_primitives))]
    include_claim_only_frames = bool(args.include_claim_only_frames) and str(args.replay_mode) == "recorded"
    if not primitives:
        raise SystemExit("no executable primitives found in result log")
    pose_entries = _pose_entries_from_log(
        log_entries,
        max_entries=None if args.max_primitives is None else int(args.max_primitives),
        include_claim_only=include_claim_only_frames,
    )
    if str(args.replay_mode) == "recorded":
        if not pose_entries:
            raise SystemExit("recorded replay requested but result log has no post_xy/post_yaw poses")
        primitives = [str(entry["primitive"]) for entry in pose_entries]

    scene_id = str(args.scene_id or result.get("scene") or "")
    if not scene_id:
        raise SystemExit("scene id missing; pass --scene-id or use a result with result.scene")

    platform = load_platform_manifest(args.platform_manifest.resolve())
    scene_dir = _select_scene(
        scene_corpus=args.scene_corpus,
        split=str(args.split),
        family=str(args.family),
        scene_id=scene_id,
    )
    pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    safety = SafetyLimits.from_manifest(platform)
    build = build_scene_from_pack(
        pack,
        n_envs=1,
        backend=str(args.backend),
        show_viewer=False,
        render_robot=bool(args.render_robot),
        apply_textures=bool(args.apply_textures),
    )
    third_person_build = None
    if args.demo_video is not None and not bool(args.render_robot):
        third_person_build = build_scene_from_pack(
            pack,
            n_envs=1,
            backend=str(args.backend),
            show_viewer=False,
            render_robot=True,
            apply_textures=bool(args.apply_textures),
        )
    policy = GenesisGo2PPOPolicy.from_platform_manifest(platform, REPO_ROOT, device=str(args.policy_device))
    runner = RolloutRunner(
        build,
        policy,
        registry,
        safety,
        config=RolloutConfig(
            n_blocks=len(primitives),
            fall_z_threshold_m=float(args.fall_z_threshold_m),
            rgb_capture_per_block=False,
            seed=int(args.seed),
            log_progress_every_blocks=0,
            foot_contact_source="zero",
            randomize_spawn_pose=False,
        ),
    )
    spawn_pos, spawn_quat = _scene_spawn(scene_dir)
    replay_start_source = "scene_spawn"
    if bool(args.use_slice_start):
        slice_start = wall_metrics.get("slice_start") if isinstance(wall_metrics, dict) else None
        if not isinstance(slice_start, dict):
            raise SystemExit("--use-slice-start requested but result has no wall_metrics.slice_start")
        start_xy = slice_start.get("start_xy")
        start_yaw = slice_start.get("start_yaw")
        if (
            not isinstance(start_xy, list)
            or len(start_xy) < 2
            or start_yaw is None
        ):
            raise SystemExit("--use-slice-start result slice_start lacks start_xy/start_yaw")
        spawn_pos = np.asarray(
            [float(start_xy[0]), float(start_xy[1]), float(spawn_pos[2])],
            dtype=np.float32,
        )
        spawn_quat = _quat_wxyz_from_yaw(float(start_yaw))
        replay_start_source = "slice_start"
    _set_pose(build=build, runner=runner, pos_xyz=spawn_pos, quat_wxyz=spawn_quat)

    if args.demo_fps is not None:
        demo_fps = float(args.demo_fps)
    elif str(args.capture_rate) == "policy":
        demo_fps = 1.0 / float(pack.timing.policy_dt_s)
    else:
        demo_fps = 1.0 / float(pack.timing.command_dt_s)

    frames_per_primitive = int(registry.block_size)
    if str(args.capture_rate) == "policy":
        frames_per_primitive *= int(pack.timing.command_ticks_per_block)
    expected_frames = int(len(primitives) * frames_per_primitive)
    print(
        f"replay scene={scene_id} mode={args.replay_mode} primitives={len(primitives)} "
        f"capture={args.capture_rate} expected_frames={expected_frames} fps={demo_fps:.2f}",
        flush=True,
    )

    review_ui = None
    if bool(args.review_ui):
        manifest_path = scene_dir / "manifest.json"
        scene_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        review_ui = ReplayReviewUi(
            result=result,
            pose_entries=pose_entries,
            scene_manifest=scene_manifest,
            frames_per_entry=frames_per_primitive,
            width=int(args.ui_width),
            height=int(args.ui_height),
        )
    sink = (
        StreamingVideoSink(args.demo_video, demo_fps, ui=review_ui)
        if args.demo_video is not None
        else None
    )
    min_base_z = float("inf")
    max_abs_roll_pitch = 0.0
    fall_events = 0
    tip_events = 0
    first_unstable: dict[str, Any] | None = None
    try:
        if str(args.replay_mode) == "recorded":
            if sink is None:
                raise SystemExit("recorded replay requires --demo-video")
            prev_pos = np.asarray(spawn_pos, dtype=np.float32).copy()
            prev_yaw = _yaw_from_quat_wxyz(spawn_quat)
            prev_roll = 0.0
            prev_pitch = 0.0
            for index, entry in enumerate(pose_entries, start=1):
                end_pos = np.asarray(
                    [entry["post_xy"][0], entry["post_xy"][1], entry["post_z"]],
                    dtype=np.float32,
                )
                end_yaw = float(entry["post_yaw"])
                end_roll = float(entry["post_roll"])
                end_pitch = float(entry["post_pitch"])
                dyaw = _wrap_pi(end_yaw - prev_yaw)
                for step_idx in range(1, frames_per_primitive + 1):
                    alpha = float(step_idx) / float(frames_per_primitive)
                    pos = prev_pos * (1.0 - alpha) + end_pos * alpha
                    yaw = _wrap_pi(prev_yaw + dyaw * alpha)
                    roll = float(prev_roll * (1.0 - alpha) + end_roll * alpha)
                    pitch = float(prev_pitch * (1.0 - alpha) + end_pitch * alpha)
                    quat = _quat_wxyz_from_rpy(roll, pitch, yaw)
                    _render_recorded_pose_frame(
                        build=build,
                        pack=pack,
                        sink=sink,
                        pos_xyz=pos,
                        quat_wxyz=quat,
                        yaw=yaw,
                        device=torch.device("cpu"),
                        third_person_build=third_person_build,
                        leg_dof_idx=runner._leg_dof_idx,
                    )
                    tip = max(abs(roll), abs(pitch))
                    min_base_z = min(min_base_z, float(pos[2]))
                    max_abs_roll_pitch = max(max_abs_roll_pitch, tip)
                    fell = float(pos[2]) < float(args.fall_z_threshold_m)
                    tipped = tip > float(args.tip_threshold_rad)
                    if fell:
                        fall_events += 1
                    if tipped:
                        tip_events += 1
                    if first_unstable is None and (fell or tipped):
                        first_unstable = {
                            "primitive_index": index - 1,
                            "frame_index": int(sink.count - 1),
                            "time_s": float(sink.count) / float(demo_fps),
                            "primitive": str(entry["primitive"]),
                            "reason": "fall_and_tip" if fell and tipped else ("fall" if fell else "tip"),
                            "base_z_m": float(pos[2]),
                            "roll_rad": float(roll),
                            "pitch_rad": float(pitch),
                            "tip_rad": float(tip),
                        }
                prev_pos = end_pos
                prev_yaw = end_yaw
                prev_roll = end_roll
                prev_pitch = end_pitch
                if args.progress_every > 0 and (index == 1 or index % int(args.progress_every) == 0):
                    print(f"rendered {index}/{len(pose_entries)} recorded poses ({sink.count} frames)", flush=True)
        else:
            for index, primitive in enumerate(primitives, start=1):
                _execute_physical_primitive(
                    runner,
                    registry,
                    primitive,
                    frame_sink=sink,
                    build=build,
                    pack=pack,
                    device=torch.device("cpu"),
                    capture_policy_steps=(str(args.capture_rate) == "policy"),
                    third_person_build=third_person_build,
                )
                pos, quat = _current_pose(build)
                roll = _roll_from_quat_wxyz(quat)
                pitch = _pitch_from_quat_wxyz(quat)
                tip = max(abs(roll), abs(pitch))
                min_base_z = min(min_base_z, float(pos[2]))
                max_abs_roll_pitch = max(max_abs_roll_pitch, tip)
                fell = float(pos[2]) < float(args.fall_z_threshold_m)
                tipped = tip > float(args.tip_threshold_rad)
                if fell:
                    fall_events += 1
                if tipped:
                    tip_events += 1
                if first_unstable is None and (fell or tipped):
                    first_unstable = {
                        "primitive_index": index - 1,
                        "time_s": float(index) * float(registry.block_size) * float(pack.timing.command_dt_s),
                        "primitive": primitive,
                        "reason": "fall_and_tip" if fell and tipped else ("fall" if fell else "tip"),
                        "base_z_m": float(pos[2]),
                        "roll_rad": float(roll),
                        "pitch_rad": float(pitch),
                        "tip_rad": float(tip),
                    }
                if args.progress_every > 0 and (index == 1 or index % int(args.progress_every) == 0):
                    frame_count = 0 if sink is None else sink.count
                    print(f"rendered {index}/{len(primitives)} primitives ({frame_count} frames)", flush=True)
    finally:
        if sink is not None:
            sink.close()

    final_pos, final_quat = _current_pose(build)
    frame_count = 0 if sink is None else sink.count
    duration_s = float(frame_count) / float(demo_fps) if frame_count else (
        float(len(primitives)) * float(registry.block_size) * float(pack.timing.command_dt_s)
    )
    unstable_events = int(fall_events + tip_events)
    report = {
        "source_result": str(args.result),
        "video": None if args.demo_video is None else str(args.demo_video),
        "scene": scene_id,
        "source_slice_benchmark": bool(source_slice_benchmark),
        "source_slice_start": wall_metrics.get("slice_start"),
        "replay_mode": str(args.replay_mode),
        "locomotion_policy_replayed": str(args.replay_mode) == "physical",
        "recorded_pose_interpolation": str(args.replay_mode) == "recorded",
        "replay_start_source": replay_start_source,
        "replay_start_xy": [float(spawn_pos[0]), float(spawn_pos[1])],
        "replay_start_yaw": float(_yaw_from_quat_wxyz(spawn_quat)),
        "capture_rate": str(args.capture_rate),
        "fps": demo_fps,
        "frame_count": int(frame_count),
        "expected_frame_count": expected_frames,
        "duration_s": duration_s,
        "primitive_count": len(primitives),
        "recorded_pose_count": len(pose_entries),
        "include_claim_only_frames": bool(include_claim_only_frames),
        "registry_block_size": int(registry.block_size),
        "policy_dt_s": float(pack.timing.policy_dt_s),
        "command_dt_s": float(pack.timing.command_dt_s),
        "fall_z_threshold_m": float(args.fall_z_threshold_m),
        "tip_threshold_rad": float(args.tip_threshold_rad),
        "stable_base": bool(unstable_events == 0),
        "fall_events": int(fall_events),
        "tip_events": int(tip_events),
        "unstable_base_events": int(unstable_events),
        "min_base_z_m": None if min_base_z == float("inf") else float(min_base_z),
        "max_abs_roll_pitch_rad": float(max_abs_roll_pitch),
        "first_unstable": first_unstable,
        "final_xy": [float(final_pos[0]), float(final_pos[1])],
        "final_yaw": float(_yaw_from_quat_wxyz(final_quat)),
        "source_final_xy": result.get("final_xy"),
        "source_claimed_colors": result.get("claimed_colors"),
        "first_render_pose": None if sink is None else sink.first_pose,
        "last_render_pose": None if sink is None else sink.last_pose,
    }
    if result.get("final_xy") is not None:
        try:
            src_xy = np.asarray(result["final_xy"], dtype=np.float64)
            out_xy = np.asarray(report["final_xy"], dtype=np.float64)
            report["source_final_xy_error_m"] = float(np.linalg.norm(src_xy[:2] - out_xy[:2]))
        except Exception:
            report["source_final_xy_error_m"] = None
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
