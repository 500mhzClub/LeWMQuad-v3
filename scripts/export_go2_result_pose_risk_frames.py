#!/usr/bin/env python3
"""Render selected recorded Go2 result poses into risk-classifier RGB rows."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_lewm_closed_loop_mpc import _render_tensor_from_base  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from render_go2_closed_loop_result_replay import (  # noqa: E402
    _pose_entries_from_log,
    _quat_wxyz_from_rpy,
    _select_scene,
    _set_robot_pose,
)


def _parse_ticks(spec: str) -> set[int]:
    ticks: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                start, end = end, start
            ticks.update(range(start, end + 1))
        else:
            ticks.add(int(part))
    return ticks


def _selected_entries(result_path: Path, ticks: set[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    log = payload.get("log") or result.get("log") or []
    entries = _pose_entries_from_log(log)
    return result, [entry for entry in entries if int(entry.get("tick", -1)) in ticks]


def _source_clearance_by_tick(result_path: Path) -> dict[int, float]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    log = payload.get("log") or result.get("log") or []
    out: dict[int, float] = {}
    for row in log:
        if not isinstance(row, dict):
            continue
        if row.get("post_body_clearance_m") is None:
            continue
        out[int(row.get("tick", len(out)))] = float(row["post_body_clearance_m"])
    return out


def _label_traversability(
    clearance_m: float,
    *,
    positive_clearance_threshold_m: float,
    negative_clearance_threshold_m: float | None,
) -> float | None:
    if float(clearance_m) <= float(positive_clearance_threshold_m):
        return 0.0
    if negative_clearance_threshold_m is not None and float(clearance_m) < float(negative_clearance_threshold_m):
        return None
    return 2.0


def _jittered_poses(
    entry: dict[str, Any],
    *,
    count: int,
    xy_m: float,
    yaw_rad: float,
    rng: random.Random,
) -> list[tuple[np.ndarray, float, float, float]]:
    base_xy = entry["post_xy"]
    base_z = float(entry.get("post_z", 0.34))
    base_roll = float(entry.get("post_roll", 0.0))
    base_pitch = float(entry.get("post_pitch", 0.0))
    base_yaw = float(entry.get("post_yaw", 0.0))
    poses: list[tuple[np.ndarray, float, float, float]] = [
        (np.asarray([float(base_xy[0]), float(base_xy[1]), base_z], dtype=np.float32), base_roll, base_pitch, base_yaw)
    ]
    for _ in range(max(0, int(count))):
        dx = rng.uniform(-float(xy_m), float(xy_m))
        dy = rng.uniform(-float(xy_m), float(xy_m))
        dyaw = rng.uniform(-float(yaw_rad), float(yaw_rad))
        poses.append((
            np.asarray([float(base_xy[0]) + dx, float(base_xy[1]) + dy, base_z], dtype=np.float32),
            base_roll,
            base_pitch,
            base_yaw + dyaw,
        ))
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--ticks", required=True, help="Comma-separated ticks and ranges, for example 580-592,601.")
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z",
    )
    parser.add_argument("--platform-manifest", type=Path, default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--positive-clearance-threshold-m", type=float, default=0.06)
    parser.add_argument("--negative-clearance-threshold-m", type=float, default=0.08)
    parser.add_argument("--positive-jitter-count", type=int, default=0)
    parser.add_argument("--negative-jitter-count", type=int, default=0)
    parser.add_argument("--jitter-xy-m", type=float, default=0.015)
    parser.add_argument("--jitter-yaw-rad", type=float, default=0.08)
    parser.add_argument(
        "--counterfactual-pose-rows",
        action="store_true",
        help="Emit rows compatible with train_go2_jepa_primitive_outcome_predictor.py "
             "--label-mode counterfactual_body_clearance. The rendered image is written "
             "as start_frame and pose/scene_manifest metadata is included.",
    )
    parser.add_argument("--command-dt-s", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    ticks = _parse_ticks(args.ticks)
    result, entries = _selected_entries(args.result.resolve(), ticks)
    if not entries:
        raise SystemExit("no selected pose entries")
    clearance_by_tick = _source_clearance_by_tick(args.result.resolve())
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
    build = build_scene_from_pack(
        pack,
        n_envs=1,
        backend=str(args.backend),
        show_viewer=False,
        render_robot=False,
        apply_textures=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    rows: list[dict[str, Any]] = []
    from PIL import Image

    for entry in entries:
        tick = int(entry["tick"])
        clearance_m = clearance_by_tick.get(tick)
        if clearance_m is None:
            continue
        trav = _label_traversability(
            clearance_m,
            positive_clearance_threshold_m=float(args.positive_clearance_threshold_m),
            negative_clearance_threshold_m=args.negative_clearance_threshold_m,
        )
        if trav is None:
            continue
        jitter_count = (
            int(args.positive_jitter_count)
            if float(trav) == 0.0
            else int(args.negative_jitter_count)
        )
        for variant_idx, (pos_xyz, roll, pitch, yaw) in enumerate(
            _jittered_poses(
                entry,
                count=jitter_count,
                xy_m=float(args.jitter_xy_m),
                yaw_rad=float(args.jitter_yaw_rad),
                rng=rng,
            )
        ):
            quat = _quat_wxyz_from_rpy(float(roll), float(pitch), float(yaw))
            _set_robot_pose(build, pos_xyz, quat)
            try:
                build.scene.step()
            except Exception:
                pass
            ego = _render_tensor_from_base(
                build,
                pack,
                base_xyz_m=pos_xyz,
                base_quat_wxyz=quat,
                device=torch.device(str(args.policy_device)),
            )
            image = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            rgb_path = args.output_dir / f"{scene_id}_tick{tick:05d}_v{variant_idx:03d}.png"
            Image.fromarray(image).save(rgb_path)
            if bool(args.counterfactual_pose_rows):
                rows.append({
                    "command_dt_s": float(args.command_dt_s),
                    "scene_id": scene_id,
                    "scene_manifest": str((scene_dir / "manifest.json").resolve()),
                    "schema": "lewm_go2_result_pose_counterfactual_row_v0",
                    "source_clearance_m": float(clearance_m),
                    "start_base_pose_world": {
                        "position": {
                            "x": float(pos_xyz[0]),
                            "y": float(pos_xyz[1]),
                            "z": float(pos_xyz[2]),
                        },
                    },
                    "start_base_rpy_rad": {
                        "pitch": float(pitch),
                        "roll": float(roll),
                        "yaw": float(yaw),
                    },
                    "start_frame": str(rgb_path.resolve()),
                    "tick": tick,
                    "variant": int(variant_idx),
                })
            else:
                rows.append({
                    "rgb_path": str(rgb_path.resolve()),
                    "scene_id": scene_id,
                    "tick": tick,
                    "variant": int(variant_idx),
                    "source_clearance_m": float(clearance_m),
                    "traversability_forward_m": float(trav),
                })

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    positives = sum(
        1
        for row in rows
        if row.get("traversability_forward_m") is not None
        and float(row["traversability_forward_m"]) == 0.0
    )
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "frames": len(rows),
        "positives": int(positives),
        "negatives": (
            None
            if bool(args.counterfactual_pose_rows)
            else int(len(rows) - positives)
        ),
        "counterfactual_pose_rows": bool(args.counterfactual_pose_rows),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
