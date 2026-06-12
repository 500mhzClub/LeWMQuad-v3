#!/usr/bin/env python3
"""Build a goal-aligned all-primitives first-action dataset on frozen LeWM features.

Each sample group fixes one rendered start observation and one visible-beacon
goal, then evaluates every registered primitive from that identical state under
the benchmark's kinematic/collision model. The output is grouped NPZ data for
``train_first_action_ranker.py``; no LeWM parameters are updated.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_lewm_closed_loop_mpc as B  # noqa: E402


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _balanced_family_order(scene_dirs: list[Path]) -> list[Path]:
    """Round-robin families so a bounded all-family screen is representative."""
    by_family: dict[str, list[Path]] = {}
    for scene_dir in scene_dirs:
        by_family.setdefault(scene_dir.parent.name, []).append(scene_dir)
    for paths in by_family.values():
        paths.sort(key=lambda path: path.name)
    ordered: list[Path] = []
    for index in range(max(map(len, by_family.values()), default=0)):
        for family in sorted(by_family):
            if index < len(by_family[family]):
                ordered.append(by_family[family][index])
    return ordered


def _kinematic_outcome(
    primitive_name: str,
    registry,
    grid,
    start_xy: np.ndarray,
    start_yaw: float,
    command_dt_s: float,
) -> tuple[tuple[float, float], bool]:
    x, y, yaw = float(start_xy[0]), float(start_xy[1]), float(start_yaw)
    collided = False
    for vx, vy, yaw_rate in B.expand_primitive_to_block(registry, primitive_name):
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        nx = x + (float(vx) * cos_y - float(vy) * sin_y) * command_dt_s
        ny = y + (float(vx) * sin_y + float(vy) * cos_y) * command_dt_s
        if not grid.is_free((nx, ny)):
            collided = True
            break
        x, y = nx, ny
        yaw = B.wrap_angle_pi(yaw + float(yaw_rate) * command_dt_s)
    return (x, y), collided


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=REPO_ROOT / ".generated" / "scene_corpus" / "minimum_20260520T080420Z",
    )
    parser.add_argument(
        "--platform-manifest",
        type=Path,
        default=REPO_ROOT / "config" / "go2_platform_manifest.yaml",
    )
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=REPO_ROOT / "config" / "go2_primitive_registry.yaml",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="open_obstacle_field", help="Family name or 'all'.")
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--trials-per-scene", type=int, default=8)
    parser.add_argument("--goal-standoff-m", type=float, default=0.85)
    parser.add_argument("--beacon-approach-distance-m", type=float, default=1.5)
    parser.add_argument("--start-yaw-jitter-rad", type=float, default=math.pi)
    parser.add_argument(
        "--primitive-names",
        default="hold,forward_medium,arc_left,arc_right,yaw_left,yaw_right,backward",
    )
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, _ = B.load_model(
        SimpleNamespace(checkpoint=args.checkpoint.resolve(), max_seq_len=None, sigreg_lambda=None),
        device,
    )
    platform = B.load_platform_manifest(args.platform_manifest.resolve())
    registry = B.PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    command_dt_s = float(platform.get("timing", {}).get("command_dt_s", 0.10))
    primitive_names = _parse_csv(args.primitive_names)
    primitive_blocks = B._primitive_active_blocks(registry, primitive_names)
    primitive_actions = np.stack([primitive_blocks[name] for name in primitive_names]).astype(
        np.float32
    )

    family = None if args.family == "all" else args.family
    scene_dirs = list(B.find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=family))
    scene_dirs = (
        _balanced_family_order(scene_dirs)
        if family is None
        else sorted(scene_dirs, key=lambda path: path.name)
    )
    scene_dirs = scene_dirs[args.scene_offset :]
    if args.scene_limit > 0:
        scene_dirs = scene_dirs[: args.scene_limit]
    if not scene_dirs:
        raise SystemExit(f"no scenes for split={args.split!r} family={args.family!r}")

    start_raw_latents: list[np.ndarray] = []
    start_proj_latents: list[np.ndarray] = []
    goal_raw_latents: list[np.ndarray] = []
    goal_proj_latents: list[np.ndarray] = []
    initial_distances: list[float] = []
    after_distances: list[np.ndarray] = []
    collisions: list[np.ndarray] = []
    scene_ids: list[str] = []
    goal_ids: list[str] = []
    skipped: list[dict[str, str]] = []

    for scene_index, scene_dir in enumerate(scene_dirs):
        pack = B.load_scene_pack(
            scene_dir,
            platform_manifest=platform,
            workspace_root=REPO_ROOT,
        )
        print(f"[{scene_index + 1}/{len(scene_dirs)}] {pack.split}/{pack.family}/{pack.scene_id}")
        try:
            build = B.build_scene_from_pack(
                pack,
                n_envs=1,
                backend=args.backend,
                show_viewer=False,
                render_robot=False,
            )
            grid = B.InflatedOccupancyGrid(
                pack.scene_graph.manifest,
                cell_size_m=0.05,
                inflation_m=0.20,
            )
            for trial_index in range(args.trials_per_scene):
                trial_seed = args.seed + (args.scene_offset + scene_index) * 1000 + trial_index
                start_pos, start_quat, goal = B._select_visible_beacon_setup(
                    pack,
                    random.Random(trial_seed),
                    device=device,
                    build=build,
                    grid=grid,
                    approach_distance_m=args.beacon_approach_distance_m,
                    goal_standoff_m=args.goal_standoff_m,
                    start_yaw_jitter_rad=args.start_yaw_jitter_rad,
                    n_goal_views=0,
                )
                start_image = B._render_tensor_from_base(
                    build,
                    pack,
                    base_xyz_m=start_pos,
                    base_quat_wxyz=start_quat,
                    device=device,
                )
                start_raw, start_proj = B._encode_frame(model, start_image)
                goal_raw, goal_proj = B._encode_frame(model, goal.image)

                start_yaw = B._yaw_from_quat_wxyz(start_quat)
                initial_distance = B._xy_distance(start_pos[:2], goal.target_xy)
                trial_after: list[float] = []
                trial_collisions: list[bool] = []
                for primitive_name in primitive_names:
                    endpoint, collided = _kinematic_outcome(
                        primitive_name,
                        registry,
                        grid,
                        start_pos[:2],
                        start_yaw,
                        command_dt_s,
                    )
                    trial_after.append(B._xy_distance(endpoint, goal.target_xy))
                    trial_collisions.append(collided)

                start_raw_latents.append(start_raw[0].float().cpu().numpy())
                start_proj_latents.append(start_proj[0].float().cpu().numpy())
                goal_raw_latents.append(goal_raw[0].float().cpu().numpy())
                goal_proj_latents.append(goal_proj[0].float().cpu().numpy())
                initial_distances.append(initial_distance)
                after_distances.append(np.asarray(trial_after, dtype=np.float32))
                collisions.append(np.asarray(trial_collisions, dtype=np.bool_))
                scene_ids.append(str(pack.scene_id))
                goal_ids.append(str(goal.object_id))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"scene": str(scene_dir), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[SKIP] {scene_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not start_proj_latents:
        raise SystemExit("no first-action sample groups were generated")

    initial_array = np.asarray(initial_distances, dtype=np.float32)
    after_array = np.stack(after_distances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray("first_action_dataset_v0"),
        source_checkpoint=np.asarray(str(args.checkpoint.resolve())),
        split=np.asarray(args.split),
        family=np.asarray(args.family),
        primitive_names=np.asarray(primitive_names),
        primitive_actions=primitive_actions,
        start_raw=np.stack(start_raw_latents),
        start_proj=np.stack(start_proj_latents),
        goal_raw=np.stack(goal_raw_latents),
        goal_proj=np.stack(goal_proj_latents),
        initial_distance_m=initial_array,
        after_distance_m=after_array,
        progress_m=initial_array[:, None] - after_array,
        collision=np.stack(collisions),
        scene_id=np.asarray(scene_ids),
        goal_id=np.asarray(goal_ids),
    )
    summary = {
        "schema": "first_action_dataset_summary_v0",
        "output": str(args.output.resolve()),
        "split": args.split,
        "family": args.family,
        "sample_groups": len(start_proj_latents),
        "primitives": primitive_names,
        "mean_random_regret_m": float((after_array.mean(1) - after_array.min(1)).mean()),
        "mean_oracle_progress_m": float((initial_array - after_array.min(1)).mean()),
        "collision_rate": float(np.stack(collisions).mean()),
        "skipped": skipped,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
