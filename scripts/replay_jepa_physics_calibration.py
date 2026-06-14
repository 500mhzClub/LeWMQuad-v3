#!/usr/bin/env python3
"""Replay sampled JEPA counterfactual candidates with the Genesis Go2 gait."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
os.environ.setdefault("HOME", str(REPO_ROOT / ".generated/benchmark_home"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".generated/cache"))
os.environ.setdefault("TI_CACHE_HOME", str(REPO_ROOT / ".generated/cache/taichi"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".generated/mplconfig"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark_lewm_closed_loop_mpc as B  # noqa: E402
from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits, expand_primitive_to_block  # noqa: E402
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import load_platform_manifest, load_scene_pack  # noqa: E402
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402


def _quat_wxyz_from_row(row: dict) -> np.ndarray:
    orientation = row["start_base_pose_world"]["orientation"]
    return np.asarray(
        [orientation["w"], orientation["x"], orientation["y"], orientation["z"]],
        dtype=np.float32,
    )


def _position_from_row(row: dict) -> np.ndarray:
    position = row["start_base_pose_world"]["position"]
    return np.asarray([position["x"], position["y"], position["z"]], dtype=np.float32)


def _wrap_angle_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _group_summary(rows: list[dict]) -> dict:
    enters_confusion = Counter(
        (
            bool(row["kinematic_candidate"]["enters_grid_unsafe"]),
            bool(row["physical_enters_grid_unsafe"]),
        )
        for row in rows
    )
    ends_confusion = Counter(
        (
            bool(row["kinematic_candidate"]["ends_grid_unsafe"]),
            bool(row["physical_ends_grid_unsafe"]),
        )
        for row in rows
    )

    def confusion_dict(confusion: Counter) -> dict:
        return {
            f"kinematic_{str(kinematic).lower()}_physical_{str(physical).lower()}": int(
                confusion[(kinematic, physical)]
            )
            for kinematic in (False, True)
            for physical in (False, True)
        }

    return {
        "row_count": len(rows),
        "fall_rate": float(np.mean([row["physical_fell"] for row in rows])),
        "mean_endpoint_error_vs_kinematic_m": float(
            np.mean([row["endpoint_error_vs_kinematic_m"] for row in rows])
        ),
        "mean_yaw_error_vs_kinematic_rad": float(
            np.mean([row["yaw_error_vs_kinematic_rad"] for row in rows])
        ),
        "enters_grid_unsafe_agreement_rate": float(
            np.mean([row["enters_grid_unsafe_agrees"] for row in rows])
        ),
        "ends_grid_unsafe_agreement_rate": float(
            np.mean([row["ends_grid_unsafe_agrees"] for row in rows])
        ),
        "enters_grid_unsafe_confusion": confusion_dict(enters_confusion),
        "ends_grid_unsafe_confusion": confusion_dict(ends_confusion),
    }


def _replay_candidate(
    row: dict,
    *,
    build,
    runner: RolloutRunner,
    registry: PrimitiveRegistry,
    grid: InflatedOccupancyGrid,
    fall_z_threshold_m: float,
) -> dict:
    start_pos = _position_from_row(row)
    start_quat = _quat_wxyz_from_row(row)
    B._set_pose(build=build, runner=runner, pos_xyz=start_pos, quat_wxyz=start_quat)
    positions = [np.asarray(B._current_pose(build)[0], dtype=np.float64)]
    yaws = [B._yaw_from_quat_wxyz(B._current_pose(build)[1])]
    clearances = [grid.configuration_clearance_m(tuple(positions[-1][:2]))]
    fell = positions[-1][2] < fall_z_threshold_m
    executed_blocks = []
    block_endpoints_xyz = []
    block_endpoint_yaws = []
    for primitive_name in row["kinematic_candidate"]["primitive_sequence"]:
        requested = expand_primitive_to_block(registry, primitive_name)
        clipped = runner._clip_block(requested[None, :, :]).executed[0]
        executed_blocks.append(clipped.tolist())
        for tick in clipped:
            runner._step_command_tick(tick[None, :])
            pos, quat = B._current_pose(build)
            positions.append(np.asarray(pos, dtype=np.float64))
            yaws.append(B._yaw_from_quat_wxyz(quat))
            clearances.append(grid.configuration_clearance_m(tuple(positions[-1][:2])))
            fell = fell or positions[-1][2] < fall_z_threshold_m
        runner._last_executed[0] = clipped[-1]
        block_endpoints_xyz.append(positions[-1].tolist())
        block_endpoint_yaws.append(float(yaws[-1]))

    start_xy = positions[0][:2]
    end_xy = positions[-1][:2]
    path_length = sum(
        float(np.linalg.norm(current[:2] - previous[:2]))
        for previous, current in zip(positions, positions[1:])
    )
    target_xy = row.get("target_xy")
    target_progress = None
    if target_xy is not None:
        target = np.asarray(target_xy, dtype=np.float64)
        target_progress = float(np.linalg.norm(start_xy - target) - np.linalg.norm(end_xy - target))
    kinematic = row["kinematic_candidate"]
    kin_endpoint = kinematic["endpoint"]
    endpoint_error = math.dist(
        (float(end_xy[0]), float(end_xy[1])),
        (float(kin_endpoint["x_m"]), float(kin_endpoint["y_m"])),
    )
    yaw_error = abs(_wrap_angle_pi(float(yaws[-1]) - float(kin_endpoint["yaw_rad"])))
    physical_enters_grid_unsafe = any(
        clearances[index - 1] >= 0.0 and clearances[index] < 0.0
        for index in range(1, len(clearances))
    )
    return {
        **row,
        "physics_schema": "jepa_physics_calibration_result_v0",
        "physics_backend": "genesis_go2_contract_ppo",
        "physics_validated": True,
        "executed_blocks": executed_blocks,
        "physical_block_endpoints_xyz": block_endpoints_xyz,
        "physical_block_endpoint_yaws_rad": block_endpoint_yaws,
        "physical_endpoint_xyz": positions[-1].tolist(),
        "physical_endpoint_yaw_rad": float(yaws[-1]),
        "physical_path_length_m": path_length,
        "physical_target_progress_m": target_progress,
        "physical_minimum_configuration_clearance_m": float(min(clearances)),
        "physical_enters_grid_unsafe": physical_enters_grid_unsafe,
        "physical_ends_grid_unsafe": clearances[-1] < 0.0,
        "physical_fell": bool(fell),
        "endpoint_error_vs_kinematic_m": endpoint_error,
        "yaw_error_vs_kinematic_rad": yaw_error,
        "enters_grid_unsafe_agrees": (
            bool(kinematic["enters_grid_unsafe"]) == physical_enters_grid_unsafe
        ),
        "ends_grid_unsafe_agrees": (
            bool(kinematic["ends_grid_unsafe"]) == (clearances[-1] < 0.0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15)
    parser.add_argument(
        "--platform-manifest",
        type=Path,
        default=REPO_ROOT / "config/go2_platform_manifest.yaml",
    )
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=REPO_ROOT / "config/go2_primitive_registry.yaml",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open()]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit("calibration sample is empty")

    platform = load_platform_manifest(args.platform_manifest.resolve())
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    safety = SafetyLimits.from_manifest(platform)
    policy = GenesisGo2PPOPolicy.from_platform_manifest(
        platform, REPO_ROOT, device=args.policy_device
    )
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        current_manifest = None
        build = None
        runner = None
        grid = None
        for index, row in enumerate(rows, 1):
            manifest = str(row["scene_manifest"])
            if manifest != current_manifest:
                scene_dir = Path(manifest).parent
                pack = load_scene_pack(
                    scene_dir,
                    platform_manifest=platform,
                    workspace_root=REPO_ROOT,
                )
                build = build_scene_from_pack(
                    pack,
                    n_envs=1,
                    backend=args.backend,
                    show_viewer=False,
                    render_robot=False,
                )
                runner = RolloutRunner(
                    build,
                    policy,
                    registry,
                    safety,
                    config=RolloutConfig(
                        n_blocks=2,
                        fall_z_threshold_m=args.fall_z_threshold_m,
                        rgb_capture_per_block=False,
                        seed=20260614,
                        log_progress_every_blocks=0,
                        foot_contact_source="zero",
                        randomize_spawn_pose=False,
                    ),
                )
                grid = InflatedOccupancyGrid(
                    pack.scene_graph.manifest, cell_size_m=0.05, inflation_m=0.20
                )
                current_manifest = manifest
            assert build is not None and runner is not None and grid is not None
            result = _replay_candidate(
                row,
                build=build,
                runner=runner,
                registry=registry,
                grid=grid,
                fall_z_threshold_m=args.fall_z_threshold_m,
            )
            results.append(result)
            destination.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"[{index}/{len(rows)}] {row['family']} {row['candidate_bucket']} "
                f"endpoint_err={result['endpoint_error_vs_kinematic_m']:.3f}m "
                f"fell={int(result['physical_fell'])}",
                flush=True,
            )

    overall = _group_summary(results)
    summary = {
        "schema": "jepa_physics_calibration_summary_v0",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "family_counts": dict(Counter(row["family"] for row in results)),
        "bucket_counts": dict(Counter(row["candidate_bucket"] for row in results)),
        **overall,
        "by_bucket": {
            bucket: _group_summary(
                [row for row in results if row["candidate_bucket"] == bucket]
            )
            for bucket in sorted({row["candidate_bucket"] for row in results})
        },
        "by_family": {
            family: _group_summary([row for row in results if row["family"] == family])
            for family in sorted({row["family"] for row in results})
        },
        "limitations": [
            "Genesis foot-contact force reads are disabled on the current AMD path",
            "minimum physical configuration clearance is sampled once per command tick",
            "sample is bounded and stratified rather than exhaustive",
        ],
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
