#!/usr/bin/env python3
"""Build the JEPA multi-block counterfactual navigation benchmark.

The input is an existing scene-disjoint task-aligned decision index. Every row
fixes one observation/history/start pose and target. This script branches all
configured primitive sequences from that identical state and adds privileged
kinematic labels for swept safety, clearance, progress, heading, and
recoverability.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm.benchmarks.counterfactual import (  # noqa: E402
    Pose2D,
    oracle_sort_key,
    simulate_candidate_trajectory,
)
from lewm.benchmarks.phase2d_generation import (  # noqa: E402
    factorial_primitive_sequences,
    phase2d_lineage_fields,
    sequence_grid_audit,
)
from lewm.actions import encode_active_block  # noqa: E402
from lewm_genesis.lewm_contract import (  # noqa: E402
    PrimitiveRegistry,
    expand_primitive_to_block,
)
from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402


class _CachedRecoverabilityGrid:
    """Delegate grid queries while caching repeated endpoint-to-target A* calls."""

    def __init__(self, grid: InflatedOccupancyGrid):
        self.grid = grid
        self.astar_cache: dict[tuple[tuple[int, int], tuple[int, int]], object] = {}

    def is_free(self, xy: tuple[float, float]) -> bool:
        return self.grid.is_free(xy)

    def configuration_clearance_m(self, xy: tuple[float, float]) -> float:
        return self.grid.configuration_clearance_m(xy)

    def astar(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]):
        key = (self.grid.to_grid(start_xy), self.grid.to_grid(goal_xy))
        if key not in self.astar_cache:
            self.astar_cache[key] = self.grid.astar(start_xy, goal_xy)
        return self.astar_cache[key]


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _target_cell(row: dict) -> int | None:
    oracle_next = row.get("oracle_next_cell_id")
    if oracle_next is not None:
        return int(oracle_next)
    route_target = int(row.get("route_target_id", -1))
    return route_target if route_target >= 0 else None


def _active_block(block) -> list[float]:
    return encode_active_block(block[:, 0], block[:, 1], block[:, 2]).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=REPO_ROOT / "config/go2_primitive_registry.yaml",
    )
    parser.add_argument(
        "--primitive-names",
        default="hold,forward_slow,forward_medium,forward_fast,backward,yaw_left,yaw_right,arc_left,arc_right",
    )
    parser.add_argument("--horizon-blocks", type=int, default=2)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--inflation-m", type=float, default=0.20)
    parser.add_argument("--sweep-step-m", type=float, default=0.025)
    parser.add_argument("--sweep-step-yaw-rad", type=float, default=0.05)
    parser.add_argument("--include-trajectories", action="store_true")
    parser.add_argument(
        "--require-local-target-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip goal-conditioned rows without the matched local-target image.",
    )
    parser.add_argument(
        "--require-phase2d-lineage",
        action="store_true",
        help=(
            "Fail if a generated source state cannot be assigned both topology "
            "and visual lineage fields."
        ),
    )
    args = parser.parse_args()

    if args.horizon_blocks < 1:
        raise SystemExit("--horizon-blocks must be at least 1")

    registry = PrimitiveRegistry.from_yaml(args.primitive_registry)
    primitive_names = _parse_csv(args.primitive_names)
    unknown = [name for name in primitive_names if name not in registry.primitives]
    if unknown:
        raise SystemExit(f"unknown primitive names: {unknown}")
    action_blocks = {
        name: expand_primitive_to_block(registry, name) for name in primitive_names
    }
    active_blocks = {name: _active_block(block) for name, block in action_blocks.items()}
    sequences = factorial_primitive_sequences(
        primitive_names,
        horizon_blocks=args.horizon_blocks,
    )
    sequence_audit = sequence_grid_audit(
        primitive_names=primitive_names,
        horizon_blocks=args.horizon_blocks,
        sequences=sequences,
    )

    assets: dict[str, tuple[dict, SceneGraph, InflatedOccupancyGrid]] = {}
    input_row_count = 0
    row_count = 0
    skipped_missing_local_target_frame = 0
    target_rows = 0
    target_rows_with_local_target_frame = 0
    candidate_count = 0
    starts_grid_unsafe = 0
    candidate_enters_unsafe = 0
    candidate_ends_unsafe = 0
    candidate_recoverable = 0
    oracle_first_primitive: Counter[str] = Counter()
    phase2d_lineage_verified_rows = 0
    phase2d_lineage_missing_field_counts: Counter[str] = Counter()
    scene_ids: set[str] = set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as source, args.output.open("w") as destination:
        for source_index, line in enumerate(source):
            if source_index < args.start_row:
                continue
            if args.max_rows > 0 and row_count >= args.max_rows:
                break
            input_row_count += 1
            row = json.loads(line)
            target_cell = _target_cell(row)
            local_target_frame = row.get("local_target_frame")
            target_rows += target_cell is not None
            target_rows_with_local_target_frame += (
                target_cell is not None
                and local_target_frame is not None
                and Path(str(local_target_frame)).is_file()
            )
            if (
                args.require_local_target_frame
                and target_cell is not None
                and (
                    local_target_frame is None
                    or not Path(str(local_target_frame)).is_file()
                )
            ):
                skipped_missing_local_target_frame += 1
                continue
            manifest_path = str(row["scene_manifest"])
            if manifest_path not in assets:
                manifest_payload = json.loads(Path(manifest_path).read_text())
                manifest = parse_scene_manifest_dict(
                    manifest_payload
                )
                assets[manifest_path] = (
                    manifest_payload,
                    SceneGraph(manifest),
                    InflatedOccupancyGrid(
                        manifest,
                        cell_size_m=args.cell_size_m,
                        inflation_m=args.inflation_m,
                    ),
                )
            manifest_payload, graph, grid = assets[manifest_path]
            lineage_fields = phase2d_lineage_fields(
                row,
                scene_manifest=manifest_payload,
            )
            lineage_audit = lineage_fields["phase2d_source_state_lineage"]
            if lineage_audit["lineage_verified"]:
                phase2d_lineage_verified_rows += 1
            else:
                for field in lineage_audit["missing_fields"]:
                    phase2d_lineage_missing_field_counts[str(field)] += 1
                if args.require_phase2d_lineage:
                    raise SystemExit(
                        "missing Phase 2D lineage for source row "
                        f"{source_index}: {lineage_audit['missing_fields']}"
                    )
            cached_grid = _CachedRecoverabilityGrid(grid)
            target_xy = graph.cell_center(target_cell) if target_cell is not None else None
            position = row["start_base_pose_world"]["position"]
            start = Pose2D(
                x_m=float(position["x"]),
                y_m=float(position["y"]),
                yaw_rad=float(row["start_base_rpy_rad"]["yaw"]),
            )

            candidates = []
            candidate_objects = []
            for sequence in sequences:
                candidate = simulate_candidate_trajectory(
                    primitive_sequence=sequence,
                    action_blocks=[action_blocks[name] for name in sequence],
                    start=start,
                    command_dt_s=registry.command_dt_s,
                    grid=cached_grid,
                    target_xy=target_xy,
                    sweep_step_m=args.sweep_step_m,
                    sweep_step_yaw_rad=args.sweep_step_yaw_rad,
                    include_trajectory=args.include_trajectories,
                )
                payload = candidate.to_jsonable()
                payload["active_blocks"] = [active_blocks[name] for name in sequence]
                candidates.append(payload)
                candidate_objects.append(candidate)
                candidate_count += 1
                candidate_enters_unsafe += candidate.enters_grid_unsafe
                candidate_ends_unsafe += candidate.ends_grid_unsafe
                candidate_recoverable += candidate.target_recoverable is True

            oracle_index = min(
                range(len(candidate_objects)),
                key=lambda index: oracle_sort_key(candidate_objects[index]),
            )
            oracle = candidate_objects[oracle_index]
            oracle_first_primitive[oracle.primitive_sequence[0]] += 1
            starts_grid_unsafe += oracle.starts_grid_unsafe
            scene_ids.add(str(row["scene_id"]))

            output = {
                **row,
                "benchmark_schema": "jepa_counterfactual_decision_v0",
                "label_source": "privileged_kinematic_grid_v0",
                "physics_validated": False,
                "topology_seed": lineage_fields["topology_seed"],
                "visual_seed": lineage_fields["visual_seed"],
                "phase2d_source_state_lineage": lineage_audit,
                "counterfactual_sequence_grid": sequence_audit,
                "counterfactual_generation_contract": {
                    "schema": "jepa_phase2d_counterfactual_generation_contract_v0",
                    "same_source_state_for_all_candidates": True,
                    "full_factorial_sequence_grid": sequence_audit[
                        "full_factorial_passed"
                    ],
                    "phase2d_full_81_two_block_grid": sequence_audit[
                        "phase2d_full_81_two_block_grid"
                    ],
                    "lineage_verified": lineage_audit["lineage_verified"],
                },
                "counterfactual_horizon_blocks": args.horizon_blocks,
                "counterfactual_target_cell_id": target_cell,
                "counterfactual_target_xy": list(target_xy) if target_xy is not None else None,
                "counterfactual_primitive_names": primitive_names,
                "counterfactual_oracle_index": oracle_index,
                "counterfactual_oracle_primitive_sequence": list(
                    oracle.primitive_sequence
                ),
                "counterfactual_candidates": candidates,
            }
            destination.write(json.dumps(output, sort_keys=True) + "\n")
            row_count += 1

    if row_count == 0:
        raise SystemExit("no counterfactual benchmark rows generated")
    summary = {
        "schema": "jepa_counterfactual_decision_summary_v0",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "start_row": args.start_row,
        "label_source": "privileged_kinematic_grid_v0",
        "physics_validated": False,
        "input_row_count": input_row_count,
        "row_count": row_count,
        "require_phase2d_lineage": bool(args.require_phase2d_lineage),
        "phase2d_lineage_verified_rows": phase2d_lineage_verified_rows,
        "phase2d_lineage_missing_rows": row_count - phase2d_lineage_verified_rows,
        "phase2d_lineage_missing_field_counts": dict(
            sorted(phase2d_lineage_missing_field_counts.items())
        ),
        "require_local_target_frame": bool(args.require_local_target_frame),
        "skipped_missing_local_target_frame": skipped_missing_local_target_frame,
        "input_target_rows": target_rows,
        "input_target_rows_with_local_target_frame": target_rows_with_local_target_frame,
        "scene_count": len(scene_ids),
        "horizon_blocks": args.horizon_blocks,
        "primitive_names": primitive_names,
        "sequences_per_row": len(sequences),
        "counterfactual_sequence_grid": sequence_audit,
        "candidate_count": candidate_count,
        "starts_grid_unsafe_rate": starts_grid_unsafe / row_count,
        "candidate_enters_grid_unsafe_rate": candidate_enters_unsafe / candidate_count,
        "candidate_ends_grid_unsafe_rate": candidate_ends_unsafe / candidate_count,
        "candidate_target_recoverable_rate": candidate_recoverable / candidate_count,
        "oracle_first_primitive_counts": dict(oracle_first_primitive),
        "contract": {
            "separate_labels": [
                "enters_grid_unsafe",
                "ends_grid_unsafe",
                "minimum_swept_configuration_clearance_m",
                "p05_swept_configuration_clearance_m",
                "target_progress_m",
                "target_heading_error_rad",
                "target_recoverable",
            ],
            "oracle_order": [
                "avoid entering unsafe space",
                "end outside unsafe space",
                "preserve target recoverability",
                "maximize target progress or targetless clearance gain",
                "maximize p05 swept clearance",
                "minimize target heading error",
                "minimize path length",
            ],
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
