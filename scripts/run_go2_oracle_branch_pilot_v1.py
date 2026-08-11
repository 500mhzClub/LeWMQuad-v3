#!/usr/bin/env python3
"""Thin mid-episode branch runner + oracle pilot.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No world-model checkpoint is loaded.

Three narrow operations against the deployed runtime, nothing more:

    capture_branch_state(runner)            at the frozen canonical boundary
    restore_branch_state(runner, snapshot)  without advancing policy or solver
    execute_candidate(runner, snapshot, c)  one four-block branch

Frozen bindings: boundary 1faae05f..., inventory v2 9b08939a..., design v1.1
e3176d93....  Spatial labels are intentionally deferred and are not part of the
gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
BOUNDARY_DIGEST = "1faae05f843e6f02f0f354c63ab3bcad9404111140146b1355d025da3d0c7a92"
INVENTORY_V2 = "9b08939adabff5650b570c7f7b806524c0a0a38332946cd6d715313823e3595a"
DESIGN_V1_1 = "e3176d93fe3028d659359bc76606e7f1480d1742bbcd57cd8635251ea282f2a3"

BLOCKS = 4
TICKS = 5
TIE_TOLERANCE = 0.02

# ---- frozen 12-candidate bank (design v1.1: forward / arc / yaw / backward / hold)
PRIMITIVES = {
    "hold": (0.0, 0.0, 0.0), "forward_slow": (0.20, 0.0, 0.0),
    "forward_medium": (0.25, 0.0, 0.0), "forward_fast": (0.30, 0.0, 0.0),
    "backward": (-0.20, 0.0, 0.0), "yaw_left": (0.0, 0.0, 0.45),
    "yaw_right": (0.0, 0.0, -0.45), "arc_left": (0.20, 0.0, 0.45),
    "arc_right": (0.20, 0.0, -0.45),
}
CANDIDATE_BANK = (
    ("straight_fast", ("forward_fast",) * 4),
    ("straight_medium", ("forward_medium",) * 4),
    ("straight_slow", ("forward_slow",) * 4),
    ("arc_left_sustained", ("arc_left",) * 4),
    ("arc_right_sustained", ("arc_right",) * 4),
    ("turn_left_sustained", ("yaw_left",) * 4),
    ("turn_right_sustained", ("yaw_right",) * 4),
    ("turn_left_then_go", ("yaw_left", "yaw_left", "forward_medium", "forward_medium")),
    ("turn_right_then_go", ("yaw_right", "yaw_right", "forward_medium", "forward_medium")),
    ("go_then_turn_left", ("forward_medium", "forward_medium", "yaw_left", "yaw_left")),
    ("reverse_then_turn", ("backward", "backward", "yaw_left", "yaw_left")),
    ("hold_all", ("hold",) * 4),
)
UTILITY = {"progress": 1.0, "safety": -2.0, "completion": 0.5}
PROGRESS_SCALE = 20.0          # BFS cells; progress clipped to [-1, 1]
CLEARANCE_HAZARD_M = 0.15


def bank_digest() -> str:
    payload = {"bank": [[n, list(s)] for n, s in CANDIDATE_BANK],
               "primitives": {k: list(v) for k, v in PRIMITIVES.items()},
               "blocks": BLOCKS, "ticks": TICKS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def oracle_digest() -> str:
    payload = {"weights": UTILITY, "progress_scale": PROGRESS_SCALE,
               "clearance_hazard_m": CLEARANCE_HAZARD_M,
               "safety": "path-level max over the whole four-block trace",
               "completion": "at or before the branch horizon",
               "tie_tolerance": TIE_TOLERANCE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def block_for(primitive: str) -> np.ndarray:
    return np.tile(np.asarray(PRIMITIVES[primitive], dtype=np.float32), (TICKS, 1))


# ------------------------------------------------------------------ snapshot --
class BranchSnapshot:
    """In-process snapshot captured only at the canonical boundary."""

    __slots__ = ("checkpoint", "last_actions", "last_executed", "sim_time_ns",
                 "rng", "goal", "identity", "digest")

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _assert_boundary(runner) -> None:
    """The frozen invariants; capture is refused elsewhere."""
    if int(runner.n_envs) != 1:
        raise RuntimeError("branch capture requires a single-env build")
    if getattr(runner.policy, "_last_actions", None) is None:
        raise RuntimeError("policy action history is uninitialised at capture")


def capture_branch_state(runner, *, goal, identity) -> BranchSnapshot:
    _assert_boundary(runner)
    path = Path(tempfile.mkdtemp(prefix="branchsnap_")) / "solver.ckpt"
    runner.build.scene.save_checkpoint(str(path))
    snap = BranchSnapshot(
        checkpoint=path,
        last_actions=np.array(runner.policy._last_actions, dtype=np.float32, copy=True),
        last_executed=np.array(runner._last_executed, dtype=np.float32, copy=True),
        sim_time_ns=int(getattr(runner, "_sim_time_ns", 0)),
        rng={"python": random.getstate(), "numpy": np.random.get_state()},
        goal=dict(goal), identity=dict(identity), digest=None)
    snap.digest = hashlib.sha256(b"|".join([
        b"GENESIS_SOLVER_V1", path.read_bytes(),
        b"CONTROLLER_RNG_V1", snap.last_actions.tobytes(), snap.last_executed.tobytes(),
        b"HARNESS_STATE_V1", json.dumps(snap.identity, sort_keys=True).encode(),
    ])).hexdigest()
    return snap


def restore_branch_state(runner, snapshot: BranchSnapshot) -> None:
    """Restore all three layers. No policy or simulator step is taken."""
    runner.build.scene.load_checkpoint(str(snapshot.checkpoint))
    runner.policy._last_actions = np.array(snapshot.last_actions, dtype=np.float32, copy=True)
    runner._last_executed = np.array(snapshot.last_executed, dtype=np.float32, copy=True)
    if hasattr(runner, "_sim_time_ns"):
        runner._sim_time_ns = int(snapshot.sim_time_ns)
    random.setstate(snapshot.rng["python"])
    np.random.set_state(snapshot.rng["numpy"])
    _assert_boundary(runner)


# ------------------------------------------------------------------- branch ---
def execute_candidate(runner, snapshot, candidate, *, scene_graph=None, trace=False):
    """Restore, then run the four-block candidate through the production path."""
    name, primitives = candidate
    restore_branch_state(runner, snapshot)
    robot = runner.build.robot
    start_xy = np.asarray(robot.get_pos())[0][:2]
    ticks = []

    def after_policy_step(tick_idx, step_idx):
        if not trace:
            return
        ticks.append({
            "tick": int(tick_idx), "step": int(step_idx),
            "last_actions": np.asarray(runner.policy._last_actions, dtype=np.float32).copy(),
            "pos": np.asarray(robot.get_pos(), dtype=np.float32).copy(),
            "quat": np.asarray(robot.get_quat(), dtype=np.float32).copy(),
            "vel": np.asarray(robot.get_vel(), dtype=np.float32).copy(),
        })

    requested_all, executed_all, hazard, fell = [], [], 0.0, False
    for primitive in primitives:
        requested = block_for(primitive)[None, ...]
        block = runner.execute_requested_block(requested, after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        z = float(np.asarray(robot.get_pos())[0][2])
        if z < float(runner.config.fall_z_threshold_m):
            fell = True
            hazard = 1.0
            break
    final_xy = np.asarray(robot.get_pos())[0][:2]
    return {"candidate": name, "requested": requested_all, "executed": executed_all,
            "start_xy": start_xy.tolist(), "final_xy": final_xy.tolist(),
            "fell": fell, "hazard": hazard, "ticks": ticks}


# ------------------------------------------------------------------- oracle ---
def oracle_components(branch, *, scene_graph, goal_cell, start_distance):
    """progress / safety / completion under the frozen definitions."""
    hit = scene_graph.locate((float(branch["final_xy"][0]), float(branch["final_xy"][1])))
    final_cell = getattr(hit, "cell_id", None)
    final_distance = (scene_graph.bfs_distance(final_cell, goal_cell)
                      if final_cell is not None else None)
    if final_distance is None or start_distance is None:
        return None
    progress = float(np.clip((start_distance - final_distance) / PROGRESS_SCALE, -1.0, 1.0))
    safety = float(branch["hazard"])
    completion = 1.0 if final_distance == 0 else 0.0
    utility = (UTILITY["progress"] * progress + UTILITY["safety"] * safety
               + UTILITY["completion"] * completion)
    return {"progress": progress, "safety": safety, "completion": completion,
            "utility": utility, "start_bfs": start_distance, "final_bfs": final_distance}


# --------------------------------------------------------------------- gate ---
def identifiability(records) -> dict:
    by_state = {}
    for row in records:
        by_state.setdefault(row["state_id"], []).append(row)
    valid_states, separated, levels, spreads, families = 0, 0, [], [], {}
    attempted = len(records)
    invalid = sum(1 for r in records if not r["valid"])
    for state_id, rows in by_state.items():
        good = [r for r in rows if r["valid"]]
        if len(good) < 2:
            continue
        valid_states += 1
        families.setdefault(rows[0]["family"], 0)
        families[rows[0]["family"]] += 1
        utilities = sorted((r["utility"] for r in good), reverse=True)
        if utilities[0] - utilities[1] > TIE_TOLERANCE:
            separated += 1
        distinct = []
        for value in utilities:
            if not distinct or abs(value - distinct[-1]) > TIE_TOLERANCE:
                distinct.append(value)
        levels.append(len(distinct))
        spreads.append(utilities[0] - utilities[-1])
    return {
        "attempted": attempted, "valid": attempted - invalid, "invalid": invalid,
        "invalid_rate": invalid / attempted if attempted else 1.0,
        "states_scored": valid_states,
        "uniquely_separated_fraction": separated / valid_states if valid_states else 0.0,
        "median_distinct_levels": float(np.median(levels)) if levels else 0.0,
        "median_spread": float(np.median(spreads)) if spreads else 0.0,
        "families_with_two_valid_states": sum(1 for v in families.values() if v >= 2),
        "families_present": len(families),
    }


def gate_verdict(stats) -> dict:
    checks = {
        "uniquely_separated_ge_0.70": stats["uniquely_separated_fraction"] >= 0.70,
        "median_distinct_levels_ge_5": stats["median_distinct_levels"] >= 5,
        "median_spread_ge_0.10": stats["median_spread"] >= 0.10,
        "invalid_rate_le_0.20": stats["invalid_rate"] <= 0.20,
        "all_eight_families_two_valid_states": stats["families_with_two_valid_states"] >= 8,
    }
    return {"components": checks, "pass": all(checks.values())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-digests", action="store_true")
    args = ap.parse_args()
    if args.print_digests:
        print(json.dumps({"candidate_bank_digest": bank_digest(),
                          "oracle_contract_digest": oracle_digest(),
                          "candidates": [n for n, _ in CANDIDATE_BANK],
                          "boundary": BOUNDARY_DIGEST, "inventory_v2": INVENTORY_V2,
                          "design_v1_1": DESIGN_V1_1}, indent=2))
        raise SystemExit(0)
    raise SystemExit("driver stages are invoked by the pilot entry point")
