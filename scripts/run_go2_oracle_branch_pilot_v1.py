#!/usr/bin/env python3
"""Mid-episode branch runner, frozen oracle, and the 20-state oracle pilot.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No world-model checkpoint is loaded.

Three narrow operations against the deployed runtime:

    capture_branch_state(ctx)             at the frozen canonical boundary
    restore_branch_state(ctx, snapshot)   without advancing policy or solver
    execute_candidate(ctx, snapshot, c)   one four-block branch

plus the driver that selects states, qualifies deterministic replay, and runs
the pilot.

Frozen bindings: boundary 1faae05f..., inventory v2 9b08939a..., design v1.1
e3176d93....  Spatial labels at H=1-4 are intentionally deferred and are not
part of the identifiability gate.

Stages::

    --stage digests      print the frozen digests, simulate nothing
    --stage identities   write the frozen identity manifests, simulate nothing
    --stage qualify      replay qualification + sensitivity + branch order
    --stage pilot        the 20-state / 240-branch pilot
    --stage gate         apply the frozen gate to the pilot JSONL
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
BOUNDARY_DIGEST = "1faae05f843e6f02f0f354c63ab3bcad9404111140146b1355d025da3d0c7a92"
INVENTORY_V2 = "9b08939adabff5650b570c7f7b806524c0a0a38332946cd6d715313823e3595a"
DESIGN_V1_1 = "e3176d93fe3028d659359bc76606e7f1480d1742bbcd57cd8635251ea282f2a3"

# The pre-run oracle digest published at commit 769d3a4.  Its safety component
# was fall-only, which is weaker than the design-v1.1 path-level definition, so
# it is superseded *before* any branch was executed.  Recorded, never reused.
SUPERSEDED_ORACLE_DIGEST_PRE_RUN = (
    "4849e6dd99cc40ab721eaf9065a553d652513e36ef476d4df98f402ef903c399"
)

BLOCKS = 4
TICKS = 5
TIE_TOLERANCE = 0.02

# ---- frozen 12-candidate bank (design v1.1: forward / arc / yaw / backward / hold)
# Values are the ``config/go2_primitive_registry.yaml`` set-points verbatim.
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

# ---- path-level safety, design v1.1 §"Safety preserves path events".
# Every quantity below is an existing production definition; none is new.
SAFETY_CONTRACT = {
    "aggregation": "max over every command tick of the four-block branch",
    "terminal_only": False,
    "clearance": {
        "source": "lewm_worlds.scene_graph.SceneGraph.clearance_to_walls",
        "hazard_if_below_m": CLEARANCE_HAZARD_M,
    },
    "contact": {
        "source": "analyze_go2_closed_loop_quality._body_probe_configuration_clearance_m"
                  " over lewm_worlds.planning_grid.InflatedOccupancyGrid",
        "body_forward_m": 0.35,
        "body_half_width_m": 0.18,
        "body_probe_margin_m": 0.03,
        "hazard_if_at_or_below_m": 1e-4,
    },
    "stuck": {
        "source": "lewm_worlds.labels.derived.DerivedLabelComputer.stuck_label",
        "note": "PoseStep.last_command is the executed command of the tick that "
                "produced the displacement being measured",
    },
    "termination": {
        "source": "lewm_genesis.rollout.RolloutRunner._check_and_reset_fallen_envs "
                  "predicates, evaluated per tick",
        "events": ["fall_z_threshold", "out_of_bounds", "tipped"],
    },
}

CONTACT_BODY_FORWARD_M = 0.35
CONTACT_BODY_HALF_WIDTH_M = 0.18
CONTACT_BODY_PROBE_MARGIN_M = 0.03
CONTACT_HAZARD_M = 1e-4

# ---- frozen state-selection contract (candidate-blind; fixed before any branch)
SELECTION = {
    "pilot_pool_split": "val",
    "qualification_pool_split": "test_hard",
    "excluded_scene_panel": "docs/lewm_go2_world_model_counterfactual_calibration"
                            "_scene_panel_v1_2026-08-02.json",
    "states_per_family_base": 2,
    "extra_states_to_first_n_families": 4,
    "pilot_states": 20,
    "one_state_per_scene": True,
    "warmup_blocks_min": 40,
    "warmup_blocks_max": 120,
    "drive": "production RolloutRunner._collect_block collector mix, manifest spawn",
    "drive_seed_rule": "20260811 ^ crc32(scene_id)",
    "eligibility": [
        "block index >= warmup_blocks_min",
        "no reset fired in this block and reset_count unchanged since episode start",
        "production termination predicates all false",
        "SceneGraph.locate distance <= 2.0 m",
        "a landmark binds with bfs distance >= 1",
        "canonical boundary assertions pass",
    ],
    "landmark_binding": "min positive bfs_distance from the snapshot cell using "
                        "transit_blocked=nav_blocked_cells; ties by object_id",
    "capture_rule": "first eligible boundary at or after warmup_blocks_min",
    "qualification_capture_rule_contact_case": "eligible boundary in [20, 60) "
                                               "minimising body-probe clearance",
}
LOCATE_MAX_DISTANCE_M = 2.0
DRIVE_SEED_BASE = 20260811

# ---- development-only replay-qualification sequences (four required types)
QUALIFICATION_SEQUENCES = (
    ("sign_reversal", ("forward_fast", "backward", "forward_fast", "backward")),
    ("sustained_turn", ("yaw_left", "yaw_left", "yaw_left", "yaw_left")),
    ("obstacle_adjacent", ("forward_fast", "forward_fast", "forward_fast", "forward_fast")),
    ("slew_limited", ("forward_fast", "backward", "yaw_left", "arc_right")),
)

CORPUS = ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z"
PLATFORM_MANIFEST = ROOT / "config/go2_platform_manifest.yaml"
PRIMITIVE_REGISTRY = ROOT / "config/go2_primitive_registry.yaml"
OUT_DIR = ROOT / ".generated/go2_oracle_branch_pilot_v1"


# ---------------------------------------------------------------- digests ----
def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def bank_digest() -> str:
    return _sha({"bank": [[n, list(s)] for n, s in CANDIDATE_BANK],
                 "primitives": {k: list(v) for k, v in PRIMITIVES.items()},
                 "blocks": BLOCKS, "ticks": TICKS})


def oracle_digest() -> str:
    """Corrected oracle: path-level safety replaces the fall-only pre-run version."""

    return _sha({"weights": UTILITY, "progress_scale": PROGRESS_SCALE,
                 "clearance_hazard_m": CLEARANCE_HAZARD_M,
                 "safety": SAFETY_CONTRACT,
                 "completion": "bound landmark reached at or before the branch horizon",
                 "progress": "(start_bfs - final_bfs) / progress_scale, clipped to [-1, 1]",
                 "tie_tolerance": TIE_TOLERANCE,
                 "supersedes": SUPERSEDED_ORACLE_DIGEST_PRE_RUN})


def qualification_digest() -> str:
    return _sha({"sequences": [[n, list(s)] for n, s in QUALIFICATION_SEQUENCES],
                 "repetitions": 3, "controls": ["last_actions", "last_executed"],
                 "branch_order": "12 candidates forward then reverse, restore before each"})


def selection_digest() -> str:
    return _sha(SELECTION)


def block_for(primitive: str) -> np.ndarray:
    return np.tile(np.asarray(PRIMITIVES[primitive], dtype=np.float32), (TICKS, 1))


# ------------------------------------------------------------------ context --
@dataclass
class BranchContext:
    """Everything one scene's branch operations need, and the drive counters."""

    runner: Any
    policy: Any
    build: Any
    pack: Any
    scene_graph: Any
    grid: Any
    manifest: Any
    solver_fields: list = field(default_factory=list)
    # ``ticks_executed`` / ``policy_steps`` are *global* counters: the runner's
    # ``_sim_time_ns`` is a global sim clock that production never rewinds on an
    # episode reset, so the exact clock relation must be measured against them.
    # ``episode_ticks`` is the per-episode counter the factorial manifest's
    # block alignment is defined on, and follows production resets.
    ticks_executed: int = 0
    policy_steps: int = 0
    episode_ticks: int = 0
    last_block_executed: np.ndarray | None = None
    reset_in_last_block: bool = False
    episode_start_reset_count: int = 0

    # ---- drive ------------------------------------------------------------
    def begin_episode(self) -> None:
        runner = self.runner
        runner._reset_robot_to_spawn(envs_idx=None)
        for env_idx in range(runner.n_envs):
            runner._scheduler.on_episode_reset(env_idx)
            runner._blocks_in_episode[env_idx] = 0
            runner.episode_states[env_idx].episode_step = 0
        self.episode_ticks = 0
        self.last_block_executed = None
        self.reset_in_last_block = False
        self.episode_start_reset_count = int(runner.episode_states[0].reset_count)

    def drive_one_block(self) -> Any:
        """One production block: collect, execute, advance counters, run resets."""

        runner = self.runner
        requested, choices = runner._collect_block()
        block = runner.execute_requested_block(requested)
        # ``run()`` advances episode_step once per command tick via
        # _emit_per_tick_records; mirror it so the block-alignment invariant is
        # measured on the same counter the factorial manifest uses.
        for _ in range(runner._block_size):
            for state in runner.episode_states:
                state.step()
        runner._blocks_in_episode += 1
        self.ticks_executed += runner._block_size
        self.episode_ticks += runner._block_size
        self.policy_steps += runner._block_size * runner._policy_steps_per_command_tick
        self.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()

        before = int(runner.episode_states[0].reset_count)
        writer = _NullWriter()
        from lewm_genesis import ros_msg_adapter as adapter
        runner._check_and_reset_fallen_envs(writer, adapter)
        runner._check_and_reset_completed_envs(writer, adapter)
        after = int(runner.episode_states[0].reset_count)
        self.reset_in_last_block = after != before
        if self.reset_in_last_block:
            # ``EpisodeState.reset`` already zeroed episode_step; the global sim
            # clock and the global counters keep running, exactly as production.
            self.episode_ticks = 0
            self.last_block_executed = None
            self.episode_start_reset_count = after
        return block

    # ---- geometry ---------------------------------------------------------
    def pose(self) -> tuple[tuple[float, float], float, float]:
        robot = self.build.robot
        pos = np.asarray(self.runner._as_np(robot.get_pos()), dtype=np.float64)
        quat = np.asarray(self.runner._as_np(robot.get_quat()), dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[None, :]
        if quat.ndim == 1:
            quat = quat[None, :]
        from lewm_genesis.rollout import _yaw_from_quat_wxyz
        yaw = _yaw_from_quat_wxyz(*(float(v) for v in quat[0]))
        return (float(pos[0, 0]), float(pos[0, 1])), float(yaw), float(pos[0, 2])


class _NullWriter:
    """Production reset paths need a writer; the pilot records nothing."""

    def write_env(self, *args: Any, **kwargs: Any) -> None:
        return None

    def write_clock(self, *args: Any, **kwargs: Any) -> None:
        return None


# ------------------------------------------------------------------ boundary -
class BoundaryRefused(RuntimeError):
    """Raised instead of normalising a state into the canonical boundary."""


def _termination_flags(ctx: BranchContext) -> dict[str, bool]:
    """The production reset predicates, evaluated without resetting anything."""

    from lewm_genesis.rollout import _roll_from_quat_wxyz, _pitch_from_quat_wxyz
    runner = ctx.runner
    robot = ctx.build.robot
    pos = np.asarray(runner._as_np(robot.get_pos()), dtype=np.float64)
    quat = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64)
    if pos.ndim == 1:
        pos = pos[None, :]
    if quat.ndim == 1:
        quat = quat[None, :]
    fell = bool(pos[0, 2] < float(runner.config.fall_z_threshold_m))
    (xmin, ymin), (xmax, ymax) = ctx.pack.world_bounds_xy_m
    pad = float(runner.config.out_of_bounds_pad_m)
    oob = bool(pos[0, 0] < xmin - pad or pos[0, 0] > xmax + pad
               or pos[0, 1] < ymin - pad or pos[0, 1] > ymax + pad)
    qw, qx, qy, qz = (float(v) for v in quat[0])
    tip = max(abs(_roll_from_quat_wxyz(qw, qx, qy, qz)),
              abs(_pitch_from_quat_wxyz(qw, qx, qy, qz)))
    tipped = bool(tip > float(runner.config.tip_threshold_rad))
    nan = bool(not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)))
    return {"fall": fell, "out_of_bounds": oob, "tipped": tipped, "nan": nan}


def assert_canonical_boundary(ctx: BranchContext) -> dict[str, Any]:
    """The frozen boundary 1faae05f..., checked field by field.

    Capture is refused outside it.  Nothing here advances the simulator or the
    policy, and nothing normalises a misaligned state.
    """

    runner, policy = ctx.runner, ctx.policy
    fail: list[str] = []
    if int(runner.n_envs) != 1:
        fail.append(f"n_envs={runner.n_envs}, branch capture requires a single-env build")

    block_size = int(runner._block_size)
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    command_tick_phase = int(ctx.ticks_executed) % block_size
    decimation_phase = int(ctx.policy_steps) % steps_per_tick
    if command_tick_phase != 0:
        fail.append(f"command-block tick phase={command_tick_phase}, expected 0")
    if decimation_phase != 0:
        fail.append(f"low-level decimation phase={decimation_phase}, expected 0")

    # The sim clock is advanced by exactly one policy_dt per policy step, so the
    # emission phase is an exact integer relation, not an inferred one.
    expected_ns = int(ctx.policy_steps) * int(runner._policy_dt_ns)
    if int(runner._sim_time_ns) != expected_ns:
        fail.append(f"_sim_time_ns={runner._sim_time_ns}, expected {expected_ns}")
    emission_phase = int(runner._sim_time_ns) % int(runner._command_dt_ns)
    if emission_phase != 0:
        fail.append(f"observation-emission phase={emission_phase} ns, expected 0")

    if ctx.reset_in_last_block:
        fail.append("a reset fired in the preceding block")
    if int(runner.episode_states[0].reset_count) != int(ctx.episode_start_reset_count):
        fail.append("reset_count moved since the episode began")
    flags = _termination_flags(ctx)
    for name, value in flags.items():
        if value:
            fail.append(f"termination flag {name} is True")
    if int(runner._consecutive_tipped_blocks[0]) != 0:
        fail.append("consecutive-tipped accumulator is non-zero")

    source_step = int(ctx.episode_ticks) + 1
    if (source_step - 1) % TICKS != 0:
        fail.append(f"source step {source_step} violates (s-1) mod {TICKS} == 0")
    episode_step = int(runner.episode_states[0].episode_step)
    if episode_step != int(ctx.episode_ticks):
        fail.append(f"episode_step={episode_step} disagrees with driver episode ticks="
                    f"{ctx.episode_ticks}")

    last_actions = getattr(policy, "_last_actions", None)
    if last_actions is None:
        fail.append("policy._last_actions is uninitialised at capture")
    else:
        arr = np.asarray(last_actions)
        if arr.shape != (1, len(policy.policy_joint_names)):
            fail.append(f"_last_actions shape {arr.shape} is not (1, "
                        f"{len(policy.policy_joint_names)})")
        elif not np.all(np.isfinite(arr)):
            fail.append("_last_actions contains non-finite entries")
    last_executed = np.asarray(runner._last_executed, dtype=np.float64)
    if last_executed.shape != (1, 3):
        fail.append(f"_last_executed shape {last_executed.shape} is not (1, 3)")
    elif not np.all(np.isfinite(last_executed)):
        fail.append("_last_executed contains non-finite entries")
    else:
        limits = runner.safety
        vx, vy, yaw_rate = (float(v) for v in last_executed[0])
        if not (limits.min_vx_mps - 1e-6 <= vx <= limits.max_vx_mps + 1e-6):
            fail.append(f"_last_executed vx={vx} outside the safety envelope")
        if not (limits.min_vy_mps - 1e-6 <= vy <= limits.max_vy_mps + 1e-6):
            fail.append(f"_last_executed vy={vy} outside the safety envelope")
        if abs(yaw_rate) > limits.max_yaw_rate_radps + 1e-6:
            fail.append(f"_last_executed yaw_rate={yaw_rate} outside the safety envelope")
    if ctx.last_block_executed is None:
        fail.append("no executed block precedes the boundary")
    elif last_executed.shape == (1, 3):
        expected_last = np.asarray(ctx.last_block_executed[0, -1, :], dtype=np.float64)
        if not np.array_equal(expected_last, last_executed[0]):
            fail.append("_last_executed disagrees with the final tick of the last "
                        f"executed block ({last_executed[0]} vs {expected_last})")

    if fail:
        raise BoundaryRefused("; ".join(fail))
    return {
        "command_block_tick": command_tick_phase,
        "decimation_phase": decimation_phase,
        "observation_emission_phase_ns": emission_phase,
        "reset": False, "terminated": False, "truncated": False,
        "source_step": source_step,
        "episode_step": episode_step,
        "sim_time_ns": int(runner._sim_time_ns),
        "boundary_digest": BOUNDARY_DIGEST,
    }


# ------------------------------------------------------------------ snapshot --
_HARNESS_ARRAY_FIELDS = (
    "_last_executed", "_blocks_in_episode", "_consecutive_tipped_blocks",
    "_recovery_interlock_blocks_remaining", "_spawn_xyz_per_env",
    "_spawn_quat_wxyz_per_env", "_per_env_path_length_m",
    "_per_env_target_changes", "_per_env_recovery_handoffs",
)
_HARNESS_OBJECT_FIELDS = (
    "_per_env_last_xy", "_per_env_cells_visited", "_per_env_primitive_counts",
    "_per_env_source_counts", "_per_env_goals_targeted", "_per_env_prev_target",
    "_per_env_achieved_landmarks", "_per_env_route_completions",
    "_isotropy_block_counts", "_isotropy_cell_yaw_counts",
    "_isotropy_primitive_counts",
)


class BranchSnapshot:
    """In-process snapshot captured only at the canonical boundary."""

    __slots__ = ("solver_state", "step_index", "last_actions", "harness", "rng",
                 "counters", "goal", "identity", "boundary", "digest")

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


# ``Scene.save_checkpoint`` walks only ``solver.__dict__`` and the solver's
# data_manager.  Replay qualification proved that leaves out the collider,
# constraint-solver, contact-island and GJK state that the solver genuinely
# carries between steps, so branches from the same snapshot diverged at the
# first physics step.  The snapshot therefore captures every gstaichi field
# reachable from the scene and its solvers, minus the static geometry tables
# below (1.89 GB of SDF and support-field lookup data that never mutates).
_STATIC_FIELD_MARKERS = (
    "._sdf_info.geoms_sdf_",
    "._support_field_info.support_",
)
_FIELD_WALK_DEPTH = 4


def _is_static_field(path: str) -> bool:
    return any(marker in path for marker in _STATIC_FIELD_MARKERS)


def collect_solver_fields(scene: Any) -> list[tuple[str, Any]]:
    """Every mutable gstaichi field reachable from the scene, in a stable order."""

    import gstaichi as ti
    found: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def walk(obj: Any, path: str, depth: int) -> None:
        if id(obj) in seen or depth > _FIELD_WALK_DEPTH:
            return
        seen.add(id(obj))
        for name, value in sorted(getattr(obj, "__dict__", {}).items()):
            if name.startswith("__"):
                continue
            sub = f"{path}.{name}"
            if isinstance(value, (ti.Field, ti.Ndarray)):
                if not _is_static_field(sub):
                    found.append((sub, value))
            elif hasattr(value, "__dict__") and not isinstance(
                    value, (str, bytes, np.ndarray, type)):
                walk(value, sub, depth + 1)

    walk(scene, type(scene).__name__, 0)
    for solver in scene.active_solvers:
        walk(solver, type(solver).__name__, 0)
    return found


def dump_solver_state(fields: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {path: field.to_numpy() for path, field in fields}


def load_solver_state(fields: Sequence[tuple[str, Any]],
                      state: dict[str, Any]) -> None:
    for path, field in fields:
        value = state.get(path)
        if value is not None:
            field.from_numpy(value)


def _solver_state_digest(state: dict[str, Any], step_index: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(int(step_index)).encode())

    def absorb(key: str, value: Any) -> None:
        if isinstance(value, dict):
            for sub in sorted(value):
                absorb(f"{key}.{sub}", value[sub])
            return
        array = np.ascontiguousarray(value)
        hasher.update(key.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())

    for path in sorted(state):
        absorb(path, state[path])
    return hasher.hexdigest()


def _domain_digest(parts: Sequence[tuple[bytes, bytes]]) -> str:
    """Snapshot schema v1.1: len(tag) || tag || len(payload) || payload."""

    hasher = hashlib.sha256()
    for tag, payload in parts:
        hasher.update(len(tag).to_bytes(4, "big"))
        hasher.update(tag)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return hasher.hexdigest()


def capture_branch_state(ctx: BranchContext, *, goal: dict, identity: dict,
                         checkpoint_dir: Path | None = None) -> BranchSnapshot:
    boundary = assert_canonical_boundary(ctx)
    solver_state = dump_solver_state(ctx.solver_fields)
    step_index = int(getattr(ctx.build.scene, "t", -1))

    runner, policy = ctx.runner, ctx.policy
    harness = {name: np.array(getattr(runner, name), copy=True)
               for name in _HARNESS_ARRAY_FIELDS}
    harness_obj = {name: copy.deepcopy(getattr(runner, name))
                   for name in _HARNESS_OBJECT_FIELDS}
    harness_obj["episode_states"] = copy.deepcopy(runner.episode_states)
    harness_obj["_sim_time_ns"] = int(runner._sim_time_ns)
    harness_obj["_sequence_id_counter"] = int(runner._sequence_id_counter)

    import torch
    rng = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "runner_rng": runner._rng.bit_generator.state,
        "spawn_rng": runner._spawn_rng.getstate(),
        "torch": torch.get_rng_state().clone(),
    }
    counters = {"ticks_executed": int(ctx.ticks_executed),
                "policy_steps": int(ctx.policy_steps),
                "episode_ticks": int(ctx.episode_ticks),
                "reset_in_last_block": bool(ctx.reset_in_last_block),
                "episode_start_reset_count": int(ctx.episode_start_reset_count)}
    last_block = (None if ctx.last_block_executed is None
                  else np.array(ctx.last_block_executed, copy=True))

    snap = BranchSnapshot(
        solver_state=solver_state, step_index=step_index,
        last_actions=np.array(policy._last_actions, dtype=np.float32, copy=True),
        harness={"arrays": harness, "objects": harness_obj, "last_block": last_block},
        rng=rng, counters=counters,
        goal=dict(goal), identity=dict(identity), boundary=boundary, digest=None,
    )

    controller_payload = b"".join([
        snap.last_actions.tobytes(),
        np.ascontiguousarray(harness["_last_executed"], dtype=np.float32).tobytes(),
        pickle.dumps(rng["python"], protocol=4),
        pickle.dumps(rng["numpy_global"], protocol=4),
        pickle.dumps(rng["runner_rng"], protocol=4),
        pickle.dumps(rng["spawn_rng"], protocol=4),
        rng["torch"].numpy().tobytes(),
    ])
    harness_payload = json.dumps({
        "identity": snap.identity, "goal": snap.goal, "boundary": boundary,
        "counters": counters,
        "episode": [{"episode_id": s.episode_id, "episode_step": s.episode_step,
                     "reset_count": s.reset_count}
                    for s in harness_obj["episode_states"]],
        "blocks_in_episode": harness["_blocks_in_episode"].tolist(),
        "consecutive_tipped_blocks": harness["_consecutive_tipped_blocks"].tolist(),
    }, sort_keys=True).encode()
    snap.digest = _domain_digest([
        (b"GENESIS_SOLVER_V1", _solver_state_digest(solver_state, step_index).encode()),
        (b"CONTROLLER_RNG_V1", controller_payload),
        (b"HARNESS_STATE_V1", harness_payload),
    ])
    return snap


def restore_branch_state(ctx: BranchContext, snapshot: BranchSnapshot) -> None:
    """Restore all three layers.  No policy or simulator step is taken."""

    runner, policy = ctx.runner, ctx.policy
    load_solver_state(ctx.solver_fields, snapshot.solver_state)
    if snapshot.step_index >= 0:
        ctx.build.scene._t = int(snapshot.step_index)
    policy._last_actions = np.array(snapshot.last_actions, dtype=np.float32, copy=True)
    for name, value in snapshot.harness["arrays"].items():
        setattr(runner, name, np.array(value, copy=True))
    objects = snapshot.harness["objects"]
    for name in _HARNESS_OBJECT_FIELDS:
        setattr(runner, name, copy.deepcopy(objects[name]))
    runner.episode_states = copy.deepcopy(objects["episode_states"])
    runner._sim_time_ns = int(objects["_sim_time_ns"])
    runner._sequence_id_counter = int(objects["_sequence_id_counter"])

    import torch
    random.setstate(snapshot.rng["python"])
    np.random.set_state(snapshot.rng["numpy_global"])
    runner._rng.bit_generator.state = copy.deepcopy(snapshot.rng["runner_rng"])
    runner._spawn_rng.setstate(snapshot.rng["spawn_rng"])
    torch.set_rng_state(snapshot.rng["torch"].clone())

    ctx.ticks_executed = int(snapshot.counters["ticks_executed"])
    ctx.policy_steps = int(snapshot.counters["policy_steps"])
    ctx.episode_ticks = int(snapshot.counters["episode_ticks"])
    ctx.reset_in_last_block = bool(snapshot.counters["reset_in_last_block"])
    ctx.episode_start_reset_count = int(snapshot.counters["episode_start_reset_count"])
    last_block = snapshot.harness["last_block"]
    ctx.last_block_executed = None if last_block is None else np.array(last_block, copy=True)
    assert_canonical_boundary(ctx)


# ------------------------------------------------------------------- branch ---
def _instrumented_act(policy: Any, sink: list[dict[str, Any]]) -> Callable:
    """Wrap ``policy.act`` so a trace can see every controller quantity.

    The wrapper only *reads*; ``_build_policy_observation`` is a pure function
    of the observation and ``_last_actions``, so the numerical path is
    unchanged.
    """

    original = policy.act

    def wrapped(observation: dict[str, np.ndarray]) -> np.ndarray:
        before = (None if policy._last_actions is None
                  else np.array(policy._last_actions, copy=True))
        obs_np = (None if before is None
                  else np.array(policy._build_policy_observation(observation), copy=True))
        targets = original(observation)
        after = np.array(policy._last_actions, copy=True)
        applied = before if policy.simulate_action_latency else after
        sink.append({
            "policy_observation": obs_np,
            "raw_actor_output": after,           # act() stores the raw actor output
            "last_actions_before": before,
            "last_actions_after": after,
            "applied_action_under_latency": applied,
            "requested_command": np.array(observation["command"], copy=True),
            "joint_targets": np.array(targets, copy=True),
        })
        return targets

    return wrapped


def _tick_evidence(ctx: BranchContext, label_computer: Any, pose_step_cls: Any,
                   *, env_idx: int, episode_id: int, episode_step: int,
                   stamp_ns: int, executed_cmd: Sequence[float],
                   goal_cell: int) -> dict[str, Any]:
    """One command tick of path-level safety and goal evidence."""

    (x, y), yaw, z = ctx.pose()
    label = label_computer.step(pose_step_cls(
        timestamp_ns=int(stamp_ns), env_idx=int(env_idx), episode_id=int(episode_id),
        episode_step=int(episode_step), position_xy_world=(x, y),
        yaw_world_rad=float(yaw),
        last_command=(float(executed_cmd[0]), float(executed_cmd[1]),
                      float(executed_cmd[2])),
    ))
    from analyze_go2_closed_loop_quality import _body_probe_configuration_clearance_m
    body_clearance = _body_probe_configuration_clearance_m(
        ctx.grid, [x, y], yaw,
        body_forward_m=CONTACT_BODY_FORWARD_M,
        body_half_width_m=CONTACT_BODY_HALF_WIDTH_M,
        body_probe_margin_m=CONTACT_BODY_PROBE_MARGIN_M)
    flags = _termination_flags(ctx)
    blocked = getattr(ctx.scene_graph, "nav_blocked_cells", frozenset())
    hit = ctx.scene_graph.locate((x, y))
    cell = int(hit.cell_id)
    located = bool(float(hit.distance_m) <= LOCATE_MAX_DISTANCE_M)
    bfs = (ctx.scene_graph.bfs_distance(cell, int(goal_cell), transit_blocked=blocked)
           if located else None)

    contact = bool(body_clearance <= CONTACT_HAZARD_M)
    low_clearance = bool(float(label.clearance_m) < CLEARANCE_HAZARD_M)
    stuck = bool(label.stuck_label)
    terminated = bool(flags["fall"] or flags["out_of_bounds"] or flags["tipped"])
    return {
        "xy": [x, y], "yaw": yaw, "z": z,
        "cell_id": cell, "located": located, "cell_distance_m": float(hit.distance_m),
        "bfs_to_goal": None if bfs is None else int(bfs),
        "clearance_m": float(label.clearance_m),
        "body_clearance_m": float(body_clearance),
        "contact": contact, "low_clearance": low_clearance, "stuck": stuck,
        "termination": flags, "terminated": terminated,
        "nan": bool(flags["nan"]),
        "hazard": bool(contact or low_clearance or stuck or terminated),
    }


def execute_candidate(ctx: BranchContext, snapshot: BranchSnapshot,
                      candidate: tuple[str, tuple[str, ...]], *,
                      trace: bool = False) -> dict[str, Any]:
    """Restore, then run the four-block candidate through the production path."""

    from lewm_worlds.labels.derived import (DerivedLabelComputer, DerivedLabelConfig,
                                            PoseStep)
    name, primitives = candidate
    restore_branch_state(ctx, snapshot)
    runner, policy = ctx.runner, ctx.policy
    goal_cell = int(snapshot.goal["landmark_cell"])

    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    episode_step = int(runner.episode_states[0].episode_step)
    stamp_ns = int(runner._sim_time_ns)
    # Seed the label history with the boundary pose so the first tick's stuck
    # label has a displacement to measure against.
    start_evidence = _tick_evidence(
        ctx, label_computer, PoseStep, env_idx=0, episode_id=episode_id,
        episode_step=episode_step, stamp_ns=stamp_ns,
        executed_cmd=np.asarray(runner._last_executed, dtype=np.float64)[0],
        goal_cell=goal_cell)

    controller_sink: list[dict[str, Any]] = []
    if trace:
        policy.act = _instrumented_act(policy, controller_sink)

    requested_all: list[list[list[float]]] = []
    executed_all: list[list[list[float]]] = []
    clipped_any = False
    tick_rows: list[dict[str, Any]] = [start_evidence]
    step_rows: list[dict[str, Any]] = []
    truncated_at_block: int | None = None
    nan_seen = False

    try:
        for block_idx, primitive in enumerate(primitives):
            requested = block_for(primitive)[None, ...]
            executed_block = np.asarray(
                runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
                dtype=np.float64)
            pending: list[dict[str, Any]] = []

            def after_policy_step(tick_idx: int, step_idx: int,
                                  _block=executed_block) -> None:
                if trace and controller_sink:
                    row = controller_sink[-1]
                    step_rows.append({
                        "block": block_idx, "tick": int(tick_idx), "step": int(step_idx),
                        "policy_observation": row["policy_observation"],
                        "raw_actor_output": row["raw_actor_output"],
                        "last_actions_before": row["last_actions_before"],
                        "last_actions_after": row["last_actions_after"],
                        "applied_action_under_latency": row["applied_action_under_latency"],
                        "requested_command": row["requested_command"],
                        "post_slew_command": _block[0, tick_idx].copy(),
                        "joint_targets": row["joint_targets"],
                        "root_pos": np.asarray(runner._as_np(ctx.build.robot.get_pos())).copy(),
                        "root_quat": np.asarray(runner._as_np(ctx.build.robot.get_quat())).copy(),
                        "root_lin_vel": np.asarray(runner._as_np(ctx.build.robot.get_vel())).copy(),
                        "root_ang_vel": np.asarray(runner._as_np(ctx.build.robot.get_ang())).copy(),
                        "joint_pos": np.asarray(runner._as_np(
                            ctx.build.robot.get_dofs_position(runner._leg_dof_idx.tolist()))).copy(),
                        "joint_vel": np.asarray(runner._as_np(
                            ctx.build.robot.get_dofs_velocity(runner._leg_dof_idx.tolist()))).copy(),
                    })
                if step_idx != runner._policy_steps_per_command_tick - 1:
                    return
                pending.append({"tick_idx": tick_idx})

            block = runner.execute_requested_block(
                requested, after_policy_step=after_policy_step)
            # Evidence is sampled at the production 10 Hz emission instant, i.e.
            # after each command tick completes.  ``after_policy_step`` cannot
            # reach the label computer safely mid-callback, so the ticks are
            # replayed here from the recorded per-tick boundary poses.
            del pending
            requested_all.append(np.asarray(block.requested)[0].tolist())
            executed_all.append(np.asarray(block.executed)[0].tolist())
            clipped_any = clipped_any or bool(np.asarray(block.clipped)[0])
            ctx.ticks_executed += runner._block_size
            ctx.episode_ticks += runner._block_size
            ctx.policy_steps += runner._block_size * runner._policy_steps_per_command_tick
            ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()

            executed = np.asarray(block.executed, dtype=np.float64)[0]
            for tick_idx in range(runner._block_size):
                episode_step += 1
                stamp_ns += int(runner._command_dt_ns)
                evidence = _tick_evidence(
                    ctx, label_computer, PoseStep, env_idx=0, episode_id=episode_id,
                    episode_step=episode_step, stamp_ns=stamp_ns,
                    executed_cmd=executed[tick_idx], goal_cell=goal_cell)
                evidence["block"] = block_idx
                evidence["tick"] = tick_idx
                tick_rows.append(evidence)
            if tick_rows[-1]["nan"]:
                nan_seen = True
                truncated_at_block = block_idx
                break
            if tick_rows[-1]["terminated"]:
                truncated_at_block = block_idx
                break
    finally:
        if trace:
            policy.act = type(policy).act.__get__(policy, type(policy))

    return {
        "candidate": name, "primitives": list(primitives),
        "requested": requested_all, "executed": executed_all,
        "clipped": clipped_any,
        "blocks_completed": len(executed_all),
        "truncated_at_block": truncated_at_block,
        "nan": nan_seen,
        "ticks": tick_rows,
        "steps": step_rows,
    }


# ------------------------------------------------------------------- oracle ---
def oracle_components(branch: dict[str, Any], *, start_bfs: int) -> dict[str, Any] | None:
    """progress / safety / completion under the frozen design-v1.1 definitions."""

    ticks = branch["ticks"]
    if branch["nan"]:
        return None
    moving = ticks[1:]
    if not moving:
        return None
    final = moving[-1]
    if not final["located"] or final["bfs_to_goal"] is None:
        return None
    final_bfs = int(final["bfs_to_goal"])
    progress = float(np.clip((start_bfs - final_bfs) / PROGRESS_SCALE, -1.0, 1.0))
    # Path-level: max over every tick of the branch, never terminal-only.
    safety = 1.0 if any(bool(row["hazard"]) for row in moving) else 0.0
    completion = 1.0 if any(row["bfs_to_goal"] == 0 for row in moving
                            if row["bfs_to_goal"] is not None) else 0.0
    utility = (UTILITY["progress"] * progress + UTILITY["safety"] * safety
               + UTILITY["completion"] * completion)
    evidence = {
        "contact": any(row["contact"] for row in moving),
        "low_clearance": any(row["low_clearance"] for row in moving),
        "stuck": any(row["stuck"] for row in moving),
        "terminated": any(row["terminated"] for row in moving),
    }
    return {"progress": progress, "safety": safety, "completion": completion,
            "utility": utility, "start_bfs": int(start_bfs), "final_bfs": final_bfs,
            "min_bfs": min(int(r["bfs_to_goal"]) for r in moving
                           if r["bfs_to_goal"] is not None),
            "safety_evidence": evidence,
            "min_clearance_m": min(float(r["clearance_m"]) for r in moving),
            "min_body_clearance_m": min(float(r["body_clearance_m"]) for r in moving)}


# --------------------------------------------------------------------- gate ---
def identifiability(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_state.setdefault(row["state_id"], []).append(row)
    valid_states, separated, levels, spreads = 0, 0, [], []
    families: dict[str, int] = {}
    attempted = len(records)
    invalid = sum(1 for r in records if not r["valid"])
    for _state_id, rows in by_state.items():
        good = [r for r in rows if r["valid"]]
        if len(good) < 2:
            continue
        valid_states += 1
        family = rows[0]["family"]
        families[family] = families.get(family, 0) + 1
        utilities = sorted((float(r["utility"]) for r in good), reverse=True)
        if utilities[0] - utilities[1] > TIE_TOLERANCE:
            separated += 1
        distinct: list[float] = []
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
        "per_family_valid_states": families,
    }


def gate_verdict(stats: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "uniquely_separated_ge_0.70": stats["uniquely_separated_fraction"] >= 0.70,
        "median_distinct_levels_ge_5": stats["median_distinct_levels"] >= 5,
        "median_spread_ge_0.10": stats["median_spread"] >= 0.10,
        "invalid_rate_le_0.20": stats["invalid_rate"] <= 0.20,
        "all_eight_families_two_valid_states": stats["families_with_two_valid_states"] >= 8,
    }
    return {"components": checks, "pass": all(checks.values())}


# ---------------------------------------------------------------- identities --
def _excluded_scene_ids() -> set[str]:
    panel = ROOT / SELECTION["excluded_scene_panel"]
    if not panel.is_file():
        raise FileNotFoundError(f"missing factorial scene panel: {panel}")
    data = json.loads(panel.read_text())
    return {str(entry["scene_id"]) for entry in data.get("scenes", [])}


def _scene_index(split: str) -> dict[str, list[Path]]:
    root = CORPUS / split
    families: dict[str, list[Path]] = {}
    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir():
            continue
        scenes = [d for d in sorted(family_dir.iterdir())
                  if (d / "manifest.json").is_file() and (d / "genesis_scene.json").is_file()]
        if scenes:
            families[family_dir.name] = scenes
    return families


def build_identity_manifests() -> dict[str, Any]:
    """Freeze the pilot and replay identities.  Simulates nothing."""

    excluded = _excluded_scene_ids()
    pilot_families = _scene_index(SELECTION["pilot_pool_split"])
    family_names = sorted(pilot_families)
    if len(family_names) != 8:
        raise RuntimeError(f"expected 8 families in the pilot pool, found {family_names}")

    pilot: list[dict[str, Any]] = []
    for family_index, family in enumerate(family_names):
        count = SELECTION["states_per_family_base"]
        if family_index < SELECTION["extra_states_to_first_n_families"]:
            count += 1
        picked = [d for d in pilot_families[family] if d.name not in excluded][:count]
        if len(picked) < count:
            raise RuntimeError(f"family {family} has {len(picked)} usable scenes, need {count}")
        for slot, scene_dir in enumerate(picked):
            pilot.append({
                "state_id": f"pilot-{family}-{slot}",
                "family": family, "split": SELECTION["pilot_pool_split"],
                "scene_id": scene_dir.name, "scene_dir": str(scene_dir),
                "drive_seed": _drive_seed(scene_dir.name),
                "capture_rule": "first_eligible",
            })
    if len(pilot) != SELECTION["pilot_states"]:
        raise RuntimeError(f"pilot manifest has {len(pilot)} states, expected 20")

    qual_families = _scene_index(SELECTION["qualification_pool_split"])
    # rough_local_dynamics is excluded from the *development* qualification pool
    # only: that family is the known rigid-solver NaN family, and a qualification
    # case must fail on replay determinism, not on a solver blow-up.  It remains
    # fully in the pilot pool.
    qual_family_names = [f for f in sorted(qual_families) if f != "rough_local_dynamics"]
    cases = [name for name, _ in QUALIFICATION_SEQUENCES]
    if len(qual_family_names) < len(cases):
        raise RuntimeError("not enough qualification families")
    pilot_scene_ids = {row["scene_id"] for row in pilot}
    replay: list[dict[str, Any]] = []
    for case, family in zip(cases, qual_family_names):
        scene_dir = qual_families[family][0]
        if scene_dir.name in pilot_scene_ids:
            raise RuntimeError("replay/pilot scene collision")
        replay.append({
            "state_id": f"replay-{case}",
            "case": case, "family": family,
            "split": SELECTION["qualification_pool_split"],
            "scene_id": scene_dir.name, "scene_dir": str(scene_dir),
            "drive_seed": _drive_seed(scene_dir.name),
            "capture_rule": ("min_body_clearance" if case == "obstacle_adjacent"
                             else "first_eligible"),
            "sequence": list(dict(QUALIFICATION_SEQUENCES)[case]),
        })

    manifest = {
        "schema": "go2_oracle_branch_pilot_identity_manifest_v1",
        "status": STATUS,
        "selection_digest": selection_digest(),
        "candidate_bank_digest": bank_digest(),
        "oracle_contract_digest": oracle_digest(),
        "qualification_digest": qualification_digest(),
        "boundary": BOUNDARY_DIGEST, "inventory_v2": INVENTORY_V2,
        "design_v1_1": DESIGN_V1_1,
        "selection": SELECTION,
        "pilot_states": pilot,
        "replay_states": replay,
        "disjointness": {
            "replay_disjoint_from_pilot": True,
            "pilot_scenes_excluded_from_factorial_panel": True,
            "pilot_identities_unavailable_for_scorer_fit_and_evaluation": True,
        },
    }
    manifest["identity_manifest_digest"] = _sha(
        {k: v for k, v in manifest.items() if k != "identity_manifest_digest"})
    return manifest


def _drive_seed(scene_id: str) -> int:
    import zlib
    return int(DRIVE_SEED_BASE ^ (zlib.crc32(scene_id.encode()) & 0x7FFFFFFF))


# ------------------------------------------------------------------- runtime --
def build_context(scene_dir: Path, *, seed: int, backend: str,
                  shared: dict[str, Any]) -> BranchContext:
    from lewm_genesis.scene_loader import load_scene_pack
    from lewm_genesis.scene_builder import build_scene_from_pack
    from lewm_genesis.rollout import RolloutRunner, RolloutConfig
    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    pack = load_scene_pack(scene_dir, platform_manifest=shared["platform"],
                           workspace_root=ROOT)
    build = build_scene_from_pack(pack, n_envs=1, backend=backend, show_viewer=False,
                                  render_robot=False, apply_textures=False)
    runner = RolloutRunner(build, shared["policy"], shared["registry"], shared["safety"],
                           config=RolloutConfig(
                               n_blocks=1, fall_z_threshold_m=0.15,
                               rgb_capture_per_block=False, seed=int(seed),
                               log_progress_every_blocks=0, foot_contact_source="zero",
                               randomize_spawn_pose=False))
    manifest = pack.scene_graph.manifest if pack.scene_graph is not None else None
    if manifest is None:
        raise RuntimeError(f"scene {pack.scene_id} has no scene graph")
    return BranchContext(runner=runner, policy=shared["policy"], build=build, pack=pack,
                         scene_graph=pack.scene_graph, manifest=manifest,
                         grid=InflatedOccupancyGrid(manifest),
                         solver_fields=collect_solver_fields(build.scene))


def bind_landmark(ctx: BranchContext, cell_id: int) -> dict[str, Any] | None:
    """The frozen snapshot-time goal binding.  Never re-chosen after the fact."""

    graph = ctx.scene_graph
    blocked = getattr(graph, "nav_blocked_cells", frozenset())
    best: tuple[int, str, int] | None = None
    for name, landmark_cell in sorted(graph.landmark_cells, key=lambda kv: str(kv[0])):
        distance = graph.bfs_distance(int(cell_id), int(landmark_cell),
                                      transit_blocked=blocked)
        if distance is None or int(distance) < 1:
            continue
        key = (int(distance), str(name), int(landmark_cell))
        if best is None or key < best:
            best = key
    if best is None:
        return None
    distance, name, landmark_cell = best
    centre = graph.cell_center(int(landmark_cell))
    return {"landmark_id": name, "landmark_cell": int(landmark_cell),
            "bfs_distance_cells": int(distance),
            "landmark_xy_m": [float(centre[0]), float(centre[1])]}


def find_capture(ctx: BranchContext, *, rule: str,
                 identity: dict[str, Any]) -> dict[str, Any] | None:
    """Drive the production collectors and stop at the frozen eligible boundary.

    Candidate-blind by construction: no candidate is executed, and no branch
    outcome is available, while the state is being chosen.
    """

    ctx.begin_episode()
    low_block, high_block = (20, 60) if rule == "min_body_clearance" else (
        SELECTION["warmup_blocks_min"], SELECTION["warmup_blocks_max"])
    best: dict[str, Any] | None = None
    from analyze_go2_closed_loop_quality import _body_probe_configuration_clearance_m
    for block_idx in range(high_block):
        ctx.drive_one_block()
        if block_idx + 1 < low_block:
            continue
        try:
            boundary = assert_canonical_boundary(ctx)
        except BoundaryRefused:
            continue
        (x, y), yaw, _z = ctx.pose()
        hit = ctx.scene_graph.locate((x, y))
        if float(hit.distance_m) > LOCATE_MAX_DISTANCE_M:
            continue
        goal = bind_landmark(ctx, int(hit.cell_id))
        if goal is None:
            continue
        record = {"boundary": boundary, "block_index": block_idx + 1,
                  "cell_id": int(hit.cell_id), "goal": goal,
                  "body_clearance_m": float(_body_probe_configuration_clearance_m(
                      ctx.grid, [x, y], yaw,
                      body_forward_m=CONTACT_BODY_FORWARD_M,
                      body_half_width_m=CONTACT_BODY_HALF_WIDTH_M,
                      body_probe_margin_m=CONTACT_BODY_PROBE_MARGIN_M))}
        # The snapshot is taken inline: a candidate boundary cannot be revisited
        # by re-driving, because the global sim clock never rewinds.
        record["snapshot"] = capture_branch_state(
            ctx, goal=goal,
            identity=dict(identity, block_index=block_idx + 1,
                          source_step=boundary["source_step"],
                          episode_id=int(ctx.runner.episode_states[0].episode_id)))
        if rule == "first_eligible":
            return record
        if best is None or record["body_clearance_m"] < best["body_clearance_m"]:
            best = record
    return best


def _load_shared(backend: str) -> dict[str, Any]:
    from lewm_genesis.scene_loader import load_platform_manifest
    from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits
    from lewm_genesis.rollout import GenesisGo2PPOPolicy
    platform = load_platform_manifest(PLATFORM_MANIFEST)
    registry = PrimitiveRegistry.from_yaml(PRIMITIVE_REGISTRY)
    safety = SafetyLimits.from_manifest(platform)
    policy = GenesisGo2PPOPolicy.from_platform_manifest(platform, ROOT, device="cpu")
    return {"platform": platform, "registry": registry, "safety": safety,
            "policy": policy, "backend": backend}


# ------------------------------------------------------------- trace digest ---
_TRACE_FIELDS = (
    "policy_observation", "raw_actor_output", "last_actions_before",
    "last_actions_after", "applied_action_under_latency", "requested_command",
    "post_slew_command", "joint_targets", "root_pos", "root_quat",
    "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel",
)
_TICK_FIELDS = ("xy", "yaw", "z", "cell_id", "bfs_to_goal", "clearance_m",
                "body_clearance_m", "contact", "low_clearance", "stuck",
                "terminated", "hazard")


def trace_digest(branch: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for row in branch["steps"]:
        for field_name in _TRACE_FIELDS:
            value = row.get(field_name)
            if value is None:
                hasher.update(b"None")
                continue
            hasher.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    for row in branch["ticks"]:
        for field_name in _TICK_FIELDS:
            hasher.update(repr(row.get(field_name)).encode())
    hasher.update(json.dumps({"requested": branch["requested"],
                              "executed": branch["executed"],
                              "clipped": branch["clipped"],
                              "blocks_completed": branch["blocks_completed"],
                              "truncated_at_block": branch["truncated_at_block"]},
                             sort_keys=True).encode())
    return hasher.hexdigest()


def first_divergence(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    for index, (row_a, row_b) in enumerate(zip(a["steps"], b["steps"])):
        for field_name in _TRACE_FIELDS:
            va, vb = row_a.get(field_name), row_b.get(field_name)
            if va is None and vb is None:
                continue
            if va is None or vb is None or not np.array_equal(np.asarray(va), np.asarray(vb)):
                return {"where": "steps", "index": index, "field": field_name,
                        "block": row_a.get("block"), "tick": row_a.get("tick"),
                        "step": row_a.get("step"),
                        "a": np.asarray(va).ravel()[:4].tolist() if va is not None else None,
                        "b": np.asarray(vb).ravel()[:4].tolist() if vb is not None else None}
    if len(a["steps"]) != len(b["steps"]):
        return {"where": "steps", "field": "length", "a": len(a["steps"]),
                "b": len(b["steps"])}
    for index, (row_a, row_b) in enumerate(zip(a["ticks"], b["ticks"])):
        for field_name in _TICK_FIELDS:
            if repr(row_a.get(field_name)) != repr(row_b.get(field_name)):
                return {"where": "ticks", "index": index, "field": field_name,
                        "block": row_a.get("block"), "tick": row_a.get("tick"),
                        "a": row_a.get(field_name), "b": row_b.get(field_name)}
    if len(a["ticks"]) != len(b["ticks"]):
        return {"where": "ticks", "field": "length", "a": len(a["ticks"]),
                "b": len(b["ticks"])}
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ------------------------------------------------------------------ stages ----
def stage_qualify(args: argparse.Namespace) -> int:
    manifest = json.loads((OUT_DIR / "identity_manifest.json").read_text())
    shared = _load_shared(args.backend)
    results: dict[str, Any] = {"schema": "go2_oracle_branch_replay_qualification_v1",
                               "status": STATUS,
                               "candidate_bank_digest": bank_digest(),
                               "oracle_contract_digest": oracle_digest(),
                               "qualification_digest": qualification_digest(),
                               "identity_manifest_digest": manifest["identity_manifest_digest"],
                               "genesis_backend": args.backend,
                               "snapshot_scope": "all reachable gstaichi fields minus "
                                                 "the static SDF/support-field tables",
                               "cases": [], "controls": {}, "branch_order": {}}
    started = time.time()
    order_state_done = False

    for entry in manifest["replay_states"]:
        case_started = time.time()
        print(f"[qualify] {entry['state_id']} ({entry['family']})", flush=True)
        ctx = build_context(Path(entry["scene_dir"]), seed=entry["drive_seed"],
                            backend=args.backend, shared=shared)
        capture = find_capture(
            ctx, rule=entry["capture_rule"],
            identity={"state_id": entry["state_id"], "case": entry["case"],
                      "scene_id": entry["scene_id"], "family": entry["family"],
                      "split": entry["split"]})
        if capture is None:
            results["cases"].append({"case": entry["case"], "state_id": entry["state_id"],
                                     "result": "no_eligible_boundary"})
            continue
        snapshot = capture["snapshot"]
        sequence = (entry["case"], tuple(entry["sequence"]))
        traces = []
        for repetition in range(3):
            branch = execute_candidate(ctx, snapshot, sequence, trace=True)
            traces.append(branch)
            print(f"    rep {repetition}: digest={trace_digest(branch)[:16]} "
                  f"steps={len(branch['steps'])} clipped={branch['clipped']}", flush=True)
        digests = [trace_digest(t) for t in traces]
        identical = len(set(digests)) == 1
        divergence = None if identical else _jsonable(first_divergence(traces[0], traces[1])
                                                      or first_divergence(traces[0], traces[2]))
        results["cases"].append({
            "case": entry["case"], "state_id": entry["state_id"],
            "scene_id": entry["scene_id"], "family": entry["family"],
            "snapshot_digest": snapshot.digest,
            "boundary": capture["boundary"],
            "body_clearance_m": capture["body_clearance_m"],
            "slew_clipped": bool(traces[0]["clipped"]),
            "trace_digests": digests, "identical": identical,
            "first_divergence": divergence,
            "wall_time_s": round(time.time() - case_started, 2),
        })

        if entry["case"] == "sign_reversal":
            results["controls"] = _sensitivity_controls(ctx, snapshot, sequence)
        if entry["case"] == "sign_reversal" and not order_state_done:
            results["branch_order"] = _branch_order_test(ctx, snapshot)
            order_state_done = True
        del ctx

    results["wall_time_s"] = round(time.time() - started, 2)
    results["pass"] = (
        all(case.get("identical") for case in results["cases"])
        and len(results["cases"]) == 4
        and results["controls"].get("last_actions_diverges") is True
        and results["controls"].get("last_executed_diverges") is True
        and results["branch_order"].get("identical") is True
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "replay_qualification.json").write_text(
        json.dumps(_jsonable(results), indent=2))
    print(json.dumps({"pass": results["pass"],
                      "cases": [(c["case"], c.get("identical")) for c in results["cases"]],
                      "controls": results["controls"],
                      "branch_order": results["branch_order"].get("identical")}, indent=2))
    return 0 if results["pass"] else 1


def _sensitivity_controls(ctx: BranchContext, snapshot: BranchSnapshot,
                          sequence: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    """Omission-sensitivity: a wrong controller field must change the branch."""

    baseline = execute_candidate(ctx, snapshot, sequence, trace=True)
    base_first = baseline["steps"][0]

    restore_branch_state(ctx, snapshot)
    ctx.policy._last_actions = ctx.policy._last_actions + np.float32(0.10)
    perturbed_actions = _run_one_policy_step(ctx, sequence)
    obs_diff = not np.array_equal(np.asarray(base_first["policy_observation"]),
                                  np.asarray(perturbed_actions["policy_observation"]))
    applied_diff = not np.array_equal(
        np.asarray(base_first["applied_action_under_latency"]),
        np.asarray(perturbed_actions["applied_action_under_latency"]))

    restore_branch_state(ctx, snapshot)
    ctx.runner._last_executed = ctx.runner._last_executed + np.float32(0.20)
    perturbed_exec = _run_one_policy_step(ctx, sequence)
    slew_diff = not np.array_equal(np.asarray(base_first["post_slew_command"]),
                                   np.asarray(perturbed_exec["post_slew_command"]))
    restore_branch_state(ctx, snapshot)
    return {
        "last_actions_diverges": bool(obs_diff or applied_diff),
        "last_actions_observation_differs": bool(obs_diff),
        "last_actions_applied_differs": bool(applied_diff),
        "last_executed_diverges": bool(slew_diff),
        "baseline_post_slew_first_tick": np.asarray(base_first["post_slew_command"]).tolist(),
        "perturbed_post_slew_first_tick":
            np.asarray(perturbed_exec["post_slew_command"]).tolist(),
    }


def _run_one_policy_step(ctx: BranchContext,
                         sequence: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    """Execute the branch's first block and return its first policy-step row."""

    sink: list[dict[str, Any]] = []
    policy = ctx.policy
    policy.act = _instrumented_act(policy, sink)
    try:
        requested = block_for(sequence[1][0])[None, ...]
        executed = np.asarray(ctx.runner._clip_block(
            np.asarray(requested, dtype=np.float32)).executed, dtype=np.float64)
        rows: list[dict[str, Any]] = []

        def after_policy_step(tick_idx: int, step_idx: int) -> None:
            if rows:
                return
            row = dict(sink[-1])
            row["post_slew_command"] = executed[0, tick_idx].copy()
            rows.append(row)

        ctx.runner.execute_requested_block(requested, after_policy_step=after_policy_step)
        return rows[0]
    finally:
        policy.act = type(policy).act.__get__(policy, type(policy))


def _branch_order_test(ctx: BranchContext, snapshot: BranchSnapshot) -> dict[str, Any]:
    forward = {}
    for candidate in CANDIDATE_BANK:
        branch = execute_candidate(ctx, snapshot, candidate, trace=True)
        forward[candidate[0]] = trace_digest(branch)
    reverse = {}
    for candidate in reversed(CANDIDATE_BANK):
        branch = execute_candidate(ctx, snapshot, candidate, trace=True)
        reverse[candidate[0]] = trace_digest(branch)
    mismatches = [name for name in forward if forward[name] != reverse[name]]
    return {"identical": not mismatches, "mismatches": mismatches,
            "forward_digests": forward, "reverse_digests": reverse}


def stage_pilot(args: argparse.Namespace) -> int:
    manifest = json.loads((OUT_DIR / "identity_manifest.json").read_text())
    qual_path = OUT_DIR / "replay_qualification.json"
    if not qual_path.is_file():
        raise SystemExit("replay qualification has not run; the pilot is gated on it")
    qualification = json.loads(qual_path.read_text())
    if not qualification.get("pass"):
        raise SystemExit("replay qualification did not pass; the pilot is gated on it")

    shared = _load_shared(args.backend)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pilot_branches.jsonl"
    states = manifest["pilot_states"][args.state_offset:args.state_offset + args.state_limit]
    mode = "a" if args.state_offset else "w"
    started = time.time()
    written = 0
    with out_path.open(mode) as sink:
        for entry in states:
            state_started = time.time()
            print(f"[pilot] {entry['state_id']} ({entry['scene_id']})", flush=True)
            ctx = build_context(Path(entry["scene_dir"]), seed=entry["drive_seed"],
                                backend=args.backend, shared=shared)
            capture = find_capture(
                ctx, rule="first_eligible",
                identity={"state_id": entry["state_id"], "scene_id": entry["scene_id"],
                          "family": entry["family"], "split": entry["split"]})
            if capture is None:
                for candidate, _ in CANDIDATE_BANK:
                    sink.write(json.dumps(_jsonable({
                        "state_id": entry["state_id"], "family": entry["family"],
                        "split": entry["split"], "scene_id": entry["scene_id"],
                        "candidate": candidate, "valid": False,
                        "invalid_reason": "no_eligible_canonical_boundary",
                        "utility": None,
                        "candidate_bank_digest": bank_digest(),
                        "oracle_contract_digest": oracle_digest(),
                    })) + "\n")
                    written += 1
                del ctx
                continue
            snapshot = capture["snapshot"]
            start_bfs = int(capture["goal"]["bfs_distance_cells"])
            for candidate in CANDIDATE_BANK:
                branch_started = time.time()
                try:
                    branch = execute_candidate(ctx, snapshot, candidate, trace=False)
                    components = oracle_components(branch, start_bfs=start_bfs)
                    invalid_reason = None
                    if components is None:
                        invalid_reason = ("solver_nan" if branch["nan"]
                                          else "final_pose_unlocatable_or_unreachable")
                except Exception as exc:                      # noqa: BLE001
                    branch, components = None, None
                    invalid_reason = f"branch_execution_error: {type(exc).__name__}: {exc}"
                row = {
                    "state_id": entry["state_id"], "scene_id": entry["scene_id"],
                    "family": entry["family"], "split": entry["split"],
                    "episode_id": int(snapshot.identity["episode_id"]),
                    "source_step": int(snapshot.identity["source_step"]),
                    "block_index": int(snapshot.identity["block_index"]),
                    "landmark_id": capture["goal"]["landmark_id"],
                    "landmark_cell": capture["goal"]["landmark_cell"],
                    "candidate": candidate[0], "primitives": list(candidate[1]),
                    "requested": None if branch is None else branch["requested"],
                    "post_slew": None if branch is None else branch["executed"],
                    "clipped": None if branch is None else branch["clipped"],
                    "blocks_completed": None if branch is None else branch["blocks_completed"],
                    "truncated_at_block": None if branch is None else branch["truncated_at_block"],
                    "valid": components is not None,
                    "invalid_reason": invalid_reason,
                    "start_bfs": start_bfs,
                    "final_bfs": None if components is None else components["final_bfs"],
                    "progress": None if components is None else components["progress"],
                    "safety": None if components is None else components["safety"],
                    "completion": None if components is None else components["completion"],
                    "utility": None if components is None else components["utility"],
                    "safety_evidence": None if components is None else components["safety_evidence"],
                    "min_clearance_m": None if components is None else components["min_clearance_m"],
                    "min_body_clearance_m": (None if components is None
                                             else components["min_body_clearance_m"]),
                    "termination_events": (None if branch is None else
                                           [r["termination"] for r in branch["ticks"][1:]
                                            if r["terminated"]]),
                    "snapshot_digest": snapshot.digest,
                    "candidate_bank_digest": bank_digest(),
                    "oracle_contract_digest": oracle_digest(),
                    "identity_manifest_digest": manifest["identity_manifest_digest"],
                    "wall_time_s": round(time.time() - branch_started, 3),
                }
                sink.write(json.dumps(_jsonable(row)) + "\n")
                sink.flush()
                written += 1
            print(f"    {entry['state_id']} done in {time.time() - state_started:.1f}s",
                  flush=True)
            del ctx
    print(f"[pilot] wrote {written} branch records in {time.time() - started:.1f}s")
    return 0


def _pilot_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Why the gate landed where it did.  Reporting only; changes no threshold."""

    from collections import Counter
    invalid_reasons = Counter(r["invalid_reason"] for r in records if not r["valid"])
    valid = [r for r in records if r["valid"]]
    evidence = Counter()
    for row in valid:
        for name, fired in (row.get("safety_evidence") or {}).items():
            if fired:
                evidence[name] += 1
    delta_bfs = Counter(int(r["start_bfs"]) - int(r["final_bfs"]) for r in valid)
    per_state = {}
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_state.setdefault(row["state_id"], []).append(row)
    for state_id, rows in by_state.items():
        good = [r for r in rows if r["valid"]]
        if not good:
            per_state[state_id] = {"valid": 0}
            continue
        utilities = sorted((float(r["utility"]) for r in good), reverse=True)
        distinct: list[float] = []
        for value in utilities:
            if not distinct or abs(value - distinct[-1]) > TIE_TOLERANCE:
                distinct.append(value)
        per_state[state_id] = {
            "valid": len(good), "distinct_levels": len(distinct),
            "spread": round(utilities[0] - utilities[-1], 4),
            "top_margin": round(utilities[0] - utilities[1], 4) if len(utilities) > 1 else None,
            "safety_rate": round(sum(r["safety"] for r in good) / len(good), 3),
        }
    return {
        "invalid_reasons": dict(invalid_reasons),
        "valid_branch_safety_rate": (round(sum(r["safety"] for r in valid) / len(valid), 4)
                                     if valid else None),
        "valid_branch_completion_count": int(sum(r["completion"] for r in valid)),
        "safety_evidence_counts": dict(evidence),
        "delta_bfs_cells_histogram": {str(k): v for k, v in sorted(delta_bfs.items())},
        "distinct_utility_values_observed": sorted(
            {round(float(r["utility"]), 4) for r in valid}),
        "per_state": per_state,
    }


def stage_gate(args: argparse.Namespace) -> int:
    records = [json.loads(line) for line in
               (OUT_DIR / "pilot_branches.jsonl").read_text().splitlines() if line.strip()]
    stats = identifiability(records)
    verdict = gate_verdict(stats)
    total_wall = sum(float(r.get("wall_time_s") or 0.0) for r in records)
    diagnostics = _pilot_diagnostics(records)
    report = {
        "schema": "go2_oracle_branch_pilot_gate_report_v1", "status": STATUS,
        "candidate_bank_digest": bank_digest(),
        "oracle_contract_digest": oracle_digest(),
        "superseded_oracle_digest_pre_run": SUPERSEDED_ORACLE_DIGEST_PRE_RUN,
        "tie_tolerance": TIE_TOLERANCE,
        "genesis_backend": args.backend,
        "statistics": stats, "gate": verdict,
        "diagnostics": diagnostics,
        "branch_wall_time_s": round(total_wall, 1),
    }
    (OUT_DIR / "gate_report.json").write_text(json.dumps(_jsonable(report), indent=2))
    print(json.dumps(_jsonable(report), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["digests", "identities", "qualify", "pilot", "gate"],
                        default="digests")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-limit", type=int, default=20)
    args = parser.parse_args()

    if args.stage == "digests":
        print(json.dumps({"status": STATUS,
                          "candidate_bank_digest": bank_digest(),
                          "oracle_contract_digest": oracle_digest(),
                          "superseded_oracle_digest_pre_run": SUPERSEDED_ORACLE_DIGEST_PRE_RUN,
                          "qualification_digest": qualification_digest(),
                          "selection_digest": selection_digest(),
                          "candidates": [n for n, _ in CANDIDATE_BANK],
                          "boundary": BOUNDARY_DIGEST, "inventory_v2": INVENTORY_V2,
                          "design_v1_1": DESIGN_V1_1}, indent=2))
        return 0
    if args.stage == "identities":
        manifest = build_identity_manifests()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "identity_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"identity_manifest_digest": manifest["identity_manifest_digest"],
                          "pilot_states": len(manifest["pilot_states"]),
                          "replay_states": len(manifest["replay_states"]),
                          "families": sorted({s["family"] for s in manifest["pilot_states"]})},
                         indent=2))
        return 0
    if args.stage == "qualify":
        return stage_qualify(args)
    if args.stage == "pilot":
        return stage_pilot(args)
    return stage_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
