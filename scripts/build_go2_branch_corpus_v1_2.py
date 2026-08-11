#!/usr/bin/env python3
"""Branch corpora for the v1.2 utility scorer and the final evaluation.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No predictor checkpoint is opened here.

Two pools, one generator:

* ``scorer_fit``  120 states (15/family), 6 candidates each  -> 720 branches
* ``final_eval``  200 states (25/family), 12 candidates each -> 2400 branches

The snapshot mechanism, the canonical boundary, the candidate bank and oracle
v1.2 are imported unchanged.  What is new here is only the corpus: state
selection with strata, the frozen candidate rotation, the proprioceptive and
control history, and the textured_v03 renders at the three context slots and the
four horizons.

Stages::

    --stage states    resolve + freeze the identities (no branch, no render)
    --stage branches  execute, render and label every allocated branch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_go2_oracle_branch_pilot_v1 as V1
import run_go2_oracle_branch_pilot_v1_2 as V12
from lewm.oracle.go2_branch_oracle_v1_2 import (
    GeodesicField, PROGRESS_NORMALISER_M, V_MAX_MPS, HORIZON_S,
    progress_digest, safety_digest, oracle_digest as v12_oracle_digest,
)
from lewm.oracle.go2_scorer_contract_v1_2 import contract_digest as scorer_contract_digest
from scripts import dev_action_slew_reconstruction_v1 as SLEW
from scripts import build_dev_v03_proprio_action_manifest_v1 as M

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
CORPUS = V1.CORPUS
SPLITS = ("train", "val", "test_id", "test_hard")

CONTEXT_SLOTS = 3
SAMPLES_PER_SLOT = M.SAMPLES_PER_SLOT           # 5
PROPRIO_HISTORY = CONTEXT_SLOTS * SAMPLES_PER_SLOT   # 15 trailing 10 Hz samples
HORIZONS = 4
STORE_RESOLUTION_WH = (224, 224)

# ---- frozen candidate rotation ------------------------------------------------
# Cyclic 12-subset design: subset_k = {k+o mod 12 : o in ROTATION_OFFSETS}.
# Each candidate appears in exactly 6 of the 12 subsets, hence exactly 60 times
# across 120 states, as the frozen scorer-fit design requires.  Every subset
# contains at least one forward and at least one turning candidate.
#
# ARITHMETIC CONFLICT, resolved before execution and recorded here: the frozen
# design also asks for at least one *reversing* candidate in every subset, but
# ``reverse_then_turn`` is the only reversing member of the bank, so requiring it
# in all 120 subsets would give it 120 appearances and break the exact-60
# balance the same design fixes numerically.  The exact-60 count is the
# checkable constraint and is preserved; reversing appears in the maximum
# compatible 6 of 12 subsets (60 of 120 states).
ROTATION_OFFSETS = (0, 1, 3, 5, 7, 9)
FORWARD_CANDIDATES = frozenset({0, 1, 2})
TURNING_CANDIDATES = frozenset({3, 4, 5, 6, 7, 8, 9})
REVERSING_CANDIDATES = frozenset({10})

# ---- frozen strata (scorer-fit only), snapshot-time geometry only -------------
STRATA = ("general", "safety_enriched", "completion_enriched")
SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M = 0.10
COMPLETION_ENRICHED_MAX_GEODESIC_M = 0.75
COMPLETION_ENRICHED_MAX_BEARING_RAD = math.radians(75.0)

POOLS = {
    "scorer_fit": {
        "states_per_family": 15, "candidates_per_state": 6,
        "strata": {"general": 5, "safety_enriched": 5, "completion_enriched": 5},
        "scene_slice": [0, 200],
        "calibration_per_stratum_per_family": 1,
    },
    "final_eval": {
        "states_per_family": 25, "candidates_per_state": 12,
        "strata": None,
        "scene_slice": [200, 500],
        "calibration_per_stratum_per_family": 0,
    },
}

SELECTION = {
    "name": "go2_branch_corpus_selection_v1_2",
    "scene_order": "every corpus scene, excluding the exclusion set, sorted by "
                   "(family, scene_id); pools take disjoint slices of that order",
    "excluded": [
        "the 80 factorial scenes the 32 predictors were trained on",
        "the 20 v1.1 pilot states' scenes",
        "the 4 v1.1 replay-qualification scenes",
        "the 20 v1.2 pilot states' scenes",
    ],
    "pool_scene_disjointness": "scorer_fit and final_eval take disjoint slices, so "
                               "no scene, episode cluster, state or branch is shared",
    "one_state_per_scene": True,
    "warmup_blocks_min": 40, "warmup_blocks_max": 120,
    "drive_seed_rule": "20260811 ^ crc32(scene_id)",
    "backend": "cpu",
    "eligibility_common": [
        "the canonical branch boundary 1faae05f... holds",
        "a designated landmark binds with finite metric graph reachability",
        "not already terminated or truncated",
        "not already in a disallowed body/environment contact",
        "SceneGraph.locate distance <= 2.0 m",
        "at least 14 prior command ticks of proprioceptive history in the "
        "current episode (the three context slots)",
    ],
    "strata": {
        "general": "graph_edges >= 2 (the v1.2 pilot precondition)",
        "safety_enriched": f"graph_edges >= 2 and snapshot-time body-probe "
                           f"clearance <= {SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M} m "
                           f"(snapshot-time geometry only, never an outcome)",
        "completion_enriched": f"geodesic distance <= "
                               f"{COMPLETION_ENRICHED_MAX_GEODESIC_M} m and "
                               f"|landmark bearing| <= 75 deg, so completion is "
                               f"physically reachable within v_max*T = "
                               f"{PROGRESS_NORMALISER_M} m; snapshot-time "
                               f"geometry and the frozen candidate limits only",
    },
    "landmark_binding": "smallest finite geodesic remaining distance among the "
                        "stratum-eligible landmarks; ties by object id",
    "candidate_allocation": {
        "rule": "cyclic 12-subset rotation indexed by the state's index within "
                "its pool",
        "offsets": list(ROTATION_OFFSETS),
        "appearances_per_candidate_scorer_fit": 60,
        "reversing_conflict": "documented above; exact-60 balance preserved",
    },
    "fit_calibration_split": "BY SCENE: one calibration state per stratum per "
                             "family (24 states), the remaining 96 are fit",
}


def selection_digest() -> str:
    return hashlib.sha256(json.dumps(SELECTION, sort_keys=True).encode()).hexdigest()


def candidate_subset(state_index: int) -> tuple[int, ...]:
    return tuple(sorted((state_index + offset) % len(V1.CANDIDATE_BANK)
                        for offset in ROTATION_OFFSETS))


# ------------------------------------------------------------------ rendering --
def render_frame(ctx: V1.BranchContext) -> np.ndarray:
    """One textured_v03 frame: production camera pose, 640x480, Lanczos to 224."""

    from PIL import Image
    from lewm_genesis.scene_loader import effective_camera_mount_xyz_rpy
    from lewm_genesis.rollout import (camera_safety_config_from_pack,
                                      safe_camera_pose_from_base)
    runner = ctx.runner
    pos = np.asarray(runner._as_np(ctx.build.robot.get_pos()), dtype=np.float32)
    quat_wxyz = np.asarray(runner._as_np(ctx.build.robot.get_quat()), dtype=np.float32)
    if pos.ndim == 1:
        pos = pos[None, :]
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz[None, :]
    quat_xyzw = np.stack([quat_wxyz[..., 1], quat_wxyz[..., 2],
                          quat_wxyz[..., 3], quat_wxyz[..., 0]], axis=-1)
    mount_xyz, mount_rpy = effective_camera_mount_xyz_rpy(ctx.pack)
    pose, _safety = safe_camera_pose_from_base(
        pos[0], quat_xyzw[0], mount_xyz_body=mount_xyz, mount_rpy_body=mount_rpy,
        objects=ctx.pack.static_objects,
        config=camera_safety_config_from_pack(ctx.pack))
    ctx.build.camera.set_pose(pos=pose.position, lookat=pose.lookat, up=pose.up)
    rendered = runner._extract_rgb(ctx.build.camera.render())
    if rendered is None:
        raise RuntimeError("camera render returned no RGB")
    rgb = np.asarray(rendered)
    if rgb.ndim == 4:
        rgb = rgb[0]
    arr = np.asarray(rgb, dtype=np.uint8)
    w, h = STORE_RESOLUTION_WH
    if arr.shape[1] != w or arr.shape[0] != h:
        arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.LANCZOS),
                         dtype=np.uint8)
    return arr


def write_png(array: np.ndarray, path: Path) -> str:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------- proprioception --
def proprio_sample(ctx: V1.BranchContext) -> list[float]:
    """The frozen 30-D sensed-state vector, in the corpus channel order."""

    from lewm_genesis.rollout import _roll_from_quat_wxyz, _pitch_from_quat_wxyz
    runner = ctx.runner
    robot = ctx.build.robot
    quat_wxyz = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64)
    ang_world = np.asarray(runner._as_np(robot.get_ang()), dtype=np.float64)
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz[None, :]
    if ang_world.ndim == 1:
        ang_world = ang_world[None, :]
    qw, qx, qy, qz = (float(v) for v in quat_wxyz[0])
    quat_xyzw = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    gyro = runner._world_to_body(ang_world[0], quat_xyzw)
    gravity = M.projected_gravity(_roll_from_quat_wxyz(qw, qx, qy, qz),
                                  _pitch_from_quat_wxyz(qw, qx, qy, qz))
    feature = [g - o for g, o in zip(gravity, M.GRAVITY_OFFSET)]
    joint_pos = np.asarray(runner._as_np(
        robot.get_dofs_position(runner._leg_dof_idx.tolist())), dtype=np.float64)
    joint_vel = np.asarray(runner._as_np(
        robot.get_dofs_velocity(runner._leg_dof_idx.tolist())), dtype=np.float64)
    if joint_pos.ndim == 2:
        joint_pos, joint_vel = joint_pos[0], joint_vel[0]
    return ([float(v) for v in feature] + [float(v) for v in gyro]
            + [float(v) for v in joint_pos] + [float(v) for v in joint_vel])


def control_sample(previous_applied: Sequence[float]) -> list[float]:
    """Efference copy: the applied command at the tick BEFORE this sample."""

    return [float(previous_applied[c]) for c in SLEW.ACTIVE_CHANNELS]


def action_block_10d(executed_block: np.ndarray) -> list[float]:
    """The frozen 10-D five-tick post-slew action for one block."""

    return SLEW.flatten([[float(v) for v in tick] for tick in executed_block])


# ------------------------------------------------------------------- driving ---
def drive_block_with_probe(ctx: V1.BranchContext,
                           probe: Callable[[int, Sequence[float]], None]) -> Any:
    """V1.drive_one_block, with a per-command-tick probe.

    Replicated rather than patched so the frozen driver stays untouched.
    """

    runner = ctx.runner
    requested, _choices = runner._collect_block()
    executed = np.asarray(
        runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
        dtype=np.float64)
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    previous = np.asarray(runner._last_executed, dtype=np.float64)[0].copy()
    carry = {"prev": previous}

    def after_policy_step(tick_idx: int, step_idx: int) -> None:
        if step_idx != steps_per_tick - 1:
            return
        probe(int(tick_idx), carry["prev"])
        carry["prev"] = executed[0, tick_idx].copy()

    block = runner.execute_requested_block(requested,
                                           after_policy_step=after_policy_step)
    for _ in range(runner._block_size):
        for state in runner.episode_states:
            state.step()
    runner._blocks_in_episode += 1
    ctx.ticks_executed += runner._block_size
    ctx.episode_ticks += runner._block_size
    ctx.policy_steps += runner._block_size * steps_per_tick
    ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()

    before = int(runner.episode_states[0].reset_count)
    from lewm_genesis import ros_msg_adapter as adapter
    runner._check_and_reset_fallen_envs(V1._NullWriter(), adapter)
    runner._check_and_reset_completed_envs(V1._NullWriter(), adapter)
    after = int(runner.episode_states[0].reset_count)
    ctx.reset_in_last_block = after != before
    if ctx.reset_in_last_block:
        ctx.episode_ticks = 0
        ctx.last_block_executed = None
        ctx.episode_start_reset_count = after
    return block


# ------------------------------------------------------------------ eligible --
def landmark_bearing_range(ctx: V1.BranchContext, landmark_xy: Sequence[float]
                           ) -> tuple[float, float]:
    """The planning-time observable goal binding: body bearing and range."""

    from lewm_worlds.scene_graph import wrap_angle_pi
    (x, y), yaw, _z = ctx.pose()
    dx = float(landmark_xy[0]) - x
    dy = float(landmark_xy[1]) - y
    return float(wrap_angle_pi(math.atan2(dy, dx) - yaw)), float(math.hypot(dx, dy))


_FIELD_CACHE: dict[tuple[int, int], GeodesicField] = {}


def geodesic_field(ctx: V1.BranchContext, landmark_cell: int,
                   blocked: frozenset) -> GeodesicField:
    """Dijkstra depends only on the scene and the goal, so cache it per scene."""

    key = (id(ctx.scene_graph), int(landmark_cell))
    field = _FIELD_CACHE.get(key)
    if field is None:
        field = GeodesicField(ctx.scene_graph, int(landmark_cell),
                              transit_blocked=blocked)
        _FIELD_CACHE[key] = field
    return field


def classify_state(ctx: V1.BranchContext, topology: dict[str, Any]
                   ) -> tuple[dict[str, Any], GeodesicField, set[str]] | str:
    """Common eligibility plus the strata this state qualifies for."""

    try:
        boundary = V1.assert_canonical_boundary(ctx)
    except V1.BoundaryRefused as exc:
        return f"boundary_refused: {str(exc)[:50]}"
    if ctx.episode_ticks < PROPRIO_HISTORY - 1:
        return "insufficient_proprioceptive_history"
    (x, y), _yaw, _z = ctx.pose()
    hit = ctx.scene_graph.locate((x, y))
    if float(hit.distance_m) > V1.LOCATE_MAX_DISTANCE_M:
        return "locate_distance_gt_2m"
    if V12._contact_count(ctx, topology) > 0:
        return "already_in_disallowed_contact"

    graph = ctx.scene_graph
    blocked = getattr(graph, "nav_blocked_cells", frozenset())
    cell = int(hit.cell_id)
    best: tuple[float, str, int, int, GeodesicField] | None = None
    for name, landmark_cell in sorted(graph.landmark_cells, key=lambda kv: str(kv[0])):
        hops = graph.bfs_distance(cell, int(landmark_cell), transit_blocked=blocked)
        if hops is None or int(hops) < 1:
            continue
        field = geodesic_field(ctx, int(landmark_cell), blocked)
        distance = field.remaining_distance((x, y), cell)
        if not math.isfinite(distance):
            continue
        key = (float(distance), str(name), int(landmark_cell), int(hops))
        if best is None or key < best[:4]:
            best = (*key, field)
    if best is None:
        return "no_reachable_landmark"
    distance, name, landmark_cell, hops, field = best

    from analyze_go2_closed_loop_quality import _body_probe_configuration_clearance_m
    (_x, _y), yaw, _z2 = ctx.pose()
    body_clearance = float(_body_probe_configuration_clearance_m(
        ctx.grid, [x, y], yaw,
        body_forward_m=V1.CONTACT_BODY_FORWARD_M,
        body_half_width_m=V1.CONTACT_BODY_HALF_WIDTH_M,
        body_probe_margin_m=V1.CONTACT_BODY_PROBE_MARGIN_M))
    centre = graph.cell_center(int(landmark_cell))
    bearing, range_m = landmark_bearing_range(ctx, centre)

    strata: set[str] = set()
    if hops >= 2:
        strata.add("general")
        if body_clearance <= SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M:
            strata.add("safety_enriched")
    if (distance <= COMPLETION_ENRICHED_MAX_GEODESIC_M
            and abs(bearing) <= COMPLETION_ENRICHED_MAX_BEARING_RAD):
        strata.add("completion_enriched")
    if not strata:
        return "no_stratum"

    record = {
        "boundary": boundary, "cell_id": cell,
        "goal": {"landmark_id": name, "landmark_cell": int(landmark_cell),
                 "graph_edges": int(hops), "start_geodesic_m": float(distance),
                 "bearing_body_rad": float(bearing), "range_m": float(range_m),
                 "landmark_xy_m": [float(centre[0]), float(centre[1])]},
        "body_clearance_m": body_clearance,
        "clearance_m": float(graph.clearance_to_walls((x, y))),
    }
    return record, field, strata


# ------------------------------------------------------------------- stage A --
def scene_pool() -> dict[str, list[Path]]:
    excluded = set(V1._excluded_scene_ids())
    v11 = json.loads((V1.OUT_DIR / "identity_manifest.json").read_text())
    excluded |= {r["scene_id"] for r in v11["pilot_states"]}
    excluded |= {r["scene_id"] for r in v11["replay_states"]}
    v12 = json.loads((V12.OUT_DIR / "state_manifest.json").read_text())
    excluded |= {r["scene_id"] for r in v12["states"]}
    families: dict[str, list[Path]] = {}
    for split in SPLITS:
        root = CORPUS / split
        if not root.is_dir():
            continue
        for family_dir in sorted(root.iterdir()):
            if not family_dir.is_dir():
                continue
            for scene_dir in sorted(family_dir.iterdir()):
                if scene_dir.name in excluded:
                    continue
                if not (scene_dir / "manifest.json").is_file():
                    continue
                families.setdefault(family_dir.name, []).append(scene_dir)
    for family in families:
        families[family].sort(key=lambda p: p.name)
    return families


def resolve_states(args: argparse.Namespace) -> dict[str, Any]:
    spec = POOLS[args.pool]
    pool = scene_pool()
    shared = V1._load_shared(args.backend)
    lo, hi = spec["scene_slice"]
    states: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}

    wanted_families = ([args.family] if getattr(args, "family", None)
                       else sorted(pool))
    for family in wanted_families:
        scenes = pool[family][lo:hi]
        if spec["strata"] is None:
            need = {"any": spec["states_per_family"]}
        else:
            need = dict(spec["strata"])
        found = {k: 0 for k in need}
        for scene_dir in scenes:
            if all(found[k] >= need[k] for k in need):
                break
            seed = V1._drive_seed(scene_dir.name)
            ctx = V1.build_context(scene_dir, seed=seed, backend=args.backend,
                                   shared=shared)
            topology = V12.link_topology(ctx)
            ctx.begin_episode()
            reasons: dict[str, int] = {}
            chosen: dict[str, Any] | None = None
            for block_idx in range(SELECTION["warmup_blocks_max"]):
                ctx.drive_one_block()
                if block_idx + 1 < SELECTION["warmup_blocks_min"]:
                    continue
                verdict = classify_state(ctx, topology)
                if isinstance(verdict, str):
                    key = verdict.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
                    continue
                record, _field, strata = verdict
                wanted = [s for s in STRATA
                          if s in strata and found.get(s, 0) < need.get(s, 0)]
                if spec["strata"] is None:
                    wanted = ["any"] if found["any"] < need["any"] else []
                if not wanted:
                    reasons["stratum_already_full"] = reasons.get(
                        "stratum_already_full", 0) + 1
                    continue
                stratum = wanted[0]
                chosen = {
                    "family": family, "scene_id": scene_dir.name,
                    "scene_dir": str(scene_dir),
                    "split": scene_dir.parent.parent.name,
                    "drive_seed": seed, "stratum": stratum,
                    "warmup_blocks": block_idx + 1,
                    "source_step": int(record["boundary"]["source_step"]),
                    "episode_id": int(ctx.runner.episode_states[0].episode_id),
                    "cell_id": int(record["cell_id"]),
                    "goal": record["goal"],
                    "body_clearance_m": record["body_clearance_m"],
                    "clearance_m": record["clearance_m"],
                }
                found[stratum] += 1
                break
            rejections[scene_dir.name] = reasons
            _FIELD_CACHE.clear()
            del ctx
            if chosen is not None:
                states.append(chosen)
                print(f"[states] {args.pool} {family[:22]:22s} {chosen['stratum'][:20]:20s} "
                      f"{scene_dir.name} blocks={chosen['warmup_blocks']} "
                      f"edges={chosen['goal']['graph_edges']} "
                      f"d0={chosen['goal']['start_geodesic_m']:.2f}m", flush=True)
        for key in need:
            if found[key] < need[key]:
                print(f"[states] WARNING {family} {key}: {found[key]}/{need[key]}",
                      flush=True)

    # Frozen ordering, candidate allocation and fit/calibration split.
    states.sort(key=lambda s: (s["family"], STRATA.index(s["stratum"])
                               if s["stratum"] in STRATA else 0, s["scene_id"]))
    per_family_stratum_seen: dict[tuple[str, str], int] = {}
    for index, state in enumerate(states):
        state["state_index"] = index
        state["state_id"] = f"{args.pool}-{index:03d}-{state['family']}"
        state["candidate_indices"] = list(candidate_subset(index)) \
            if spec["candidates_per_state"] < len(V1.CANDIDATE_BANK) \
            else list(range(len(V1.CANDIDATE_BANK)))
        key = (state["family"], state["stratum"])
        seen = per_family_stratum_seen.get(key, 0)
        per_family_stratum_seen[key] = seen + 1
        state["split_role"] = ("calibration"
                               if seen < spec["calibration_per_stratum_per_family"]
                               else "fit")

    counts: dict[str, int] = {}
    for state in states:
        for c in state["candidate_indices"]:
            counts[V1.CANDIDATE_BANK[c][0]] = counts.get(V1.CANDIDATE_BANK[c][0], 0) + 1

    manifest = {
        "schema": "go2_branch_corpus_v1_2_state_manifest", "status": STATUS,
        "pool": args.pool, "spec": spec,
        "selection": SELECTION, "selection_digest": selection_digest(),
        "candidate_bank_digest": V1.bank_digest(),
        "progress_contract_digest": progress_digest(),
        "safety_contract_digest": safety_digest(),
        "oracle_v1_2_digest": v12_oracle_digest(),
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "boundary": V1.BOUNDARY_DIGEST, "genesis_backend": args.backend,
        "states": states,
        "candidate_appearances": counts,
        "scene_rejection_reasons": rejections,
    }
    manifest["state_manifest_digest"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items()
                    if k != "state_manifest_digest"}, sort_keys=True).encode()).hexdigest()
    return manifest


# ------------------------------------------------------------------- stage B --
def stage_branches(args: argparse.Namespace) -> int:
    out = OUT_ROOT / args.pool
    manifest = json.loads((out / "state_manifest.json").read_text())
    shared = V1._load_shared(args.backend)
    frames_dir = out / "frames"
    rows_path = out / "branch_rows.jsonl"
    states = manifest["states"][args.state_offset:args.state_offset + args.state_limit]
    mode = "a" if args.state_offset else "w"
    started = time.time()
    written = 0
    self_check: dict[str, Any] | None = None

    with rows_path.open(mode) as sink:
        for entry in states:
            state_started = time.time()
            print(f"[branches] {entry['state_id']} ({entry['scene_id']})", flush=True)
            ctx = V1.build_context(Path(entry["scene_dir"]), seed=entry["drive_seed"],
                                   backend=args.backend, shared=shared)
            topology = V12.link_topology(ctx)
            ctx.begin_episode()

            proprio_log: list[list[float]] = []
            control_log: list[list[float]] = []
            context_frames: list[np.ndarray] = []
            warmup = int(entry["warmup_blocks"])

            def probe(tick_idx: int, previous_applied: Sequence[float]) -> None:
                proprio_log.append(proprio_sample(ctx))
                control_log.append(control_sample(previous_applied))

            for block_idx in range(warmup):
                drive_block_with_probe(ctx, probe)
                if block_idx >= warmup - CONTEXT_SLOTS:
                    context_frames.append(render_frame(ctx))

            verdict = classify_state(ctx, topology)
            if isinstance(verdict, str) or len(proprio_log) < PROPRIO_HISTORY:
                reason = verdict if isinstance(verdict, str) else "short_proprio_history"
                for c in entry["candidate_indices"]:
                    sink.write(json.dumps(V1._jsonable(_invalid_row(
                        entry, manifest, V1.CANDIDATE_BANK[c][0],
                        f"redrive_failed: {reason}"))) + "\n")
                    written += 1
                del ctx
                continue
            record, field, _strata = verdict
            if int(record["cell_id"]) != int(entry["cell_id"]):
                for c in entry["candidate_indices"]:
                    sink.write(json.dumps(V1._jsonable(_invalid_row(
                        entry, manifest, V1.CANDIDATE_BANK[c][0],
                        "redrive_cell_mismatch"))) + "\n")
                    written += 1
                del ctx
                continue

            stem = f"{entry['state_id']}"
            context_paths = []
            for slot, frame in enumerate(context_frames[-CONTEXT_SLOTS:]):
                path = frames_dir / entry["family"] / f"{stem}_ctx{slot}.png"
                write_png(frame, path)
                context_paths.append(str(path))
            proprio = np.asarray(proprio_log[-PROPRIO_HISTORY:], dtype=np.float32)
            control = np.asarray(control_log[-PROPRIO_HISTORY:], dtype=np.float32)

            snapshot = V1.capture_branch_state(
                ctx, goal=record["goal"],
                identity={"state_id": entry["state_id"], "scene_id": entry["scene_id"],
                          "family": entry["family"], "split": entry["split"],
                          "block_index": warmup,
                          "source_step": record["boundary"]["source_step"],
                          "episode_id": int(ctx.runner.episode_states[0].episode_id)})

            for c in entry["candidate_indices"]:
                candidate = V1.CANDIDATE_BANK[c]
                branch_started = time.time()
                horizon_paths: list[str] = []

                def on_block_end(block_index: int) -> None:
                    frame = render_frame(ctx)
                    path = (frames_dir / entry["family"]
                            / f"{stem}_{candidate[0]}_h{block_index + 1}.png")
                    write_png(frame, path)
                    horizon_paths.append(str(path))

                branch = _execute_and_render(ctx, snapshot, candidate, field=field,
                                             topology=topology,
                                             on_block_end=on_block_end)
                scored = V12.score_branch_v12(branch)
                reason = None
                if scored is None:
                    reason = ("solver_nan" if branch["nan"]
                              else "unlocatable_or_unreachable_geodesic")
                row = {
                    "pool": args.pool, "state_id": entry["state_id"],
                    "state_index": int(entry["state_index"]),
                    "split_role": entry["split_role"], "stratum": entry["stratum"],
                    "scene_id": entry["scene_id"], "family": entry["family"],
                    "split": entry["split"],
                    "episode_id": int(snapshot.identity["episode_id"]),
                    "source_step": int(snapshot.identity["source_step"]),
                    "candidate": candidate[0], "candidate_index": int(c),
                    "primitives": list(candidate[1]),
                    "action_blocks": [action_block_10d(np.asarray(b))
                                      for b in branch["post_slew"]],
                    "requested": branch["requested"], "post_slew": branch["post_slew"],
                    "goal": record["goal"],
                    "context_paths": context_paths,
                    "horizon_paths": horizon_paths,
                    "proprio": proprio.tolist(), "control": control.tolist(),
                    "valid": scored is not None, "invalid_reason": reason,
                    "blocks_completed": branch["blocks_completed"],
                    "truncated_at_block": branch["truncated_at_block"],
                    "snapshot_digest": snapshot.digest,
                    "state_manifest_digest": manifest["state_manifest_digest"],
                    "oracle_v1_2_digest": v12_oracle_digest(),
                    "scorer_contract_v1_2_digest": scorer_contract_digest(),
                    "wall_time_s": round(time.time() - branch_started, 3),
                }
                row.update({key: (None if scored is None else scored[key]) for key in (
                    "start_geodesic_m", "final_geodesic_m", "progress",
                    "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
                    "safety", "completion", "utility", "min_clearance_m",
                    "evaluation_points")})
                sink.write(json.dumps(V1._jsonable(row)) + "\n")
                sink.flush()
                written += 1

                if self_check is None:
                    repeat = _execute_and_render(ctx, snapshot, candidate, field=field,
                                                 topology=topology, on_block_end=None)
                    again = V12.score_branch_v12(repeat)
                    self_check = {
                        "state_id": entry["state_id"], "candidate": candidate[0],
                        "rendering_is_physically_inert": bool(
                            scored is not None and again is not None
                            and abs(scored["final_geodesic_m"]
                                    - again["final_geodesic_m"]) < 1e-12
                            and abs(scored["safety"] - again["safety"]) < 1e-12),
                    }
                    print(f"    self-check {self_check}", flush=True)
            print(f"    done in {time.time() - state_started:.1f}s", flush=True)
            del ctx

    summary = {"pool": args.pool, "rows_written": written,
               "wall_time_s": round(time.time() - started, 1),
               "render_self_check": self_check}
    (out / f"branch_summary_{args.state_offset}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def _execute_and_render(ctx, snapshot, candidate, *, field, topology, on_block_end):
    """V12.execute_branch_v12 with an optional per-block render hook."""

    from lewm_worlds.labels.derived import (DerivedLabelComputer, DerivedLabelConfig,
                                            PoseStep)
    V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    goal_cell = int(snapshot.goal["landmark_cell"])
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    state = {"episode_step": int(runner.episode_states[0].episode_step),
             "stamp_ns": int(runner._sim_time_ns)}

    def sample(executed_cmd):
        (x, y), yaw, z = ctx.pose()
        label = label_computer.step(PoseStep(
            timestamp_ns=int(state["stamp_ns"]), env_idx=0, episode_id=episode_id,
            episode_step=int(state["episode_step"]), position_xy_world=(x, y),
            yaw_world_rad=float(yaw),
            last_command=(float(executed_cmd[0]), float(executed_cmd[1]),
                          float(executed_cmd[2]))))
        flags = V1._termination_flags(ctx)
        hit = ctx.scene_graph.locate((x, y))
        located = bool(float(hit.distance_m) <= V1.LOCATE_MAX_DISTANCE_M)
        cell = int(hit.cell_id)
        return {"xy": [x, y], "yaw": yaw, "z": z, "cell_id": cell, "located": located,
                "geodesic_m": float(field.remaining_distance((x, y), cell)
                                    if located else math.inf),
                "at_goal_cell": bool(cell == goal_cell),
                "clearance_m": float(label.clearance_m),
                "stuck": bool(label.stuck_label),
                "disallowed_contacts": int(V12._contact_count(ctx, topology)),
                "terminated": bool(flags["fall"] or flags["out_of_bounds"]
                                   or flags["tipped"]),
                "nan": bool(flags["nan"])}

    start_row = sample(np.asarray(runner._last_executed, dtype=np.float64)[0])
    tick_rows, requested_all, executed_all = [], [], []
    truncated_at_block, nan_seen = None, False

    for block_idx, primitive in enumerate(candidate[1]):
        requested = V1.block_for(primitive)[None, ...]
        executed_block = np.asarray(
            runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
            dtype=np.float64)

        def after_policy_step(tick_idx, step_idx, _b=executed_block, _i=block_idx):
            if step_idx != steps_per_tick - 1:
                return
            state["episode_step"] += 1
            state["stamp_ns"] += int(runner._command_dt_ns)
            row = sample(_b[0, tick_idx])
            row["block"] = _i
            row["tick"] = int(tick_idx)
            tick_rows.append(row)

        block = runner.execute_requested_block(requested,
                                               after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if on_block_end is not None:
            on_block_end(block_idx)
        if tick_rows and tick_rows[-1]["nan"]:
            nan_seen, truncated_at_block = True, block_idx
            break
        if tick_rows and tick_rows[-1]["terminated"]:
            truncated_at_block = block_idx
            break

    return {"candidate": candidate[0], "primitives": list(candidate[1]),
            "requested": requested_all, "post_slew": executed_all,
            "blocks_completed": len(executed_all),
            "truncated_at_block": truncated_at_block, "nan": nan_seen,
            "start": start_row, "ticks": tick_rows}


def _invalid_row(entry, manifest, candidate, reason):
    return {"state_id": entry["state_id"], "scene_id": entry["scene_id"],
            "family": entry["family"], "split": entry["split"],
            "stratum": entry["stratum"], "split_role": entry["split_role"],
            "candidate": candidate, "valid": False, "invalid_reason": reason,
            "utility": None,
            "state_manifest_digest": manifest["state_manifest_digest"],
            "oracle_v1_2_digest": v12_oracle_digest()}


def merge_states(out: Path) -> int:
    """Merge per-family shards into the one frozen manifest, then index it."""

    shards = sorted(out.glob("state_shard_*.json"))
    if not shards:
        raise SystemExit("no state shards to merge")
    merged = json.loads(shards[0].read_text())
    states: list[dict[str, Any]] = []
    rejections: dict[str, Any] = {}
    for shard in shards:
        payload = json.loads(shard.read_text())
        states.extend(payload["states"])
        rejections.update(payload["scene_rejection_reasons"])
    states.sort(key=lambda s: (s["family"],
                               STRATA.index(s["stratum"]) if s["stratum"] in STRATA else 0,
                               s["scene_id"]))
    spec = merged["spec"]
    per_family_stratum_seen: dict[tuple[str, str], int] = {}
    counts: dict[str, int] = {}
    for index, state in enumerate(states):
        state["state_index"] = index
        state["state_id"] = f"{merged['pool']}-{index:03d}-{state['family']}"
        state["candidate_indices"] = (
            list(candidate_subset(index))
            if spec["candidates_per_state"] < len(V1.CANDIDATE_BANK)
            else list(range(len(V1.CANDIDATE_BANK))))
        key = (state["family"], state["stratum"])
        seen = per_family_stratum_seen.get(key, 0)
        per_family_stratum_seen[key] = seen + 1
        state["split_role"] = ("calibration"
                               if seen < spec["calibration_per_stratum_per_family"]
                               else "fit")
        for c in state["candidate_indices"]:
            name = V1.CANDIDATE_BANK[c][0]
            counts[name] = counts.get(name, 0) + 1
    merged["states"] = states
    merged["candidate_appearances"] = counts
    merged["scene_rejection_reasons"] = rejections
    merged.pop("state_manifest_digest", None)
    merged["state_manifest_digest"] = hashlib.sha256(
        json.dumps(merged, sort_keys=True).encode()).hexdigest()
    (out / "state_manifest.json").write_text(json.dumps(merged, indent=2))
    from collections import Counter
    print(json.dumps({
        "state_manifest_digest": merged["state_manifest_digest"],
        "states": len(states),
        "per_family": dict(Counter(s["family"] for s in states)),
        "per_stratum": dict(Counter(s["stratum"] for s in states)),
        "split_roles": dict(Counter(s["split_role"] for s in states)),
        "candidate_appearances": counts}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=sorted(POOLS), required=True)
    parser.add_argument("--stage",
                        choices=["states", "merge-states", "branches"], required=True)
    parser.add_argument("--family", default=None,
                        help="resolve one family only; shards merge via merge-states")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-limit", type=int, default=10**6)
    args = parser.parse_args()

    out = OUT_ROOT / args.pool
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "merge-states":
        return merge_states(out)
    if args.stage == "states":
        manifest = resolve_states(args)
        name = ("state_manifest.json" if not args.family
                else f"state_shard_{args.family}.json")
        (out / name).write_text(json.dumps(manifest, indent=2))
        from collections import Counter
        print(json.dumps({
            "state_manifest_digest": manifest["state_manifest_digest"],
            "states": len(manifest["states"]),
            "per_family": dict(Counter(s["family"] for s in manifest["states"])),
            "per_stratum": dict(Counter(s["stratum"] for s in manifest["states"])),
            "split_roles": dict(Counter(s["split_role"] for s in manifest["states"])),
            "candidate_appearances": manifest["candidate_appearances"],
        }, indent=2))
        return 0
    return stage_branches(args)


if __name__ == "__main__":
    raise SystemExit(main())
