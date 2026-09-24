#!/usr/bin/env python3
"""Oracle v1.2 pilot — continuous geodesic progress, graded path safety.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No world-model checkpoint is loaded.

This is an **oracle redesign only**.  The snapshot and restoration
implementation, the canonical boundary, the branch runner, the twelve-candidate
bank (`85471e44…`), the four-block / two-second horizon and the CPU backend are
imported unchanged from ``run_go2_oracle_branch_pilot_v1`` and are not modified
here.  Only the oracle and the state-selection preconditions are new.

Stages::

    --stage digests   print the frozen v1.2 digests, simulate nothing
    --stage states    resolve and freeze the twenty new state identities
    --stage smoke     compact replay / sensitivity / branch-order regression
    --stage pilot     240 branches through the unchanged twelve candidates
    --stage gate      apply the unchanged identifiability gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_go2_oracle_branch_pilot_v1 as V1
from lewm.oracle.go2_branch_oracle_v1_2 import (
    CLEARANCE_SAFE_M,
    COMPLETION_CONTRACT,
    HORIZON_BLOCKS,
    PROGRESS_CONTRACT,
    SAFETY_CONTRACT,
    TICKS_PER_BLOCK,
    GeodesicField,
    TickSafetyEvidence,
    composite_utility,
    disallowed_contact_present,
    graded_safety,
    oracle_digest as v12_oracle_digest,
    progress_digest,
    progress_from_distances,
    safety_digest,
)

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
SUPERSEDED_ORACLE_V1_1 = "03dbe01100870cb4cc082f936bc3d0b62aa1e5d23e8eedb44188e88992acbb53"

OUT_DIR = ROOT / ".generated/go2_oracle_branch_pilot_v1_2"
MIN_GRAPH_EDGES_TO_LANDMARK = 2

# ---- frozen v1.2 state-selection contract ------------------------------------
SELECTION = {
    "name": "go2_branch_state_selection_v1_2",
    "pool_split": "test_id",
    "disjoint_from": [
        "the 80-scene factorial calibration panel",
        "the twenty v1.1 pilot states (val split)",
        "the four v1.1 replay-qualification states (test_hard split)",
        "the future 120-state scorer-fit pool",
        "the future 200-state final-evaluation pool",
    ],
    "states": 20,
    "states_per_family_base": 2,
    "extra_states_to_first_n_families": 4,
    "one_state_per_scene": True,
    "max_scenes_tried_per_family": 10,
    "warmup_blocks_min": 40,
    "warmup_blocks_max": 120,
    "drive": "production RolloutRunner._collect_block collector mix, manifest spawn",
    "drive_seed_rule": "20260811 ^ crc32(scene_id)",
    "backend": "cpu",
    "eligibility": [
        "the canonical branch boundary 1faae05f... holds",
        "a designated landmark binds with finite metric graph reachability",
        "the state is at least two graph edges from that landmark",
        "not already completed, terminated or truncated",
        "not already in a disallowed body/environment contact",
        "every oracle input is available (locate distance <= 2.0 m)",
    ],
    "landmark_binding": "smallest finite geodesic remaining distance among "
                        "landmarks at >= 2 graph edges; ties by object id",
    "capture_rule": "first eligible boundary at or after warmup_blocks_min; if a "
                    "scene yields none, advance to the next scene of that family",
}


def selection_digest() -> str:
    return hashlib.sha256(json.dumps(SELECTION, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------- link topology --
def link_topology(ctx: V1.BranchContext) -> dict[str, Any]:
    """Robot / foot / ground link indices used to classify actual contacts."""

    robot = ctx.build.robot
    foot_names = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
    feet = frozenset(int(link.idx) for link in robot.links
                     if str(link.name) in foot_names)
    ground: set[int] = set()
    for entity in ctx.build.scene.entities:
        if type(entity.morph).__name__ == "Plane":
            ground.update(range(int(entity.link_start), int(entity.link_end)))
    return {"robot_link_range": (int(robot.link_start), int(robot.link_end)),
            "foot_link_indices": feet,
            "ground_link_indices": frozenset(ground)}


def _contact_count(ctx: V1.BranchContext, topology: dict[str, Any]) -> int:
    contacts = ctx.build.robot.get_contacts()
    if not contacts:
        return 0
    link_a = _to_numpy(contacts["link_a"]).ravel()
    if link_a.size == 0:
        return 0
    link_b = _to_numpy(contacts["link_b"]).ravel()
    force = _to_numpy(contacts.get("force_a")) if "force_a" in contacts else None
    magnitudes = (np.linalg.norm(force.reshape(-1, 3), axis=-1).tolist()
                  if force is not None and force.size else None)
    return disallowed_contact_present(
        {"link_a": link_a.tolist(), "link_b": link_b.tolist()},
        robot_link_range=topology["robot_link_range"],
        foot_link_indices=topology["foot_link_indices"],
        ground_link_indices=topology["ground_link_indices"],
        forces=magnitudes)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(value)


# ------------------------------------------------------------------- branch ---
def execute_branch_v12(ctx: V1.BranchContext, snapshot: Any,
                       candidate: tuple[str, tuple[str, ...]], *,
                       field: GeodesicField, topology: dict[str, Any]) -> dict[str, Any]:
    """Restore, run the unchanged four-block candidate, collect v1.2 evidence.

    Evidence is sampled *inside* the per-policy-step callback at the final policy
    step of each command tick — the production 10 Hz emission instant.  (v1.1
    sampled after the block returned, so all five of a block's tick rows read the
    same end-of-block pose; that zero displacement forced the production stuck
    predicate true and inflated its stuck component.)
    """

    from lewm_worlds.labels.derived import (DerivedLabelComputer, DerivedLabelConfig,
                                            PoseStep)
    name, primitives = candidate
    V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    goal_cell = int(snapshot.goal["landmark_cell"])
    steps_per_tick = int(runner._policy_steps_per_command_tick)

    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    state = {"episode_step": int(runner.episode_states[0].episode_step),
             "stamp_ns": int(runner._sim_time_ns)}

    def sample(executed_cmd: Sequence[float]) -> dict[str, Any]:
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
        remaining = field.remaining_distance((x, y), cell) if located else math.inf
        return {
            "xy": [x, y], "yaw": yaw, "z": z, "cell_id": cell, "located": located,
            "geodesic_m": float(remaining),
            "at_goal_cell": bool(cell == goal_cell),
            "clearance_m": float(label.clearance_m),
            "stuck": bool(label.stuck_label),
            "disallowed_contacts": int(_contact_count(ctx, topology)),
            "terminated": bool(flags["fall"] or flags["out_of_bounds"] or flags["tipped"]),
            "nan": bool(flags["nan"]),
        }

    start_row = sample(np.asarray(runner._last_executed, dtype=np.float64)[0])
    tick_rows: list[dict[str, Any]] = []
    requested_all: list[Any] = []
    executed_all: list[Any] = []
    clipped_any = False
    truncated_at_block: int | None = None
    nan_seen = False

    for block_idx, primitive in enumerate(primitives):
        requested = V1.block_for(primitive)[None, ...]
        executed_block = np.asarray(
            runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
            dtype=np.float64)

        def after_policy_step(tick_idx: int, step_idx: int,
                              _block=executed_block, _b=block_idx) -> None:
            if step_idx != steps_per_tick - 1:
                return
            state["episode_step"] += 1
            state["stamp_ns"] += int(runner._command_dt_ns)
            row = sample(_block[0, tick_idx])
            row["block"] = _b
            row["tick"] = int(tick_idx)
            tick_rows.append(row)

        block = runner.execute_requested_block(requested,
                                               after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        clipped_any = clipped_any or bool(np.asarray(block.clipped)[0])
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if tick_rows and tick_rows[-1]["nan"]:
            nan_seen = True
            truncated_at_block = block_idx
            break
        if tick_rows and tick_rows[-1]["terminated"]:
            truncated_at_block = block_idx
            break

    return {"candidate": name, "primitives": list(primitives),
            "requested": requested_all, "post_slew": executed_all,
            "clipped": clipped_any, "blocks_completed": len(executed_all),
            "truncated_at_block": truncated_at_block, "nan": nan_seen,
            "start": start_row, "ticks": tick_rows}


def score_branch_v12(branch: dict[str, Any]) -> dict[str, Any] | None:
    """The frozen v1.2 oracle applied to one executed branch."""

    ticks = branch["ticks"]
    if branch["nan"] or not ticks:
        return None
    start_m = float(branch["start"]["geodesic_m"])
    final_m = float(ticks[-1]["geodesic_m"])
    if not branch["start"]["located"] or not ticks[-1]["located"]:
        return None
    progress = progress_from_distances(start_m, final_m)
    if progress is None:
        return None
    evidence = [TickSafetyEvidence(
        disallowed_contact=int(row["disallowed_contacts"]) > 0,
        clearance_m=float(row["clearance_m"]),
        stuck=bool(row["stuck"]),
        terminated=bool(row["terminated"])) for row in ticks]
    safety = graded_safety(evidence)
    if safety is None:
        return None
    completion = 1.0 if any(row["at_goal_cell"] for row in ticks) else 0.0
    return {
        "start_geodesic_m": start_m, "final_geodesic_m": final_m,
        "progress": float(progress),
        "contact_fraction": safety["contact_fraction"],
        "clearance_cost": safety["clearance_cost"],
        "stuck_fraction": safety["stuck_fraction"],
        "fall": safety["fall"], "safety": safety["safety"],
        "completion": completion,
        "utility": composite_utility(progress, safety["safety"], completion),
        "min_clearance_m": min(float(r["clearance_m"]) for r in ticks),
        "evaluation_points": len(ticks),
    }


# ---------------------------------------------------------------- selection --
def bind_landmark_v12(ctx: V1.BranchContext, cell_id: int
                      ) -> tuple[dict[str, Any], GeodesicField] | None:
    """Snapshot-time binding under the v1.2 metric, with the >= 2 edge floor."""

    graph = ctx.scene_graph
    blocked = getattr(graph, "nav_blocked_cells", frozenset())
    best: tuple[float, str, int, int, GeodesicField] | None = None
    for name, landmark_cell in sorted(graph.landmark_cells, key=lambda kv: str(kv[0])):
        hops = graph.bfs_distance(int(cell_id), int(landmark_cell),
                                  transit_blocked=blocked)
        if hops is None or int(hops) < MIN_GRAPH_EDGES_TO_LANDMARK:
            continue
        field = GeodesicField(graph, int(landmark_cell), transit_blocked=blocked)
        distance = field.cell_distance(int(cell_id))
        if not math.isfinite(distance):
            continue
        key = (float(distance), str(name), int(landmark_cell), int(hops))
        if best is None or key < best[:4]:
            best = (*key, field)
    if best is None:
        return None
    distance, name, landmark_cell, hops, field = best
    centre = ctx.scene_graph.cell_center(int(landmark_cell))
    return ({"landmark_id": name, "landmark_cell": int(landmark_cell),
             "graph_edges": int(hops),
             "geodesic_distance_m": float(distance),
             "landmark_xy_m": [float(centre[0]), float(centre[1])]}, field)


def eligible_here(ctx: V1.BranchContext, topology: dict[str, Any]
                  ) -> tuple[dict[str, Any], GeodesicField] | str:
    """Full v1.2 precondition set, evaluated with no candidate outcome in view."""

    try:
        boundary = V1.assert_canonical_boundary(ctx)
    except V1.BoundaryRefused as exc:
        return f"boundary_refused: {str(exc)[:60]}"
    (x, y), _yaw, _z = ctx.pose()
    hit = ctx.scene_graph.locate((x, y))
    if float(hit.distance_m) > V1.LOCATE_MAX_DISTANCE_M:
        return "locate_distance_gt_2m"
    if _contact_count(ctx, topology) > 0:
        return "already_in_disallowed_contact"
    bound = bind_landmark_v12(ctx, int(hit.cell_id))
    if bound is None:
        return "no_landmark_at_two_or_more_graph_edges"
    goal, field = bound
    start_m = field.remaining_distance((x, y), int(hit.cell_id))
    if not math.isfinite(start_m):
        return "start_geodesic_unreachable"
    clearance = float(ctx.scene_graph.clearance_to_walls((x, y)))
    if not math.isfinite(clearance):
        return "clearance_unavailable"
    goal = dict(goal, start_geodesic_m=float(start_m))
    return ({"boundary": boundary, "cell_id": int(hit.cell_id), "goal": goal,
             "clearance_m": clearance}, field)


def resolve_states(args: argparse.Namespace) -> dict[str, Any]:
    """Phase A: drive the frozen pool and record twenty eligible identities.

    Executes no candidate branch and captures no snapshot, so the manifest is
    frozen strictly before any outcome exists.
    """

    v11 = json.loads((V1.OUT_DIR / "identity_manifest.json").read_text())
    excluded = set(V1._excluded_scene_ids())
    excluded |= {row["scene_id"] for row in v11["pilot_states"]}
    excluded |= {row["scene_id"] for row in v11["replay_states"]}

    families = V1._scene_index(SELECTION["pool_split"])
    family_names = sorted(families)
    if len(family_names) != 8:
        raise RuntimeError(f"expected 8 families, found {family_names}")
    shared = V1._load_shared(args.backend)

    states: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}
    for family_index, family in enumerate(family_names):
        wanted = SELECTION["states_per_family_base"]
        if family_index < SELECTION["extra_states_to_first_n_families"]:
            wanted += 1
        pool = [d for d in families[family] if d.name not in excluded]
        found = 0
        for scene_dir in pool[:SELECTION["max_scenes_tried_per_family"]]:
            if found >= wanted:
                break
            seed = V1._drive_seed(scene_dir.name)
            ctx = V1.build_context(scene_dir, seed=seed, backend=args.backend,
                                   shared=shared)
            topology = link_topology(ctx)
            ctx.begin_episode()
            outcome: dict[str, Any] | None = None
            reasons: dict[str, int] = {}
            for block_idx in range(SELECTION["warmup_blocks_max"]):
                ctx.drive_one_block()
                if block_idx + 1 < SELECTION["warmup_blocks_min"]:
                    continue
                verdict = eligible_here(ctx, topology)
                if isinstance(verdict, str):
                    reasons[verdict.split(":")[0]] = reasons.get(
                        verdict.split(":")[0], 0) + 1
                    continue
                record, _field = verdict
                outcome = {
                    "state_id": f"v12-{family}-{found}",
                    "family": family, "split": SELECTION["pool_split"],
                    "scene_id": scene_dir.name, "scene_dir": str(scene_dir),
                    "drive_seed": seed,
                    "warmup_blocks": block_idx + 1,
                    "source_step": int(record["boundary"]["source_step"]),
                    "cell_id": int(record["cell_id"]),
                    "landmark_id": record["goal"]["landmark_id"],
                    "landmark_cell": int(record["goal"]["landmark_cell"]),
                    "graph_edges_to_landmark": int(record["goal"]["graph_edges"]),
                    "start_geodesic_m": float(record["goal"]["start_geodesic_m"]),
                    "clearance_m": float(record["clearance_m"]),
                }
                break
            rejections[scene_dir.name] = reasons
            del ctx
            if outcome is not None:
                states.append(outcome)
                found += 1
                print(f"[states] {outcome['state_id']:34s} {scene_dir.name} "
                      f"blocks={outcome['warmup_blocks']} "
                      f"edges={outcome['graph_edges_to_landmark']} "
                      f"d0={outcome['start_geodesic_m']:.2f}m", flush=True)
        if found < wanted:
            print(f"[states] WARNING family {family}: {found}/{wanted} states",
                  flush=True)

    manifest = {
        "schema": "go2_oracle_branch_pilot_v1_2_state_manifest",
        "status": STATUS,
        "candidate_bank_digest": V1.bank_digest(),
        "progress_contract_digest": progress_digest(),
        "safety_contract_digest": safety_digest(),
        "oracle_v1_2_digest": v12_oracle_digest(),
        "superseded_oracle_v1_1": SUPERSEDED_ORACLE_V1_1,
        "selection_digest": selection_digest(),
        "boundary": V1.BOUNDARY_DIGEST, "inventory_v2": V1.INVENTORY_V2,
        "genesis_backend": args.backend,
        "selection": SELECTION,
        "progress_contract": PROGRESS_CONTRACT,
        "safety_contract": SAFETY_CONTRACT,
        "completion_contract": COMPLETION_CONTRACT,
        "states": states,
        "scene_rejection_reasons": rejections,
    }
    manifest["state_manifest_digest"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items()
                    if k != "state_manifest_digest"}, sort_keys=True).encode()).hexdigest()
    return manifest


# -------------------------------------------------------------------- smoke --
def stage_smoke(args: argparse.Namespace) -> int:
    """Compact regression only — the full replay programme already qualified."""

    v11 = json.loads((V1.OUT_DIR / "identity_manifest.json").read_text())
    entry = v11["replay_states"][0]
    shared = V1._load_shared(args.backend)
    started = time.time()
    ctx = V1.build_context(Path(entry["scene_dir"]), seed=entry["drive_seed"],
                           backend=args.backend, shared=shared)
    capture = V1.find_capture(ctx, rule="first_eligible",
                              identity={"state_id": "v12-smoke", "case": entry["case"],
                                        "scene_id": entry["scene_id"],
                                        "family": entry["family"],
                                        "split": entry["split"]})
    if capture is None:
        raise SystemExit("smoke: no eligible boundary on the qualified-style state")
    snapshot = capture["snapshot"]
    sequence = (entry["case"], tuple(entry["sequence"]))

    digests = [V1.trace_digest(V1.execute_candidate(ctx, snapshot, sequence, trace=True))
               for _ in range(3)]
    replay_identical = len(set(digests)) == 1
    controls = V1._sensitivity_controls(ctx, snapshot, sequence)
    order = V1._branch_order_test(ctx, snapshot)

    result = {
        "schema": "go2_oracle_branch_pilot_v1_2_smoke", "status": STATUS,
        "genesis_backend": args.backend,
        "state": {"scene_id": entry["scene_id"], "family": entry["family"],
                  "source_step": capture["boundary"]["source_step"]},
        "snapshot_digest": snapshot.digest,
        "exact_replay": {"trace_digests": digests, "identical": replay_identical},
        "last_actions_sensitivity": {
            "diverges": controls["last_actions_diverges"],
            "observation_differs": controls["last_actions_observation_differs"],
            "applied_action_differs": controls["last_actions_applied_differs"]},
        "last_executed_sensitivity": {
            "diverges": controls["last_executed_diverges"],
            "baseline_first_post_slew": controls["baseline_post_slew_first_tick"],
            "perturbed_first_post_slew": controls["perturbed_post_slew_first_tick"]},
        "branch_order": {"identical": order["identical"],
                         "mismatches": order["mismatches"],
                         "candidates": len(order["forward_digests"])},
        "wall_time_s": round(time.time() - started, 2),
    }
    result["pass"] = bool(replay_identical and controls["last_actions_diverges"]
                          and controls["last_executed_diverges"]
                          and order["identical"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "smoke.json").write_text(json.dumps(V1._jsonable(result), indent=2))
    print(json.dumps({k: v for k, v in V1._jsonable(result).items()
                      if k not in {"state", "snapshot_digest"}}, indent=2))
    return 0 if result["pass"] else 1


# -------------------------------------------------------------------- pilot --
def stage_pilot(args: argparse.Namespace) -> int:
    manifest_path = OUT_DIR / "state_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    smoke_path = OUT_DIR / "smoke.json"
    if not smoke_path.is_file() or not json.loads(smoke_path.read_text()).get("pass"):
        raise SystemExit("compact regression smoke has not passed; the pilot is gated on it")

    shared = V1._load_shared(args.backend)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pilot_branches.jsonl"
    started = time.time()
    written = 0
    with out_path.open("w") as sink:
        for entry in manifest["states"]:
            state_started = time.time()
            print(f"[pilot] {entry['state_id']} ({entry['scene_id']})", flush=True)
            ctx = V1.build_context(Path(entry["scene_dir"]), seed=entry["drive_seed"],
                                   backend=args.backend, shared=shared)
            topology = link_topology(ctx)
            ctx.begin_episode()
            for _ in range(int(entry["warmup_blocks"])):
                ctx.drive_one_block()
            verdict = eligible_here(ctx, topology)
            mismatch = None
            if isinstance(verdict, str):
                mismatch = f"redrive_ineligible: {verdict}"
            else:
                record, field = verdict
                if int(record["cell_id"]) != int(entry["cell_id"]):
                    mismatch = (f"redrive_cell_mismatch "
                                f"{record['cell_id']} != {entry['cell_id']}")
                elif int(record["goal"]["landmark_cell"]) != int(entry["landmark_cell"]):
                    mismatch = "redrive_landmark_mismatch"
            if mismatch is not None:
                for candidate, _ in V1.CANDIDATE_BANK:
                    sink.write(json.dumps(V1._jsonable(
                        _invalid_row(entry, manifest, candidate, mismatch))) + "\n")
                    written += 1
                del ctx
                continue

            snapshot = V1.capture_branch_state(
                ctx, goal=record["goal"],
                identity={"state_id": entry["state_id"], "scene_id": entry["scene_id"],
                          "family": entry["family"], "split": entry["split"],
                          "block_index": int(entry["warmup_blocks"]),
                          "source_step": record["boundary"]["source_step"],
                          "episode_id": int(ctx.runner.episode_states[0].episode_id)})
            for candidate in V1.CANDIDATE_BANK:
                branch_started = time.time()
                try:
                    branch = execute_branch_v12(ctx, snapshot, candidate,
                                                field=field, topology=topology)
                    scored = score_branch_v12(branch)
                    reason = None
                    if scored is None:
                        reason = ("solver_nan" if branch["nan"]
                                  else "unlocatable_or_unreachable_geodesic")
                except Exception as exc:                       # noqa: BLE001
                    branch, scored = None, None
                    reason = f"branch_execution_error: {type(exc).__name__}: {exc}"
                row = {
                    "state_id": entry["state_id"], "scene_id": entry["scene_id"],
                    "family": entry["family"], "split": entry["split"],
                    "episode_id": int(snapshot.identity["episode_id"]),
                    "source_step": int(snapshot.identity["source_step"]),
                    "warmup_blocks": int(entry["warmup_blocks"]),
                    "landmark_id": entry["landmark_id"],
                    "landmark_cell": int(entry["landmark_cell"]),
                    "graph_edges_to_landmark": int(entry["graph_edges_to_landmark"]),
                    "candidate": candidate[0], "primitives": list(candidate[1]),
                    "requested": None if branch is None else branch["requested"],
                    "post_slew": None if branch is None else branch["post_slew"],
                    "clipped": None if branch is None else branch["clipped"],
                    "blocks_completed": None if branch is None else branch["blocks_completed"],
                    "truncated_at_block": None if branch is None else branch["truncated_at_block"],
                    "valid": scored is not None,
                    "invalid_reason": reason,
                    "snapshot_digest": snapshot.digest,
                    "candidate_bank_digest": V1.bank_digest(),
                    "progress_contract_digest": progress_digest(),
                    "safety_contract_digest": safety_digest(),
                    "oracle_v1_2_digest": v12_oracle_digest(),
                    "state_manifest_digest": manifest["state_manifest_digest"],
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
            print(f"    done in {time.time() - state_started:.1f}s", flush=True)
            del ctx
    print(f"[pilot] wrote {written} branch records in {time.time() - started:.1f}s")
    return 0


def _invalid_row(entry: dict[str, Any], manifest: dict[str, Any], candidate: str,
                 reason: str) -> dict[str, Any]:
    return {"state_id": entry["state_id"], "scene_id": entry["scene_id"],
            "family": entry["family"], "split": entry["split"],
            "candidate": candidate, "valid": False, "invalid_reason": reason,
            "utility": None,
            "candidate_bank_digest": V1.bank_digest(),
            "oracle_v1_2_digest": v12_oracle_digest(),
            "state_manifest_digest": manifest["state_manifest_digest"]}


# --------------------------------------------------------------------- gate --
def stage_gate(_args: argparse.Namespace) -> int:
    records = [json.loads(line) for line in
               (OUT_DIR / "pilot_branches.jsonl").read_text().splitlines() if line.strip()]
    stats = V1.identifiability(records)
    verdict = V1.gate_verdict(stats)
    valid = [r for r in records if r["valid"]]

    def summary(key: str) -> dict[str, float] | None:
        values = [float(r[key]) for r in valid if r.get(key) is not None]
        if not values:
            return None
        array = np.asarray(values)
        return {"min": float(array.min()), "p25": float(np.percentile(array, 25)),
                "median": float(np.median(array)), "p75": float(np.percentile(array, 75)),
                "max": float(array.max()), "mean": float(array.mean()),
                "distinct": int(len(set(np.round(array, 6).tolist())))}

    from collections import Counter
    report = {
        "schema": "go2_oracle_branch_pilot_v1_2_gate_report", "status": STATUS,
        "candidate_bank_digest": V1.bank_digest(),
        "progress_contract_digest": progress_digest(),
        "safety_contract_digest": safety_digest(),
        "oracle_v1_2_digest": v12_oracle_digest(),
        "superseded_oracle_v1_1": SUPERSEDED_ORACLE_V1_1,
        "state_manifest_digest": (records[0].get("state_manifest_digest")
                                  if records else None),
        "genesis_backend": "cpu", "tie_tolerance": V1.TIE_TOLERANCE,
        "statistics": stats, "gate": verdict,
        "distributions": {name: summary(name) for name in (
            "progress", "contact_fraction", "clearance_cost", "stuck_fraction",
            "safety", "utility", "start_geodesic_m", "final_geodesic_m")},
        "invalid_reasons": dict(Counter(r["invalid_reason"] for r in records
                                        if not r["valid"])),
        "completion_count": int(sum(r["completion"] for r in valid)),
        "fall_count": int(sum(r["fall"] for r in valid)),
        "branch_wall_time_s": round(sum(float(r.get("wall_time_s") or 0.0)
                                        for r in records), 1),
    }
    (OUT_DIR / "gate_report.json").write_text(json.dumps(V1._jsonable(report), indent=2))
    print(json.dumps(V1._jsonable(report), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["digests", "states", "smoke", "pilot", "gate"],
                        default="digests")
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()

    if args.stage == "digests":
        print(json.dumps({
            "status": STATUS,
            "candidate_bank_digest": V1.bank_digest(),
            "progress_contract_digest": progress_digest(),
            "safety_contract_digest": safety_digest(),
            "oracle_v1_2_digest": v12_oracle_digest(),
            "superseded_oracle_v1_1": SUPERSEDED_ORACLE_V1_1,
            "selection_digest": selection_digest(),
            "boundary": V1.BOUNDARY_DIGEST, "inventory_v2": V1.INVENTORY_V2,
            "horizon_blocks": HORIZON_BLOCKS, "ticks_per_block": TICKS_PER_BLOCK,
        }, indent=2))
        return 0
    if args.stage == "states":
        manifest = resolve_states(args)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "state_manifest.json").write_text(json.dumps(manifest, indent=2))
        from collections import Counter
        print(json.dumps({
            "state_manifest_digest": manifest["state_manifest_digest"],
            "states": len(manifest["states"]),
            "per_family": dict(Counter(s["family"] for s in manifest["states"])),
        }, indent=2))
        return 0
    if args.stage == "smoke":
        return stage_smoke(args)
    if args.stage == "pilot":
        return stage_pilot(args)
    return stage_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
