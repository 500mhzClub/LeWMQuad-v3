#!/usr/bin/env python3
"""Replay one frozen waypoint state, recover safety components and H1-H3 RGB."""
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    sys.path.insert(0, str(extra))
import run_go2_oracle_branch_pilot_v1_2 as V
from lewm.oracle.go2_branch_oracle_v1_2 import CLEARANCE_SAFE_M
from lewm.oracle.go2_textured_v03_renderer import BasePose, TexturedV03Renderer

V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
OUT = ROOT / ".generated/safe_local_waypoint_route_intent_v2/replay"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/safe_local_waypoint_route_intent_v2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(value)


def execute_capture(ctx, snapshot, candidate, *, field, topology):
    from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep
    V.V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    state = {"episode_step": int(runner.episode_states[0].episode_step), "stamp_ns": int(runner._sim_time_ns)}

    def sample(executed_cmd: Sequence[float]):
        robot = ctx.build.robot
        pos = _to_numpy(robot.get_pos()).astype(np.float64)
        quat = _to_numpy(robot.get_quat()).astype(np.float64)
        if pos.ndim > 1: pos = pos[0]
        if quat.ndim > 1: quat = quat[0]
        (x, y), yaw, z = ctx.pose()
        label = label_computer.step(PoseStep(
            timestamp_ns=int(state["stamp_ns"]), env_idx=0, episode_id=episode_id,
            episode_step=int(state["episode_step"]), position_xy_world=(x, y),
            yaw_world_rad=float(yaw), last_command=tuple(float(v) for v in executed_cmd)))
        flags = V.V1._termination_flags(ctx)
        return {"xy": [x, y], "yaw": yaw, "z": z,
                "position_world_xyz": [float(v) for v in pos],
                "quaternion_world_wxyz": [float(v) for v in quat],
                "clearance_m": float(label.clearance_m), "stuck": bool(label.stuck_label),
                "disallowed_contacts": int(V._contact_count(ctx, topology)),
                "fall": bool(flags["fall"]), "out_of_bounds": bool(flags["out_of_bounds"]),
                "tipped": bool(flags["tipped"]), "nan": bool(flags["nan"]),
                "terminated": bool(flags["fall"] or flags["out_of_bounds"] or flags["tipped"])}

    ticks, requested_all, executed_all = [], [], []
    name, primitives = candidate
    for block_idx, primitive in enumerate(primitives):
        requested = V.V1.block_for(primitive)[None, ...]
        executed_block = np.asarray(runner._clip_block(np.asarray(requested, dtype=np.float32)).executed, dtype=np.float64)
        def after_policy_step(tick_idx, step_idx, _block=executed_block, _b=block_idx):
            if step_idx != steps_per_tick - 1: return
            state["episode_step"] += 1; state["stamp_ns"] += int(runner._command_dt_ns)
            row = sample(_block[0, tick_idx]); row["block"] = _b; row["tick"] = int(tick_idx); ticks.append(row)
        block = runner.execute_requested_block(requested, after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if ticks and (ticks[-1]["nan"] or ticks[-1]["terminated"]): break
    return {"candidate": name, "primitives": list(primitives), "requested": requested_all,
            "post_slew": executed_all, "ticks": ticks}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    entry = manifest["state_candidates"][args.state_index]
    sid = entry["state_id"]
    source_rows = {int(r["candidate_index"]): r for r in
                   (json.loads(x) for x in (V1 / "branch_labels.jsonl").read_text().splitlines())
                   if r["state_id"] == sid}
    if len(source_rows) != 12: raise RuntimeError(f"{sid}: expected 12 source rows")
    started = time.time(); shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(entry["scene_dir"]), seed=entry["seed"], backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40): ctx.drive_one_block()
    topo = V.link_topology(ctx); eligible = V.eligible_here(ctx, topo)
    if isinstance(eligible, str): raise RuntimeError(f"{sid}: replay eligibility changed: {eligible}")
    record, field = eligible; goal = dict(record["goal"])
    snapshot = V.V1.capture_branch_state(ctx, goal=goal, identity={"state_id": sid, "scene_id": entry["scene_id"], "family": entry["family"]})
    expected_snapshot_digest = entry.get("snapshot_digest")
    snapshot_digest_match = expected_snapshot_digest is None or snapshot.digest == expected_snapshot_digest
    captures = []
    for ci, candidate in enumerate(V.V1.CANDIDATE_BANK):
        branch = execute_capture(ctx, snapshot, candidate, field=field, topology=topo)
        source = source_rows[ci]
        if not np.allclose(np.asarray(branch["post_slew"]), np.asarray(source["post_slew"]), atol=1e-7, rtol=0):
            raise RuntimeError(f"{sid}:{ci}: post-slew trace mismatch")
        tp = max(1, len(branch["ticks"]) // max(1, len(branch["primitives"])))
        horizons = {}
        for h in (1, 2, 3):
            tick = branch["ticks"][min(len(branch["ticks"])-1, h*tp-1)]
            old = source["horizons"][str(h)]
            if not np.allclose([*tick["xy"], tick["yaw"]], old["pose"], atol=2e-5, rtol=0):
                raise RuntimeError(f"{sid}:{ci}: H{h} pose mismatch")
            prefix = branch["ticks"][:min(len(branch["ticks"]), h*tp)]
            component = {"collision_or_disallowed_contact": any(t["disallowed_contacts"] > 0 for t in prefix),
                         "clearance_violation": min(t["clearance_m"] for t in prefix) < CLEARANCE_SAFE_M,
                         "stuck": any(t["stuck"] for t in prefix), "fall": any(t["fall"] for t in prefix),
                         "unsafe_termination": any(t["out_of_bounds"] or t["tipped"] for t in prefix)}
            aggregate = component["collision_or_disallowed_contact"] or component["stuck"] or component["fall"] or component["unsafe_termination"]
            horizons[str(h)] = {"pose": tick["position_world_xyz"], "quaternion_wxyz": tick["quaternion_world_wxyz"],
                                "components": component, "replay_path_unsafe": aggregate,
                                "frozen_path_unsafe": bool(old["unsafe"]),
                                "safety_replay_match": aggregate == bool(old["unsafe"])}
        captures.append({"state_id": sid, "candidate_index": ci, "horizons": horizons})

    if not args.no_render:
        import genesis as gs
        raw_manifest = json.loads((Path(entry["scene_dir"]) / "genesis_scene.json").read_text())
        renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
        rgb_dir = CACHE / "rgb" / sid; rgb_dir.mkdir(parents=True, exist_ok=True)
        for row in captures:
            for h in (1, 2, 3):
                item = row["horizons"][str(h)]
                result = renderer.render_pose(BasePose(tuple(item["pose"]), tuple(item["quaternion_wxyz"])))
                path = rgb_dir / f"candidate_{row['candidate_index']:02d}_h{h}.png"
                Image.fromarray(result.image, mode="RGB").save(path)
                item["rgb_path"] = str(path); item["rgb_sha256"] = sha(path)
    output = {"schema": "safe_local_waypoint_route_intent_v2_replay_state", "state_id": sid,
              "scene_id": entry["scene_id"], "family": entry["family"], "snapshot_digest": snapshot.digest,
              "snapshot_digest_was_bound_in_v1": expected_snapshot_digest is not None,
              "snapshot_digest_match": snapshot_digest_match,
              "rows": captures, "runtime_s": time.time()-started,
              "render_status": "DEFERRED" if args.no_render else "COMPLETE", "status": "PASS"}
    out_path = OUT / f"{sid}.json"; out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"state_id": sid, "rows": len(captures), "runtime_s": output["runtime_s"]}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
