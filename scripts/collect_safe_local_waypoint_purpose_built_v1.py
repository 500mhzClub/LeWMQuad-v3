#!/usr/bin/env python3
"""Purpose-built H1-H3 waypoint label collector (development only).

Uses the qualified Genesis snapshot/restore and candidate bank.  Visual target
encoding is intentionally a separate post-collection gate because the encoder
checkpoint is not required to derive poses, waypoint geometry, or tick safety.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import gc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    sys.path.insert(0, str(extra))
import run_go2_oracle_branch_pilot_v1_2 as V

OUT = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")


def body_relative(start, point):
    # BranchContext.pose() returns ((x,y), yaw, z).
    if len(start) == 3 and isinstance(start[0], (tuple, list)):
        (x0, y0), yaw, _ = start
    else:
        x0, y0, yaw = start[0], start[1], start[2]
    x0, y0, yaw = float(x0), float(y0), float(yaw)
    dx, dy = float(point[0]) - x0, float(point[1]) - y0
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * dx + s * dy, -s * dx + c * dy]


def wrap(a):
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def candidates():
    return list(V.V1.CANDIDATE_BANK)


def scene_candidates():
    excluded = set()
    old = ROOT / ".generated/go2_oracle_branch_pilot_v1_2/state_manifest.json"
    if old.exists():
        excluded = {s["scene_id"] for s in json.loads(old.read_text())["states"]}
    root = ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z"
    result = {}
    for family in FAMILIES:
        dirs = sorted(root.glob(f"*/{family}/*"), key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())
        result[family] = [p for p in dirs if p.is_dir() and p.name not in excluded]
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    resume = os.environ.get("PURPOSE_RESUME_MANIFEST") == "1" and (OUT / "state_manifest.json").exists()
    selected = []
    if resume:
        manifest = json.loads((OUT / "state_manifest.json").read_text())
        selected = list(manifest.get("state_candidates", []))
    else:
        manifest = None
    for family, dirs in scene_candidates().items():
        for scene_dir in dirs[:30]:
            selected.append({"family": family, "scene_dir": str(scene_dir),
                             "scene_id": scene_dir.name,
                             "seed": int(hashlib.sha256(scene_dir.name.encode()).hexdigest()[:8], 16)})
    if manifest is None:
        manifest = {"schema": "safe_local_waypoint_purpose_built_v1_state_manifest",
                    "families": list(FAMILIES), "states_requested": 48,
                    "state_candidates": selected, "candidate_count": 12,
                    "horizons": [1, 2, 3], "visual_target_status": "pending_encoder_gate"}
        (OUT / "state_manifest.json").write_text(json.dumps(manifest, indent=2))
    state_index = os.environ.get("PURPOSE_STATE_INDEX")
    if state_index is not None:
        selected = [selected[int(state_index)]]
    shared = V.V1._load_shared("cpu")
    # Candidate-blind eligibility scan.  No candidate branch is executed here;
    # the first twelve eligible identities per family are frozen before the
    # branch pass begins.
    if resume:
        eligible_states = list(manifest.get("state_candidates", []))
    else:
        eligible_states = []
    eligible_by_family = {f: 0 for f in FAMILIES}
    scan_candidates = selected if not resume else []
    for ordinal, entry in enumerate(scan_candidates):
        if eligible_by_family[entry["family"]] >= 12:
            continue
        print(f"[eligibility {ordinal+1}/{len(selected)}] {entry['scene_id']}", flush=True)
        try:
            ctx = V.V1.build_context(Path(entry["scene_dir"]), seed=entry["seed"], backend="cpu", shared=shared)
            ctx.begin_episode()
            for _ in range(40): ctx.drive_one_block()
            topo = V.link_topology(ctx)
            eligible = V.eligible_here(ctx, topo)
            if isinstance(eligible, str):
                continue
            record, _field = eligible
            path = ctx.scene_graph.shortest_path(int(record["cell_id"]), int(record["goal"]["landmark_cell"]))
            if path is None or len(path) < 3:
                continue
            entry = dict(entry)
            entry.update({"status": "eligible", "state_id": f"purpose-{len(eligible_states)}",
                          "start_pose": list(ctx.pose()), "waypoint_path_cells": list(map(int, path[:3])),
                          "goal": dict(record["goal"])})
            eligible_states.append(entry)
            eligible_by_family[entry["family"]] += 1
            del ctx
        except Exception:
            continue
    if not resume:
        selected = [x for x in eligible_states if sum(y["family"] == x["family"] for y in eligible_states[:eligible_states.index(x)+1]) <= 12]
        selected = [x for x in selected if sum(y["family"] == x["family"] for y in selected) >= 0]
        manifest["state_candidates"] = selected
        manifest["states_frozen_before_branching"] = True
        (OUT / "state_manifest.json").write_text(json.dumps(manifest, indent=2))
    if state_index is not None and resume:
        selected = [manifest["state_candidates"][int(state_index)]]
    rows = []
    started = time.time()
    for ordinal, entry in enumerate(selected):
        print(f"[state {ordinal+1}/{len(selected)}] {entry['scene_id']}", flush=True)
        try:
            entry = dict(entry)
            ctx = V.V1.build_context(Path(entry["scene_dir"]), seed=entry["seed"], backend="cpu", shared=shared)
            ctx.begin_episode()
            for _ in range(40):
                ctx.drive_one_block()
            topo = V.link_topology(ctx)
            eligible = V.eligible_here(ctx, topo)
            if isinstance(eligible, str):
                entry["status"] = "ineligible"; entry["reason"] = eligible; del ctx; continue
            record, field = eligible
            path = ctx.scene_graph.shortest_path(int(record["cell_id"]), int(record["goal"]["landmark_cell"]))
            if path is None or len(path) < 3:
                entry["status"] = "ineligible"; entry["reason"] = "no_two_hop_route"; del ctx; continue
            waypoint_xy = ctx.scene_graph.cell_center(int(path[min(2, len(path)-1)]))
            start_pose = ctx.pose()
            waypoint_body = body_relative(start_pose, waypoint_xy)
            goal = dict(record["goal"])
            frozen_state_id = str(entry.get("state_id", f"purpose-{ordinal}"))
            snapshot = V.V1.capture_branch_state(ctx, goal=goal, identity={"state_id": frozen_state_id, "scene_id": entry["scene_id"], "family": entry["family"]})
            entry.update({"status": "eligible", "state_id": frozen_state_id, "start_pose": list(start_pose),
                          "waypoint_xy": list(map(float, waypoint_xy)), "waypoint_body_xy": waypoint_body,
                          "waypoint_path_cells": list(map(int, path[:3])), "snapshot_digest": snapshot.digest,
                          "goal": goal})
            for ci, candidate in enumerate(candidates()):
                branch = V.execute_branch_v12(ctx, snapshot, candidate, field=field, topology=topo)
                tp = max(1, len(branch["ticks"]) // max(1, len(branch["primitives"])))
                horizons = {}
                for h in (1, 2, 3):
                    idx = min(len(branch["ticks"])-1, h * tp - 1)
                    tick = branch["ticks"][idx]
                    xy = tick["xy"]
                    rel = body_relative(start_pose, xy)
                    dist0 = math.hypot(*waypoint_body)
                    dist = math.hypot(waypoint_body[0]-rel[0], waypoint_body[1]-rel[1])
                    horizons[str(h)] = {"pose": [float(xy[0]), float(xy[1]), float(tick["yaw"])],
                                        "delta_body": [float(rel[0]), float(rel[1])],
                                        "distance_to_waypoint": float(dist),
                                        "progress": float(dist0 - dist),
                                        "completed": bool(dist <= 0.35),
                                        "delta_yaw": wrap(float(tick["yaw"]) - float(start_pose[2])),
                                        "min_clearance": float(min(float(t["clearance_m"]) for t in branch["ticks"][:idx+1])),
                                        "unsafe": bool(any(t["disallowed_contacts"] or t["stuck"] or t["terminated"] for t in branch["ticks"][:idx+1]))}
                rows.append({"state_id": frozen_state_id, "scene_id": entry["scene_id"], "family": entry["family"],
                             "candidate_index": ci, "candidate": candidate[0], "primitives": list(candidate[1]),
                             "requested": branch["requested"], "post_slew": branch["post_slew"], "horizons": horizons,
                             "snapshot_digest": snapshot.digest, "visual_target_status": "missing_encoder_cache"})
            del ctx
        except Exception as exc:
            entry["status"] = "error"; entry["reason"] = f"{type(exc).__name__}: {exc}"
        gc.collect()
    if not resume:
        (OUT / "state_manifest.json").write_text(json.dumps(manifest, indent=2))
    mode = "a" if state_index is not None else "w"
    with (OUT / "branch_labels.jsonl").open(mode) as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    existing_rows = sum(1 for _ in (OUT / "branch_labels.jsonl").open())
    summary = {"schema": "safe_local_waypoint_purpose_built_v1_collection_summary",
               "states_requested": 48, "states_eligible": len({r["state_id"] for r in rows}),
               "branches": existing_rows, "branches_this_invocation": len(rows), "expected_branches": 576,
               "runtime_s": round(time.time()-started, 2),
               "visual_target_status": "missing_frozen_encoder_checkpoint_or_cache",
               "family_counts": {f: sum(1 for r in rows if r["family"] == f) for f in FAMILIES}}
    (OUT / "collection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
