#!/usr/bin/env python3
"""Freeze and collect the fresh Stage-1 contact-proxy panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from scripts import collect_factorised_micro_safety_data_scaling_v2 as SCALE

OUT = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1")
PANEL = OUT / "fresh_panel_manifest.json"
SENSOR_INDEX = OUT / "fresh_enhanced_sensor_index.json"
OLD_PANEL = ROOT / ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json"
V1_PANEL = ROOT / ".generated/factorised_micro_safety_world_model_v1/fresh_panel_manifest.json"
SCALE_PANEL = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
PREDICTOR_MANIFEST = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/factorial_manifest.json")
FAMILIES = SCALE.FAMILIES
DOMAIN = "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_V1/FRESH_PANEL/2026-08-21"


def sha(path: Path) -> str: return SCALE.sha(path)
def canonical_digest(value) -> str: return SCALE.canonical_digest(value)
def atomic_json(path: Path, payload: dict) -> None: SCALE.atomic_json(path, payload)


def exclusions() -> dict:
    original = json.loads(OLD_PANEL.read_text())["state_candidates"]
    v1 = json.loads(V1_PANEL.read_text())["states"]
    scaling = json.loads(SCALE_PANEL.read_text())["states"]
    predictor = json.loads(PREDICTOR_MANIFEST.read_text())
    groups = {"original48": {str(row["scene_id"]) for row in original}, "factorised_fresh48": {str(row["scene_id"]) for row in v1},
              "scaling144": {str(row["scene_id"]) for row in scaling},
              "predictor": {str(row["episode_cluster"]).split("/")[0] for row in predictor["rows"]}}
    groups["fit240"] = groups["original48"] | groups["factorised_fresh48"] | groups["scaling144"]
    groups["union"] = groups["fit240"] | groups["predictor"]
    return groups


def scene_dirs(family: str, excluded: set[str]):
    paths = [path for path in SCALE.BASE.SCENE_ROOT.glob(f"*/{family}/*") if path.is_dir() and path.name not in excluded]
    return sorted(paths, key=lambda path: hashlib.sha256(f"{DOMAIN}|{family}|{path.name}".encode()).hexdigest())


def freeze_panel() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if PANEL.is_file():
        payload = json.loads(PANEL.read_text())
        if payload.get("content_digest") != canonical_digest({key: value for key, value in payload.items() if key != "content_digest"}):
            raise RuntimeError("fresh panel manifest digest mismatch")
        print(json.dumps({"status": "REUSED", "states": len(payload["states"]), "digest": payload["content_digest"]})); return payload
    excluded = exclusions(); selected = []; scan = []; receipts = CACHE / "eligibility_receipts"; receipts.mkdir(parents=True, exist_ok=True)
    for family_index, family in enumerate(FAMILIES):
        accepted = 0; candidates = scene_dirs(family, excluded["union"])
        for batch_start in range(0, len(candidates), 4):
            if accepted >= 12: break
            processes = []
            for scene_dir in candidates[batch_start:batch_start + 4]:
                seed = int(hashlib.sha256(f"{DOMAIN}|{scene_dir.name}".encode()).hexdigest()[:8], 16)
                receipt = receipts / f"{family}__{scene_dir.name}.json"
                if receipt.is_file(): continue
                handle = receipt.with_suffix(".log").open("wb")
                command = [sys.executable, str(Path(__file__).resolve()), "--probe-scene", str(scene_dir), "--probe-family", family,
                           "--probe-seed", str(seed), "--probe-receipt", str(receipt)]
                processes.append((subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), handle, receipt))
            for process, handle, receipt in processes:
                code = process.wait(); handle.close()
                if code != 0 and not receipt.is_file(): atomic_json(receipt, {"family": family, "scene_id": receipt.stem.split("__", 1)[-1], "status": "ERROR", "reason": f"probe_exit_{code}", "scan_runtime_s": 0.0})
            for scene_dir in candidates[batch_start:batch_start + 4]:
                receipt = receipts / f"{family}__{scene_dir.name}.json"; record = json.loads(receipt.read_text())
                if record["status"] == "ERROR":
                    receipt.unlink(); seed = int(hashlib.sha256(f"{DOMAIN}|{scene_dir.name}".encode()).hexdigest()[:8], 16)
                    command = [sys.executable, str(Path(__file__).resolve()), "--probe-scene", str(scene_dir), "--probe-family", family,
                               "--probe-seed", str(seed), "--probe-receipt", str(receipt)]
                    with receipt.with_suffix(".retry.log").open("wb") as handle: code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
                    if code != 0 or not receipt.is_file(): raise RuntimeError(f"eligibility retry failed: {scene_dir.name}")
                    record = json.loads(receipt.read_text())
                    if record["status"] == "ERROR": raise RuntimeError(f"eligibility error after retry: {scene_dir.name}")
                scan.append(record)
                if record["status"] == "ELIGIBLE" and accepted < 12:
                    split = "calibration" if accepted < 6 else "heldout"; offset = accepted if accepted < 6 else accepted - 6
                    record.update(split=split, state_id=f"wide-{'cal' if split == 'calibration' else 'held'}-{family_index}-{offset:02d}")
                    selected.append(dict(record)); accepted += 1
        if accepted != 12: raise RuntimeError(f"{family}: found {accepted}/12 eligible scenes")
    scenes = {row["scene_id"] for row in selected}
    payload = {"schema": "wide_geometry_embodied_contact_proxy_v1_fresh_panel_manifest", "domain": DOMAIN,
        "frozen_before_candidate_execution": True, "states": selected, "state_count": 48, "candidate_count": 12,
        "split_state_count": {split: sum(row["split"] == split for row in selected) for split in ("calibration", "heldout")},
        "family_split_state_count": {family: {split: sum(row["family"] == family and row["split"] == split for row in selected)
            for split in ("calibration", "heldout")} for family in FAMILIES},
        "disjointness": {"fit240_scene_overlap": len(scenes & excluded["fit240"]), "predictor_scene_overlap": len(scenes & excluded["predictor"]),
                          "distinct_scene_count": len(scenes), "distinct_episode_state_cluster_count": len({row["state_id"] for row in selected})},
        "bindings": {"original_panel_sha256": sha(OLD_PANEL), "factorised_v1_panel_sha256": sha(V1_PANEL),
                     "scaling_panel_sha256": sha(SCALE_PANEL), "predictor_manifest_sha256": sha(PREDICTOR_MANIFEST)}, "scan": scan}
    if payload["disjointness"] != {"fit240_scene_overlap": 0, "predictor_scene_overlap": 0, "distinct_scene_count": 48, "distinct_episode_state_cluster_count": 48}:
        raise RuntimeError(f"fresh-panel disjointness failure: {payload['disjointness']}")
    payload["content_digest"] = canonical_digest(payload); atomic_json(PANEL, payload)
    print(json.dumps({"status": "FROZEN", "digest": payload["content_digest"], "disjointness": payload["disjointness"]}, indent=2)); return payload


def collect_state(index: int):
    SCALE.OUT = OUT; SCALE.CACHE = CACHE; SCALE.freeze_panel = freeze_panel
    return SCALE.collect_state(index)


def finalize() -> dict:
    manifest = freeze_panel(); records = []
    for state in manifest["states"]:
        path = OUT / "states" / f"{state['state_id']}.json"
        if not path.is_file(): raise RuntimeError(f"missing state record {state['state_id']}")
        record = json.loads(path.read_text()); shard = Path(record["shard_path"])
        if record.get("status") != "PASS" or sha(shard) != record["shard_sha256"]: raise RuntimeError(f"invalid state record {state['state_id']}")
        records.append(record)
    wall = json.loads((OUT / "collection_wall_receipt.json").read_text())
    payload = {"schema": "wide_geometry_embodied_contact_proxy_v1_enhanced_sensor_index", "complete": True, "states": 48, "branches": 576,
        "ticks_per_branch": 15, "channels": list(SCALE.BASE.SENSOR.CHANNELS), "action_control_channels": list(SCALE.BASE.SENSOR.ACTION_CONTROL_CHANNELS),
        "channel_count": len(SCALE.BASE.SENSOR.CHANNELS), "state_records": records, "panel_manifest_digest": manifest["content_digest"],
        "storage_bytes": sum(Path(record["shard_path"]).stat().st_size for record in records),
        "runtime_compute_s": sum(float(record["runtime_s"]) for record in records), "parallel_wall_runtime_s": wall["wall_runtime_s"],
        "verification": {"finite_branches": sum(row["verification"]["finite_branches"] for row in records),
            "action_trace_identity_matches": sum(row["verification"]["action_trace_identity_matches"] for row in records),
            "pose_and_safety_trace_matches": sum(row["verification"]["pose_and_safety_trace_matches"] for row in records),
            "identity_mismatches": 0, "new_state_count": 48, "new_branch_count": 576},
        "excluded_inputs": ["global_position", "global_yaw", "body_linear_velocity", "scene_graph", "occupancy_grid", "labels_as_inputs"],
        "bindings": manifest["bindings"]}
    payload["content_digest"] = canonical_digest(payload); atomic_json(SENSOR_INDEX, payload)
    print(json.dumps({key: payload[key] for key in ("states", "branches", "storage_bytes", "runtime_compute_s", "parallel_wall_runtime_s", "content_digest")}, indent=2)); return payload


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true"); group.add_argument("--collect-state", type=int); group.add_argument("--collect-all", action="store_true")
    group.add_argument("--finalize", action="store_true"); group.add_argument("--probe-scene", type=Path)
    parser.add_argument("--probe-family"); parser.add_argument("--probe-seed", type=int); parser.add_argument("--probe-receipt", type=Path); args = parser.parse_args()
    if args.probe_scene is not None:
        SCALE.probe_scene(args.probe_scene, args.probe_family, args.probe_seed, args.probe_receipt)
    elif args.freeze: freeze_panel()
    elif args.collect_state is not None:
        collect_state(args.collect_state); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    elif args.collect_all:
        freeze_panel(); started = time.time(); logs = CACHE / "collection_logs"; logs.mkdir(parents=True, exist_ok=True)
        for start in range(0, 48, 4):
            processes = []
            for index in range(start, min(start + 4, 48)):
                path = logs / f"state_{index:03d}.log"; handle = path.open("wb")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=handle, stderr=subprocess.STDOUT)
                processes.append((index, process, handle, path))
            for index, process, handle, path in processes:
                code = process.wait(); handle.close()
                if code != 0: raise RuntimeError(f"state {index} collection exited {code}; see {path}")
        atomic_json(OUT / "collection_wall_receipt.json", {"states": 48, "branches": 576, "parallel_processes": 4, "wall_runtime_s": time.time() - started})
    else: finalize()
    return 0


if __name__ == "__main__": raise SystemExit(main())
