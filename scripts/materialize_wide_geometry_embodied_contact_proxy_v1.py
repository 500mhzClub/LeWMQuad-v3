#!/usr/bin/env python3
"""Materialise frozen depth/LiDAR trajectories for the fresh wide panel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from scripts import collect_wide_geometry_embodied_contact_proxy_v1 as COLLECT
from scripts import materialize_geometry_modality_safety_sufficiency_v1 as BASE

OUT = COLLECT.OUT
GEOMETRY_OUT = OUT / "geometry"
CACHE = COLLECT.CACHE / "geometry"
INDEX = OUT / "fresh_geometry_sensor_index.json"


def state_sources():
    manifest = json.loads(COLLECT.PANEL.read_text()); sensors = json.loads(COLLECT.SENSOR_INDEX.read_text())
    records = {row["state_id"]: row for row in sensors["state_records"]}; output = []
    for state in manifest["states"]:
        output.append({**state, "lineage": "wide_geometry_fresh", "pose_source": records[state["state_id"]]["shard_path"]})
    if len(output) != 48 or len({row["state_id"] for row in output}) != 48: raise RuntimeError("fresh geometry identity mismatch")
    return output


def configure() -> None:
    BASE.OUT = GEOMETRY_OUT; BASE.CACHE = CACHE; BASE.state_sources = state_sources


def collect_state(index: int):
    configure(); return BASE.collect_state(index)


def finalize() -> dict:
    configure(); states = state_sources(); records = []
    for state in states:
        path = GEOMETRY_OUT / "states" / f"{state['state_id']}.json"; row = json.loads(path.read_text())
        if row.get("schema") != "geometry_modality_state_v1" or BASE.sha(Path(row["shard_path"])) != row["shard_sha256"]:
            raise RuntimeError(f"invalid geometry state {state['state_id']}")
        records.append(row)
    payload = {"schema": "wide_geometry_embodied_contact_proxy_v1_geometry_sensor_index", "complete": True, "states": 48, "branches": 576,
        "split_states": {split: sum(row["split"] == split for row in records) for split in ("calibration", "heldout")}, "ticks_per_branch": 15,
        "state_records": records, "simulator_steps": 0, "static_pose_materialized_states": 48, "verification_failures": 0,
        "depth_contract": {"modality": "ideal metric pinhole depth", "resolution_wh": [BASE.DEPTH_WIDTH, BASE.DEPTH_HEIGHT],
            "camera_pose": "frozen RGB-camera mount and per-scene extrinsic jitter", "horizontal_fov_deg": BASE.DEPTH_HORIZONTAL_FOV_DEG,
            "range_m": [BASE.DEPTH_NEAR_M, BASE.DEPTH_FAR_M], "validity_mask": True, "noise": "ideal/noiseless"},
        "lidar_contract": {"modality": "ideal 360-degree range scan", "mount_xyz_body_m": list(BASE.LIDAR_XYZ_BODY_M),
            "azimuth_bins": BASE.LIDAR_AZIMUTH_BINS, "vertical_channels_deg": list(BASE.LIDAR_VERTICAL_DEG),
            "range_m": [BASE.LIDAR_NEAR_M, BASE.LIDAR_FAR_M], "validity_mask": True, "noise": "ideal/noiseless",
            "deployment_status": "CHANGED_DEPLOYMENT_SENSOR_CONTRACT"},
        "storage_bytes": sum(Path(row["shard_path"]).stat().st_size for row in records), "runtime_compute_s": sum(float(row["runtime_s"]) for row in records),
        "bindings": {"fresh_panel_digest": json.loads(COLLECT.PANEL.read_text())["content_digest"],
                     "enhanced_sensor_index_digest": json.loads(COLLECT.SENSOR_INDEX.read_text())["content_digest"]}}
    payload["content_digest"] = COLLECT.canonical_digest(payload); COLLECT.atomic_json(INDEX, payload)
    print(json.dumps({key: payload[key] for key in ("states", "branches", "split_states", "runtime_compute_s", "storage_bytes", "content_digest")}, indent=2)); return payload


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-state", type=int); group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true"); args = parser.parse_args()
    if args.collect_state is not None: collect_state(args.collect_state)
    elif args.collect_all:
        for index in range(48): collect_state(index)
    else: finalize()
    return 0


if __name__ == "__main__": raise SystemExit(main())
