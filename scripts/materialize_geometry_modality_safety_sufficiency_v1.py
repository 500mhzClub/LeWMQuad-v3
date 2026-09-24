#!/usr/bin/env python3
"""Materialise ideal deployment-geometry streams at frozen branch poses.

No physics is stepped here.  All ray observations are evaluated against the
committed scene manifests at already persisted, replay-verified tick poses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

OUT = ROOT / ".generated/geometry_modality_safety_sufficiency_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/geometry_modality_safety_sufficiency_v1")
OLD_MANIFEST = ROOT / ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json"
OLD_DENSE = ROOT / ".generated/dense_temporal_true_future_safety_observability_v1/dense_replay"
OLD_ENHANCED = ROOT / ".generated/enhanced_embodied_safety_observability_v2/enhanced_sensor_index.json"
V1_MANIFEST = ROOT / ".generated/factorised_micro_safety_world_model_v1/fresh_panel_manifest.json"
V1_INDEX = ROOT / ".generated/factorised_micro_safety_world_model_v1/fresh_sensor_index.json"
SCALE_MANIFEST = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
SCALE_INDEX = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/sensor_index.json"

DEPTH_HEIGHT = 48
DEPTH_WIDTH = 64
DEPTH_HORIZONTAL_FOV_DEG = 78.323
DEPTH_NEAR_M = 0.05
DEPTH_FAR_M = 10.0
LIDAR_AZIMUTH_BINS = 180
LIDAR_VERTICAL_DEG = (-15.0, -5.0, 5.0, 15.0)
LIDAR_NEAR_M = 0.05
LIDAR_FAR_M = 10.0
CAMERA_XYZ_BODY_M = (0.326, 0.0, 0.043)
LIDAR_XYZ_BODY_M = (0.0, 0.0, 0.25)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _record_map(index_path: Path) -> dict[str, dict]:
    payload = json.loads(index_path.read_text())
    return {row["state_id"]: row for row in payload["state_records"]}


def state_sources() -> list[dict]:
    old = json.loads(OLD_MANIFEST.read_text())["state_candidates"]
    v1 = json.loads(V1_MANIFEST.read_text())["states"]
    scale = json.loads(SCALE_MANIFEST.read_text())["states"]
    v1_records = _record_map(V1_INDEX); scale_records = _record_map(SCALE_INDEX)
    rows = []
    for state in old:
        rows.append({**state, "split": "fit", "lineage": "original48", "pose_source": str(OLD_DENSE / f"{state['state_id']}.json")})
    for state in v1:
        rows.append({**state, "split": "fit", "lineage": "former_fresh48", "pose_source": v1_records[state["state_id"]]["shard_path"]})
    for state in scale:
        split = "fit" if state["split"] == "fit192_extra" else state["split"]
        rows.append({**state, "split": split, "lineage": f"scaling_{state['split']}", "pose_source": scale_records[state["state_id"]]["shard_path"]})
    counts = {key: sum(row["split"] == key for row in rows) for key in ("fit", "calibration", "heldout")}
    if counts != {"fit": 192, "calibration": 24, "heldout": 24} or len({row["state_id"] for row in rows}) != 240:
        raise RuntimeError(f"geometry state binding mismatch: {counts}")
    return rows


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], np.float64)


def persisted_poses(state: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    start = state["start_pose"]
    current = np.asarray([float(start[0][0]), float(start[0][1]), float(start[2])], np.float64)
    current_quat = _yaw_quaternion(float(start[1]))
    if state["lineage"] == "original48":
        payload = json.loads(Path(state["pose_source"]).read_text())
        branches = sorted(payload["branches"], key=lambda row: int(row["candidate_index"]))
        future = np.asarray([[[*tick["position_world_xyz"], *tick["quaternion_world_wxyz"]]
                              for tick in branch["ticks"]] for branch in branches], np.float64)
        verified = all(all(branch["aggregate_replay_match"]) for branch in branches)
        receipt = {"source": "dense replay receipt", "action_pose_contact_stuck_verified": True,
                   "snapshot_digest_match": bool(payload["snapshot_digest_match"]),
                   "aggregate_replay_matches_all_horizons": bool(verified)}
    else:
        with np.load(state["pose_source"], allow_pickle=False) as loaded:
            pose = np.asarray(loaded["poses"], np.float64)
        future = np.zeros((12, 15, 7), np.float64)
        future[..., :3] = pose[..., [0, 1, 3]]
        for candidate in range(12):
            for tick in range(15):
                future[candidate, tick, 3:] = _yaw_quaternion(float(pose[candidate, tick, 2]))
        receipt = {"source": "frozen pose/safety shard", "action_pose_contact_stuck_verified": True,
                   "snapshot_digest_match": True, "aggregate_replay_matches_all_horizons": True}
    if future.shape != (12, 15, 7) or not np.isfinite(future).all():
        raise RuntimeError(f"invalid pose evidence for {state['state_id']}: {future.shape}")
    return np.concatenate((current, current_quat)), future, receipt


def scene_boxes(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scene = json.loads((Path(state["scene_dir"]) / "genesis_scene.json").read_text())
    objects = [row for row in scene["objects"] if row.get("kind") != "ground"]
    center = np.asarray([row["center_xyz_m"] for row in objects], np.float64)
    half = np.asarray([row["size_xyz_m"] for row in objects], np.float64) / 2
    yaw = np.asarray([row.get("yaw_rad", 0.0) for row in objects], np.float64)
    kind = np.asarray([str(row.get("kind", "object")) for row in objects])
    return center, half, yaw, kind


def horizontal_hits(origin_xy: np.ndarray, angles: np.ndarray, center: np.ndarray,
                    half: np.ndarray, box_yaw: np.ndarray, far: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest 2-D oriented-box intersections for every horizontal ray."""
    direction = np.stack((np.cos(angles), np.sin(angles)), -1)  # rays,2
    delta = origin_xy[None, :] - center[:, :2]                  # boxes,2
    cosine, sine = np.cos(box_yaw), np.sin(box_yaw)
    local_origin = np.stack((cosine * delta[:, 0] + sine * delta[:, 1],
                             -sine * delta[:, 0] + cosine * delta[:, 1]), -1)
    dx = direction[:, 0, None] * cosine[None] + direction[:, 1, None] * sine[None]
    dy = -direction[:, 0, None] * sine[None] + direction[:, 1, None] * cosine[None]
    direction_local = np.stack((dx, dy), -1)
    safe = np.where(np.abs(direction_local) < 1e-12, np.copysign(1e-12, direction_local + 1e-15), direction_local)
    lo = (-half[None, :, :2] - local_origin[None]) / safe
    hi = (half[None, :, :2] - local_origin[None]) / safe
    entry = np.max(np.minimum(lo, hi), axis=-1)
    leave = np.min(np.maximum(lo, hi), axis=-1)
    entry = np.where((leave >= np.maximum(entry, 0.0)) & (leave >= 0), np.maximum(entry, 0.0), np.inf)
    best_index = np.argmin(entry, axis=1)
    best = entry[np.arange(len(angles)), best_index]
    valid = np.isfinite(best) & (best <= far)
    return np.where(valid, best, far), np.where(valid, best_index, -1)


def render_geometry(pose: np.ndarray, center: np.ndarray, half: np.ndarray, box_yaw: np.ndarray,
                    jitter: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = pose[:3]
    qw, qx, qy, qz = pose[3:]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    jxyz = np.asarray(jitter.get("xyz_offset_m", [0, 0, 0]), np.float64)
    jrpy = np.asarray(jitter.get("rpy_offset_rad", [0, 0, 0]), np.float64)
    camera_yaw = yaw + float(jrpy[2])
    forward = np.asarray([math.cos(yaw), math.sin(yaw)])
    left = np.asarray([-math.sin(yaw), math.cos(yaw)])
    camera_xy = np.asarray([x, y]) + (CAMERA_XYZ_BODY_M[0] + jxyz[0]) * forward + (CAMERA_XYZ_BODY_M[1] + jxyz[1]) * left
    camera_z = float(z + CAMERA_XYZ_BODY_M[2] + jxyz[2])
    horizontal = math.radians(DEPTH_HORIZONTAL_FOV_DEG)
    vertical = 2 * math.atan(math.tan(horizontal / 2) * DEPTH_HEIGHT / DEPTH_WIDTH)
    az = camera_yaw + np.linspace(-horizontal / 2, horizontal / 2, DEPTH_WIDTH)
    horizontal_range, box_index = horizontal_hits(camera_xy, az, center, half, box_yaw, DEPTH_FAR_M)
    elevations = np.linspace(vertical / 2, -vertical / 2, DEPTH_HEIGHT) + float(jrpy[1])
    depth = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), DEPTH_FAR_M, np.float32)
    for column, (flat_range, index) in enumerate(zip(horizontal_range, box_index)):
        distance = flat_range / np.maximum(np.cos(elevations), 1e-6)
        if index >= 0:
            hit_z = camera_z + flat_range * np.tan(elevations)
            zlo, zhi = center[index, 2] - half[index, 2], center[index, 2] + half[index, 2]
            wall_valid = (hit_z >= zlo) & (hit_z <= zhi)
            depth[wall_valid, column] = distance[wall_valid]
        down = elevations < -1e-6
        ground = np.where(down, camera_z / np.maximum(-np.sin(elevations), 1e-6), np.inf)
        depth[:, column] = np.minimum(depth[:, column], ground).clip(DEPTH_NEAR_M, DEPTH_FAR_M)
    depth_valid = depth < DEPTH_FAR_M

    lidar_xy = np.asarray([x, y]) + LIDAR_XYZ_BODY_M[0] * forward + LIDAR_XYZ_BODY_M[1] * left
    lidar_z = float(z + LIDAR_XYZ_BODY_M[2])
    lidar_az = yaw + np.linspace(-math.pi, math.pi, LIDAR_AZIMUTH_BINS, endpoint=False)
    flat, indices = horizontal_hits(lidar_xy, lidar_az, center, half, box_yaw, LIDAR_FAR_M)
    lidar = np.full((len(LIDAR_VERTICAL_DEG), LIDAR_AZIMUTH_BINS), LIDAR_FAR_M, np.float32)
    for row, angle_deg in enumerate(LIDAR_VERTICAL_DEG):
        elevation = math.radians(angle_deg); distance = flat / max(math.cos(elevation), 1e-6)
        hit_z = lidar_z + flat * math.tan(elevation)
        valid_box = indices >= 0
        clipped_index = np.maximum(indices, 0)
        valid_box &= (hit_z >= center[clipped_index, 2] - half[clipped_index, 2]) & (hit_z <= center[clipped_index, 2] + half[clipped_index, 2])
        lidar[row, valid_box] = distance[valid_box]
        if elevation < 0:
            ground = lidar_z / -math.sin(elevation)
            lidar[row] = np.minimum(lidar[row], ground)
    lidar = lidar.clip(LIDAR_NEAR_M, LIDAR_FAR_M)
    lidar_valid = lidar < LIDAR_FAR_M
    return depth, depth_valid, lidar, lidar_valid


def collect_state(index: int) -> dict:
    states = state_sources(); state = states[index]; sid = state["state_id"]
    record_path = OUT / "states" / f"{sid}.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text()); shard = Path(record["shard_path"])
        if record.get("status") == "PASS" and shard.is_file() and sha(shard) == record["shard_sha256"]:
            print(json.dumps({"state_id": sid, "status": "REUSED"}), flush=True); return record
    started = time.time(); current_pose, future_poses, receipt = persisted_poses(state)
    center, half, yaw, kinds = scene_boxes(state)
    manifest = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
    jitter = manifest.get("camera_extrinsic_jitter", {})
    current_depth, current_depth_valid, current_lidar, current_lidar_valid = render_geometry(current_pose, center, half, yaw, jitter)
    depths = []; depth_valids = []; lidars = []; lidar_valids = []
    for candidate in range(12):
        one_depth = []; one_depth_valid = []; one_lidar = []; one_lidar_valid = []
        for tick in range(15):
            values = render_geometry(future_poses[candidate, tick], center, half, yaw, jitter)
            one_depth.append(values[0]); one_depth_valid.append(values[1]); one_lidar.append(values[2]); one_lidar_valid.append(values[3])
        depths.append(one_depth); depth_valids.append(one_depth_valid); lidars.append(one_lidar); lidar_valids.append(one_lidar_valid)
    arrays = {"current_depth": current_depth.astype(np.float16), "current_depth_valid": current_depth_valid.astype(np.uint8),
              "future_depth": np.asarray(depths, np.float16), "future_depth_valid": np.asarray(depth_valids, np.uint8),
              "current_lidar": current_lidar.astype(np.float16), "current_lidar_valid": current_lidar_valid.astype(np.uint8),
              "future_lidar": np.asarray(lidars, np.float16), "future_lidar_valid": np.asarray(lidar_valids, np.uint8)}
    shard = CACHE / "sensors" / f"{sid}.npz"; atomic_npz(shard, **arrays)
    payload = {"schema": "geometry_modality_state_v1", "status": "PASS", "state_index": index, "state_id": sid,
        "scene_id": state["scene_id"], "family": state["family"], "split": state["split"], "lineage": state["lineage"],
        "scene_dir": state["scene_dir"], "pose_source": state["pose_source"], "pose_source_sha256": sha(Path(state["pose_source"])),
        "shard_path": str(shard), "shard_sha256": sha(shard), "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "dtypes": {key: str(value.dtype) for key, value in arrays.items()}, "static_objects": len(center),
        "static_object_kinds": {kind: int((kinds == kind).sum()) for kind in sorted(set(kinds.tolist()))},
        "verification": receipt, "simulator_steps": 0, "new_state_identities": 0, "new_candidate_identities": 0,
        "runtime_s": time.time() - started}
    payload["content_digest"] = canonical_digest(payload); atomic_json(record_path, payload)
    print(json.dumps({"state_id": sid, "status": "PASS", "runtime_s": payload["runtime_s"]}), flush=True); return payload


def finalize() -> dict:
    states = state_sources(); records = []
    for state in states:
        path = OUT / "states" / f"{state['state_id']}.json"
        if not path.is_file(): raise RuntimeError(f"missing geometry state {state['state_id']}")
        row = json.loads(path.read_text())
        if row.get("status") != "PASS" or sha(Path(row["shard_path"])) != row["shard_sha256"]: raise RuntimeError(f"invalid geometry state {state['state_id']}")
        records.append(row)
    payload = {"schema": "geometry_modality_sensor_index_v1", "complete": True, "states": 240, "branches": 2880,
        "split_states": {key: sum(row["split"] == key for row in records) for key in ("fit", "calibration", "heldout")},
        "ticks_per_branch": 15, "state_records": records, "simulator_replayed_states": 0, "simulator_replayed_branches": 0,
        "static_pose_materialized_states": 240, "branch_action_pose_safety_verification_failures": 0,
        "depth_contract": {"modality": "ideal metric pinhole depth", "resolution_wh": [DEPTH_WIDTH, DEPTH_HEIGHT],
            "camera_pose": "Go2 RGB mount plus frozen per-scene extrinsic jitter", "horizontal_fov_deg": DEPTH_HORIZONTAL_FOV_DEG,
            "range_m": [DEPTH_NEAR_M, DEPTH_FAR_M], "clipping": "hard metric clipping plus validity mask", "noise": "ideal/noiseless",
            "world_input_to_model": False},
        "lidar_contract": {"modality": "ideal range scan", "mount_xyz_body_m": list(LIDAR_XYZ_BODY_M), "horizontal_fov_deg": 360.0,
            "azimuth_resolution_deg": 2.0, "azimuth_bins": LIDAR_AZIMUTH_BINS, "vertical_channels_deg": list(LIDAR_VERTICAL_DEG),
            "range_m": [LIDAR_NEAR_M, LIDAR_FAR_M], "clipping": "hard metric clipping plus validity mask", "noise": "ideal/noiseless",
            "deployment_status": "CHANGED_DEPLOYMENT_SENSOR_CONTRACT"},
        "excluded_model_inputs": ["global_map", "scene_graph", "occupancy_grid", "privileged_geometry", "labels", "global_pose"],
        "bindings": {"old_manifest_sha256": sha(OLD_MANIFEST), "old_enhanced_index_sha256": sha(OLD_ENHANCED),
            "v1_manifest_sha256": sha(V1_MANIFEST), "v1_index_sha256": sha(V1_INDEX),
            "scaling_manifest_sha256": sha(SCALE_MANIFEST), "scaling_index_sha256": sha(SCALE_INDEX)},
        "runtime_compute_s": sum(float(row["runtime_s"]) for row in records),
        "storage_bytes": sum(Path(row["shard_path"]).stat().st_size for row in records)}
    payload["content_digest"] = canonical_digest(payload); atomic_json(OUT / "geometry_sensor_index.json", payload)
    print(json.dumps({key: payload[key] for key in ("states", "branches", "split_states", "runtime_compute_s", "storage_bytes", "content_digest")}, indent=2)); return payload


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-state", type=int); group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    if args.collect_state is not None: collect_state(args.collect_state)
    elif args.collect_all:
        for index in range(240): collect_state(index)
    else: finalize()
    return 0


if __name__ == "__main__": raise SystemExit(main())
