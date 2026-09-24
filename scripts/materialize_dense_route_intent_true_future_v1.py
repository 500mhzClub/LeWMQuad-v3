#!/usr/bin/env python3
"""Materialise replay-verified, dense true-future RGB/tokens through H3.

Large RGB and token artefacts are written only below the high-capacity temporal
cache.  The workspace receives compact JSON indices.  Existing state and
candidate identities are never changed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import replay_safe_local_waypoint_route_intent_v2 as REPLAY
from scripts import run_go2_oracle_branch_pilot_v1_2 as V

OUT = ROOT / ".generated/dense_temporal_true_future_safety_observability_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/dense_temporal_true_future_safety_observability_v1")
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
V2 = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
EXPECTED_TARGET = "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874"
EXPECTED_ENCODER = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
TOKEN_SHAPE = (768, 1024)
TOKEN_BYTES = int(np.prod(TOKEN_SHAPE) * np.dtype(np.float16).itemsize)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.png")
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(tmp)
    os.replace(tmp, path)
    return sha(path)


def source_rows() -> dict[tuple[str, int], dict]:
    rows = [json.loads(line) for line in (V1 / "branch_labels.jsonl").read_text().splitlines() if line]
    if len(rows) != 576:
        raise RuntimeError("frozen branch ledger is incomplete")
    return {(row["state_id"], int(row["candidate_index"])): row for row in rows}


def frozen_replay_rows() -> dict[tuple[str, int], dict]:
    result = {}
    for path in sorted((V2 / "replay").glob("purpose-*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            result[(row["state_id"], int(row["candidate_index"]))] = row
    if len(result) != 576:
        raise RuntimeError("frozen endpoint replay is incomplete")
    return result


def tick_components(ticks: list[dict], stop: int) -> dict:
    prefix = ticks[:stop]
    return {
        "collision_or_disallowed_contact": any(int(t["disallowed_contacts"]) > 0 for t in prefix),
        "clearance_violation": min(float(t["clearance_m"]) for t in prefix) < float(REPLAY.CLEARANCE_SAFE_M),
        "stuck": any(bool(t["stuck"]) for t in prefix),
        "fall": any(bool(t["fall"]) for t in prefix),
        "unsafe_termination": any(bool(t["out_of_bounds"]) or bool(t["tipped"]) for t in prefix),
    }


def aggregate(component: dict) -> bool:
    return bool(component["collision_or_disallowed_contact"] or component["stuck"]
                or component["fall"] or component["unsafe_termination"])


def validate_existing_state(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "dense_route_intent_true_future_state_v1" or payload.get("status") != "PASS":
            return False
        if len(payload.get("branches", [])) != 12 or int(payload.get("h3_tick_count", -1)) <= 0:
            return False
        frames = [payload["current_frame"]]
        for branch in payload["branches"]:
            if len(branch["ticks"]) != payload["h3_tick_count"]:
                return False
            if "aggregate_replay_match" not in branch or any("training_cumulative_unsafe" not in tick for tick in branch["ticks"]):
                return False
            frames.extend(branch["ticks"])
        for frame in frames:
            p = Path(frame["rgb_path"])
            if not p.is_file() or sha(p) != frame["rgb_sha256"]:
                return False
        return True
    except (KeyError, OSError, ValueError, TypeError):
        return False


def collect_state(state_index: int) -> dict:
    states = json.loads((V1 / "state_manifest.json").read_text())["state_candidates"]
    if state_index < 0 or state_index >= len(states):
        raise RuntimeError("state index is outside the frozen manifest")
    state = states[state_index]
    sid = str(state["state_id"])
    out_path = OUT / "dense_replay" / f"{sid}.json"
    if validate_existing_state(out_path):
        payload = json.loads(out_path.read_text())
        print(json.dumps({"state_id": sid, "status": "REUSED", "frames": 1 + 12 * payload["h3_tick_count"]}), flush=True)
        return payload

    ledger = source_rows()
    endpoint_replay = frozen_replay_rows()
    started = time.time()
    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40):
        ctx.drive_one_block()
    topo = V.link_topology(ctx)
    eligible = V.eligible_here(ctx, topo)
    if isinstance(eligible, str):
        raise RuntimeError(f"{sid}: replay eligibility changed: {eligible}")
    record, field = eligible
    goal = dict(record["goal"])
    snapshot = V.V1.capture_branch_state(
        ctx, goal=goal,
        identity={"state_id": sid, "scene_id": state["scene_id"], "family": state["family"]},
    )
    expected_snapshot_digest = state.get("snapshot_digest")
    snapshot_digest_match = expected_snapshot_digest is None or snapshot.digest == expected_snapshot_digest

    import genesis as gs
    from lewm.oracle.go2_textured_v03_renderer import BasePose, TexturedV03Renderer, capture_base_pose
    raw_manifest = json.loads((Path(state["scene_dir"]) / "genesis_scene.json").read_text())
    renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
    rgb_root = CACHE / "rgb" / sid
    current_pose = capture_base_pose(ctx)
    current_render = renderer.render_pose(current_pose)
    current_path = rgb_root / "current.png"
    current_sha = atomic_png(current_path, current_render.image)
    current_frame = {
        "identity": f"{sid}:current", "rgb_path": str(current_path), "rgb_sha256": current_sha,
        "position_world_xyz": list(current_pose.position_world_xyz),
        "quaternion_world_wxyz": list(current_pose.quaternion_world_wxyz),
        "camera_pose_world": current_render.camera_pose_world,
    }

    branches = []
    tick_counts = set()
    boundary_sets = set()
    for ci, candidate in enumerate(V.V1.CANDIDATE_BANK):
        source = ledger[(sid, ci)]
        frozen = endpoint_replay[(sid, ci)]
        branch = REPLAY.execute_capture(ctx, snapshot, candidate, field=field, topology=topo)
        if not np.allclose(np.asarray(branch["post_slew"]), np.asarray(source["post_slew"]), atol=1e-7, rtol=0):
            raise RuntimeError(f"{sid}:{ci}: post-slew action mismatch")
        boundaries = tuple(sum(len(block) for block in source["post_slew"][:h]) for h in (1, 2, 3))
        if len(branch["ticks"]) < boundaries[-1]:
            raise RuntimeError(f"{sid}:{ci}: branch ended before registered H3 boundary")
        boundary_sets.add(boundaries)
        tick_counts.add(boundaries[-1])
        for h, stop in enumerate(boundaries, 1):
            tick = branch["ticks"][stop - 1]
            old = source["horizons"][str(h)]
            if not np.allclose([*tick["xy"], tick["yaw"]], old["pose"], atol=2e-5, rtol=0):
                raise RuntimeError(f"{sid}:{ci}: H{h} pose mismatch")
            frozen_h = frozen["horizons"][str(h)]
            if not np.allclose(tick["position_world_xyz"], frozen_h["pose"], atol=2e-5, rtol=0):
                raise RuntimeError(f"{sid}:{ci}: H{h} full pose mismatch")
            component = tick_components(branch["ticks"], stop)
            if component != frozen_h["components"]:
                raise RuntimeError(f"{sid}:{ci}: H{h} component label mismatch")
        horizon_unsafe = [bool(source["horizons"][str(h)]["unsafe"]) for h in (1, 2, 3)]
        replay_horizon_unsafe = [aggregate(tick_components(branch["ticks"], stop)) for stop in boundaries]

        ticks = []
        cumulative_contact = cumulative_stuck = cumulative_unsafe = False
        for offset, tick in enumerate(branch["ticks"][:boundaries[-1]], 1):
            active_contact = bool(int(tick["disallowed_contacts"]) > 0)
            active_stuck = bool(tick["stuck"])
            active_unsafe = bool(active_contact or active_stuck or tick["fall"]
                                 or tick["out_of_bounds"] or tick["tipped"])
            cumulative_contact |= active_contact
            cumulative_stuck |= active_stuck
            cumulative_unsafe |= active_unsafe
            training_cumulative_unsafe = cumulative_unsafe
            # The original path-level ledger is authoritative.  When a replay
            # misses a historically registered transient, its first known
            # right-censoring boundary is carried forward without inventing a
            # component identity or an earlier event tick.
            for horizon_index, stop in enumerate(boundaries):
                if offset >= stop and horizon_unsafe[horizon_index]:
                    training_cumulative_unsafe = True
            render = renderer.render_pose(BasePose(
                tuple(tick["position_world_xyz"]), tuple(tick["quaternion_world_wxyz"])
            ))
            path = rgb_root / f"candidate_{ci:02d}_tick_{offset:02d}.png"
            digest = atomic_png(path, render.image)
            block_index = int(tick["block"])
            action = source["post_slew"][block_index][int(tick["tick"])]
            ticks.append({
                "tick": offset,
                "time_s": offset * 0.1,
                "block": block_index,
                "tick_in_block": int(tick["tick"]),
                "applied_action": [float(value) for value in action],
                "position_world_xyz": [float(value) for value in tick["position_world_xyz"]],
                "quaternion_world_wxyz": [float(value) for value in tick["quaternion_world_wxyz"]],
                "rgb_path": str(path), "rgb_sha256": digest,
                "active_contact": active_contact, "active_stuck": active_stuck,
                "active_unsafe": active_unsafe,
                "cumulative_contact": cumulative_contact,
                "cumulative_stuck": cumulative_stuck,
                "cumulative_unsafe": cumulative_unsafe,
                "training_cumulative_unsafe": training_cumulative_unsafe,
                "clearance_m": float(tick["clearance_m"]),
            })
        branches.append({
            "state_id": sid, "candidate_index": ci,
            "branch_identity": f"{sid}:{ci:02d}", "candidate": source["candidate"],
            "horizon_tick_boundaries": list(boundaries),
            "ticks": ticks,
            "h3_labels": {
                "contact": cumulative_contact, "stuck": cumulative_stuck,
                "aggregate_unsafe": horizon_unsafe[-1],
                "replay_aggregate_unsafe": cumulative_unsafe,
            },
            "frozen_horizon_unsafe": horizon_unsafe,
            "replay_horizon_unsafe": replay_horizon_unsafe,
            "aggregate_replay_match": [a == b for a, b in zip(horizon_unsafe, replay_horizon_unsafe)],
        })
        print(json.dumps({"state_id": sid, "candidate": ci, "ticks": len(ticks)}), flush=True)
    if len(tick_counts) != 1 or len(boundary_sets) != 1:
        raise RuntimeError(f"{sid}: registered tick boundary differs across candidates")
    payload = {
        "schema": "dense_route_intent_true_future_state_v1",
        "status": "PASS", "state_index": state_index, "state_id": sid,
        "scene_id": state["scene_id"], "family": state["family"],
        "snapshot_digest": snapshot.digest,
        "expected_snapshot_digest": expected_snapshot_digest,
        "snapshot_digest_match": snapshot_digest_match,
        "horizon_tick_boundaries": list(next(iter(boundary_sets))),
        "h3_tick_count": next(iter(tick_counts)),
        "command_tick_s": 0.1,
        "materialisation": "deterministic registered-candidate replay plus static textured-v03 rendering of captured tick poses",
        "current_frame": current_frame, "branches": branches,
        "renderer_contract_digest": renderer.contract_digest,
        "runtime_s": time.time() - started,
    }
    payload["content_digest"] = canonical_digest(payload)
    atomic_json(out_path, payload)
    del ctx, renderer
    gc.collect()
    print(json.dumps({"state_id": sid, "status": "PASS", "frames": 1 + 12 * payload["h3_tick_count"],
                      "runtime_s": payload["runtime_s"]}), flush=True)
    return payload


def all_dense_states() -> list[dict]:
    states = json.loads((V1 / "state_manifest.json").read_text())["state_candidates"]
    payloads = []
    for state in states:
        path = OUT / "dense_replay" / f"{state['state_id']}.json"
        if not validate_existing_state(path):
            raise RuntimeError(f"dense replay missing or invalid: {state['state_id']}")
        payloads.append(json.loads(path.read_text()))
    return payloads


def frame_inventory() -> tuple[dict[str, dict], list[dict]]:
    unique = {}
    occurrences = []
    for state in all_dense_states():
        frames = [(state["current_frame"], "current", None, 0)]
        for branch in state["branches"]:
            frames.extend((tick, "future", branch["candidate_index"], tick["tick"])
                          for tick in branch["ticks"])
        for frame, kind, candidate, tick in frames:
            digest = frame["rgb_sha256"]
            record = {"rgb_sha256": digest, "rgb_path": frame["rgb_path"]}
            if digest in unique and unique[digest]["rgb_path"] != frame["rgb_path"]:
                if sha(Path(frame["rgb_path"])) != digest:
                    raise RuntimeError("duplicate RGB identity has invalid bytes")
            unique.setdefault(digest, record)
            occurrences.append({
                "state_id": state["state_id"], "kind": kind,
                "candidate_index": candidate, "tick": tick, "rgb_sha256": digest,
            })
    return unique, occurrences


def valid_token(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != TOKEN_BYTES:
        return False
    value = np.memmap(path, mode="r", dtype=np.float16, shape=TOKEN_SHAPE)
    return bool(np.isfinite(value).all())


def encode_all(batch_size: int = 16) -> dict:
    import torch
    from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21CroppedV03Arm, preprocessing_hash

    if sha(V2 / "target_latent_index.json") != EXPECTED_TARGET:
        raise RuntimeError("frozen target lineage mismatch")
    unique, occurrences = frame_inventory()
    cache_tokens = CACHE / "tokens"
    cache_tokens.mkdir(parents=True, exist_ok=True)
    arm = VJepa21CroppedV03Arm()
    if sha(Path(arm.checkpoint)) != EXPECTED_ENCODER:
        raise RuntimeError("frozen encoder checkpoint mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm.build(device, torch.float32)
    pending = []
    reused = 0
    for digest, record in sorted(unique.items()):
        path = cache_tokens / digest[:2] / f"{digest}.f16"
        record["token_path"] = str(path)
        if valid_token(path):
            reused += 1
        else:
            pending.append((digest, record, path))
    started = time.time()
    peak = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        pixels = torch.stack([arm.preprocess(item[1]["rgb_path"]) for item in batch]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            encoded = arm.tokens(pixels).float().cpu().numpy().astype(np.float16)
        for (_digest, record, path), value in zip(batch, encoded):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            np.ascontiguousarray(value).tofile(tmp)
            os.replace(tmp, path)
            if not valid_token(path):
                raise RuntimeError(f"invalid token shard: {path}")
            record["token_sha256"] = sha(path)
        if device.type == "cuda":
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
        print(json.dumps({"encoded": min(offset + len(batch), len(pending)),
                          "pending": len(pending), "unique": len(unique)}), flush=True)
    for record in unique.values():
        path = Path(record["token_path"])
        record["token_sha256"] = sha(path)
        record["shape"] = list(TOKEN_SHAPE)
        record["dtype"] = "float16"
    payload = {
        "schema": "dense_route_intent_true_future_token_index_v1",
        "complete": True,
        "encoder_checkpoint_sha256": EXPECTED_ENCODER,
        "preprocessing_digest": preprocessing_hash(arm),
        "preprocessing": "RGB rows 28:196, bicubic 512x384, ImageNet, frozen ViT-L final [768,1024]",
        "device": str(device), "batch_size": batch_size,
        "frame_occurrences": len(occurrences), "unique_frames": len(unique),
        "reused_tokens": reused, "new_tokens": len(pending),
        "runtime_s": time.time() - started, "peak_vram_bytes": peak,
        "cache_bytes": sum(Path(record["token_path"]).stat().st_size for record in unique.values()),
        "records": [unique[key] for key in sorted(unique)],
        "occurrences": occurrences,
    }
    payload["token_index_digest"] = canonical_digest(payload)
    atomic_json(OUT / "token_index.json", payload)
    print(json.dumps({k: payload[k] for k in (
        "frame_occurrences", "unique_frames", "reused_tokens", "new_tokens",
        "runtime_s", "peak_vram_bytes", "cache_bytes", "token_index_digest"
    )}, indent=2), flush=True)
    return payload


def finalize_evidence() -> dict:
    states = all_dense_states()
    token_index = json.loads((OUT / "token_index.json").read_text())
    if token_index.get("complete") is not True:
        raise RuntimeError("token index is incomplete")
    split = json.loads((V1 / "split.json").read_text())
    payload = {
        "schema": "dense_route_intent_true_future_evidence_receipt_v1",
        "complete": True,
        "bindings": {
            "state_manifest_sha256": sha(V1 / "state_manifest.json"),
            "branch_ledger_sha256": sha(V1 / "branch_labels.jsonl"),
            "split_sha256": sha(V1 / "split.json"),
            "route_target_index_sha256": sha(V2 / "target_latent_index.json"),
            "token_index_sha256": sha(OUT / "token_index.json"),
            "token_index_digest": token_index["token_index_digest"],
        },
        "states": len(states), "branches": sum(len(x["branches"]) for x in states),
        "h3_tick_counts": sorted({int(x["h3_tick_count"]) for x in states}),
        "horizon_tick_boundaries": sorted({tuple(x["horizon_tick_boundaries"]) for x in states}),
        "split_states": {name: len(values) for name, values in split.items() if isinstance(values, list)},
        "rgb_occurrences": token_index["frame_occurrences"],
        "unique_rgb_frames": token_index["unique_frames"],
        "token_cache_bytes": token_index["cache_bytes"],
        "replay_runtime_s": sum(float(x["runtime_s"]) for x in states),
        "encoding_runtime_s": token_index["runtime_s"],
        "peak_vram_bytes": token_index["peak_vram_bytes"],
        "method": "deterministic replay because the frozen ledger lacked complete tick poses; static render from replay-captured poses",
        "custody": {"new_states": 0, "new_branches": 0, "physics_replay": True,
                    "predictor_opened": False, "model_training": False},
    }
    payload["evidence_digest"] = canonical_digest(payload)
    atomic_json(OUT / "evidence_receipt.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true")
    group.add_argument("--encode-all", action="store_true")
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.collect_state is not None:
        collect_state(args.collect_state)
    elif args.collect_all:
        states = json.loads((V1 / "state_manifest.json").read_text())["state_candidates"]
        for index in range(len(states)):
            collect_state(index)
    elif args.encode_all:
        encode_all(args.batch_size)
    else:
        finalize_evidence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
