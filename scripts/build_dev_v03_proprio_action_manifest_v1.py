#!/usr/bin/env python3
"""Corrected-action + deployment-valid proprioception manifest.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only over the corpus; writes one new
manifest.  No image is re-rendered and no existing manifest is modified.

Row set
-------
Exactly the rows of ``temporal_rows.jsonl`` (4,566: 4,075 train / 491
checkpoint-selection).  Rows are only ever DROPPED, never added, so every cell of
the prospective factorial sees the same sequences.

Action contract (replaces the primitive-only representation)
------------------------------------------------------------
The action for the 0.5 s transition ``t -> t+240`` is the **five-tick post-slew
command trajectory**, 15 dims (tick-major, ``[vx, vy, yaw]`` per tick), rebuilt
deterministically by ``dev_action_slew_reconstruction_v1`` from the requested
primitive and the previous applied command.  It is then **verified tick-by-tick
against the logged post-limiter block**; a row whose action (or whose future
action blocks, for the rollout horizons) fails to verify is dropped.  The same
pure function serves hypothetical planning actions -- no measured body motion is
consulted at any point.

Proprioceptive contract (deployment-valid subset only)
-----------------------------------------------------
Per slot, the **trailing** five 10 Hz samples ending at the slot's own step, so
every proprioceptive timestamp is <= that slot's image timestamp.  Three slots
tile ``[s-14 .. s]`` contiguously.  32 channels per sample:

    [ 0: 3)  projected gravity      -- from roll/pitch only, yaw-free by construction
    [ 3: 6)  body angular velocity  -- gyro, body frame
    [ 6:18)  joint positions        -- 12 joints
    [18:30)  joint velocities       -- 12 joints
    [30:32)  previous applied command -- applied[k-1] on (vx, yaw), strictly historical

Excluded by decision: body linear velocity (simulator ground truth), absolute
yaw, world pose, camera extrinsics, foot contacts, joint effort, IMU linear
acceleration -- every empty, constant, privileged or deployment-invalid field.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_action_slew_reconstruction_v1 as SLEW  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
ROWS = CACHE / "temporal_rows.jsonl"
OUT = CACHE / "proprio_v1"
ROLLOUT = ROOT / ".generated/datagen_full/rollout"

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

FRAMES_PER_TIMESTEP = 48
SLOTS = 3                       # context slots, unchanged
SAMPLES_PER_SLOT = 5            # 10 Hz proprioception over a 0.5 s slot
PROPRIO_HISTORY = SLOTS * SAMPLES_PER_SLOT      # 15 samples, steps [s-14 .. s]
MAX_ACTION_BLOCKS = 4           # H = 1..4

JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
# Unitree SDK motor order: per leg (FR, FL, RR, RL) x (hip, thigh, calf).
UNITREE_ORDER = tuple(
    f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
TO_UNITREE = tuple(JOINT_ORDER.index(name) for name in UNITREE_ORDER)

CHANNELS = (
    ("projected_gravity", 3), ("body_angular_velocity", 3),
    ("joint_positions", 12), ("joint_velocities", 12),
    ("previous_applied_command", len(SLEW.ACTIVE_CHANNELS)),
)
PROPRIO_DIM = sum(width for _, width in CHANNELS)   # 32: vy is constant, excluded


def projected_gravity(roll: float, pitch: float):
    """Unit gravity in the body frame from roll/pitch alone.

    g_body = Rx(roll)^T Ry(pitch)^T Rz(yaw)^T (0, 0, -1); the yaw factor leaves
    the world z-axis invariant, so absolute heading cancels exactly and no
    privileged field enters.
    """
    return [math.sin(pitch),
            -math.sin(roll) * math.cos(pitch),
            -math.cos(roll) * math.cos(pitch)]


# --------------------------------------------------------------------------
def _scene_paths(scene: str):
    for frames in ROLLOUT.glob(f"*/*/chunk_*/plan/*_{scene}/frames.jsonl"):
        messages = frames.parent.parent.parent / "raw" / scene / "messages.jsonl"
        if messages.is_file():
            return frames, messages
    raise FileNotFoundError(f"no frames.jsonl/messages.jsonl pair for {scene}")


def _load_logged_blocks(messages: Path):
    """(env, block_index) -> logged post-limiter trajectory, from the raw stream."""
    needle = '"/lewm/go2/executed_command_block"'
    per_env = collections.defaultdict(list)
    with open(messages, "r", encoding="utf-8") as stream:
        for line in stream:
            if needle not in line:
                continue
            record = json.loads(line)
            if record.get("canonical_topic") != "/lewm/go2/executed_command_block":
                continue
            per_env[record["env_index"]].append((record["timestamp_ns"], record["payload"]))
    logged = {}
    for env_index, entries in per_env.items():
        entries.sort(key=lambda item: item[0])
        for block_index, (_, payload) in enumerate(entries):
            logged[(env_index, block_index)] = [
                [payload["executed_vx_body_mps"][t],
                 payload["executed_vy_body_mps"][t],
                 payload["executed_yaw_rate_radps"][t]]
                for t in range(SLEW.TICKS)]
    return logged


def _load_frames(frames: Path):
    """(env, global_step) -> the per-frame record, plus the requested set-point.

    ``episode_step`` RESTARTS at a reset, so it is not unique within a scene; the
    key is the global step derived from ``frame_index``, which is monotone in
    time and unique per (scene, env).  Getting this wrong silently merges two
    episodes.
    """
    table = {}
    with open(frames, "r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            env_index = record["env_index"]
            step = record["frame_index"] // FRAMES_PER_TIMESTEP + 1
            joints = record["joint_state"]
            index = {name: i for i, name in enumerate(joints["names"])}
            order = [index[name] for name in JOINT_ORDER]
            rpy = record["base_rpy_rad"]
            command = record["command_context"]
            tick = (step - 1) % SLEW.TICKS   # global step: block-aligned by construction
            table[(env_index, step)] = {
                "roll": rpy["roll"], "pitch": rpy["pitch"],
                "gyro": [record["twist_body"]["angular"][axis] for axis in "xyz"],
                "q": [joints["position"][i] for i in order],
                "dq": [joints["velocity"][i] for i in order],
                "requested": [command["vx_body_mps"][tick], command["vy_body_mps"][tick],
                              command["yaw_rate_radps"][tick]],
                "episode": (env_index, record["episode"]["episode_id"],
                            record["episode"]["reset_count"]),
                "timestamp_ns": record["timestamp_ns"],
                "frame_index": record["frame_index"],
            }
    return table


def _reconstruct_applied(table, envs, max_step):
    """Run the limiter in global time order; return applied[(env, global_step)].

    ``previous`` is reset to zero whenever the episode identity changes, which is
    the physical reset contract: after a respawn the controller starts from a
    standing command.  Also returns the reconstructed 5-tick trajectory per
    (env, block_index) so it can be verified against the logged block.
    """
    applied, blocks = {}, {}
    for env_index in envs:
        previous = SLEW.RESET_APPLIED
        episode = None
        for block_index in range((max_step + SLEW.TICKS - 1) // SLEW.TICKS):
            steps = [block_index * SLEW.TICKS + 1 + t for t in range(SLEW.TICKS)]
            if any((env_index, s) not in table for s in steps):
                break
            here = table[(env_index, steps[0])]["episode"]
            if episode is not None and here != episode:
                previous = SLEW.RESET_APPLIED     # respawn: limiter starts from stand
            episode = here
            requested = [table[(env_index, s)]["requested"] for s in steps]
            trajectory, previous = SLEW.apply_slew(requested, previous)
            blocks[(env_index, block_index)] = trajectory
            for s, value in zip(steps, trajectory):
                applied[(env_index, s)] = value
    return applied, blocks


def _verify(blocks, logged):
    """Tick-exact comparison; returns the set of verified (env, block_index)."""
    verified, mismatched = set(), 0
    for key, trajectory in blocks.items():
        reference = logged.get(key)
        if reference is None:
            continue
        if all(abs(a - b) < 1e-6 for pr, lo in zip(trajectory, reference)
               for a, b in zip(pr, lo)):
            verified.add(key)
        else:
            mismatched += 1
    return verified, mismatched


def build_scene(task):
    scene, rows = task
    frames_path, messages_path = _scene_paths(scene)
    table = _load_frames(frames_path)
    logged = _load_logged_blocks(messages_path)
    envs = sorted({env for env, _ in table})
    max_step = max(step for _, step in table)
    applied, blocks = _reconstruct_applied(table, envs, max_step)
    verified, mismatched = _verify(blocks, logged)

    dropped = collections.Counter()
    built = []
    for row in rows:
        frame_index = [f for f in row["frames"] if f["offset"] == 0][0]["frame_index"]
        step = frame_index // FRAMES_PER_TIMESTEP + 1
        env_index = row["env_index"]
        episode = (env_index, row["episode_id"], row["reset_count"])

        history = [step - PROPRIO_HISTORY + 1 + i for i in range(PROPRIO_HISTORY)]
        if any((env_index, s) not in table for s in history):
            dropped["proprio_history_absent"] += 1
            continue
        if any(table[(env_index, s)]["episode"] != episode for s in history):
            dropped["proprio_history_crosses_reset"] += 1
            continue
        if any((env_index, s - 1) not in applied and s > 1 for s in history):
            dropped["previous_applied_absent"] += 1
            continue

        first_block = (step - 1) // SLEW.TICKS
        action_blocks, ok = [], True
        for offset in range(MAX_ACTION_BLOCKS):
            key = (env_index, first_block + offset)
            if key not in blocks:
                break
            if key not in verified:
                ok = False
                break
            action_blocks.append(SLEW.flatten(blocks[key]))
        if not ok:
            dropped["action_block_failed_verification"] += 1
            continue
        if not action_blocks:
            dropped["action_block_absent"] += 1
            continue

        proprio = []
        for s in history:
            record = table[(env_index, s)]
            previous_applied = applied.get((env_index, s - 1), list(SLEW.RESET_APPLIED))
            proprio.append(projected_gravity(record["roll"], record["pitch"])
                           + record["gyro"] + record["q"] + record["dq"]
                           + [previous_applied[c] for c in SLEW.ACTIVE_CHANNELS])

        built.append({
            "pair_sha256": row["pair_sha256"], "role": row["role"],
            "scene": row["scene"], "family": row["family"],
            "env_index": env_index, "episode_id": row["episode_id"],
            "reset_count": row["reset_count"],
            "step": step, "t": row["t"],
            "proprio_steps": history,
            "proprio_timestamps_ns": [table[(env_index, s)]["timestamp_ns"] for s in history],
            "image_timestamp_ns": table[(env_index, step)]["timestamp_ns"],
            "proprio": proprio,
            "action_blocks": action_blocks,
            "action_block_indices": [first_block + o for o in range(len(action_blocks))],
            "primitive": row["primitive"],
        })
    return scene, built, dropped, {"blocks": len(blocks), "verified": len(verified),
                                   "mismatched": mismatched}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    by_scene = collections.defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    tasks = sorted(by_scene.items())
    print(f"building {len(rows)} rows over {len(tasks)} scenes, {args.workers} workers",
          flush=True)

    from multiprocessing import Pool
    built, dropped, verify_stats = [], collections.Counter(), collections.Counter()
    with Pool(args.workers) as pool:
        for index, (scene, rows_out, drops, stats) in enumerate(
                pool.imap_unordered(build_scene, tasks), 1):
            built.extend(rows_out)
            dropped.update(drops)
            verify_stats.update(stats)
            if index % 10 == 0:
                print(f"  {index}/{len(tasks)} scenes, {len(built)} rows", flush=True)

    built.sort(key=lambda r: r["pair_sha256"])
    rows_path = out / "proprio_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as stream:
        for row in built:
            stream.write(json.dumps(row) + "\n")

    # -- normalisation statistics: TRAIN SPLIT ONLY, frozen and hashed
    train = [r for r in built if r["role"] == "train"]
    total = [0.0] * PROPRIO_DIM
    total_sq = [0.0] * PROPRIO_DIM
    count = 0
    for row in train:
        for sample in row["proprio"]:
            count += 1
            for c, value in enumerate(sample):
                total[c] += value
                total_sq[c] += value * value
    mean = [t / count for t in total]
    std = [max(math.sqrt(max(total_sq[c] / count - mean[c] ** 2, 0.0)), 1e-3)
           for c in range(PROPRIO_DIM)]
    stats = {"mean": mean, "std": std, "samples": count,
             "source": "train split only", "channels": [list(c) for c in CHANNELS]}
    stats_text = json.dumps(stats, sort_keys=True)
    stats["sha256"] = hashlib.sha256(stats_text.encode()).hexdigest()
    (out / "proprio_norm_stats.json").write_text(json.dumps(stats, indent=2))

    kept = collections.Counter(r["role"] for r in built)
    manifest = {
        "status": STATUS, "claim_bearing": False,
        "source_rows": str(ROWS), "source_rows_count": len(rows),
        "rows_kept": len(built), "rows_kept_by_role": dict(kept),
        "rows_dropped": dict(dropped),
        "proprio": {
            "dim": PROPRIO_DIM, "samples_per_slot": SAMPLES_PER_SLOT,
            "slots": SLOTS, "history_samples": PROPRIO_HISTORY,
            "window": "trailing: steps [s-14 .. s], every timestamp <= the slot image",
            "channels": [list(c) for c in CHANNELS],
            "excluded": ["lateral command vy (identically zero: constant field)",
                         "body linear velocity (simulator ground truth)", "absolute yaw",
                         "world pose", "camera extrinsics", "foot contacts",
                         "joint effort", "IMU linear acceleration"],
            "joint_order_manifest": list(JOINT_ORDER),
            "joint_order_unitree": list(UNITREE_ORDER),
            "to_unitree_permutation": list(TO_UNITREE),
        },
        "action": {
            "dim": SLEW.ACTION_DIM, "ticks": SLEW.TICKS,
            "representation": "five-tick post-slew command trajectory, tick-major (vx, vy, yaw)",
            "reconstruction": "applied[k] = prev + clip(requested[k] - prev, +-rate)",
            "rates": {"vx": SLEW.VX_RATE, "vy": SLEW.VY_RATE, "yaw": SLEW.YAW_RATE},
            "verified_against": "logged executed_command_block, tick-exact",
            "planning_time_identical": True,
            "inputs_not_used": ["measured body motion", "future proprioception"],
        },
        "verification": dict(verify_stats),
        "normalisation_sha256": stats["sha256"],
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    (out / "proprio_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "proprio"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
