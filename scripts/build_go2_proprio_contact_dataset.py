#!/usr/bin/env python3
"""Build a proprioceptive contact-detection dataset from closed-loop result logs.

Each row is an H-tick window of nonprivileged proprioceptive features
(executed displacement, yaw/z deltas, roll/pitch, executed primitive one-hot,
requested-vs-executed mismatch) ending at tick t, labeled with the privileged
``body_clearance_violation`` flag at tick t. Labels are offline-only; the
runtime detector consumes the same features from live proprioception.

Windows are deduplicated by rounded feature hash. Rows whose hash also occurs
in a validation file are dropped from train (deterministic-prefix leak guard).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PRIMITIVES = (
    "forward_slow",
    "forward_medium",
    "forward_fast",
    "arc_left",
    "arc_right",
    "yaw_left",
    "yaw_right",
    "backward",
    "hold",
)
PRIMITIVE_INDEX = {name: idx for idx, name in enumerate(PRIMITIVES)}
# Nominal contact-free execution magnitudes per primitive (clear-tick medians
# over the physical corpus); runtime-known constants of the primitive bank.
NOMINAL_DISP_M = {
    "forward_slow": 0.05,
    "forward_medium": 0.100,
    "forward_fast": 0.150,
    "arc_left": 0.086,
    "arc_right": 0.080,
    "yaw_left": 0.031,
    "yaw_right": 0.030,
    "backward": 0.057,
    "hold": 0.001,
}
NOMINAL_ABS_DYAW = {
    "forward_slow": 0.05,
    "forward_medium": 0.050,
    "forward_fast": 0.050,
    "arc_left": 0.096,
    "arc_right": 0.164,
    "yaw_left": 0.152,
    "yaw_right": 0.172,
    "backward": 0.053,
    "hold": 0.0,
}
# displacement, dyaw, roll, pitch, dz, mismatch, tip, shortfall_disp,
# shortfall_yaw, primitive one-hot (+other)
FEATURE_DIM = 9 + len(PRIMITIVES) + 1
TICK_FIELDS = (
    "executed_displacement_m",
    "post_yaw",
    "post_roll",
    "post_pitch",
    "post_z",
    "primitive",
)


def _wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def _tick_features(tick: dict, prev: dict | None) -> np.ndarray | None:
    for field in TICK_FIELDS:
        if tick.get(field) is None:
            return None
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    disp = float(np.clip(float(tick["executed_displacement_m"]), 0.0, 0.3))
    features[0] = disp
    dyaw = 0.0
    if prev is not None and prev.get("post_yaw") is not None:
        dyaw = float(
            np.clip(_wrap_angle(float(tick["post_yaw"]) - float(prev["post_yaw"])), -0.6, 0.6)
        )
        features[1] = dyaw
        if prev.get("post_z") is not None:
            features[4] = float(np.clip(float(tick["post_z"]) - float(prev["post_z"]), -0.1, 0.1))
    features[2] = float(np.clip(float(tick["post_roll"]), -0.8, 0.8))
    features[3] = float(np.clip(float(tick["post_pitch"]), -0.8, 0.8))
    requested = tick.get("requested_primitive")
    executed = str(tick["primitive"])
    features[5] = 1.0 if requested is not None and str(requested) != executed else 0.0
    tip = tick.get("post_tip_rad")
    features[6] = float(np.clip(float(tip), 0.0, 1.0)) if tip is not None else 0.0
    features[7] = float(NOMINAL_DISP_M.get(executed, 0.05)) - disp
    features[8] = float(NOMINAL_ABS_DYAW.get(executed, 0.05)) - abs(dyaw)
    features[9 + PRIMITIVE_INDEX.get(executed, len(PRIMITIVES))] = 1.0
    return features


def _load_rows(
    path: Path,
    *,
    window: int,
) -> tuple[list[np.ndarray], list[int], list[dict]] | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    result = data.get("result") or {}
    if str(result.get("execution_mode")) != "physical":
        return None
    log = data.get("log")
    if not isinstance(log, list) or len(log) < window:
        return None
    scene_id = str(result.get("scene") or "")
    per_tick: list[np.ndarray | None] = []
    prev = None
    for tick in log:
        if not isinstance(tick, dict):
            per_tick.append(None)
            prev = None
            continue
        per_tick.append(_tick_features(tick, prev))
        prev = tick
    windows: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict] = []
    for end in range(window - 1, len(log)):
        chunk = per_tick[end - window + 1 : end + 1]
        if any(item is None for item in chunk):
            continue
        tick = log[end]
        label_value = tick.get("body_clearance_violation")
        if label_value is None:
            continue
        windows.append(np.stack(chunk))  # type: ignore[arg-type]
        labels.append(1 if bool(label_value) else 0)
        meta.append(
            {
                "scene_id": scene_id,
                "tick": int(tick.get("tick", end)),
                "state": str(tick.get("state", "")),
                "primitive": str(tick.get("primitive", "")),
            }
        )
    if not windows:
        return None
    return windows, labels, meta


def _window_hash(window: np.ndarray, label: int) -> bytes:
    rounded = np.round(window.astype(np.float64), 4)
    return hashlib.blake2b(
        rounded.tobytes() + bytes([label]), digest_size=16
    ).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="result JSON paths or globs")
    parser.add_argument("--val-file", action="append", default=[], help="result JSONs routed to validation split")
    parser.add_argument("--val-scene", action="append", default=[], help="scene ids routed to validation split")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    paths: list[Path] = []
    seen_paths: set[Path] = set()
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for match in matches:
            path = Path(match).resolve()
            if path not in seen_paths and path.is_file():
                seen_paths.add(path)
                paths.append(path)
    val_paths = {Path(item).resolve() for item in args.val_file}
    for val_path in val_paths:
        if val_path not in seen_paths and val_path.is_file():
            paths.append(val_path)
    if args.max_files is not None:
        non_val = [p for p in paths if p not in val_paths]
        paths = non_val[: int(args.max_files)] + [p for p in paths if p in val_paths]
    val_scenes = set(args.val_scene)

    splits: dict[str, dict[str, list]] = {
        name: {"X": [], "y": [], "meta": []} for name in ("train", "val")
    }
    val_hashes: set[bytes] = set()
    train_pending: list[tuple[np.ndarray, int, dict, bytes]] = []
    skipped_files = 0
    for path in paths:
        loaded = _load_rows(path, window=int(args.window))
        if loaded is None:
            skipped_files += 1
            continue
        windows, labels, meta = loaded
        is_val = path in val_paths or (meta and meta[0]["scene_id"] in val_scenes)
        for window_arr, label, row_meta in zip(windows, labels, meta):
            digest = _window_hash(window_arr, label)
            row_meta = dict(row_meta, file=path.name)
            if is_val:
                if digest in val_hashes:
                    continue
                val_hashes.add(digest)
                splits["val"]["X"].append(window_arr)
                splits["val"]["y"].append(label)
                splits["val"]["meta"].append(row_meta)
            else:
                train_pending.append((window_arr, label, row_meta, digest))
        print(f"[{len(splits['val']['y']) + len(train_pending):>8}] {path.name}", file=sys.stderr)

    train_hashes: set[bytes] = set()
    dropped_dup = 0
    dropped_leak = 0
    for window_arr, label, row_meta, digest in train_pending:
        if digest in val_hashes:
            dropped_leak += 1
            continue
        if digest in train_hashes:
            dropped_dup += 1
            continue
        train_hashes.add(digest)
        splits["train"]["X"].append(window_arr)
        splits["train"]["y"].append(label)
        splits["train"]["meta"].append(row_meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    summary: dict[str, object] = {
        "schema": "go2_proprio_contact_dataset_v0",
        "window": int(args.window),
        "feature_dim": FEATURE_DIM,
        "primitives": list(PRIMITIVES),
        "files": len(paths),
        "skipped_files": skipped_files,
        "dropped_duplicate_train_rows": dropped_dup,
        "dropped_val_leak_train_rows": dropped_leak,
    }
    for name, split in splits.items():
        if not split["y"]:
            raise SystemExit(f"empty split: {name}")
        payload[f"{name}_X"] = np.stack(split["X"]).astype(np.float32)
        payload[f"{name}_y"] = np.asarray(split["y"], dtype=np.int64)
        payload[f"{name}_meta"] = np.asarray(
            [json.dumps(m, sort_keys=True) for m in split["meta"]], dtype=object
        )
        summary[name] = {
            "rows": int(len(split["y"])),
            "positives": int(sum(split["y"])),
            "positive_rate": float(np.mean(split["y"])),
        }
    np.savez_compressed(args.output, **payload)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
