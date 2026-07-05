#!/usr/bin/env python3
"""Build synthetic Go2 learned-local examples for online-map frontier behavior.

The generated rows use the standard learned-local dataset schema, but the
features are already materialized as ``clock_state_online_map_v1``.  Labels come
from a runtime-safe egomotion frontier teacher over visited/blocked/claim cells;
no scene map, route table, landmark coordinate, or privileged planner is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PRIMITIVES = (
    "forward_fast",
    "forward_medium",
    "arc_left",
    "arc_right",
    "yaw_left",
    "yaw_right",
    "backward",
    "hold",
)
STATE_FEATURES = ("EXPLORE", "SEEK", "SERVO", "CLAIM")
ONLINE_MAP_CHANNELS = 8
FEATURE_VARIANT = "clock_state_online_map_edge_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=60000)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--validation-examples", type=int, default=12000)
    parser.add_argument("--base-dim", type=int, default=1600)
    parser.add_argument(
        "--base-sample-datasets",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Optional learned-local NPZ datasets whose leading --base-dim columns "
            "will be sampled into synthetic rows. This makes the frontier labels "
            "robust to realistic runtime JEPA/memory/outcome feature scale without "
            "using any privileged map or route input."
        ),
    )
    parser.add_argument("--base-sample-limit", type=int, default=20000)
    parser.add_argument("--base-noise-std", type=float, default=0.0)
    parser.add_argument("--map-size", type=int, default=11)
    parser.add_argument(
        "--post-claim-corridor-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of rows sampled from a scene-agnostic post-claim corridor "
            "template: the online map has a claimed/visited cell at or behind "
            "the robot and an unvisited frontier ahead."
        ),
    )
    parser.add_argument(
        "--post-claim-junction-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of rows sampled from a scene-agnostic post-claim junction "
            "template: the online map has a claimed/visited trail behind the "
            "robot and an unvisited left/right branch at the corridor exit."
        ),
    )
    parser.add_argument(
        "--post-claim-escape-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of rows sampled from a scene-agnostic post-claim escape "
            "template: the online map has a nearby claimed trail, guard-like "
            "attempted directions, and one open frontier direction."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument(
        "--state",
        default="EXPLORE",
        choices=STATE_FEATURES,
        help="Controller state encoded in synthetic rows.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    base_pool = _load_base_feature_pool(
        args.base_sample_datasets or [],
        base_dim=int(args.base_dim),
        limit=int(args.base_sample_limit),
        rng=rng,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_report = _write_dataset(
        args.output,
        examples=int(args.examples),
        base_dim=int(args.base_dim),
        base_pool=base_pool,
        base_noise_std=float(args.base_noise_std),
        map_size=int(args.map_size),
        post_claim_corridor_fraction=float(args.post_claim_corridor_fraction),
        post_claim_junction_fraction=float(args.post_claim_junction_fraction),
        post_claim_escape_fraction=float(args.post_claim_escape_fraction),
        state=str(args.state),
        rng=rng,
        seed=int(args.seed),
        split="train",
    )
    reports = {"train": train_report}
    if args.validation_output is not None:
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        reports["validation"] = _write_dataset(
            args.validation_output,
            examples=int(args.validation_examples),
            base_dim=int(args.base_dim),
            base_pool=base_pool,
            base_noise_std=float(args.base_noise_std),
            map_size=int(args.map_size),
            post_claim_corridor_fraction=float(args.post_claim_corridor_fraction),
            post_claim_junction_fraction=float(args.post_claim_junction_fraction),
            post_claim_escape_fraction=float(args.post_claim_escape_fraction),
            state=str(args.state),
            rng=np.random.default_rng(int(args.seed) + 1),
            seed=int(args.seed) + 1,
            split="validation",
        )
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


def _write_dataset(
    path: Path,
    *,
    examples: int,
    base_dim: int,
    base_pool: np.ndarray | None,
    base_noise_std: float,
    map_size: int,
    post_claim_corridor_fraction: float,
    post_claim_junction_fraction: float,
    post_claim_escape_fraction: float,
    state: str,
    rng: np.random.Generator,
    seed: int,
    split: str,
) -> dict[str, object]:
    if examples <= 0:
        raise SystemExit("--examples must be positive")
    size = _odd_size(map_size)
    feature_dim = int(base_dim) + 3 + len(STATE_FEATURES) + ONLINE_MAP_CHANNELS * size * size
    features = np.zeros((examples, feature_dim), dtype=np.float32)
    labels = np.zeros((examples,), dtype=np.int64)
    meta_json: list[str] = []
    post_claim_examples = 0
    post_claim_junction_examples = 0
    post_claim_escape_examples = 0
    for idx in range(examples):
        corridor_fraction = max(0.0, min(1.0, float(post_claim_corridor_fraction)))
        junction_fraction = max(0.0, min(1.0, float(post_claim_junction_fraction)))
        escape_fraction = max(0.0, min(1.0, float(post_claim_escape_fraction)))
        total_special_fraction = min(1.0, corridor_fraction + junction_fraction + escape_fraction)
        draw = float(rng.random())
        post_claim = bool(draw < total_special_fraction)
        post_claim_escape = bool(post_claim and draw < escape_fraction)
        post_claim_junction = bool(
            post_claim
            and not post_claim_escape
            and draw < escape_fraction + junction_fraction
        )
        label_override: str | None = None
        if post_claim_escape:
            channels, tick, label_override = _sample_post_claim_escape_map(rng, size=size)
            post_claim_escape_examples += 1
        elif post_claim_junction:
            channels, tick, label_override = _sample_post_claim_junction_map(rng, size=size)
            post_claim_junction_examples += 1
        elif post_claim:
            channels, tick = _sample_post_claim_corridor_map(rng, size=size)
            post_claim_examples += 1
        else:
            channels, tick = _sample_online_map(rng, size=size)
        label_name = label_override or _frontier_teacher_label(channels, rng=rng)
        labels[idx] = PRIMITIVES.index(label_name)
        _encode_base_features(
            features[idx],
            base_dim=int(base_dim),
            base_pool=base_pool,
            base_noise_std=float(base_noise_std),
            rng=rng,
        )
        _encode_clock_state_map(
            features[idx],
            base_dim=int(base_dim),
            state=state,
            map_channels=channels,
            tick=int(tick),
            max_tick=700,
        )
        meta_json.append(
            json.dumps(
                {
                    "tick": int(tick),
                    "state": state,
                    "label": label_name,
                    "target_color": "",
                    "target_index": 0,
                    "pose_xy": [0.0, 0.0],
                    "yaw_rad": 0.0,
                    "claimed_count": 1 if post_claim else 0,
                    "synthetic_post_claim_corridor": bool(post_claim),
                    "synthetic_post_claim_junction": bool(post_claim_junction),
                    "synthetic_post_claim_escape": bool(post_claim_escape),
                    "synthetic_online_map_frontier": True,
                },
                sort_keys=True,
            )
        )

    label_counts = {
        PRIMITIVES[int(label)]: int(count)
        for label, count in sorted(Counter(labels.tolist()).items())
    }
    result_json = json.dumps(
        {
            "schema": "lewm_go2_online_map_frontier_synthetic_result_v0",
            "scene": f"synthetic_online_map_frontier_{split}",
            "success": True,
            "wall_metrics": {
                "learned_local_policy_feature_variant": FEATURE_VARIANT,
                "synthetic_online_map_frontier_examples": int(examples),
                "synthetic_online_map_frontier_seed": int(seed),
                "synthetic_online_map_frontier_label_counts": label_counts,
                "synthetic_online_map_frontier_base_sample_rows": (
                    0 if base_pool is None else int(base_pool.shape[0])
                ),
                "synthetic_online_map_frontier_base_noise_std": float(base_noise_std),
                "synthetic_post_claim_corridor_examples": int(post_claim_examples),
                "synthetic_post_claim_corridor_fraction": float(post_claim_corridor_fraction),
                "synthetic_post_claim_junction_examples": int(post_claim_junction_examples),
                "synthetic_post_claim_junction_fraction": float(post_claim_junction_fraction),
                "synthetic_post_claim_escape_examples": int(post_claim_escape_examples),
                "synthetic_post_claim_escape_fraction": float(post_claim_escape_fraction),
            },
        },
        sort_keys=True,
    )
    np.savez_compressed(
        path,
        schema=np.asarray(["lewm_go2_closed_loop_learned_local_policy_dataset_v0"]),
        features=features,
        labels=labels,
        primitive_vocab=np.asarray(PRIMITIVES),
        result_json=np.asarray([result_json]),
        meta_json=np.asarray(meta_json),
    )
    return {
        "path": str(path),
        "examples": int(examples),
        "feature_dim": int(feature_dim),
        "feature_variant": FEATURE_VARIANT,
        "label_counts": label_counts,
        "base_sample_rows": 0 if base_pool is None else int(base_pool.shape[0]),
        "base_noise_std": float(base_noise_std),
        "post_claim_corridor_examples": int(post_claim_examples),
        "post_claim_corridor_fraction": float(post_claim_corridor_fraction),
        "post_claim_junction_examples": int(post_claim_junction_examples),
        "post_claim_junction_fraction": float(post_claim_junction_fraction),
        "post_claim_escape_examples": int(post_claim_escape_examples),
        "post_claim_escape_fraction": float(post_claim_escape_fraction),
    }


def _load_base_feature_pool(
    paths: list[Path],
    *,
    base_dim: int,
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "features" not in data:
                continue
            features = np.asarray(data["features"], dtype=np.float32)
            if features.ndim != 2 or int(features.shape[1]) < int(base_dim):
                raise SystemExit(
                    f"{path} has feature shape {features.shape}, cannot sample base_dim={base_dim}"
                )
            if int(features.shape[0]) == 0:
                continue
            rows.append(np.asarray(features[:, : int(base_dim)], dtype=np.float32))
    if not rows:
        return None
    pool = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    max_rows = int(limit)
    if max_rows > 0 and int(pool.shape[0]) > max_rows:
        keep = rng.choice(int(pool.shape[0]), size=max_rows, replace=False)
        pool = pool[np.asarray(keep, dtype=np.int64)]
    return pool


def _encode_base_features(
    row: np.ndarray,
    *,
    base_dim: int,
    base_pool: np.ndarray | None,
    base_noise_std: float,
    rng: np.random.Generator,
) -> None:
    if int(base_dim) <= 0:
        return
    if base_pool is not None and int(base_pool.shape[0]) > 0:
        sample_idx = int(rng.integers(0, int(base_pool.shape[0])))
        row[: int(base_dim)] = base_pool[sample_idx]
    if float(base_noise_std) > 0.0:
        row[: int(base_dim)] += rng.normal(
            loc=0.0,
            scale=float(base_noise_std),
            size=(int(base_dim),),
        ).astype(np.float32)


def _odd_size(size: int) -> int:
    out = max(3, int(size))
    return out + 1 if out % 2 == 0 else out


def _sample_online_map(rng: np.random.Generator, *, size: int) -> tuple[np.ndarray, int]:
    radius = size // 2
    channels = np.zeros((ONLINE_MAP_CHANNELS, size, size), dtype=np.float32)
    tick = int(rng.integers(20, 700))
    current = (radius, radius)
    visited: set[tuple[int, int]] = {current}
    blocked: set[tuple[int, int]] = set()

    walk_len = int(rng.integers(4, 32))
    row, col = current
    for step in range(walk_len):
        visited.add((row, col))
        if rng.random() < 0.22:
            ahead = (row - 1, col)
            if _in_bounds(ahead, size) and ahead not in visited:
                blocked.add(ahead)
        moves = [(-1, 0), (0, 1), (0, -1), (1, 0)]
        rng.shuffle(moves)
        for dr, dc in moves:
            nxt = (row + dr, col + dc)
            if _in_bounds(nxt, size) and nxt not in blocked:
                row, col = nxt
                break

    local_visit_prob = float(rng.uniform(0.03, 0.18))
    local_block_prob = float(rng.uniform(0.01, 0.09))
    for r in range(size):
        for c in range(size):
            dist = abs(r - radius) + abs(c - radius)
            if dist <= 1:
                continue
            if rng.random() < local_visit_prob * max(0.25, 1.0 - dist / (size + 1)):
                visited.add((r, c))
            if rng.random() < local_block_prob:
                blocked.add((r, c))

    blocked.discard(current)
    for r, c in visited:
        channels[0, r, c] = 1.0
        age = float(rng.integers(0, 220))
        channels[2, r, c] = max(channels[2, r, c], max(0.0, 1.0 - age / 160.0))
    channels[0, radius, radius] = 1.0
    channels[2, radius, radius] = 1.0
    for r, c in blocked:
        channels[1, r, c] = 1.0
        channels[0, r, c] = 0.0
        channels[2, r, c] = 0.0
    if rng.random() < 0.10:
        r = int(rng.integers(max(0, radius - 2), min(size, radius + 3)))
        c = int(rng.integers(max(0, radius - 2), min(size, radius + 3)))
        if (r, c) != current and channels[1, r, c] < 0.5:
            channels[3, r, c] = 1.0
            channels[0, r, c] = 1.0
    _add_edge_frontier_channels(channels, current=current)
    return channels, tick


def _sample_post_claim_corridor_map(
    rng: np.random.Generator,
    *,
    size: int,
) -> tuple[np.ndarray, int]:
    radius = size // 2
    channels = np.zeros((ONLINE_MAP_CHANNELS, size, size), dtype=np.float32)
    tick = int(rng.integers(20, 700))
    current = (radius, radius)
    visited: set[tuple[int, int]] = {current}
    blocked: set[tuple[int, int]] = set()

    tail = int(rng.integers(2, max(3, min(7, radius + 1))))
    lateral = int(rng.choice([-1, 0, 1], p=[0.18, 0.64, 0.18]))
    for step in range(1, tail + 1):
        row = min(size - 1, radius + step)
        col = max(0, min(size - 1, radius + (lateral if step > 1 else 0)))
        visited.add((row, col))
    for step in range(1, min(3, radius) + 1):
        if rng.random() < 0.35:
            visited.add((max(0, radius - step), radius))

    for row in range(max(0, radius - 1), min(size, radius + tail + 1)):
        if rng.random() < 0.72:
            blocked.add((row, max(0, radius - 1)))
        if rng.random() < 0.72:
            blocked.add((row, min(size - 1, radius + 1)))
    blocked.discard(current)
    blocked.discard((max(0, radius - 1), radius))

    claim_cell = current if rng.random() < 0.55 else (min(size - 1, radius + 1), radius)
    visited.add(claim_cell)
    for r, c in visited:
        channels[0, r, c] = 1.0
        age = 0.0 if (r, c) == current else float(rng.integers(0, 180))
        channels[2, r, c] = max(0.0, 1.0 - age / 160.0)
    for r, c in blocked:
        if (r, c) in visited:
            continue
        channels[1, r, c] = 1.0
    channels[0, radius, radius] = 1.0
    channels[2, radius, radius] = 1.0
    channels[3, claim_cell[0], claim_cell[1]] = 1.0
    if rng.random() < 0.35:
        channels[7, min(size - 1, radius + 1), radius] = 1.0
    _add_edge_frontier_channels(channels, current=current)
    return channels, tick


def _sample_post_claim_junction_map(
    rng: np.random.Generator,
    *,
    size: int,
) -> tuple[np.ndarray, int, str]:
    radius = size // 2
    channels = np.zeros((ONLINE_MAP_CHANNELS, size, size), dtype=np.float32)
    tick = int(rng.integers(20, 700))
    current = (radius, radius)
    visited: set[tuple[int, int]] = {current}
    blocked: set[tuple[int, int]] = set()

    side = int(rng.choice([-1, 1]))
    side_label = "left" if side > 0 else "right"
    mode = str(rng.choice(["arc", "yaw", "commit"], p=[0.50, 0.28, 0.22]))

    tail = int(rng.integers(2, max(3, min(7, radius + 1))))
    for step in range(1, tail + 1):
        visited.add((min(size - 1, radius + step), radius))
    claim_cell = (
        min(size - 1, radius + int(rng.integers(1, min(3, tail) + 1))),
        radius,
    )
    visited.add(claim_cell)

    # Approach corridor walls behind the robot plus a blocked dead-end/opposite
    # branch at the junction. The open branch itself is left unknown.
    for row in range(radius, min(size, radius + tail + 1)):
        blocked.add((row, max(0, radius - 1)))
        blocked.add((row, min(size - 1, radius + 1)))
    for cell in (
        (radius, radius - side),
        (max(0, radius - 1), radius - side),
        (max(0, radius - 2), radius - side),
    ):
        if _in_bounds(cell, size):
            blocked.add(cell)

    if mode in {"arc", "yaw"}:
        if rng.random() < 0.82:
            blocked.add((max(0, radius - 1), radius))
        for cell in (
            (radius, radius + side),
            (max(0, radius - 1), radius + side),
            (max(0, radius - 2), radius + side),
            (radius, radius + 2 * side),
        ):
            if _in_bounds(cell, size):
                blocked.discard(cell)
        if rng.random() < 0.42:
            visited.add((min(size - 1, radius + 1), radius + side))
        label = f"arc_{side_label}" if mode == "arc" else f"yaw_{side_label}"
    else:
        for step in range(1, int(rng.integers(2, 5))):
            visited.add((radius, max(0, min(size - 1, radius - side * step))))
        if rng.random() < 0.55:
            blocked.add((radius, max(0, min(size - 1, radius - side * 2))))
        label = "forward_medium"

    blocked.discard(current)
    blocked.discard(claim_cell)
    for r, c in visited:
        channels[0, r, c] = 1.0
        age = 0.0 if (r, c) == current else float(rng.integers(0, 180))
        channels[2, r, c] = max(0.0, 1.0 - age / 160.0)
    for r, c in blocked:
        if (r, c) in visited:
            continue
        channels[1, r, c] = 1.0
        channels[0, r, c] = 0.0
        channels[2, r, c] = 0.0
    channels[0, radius, radius] = 1.0
    channels[2, radius, radius] = 1.0
    channels[3, claim_cell[0], claim_cell[1]] = 1.0
    if rng.random() < 0.55:
        channels[7, min(size - 1, radius + 1), radius] = 1.0
    _add_edge_frontier_channels(channels, current=current)
    return channels, tick, label


def _sample_post_claim_escape_map(
    rng: np.random.Generator,
    *,
    size: int,
) -> tuple[np.ndarray, int, str]:
    radius = size // 2
    channels = np.zeros((ONLINE_MAP_CHANNELS, size, size), dtype=np.float32)
    tick = int(rng.integers(20, 700))
    current = (radius, radius)
    visited: set[tuple[int, int]] = {current}
    blocked: set[tuple[int, int]] = set()
    attempted: set[tuple[int, int]] = set()

    label = str(
        rng.choice(
            [
                "forward_medium",
                "arc_left",
                "arc_right",
                "yaw_left",
                "yaw_right",
                "backward",
            ],
            p=[0.20, 0.24, 0.24, 0.07, 0.07, 0.18],
        )
    )
    primary_offsets = {
        "forward_medium": (-1, 0),
        "arc_left": (-1, 1),
        "arc_right": (-1, -1),
        "yaw_left": (0, 1),
        "yaw_right": (0, -1),
        "backward": (1, 0),
    }
    probe_offsets = {
        "forward_medium": [(-1, 0), (-2, 0)],
        "arc_left": [(-1, 1), (0, 1), (-1, 2)],
        "arc_right": [(-1, -1), (0, -1), (-1, -2)],
        "yaw_left": [(0, 1), (-1, 1), (1, 1)],
        "yaw_right": [(0, -1), (-1, -1), (1, -1)],
        "backward": [(1, 0), (2, 0)],
    }
    label_offsets = set(probe_offsets[label])
    label_primary = primary_offsets[label]
    claim_offset = (-label_primary[0], -label_primary[1])
    if claim_offset == (0, 0):
        claim_offset = (1, 0)
    claim_cell = (
        max(0, min(size - 1, radius + claim_offset[0])),
        max(0, min(size - 1, radius + claim_offset[1])),
    )
    visited.add(claim_cell)
    for step in range(2, int(rng.integers(3, min(6, radius + 1)))):
        trail = (
            max(0, min(size - 1, radius + claim_offset[0] * step)),
            max(0, min(size - 1, radius + claim_offset[1] * step)),
        )
        visited.add(trail)
        if rng.random() < 0.65:
            side = 1 if rng.random() < 0.5 else -1
            side_cell = (
                max(0, min(size - 1, trail[0] + side * claim_offset[1])),
                max(0, min(size - 1, trail[1] - side * claim_offset[0])),
            )
            blocked.add(side_cell)

    open_cells: set[tuple[int, int]] = set()
    for dr, dc in label_offsets:
        cell = (radius + dr, radius + dc)
        if _in_bounds(cell, size):
            open_cells.add(cell)

    for name, offsets in probe_offsets.items():
        if name == label:
            continue
        for dr, dc in offsets:
            cell = (radius + dr, radius + dc)
            if not _in_bounds(cell, size) or cell == current or cell in open_cells:
                continue
            if rng.random() < 0.55:
                blocked.add(cell)
            if rng.random() < 0.78:
                attempted.add(cell)

    # Add a few asymmetric guard-like cells near the non-selected directions so
    # the model learns to use attempted edges without overfitting to hard blocks.
    for _ in range(int(rng.integers(1, 5))):
        dr = int(rng.integers(-2, 3))
        dc = int(rng.integers(-2, 3))
        if abs(dr) + abs(dc) == 0 or (dr, dc) in label_offsets:
            continue
        cell = (radius + dr, radius + dc)
        if not _in_bounds(cell, size) or cell in open_cells or cell == current:
            continue
        if rng.random() < 0.35:
            blocked.add(cell)
        else:
            attempted.add(cell)

    blocked.discard(current)
    blocked.discard(claim_cell)
    for cell in open_cells:
        blocked.discard(cell)
        visited.discard(cell)
    for r, c in visited:
        channels[0, r, c] = 1.0
        age = 0.0 if (r, c) == current else float(rng.integers(0, 180))
        channels[2, r, c] = max(0.0, 1.0 - age / 160.0)
    for r, c in blocked:
        if (r, c) in visited:
            continue
        channels[1, r, c] = 1.0
        channels[0, r, c] = 0.0
        channels[2, r, c] = 0.0
    channels[0, radius, radius] = 1.0
    channels[2, radius, radius] = 1.0
    channels[3, claim_cell[0], claim_cell[1]] = 1.0
    _add_edge_frontier_channels(channels, current=current)
    for r, c in attempted:
        if _in_bounds((r, c), size) and channels[1, r, c] <= 0.5:
            channels[7, r, c] = 1.0
    return channels, tick, label


def _frontier_teacher_label(channels: np.ndarray, *, rng: np.random.Generator) -> str:
    edge_label = _edge_frontier_teacher_label(channels, rng=rng)
    if edge_label is not None:
        return edge_label

    ahead = _offset_cell(channels, (-1, 0))
    ahead_left = _offset_cell(channels, (-1, 1))
    ahead_right = _offset_cell(channels, (-1, -1))
    if ahead is not None and _unknown_clear(channels, *ahead) and rng.random() < 0.72:
        return "forward_medium"
    if ahead_left is not None and _unknown_clear(channels, *ahead_left) and rng.random() < 0.46:
        return "arc_left"
    if ahead_right is not None and _unknown_clear(channels, *ahead_right) and rng.random() < 0.46:
        return "arc_right"

    candidates = (
        ("forward_medium", ((-1, 0), (-2, 0)), 0.72),
        ("arc_left", ((-1, 1), (-2, 1), (-1, 2)), 0.42),
        ("arc_right", ((-1, -1), (-2, -1), (-1, -2)), 0.42),
        ("yaw_left", ((0, 1), (-1, 1), (1, 1), (0, 2)), -0.20),
        ("yaw_right", ((0, -1), (-1, -1), (1, -1), (0, -2)), -0.20),
        ("backward", ((1, 0), (2, 0)), -0.72),
    )
    scored: list[tuple[float, str]] = []
    for name, offsets, bias in candidates:
        score = float(bias)
        blocked_hits = 0
        unknown_hits = 0
        stale_hits = 0.0
        frontier_hits = 0
        valid_hits = 0
        for offset in offsets:
            cell = _offset_cell(channels, offset)
            if cell is None:
                score -= 0.8
                continue
            valid_hits += 1
            r, c = cell
            blocked = bool(channels[1, r, c] > 0.5)
            visited = bool(channels[0, r, c] > 0.5)
            claimed = bool(channels[3, r, c] > 0.5)
            if blocked:
                blocked_hits += 1
                score -= 1.4
                continue
            if not visited:
                unknown_hits += 1
                score += 1.0
            else:
                stale = 1.0 - float(channels[2, r, c])
                stale_hits += stale
                score += 0.28 * stale
            score += 0.18 * _unknown_cardinal_neighbors(channels, r, c)
            frontier_hits += _unknown_cardinal_neighbors(channels, r, c)
            if claimed:
                score -= 0.35
        if valid_hits > 0:
            score += 0.04 * float(frontier_hits)
        if blocked_hits and name in {"forward_medium", "arc_left", "arc_right"}:
            score -= 0.8 * float(blocked_hits)
        if unknown_hits == 0 and stale_hits < 0.6 and name in {"forward_medium", "arc_left", "arc_right"}:
            score -= 0.35
        score += float(rng.normal(0.0, 0.025))
        score += _directional_frontier_score(channels, name)
        scored.append((score, name))
    scored.sort(reverse=True)
    best = scored[0][1]
    if best == "forward_medium" and scored[0][0] < -0.10:
        left = next(score for score, name in scored if name == "yaw_left")
        right = next(score for score, name in scored if name == "yaw_right")
        best = "yaw_left" if left >= right else "yaw_right"
    return best


def _add_edge_frontier_channels(
    channels: np.ndarray,
    *,
    current: tuple[int, int],
) -> None:
    size = int(channels.shape[-1])
    visited = {
        (r, c)
        for r in range(size)
        for c in range(size)
        if channels[0, r, c] > 0.5 and channels[1, r, c] <= 0.5
    }
    blocked = {
        (r, c)
        for r in range(size)
        for c in range(size)
        if channels[1, r, c] > 0.5
    }
    claimed = {
        (r, c)
        for r in range(size)
        for c in range(size)
        if channels[3, r, c] > 0.5
    }
    connected = _connected_visited(visited, blocked, current=current, size=size)
    frontiers: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    for cell in connected:
        cx, cy = cell
        local_targets: list[tuple[int, int]] = []
        for neighbor in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if not _in_bounds(neighbor, size):
                continue
            if neighbor not in visited and neighbor not in blocked and neighbor not in claimed:
                local_targets.append(neighbor)
        if local_targets:
            frontiers.append(cell)
            targets.extend(local_targets)
    path = _path_to_nearest_frontier(connected, blocked, frontiers, current=current, size=size)
    for r, c in frontiers:
        channels[4, r, c] = 1.0
    for r, c in path:
        channels[5, r, c] = 1.0
    for r, c in targets:
        channels[6, r, c] = 1.0
    attempted_prob = 0.10
    for r, c in visited:
        if (r, c) == current:
            continue
        if channels[5, r, c] <= 0.5 and np.random.default_rng(r * 8191 + c * 131 + size).random() < attempted_prob:
            channels[7, r, c] = 1.0


def _connected_visited(
    visited: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    *,
    current: tuple[int, int],
    size: int,
) -> set[tuple[int, int]]:
    if current not in visited or current in blocked:
        return set()
    queue = [current]
    seen = {current}
    for cell in queue:
        r, c = cell
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                neighbor in seen
                or not _in_bounds(neighbor, size)
                or neighbor not in visited
                or neighbor in blocked
            ):
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def _path_to_nearest_frontier(
    visited: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    frontiers: list[tuple[int, int]],
    *,
    current: tuple[int, int],
    size: int,
) -> list[tuple[int, int]]:
    if current not in visited or current in blocked or not frontiers:
        return []
    frontier_set = set(frontiers)
    queue = [current]
    seen = {current}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {current: None}
    for cell in queue:
        if cell in frontier_set:
            out: list[tuple[int, int]] = []
            cursor: tuple[int, int] | None = cell
            while cursor is not None:
                out.append(cursor)
                cursor = parent.get(cursor)
            out.reverse()
            return out
        r, c = cell
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                neighbor in seen
                or not _in_bounds(neighbor, size)
                or neighbor not in visited
                or neighbor in blocked
            ):
                continue
            seen.add(neighbor)
            parent[neighbor] = cell
            queue.append(neighbor)
    return []


def _edge_frontier_teacher_label(
    channels: np.ndarray,
    *,
    rng: np.random.Generator,
) -> str | None:
    radius = int(channels.shape[-1]) // 2
    path_cells = _channel_cells(channels, 5)
    target_cells = _channel_cells(channels, 6)
    current = (radius, radius)
    if current in path_cells and len(path_cells) > 1:
        candidates = [
            cell for cell in path_cells
            if abs(cell[0] - radius) + abs(cell[1] - radius) == 1
        ]
        if candidates:
            cell = min(candidates, key=lambda item: (item[0] - radius) ** 2 + (item[1] - radius) ** 2)
            return _label_toward_cell(cell, radius=radius)
    near_targets = [
        cell for cell in target_cells
        if abs(cell[0] - radius) + abs(cell[1] - radius) <= 2
    ]
    if near_targets:
        near_targets.sort(key=lambda item: (item[0] - radius) ** 2 + (item[1] - radius) ** 2)
        if rng.random() < 0.92:
            return _label_toward_cell(near_targets[0], radius=radius)
    if path_cells:
        path_cells.sort(key=lambda item: (item[0] - radius) ** 2 + (item[1] - radius) ** 2)
        return _label_toward_cell(path_cells[0], radius=radius)
    return None


def _channel_cells(channels: np.ndarray, channel: int) -> list[tuple[int, int]]:
    size = int(channels.shape[-1])
    return [
        (r, c)
        for r in range(size)
        for c in range(size)
        if channels[int(channel), r, c] > 0.5
    ]


def _label_toward_cell(cell: tuple[int, int], *, radius: int) -> str:
    dr = int(cell[0]) - int(radius)
    dc = int(cell[1]) - int(radius)
    if dr < 0 and abs(dc) <= 1:
        if dc > 0:
            return "arc_left"
        if dc < 0:
            return "arc_right"
        return "forward_medium"
    if dr < 0 and dc > 1:
        return "yaw_left"
    if dr < 0 and dc < -1:
        return "yaw_right"
    if dc > 0:
        return "yaw_left"
    if dc < 0:
        return "yaw_right"
    if dr > 0:
        return "backward"
    return "forward_medium"


def _unknown_clear(channels: np.ndarray, row: int, col: int) -> bool:
    return bool(channels[0, row, col] <= 0.5 and channels[1, row, col] <= 0.5)


def _directional_frontier_score(channels: np.ndarray, primitive: str) -> float:
    direction = {
        "forward_medium": (-1.0, 0.0),
        "arc_left": (-0.85, 0.55),
        "arc_right": (-0.85, -0.55),
        "yaw_left": (-0.20, 1.0),
        "yaw_right": (-0.20, -1.0),
        "backward": (1.0, 0.0),
    }.get(primitive)
    if direction is None:
        return 0.0
    size = int(channels.shape[-1])
    radius = size // 2
    dr, dc = direction
    norm = max(1e-6, float(np.hypot(dr, dc)))
    dr /= norm
    dc /= norm
    score = 0.0
    weight_sum = 0.0
    for row in range(size):
        for col in range(size):
            if row == radius and col == radius:
                continue
            rel_r = float(row - radius)
            rel_c = float(col - radius)
            dist = max(1.0, float(np.hypot(rel_r, rel_c)))
            proj = (rel_r * dr + rel_c * dc) / dist
            if proj <= 0.15:
                continue
            lateral = abs(rel_r * dc - rel_c * dr)
            if lateral > max(1.25, 0.45 * dist + 0.75):
                continue
            blocked = bool(channels[1, row, col] > 0.5)
            if blocked:
                score -= 0.10 * proj / dist
                continue
            visited = bool(channels[0, row, col] > 0.5)
            claimed = bool(channels[3, row, col] > 0.5)
            value = 0.0
            if not visited:
                value += 1.0
            else:
                value += 0.25 * (1.0 - float(channels[2, row, col]))
            value += 0.10 * float(_unknown_cardinal_neighbors(channels, row, col))
            if claimed:
                value -= 0.3
            weight = proj / np.sqrt(dist)
            score += float(weight * value)
            weight_sum += float(weight)
    if weight_sum <= 0.0:
        return 0.0
    scale = 1.35 if primitive in {"forward_medium", "arc_left", "arc_right"} else 0.95
    if primitive == "backward":
        scale = 0.35
    return float(scale * score / weight_sum)


def _offset_cell(channels: np.ndarray, offset: tuple[int, int]) -> tuple[int, int] | None:
    size = int(channels.shape[-1])
    radius = size // 2
    row = radius + int(offset[0])
    col = radius + int(offset[1])
    if 0 <= row < size and 0 <= col < size:
        return row, col
    return None


def _unknown_cardinal_neighbors(channels: np.ndarray, row: int, col: int) -> int:
    count = 0
    size = int(channels.shape[-1])
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r = row + dr
        c = col + dc
        if not (0 <= r < size and 0 <= c < size):
            continue
        if channels[0, r, c] <= 0.5 and channels[1, r, c] <= 0.5:
            count += 1
    return count


def _encode_clock_state_map(
    row: np.ndarray,
    *,
    base_dim: int,
    state: str,
    map_channels: np.ndarray,
    tick: int,
    max_tick: int,
) -> None:
    cursor = int(base_dim)
    denom = max(1.0, float(max_tick))
    row[cursor : cursor + 3] = np.asarray(
        [float(tick) / denom, float(tick) / denom, 0.0],
        dtype=np.float32,
    )
    cursor += 3
    state_idx = STATE_FEATURES.index(state)
    row[cursor + state_idx] = 1.0
    cursor += len(STATE_FEATURES)
    row[cursor:] = map_channels.reshape(-1).astype(np.float32)


def _in_bounds(cell: tuple[int, int], size: int) -> bool:
    return 0 <= cell[0] < size and 0 <= cell[1] < size


if __name__ == "__main__":
    raise SystemExit(main())
