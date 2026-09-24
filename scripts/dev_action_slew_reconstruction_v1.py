#!/usr/bin/env python3
"""Deterministic post-slew command-trajectory reconstruction for the Go2 corpus.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The corpus logs a *requested* command block (5 ticks x 0.1 s) and, separately, an
*executed* (post-limiter) block.  40 % of executed blocks differ from what was
requested, and the deviation lands almost entirely on the first tick -- the tick
at which the visual change begins.  The nominal primitive triple therefore
misstates the action the robot actually applied.

This module reconstructs the executed trajectory **deterministically** from
information a planner also has at inference time:

    applied[k] = prev + clip(requested[k] - prev, -rate, +rate)
    prev <- applied[k]

with ``prev`` carried across block boundaries and initialised to zero at an
episode reset.  Nothing here reads measured body motion: the reconstruction is a
function of the requested command and the previous *applied command* only, so the
identical function serves hypothetical planning actions.

Rates were inferred from logged post-limiter values (max observed tick-to-tick
delta per channel) and then verified by exact reconstruction; see
``validate()`` and the accompanying report.

``vy`` is identically zero throughout the corpus, so its rate is not identifiable
from data; VY_RATE is set equal to VX_RATE and is inert for every logged block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

TICKS = 5
TICK_DT_S = 0.1
VX_RATE = 0.25          # m/s per 0.1 s tick   (identified, exact)
VY_RATE = 0.25          # not identifiable: vy is identically zero in the corpus
YAW_RATE = 0.35         # rad/s per 0.1 s tick (identified, exact)
RATES = (VX_RATE, VY_RATE, YAW_RATE)

RESET_APPLIED = (0.0, 0.0, 0.0)   # applied command immediately after a reset

# The nine primitives and their requested (vx, vy, yaw_rate) set-points.  A
# hypothetical planning action is named by its primitive; the requested block is
# that set-point held for all five ticks.
PRIMITIVES = ("arc_left", "arc_right", "backward", "forward_fast", "forward_medium",
              "forward_slow", "hold", "yaw_left", "yaw_right")
COMMANDS = {
    "arc_left": (0.20, 0.0, 0.45), "arc_right": (0.20, 0.0, -0.45),
    "backward": (-0.20, 0.0, 0.0), "forward_fast": (0.30, 0.0, 0.0),
    "forward_medium": (0.25, 0.0, 0.0), "forward_slow": (0.20, 0.0, 0.0),
    "hold": (0.0, 0.0, 0.0), "yaw_left": (0.0, 0.0, 0.45), "yaw_right": (0.0, 0.0, -0.45),
}

# The corpus contains no lateral command: requested and executed vy are
# identically zero in every block of every scene, and none of the nine
# primitives commands vy.  It is therefore a CONSTANT field and is excluded from
# the model-facing action, exactly as constant proprioceptive fields are.  The
# limiter arithmetic below keeps all three channels, so a future corpus with
# lateral commands needs no change here beyond widening ACTIVE_CHANNELS.
ACTIVE_CHANNELS = (0, 2)          # vx, yaw_rate
ACTIVE_CHANNEL_NAMES = ("vx_body_mps", "yaw_rate_radps")
ACTION_DIM = TICKS * len(ACTIVE_CHANNELS)   # 10


class LateralMotionRejected(ValueError):
    """A planning candidate or state carrying lateral motion: a hard failure."""


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def apply_slew(requested, previous_applied):
    """Post-limiter trajectory for one block.

    ``requested``        : (TICKS, 3) sequence of requested (vx, vy, yaw_rate)
    ``previous_applied`` : (3,) the applied command at the tick BEFORE this block
    returns              : (TICKS, 3) list of lists, and the final applied command
    """
    prev = list(previous_applied)
    out = []
    for tick in range(len(requested)):
        step = []
        for channel in range(3):
            target = requested[tick][channel]
            prev[channel] = prev[channel] + _clip(target - prev[channel], RATES[channel])
            step.append(prev[channel])
        out.append(step)
    return out, tuple(prev)


def reconstruct_block(primitive: str, previous_applied):
    """The planning-time entry point: primitive name -> post-slew trajectory.

    Identical arithmetic to ``apply_slew``; a hypothetical action is a set-point
    held for five ticks.  No measured body motion is consulted.
    """
    if primitive not in COMMANDS:
        raise LateralMotionRejected(f"unknown primitive {primitive!r}")
    setpoint = COMMANDS[primitive]
    if abs(setpoint[1]) > 0.0:
        raise LateralMotionRejected(
            f"planning candidate {primitive!r} commands lateral motion vy={setpoint[1]}; "
            "the contract forbids it and no training data supports it")
    if abs(previous_applied[1]) > 0.0:
        raise LateralMotionRejected(
            f"previous applied command carries lateral motion vy={previous_applied[1]}")
    return apply_slew([list(setpoint)] * TICKS, previous_applied)


def flatten(trajectory):
    """(TICKS, 3) -> (ACTION_DIM,), tick-major, active channels only."""
    return [tick[channel] for tick in trajectory for channel in ACTIVE_CHANNELS]


# --------------------------------------------------------------------------
def _iter_executed_blocks(path: Path):
    """Yield (env_index, timestamp, kind, payload) for blocks and reset events.

    Reset events matter: the limiter state returns to a standing command at a
    respawn, and an env can respawn mid-episode.  Carrying the previous applied
    command across a reset is the one way this reconstruction can go wrong.
    """
    block_needle = '"/lewm/go2/executed_command_block"'
    reset_needle = '"/lewm/go2/reset_event"'
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            has_block = block_needle in line
            if not has_block and reset_needle not in line:
                continue
            record = json.loads(line)
            topic = record.get("canonical_topic")
            if topic == "/lewm/go2/executed_command_block":
                yield record["env_index"], record["timestamp_ns"], "block", record["payload"]
            elif topic == "/lewm/go2/reset_event":
                yield record["env_index"], record["timestamp_ns"], "reset", record["payload"]


def validate_scene(path: Path) -> dict:
    """Reconstruct every logged block in one scene and compare tick by tick."""
    import collections
    events = collections.defaultdict(list)
    for env_index, timestamp, kind, payload in _iter_executed_blocks(path):
        events[env_index].append((timestamp, kind, payload))
    blocks = collections.defaultdict(list)
    resets = collections.defaultdict(list)
    for env_index, entries in events.items():
        entries.sort(key=lambda item: (item[0], item[1] != "reset"))
        for timestamp, kind, payload in entries:
            (blocks if kind == "block" else resets)[env_index].append((timestamp, payload))

    stats = {"blocks": 0, "ticks": 0, "block_exact": 0, "tick_exact": 0,
             "sign_reversal_blocks": 0, "sign_reversal_exact": 0,
             "first_block_after_reset": 0, "first_block_after_reset_exact": 0,
             "clipped_blocks": 0, "clipped_exact": 0, "mismatches": []}

    for env_index, sequence in blocks.items():
        sequence.sort(key=lambda item: item[0])
        reset_times = sorted(t for t, _ in resets.get(env_index, []))
        previous = RESET_APPLIED
        pending = list(reset_times)
        block_span_ns = int(TICKS * TICK_DT_S * 1e9)
        for position, (timestamp, payload) in enumerate(sequence):
            # a block is stamped at its END, so it covers [timestamp - span, timestamp);
            # a reset takes effect for the first block that STARTS at or after it.
            block_start = timestamp - block_span_ns
            while pending and pending[0] <= block_start:
                pending.pop(0)
                previous = RESET_APPLIED       # respawn: limiter restarts from stand
                stats["reset_restarts"] = stats.get("reset_restarts", 0) + 1
            requested = [[payload["requested_vx_body_mps"][t],
                          payload["requested_vy_body_mps"][t],
                          payload["requested_yaw_rate_radps"][t]] for t in range(TICKS)]
            logged = [[payload["executed_vx_body_mps"][t],
                       payload["executed_vy_body_mps"][t],
                       payload["executed_yaw_rate_radps"][t]] for t in range(TICKS)]
            predicted, previous = apply_slew(requested, previous)

            exact = all(abs(p - l) < 1e-6 for pr, lo in zip(predicted, logged)
                        for p, l in zip(pr, lo))
            ticks_exact = sum(1 for pr, lo in zip(predicted, logged)
                              if all(abs(p - l) < 1e-6 for p, l in zip(pr, lo)))
            # a sign reversal: a logged tick whose sign opposes the request
            reversal = any(l * r < -1e-9 for lo, rq in zip(logged, requested)
                           for l, r in zip(lo, rq))

            stats["blocks"] += 1
            stats["ticks"] += TICKS
            stats["tick_exact"] += ticks_exact
            stats["block_exact"] += int(exact)
            if reversal:
                stats["sign_reversal_blocks"] += 1
                stats["sign_reversal_exact"] += int(exact)
            if position == 0 or (reset_times and any(
                    sequence[position - 1][0] < r <= timestamp for r in reset_times)):
                stats["first_block_after_reset"] += 1
                stats["first_block_after_reset_exact"] += int(exact)
            if payload.get("clipped"):
                stats["clipped_blocks"] += 1
                stats["clipped_exact"] += int(exact)
            if not exact and len(stats["mismatches"]) < 5:
                stats["mismatches"].append(
                    {"env": env_index, "sequence_id": payload["sequence_id"],
                     "primitive": payload["primitive_name"],
                     "requested": requested, "logged": logged, "predicted": predicted})
    return stats


def _merge(total: dict, part: dict) -> dict:
    for key, value in part.items():
        if key == "mismatches":
            total[key] = (total.get(key, []) + value)[:10]
        else:
            total[key] = total.get(key, 0) + value
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="validate the slew reconstruction")
    ap.add_argument("--scenes-per-family", type=int, default=3)
    ap.add_argument("--scenes-file", default=None,
                    help="newline-separated scene ids; validates exactly these")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rollout = ROOT / ".generated/datagen_full/rollout"
    paths = []
    if args.scenes_file:
        wanted = [line.strip() for line in Path(args.scenes_file).read_text().splitlines()
                  if line.strip()]
        for scene in wanted:
            found = sorted(rollout.glob(f"*/*/chunk_*/raw/{scene}/messages.jsonl"))
            if not found:
                raise SystemExit(f"no messages.jsonl for scene {scene}")
            paths.append(found[0])
    else:
        for split in sorted(p.name for p in rollout.iterdir() if p.is_dir()):
            for family in sorted(p.name for p in (rollout / split).iterdir() if p.is_dir()):
                found = sorted((rollout / split / family).glob("chunk_*/raw/*/messages.jsonl"))
                paths.extend(found[: args.scenes_per_family])
    print(f"validating {len(paths)} scenes with {args.workers} workers", flush=True)

    from multiprocessing import Pool
    total = {}
    with Pool(args.workers) as pool:
        for index, stats in enumerate(pool.imap_unordered(validate_scene, paths), 1):
            total = _merge(total, stats)
            if index % 5 == 0:
                print(f"  {index}/{len(paths)} scenes", flush=True)

    report = {
        "status": STATUS, "claim_bearing": False,
        "reconstruction": "applied[k] = prev + clip(requested[k] - prev, +-rate)",
        "rates": {"vx": VX_RATE, "vy": VY_RATE, "yaw": YAW_RATE},
        "vy_note": "vy is identically zero in the corpus; its rate is not identifiable and is inert",
        "reset_applied": list(RESET_APPLIED),
        "inputs_used": ["requested command", "previous applied command"],
        "inputs_not_used": ["measured body motion", "future proprioception", "world pose"],
        "scenes": len(paths), "totals": total,
        "scene_ids": sorted(path.parent.name for path in paths),
    }
    for label, num, den in (("block_exact", "block_exact", "blocks"),
                            ("tick_exact", "tick_exact", "ticks"),
                            ("sign_reversal_exact", "sign_reversal_exact", "sign_reversal_blocks"),
                            ("post_reset_exact", "first_block_after_reset_exact", "first_block_after_reset"),
                            ("clipped_exact", "clipped_exact", "clipped_blocks")):
        d = total.get(den, 0)
        report[f"{label}_rate"] = (total.get(num, 0) / d) if d else None
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
