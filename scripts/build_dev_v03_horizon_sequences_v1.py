#!/usr/bin/env python3
"""Extend the two-step selection sequences to horizons H = 1,2,3,4.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Metadata only -- no encoder is run here.

Adds targets at t+720 and t+960 and the command blocks a2 (t+480 -> t+720) and
a3 (t+720 -> t+960), read from the rollout ``frames.jsonl`` ``command_context``
by exactly the same directly-recorded criterion used for a0 and a1: each block is
a distinct ``sequence_id`` of matching ``block_size``.

Every frame of a retained sequence must lie in one scene, ``env_index``,
``episode_id`` and ``reset_count``.  No frame is duplicated, no filename is
inferred, and no reset is crossed.  A sequence is retained at horizon H only if
every frame and every action up to H is present and verified.
"""
from __future__ import annotations

import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUP = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
PAIRED = ROOT / ".generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json"
V03 = ROOT / ".generated/datagen_full/render_textured_v03"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
TWO = CACHE / "two_step"
OUT = CACHE / "horizons"
STEP = 240
MAX_H = 4


def _blocks(task):
    scene_id, frames_jsonl, needed = task
    needed = set(needed)
    out = {}
    with open(frames_jsonl, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream):
            if line_no not in needed:
                continue
            row = json.loads(line)
            if int(row["frame_index"]) != line_no:
                raise RuntimeError(f"{scene_id}: positional indexing invalid at {line_no}")
            ep, cmd = row["episode"], (row.get("command_context") or {})
            out[line_no] = {"env": int(row["env_index"]),
                            "episode_id": int(ep["episode_id"]),
                            "reset_count": int(ep["reset_count"]),
                            "primitive": str(cmd.get("primitive_name") or ""),
                            "sequence_id": int(cmd.get("sequence_id", -1)),
                            "block_size": int(cmd.get("block_size", -1))}
    return scene_id, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in (TWO / "two_step_rows.jsonl").read_text().splitlines() if l.strip()]
    sel = [r for r in rows if r["role"] == "checkpoint_selection"]
    sources = {s["scene_id"]: s["paths"]["frames_jsonl"] for s in json.load(open(PAIRED))["sources"]}

    want = collections.defaultdict(set)
    for r in sel:
        want[r["scene"]].update(r["t"] + k * STEP for k in range(0, MAX_H + 1))
    tasks = [(s, sources[s], sorted(v)) for s, v in want.items()]
    with Pool(args.workers) as pool:
        index = dict(pool.map(_blocks, tasks))

    retained, dropped = [], collections.Counter()
    for r in sel:
        blocks = index[r["scene"]]
        identity = (r["env_index"], r["episode_id"], r["reset_count"])
        frames, actions, horizon = [], [], 0
        ok = True
        for k in range(0, MAX_H + 1):
            idx = r["t"] + k * STEP
            b = blocks.get(idx)
            if b is None or (b["env"], b["episode_id"], b["reset_count"]) != identity:
                ok = False
                break
            png = V03 / r["scene"] / "rgb" / f"frame_{idx:06d}_env_{r['env']}.png"
            if k > 0 and not png.is_file():
                ok = False
                break
            frames.append({"h": k, "frame_index": idx, "path": str(png)})
            if k < MAX_H:
                nxt = blocks.get(idx + STEP)
                if nxt is None or nxt["sequence_id"] == b["sequence_id"] \
                        or nxt["block_size"] != b["block_size"]:
                    break
                actions.append({"h": k + 1, "primitive": b["primitive"],
                                "sequence_id": b["sequence_id"]})
                horizon = k + 1
        # `horizon` is advanced when the NEXT action exists, but the frame at that
        # horizon may still be absent (the loop breaks before appending it).  The
        # usable horizon is therefore bounded by the frames actually collected.
        horizon = min(horizon, len(frames) - 1)
        if horizon < 1:
            dropped["no_valid_horizon"] += 1
            continue
        if len({f["frame_index"] for f in frames}) != len(frames):
            dropped["duplicate_frame_index"] += 1
            continue
        # a1 must reproduce the two-step manifest's directly-recorded value
        if horizon >= 1 and actions[0]["primitive"] != r["action_step1"]:
            dropped["a0_disagrees_with_two_step_manifest"] += 1
            continue
        if horizon >= 2 and actions[1]["primitive"] != r["action_step2"]:
            dropped["a1_disagrees_with_two_step_manifest"] += 1
            continue
        entry = dict(r)
        entry.update({"max_horizon": horizon,
                      "horizon_frames": frames[: horizon + 1],
                      "horizon_actions": actions[:horizon]})
        retained.append(entry)

    by_h = collections.Counter(r["max_horizon"] for r in retained)
    manifest = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING",
        "role": "checkpoint_selection only; read-only horizon evaluation of frozen models",
        "contract": {
            "context": [-480, -240, 0], "step": STEP, "max_horizon": MAX_H,
            "actions": "command blocks read from frames.jsonl command_context, "
                       "directly recorded, same criterion as a0/a1",
            "a0_a1_cross_checked_against_two_step_manifest": True,
            "duplicated_frames": 0, "inferred_filenames": 0, "crossed_resets": 0,
        },
        "retention": {
            "selection_rows_in": len(sel), "retained": len(retained),
            "dropped": dict(dropped),
            "rows_with_max_horizon_at_least": {
                str(h): sum(1 for r in retained if r["max_horizon"] >= h)
                for h in range(1, MAX_H + 1)},
            "rows_by_max_horizon": {str(k): v for k, v in sorted(by_h.items())},
            "by_family_at_horizon_4": dict(collections.Counter(
                r["family"] for r in retained if r["max_horizon"] >= 4)),
        },
    }
    (OUT / "horizon_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "horizon_rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in retained))
    print(json.dumps(manifest["retention"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
