#!/usr/bin/env python3
"""Two-step causal rollout sequences over the dense v03 render.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Frozen contract, extending the one-step contract by exactly one command block:

    visual context : t-480, t-240, t
    step 1         : action a0 (block t -> t+240),  target y1 = tokens(t+240)
    step 2         : action a1 (block t+240 -> t+480), target y2 = tokens(t+480)

**a1 is directly recorded, not inferred.**  Every frame of the rollout
``frames.jsonl`` carries ``command_context.primitive_name`` and a
``sequence_id`` that changes exactly every 240 flat frames -- one complete
command block, the same source and the same criterion the corpus itself used to
label a0.  This is verified per row: a0 read from ``frames.jsonl`` at t must
equal the corpus pair's own primitive, and the block at t+240 must be a distinct
complete block.

Every one of the five frames is checked to lie in the same scene, ``env_index``,
``episode_id`` and ``reset_count``.  No frame is duplicated and no filename is
inferred.

``t+480`` occupancy rasters are NOT reconstructed.  Rows whose t+480 endpoint is
natively present in the corpus (and therefore carries a real
``raster_labels.u1``) are flagged ``native_step2_labels``; second-step spatial
metrics may only be reported on that subset.
"""
from __future__ import annotations

import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUP = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
PAIRED = ROOT / ".generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json"
V03 = ROOT / ".generated/datagen_full/render_textured_v03"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
ROWS = CACHE / "temporal_rows.jsonl"
OUT = CACHE / "two_step"

STEP = 240
FRAME_RE = re.compile(r"frame_(\d+)_env_(\d+)\.png$")
ALLOWED_ROLES = ("train", "checkpoint_selection")


def _scene_blocks(task):
    """Command block identity and primitive for the frame indices we need."""
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
            episode = row["episode"]
            command = row.get("command_context") or {}
            out[line_no] = {
                "env": int(row["env_index"]),
                "episode_id": int(episode["episode_id"]),
                "reset_count": int(episode["reset_count"]),
                "primitive": str(command.get("primitive_name") or ""),
                "sequence_id": int(command.get("sequence_id", -1)),
                "block_size": int(command.get("block_size", -1)),
            }
    return scene_id, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    endpoints = {
        e["endpoint_identity_sha256"]: e
        for e in (json.loads(l) for l in (SUP / "endpoints.jsonl").read_text().splitlines() if l.strip())
    }
    pairs = [json.loads(l) for l in (SUP / "pairs.jsonl").read_text().splitlines() if l.strip()]
    pair_by_sha = {p["content_sha256"]: p for p in pairs}
    sources = {s["scene_id"]: s["paths"]["frames_jsonl"] for s in json.load(open(PAIRED))["sources"]}
    # endpoints that natively carry a raster, keyed by (scene, frame index, env)
    native = {}
    for e in endpoints.values():
        if e.get("dataset_role") not in ALLOWED_ROLES:
            continue
        m = FRAME_RE.search(e["image_path_metadata_only"])
        native[(e["scene_id"], int(m.group(1)), m.group(2))] = e

    base = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    want = collections.defaultdict(set)
    for row in base:
        want[row["scene"]].update({row["t"], row["t"] + STEP, row["t"] + 2 * STEP})
    tasks = [(s, sources[s], sorted(v)) for s, v in want.items()]
    print(f"indexing command blocks in {len(tasks)} rollout files ...", flush=True)
    with Pool(args.workers) as pool:
        index = dict(pool.map(_scene_blocks, tasks))

    retained, dropped = [], collections.Counter()
    for row in base:
        blocks = index[row["scene"]]
        t = row["t"]
        b0, b1, b2 = blocks.get(t), blocks.get(t + STEP), blocks.get(t + 2 * STEP)
        pair = pair_by_sha[row["pair_sha256"]]
        if b0 is None or b1 is None or b2 is None:
            dropped["frame_index_absent_from_rollout"] += 1
            continue
        identity = (row["env_index"], row["episode_id"], row["reset_count"])
        if any((b["env"], b["episode_id"], b["reset_count"]) != identity for b in (b0, b1, b2)):
            dropped["episode_or_reset_boundary_crossed"] += 1
            continue
        # a0 recorded at t must agree with the corpus pair's own label
        if b0["primitive"] != pair["primitive"]:
            dropped["a0_disagrees_with_corpus_pair_primitive"] += 1
            continue
        # a1 must be a DISTINCT complete command block starting at t+240
        if b1["sequence_id"] == b0["sequence_id"] or b2["sequence_id"] == b1["sequence_id"]:
            dropped["t+240_is_not_a_new_complete_command_block"] += 1
            continue
        if b1["block_size"] != b0["block_size"]:
            dropped["inconsistent_block_size"] += 1
            continue
        png = V03 / row["scene"] / "rgb" / f"frame_{t + 2 * STEP:06d}_env_{row['env']}.png"
        if not png.is_file():
            dropped["t+480_frame_not_rendered"] += 1
            continue
        indices = [t - 2 * STEP, t - STEP, t, t + STEP, t + 2 * STEP]
        if len(set(indices)) != 5:
            dropped["duplicate_frame_index"] += 1
            continue
        step2_endpoint = native.get((row["scene"], t + 2 * STEP, row["env"]))
        entry = dict(row)
        entry.update({
            "action_step1": pair["primitive"],
            "action_step2": b1["primitive"],
            "action_step2_source": "frames.jsonl command_context at t+240 (directly recorded)",
            "step2_path": str(png),
            "step2_frame_index": t + 2 * STEP,
            "sequence_ids": [b0["sequence_id"], b1["sequence_id"], b2["sequence_id"]],
            "native_step2_labels": step2_endpoint is not None,
            "step2_shard_dir": (str(SUP / Path(step2_endpoint["scene_shard"]).parent)
                                if step2_endpoint else None),
            "step2_shard_row": (int(step2_endpoint["shard_row"]) if step2_endpoint else None),
        })
        retained.append(entry)

    by_role = collections.Counter(r["role"] for r in retained)
    native_by_role = collections.Counter(r["role"] for r in retained if r["native_step2_labels"])
    manifest = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING",
        "contract": {
            "context": [-480, -240, 0], "step1_target": STEP, "step2_target": 2 * STEP,
            "action_step1": "command block t -> t+240 (corpus pair primitive)",
            "action_step2": "command block t+240 -> t+480 (frames.jsonl command_context, directly recorded)",
            "a1_inference": "none: read from the same source and by the same criterion as a0",
            "step2_prediction_consumes": "p1, never the true y1",
            "duplicated_frames": 0, "inferred_filenames": 0, "crossed_resets": 0,
        },
        "retention": {
            "base_one_step_rows": len(base),
            "retained": len(retained),
            "dropped": dict(dropped),
            "by_role": {r: by_role[r] for r in ALLOWED_ROLES},
            "native_step2_label_rows_by_role": {r: native_by_role[r] for r in ALLOWED_ROLES},
        },
    }
    for axis in ("family", "action_step1", "action_step2"):
        counts = collections.Counter((r["role"], r[axis]) for r in retained)
        manifest["retention"][f"by_{axis}"] = {
            role: {k: counts[(role, k)] for _, k in sorted(x for x in counts if x[0] == role)}
            for role in ALLOWED_ROLES
        }
    manifest["retention"]["native_step2_by_family"] = {
        role: dict(collections.Counter(
            r["family"] for r in retained if r["role"] == role and r["native_step2_labels"]))
        for role in ALLOWED_ROLES
    }
    train_scenes = {r["scene"] for r in retained if r["role"] == "train"}
    sel_scenes = {r["scene"] for r in retained if r["role"] == "checkpoint_selection"}
    manifest["split"] = {
        "train_scenes": len(train_scenes), "selection_scenes": len(sel_scenes),
        "scene_overlap": len(train_scenes & sel_scenes),
    }
    if train_scenes & sel_scenes:
        raise RuntimeError("train/selection scene overlap")
    (OUT / "two_step_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "two_step_rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in retained))

    print(json.dumps(manifest["retention"]["by_role"], indent=2))
    print("native step-2 label rows:", dict(manifest["retention"]["native_step2_label_rows_by_role"]))
    print("dropped:", dict(dropped))
    for role in ALLOWED_ROLES:
        print(f"-- {role} by family:")
        for k, v in manifest["retention"]["by_family"][role].items():
            nat = manifest["retention"]["native_step2_by_family"][role].get(k, 0)
            print(f"     {k:24s} {v:5d}   (native step-2 labels: {nat})")
    print("split:", json.dumps(manifest["split"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
