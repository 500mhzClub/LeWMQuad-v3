#!/usr/bin/env python3
"""Build the causal temporal sequence manifest over the dense v03 render.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Frozen contract, one sequence per retained WP-E pair:

    context frames = t-480, t-240, t     (v03, centre-cropped 224x224 -> 224x168)
    future target  = t+240
    action         = the command block executed from t to t+240

Every frame is verified against the rollout ``frames.jsonl`` to lie in the same
scene, ``env_index``, ``episode_id`` and ``reset_count`` as the pair.  Nothing is
duplicated, no filename is inferred, and no reset is crossed: a sequence is
retained only when all four frames are present on disk *and* their episode
identity matches exactly.

Derived output goes to the root filesystem -- the workspace pool is full.
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
OUT = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")

CONTEXT_OFFSETS = (-480, -240, 0)     # frozen
TARGET_OFFSET = 240                   # frozen
ALLOWED_ROLES = ("train", "checkpoint_selection")
FRAME_RE = re.compile(r"frame_(\d+)_env_(\d+)\.png$")


def _scene_index(task):
    """Extract episode identity and primitive for the frame indices we need."""
    scene_id, frames_jsonl, needed = task
    needed = set(needed)
    out = {}
    with open(frames_jsonl, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream):
            if line_no not in needed:
                continue
            row = json.loads(line)
            if int(row["frame_index"]) != line_no:
                raise RuntimeError(
                    f"{scene_id}: frames.jsonl line {line_no} carries frame_index "
                    f"{row['frame_index']}; positional indexing is invalid"
                )
            episode = row["episode"]
            out[line_no] = (
                int(row["env_index"]),
                int(episode["episode_id"]),
                int(episode["reset_count"]),
                str((row.get("command_context") or {}).get("primitive_name") or ""),
            )
    return scene_id, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    endpoints = {
        e["endpoint_identity_sha256"]: e
        for e in (
            json.loads(l)
            for l in (SUP / "endpoints.jsonl").read_text().splitlines()
            if l.strip()
        )
    }
    pairs = [
        p for p in (json.loads(l) for l in (SUP / "pairs.jsonl").read_text().splitlines() if l.strip())
        if p["dataset_role"] in ALLOWED_ROLES
    ]
    sources = {
        s["scene_id"]: s["paths"]["frames_jsonl"]
        for s in json.load(open(PAIRED))["sources"]
    }

    # what each pair needs, before any verification
    want: dict[str, set[int]] = collections.defaultdict(set)
    staged = []
    for pair in pairs:
        current = endpoints[pair["current_endpoint_sha256"]]
        nxt = endpoints[pair["next_endpoint_sha256"]]
        m = FRAME_RE.search(current["image_path_metadata_only"])
        t, env = int(m.group(1)), m.group(2)
        mn = FRAME_RE.search(nxt["image_path_metadata_only"])
        if int(mn.group(1)) != t + TARGET_OFFSET or mn.group(2) != env:
            raise RuntimeError(f"pair violates the +240 same-env contract: {pair['content_sha256']}")
        indices = [t + o for o in CONTEXT_OFFSETS] + [t + TARGET_OFFSET]
        want[pair["scene_id"]].update(indices)
        staged.append(
            {
                "pair_sha256": pair["content_sha256"],
                "scene": pair["scene_id"],
                "family": pair["family"],
                "role": pair["dataset_role"],
                "primitive": pair["primitive"],
                "env": env,
                "t": t,
                "indices": indices,
                "episode_id": int(pair["episode_id"]),
                "reset_count": int(pair["reset_count"]),
                "env_index": int(pair["env_index"]),
                "relative_se2_current_frame": pair["relative_se2_current_frame"],
                "shard_dir": str(SUP / Path(current["scene_shard"]).parent),
                "shard_row": int(current["shard_row"]),
                "endpoint_sha256": current["endpoint_identity_sha256"],
                "raster_content_sha256": current["raster_content_sha256"],
            }
        )

    tasks = [(s, sources[s], sorted(v)) for s, v in want.items()]
    print(f"indexing {len(tasks)} rollout frames.jsonl files ...", flush=True)
    with Pool(args.workers) as pool:
        index = dict(pool.map(_scene_index, tasks))

    retained, dropped = [], collections.Counter()
    for row in staged:
        scene_index = index[row["scene"]]
        frames, ok = [], True
        for offset, idx in zip(CONTEXT_OFFSETS + (TARGET_OFFSET,), row["indices"]):
            meta = scene_index.get(idx)
            if meta is None:
                dropped["frame_index_absent_from_rollout"] += 1
                ok = False
                break
            env_index, episode_id, reset_count, primitive = meta
            if (env_index, episode_id, reset_count) != (
                row["env_index"], row["episode_id"], row["reset_count"]
            ):
                dropped["episode_or_reset_boundary_crossed"] += 1
                ok = False
                break
            png = V03 / row["scene"] / "rgb" / f"frame_{idx:06d}_env_{row['env']}.png"
            if not png.is_file():
                dropped["v03_frame_not_rendered"] += 1
                ok = False
                break
            frames.append({"offset": offset, "frame_index": idx, "path": str(png),
                           "primitive_at_frame": primitive})
        if not ok:
            continue
        if len({f["frame_index"] for f in frames}) != 4:
            dropped["duplicate_frame_index"] += 1
            continue
        row = dict(row)
        row["frames"] = frames
        row["context_paths"] = [f["path"] for f in frames[:3]]
        row["target_path"] = frames[3]["path"]
        retained.append(row)

    by_role = collections.Counter(r["role"] for r in retained)
    staged_by_role = collections.Counter(r["role"] for r in staged)
    manifest = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING",
        "temporal_contract": {
            "context_offsets_frames": list(CONTEXT_OFFSETS),
            "target_offset_frames": TARGET_OFFSET,
            "frames_per_timestep": 48,
            "timesteps_per_offset": TARGET_OFFSET // 48,
            "command_block": {"block_size": 5, "command_dt_s": 0.1, "duration_s": 0.5},
            "action": "the command block executed from t to t+240",
            "image_source": str(V03),
            "visuals": "textured_v03",
            "native_wh": [224, 224],
            "preprocessing": "centre-crop rows to 224x168, then the official V-JEPA 2.1 path",
            "mixing_textured_v03_and_v04_within_a_sequence": "forbidden; every frame is v03",
            "duplicated_frames": 0,
            "inferred_filenames": 0,
            "crossed_resets": 0,
        },
        "retention": {
            "staged": len(staged),
            "retained": len(retained),
            "dropped": dict(dropped),
            "by_role": {
                role: {"staged": staged_by_role[role], "retained": by_role[role]}
                for role in ALLOWED_ROLES
            },
        },
    }
    for axis in ("family", "primitive", "scene"):
        st = collections.Counter((r["role"], r[axis]) for r in staged)
        rt = collections.Counter((r["role"], r[axis]) for r in retained)
        manifest["retention"][f"by_{axis}"] = {
            role: {
                key: {"staged": st[(role, key)], "retained": rt[(role, key)]}
                for _, key in sorted(k for k in st if k[0] == role)
            }
            for role in ALLOWED_ROLES
        }
    (OUT / "temporal_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "temporal_rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in retained)
    )

    print(json.dumps(manifest["retention"]["by_role"], indent=2))
    print("dropped:", dict(dropped))
    for role in ALLOWED_ROLES:
        fam = manifest["retention"]["by_family"][role]
        print(f"-- {role} by family:")
        for k, v in fam.items():
            print(f"     {k:24s} {v['retained']:5d} / {v['staged']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
