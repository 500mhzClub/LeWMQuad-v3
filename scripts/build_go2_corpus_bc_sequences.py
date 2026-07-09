#!/usr/bin/env python3
"""Build broad-explorer BC sequences from the mass datagen corpus.

Per scene/env stream: frozen-JEPA latents of the rendered ego frames plus the
same nonprivileged proprioceptive features the runtime already buffers
(build_go2_proprio_contact_dataset._tick_features contract), labeled with the
collector's executed primitive. Rows are filtered to sweeping collector
sources (route_teacher, frontier) so the head imitates movement, not noise.

Output: per-scene NPZ shards (latents (T,D), proprio (T,F), labels (T,),
sources (T,), ticks (T,)) consumable by train_go2_broad_explorer_bc.py.
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

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
PRIM_INDEX = {p: i for i, p in enumerate(PRIMITIVES)}


def _scene_plans(rollout_root: Path, family: str, split: str) -> list[Path]:
    out = []
    base = rollout_root / split / family
    for chunk in sorted(base.glob("chunk_*")):
        plan_dir = chunk / "plan"
        if plan_dir.is_dir():
            for scene_plan in sorted(plan_dir.iterdir()):
                if (scene_plan / "frames.jsonl").is_file():
                    out.append(scene_plan)
    return out


def _prep_scene(job: tuple[str, str, str, int, tuple[str, ...]]) -> dict | None:
    """CPU part: parse frames.jsonl, compute proprio features + labels."""
    plan_dir_s, render_root_s, _out_dir_s, envs_per_scene, sources = job
    plan_dir = Path(plan_dir_s)
    scene_id = plan_dir.name.split("_", 1)[1]
    rgb_dir = Path(render_root_s) / scene_id / "rgb"
    if not rgb_dir.is_dir():
        return None
    per_env: dict[int, list[dict]] = {}
    with (plan_dir / "frames.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            env = int(rec["env_index"])
            if env >= envs_per_scene:
                continue
            per_env.setdefault(env, []).append(rec)
    frames: list[str] = []
    proprio: list[list[float]] = []
    labels: list[int] = []
    srcs: list[str] = []
    ticks: list[int] = []
    for env, recs in sorted(per_env.items()):
        recs.sort(key=lambda r: int(r["timestamp_ns"]))
        prev = None
        for rec in recs:
            cc = rec.get("command_context") or {}
            prim = str(cc.get("primitive_name") or "")
            src = str(cc.get("command_source") or "")
            pose = rec["base_pose_world"]["position"]
            rpy = rec["base_rpy_rad"]
            x, y = float(pose["x"]), float(pose["y"])
            yaw = float(rpy["yaw"])
            z = float(pose.get("z", 0.34))
            roll, pitch = float(rpy.get("roll", 0.0)), float(rpy.get("pitch", 0.0))
            episode = rec.get("episode") or {}
            ep = int(episode.get("episode_id", 0))
            if prev is not None and prev[4] == ep:
                disp = math.hypot(x - prev[0], y - prev[1])
                dyaw = math.atan2(math.sin(yaw - prev[2]), math.cos(yaw - prev[2]))
                dz = z - prev[3]
            else:
                disp, dyaw, dz = 0.0, 0.0, 0.0
            prev = (x, y, yaw, z, ep)
            if prim not in PRIM_INDEX:
                continue
            import sys as _sys
            if str(REPO_ROOT / "scripts") not in _sys.path:
                _sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from build_go2_proprio_contact_dataset import NOMINAL_ABS_DYAW, NOMINAL_DISP_M

            feat = [0.0] * (9 + len(PRIMITIVES) + 1)
            feat[0] = min(0.3, max(0.0, disp))
            feat[1] = max(-0.6, min(0.6, dyaw))
            feat[2] = max(-0.8, min(0.8, roll))
            feat[3] = max(-0.8, min(0.8, pitch))
            feat[4] = max(-0.1, min(0.1, dz))
            feat[5] = 0.0
            feat[6] = min(1.0, max(abs(roll), abs(pitch)))
            feat[7] = float(NOMINAL_DISP_M.get(prim, 0.05)) - feat[0]
            feat[8] = float(NOMINAL_ABS_DYAW.get(prim, 0.05)) - abs(feat[1])
            feat[9 + PRIM_INDEX[prim]] = 1.0
            frame = rgb_dir / f"frame_{int(rec['frame_index']):06d}_env_{env:02d}.png"
            frames.append(str(frame))
            proprio.append(feat)
            labels.append(PRIM_INDEX[prim])
            srcs.append(src)
            ticks.append(int(rec["frame_index"]))
    keep = [i for i, s in enumerate(srcs) if s in sources]
    if len(keep) < 64:
        return None
    return {
        "scene_id": scene_id,
        "frames": frames,
        "proprio": proprio,
        "labels": labels,
        "sources": srcs,
        "ticks": ticks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--envs-per-scene", type=int, default=2)
    parser.add_argument("--sources", default="route_teacher,frontier")
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from lewm.models.go2_jepa import load_go2_jepa_encoder
    from train_go2_hidden_target_memory_probe import _load_image, _resolve_device

    device = _resolve_device(str(args.device))
    encoder, encoder_ck = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint, device=device, freeze=True
    )
    encoder.eval()

    sources = tuple(s.strip() for s in str(args.sources).split(",") if s.strip())
    plans = _scene_plans(args.rollout_root, str(args.family), str(args.split))
    if args.max_scenes is not None:
        plans = plans[: int(args.max_scenes)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (str(p), str(args.render_root), str(args.output_dir), int(args.envs_per_scene), sources)
        for p in plans
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for prep in pool.map(_prep_scene, jobs, chunksize=1):
            if prep is None:
                continue
            out_path = args.output_dir / f"{prep['scene_id']}.npz"
            if out_path.is_file():
                continue
            frames = prep["frames"]
            latents = []
            with torch.no_grad():
                for start in range(0, len(frames), int(args.batch_size)):
                    batch_paths = frames[start : start + int(args.batch_size)]
                    images = torch.stack(
                        [_load_image(Path(p), image_size=int(args.image_size)) for p in batch_paths]
                    ).to(device)
                    latents.append(encoder(images).cpu())
            np.savez_compressed(
                out_path,
                latents=torch.cat(latents).numpy().astype(np.float32),
                proprio=np.asarray(prep["proprio"], dtype=np.float32),
                labels=np.asarray(prep["labels"], dtype=np.int64),
                sources=np.asarray(prep["sources"]),
                ticks=np.asarray(prep["ticks"], dtype=np.int64),
            )
            done += 1
            print(f"[{done:4d}] {prep['scene_id']} rows={len(frames)}", flush=True)
    meta = {
        "schema": "go2_corpus_bc_sequences_v0",
        "primitives": list(PRIMITIVES),
        "proprio_feature_dim": 9 + len(PRIMITIVES) + 1,
        "latent_dim": int(encoder_ck.get("latent_dim", 192)),
        "image_size": int(args.image_size),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "sources": list(sources),
        "scenes": done,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
