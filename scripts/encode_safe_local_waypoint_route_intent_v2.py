#!/usr/bin/env python3
"""Encode the frozen V2 RGB index into H1-H3 ViT-L trajectory shards."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "lewm_worlds")]
from dev_frozen_dense_representation_encoders_v1 import VJepa21CroppedV03Arm

REPLAY = ROOT / ".generated/safe_local_waypoint_route_intent_v2/replay"
OUT = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/safe_local_waypoint_route_intent_v2/latents")
EXPECTED = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    files = sorted(REPLAY.glob("purpose-*.json"), key=lambda p: int(p.stem.split("-")[1]))
    if len(files) != 48:
        raise RuntimeError(f"expected 48 replay states, found {len(files)}")
    records = []
    for path in files:
        state = json.loads(path.read_text())
        if state.get("render_status") not in (None, "COMPLETE"):
            raise RuntimeError(f"render incomplete: {path}")
        for row in state["rows"]:
            for h in (1, 2, 3):
                item = row["horizons"][str(h)]
                rgb = Path(item["rgb_path"])
                if sha(rgb) != item["rgb_sha256"]:
                    raise RuntimeError(f"RGB digest mismatch: {rgb}")
                records.append((state["state_id"], row["candidate_index"], h, rgb))
    CACHE.mkdir(parents=True, exist_ok=True)
    arm = VJepa21CroppedV03Arm()
    if sha(Path(arm.checkpoint)) != EXPECTED:
        raise RuntimeError("encoder checkpoint digest mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm.build(device, torch.float32)
    started = time.time(); batch_size = 16; index = []; peak = 0
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        tensors = torch.stack([arm.preprocess(str(r[3])) for r in batch]).to(device)
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded = arm.tokens(tensors).float().cpu().numpy().astype(np.float16)
        for (sid, ci, h, rgb), tokens in zip(batch, encoded):
            path = CACHE / f"{sid}_candidate_{ci:02d}_h{h}.npy"
            np.save(path, tokens)
            index.append({"state_id": sid, "candidate_index": ci, "horizon": h,
                          "rgb_path": str(rgb), "latent_path": str(path),
                          "shape": [768, 1024], "dtype": "float16", "sha256": sha(path)})
        if device.type == "cuda": peak = max(peak, int(torch.cuda.max_memory_allocated()))
        if offset % 160 == 0:
            print(json.dumps({"encoded": min(offset + len(batch), len(records)), "total": len(records)}), flush=True)
    payload = {"schema": "safe_local_waypoint_route_intent_v2_target_latent_index",
               "encoder_sha256": EXPECTED, "count": len(index), "entries": index,
               "runtime_s": time.time() - started, "peak_vram_bytes": peak}
    target = OUT / "target_latent_index.json"
    target.write_text(json.dumps(payload, sort_keys=True, indent=2))
    print(json.dumps({"count": len(index), "runtime_s": payload["runtime_s"],
                      "peak_vram_bytes": peak, "index_sha256": sha(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
