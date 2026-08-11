#!/usr/bin/env python3
"""Encode a v1.2 branch corpus with the single frozen target encoder.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No predictor checkpoint is opened.

The encoder, its preprocessing and the target normalisation are exactly the ones
the 32-model factorial used: V-JEPA 2.1 ViT-L/384 through its single-frame image
tokenizer, ``preprocess_vjepa`` (PIL RGB -> 512x384 BICUBIC -> ImageNet
normalisation), 24x32 = 768 tokens of width 1024, ``F.layer_norm`` over the token
dimension.

Writes one float16 blob per corpus:

    context.f16   (states, 3, 768, 1024)   the three context slots
    horizon.f16   (branches, 4, 768, 1024) the realised H=1..4 targets
    index.json    row -> blob offsets, plus the encoder identity digests

Run under the GPU torch environment (``~/TinyQuadJEPA/bin/python``); it needs no
Genesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
TOKENS = 768
TOKEN_DIM = 1024
HORIZONS = 4
CONTEXT_SLOTS = 3


def normalise(tokens: torch.Tensor) -> torch.Tensor:
    """The frozen target normalisation (run_dev_v03_temporal_action_jepa_v1)."""

    return F.layer_norm(tokens, (tokens.shape[-1],))


def encode_paths(arm, encoder, paths, device, dtype, batch: int) -> np.ndarray:
    """Dense token grids for a list of PNG paths, normalised, float16."""

    out = np.empty((len(paths), TOKENS, TOKEN_DIM), dtype=np.float16)
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        pixels = torch.stack([arm.preprocess(p) for p in chunk]).to(device=device,
                                                                   dtype=dtype)
        with torch.no_grad():
            tokens = encoder(pixels.unsqueeze(2))
        tokens = normalise(tokens.float())
        if tokens.shape[-2:] != (TOKENS, TOKEN_DIM):
            raise RuntimeError(f"unexpected token shape {tuple(tokens.shape)}")
        out[start:start + len(chunk)] = tokens.cpu().numpy().astype(np.float16)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    out = OUT_ROOT / args.pool
    rows = [json.loads(l) for l in (out / "branch_rows.jsonl").read_text().splitlines()
            if l.strip()]
    valid = [r for r in rows if r.get("valid") and r.get("horizon_paths")]
    print(f"{len(rows)} rows, {len(valid)} valid with renders", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    arm = E.VJepa21Arm()
    encoder = arm.build(device, dtype)
    identity = arm.identity()
    print(f"encoder {identity['name']} on {device}", flush=True)

    # One context triple per state; shared by every branch of that state.
    state_context: dict[str, list[str]] = {}
    for row in valid:
        state_context.setdefault(row["state_id"], row["context_paths"])
    state_ids = sorted(state_context)
    context_paths = [p for sid in state_ids for p in state_context[sid]]
    started = time.time()
    context = encode_paths(arm, encoder, context_paths, device, dtype, args.batch)
    context = context.reshape(len(state_ids), CONTEXT_SLOTS, TOKENS, TOKEN_DIM)
    print(f"context: {context.shape} in {time.time() - started:.1f}s", flush=True)

    horizon_rows = [r for r in valid if len(r["horizon_paths"]) == HORIZONS]
    horizon_paths = [p for r in horizon_rows for p in r["horizon_paths"]]
    started = time.time()
    horizon = encode_paths(arm, encoder, horizon_paths, device, dtype, args.batch)
    horizon = horizon.reshape(len(horizon_rows), HORIZONS, TOKENS, TOKEN_DIM)
    print(f"horizon: {horizon.shape} in {time.time() - started:.1f}s", flush=True)

    context.tofile(out / "context.f16")
    horizon.tofile(out / "horizon.f16")
    index = {
        "status": STATUS, "pool": args.pool,
        "encoder": identity, "tokens": TOKENS, "token_dim": TOKEN_DIM,
        "target_normalisation": "F.layer_norm over the token dimension",
        "preprocess": "dev_frozen_dense_representation_encoders_v1.preprocess_vjepa",
        "context_states": state_ids,
        "context_shape": list(context.shape),
        "horizon_shape": list(horizon.shape),
        "horizon_keys": [f"{r['state_id']}|{r['candidate']}" for r in horizon_rows],
        "context_sha256": hashlib.sha256((out / "context.f16").read_bytes()).hexdigest(),
        "horizon_sha256": hashlib.sha256((out / "horizon.f16").read_bytes()).hexdigest(),
    }
    (out / "latents_index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps({k: v for k, v in index.items()
                      if k not in {"context_states", "horizon_keys", "encoder"}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
