#!/usr/bin/env python3
"""Official-scale (24x1024x16) frozen-encoder action-conditioned predictor.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Capacity intervention ONLY, against the completed 17.2M one-step supervision
control.  Depth, width and head count change because they define capacity; the
predictor's role, inputs, outputs, conditioning and objective do not.

Frozen throughout: the official V-JEPA 2.1 ViT-L encoder never runs here at all
-- the run consumes its cached token tensors, so there is no encoder graph, no
encoder gradient, no EMA and no fine-tuning by construction.

Supervision bundle retained exactly from the completed run:
  * fully visible three-frame context (t-480, t-240, t)
  * predict all 768 future patch tokens
  * dense L1 loss
  * per-token LayerNorm on true-future targets AND on predictor outputs
  * one-step teacher-forced prediction only

No rollout, no proprioception, no action tokens, no extra context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import dev_checkpoint_v1 as CK  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
DIAG = CACHE / "temporal_action_jepa_v1" / "predicted_token_diagnostic"
OUT = CACHE / "temporal_action_jepa_v1" / "arm_frozen_official_scale"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

TOKENS = T.TOKENS
DIM = T.TOKEN_DIM
SEED = T.SEED                      # unchanged from the control
EPOCHS = 6                         # unchanged
EFFECTIVE_BATCH = 4                # unchanged
LR = 3.0e-4                        # unchanged
WEIGHT_DECAY = 0.01                # unchanged
GRAD_CLIP = 1.0                    # unchanged

# predeclared official scale -- do not silently reduce
PRED_DEPTH, PRED_WIDTH, PRED_HEADS = 24, 1024, 16

CACHES = {
    "train_ctx0": (DIAG / "frozen_train_ctx0.f16", 4075),
    "train_ctx1": (DIAG / "frozen_train_ctx1.f16", 4075),
    "current_all": (EVAL / "frozen_current.f16", 4566),
    "train_future": (EVAL / "frozen_train_future.f16", 4075),
}


def load_cache(path: Path, count: int) -> torch.Tensor:
    expected = count * TOKENS * DIM * 2
    if not path.is_file() or path.stat().st_size != expected:
        raise RuntimeError(f"cache missing or wrong size: {path}")
    return torch.from_numpy(
        np.ascontiguousarray(
            np.memmap(path, dtype=np.float16, mode="r", shape=(count, TOKENS, DIM))
        )
    )


def cache_fingerprint(path: Path, sample_bytes: int = 1 << 22) -> str:
    """Cheap stable fingerprint: size plus head/tail bytes."""
    digest = hashlib.sha256()
    digest.update(str(path.stat().st_size).encode())
    with open(path, "rb") as handle:
        digest.update(handle.read(sample_bytes))
        handle.seek(-sample_bytes, 2)
        digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def build(device):
    torch.manual_seed(SEED)
    return T.Predictor(width=PRED_WIDTH, depth=PRED_DEPTH, heads=PRED_HEADS).to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--microbatch", type=int, default=4)
    ap.add_argument("--feasibility-only", action="store_true")
    ap.add_argument("--activation-checkpointing", action="store_true")
    ap.add_argument("--amp", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest resumable checkpoint, restoring optimiser "
                         "moments, epoch index and data-order RNG so the trajectory is faithful")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if EFFECTIVE_BATCH % args.microbatch:
        raise RuntimeError("microbatch must divide the effective batch of 4")
    accumulation = EFFECTIVE_BATCH // args.microbatch

    train_rows, sel_rows = T.load_rows()
    n_train = len(train_rows)
    fingerprints = {k: cache_fingerprint(p) for k, (p, _) in CACHES.items()}

    ctx0 = load_cache(*CACHES["train_ctx0"])
    ctx1 = load_cache(*CACHES["train_ctx1"])
    current = load_cache(*CACHES["current_all"])[:n_train]
    target = load_cache(*CACHES["train_future"])

    predictor = build(device)
    parameters = int(sum(p.numel() for p in predictor.parameters()))
    trainable = int(sum(p.numel() for p in predictor.parameters() if p.requires_grad))
    optimiser = torch.optim.AdamW(
        predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16")

    record = {
        "status": STATUS, "claim_bearing": False,
        "intervention": "predictor capacity only",
        "architecture": {
            "depth": PRED_DEPTH, "width": PRED_WIDTH, "heads": PRED_HEADS,
            "mlp_ratio": 4, "token_dim_in_out": DIM,
            "parameters_total": parameters, "parameters_trainable": trainable,
            "action_conditioning": "AdaLN, unchanged",
            "positional_scheme": "learned spatial (768) + learned temporal (3 context + 1 target), unchanged",
        },
        "control_architecture": {"depth": 6, "width": 384, "heads": 6,
                                 "parameters_total": 17198080},
        "capacity_ratio": round(parameters / 17198080, 3),
        "frozen_encoder": {
            "encoder_executed_in_this_run": False,
            "reason": "the run consumes cached frozen-encoder tokens; there is no encoder graph",
            "encoder_gradients": 0, "ema": False, "top_block_finetuning": False,
        },
        "feature_caches": {k: {"path": str(p), "rows": c,
                               "fingerprint_sha256": fingerprints[k]}
                           for k, (p, c) in CACHES.items()},
        "cache_parity_note": (
            "the 17.2M control trained on live float32 encoder features; this run "
            "trains on the float16 caches produced by the SAME frozen encoder. "
            "Measured parity on a fixed 24-row subset: feature max|d| 0.0156, "
            "mean|d| 2.0e-4, relative 1.76e-4, derived cosine/error agreeing to "
            "1e-6. Reported as a deviation, not hidden."
        ),
        "supervision": {
            "context": "fully visible three frames", "targets": "all 768 future tokens",
            "loss": "dense L1", "target_norm": "per-token LayerNorm",
            "output_norm": "per-token LayerNorm", "rollout": "one-step teacher-forced only",
            "reintroduced_mask_or_smooth_l1": False,
        },
        "training_controls": {
            "seed": SEED, "epochs": EPOCHS, "effective_batch": EFFECTIVE_BATCH,
            "microbatch": args.microbatch, "gradient_accumulation": accumulation,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "optimiser": "AdamW", "amp": args.amp,
            "activation_checkpointing": bool(args.activation_checkpointing),
            "ordered_rows": "unchanged", "checkpoint_selection": "final epoch, as the control",
        },
    }

    # ---------------------------------------------------------- feasibility
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(n_train, generator=generator).tolist()
    probe_batch = order[:EFFECTIVE_BATCH]
    torch.cuda.reset_peak_memory_stats(device)
    step_started = time.time()
    optimiser.zero_grad(set_to_none=True)
    losses = []
    for micro in range(accumulation):
        sel = probe_batch[micro * args.microbatch : (micro + 1) * args.microbatch]
        context = torch.stack(
            [ctx0[sel].float(), ctx1[sel].float(), current[sel].float()], dim=1
        ).to(device)
        future = T.normalise(target[sel].float().to(device))
        action = T.action_tensor([train_rows[i]["primitive"] for i in sel], device)
        with autocast:
            predicted = T.normalise(
                predictor(T.normalise(context), action,
                          torch.ones(len(sel), TOKENS, dtype=torch.bool, device=device))
            )
            loss = (predicted - future).abs().mean() / accumulation
        loss.backward()
        losses.append(float(loss.detach()) * accumulation)
    grad_norm = float(nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP))
    blocks_with_grad = sum(
        1 for block in predictor.blocks
        if any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in block.parameters())
    )
    optimiser.step()
    torch.cuda.synchronize()
    step_seconds = time.time() - step_started
    peak = int(torch.cuda.max_memory_allocated(device))
    steps_per_epoch = (n_train + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH

    record["feasibility"] = {
        "parameter_count": parameters,
        "peak_vram_bytes": peak, "peak_vram_gib": round(peak / 2**30, 3),
        "card_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "microbatch": args.microbatch, "accumulation_factor": accumulation,
        "step_seconds": round(step_seconds, 3),
        "steps_per_epoch": steps_per_epoch,
        "estimated_epoch_minutes": round(steps_per_epoch * step_seconds / 60, 1),
        "estimated_full_run_hours": round(EPOCHS * steps_per_epoch * step_seconds / 3600, 2),
        "loss_finite": bool(np.isfinite(losses).all()), "loss_values": losses,
        "gradient_norm": grad_norm, "gradient_norm_finite": bool(np.isfinite(grad_norm)),
        "blocks_receiving_gradient": blocks_with_grad,
        "blocks_total": len(predictor.blocks),
        "encoder_gradients": 0,
    }
    (OUT / "feasibility.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record["feasibility"], indent=2))
    if blocks_with_grad != len(predictor.blocks):
        raise RuntimeError(f"only {blocks_with_grad}/{len(predictor.blocks)} blocks got gradient")
    if args.feasibility_only:
        return 0

    # ------------------------------------------------------------- training
    predictor = build(device)                       # discard the probe step
    optimiser = torch.optim.AdamW(
        predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    generator = torch.Generator().manual_seed(SEED)
    record["epochs"] = []
    first_epoch = 0
    if args.resume:
        newest = CK.newest_resumable(OUT)
        if newest is not None:
            state = CK.load_for_resume(newest, model=predictor, optimizer=optimiser,
                                       data_order_generator=generator)
            first_epoch = int(state["epoch"]) + 1
            record["epochs"] = state.get("epochs", [])
            record["resumed_from"] = {"path": str(newest), "next_epoch": first_epoch}
            print(f"resumed from {newest.name}, continuing at epoch {first_epoch}", flush=True)
        elif list(OUT.glob("checkpoint_epoch*.pt")):
            raise RuntimeError(
                "checkpoints exist but none carries complete resumable state; resuming "
                "from them would change the optimisation trajectory. Restart from scratch."
            )
    for epoch in range(first_epoch, EPOCHS):
        order = torch.randperm(n_train, generator=generator).tolist()
        predictor.train()
        running, seen = 0.0, 0
        for start in range(0, len(order), EFFECTIVE_BATCH):
            batch = order[start : start + EFFECTIVE_BATCH]
            optimiser.zero_grad(set_to_none=True)
            for micro in range(0, len(batch), args.microbatch):
                sel = batch[micro : micro + args.microbatch]
                context = torch.stack(
                    [ctx0[sel].float(), ctx1[sel].float(), current[sel].float()], dim=1
                ).to(device)
                future = T.normalise(target[sel].float().to(device))
                action = T.action_tensor([train_rows[i]["primitive"] for i in sel], device)
                with autocast:
                    predicted = T.normalise(
                        predictor(T.normalise(context), action,
                                  torch.ones(len(sel), TOKENS, dtype=torch.bool, device=device))
                    )
                    loss = (predicted - future).abs().mean() * (len(sel) / len(batch))
                loss.backward()
                running += float(loss.detach()) * len(batch)
                seen += len(sel)
            nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP)
            optimiser.step()
            if (start // EFFECTIVE_BATCH) % 50 == 0:
                print(f"  [official_scale] epoch {epoch} step {start//EFFECTIVE_BATCH} "
                      f"loss {running/max(seen,1):.5f}", flush=True)
        mean_loss = running / max(seen, 1)
        record["epochs"].append({"epoch": epoch, "train_loss": mean_loss})
        print(f"[official_scale] epoch {epoch} mean loss {mean_loss:.5f}", flush=True)
        receipt = CK.save(
            OUT / f"checkpoint_epoch{epoch}.pt",
            model=predictor, optimizer=optimiser,
            epoch=epoch, global_step=(epoch + 1) * steps_per_epoch, seed=SEED,
            model_config={"width": PRED_WIDTH, "depth": PRED_DEPTH, "heads": PRED_HEADS,
                          "token_dim": DIM, "tokens": TOKENS,
                          "class": "run_dev_v03_temporal_action_jepa_v1.Predictor"},
            scheduler=None,
            scheduler_absent_reason="fixed learning rate; no scheduler is constructed in this run",
            data_order_generator=generator,
            extra={"encoder_trainable": None, "epochs": record["epochs"]},
        )
        record.setdefault("checkpoint_receipts", []).append(receipt)
        record["wall_seconds"] = round(time.time() - started, 1)
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"epochs": record["epochs"],
                      "wall_seconds": record.get("wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
