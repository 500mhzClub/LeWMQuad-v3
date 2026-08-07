#!/usr/bin/env python3
"""Matched temporal action-conditioned JEPA: frozen vs top-block encoder movement.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Frozen causal contract (see ``build_dev_v03_temporal_sequences_v1.py``):

    context = t-480, t-240, t   (v03, centre-cropped to the v04 field of view)
    target  = t+240
    action  = the command block executed from t to t+240

Both arms share rows, ordering, seed, optimiser, predictor, action representation
and schedule.  They differ in exactly one thing:

    A  frozen encoder, predictor trainable
    B  identical initialisation, final encoder block(s) + final norms trainable
       at a lower learning rate than the predictor, with an EMA target encoder

Neither arm caches encoder features: both recompute them every step, so the two
arms see numerically identical data paths.  The predictive state is the dense
24x32 token grid throughout -- no CLS, no global pooling, no BEV bottleneck.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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

from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import dev_checkpoint_v1 as CK  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
ROWS = CACHE / "temporal_rows.jsonl"
OUT = CACHE / "temporal_action_jepa_v1"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

GRID = (24, 32)
TOKENS = GRID[0] * GRID[1]
TOKEN_DIM = 1024
CONTEXT_POSITIONS = 3

PRIMITIVES = ("arc_left", "arc_right", "backward", "forward_fast", "forward_medium",
              "forward_slow", "hold", "yaw_left", "yaw_right")
COMMANDS = {
    "arc_left": (0.20, 0.0, 0.45), "arc_right": (0.20, 0.0, -0.45),
    "backward": (-0.20, 0.0, 0.0), "forward_fast": (0.30, 0.0, 0.0),
    "forward_medium": (0.25, 0.0, 0.0), "forward_slow": (0.20, 0.0, 0.0),
    "hold": (0.0, 0.0, 0.0), "yaw_left": (0.0, 0.0, 0.45), "yaw_right": (0.0, 0.0, -0.45),
}
SEED = 2_026_080_651
MASK_RATIO = 0.5


# --------------------------------------------------------------------------
class Predictor(nn.Module):
    """Action-conditioned masked context->target predictor over dense tokens.

    Context tokens carry a learned temporal position (one per context frame);
    target queries carry the target temporal position.  Action conditioning is
    AdaLN over every block.  Output is the dense target token grid.
    """

    def __init__(self, token_dim=TOKEN_DIM, width=384, depth=6, heads=6):
        super().__init__()
        self.width = width
        self.input = nn.Linear(token_dim, width)
        self.output = nn.Linear(width, token_dim)
        self.spatial = nn.Parameter(torch.zeros(1, TOKENS, width))
        self.temporal = nn.Parameter(torch.zeros(CONTEXT_POSITIONS + 1, 1, width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.spatial, std=0.02)
        nn.init.trunc_normal_(self.temporal, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.action = nn.Sequential(
            nn.Linear(len(PRIMITIVES) + 3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [PredictorBlock(width, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, context, action, mask):
        """context: (B, 3, N, D); action: (B, A); mask: (B, N) bool, True = predict."""
        b, t, n, _ = context.shape
        x = self.input(context.reshape(b * t, n, -1)).reshape(b, t, n, self.width)
        x = x + self.spatial.unsqueeze(1) + self.temporal[:CONTEXT_POSITIONS].unsqueeze(0)
        x = x.reshape(b, t * n, self.width)
        query = self.mask_token.expand(b, n, -1) + self.spatial + self.temporal[CONTEXT_POSITIONS]
        sequence = torch.cat([x, query], dim=1)
        conditioning = self.action(action)
        for block in self.blocks:
            sequence = block(sequence, conditioning)
        return self.output(self.norm(sequence[:, t * n :]))


class PredictorBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Linear(width * 4, width))
        self.ada = nn.Linear(width, width * 6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, conditioning):
        s1, b1, g1, s2, b2, g2 = self.ada(conditioning).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        b, n, w = h.shape
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, w // self.heads).permute(
            2, 0, 3, 1, 4
        )
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(b, n, w))
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


# --------------------------------------------------------------------------
def load_rows():
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    return ([r for r in rows if r["role"] == "train"],
            [r for r in rows if r["role"] == "checkpoint_selection"])


def action_tensor(primitives, device):
    idx = torch.tensor([PRIMITIVES.index(p) for p in primitives])
    one_hot = F.one_hot(idx, num_classes=len(PRIMITIVES)).float()
    command = torch.tensor([COMMANDS[p] for p in primitives], dtype=torch.float32)
    return torch.cat([one_hot, command], dim=-1).to(device)


def load_batch(rows, indices, arm, device, dtype):
    """Decode and preprocess the four frames of each sequence."""
    context, target = [], []
    for i in indices:
        row = rows[i]
        context.append(torch.stack([arm.preprocess(p) for p in row["context_paths"]]))
        target.append(arm.preprocess(row["target_path"]))
    return (torch.stack(context).to(device=device, dtype=dtype),
            torch.stack(target).to(device=device, dtype=dtype))


def encode(module, frames, grad: bool):
    """Dense image-token path, one frame at a time, grids preserved."""
    b = frames.shape[0]
    flat = frames.reshape(-1, *frames.shape[-3:])
    with torch.set_grad_enabled(grad):
        tokens = module(flat.unsqueeze(2))
    return tokens.reshape(b, -1, TOKENS, TOKEN_DIM)


def normalise(tokens):
    """Target normalisation: the WP-E contraction shortcut fix."""
    return F.layer_norm(tokens, (tokens.shape[-1],))


def token_health(tokens):
    flat = tokens.reshape(-1, tokens.shape[-1]).float()
    centred = flat - flat.mean(0, keepdim=True)
    variance = float(centred.pow(2).mean())
    cov = (centred.T @ centred) / max(1, centred.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    p = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    entropy = -(p * (p + 1e-12).log()).sum()
    return {"variance": variance, "effective_rank": float(entropy.exp())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", required=True, choices=("frozen", "moving"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--predictor-lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr-ratio", type=float, default=0.05)
    ap.add_argument("--trainable-blocks", type=int, default=1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--amp", default="bf16", choices=("bf16", "fp32"))
    # Supervision contract.  "smooth_l1_masked" is the original run.  "l1_dense"
    # reproduces the official DROID robot post-training `loss:` block minus
    # auto_steps: loss_exp 1.0 (L1), dense over every future token (the official
    # path masks nothing), and normalize_reps applied to the predictor output as
    # well as the target.
    ap.add_argument("--loss-mode", default="smooth_l1_masked",
                    choices=("smooth_l1_masked", "l1_dense"))
    ap.add_argument("--tag", default=None, help="output subdirectory suffix")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    out = OUT / (f"arm_{args.arm}" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.time()

    train_rows, sel_rows = load_rows()
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    arm = E.VJepa21CroppedV03Arm()
    online = arm.build(device, dtype)
    target_encoder = copy.deepcopy(online).eval().requires_grad_(False)

    for p in online.parameters():
        p.requires_grad_(False)
    trainable, trainable_names = [], []
    # WP-E defect: parameter freezing alone does not disable dropout or mutable
    # buffers.  Put the WHOLE encoder in eval mode first, then opt the trainable
    # blocks back into train mode -- never the reverse.
    online.eval()
    if args.arm == "moving":
        blocks = list(online.blocks)[-args.trainable_blocks :]
        for module in blocks + [online.norms_block]:
            module.train()
            for name, p in module.named_parameters():
                p.requires_grad_(True)
                trainable.append(p)
                trainable_names.append(name)
    allowed = set()
    if args.arm == "moving":
        for module in blocks + [online.norms_block]:
            allowed.update(id(m) for m in module.modules())
    stray = [n for n, m in online.named_modules() if m.training and id(m) not in allowed]
    if stray:
        raise RuntimeError(f"frozen modules left in train mode: {stray[:5]}")
    if any(p.requires_grad for p in target_encoder.parameters()):
        raise RuntimeError("target encoder must not receive gradient")

    predictor = Predictor().to(device)
    groups = [{"params": list(predictor.parameters()), "lr": args.predictor_lr}]
    if trainable:
        groups.append({"params": trainable, "lr": args.predictor_lr * args.encoder_lr_ratio})
    optimiser = torch.optim.AdamW(groups, weight_decay=0.01, foreach=False)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16")

    record = {
        "status": STATUS, "claim_bearing": False, "arm": args.arm,
        "temporal_contract": {"context_offsets": [-480, -240, 0], "target_offset": 240,
                              "seconds_per_offset": 0.5, "action": "command block t -> t+240"},
        "encoder": arm.identity(),
        "visual_contract": E.preprocessing_identity(arm),
        "rows": {"train": len(train_rows), "checkpoint_selection": len(sel_rows)},
        "trainable_encoder_parameters": int(sum(p.numel() for p in trainable)),
        "total_encoder_parameters": int(sum(p.numel() for p in online.parameters())),
        "predictor_parameters": int(sum(p.numel() for p in predictor.parameters())),
        "schedule": {"epochs": args.epochs, "batch": args.batch, "seed": SEED,
                     "predictor_lr": args.predictor_lr,
                     "encoder_lr": args.predictor_lr * args.encoder_lr_ratio if trainable else None,
                     "ema": args.ema,
                     "loss_mode": args.loss_mode,
                     "mask_ratio": 0.0 if args.loss_mode == "l1_dense" else MASK_RATIO,
                     "trainable_blocks": args.trainable_blocks if trainable else 0},
        "feature_cache_used": False,
        "epochs": [],
    }

    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        running, seen = 0.0, 0
        for start in range(0, len(order), args.batch):
            batch = order[start : start + args.batch]
            context_px, target_px = load_batch(train_rows, batch, arm, device, dtype)
            action = action_tensor([train_rows[i]["primitive"] for i in batch], device)
            if args.loss_mode == "l1_dense":
                # the official robot path supervises every future token
                mask = torch.ones(len(batch), TOKENS, dtype=torch.bool, device=device)
            else:
                mask = (torch.rand(len(batch), TOKENS, generator=generator).to(device) < MASK_RATIO)
                mask[:, 0] = True                              # never an empty target
            optimiser.zero_grad(set_to_none=True)
            with autocast:
                context = encode(online, context_px, grad=bool(trainable))
                with torch.no_grad():
                    future = normalise(encode(target_encoder, target_px.unsqueeze(1), grad=False))[:, 0]
                predicted = predictor(normalise(context), action, mask)
                if args.loss_mode == "l1_dense":
                    predicted = normalise(predicted)           # normalize_reps on the output
                    loss = (predicted - future).abs().mean()   # loss_exp = 1.0
                else:
                    loss = F.smooth_l1_loss(predicted[mask], future[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(predictor.parameters()) + trainable, 1.0
            )
            optimiser.step()
            if trainable:
                with torch.no_grad():
                    for tp, op in zip(target_encoder.parameters(), online.parameters()):
                        tp.mul_(args.ema).add_(op.detach(), alpha=1 - args.ema)
            running += float(loss.detach()) * len(batch)
            seen += len(batch)
            if (start // args.batch) % 25 == 0:
                print(f"  [{args.arm}] epoch {epoch} step {start//args.batch} "
                      f"loss {running/max(seen,1):.5f}", flush=True)
        record["epochs"].append({"epoch": epoch, "train_loss": running / max(seen, 1)})
        print(f"[{args.arm}] epoch {epoch} mean loss {running/max(seen,1):.5f}", flush=True)
        receipt = CK.save(
            out / f"checkpoint_epoch{epoch}.pt",
            model=predictor, optimizer=optimiser,
            epoch=epoch, global_step=(epoch + 1) * ((len(train_rows) + args.batch - 1) // args.batch),
            seed=SEED,
            model_config={"width": 384, "depth": 6, "heads": 6,
                          "token_dim": TOKEN_DIM, "tokens": TOKENS,
                          "class": "run_dev_v03_temporal_action_jepa_v1.Predictor"},
            scheduler=None,
            scheduler_absent_reason="fixed learning rate; no scheduler is constructed in this run",
            data_order_generator=generator,
            extra={
                "encoder_trainable": (
                    {n: q.detach().cpu() for n, q in online.named_parameters()
                     if q.requires_grad} if trainable else None
                ),
                "target_encoder_trainable": (
                    {n: q.detach().cpu() for n, q in target_encoder.named_parameters()
                     if n in {m for m, r in online.named_parameters() if r.requires_grad}}
                    if trainable else None
                ),
                "epochs": record["epochs"],
            },
        )
        record.setdefault("checkpoint_receipts", []).append(receipt)
        record["wall_seconds"] = round(time.time() - started, 1)
        (out / "result.json").write_text(json.dumps(record, indent=2))

    (out / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"arm": args.arm, "epochs": record["epochs"],
                      "wall_seconds": record.get("wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
