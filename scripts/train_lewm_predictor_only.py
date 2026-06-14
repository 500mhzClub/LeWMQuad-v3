#!/usr/bin/env python3
"""Stage 1: predictor-only rollout-objective ablation on cached latents (docs §12).

Trains a FRESH predictor (+projector) on cached frozen-encoder latents, with or
without the free-running rollout loss, then measures the teacher-forced vs
free-running decomposition on a held-out cache. No images / no encoder => a whole
sweep runs in minutes. Doubles as a preview of the freeze-encoder (DINO-WM) path:
the encoder here is exactly the frozen e9 encoder that produced the cache.

Key question: does adding the rollout loss collapse the free-running/teacher-forced
compounding ratio (≈22x at h=10 for production e9) toward ~1-3, and make rollout
beat persistence at h>=3?
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.models.lewm import LeWorldModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _horizon_weights(H: int, gamma: float, device) -> torch.Tensor:
    return gamma ** torch.arange(H, dtype=torch.float32, device=device)


def _onestep_loss(model, z_raw, z_proj, cmd):
    if z_raw.shape[1] <= model.predictor.max_seq_len:
        z_pred_raw = model.predictor(z_raw, cmd)             # (B,T,D) teacher-forced
        z_pred_proj = model.pred_projector.forward_seq(z_pred_raw)
        return (z_pred_proj[:, :-1] - z_proj[:, 1:]).square().mean()

    # Keep seq4 production checkpoints usable on longer cached windows: predict
    # each transition from the true z[t] with one action, matching the
    # teacher-forced decomposition probe.
    B, T, D = z_raw.shape
    starts = z_raw[:, :-1].reshape(B * (T - 1), D)
    acts = cmd[:, :-1].reshape(B * (T - 1), 1, cmd.shape[-1])
    z_pred = model.predictor.rollout(starts, acts).reshape(B, T - 1, D)
    z_pred_proj = model.pred_projector.forward_seq(z_pred)
    return (z_pred_proj - z_proj[:, 1:]).square().mean()


def _rollout_loss(model, z_raw, z_proj, cmd, H, gamma):
    roll_raw = model.predictor.rollout(z_raw[:, 0], cmd[:, :H])   # (B,H,D) free-running
    roll_proj = model.pred_projector.forward_seq(roll_raw)
    per_step = (roll_proj - z_proj[:, 1:H + 1]).square().mean(-1)  # (B,H)
    w = _horizon_weights(H, gamma, per_step.device)
    return (per_step * w).sum(-1).mean() / w.sum()


def _load_predictor_init(model: LeWorldModel, checkpoint_path: Path) -> dict:
    """Load predictor/pred_projector weights, tolerating max_seq_len changes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    filtered = {}
    copied, skipped = [], []
    for name, tensor in state.items():
        if not (name.startswith("predictor.") or name.startswith("pred_projector.")):
            continue
        if name not in target:
            skipped.append({"name": name, "reason": "missing_in_target"})
            continue
        if tuple(target[name].shape) == tuple(tensor.shape):
            filtered[name] = tensor
            copied.append(name)
            continue
        if (
            name == "predictor.pos_embed"
            and tensor.ndim == 3
            and target[name].ndim == 3
            and tensor.shape[-1] == target[name].shape[-1]
        ):
            patched = target[name].clone()
            n = min(int(tensor.shape[1]), int(target[name].shape[1]))
            patched[:, :n] = tensor[:, :n]
            filtered[name] = patched
            copied.append(f"{name}[:,:{n}]")
            continue
        skipped.append({
            "name": name,
            "reason": f"shape {tuple(tensor.shape)} -> {tuple(target[name].shape)}",
        })
    model.load_state_dict(filtered, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "copied_count": len(copied),
        "copied_examples": copied[:8],
        "skipped": skipped,
    }


@torch.no_grad()
def evaluate(model, z_raw, z_proj, cmd, horizons, device, batch=256):
    model.eval()
    T = z_raw.shape[1]
    max_h = min(max(horizons), T - 1)
    sums = {k: torch.zeros(max_h, dtype=torch.float64) for k in ("tf", "free", "persist", "step_delta")}
    n = 0
    for i in range(0, z_raw.shape[0], batch):
        zr = z_raw[i:i + batch].to(device)
        zp = z_proj[i:i + batch].to(device)
        cm = cmd[i:i + batch].to(device)
        B = zr.shape[0]
        targets = zp[:, 1:max_h + 1]
        # free-running rollout from z[0]
        free = model.pred_projector.forward_seq(model.predictor.rollout(zr[:, 0], cm[:, :max_h]))
        # teacher-forced single step from the TRUE latent at each step t
        tf_starts = zr[:, :max_h].reshape(B * max_h, -1)
        tf_acts = cm[:, :max_h].reshape(B * max_h, 1, cm.shape[-1])
        tf = model.pred_projector.forward_seq(model.predictor.rollout(tf_starts, tf_acts)).reshape(B, max_h, -1)
        persist = zp[:, :1].expand_as(targets)
        prev = zp[:, :max_h]
        batch_m = {
            "tf": (tf - targets).square().mean(-1),
            "free": (free - targets).square().mean(-1),
            "persist": (persist - targets).square().mean(-1),
            "step_delta": (targets - prev).square().mean(-1),
        }
        for k, v in batch_m.items():
            sums[k] += v.double().sum(0).cpu()
        n += B

    reports = []
    for h in horizons:
        if h > max_h:
            continue
        i = h - 1
        pt = {k: float(sums[k][i] / n) for k in sums}
        reports.append({
            "horizon": h,
            "tf_1step_mse": pt["tf"],
            "free_running_mse": pt["free"],
            "persistence_mse": pt["persist"],
            "target_step_delta_mse": pt["step_delta"],
            "free_over_tf": (pt["free"] / pt["tf"]) if pt["tf"] > 0 else None,
            "tf_over_persistence": (pt["tf"] / pt["persist"]) if pt["persist"] > 0 else None,
            "free_over_persistence": (pt["free"] / pt["persist"]) if pt["persist"] > 0 else None,
        })
    return reports


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--rollout-lambda", type=float, default=0.0)
    p.add_argument("--rollout-horizon", type=int, default=None)
    p.add_argument("--rollout-gamma", type=float, default=0.9)
    p.add_argument("--max-seq-len", type=int, default=11)
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="optional checkpoint to initialize predictor + pred_projector")
    p.add_argument("--freeze-pred-projector", action="store_true",
                   help="train only predictor weights; keep pred_projector fixed")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--horizons", type=str, default="1,2,3,5,8,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for predictor-only training, but torch.cuda.is_available() is false")
    horizons = [int(x) for x in args.horizons.split(",")]

    tr = torch.load(args.train_cache, map_location="cpu", weights_only=True)
    ev = torch.load(args.eval_cache, map_location="cpu", weights_only=True)
    latent_dim = int(tr["z_raw"].shape[-1])
    cmd_dim = int(tr["cmd"].shape[-1])
    T = int(tr["z_raw"].shape[1])
    if int(ev["z_raw"].shape[1]) != T:
        raise SystemExit(
            f"eval cache seq_len {ev['z_raw'].shape[1]} does not match train cache {T}"
        )
    if int(ev["z_raw"].shape[-1]) != latent_dim or int(ev["cmd"].shape[-1]) != cmd_dim:
        raise SystemExit("eval cache latent/cmd dimensions do not match train cache")
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")

    model = LeWorldModel(latent_dim=latent_dim, cmd_dim=cmd_dim, max_seq_len=args.max_seq_len).to(device)
    init_summary = None
    if args.init_checkpoint is not None:
        init_summary = _load_predictor_init(model, args.init_checkpoint)
        model.to(device)
        logger.info(
            "initialized predictor/pred_projector from %s (%d tensors copied)",
            args.init_checkpoint,
            init_summary["copied_count"],
        )
    for prm in model.encoder.parameters():
        prm.requires_grad_(False)
    for prm in model.enc_projector.parameters():
        prm.requires_grad_(False)
    if args.freeze_pred_projector:
        for prm in model.pred_projector.parameters():
            prm.requires_grad_(False)
        params = list(model.predictor.parameters())
    else:
        params = list(model.predictor.parameters()) + list(model.pred_projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    ds = TensorDataset(tr["z_raw"], tr["z_proj"], tr["cmd"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    H = min(args.rollout_horizon or (T - 1), T - 1)
    logger.info("train=%d eval=%d seq_len=%d latent=%d cmd=%d rollout_lambda=%.2f H=%d max_seq_len=%d",
                tr["z_raw"].shape[0], ev["z_raw"].shape[0], T, latent_dim, cmd_dim, args.rollout_lambda, H, args.max_seq_len)

    for epoch in range(args.epochs):
        model.train()
        rl = args.rollout_lambda * min(1.0, (epoch + 1) / max(1, args.warmup_epochs)) if args.rollout_lambda > 0 else 0.0
        tot = tot1 = totr = nb = 0.0
        for z_raw, z_proj, cmd in loader:
            z_raw, z_proj, cmd = z_raw.to(device), z_proj.to(device), cmd.to(device)
            l1 = _onestep_loss(model, z_raw, z_proj, cmd)
            lr_ = _rollout_loss(model, z_raw, z_proj, cmd, H, args.rollout_gamma) if rl > 0 else z_raw.new_zeros(())
            loss = l1 + rl * lr_
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.detach())
            tot1 += float(l1.detach())
            totr += float(lr_.detach())
            nb += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("epoch %d/%d loss=%.4f 1step=%.4f rollout=%.4f (rl=%.2f)",
                        epoch + 1, args.epochs, tot / nb, tot1 / nb, totr / nb, rl)

    reports = evaluate(model, ev["z_raw"], ev["z_proj"], ev["cmd"], horizons, device)
    record = {
        "schema": "lewm_predictor_only_rollout_ablation_v0",
        "interpretation_note": (
            "Predictor + prediction projector trained on frozen e9 encoder "
            "latents; positive results isolate rollout-objective efficacy, "
            "negative results remain inconclusive for end-to-end training."
        ),
        "config": {
            "rollout_lambda": args.rollout_lambda, "rollout_horizon": H,
            "rollout_gamma": args.rollout_gamma, "max_seq_len": args.max_seq_len,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "init_summary": init_summary,
            "freeze_pred_projector": bool(args.freeze_pred_projector),
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs, "seed": args.seed,
            "train_cache": str(args.train_cache), "eval_cache": str(args.eval_cache),
            "n_train": int(tr["z_raw"].shape[0]), "n_eval": int(ev["z_raw"].shape[0]),
            "seq_len": T, "latent_dim": latent_dim, "cmd_dim": cmd_dim,
            "source_checkpoint": tr.get("checkpoint"),
            "train_cache_meta": {
                "holdout_role": tr.get("holdout_role"),
                "holdout_fraction": tr.get("holdout_fraction"),
                "holdout_seed": tr.get("holdout_seed"),
                "render_root": tr.get("render_root"),
            },
            "eval_cache_meta": {
                "holdout_role": ev.get("holdout_role"),
                "holdout_fraction": ev.get("holdout_fraction"),
                "holdout_seed": ev.get("holdout_seed"),
                "render_root": ev.get("render_root"),
            },
        },
        "horizons": reports,
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
