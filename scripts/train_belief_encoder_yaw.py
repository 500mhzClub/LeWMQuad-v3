#!/usr/bin/env python3
"""Stage 3a reassessment — v7: yaw-CONDITIONED BeliefEncoder.

The yaw probe (``probe_loop_closure_yaw.py``,
``docs/lewm_topological_nav_stage3a_loop_closure_2026-06-09.md``) showed
same-yaw-bin verification is dramatically stronger than any-yaw (belief AP 0.28
-> 0.62; recall@P90 0.08 -> 0.27) but still short of the registered usable band
(R@P95 >= 0.3) — *with an encoder trained to be yaw-INVARIANT* (v6's supcon
pulls any-yaw same-cell pairs together), which actively fights the (cell x
yaw-bin) keyframe node design the probe motivates.

v7 retrains the same BeliefEncoder (v4/v6 winning config) with **yaw-conditioned
contrastive targets** — the spec §5.1 yaw scheme at the registered knob
λ_yaw_weak -> 0:

  - strong positive: same (scene, cell) AND same terminal yaw_bin;
  - masked (ambiguous-ignore): same cell, different yaw_bin — neither pulled
    nor pushed (pushing would fight place identity; pulling fights the node
    design);
  - valid negative: different cell, BFS >= 2, any yaw (same-heading aliased
    corridors arrive as the hard negatives naturally).

Evaluation is NOT retrieval R@5 (the misleading proxy): run
``probe_loop_closure_yaw.py --belief-encoder-dir <v7 encoders>`` and compare
same-yaw verification AP / recall@P{95,90} against v6 on the identical cached
eval banks. Registered bar (Stage 3a doc): same-yaw recall >= 0.3 at P >= 0.95.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.belief_encoder import BeliefEncoder  # noqa: E402
from lewm.models.place_retrieval import masked_supervised_contrastive_loss  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_loop_closure_yaw import build_yaw_banks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_belief_encoder_yaw")


def yaw_conditioned_masks(bank) -> tuple[torch.Tensor, torch.Tensor]:
    """(positive, valid) under the λ_yaw_weak->0 scheme; positive = same cell+yaw."""
    yaws = torch.from_numpy(bank["yaws"])
    diff = (yaws[:, None] - yaws[None, :]).abs()
    same_yaw = torch.minimum(diff, 8 - diff) == 0
    positive = torch.from_numpy(bank["positive"]) & same_yaw
    valid = torch.from_numpy(bank["negative"])  # diff-cell BFS>=2, any yaw
    return positive, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-cache", type=Path, required=True,
                        help="yaw-annotated TRAIN banks cache (built if missing)")
    parser.add_argument("--rollout-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/rollout")
    parser.add_argument("--render-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    parser.add_argument("--manifest-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-scenes-per-family", type=int, default=32)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--windows-per-scene", type=int, default=160)
    parser.add_argument("--max-per-cell", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--seeds", default="20260609,20260610,20260611")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.seed = seeds[0]
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])

    if args.bank_cache.exists():
        logger.info("Loading cached yaw train banks from %s", args.bank_cache)
        banks = torch.load(args.bank_cache, weights_only=False)
    else:
        model, _ = load_model(args, device)
        logger.info("Building yaw-annotated train banks on %s", device)
        banks = build_yaw_banks(model, args.train_split, args.train_scenes_per_family, args, device)
        args.bank_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(banks, args.bank_cache)
        logger.info("Cached %d yaw train banks to %s", len(banks), args.bank_cache)

    # Keep only scenes with at least one anchor having both a same-yaw positive
    # and a valid negative (the loss requires it).
    prepared = []
    for bank in banks:
        positive, valid = yaw_conditioned_masks(bank)
        negative = valid & ~positive
        eligible = (positive.any(dim=1) & negative.any(dim=1)).any()
        if not bool(eligible):
            continue
        prepared.append((torch.from_numpy(bank["z_history"]).float(), positive, valid))
    logger.info("train scenes usable under yaw-conditioned masks: %d / %d", len(prepared), len(banks))
    if len(prepared) < 32:
        raise SystemExit("too few usable train scenes for yaw-conditioned training")

    head_dir = args.output.parent / f"{args.output.stem}_encoders"
    head_dir.mkdir(parents=True, exist_ok=True)
    seed_losses = {}
    for seed in seeds:
        torch.manual_seed(seed)
        latent_dim = int(prepared[0][0].shape[-1])
        encoder = BeliefEncoder(
            latent_dim=latent_dim, embedding_dim=args.embedding_dim, hidden=args.hidden,
            n_heads=args.n_heads, n_layers=args.n_layers, max_len=max(args.history, 16),
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        final_loss = 0.0
        for epoch in range(args.epochs):
            encoder.train()
            order = list(range(len(prepared)))
            random.Random(seed + epoch).shuffle(order)
            total = 0.0
            for idx in order:
                z, positive, valid = prepared[idx]
                emb = encoder(z.to(device))
                loss = masked_supervised_contrastive_loss(
                    emb, positive.to(device), valid.to(device), temperature=args.temperature
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach())
            final_loss = total / len(order)
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                logger.info("seed=%d epoch=%d train_loss=%.4f", seed, epoch, final_loss)
        seed_losses[seed] = final_loss
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
             "latent_dim": encoder.latent_dim, "embedding_dim": encoder.embedding_dim,
             "source_checkpoint": str(args.checkpoint), "seed": seed,
             "metrics": {"final_train_loss": final_loss},
             "config": {k: str(v) for k, v in vars(args).items()}},
            head_dir / f"belief_encoder_seed{seed}.pt",
        )
        logger.info("saved seed %d encoder (final loss %.4f)", seed, final_loss)

    report = {
        "schema": "lewm_belief_encoder_yaw_v0",
        "train_scene_count": len(prepared),
        "objective": "supcon, positives=same(cell,yaw_bin), same-cell-diff-yaw masked, negatives=BFS>=2 any-yaw",
        "seed_final_losses": {str(k): v for k, v in seed_losses.items()},
        "encoders_dir": str(head_dir),
        "evaluation": "run probe_loop_closure_yaw.py --belief-encoder-dir on the cached eval banks",
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
