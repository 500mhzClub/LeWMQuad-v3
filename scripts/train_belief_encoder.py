#!/usr/bin/env python3
"""Train + gate the Stage 2 BeliefEncoder on frozen-LeWM history windows.

Stage 1 (``docs/lewm_topological_nav_stage1_retrieval_2026-06-09.md``) showed
naive mean/concat pooling of a frozen-latent history barely lifts same-place
recall, yet a history window separates single-frame-aliased different-cell pairs
with AUC ~0.86. This script trains the learned temporal aggregator
(``lewm.models.belief_encoder.BeliefEncoder``) with supervised-contrastive loss
(same masking as ``PlaceRetrievalHead``: same-cell positives, BFS-distance >= 2
valid negatives) and gates it against the **naive pooling** baseline (not just
single-frame) — the registered Stage 2 bar. No anti-collapse regularizer: the
contrastive negatives prevent collapse (the repo's anti-collapse mechanism,
SIGReg, is for the negative-free world-model objective, not a contrastive head).

Reuses the Stage 1 data path verbatim: ``build_scene_bank`` +
``_select_history_windows`` -> per-scene ``(z_history (Nw, H, D), cells)`` over
frozen *raw* latents (same space the naive baselines live in). The LeWM encoder
stays frozen.
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
from probe_lewm_latent_aliasing import _encode_frames  # noqa: E402
from probe_lewm_history_retrieval import _select_history_windows  # noqa: E402
from probe_lewm_reachability_a3 import _select, build_scene_bank, same_cell_retrieval  # noqa: E402
from train_place_retrieval_head import _aggregate, _pair_masks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_belief_encoder")


def build_history_banks(model, split, per_family, args, device):
    """Per-scene frozen history windows: {scene_id, z_history (Nw,H,D), cells, graph}."""
    banks = []
    for index, (family, label_file) in enumerate(_select(args.rollout_root, split, per_family, args.seed)):
        bank = build_scene_bank(
            model,
            label_file=label_file,
            family=family,
            split=split,
            render_root=args.render_root,
            corpus_root=args.manifest_corpus,
            device=device,
            frames_per_scene=max(12, args.min_cells * 2),
            max_per_cell=2,
            batch_size=args.batch_size,
            min_cells=args.min_cells,
            rng=random.Random(args.seed + index * 7919),
        )
        if bank is None:
            continue
        windows, cells = _select_history_windows(
            bank,
            history=args.history,
            windows_per_scene=args.windows_per_scene,
            max_per_cell=args.max_per_cell,
            rng=random.Random(args.seed + index * 104729),
        )
        if len(set(cells.tolist())) < args.min_cells or len(windows) < 12:
            continue
        unique_paths = list(dict.fromkeys(path for window in windows for path in window))
        z_unique, _z_proj = _encode_frames(model, unique_paths, device, args.batch_size)
        path_index = {path: offset for offset, path in enumerate(unique_paths)}
        window_index = np.asarray([[path_index[p] for p in w] for w in windows], dtype=np.int64)
        z_history = z_unique[window_index].astype(np.float32)  # (Nw, H, D)
        # Precompute contrastive pair masks now (needs the graph) so cached banks
        # are self-contained and training/eval never touch the SceneGraph.
        positive, valid = _pair_masks({"cells": cells, "graph": bank["graph"]})
        banks.append({
            "scene_id": bank["scene_id"],
            "z_history": z_history,
            "cells": cells,
            "positive": positive.numpy(),
            "valid": valid.numpy(),
        })
        logger.info("bank %s %s windows=%d cells=%d", split, bank["scene_id"], len(windows), len(set(cells.tolist())))
    return banks


def naive_baselines(banks) -> dict:
    """Single-frame terminal + history mean/concat recall (the Stage 1 baselines)."""
    out = {"terminal": {"r1": [], "r5": []}, "mean": {"r1": [], "r5": []}, "concat": {"r1": [], "r5": []}}
    for bank in banks:
        z = bank["z_history"].astype(np.float64)
        descriptors = {
            "terminal": z[:, -1],
            "mean": z.mean(axis=1),
            "concat": z.reshape(len(z), -1),
        }
        for name, desc in descriptors.items():
            ret = same_cell_retrieval(desc, bank["cells"])
            if ret is not None:
                out[name]["r1"].append(ret["retrieval_at_1"])
                out[name]["r5"].append(ret["retrieval_at_5"])
    return {name: {"retrieval_at_1": _aggregate(v["r1"]), "retrieval_at_5": _aggregate(v["r5"])} for name, v in out.items()}


@torch.no_grad()
def evaluate(encoder, banks, device) -> dict:
    encoder.eval()
    r1, r5 = [], []
    for bank in banks:
        z = torch.from_numpy(bank["z_history"]).float().to(device)
        emb = encoder(z).cpu().numpy().astype(np.float64)
        ret = same_cell_retrieval(emb, bank["cells"])
        if ret is not None:
            r1.append(ret["retrieval_at_1"])
            r5.append(ret["retrieval_at_5"])
    return {"retrieval_at_1": _aggregate(r1), "retrieval_at_5": _aggregate(r5)}


def train_seed(train_banks, eval_banks, *, seed, args, device):
    torch.manual_seed(seed)
    latent_dim = int(train_banks[0]["z_history"].shape[-1])
    encoder = BeliefEncoder(
        latent_dim=latent_dim,
        embedding_dim=args.embedding_dim,
        hidden=args.hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=max(args.history, 16),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    prepared = [
        (
            torch.from_numpy(b["z_history"]).float(),
            torch.from_numpy(b["positive"]),
            torch.from_numpy(b["valid"]),
        )
        for b in train_banks
    ]

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

    metrics = evaluate(encoder, eval_banks, device)
    metrics["seed"] = seed
    metrics["final_train_loss"] = final_loss
    return encoder, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/rollout")
    parser.add_argument("--render-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    parser.add_argument("--manifest-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test_id")
    parser.add_argument("--train-scenes-per-family", type=int, default=4)
    parser.add_argument("--eval-scenes-per-family", type=int, default=4)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--windows-per-scene", type=int, default=160)
    parser.add_argument("--max-per-cell", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--bank-cache", type=Path, default=None,
                        help="cache the encoded history banks here; reused (and model load skipped) if it exists")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", default="20260609,20260610,20260611")
    parser.add_argument("--gate-recall5-improvement", type=float, default=0.05)
    parser.add_argument("--min-eval-scenes", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.seed = seeds[0]
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    if args.bank_cache is not None and args.bank_cache.exists():
        logger.info("Loading cached history banks from %s (skipping model load)", args.bank_cache)
        blob = torch.load(args.bank_cache, weights_only=False)
        train_banks, eval_banks = blob["train"], blob["eval"]
    else:
        model, _config = load_model(args, device)
        logger.info("Building frozen train history banks on %s", device)
        train_banks = build_history_banks(model, args.train_split, args.train_scenes_per_family, args, device)
        logger.info("Building frozen eval history banks")
        eval_banks = build_history_banks(model, args.eval_split, args.eval_scenes_per_family, args, device)
        if args.bank_cache is not None:
            args.bank_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"train": train_banks, "eval": eval_banks}, args.bank_cache)
            logger.info("Cached history banks to %s", args.bank_cache)
    if not train_banks or not eval_banks:
        raise SystemExit("no usable train or eval history banks")

    baseline = naive_baselines(eval_banks)
    # The bar to beat = best naive history descriptor (mean vs concat) on R@5.
    naive_r5 = max(baseline["mean"]["retrieval_at_5"]["mean"], baseline["concat"]["retrieval_at_5"]["mean"])
    naive_r1 = max(baseline["mean"]["retrieval_at_1"]["mean"], baseline["concat"]["retrieval_at_1"]["mean"])

    head_dir = args.output.parent / f"{args.output.stem}_encoders"
    head_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    for seed in seeds:
        encoder, metrics = train_seed(train_banks, eval_banks, seed=seed, args=args, device=device)
        seed_reports.append(metrics)
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
             "latent_dim": encoder.latent_dim, "embedding_dim": encoder.embedding_dim,
             "source_checkpoint": str(args.checkpoint), "seed": seed, "metrics": metrics,
             "config": {k: str(v) for k, v in vars(args).items()}},
            head_dir / f"belief_encoder_seed{seed}.pt",
        )

    # Train-set recall of the last seed's encoder: a large train>>eval gap = the
    # encoder overfits scene-specific structure (the cross-scene generalization
    # diagnostic), distinct from a representational ceiling.
    train_recall_last_seed = evaluate(encoder, train_banks, device)

    seed_r1 = [r["retrieval_at_1"]["mean"] for r in seed_reports]
    seed_r5 = [r["retrieval_at_5"]["mean"] for r in seed_reports]
    learned = {
        "retrieval_at_1_across_seeds": _aggregate(seed_r1),
        "retrieval_at_5_across_seeds": _aggregate(seed_r5),
        "mean_recall5_improvement_vs_naive": float(np.mean(seed_r5) - naive_r5),
        "mean_recall1_improvement_vs_naive": float(np.mean(seed_r1) - naive_r1),
    }
    gate_passed = bool(
        len(eval_banks) >= args.min_eval_scenes
        and learned["mean_recall5_improvement_vs_naive"] >= args.gate_recall5_improvement
        and all(v > naive_r5 for v in seed_r5)
        and np.mean(seed_r1) >= naive_r1
    )
    report = {
        "schema": "lewm_belief_encoder_gate_v0",
        "source_checkpoint": str(args.checkpoint),
        "train_scene_count": len(train_banks),
        "eval_scene_count": len(eval_banks),
        "history": args.history,
        "train_recall_last_seed": train_recall_last_seed,
        "naive_baselines": baseline,
        "naive_bar": {"retrieval_at_5": naive_r5, "retrieval_at_1": naive_r1},
        "seed_reports": seed_reports,
        "learned_summary": learned,
        "gate": {
            "passed": gate_passed,
            "required_recall5_improvement_vs_naive": args.gate_recall5_improvement,
            "min_eval_scenes": args.min_eval_scenes,
            "requires_every_seed_beats_naive_recall5": True,
            "requires_mean_recall1_non_regression": True,
        },
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
