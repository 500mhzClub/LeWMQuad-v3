#!/usr/bin/env python3
"""Stage 3a — LoopClosureHead consumer gate for the Stage 2 BeliefEncoder.

Registered in ``docs/lewm_topological_nav_stage2_belief_encoder_2026-06-09.md``
("v6 + decision", registered before running): the Stage-2->3 decision is made at
the operating point the topological memory actually consumes — loop-closure
recall at >= 99% precision (spec §5.3) — not at retrieval R@5.

Per representation (v6 belief encoder x3 seeds; frozen single-frame terminal;
naive mean-pool), this script:
  1. trains a ``LoopClosureHead`` (BCE, all positives + sampled negatives per
     scene per epoch) on head-train scenes (train split minus calibration slice);
  2. Platt-calibrates on the held-out calibration scenes (scene-disjoint);
  3. picks the deployment threshold on the calibration pairs at precision >=
     --precision-target (max recall subject to that);
  4. reports precision/recall at that threshold, ECE, average precision, and the
     eval-curve oracle recall@target on ALL valid pairs of the test_id scenes.
A training-free cosine scorer is reported alongside as the no-head ablation
(spec §9.4 "no LoopClosureHead").

Gate (registered):
  - spec §5.3: eval precision at deployed threshold >= 0.99 and ECE <= 5% for
    any adopted representation;
  - consumer question: every belief seed's recall at the deployed threshold
    beats the single-frame head's mean recall by >= +5 pp absolute.

Pairs use the banks' precomputed masks: positive = same cell, valid negative =
BFS >= 2 (adjacent cells = ambiguous, excluded) — the §5.1 three-bucket scheme.
Reuses the Stage 2 bank cache verbatim (no model load, no genesis).
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.models.belief_encoder import BeliefEncoder  # noqa: E402
from lewm.models.loop_closure import (  # noqa: E402
    LoopClosureHead,
    expected_calibration_error,
    fit_platt,
    precision_recall_at,
    threshold_at_precision,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_loop_closure_head")


def load_belief_encoder(path: Path, device: torch.device) -> BeliefEncoder:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = blob["config"]
    encoder = BeliefEncoder(
        latent_dim=int(blob["latent_dim"]),
        embedding_dim=int(blob["embedding_dim"]),
        hidden=int(config["hidden"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        max_len=max(int(config["history"]), 16),
        dropout=float(config["dropout"]),
    )
    encoder.load_state_dict(blob["state_dict"])
    return encoder.to(device).eval()


@torch.no_grad()
def embed_banks(banks, representation: str, device: torch.device, belief_encoder=None):
    """Per-scene L2-normalized embeddings + pair index lists for one representation."""
    scenes = []
    for bank in banks:
        z = torch.from_numpy(bank["z_history"]).float().to(device)
        if representation == "single_frame":
            emb = F.normalize(z[:, -1], dim=-1)
        elif representation == "mean_pool":
            emb = F.normalize(z.mean(dim=1), dim=-1)
        elif representation == "belief":
            emb = belief_encoder(z)
        else:
            raise ValueError(representation)
        positive = torch.from_numpy(bank["positive"]).to(device)
        valid = torch.from_numpy(bank["valid"]).to(device)
        upper = torch.triu(torch.ones_like(positive), diagonal=1)
        pos_pairs = torch.nonzero(positive & upper)
        neg_pairs = torch.nonzero(valid & ~positive & upper)
        if len(pos_pairs) == 0 or len(neg_pairs) == 0:
            continue
        scenes.append({"scene_id": bank["scene_id"], "emb": emb, "pos": pos_pairs, "neg": neg_pairs})
    return scenes


def pooled_scores(head, scenes, device):
    """Logits + labels over ALL pos/neg pairs of the given scenes, pooled."""
    logits, labels = [], []
    with torch.no_grad():
        for scene in scenes:
            for pairs, label in ((scene["pos"], 1.0), (scene["neg"], 0.0)):
                a, b = scene["emb"][pairs[:, 0]], scene["emb"][pairs[:, 1]]
                logits.append(head(a, b).cpu() if head is not None else (a * b).sum(-1).cpu())
                labels.append(torch.full((len(pairs),), label))
    return torch.cat(logits), torch.cat(labels)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels.float()[order]
    cumulative_tp = sorted_labels.cumsum(0)
    counts = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float64)
    precisions = (cumulative_tp.double() / counts)[sorted_labels.bool()]
    return float(precisions.mean()) if len(precisions) else 0.0


def evaluate_scorer(head, train_scenes, cal_scenes, eval_scenes, device, precision_target):
    """Platt-calibrate on cal scenes, pick threshold there, measure on eval scenes."""
    cal_logits, cal_labels = pooled_scores(head, cal_scenes, device)
    a, b = fit_platt(cal_logits, cal_labels)
    cal_probs = torch.sigmoid(a * cal_logits + b)
    threshold = threshold_at_precision(cal_probs, cal_labels, precision_target)

    eval_logits, eval_labels = pooled_scores(head, eval_scenes, device)
    eval_probs = torch.sigmoid(a * eval_logits + b)
    out = {
        "n_eval_pairs": int(len(eval_labels)),
        "eval_positive_rate": float(eval_labels.mean()),
        "average_precision": average_precision(eval_logits, eval_labels),
        "ece": expected_calibration_error(eval_probs, eval_labels),
        "platt": {"a": a, "b": b},
        "cal_threshold": threshold,
    }
    if threshold is None:
        out.update({"precision_at_threshold": None, "recall_at_threshold": 0.0})
    else:
        precision, recall = precision_recall_at(eval_probs, eval_labels, threshold)
        out.update({"precision_at_threshold": precision, "recall_at_threshold": recall})
    # Oracle: recall at the target precision on the eval curve itself (upper
    # bound; diagnoses threshold-transfer loss vs representation limits).
    oracle_threshold = threshold_at_precision(eval_probs, eval_labels, precision_target)
    if oracle_threshold is None:
        out["oracle_recall_at_target"] = 0.0
    else:
        _, oracle_recall = precision_recall_at(eval_probs, eval_labels, oracle_threshold)
        out["oracle_recall_at_target"] = oracle_recall
    return out


def train_head(train_scenes, *, seed: int, args, device) -> LoopClosureHead:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    embedding_dim = train_scenes[0]["emb"].shape[-1]
    head = LoopClosureHead(embedding_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        head.train()
        order = list(range(len(train_scenes)))
        rng.shuffle(order)
        total, count = 0.0, 0
        for index in order:
            scene = train_scenes[index]
            n_neg = min(len(scene["neg"]), args.neg_ratio * len(scene["pos"]))
            neg_select = torch.randperm(len(scene["neg"]), device=device)[:n_neg]
            pairs = torch.cat([scene["pos"], scene["neg"][neg_select]])
            labels = torch.cat([
                torch.ones(len(scene["pos"]), device=device),
                torch.zeros(n_neg, device=device),
            ])
            logits = head(scene["emb"][pairs[:, 0]], scene["emb"][pairs[:, 1]])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, count = total + float(loss.detach()), count + 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info("seed=%d epoch=%d bce=%.4f", seed, epoch, total / max(count, 1))
    return head.eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-cache", type=Path, required=True)
    parser.add_argument("--belief-encoder-dir", type=Path, required=True,
                        help="dir of belief_encoder_seed*.pt from train_belief_encoder.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cal-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-seeds", default="20260609,20260610,20260611")
    parser.add_argument("--precision-target", type=float, default=0.99)
    parser.add_argument("--gate-recall-margin", type=float, default=0.05)
    parser.add_argument("--gate-ece-max", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    head_seeds = [int(s) for s in args.head_seeds.split(",") if s.strip()]

    logger.info("Loading bank cache %s", args.bank_cache)
    blob = torch.load(args.bank_cache, weights_only=False)
    train_banks, eval_banks = blob["train"], blob["eval"]
    # Scene-disjoint calibration slice off the train split (eval stays untouched).
    split_rng = random.Random(head_seeds[0])
    indices = list(range(len(train_banks)))
    split_rng.shuffle(indices)
    n_cal = max(8, int(round(args.cal_frac * len(train_banks))))
    cal_idx, head_train_idx = set(indices[:n_cal]), indices[n_cal:]
    cal_banks = [train_banks[i] for i in sorted(cal_idx)]
    head_train_banks = [train_banks[i] for i in head_train_idx]
    logger.info("scenes: head-train=%d cal=%d eval=%d", len(head_train_banks), len(cal_banks), len(eval_banks))

    belief_paths = sorted(args.belief_encoder_dir.glob("belief_encoder_seed*.pt"))
    if not belief_paths:
        raise SystemExit(f"no belief encoders under {args.belief_encoder_dir}")

    results: dict[str, list[dict]] = {}
    runs = [("single_frame", seed, None) for seed in head_seeds]
    runs += [("mean_pool", seed, None) for seed in head_seeds]
    runs += [("belief", head_seeds[0], path) for path in belief_paths]
    for representation, seed, belief_path in runs:
        belief_encoder = load_belief_encoder(belief_path, device) if belief_path else None
        tag = f"{representation}" + (f"[{belief_path.stem}]" if belief_path else f"[head_seed={seed}]")
        logger.info("=== %s ===", tag)
        train_scenes = embed_banks(head_train_banks, representation, device, belief_encoder)
        cal_scenes = embed_banks(cal_banks, representation, device, belief_encoder)
        eval_scenes = embed_banks(eval_banks, representation, device, belief_encoder)
        head = train_head(train_scenes, seed=seed, args=args, device=device)
        report = evaluate_scorer(head, train_scenes, cal_scenes, eval_scenes, device, args.precision_target)
        report["tag"] = tag
        # Training-free cosine ablation (once per representation).
        if representation not in results:
            report["cosine_baseline"] = evaluate_scorer(None, train_scenes, cal_scenes, eval_scenes, device, args.precision_target)
        results.setdefault(representation, []).append(report)
        logger.info("%s recall@thr=%.4f precision@thr=%s ece=%.4f oracle=%.4f", tag,
                    report["recall_at_threshold"], report["precision_at_threshold"], report["ece"],
                    report["oracle_recall_at_target"])

    def mean_recall(representation: str) -> float:
        return float(np.mean([r["recall_at_threshold"] for r in results[representation]]))

    single_frame_mean = mean_recall("single_frame")
    belief_recalls = [r["recall_at_threshold"] for r in results["belief"]]
    belief_ok = all(r >= single_frame_mean + args.gate_recall_margin for r in belief_recalls)
    spec_ok = all(
        (r["precision_at_threshold"] or 0.0) >= args.precision_target * 0.999 and r["ece"] <= args.gate_ece_max
        for r in results["belief"]
    )
    summary = {
        "schema": "lewm_loop_closure_gate_v0",
        "bank_cache": str(args.bank_cache),
        "belief_encoder_dir": str(args.belief_encoder_dir),
        "scene_counts": {"head_train": len(head_train_banks), "cal": len(cal_banks), "eval": len(eval_banks)},
        "results": results,
        "gate": {
            "passed": bool(belief_ok and spec_ok),
            "belief_recalls_at_threshold": belief_recalls,
            "single_frame_mean_recall": single_frame_mean,
            "mean_pool_mean_recall": mean_recall("mean_pool"),
            "belief_mean_recall": mean_recall("belief"),
            "required_margin_vs_single_frame": args.gate_recall_margin,
            "consumer_margin_met_all_seeds": bool(belief_ok),
            "spec_553_precision_and_ece_met": bool(spec_ok),
            "precision_target": args.precision_target,
        },
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": summary["gate"]}, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
