#!/usr/bin/env python3
"""Probe #3 — offline replay filter test (the Stage 3 buildability question).

Registered (re-ordered before the action/motion probe) in
``docs/lewm_topological_nav_stage3a_loop_closure_2026-06-09.md``: pairwise
loop-closure verification converged at R~0.27 @P90 on frozen seq4 — but the
deployed §5.4 mechanism aggregates per-step likelihoods over consecutive steps
under a transition prior. This probe answers the question at the level that
matters: replay *contiguous held-out rollouts* through the minimal
``OnlineTopologicalMemory`` (view-keyframe nodes = the (cell x yaw-bin) design;
top-k Bayes filter; global novelty commit) and measure the §5.5 gate.

Pipeline (pure torch; no genesis):
  1. Train the same-yaw LoopClosureHead on the yaw-annotated TRAIN banks
     (positives = same (cell, yaw_bin); negatives = BFS>=2 same-yaw), Platt-
     calibrate on held-out calibration scenes; derive candidate tau_new values
     from calibration precision targets (the spec's nominal 0.70 is meaningless
     under a 5%-base-rate calibration).
  2. Build contiguous TRAJECTORY banks for the eval scenes (the Stage-2 banks
     are shuffled/per-cell-capped — useless for replay): one env per scene, a
     contiguous step span, H=8 sliding windows, per-step (cell, yaw_bin) labels
     and a boundary flag (cell-transition frames).
  3. Replay per scene: v6 BeliefEncoder embeds each window; the filter updates;
     each frame is assigned to the MAP node.
  4. Score per §5.5: map nodes to majority labels (§6.1 purity rule);
     **trajectory coherence** on non-boundary frames (cell-level, and the
     stricter (cell,yaw) level), false-merge rate (impure-node fraction),
     fragmentation (reliable nodes / unique true labels) — fragmentation is
     reported, NOT gated (§5.5: false merges are fatal, fragmentation is a
     minor inefficiency).
  5. Ablation: same mechanics with ``uniform_leak=1.0`` (prediction = uniform,
     i.e. per-step likelihood only) isolates the transition prior's value.

Gate (registered): cell-level coherence >= 0.90 on non-boundary frames at some
tau_new. Pass -> the memory is buildable; §5.3's single-pair 99% bar is
re-registered as commit-only. Fail -> probe #2 (action tokens + motion-aux).
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.memory.online_topological_memory import OnlineTopologicalMemory  # noqa: E402
from lewm.models.loop_closure import fit_platt, threshold_at_precision  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import _encode_frames  # noqa: E402
from probe_lewm_reachability_a3 import _select, build_scene_bank  # noqa: E402
from train_loop_closure_head import load_belief_encoder, pooled_scores, train_head  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe_topo_filter_replay")


# ------------------------------------------------------------------------- #
# Same-yaw head training (reuses the Stage 3a machinery + yaw banks)
# ------------------------------------------------------------------------- #


def same_yaw_scenes(banks, encoder, device):
    out = []
    with torch.no_grad():
        for bank in banks:
            z = torch.from_numpy(bank["z_history"]).float().to(device)
            emb = encoder(z)
            positive = torch.from_numpy(bank["positive"])
            negative = torch.from_numpy(bank["negative"])
            yaws = torch.from_numpy(bank["yaws"])
            diff = (yaws[:, None] - yaws[None, :]).abs()
            same = torch.minimum(diff, 8 - diff) == 0
            upper = torch.triu(torch.ones_like(positive), diagonal=1)
            pos = torch.nonzero(positive & same & upper).to(device)
            neg = torch.nonzero(negative & same & upper).to(device)
            if len(pos) and len(neg):
                out.append({"scene_id": bank["scene_id"], "emb": emb, "pos": pos, "neg": neg})
    return out


def train_same_yaw_head(encoder, args, device):
    train_banks = torch.load(args.yaw_train_banks, weights_only=False)
    rng = random.Random(args.seed)
    indices = list(range(len(train_banks)))
    rng.shuffle(indices)
    n_cal = max(8, int(round(0.2 * len(train_banks))))
    cal = [train_banks[i] for i in sorted(set(indices[:n_cal]))]
    head_train = [train_banks[i] for i in indices[n_cal:]]

    class HeadArgs:
        epochs, hidden, dropout, neg_ratio, lr, weight_decay = 30, 128, 0.1, 4, 1e-3, 1e-4

    head = train_head(same_yaw_scenes(head_train, encoder, device), seed=args.seed, args=HeadArgs, device=device)
    cal_logits, cal_labels = pooled_scores(head, same_yaw_scenes(cal, encoder, device), device)
    a, b = fit_platt(cal_logits, cal_labels)
    cal_probs = torch.sigmoid(a * cal_logits + b)
    tau_candidates = {}
    for target in (0.95, 0.90, 0.80, 0.50):
        thr = threshold_at_precision(cal_probs, cal_labels, target)
        if thr is not None:
            tau_candidates[f"calP{int(target * 100)}"] = float(thr)
    logger.info("Platt a=%.3f b=%.3f; tau candidates: %s", a, b, tau_candidates)
    return head, (a, b), tau_candidates


# ------------------------------------------------------------------------- #
# Contiguous trajectory banks
# ------------------------------------------------------------------------- #


def build_trajectory_banks(model, args, device):
    banks = []
    for index, (family, label_file) in enumerate(
        _select(args.rollout_root, args.eval_split, args.eval_scenes_per_family, args.seed)
    ):
        bank = build_scene_bank(
            model, label_file=label_file, family=family, split=args.eval_split,
            render_root=args.render_root, corpus_root=args.manifest_corpus, device=device,
            frames_per_scene=12, max_per_cell=2, batch_size=args.batch_size, min_cells=args.min_cells,
            rng=random.Random(args.seed + index * 7919),
        )
        if bank is None:
            continue
        graph_cells = {n.node_id for n in bank["graph"].manifest.graph_nodes}
        # Env with the most valid labels wins.
        env_idx, observations = max(
            bank["by_env"].items(),
            key=lambda kv: sum(1 for c, y in kv[1] if c in graph_cells and y >= 0),
        )
        steps, paths = [], []
        for step in range(args.history - 1, min(len(observations), args.max_steps + args.history)):
            cell, yaw = observations[step]
            if cell not in graph_cells or yaw < 0:
                continue
            window = [
                bank["render_dir"] / "rgb" /
                f"frame_{(s * bank['n_envs'] + env_idx):06d}_env_{env_idx:02d}.png"
                for s in range(step - args.history + 1, step + 1)
            ]
            if all(p.exists() for p in window):
                steps.append((step, int(cell), int(yaw)))
                paths.append(window)
        if len(steps) < 50:
            continue
        unique = list(dict.fromkeys(p for w in paths for p in w))
        z_unique, _ = _encode_frames(model, unique, device, args.batch_size)
        pindex = {p: i for i, p in enumerate(unique)}
        windows = np.asarray([[pindex[p] for p in w] for w in paths], dtype=np.int64)
        cells = np.asarray([c for _, c, _ in steps], dtype=np.int64)
        yaws = np.asarray([y for _, _, y in steps], dtype=np.int64)
        boundary = np.zeros(len(cells), dtype=bool)
        boundary[1:] |= cells[1:] != cells[:-1]
        boundary[:-1] |= cells[:-1] != cells[1:]
        banks.append({
            "scene_id": bank["scene_id"], "z_history": z_unique[windows].astype(np.float32),
            "cells": cells, "yaws": yaws, "boundary": boundary,
        })
        logger.info("traj %s steps=%d cells=%d boundary_frac=%.2f", bank["scene_id"],
                    len(cells), len(set(cells.tolist())), boundary.mean())
    return banks


# ------------------------------------------------------------------------- #
# Replay + §5.5 scoring
# ------------------------------------------------------------------------- #


def replay_scene(bank, encoder, head, platt, *, tau_new, uniform_leak, device, tau_purity=0.8):
    a, b = platt

    def scorer(query: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        logits = head(query.unsqueeze(0).expand(len(nodes), -1), nodes)
        return torch.sigmoid(a * logits + b)

    memory = OnlineTopologicalMemory(
        scorer, tau_new=tau_new, new_node_streak=3, top_k=8,
        self_stay_prob=0.6, uniform_leak=uniform_leak,
    )
    with torch.no_grad():
        embeddings = encoder(torch.from_numpy(bank["z_history"]).float().to(device))
    assignments = []
    for i in range(len(embeddings)):
        label = (int(bank["cells"][i]), int(bank["yaws"][i]))
        assignments.append(memory.update(embeddings[i], label=label))

    majority = memory.node_majority_labels(tau_purity)
    correct_cell = correct_view = scored = 0
    for i, node in enumerate(assignments):
        if bank["boundary"][i] or node is None or node not in majority:
            continue
        scored += 1
        label, _purity, _reliable = majority[node]
        correct_cell += int(label[0] == int(bank["cells"][i]))
        correct_view += int(label == (int(bank["cells"][i]), int(bank["yaws"][i])))
    n_nodes = len(memory.nodes)
    impure = sum(1 for _, _, reliable in majority.values() if not reliable)
    unique_views = len({(int(c), int(y)) for c, y in zip(bank["cells"], bank["yaws"])})
    return {
        "scene_id": bank["scene_id"],
        "n_steps": len(assignments),
        "n_scored": scored,
        "coherence_cell": correct_cell / max(scored, 1),
        "coherence_view": correct_view / max(scored, 1),
        "n_nodes": n_nodes,
        "false_merge_rate": impure / max(len(majority), 1),
        "fragmentation": n_nodes / max(unique_views, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--belief-encoder", type=Path, required=True,
                        help="a single belief_encoder_seed*.pt (v6)")
    parser.add_argument("--yaw-train-banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traj-cache", type=Path, default=None)
    parser.add_argument("--rollout-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/rollout")
    parser.add_argument("--render-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    parser.add_argument("--manifest-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    parser.add_argument("--eval-split", default="test_id")
    parser.add_argument("--eval-scenes-per-family", type=int, default=4)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--gate-coherence", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    encoder = load_belief_encoder(args.belief_encoder, device)
    head, platt, tau_candidates = train_same_yaw_head(encoder, args, device)

    if args.traj_cache is not None and args.traj_cache.exists():
        logger.info("Loading cached trajectory banks from %s", args.traj_cache)
        banks = torch.load(args.traj_cache, weights_only=False)
    else:
        model, _ = load_model(args, device)
        banks = build_trajectory_banks(model, args, device)
        if args.traj_cache is not None:
            args.traj_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(banks, args.traj_cache)
    if not banks:
        raise SystemExit("no usable trajectory banks")

    results = {}
    for tau_name, tau_value in tau_candidates.items():
        for mode, leak in (("filter", 0.05), ("no_prior", 1.0)):
            per_scene = [replay_scene(b, encoder, head, platt, tau_new=tau_value,
                                      uniform_leak=leak, device=device) for b in banks]
            agg = {
                "tau_new": tau_value,
                "mean_coherence_cell": float(np.mean([s["coherence_cell"] for s in per_scene])),
                "median_coherence_cell": float(np.median([s["coherence_cell"] for s in per_scene])),
                "mean_coherence_view": float(np.mean([s["coherence_view"] for s in per_scene])),
                "mean_false_merge_rate": float(np.mean([s["false_merge_rate"] for s in per_scene])),
                "mean_fragmentation": float(np.mean([s["fragmentation"] for s in per_scene])),
                "mean_nodes": float(np.mean([s["n_nodes"] for s in per_scene])),
                "per_scene": per_scene,
            }
            results[f"{tau_name}/{mode}"] = agg
            logger.info("%s/%s: coh_cell=%.3f coh_view=%.3f false_merge=%.3f frag=%.2f nodes=%.1f",
                        tau_name, mode, agg["mean_coherence_cell"], agg["mean_coherence_view"],
                        agg["mean_false_merge_rate"], agg["mean_fragmentation"], agg["mean_nodes"])

    best = max((r for k, r in results.items() if k.endswith("/filter")),
               key=lambda r: r["mean_coherence_cell"])
    gate_passed = bool(best["mean_coherence_cell"] >= args.gate_coherence)
    report = {
        "schema": "lewm_topo_filter_replay_v0",
        "belief_encoder": str(args.belief_encoder),
        "n_scenes": len(banks),
        "tau_candidates": tau_candidates,
        "results": results,
        "gate": {
            "passed": gate_passed,
            "required_mean_coherence_cell": args.gate_coherence,
            "best_filter_coherence_cell": best["mean_coherence_cell"],
            "best_tau_new": best["tau_new"],
        },
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": report["gate"]}, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
