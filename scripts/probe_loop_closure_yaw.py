#!/usr/bin/env python3
"""Stage 3a follow-up — is loop-closure verification yaw-limited?

Registered probe #1 of the Stage 3a reassessment
(``docs/lewm_topological_nav_stage3a_loop_closure_2026-06-09.md``): pairwise
any-yaw place verification is too weak at every precision on frozen seq4 (and
the v6 BeliefEncoder lifts it 2-3x but not into a usable band). The prime
suspect is the heading-dominated latent (yaw R^2 0.81 vs pos 0.16): retrieval
tolerates it (rank-based), verification cannot.

This probe rebuilds the held-out eval banks with the per-window **terminal
yaw_bin** (8 bins, already in ``labels.jsonl``; dropped by the Stage 1/2 window
selector) and compares verification PR under three pair scopes:

  - ``all``        — any relative yaw (replicates the Stage 3a condition);
  - ``same_yaw``   — both windows in the same yaw bin (the operating
                     distribution of a (cell x yaw-bin) keyframe memory);
  - ``adjacent_yaw`` — within +/-1 bin (45deg tolerance).

Positives AND negatives are both restricted per scope — under a (cell x
yaw-bin) node design the negatives are different cells seen at the same
heading, i.e. the classic aliased-corridor cases, so this is not a giveaway.
Base rates per scope are reported alongside. Scorers are training-free cosine
(single-frame terminal, mean-pool, v6 belief encoders x3 seeds): the Stage 3a
trained head added little over cosine (AP 0.286 vs 0.279), so cosine curves
answer the diagnostic without retraining heads.

Decision rule (registered): if same-yaw verification reaches a usable band
(recall >= ~0.3 at P>=0.95 for belief), adopt (cell x yaw-bin) keyframe nodes
in the Stage 3 memory design (converging with the goal-facing-keyframe
constraint); cross-yaw same-place association then comes from graph edges, not
visual verification. If it stays flat, yaw is NOT the limiter -> escalate to
probe #2 (action-token + motion-aux encoder) / substrate fork for the memory key.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.loop_closure import precision_recall_at, threshold_at_precision  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import _encode_frames  # noqa: E402
from probe_lewm_reachability_a3 import _select, build_scene_bank  # noqa: E402
from train_loop_closure_head import load_belief_encoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe_loop_closure_yaw")

PRECISION_TARGETS = (0.99, 0.95, 0.90, 0.80, 0.50)


def _select_history_windows_with_yaw(bank, *, history, windows_per_scene, max_per_cell, rng):
    """Stage 1 `_select_history_windows` + the terminal yaw_bin it drops."""
    graph_cells = {node.node_id for node in bank["graph"].manifest.graph_nodes}
    by_cell = defaultdict(list)
    for env_index, observations in bank["by_env"].items():
        for step in range(history - 1, len(observations)):
            terminal_cell, terminal_yaw = observations[step]
            if terminal_cell not in graph_cells or terminal_yaw < 0:
                continue
            paths = []
            for history_step in range(step - history + 1, step + 1):
                global_index = history_step * bank["n_envs"] + env_index
                paths.append(bank["render_dir"] / "rgb" / f"frame_{global_index:06d}_env_{env_index:02d}.png")
            if all(path.exists() for path in paths):
                by_cell[int(terminal_cell)].append((paths, int(terminal_yaw)))
    selected = []
    for cell, windows in by_cell.items():
        rng.shuffle(windows)
        selected.extend((paths, cell, yaw) for paths, yaw in windows[:max_per_cell])
    rng.shuffle(selected)
    selected = selected[:windows_per_scene]
    return (
        [paths for paths, _cell, _yaw in selected],
        np.asarray([cell for _paths, cell, _yaw in selected], dtype=np.int64),
        np.asarray([yaw for _paths, _cell, yaw in selected], dtype=np.int64),
    )


def build_yaw_banks(model, split, per_family, args, device):
    banks = []
    for index, (family, label_file) in enumerate(_select(args.rollout_root, split, per_family, args.seed)):
        bank = build_scene_bank(
            model, label_file=label_file, family=family, split=split,
            render_root=args.render_root, corpus_root=args.manifest_corpus, device=device,
            frames_per_scene=max(12, args.min_cells * 2), max_per_cell=2,
            batch_size=args.batch_size, min_cells=args.min_cells,
            rng=random.Random(args.seed + index * 7919),
        )
        if bank is None:
            continue
        windows, cells, yaws = _select_history_windows_with_yaw(
            bank, history=args.history, windows_per_scene=args.windows_per_scene,
            max_per_cell=args.max_per_cell, rng=random.Random(args.seed + index * 104729),
        )
        if len(set(cells.tolist())) < args.min_cells or len(windows) < 12:
            continue
        unique_paths = list(dict.fromkeys(path for window in windows for path in window))
        z_unique, _ = _encode_frames(model, unique_paths, device, args.batch_size)
        path_index = {path: offset for offset, path in enumerate(unique_paths)}
        window_index = np.asarray([[path_index[p] for p in w] for w in windows], dtype=np.int64)
        # Valid-negative mask via BFS >= 2 (same scheme as the Stage 2 banks).
        graph = bank["graph"]
        unique_cells = sorted({int(c) for c in cells})
        far = {(a, b) for i, a in enumerate(unique_cells) for b in unique_cells[i + 1:]
               if (d := graph.bfs_distance(a, b)) is not None and d >= 2}
        far |= {(b, a) for a, b in far}
        positive = (cells[:, None] == cells[None, :])
        np.fill_diagonal(positive, False)
        negative = np.zeros_like(positive)
        for i, ca in enumerate(cells):
            for j, cb in enumerate(cells):
                if (int(ca), int(cb)) in far:
                    negative[i, j] = True
        banks.append({
            "scene_id": bank["scene_id"], "z_history": z_unique[window_index].astype(np.float32),
            "cells": cells, "yaws": yaws, "positive": positive, "negative": negative,
        })
        logger.info("bank %s windows=%d cells=%d yaw_bins=%d", bank["scene_id"], len(windows),
                    len(set(cells.tolist())), len(set(yaws.tolist())))
    return banks


def curve(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels.float()[order]
    cum_tp = sorted_labels.cumsum(0)
    counts = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float64)
    ap = float((cum_tp.double() / counts)[sorted_labels.bool()].mean()) if labels.sum() else 0.0
    out = {"n_pairs": int(len(labels)), "base_rate": float(labels.float().mean()), "average_precision": ap}
    for target in PRECISION_TARGETS:
        thr = threshold_at_precision(scores, labels, target)
        recall = 0.0 if thr is None else precision_recall_at(scores, labels, thr)[1]
        out[f"recall_at_p{int(target * 100)}"] = recall
    return out


@torch.no_grad()
def probe(banks, representation, device, belief_encoder=None) -> dict:
    pooled = {scope: {"scores": [], "labels": []} for scope in ("all", "same_yaw", "adjacent_yaw")}
    for bank in banks:
        z = torch.from_numpy(bank["z_history"]).float().to(device)
        if representation == "single_frame":
            emb = F.normalize(z[:, -1], dim=-1)
        elif representation == "mean_pool":
            emb = F.normalize(z.mean(dim=1), dim=-1)
        else:
            emb = belief_encoder(z)
        sim = (emb @ emb.T).cpu()
        positive = torch.from_numpy(bank["positive"])
        negative = torch.from_numpy(bank["negative"])
        yaws = torch.from_numpy(bank["yaws"])
        diff = (yaws[:, None] - yaws[None, :]).abs()
        circ = torch.minimum(diff, 8 - diff)
        upper = torch.triu(torch.ones_like(positive), diagonal=1)
        scopes = {
            "all": torch.ones_like(positive),
            "same_yaw": circ == 0,
            "adjacent_yaw": circ <= 1,
        }
        for name, scope in scopes.items():
            mask = (positive | negative) & scope & upper
            idx = torch.nonzero(mask)
            if len(idx) == 0:
                continue
            pooled[name]["scores"].append(sim[idx[:, 0], idx[:, 1]])
            pooled[name]["labels"].append(positive[idx[:, 0], idx[:, 1]].float())
    return {name: curve(torch.cat(v["scores"]), torch.cat(v["labels"]))
            for name, v in pooled.items() if v["scores"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--belief-encoder-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-cache", type=Path, default=None)
    parser.add_argument("--rollout-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/rollout")
    parser.add_argument("--render-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    parser.add_argument("--manifest-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    parser.add_argument("--eval-split", default="test_id")
    parser.add_argument("--eval-scenes-per-family", type=int, default=4)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--windows-per-scene", type=int, default=160)
    parser.add_argument("--max-per-cell", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.bank_cache is not None and args.bank_cache.exists():
        logger.info("Loading cached yaw banks from %s", args.bank_cache)
        banks = torch.load(args.bank_cache, weights_only=False)
    else:
        model, _ = load_model(args, device)
        banks = build_yaw_banks(model, args.eval_split, args.eval_scenes_per_family, args, device)
        if args.bank_cache is not None:
            args.bank_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(banks, args.bank_cache)
            logger.info("Cached yaw banks to %s", args.bank_cache)
    if not banks:
        raise SystemExit("no usable eval banks")

    results: dict[str, dict] = {}
    for representation in ("single_frame", "mean_pool"):
        results[representation] = probe(banks, representation, device)
        logger.info("%s: %s", representation, json.dumps(results[representation], indent=None))
    for path in sorted(args.belief_encoder_dir.glob("belief_encoder_seed*.pt")):
        encoder = load_belief_encoder(path, device)
        results[f"belief[{path.stem}]"] = probe(banks, "belief", device, encoder)
        logger.info("%s: %s", path.stem, json.dumps(results[f"belief[{path.stem}]"], indent=None))

    report = {
        "schema": "lewm_loop_closure_yaw_probe_v0",
        "eval_scene_count": len(banks),
        "precision_targets": list(PRECISION_TARGETS),
        "results": results,
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
