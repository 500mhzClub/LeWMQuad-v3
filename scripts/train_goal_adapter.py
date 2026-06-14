#!/usr/bin/env python3
"""Stage 3 Unit A — train + gate the GoalAdapter (§5.2) on the cached yaw banks.

The adapter maps a frozen single-frame LeWM latent (a goal image) into the v6
BeliefEncoder retrieval space, so Level-1 goal matching can score a goal image
against memory-node embeddings. Trained cross-modally (see
``lewm/models/goal_adapter.py``); the v6 BeliefEncoder stays frozen.

Gate (registered before running; recorded relaxation of the spec §5.2 ≥15 pp,
which was written pre-evidence — the whole encoder's margin over single-frame
is ~5 pp, so 15 pp for the adapter alone is not a meaningful bar):
  - eval goal->window retrieval R@5 (cell-level, self-pair excluded) beats the
    frozen single-frame cosine goal baseline by >= +5 pp, all 3 seeds;
  - view-level ((cell, yaw_bin)) R@1 non-regression vs the same baseline.
Report alongside: the belief->belief oracle (window's own belief embedding as
query) as the alignment ceiling.
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

from lewm.models.goal_adapter import GoalAdapter, masked_cross_modal_supcon  # noqa: E402
from train_loop_closure_head import load_belief_encoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_goal_adapter")


def yaw_masks(bank) -> tuple[torch.Tensor, torch.Tensor]:
    yaws = torch.from_numpy(bank["yaws"])
    diff = (yaws[:, None] - yaws[None, :]).abs()
    same_yaw = torch.minimum(diff, 8 - diff) == 0
    positive = torch.from_numpy(bank["positive"]) & same_yaw
    valid = torch.from_numpy(bank["negative"])
    return positive, valid


@torch.no_grad()
def retrieval(query: torch.Tensor, database: torch.Tensor, cells, yaws, ks=(1, 5)):
    """Self-excluded retrieval; returns per-k cell-level and view-level hit rates."""
    sim = (query @ database.T).cpu()
    sim.fill_diagonal_(-torch.inf)
    out = {}
    order = sim.argsort(dim=1, descending=True)
    cells_t = torch.as_tensor(cells)
    yaws_t = torch.as_tensor(yaws)
    for k in ks:
        top = order[:, :k]
        cell_hit = (cells_t[top] == cells_t[:, None]).any(dim=1).float().mean()
        view_hit = ((cells_t[top] == cells_t[:, None]) & (yaws_t[top] == yaws_t[:, None])).any(dim=1).float().mean()
        out[f"cell_r{k}"] = float(cell_hit)
        out[f"view_r{k}"] = float(view_hit)
    return out


@torch.no_grad()
def evaluate(adapter, encoder, banks, device):
    metrics = {"adapter": [], "baseline": [], "oracle": []}
    for bank in banks:
        z = torch.from_numpy(bank["z_history"]).float().to(device)
        belief = encoder(z)
        goal_latents = z[:, -1]
        queries = {
            "adapter": adapter(goal_latents),
            "baseline": F.normalize(goal_latents, dim=-1),
            "oracle": belief,
        }
        databases = {"adapter": belief, "baseline": F.normalize(goal_latents, dim=-1), "oracle": belief}
        for name in metrics:
            metrics[name].append(retrieval(queries[name], databases[name].clone(), bank["cells"], bank["yaws"]))
    def agg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return {name: agg(rows) for name, rows in metrics.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--belief-encoder", type=Path, required=True)
    parser.add_argument("--yaw-train-banks", type=Path, required=True)
    parser.add_argument("--yaw-eval-banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seeds", default="20260609,20260610,20260611")
    parser.add_argument("--gate-cell-r5-margin", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    encoder = load_belief_encoder(args.belief_encoder, device)
    train_banks = torch.load(args.yaw_train_banks, weights_only=False)
    eval_banks = torch.load(args.yaw_eval_banks, weights_only=False)

    # Precompute per-scene tensors once (belief embeddings are frozen).
    prepared = []
    with torch.no_grad():
        for bank in train_banks:
            positive, valid = yaw_masks(bank)
            negative = valid & ~positive
            if not bool(((positive.any(1)) & (negative.any(1))).any()):
                continue
            z = torch.from_numpy(bank["z_history"]).float().to(device)
            prepared.append((z[:, -1], encoder(z), positive.to(device), valid.to(device)))
    logger.info("train scenes usable: %d/%d; eval scenes: %d", len(prepared), len(train_banks), len(eval_banks))

    head_dir = args.output.parent / f"{args.output.stem}_adapters"
    head_dir.mkdir(parents=True, exist_ok=True)
    latent_dim = int(prepared[0][0].shape[-1])
    embedding_dim = int(prepared[0][1].shape[-1])

    seed_reports = []
    for seed in seeds:
        torch.manual_seed(seed)
        adapter = GoalAdapter(latent_dim, embedding_dim, hidden=args.hidden, dropout=args.dropout).to(device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            adapter.train()
            order = list(range(len(prepared)))
            random.Random(seed + epoch).shuffle(order)
            total = 0.0
            for index in order:
                goal_latents, belief, positive, valid = prepared[index]
                loss = masked_cross_modal_supcon(
                    adapter(goal_latents), belief, positive, valid, temperature=args.temperature
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                logger.info("seed=%d epoch=%d loss=%.4f", seed, epoch, total / len(order))
        adapter.eval()
        report = evaluate(adapter, encoder, eval_banks, device)
        report["seed"] = seed
        seed_reports.append(report)
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
             "latent_dim": latent_dim, "embedding_dim": embedding_dim, "seed": seed,
             "config": {k: str(v) for k, v in vars(args).items()}},
            head_dir / f"goal_adapter_seed{seed}.pt",
        )
        logger.info("seed %d: adapter=%s baseline=%s", seed, report["adapter"], report["baseline"])

    baseline = seed_reports[0]["baseline"]  # deterministic (no trained parts)
    gate_passed = bool(
        all(r["adapter"]["cell_r5"] >= baseline["cell_r5"] + args.gate_cell_r5_margin for r in seed_reports)
        and all(r["adapter"]["view_r1"] >= baseline["view_r1"] for r in seed_reports)
    )
    summary = {
        "schema": "lewm_goal_adapter_gate_v0",
        "belief_encoder": str(args.belief_encoder),
        "seed_reports": seed_reports,
        "baseline": baseline,
        "oracle": seed_reports[0]["oracle"],
        "gate": {
            "passed": gate_passed,
            "required_cell_r5_margin": args.gate_cell_r5_margin,
            "adapter_cell_r5": [r["adapter"]["cell_r5"] for r in seed_reports],
            "baseline_cell_r5": baseline["cell_r5"],
            "adapter_view_r1": [r["adapter"]["view_r1"] for r in seed_reports],
            "baseline_view_r1": baseline["view_r1"],
        },
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
