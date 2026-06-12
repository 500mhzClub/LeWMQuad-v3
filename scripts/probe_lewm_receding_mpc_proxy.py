#!/usr/bin/env python3
"""Offline receding-horizon proxy for LeWM local MPC usefulness.

This is not a simulator rollout and it is not a full CEM benchmark. It answers a
more bounded question: when the real observation is re-encoded at every macro
step, does the model assign lower terminal latent cost to the recorded action
sequence than to simple corruptions such as zero or batch-shuffled actions?

That is the offline proxy closest to the future LocalMPC role: use LeWM to rank
short local candidate action sequences, execute one block, observe again, and
replan from the fresh encoded observation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lewm_checkpoint import load_model  # noqa: E402
from train_lewm import GenesisWMDataset, make_loader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or horizons[0] < 1:
        raise argparse.ArgumentTypeError("horizons must contain positive integers")
    return horizons


def mean_or_none(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return total / count


def ratio_or_none(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0.0):
        return None
    return num / den


def _mean_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean(dim=-1)


@torch.no_grad()
def probe_receding_proxy(
    model,
    loader,
    device: torch.device,
    *,
    horizons: list[int],
    replan_steps: int,
    max_batches: int,
    precision: str,
    use_history: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    model.eval()
    max_horizon = max(horizons)
    autocast_enabled = precision == "bf16" and device.type == "cuda"
    totals = {
        horizon: {
            "recorded": 0.0,
            "zero": 0.0,
            "shuffled": 0.0,
            "persistence": 0.0,
            "recorded_wins_zero": 0,
            "recorded_wins_shuffled": 0,
            "recorded_wins_persistence": 0,
            "recorded_top1": 0,
            "count": 0,
        }
        for horizon in horizons
    }
    evaluated_samples = 0
    evaluated_decisions = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        vis = batch["vis_seq"].to(device, non_blocking=False)
        cmd = batch["cmd_seq"].to(device, non_blocking=False)
        batch_size = int(vis.shape[0])

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            z_raw, z_proj = model.encode_seq(vis, prop_seq=None)

            for step in range(replan_steps):
                z_start = z_raw[:, step]
                if use_history:
                    hist_start = max(0, step - model.predictor.max_seq_len + 1)
                    z_history = z_raw[:, hist_start : step + 1]
                    action_history = cmd[:, hist_start:step] if step > hist_start else None
                else:
                    z_history = None
                    action_history = None

                for horizon in horizons:
                    actions = cmd[:, step : step + horizon]
                    target = z_proj[:, step + horizon]
                    recorded_rollout = model.plan_rollout(
                        z_start,
                        actions,
                        z_history_raw=z_history,
                        action_history=action_history,
                    )[:, -1]
                    zero_rollout = model.plan_rollout(
                        z_start,
                        torch.zeros_like(actions),
                        z_history_raw=z_history,
                        action_history=action_history,
                    )[:, -1]
                    shuffled_actions = (
                        actions.roll(shifts=1, dims=0)
                        if batch_size > 1
                        else torch.zeros_like(actions)
                    )
                    shuffled_rollout = model.plan_rollout(
                        z_start,
                        shuffled_actions,
                        z_history_raw=z_history,
                        action_history=action_history,
                    )[:, -1]

                    recorded_mse = _mean_last_dim(recorded_rollout - target)
                    zero_mse = _mean_last_dim(zero_rollout - target)
                    shuffled_mse = _mean_last_dim(shuffled_rollout - target)
                    persistence_mse = _mean_last_dim(z_proj[:, step] - target)
                    stacked = torch.stack((recorded_mse, zero_mse, shuffled_mse), dim=0)

                    item = totals[horizon]
                    item["recorded"] += float(recorded_mse.double().sum().cpu())
                    item["zero"] += float(zero_mse.double().sum().cpu())
                    item["shuffled"] += float(shuffled_mse.double().sum().cpu())
                    item["persistence"] += float(persistence_mse.double().sum().cpu())
                    item["recorded_wins_zero"] += int((recorded_mse < zero_mse).sum().cpu())
                    item["recorded_wins_shuffled"] += int(
                        (recorded_mse < shuffled_mse).sum().cpu()
                    )
                    item["recorded_wins_persistence"] += int(
                        (recorded_mse < persistence_mse).sum().cpu()
                    )
                    item["recorded_top1"] += int((stacked.argmin(dim=0) == 0).sum().cpu())
                    item["count"] += batch_size
                    evaluated_decisions += batch_size

        evaluated_samples += batch_size

    reports: list[dict[str, Any]] = []
    for horizon in horizons:
        item = totals[horizon]
        count = int(item["count"])
        recorded = mean_or_none(float(item["recorded"]), count)
        zero = mean_or_none(float(item["zero"]), count)
        shuffled = mean_or_none(float(item["shuffled"]), count)
        persistence = mean_or_none(float(item["persistence"]), count)
        reports.append(
            {
                "horizon": horizon,
                "terminal_mse": {
                    "recorded": recorded,
                    "zero_action": zero,
                    "shuffled_action": shuffled,
                    "persistence": persistence,
                },
                "recorded_over_zero": ratio_or_none(recorded, zero),
                "recorded_over_shuffled": ratio_or_none(recorded, shuffled),
                "recorded_over_persistence": ratio_or_none(recorded, persistence),
                "recorded_win_rate_vs_zero": mean_or_none(
                    int(item["recorded_wins_zero"]), count
                ),
                "recorded_win_rate_vs_shuffled": mean_or_none(
                    int(item["recorded_wins_shuffled"]), count
                ),
                "recorded_win_rate_vs_persistence": mean_or_none(
                    int(item["recorded_wins_persistence"]), count
                ),
                "recorded_top1_rate_among_action_candidates": mean_or_none(
                    int(item["recorded_top1"]), count
                ),
                "decisions": count,
            }
        )
    return reports, evaluated_samples, evaluated_decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--horizons", type=parse_horizons, default=parse_horizons("1,2,3"))
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.02)
    parser.add_argument("--holdout-role", choices=("all", "train", "eval"), default="eval")
    parser.add_argument("--holdout-seed", type=int, default=20260524)
    parser.add_argument("--sample-seed", type=int, default=20260601)
    parser.add_argument("--allow-material-color-render", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-history", action="store_true")
    args = parser.parse_args()

    if args.replan_steps < 1:
        raise SystemExit("--replan-steps must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Running receding MPC proxy on %s", device)

    model, model_config = load_model(args, device)
    max_horizon = max(args.horizons)
    render_root = args.render_root or (
        Path(model_config["render_root"]) if "render_root" in model_config else None
    )
    allow_material = bool(
        args.allow_material_color_render
        or model_config.get("allow_material_color_render", False)
    )
    dataset_seq_len = args.replan_steps + max_horizon
    dataset = GenesisWMDataset(
        root_dir=args.data_root,
        render_root=render_root,
        seq_len=dataset_seq_len,
        stride=args.stride,
        max_sessions=args.max_sessions,
        allow_material_color_render=allow_material,
        holdout_fraction=args.holdout_fraction,
        holdout_role=args.holdout_role,
        holdout_seed=args.holdout_seed,
    )
    if len(dataset) == 0:
        raise SystemExit("proxy dataset is empty")

    requested_samples = args.max_batches * args.batch_size if args.max_batches > 0 else len(dataset)
    sampled_count = min(len(dataset), requested_samples)
    rng = random.Random(args.sample_seed)
    sampled_indices = rng.sample(range(len(dataset)), sampled_count)
    sampled_dataset = Subset(dataset, sampled_indices)
    loader = make_loader(
        sampled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=3,
    )
    reports, evaluated_samples, evaluated_decisions = probe_receding_proxy(
        model,
        loader,
        device,
        horizons=args.horizons,
        replan_steps=args.replan_steps,
        max_batches=args.max_batches,
        precision=args.precision,
        use_history=not args.no_history,
    )
    record = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "render_root": str(dataset.render_root),
        "trained_max_seq_len": int(model.predictor.max_seq_len),
        "dataset_seq_len": dataset_seq_len,
        "stride": args.stride,
        "approx_seconds_per_macro_step": 0.5 if args.stride == 5 else None,
        "horizons": reports,
        "replan_steps": args.replan_steps,
        "use_history": not args.no_history,
        "holdout_fraction": args.holdout_fraction,
        "holdout_role": args.holdout_role,
        "holdout_seed": args.holdout_seed,
        "sample_seed": args.sample_seed,
        "dataset_sequences": len(dataset),
        "evaluated_samples": evaluated_samples,
        "evaluated_decisions": evaluated_decisions,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
