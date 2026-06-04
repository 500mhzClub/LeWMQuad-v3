#!/usr/bin/env python3
"""Bounded long-horizon rollout diagnostics for a trained LeWM checkpoint.

This probe is intentionally separate from the in-training evaluation. Training
samples only expose ``max_seq_len - 1`` target transitions, while planning may
roll the predictor forward for substantially longer horizons using its sliding
context window. The report compares the learned rollout against persistence,
zero-action, and shuffled-action baselines on a deterministic random holdout
sample.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

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


def mean_or_none(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def delta_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


@torch.no_grad()
def probe_rollout_horizons(
    model,
    loader,
    device: torch.device,
    *,
    horizons: list[int],
    max_batches: int,
    precision: str,
) -> tuple[list[dict[str, float | int | None]], int]:
    max_horizon = max(horizons)
    sums = {
        name: torch.zeros(max_horizon, dtype=torch.float64)
        for name in (
            "rollout",
            "persistence",
            "zero_action",
            "shuffled_action",
            "target_from_start",
            "target_step_delta",
        )
    }
    sample_count = 0
    autocast_enabled = precision == "bf16" and device.type == "cuda"
    model.eval()

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
            actions = cmd[:, :max_horizon]
            targets = z_proj[:, 1 : max_horizon + 1]
            rollout = model.plan_rollout(z_raw[:, 0], actions)
            zero_rollout = model.plan_rollout(z_raw[:, 0], torch.zeros_like(actions))
            shuffled_rollout = model.plan_rollout(
                z_raw[:, 0],
                actions.roll(shifts=1, dims=0) if batch_size > 1 else torch.zeros_like(actions),
            )
            persistence = z_proj[:, :1].expand_as(targets)
            previous_targets = z_proj[:, :max_horizon]

            batch_metrics = {
                "rollout": (rollout - targets).square().mean(dim=-1),
                "persistence": (persistence - targets).square().mean(dim=-1),
                "zero_action": (zero_rollout - targets).square().mean(dim=-1),
                "shuffled_action": (shuffled_rollout - targets).square().mean(dim=-1),
                "target_from_start": (targets - persistence).square().mean(dim=-1),
                "target_step_delta": (targets - previous_targets).square().mean(dim=-1),
            }

        for name, values in batch_metrics.items():
            sums[name] += values.detach().double().sum(dim=0).cpu()
        sample_count += batch_size

    reports = []
    for horizon in horizons:
        index = horizon - 1
        point = {
            name: mean_or_none(float(values[index]), sample_count)
            for name, values in sums.items()
        }
        cumulative = {
            name: mean_or_none(float(values[:horizon].sum()), sample_count * horizon)
            for name, values in sums.items()
        }
        reports.append(
            {
                "horizon": horizon,
                "point_mse": point,
                "cumulative_mse": cumulative,
                "point_rollout_over_persistence": ratio_or_none(
                    point["rollout"],
                    point["persistence"],
                ),
                "point_shuffled_minus_rollout": delta_or_none(
                    point["shuffled_action"],
                    point["rollout"],
                ),
                "point_zero_minus_rollout": delta_or_none(
                    point["zero_action"],
                    point["rollout"],
                ),
            }
        )
    return reports, sample_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--horizons", type=parse_horizons, default=parse_horizons("1,2,3,5,8,10,16,20"))
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.02)
    parser.add_argument("--holdout-role", choices=("all", "train", "eval"), default="eval")
    parser.add_argument("--holdout-seed", type=int, default=20260524)
    parser.add_argument("--sample-seed", type=int, default=20260601)
    parser.add_argument("--allow-material-color-render", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Probing long-horizon rollouts on %s", device)

    model, model_config = load_model(args, device)
    max_horizon = max(args.horizons)
    render_root = args.render_root or (
        Path(model_config["render_root"]) if "render_root" in model_config else None
    )
    allow_material = bool(
        args.allow_material_color_render
        or model_config.get("allow_material_color_render", False)
    )
    dataset = GenesisWMDataset(
        root_dir=args.data_root,
        render_root=render_root,
        seq_len=max_horizon + 1,
        stride=args.stride,
        max_sessions=args.max_sessions,
        allow_material_color_render=allow_material,
        holdout_fraction=args.holdout_fraction,
        holdout_role=args.holdout_role,
        holdout_seed=args.holdout_seed,
    )
    if len(dataset) == 0:
        raise SystemExit("long-horizon probe dataset is empty")

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
        persistent_workers=True,
        prefetch_factor=3,
    )
    horizon_reports, evaluated_samples = probe_rollout_horizons(
        model,
        loader,
        device,
        horizons=args.horizons,
        max_batches=args.max_batches,
        precision=args.precision,
    )
    record = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "render_root": str(dataset.render_root),
        "trained_max_seq_len": int(model.predictor.max_seq_len),
        "probe_max_horizon": max_horizon,
        "stride": args.stride,
        "approx_seconds_per_macro_step": 0.5 if args.stride == 5 else None,
        "holdout_fraction": args.holdout_fraction,
        "holdout_role": args.holdout_role,
        "holdout_seed": args.holdout_seed,
        "sample_seed": args.sample_seed,
        "dataset_sequences": len(dataset),
        "evaluated_samples": evaluated_samples,
        "horizons": horizon_reports,
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
