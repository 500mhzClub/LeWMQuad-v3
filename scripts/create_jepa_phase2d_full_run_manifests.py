#!/usr/bin/env python3
"""Freeze Phase 2D full-training launch manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase2d_run_manifest import (  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_SOURCE_STATES_PER_BATCH,
    PRIMARY_CELLS,
    REGISTERED_OPTIMIZATION_SEEDS,
    create_phase2d_training_run_manifests,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cell", action="append", choices=PRIMARY_CELLS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--python-executable",
        default=".generated/venvs/genesis_render_vulkan/bin/python",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--source-states-per-batch",
        type=int,
        default=DEFAULT_SOURCE_STATES_PER_BATCH,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Freeze gradient clipping into generated trainer commands.",
    )
    parser.add_argument(
        "--detach-action-control-state",
        action="store_true",
        help="Freeze detached C2 contrast-control state into trainer commands.",
    )
    parser.add_argument(
        "--target-geometry",
        choices=("patch", "slot"),
        default="patch",
        help="Freeze the trainer target geometry into generated commands.",
    )
    parser.add_argument(
        "--num-target-slots",
        type=int,
        default=16,
        help="Freeze the learned slot count when --target-geometry slot is used.",
    )
    parser.add_argument(
        "--consequence-loss-lambda",
        type=float,
        default=0.0,
        help="Freeze the Phase 2F consequence auxiliary loss weight.",
    )
    parser.add_argument(
        "--action-utility-loss-lambda",
        type=float,
        default=0.0,
        help="Freeze the Phase 2G source-local action-utility loss weight.",
    )
    parser.add_argument(
        "--action-utility-regression-weight",
        type=float,
        default=0.1,
        help="Freeze the Phase 2G utility regression scale term.",
    )
    args = parser.parse_args()

    summary = create_phase2d_training_run_manifests(
        repository_root=REPO_ROOT,
        split_manifest_path=args.split_manifest,
        train_data_path=args.train_data,
        validation_data_path=args.validation_data,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        cells=tuple(args.cell or PRIMARY_CELLS),
        seeds=tuple(args.seed or REGISTERED_OPTIMIZATION_SEEDS),
        python_executable=args.python_executable,
        device=args.device,
        source_states_per_batch=args.source_states_per_batch,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        detach_action_control_state=args.detach_action_control_state,
        target_geometry=args.target_geometry,
        num_target_slots=args.num_target_slots,
        consequence_loss_lambda=args.consequence_loss_lambda,
        action_utility_loss_lambda=args.action_utility_loss_lambda,
        action_utility_regression_weight=args.action_utility_regression_weight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
