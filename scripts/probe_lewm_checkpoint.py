#!/usr/bin/env python3
"""Small offline probe for flat LeWM checkpoints.

The probe is intentionally bounded: it reports teacher-forced prediction loss,
short autoregressive rollout loss, latent variance, and action-sensitivity
deltas over a small deterministic scene holdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.actions import (  # noqa: E402
    ACTIVE_BLOCK_DIM,
    assert_active_block_metadata_compatible,
)
from lewm.models.lewm import LeWorldModel  # noqa: E402
from train_lewm import GenesisWMDataset, evaluate_model, make_loader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[LeWorldModel, dict]:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    max_seq_len = args.max_seq_len or int(model_config.get("max_seq_len", 4))
    sigreg_lambda = (
        args.sigreg_lambda
        if args.sigreg_lambda is not None
        else float(model_config.get("sigreg_lambda", 0.09))
    )

    model = LeWorldModel(
        max_seq_len=max_seq_len,
        cmd_dim=ACTIVE_BLOCK_DIM,
        sigreg_lambda=sigreg_lambda,
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata = checkpoint.get("action_metadata")
        if metadata is not None:
            assert_active_block_metadata_compatible(metadata)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, model_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.02)
    parser.add_argument("--holdout-role", choices=("all", "train", "eval"), default="eval")
    parser.add_argument("--holdout-seed", type=int, default=20260524)
    parser.add_argument("--allow-material-color-render", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Probing on %s", device)

    model, model_config = load_model(args, device)
    max_seq_len = args.max_seq_len or int(model_config.get("max_seq_len", 4))
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
        seq_len=max_seq_len,
        stride=args.stride,
        max_sessions=args.max_sessions,
        allow_material_color_render=allow_material,
        holdout_fraction=args.holdout_fraction,
        holdout_role=args.holdout_role,
        holdout_seed=args.holdout_seed,
    )
    if len(dataset) == 0:
        raise SystemExit("probe dataset is empty")

    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=3,
    )
    metrics = evaluate_model(
        model,
        loader,
        device,
        max_batches=args.max_batches,
        precision=args.precision,
    )
    record = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "render_root": str(dataset.render_root),
        "seq_len": max_seq_len,
        "stride": args.stride,
        "holdout_fraction": args.holdout_fraction,
        "holdout_role": args.holdout_role,
        "holdout_seed": args.holdout_seed,
        "max_batches": args.max_batches,
        "num_sequences": len(dataset),
        **metrics,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
