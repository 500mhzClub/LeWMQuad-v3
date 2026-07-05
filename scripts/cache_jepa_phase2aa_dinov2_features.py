#!/usr/bin/env python3
"""Cache frozen DINOv2 features for the bounded Phase 2AA patch-dynamics screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase2_data import load_spatial_future_rows  # noqa: E402
from lewm.benchmarks.phase2aa_dinov2_cache import (  # noqa: E402
    PHASE2AA_DINOV2_CACHE_SCHEMA,
    phase2aa_frame_cache_audit,
    phase2aa_unique_frame_records,
)
from lewm.benchmarks.phase2d_training import image_tensor  # noqa: E402

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _args_for_json(args: argparse.Namespace) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _load_dinov2_model(args: argparse.Namespace, device: torch.device):
    try:
        model = torch.hub.load(
            args.dinov2_repo,
            args.dinov2_model,
            pretrained=True,
            trust_repo=True,
        )
    except Exception as error:  # pragma: no cover - depends on local torch cache.
        raise RuntimeError(
            "failed to load DINOv2. Phase 2AA must not silently fall back to a "
            "weaker encoder; either ensure the torch hub cache is present or "
            "explicitly approve the one-time pretrained-weight download."
        ) from error
    return model.to(device).eval()


@torch.no_grad()
def _encode_records(
    model,
    records,
    *,
    device: torch.device,
    batch_size: int,
    image_size: int,
    feature_kind: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    cls_batches = []
    patch_mean_batches = []
    patch_grid_batches = []
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        vision = torch.stack(
            [
                image_tensor(Path(record.frame_path), image_size=image_size)
                for record in batch_records
            ]
        ).to(device)
        vision = (vision - mean) / std
        features = model.forward_features(vision)
        cls_batches.append(features["x_norm_clstoken"].detach().cpu().to(dtype))
        patches = features["x_norm_patchtokens"].detach().cpu().to(dtype)
        patch_mean_batches.append(patches.mean(dim=1))
        if feature_kind == "patch_grid":
            patch_grid_batches.append(patches)
    output = {
        "cls": torch.cat(cls_batches, dim=0) if cls_batches else torch.empty(0),
        "patch_mean": (
            torch.cat(patch_mean_batches, dim=0)
            if patch_mean_batches
            else torch.empty(0)
        ),
    }
    if feature_kind == "patch_grid":
        output["patch_grid"] = (
            torch.cat(patch_grid_batches, dim=0)
            if patch_grid_batches
            else torch.empty(0)
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("all", "any_transition", "complete"),
        default="complete",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dinov2-repo", default="facebookresearch/dinov2")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument(
        "--feature-kind",
        choices=("patch_mean", "patch_grid"),
        default="patch_grid",
    )
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.max_rows < 0:
        parser.error("--max-rows must be non-negative")
    if args.image_size < 14 or args.image_size % 14 != 0:
        parser.error("--image-size must be a positive multiple of 14")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32

    rows, load_audit = load_spatial_future_rows(
        args.data,
        mode=args.mode,
        max_rows=args.max_rows,
    )
    records = phase2aa_unique_frame_records(rows)
    audit = phase2aa_frame_cache_audit(
        rows,
        records,
        split_name=args.split_name,
    )
    if not audit["all_frames_exist"]:
        raise SystemExit(
            json.dumps(
                {
                    "error": "phase2aa_missing_frame_paths",
                    "missing_unique_frames": audit["missing_unique_frames"],
                    "data": str(args.data),
                },
                sort_keys=True,
            )
        )
    model = _load_dinov2_model(args, device)
    features = _encode_records(
        model,
        records,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        feature_kind=args.feature_kind,
        dtype=dtype,
    )
    report = {
        "schema": PHASE2AA_DINOV2_CACHE_SCHEMA,
        "split": args.split_name,
        "data": str(args.data.resolve()),
        "load_audit": load_audit,
        "frame_cache_audit": audit,
        "dinov2": {
            "repo": args.dinov2_repo,
            "model": args.dinov2_model,
            "image_size": args.image_size,
            "feature_kind": args.feature_kind,
            "storage_dtype": args.storage_dtype,
        },
        "device": str(device),
        "feature_shapes": {
            key: list(value.shape) for key, value in features.items()
        },
        "args": _args_for_json(args),
    }
    payload = {
        "report": report,
        "records": [record.__dict__ for record in records],
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
