#!/usr/bin/env python3
"""Screen current LeWM spatial patch tokens for held-out place retrieval."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import _load_image_tensor  # noqa: E402
from probe_lewm_reachability_a3 import (  # noqa: E402
    _select,
    build_scene_bank,
    same_cell_retrieval,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe_lewm_patch_retrieval")


def _aggregate(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def spatial_pyramid_descriptor(patches: torch.Tensor, levels: tuple[int, ...] = (1, 2, 4)) -> torch.Tensor:
    """Concatenate regional mean patch features while preserving coarse layout."""
    batch, count, dimensions = patches.shape
    grid_size = math.isqrt(count)
    if grid_size * grid_size != count:
        raise ValueError(f"patch count {count} is not a square grid")
    grid = patches.reshape(batch, grid_size, grid_size, dimensions)
    pooled = []
    for level in levels:
        if grid_size % level:
            raise ValueError(f"grid size {grid_size} is not divisible by level {level}")
        region = grid_size // level
        pooled.append(
            grid.reshape(batch, level, region, level, region, dimensions)
            .mean(dim=(2, 4))
            .reshape(batch, level * level * dimensions)
        )
    return torch.cat(pooled, dim=-1)


@torch.no_grad()
def _encode_patch_descriptors(
    model,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    mean_all: list[np.ndarray] = []
    pyramid_all: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        array = np.stack([_load_image_tensor(path) for path in batch_paths]).astype(np.float32) / 255.0
        vision = torch.from_numpy(array).to(device)
        patches = model.encoder.vis_enc.forward_tokens(vision)[:, 1:]
        mean_all.append(patches.mean(dim=1).cpu().float().numpy())
        pyramid_all.append(spatial_pyramid_descriptor(patches).cpu().float().numpy())
    return {
        "patch_mean": np.concatenate(mean_all).astype(np.float64),
        "patch_spatial_pyramid": np.concatenate(pyramid_all).astype(np.float64),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/rollout")
    parser.add_argument("--render-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    parser.add_argument(
        "--manifest-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z",
    )
    parser.add_argument("--eval-split", default="test_id")
    parser.add_argument("--eval-scenes-per-family", type=int, default=4)
    parser.add_argument("--eval-frames-per-scene", type=int, default=240)
    parser.add_argument("--eval-max-per-cell", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gate-recall5-improvement", type=float, default=0.05)
    parser.add_argument("--min-eval-scenes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, _config = load_model(args, device)
    descriptor_metrics: dict[str, dict[str, list[float]]] = {
        "raw_cls": {"retrieval_at_1": [], "retrieval_at_5": []},
        "patch_mean": {"retrieval_at_1": [], "retrieval_at_5": []},
        "patch_spatial_pyramid": {"retrieval_at_1": [], "retrieval_at_5": []},
    }
    scene_count = 0
    for index, (family, label_file) in enumerate(
        _select(args.rollout_root, args.eval_split, args.eval_scenes_per_family, args.seed)
    ):
        bank = build_scene_bank(
            model,
            label_file=label_file,
            family=family,
            split=args.eval_split,
            render_root=args.render_root,
            corpus_root=args.manifest_corpus,
            device=device,
            frames_per_scene=args.eval_frames_per_scene,
            max_per_cell=args.eval_max_per_cell,
            batch_size=args.batch_size,
            min_cells=args.min_cells,
            rng=random.Random(args.seed + index * 7919),
        )
        if bank is None:
            continue
        descriptors = {"raw_cls": bank["z_raw"]}
        descriptors.update(_encode_patch_descriptors(model, bank["paths"], device, args.batch_size))
        for name, descriptor in descriptors.items():
            retrieval = same_cell_retrieval(descriptor, bank["cells"])
            if retrieval is not None:
                descriptor_metrics[name]["retrieval_at_1"].append(retrieval["retrieval_at_1"])
                descriptor_metrics[name]["retrieval_at_5"].append(retrieval["retrieval_at_5"])
        scene_count += 1
        logger.info("scene=%s frames=%d cells=%d", bank["scene_id"], len(bank["cells"]), len(set(bank["cells"])))

    metrics = {
        name: {key: _aggregate(values) for key, values in readouts.items()}
        for name, readouts in descriptor_metrics.items()
    }
    baseline_r1 = metrics["raw_cls"]["retrieval_at_1"]["mean"]
    baseline_r5 = metrics["raw_cls"]["retrieval_at_5"]["mean"]
    candidates = {}
    for name in ("patch_mean", "patch_spatial_pyramid"):
        candidates[name] = {
            "recall1_improvement": metrics[name]["retrieval_at_1"]["mean"] - baseline_r1,
            "recall5_improvement": metrics[name]["retrieval_at_5"]["mean"] - baseline_r5,
        }
    passing = [
        name
        for name, result in candidates.items()
        if result["recall5_improvement"] >= args.gate_recall5_improvement
        and result["recall1_improvement"] >= 0.0
    ]
    report = {
        "schema": "lewm_patch_retrieval_screen_v0",
        "source_checkpoint": str(args.checkpoint),
        "eval_scene_count": scene_count,
        "metrics": metrics,
        "candidate_improvements": candidates,
        "gate": {
            "passed": scene_count >= args.min_eval_scenes and bool(passing),
            "passing_descriptors": passing,
            "required_recall5_improvement": args.gate_recall5_improvement,
            "min_eval_scenes": args.min_eval_scenes,
            "requires_recall1_non_regression": True,
        },
        "notes": (
            "This screens spatial aggregation of the current from-scratch LeWM ViT only; "
            "it does not evaluate pretrained DINO-style patch features."
        ),
        "config": {key: str(value) for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "config"}, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

