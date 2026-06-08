#!/usr/bin/env python3
"""Screen short frozen-LeWM histories for held-out same-place retrieval."""
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import _encode_frames  # noqa: E402
from probe_lewm_reachability_a3 import (  # noqa: E402
    _select,
    build_scene_bank,
    same_cell_retrieval,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe_lewm_history_retrieval")


def _aggregate(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def history_descriptors(z_history: np.ndarray, h_values: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Build terminal, mean, and ordered-concatenation descriptors."""
    descriptors = {"terminal_raw": z_history[:, -1]}
    for history in h_values:
        suffix = z_history[:, -history:]
        descriptors[f"h{history}_mean"] = suffix.mean(axis=1)
        descriptors[f"h{history}_concat"] = suffix.reshape(len(suffix), -1)
    return descriptors


def _select_history_windows(
    bank: dict,
    *,
    history: int,
    windows_per_scene: int,
    max_per_cell: int,
    rng: random.Random,
) -> tuple[list[list[Path]], np.ndarray]:
    graph_cells = {node.node_id for node in bank["graph"].manifest.graph_nodes}
    by_cell: dict[int, list[list[Path]]] = defaultdict(list)
    for env_index, observations in bank["by_env"].items():
        for step in range(history - 1, len(observations)):
            terminal_cell, _yaw = observations[step]
            if terminal_cell not in graph_cells:
                continue
            paths = []
            for history_step in range(step - history + 1, step + 1):
                global_index = history_step * bank["n_envs"] + env_index
                paths.append(
                    bank["render_dir"] / "rgb" / f"frame_{global_index:06d}_env_{env_index:02d}.png"
                )
            if all(path.exists() for path in paths):
                by_cell[int(terminal_cell)].append(paths)
    selected: list[tuple[list[Path], int]] = []
    for cell, windows in by_cell.items():
        rng.shuffle(windows)
        selected.extend((window, cell) for window in windows[:max_per_cell])
    rng.shuffle(selected)
    selected = selected[:windows_per_scene]
    return [window for window, _cell in selected], np.asarray(
        [cell for _window, cell in selected],
        dtype=np.int64,
    )


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
    parser.add_argument("--history", default="4,8")
    parser.add_argument("--windows-per-scene", type=int, default=160)
    parser.add_argument("--max-per-cell", type=int, default=8)
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
    h_values = tuple(sorted({int(value) for value in args.history.split(",") if value.strip()}))
    max_history = max(h_values)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, _config = load_model(args, device)

    names = ("terminal_raw",) + tuple(
        name
        for history in h_values
        for name in (f"h{history}_mean", f"h{history}_concat")
    )
    descriptor_metrics = {
        name: {"retrieval_at_1": [], "retrieval_at_5": []}
        for name in names
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
            frames_per_scene=max(12, args.min_cells * 2),
            max_per_cell=2,
            batch_size=args.batch_size,
            min_cells=args.min_cells,
            rng=random.Random(args.seed + index * 7919),
        )
        if bank is None:
            continue
        windows, cells = _select_history_windows(
            bank,
            history=max_history,
            windows_per_scene=args.windows_per_scene,
            max_per_cell=args.max_per_cell,
            rng=random.Random(args.seed + index * 104729),
        )
        if len(set(cells.tolist())) < args.min_cells or len(windows) < 12:
            continue
        unique_paths = list(dict.fromkeys(path for window in windows for path in window))
        z_unique, _z_projected = _encode_frames(model, unique_paths, device, args.batch_size)
        path_index = {path: offset for offset, path in enumerate(unique_paths)}
        window_index = np.asarray(
            [[path_index[path] for path in window] for window in windows],
            dtype=np.int64,
        )
        z_history = z_unique[window_index].astype(np.float64)
        for name, descriptor in history_descriptors(z_history, h_values).items():
            retrieval = same_cell_retrieval(descriptor, cells)
            if retrieval is not None:
                descriptor_metrics[name]["retrieval_at_1"].append(retrieval["retrieval_at_1"])
                descriptor_metrics[name]["retrieval_at_5"].append(retrieval["retrieval_at_5"])
        scene_count += 1
        logger.info(
            "scene=%s windows=%d unique_frames=%d cells=%d",
            bank["scene_id"],
            len(windows),
            len(unique_paths),
            len(set(cells.tolist())),
        )

    metrics = {
        name: {key: _aggregate(values) for key, values in readouts.items()}
        for name, readouts in descriptor_metrics.items()
    }
    baseline_r1 = metrics["terminal_raw"]["retrieval_at_1"]["mean"]
    baseline_r5 = metrics["terminal_raw"]["retrieval_at_5"]["mean"]
    candidates = {}
    for name in names[1:]:
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
        "schema": "lewm_history_retrieval_screen_v0",
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
            "All descriptors are evaluated on identical H=max(history) terminal windows. "
            "This tests visual history only and does not include actions or a learned belief encoder."
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
