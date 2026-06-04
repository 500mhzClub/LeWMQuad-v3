#!/usr/bin/env python3
"""View-vs-place diagnostic: is the frozen LeWM latent a PLACE code or a VIEW code?

Roadmap step 1. The A2 aliasing audit reported yaw-matched rho >> raw rho, which
hints that heading (what the robot sees) dominates the embedding more than
position (where the robot is). This probe quantifies that directly on held-out
scenes, two complementary ways, reusing A2's exact frame sampler/encoder so the
latents, cells and yaw bins line up with the aliasing audit:

  (A) Same-place-across-yaw vs across-place latent distance.
      For frames sharing a ground-truth cell but differing in yaw_bin, the
      latent L2 distance (same place, different heading) vs the distance to
      frames of *other* cells. ``yaw_place_ratio = median(within)/median(across)``.
      ~1 => turning in place moves the latent as much as walking to a new place
      (a VIEW code); << 1 => place dominates (a PLACE code).

  (B) Decode yaw vs decode position from the latent (held-out ridge readout, on
      the SAME frames/split so the two R^2 are comparable). Heading is decoded as
      [sin, cos] of the bin angle; position as the ground-truth cell-centre xy.
      yaw_decode_r2 >> pos_decode_r2 => heading/appearance-dominated latent.

CPU by default so it does not contend with a live GPU training run.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import (  # noqa: E402
    _agg,
    _encode_frames,
    _find_manifest,
    _iter_label_files,
    _load_observations,
    _sample_scene_frames,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tiny self-contained held-out readout (standardize -> PCA -> ridge -> R^2).
# --------------------------------------------------------------------------- #
def _standardize(Xtr: np.ndarray, Xte: np.ndarray):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _pca(Xtr: np.ndarray, Xte: np.ndarray, k: int):
    mu = Xtr.mean(0)
    _, _, vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    comp = vt[:k]
    return (Xtr - mu) @ comp.T, (Xte - mu) @ comp.T


def _ridge_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, alpha: float):
    n, d = Xtr.shape
    xb = np.hstack([Xtr, np.ones((n, 1))])
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0  # do not regularise the bias
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ Ytr)
    return np.hstack([Xte, np.ones((Xte.shape[0], 1))]) @ w


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def _decode_r2(Z: np.ndarray, Y: np.ndarray, *, seed: int, k: int = 32,
               test_frac: float = 0.3, alpha: float = 1.0):
    """Held-out R^2 of predicting Y from latents Z. Fixed seed => reproducible
    split shared across targets so yaw vs position R^2 are comparable."""
    n = len(Z)
    if n < 12:
        return None
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    ncut = max(4, int(n * (1 - test_frac)))
    tr, te = np.array(idx[:ncut]), np.array(idx[ncut:])
    if len(te) < 3:
        return None
    ztr, zte = _standardize(Z[tr], Z[te])
    kk = min(k, ztr.shape[0] - 1, ztr.shape[1])
    if kk >= 2:
        ztr, zte = _pca(ztr, zte, kk)
    y_pred = _ridge_predict(ztr, Y[tr], zte, alpha)
    return _r2(Y[te], y_pred)


# --------------------------------------------------------------------------- #
# Per-scene analysis
# --------------------------------------------------------------------------- #
def analyze_scene(model, *, label_file, family, split, rollout_root, render_root,
                  corpus_root, device, frames_per_scene, max_per_cell, batch_size,
                  min_cells, n_yaw_bins, max_pairs, rng):
    scene_id = label_file.parent.name
    render_dir = render_root / scene_id
    if not (render_dir / "summary.json").exists():
        return {"scene_id": scene_id, "family": family, "skipped": "no_render"}
    manifest_path = _find_manifest(corpus_root, split, family, scene_id)
    if manifest_path is None:
        return {"scene_id": scene_id, "family": family, "skipped": "no_manifest"}

    manifest = parse_scene_manifest_dict(json.loads(manifest_path.read_text()))
    graph = SceneGraph(manifest)
    graph_cells = {n.node_id for n in manifest.graph_nodes}

    chunk_dir = label_file.parents[1]
    rsum_path = chunk_dir / "rollout" / scene_id / "summary.json"
    by_env = _load_observations(label_file)
    if rsum_path.exists():
        n_envs = int(json.loads(rsum_path.read_text()).get("n_envs", len(by_env)))
    else:
        n_envs = max(by_env) + 1

    frames = _sample_scene_frames(
        by_env, render_dir=render_dir, n_envs=n_envs, graph_cells=graph_cells,
        frames_per_scene=frames_per_scene, max_per_cell=max_per_cell, rng=rng,
    )
    cells = np.array([f[1] for f in frames], dtype=np.int64)
    yaws = np.array([f[2] for f in frames], dtype=np.int64)
    distinct = len(set(cells.tolist()))
    if distinct < min_cells or len(frames) < min_cells:
        return {"scene_id": scene_id, "family": family, "skipped": "too_few_cells",
                "distinct_cells": distinct, "frames": len(frames)}

    paths = [f[0] for f in frames]
    z_raw, z_proj = _encode_frames(model, paths, device, batch_size)
    xy = np.array([graph.cell_center(int(c)) for c in cells], dtype=np.float64)

    res: dict = {"scene_id": scene_id, "family": family, "split": split,
                 "n_frames": len(frames), "distinct_cells": distinct}
    n = len(frames)
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        all_pairs = rng.sample(all_pairs, max_pairs)

    for space, z in (("proj", z_proj.astype(np.float64)), ("raw", z_raw.astype(np.float64))):
        # (A) within-cell (across-yaw) vs across-cell latent distance.
        within, across = [], []
        for i, j in all_pairs:
            d = float(np.linalg.norm(z[i] - z[j]))
            if cells[i] == cells[j]:
                if yaws[i] != yaws[j] and yaws[i] >= 0 and yaws[j] >= 0:
                    within.append(d)
            else:
                across.append(d)
        if len(within) >= 5 and len(across) >= 5:
            dw, da = float(np.median(within)), float(np.median(across))
            res[f"within_cell_acrossyaw_dist_median_{space}"] = dw
            res[f"across_cell_dist_median_{space}"] = da
            res[f"yaw_place_ratio_{space}"] = dw / (da if da else 1e-9)
            res[f"n_within_{space}"] = len(within)
            res[f"n_across_{space}"] = len(across)

        # (B) decode yaw vs position on the SAME yaw-valid frames/split.
        ok = yaws >= 0
        if ok.sum() >= 12 and n_yaw_bins >= 2:
            theta = 2.0 * np.pi * yaws[ok] / float(n_yaw_bins)
            y_yaw = np.stack([np.sin(theta), np.cos(theta)], axis=1)
            split_seed = abs(hash((scene_id, space))) % (2 ** 31)
            res[f"yaw_decode_r2_{space}"] = _decode_r2(z[ok], y_yaw, seed=split_seed)
            res[f"pos_decode_r2_{space}"] = _decode_r2(z[ok], xy[ok].copy(), seed=split_seed)

    yr, pr = res.get("yaw_decode_r2_proj"), res.get("pos_decode_r2_proj")
    if yr is not None and pr is not None:
        res["yaw_minus_pos_decode_r2_proj"] = yr - pr
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path,
                        default=REPO_ROOT / ".generated/datagen_full/rollout")
    parser.add_argument("--render-root", type=Path,
                        default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    parser.add_argument("--manifest-corpus", type=Path,
                        default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--family", default=None)
    parser.add_argument("--scenes-per-family", type=int, default=4)
    parser.add_argument("--frames-per-scene", type=int, default=200)
    parser.add_argument("--max-per-cell", type=int, default=8)
    parser.add_argument("--max-pairs-per-scene", type=int, default=60000)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--n-yaw-bins", type=int, default=None,
                        help="override; otherwise inferred as max(yaw_bin)+1 over selected scenes")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    logger.info("Yaw-invariance probe on %s (split=%s)", device, args.split)
    model, _ = load_model(args, device)

    per_family: dict[str, list] = defaultdict(list)
    for fam, lf in _iter_label_files(args.rollout_root, args.split, args.family):
        per_family[fam].append(lf)
    rng = random.Random(args.seed)
    selected: list[tuple[str, Path]] = []
    for fam in sorted(per_family):
        files = sorted(per_family[fam])
        rng.shuffle(files)
        for lf in files[: args.scenes_per_family]:
            selected.append((fam, lf))
    logger.info("Selected %d scenes across %d families", len(selected), len(per_family))

    # Infer the global yaw-bin count from the selected scenes' labels.
    n_yaw_bins = args.n_yaw_bins
    if n_yaw_bins is None:
        max_yaw = -1
        for _, lf in selected:
            for obs in _load_observations(lf).values():
                for _cell, yaw in obs:
                    if yaw > max_yaw:
                        max_yaw = yaw
        n_yaw_bins = max_yaw + 1 if max_yaw >= 1 else 0
    logger.info("Using n_yaw_bins=%d", n_yaw_bins)

    results: list[dict] = []
    for idx, (fam, lf) in enumerate(selected):
        scene_rng = random.Random(args.seed + idx * 7919)
        try:
            res = analyze_scene(
                model, label_file=lf, family=fam, split=args.split,
                rollout_root=args.rollout_root, render_root=args.render_root,
                corpus_root=args.manifest_corpus, device=device,
                frames_per_scene=args.frames_per_scene, max_per_cell=args.max_per_cell,
                batch_size=args.batch_size, min_cells=args.min_cells,
                n_yaw_bins=n_yaw_bins, max_pairs=args.max_pairs_per_scene, rng=scene_rng,
            )
        except Exception as exc:  # noqa: BLE001
            res = {"scene_id": lf.parent.name, "family": fam, "skipped": f"error:{exc}"}
        results.append(res)
        logger.info(
            "[%d/%d] %s yaw_place_ratio_proj=%s yawR2=%s posR2=%s%s",
            idx + 1, len(selected), res.get("scene_id"),
            f"{res['yaw_place_ratio_proj']:.3f}" if isinstance(res.get("yaw_place_ratio_proj"), float) else res.get("yaw_place_ratio_proj"),
            f"{res['yaw_decode_r2_proj']:.3f}" if isinstance(res.get("yaw_decode_r2_proj"), float) else res.get("yaw_decode_r2_proj"),
            f"{res['pos_decode_r2_proj']:.3f}" if isinstance(res.get("pos_decode_r2_proj"), float) else res.get("pos_decode_r2_proj"),
            f"  SKIP({res['skipped']})" if "skipped" in res else "",
        )

    used = [r for r in results if "yaw_place_ratio_proj" in r]
    def col(key):
        return _agg([r.get(key) for r in used])

    ratio_proj = col("yaw_place_ratio_proj").get("median")
    yaw_r2 = col("yaw_decode_r2_proj").get("median")
    pos_r2 = col("pos_decode_r2_proj").get("median")
    interp = "inconclusive"
    if ratio_proj is not None and yaw_r2 is not None and pos_r2 is not None:
        view_dominated = (ratio_proj >= 0.70) or (yaw_r2 - pos_r2 >= 0.20)
        interp = ("view/heading-dominated latent (place is partly washed out by "
                  "orientation)") if view_dominated else (
                  "place-leaning latent (orientation is not the dominant axis)")

    record = {
        "schema": "lewm_yaw_invariance_probe_v0",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "manifest_corpus": str(args.manifest_corpus),
        "n_yaw_bins": n_yaw_bins,
        "scenes_used": len(used),
        "scenes_selected": len(selected),
        "interpretation": interp,
        "yaw_place_ratio_proj": col("yaw_place_ratio_proj"),
        "yaw_place_ratio_raw": col("yaw_place_ratio_raw"),
        "yaw_decode_r2_proj": col("yaw_decode_r2_proj"),
        "pos_decode_r2_proj": col("pos_decode_r2_proj"),
        "yaw_decode_r2_raw": col("yaw_decode_r2_raw"),
        "pos_decode_r2_raw": col("pos_decode_r2_raw"),
        "yaw_minus_pos_decode_r2_proj": col("yaw_minus_pos_decode_r2_proj"),
        "notes": {
            "yaw_place_ratio": "median(within-cell across-yaw L2) / median(across-cell L2); ~1 = view code, <<1 = place code",
            "decode_r2": "held-out ridge R^2 on identical yaw-valid frames/split; yaw=[sin,cos] of bin angle, pos=cell-centre xy",
        },
        "scenes": results,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
