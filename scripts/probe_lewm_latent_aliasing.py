#!/usr/bin/env python3
"""Phase A2 — frozen-latent visual-aliasing audit.

Decision-relevant question (v3 plan §4.2/§4.5): does the frozen LeWM latent
distance track ground-truth topological (BFS graph) distance on held-out scene
topologies? If it does (Spearman rho high), the latents already encode place
structure and the belief/memory stack may be unnecessary; if it does not
(rho low), partial observability / aliasing is real and BeliefEncoder is
motivated.

This probe does NOT train any head. For each held-out scene it:

  1. Loads the scene manifest -> ``SceneGraph`` (ground-truth cell adjacency).
  2. Reads ``labels.jsonl`` to map every macro-step (env, step) -> ``cell_id``
     and ``yaw_bin``.
  3. Locates the matching rendered RGB frame, encodes it with the frozen
     encoder (projected embedding ``z_proj`` = the goal-matching cost space,
     plus raw backbone ``z_raw``).
  4. Computes, within the scene, pairwise latent L2 distance vs pairwise BFS
     graph distance, and the Spearman rank correlation between them.

Aggregates per-scene rho across scenes (cross-scene pairs are never formed —
BFS distance is only defined within a scene). Also reports a yaw-matched
variant (pairs sharing ``yaw_bin``) to separate place structure from heading,
a per-scene-normalised latent-distance-by-bucket table, and a near/far
confusion proxy.

Runs on CPU by default so it does not contend with a live GPU training run.
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Graph-distance buckets, mirroring v3 plan §4.4 (collapsed to inclusive ranges).
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0-1_same_adjacent", 0, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16+", 16, 10**9),
)


def _bucket_name(d: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= d <= hi:
            return name
    return BUCKETS[-1][0]


# ---------------------------------------------------------------------------
# Dependency-free rank statistics
# ---------------------------------------------------------------------------


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Average-rank of ``a`` (ties share the mean rank), matching scipy."""
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], a.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return _pearson(_rankdata_average(x), _rankdata_average(y))


# ---------------------------------------------------------------------------
# Scene discovery and observation sampling
# ---------------------------------------------------------------------------


def _find_manifest(corpus_root: Path, split: str, family: str, scene_id: str) -> Path | None:
    candidate = corpus_root / split / family / scene_id / "manifest.json"
    if candidate.exists():
        return candidate
    matches = list(corpus_root.glob(f"**/{scene_id}/manifest.json"))
    return matches[0] if matches else None


def _iter_label_files(rollout_root: Path, split: str, family: str | None):
    pattern = f"{split}/*/chunk_*/labels/*/labels.jsonl"
    for lf in sorted(rollout_root.glob(pattern)):
        fam = lf.parents[3].name
        if family is not None and fam != family:
            continue
        yield fam, lf


def _load_observations(label_file: Path):
    """Return per-env ordered list of (cell_id, yaw_bin), in global-step order."""
    by_env: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with label_file.open() as f:
        for line in f:
            row = json.loads(line)
            cell = row.get("cell_id")
            if cell is None:
                by_env[int(row.get("env_idx", 0))].append((-1, -1))
                continue
            by_env[int(row.get("env_idx", 0))].append(
                (int(cell), int(row.get("yaw_bin", -1)))
            )
    return by_env


def _sample_scene_frames(
    by_env: dict[int, list[tuple[int, int]]],
    *,
    render_dir: Path,
    n_envs: int,
    graph_cells: set[int],
    frames_per_scene: int,
    max_per_cell: int,
    rng: random.Random,
):
    """Pick frame paths with per-cell capping for even graph coverage.

    Returns list of (frame_path, cell_id, yaw_bin).
    """
    by_cell: dict[int, list[tuple[Path, int, int]]] = defaultdict(list)
    for env_idx, obs in by_env.items():
        for step, (cell, yaw) in enumerate(obs):
            if cell not in graph_cells:
                continue
            global_idx = step * n_envs + env_idx
            fp = render_dir / "rgb" / f"frame_{global_idx:06d}_env_{env_idx:02d}.png"
            by_cell[cell].append((fp, cell, yaw))

    chosen: list[tuple[Path, int, int]] = []
    for cell, items in by_cell.items():
        rng.shuffle(items)
        chosen.extend(items[:max_per_cell])
    rng.shuffle(chosen)
    # Verify existence lazily but cap total.
    out: list[tuple[Path, int, int]] = []
    for fp, cell, yaw in chosen:
        if fp.exists():
            out.append((fp, cell, yaw))
        if len(out) >= frames_per_scene:
            break
    return out


def _load_image_tensor(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.asarray(img).transpose(2, 0, 1)


@torch.no_grad()
def _encode_frames(model, paths: list[Path], device: torch.device, batch_size: int):
    z_proj_all: list[np.ndarray] = []
    z_raw_all: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        arr = np.stack([_load_image_tensor(p) for p in batch_paths]).astype(np.float32) / 255.0
        vis = torch.from_numpy(arr).to(device)
        z_raw, z_proj = model.encode(vis, None)
        z_raw_all.append(z_raw.detach().cpu().float().numpy())
        z_proj_all.append(z_proj.detach().cpu().float().numpy())
    return np.concatenate(z_raw_all, axis=0), np.concatenate(z_proj_all, axis=0)


# ---------------------------------------------------------------------------
# Per-scene audit
# ---------------------------------------------------------------------------


def _pairwise(latents: np.ndarray, cells: list[int], yaws: list[int], graph: SceneGraph,
              *, max_pairs: int, rng: random.Random):
    """Return arrays (latent_dist, graph_dist, same_yaw) over within-scene pairs."""
    n = len(cells)
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        all_pairs = rng.sample(all_pairs, max_pairs)
    bfs_cache: dict[tuple[int, int], int | None] = {}
    lat_d: list[float] = []
    grp_d: list[int] = []
    same_yaw: list[bool] = []
    for i, j in all_pairs:
        ci, cj = cells[i], cells[j]
        key = (ci, cj) if ci <= cj else (cj, ci)
        if key not in bfs_cache:
            bfs_cache[key] = graph.bfs_distance(ci, cj)
        gd = bfs_cache[key]
        if gd is None:  # different connected components
            continue
        lat_d.append(float(np.linalg.norm(latents[i] - latents[j])))
        grp_d.append(int(gd))
        same_yaw.append(yaws[i] == yaws[j] and yaws[i] >= 0)
    return np.asarray(lat_d), np.asarray(grp_d, dtype=np.int64), np.asarray(same_yaw, dtype=bool)


def audit_scene(model, *, label_file: Path, family: str, split: str, rollout_root: Path,
                render_root: Path, corpus_root: Path, device: torch.device,
                frames_per_scene: int, max_per_cell: int, max_pairs: int,
                batch_size: int, min_cells: int, rng: random.Random):
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

    # n_envs from rollout summary, fallback to label env count.
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
    distinct_cells = len({c for _, c, _ in frames})
    if distinct_cells < min_cells or len(frames) < min_cells:
        return {"scene_id": scene_id, "family": family, "skipped": "too_few_cells",
                "distinct_cells": distinct_cells, "frames": len(frames)}

    paths = [f[0] for f in frames]
    cells = [f[1] for f in frames]
    yaws = [f[2] for f in frames]
    z_raw, z_proj = _encode_frames(model, paths, device, batch_size)

    result: dict = {
        "scene_id": scene_id, "family": family, "split": split,
        "n_frames": len(frames), "distinct_cells": distinct_cells,
        "graph_nodes": len(graph_cells),
    }
    for space, latents in (("proj", z_proj), ("raw", z_raw)):
        lat_d, grp_d, same_yaw = _pairwise(
            latents, cells, yaws, graph, max_pairs=max_pairs, rng=rng,
        )
        if lat_d.size < 2:
            result[f"spearman_{space}"] = None
            continue
        result[f"spearman_{space}"] = _spearman(lat_d, grp_d)
        result[f"pearson_{space}"] = _pearson(lat_d, grp_d)
        result[f"n_pairs_{space}"] = int(lat_d.size)
        if space == "proj":
            ym = same_yaw
            if ym.sum() >= 2:
                result["spearman_proj_yawmatched"] = _spearman(lat_d[ym], grp_d[ym])
                result["n_pairs_yawmatched"] = int(ym.sum())
            # per-scene-median-normalised latent distance by bucket
            med = float(np.median(lat_d)) or 1.0
            norm = lat_d / med
            buckets: dict[str, dict] = {}
            for name, lo, hi in BUCKETS:
                m = (grp_d >= lo) & (grp_d <= hi)
                if m.any():
                    buckets[name] = {
                        "n": int(m.sum()),
                        "norm_latent_median": float(np.median(norm[m])),
                        "norm_latent_mean": float(norm[m].mean()),
                    }
            result["bucket_norm_latent"] = buckets
            # near/far confusion proxy: of the closest-10%-in-latent pairs,
            # fraction that are actually graph-far (>=8 hops).
            k = max(1, int(0.10 * lat_d.size))
            nearest = np.argsort(lat_d)[:k]
            result["confusion_nearest10pct_graphfar_frac"] = float(
                (grp_d[nearest] >= 8).mean()
            )
    return result


def _agg(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)),
        "min": float(arr.min()), "max": float(arr.max()), "std": float(arr.std()),
    }


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
    parser.add_argument("--family", default=None, help="restrict to one scene family")
    parser.add_argument("--scenes-per-family", type=int, default=4)
    parser.add_argument("--frames-per-scene", type=int, default=160)
    parser.add_argument("--max-per-cell", type=int, default=4)
    parser.add_argument("--max-pairs-per-scene", type=int, default=40000)
    parser.add_argument("--min-cells", type=int, default=6)
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
    logger.info("Aliasing audit on %s (split=%s)", device, args.split)
    model, model_config = load_model(args, device)

    # Balanced scene selection across families.
    per_family: dict[str, list] = defaultdict(list)
    for fam, lf in _iter_label_files(args.rollout_root, args.split, args.family):
        per_family[fam].append(lf)
    selected: list[tuple[str, Path]] = []
    rng = random.Random(args.seed)
    for fam in sorted(per_family):
        files = sorted(per_family[fam])
        rng.shuffle(files)
        for lf in files[: args.scenes_per_family]:
            selected.append((fam, lf))
    logger.info("Selected %d scenes across %d families", len(selected), len(per_family))

    scene_results: list[dict] = []
    for idx, (fam, lf) in enumerate(selected):
        scene_rng = random.Random(args.seed + idx * 7919)
        try:
            res = audit_scene(
                model, label_file=lf, family=fam, split=args.split,
                rollout_root=args.rollout_root, render_root=args.render_root,
                corpus_root=args.manifest_corpus, device=device,
                frames_per_scene=args.frames_per_scene, max_per_cell=args.max_per_cell,
                max_pairs=args.max_pairs_per_scene, batch_size=args.batch_size,
                min_cells=args.min_cells, rng=scene_rng,
            )
        except Exception as exc:  # noqa: BLE001
            res = {"scene_id": lf.parent.name, "family": fam, "skipped": f"error:{exc}"}
        scene_results.append(res)
        tag = res.get("spearman_proj")
        logger.info(
            "[%d/%d] %s rho_proj=%s frames=%s cells=%s%s",
            idx + 1, len(selected), res["scene_id"],
            f"{tag:.3f}" if isinstance(tag, float) else tag,
            res.get("n_frames"), res.get("distinct_cells"),
            f"  SKIP({res['skipped']})" if "skipped" in res else "",
        )

    used = [r for r in scene_results if "spearman_proj" in r and r["spearman_proj"] is not None]
    rho_proj = [r["spearman_proj"] for r in used]
    rho_proj_ym = [r.get("spearman_proj_yawmatched") for r in used]
    rho_raw = [r.get("spearman_raw") for r in used]

    # Aggregate the bucket table across scenes.
    bucket_acc: dict[str, list[float]] = defaultdict(list)
    bucket_n: dict[str, int] = defaultdict(int)
    for r in used:
        for name, b in r.get("bucket_norm_latent", {}).items():
            bucket_acc[name].append(b["norm_latent_median"])
            bucket_n[name] += b["n"]
    bucket_summary = {
        name: {"scenes": len(bucket_acc[name]),
               "pairs": bucket_n[name],
               "norm_latent_median_across_scenes": float(np.median(bucket_acc[name]))}
        for name, _, _ in BUCKETS if name in bucket_acc
    }

    agg_proj = _agg(rho_proj)
    interp = "insufficient (<0.40)"
    if agg_proj.get("median", 0) >= 0.70:
        interp = "strong (>=0.70)"
    elif agg_proj.get("median", 0) >= 0.40:
        interp = "ambiguous (0.40-0.70)"

    record = {
        "schema": "lewm_latent_aliasing_audit_v0",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "manifest_corpus": str(args.manifest_corpus),
        "scenes_used": len(used),
        "scenes_selected": len(selected),
        "spearman_proj_per_scene": _agg(rho_proj),
        "spearman_proj_yawmatched_per_scene": _agg(rho_proj_ym),
        "spearman_raw_per_scene": _agg(rho_raw),
        "confusion_nearest10pct_graphfar_frac": _agg(
            [r.get("confusion_nearest10pct_graphfar_frac") for r in used]
        ),
        "bucket_norm_latent_across_scenes": bucket_summary,
        "a4_interpretation_proj_median": interp,
        "config": {
            "scenes_per_family": args.scenes_per_family,
            "frames_per_scene": args.frames_per_scene,
            "max_per_cell": args.max_per_cell,
            "max_pairs_per_scene": args.max_pairs_per_scene,
            "seed": args.seed,
        },
        "scenes": scene_results,
    }

    print(json.dumps({k: v for k, v in record.items() if k != "scenes"}, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
