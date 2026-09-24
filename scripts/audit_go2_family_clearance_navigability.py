#!/usr/bin/env python3
"""Audit per-family clearance with a yaw-invariant disc proxy.

Answers the option-C question from
`docs/lewm_go2_campaign_first_principles_review_2026-07-31.md` §5:
which scene families are connected under the repository's conservative
yaw-invariant planning footprint, and how much of the existing corpus already
lives in them?

For every scene manifest it rasterizes walls/obstacles into an occupancy grid,
takes a Euclidean distance transform to get per-cell wall clearance, then:

- `frac_free_tight`   fraction of free area with clearance <= configured radius
                      (the statistic reported as 36-44% for medium mazes)
- `frac_reachable`    fraction of free area connected to the *actual spawn* by
                      the configured disc; zero when spawn is infeasible
- `frac_largest_component` diagnostic connectivity independent of spawn; it is
                           never substituted for spawn reachability
- `ray_*`             mean free-ray distance from reachable poses, a proxy for
                      how wall-filled the ego camera frames are

This is a raster-disc planning diagnostic, not proof of physical Go2
navigability.  It does not model body orientation, turning sweeps, controller
dynamics, contacts, rough-terrain feasibility, or gait-dependent directional
clearance.  Connectivity is four-neighbour and therefore conservative with
respect to the privileged grid's corner-safe eight-neighbour A*.  Walls and
obstacles block at all heights (conservative; the robot cannot step over them).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from collections import defaultdict

import numpy as np
from scipy import ndimage

from lewm.datasets import go2_paired_navigation as paired_navigation

DEFAULT_FOOTPRINT_RADIUS_M = paired_navigation.OBSERVABLE_FOOTPRINT_RADIUS_M
GRID_RESOLUTION_M = 0.02
CONNECTIVITY = 4
REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_OUTPUT_ROOT = (REPO_ROOT / ".generated/dev").resolve()
SCENE_CORPUS_ROOT = (REPO_ROOT / ".generated/scene_corpus").resolve()
RAY_COUNT = 16
RAY_SAMPLE_CELLS = 160
RAY_MAX_M = 6.0


def read_bound_file(path: Path) -> tuple[bytes, dict[str, object]]:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {selected}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with selected.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise RuntimeError(f"input changed while it was read: {selected}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError(f"input byte count changed while it was read: {selected}")
    return raw, {
        "path": str(selected.resolve()),
        "byte_count": len(raw),
        "sha256": digest.hexdigest(),
    }


def assert_binding_unchanged(binding: dict[str, object]) -> None:
    _raw, current = read_bound_file(Path(str(binding["path"])))
    if current != binding:
        raise RuntimeError(f"input changed after audit: {binding['path']}")


def _occupancy(manifest: dict, resolution: float) -> tuple[np.ndarray, tuple[float, float], float]:
    (min_x, min_y), (max_x, max_y) = manifest["world_bounds_xy_m"]
    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))
    grid = np.zeros((height, width), dtype=bool)
    ys = min_y + (np.arange(height) + 0.5) * resolution
    xs = min_x + (np.arange(width) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)

    for body in (*manifest.get("walls", []), *manifest.get("obstacles", [])):
        cx, cy = body["center_xyz_m"][0], body["center_xyz_m"][1]
        sx, sy = body["size_xyz_m"][0], body["size_xyz_m"][1]
        yaw = float(body.get("yaw_rad", 0.0) or 0.0)
        dx, dy = gx - cx, gy - cy
        if yaw:
            cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
            dx, dy = dx * cos_y - dy * sin_y, dx * sin_y + dy * cos_y
        grid |= (np.abs(dx) <= sx / 2.0) & (np.abs(dy) <= sy / 2.0)
    return grid, (min_x, min_y), resolution


def _ray_statistics(free: np.ndarray, cells: np.ndarray, resolution: float) -> dict:
    """Mean free-ray distance from sampled poses -- ego-frame openness proxy."""
    if cells.shape[0] == 0:
        return {"ray_mean_m": 0.0, "ray_median_m": 0.0}
    height, width = free.shape
    max_steps = int(RAY_MAX_M / resolution)
    angles = np.arange(RAY_COUNT) * (2.0 * math.pi / RAY_COUNT)
    dir_r = np.sin(angles)
    dir_c = np.cos(angles)
    means = []
    for row, col in cells:
        distances = np.full(RAY_COUNT, RAY_MAX_M)
        for index in range(RAY_COUNT):
            for step in range(1, max_steps):
                r = int(row + dir_r[index] * step)
                c = int(col + dir_c[index] * step)
                if r < 0 or c < 0 or r >= height or c >= width or not free[r, c]:
                    distances[index] = step * resolution
                    break
        means.append(float(distances.mean()))
    return {
        "ray_mean_m": float(statistics.fmean(means)),
        "ray_median_m": float(statistics.median(means)),
    }


def audit_scene(args) -> dict:
    scene_id, family, split, manifest_path, footprint_radius = args
    path = Path(manifest_path)
    try:
        manifest_bytes, manifest_binding = read_bound_file(path)
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read scene manifest {manifest_path}: {exc}") from exc

    occupied, (min_x, min_y), resolution = _occupancy(manifest, GRID_RESOLUTION_M)
    free = ~occupied
    free_cells = int(free.sum())
    if free_cells == 0:
        raise ValueError(f"scene {scene_id} has no free raster cells")

    # Clearance: distance from each free cell to the nearest occupied cell.
    # Treat outside-the-grid as occupied so boundary walls count.
    padded = np.pad(free, 1, mode="constant", constant_values=False)
    clearance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * resolution

    traversable = clearance >= footprint_radius
    spawn = manifest.get("spawn", {}).get("xyz_m", [0.0, 0.0, 0.0])
    srow = math.floor((spawn[1] - min_y) / resolution)
    scol = math.floor((spawn[0] - min_x) / resolution)
    spawn_in_bounds = 0 <= srow < free.shape[0] and 0 <= scol < free.shape[1]

    labels, component_count = ndimage.label(traversable)
    spawn_label = int(labels[srow, scol]) if spawn_in_bounds else 0
    spawn_traversable = spawn_label > 0
    reachable = (labels == spawn_label) if spawn_label else np.zeros_like(traversable)
    sizes = np.bincount(labels.ravel())
    largest_component_cells = (
        int(sizes[1:].max()) if component_count > 0 and sizes.size > 1 else 0
    )
    traversable_cells = int(traversable.sum())
    reachable_cells = int(reachable.sum())

    free_clear = clearance[free]
    sample = np.argwhere(reachable)
    if sample.shape[0] > RAY_SAMPLE_CELLS:
        stride = sample.shape[0] // RAY_SAMPLE_CELLS
        sample = sample[::stride][:RAY_SAMPLE_CELLS]

    return {
        "scene_id": scene_id,
        "family": family,
        "split": split,
        "manifest_path": manifest_binding["path"],
        "manifest_bytes": manifest_binding["byte_count"],
        "manifest_sha256": manifest_binding["sha256"],
        "free_area_m2": free_cells * resolution * resolution,
        "footprint_radius_m": footprint_radius,
        "geometry_model": "yaw_invariant_raster_disc_proxy_v1",
        "connectivity": CONNECTIVITY,
        "physical_navigability_proven": False,
        "frac_free_tight": float((free_clear <= footprint_radius).mean()),
        "frac_free_traversable": float(traversable_cells / free_cells),
        "frac_reachable": float(reachable_cells / free_cells),
        "frac_reachable_of_traversable": (
            float(reachable_cells / traversable_cells) if traversable_cells else 0.0
        ),
        "frac_largest_component": float(largest_component_cells / free_cells),
        "spawn_in_bounds": spawn_in_bounds,
        "spawn_traversable": spawn_traversable,
        "clearance_median_m": float(np.median(free_clear)),
        "clearance_p90_m": float(np.percentile(free_clear, 90)),
        **_ray_statistics(free, sample, resolution),
    }


def summarize_records(records: list[dict]) -> dict:
    """Return machine-derived corpus and per-family summaries."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["family"])].append(record)

    def summarize(rows: list[dict]) -> dict:
        connected = [row["frac_reachable_of_traversable"] for row in rows]
        return {
            "scene_count": len(rows),
            "spawn_in_bounds_count": sum(bool(row["spawn_in_bounds"]) for row in rows),
            "spawn_traversable_count": sum(
                bool(row["spawn_traversable"]) for row in rows
            ),
            "fully_spawn_connected_count": sum(
                math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
                for value in connected
            ),
            "frac_free_traversable_mean": float(
                statistics.fmean(row["frac_free_traversable"] for row in rows)
            ),
            "frac_reachable_mean": float(
                statistics.fmean(row["frac_reachable"] for row in rows)
            ),
            "frac_reachable_of_traversable_mean": float(
                statistics.fmean(connected)
            ),
            "frac_reachable_of_traversable_median": float(
                statistics.median(connected)
            ),
            "frac_largest_component_mean": float(
                statistics.fmean(row["frac_largest_component"] for row in rows)
            ),
        }

    return {
        "all": summarize(records) if records else {"scene_count": 0},
        "families": {
            family: summarize(rows) for family, rows in sorted(grouped.items())
        },
    }


def require_development_output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(DEV_OUTPUT_ROOT):
        raise ValueError(f"development output must remain under {DEV_OUTPUT_ROOT}")
    return resolved


def require_scene_corpus(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(SCENE_CORPUS_ROOT):
        raise ValueError(f"corpus input must remain under {SCENE_CORPUS_ROOT}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=".generated/scene_corpus/minimum_tex_20260520T211541Z",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--footprint-radius",
        "--robot-half-width",
        dest="footprint_radius",
        type=float,
        default=DEFAULT_FOOTPRINT_RADIUS_M,
        help=(
            "Yaw-invariant disc radius in metres. The legacy "
            "--robot-half-width spelling is retained as an alias."
        ),
    )
    parser.add_argument(
        "--out", default=".generated/dev/family_clearance_audit/audit.json"
    )
    args = parser.parse_args()
    if not math.isfinite(args.footprint_radius) or args.footprint_radius <= 0.0:
        parser.error("--footprint-radius must be finite and positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    try:
        out_path = require_development_output(Path(args.out))
    except ValueError as exc:
        parser.error(str(exc))

    try:
        corpus_root = require_scene_corpus(Path(args.corpus))
    except ValueError as exc:
        parser.error(str(exc))
    corpus_path = corpus_root / "corpus.json"
    corpus_bytes, corpus_binding = read_bound_file(corpus_path)
    corpus = json.loads(corpus_bytes)
    source_bindings = [
        read_bound_file(Path(path))[1]
        for path in (__file__, paired_navigation.__file__)
    ]
    jobs = []
    for scene in corpus["scenes"]:
        manifest = (corpus_root / scene["relative_dir"] / "manifest.json").resolve()
        if not manifest.is_relative_to(corpus_root):
            raise ValueError(
                f"scene manifest escapes corpus root: {scene['relative_dir']}"
            )
        if not manifest.is_file():
            raise FileNotFoundError(f"missing scene manifest: {manifest}")
        jobs.append(
            (
                scene["scene_id"],
                scene["family"],
                scene["split"],
                str(manifest),
                args.footprint_radius,
            )
        )
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"auditing {len(jobs)} scenes with {args.workers} workers", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(audit_scene, jobs, chunksize=4), 1):
            records.append(record)
            if index % 200 == 0:
                print(f"  {index}/{len(jobs)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_binding_unchanged(corpus_binding)
    for record in records:
        assert_binding_unchanged(
            {
                "path": record["manifest_path"],
                "byte_count": record["manifest_bytes"],
                "sha256": record["manifest_sha256"],
            }
        )
    for binding in source_bindings:
        assert_binding_unchanged(binding)
    payload = {
        "schema": "go2_family_clearance_navigability_audit_v2",
        "citable_as_scientific_evidence": False,
        "geometry_model": "yaw_invariant_raster_disc_proxy_v1",
        "connectivity": CONNECTIVITY,
        "physical_navigability_proven": False,
        "footprint_radius_m": args.footprint_radius,
        "canonical_footprint_radius_m": DEFAULT_FOOTPRINT_RADIUS_M,
        "grid_resolution_m": GRID_RESOLUTION_M,
        "corpus": str(corpus_root),
        "corpus_manifest_bytes": corpus_binding["byte_count"],
        "corpus_manifest_sha256": corpus_binding["sha256"],
        "source_bindings": source_bindings,
        "corpus_declared_scene_count": len(corpus["scenes"]),
        "audited_scene_count": len(records),
        "limit": args.limit,
        "summary": summarize_records(records),
        "records": records,
    }
    temporary_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if (
        out_path.exists()
        or out_path.is_symlink()
        or temporary_path.exists()
        or temporary_path.is_symlink()
    ):
        raise FileExistsError(f"refusing to overwrite geometry audit: {out_path}")
    with temporary_path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    try:
        os.link(temporary_path, out_path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite geometry audit: {out_path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"wrote {out_path} ({len(records)} scenes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
