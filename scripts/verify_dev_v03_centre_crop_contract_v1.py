#!/usr/bin/env python3
"""Gate: does a 224x224 -> 224x168 centre crop of the v03 render reproduce v04?

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The dense ``render_textured_v03`` frames are the only source of genuine
three-frame history, but the corpus, its ``raster_labels.u1`` observability mask
and the frozen dense-representation screen all live in the ``textured_v04``
contract (224x168).  This checks the crop analytically and empirically before any
temporal work depends on it.

Analytic: same focal length => cropping rows only narrows the vertical FOV.
Empirical: gradient NCC against the co-indexed v04 frame, over many endpoints,
with a crop-offset and scale sweep so "centre" and "same scale" are measured
rather than assumed.  The texture sets differ (textured_v03 vs textured_v04), so
a perfect correlation is not expected; the geometry is what must agree.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUP = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
V03 = ROOT / ".generated/datagen_full/render_textured_v03"
OUT = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/crop_gate")

HORIZONTAL_FOV_DEG = 78.323          # config/go2_generalization_geometry_v2.json
V04_WH = (224, 168)
V03_WH = (224, 224)
UNKNOWN = 0


def analytic() -> dict:
    """Pinhole geometry of both contracts, from the shared horizontal FOV."""
    f = (V04_WH[0] / 2) / math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2)
    v04_v = math.degrees(2 * math.atan((V04_WH[1] / 2) / f))
    v03_v = math.degrees(2 * math.atan((V03_WH[1] / 2) / f))
    crop_v = math.degrees(2 * math.atan((V04_WH[1] / 2) / f))
    return {
        "shared_focal_length_px": f,
        "v04": {"wh": list(V04_WH), "h_fov_deg": HORIZONTAL_FOV_DEG, "v_fov_deg": v04_v},
        "v03_native": {"wh": list(V03_WH), "h_fov_deg": HORIZONTAL_FOV_DEG, "v_fov_deg": v03_v},
        "v03_centre_cropped": {
            "wh": list(V04_WH),
            "h_fov_deg": HORIZONTAL_FOV_DEG,
            "v_fov_deg": crop_v,
            "rows_removed_top_bottom": (V03_WH[1] - V04_WH[1]) // 2,
        },
        "vertical_fov_error_deg": crop_v - v04_v,
        "horizontal_fov_error_deg": 0.0,
        "note": (
            "cropping rows changes only the vertical extent at fixed focal length, "
            "so the horizontal FOV is preserved exactly and the vertical FOV becomes "
            "the v04 value by construction -- CONDITIONAL on the two renders sharing "
            "a focal length, which the empirical scale sweep below measures."
        ),
    }


def centre_crop(image: Image.Image) -> Image.Image:
    if image.size != V03_WH:
        raise ValueError(f"expected {V03_WH}, got {image.size}")
    top = (V03_WH[1] - V04_WH[1]) // 2
    return image.crop((0, top, V03_WH[0], top + V04_WH[1]))


def _grad(x: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(x)
    g = np.hypot(gx, gy)
    return (g - g.mean()) / (g.std() + 1e-8)


def _ncc(x: np.ndarray, y: np.ndarray) -> float:
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float((x * y).mean())


def load_rows() -> list[dict]:
    endpoints = {
        e["endpoint_identity_sha256"]: e
        for e in (
            json.loads(line)
            for line in (SUP / "endpoints.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    rows = []
    for line in (SUP / "pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        if pair["dataset_role"] not in ("train", "checkpoint_selection"):
            continue
        e = endpoints[pair["current_endpoint_sha256"]]
        name = Path(e["image_path_metadata_only"]).name
        v03 = V03 / pair["scene_id"] / "rgb" / name
        if not v03.is_file():
            continue
        rows.append(
            {
                "v04": e["image_path_metadata_only"],
                "v03": str(v03),
                "scene": pair["scene_id"],
                "family": pair["family"],
                "role": pair["dataset_role"],
                "shard_dir": str(SUP / Path(e["scene_shard"]).parent),
                "shard_row": int(e["shard_row"]),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=240)
    ap.add_argument("--overlays", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    geometry = analytic()
    rows = load_rows()
    rng = random.Random(20260806)
    sample = rng.sample(rows, min(args.samples, len(rows)))

    # --- empirical: is the crop centred, and are the scales equal? -----------
    offsets = list(range(0, 57, 4))
    scales = [0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 4 / 3]
    offset_scores = {o: [] for o in offsets}
    scale_scores = {s: [] for s in scales}
    crop_scores, resize_scores, per_family = [], [], {}

    for row in sample:
        v4 = np.asarray(Image.open(row["v04"]).convert("L"), dtype=np.float32)
        v3i = Image.open(row["v03"]).convert("L")
        v3 = np.asarray(v3i, dtype=np.float32)
        g4 = _grad(v4)
        for o in offsets:
            offset_scores[o].append(_ncc(_grad(v3[o : o + 168, :]), g4))
        for s in scales:
            w = int(round(224 * s))
            im = np.asarray(v3i.resize((w, w), Image.BICUBIC), dtype=np.float32)
            if w < 224:
                pad = (224 - w) // 2
                im = np.pad(im, ((pad, 224 - w - pad), (pad, 224 - w - pad)), mode="edge")
            x0, y0 = (im.shape[1] - 224) // 2, (im.shape[0] - 168) // 2
            scale_scores[s].append(_ncc(_grad(im[y0 : y0 + 168, x0 : x0 + 224]), g4))
        crop_scores.append(_ncc(_grad(np.asarray(centre_crop(v3i), dtype=np.float32)), g4))
        resize_scores.append(
            _ncc(_grad(np.asarray(v3i.resize((224, 168), Image.BICUBIC), dtype=np.float32)), g4)
        )
        per_family.setdefault(row["family"], []).append(crop_scores[-1])

    best_offset = max(offsets, key=lambda o: float(np.mean(offset_scores[o])))
    best_scale = max(scales, key=lambda s: float(np.mean(scale_scores[s])))
    crop_mean = float(np.mean(crop_scores))

    # --- raster overlays: crop vs v04 vs the labelled observability mask -----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_cache: dict[str, np.ndarray] = {}
    picks = rng.sample(sample, min(args.overlays, len(sample)))
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 2.7 * len(picks)))
    colours = np.array([[40, 40, 48], [70, 130, 180], [220, 80, 60]], dtype=np.uint8)
    for r, row in enumerate(picks):
        if row["shard_dir"] not in label_cache:
            label_cache[row["shard_dir"]] = np.fromfile(
                Path(row["shard_dir"]) / "raster_labels.u1", dtype=np.uint8
            ).reshape(-1, 64, 64)
        label = label_cache[row["shard_dir"]][row["shard_row"]]
        axes[r, 0].imshow(Image.open(row["v04"]).convert("RGB"))
        axes[r, 0].set_title("v04 224x168 (contract)", fontsize=7)
        axes[r, 1].imshow(centre_crop(Image.open(row["v03"]).convert("RGB")))
        axes[r, 1].set_title("v03 centre-cropped 224x168", fontsize=7)
        axes[r, 2].imshow(np.asarray(Image.open(row["v03"]).convert("RGB")))
        axes[r, 2].axhline(28, color="lime", lw=1)
        axes[r, 2].axhline(196, color="lime", lw=1)
        axes[r, 2].set_title("v03 native 224x224 + crop rows", fontsize=7)
        axes[r, 3].imshow(colours[label])
        axes[r, 3].set_title(
            f"raster_labels observable {(label != UNKNOWN).mean():.1%}\n{row['family']}",
            fontsize=7,
        )
        for c in range(4):
            axes[r, c].axis("off")
    fig.suptitle("v03 centre-crop contract check vs v04 and the labelled observability mask",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "crop_overlays.png", dpi=110)
    plt.close(fig)

    passed = (
        best_offset == (V03_WH[1] - V04_WH[1]) // 2
        and abs(best_scale - 1.0) < 1e-9
        and crop_mean > float(np.mean(resize_scores))
    )
    result = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING",
        "analytic_geometry": geometry,
        "samples": len(sample),
        "empirical": {
            "metric": "gradient-magnitude normalised cross-correlation vs the co-indexed v04 frame",
            "crop_offset_sweep_mean": {str(o): float(np.mean(offset_scores[o])) for o in offsets},
            "best_crop_offset_row": best_offset,
            "expected_centre_offset_row": (V03_WH[1] - V04_WH[1]) // 2,
            "scale_sweep_mean": {f"{s:.4f}": float(np.mean(scale_scores[s])) for s in scales},
            "best_scale": best_scale,
            "centre_crop_mean_ncc": crop_mean,
            "centre_crop_min_ncc": float(np.min(crop_scores)),
            "anisotropic_resize_mean_ncc": float(np.mean(resize_scores)),
            "per_family_centre_crop_mean_ncc": {
                k: float(np.mean(v)) for k, v in sorted(per_family.items())
            },
        },
        "gate": {
            "best_offset_is_centre": best_offset == (V03_WH[1] - V04_WH[1]) // 2,
            "best_scale_is_unity": abs(best_scale - 1.0) < 1e-9,
            "crop_beats_anisotropic_resize": crop_mean > float(np.mean(resize_scores)),
            "PASSED": bool(passed),
        },
        "overlays": str(OUT / "crop_overlays.png"),
        "caveat": (
            "texture sets differ (textured_v03 vs textured_v04), so NCC < 1 is expected; "
            "the claim is geometric correspondence, not pixel identity"
        ),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "empirical"}, indent=2))
    print(json.dumps(result["empirical"], indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
