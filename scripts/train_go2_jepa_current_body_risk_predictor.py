#!/usr/bin/env python3
"""Train a frozen-JEPA current-body-risk classifier for Go2 wall clearance.

Labels are built offline from rendered RGB rows, matching raw base-state poses,
and the scene occupancy grid. The saved model is queried from RGB/JEPA latent at
runtime, so closed-loop traversal decisions remain nonprivileged.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.go2_jepa import Go2FrontBlockedHead, load_go2_jepa_encoder  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from train_go2_hidden_target_memory_probe import _resolve_device  # noqa: E402
from train_go2_jepa_front_blocked_predictor import (  # noqa: E402
    _metrics,
    _precompute_latents,
    _train,
)


@dataclass(frozen=True)
class Example:
    rgb_path: Path
    scene_id: str
    env_idx: int
    timestamp_ns: int
    body_clearance_m: float
    label: float


def _body_probe_configuration_clearance_m(
    grid: InflatedOccupancyGrid,
    xy: tuple[float, float],
    yaw: float,
    *,
    body_forward_m: float,
    body_half_width_m: float,
) -> float:
    x, y = float(xy[0]), float(xy[1])
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    lx, ly = -fy, fx
    probes = (
        (0.0, 0.0),
        (body_forward_m, 0.0),
        (body_forward_m, body_half_width_m),
        (body_forward_m, -body_half_width_m),
        (0.0, body_half_width_m),
        (0.0, -body_half_width_m),
    )
    return float(
        min(
            grid.configuration_clearance_m(
                (x + forward * fx + lateral * lx, y + forward * fy + lateral * ly)
            )
            for forward, lateral in probes
        )
    )


def _scene_grid(
    scene_id: str,
    *,
    scene_dirs_by_id: dict[str, Path],
    platform_manifest: dict[str, Any],
    inflation_m: float,
    cache: dict[str, InflatedOccupancyGrid],
) -> InflatedOccupancyGrid | None:
    if scene_id in cache:
        return cache[scene_id]
    scene_dir = scene_dirs_by_id.get(scene_id)
    if scene_dir is None:
        return None
    pack = load_scene_pack(
        scene_dir,
        platform_manifest=platform_manifest,
        workspace_root=REPO_ROOT,
    )
    grid = InflatedOccupancyGrid(
        pack.scene_graph.manifest,
        cell_size_m=0.05,
        inflation_m=float(inflation_m),
    )
    cache[scene_id] = grid
    return grid


def _infer_scene_id(row: dict[str, Any], dataset_path: Path, rgb_path: Path) -> str:
    explicit_scene_id = row.get("scene_id")
    if explicit_scene_id:
        return str(explicit_scene_id)
    if rgb_path.parent.name == "rgb":
        return rgb_path.parent.parent.name
    return dataset_path.parent.name


def _infer_raw_messages_path(
    dataset_path: Path,
    scene_id: str,
    *,
    rgb_path: Path | None = None,
) -> Path | None:
    family = "_".join(str(scene_id).split("_")[:-1])
    for parent in (dataset_path.parent, *dataset_path.parents):
        candidates = [parent / "raw" / scene_id / "messages.jsonl"]
        if family:
            candidates.append(parent / "raw" / family / scene_id / "messages.jsonl")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    if rgb_path is not None:
        for parent in (rgb_path.parent, *rgb_path.parents):
            if parent.name != "render":
                continue
            candidates = [parent.parent / "raw" / scene_id / "messages.jsonl"]
            if family:
                candidates.append(parent.parent / "raw" / family / scene_id / "messages.jsonl")
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    return None


def _load_pose_index(
    raw_messages_path: Path,
    *,
    cache: dict[Path, dict[tuple[int, int], tuple[float, float, float]]],
) -> dict[tuple[int, int], tuple[float, float, float]]:
    raw_messages_path = raw_messages_path.resolve()
    if raw_messages_path in cache:
        return cache[raw_messages_path]
    poses: dict[tuple[int, int], tuple[float, float, float]] = {}
    with raw_messages_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            msg = json.loads(line)
            if str(msg.get("canonical_topic", "")) != "/lewm/go2/base_state":
                continue
            payload = msg.get("payload") or {}
            pose_world = payload.get("pose_world") or {}
            position = pose_world.get("position") or {}
            env_idx = int(msg.get("env_index", msg.get("env_idx", -1)))
            timestamp_ns = int(msg.get("timestamp_ns", -1))
            if env_idx < 0 or timestamp_ns < 0:
                continue
            poses[(env_idx, timestamp_ns)] = (
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(payload.get("yaw_rad", 0.0)),
            )
    cache[raw_messages_path] = poses
    return poses


def _load_examples(
    paths: list[Path],
    *,
    scene_dirs_by_id: dict[str, Path],
    platform_manifest: dict[str, Any],
    inflation_m: float,
    body_forward_m: float,
    body_half_width_m: float,
    risk_margin_m: float,
    max_examples: int | None,
) -> tuple[list[Example], dict[str, int]]:
    examples: list[Example] = []
    seen_rgb: set[str] = set()
    pose_cache: dict[Path, dict[tuple[int, int], tuple[float, float, float]]] = {}
    grid_cache: dict[str, InflatedOccupancyGrid] = {}
    skipped = {
        "camera_invalid": 0,
        "missing_rgb": 0,
        "missing_scene": 0,
        "missing_raw": 0,
        "missing_pose": 0,
        "duplicate_rgb": 0,
        "direct_label_rows": 0,
    }
    for dataset_path in paths:
        with dataset_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not bool(row.get("camera_valid", True)):
                    skipped["camera_invalid"] += 1
                    continue
                rgb_path = Path(str(row.get("rgb_path", "")))
                if not rgb_path.is_file():
                    skipped["missing_rgb"] += 1
                    continue
                rgb_key = str(rgb_path.resolve())
                if rgb_key in seen_rgb:
                    skipped["duplicate_rgb"] += 1
                    continue
                scene_id = _infer_scene_id(row, dataset_path, rgb_path)
                direct_traversability = row.get("traversability_forward_m")
                if direct_traversability is not None and (
                    row.get("env_idx", row.get("env_index")) is None
                    or row.get("timestamp_ns") is None
                ):
                    direct_label = 1.0 if float(direct_traversability) <= 0.0 else 0.0
                    if row.get("source_clearance_m") is not None:
                        direct_clearance = float(row["source_clearance_m"])
                    else:
                        direct_clearance = (
                            0.0 if direct_label >= 0.5 else float(risk_margin_m) * 2.0
                        )
                    seen_rgb.add(rgb_key)
                    skipped["direct_label_rows"] += 1
                    examples.append(
                        Example(
                            rgb_path=rgb_path,
                            scene_id=scene_id,
                            env_idx=-1,
                            timestamp_ns=-1,
                            body_clearance_m=direct_clearance,
                            label=direct_label,
                        )
                    )
                    if max_examples is not None and len(examples) >= int(max_examples):
                        return examples, skipped
                    continue
                grid = _scene_grid(
                    scene_id,
                    scene_dirs_by_id=scene_dirs_by_id,
                    platform_manifest=platform_manifest,
                    inflation_m=float(inflation_m),
                    cache=grid_cache,
                )
                if grid is None:
                    skipped["missing_scene"] += 1
                    continue
                raw_messages_path = _infer_raw_messages_path(
                    dataset_path,
                    scene_id,
                    rgb_path=rgb_path,
                )
                if raw_messages_path is None:
                    skipped["missing_raw"] += 1
                    continue
                poses = _load_pose_index(raw_messages_path, cache=pose_cache)
                env_idx = int(row.get("env_idx", row.get("env_index", -1)))
                timestamp_ns = int(row.get("timestamp_ns", -1))
                pose = poses.get((env_idx, timestamp_ns))
                if pose is None:
                    skipped["missing_pose"] += 1
                    continue
                clearance_m = _body_probe_configuration_clearance_m(
                    grid,
                    (pose[0], pose[1]),
                    pose[2],
                    body_forward_m=float(body_forward_m),
                    body_half_width_m=float(body_half_width_m),
                )
                seen_rgb.add(rgb_key)
                examples.append(
                    Example(
                        rgb_path=rgb_path,
                        scene_id=scene_id,
                        env_idx=env_idx,
                        timestamp_ns=timestamp_ns,
                        body_clearance_m=clearance_m,
                        label=1.0 if clearance_m < float(risk_margin_m) else 0.0,
                    )
                )
                if max_examples is not None and len(examples) >= int(max_examples):
                    return examples, skipped
    return examples, skipped


def _clearance_stats(examples: list[Example]) -> dict[str, Any]:
    if not examples:
        return {}
    vals = np.asarray([item.body_clearance_m for item in examples], dtype=np.float32)
    return {
        "min": float(np.min(vals)),
        "p10": float(np.percentile(vals, 10)),
        "median": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "share_under_0p00": float(np.mean(vals < 0.0)),
        "share_under_0p03": float(np.mean(vals < 0.03)),
        "share_under_0p06": float(np.mean(vals < 0.06)),
        "share_under_0p08": float(np.mean(vals < 0.08)),
    }


def _threshold_sweep(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return {
        f"{threshold:.2f}": _metrics(logits, labels, threshold=threshold)
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="*", type=Path, default=None)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--scene-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z")
    parser.add_argument("--extra-scene-corpus", action="append", type=Path, default=[])
    parser.add_argument("--platform-manifest", type=Path, default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--inflation-m", type=float, default=0.12)
    parser.add_argument("--body-forward-m", type=float, default=0.40)
    parser.add_argument("--body-half-width-m", type=float, default=0.24)
    parser.add_argument("--risk-margin-m", type=float, default=0.06)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    platform_manifest = load_platform_manifest(args.platform_manifest.resolve())
    scene_dirs_by_id: dict[str, Path] = {}
    scene_corpora = [args.scene_corpus.resolve(), *(path.resolve() for path in args.extra_scene_corpus)]
    for scene_corpus in scene_corpora:
        for scene_dir in find_scene_dirs(scene_corpus, family=str(args.family)):
            scene_dirs_by_id.setdefault(scene_dir.name, scene_dir)
    train_examples, train_skipped = _load_examples(
        args.datasets,
        scene_dirs_by_id=scene_dirs_by_id,
        platform_manifest=platform_manifest,
        inflation_m=float(args.inflation_m),
        body_forward_m=float(args.body_forward_m),
        body_half_width_m=float(args.body_half_width_m),
        risk_margin_m=float(args.risk_margin_m),
        max_examples=args.max_train_examples,
    )
    if args.validation_datasets:
        val_examples, val_skipped = _load_examples(
            args.validation_datasets,
            scene_dirs_by_id=scene_dirs_by_id,
            platform_manifest=platform_manifest,
            inflation_m=float(args.inflation_m),
            body_forward_m=float(args.body_forward_m),
            body_half_width_m=float(args.body_half_width_m),
            risk_margin_m=float(args.risk_margin_m),
            max_examples=args.max_validation_examples,
        )
    else:
        random.shuffle(train_examples)
        cut = max(1, int(len(train_examples) * 0.8))
        val_examples = train_examples[cut:]
        train_examples = train_examples[:cut]
        val_skipped = {}
    if not train_examples:
        raise SystemExit("no train examples")
    if not val_examples:
        raise SystemExit("no validation examples")

    device = _resolve_device(str(args.device))
    encoder, encoder_checkpoint = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint,
        device=device,
        freeze=True,
    )
    train_latents, train_labels = _precompute_latents(
        encoder,
        train_examples,
        image_size=int(args.image_size),
        device=device,
        batch_size=int(args.batch_size),
    )
    val_latents, val_labels = _precompute_latents(
        encoder,
        val_examples,
        image_size=int(args.image_size),
        device=device,
        batch_size=int(args.batch_size),
    )
    latent_dim = int(encoder_checkpoint.get("latent_dim", train_latents.shape[-1]))
    model = Go2FrontBlockedHead(latent_dim=latent_dim, hidden_dim=int(args.hidden_dim))
    best_state, best_metrics, history = _train(
        model,
        train_latents,
        train_labels,
        val_latents,
        val_labels,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        threshold=float(args.threshold),
    )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(train_latents).cpu()
        val_logits = model(val_latents).cpu()
    train_metrics = _metrics(train_logits, train_labels.cpu(), threshold=float(args.threshold))
    val_metrics = _metrics(val_logits, val_labels.cpu(), threshold=float(args.threshold))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "go2_jepa_current_body_risk_predictor_v0",
            "model_state_dict": best_state,
            "latent_dim": latent_dim,
            "hidden_dim": int(args.hidden_dim),
            "image_size": int(args.image_size),
            "threshold": float(args.threshold),
            "risk_margin_m": float(args.risk_margin_m),
            "inflation_m": float(args.inflation_m),
            "body_forward_m": float(args.body_forward_m),
            "body_half_width_m": float(args.body_half_width_m),
            "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        },
        args.output,
    )
    report = {
        "schema": "go2_jepa_current_body_risk_predictor_report_v0",
        "checkpoint": str(args.output),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "train_scenes": sorted({item.scene_id for item in train_examples}),
        "validation_scenes": sorted({item.scene_id for item in val_examples}),
        "train_skipped": train_skipped,
        "validation_skipped": val_skipped,
        "risk_margin_m": float(args.risk_margin_m),
        "inflation_m": float(args.inflation_m),
        "body_forward_m": float(args.body_forward_m),
        "body_half_width_m": float(args.body_half_width_m),
        "threshold": float(args.threshold),
        "scene_corpora": [str(path) for path in scene_corpora],
        "train_clearance": _clearance_stats(train_examples),
        "validation_clearance": _clearance_stats(val_examples),
        "train": train_metrics,
        "validation": val_metrics,
        "best_validation": best_metrics,
        "threshold_sweep": {
            "train": _threshold_sweep(train_logits, train_labels.cpu()),
            "validation": _threshold_sweep(val_logits, val_labels.cpu()),
        },
        "history": history,
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "schema",
                    "checkpoint",
                    "train_examples",
                    "validation_examples",
                    "train",
                    "validation",
                    "train_clearance",
                    "validation_clearance",
                )
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
