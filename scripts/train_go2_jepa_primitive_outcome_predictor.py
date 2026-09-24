#!/usr/bin/env python3
"""Train an action-conditioned Go2 primitive outcome predictor."""
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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_genesis" / "lewm_genesis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_genesis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_worlds"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lewm.benchmarks.go2_primitive_outcome import (  # noqa: E402
    primitive_body_clearance_and_progress,
)
from lewm.models.go2_jepa import Go2PrimitiveOutcomeHead, load_go2_jepa_encoder  # noqa: E402
from lewm_contract import PrimitiveRegistry  # noqa: E402
from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from train_go2_hidden_target_memory_probe import _load_image, _resolve_device  # noqa: E402
from train_go2_rgb_jepa_vector_memory_controller import PRIMITIVE_NAMES  # noqa: E402


_TRANSLATING_PRIMITIVES = {"forward_medium", "arc_left", "arc_right", "backward"}


@dataclass(frozen=True)
class Example:
    rgb_path: Path
    scene_id: str
    primitive: str
    progress_m: float
    blocked: float
    clearance_m: float | None = None


def _as_motion_list(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, list | tuple) and len(value) >= 3:
        return float(value[0]), float(value[1]), float(value[2])
    return None


def _primitive(row: dict[str, Any]) -> str:
    if "primitive" in row:
        return str(row.get("primitive") or "")
    return str((row.get("command") or {}).get("primitive_name", ""))


def _progress_m(row: dict[str, Any]) -> float | None:
    if row.get("executed_displacement_m") is not None:
        return float(row["executed_displacement_m"])
    motion = _as_motion_list(row.get("integrated_body_motion_block"))
    if motion is not None:
        dx, dy, _dyaw = motion
        return float(math.hypot(dx, dy))
    return None


def _blocked_label(
    row: dict[str, Any],
    *,
    primitive: str,
    progress_m: float,
    min_progress_m: float,
    block_distance_m: float,
    include_traversability: bool,
    include_guard: bool,
) -> float:
    blocked = False
    if primitive in _TRANSLATING_PRIMITIVES and progress_m < float(min_progress_m):
        blocked = True
    if bool(row.get("stalled")):
        blocked = True
    if include_traversability and primitive in {"forward_medium", "arc_left", "arc_right"}:
        trav = row.get("traversability_forward_m")
        if trav is not None and float(trav) < float(block_distance_m):
            blocked = True
    if include_guard:
        guard = row.get("wall_guard")
        if isinstance(guard, dict) and bool(guard.get("requested_blocked")):
            blocked = True
    return 1.0 if blocked else 0.0


def _load_rows(
    paths: list[Path],
    *,
    primitive_vocab: list[str],
    min_progress_m: float,
    block_distance_m: float,
    include_traversability: bool,
    include_guard: bool,
) -> list[Example]:
    examples: list[Example] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not bool(row.get("camera_valid", True)):
                continue
            rgb_path = Path(str(row.get("rgb_path", "")))
            if not rgb_path.exists():
                continue
            primitive = _primitive(row)
            if primitive not in primitive_vocab:
                continue
            progress = _progress_m(row)
            if progress is None:
                continue
            key = (str(rgb_path), primitive)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                Example(
                    rgb_path=rgb_path,
                    scene_id=str(row.get("scene_id", path.parent.name)),
                    primitive=primitive,
                    progress_m=float(progress),
                    blocked=_blocked_label(
                        row,
                        primitive=primitive,
                        progress_m=float(progress),
                        min_progress_m=float(min_progress_m),
                        block_distance_m=float(block_distance_m),
                        include_traversability=bool(include_traversability),
                        include_guard=bool(include_guard),
                    ),
                )
            )
    return examples


def _load_counterfactual_body_clearance_rows(
    paths: list[Path],
    *,
    primitive_vocab: list[str],
    primitive_registry: Path,
    body_forward_m: float,
    body_half_width_m: float,
    body_clearance_margin_m: float,
    body_clearance_label_target: str,
    body_clearance_source: str,
    cell_size_m: float,
    inflation_m: float,
    max_source_rows: int,
    blocked_label_source: str = "clearance",
    progress_floor_m: float = 0.04,
) -> list[Example]:
    registry = PrimitiveRegistry.from_yaml(primitive_registry)
    known_primitives = [name for name in primitive_vocab if name in registry.primitives]
    if not known_primitives:
        raise SystemExit("no primitive vocab entries are present in the primitive registry")
    examples: list[Example] = []
    grids: dict[str, InflatedOccupancyGrid] = {}
    seen: set[tuple[str, str]] = set()
    source_rows = 0
    for path in paths:
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                if max_source_rows > 0 and source_rows >= max_source_rows:
                    break
                row = json.loads(line)
                rgb_path = Path(str(row.get("start_frame", "")))
                if not rgb_path.exists():
                    continue
                pose = row.get("start_base_pose_world") or {}
                position = pose.get("position") or {}
                rpy = row.get("start_base_rpy_rad") or {}
                if (
                    position.get("x") is None
                    or position.get("y") is None
                    or rpy.get("yaw") is None
                ):
                    continue
                manifest_path = str(row.get("scene_manifest", ""))
                if not manifest_path:
                    continue
                if manifest_path not in grids:
                    manifest_payload = json.loads(Path(manifest_path).read_text())
                    grids[manifest_path] = InflatedOccupancyGrid(
                        parse_scene_manifest_dict(manifest_payload),
                        cell_size_m=float(cell_size_m),
                        inflation_m=float(inflation_m),
                    )
                grid = grids[manifest_path]
                command_dt_s = float(row.get("command_dt_s", registry.command_dt_s))
                scene_id = str(row.get("scene_id", path.parent.name))
                x_m = float(position["x"])
                y_m = float(position["y"])
                yaw_rad = float(rpy["yaw"])
                source_rows += 1
                for primitive in known_primitives:
                    key = (str(rgb_path), primitive)
                    if key in seen:
                        continue
                    seen.add(key)
                    (
                        swept_clearance_m,
                        after_start_clearance_m,
                        final_clearance_m,
                        progress_m,
                    ) = primitive_body_clearance_and_progress(
                        registry=registry,
                        primitive=primitive,
                        grid=grid,
                        x_m=x_m,
                        y_m=y_m,
                        yaw_rad=yaw_rad,
                        command_dt_s=command_dt_s,
                        body_forward_m=float(body_forward_m),
                        body_half_width_m=float(body_half_width_m),
                        clearance_source=str(body_clearance_source),
                        progress_collision_stop_m=(
                            0.0 if str(blocked_label_source) == "progress" else None
                        ),
                    )
                    if body_clearance_label_target == "final":
                        clearance_m = final_clearance_m
                    elif body_clearance_label_target == "after_start_min":
                        clearance_m = after_start_clearance_m
                    else:
                        clearance_m = swept_clearance_m
                    examples.append(
                        Example(
                            rgb_path=rgb_path,
                            scene_id=scene_id,
                            primitive=primitive,
                            progress_m=progress_m,
                            blocked=(
                                (1.0 if progress_m < float(progress_floor_m) else 0.0)
                                if (
                                    str(blocked_label_source) == "progress"
                                    and primitive in _TRANSLATING_PRIMITIVES
                                )
                                else (1.0 if clearance_m < float(body_clearance_margin_m) else 0.0)
                            ),
                            clearance_m=clearance_m,
                        )
                    )
        if max_source_rows > 0 and source_rows >= max_source_rows:
            break
    return examples


def _split_by_scene(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    scenes = sorted({item.scene_id for item in examples})
    if len(scenes) <= 1:
        cut = max(1, int(0.8 * len(examples)))
        return examples[:cut], examples[cut:]
    val_scenes = set(scenes[::4] or scenes[-1:])
    train = [item for item in examples if item.scene_id not in val_scenes]
    val = [item for item in examples if item.scene_id in val_scenes]
    if not train or not val:
        cut = max(1, int(0.8 * len(examples)))
        train, val = examples[:cut], examples[cut:]
    return train, val


def _precompute(
    encoder: torch.nn.Module,
    examples: list[Example],
    *,
    primitive_to_idx: dict[str, int],
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_paths = list(dict.fromkeys(item.rgb_path for item in examples))
    latent_by_path: dict[Path, torch.Tensor] = {}
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(unique_paths), int(batch_size)):
            batch_paths = unique_paths[start:start + int(batch_size)]
            images = torch.stack([
                _load_image(path, image_size=image_size) for path in batch_paths
            ]).to(device)
            encoded = encoder(images).cpu()
            for path, latent in zip(batch_paths, encoded):
                latent_by_path[path] = latent
    return (
        torch.stack([latent_by_path[item.rgb_path] for item in examples], dim=0),
        torch.tensor([primitive_to_idx[item.primitive] for item in examples], dtype=torch.long),
        torch.tensor([float(item.blocked) for item in examples], dtype=torch.float32),
        torch.tensor([float(item.progress_m) for item in examples], dtype=torch.float32),
    )


def _one_hot(indices: torch.Tensor, primitive_count: int) -> torch.Tensor:
    return F.one_hot(indices, num_classes=int(primitive_count)).float()


def _metrics(
    blocked_logits: torch.Tensor,
    progress_pred: torch.Tensor,
    blocked: torch.Tensor,
    progress: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, Any]:
    probs = torch.sigmoid(blocked_logits)
    pred = probs >= float(threshold)
    target = blocked >= 0.5
    tp = int((pred & target).sum().item())
    fp = int((pred & ~target).sum().item())
    tn = int((~pred & ~target).sum().item())
    fn = int((~pred & target).sum().item())
    total = max(1, int(blocked.numel()))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    err = (progress_pred - progress).abs()
    return {
        "count": total,
        "positive_count": int(target.sum().item()),
        "positive_rate": float(target.float().mean().item()) if total else 0.0,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "progress_mae_m": float(err.mean().item()) if total else None,
        "progress_mean_m": float(progress.mean().item()) if total else None,
        "progress_pred_mean_m": float(progress_pred.mean().item()) if total else None,
    }


def _evaluate(
    model: Go2PrimitiveOutcomeHead,
    latents: torch.Tensor,
    primitive_idx: torch.Tensor,
    blocked: torch.Tensor,
    progress: torch.Tensor,
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits, progress_pred = model(
            latents.to(device),
            _one_hot(primitive_idx.to(device), model.primitive_count),
        )
    return _metrics(
        logits.cpu(),
        progress_pred.cpu().clamp_min(0.0),
        blocked.cpu(),
        progress.cpu(),
        threshold=float(threshold),
    )


def _train(
    model: Go2PrimitiveOutcomeHead,
    train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold: float,
    progress_loss_weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    train_latents, train_prim, train_blocked, train_progress = [x.to(device) for x in train_data]
    val_latents, val_prim, val_blocked, val_progress = [x.to(device) for x in val_data]
    model.to(device)
    pos = float(train_blocked.sum().item())
    neg = float(train_blocked.numel() - pos)
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -1e9
    history: list[dict[str, Any]] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(train_latents.shape[0], generator=gen)
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            idx = order[start:start + int(batch_size)].to(device)
            one_hot = _one_hot(train_prim[idx], model.primitive_count)
            logits, progress_pred = model(train_latents[idx], one_hot)
            blocked_loss = F.binary_cross_entropy_with_logits(
                logits,
                train_blocked[idx],
                pos_weight=pos_weight,
            )
            progress_loss = F.smooth_l1_loss(progress_pred, train_progress[idx])
            loss = blocked_loss + float(progress_loss_weight) * progress_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_metrics = _evaluate(
            model,
            train_latents,
            train_prim,
            train_blocked,
            train_progress,
            device=device,
            threshold=float(threshold),
        )
        val_metrics = _evaluate(
            model,
            val_latents,
            val_prim,
            val_blocked,
            val_progress,
            device=device,
            threshold=float(threshold),
        )
        score = float(val_metrics["f1"]) - 0.5 * float(val_metrics["progress_mae_m"] or 0.0)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": train_metrics,
            "validation": val_metrics,
        })
        if score >= best_score:
            best_score = score
            best_metrics = val_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch == int(epochs) or epoch % 10 == 0:
            print(
                f"epoch={epoch} loss={history[-1]['loss']:.4f}"
                f" val_f1={val_metrics['f1']:.3f}"
                f" val_progress_mae={val_metrics['progress_mae_m']:.3f}",
                flush=True,
            )
    assert best_state is not None and best_metrics is not None
    return best_state, best_metrics, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="*", type=Path, default=None)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--primitive-vocab", nargs="*", default=list(PRIMITIVE_NAMES))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--feature-mode",
        choices=("global", "spatial"),
        default="global",
        help="global uses the frozen encoder's pooled latent. spatial taps the "
             "conv feature map before global pooling (adaptive-pooled to "
             "--spatial-pool x --spatial-pool and flattened) so the head can "
             "see where free space and obstacles are in the frame.",
    )
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-progress-m", type=float, default=0.035)
    parser.add_argument("--block-distance-m", type=float, default=1.0)
    parser.add_argument("--no-traversability-labels", action="store_true")
    parser.add_argument("--include-guard-block-label", action="store_true")
    parser.add_argument("--progress-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--counterfactual-blocked-source",
        choices=("clearance", "progress"),
        default="clearance",
        help="In counterfactual mode, label a TRANSLATING primitive blocked when "
             "its swept counterfactual progress falls below "
             "--counterfactual-progress-floor-m (operationally exact for the "
             "kinematic sim) instead of the clearance-margin label. Yaw/hold "
             "keep the clearance label.",
    )
    parser.add_argument("--counterfactual-progress-floor-m", type=float, default=0.04)
    parser.add_argument(
        "--label-mode",
        choices=("closed_loop_progress", "counterfactual_body_clearance"),
        default="closed_loop_progress",
        help="closed_loop_progress uses logged execution/stall labels. "
             "counterfactual_body_clearance computes offline swept Go2 body "
             "labels from source poses and scene geometry, but still trains "
             "the runtime head from RGB/JEPA latents.",
    )
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/go2_primitive_registry.yaml",
    )
    parser.add_argument("--body-forward-m", type=float, default=0.40)
    parser.add_argument("--body-half-width-m", type=float, default=0.24)
    parser.add_argument("--body-clearance-margin-m", type=float, default=0.03)
    parser.add_argument(
        "--body-clearance-source",
        choices=("configuration", "obstacle"),
        default="configuration",
        help="Clearance field used for counterfactual body labels. configuration "
             "matches the original inflated-grid label; obstacle uses raw "
             "obstacle distance at body probe points, matching the physical "
             "runtime body-envelope clearance check.",
    )
    parser.add_argument(
        "--body-clearance-label-target",
        choices=("swept_min", "after_start_min", "final"),
        default="swept_min",
        help="Which counterfactual body-clearance value becomes the blocked label. "
             "swept_min preserves the old conservative behavior; after_start_min "
             "ignores the starting pose so the head can distinguish backing out "
             "from continuing into a corner.",
    )
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--inflation-m", type=float, default=0.12)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    primitive_vocab = [str(item) for item in args.primitive_vocab]
    primitive_to_idx = {name: idx for idx, name in enumerate(primitive_vocab)}
    if args.label_mode == "counterfactual_body_clearance":
        train_examples = _load_counterfactual_body_clearance_rows(
            args.datasets,
            primitive_vocab=primitive_vocab,
            primitive_registry=args.primitive_registry,
            body_forward_m=float(args.body_forward_m),
            body_half_width_m=float(args.body_half_width_m),
            body_clearance_margin_m=float(args.body_clearance_margin_m),
            body_clearance_label_target=str(args.body_clearance_label_target),
            body_clearance_source=str(args.body_clearance_source),
            cell_size_m=float(args.cell_size_m),
            inflation_m=float(args.inflation_m),
            max_source_rows=int(args.max_source_rows),
            blocked_label_source=str(args.counterfactual_blocked_source),
            progress_floor_m=float(args.counterfactual_progress_floor_m),
        )
        if args.validation_datasets:
            val_examples = _load_counterfactual_body_clearance_rows(
                args.validation_datasets,
                primitive_vocab=primitive_vocab,
                primitive_registry=args.primitive_registry,
                body_forward_m=float(args.body_forward_m),
                body_half_width_m=float(args.body_half_width_m),
                body_clearance_margin_m=float(args.body_clearance_margin_m),
                body_clearance_label_target=str(args.body_clearance_label_target),
                body_clearance_source=str(args.body_clearance_source),
                cell_size_m=float(args.cell_size_m),
                inflation_m=float(args.inflation_m),
                max_source_rows=int(args.max_source_rows),
                blocked_label_source=str(args.counterfactual_blocked_source),
                progress_floor_m=float(args.counterfactual_progress_floor_m),
            )
        else:
            random.shuffle(train_examples)
            train_examples, val_examples = _split_by_scene(train_examples)
    else:
        train_examples = _load_rows(
            args.datasets,
            primitive_vocab=primitive_vocab,
            min_progress_m=float(args.min_progress_m),
            block_distance_m=float(args.block_distance_m),
            include_traversability=not bool(args.no_traversability_labels),
            include_guard=bool(args.include_guard_block_label),
        )
        if args.validation_datasets:
            val_examples = _load_rows(
                args.validation_datasets,
                primitive_vocab=primitive_vocab,
                min_progress_m=float(args.min_progress_m),
                block_distance_m=float(args.block_distance_m),
                include_traversability=not bool(args.no_traversability_labels),
                include_guard=bool(args.include_guard_block_label),
            )
        else:
            random.shuffle(train_examples)
            train_examples, val_examples = _split_by_scene(train_examples)
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
    if str(args.feature_mode) == "spatial":
        # Tap the last conv activation (before AdaptiveAvgPool2d/Flatten/Linear)
        # so spatial obstacle layout survives into the head input.
        pool = max(1, int(args.spatial_pool))
        feature_encoder: torch.nn.Module = torch.nn.Sequential(
            *list(encoder.net[:8]),
            torch.nn.AdaptiveAvgPool2d((pool, pool)),
            torch.nn.Flatten(),
        ).to(device)
        feature_encoder.eval()
    else:
        feature_encoder = encoder
    train_data = _precompute(
        feature_encoder,
        train_examples,
        primitive_to_idx=primitive_to_idx,
        image_size=int(args.image_size),
        device=device,
        batch_size=int(args.batch_size),
    )
    val_data = _precompute(
        feature_encoder,
        val_examples,
        primitive_to_idx=primitive_to_idx,
        image_size=int(args.image_size),
        device=device,
        batch_size=int(args.batch_size),
    )
    if str(args.feature_mode) == "spatial":
        latent_dim = int(train_data[0].shape[-1])
    else:
        latent_dim = int(encoder_checkpoint.get("latent_dim", train_data[0].shape[-1]))
    model = Go2PrimitiveOutcomeHead(
        latent_dim=latent_dim,
        primitive_count=len(primitive_vocab),
        hidden_dim=int(args.hidden_dim),
    )
    best_state, best_metrics, history = _train(
        model,
        train_data,
        val_data,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        threshold=float(args.threshold),
        progress_loss_weight=float(args.progress_loss_weight),
    )
    model.load_state_dict(best_state)
    train_metrics = _evaluate(
        model,
        *train_data,
        device=device,
        threshold=float(args.threshold),
    )
    val_metrics = _evaluate(
        model,
        *val_data,
        device=device,
        threshold=float(args.threshold),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "go2_jepa_primitive_outcome_predictor_v0",
        "model_state_dict": best_state,
        "latent_dim": latent_dim,
        "hidden_dim": int(args.hidden_dim),
        "image_size": int(args.image_size),
        "feature_mode": str(args.feature_mode),
        "spatial_pool": int(args.spatial_pool),
        "primitive_vocab": primitive_vocab,
        "threshold": float(args.threshold),
        "min_progress_m": float(args.min_progress_m),
        "block_distance_m": float(args.block_distance_m),
        "label_mode": str(args.label_mode),
        "body_forward_m": float(args.body_forward_m),
        "body_half_width_m": float(args.body_half_width_m),
        "body_clearance_margin_m": float(args.body_clearance_margin_m),
        "body_clearance_source": str(args.body_clearance_source),
        "body_clearance_label_target": str(args.body_clearance_label_target),
        "cell_size_m": float(args.cell_size_m),
        "inflation_m": float(args.inflation_m),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
    }
    torch.save(checkpoint, args.output)
    report = {
        "schema": "go2_jepa_primitive_outcome_predictor_report_v0",
        "checkpoint": str(args.output),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "train_scenes": sorted({item.scene_id for item in train_examples}),
        "validation_scenes": sorted({item.scene_id for item in val_examples}),
        "primitive_vocab": primitive_vocab,
        "threshold": float(args.threshold),
        "min_progress_m": float(args.min_progress_m),
        "block_distance_m": float(args.block_distance_m),
        "label_mode": str(args.label_mode),
        "body_forward_m": float(args.body_forward_m),
        "body_half_width_m": float(args.body_half_width_m),
        "body_clearance_margin_m": float(args.body_clearance_margin_m),
        "body_clearance_source": str(args.body_clearance_source),
        "body_clearance_label_target": str(args.body_clearance_label_target),
        "cell_size_m": float(args.cell_size_m),
        "inflation_m": float(args.inflation_m),
        "include_traversability_labels": not bool(args.no_traversability_labels),
        "include_guard_block_label": bool(args.include_guard_block_label),
        "train": train_metrics,
        "validation": val_metrics,
        "best_validation": best_metrics,
        "history": history,
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": report["schema"],
        "checkpoint": report["checkpoint"],
        "train_examples": report["train_examples"],
        "validation_examples": report["validation_examples"],
        "train": report["train"],
        "validation": report["validation"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
