#!/usr/bin/env python3
"""Train a Phase 3A latent map head from JEPA spatial tokens."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_marker_memory import marker_position_in_observation  # noqa: E402
from lewm.benchmarks.phase3a_positive_control import read_jsonl, render_observation  # noqa: E402
from lewm.benchmarks.phase3a_training import source_key  # noqa: E402
from lewm.models.phase3a_latent_map import Phase3ALatentMapHead  # noqa: E402
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _goal_scene_from_row,
    _infer_scene_seed,
    _state_from_dict,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402


@dataclass(frozen=True)
class MapFrame:
    observation: list
    palette: dict[str, tuple[float, float, float]]


def _json_safe_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe_arg(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_arg(item) for item in value]
    return value


def _color_distance_sq(
    color: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    return sum((color[index] - target[index]) ** 2 for index in range(3))


def _target_from_observation(
    observation: list,
    *,
    palette: dict[str, tuple[float, float, float]],
) -> torch.Tensor:
    red, green, blue = observation
    view_size = len(red)
    target = torch.zeros(3, view_size, view_size, dtype=torch.float32)
    wall = palette.get("wall")
    outside = palette.get("outside")
    for row in range(view_size):
        for col in range(view_size):
            color = (
                float(red[row][col]),
                float(green[row][col]),
                float(blue[row][col]),
            )
            blocked = False
            if wall is not None and _color_distance_sq(color, wall) <= 1e-8:
                blocked = True
            if outside is not None and _color_distance_sq(color, outside) <= 1e-8:
                blocked = True
            target[0, row, col] = float(blocked)
            target[1, row, col] = float(not blocked)
    marker_color = palette.get("goal")
    marker = marker_position_in_observation(
        observation,
        marker_color=marker_color,
    )
    if marker is not None:
        ahead, lateral = marker
        radius = view_size // 2
        row = radius - ahead
        col = lateral + radius
        if 0 <= row < view_size and 0 <= col < view_size:
            target[2, row, col] = 1.0
    return target


def _palette_from_row(row: dict) -> dict[str, tuple[float, float, float]]:
    return {
        key: tuple(float(channel) for channel in value)
        for key, value in row.get("render_palette", {}).items()
    }


def _collect_frames(rows: list[dict], *, include_history: bool, include_future: bool) -> list[MapFrame]:
    frames: list[MapFrame] = []
    seen: set[str] = set()
    for row in rows:
        palette = _palette_from_row(row)
        observations = [row["start_observation_rgb"]]
        if include_history:
            observations.extend(row.get("history_observations_rgb", []))
        if include_future:
            observations.extend(
                item["observation_rgb"]
                for item in row.get("future_observations", [])
                if item.get("observation_valid", False)
            )
        for observation in observations:
            key = json.dumps(
                {
                    "observation": observation,
                    "palette": row.get("render_palette", {}),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if key in seen:
                continue
            seen.add(key)
            frames.append(MapFrame(observation=observation, palette=palette))
    return frames


def _group_sources(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault(source_key(row), []).append(row)
    return [grouped[key] for key in sorted(grouped)]


def _collect_trace_frames(
    trace_paths: list[Path],
    rows: list[dict],
    *,
    scene_seed: int,
    width: int,
    height: int,
    view_size: int,
    failed_only: bool,
    current_marker_only: bool,
    repeat: int,
) -> list[MapFrame]:
    groups = _group_sources(rows)
    frames: list[MapFrame] = []
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text())
        for episode_index, episode in enumerate(trace.get("episodes", [])):
            if episode_index >= len(groups):
                raise SystemExit(
                    f"{trace_path} has episode index {episode_index}, "
                    f"but only {len(groups)} source groups are available"
                )
            if failed_only and bool(episode.get("claimed", False)):
                continue
            template = groups[episode_index][0]
            scene = _goal_scene_from_row(
                template,
                seed=scene_seed,
                width=width,
                height=height,
            )
            palette = scene.render_palette or _palette_from_row(template)
            for item in episode.get("trajectory", []):
                state = _state_from_dict(item["state"])
                observation = render_observation(
                    scene,
                    state,
                    view_size=view_size,
                    show_goal_marker=True,
                )
                frame = MapFrame(observation=observation, palette=palette)
                if current_marker_only and (
                    _target_from_observation(observation, palette=palette)[2].sum()
                    <= 0.5
                ):
                    continue
                frames.append(frame)
    if repeat <= 1:
        return frames
    return frames * repeat


class LatentMapFrameDataset(Dataset):
    def __init__(self, frames: list[MapFrame]) -> None:
        self.observations = torch.stack(
            [
                torch.tensor(frame.observation, dtype=torch.float32)
                for frame in frames
            ]
        )
        self.targets = torch.stack(
            [
                _target_from_observation(frame.observation, palette=frame.palette)
                for frame in frames
            ]
        )

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[index], self.targets[index]


@torch.no_grad()
def _evaluate(
    base_model: nn.Module,
    head: Phase3ALatentMapHead,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict:
    base_model.eval()
    head.eval()
    total_loss = 0.0
    total_frames = 0
    total_cells = 0
    blocked_correct = 0
    free_correct = 0
    marker_visible = 0
    marker_top1 = 0
    marker_present_pred = 0
    marker_absent = 0
    marker_false_present = 0
    for observation, target in loader:
        observation = observation.to(device)
        target = target.to(device)
        tokens = base_model.encoder(observation)
        logits = head(tokens)
        loss = _loss(logits, target)
        probs = logits.sigmoid()
        blocked_pred = probs[:, 0] >= 0.5
        free_pred = probs[:, 1] >= 0.5
        blocked_target = target[:, 0] >= 0.5
        free_target = target[:, 1] >= 0.5
        blocked_correct += int((blocked_pred == blocked_target).sum().item())
        free_correct += int((free_pred == free_target).sum().item())
        total_cells += int(blocked_target.numel())
        marker_target = target[:, 2].flatten(start_dim=1)
        marker_logits = logits[:, 2].flatten(start_dim=1)
        visible = marker_target.sum(dim=1) > 0.5
        marker_visible += int(visible.sum().item())
        marker_absent += int((~visible).sum().item())
        if bool(visible.any()):
            marker_top1 += int(
                (
                    marker_logits[visible].argmax(dim=1)
                    == marker_target[visible].argmax(dim=1)
                )
                .sum()
                .item()
            )
        marker_presence = marker_logits.max(dim=1).values.sigmoid() >= 0.5
        marker_present_pred += int((marker_presence & visible).sum().item())
        marker_false_present += int((marker_presence & ~visible).sum().item())
        total_loss += float(loss.item()) * observation.shape[0]
        total_frames += int(observation.shape[0])
    return {
        "frames": total_frames,
        "loss": total_loss / max(total_frames, 1),
        "blocked_accuracy": blocked_correct / max(total_cells, 1),
        "free_accuracy": free_correct / max(total_cells, 1),
        "marker_visible_frames": marker_visible,
        "marker_top1_when_visible": marker_top1 / max(marker_visible, 1),
        "marker_presence_recall": marker_present_pred / max(marker_visible, 1),
        "marker_false_presence_rate": marker_false_present / max(marker_absent, 1),
    }


def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    marker_pos_weight = logits.new_tensor(float(logits.shape[-1] * logits.shape[-2]))
    marker_loss = F.binary_cross_entropy_with_logits(
        logits[:, 2],
        target[:, 2],
        pos_weight=marker_pos_weight,
    )
    occupancy_loss = F.binary_cross_entropy_with_logits(logits[:, :2], target[:, :2])
    return occupancy_loss + marker_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument(
        "--extra-train-data",
        type=Path,
        action="append",
        default=[],
        help="additional train JSONL files to include in frame collection",
    )
    parser.add_argument(
        "--extra-validation-data",
        type=Path,
        action="append",
        default=[],
        help="additional validation JSONL files to include in checkpoint selection",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--init-head", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimization-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260644)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--include-future", action="store_true")
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument(
        "--trace-frame-data",
        type=Path,
        action="append",
        default=[],
        help="closed-loop trace JSON to render into additional training frames",
    )
    parser.add_argument(
        "--trace-frame-source-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional JSONL source rows paired one-for-one with "
            "--trace-frame-data; use this when replaying traces from multiple "
            "generated splits"
        ),
    )
    parser.add_argument(
        "--trace-frame-source",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--trace-frame-failed-only", action="store_true")
    parser.add_argument("--trace-frame-current-marker-only", action="store_true")
    parser.add_argument("--trace-frame-repeat", type=int, default=1)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--log-every", type=int, default=128)
    args = parser.parse_args()

    if args.trace_frame_repeat < 1:
        raise SystemExit("--trace-frame-repeat must be positive")
    if args.trace_frame_source_data and (
        len(args.trace_frame_source_data) != len(args.trace_frame_data)
    ):
        raise SystemExit(
            "--trace-frame-source-data must be passed once for each "
            "--trace-frame-data path"
        )

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    train_rows = read_jsonl(args.train_data)
    for extra_path in args.extra_train_data:
        train_rows.extend(read_jsonl(extra_path))
    validation_rows = read_jsonl(args.validation_data)
    for extra_path in args.extra_validation_data:
        validation_rows.extend(read_jsonl(extra_path))
    train_frames = _collect_frames(
        train_rows,
        include_history=args.include_history,
        include_future=args.include_future,
    )
    validation_frames = _collect_frames(
        validation_rows,
        include_history=args.include_history,
        include_future=args.include_future,
    )
    trace_frame_count = 0
    if args.trace_frame_data:
        if args.trace_frame_source_data:
            trace_frames = []
            for trace_path, source_path in zip(
                args.trace_frame_data,
                args.trace_frame_source_data,
                strict=True,
            ):
                trace_seed = _infer_scene_seed(source_path)
                if trace_seed is None:
                    raise SystemExit(
                        f"could not infer scene seed from {source_path}"
                    )
                trace_rows = read_jsonl(source_path)
                trace_view_size = len(trace_rows[0]["start_observation_rgb"][0])
                trace_frames.extend(
                    _collect_trace_frames(
                        [trace_path],
                        trace_rows,
                        scene_seed=int(trace_seed),
                        width=args.width_cells,
                        height=args.height_cells,
                        view_size=trace_view_size,
                        failed_only=bool(args.trace_frame_failed_only),
                        current_marker_only=bool(args.trace_frame_current_marker_only),
                        repeat=int(args.trace_frame_repeat),
                    )
                )
        else:
            trace_rows = (
                train_rows if args.trace_frame_source == "train" else validation_rows
            )
            trace_data_path = (
                args.train_data
                if args.trace_frame_source == "train"
                else args.validation_data
            )
            trace_seed = _infer_scene_seed(trace_data_path)
            if trace_seed is None:
                raise SystemExit(f"could not infer scene seed from {trace_data_path}")
            trace_view_size = len(trace_rows[0]["start_observation_rgb"][0])
            trace_frames = _collect_trace_frames(
                list(args.trace_frame_data),
                trace_rows,
                scene_seed=int(trace_seed),
                width=args.width_cells,
                height=args.height_cells,
                view_size=trace_view_size,
                failed_only=bool(args.trace_frame_failed_only),
                current_marker_only=bool(args.trace_frame_current_marker_only),
                repeat=int(args.trace_frame_repeat),
            )
        trace_frame_count = len(trace_frames)
        train_frames.extend(trace_frames)
    if not train_frames or not validation_frames:
        raise SystemExit("train and validation frames must be non-empty")

    base_model, base_report = load_model(args.base_checkpoint, device=device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    head = Phase3ALatentMapHead(
        view_size=base_model.view_size,
        latent_dim=base_model.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    init_report = None
    if args.init_head is not None:
        try:
            init_checkpoint = torch.load(
                args.init_head,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            init_checkpoint = torch.load(args.init_head, map_location=device)
        init_report = init_checkpoint.get("report", {})
        init_config = init_report.get("model_config", {})
        expected = {
            "view_size": int(base_model.view_size),
            "latent_dim": int(base_model.latent_dim),
            "hidden_dim": int(args.hidden_dim),
            "output_channels": 3,
        }
        actual = {
            "view_size": int(init_config.get("view_size", base_model.view_size)),
            "latent_dim": int(init_config.get("latent_dim", base_model.latent_dim)),
            "hidden_dim": int(init_config.get("hidden_dim", args.hidden_dim)),
            "output_channels": int(init_config.get("output_channels", 3)),
        }
        if actual != expected:
            raise SystemExit(
                "--init-head config does not match requested model config "
                f"({actual} != {expected})"
            )
        head.load_state_dict(init_checkpoint["head_state_dict"])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(
        LatentMapFrameDataset(train_frames),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        LatentMapFrameDataset(validation_frames),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_step = None
    best_validation = None
    best_score = (-1.0, -1.0, -1.0, -1.0, float("inf"), float("inf"))
    for step in range(1, args.optimization_steps + 1):
        try:
            observation, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            observation, target = next(iterator)
        observation = observation.to(device)
        target = target.to(device)
        with torch.no_grad():
            tokens = base_model.encoder(observation)
        logits = head(tokens)
        loss = _loss(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate(base_model, head, validation_loader, device=device)
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            score = (
                float(metrics["marker_presence_recall"]),
                float(metrics["marker_top1_when_visible"]),
                float(metrics["blocked_accuracy"]),
                float(metrics["free_accuracy"]),
                -float(metrics["marker_false_presence_rate"]),
                -float(metrics["loss"]),
            )
            if args.save_best and score > best_score:
                best_score = score
                best_step = step
                best_validation = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }

    final_validation = _evaluate(base_model, head, validation_loader, device=device)
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_validation or final_validation
    report = {
        "schema": "jepa_phase3a_latent_map_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "train_data": str(args.train_data.resolve()),
        "extra_train_data": [
            str(path.resolve()) for path in args.extra_train_data
        ],
        "validation_data": str(args.validation_data.resolve()),
        "extra_validation_data": [
            str(path.resolve()) for path in args.extra_validation_data
        ],
        "train_frames": len(train_frames),
        "validation_frames": len(validation_frames),
        "trace_frame_count": int(trace_frame_count),
        "trace_frame_source_data": [
            str(path.resolve()) for path in args.trace_frame_source_data
        ],
        "init_head": str(args.init_head.resolve()) if args.init_head else None,
        "init_head_completed_steps": (
            init_report.get("completed_steps") if init_report else None
        ),
        "selected_step": int(selected_step),
        "selected_validation": selected_validation,
        "args": {
            key: _json_safe_arg(value)
            for key, value in vars(args).items()
        },
        "completed_steps": args.optimization_steps,
        "final_validation": final_validation,
        "logs": logs,
        "model_config": {
            "view_size": base_model.view_size,
            "latent_dim": base_model.latent_dim,
            "hidden_dim": args.hidden_dim,
            "output_channels": 3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
