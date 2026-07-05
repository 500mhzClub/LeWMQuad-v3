#!/usr/bin/env python3
"""Train a closed-loop Go2 learned-local EXPLORE primitive policy.

The dataset is produced by benchmark_go2_memory_closed_loop.py with
--learned-local-dataset-output. Labels may come from a privileged offline
teacher, but the saved features are runtime-safe: learned controller state,
RGB/JEPA-derived outcome predictions, target query color, claimed-color mask,
and last command.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_ONLINE_MAP_CHANNELS = 8
_BASE_LEARNED_LOCAL_FEATURE_DIM = 1600
_LEARNED_LOCAL_POLICY_PRIMITIVE_COUNT = 8
_PRIMITIVE_OUTCOME_FEATURE_WIDTH = _LEARNED_LOCAL_POLICY_PRIMITIVE_COUNT * 3
_FORWARD_LABELS = frozenset(("forward_fast", "forward_medium", "forward_slow", "arc_left", "arc_right"))
_TRANSLATING_LABELS = frozenset((*_FORWARD_LABELS, "backward"))
_STATE_FEATURES = ("EXPLORE", "SEEK", "SERVO", "CLAIM")


class LearnedLocalPolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        primitive_count: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(float(dropout))
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(primitive_count)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_dropout(features))


class LearnedLocalRecurrentPolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        primitive_count: int,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(input_dim))
        self.input_dropout = nn.Dropout(float(dropout))
        self.output_dropout = nn.Dropout(float(dropout))
        self.embed = nn.Sequential(
            nn.Linear(int(input_dim), int(embed_dim)),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=int(embed_dim),
            hidden_size=int(hidden_dim),
            batch_first=True,
        )
        self.out = nn.Linear(int(hidden_dim), int(primitive_count))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(self.norm(self.input_dropout(features)))
        sequence, _ = self.gru(embedded)
        return self.out(self.output_dropout(sequence))


class LearnedLocalMapCnnPolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        primitive_count: int,
        *,
        map_size: int,
        map_channels: int = _ONLINE_MAP_CHANNELS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.map_size = _odd_online_map_size(int(map_size))
        self.map_channels = int(map_channels)
        self.map_feature_dim = self.map_channels * self.map_size * self.map_size
        self.base_dim = self.input_dim - self.map_feature_dim
        if self.base_dim <= 0:
            raise ValueError(
                "map_cnn input_dim must include a non-map prefix before the online-map suffix"
            )
        self.input_dropout = nn.Dropout(float(dropout))
        self.base_norm = nn.LayerNorm(self.base_dim)
        self.map_norm = nn.LayerNorm(self.map_feature_dim)
        self.base_embed = nn.Sequential(
            nn.Linear(self.base_dim, int(hidden_dim) // 2),
            nn.GELU(),
        )
        self.map_conv = nn.Sequential(
            nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, self.map_channels, self.map_size, self.map_size)
            map_embed_dim = int(self.map_conv(probe).shape[1])
        fused_dim = int(hidden_dim) // 2 + map_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(primitive_count)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        dropped = self.input_dropout(features)
        base = dropped[:, : self.base_dim]
        map_flat = dropped[:, self.base_dim : self.base_dim + self.map_feature_dim]
        map_tensor = self.map_norm(map_flat).reshape(
            -1,
            self.map_channels,
            self.map_size,
            self.map_size,
        )
        return self.head(
            torch.cat(
                [
                    self.base_embed(self.base_norm(base)),
                    self.map_conv(map_tensor),
                ],
                dim=1,
            )
        )


class OfflineOnlineEgomotionMap:
    """Replay the runtime egomotion visit/stall map from dataset metadata."""

    def __init__(self, *, size: int, cell_m: float) -> None:
        size = int(size)
        if size < 3:
            size = 3
        if size % 2 == 0:
            size += 1
        self.size = size
        self.radius = size // 2
        self.cell_m = max(1e-3, float(cell_m))
        self.visited: dict[tuple[int, int], int] = {}
        self.blocked: set[tuple[int, int]] = set()
        self.claimed: set[tuple[int, int]] = set()
        self.attempted_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def observe_pose(self, pose_xy: Any, *, tick: int) -> None:
        cell = self._cell(pose_xy)
        self.visited[cell] = int(tick)
        self.blocked.discard(cell)

    def mark_claim(self, pose_xy: Any) -> None:
        self.claimed.add(self._cell(pose_xy))

    def reset_after_claim(self, pose_xy: Any, *, tick: int) -> None:
        claimed = set(self.claimed)
        self.visited.clear()
        self.blocked.clear()
        self.attempted_edges.clear()
        self.blocked_edges.clear()
        self.claimed = claimed
        self.observe_pose(pose_xy, tick=int(tick))

    def mark_blocked_ahead(self, pose_xy: Any, yaw_rad: float) -> None:
        arr = _pose_xy_array(pose_xy)
        self.blocked.add(
            self._cell(
                (
                    float(arr[0]) + self.cell_m * float(np.cos(float(yaw_rad))),
                    float(arr[1]) + self.cell_m * float(np.sin(float(yaw_rad))),
                )
            )
        )

    def mark_guard_blocked_primitive(
        self,
        pose_xy: Any,
        yaw_rad: float,
        primitive: str,
    ) -> None:
        if str(primitive) not in _TRANSLATING_LABELS:
            return
        start_cell = self._cell(pose_xy)
        target_cell = self._primitive_target_cell(pose_xy, yaw_rad, primitive)
        edge_target = self._cardinal_step_toward(start_cell, target_cell)
        if edge_target != start_cell:
            self.attempted_edges.add((start_cell, edge_target))

    def update_after_action(
        self,
        *,
        pose_xy: Any,
        post_xy: Any,
        yaw_rad: float,
        primitive: str,
        stalled: bool,
        tick: int,
    ) -> None:
        start_cell = self._cell(pose_xy)
        post_cell = self._cell(post_xy)
        target_cell = self._primitive_target_cell(pose_xy, yaw_rad, primitive)
        edge_target = (
            post_cell
            if post_cell != start_cell and not bool(stalled)
            else self._cardinal_step_toward(start_cell, target_cell)
        )
        if edge_target != start_cell and str(primitive) in _TRANSLATING_LABELS:
            self.attempted_edges.add((start_cell, edge_target))
            if bool(stalled):
                self.blocked_edges.add((start_cell, edge_target))
                self.blocked_edges.add((edge_target, start_cell))
                self.blocked.add(edge_target)
            else:
                self.attempted_edges.add((edge_target, start_cell))
                self.blocked_edges.discard((start_cell, edge_target))
                self.blocked_edges.discard((edge_target, start_cell))
                self.blocked.discard(edge_target)
                self.blocked.discard(post_cell)
                self.visited[edge_target] = int(tick) + 1
        self.observe_pose(post_xy, tick=int(tick) + 1)
        if bool(stalled) and str(primitive) in _FORWARD_LABELS:
            self.mark_blocked_ahead(pose_xy, yaw_rad)

    def feature(self, pose_xy: Any, yaw_rad: float, *, tick: int) -> np.ndarray:
        channels = np.zeros((_ONLINE_MAP_CHANNELS, self.size, self.size), dtype=np.float32)
        self._scatter(channels[0], self.visited.keys(), pose_xy, yaw_rad, value=1.0)
        self._scatter(channels[1], self.blocked, pose_xy, yaw_rad, value=1.0)
        self._scatter(channels[3], self.claimed, pose_xy, yaw_rad, value=1.0)
        current_cell = self._cell(pose_xy)
        frontier_cells = self._frontier_cells()
        frontier_path = self._path_to_frontier(current_cell)
        frontier_targets = self._frontier_targets(frontier_cells)
        attempted_targets = [dst for _, dst in self.attempted_edges]
        self._scatter(channels[4], frontier_cells, pose_xy, yaw_rad, value=1.0)
        self._scatter(channels[5], frontier_path, pose_xy, yaw_rad, value=1.0)
        self._scatter(channels[6], frontier_targets, pose_xy, yaw_rad, value=1.0)
        self._scatter(channels[7], attempted_targets, pose_xy, yaw_rad, value=1.0)
        for cell, seen_tick in self.visited.items():
            row_col = self._row_col(cell, pose_xy, yaw_rad)
            if row_col is None:
                continue
            age = max(0, int(tick) - int(seen_tick))
            channels[2, row_col[0], row_col[1]] = max(
                channels[2, row_col[0], row_col[1]],
                max(0.0, 1.0 - float(age) / 160.0),
            )
        return channels.reshape(-1)

    def _cell(self, pose_xy: Any) -> tuple[int, int]:
        arr = _pose_xy_array(pose_xy)
        return (int(round(float(arr[0]) / self.cell_m)), int(round(float(arr[1]) / self.cell_m)))

    def _primitive_target_cell(self, pose_xy: Any, yaw_rad: float, primitive: str) -> tuple[int, int]:
        arr = _pose_xy_array(pose_xy)
        x = float(arr[0])
        y = float(arr[1])
        name = str(primitive)
        yaw_delta = 0.0
        distance_m = 0.0
        if name == "forward_fast":
            distance_m = 1.6 * self.cell_m
        elif name == "forward_medium":
            distance_m = 1.25 * self.cell_m
        elif name == "forward_slow":
            distance_m = 1.0 * self.cell_m
        elif name == "arc_left":
            distance_m = 1.1 * self.cell_m
            yaw_delta = 0.45
        elif name == "arc_right":
            distance_m = 1.1 * self.cell_m
            yaw_delta = -0.45
        elif name == "backward":
            distance_m = -1.0 * self.cell_m
        yaw = float(yaw_rad) + yaw_delta
        return self._cell((x + distance_m * np.cos(yaw), y + distance_m * np.sin(yaw)))

    def _cardinal_step_toward(
        self,
        source: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int]:
        sx, sy = int(source[0]), int(source[1])
        dx = int(target[0]) - sx
        dy = int(target[1]) - sy
        if dx == 0 and dy == 0:
            return (sx, sy)
        if abs(dx) >= abs(dy) and dx != 0:
            return (sx + (1 if dx > 0 else -1), sy)
        if dy != 0:
            return (sx, sy + (1 if dy > 0 else -1))
        return (sx, sy)

    def _center(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (float(cell[0]) * self.cell_m, float(cell[1]) * self.cell_m)

    def _row_col(self, cell: tuple[int, int], pose_xy: Any, yaw_rad: float) -> tuple[int, int] | None:
        pose = _pose_xy_array(pose_xy)
        cx, cy = self._center(cell)
        dxw = cx - float(pose[0])
        dyw = cy - float(pose[1])
        yaw = float(yaw_rad)
        ahead = float(np.cos(yaw)) * dxw + float(np.sin(yaw)) * dyw
        lateral = -float(np.sin(yaw)) * dxw + float(np.cos(yaw)) * dyw
        row = self.radius - int(round(ahead / self.cell_m))
        col = self.radius + int(round(lateral / self.cell_m))
        if 0 <= row < self.size and 0 <= col < self.size:
            return row, col
        return None

    def _scatter(self, channel: np.ndarray, cells: Any, pose_xy: Any, yaw_rad: float, *, value: float) -> None:
        for cell in cells:
            row_col = self._row_col(tuple(cell), pose_xy, yaw_rad)
            if row_col is not None:
                channel[row_col[0], row_col[1]] = max(float(channel[row_col[0], row_col[1]]), value)

    def _frontier_cells(self) -> list[tuple[int, int]]:
        return [cell for cell in self.visited if self._is_frontier(cell)]

    def _frontier_targets(self, frontier_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        targets: list[tuple[int, int]] = []
        for cell in frontier_cells:
            cx, cy = int(cell[0]), int(cell[1])
            for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (
                    neighbor not in self.visited
                    and neighbor not in self.blocked
                    and neighbor not in self.claimed
                    and (cell, neighbor) not in self.attempted_edges
                    and (cell, neighbor) not in self.blocked_edges
                ):
                    targets.append(neighbor)
        return targets

    def _is_frontier(self, cell: tuple[int, int]) -> bool:
        if cell not in self.visited:
            return False
        cx, cy = int(cell[0]), int(cell[1])
        for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (
                neighbor not in self.visited
                and neighbor not in self.blocked
                and neighbor not in self.claimed
                and (cell, neighbor) not in self.attempted_edges
                and (cell, neighbor) not in self.blocked_edges
            ):
                return True
        return False

    def _path_to_frontier(self, start: tuple[int, int]) -> list[tuple[int, int]]:
        if start not in self.visited:
            return []
        queue: list[tuple[int, int]] = [start]
        seen = {start}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        for cell in queue:
            if self._is_frontier(cell):
                out: list[tuple[int, int]] = []
                cursor: tuple[int, int] | None = cell
                while cursor is not None:
                    out.append(cursor)
                    cursor = parent.get(cursor)
                out.reverse()
                return out
            cx, cy = int(cell[0]), int(cell[1])
            for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (
                    neighbor in seen
                    or neighbor not in self.visited
                    or (neighbor in self.blocked and neighbor not in self.visited)
                    or (cell, neighbor) in self.blocked_edges
                ):
                    continue
                seen.add(neighbor)
                parent[neighbor] = cell
                queue.append(neighbor)
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", type=Path)
    parser.add_argument(
        "--dataset-list",
        action="append",
        type=Path,
        default=[],
        help="Optional newline-delimited file of training dataset paths.",
    )
    parser.add_argument(
        "--dataset-glob",
        action="append",
        default=[],
        help="Optional training dataset glob, optionally prefixed as REPEAT:PATTERN.",
    )
    parser.add_argument("--validation-datasets", nargs="+", type=Path, default=None)
    parser.add_argument(
        "--validation-dataset-list",
        action="append",
        type=Path,
        default=[],
        help="Optional newline-delimited file of validation dataset paths.",
    )
    parser.add_argument(
        "--validation-dataset-glob",
        action="append",
        default=[],
        help="Optional validation dataset glob, optionally prefixed as REPEAT:PATTERN.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional compatible learned-local checkpoint to initialize from before training.",
    )
    parser.add_argument("--model-type", choices=("mlp", "gru", "map_cnn"), default="mlp")
    parser.add_argument(
        "--include-states",
        default="",
        help="Optional comma-separated controller states to keep from dataset meta_json.",
    )
    parser.add_argument(
        "--meta-min",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep rows whose numeric metadata KEY is at least VALUE.",
    )
    parser.add_argument(
        "--meta-max",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep rows whose numeric metadata KEY is at most VALUE.",
    )
    parser.add_argument(
        "--meta-eq",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep rows whose metadata KEY stringifies exactly to VALUE.",
    )
    parser.add_argument(
        "--meta-not-eq",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Drop rows whose metadata KEY stringifies exactly to VALUE.",
    )
    parser.add_argument(
        "--append-clock-features",
        action="store_true",
        help="Append runtime-safe normalized tick/target-progress scalars from meta_json.",
    )
    parser.add_argument("--clock-max-ticks", type=float, default=560.0)
    parser.add_argument(
        "--append-state-features",
        action="store_true",
        help=(
            "Append a one-hot controller-state feature from meta_json "
            "(EXPLORE/SEEK/SERVO/CLAIM). This is runtime-safe and scene-independent."
        ),
    )
    parser.add_argument(
        "--append-visual-readout-features",
        action="store_true",
        help=(
            "Append runtime-safe active-target RGB/controller readout scalars "
            "from meta_json: area, bearing, memory confidence, read score, "
            "in-cone flag, and claimed count."
        ),
    )
    parser.add_argument(
        "--append-pose-topology-features",
        action="store_true",
        help="Append same-scene odometry-like pose/topology scalars from meta_json.",
    )
    parser.add_argument("--pose-scale-m", type=float, default=4.0)
    parser.add_argument(
        "--append-online-map-features",
        action="store_true",
        help=(
            "Append a runtime-safe egomotion visit/stall/claim map reconstructed "
            "from meta_json. This does not append absolute pose scalars."
        ),
    )
    parser.add_argument("--online-map-size", type=int, default=11)
    parser.add_argument("--online-map-cell-m", type=float, default=0.45)
    parser.add_argument("--online-map-stall-displacement-m", type=float, default=0.015)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Mini-batch size for training. <=0 preserves full-batch training where supported.",
    )
    parser.add_argument(
        "--sequence-chunk-len",
        type=int,
        default=0,
        help=(
            "For GRU training, split each loaded sequence into chunks of this "
            "many rows before padding. <=0 preserves one sequence per dataset."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate and checkpoint every N epochs. Defaults to every epoch.",
    )
    parser.add_argument(
        "--save-best-every-eval",
        action="store_true",
        help=(
            "Write --output immediately when validation macro-F1 improves. "
            "The final checkpoint is still rewritten after training."
        ),
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help=(
            "Exponent for inverse-frequency class weighting. 1.0 preserves the "
            "original balanced loss; 0.0 disables class weighting."
        ),
    )
    parser.add_argument(
        "--primitive-loss-multipliers",
        default="",
        help=(
            "Optional comma-separated primitive=multiplier pairs applied on top "
            "of class weights, e.g. forward_medium=1.4,arc_left=1.2."
        ),
    )
    parser.add_argument(
        "--forbid-output-primitives",
        default="",
        help=(
            "Optional comma-separated primitive names to mask at runtime. This is "
            "saved in the checkpoint; training labels must not include these names."
        ),
    )
    parser.add_argument(
        "--input-mask-mode",
        choices=(
            "none",
            "visual_readout",
            "visual_readout_state",
            "visual_readout_state_clock",
            "visual_readout_state_clock_outcome",
            "visual_readout_state_clock_outcome_online_map",
        ),
        default="none",
        help=(
            "Optionally zero all features outside a runtime-safe subset before "
            "training and save the same mask in the checkpoint for inference."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    args.datasets = _expand_dataset_paths(
        args.datasets,
        args.dataset_list,
        args.dataset_glob,
    )
    args.validation_datasets = _expand_dataset_paths(
        args.validation_datasets or [],
        args.validation_dataset_list,
        args.validation_dataset_glob,
    )
    if not args.datasets:
        parser.error("at least one training dataset is required")
    if not args.validation_datasets:
        args.validation_datasets = None

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _resolve_device(str(args.device))
    _initialize_device(device)

    if args.model_type == "gru":
        return _main_recurrent(args)

    include_states = _parse_state_filter(str(args.include_states))
    meta_min = _numeric_filters(args.meta_min, flag="--meta-min")
    meta_max = _numeric_filters(args.meta_max, flag="--meta-max")
    meta_eq = _string_filters(args.meta_eq, flag="--meta-eq")
    meta_not_eq = _string_filters(args.meta_not_eq, flag="--meta-not-eq")
    x, y, primitive_vocab, dataset_reports = _load_many(
        args.datasets,
        include_states=include_states,
        meta_min=meta_min,
        meta_max=meta_max,
        meta_eq=meta_eq,
        meta_not_eq=meta_not_eq,
        append_clock_features=bool(args.append_clock_features),
        clock_max_ticks=float(args.clock_max_ticks),
        append_state_features=bool(args.append_state_features),
        append_visual_readout_features=bool(args.append_visual_readout_features),
        append_pose_topology_features=bool(args.append_pose_topology_features),
        pose_scale_m=float(args.pose_scale_m),
        append_online_map_features=bool(args.append_online_map_features),
        online_map_size=int(args.online_map_size),
        online_map_cell_m=float(args.online_map_cell_m),
        online_map_stall_displacement_m=float(args.online_map_stall_displacement_m),
    )
    if x.shape[0] == 0:
        raise SystemExit("no training examples")
    if args.validation_datasets:
        train_x, train_y = x, y
        val_x, val_y, val_vocab, validation_reports = _load_many(
            args.validation_datasets,
            include_states=include_states,
            meta_min=meta_min,
            meta_max=meta_max,
            meta_eq=meta_eq,
            meta_not_eq=meta_not_eq,
            append_clock_features=bool(args.append_clock_features),
            clock_max_ticks=float(args.clock_max_ticks),
            append_state_features=bool(args.append_state_features),
            append_visual_readout_features=bool(args.append_visual_readout_features),
            append_pose_topology_features=bool(args.append_pose_topology_features),
            pose_scale_m=float(args.pose_scale_m),
            append_online_map_features=bool(args.append_online_map_features),
            online_map_size=int(args.online_map_size),
            online_map_cell_m=float(args.online_map_cell_m),
            online_map_stall_displacement_m=float(args.online_map_stall_displacement_m),
        )
        if list(val_vocab) != list(primitive_vocab):
            raise SystemExit("validation primitive vocab does not match training vocab")
    else:
        train_x, train_y, val_x, val_y = _split_train_validation(
            x,
            y,
            validation_fraction=float(args.validation_fraction),
            seed=int(args.seed),
        )
        validation_reports = []
    if train_x.shape[0] == 0:
        raise SystemExit("train/validation split produced an empty training side")
    if val_x.shape[0] == 0:
        val_x, val_y = train_x, train_y

    feature_variant = _feature_variant(
        append_clock=bool(args.append_clock_features),
        append_state=bool(args.append_state_features),
        append_visual_readout=bool(args.append_visual_readout_features),
        append_pose_topology=bool(args.append_pose_topology_features),
        append_online_map=bool(args.append_online_map_features),
    )
    input_mask_np = _build_input_mask(
        mode=str(args.input_mask_mode),
        input_dim=int(train_x.shape[1]),
        feature_variant=feature_variant,
    )
    model_type = str(args.model_type)
    online_map_size = _odd_online_map_size(int(args.online_map_size))
    online_map_channels = _ONLINE_MAP_CHANNELS
    online_map_feature_dim = int(online_map_channels * online_map_size * online_map_size)
    base_input_dim = (
        int(train_x.shape[1]) - online_map_feature_dim
        if model_type == "map_cnn"
        else None
    )
    if model_type == "map_cnn" and not bool(args.append_online_map_features):
        raise SystemExit("--model-type map_cnn requires --append-online-map-features")
    if model_type == "map_cnn":
        model = LearnedLocalMapCnnPolicyHead(
            input_dim=int(train_x.shape[1]),
            hidden_dim=int(args.hidden_dim),
            primitive_count=len(primitive_vocab),
            map_size=online_map_size,
            map_channels=online_map_channels,
            dropout=float(args.dropout),
        ).to(device)
    else:
        model = LearnedLocalPolicyHead(
            input_dim=int(train_x.shape[1]),
            hidden_dim=int(args.hidden_dim),
            primitive_count=len(primitive_vocab),
            dropout=float(args.dropout),
        ).to(device)
    _load_initial_checkpoint_if_requested(
        model,
        args=args,
        model_type=model_type,
        input_dim=int(train_x.shape[1]),
        primitive_vocab=primitive_vocab,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    class_weights = _class_weights(
        train_y,
        primitive_count=len(primitive_vocab),
        power=float(args.class_weight_power),
    )
    primitive_loss_multipliers = _primitive_loss_multipliers(
        str(args.primitive_loss_multipliers),
        primitive_vocab=primitive_vocab,
    )
    forbid_output_primitives = _forbid_output_primitives(
        str(args.forbid_output_primitives),
        primitive_vocab=primitive_vocab,
        labels=train_y,
    )
    class_weights = _apply_primitive_loss_multipliers(
        class_weights,
        primitive_loss_multipliers,
        primitive_vocab=primitive_vocab,
    ).to(device)
    train_xt = torch.from_numpy(train_x).float().to(device)
    train_yt = torch.from_numpy(train_y).long().to(device)
    val_xt = torch.from_numpy(val_x).float().to(device)
    val_yt = torch.from_numpy(val_y).long().to(device)
    input_mask_t: torch.Tensor | None = None
    if input_mask_np is not None:
        input_mask_t = torch.from_numpy(input_mask_np).float().to(device)
        train_xt = train_xt * input_mask_t
        val_xt = val_xt * input_mask_t

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    history = []
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = int(train_xt.shape[0])
    eval_every = max(1, int(args.eval_every))
    last_train_metrics: dict[str, Any] | None = None
    last_val_metrics: dict[str, Any] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = torch.randperm(int(train_xt.shape[0]), device=device)
        loss_total = 0.0
        loss_count = 0
        for start in range(0, int(order.shape[0]), batch_size):
            batch_idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_xt[batch_idx])
            loss = F.cross_entropy(
                logits,
                train_yt[batch_idx],
                weight=class_weights,
                label_smoothing=float(args.label_smoothing),
            )
            loss.backward()
            optimizer.step()
            batch_count = int(batch_idx.shape[0])
            loss_total += float(loss.detach().cpu()) * float(batch_count)
            loss_count += batch_count
        epoch_loss = loss_total / float(max(1, loss_count))

        should_eval = epoch == 1 or epoch == int(args.epochs) or epoch % eval_every == 0
        if should_eval:
            train_metrics = _evaluate(
                model,
                train_xt,
                train_yt,
                primitive_vocab=primitive_vocab,
                batch_size=batch_size,
            )
            val_metrics = _evaluate(
                model,
                val_xt,
                val_yt,
                primitive_vocab=primitive_vocab,
                batch_size=batch_size,
            )
            last_train_metrics = train_metrics
            last_val_metrics = val_metrics
            score = float(val_metrics["macro_f1"])
            if score >= best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if bool(args.save_best_every_eval):
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "schema": "lewm_go2_closed_loop_learned_local_policy_v0",
                            "model_type": model_type,
                            "model_state_dict": best_state,
                            "input_dim": int(train_x.shape[1]),
                            "hidden_dim": int(args.hidden_dim),
                            "embed_dim": None,
                            "online_map_size": int(online_map_size),
                            "online_map_channels": int(online_map_channels),
                            "online_map_feature_dim": int(online_map_feature_dim),
                            "base_input_dim": (
                                int(base_input_dim) if base_input_dim is not None else None
                            ),
                            "dropout": float(args.dropout),
                            "label_smoothing": float(args.label_smoothing),
                            "batch_size": int(args.batch_size),
                            "eval_every": int(args.eval_every),
                            "feature_variant": feature_variant,
                            "input_mask_mode": str(args.input_mask_mode),
                            "input_mask": (
                                None
                                if input_mask_np is None
                                else torch.from_numpy(input_mask_np).float()
                            ),
                            "primitive_vocab": list(primitive_vocab),
                            "forbid_output_primitives": list(forbid_output_primitives),
                            "args": vars(args),
                        },
                        args.output,
                    )
        else:
            train_metrics = last_train_metrics or {"accuracy": 0.0}
            val_metrics = last_val_metrics or {"accuracy": 0.0, "macro_f1": 0.0}
            score = float(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": int(epoch),
                "loss": float(epoch_loss),
                "evaluated": bool(should_eval),
                "train_accuracy": float(train_metrics["accuracy"]),
                "validation_accuracy": float(val_metrics["accuracy"]),
                "validation_macro_f1": float(val_metrics["macro_f1"]),
            }
        )
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={epoch_loss:.4f}"
                f" val_acc={val_metrics['accuracy']:.3f}"
                f" val_macro_f1={val_metrics['macro_f1']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = _evaluate(
        model,
        train_xt,
        train_yt,
        primitive_vocab=primitive_vocab,
        batch_size=batch_size,
    )
    final_validation = _evaluate(
        model,
        val_xt,
        val_yt,
        primitive_vocab=primitive_vocab,
        batch_size=batch_size,
    )
    checkpoint = {
        "schema": "lewm_go2_closed_loop_learned_local_policy_v0",
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "embed_dim": None,
        "online_map_size": int(online_map_size),
        "online_map_channels": int(online_map_channels),
        "online_map_feature_dim": int(online_map_feature_dim),
        "base_input_dim": int(base_input_dim) if base_input_dim is not None else None,
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "batch_size": int(args.batch_size),
        "eval_every": int(args.eval_every),
        "feature_variant": feature_variant,
        "input_mask_mode": str(args.input_mask_mode),
        "input_mask": None if input_mask_np is None else torch.from_numpy(input_mask_np).float(),
        "primitive_vocab": list(primitive_vocab),
        "forbid_output_primitives": list(forbid_output_primitives),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema": "lewm_go2_closed_loop_learned_local_policy_report_v0",
        "output": str(args.output),
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets or []],
        "dataset_reports": dataset_reports,
        "validation_dataset_reports": validation_reports,
        "device": str(device),
        "model_type": model_type,
        "include_states": sorted(include_states),
        "meta_min": dict(sorted(meta_min.items())),
        "meta_max": dict(sorted(meta_max.items())),
        "meta_eq": dict(sorted(meta_eq.items())),
        "meta_not_eq": dict(sorted(meta_not_eq.items())),
        "append_clock_features": bool(args.append_clock_features),
        "append_state_features": bool(args.append_state_features),
        "append_visual_readout_features": bool(args.append_visual_readout_features),
        "append_pose_topology_features": bool(args.append_pose_topology_features),
        "append_online_map_features": bool(args.append_online_map_features),
        "clock_max_ticks": float(args.clock_max_ticks),
        "pose_scale_m": float(args.pose_scale_m),
        "online_map_size": int(args.online_map_size),
        "online_map_channels": int(online_map_channels),
        "online_map_cell_m": float(args.online_map_cell_m),
        "online_map_stall_displacement_m": float(args.online_map_stall_displacement_m),
        "class_weight_power": float(args.class_weight_power),
        "primitive_loss_multipliers": primitive_loss_multipliers,
        "forbid_output_primitives": list(forbid_output_primitives),
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "batch_size": int(args.batch_size),
        "eval_every": int(args.eval_every),
        "feature_variant": feature_variant,
        "input_mask_mode": str(args.input_mask_mode),
        "input_mask_nonzero": (
            None if input_mask_np is None else int(np.count_nonzero(input_mask_np))
        ),
        "input_dim": int(train_x.shape[1]),
        "base_input_dim": int(base_input_dim) if base_input_dim is not None else None,
        "primitive_vocab": list(primitive_vocab),
        "train_count": int(train_x.shape[0]),
        "validation_count": int(val_x.shape[0]),
        "train_label_counts": _label_counts(train_y, primitive_vocab),
        "validation_label_counts": _label_counts(val_y, primitive_vocab),
        "final_train": final_train,
        "final_validation": final_validation,
        "history": history,
        "claim_boundary": (
            "Teacher-trained learned-local policy. Runtime benchmark inference "
            "must run with --explore-standoff-route disabled for route-free evidence."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"learned_local_policy: output={args.output} "
        f"model={model_type} "
        f"val_acc={final_validation['accuracy']:.3f} "
        f"val_macro_f1={final_validation['macro_f1']:.3f}",
        flush=True,
    )
    return 0


def _main_recurrent(args: argparse.Namespace) -> int:
    include_states = _parse_state_filter(str(args.include_states))
    meta_min = _numeric_filters(args.meta_min, flag="--meta-min")
    meta_max = _numeric_filters(args.meta_max, flag="--meta-max")
    meta_eq = _string_filters(args.meta_eq, flag="--meta-eq")
    meta_not_eq = _string_filters(args.meta_not_eq, flag="--meta-not-eq")
    sequences, primitive_vocab, dataset_reports = _load_many_sequences(
        args.datasets,
        include_states=include_states,
        meta_min=meta_min,
        meta_max=meta_max,
        meta_eq=meta_eq,
        meta_not_eq=meta_not_eq,
        append_clock_features=bool(args.append_clock_features),
        clock_max_ticks=float(args.clock_max_ticks),
        append_state_features=bool(args.append_state_features),
        append_visual_readout_features=bool(args.append_visual_readout_features),
        append_pose_topology_features=bool(args.append_pose_topology_features),
        pose_scale_m=float(args.pose_scale_m),
        append_online_map_features=bool(args.append_online_map_features),
        online_map_size=int(args.online_map_size),
        online_map_cell_m=float(args.online_map_cell_m),
        online_map_stall_displacement_m=float(args.online_map_stall_displacement_m),
    )
    sequences = _chunk_sequences(sequences, chunk_len=int(args.sequence_chunk_len))
    if not sequences:
        raise SystemExit("no training sequences")
    if args.validation_datasets:
        train_sequences = sequences
        validation_sequences, val_vocab, validation_reports = _load_many_sequences(
            args.validation_datasets,
            include_states=include_states,
            meta_min=meta_min,
            meta_max=meta_max,
            meta_eq=meta_eq,
            meta_not_eq=meta_not_eq,
            append_clock_features=bool(args.append_clock_features),
            clock_max_ticks=float(args.clock_max_ticks),
            append_state_features=bool(args.append_state_features),
            append_visual_readout_features=bool(args.append_visual_readout_features),
            append_pose_topology_features=bool(args.append_pose_topology_features),
            pose_scale_m=float(args.pose_scale_m),
            append_online_map_features=bool(args.append_online_map_features),
            online_map_size=int(args.online_map_size),
            online_map_cell_m=float(args.online_map_cell_m),
            online_map_stall_displacement_m=float(args.online_map_stall_displacement_m),
        )
        validation_sequences = _chunk_sequences(
            validation_sequences,
            chunk_len=int(args.sequence_chunk_len),
        )
        if list(val_vocab) != list(primitive_vocab):
            raise SystemExit("validation primitive vocab does not match training vocab")
    else:
        train_sequences, validation_sequences = _split_sequences(
            sequences,
            validation_fraction=float(args.validation_fraction),
            seed=int(args.seed),
        )
        validation_reports = []
    if not train_sequences:
        raise SystemExit("train/validation split produced an empty training side")
    if not validation_sequences:
        validation_sequences = train_sequences

    train_labels_np = _concat_sequence_labels(train_sequences)
    val_labels_np = _concat_sequence_labels(validation_sequences)
    feature_dim = int(train_sequences[0]["features"].shape[1])
    device = _resolve_device(str(args.device))
    batch_size = int(args.batch_size)
    full_batch = batch_size <= 0
    if full_batch:
        train_xt, train_yt, train_mask = _pad_sequences(train_sequences, device=device)
        val_xt, val_yt, val_mask = _pad_sequences(validation_sequences, device=device)
    else:
        train_xt = train_yt = train_mask = None
        val_xt = val_yt = val_mask = None
    model = LearnedLocalRecurrentPolicyHead(
        input_dim=feature_dim,
        hidden_dim=int(args.hidden_dim),
        primitive_count=len(primitive_vocab),
        embed_dim=int(args.embed_dim),
        dropout=float(args.dropout),
    ).to(device)
    _load_initial_checkpoint_if_requested(
        model,
        args=args,
        model_type="gru",
        input_dim=feature_dim,
        primitive_vocab=primitive_vocab,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    class_weights = _class_weights(
        train_labels_np,
        primitive_count=len(primitive_vocab),
        power=float(args.class_weight_power),
    )
    primitive_loss_multipliers = _primitive_loss_multipliers(
        str(args.primitive_loss_multipliers),
        primitive_vocab=primitive_vocab,
    )
    forbid_output_primitives = _forbid_output_primitives(
        str(args.forbid_output_primitives),
        primitive_vocab=primitive_vocab,
        labels=train_labels_np,
    )
    class_weights = _apply_primitive_loss_multipliers(
        class_weights,
        primitive_loss_multipliers,
        primitive_vocab=primitive_vocab,
    ).to(device)
    feature_variant = _feature_variant(
        append_clock=bool(args.append_clock_features),
        append_state=bool(args.append_state_features),
        append_visual_readout=bool(args.append_visual_readout_features),
        append_pose_topology=bool(args.append_pose_topology_features),
        append_online_map=bool(args.append_online_map_features),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    history = []
    eval_every = max(1, int(args.eval_every))
    last_train_metrics: dict[str, Any] | None = None
    last_val_metrics: dict[str, Any] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        if full_batch:
            assert train_xt is not None and train_yt is not None and train_mask is not None
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_xt)
            loss = F.cross_entropy(
                logits[train_mask],
                train_yt[train_mask],
                weight=class_weights,
                label_smoothing=float(args.label_smoothing),
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        else:
            for batch in _sequence_minibatches(
                train_sequences,
                batch_size=batch_size,
                shuffle=True,
                seed=int(args.seed) + int(epoch),
            ):
                batch_xt, batch_yt, batch_mask = _pad_sequences(batch, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_xt)
                loss = F.cross_entropy(
                    logits[batch_mask],
                    batch_yt[batch_mask],
                    weight=class_weights,
                    label_smoothing=float(args.label_smoothing),
                )
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        should_eval = epoch == 1 or epoch == int(args.epochs) or epoch % eval_every == 0
        if should_eval:
            if full_batch:
                assert train_xt is not None and train_yt is not None and train_mask is not None
                assert val_xt is not None and val_yt is not None and val_mask is not None
                train_metrics = _evaluate_sequence(
                    model,
                    train_xt,
                    train_yt,
                    train_mask,
                    primitive_vocab=primitive_vocab,
                )
                val_metrics = _evaluate_sequence(
                    model,
                    val_xt,
                    val_yt,
                    val_mask,
                    primitive_vocab=primitive_vocab,
                )
            else:
                eval_batch_size = max(1, batch_size)
                train_metrics = _evaluate_sequence_batches(
                    model,
                    train_sequences,
                    device=device,
                    batch_size=eval_batch_size,
                    primitive_vocab=primitive_vocab,
                )
                val_metrics = _evaluate_sequence_batches(
                    model,
                    validation_sequences,
                    device=device,
                    batch_size=eval_batch_size,
                    primitive_vocab=primitive_vocab,
                )
            last_train_metrics = train_metrics
            last_val_metrics = val_metrics
            score = float(val_metrics["macro_f1"])
            if score >= best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if bool(args.save_best_every_eval):
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "schema": "lewm_go2_closed_loop_learned_local_policy_v0",
                            "model_type": "gru",
                            "model_state_dict": best_state,
                            "input_dim": feature_dim,
                            "hidden_dim": int(args.hidden_dim),
                            "embed_dim": int(args.embed_dim),
                            "dropout": float(args.dropout),
                            "label_smoothing": float(args.label_smoothing),
                            "feature_variant": feature_variant,
                            "primitive_vocab": list(primitive_vocab),
                            "forbid_output_primitives": list(forbid_output_primitives),
                            "args": vars(args),
                        },
                        args.output,
                    )
        else:
            train_metrics = last_train_metrics or {"accuracy": 0.0}
            val_metrics = last_val_metrics or {"accuracy": 0.0, "macro_f1": 0.0}
        history.append(
            {
                "epoch": int(epoch),
                "loss": epoch_loss,
                "evaluated": bool(should_eval),
                "train_accuracy": float(train_metrics["accuracy"]),
                "validation_accuracy": float(val_metrics["accuracy"]),
                "validation_macro_f1": float(score),
            }
        )
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={epoch_loss:.4f}"
                f" val_acc={val_metrics['accuracy']:.3f}"
                f" val_macro_f1={val_metrics['macro_f1']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    if full_batch:
        assert train_xt is not None and train_yt is not None and train_mask is not None
        assert val_xt is not None and val_yt is not None and val_mask is not None
        final_train = _evaluate_sequence(
            model,
            train_xt,
            train_yt,
            train_mask,
            primitive_vocab=primitive_vocab,
        )
        final_validation = _evaluate_sequence(
            model,
            val_xt,
            val_yt,
            val_mask,
            primitive_vocab=primitive_vocab,
        )
    else:
        final_train = _evaluate_sequence_batches(
            model,
            train_sequences,
            device=device,
            batch_size=max(1, batch_size),
            primitive_vocab=primitive_vocab,
        )
        final_validation = _evaluate_sequence_batches(
            model,
            validation_sequences,
            device=device,
            batch_size=max(1, batch_size),
            primitive_vocab=primitive_vocab,
        )
    checkpoint = {
        "schema": "lewm_go2_closed_loop_learned_local_policy_v0",
        "model_type": "gru",
        "model_state_dict": model.state_dict(),
        "input_dim": feature_dim,
        "hidden_dim": int(args.hidden_dim),
        "embed_dim": int(args.embed_dim),
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "feature_variant": feature_variant,
        "primitive_vocab": list(primitive_vocab),
        "forbid_output_primitives": list(forbid_output_primitives),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema": "lewm_go2_closed_loop_learned_local_policy_report_v0",
        "output": str(args.output),
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets or []],
        "dataset_reports": dataset_reports,
        "validation_dataset_reports": validation_reports,
        "device": str(device),
        "model_type": "gru",
        "include_states": sorted(include_states),
        "meta_min": dict(sorted(meta_min.items())),
        "meta_max": dict(sorted(meta_max.items())),
        "meta_eq": dict(sorted(meta_eq.items())),
        "meta_not_eq": dict(sorted(meta_not_eq.items())),
        "append_clock_features": bool(args.append_clock_features),
        "append_state_features": bool(args.append_state_features),
        "append_visual_readout_features": bool(args.append_visual_readout_features),
        "append_pose_topology_features": bool(args.append_pose_topology_features),
        "append_online_map_features": bool(args.append_online_map_features),
        "clock_max_ticks": float(args.clock_max_ticks),
        "pose_scale_m": float(args.pose_scale_m),
        "online_map_size": int(args.online_map_size),
        "online_map_cell_m": float(args.online_map_cell_m),
        "online_map_stall_displacement_m": float(args.online_map_stall_displacement_m),
        "batch_size": int(args.batch_size),
        "sequence_chunk_len": int(args.sequence_chunk_len),
        "class_weight_power": float(args.class_weight_power),
        "primitive_loss_multipliers": primitive_loss_multipliers,
        "forbid_output_primitives": list(forbid_output_primitives),
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "feature_variant": feature_variant,
        "input_dim": feature_dim,
        "primitive_vocab": list(primitive_vocab),
        "train_count": int(train_labels_np.shape[0]),
        "validation_count": int(val_labels_np.shape[0]),
        "train_label_counts": _label_counts(train_labels_np, primitive_vocab),
        "validation_label_counts": _label_counts(val_labels_np, primitive_vocab),
        "train_sequence_lengths": [int(seq["labels"].shape[0]) for seq in train_sequences],
        "validation_sequence_lengths": [int(seq["labels"].shape[0]) for seq in validation_sequences],
        "final_train": final_train,
        "final_validation": final_validation,
        "history": history,
        "claim_boundary": (
            "Teacher-trained recurrent learned-local policy. Runtime benchmark inference "
            "must run with --explore-standoff-route disabled for route-free evidence."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"learned_local_policy: output={args.output} "
        f"model=gru "
        f"val_acc={final_validation['accuracy']:.3f} "
        f"val_macro_f1={final_validation['macro_f1']:.3f}",
        flush=True,
    )
    return 0


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _initialize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.init()


def _expand_dataset_paths(
    paths: list[Path],
    list_paths: list[Path],
    glob_specs: list[str],
) -> list[Path]:
    expanded = list(paths or [])
    for list_path in list_paths or []:
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            expanded.append(Path(line))
    for spec in glob_specs or []:
        repeat, pattern = _parse_dataset_glob_spec(str(spec))
        matched = [Path(path) for path in sorted(glob.glob(pattern))]
        if not matched:
            raise SystemExit(f"dataset glob matched no paths: {pattern}")
        for _ in range(repeat):
            expanded.extend(matched)
    return expanded


def _parse_dataset_glob_spec(spec: str) -> tuple[int, str]:
    prefix, sep, rest = spec.partition(":")
    if sep and prefix.isdigit():
        repeat = int(prefix)
        if repeat <= 0:
            raise SystemExit(f"dataset glob repeat must be positive: {spec}")
        if not rest:
            raise SystemExit(f"dataset glob pattern is empty: {spec}")
        return repeat, rest
    return 1, spec


def _load_initial_checkpoint_if_requested(
    model: nn.Module,
    *,
    args: argparse.Namespace,
    model_type: str,
    input_dim: int,
    primitive_vocab: list[str],
    device: torch.device,
) -> None:
    path = args.init_checkpoint
    if path is None:
        return
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("schema") != "lewm_go2_closed_loop_learned_local_policy_v0":
        raise SystemExit(f"unsupported init checkpoint schema: {checkpoint.get('schema')}")
    if str(checkpoint.get("model_type", "mlp")) != str(model_type):
        raise SystemExit(
            f"init checkpoint model_type={checkpoint.get('model_type')} "
            f"does not match requested {model_type}"
        )
    init_vocab = [str(item) for item in checkpoint.get("primitive_vocab", [])]
    if init_vocab != [str(item) for item in primitive_vocab]:
        raise SystemExit("init checkpoint primitive vocab does not match training vocab")
    checkpoint_input_dim = int(checkpoint.get("input_dim", -1))
    checkpoint_state = checkpoint["model_state_dict"]
    if checkpoint_input_dim != int(input_dim):
        if not (str(model_type) == "mlp" and 0 < checkpoint_input_dim < int(input_dim)):
            raise SystemExit(
                f"init checkpoint input_dim={checkpoint.get('input_dim')} "
                f"does not match training input_dim={input_dim}"
            )
        current_state = model.state_dict()
        expanded_state = dict(checkpoint_state)
        for key in ("net.0.weight", "net.0.bias"):
            if key in checkpoint_state and key in current_state:
                value = current_state[key].clone()
                value[:checkpoint_input_dim] = checkpoint_state[key]
                expanded_state[key] = value
        if "net.1.weight" in checkpoint_state and "net.1.weight" in current_state:
            value = current_state["net.1.weight"].clone()
            value[:, :checkpoint_input_dim] = checkpoint_state["net.1.weight"]
            expanded_state["net.1.weight"] = value
        model.load_state_dict(expanded_state)
        print(
            f"initialized {model_type} from {path} with input expansion "
            f"{checkpoint_input_dim}->{int(input_dim)}",
            flush=True,
        )
        return
    model.load_state_dict(checkpoint_state)
    print(f"initialized {model_type} from {path}", flush=True)


def _parse_state_filter(raw: str) -> set[str]:
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def _numeric_filters(items: list[str], *, flag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        key, sep, raw_value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        try:
            out[key] = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"{flag} value must be numeric: {item!r}") from exc
    return out


def _string_filters(items: list[str], *, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, sep, raw_value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        out[key] = raw_value.strip()
    return out


def _feature_variant(
    *,
    append_clock: bool,
    append_state: bool,
    append_visual_readout: bool,
    append_pose_topology: bool,
    append_online_map: bool,
) -> str:
    parts: list[str] = []
    if append_pose_topology:
        parts.append("pose_topology")
    elif append_clock:
        parts.append("clock")
    if append_visual_readout:
        parts.append("visual_readout")
    if append_state:
        parts.append("state")
    if append_online_map:
        parts.append("online_map_edge" if _ONLINE_MAP_CHANNELS > 4 else "online_map")
    return "_".join(parts) + "_v1" if parts else "base"


def _build_input_mask(
    *,
    mode: str,
    input_dim: int,
    feature_variant: str,
) -> np.ndarray | None:
    mode = str(mode)
    if mode == "none":
        return None
    keep_fields = {"visual_readout"}
    if mode in {
        "visual_readout_state",
        "visual_readout_state_clock",
        "visual_readout_state_clock_outcome",
        "visual_readout_state_clock_outcome_online_map",
    }:
        keep_fields.add("state")
    if mode in {
        "visual_readout_state_clock",
        "visual_readout_state_clock_outcome",
        "visual_readout_state_clock_outcome_online_map",
    }:
        keep_fields.add("clock")
    if mode in {
        "visual_readout_state_clock_outcome",
        "visual_readout_state_clock_outcome_online_map",
    }:
        keep_fields.add("primitive_outcome")
    if mode == "visual_readout_state_clock_outcome_online_map":
        keep_fields.add("online_map")
    blocks = _feature_blocks_for_variant(feature_variant, input_dim=int(input_dim))
    missing = sorted(field for field in keep_fields if field not in blocks)
    if missing:
        raise SystemExit(
            f"input mask mode {mode!r} requires feature block(s) missing from "
            f"variant {feature_variant!r}: {missing}"
        )
    mask = np.zeros((int(input_dim),), dtype=np.float32)
    for field in keep_fields:
        start, width = blocks[field]
        mask[int(start) : int(start) + int(width)] = 1.0
    return mask


def _feature_blocks_for_variant(
    feature_variant: str,
    *,
    input_dim: int,
) -> dict[str, tuple[int, int]]:
    text = str(feature_variant)
    blocks: dict[str, tuple[int, int]] = {}
    if int(input_dim) >= _BASE_LEARNED_LOCAL_FEATURE_DIM:
        blocks["primitive_outcome"] = (
            _BASE_LEARNED_LOCAL_FEATURE_DIM - _PRIMITIVE_OUTCOME_FEATURE_WIDTH,
            _PRIMITIVE_OUTCOME_FEATURE_WIDTH,
        )
    if text == "base":
        return blocks
    normalised = (
        text.replace("pose_topology", "posetopology")
        .replace("visual_readout", "visualreadout")
        .replace("online_map_edge", "onlinemap")
        .replace("online_map", "onlinemap")
        .removesuffix("_v1")
    )
    offset = _BASE_LEARNED_LOCAL_FEATURE_DIM
    for token in normalised.split("_"):
        if token == "clock":
            blocks["clock"] = (offset, 3)
            offset += 3
        elif token == "state":
            blocks["state"] = (offset, len(_STATE_FEATURES))
            offset += len(_STATE_FEATURES)
        elif token == "visualreadout":
            blocks["visual_readout"] = (offset, 8)
            offset += 8
        elif token == "posetopology":
            blocks["pose_topology"] = (offset, 5)
            offset += 5
        elif token == "onlinemap":
            blocks["online_map"] = (offset, max(0, int(input_dim) - offset))
            break
    return blocks


def _load_many(
    paths: list[Path],
    *,
    include_states: set[str] | None = None,
    meta_min: dict[str, float] | None = None,
    meta_max: dict[str, float] | None = None,
    meta_eq: dict[str, str] | None = None,
    meta_not_eq: dict[str, str] | None = None,
    append_clock_features: bool = False,
    clock_max_ticks: float = 560.0,
    append_state_features: bool = False,
    append_visual_readout_features: bool = False,
    append_pose_topology_features: bool = False,
    pose_scale_m: float = 4.0,
    append_online_map_features: bool = False,
    online_map_size: int = 11,
    online_map_cell_m: float = 0.45,
    online_map_stall_displacement_m: float = 0.015,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    features = []
    labels = []
    primitive_vocab: list[str] | None = None
    reports = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"][0]) if "schema" in data else ""
            if schema != "lewm_go2_closed_loop_learned_local_policy_dataset_v0":
                raise SystemExit(f"unsupported dataset schema in {path}: {schema}")
            current_vocab = [str(item) for item in data["primitive_vocab"].tolist()]
            if primitive_vocab is None:
                primitive_vocab = current_vocab
            elif current_vocab != primitive_vocab:
                raise SystemExit(f"primitive vocab mismatch in {path}")
            current_features = np.asarray(data["features"], dtype=np.float32)
            current_labels = np.asarray(data["labels"], dtype=np.int64)
            if append_clock_features:
                current_features = _append_clock_features(
                    data,
                    current_features,
                    clock_max_ticks=float(clock_max_ticks),
                )
            if append_visual_readout_features:
                current_features = _append_visual_readout_features(data, current_features)
            if append_pose_topology_features:
                current_features = _append_pose_topology_features(
                    data,
                    current_features,
                    pose_scale_m=float(pose_scale_m),
                )
            if append_state_features:
                current_features = _append_state_features(data, current_features)
            if append_online_map_features:
                current_features = _append_online_map_features(
                    data,
                    current_features,
                    map_size=int(online_map_size),
                    cell_m=float(online_map_cell_m),
                    stall_displacement_m=float(online_map_stall_displacement_m),
                )
            keep = _metadata_keep_mask(
                data,
                rows=int(current_labels.shape[0]),
                include_states=include_states or set(),
                meta_min=meta_min or {},
                meta_max=meta_max or {},
                meta_eq=meta_eq or {},
                meta_not_eq=meta_not_eq or {},
            )
            if not bool(np.all(keep)):
                current_features = current_features[keep]
                current_labels = current_labels[keep]
            features.append(current_features)
            labels.append(current_labels)
            if "result_json" in data and len(data["result_json"]) > 0:
                reports.append(json.loads(str(data["result_json"][0])))
    return (
        np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32),
        np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64),
        primitive_vocab or [],
        reports,
    )


def _state_keep_mask(data: np.lib.npyio.NpzFile, *, include_states: set[str]) -> np.ndarray:
    labels = np.asarray(data["labels"], dtype=np.int64)
    keep = np.ones((labels.shape[0],), dtype=bool)
    if "meta_json" not in data or len(data["meta_json"]) != labels.shape[0]:
        return keep
    keep[:] = False
    for idx, raw in enumerate(data["meta_json"].tolist()):
        try:
            state = str(json.loads(str(raw)).get("state", "")).upper()
        except json.JSONDecodeError:
            state = ""
        keep[idx] = state in include_states
    return keep


def _metadata_keep_mask(
    data: np.lib.npyio.NpzFile,
    *,
    rows: int,
    include_states: set[str],
    meta_min: dict[str, float],
    meta_max: dict[str, float],
    meta_eq: dict[str, str],
    meta_not_eq: dict[str, str],
) -> np.ndarray:
    keep = np.ones((int(rows),), dtype=bool)
    if "meta_json" not in data or len(data["meta_json"]) != int(rows):
        return keep
    meta = _meta_rows(data, rows=int(rows))
    keep &= np.asarray(
        [
            not (
                _meta_filter_bool(row.get("training_update_only"))
                or _meta_filter_bool(row.get("online_map_update_only"))
            )
            for row in meta
        ],
        dtype=bool,
    )
    if not include_states and not meta_min and not meta_max and not meta_eq and not meta_not_eq:
        return keep
    if include_states:
        keep &= np.asarray(
            [str(row.get("state", "")).upper() in include_states for row in meta],
            dtype=bool,
        )
    for key, threshold in meta_min.items():
        keep &= np.asarray(
            [
                (value := _meta_filter_float(row.get(key))) is not None
                and value >= float(threshold)
                for row in meta
            ],
            dtype=bool,
        )
    for key, threshold in meta_max.items():
        keep &= np.asarray(
            [
                (value := _meta_filter_float(row.get(key))) is not None
                and value <= float(threshold)
                for row in meta
            ],
            dtype=bool,
        )
    for key, expected in meta_eq.items():
        keep &= np.asarray(
            [_meta_filter_string(row.get(key)) == str(expected) for row in meta],
            dtype=bool,
        )
    for key, forbidden in meta_not_eq.items():
        keep &= np.asarray(
            [_meta_filter_string(row.get(key)) != str(forbidden) for row in meta],
            dtype=bool,
        )
    return keep


def _meta_filter_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _meta_filter_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _meta_filter_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _append_clock_features(
    data: np.lib.npyio.NpzFile,
    features: np.ndarray,
    *,
    clock_max_ticks: float,
) -> np.ndarray:
    if _dataset_already_has_clock_features(data):
        return features
    rows = int(features.shape[0])
    extras = np.zeros((rows, 3), dtype=np.float32)
    if "meta_json" not in data or len(data["meta_json"]) != rows:
        return np.concatenate([features, extras], axis=1)

    parsed: list[dict[str, Any]] = []
    max_target_index = 0
    for raw in data["meta_json"].tolist():
        try:
            meta = json.loads(str(raw))
        except json.JSONDecodeError:
            meta = {}
        parsed.append(meta)
        max_target_index = max(max_target_index, int(meta.get("target_index", 0)))

    denom_ticks = max(1.0, float(clock_max_ticks))
    denom_target = max(1.0, float(max_target_index))
    last_target_index: int | None = None
    target_start_tick = 0
    for idx, meta in enumerate(parsed):
        tick = int(meta.get("tick", idx))
        target_index = int(meta.get("target_index", 0))
        if last_target_index is None or target_index != last_target_index:
            target_start_tick = tick
            last_target_index = target_index
        extras[idx, 0] = float(tick) / denom_ticks
        extras[idx, 1] = float(max(0, tick - target_start_tick)) / denom_ticks
        extras[idx, 2] = float(target_index) / denom_target
    return _append_before_existing_online_map(data, features, extras)


def _dataset_already_has_clock_features(data: np.lib.npyio.NpzFile) -> bool:
    variant = _dataset_feature_variant(data)
    return "clock" in variant or "pose_topology" in variant


def _dataset_already_has_pose_topology_features(data: np.lib.npyio.NpzFile) -> bool:
    return "pose_topology" in _dataset_feature_variant(data)


def _dataset_already_has_state_features(data: np.lib.npyio.NpzFile) -> bool:
    return "state" in _dataset_feature_variant(data)


def _dataset_already_has_visual_readout_features(data: np.lib.npyio.NpzFile) -> bool:
    return "visual_readout" in _dataset_feature_variant(data)


def _dataset_already_has_online_map_features(data: np.lib.npyio.NpzFile) -> bool:
    return "online_map" in _dataset_feature_variant(data)


def _dataset_feature_variant(data: np.lib.npyio.NpzFile) -> str:
    if "result_json" not in data or len(data["result_json"]) == 0:
        return "base"
    try:
        result = json.loads(str(data["result_json"][0]))
    except json.JSONDecodeError:
        return "base"
    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        result = result["result"]
    if not isinstance(result, dict):
        return "base"
    metrics = result.get("wall_metrics", {})
    if not isinstance(metrics, dict):
        return "base"
    if _dataset_uses_post_claim_feature_slot(data):
        post_variant = metrics.get("learned_local_post_claim_policy_feature_variant")
        if post_variant is not None:
            return str(post_variant)
    return str(metrics.get("learned_local_policy_feature_variant", "base"))


def _dataset_uses_post_claim_feature_slot(data: np.lib.npyio.NpzFile) -> bool:
    if "meta_json" not in data or len(data["meta_json"]) == 0:
        return False
    saw_slot = False
    for raw in data["meta_json"].tolist():
        try:
            meta = json.loads(str(raw))
        except json.JSONDecodeError:
            return False
        if not isinstance(meta, dict):
            return False
        slot = meta.get("policy_feature_slot")
        if slot is None:
            continue
        saw_slot = True
        if str(slot) != "post_claim":
            return False
    return bool(saw_slot)


def _append_pose_topology_features(
    data: np.lib.npyio.NpzFile,
    features: np.ndarray,
    *,
    pose_scale_m: float,
) -> np.ndarray:
    if _dataset_already_has_pose_topology_features(data):
        return features
    rows = int(features.shape[0])
    extras = np.zeros((rows, 5), dtype=np.float32)
    if "meta_json" not in data or len(data["meta_json"]) != rows:
        return np.concatenate([features, extras], axis=1)
    pose_scale = max(1e-6, float(pose_scale_m))
    max_claimed = 1
    parsed = []
    for raw in data["meta_json"].tolist():
        try:
            meta = json.loads(str(raw))
        except json.JSONDecodeError:
            meta = {}
        parsed.append(meta)
        max_claimed = max(max_claimed, int(meta.get("claimed_count", 0)))
    claim_denom = max(1.0, float(max_claimed + 1))
    for idx, meta in enumerate(parsed):
        pose_xy = meta.get("pose_xy", [0.0, 0.0])
        if not isinstance(pose_xy, (list, tuple)) or len(pose_xy) < 2:
            pose_xy = [0.0, 0.0]
        yaw = float(meta.get("yaw_rad", 0.0))
        extras[idx, 0] = float(pose_xy[0]) / pose_scale
        extras[idx, 1] = float(pose_xy[1]) / pose_scale
        extras[idx, 2] = float(np.sin(yaw))
        extras[idx, 3] = float(np.cos(yaw))
        extras[idx, 4] = float(meta.get("claimed_count", 0)) / claim_denom
    return _append_before_existing_online_map(data, features, extras)


def _append_state_features(
    data: np.lib.npyio.NpzFile,
    features: np.ndarray,
) -> np.ndarray:
    if _dataset_already_has_state_features(data):
        return features
    rows = int(features.shape[0])
    states = ("EXPLORE", "SEEK", "SERVO", "CLAIM")
    extras = np.zeros((rows, len(states)), dtype=np.float32)
    if "meta_json" not in data or len(data["meta_json"]) != rows:
        return np.concatenate([features, extras], axis=1)
    state_to_idx = {name: idx for idx, name in enumerate(states)}
    for idx, raw in enumerate(data["meta_json"].tolist()):
        try:
            state = str(json.loads(str(raw)).get("state", "")).upper()
        except json.JSONDecodeError:
            state = ""
        state_idx = state_to_idx.get(state)
        if state_idx is not None:
            extras[idx, state_idx] = 1.0
    return np.concatenate([features, extras], axis=1)


def _append_visual_readout_features(
    data: np.lib.npyio.NpzFile,
    features: np.ndarray,
) -> np.ndarray:
    if _dataset_already_has_visual_readout_features(data):
        return features
    rows = int(features.shape[0])
    extras = np.zeros((rows, 8), dtype=np.float32)
    if "meta_json" not in data or len(data["meta_json"]) != rows:
        return np.concatenate([features, extras], axis=1)
    for idx, raw in enumerate(data["meta_json"].tolist()):
        try:
            meta = json.loads(str(raw))
        except json.JSONDecodeError:
            meta = {}
        area = _finite_float(meta.get("area"), default=-9.0)
        bearing = _finite_float(meta.get("bearing"), default=0.0)
        mem_conf = _finite_float(meta.get("mem_conf"), default=0.0)
        read_score = _finite_float(meta.get("read_score"), default=-1.0)
        in_cone = 1.0 if bool(meta.get("in_cone", False)) else 0.0
        seen = 1.0 if bool(meta.get("seen", False)) else 0.0
        claimed_count = _finite_float(meta.get("claimed_count"), default=0.0)
        extras[idx] = np.asarray(
            [
                area / 4.0,
                float(np.sin(bearing)),
                float(np.cos(bearing)),
                bearing / float(np.pi),
                mem_conf,
                read_score,
                in_cone,
                claimed_count / 4.0,
            ],
            dtype=np.float32,
        )
        if seen > 0.0:
            extras[idx, 5] = max(extras[idx, 5], 0.0)
    return _append_before_existing_online_map(data, features, extras)


def _append_online_map_features(
    data: np.lib.npyio.NpzFile,
    features: np.ndarray,
    *,
    map_size: int,
    cell_m: float,
    stall_displacement_m: float,
) -> np.ndarray:
    if _dataset_already_has_online_map_features(data):
        return _resize_existing_online_map_features(
            data,
            features,
            target_map_size=int(map_size),
        )
    rows = int(features.shape[0])
    size = int(map_size)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    width = _ONLINE_MAP_CHANNELS * size * size
    extras = np.zeros((rows, width), dtype=np.float32)
    if "meta_json" not in data or len(data["meta_json"]) != rows:
        return np.concatenate([features, extras], axis=1)

    primitive_vocab = [str(item) for item in data["primitive_vocab"].tolist()]
    labels = np.asarray(data["labels"], dtype=np.int64)
    meta = _meta_rows(data, rows=rows)
    replay = OfflineOnlineEgomotionMap(size=size, cell_m=float(cell_m))
    previous_claimed = 0
    for idx, row in enumerate(meta):
        pose_xy = row.get("pose_xy", [0.0, 0.0])
        yaw = float(row.get("yaw_rad", 0.0))
        tick = int(row.get("tick", idx))
        claimed_count = int(row.get("claimed_count", previous_claimed))
        if _meta_filter_bool(row.get("online_map_reset")):
            replay.reset_after_claim(pose_xy, tick=tick)
        elif claimed_count > previous_claimed:
            replay.mark_claim(pose_xy)
            replay.reset_after_claim(pose_xy, tick=tick)
        previous_claimed = max(previous_claimed, claimed_count)
        replay.observe_pose(pose_xy, tick=tick)
        extras[idx] = replay.feature(pose_xy, yaw, tick=tick)

        label_idx = int(labels[idx]) if idx < int(labels.shape[0]) else -1
        label = primitive_vocab[label_idx] if 0 <= label_idx < len(primitive_vocab) else ""
        next_row = meta[idx + 1] if idx + 1 < len(meta) else None
        if next_row is not None:
            next_pose = next_row.get("pose_xy", pose_xy)
            displacement = float(np.linalg.norm(_pose_xy_array(next_pose) - _pose_xy_array(pose_xy)))
            if _meta_filter_bool(row.get("online_map_guard_blocked_probe")):
                replay.mark_guard_blocked_primitive(
                    pose_xy,
                    float(yaw),
                    str(row.get("online_map_guard_blocked_primitive", label)),
                )
                continue
            replay.update_after_action(
                pose_xy=pose_xy,
                post_xy=next_pose,
                yaw_rad=float(yaw),
                primitive=str(label),
                stalled=bool(
                    label in _TRANSLATING_LABELS
                    and displacement < float(stall_displacement_m)
                ),
                tick=int(tick),
            )
    return np.concatenate([features, extras], axis=1)


def _finite_float(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _append_before_existing_online_map(
    data: np.lib.npyio.NpzFile,
    features: np.ndarray,
    extras: np.ndarray,
) -> np.ndarray:
    if not _dataset_already_has_online_map_features(data):
        return np.concatenate([features, extras], axis=1)
    prefix_dim = _non_map_prefix_dim_for_variant(_dataset_feature_variant(data))
    return np.concatenate(
        [features[:, :prefix_dim], extras, features[:, prefix_dim:]],
        axis=1,
    ).astype(np.float32, copy=False)


def _resize_existing_online_map_features(
    data: np.lib.npyio.NpzFile,
    features: np.ndarray,
    *,
    target_map_size: int,
) -> np.ndarray:
    target_size = _odd_online_map_size(target_map_size)
    target_channels = _ONLINE_MAP_CHANNELS
    target_width = target_channels * target_size * target_size
    variant = _dataset_feature_variant(data)
    prefix_dim = _non_map_prefix_dim_for_variant(variant)
    feature_dim = int(features.shape[1])
    rows = int(features.shape[0])
    if rows == 0 and feature_dim == 0:
        return np.zeros((0, prefix_dim + target_width), dtype=np.float32)
    if (
        "visual_readout" not in variant
        and feature_dim == prefix_dim + 8 + target_width
    ):
        prefix_dim += 8
    if feature_dim == prefix_dim + target_width:
        return features
    source_width = feature_dim - prefix_dim
    source_channels = 0
    source_size = 0
    for candidate_channels in (target_channels, 4, 8):
        if source_width <= 0 or source_width % int(candidate_channels) != 0:
            continue
        side_sq = source_width // int(candidate_channels)
        candidate_size = int(round(float(side_sq) ** 0.5))
        if candidate_size * candidate_size == side_sq and candidate_size % 2 == 1:
            source_channels = int(candidate_channels)
            source_size = int(candidate_size)
            break
    if source_channels <= 0 or source_size <= 0:
        raise SystemExit(
            "cannot resize existing online-map features: "
            f"variant={variant!r} feature_dim={feature_dim} prefix_dim={prefix_dim}"
        )
    if source_size == target_size and source_channels == target_channels:
        return features

    source_maps = features[:, prefix_dim:].reshape(
        rows,
        source_channels,
        source_size,
        source_size,
    )
    target_maps = np.zeros(
        (rows, target_channels, target_size, target_size),
        dtype=np.float32,
    )
    copy_size = min(source_size, target_size)
    copy_channels = min(source_channels, target_channels)
    source_start = (source_size - copy_size) // 2
    target_start = (target_size - copy_size) // 2
    target_maps[
        :,
        :copy_channels,
        target_start : target_start + copy_size,
        target_start : target_start + copy_size,
    ] = source_maps[
        :,
        :copy_channels,
        source_start : source_start + copy_size,
        source_start : source_start + copy_size,
    ]
    return np.concatenate(
        [
            features[:, :prefix_dim],
            target_maps.reshape(rows, target_width),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _odd_online_map_size(size: int) -> int:
    out = max(3, int(size))
    return out + 1 if out % 2 == 0 else out


def _non_map_prefix_dim_for_variant(variant: str) -> int:
    text = str(variant)
    if "online_map" not in text:
        raise SystemExit(f"dataset does not advertise online_map features: {variant!r}")
    prefix_dim = _BASE_LEARNED_LOCAL_FEATURE_DIM
    if "pose_topology" in text:
        prefix_dim += 5
    elif "clock" in text:
        prefix_dim += 3
    if "visual_readout" in text:
        prefix_dim += 8
    if "state" in text:
        prefix_dim += len(_STATE_FEATURES)
    return int(prefix_dim)


def _meta_rows(data: np.lib.npyio.NpzFile, *, rows: int) -> list[dict[str, Any]]:
    if "meta_json" not in data or len(data["meta_json"]) != rows:
        return [{} for _ in range(rows)]
    parsed: list[dict[str, Any]] = []
    for raw in data["meta_json"].tolist():
        try:
            item = json.loads(str(raw))
        except json.JSONDecodeError:
            item = {}
        parsed.append(item if isinstance(item, dict) else {})
    return parsed


def _pose_xy_array(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return np.asarray([float(value[0]), float(value[1])], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def _load_many_sequences(
    paths: list[Path],
    *,
    include_states: set[str] | None = None,
    meta_min: dict[str, float] | None = None,
    meta_max: dict[str, float] | None = None,
    meta_eq: dict[str, str] | None = None,
    meta_not_eq: dict[str, str] | None = None,
    append_clock_features: bool = False,
    clock_max_ticks: float = 560.0,
    append_state_features: bool = False,
    append_visual_readout_features: bool = False,
    append_pose_topology_features: bool = False,
    pose_scale_m: float = 4.0,
    append_online_map_features: bool = False,
    online_map_size: int = 11,
    online_map_cell_m: float = 0.45,
    online_map_stall_displacement_m: float = 0.015,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    sequences: list[dict[str, Any]] = []
    primitive_vocab: list[str] | None = None
    reports = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"][0]) if "schema" in data else ""
            if schema != "lewm_go2_closed_loop_learned_local_policy_dataset_v0":
                raise SystemExit(f"unsupported dataset schema in {path}: {schema}")
            current_vocab = [str(item) for item in data["primitive_vocab"].tolist()]
            if primitive_vocab is None:
                primitive_vocab = current_vocab
            elif current_vocab != primitive_vocab:
                raise SystemExit(f"primitive vocab mismatch in {path}")
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
            if features.shape[0] != labels.shape[0]:
                raise SystemExit(f"feature/label length mismatch in {path}")
            if labels.shape[0] == 0:
                if "result_json" in data and len(data["result_json"]) > 0:
                    reports.append(json.loads(str(data["result_json"][0])))
                continue
            if append_clock_features:
                features = _append_clock_features(
                    data,
                    features,
                    clock_max_ticks=float(clock_max_ticks),
                )
            if append_visual_readout_features:
                features = _append_visual_readout_features(data, features)
            if append_pose_topology_features:
                features = _append_pose_topology_features(
                    data,
                    features,
                    pose_scale_m=float(pose_scale_m),
                )
            if append_state_features:
                features = _append_state_features(data, features)
            if append_online_map_features:
                features = _append_online_map_features(
                    data,
                    features,
                    map_size=int(online_map_size),
                    cell_m=float(online_map_cell_m),
                    stall_displacement_m=float(online_map_stall_displacement_m),
                )
            keep = _metadata_keep_mask(
                data,
                rows=int(labels.shape[0]),
                include_states=include_states or set(),
                meta_min=meta_min or {},
                meta_max=meta_max or {},
                meta_eq=meta_eq or {},
                meta_not_eq=meta_not_eq or {},
            )
            if not bool(np.all(keep)):
                features = features[keep]
                labels = labels[keep]
            order = np.arange(labels.shape[0])
            if "meta_json" in data and len(data["meta_json"]) == keep.shape[0]:
                ticks = []
                for raw in np.asarray(data["meta_json"])[keep].tolist():
                    try:
                        ticks.append(int(json.loads(str(raw)).get("tick", len(ticks))))
                    except json.JSONDecodeError:
                        ticks.append(len(ticks))
                order = np.argsort(np.asarray(ticks), kind="stable")
            if labels.shape[0] > 0:
                sequences.append(
                    {
                        "path": str(path),
                        "features": features[order],
                        "labels": labels[order],
                    }
                )
            if "result_json" in data and len(data["result_json"]) > 0:
                reports.append(json.loads(str(data["result_json"][0])))
    return sequences, primitive_vocab or [], reports


def _chunk_sequences(
    sequences: list[dict[str, Any]],
    *,
    chunk_len: int,
) -> list[dict[str, Any]]:
    if int(chunk_len) <= 0:
        return list(sequences)
    out: list[dict[str, Any]] = []
    length_limit = max(1, int(chunk_len))
    for seq in sequences:
        features = np.asarray(seq["features"], dtype=np.float32)
        labels = np.asarray(seq["labels"], dtype=np.int64)
        row_count = int(labels.shape[0])
        for start in range(0, row_count, length_limit):
            end = min(row_count, start + length_limit)
            if end <= start:
                continue
            out.append(
                {
                    "path": f"{seq.get('path', '')}#{start}:{end}",
                    "features": features[start:end],
                    "labels": labels[start:end],
                }
            )
    return out


def _split_sequences(
    sequences: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if float(validation_fraction) <= 0.0 or len(sequences) <= 1:
        return list(sequences), []
    rng = random.Random(int(seed))
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(validation_fraction))))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _sequence_minibatches(
    sequences: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[list[dict[str, Any]]]:
    indices = list(range(len(sequences)))
    if bool(shuffle):
        rng = random.Random(int(seed))
        rng.shuffle(indices)
    size = max(1, int(batch_size))
    return [
        [sequences[idx] for idx in indices[start : start + size]]
        for start in range(0, len(indices), size)
    ]


def _concat_sequence_labels(sequences: list[dict[str, Any]]) -> np.ndarray:
    if not sequences:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(
        [np.asarray(seq["labels"], dtype=np.int64) for seq in sequences],
        axis=0,
    )


def _pad_sequences(
    sequences: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("no sequences to pad")
    feature_dim = int(sequences[0]["features"].shape[1])
    max_len = max(int(seq["labels"].shape[0]) for seq in sequences)
    features = torch.zeros(
        (len(sequences), max_len, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    labels = torch.zeros(
        (len(sequences), max_len),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros(
        (len(sequences), max_len),
        dtype=torch.bool,
        device=device,
    )
    for idx, seq in enumerate(sequences):
        seq_features = np.asarray(seq["features"], dtype=np.float32)
        seq_labels = np.asarray(seq["labels"], dtype=np.int64)
        length = int(seq_labels.shape[0])
        if int(seq_features.shape[1]) != feature_dim:
            raise SystemExit("sequence feature dimensions do not match")
        features[idx, :length] = torch.from_numpy(seq_features).to(device)
        labels[idx, :length] = torch.from_numpy(seq_labels).to(device)
        mask[idx, :length] = True
    return features, labels, mask


@torch.no_grad()
def _evaluate_sequence_batches(
    model: LearnedLocalRecurrentPolicyHead,
    sequences: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    primitive_vocab: list[str],
) -> dict[str, Any]:
    model.eval()
    pred_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for batch in _sequence_minibatches(
        sequences,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        seed=0,
    ):
        features, labels, mask = _pad_sequences(batch, device=device)
        logits = model(features)
        pred_parts.append(torch.argmax(logits[mask], dim=-1).detach().cpu())
        label_parts.append(labels[mask].detach().cpu())
    if not label_parts:
        empty = torch.zeros((0,), dtype=torch.long)
        return _evaluate_predictions(empty, empty, primitive_vocab=primitive_vocab)
    return _evaluate_predictions(
        torch.cat(pred_parts, dim=0),
        torch.cat(label_parts, dim=0),
        primitive_vocab=primitive_vocab,
    )


def _split_train_validation(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    if float(validation_fraction) <= 0.0:
        return features[indices], labels[indices], features[:0], labels[:0]
    val_count = max(1, int(round(features.shape[0] * float(validation_fraction))))
    val_count = min(val_count, features.shape[0] - 1)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]


def _class_weights(
    labels: np.ndarray,
    *,
    primitive_count: int,
    power: float = 1.0,
) -> torch.Tensor:
    counts = Counter(int(item) for item in labels.tolist())
    weights = []
    total = max(1, len(labels))
    exponent = max(0.0, float(power))
    for idx in range(int(primitive_count)):
        inverse = total / max(1, counts.get(idx, 0))
        weights.append(float(inverse) ** exponent)
    arr = torch.tensor(weights, dtype=torch.float32)
    return arr / arr.mean().clamp_min(1e-6)


def _primitive_loss_multipliers(
    spec: str,
    *,
    primitive_vocab: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    text = str(spec or "").strip()
    if not text:
        return result
    known = {str(name) for name in primitive_vocab}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                "--primitive-loss-multipliers entries must be primitive=multiplier"
            )
        name, value_text = item.split("=", 1)
        name = name.strip()
        if name not in known:
            raise SystemExit(f"unknown primitive in --primitive-loss-multipliers: {name}")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise SystemExit(f"invalid primitive multiplier for {name}: {value_text}") from exc
        if value < 0.0:
            raise SystemExit(f"primitive multiplier for {name} must be non-negative")
        result[name] = value
    return result


def _apply_primitive_loss_multipliers(
    weights: torch.Tensor,
    multipliers: dict[str, float],
    *,
    primitive_vocab: list[str],
) -> torch.Tensor:
    if not multipliers:
        return weights
    adjusted = weights.detach().clone()
    for idx, name in enumerate(primitive_vocab):
        if str(name) in multipliers:
            adjusted[idx] = adjusted[idx] * float(multipliers[str(name)])
    return adjusted / adjusted.mean().clamp_min(1e-6)


def _forbid_output_primitives(
    spec: str,
    *,
    primitive_vocab: list[str],
    labels: np.ndarray,
) -> list[str]:
    names = [item.strip() for item in str(spec or "").split(",") if item.strip()]
    if not names:
        return []
    known = {str(name) for name in primitive_vocab}
    unknown = sorted(name for name in names if name not in known)
    if unknown:
        raise SystemExit(f"unknown primitive(s) in --forbid-output-primitives: {unknown}")
    label_ids = {int(item) for item in np.asarray(labels, dtype=np.int64).tolist()}
    conflicting = [
        name
        for name in names
        if primitive_vocab.index(name) in label_ids
    ]
    if conflicting:
        raise SystemExit(
            "--forbid-output-primitives conflicts with training labels: "
            f"{conflicting}"
        )
    return sorted(dict.fromkeys(names))


@torch.no_grad()
def _evaluate(
    model: LearnedLocalPolicyHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    primitive_vocab: list[str],
    batch_size: int = 0,
) -> dict[str, Any]:
    model.eval()
    batch_size = int(batch_size)
    if batch_size <= 0 or int(features.shape[0]) <= batch_size:
        logits = model(features)
        return _evaluate_logits(logits, labels, primitive_vocab=primitive_vocab)
    preds = []
    for start in range(0, int(features.shape[0]), batch_size):
        logits = model(features[start : start + batch_size])
        preds.append(torch.argmax(logits, dim=-1).detach())
    return _evaluate_predictions(
        torch.cat(preds, dim=0),
        labels,
        primitive_vocab=primitive_vocab,
    )


@torch.no_grad()
def _evaluate_sequence(
    model: LearnedLocalRecurrentPolicyHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    primitive_vocab: list[str],
) -> dict[str, Any]:
    model.eval()
    logits = model(features)
    return _evaluate_logits(
        logits[mask],
        labels[mask],
        primitive_vocab=primitive_vocab,
    )


def _evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    primitive_vocab: list[str],
) -> dict[str, Any]:
    return _evaluate_predictions(
        torch.argmax(logits, dim=-1),
        labels,
        primitive_vocab=primitive_vocab,
    )


def _evaluate_predictions(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    primitive_vocab: list[str],
) -> dict[str, Any]:
    correct = pred.eq(labels)
    by_class = {}
    f1_values = []
    confusion: dict[str, dict[str, int]] = {}
    for idx, name in enumerate(primitive_vocab):
        idx_t = torch.tensor(idx, device=labels.device)
        tp = int((pred.eq(idx_t) & labels.eq(idx_t)).sum().detach().cpu())
        fp = int((pred.eq(idx_t) & ~labels.eq(idx_t)).sum().detach().cpu())
        fn = int((~pred.eq(idx_t) & labels.eq(idx_t)).sum().detach().cpu())
        support = int(labels.eq(idx_t).sum().detach().cpu())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        if support > 0:
            f1_values.append(f1)
        by_class[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    pred_cpu = pred.detach().cpu().tolist()
    labels_cpu = labels.detach().cpu().tolist()
    for target, predicted in zip(labels_cpu, pred_cpu):
        target_name = primitive_vocab[int(target)]
        predicted_name = primitive_vocab[int(predicted)]
        confusion.setdefault(target_name, {})
        confusion[target_name][predicted_name] = confusion[target_name].get(predicted_name, 0) + 1
    example_count = int(labels.numel())
    return {
        "accuracy": float(correct.float().mean().detach().cpu()) if example_count > 0 else 0.0,
        "correct_count": int(correct.sum().detach().cpu()),
        "example_count": example_count,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "by_class": by_class,
        "confusion": confusion,
    }


def _label_counts(labels: np.ndarray, primitive_vocab: list[str]) -> dict[str, int]:
    counts = Counter(int(item) for item in labels.tolist())
    return {
        primitive_vocab[idx]: int(counts.get(idx, 0))
        for idx in range(len(primitive_vocab))
    }


if __name__ == "__main__":
    raise SystemExit(main())
