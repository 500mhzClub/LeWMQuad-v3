#!/usr/bin/env python3
"""Train LeWorldModel (JEPA) from Genesis-rendered vision and actions.

Supports temporal context ablations (seq_len sweep) as requested in the v3 spec.
Training uses MSE prediction loss + SIGReg anti-collapse regularisation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import logging
import os
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, WeightedRandomSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.actions import (
    ACTIVE_BLOCK_DIM,
    ACTIVE_BLOCK_ORDER,
    active_block_metadata,
    assert_active_block_metadata_compatible,
    encode_executed_command_block,
)
from lewm.models.lewm import LeWorldModel
from lewm.models.pose_head import RelPoseHead, pose_aux_loss, predicted_pose_aux_loss

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MATERIAL_COLOR_SCHEMAS = {"lewm_rendered_vision_v03"}
MATERIAL_COLOR_VISUALS = {"material_color", "solid_color", "solid_material"}
CHECKPOINT_RE = re.compile(r"^lewm_seq(?P<seq_len>\d+)_e(?P<epoch>\d+)(?:_b(?P<batch>\d+))?\.pt$")


def _stable_unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(1 << 64)


class RunningAverages:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, values: Dict[str, float], weight: int) -> None:
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * weight
            self.counts[key] = self.counts.get(key, 0) + weight

    def means(self) -> Dict[str, float]:
        return {
            key: self.totals[key] / max(1, self.counts[key])
            for key in sorted(self.totals)
        }

    def state_dict(self) -> Dict[str, Dict[str, float | int]]:
        return {
            "totals": dict(self.totals),
            "counts": dict(self.counts),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningAverages":
        stats = cls()
        stats.totals = {str(key): float(value) for key, value in state["totals"].items()}
        stats.counts = {str(key): int(value) for key, value in state["counts"].items()}
        return stats


# ---------------------------------------------------------------------------
# Dataset Helpers
# ---------------------------------------------------------------------------

class GenesisWMSession:
    """A single episode's worth of rendered frames and actions."""

    def __init__(
        self,
        scene_dir: Path,
        env_index: int,
        n_envs: int,
        actions: np.ndarray,
        resets: np.ndarray,
        sources: Optional[List[str]] = None,
        poses_by_frame: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.scene_dir = scene_dir
        self.env_index = env_index
        self.n_envs = n_envs
        self.actions = actions  # (T, 15)
        self.resets = resets  # (T,) bool
        # Per-block collector source (command_source from the requested
        # CommandBlock), aligned with actions. None if the corpus lacks it.
        self.sources = sources  # list[str] length T, or None
        self.poses_by_frame = poses_by_frame

    def __len__(self) -> int:
        return len(self.actions)

    def get_rgb_path(self, step: int) -> Path:
        global_idx = step * self.n_envs + self.env_index
        return self.scene_dir / "rgb" / f"frame_{global_idx:06d}_env_{self.env_index:02d}.png"

    def get_pose(self, step: int) -> np.ndarray:
        if self.poses_by_frame is None:
            raise RuntimeError("physical pose labels were not loaded for this session")
        global_idx = step * self.n_envs + self.env_index
        try:
            return self.poses_by_frame[global_idx]
        except KeyError as exc:
            raise RuntimeError(
                f"missing physical pose label for frame {global_idx} in {self.scene_dir}"
            ) from exc


def _load_replay_plan(summary: dict, summary_file: Path) -> Tuple[dict, Path]:
    plan_path = Path(summary["plan"])
    if not plan_path.is_absolute():
        plan_path = (summary_file.parent / plan_path).resolve()
    return json.loads(plan_path.read_text(encoding="utf-8")), plan_path


def _load_pose_labels(plan: dict, plan_path: Path) -> Dict[int, np.ndarray]:
    """Load aligned rendered-frame physical SE(2) labels from the replay plan."""
    frames_path = Path(plan["frames_jsonl"])
    if not frames_path.is_absolute():
        frames_path = (plan_path.parent / frames_path).resolve()
    poses: Dict[int, np.ndarray] = {}
    with frames_path.open(encoding="utf-8") as fh:
        for line in fh:
            frame = json.loads(line)
            pos = frame["base_pose_world"]["position"]
            yaw = frame["base_rpy_rad"]["yaw"]
            poses[int(frame["frame_index"])] = np.asarray(
                [float(pos["x"]), float(pos["y"]), float(yaw)],
                dtype=np.float32,
            )
    return poses


def _is_material_color_render(summary: dict) -> bool:
    schema = str(summary.get("schema") or "")
    visuals = str(summary.get("visuals") or "")
    if visuals in MATERIAL_COLOR_VISUALS:
        return True
    # Older diagnostic renders used the v03 schema without a precise visuals
    # field. Do not classify current textured_v03 renders as material-color just
    # because they share that schema string.
    return schema in MATERIAL_COLOR_SCHEMAS and not visuals.startswith("textured")


def _validate_render_summary(
    summary: dict,
    summary_file: Path,
    *,
    allow_material_color_render: bool,
) -> None:
    if allow_material_color_render or not _is_material_color_render(summary):
        return
    raise RuntimeError(
        "Refusing to train on material-color rendered data. "
        f"summary={summary_file} schema={summary.get('schema')!r} "
        f"visuals={summary.get('visuals')!r}. "
        "Use --render-root pointing at textured renders, or pass "
        "--allow-material-color-render for an explicit diagnostic baseline."
    )


def process_scene_v03(
    scene_dir: Path,
    root_dir: Path,
    split: str,
    allow_material_color_render: bool,
    include_pose_labels: bool = False,
) -> List[GenesisWMSession]:
    """Helper for multithreaded session loading."""
    summary_file = scene_dir / "summary.json"
    if not summary_file.exists():
        return []
    
    with open(summary_file, "r") as f:
        summary = json.loads(f.read())
    _validate_render_summary(
        summary,
        summary_file,
        allow_material_color_render=allow_material_color_render,
    )
    
    if summary.get("split") != split:
        return []

    # Find raw messages
    plan_path = Path(summary["plan"])
    try:
        rollout_idx = plan_path.parts.index("rollout")
        rel_plan = Path(*plan_path.parts[rollout_idx:])
        chunk_root = rel_plan.parents[2]
        messages_file = root_dir / chunk_root / "raw" / scene_dir.name / "messages.jsonl"
        if not messages_file.exists():
            # Fallback for .generated/datagen_full structure
            messages_file = root_dir / ".generated" / "datagen_full" / chunk_root / "raw" / scene_dir.name / "messages.jsonl"
    except (ValueError, IndexError):
        messages_file = plan_path.parents[2] / "raw" / scene_dir.name / "messages.jsonl"

    if not messages_file.exists():
        return []

    actions_by_env: Dict[int, List[np.ndarray]] = {}
    resets_by_env: Dict[int, List[bool]] = {}
    seqids_by_env: Dict[int, List[object]] = {}
    # command_source rides on the requested CommandBlock, not the executed one;
    # join it onto each executed block by (env_index, sequence_id).
    source_by_key: Dict[Tuple[int, object], str] = {}
    
    with open(messages_file, "r") as f:
        for line in f:
            msg = json.loads(line)
            env_idx = msg.get("env_index", 0)
            msg_type = msg["type"]
            if msg_type == "lewm_go2_control/msg/ExecutedCommandBlock":
                payload = msg["payload"]
                block = encode_executed_command_block(payload)
                if env_idx not in actions_by_env:
                    actions_by_env[env_idx] = []
                    resets_by_env[env_idx] = []
                    seqids_by_env[env_idx] = []
                actions_by_env[env_idx].append(block)
                resets_by_env[env_idx].append(False)
                seqids_by_env[env_idx].append(payload.get("sequence_id"))
            elif msg_type == "lewm_go2_control/msg/CommandBlock":
                payload = msg["payload"]
                source_by_key[(env_idx, payload.get("sequence_id"))] = str(
                    payload.get("command_source", "unknown")
                )
            elif msg_type == "lewm_go2_control/msg/ResetEvent":
                if env_idx in resets_by_env and len(resets_by_env[env_idx]) > 0:
                    resets_by_env[env_idx][-1] = True

    plan, replay_plan_path = _load_replay_plan(summary, summary_file)
    n_envs = int(plan.get("source_env_count", len(actions_by_env)))
    if any(env_idx < 0 or env_idx >= n_envs for env_idx in actions_by_env):
        raise RuntimeError(
            f"executed-command env indexes exceed source_env_count={n_envs} in {scene_dir}"
        )
    poses_by_frame = _load_pose_labels(plan, replay_plan_path) if include_pose_labels else None
    sessions = []
    for env_idx, actions in actions_by_env.items():
        if not actions: continue
        sess_actions = np.stack(actions)
        sess_resets = np.array(resets_by_env[env_idx])
        sess_sources = [
            source_by_key.get((env_idx, sid), "unknown")
            for sid in seqids_by_env[env_idx]
        ]
        sessions.append(
            GenesisWMSession(
                scene_dir,
                env_index=env_idx,
                n_envs=n_envs,
                actions=sess_actions,
                resets=sess_resets,
                sources=sess_sources,
                poses_by_frame=poses_by_frame,
            )
        )
    return sessions


class GenesisWMDataset(Dataset):
    """DataLoader for LeWM training from Genesis artifacts."""

    def __init__(
        self,
        root_dir: Path,
        render_root: Optional[Path] = None,
        seq_len: int = 4,
        stride: int = 5,
        split: str = "train",
        max_sessions: Optional[int] = None,
        allow_material_color_render: bool = False,
        holdout_fraction: float = 0.0,
        holdout_role: str = "all",
        holdout_seed: int = 0,
        include_pose_labels: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.render_root = self._resolve_render_root(render_root)
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.allow_material_color_render = allow_material_color_render
        self.holdout_fraction = float(holdout_fraction)
        self.holdout_role = holdout_role
        self.holdout_seed = int(holdout_seed)
        self.include_pose_labels = bool(include_pose_labels)

        self.sessions: List[GenesisWMSession] = []
        self._load_corpus(max_sessions)
        self._apply_holdout_filter()

        self.indices: List[Tuple[int, int]] = []
        self.window_sources: List[str] = []
        for sess_idx, sess in enumerate(self.sessions):
            n_blocks = len(sess.actions)
            for block_start in range(n_blocks - seq_len + 1):
                if not any(sess.resets[block_start : block_start + seq_len]):
                    self.indices.append((sess_idx, block_start))
                    # Tag the window by its anchor block's collector source. A
                    # window spans seq_len blocks; a collector runs for a while,
                    # so anchor-tagging closely approximates the window marginal
                    # while keeping the index light.
                    self.window_sources.append(
                        sess.sources[block_start] if sess.sources is not None else "unknown"
                    )

        logger.info(
            f"Loaded {len(self.sessions)} environment-sessions, "
            f"{len(self.indices)} valid sequences (seq_len={seq_len}, stride={stride})"
        )
        logger.info(
            "Window collector-source mix: %s",
            dict(Counter(self.window_sources).most_common()),
        )

    def _resolve_render_root(self, render_root: Optional[Path]) -> Path:
        if render_root is not None:
            return Path(render_root)
        textured = self.root_dir / "render_textured"
        if textured.exists():
            return textured
        return self.root_dir / "render"

    def _validate_render_root(self) -> None:
        checked = 0
        for summary_file in sorted(self.render_root.glob("*/summary.json")):
            checked += 1
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            _validate_render_summary(
                summary,
                summary_file,
                allow_material_color_render=self.allow_material_color_render,
            )
        if checked == 0:
            raise RuntimeError(f"No rendered scene summaries found under {self.render_root}")

    def _apply_holdout_filter(self) -> None:
        if self.holdout_role == "all" or self.holdout_fraction <= 0.0:
            return
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("--eval-holdout-fraction must be in (0, 1)")
        if self.holdout_role not in {"train", "eval"}:
            raise ValueError(f"unknown holdout role: {self.holdout_role}")

        kept: List[GenesisWMSession] = []
        for sess in self.sessions:
            key = f"{self.holdout_seed}:{sess.scene_dir.name}"
            in_holdout = _stable_unit_interval(key) < self.holdout_fraction
            if (self.holdout_role == "eval" and in_holdout) or (
                self.holdout_role == "train" and not in_holdout
            ):
                kept.append(sess)

        logger.info(
            "Scene holdout role=%s fraction=%.4f kept %d/%d sessions",
            self.holdout_role,
            self.holdout_fraction,
            len(kept),
            len(self.sessions),
        )
        self.sessions = kept

    def _scene_passes_holdout(self, scene_name: str) -> bool:
        """Whether a scene survives the active holdout, decided from its name alone.

        Exact mirror of the per-session test in `_apply_holdout_filter`: every
        session returned by `process_scene_v03(sd, ...)` is constructed with that
        same `sd`, so `sess.scene_dir.name == sd.name` and name-level selection
        keeps exactly the same sessions. Only meaningful when a holdout is active;
        callers guard on that, and `_apply_holdout_filter` still runs afterward as
        the authoritative (now idempotent) filter.
        """
        key = f"{self.holdout_seed}:{scene_name}"
        in_holdout = _stable_unit_interval(key) < self.holdout_fraction
        return (self.holdout_role == "eval" and in_holdout) or (
            self.holdout_role == "train" and not in_holdout
        )

    def _load_corpus(self, max_sessions: Optional[int]):
        render_root = self.render_root

        if not render_root.exists():
            logger.error(f"Render root not found: {render_root}")
            return
        self._validate_render_root()
        logger.info(f"Using render root: {render_root}")

        all_scenes = sorted([d for d in render_root.glob("*") if d.is_dir()])

        if max_sessions is not None:
            for sd in tqdm(all_scenes, desc="Loading sessions"):
                self.sessions.extend(
                    process_scene_v03(
                        sd,
                        self.root_dir,
                        self.split,
                        self.allow_material_color_render,
                        self.include_pose_labels,
                    )
                )
                if len(self.sessions) >= max_sessions:
                    self.sessions = self.sessions[:max_sessions]
                    break
            return

        # When a holdout is active, decide membership from the scene name up
        # front and skip scenes we would discard anyway, instead of parsing the
        # whole corpus and dropping ~all of it in `_apply_holdout_filter`. This
        # selects exactly the same sessions (see `_scene_passes_holdout`); it only
        # avoids the wasted parse. The eval set (max_sessions=None, ~2% holdout)
        # was parsing all ~48k sessions to keep ~2%, single-threaded under the GIL.
        # The train path above (max_sessions set) is intentionally left untouched
        # so its early-break subset stays identical across cells.
        scenes_to_load = all_scenes
        if self.holdout_role in {"train", "eval"} and 0.0 < self.holdout_fraction < 1.0:
            scenes_to_load = [sd for sd in all_scenes if self._scene_passes_holdout(sd.name)]
            logger.info(
                "Holdout role=%s fraction=%.4f pre-filtered scenes to %d/%d before loading",
                self.holdout_role,
                self.holdout_fraction,
                len(scenes_to_load),
                len(all_scenes),
            )

        # Keep corpus parsing in-process: forked Python workers have proved
        # unstable with the ROCm training runtime already imported.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_scene_v03,
                    sd,
                    self.root_dir,
                    self.split,
                    self.allow_material_color_render,
                    self.include_pose_labels,
                )
                for sd in scenes_to_load
            ]
            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(scenes_to_load), desc="Loading sessions")
            for f in pbar:
                res = f.result()
                self.sessions.extend(res)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sess_idx, block_start = self.indices[i]
        sess = self.sessions[sess_idx]
        
        vis_list = []
        for t in range(self.seq_len):
            step = (block_start + t) * self.stride
            img_path = sess.get_rgb_path(step)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            vis_list.append(np.array(img).transpose(2, 0, 1))
        
        vis_seq = torch.from_numpy(np.stack(vis_list)).float() / 255.0
        
        cmd_list = []
        for t in range(self.seq_len):
            cmd_list.append(sess.actions[block_start + t])
        
        cmd_seq = torch.from_numpy(np.stack(cmd_list)).float()
        item = {
            "vis_seq": vis_seq,
            "cmd_seq": cmd_seq,
        }
        if self.include_pose_labels:
            pose_list = [
                sess.get_pose((block_start + t) * self.stride)
                for t in range(self.seq_len)
            ]
            item["pose_seq"] = torch.from_numpy(np.stack(pose_list)).float()
        return item


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


class EpochRandomSampler(Sampler[int]):
    """Deterministic epoch shuffle with a cheap sample offset for batch resume."""

    def __init__(self, data_source: Dataset, *, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.start_sample = 0

    def set_epoch(self, epoch: int, *, start_sample: int = 0) -> None:
        if not 0 <= start_sample <= len(self.data_source):
            raise ValueError(f"start_sample={start_sample} is outside the dataset")
        self.epoch = int(epoch)
        self.start_sample = int(start_sample)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        yield from itertools.islice(indices, self.start_sample, None)

    def __len__(self) -> int:
        return len(self.data_source) - self.start_sample


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    sampler: Optional[Sampler[int]] = None,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vis = batch["vis_seq"].to(device, non_blocking=non_blocking)
    cmd = batch["cmd_seq"].to(device, non_blocking=non_blocking)
    return vis, cmd


def _probe_action_sensitivity(
    model: LeWorldModel,
    vis: torch.Tensor,
    cmd: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    z_raw, z_proj = model.encode_seq(vis, prop_seq=None)
    pred = model.pred_projector.forward_seq(model.predictor(z_raw, cmd))[:, :-1]
    target = z_proj[:, 1:]

    zero_cmd = torch.zeros_like(cmd)
    zero_pred = model.pred_projector.forward_seq(model.predictor(z_raw, zero_cmd))[:, :-1]
    zero_delta = (pred - zero_pred).square().mean()

    if cmd.shape[0] > 1:
        shuffled_cmd = cmd.roll(shifts=1, dims=0)
        shuffled_pred = model.pred_projector.forward_seq(
            model.predictor(z_raw, shuffled_cmd)
        )[:, :-1]
        shuffled_delta = (pred - shuffled_pred).square().mean()
    else:
        shuffled_delta = zero_delta

    rollout_actions = cmd[:, : max(0, cmd.shape[1] - 1)]
    if rollout_actions.shape[1] > 0:
        rollout_pred = model.plan_rollout(z_raw[:, 0], rollout_actions)
        rollout_loss = (rollout_pred - target).square().mean()
    else:
        rollout_loss = zero_delta.new_tensor(float("nan"))

    target_step_delta = (z_proj[:, 1:] - z_proj[:, :-1]).square().mean()
    return {
        "action_zero_delta": zero_delta,
        "action_shuffle_delta": shuffled_delta,
        "rollout_pred_loss": rollout_loss,
        "target_step_delta": target_step_delta,
    }


@torch.no_grad()
def evaluate_model(
    model: LeWorldModel,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
    precision: str,
) -> Dict[str, float]:
    model.eval()
    stats = RunningAverages()
    autocast_enabled = precision == "bf16" and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        vis, cmd = _batch_to_device(batch, device, non_blocking=False)
        weight = int(vis.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(vis_seq=vis, prop_seq=None, cmd_seq=cmd)
            probe = _probe_action_sensitivity(model, vis, cmd)
        stats.update(
            {
                "eval_loss": out["loss"].item(),
                "eval_pred": out["pred_loss"].item(),
                "eval_rollout_loss": out["rollout_loss"].item(),
                "eval_sig": out["sigreg_loss"].item(),
                "eval_std": out["z_proj_std"].item(),
                "eval_rollout_pred": probe["rollout_pred_loss"].item(),
                "eval_action_zero_delta": probe["action_zero_delta"].item(),
                "eval_action_shuffle_delta": probe["action_shuffle_delta"].item(),
                "eval_target_step_delta": probe["target_step_delta"].item(),
            },
            weight=weight,
        )

    model.train()
    return stats.means()


def append_metrics_jsonl(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _num_batches(num_samples: int, batch_size: int, *, drop_last: bool) -> int:
    if drop_last:
        return num_samples // batch_size
    return (num_samples + batch_size - 1) // batch_size


def _checkpoint_sort_key(path: Path, *, max_seq_len: int) -> Optional[Tuple[int, int, int]]:
    match = CHECKPOINT_RE.match(path.name)
    if match is None or int(match.group("seq_len")) != max_seq_len:
        return None
    batch = match.group("batch")
    return (
        int(match.group("epoch")),
        1 if batch is None else 0,
        int(batch or 0),
    )


def _find_latest_checkpoint(directory: Path, *, max_seq_len: int) -> Optional[Path]:
    candidates = []
    for path in directory.glob(f"lewm_seq{max_seq_len}_e*.pt"):
        sort_key = _checkpoint_sort_key(path, max_seq_len=max_seq_len)
        if sort_key is not None:
            candidates.append((sort_key, path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def _save_checkpoint(
    path: Path,
    *,
    model: LeWorldModel,
    optimizer: torch.optim.Optimizer,
    dataset: GenesisWMDataset,
    train_num_samples: int,
    train_sampler_kind: str,
    train_source_mix: Dict[str, int],
    args: argparse.Namespace,
    epoch: int,
    epoch_complete: bool,
    next_batch_idx: int,
    epoch_stats: RunningAverages,
    metrics: Dict[str, float],
    pose_head: Optional[RelPoseHead] = None,
) -> None:
    train_metrics = {f"train_{key}": value for key, value in epoch_stats.means().items()}
    payload = {
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "next_batch_idx": next_batch_idx,
        "epoch_stats": epoch_stats.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "action_metadata": active_block_metadata(),
        "model_config": {
            "max_seq_len": args.max_seq_len,
            "stride": args.stride,
            "cmd_dim": ACTIVE_BLOCK_DIM,
            "sigreg_lambda": args.sigreg_lambda,
            "rollout_lambda": args.rollout_lambda,
            "rollout_horizon": args.rollout_horizon,
            "rollout_gamma": args.rollout_gamma,
            "rollout_warmup_epochs": args.rollout_warmup_epochs,
            "rollout_ss_start": args.rollout_ss_start,
            "rollout_ss_end": args.rollout_ss_end,
            "rollout_ss_ramp_epochs": args.rollout_ss_ramp_epochs,
            "render_root": str(dataset.render_root),
            "allow_material_color_render": bool(args.allow_material_color_render),
            "eval_holdout_fraction": args.eval_holdout_fraction,
            "eval_seed": args.eval_seed,
            "source_allow": args.source_allow,
            "source_cap": args.source_cap,
            "source_weight": args.source_weight,
            "pose_aux_lambda": args.pose_aux_lambda,
            "pose_aux_predicted_lambda": args.pose_aux_predicted_lambda,
            "pose_aux_hidden": args.pose_aux_hidden,
            "pose_label_source": args.pose_label_source,
            "command_dt_s": args.command_dt_s,
            "freeze_model": bool(args.freeze_model),
        },
        "data_loader_config": {
            "sampler": train_sampler_kind,
            "shuffle_seed": args.shuffle_seed,
            "batch_size": args.batch_size,
            "drop_last": bool(args.drop_last),
            "num_samples": int(train_num_samples),
            "full_dataset_num_samples": len(dataset),
            "source_allow": args.source_allow,
            "source_cap": args.source_cap,
            "source_weight": args.source_weight,
            "train_source_mix": dict(train_source_mix),
        },
        "loss": train_metrics["train_loss"],
        "metrics": metrics,
    }
    if pose_head is not None:
        payload["pose_head_state_dict"] = pose_head.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"Saved {path}")


def _validate_resume_config(
    model_config: dict,
    dataset: GenesisWMDataset,
    args: argparse.Namespace,
) -> None:
    required = {"max_seq_len", "stride", "cmd_dim", "render_root"}
    missing = sorted(required - set(model_config))
    if missing:
        raise RuntimeError(
            "Refusing to resume from checkpoint with incomplete model_config "
            f"(missing {missing}). Use --allow-legacy-resume only when you "
            "intentionally know the checkpoint matches this data/model setup."
        )
    if int(model_config["max_seq_len"]) != int(args.max_seq_len):
        raise RuntimeError(
            f"Checkpoint max_seq_len={model_config['max_seq_len']} does not "
            f"match requested max_seq_len={args.max_seq_len}."
        )
    if int(model_config["stride"]) != int(args.stride):
        raise RuntimeError(
            f"Checkpoint stride={model_config['stride']} does not match "
            f"requested stride={args.stride}."
        )
    if int(model_config["cmd_dim"]) != ACTIVE_BLOCK_DIM:
        raise RuntimeError(
            f"Checkpoint cmd_dim={model_config['cmd_dim']} does not match "
            f"active_block dim={ACTIVE_BLOCK_DIM}."
        )
    checkpoint_render_root = Path(str(model_config["render_root"])).resolve()
    current_render_root = dataset.render_root.resolve()
    if checkpoint_render_root != current_render_root:
        raise RuntimeError(
            "Refusing to resume from checkpoint trained on a different render root. "
            f"checkpoint={checkpoint_render_root} current={current_render_root}. "
            "Use a fresh --out-dir for a new render corpus."
        )
    for key in (
        "source_allow",
        "source_cap",
        "source_weight",
        "rollout_lambda",
        "rollout_horizon",
        "rollout_gamma",
        "rollout_warmup_epochs",
        "rollout_ss_start",
        "rollout_ss_end",
        "rollout_ss_ramp_epochs",
        "pose_aux_lambda",
        "pose_aux_predicted_lambda",
        "pose_aux_hidden",
        "pose_label_source",
        "command_dt_s",
        "freeze_model",
    ):
        if key in model_config and model_config[key] != getattr(args, key):
            raise RuntimeError(
                f"Checkpoint model_config[{key!r}]={model_config[key]!r} does not "
                f"match requested value {getattr(args, key)!r}. Use a separate "
                "--out-dir for each ablation cell."
            )


def _validate_partial_resume_config(
    checkpoint: dict,
    train_num_samples: int,
    train_sampler_kind: str,
    args: argparse.Namespace,
) -> None:
    config = checkpoint.get("data_loader_config")
    if not isinstance(config, dict):
        raise RuntimeError("Partial checkpoint has no data_loader_config; cannot resume within an epoch.")
    if train_sampler_kind != "epoch_random":
        raise RuntimeError(
            f"Cannot resume partial checkpoint with sampler={train_sampler_kind!r}; "
            "use epoch-complete checkpoints or rerun the epoch."
        )
    expected = {
        "sampler": "epoch_random",
        "shuffle_seed": args.shuffle_seed,
        "batch_size": args.batch_size,
        "drop_last": bool(args.drop_last),
        "num_samples": int(train_num_samples),
        "source_allow": args.source_allow,
        "source_cap": args.source_cap,
        "source_weight": args.source_weight,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"Partial checkpoint data_loader_config[{key!r}]={config.get(key)!r} "
                f"does not match requested value {value!r}."
            )
    if not isinstance(checkpoint.get("epoch_stats"), dict):
        raise RuntimeError("Partial checkpoint has no epoch_stats; cannot resume epoch metrics.")


def train(args):
    if args.init_from and args.resume:
        raise ValueError("--init-from and --resume are mutually exclusive")
    if args.pose_aux_lambda < 0.0 or args.pose_aux_predicted_lambda < 0.0:
        raise ValueError("pose auxiliary loss weights must be non-negative")
    pose_aux_enabled = args.pose_aux_lambda > 0.0 or args.pose_aux_predicted_lambda > 0.0
    if args.freeze_model and not pose_aux_enabled:
        raise ValueError("--freeze-model requires at least one pose auxiliary loss")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for LeWM training, but torch.cuda.is_available() is false")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    dataset = GenesisWMDataset(
        root_dir=Path(args.data_root),
        render_root=Path(args.render_root) if args.render_root else None,
        seq_len=args.max_seq_len,
        stride=args.stride,
        max_sessions=args.max_sessions,
        allow_material_color_render=args.allow_material_color_render,
        holdout_fraction=args.eval_holdout_fraction,
        holdout_role="train" if args.eval_every_epochs > 0 else "all",
        holdout_seed=args.eval_seed,
        include_pose_labels=(
            pose_aux_enabled and args.pose_label_source == "actual"
        ),
    )
    if len(dataset) == 0:
        logger.error("Dataset is empty. Check data paths and split.")
        return
    logger.info(f"Using active_block action order: {ACTIVE_BLOCK_ORDER}")

    pin_memory = args.pin_memory and device.type == "cuda"

    # Optional source-aware LeWM pretraining marginal (LeJEPA P1 de-risk).
    # filter+cap is the primary, paper-grade mechanism: an exact, interpretable
    # marginal plus an exact size-matched uniform control (run the control with
    # the same --source-cap and no --source-allow). --source-weight is an
    # alternative weighted sampler (disables mid-epoch resume).
    train_view: Dataset = dataset
    weighted_sampler: Optional[WeightedRandomSampler] = None
    keep: List[int] = []
    train_sampler_kind = "epoch_random"
    train_num_samples = len(dataset)
    train_source_mix: Dict[str, int] = dict(Counter(dataset.window_sources))
    if args.source_allow or args.source_cap or args.source_weight:
        allow = {s.strip() for s in args.source_allow.split(",") if s.strip()}
        keep = [
            i for i in range(len(dataset))
            if (not allow) or dataset.window_sources[i] in allow
        ]
        if allow:
            logger.info("Source filter %s -> %d/%d windows", sorted(allow), len(keep), len(dataset))
        if not keep:
            raise RuntimeError(
                f"Source filter {sorted(allow)} produced zero windows; "
                "check --source-allow against the logged collector-source mix."
            )
        if args.source_cap and not args.source_weight and 0 < args.source_cap < len(keep):
            rng = np.random.default_rng(args.shuffle_seed)
            keep = rng.permutation(np.asarray(keep))[: args.source_cap].tolist()
            logger.info("Source cap -> %d windows (seed=%d)", len(keep), args.shuffle_seed)
        if len(keep) != len(dataset):
            train_view = Subset(dataset, keep)
        train_source_mix = dict(Counter(dataset.window_sources[i] for i in keep))
        if args.source_weight:
            if args.save_every_batches > 0:
                raise RuntimeError(
                    "--source-weight is not compatible with --save-every-batches; "
                    "WeightedRandomSampler cannot resume deterministically mid-epoch."
                )
            spec: Dict[str, float] = {}
            for part in args.source_weight.split(","):
                name, _, w = part.partition(":")
                if name.strip():
                    spec[name.strip()] = float(w) if w else 1.0
            if train_view is dataset:
                src_of = lambda j: dataset.window_sources[j]  # noqa: E731
            else:
                src_of = lambda j: dataset.window_sources[keep[j]]  # noqa: E731
            weights = np.array([spec.get(src_of(j), 1.0) for j in range(len(train_view))], dtype=np.float64)
            num_samples = int(args.source_cap) if args.source_cap else len(train_view)
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
                raise RuntimeError(f"Invalid --source-weight spec {spec}; weight sum must be positive.")
            weighted_sampler = WeightedRandomSampler(
                torch.as_tensor(weights), num_samples=num_samples, replacement=True
            )
            logger.info("Weighted sampler spec=%s num_samples=%d (mid-epoch resume disabled)",
                        spec, num_samples)
            train_sampler_kind = "weighted_random"
            train_num_samples = int(num_samples)
        else:
            train_num_samples = len(train_view)
    logger.info(
        "Training view: sampler=%s samples_per_epoch=%d source_mix=%s",
        train_sampler_kind,
        train_num_samples,
        train_source_mix,
    )

    if weighted_sampler is not None:
        train_sampler = weighted_sampler
    else:
        train_sampler = EpochRandomSampler(train_view, seed=args.shuffle_seed)
    loader = make_loader(
        train_view,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        sampler=train_sampler,
    )
    non_blocking_transfer = bool(pin_memory)

    eval_loader = None
    if args.eval_every_epochs > 0:
        eval_dataset = GenesisWMDataset(
            root_dir=Path(args.data_root),
            render_root=Path(args.render_root) if args.render_root else None,
            seq_len=args.max_seq_len,
            stride=args.stride,
            max_sessions=args.eval_max_sessions,
            allow_material_color_render=args.allow_material_color_render,
            holdout_fraction=args.eval_holdout_fraction,
            holdout_role="eval",
            holdout_seed=args.eval_seed,
        )
        if len(eval_dataset) == 0:
            logger.warning("Eval probe dataset is empty; disabling eval probe.")
        else:
            eval_loader = make_loader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.eval_num_workers,
                drop_last=False,
                pin_memory=pin_memory,
                persistent_workers=args.persistent_workers,
                prefetch_factor=args.prefetch_factor,
            )

    model = LeWorldModel(
        max_seq_len=args.max_seq_len,
        cmd_dim=ACTIVE_BLOCK_DIM,
        sigreg_lambda=args.sigreg_lambda,
        rollout_lambda=args.rollout_lambda,
        rollout_horizon=args.rollout_horizon or None,
        rollout_gamma=args.rollout_gamma,
    ).to(device)

    # Fine-tune init: load ONLY model weights (fresh optimizer + epoch 0).
    if args.init_from:
        logger.info("Init-from (weights only): %s", args.init_from)
        ck = torch.load(args.init_from, map_location=device, weights_only=True)
        state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        model.load_state_dict(state)
        source_config = ck.get("model_config", {}) if isinstance(ck, dict) else {}
        changed = []
        for key in (
            "max_seq_len",
            "stride",
            "sigreg_lambda",
            "rollout_lambda",
            "rollout_horizon",
            "rollout_gamma",
            "rollout_warmup_epochs",
            "rollout_ss_start",
            "rollout_ss_end",
            "rollout_ss_ramp_epochs",
        ):
            if key in source_config and source_config[key] != getattr(args, key):
                changed.append(f"{key}: {source_config[key]!r} -> {getattr(args, key)!r}")
        if changed:
            logger.warning(
                "Init-from changes the source training objective/config: %s",
                "; ".join(changed),
            )
    if args.freeze_model:
        model.requires_grad_(False)
        logger.info("Frozen-model control ON: only the RelPoseHead will be optimized")

    # Optional RelPoseHead metric objective (trained jointly; loss backprops into encoder).
    pose_head = None
    if pose_aux_enabled:
        pose_head = RelPoseHead(latent_dim=model.latent_dim, hidden=args.pose_aux_hidden).to(device)
        logger.info(
            "Pose-aux ON: encoded_lambda=%.3g predicted_lambda=%.3g labels=%s latent_dim=%d hidden=%d",
            args.pose_aux_lambda, args.pose_aux_predicted_lambda, args.pose_label_source,
            model.latent_dim, args.pose_aux_hidden,
        )

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if pose_head is not None:
        params.extend(pose_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    start_epoch = 0
    start_batch_idx = 0
    resume_epoch_stats = None
    resume_path = None

    if args.resume:
        if os.path.isfile(args.resume):
            resume_path = Path(args.resume)
        elif os.path.isdir(args.resume):
            resume_path = _find_latest_checkpoint(Path(args.resume), max_seq_len=args.max_seq_len)

    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            checkpoint_model_config = checkpoint.get("model_config")
            if checkpoint_model_config is None:
                if not args.allow_legacy_resume:
                    raise RuntimeError(
                        "Refusing to resume from legacy checkpoint with no "
                        "model_config/render_root metadata. Use a fresh --out-dir "
                        "for new runs, or pass --allow-legacy-resume only for an "
                        "intentional same-corpus continuation."
                    )
                logger.warning(
                    "Legacy checkpoint has no model_config/render_root metadata; "
                    "continuing because --allow-legacy-resume was set."
                )
            else:
                _validate_resume_config(checkpoint_model_config, dataset, args)
            checkpoint_action_metadata = checkpoint.get("action_metadata")
            if checkpoint_action_metadata is None:
                logger.warning(
                    "Checkpoint has no action_metadata; assuming active_block order %s",
                    ACTIVE_BLOCK_ORDER,
                )
            else:
                assert_active_block_metadata_compatible(checkpoint_action_metadata)
            model.load_state_dict(checkpoint["model_state_dict"])
            if pose_head is not None:
                if "pose_head_state_dict" not in checkpoint:
                    raise RuntimeError(
                        "Cannot resume pose-aux training: checkpoint has no pose_head_state_dict"
                    )
                pose_head.load_state_dict(checkpoint["pose_head_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("epoch_complete", True):
                start_epoch = checkpoint["epoch"] + 1
            else:
                _validate_partial_resume_config(
                    checkpoint,
                    train_num_samples=train_num_samples,
                    train_sampler_kind=train_sampler_kind,
                    args=args,
                )
                start_epoch = checkpoint["epoch"]
                start_batch_idx = int(checkpoint["next_batch_idx"])
                resume_epoch_stats = checkpoint["epoch_stats"]
        else:
            # Fallback for old/test checkpoints that saved state_dict directly
            model.load_state_dict(checkpoint)
            # Find epoch from filename: lewm_seq4_e0.pt
            try:
                start_epoch = int(resume_path.stem.split("_e")[-1]) + 1
            except ValueError:
                start_epoch = 0
        logger.info(f"Starting from epoch {start_epoch}, batch {start_batch_idx}")

    metrics_path = Path(args.metrics_jsonl)
    if not metrics_path.is_absolute():
        metrics_path = Path(args.out_dir) / metrics_path

    total_batches = _num_batches(train_num_samples, args.batch_size, drop_last=args.drop_last)
    for epoch in range(start_epoch, args.epochs):
        model.eval() if args.freeze_model else model.train()
        if pose_head is not None:
            pose_head.train()
        if args.rollout_lambda > 0.0 and args.rollout_warmup_epochs > 0:
            effective_rollout_lambda = args.rollout_lambda * min(
                1.0,
                float(epoch + 1) / float(args.rollout_warmup_epochs),
            )
        else:
            effective_rollout_lambda = args.rollout_lambda
        # Scheduled sampling: ramp teacher-forcing prob start->end over ramp epochs.
        if args.rollout_ss_start > 0.0 or args.rollout_ss_end > 0.0:
            if args.rollout_ss_ramp_epochs > 0:
                ss_frac = min(1.0, float(epoch) / float(args.rollout_ss_ramp_epochs))
            else:
                ss_frac = 0.0
            effective_teacher_prob = args.rollout_ss_start + (
                args.rollout_ss_end - args.rollout_ss_start
            ) * ss_frac
        else:
            effective_teacher_prob = 0.0
        epoch_start_batch_idx = start_batch_idx if epoch == start_epoch else 0
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch, start_sample=epoch_start_batch_idx * args.batch_size)
        epoch_stats = (
            RunningAverages.from_state_dict(resume_epoch_stats)
            if epoch == start_epoch and resume_epoch_stats is not None
            else RunningAverages()
        )
        if epoch_start_batch_idx:
            logger.info(f"Resuming epoch {epoch} at batch {epoch_start_batch_idx}/{total_batches}")
        pbar = tqdm(loader, desc=f"Epoch {epoch}", initial=epoch_start_batch_idx, total=total_batches)
        for batch_idx, batch in enumerate(pbar, start=epoch_start_batch_idx):
            vis, cmd = _batch_to_device(
                batch,
                device,
                non_blocking=non_blocking_transfer,
            )
            
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = args.precision == "bf16" and device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                out = model(
                    vis_seq=vis,
                    prop_seq=None,
                    cmd_seq=cmd,
                    rollout_lambda=effective_rollout_lambda,
                    rollout_teacher_prob=effective_teacher_prob,
                    return_latents=(pose_head is not None),
                )
                loss = out["loss"]
                pose_stats = None
                if pose_head is not None:
                    pose_labels = (
                        batch["pose_seq"].to(device, non_blocking=non_blocking_transfer)
                        if args.pose_label_source == "actual" else None
                    )
                    pose_stats = {}
                    if args.pose_aux_lambda > 0.0:
                        pose_loss, encoded_pose_stats = pose_aux_loss(
                            pose_head, out["z_proj"], cmd, args.command_dt_s, poses=pose_labels
                        )
                        loss = loss + args.pose_aux_lambda * pose_loss
                        pose_stats.update(encoded_pose_stats)
                    if args.pose_aux_predicted_lambda > 0.0:
                        pose_pred_loss, pose_pred_stats = predicted_pose_aux_loss(
                            pose_head, model, out["z_raw"], out["z_proj"], cmd,
                            args.command_dt_s, poses=pose_labels,
                        )
                        loss = loss + args.pose_aux_predicted_lambda * pose_pred_loss
                        pose_stats.update(pose_pred_stats)
            loss.backward()
            grad_norm = None
            if args.gradient_clip_val > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params,
                    args.gradient_clip_val,
                )
            optimizer.step()

            weight = int(vis.shape[0])
            batch_metrics = {
                "loss": loss.item(),
                "pred": out["pred_loss"].item(),
                "rollout": out["rollout_loss"].item(),
                "rollout_lambda": out["rollout_lambda"].item(),
                "sig": out["sigreg_loss"].item(),
                "std": out["z_proj_std"].item(),
            }
            if grad_norm is not None:
                batch_metrics["grad_norm"] = float(grad_norm.detach().cpu())
            if pose_stats is not None:
                if "pose_xy_err_m" in pose_stats:
                    batch_metrics["pose_xy_err"] = pose_stats["pose_xy_err_m"]
                    batch_metrics["pose_yaw_err"] = pose_stats["pose_yaw_err_rad"]
                if "pose_pred_xy_err_m" in pose_stats:
                    batch_metrics["pose_pred_xy_err"] = pose_stats["pose_pred_xy_err_m"]
                    batch_metrics["pose_pred_yaw_err"] = pose_stats["pose_pred_yaw_err_rad"]
            epoch_stats.update(batch_metrics, weight=weight)
            epoch_means = epoch_stats.means()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pred": f"{out['pred_loss'].item():.4f}",
                "avg_pred": f"{epoch_means['pred']:.4f}",
                "rollout": f"{out['rollout_loss'].item():.4f}",
                "rlam": f"{effective_rollout_lambda:.3g}",
                "ss_tf": f"{effective_teacher_prob:.2g}",
                "sig": f"{out['sigreg_loss'].item():.4f}",
                "std": f"{out['z_proj_std'].item():.4f}"
            })

            next_batch_idx = batch_idx + 1
            if (
                args.save_every_batches > 0
                and next_batch_idx % args.save_every_batches == 0
                and next_batch_idx < total_batches
            ):
                ckpt_path = (
                    Path(args.out_dir)
                    / f"lewm_seq{args.max_seq_len}_e{epoch}_b{next_batch_idx:06d}.pt"
                )
                train_metrics = {
                    f"train_{key}": value for key, value in epoch_stats.means().items()
                }
                _save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    dataset=dataset,
                    train_num_samples=train_num_samples,
                    train_sampler_kind=train_sampler_kind,
                    train_source_mix=train_source_mix,
                    args=args,
                    epoch=epoch,
                    epoch_complete=False,
                    next_batch_idx=next_batch_idx,
                    epoch_stats=epoch_stats,
                    metrics=train_metrics,
                    pose_head=pose_head,
                )
                append_metrics_jsonl(
                    metrics_path,
                    {
                        "epoch": epoch,
                        "epoch_complete": False,
                        "next_batch_idx": next_batch_idx,
                        "checkpoint": str(ckpt_path),
                        "seq_len": args.max_seq_len,
                        "render_root": str(dataset.render_root),
                        "train_sampler": train_sampler_kind,
                        "train_num_samples": train_num_samples,
                        "source_allow": args.source_allow,
                        "source_cap": args.source_cap,
                        "source_weight": args.source_weight,
                        "rollout_lambda": args.rollout_lambda,
                        "rollout_horizon": args.rollout_horizon,
                        "rollout_gamma": args.rollout_gamma,
                        "rollout_warmup_epochs": args.rollout_warmup_epochs,
                        "pose_aux_lambda": args.pose_aux_lambda,
                        "pose_aux_predicted_lambda": args.pose_aux_predicted_lambda,
                        "pose_label_source": args.pose_label_source,
                        "freeze_model": bool(args.freeze_model),
                        **train_metrics,
                    },
                )

        train_metrics = {f"train_{key}": value for key, value in epoch_stats.means().items()}
        eval_metrics: Dict[str, float] = {}
        if eval_loader is not None and (epoch + 1) % args.eval_every_epochs == 0:
            eval_metrics = evaluate_model(
                model,
                eval_loader,
                device,
                max_batches=args.eval_max_batches,
                precision=args.precision,
            )
        log_metrics = {**train_metrics, **eval_metrics}
        logger.info(
            "Epoch %d summary: %s",
            epoch,
            " ".join(f"{key}={value:.4f}" for key, value in sorted(log_metrics.items())),
        )

        ckpt_path = Path(args.out_dir) / f"lewm_seq{args.max_seq_len}_e{epoch}.pt"
        _save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            train_num_samples=train_num_samples,
            train_sampler_kind=train_sampler_kind,
            train_source_mix=train_source_mix,
            args=args,
            epoch=epoch,
            epoch_complete=True,
            next_batch_idx=0,
            epoch_stats=epoch_stats,
            metrics=log_metrics,
            pose_head=pose_head,
        )
        if pose_head is not None:
            pose_path = Path(args.out_dir) / f"posehead_seq{args.max_seq_len}_e{epoch}.pt"
            pose_tmp_path = pose_path.with_suffix(pose_path.suffix + ".tmp")
            torch.save(
                {
                    "head_state_dict": pose_head.state_dict(),
                    "latent_dim": int(model.latent_dim),
                    "hidden": int(args.pose_aux_hidden),
                    "command_dt_s": float(args.command_dt_s),
                    "pose_aux_lambda": float(args.pose_aux_lambda),
                    "pose_aux_predicted_lambda": float(args.pose_aux_predicted_lambda),
                    "pose_label_source": str(args.pose_label_source),
                    "freeze_model": bool(args.freeze_model),
                    "epoch": int(epoch),
                    "source_checkpoint": str(ckpt_path),
                },
                pose_tmp_path,
            )
            os.replace(pose_tmp_path, pose_path)
        append_metrics_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "epoch_complete": True,
                "next_batch_idx": 0,
                "checkpoint": str(ckpt_path),
                "seq_len": args.max_seq_len,
                "render_root": str(dataset.render_root),
                "train_sampler": train_sampler_kind,
                "train_num_samples": train_num_samples,
                "source_allow": args.source_allow,
                "source_cap": args.source_cap,
                "source_weight": args.source_weight,
                "rollout_lambda": args.rollout_lambda,
                "rollout_horizon": args.rollout_horizon,
                "rollout_gamma": args.rollout_gamma,
                "rollout_warmup_epochs": args.rollout_warmup_epochs,
                "pose_aux_lambda": args.pose_aux_lambda,
                "pose_aux_predicted_lambda": args.pose_aux_predicted_lambda,
                "pose_label_source": args.pose_label_source,
                "freeze_model": bool(args.freeze_model),
                **log_metrics,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--render-root",
        type=str,
        default=None,
        help="Rendered-vision root. Defaults to <data-root>/render_textured when present, else <data-root>/render.",
    )
    parser.add_argument(
        "--allow-material-color-render",
        action="store_true",
        help="Allow training on v03/material_color renders. Intended only for diagnostic baselines.",
    )
    parser.add_argument("--out-dir", type=str, default="models/checkpoints")
    parser.add_argument("--max-seq-len", type=int, default=4, help="Sequence length ablation")
    parser.add_argument("--stride", type=int, default=5, help="Temporal stride")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--sigreg-lambda", type=float, default=0.09)
    parser.add_argument("--rollout-lambda", type=float, default=0.0)
    parser.add_argument("--rollout-horizon", type=int, default=0, help="0 = all available transitions")
    parser.add_argument("--rollout-gamma", type=float, default=0.9)
    parser.add_argument("--rollout-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--rollout-ss-start",
        type=float,
        default=0.0,
        help="Scheduled-sampling teacher-forcing prob at epoch 0 "
        "(0 = off / pure free-running rollout).",
    )
    parser.add_argument(
        "--rollout-ss-end",
        type=float,
        default=0.0,
        help="Scheduled-sampling teacher-forcing prob after the ramp.",
    )
    parser.add_argument(
        "--rollout-ss-ramp-epochs",
        type=int,
        default=0,
        help="Epochs to linearly ramp teacher prob start->end "
        "(0 = constant at --rollout-ss-start).",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--source-allow",
        type=str,
        default="",
        help=(
            "Comma list of command_source values to KEEP for LeWM training "
            "(filter mode), e.g. 'ou_noise,primitive_curriculum'. Empty = all "
            "sources (current behaviour)."
        ),
    )
    parser.add_argument(
        "--source-cap",
        type=int,
        default=0,
        help=(
            "Randomly cap training windows to this many after filtering (0 = no "
            "cap). Use the SAME value across arms for a size-matched comparison."
        ),
    )
    parser.add_argument(
        "--source-weight",
        type=str,
        default="",
        help=(
            "Weighted-sampler spec 'src:w,...' (alternative to filter); unlisted "
            "sources get weight 1.0. Disables mid-epoch resume."
        ),
    )
    parser.add_argument(
        "--save-every-batches",
        type=int,
        default=0,
        help="Save a resumable intra-epoch checkpoint every N completed batches. Disabled when 0.",
    )
    parser.add_argument("--metrics-jsonl", type=str, default="metrics.jsonl")
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.02)
    parser.add_argument("--eval-seed", type=int, default=20260524)
    parser.add_argument("--eval-max-sessions", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=32)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="Resume a checkpoint lacking model_config/render_root metadata. Use only for known same-corpus continuations.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file or directory to resume from")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Load ONLY model weights from this checkpoint (fresh optimizer + epoch 0). "
        "For fine-tuning a new objective from a trained model, e.g. the pose-aux run from e3.",
    )
    parser.add_argument(
        "--pose-aux-lambda",
        type=float,
        default=0.0,
        help="Weight of the RelPoseHead metric objective (0 = off). Backprops relative-pose "
        "decoding into the encoder so the latent becomes metric for planning.",
    )
    parser.add_argument("--pose-aux-hidden", type=int, default=512)
    parser.add_argument(
        "--pose-aux-predicted-lambda",
        type=float,
        default=0.0,
        help="Weight for deployment-aligned pose loss on predictor endpoint -> encoded final goal.",
    )
    parser.add_argument(
        "--pose-label-source",
        choices=("actual", "command"),
        default="actual",
        help="Pose-aux labels: aligned physical replay poses (primary) or command integration (ablation).",
    )
    parser.add_argument(
        "--command-dt-s", type=float, default=0.10,
        help="Tick dt for cmd-integrated pose targets (matches the kinematic nav benchmark).",
    )
    parser.add_argument(
        "--freeze-model",
        action="store_true",
        help="Train only the RelPoseHead with the world model held in eval mode; "
        "used for the frozen-head decodability ceiling.",
    )
    args = parser.parse_args()
    
    train(args)
