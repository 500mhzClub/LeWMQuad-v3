"""V3 matched-branch counterfactual groups, with the registered FOV correction.

The V3 collection renders 224x224 at 78.323 deg **vertical** FOV; the WP-E
corpus renders 224x168 at 62.837 deg vertical FOV and 78.323 deg horizontal.
``native_preprocess`` resizes to 112x112 without preserving aspect ratio, so
passing both through one encoder unmodified places identical world content on
different tokens.

    tan(62.837/2) / tan(78.323/2) = 0.750000
    224 * 0.750000                = 168.000 px

A centre vertical crop of every V3 frame from 224 to 168 px, applied **before**
the 112x112 resize, reproduces the WP-E vertical FOV to four decimals.  The
horizontal FOV and the camera mount ``(0.326, 0.0, 0.043)`` already match.  This
crop is mandatory and is applied by :func:`preprocess_v3_frame_v1`, which is the
only supported entry point for reading a V3 image.

Documented in ``docs/lewm_go2_wp_f_mechanism_settlement_2026-08-06.md``.
"""

from __future__ import annotations

import collections
import hashlib
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

# Registered FOV correction.  Do not change without re-deriving the crop.
V3_NATIVE_HEIGHT_PX = 224
WPE_NATIVE_HEIGHT_PX = 168
V3_CENTRE_CROP_RATIO = WPE_NATIVE_HEIGHT_PX / V3_NATIVE_HEIGHT_PX  # exactly 0.75
ENCODER_INPUT_PX = 112

# Registered development-selection split.  One scene per family, whole scene and
# whole branch group, carved only from the V3 *train* role.
SPLIT_SEED_V1 = "wp_f_v3_selection_split_20260806"

ACTION_NAMES_V1 = (
    "arc_left", "arc_right", "backward", "forward_fast", "forward_medium",
    "forward_slow", "hold", "yaw_left", "yaw_right",
)
BRANCHES_PER_GROUP = 9

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_v3_frame_v1(path) -> torch.Tensor:
    """Centre-crop to the WP-E vertical FOV, then the training runner's path."""

    with Image.open(path) as decoded:
        image = decoded.convert("RGB")
        width, height = image.size
        cropped_height = int(round(height * V3_CENTRE_CROP_RATIO))
        top = (height - cropped_height) // 2
        image = image.crop((0, top, width, top + cropped_height))
        image = image.resize(
            (ENCODER_INPUT_PX, ENCODER_INPUT_PX), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


class BranchGroupV1(NamedTuple):
    """One counterfactual group: a shared current state and nine actions."""

    state_id: str
    scene_id: str
    family: str
    current_path: str
    successor_paths: tuple[str, ...]      # ordered by action_id 0..8
    action_ids: tuple[int, ...]
    commands: torch.Tensor                # (9, 3) mean body-frame command per block


def _mean_command(requested_block) -> np.ndarray:
    return np.asarray(requested_block, dtype=np.float32).mean(axis=0)


def load_branch_groups_v1(role, ceiling_module, ledger):
    """Load V3 groups for ``role`` in a fixed, action-id-ordered layout."""

    groups, records = ceiling_module.load_role_v1(role, ledger=ledger)
    loaded = []
    for group in groups:
        record = records[group.state_id]
        branches = sorted(group.branches, key=lambda b: int(b.action_id))
        raw = sorted(record["branches"], key=lambda b: int(b["action_id"]))
        action_ids = tuple(int(b.action_id) for b in branches)
        if action_ids != tuple(range(BRANCHES_PER_GROUP)):
            raise RuntimeError(f"{group.state_id}: expected actions 0..8, got {action_ids}")
        loaded.append(
            BranchGroupV1(
                state_id=group.state_id,
                scene_id=group.scene_id,
                family=group.family,
                current_path=str(
                    ceiling_module.rgb_path_v1(role, group.context_rgb_artifact_ids[-1])
                ),
                successor_paths=tuple(
                    str(ceiling_module.rgb_path_v1(role, b.target_rgb_artifact_id))
                    for b in branches
                ),
                action_ids=action_ids,
                commands=torch.from_numpy(
                    np.stack([_mean_command(b["requested_block"]) for b in raw])
                ),
            )
        )
    return loaded


def selection_scene_ids_v1(groups, *, seed: str = SPLIT_SEED_V1) -> set[str]:
    """One scene per family into development-selection, deterministically."""

    by_family = collections.defaultdict(set)
    for group in groups:
        by_family[group.family].add(group.scene_id)
    chosen = set()
    for family in sorted(by_family):
        ranked = sorted(
            sorted(by_family[family]),
            key=lambda s: hashlib.sha256(f"{seed}|{family}|{s}".encode()).hexdigest(),
        )
        chosen.add(ranked[0])
    return chosen


def split_branch_groups_v1(groups, *, seed: str = SPLIT_SEED_V1):
    """Partition by whole scene and whole group; every family lands on both sides."""

    selection_scenes = selection_scene_ids_v1(groups, seed=seed)
    train = [g for g in groups if g.scene_id not in selection_scenes]
    selection = [g for g in groups if g.scene_id in selection_scenes]
    if not train or not selection:
        raise RuntimeError("counterfactual split produced an empty side")
    if {g.family for g in train} != {g.family for g in selection}:
        raise RuntimeError("counterfactual split lost family coverage")
    if {g.scene_id for g in train} & {g.scene_id for g in selection}:
        raise RuntimeError("counterfactual split leaked a scene across roles")
    return train, selection


def action_one_hot_v1(action_ids) -> torch.Tensor:
    out = torch.zeros(len(action_ids), BRANCHES_PER_GROUP)
    for row, action in enumerate(action_ids):
        out[row, int(action)] = 1.0
    return out


__all__ = [
    "ACTION_NAMES_V1",
    "BRANCHES_PER_GROUP",
    "BranchGroupV1",
    "ENCODER_INPUT_PX",
    "SPLIT_SEED_V1",
    "V3_CENTRE_CROP_RATIO",
    "action_one_hot_v1",
    "load_branch_groups_v1",
    "preprocess_v3_frame_v1",
    "selection_scene_ids_v1",
    "split_branch_groups_v1",
]
