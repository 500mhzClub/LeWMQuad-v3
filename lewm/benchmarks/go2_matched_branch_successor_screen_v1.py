"""Custody-safe data and frozen-feature helpers for the successor screen.

This module builds metadata-only training indices.  It does not read RGB
artifacts itself; callers must use the already reviewed bound RGB reader for
the artifact IDs returned here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import importlib.machinery
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType
from typing import Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import (
    materialize_go2_world_model_bounded_branch_posthoc_join_admission_v1
    as posthoc,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
POSTHOC_ROOT = (
    REPO_ROOT / ".generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1"
)
POSTHOC_TERMINAL_REVIEW_PATH = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_"
    "terminal_review_2026-08-02.json"
)
POSTHOC_MANIFEST_BINDING = MappingProxyType(
    {
        "path": str(POSTHOC_ROOT / "manifest.json"),
        "file_sha256": (
            "87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e"
        ),
        "byte_count": 11_964,
    }
)
POSTHOC_TERMINAL_BINDING = MappingProxyType(
    {
        "path": str(POSTHOC_ROOT / "terminal.json"),
        "file_sha256": (
            "a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56"
        ),
        "byte_count": 1_250,
    }
)
POSTHOC_TERMINAL_REVIEW_BINDING = MappingProxyType(
    {
        "path": str(POSTHOC_TERMINAL_REVIEW_PATH),
        "file_sha256": (
            "bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669"
        ),
        "byte_count": 2_844,
    }
)

TRAIN_STATE_COUNT = 128
CONTEXT_FRAME_COUNT = 3
HISTORY_ACTION_COUNT = 2
ACTION_COUNT = 9
ARTIFACTS_PER_STATE = CONTEXT_FRAME_COUNT + ACTION_COUNT
TRAIN_ARTIFACT_COUNT = TRAIN_STATE_COUNT * ARTIFACTS_PER_STATE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_IMAGE_SIZE = 224
VJEPA_RESIZE_SHORT_SIDE = 438
VJEPA_CROP_SIZE = 384


class MatchedBranchSuccessorScreenError(RuntimeError):
    """Raised before role-confused or malformed data can enter the screen."""


@dataclass(frozen=True)
class CandidateInputV1:
    """The complete predictor input for one requested successor action."""

    context_rgb_artifact_ids: tuple[str, str, str]
    history_action_ids: tuple[int, int]
    requested_action_id: int


@dataclass(frozen=True)
class TrainStateIndexV1:
    screen_state_index: int
    state_id: str
    family: str
    scene_id: str
    group_index: int
    state_index_in_scene: int
    context_artifact_indices: tuple[int, int, int]
    target_artifact_indices: tuple[int, ...]
    target_rgb_artifact_ids: tuple[str, ...]
    candidate_inputs: tuple[CandidateInputV1, ...]


@dataclass(frozen=True)
class TrainFeaturePlanV1:
    artifact_ids: tuple[str, ...]
    artifact_index_by_id: Mapping[str, int]
    states: tuple[TrainStateIndexV1, ...]


def load_bound_posthoc_bundle_v1() -> object:
    """Load the one preregistered split-root bundle without opening RGB leaves."""

    return posthoc.load_posthoc_bundle_v1(
        POSTHOC_ROOT,
        expected_manifest_byte_count=int(POSTHOC_MANIFEST_BINDING["byte_count"]),
        expected_manifest_sha256=str(POSTHOC_MANIFEST_BINDING["file_sha256"]),
        expected_terminal_byte_count=int(POSTHOC_TERMINAL_BINDING["byte_count"]),
        expected_terminal_sha256=str(POSTHOC_TERMINAL_BINDING["file_sha256"]),
        terminal_review_path=POSTHOC_TERMINAL_REVIEW_PATH,
        expected_terminal_review_byte_count=int(
            POSTHOC_TERMINAL_REVIEW_BINDING["byte_count"]
        ),
        expected_terminal_review_sha256=str(
            POSTHOC_TERMINAL_REVIEW_BINDING["file_sha256"]
        ),
    )


def _artifact_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MatchedBranchSuccessorScreenError(f"{label} is not a nonempty string")
    return value


def _metadata_artifact_ids(groups: Sequence[object], *, role: str) -> set[str]:
    result: set[str] = set()
    for group in groups:
        if getattr(group, "role", None) != role:
            raise MatchedBranchSuccessorScreenError(f"{role} group role changed")
        contexts = tuple(getattr(group, "context_rgb_artifact_ids", ()))
        branches = tuple(getattr(group, "branches", ()))
        if len(contexts) != CONTEXT_FRAME_COUNT or len(branches) != ACTION_COUNT:
            raise MatchedBranchSuccessorScreenError(f"{role} group geometry changed")
        result.update(
            _artifact_id(value, label=f"{role} context artifact") for value in contexts
        )
        result.update(
            _artifact_id(
                getattr(branch, "target_rgb_artifact_id", None),
                label=f"{role} target artifact",
            )
            for branch in branches
        )
    return result


def collect_train_feature_plan_v1(bundle: object) -> TrainFeaturePlanV1:
    """Collect exactly the train-role feature order and candidate inputs.

    Only context artifact IDs, two historical requested action IDs, and the
    candidate requested action ID are copied into ``CandidateInputV1``.  In
    particular, this function never accesses historical or future executed
    command tapes, physical labels, endpoint state, ranks, or target geometry.
    """

    groups_by_role = getattr(bundle, "groups_by_role", None)
    if not isinstance(groups_by_role, Mapping) or not {"train", "eval"}.issubset(
        groups_by_role
    ):
        raise MatchedBranchSuccessorScreenError("bundle train/eval roles are absent")
    train_groups = tuple(groups_by_role["train"])
    eval_groups = tuple(groups_by_role["eval"])
    if len(train_groups) != TRAIN_STATE_COUNT or len(eval_groups) != TRAIN_STATE_COUNT:
        raise MatchedBranchSuccessorScreenError("bundle role state counts changed")
    access_audit = getattr(bundle, "access_audit", None)
    if not isinstance(access_audit, Mapping) or access_audit.get(
        "rgb_leaf_open_count"
    ) != 0:
        raise MatchedBranchSuccessorScreenError(
            "bundle was not loaded at the metadata-only access boundary"
        )

    eval_artifact_ids = _metadata_artifact_ids(eval_groups, role="eval")
    try:
        ordered_groups = sorted(
            train_groups,
            key=lambda group: (
                int(getattr(group, "group_index")),
                str(getattr(group, "state_id")),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise MatchedBranchSuccessorScreenError(
            "train group ordering metadata is malformed"
        ) from exc

    artifacts = getattr(bundle, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        raise MatchedBranchSuccessorScreenError("bundle artifact manifest is absent")
    ordered_artifacts: list[str] = []
    artifact_index_by_id: dict[str, int] = {}
    states: list[TrainStateIndexV1] = []
    seen_state_ids: set[str] = set()
    seen_group_indices: set[int] = set()

    def append_artifact(artifact_id: str) -> int:
        if artifact_id in eval_artifact_ids:
            raise MatchedBranchSuccessorScreenError(
                "train feature plan names an eval-role artifact"
            )
        if artifact_id not in artifacts:
            raise MatchedBranchSuccessorScreenError(
                "train feature plan names an unbound artifact"
            )
        if artifact_id in artifact_index_by_id:
            raise MatchedBranchSuccessorScreenError(
                "train RGB artifact is reused across state slots"
            )
        index = len(ordered_artifacts)
        ordered_artifacts.append(artifact_id)
        artifact_index_by_id[artifact_id] = index
        return index

    for screen_state_index, group in enumerate(ordered_groups):
        if getattr(group, "role", None) != "train":
            raise MatchedBranchSuccessorScreenError("non-train group entered train plan")
        state_id = _artifact_id(getattr(group, "state_id", None), label="state ID")
        try:
            group_index = int(getattr(group, "group_index"))
            state_index_in_scene = int(getattr(group, "state_index_in_scene"))
        except (TypeError, ValueError) as exc:
            raise MatchedBranchSuccessorScreenError(
                "train state indices are malformed"
            ) from exc
        if (
            state_id in seen_state_ids
            or group_index in seen_group_indices
            or group_index < 0
            or state_index_in_scene < 0
        ):
            raise MatchedBranchSuccessorScreenError("train state identity repeats")
        seen_state_ids.add(state_id)
        seen_group_indices.add(group_index)

        context_ids_raw = tuple(getattr(group, "context_rgb_artifact_ids", ()))
        history_raw = tuple(getattr(group, "history_action_ids", ()))
        branches_raw = tuple(getattr(group, "branches", ()))
        if (
            len(context_ids_raw) != CONTEXT_FRAME_COUNT
            or len(history_raw) != HISTORY_ACTION_COUNT
            or len(branches_raw) != ACTION_COUNT
        ):
            raise MatchedBranchSuccessorScreenError("train group geometry changed")
        context_ids = tuple(
            _artifact_id(value, label="train context artifact")
            for value in context_ids_raw
        )
        history_actions = tuple(history_raw)
        if any(type(value) is not int or not 0 <= value < ACTION_COUNT for value in history_actions):
            raise MatchedBranchSuccessorScreenError(
                "historical requested action ID is invalid"
            )

        try:
            branches = sorted(branches_raw, key=lambda branch: int(branch.action_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise MatchedBranchSuccessorScreenError(
                "candidate requested action ID is malformed"
            ) from exc
        requested_action_ids = tuple(getattr(branch, "action_id", None) for branch in branches)
        if requested_action_ids != tuple(range(ACTION_COUNT)):
            raise MatchedBranchSuccessorScreenError(
                "candidate requested actions are not the exact nine-way grid"
            )
        target_ids = tuple(
            _artifact_id(
                getattr(branch, "target_rgb_artifact_id", None),
                label="train target artifact",
            )
            for branch in branches
        )
        context_indices = tuple(append_artifact(value) for value in context_ids)
        target_indices = tuple(append_artifact(value) for value in target_ids)
        candidate_inputs = tuple(
            CandidateInputV1(
                context_rgb_artifact_ids=context_ids,  # type: ignore[arg-type]
                history_action_ids=history_actions,  # type: ignore[arg-type]
                requested_action_id=action_id,
            )
            for action_id in requested_action_ids
        )
        states.append(
            TrainStateIndexV1(
                screen_state_index=screen_state_index,
                state_id=state_id,
                family=_artifact_id(getattr(group, "family", None), label="family"),
                scene_id=_artifact_id(
                    getattr(group, "scene_id", None), label="scene ID"
                ),
                group_index=group_index,
                state_index_in_scene=state_index_in_scene,
                context_artifact_indices=context_indices,  # type: ignore[arg-type]
                target_artifact_indices=target_indices,
                target_rgb_artifact_ids=target_ids,
                candidate_inputs=candidate_inputs,
            )
        )

    if len(ordered_artifacts) != TRAIN_ARTIFACT_COUNT:
        raise MatchedBranchSuccessorScreenError("train artifact count changed")
    return TrainFeaturePlanV1(
        artifact_ids=tuple(ordered_artifacts),
        artifact_index_by_id=MappingProxyType(artifact_index_by_id),
        states=tuple(states),
    )


def _decode_exact_rgb_png_v1(raw: bytes) -> Image.Image:
    if not isinstance(raw, bytes) or not raw:
        raise MatchedBranchSuccessorScreenError("RGB payload must be nonempty bytes")
    try:
        with Image.open(BytesIO(raw)) as probe:
            if (
                probe.format != "PNG"
                or probe.mode != "RGB"
                or probe.size != (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)
                or getattr(probe, "n_frames", 1) != 1
            ):
                raise MatchedBranchSuccessorScreenError(
                    "RGB leaf must be one exact 224x224 RGB PNG"
                )
            probe.verify()
        with Image.open(BytesIO(raw)) as decoded:
            decoded.load()
            return decoded.copy()
    except MatchedBranchSuccessorScreenError:
        raise
    except Exception as exc:
        raise MatchedBranchSuccessorScreenError("RGB PNG cannot be decoded") from exc


def _normalized_chw_v1(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (image.height, image.width, 3):
        raise MatchedBranchSuccessorScreenError("decoded RGB raster shape changed")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float32).div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor.sub_(mean).div_(std)


def preprocess_dinov2_png_bytes_v1(raw: bytes) -> torch.Tensor:
    """Return the exact 224-pixel ImageNet-normalized DINO input ``[3,H,W]``."""

    result = _normalized_chw_v1(_decode_exact_rgb_png_v1(raw))
    if result.shape != (3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE):
        raise MatchedBranchSuccessorScreenError("DINO preprocessing shape changed")
    return result


def preprocess_vjepa2_1_png_bytes_v1(raw: bytes) -> torch.Tensor:
    """Return official square-image V-JEPA 2.1 geometry as ``[3,1,384,384]``."""

    image = _decode_exact_rgb_png_v1(raw)
    image = image.resize(
        (VJEPA_RESIZE_SHORT_SIDE, VJEPA_RESIZE_SHORT_SIDE),
        resample=Image.Resampling.BILINEAR,
    )
    offset = (VJEPA_RESIZE_SHORT_SIDE - VJEPA_CROP_SIZE) // 2
    image = image.crop(
        (offset, offset, offset + VJEPA_CROP_SIZE, offset + VJEPA_CROP_SIZE)
    )
    result = _normalized_chw_v1(image).unsqueeze(1)
    if result.shape != (3, 1, VJEPA_CROP_SIZE, VJEPA_CROP_SIZE):
        raise MatchedBranchSuccessorScreenError("V-JEPA preprocessing shape changed")
    return result


def normalize_dense_token_grid_v1(tokens: torch.Tensor) -> torch.Tensor:
    """Convert a supported frozen encoder grid to normalized ``[B,256,D]``."""

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] < 1:
        raise ValueError("tokens must be a nonempty rank-three tensor")
    if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("tokens must use a floating dtype")
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError("tokens contain a nonfinite value")
    batch, token_count, feature_dim = tokens.shape
    values = tokens.to(torch.float32)
    if (token_count, feature_dim) == (256, 384):
        converted = values
    elif (token_count, feature_dim) == (576, 768):
        grid = values.transpose(1, 2).reshape(batch, feature_dim, 24, 24)
        converted = F.interpolate(grid, size=(16, 16), mode="area")
        converted = converted.flatten(2).transpose(1, 2)
    else:
        raise ValueError(
            "tokens must be DINO [B,256,384] or V-JEPA [B,576,768]"
        )
    norms = torch.linalg.vector_norm(converted, dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0.0).any()):
        raise FloatingPointError("token grid contains a zero or nonfinite vector")
    result = F.normalize(converted, p=2.0, dim=-1)
    if result.shape != (batch, 256, feature_dim) or not bool(
        torch.isfinite(result).all()
    ):
        raise FloatingPointError("normalized token grid is invalid")
    return result.contiguous()


def drop_path_compat_v1(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """The timm per-sample stochastic-depth formula, without importing timm."""

    probability = float(drop_prob)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("drop_prob must be in [0,1]")
    if probability == 0.0 or not training:
        return x
    keep_prob = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPathCompatV1(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_compat_v1(
            x,
            self.drop_prob,
            self.training,
            self.scale_by_keep,
        )


def _package_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = []  # type: ignore[attr-defined]
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, is_package=True
    )
    return module


@contextmanager
def scoped_timm_drop_path_shim_v1() -> Iterator[None]:
    """Temporarily provide only the timm symbol imported by V-JEPA 2.1.

    The official encoder imports ``drop_path`` from ``timm.models.layers``.
    The local timm import otherwise reaches an unavailable torchvision NMS
    extension.  Every prior module entry is restored on exit.
    """

    names = ("timm", "timm.models", "timm.models.layers")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    timm_module = _package_module("timm")
    models_module = _package_module("timm.models")
    layers_module = _package_module("timm.models.layers")
    layers_module.drop_path = drop_path_compat_v1  # type: ignore[attr-defined]
    layers_module.DropPath = DropPathCompatV1  # type: ignore[attr-defined]
    timm_module.models = models_module  # type: ignore[attr-defined]
    models_module.layers = layers_module  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "timm": timm_module,
            "timm.models": models_module,
            "timm.models.layers": layers_module,
        }
    )
    try:
        yield
    finally:
        for name in reversed(names):
            prior = previous[name]
            if prior is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior  # type: ignore[assignment]


__all__ = [
    "ACTION_COUNT",
    "CandidateInputV1",
    "DINO_IMAGE_SIZE",
    "DropPathCompatV1",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MatchedBranchSuccessorScreenError",
    "POSTHOC_MANIFEST_BINDING",
    "POSTHOC_ROOT",
    "POSTHOC_TERMINAL_BINDING",
    "POSTHOC_TERMINAL_REVIEW_BINDING",
    "POSTHOC_TERMINAL_REVIEW_PATH",
    "TRAIN_ARTIFACT_COUNT",
    "TRAIN_STATE_COUNT",
    "TrainFeaturePlanV1",
    "TrainStateIndexV1",
    "VJEPA_CROP_SIZE",
    "VJEPA_RESIZE_SHORT_SIDE",
    "collect_train_feature_plan_v1",
    "drop_path_compat_v1",
    "load_bound_posthoc_bundle_v1",
    "normalize_dense_token_grid_v1",
    "preprocess_dinov2_png_bytes_v1",
    "preprocess_vjepa2_1_png_bytes_v1",
    "scoped_timm_drop_path_shim_v1",
]
