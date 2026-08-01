#!/usr/bin/env python3
"""Execute the authorized existing-pool three-arm world-model experiment.

This worker is deliberately not a launcher.  It accepts only an already-bound
one-shot authority from the external supervisor, consumes only the corrected
development H6 train/validation indices, builds one fresh visible-frame pack
inside the reserved attempt, and writes immutable receipts.  It never opens a
sealed or held-out role and it never grants retry, resume, promotion, or G2--G8
authority.

The three matched arms differ at exactly the candidate-action contribution:

* ``conditioned`` uses the factual candidate action;
* ``blind`` keeps factual history actions but zeros the candidate action
  embedding (during both training and evaluation);
* ``shuffled`` uses a deterministic within-family candidate-action
  derangement during training and the factual candidate action at evaluation.

The predecessor online encoder and its hard-synchronised target encoder are
shared, frozen, and excluded from every optimizer.  All arms therefore see the
same encoded frames, targets, masks, row order, initialization, and schedule.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as temporal_metrics,
)
from lewm.benchmarks import (  # noqa: E402
    go2_world_model_existing_pool_three_arm_v1 as experiment_metrics,
)
from lewm.datasets import (  # noqa: E402
    go2_explicit_plan_discounted_successor_state_v27 as h6,
)
from lewm.models import (  # noqa: E402
    rgb_recurrent_patch_memory_temporal_jepa_v1 as temporal_model,
)
from lewm.models.rgb_single_frame_multiblock_masked_spatial_jepa_v1 import (  # noqa: E402
    _gather_spatial_tokens,
    normalized_half_squared_jepa_loss_v1,
    normalized_half_squared_token_energy_v1,
)
from scripts import dev_pack_h6_temporal_frames as packer  # noqa: E402
from scripts import dev_train_temporal_jepa_scaled as scaled  # noqa: E402
from scripts import (  # noqa: E402
    run_go2_world_model_existing_pool_three_arm_authorized_v1 as supervisor_contract,
)


SCHEMA_PREFIX = "lewm_go2_world_model_existing_pool_three_arm"
AUTHORITY_SCHEMA = supervisor_contract.AUTHORITY_SCHEMA
RESULT_SCHEMA = supervisor_contract.RESULT_SCHEMA
MEASUREMENT_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_snapshot_v1"
OVERLAP_AUDIT_SCHEMA = f"{SCHEMA_PREFIX}_overlap_audit_v1"
SHUFFLE_AUDIT_SCHEMA = f"{SCHEMA_PREFIX}_candidate_action_derangement_v1"
FAILURE_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v2_"
    "worker_failure_v1"
)

ARM_NAMES = ("conditioned", "blind", "shuffled")
PACK_ARTIFACT_RELATIVE_PATHS = {
    "train": {
        "frames": "pack/train_frames.u8",
        "actions": "pack/train_actions.npy",
        "metadata": "pack/train_meta.json",
    },
    "val": {
        "frames": "pack/val_frames.u8",
        "actions": "pack/val_actions.npy",
        "metadata": "pack/val_meta.json",
    },
}
OBSERVATION_UPDATES = tuple(range(0, 701, 100))
TRAINING_UPDATES = 700
BATCH_SIZE = 256
MICROBATCH_SIZE = 32
LR_SCALE = 4.0
WARMUP_UPDATES = 150
COSINE_SCHEDULE_UPDATES = 3_000
PREDICTOR_BASE_LR = 1.0e-4
MEMORY_BASE_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP = 1.0
TRAINING_SEED = 20_260_731
TRAIN_ORDER_NAMESPACE = (
    f"{SCHEMA_PREFIX}_v1/train-order/seed-{TRAINING_SEED}"
)
MAXIMUM_WALL_SECONDS = 43_200
MAXIMUM_GPU_SECONDS = 36_000
EVALUATION_BATCH_SIZE = 64
PREDECESSOR = (
    REPO_ROOT
    / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
    / "attempt_v1/snapshots/update_1000.pt"
)
PREDECESSOR_BYTE_COUNT = 52_282_877
PREDECESSOR_SHA256 = (
    "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873"
)
PREDECESSOR_UPDATE = 1000
EXPECTED_TRAIN_ROWS = 16_000
EXPECTED_VALIDATION_ROWS = 2_048
ACTION_COUNT = 9
EXPECTED_PYTORCH = "2.9.1+rocm7.2.1.gitff65f5bc"
EXPECTED_HIP = "7.2.53211-e1a6bc5663"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_PILLOW = "11.3.0"

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PREDICTOR_EXACT = frozenset(("predictor_position", "predictor_mask_token"))
_PREDICTOR_PREFIXES = (
    "predictor_blocks.",
    "predictor_norm.",
    "predictor_output.",
)
_MEMORY_PREFIXES = (
    "action_embedding.",
    "time_embedding.",
    "temporal_gru.",
)

REQUIRED_SOURCE_PATHS = {
    "lewm_package": Path("lewm/__init__.py"),
    "benchmarks_package": Path("lewm/benchmarks/__init__.py"),
    "counterfactual_metrics": Path("lewm/benchmarks/counterfactual.py"),
    "datasets_package": Path("lewm/datasets/__init__.py"),
    "models_package": Path("lewm/models/__init__.py"),
    "base_world_model": Path("lewm/models/lewm.py"),
    "phase2d_spatial_model": Path("lewm/models/phase2d_spatial_lewm.py"),
    "base_predictor": Path("lewm/models/predictor.py"),
    "primitive_affordance": Path("lewm/models/primitive_affordance.py"),
    "sigreg": Path("lewm/models/sigreg.py"),
    "source_action_utility": Path("lewm/models/source_action_utility.py"),
    "spatial_lewm": Path("lewm/models/spatial_lewm.py"),
    "spatial_predictor": Path("lewm/models/spatial_predictor.py"),
    "worker": Path("scripts/execute_go2_world_model_existing_pool_three_arm_v1.py"),
    "checker": Path("scripts/check_go2_world_model_existing_pool_three_arm_v1.py"),
    "external_supervisor": Path(
        "scripts/run_go2_world_model_existing_pool_three_arm_authorized_v1.py"
    ),
    "experiment_metrics": Path(
        "lewm/benchmarks/go2_world_model_existing_pool_three_arm_v1.py"
    ),
    "temporal_metrics": Path(
        "lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "h6_dataset": Path(
        "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py"
    ),
    "h6_main_pool_census": Path(
        "lewm/benchmarks/go2_recurrent_jepa_main_pool_census.py"
    ),
    "h6_sequence_contract_v2": Path(
        "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py"
    ),
    "h6_sequence_contract_v1": Path(
        "lewm/datasets/go2_recurrent_h4_rgb_sequences.py"
    ),
    "temporal_model": Path(
        "lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "spatial_model": Path(
        "lewm/models/rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "encoders": Path("lewm/models/encoders.py"),
    "temporal_training_core": Path(
        "scripts/run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "temporal_evaluator": Path(
        "scripts/evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "spatial_evaluator": Path(
        "scripts/evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "spatial_metrics": Path(
        "lewm/benchmarks/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "place_data": Path("lewm/datasets/go2_memory_role_place_triplets_v1.py"),
    "packer": Path("scripts/dev_pack_h6_temporal_frames.py"),
    "scaled_runtime": Path("scripts/dev_train_temporal_jepa_scaled.py"),
}


class ThreeArmWorkerError(RuntimeError):
    """An authority, custody, model, or experiment invariant failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite canonical JSON for stable receipt hashes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ThreeArmWorkerError("value is not canonical finite JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _reject_protected_path(path: Path, *, label: str) -> None:
    for component in Path(path).parts:
        lowered = component.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith("heldout_")
            or lowered.startswith("held_out_")
            or lowered.startswith("held-out-")
        ):
            raise ThreeArmWorkerError(f"{label} names custody-protected material")


def _hash_regular_file(path: Path) -> tuple[str, int]:
    """Hash one stable regular file and return its digest and opened size."""

    selected = Path(path)
    _reject_protected_path(selected, label="file binding")
    if selected.is_symlink() or not selected.is_file():
        raise ThreeArmWorkerError(f"not a regular non-symlink file: {selected}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    before_path = selected.stat(follow_symlinks=False)
    descriptor = os.open(selected, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ThreeArmWorkerError(f"not a regular file: {selected}")
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = selected.stat(follow_symlinks=False)
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    if before_fingerprint != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ) or before_fingerprint != (
        before_path.st_dev,
        before_path.st_ino,
        before_path.st_mode,
        before_path.st_size,
        before_path.st_mtime_ns,
    ) or before_fingerprint != (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_mode,
        after_path.st_size,
        after_path.st_mtime_ns,
    ):
        raise ThreeArmWorkerError(f"file changed while hashing: {selected}")
    return digest.hexdigest(), int(before.st_size)


def sha256_file(path: Path) -> str:
    return _hash_regular_file(path)[0]


def file_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ThreeArmWorkerError(f"bound file is absent or unsafe: {selected}")
    resolved = selected.resolve()
    reported = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    digest, byte_count = _hash_regular_file(selected)
    return {
        "path": reported,
        "file_sha256": digest,
        "byte_count": byte_count,
    }


def _binding_matches(path: Path, binding: Mapping[str, Any], *, label: str) -> None:
    if set(binding) != {"path", "file_sha256", "byte_count"}:
        raise ThreeArmWorkerError(f"{label} binding keys changed")
    if not _is_sha256(binding.get("file_sha256")):
        raise ThreeArmWorkerError(f"{label} lacks a lowercase SHA-256")
    if type(binding.get("byte_count")) is not int or binding["byte_count"] < 1:
        raise ThreeArmWorkerError(f"{label} byte count is invalid")
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ThreeArmWorkerError(f"{label} is not a regular non-symlink file")
    if selected.stat().st_size != binding["byte_count"]:
        raise ThreeArmWorkerError(f"{label} byte count changed")
    if sha256_file(selected) != binding["file_sha256"]:
        raise ThreeArmWorkerError(f"{label} SHA-256 changed")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ThreeArmWorkerError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_bound_json(
    path: Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    selected = Path(path)
    _reject_protected_path(selected, label=label)
    if type(expected_byte_count) is not int or not 1 <= expected_byte_count <= 128 * 1024 * 1024:
        raise ThreeArmWorkerError(f"{label} byte count is outside the JSON ceiling")
    if not _is_sha256(expected_sha256):
        raise ThreeArmWorkerError(f"{label} expected SHA-256 is invalid")
    if selected.is_symlink() or not selected.is_file():
        raise ThreeArmWorkerError(f"{label} is not a regular non-symlink file")
    if selected.stat().st_size != expected_byte_count:
        raise ThreeArmWorkerError(f"{label} byte count changed")
    raw = selected.read_bytes()
    if len(raw) != expected_byte_count or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ThreeArmWorkerError(f"{label} bytes disagree with the caller binding")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ThreeArmWorkerError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThreeArmWorkerError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ThreeArmWorkerError(f"{label} must be a JSON object")
    return value


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    attempt_root: Path,
) -> dict[str, Any]:
    selected = Path(path)
    temporary = selected.with_name(selected.name + ".partial")
    if selected.exists() or selected.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable JSON: {selected}")
    raw = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.link(temporary, selected)
    temporary.unlink()
    return file_binding(selected, relative_to=attempt_root)


def save_immutable_snapshot(path: Path, payload: Mapping[str, Any], *, attempt_root: Path) -> dict[str, Any]:
    selected = Path(path)
    temporary = selected.with_name(selected.name + ".partial")
    if selected.exists() or selected.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable snapshot: {selected}")
    with temporary.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, selected)
    temporary.unlink()
    return file_binding(selected, relative_to=attempt_root)


def tensor_inventory_sha256(values: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and exact CPU bytes canonically."""

    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        header = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        digest.update(canonical_json_bytes(header))
        digest.update(b"\n")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    return tensor_inventory_sha256(module.state_dict())


class ArmCore(nn.Module):
    """The independently trainable predictor/memory part of Temporal V1."""

    def __init__(
        self,
        template: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(template.config)
        self.predictor_position = nn.Parameter(template.predictor_position.detach().clone())
        self.predictor_mask_token = nn.Parameter(template.predictor_mask_token.detach().clone())
        self.predictor_blocks = copy.deepcopy(template.predictor_blocks)
        self.predictor_norm = copy.deepcopy(template.predictor_norm)
        self.predictor_output = copy.deepcopy(template.predictor_output)
        self.action_embedding = copy.deepcopy(template.action_embedding)
        self.time_embedding = copy.deepcopy(template.time_embedding)
        self.temporal_gru = copy.deepcopy(template.temporal_gru)
        # The shared template is deliberately frozen before arm allocation.
        # Deepcopy preserves that flag, so the independently allocated arm
        # inventory must explicitly re-enter the trainable state.
        self.requires_grad_(True)


@dataclass(frozen=True)
class ArmPartition:
    predictor: tuple[nn.Parameter, ...]
    memory: tuple[nn.Parameter, ...]
    predictor_names: tuple[str, ...]
    memory_names: tuple[str, ...]

    @property
    def all(self) -> tuple[nn.Parameter, ...]:
        return self.predictor + self.memory


@dataclass(frozen=True)
class ArmPrediction:
    raw: torch.Tensor
    normalized: torch.Tensor
    recurrent_memory: torch.Tensor


def partition_arm_parameters(arm: ArmCore) -> ArmPartition:
    predictor: list[nn.Parameter] = []
    memory: list[nn.Parameter] = []
    predictor_names: list[str] = []
    memory_names: list[str] = []
    for name, parameter in arm.named_parameters():
        if name in _PREDICTOR_EXACT or name.startswith(_PREDICTOR_PREFIXES):
            predictor.append(parameter)
            predictor_names.append(name)
        elif name.startswith(_MEMORY_PREFIXES):
            memory.append(parameter)
            memory_names.append(name)
        else:
            raise ThreeArmWorkerError(f"unregistered arm parameter {name!r}")
    if not predictor or not memory:
        raise ThreeArmWorkerError("arm parameter partition contains an empty role")
    all_parameters = tuple(arm.parameters())
    partitioned = tuple(predictor + memory)
    if (
        len({id(value) for value in partitioned}) != len(partitioned)
        or {id(value) for value in partitioned} != {id(value) for value in all_parameters}
        or any(not value.requires_grad or value.dtype != torch.float32 for value in partitioned)
    ):
        raise ThreeArmWorkerError("arm parameter partition is invalid")
    return ArmPartition(
        predictor=tuple(predictor),
        memory=tuple(memory),
        predictor_names=tuple(predictor_names),
        memory_names=tuple(memory_names),
    )


def build_arm_optimizer(arm: ArmCore) -> tuple[torch.optim.AdamW, ArmPartition]:
    partition = partition_arm_parameters(arm)
    optimizer = torch.optim.AdamW(
        [
            {
                "group_name": "predictor",
                "params": list(partition.predictor),
                "lr": PREDICTOR_BASE_LR * LR_SCALE,
            },
            {
                "group_name": "memory",
                "params": list(partition.memory),
                "lr": MEMORY_BASE_LR * LR_SCALE,
            },
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
    )
    observed = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(observed) != len(set(observed)) or set(observed) != {id(value) for value in partition.all}:
        raise ThreeArmWorkerError("arm optimizer membership is invalid")
    return optimizer, partition


def predict_from_shared_encoding(
    arm: ArmCore,
    encoded_history: torch.Tensor,
    actions: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    candidate_blind: bool,
    time_indices: torch.Tensor | None = None,
) -> ArmPrediction:
    """Run the exact Temporal-V1 head, optionally gating its final action."""

    if encoded_history.ndim != 4:
        raise ValueError("encoded_history must have shape (B,S,256,192)")
    batch, steps, tokens, features = encoded_history.shape
    if (
        batch < 1
        or not 1 <= steps <= 3
        or tokens != arm.config.spatial_token_count
        or features != arm.config.feature_dim
        or encoded_history.dtype != torch.float32
        or not bool(torch.isfinite(encoded_history).all())
    ):
        raise ValueError("encoded_history shape or dtype changed")
    if actions.shape != (batch, steps) or actions.dtype != torch.long:
        raise TypeError("actions must be long with shape (B,S)")
    if bool((actions < 0).any()) or bool((actions >= arm.config.action_count).any()):
        raise ValueError("action IDs left the canonical vocabulary")
    if target_indices.shape != (batch, arm.config.target_token_count) or target_indices.dtype != torch.long:
        raise TypeError("target_indices must be long with shape (B,64)")
    device = encoded_history.device
    if actions.device != device or target_indices.device != device:
        raise TypeError("arm inputs must share one device")
    if (
        bool((target_indices < 0).any())
        or bool((target_indices >= arm.config.spatial_token_count).any())
        or not bool((target_indices[:, 1:] > target_indices[:, :-1]).all())
    ):
        raise ValueError("target indices must be strictly increasing and in range")
    if arm.predictor_mask_token.device != device:
        raise TypeError("encoded history and arm must share one device")
    if time_indices is None:
        times = torch.arange(steps, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)
    else:
        times = time_indices
        if times.ndim == 1 and times.shape == (steps,):
            times = times.unsqueeze(0).expand(batch, -1)
        if times.shape != (batch, steps) or times.dtype != torch.long or times.device != device:
            raise TypeError("time_indices must be long with shape (B,S) on the input device")
    if bool((times < 0).any()) or bool((times >= arm.config.time_embedding_count).any()):
        raise ValueError("time indices left the registered range")

    action_conditioning = arm.action_embedding(actions)
    if candidate_blind:
        gate = torch.ones((1, steps, 1), dtype=action_conditioning.dtype, device=device)
        gate[:, -1] = 0.0
        action_conditioning = action_conditioning * gate
    conditioning = action_conditioning + arm.time_embedding(times)
    recurrent_input = encoded_history + conditioning.unsqueeze(2)
    streams = recurrent_input.permute(0, 2, 1, 3).reshape(
        batch * arm.config.spatial_token_count,
        steps,
        arm.config.feature_dim,
    )
    initial_hidden = torch.zeros(
        1,
        batch * arm.config.spatial_token_count,
        arm.config.temporal_hidden_dim,
        dtype=streams.dtype,
        device=device,
    )
    recurrent_streams, _ = arm.temporal_gru(streams, initial_hidden)
    recurrent_memory = recurrent_streams[:, -1].reshape(
        batch,
        arm.config.spatial_token_count,
        arm.config.temporal_hidden_dim,
    )
    memory_tokens = recurrent_memory + arm.predictor_position.unsqueeze(0)
    query_positions = _gather_spatial_tokens(
        arm.predictor_position.unsqueeze(0).expand(batch, -1, -1),
        target_indices,
    )
    queries = arm.predictor_mask_token.expand(
        batch,
        arm.config.target_token_count,
        -1,
    ) + query_positions
    predictor = torch.cat((memory_tokens, queries), dim=1)
    if predictor.shape[1] != arm.config.decoder_token_count:
        raise ThreeArmWorkerError("temporal decoder token count changed")
    for block in arm.predictor_blocks:
        predictor = block(predictor)
    predicted_queries = arm.predictor_norm(
        predictor[:, -arm.config.target_token_count :]
    )
    raw = arm.predictor_output(predicted_queries)
    normalized = F.normalize(
        raw,
        p=2.0,
        dim=-1,
        eps=arm.config.normalization_epsilon,
    )
    return ArmPrediction(raw=raw, normalized=normalized, recurrent_memory=recurrent_memory)


def build_bound_training_schedule(
    *,
    row_count: int = EXPECTED_TRAIN_ROWS,
    updates: int = TRAINING_UPDATES,
    batch_size: int = BATCH_SIZE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a hash-ordered, version-independent, tail-carrying schedule."""

    if type(row_count) is not int or type(updates) is not int or type(batch_size) is not int:
        raise TypeError("schedule dimensions must be integers")
    if row_count < 2 or updates < 1 or not 1 <= batch_size <= row_count:
        raise ValueError("schedule dimensions are invalid")
    required = updates * batch_size
    sequence: list[int] = []
    epoch = 0
    epoch_digests: list[str] = []
    while len(sequence) < required:
        ordered = sorted(
            range(row_count),
            key=lambda row: (
                hashlib.sha256(
                    f"{TRAIN_ORDER_NAMESPACE}/{epoch}/{row}".encode("ascii")
                ).digest(),
                row,
            ),
        )
        epoch_digests.append(canonical_sha256(ordered))
        sequence.extend(ordered)
        epoch += 1
    selected = sequence[:required]
    digest = hashlib.sha256()
    for row in selected:
        digest.update(struct.pack(">I", row))
    tensor = torch.tensor(selected, dtype=torch.long).reshape(updates, batch_size)
    return tensor, {
        "seed": TRAINING_SEED,
        "namespace": TRAIN_ORDER_NAMESPACE,
        "algorithm": "per_epoch_sha256_sort_then_contiguous_tail_carry_v1",
        "row_count": row_count,
        "updates": updates,
        "batch_size": batch_size,
        "presentations": required,
        "epochs_touched": epoch,
        "epoch_order_sha256": epoch_digests,
        "ordered_uint32be_sha256": digest.hexdigest(),
    }


def learning_rate_fraction(update: int) -> float:
    return scaled.learning_rate_fraction(
        update,
        warmup_updates=WARMUP_UPDATES,
        schedule_updates=COSINE_SCHEDULE_UPDATES,
    )


def _to_temporal_metrics_rows(rows: Sequence[h6.H6V2Row]) -> tuple[Any, ...]:
    return tuple(
        temporal_metrics.MetadataRow(
            index=int(row.index),
            role=str(row.role),
            family=str(row.family),
            scene_id=str(row.scene_id),
            rgb=tuple(row.rgb),
            actions=tuple(int(value) for value in row.actions),
        )
        for row in rows
    )


def _candidate_action_tensor(
    factual_actions: torch.Tensor,
    *,
    arm_name: str,
    row_indices: torch.Tensor,
    shuffled_candidate_actions: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    result = factual_actions.clone()
    if arm_name == "shuffled" and training:
        result[:, 2] = shuffled_candidate_actions[row_indices]
    if arm_name not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm_name!r}")
    return result


def _arm_is_candidate_blind(arm_name: str) -> bool:
    if arm_name not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm_name!r}")
    return arm_name == "blind"


def _encode_context_and_future(
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    normalized_frames: torch.Tensor,
    target_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalized_frames.ndim != 5 or tuple(normalized_frames.shape[1:]) != (4, 3, 112, 112):
        raise ValueError("normalized packed frames must have shape (B,4,3,112,112)")
    batch = normalized_frames.shape[0]
    with torch.no_grad():
        encoded = substrate.encode_online_full_frame(
            normalized_frames[:, :3].reshape(batch * 3, 3, 112, 112)
        ).reshape(batch, 3, 256, 192)
        target = substrate.encode_target(
            normalized_frames[:, 3], target_indices
        ).raw_target_tokens
    return encoded.detach(), target.detach()


def _energy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return normalized_half_squared_token_energy_v1(prediction, target).detach().to(
        device="cpu", dtype=torch.float64
    )


@dataclass
class EvaluationVectors:
    role: str
    row_indices: tuple[int, ...]
    factual_energy: dict[str, torch.Tensor]
    persistence_energy: torch.Tensor | None
    wrong_history_energy: dict[str, torch.Tensor]
    candidate_energy: dict[str, torch.Tensor]
    prediction_tokens: dict[str, torch.Tensor]
    target_tokens: torch.Tensor | None
    blind_candidate_max_spread: float | None


@torch.no_grad()
def evaluate_panel(
    *,
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    arms: Mapping[str, ArmCore],
    frames: torch.Tensor,
    actions: torch.Tensor,
    role: str,
    row_indices: Sequence[int],
    wrong_history_donors: Sequence[int] | None,
    include_controls: bool,
    include_nine_way: bool,
    include_rank_tokens: bool,
) -> EvaluationVectors:
    """Evaluate matched factual rows using one shared frozen encoding route."""

    selected = tuple(int(value) for value in row_indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("evaluation row indices must be unique and non-empty")
    if include_controls and (wrong_history_donors is None or len(wrong_history_donors) != frames.shape[0]):
        raise ValueError("full-role wrong-history donors are required for controls")
    if include_controls and selected != tuple(range(frames.shape[0])):
        raise ValueError("control evaluation must use the complete ordered role")
    for arm in arms.values():
        arm.eval()
    factual: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    wrong_history: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    candidates: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    prediction_tokens: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
    persistence: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    maximum_blind_spread = 0.0

    # The complete validation panel is encoded once, then cached on-device.
    # This lets persistence gather the current online tokens (the target and
    # online encoders are bit-identical and frozen) and lets wrong-history use
    # donor x0/x1 encodings without any additional RGB/frame encoding.  The
    # accounting therefore remains exactly four shared frame encodings per row.
    encoded_cache: torch.Tensor | None = None
    target_cache: torch.Tensor | None = None
    if include_controls:
        encoded_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for cache_start in range(0, len(selected), EVALUATION_BATCH_SIZE):
            cache_indices = selected[cache_start : cache_start + EVALUATION_BATCH_SIZE]
            cache_rows = torch.tensor(
                cache_indices, dtype=torch.long, device=frames.device
            )
            cache_normalized = scaled.to_float(frames[cache_rows])
            cache_targets, _ = temporal_metrics.batched_mask_indices(
                role, cache_indices, device=frames.device
            )
            cache_encoded, cache_target = _encode_context_and_future(
                substrate, cache_normalized, cache_targets
            )
            encoded_parts.append(cache_encoded)
            target_parts.append(cache_target)
        encoded_cache = torch.cat(encoded_parts, dim=0)
        target_cache = torch.cat(target_parts, dim=0)

    for start in range(0, len(selected), EVALUATION_BATCH_SIZE):
        batch_indices = selected[start : start + EVALUATION_BATCH_SIZE]
        row_cpu = torch.tensor(batch_indices, dtype=torch.long)
        row_device = row_cpu.to(frames.device)
        target_indices, _ = temporal_metrics.batched_mask_indices(
            role, batch_indices, device=frames.device
        )
        if include_controls:
            assert encoded_cache is not None and target_cache is not None
            encoded = encoded_cache[start : start + len(batch_indices)]
            target = target_cache[start : start + len(batch_indices)]
        else:
            normalized = scaled.to_float(frames[row_device])
            encoded, target = _encode_context_and_future(
                substrate, normalized, target_indices
            )
        factual_actions = actions[row_device]
        for arm_name in ARM_NAMES:
            prediction = predict_from_shared_encoding(
                arms[arm_name],
                encoded,
                factual_actions,
                target_indices,
                candidate_blind=_arm_is_candidate_blind(arm_name),
            )
            factual[arm_name].append(_energy(prediction.raw, target))
            if include_rank_tokens:
                prediction_tokens[arm_name].append(prediction.normalized.to(device="cpu"))

        if include_rank_tokens:
            targets.append(F.normalize(target, p=2.0, dim=-1, eps=1.0e-8).to(device="cpu"))

        if include_controls:
            current_target = _gather_spatial_tokens(encoded[:, 2], target_indices)
            persistence.append(_energy(current_target, target))
            donor_ids = torch.tensor(
                [int(wrong_history_donors[index]) for index in batch_indices],
                dtype=torch.long,
                device=frames.device,
            )
            assert encoded_cache is not None
            donor_encoded = encoded_cache[donor_ids, :2]
            wrong_encoded = torch.cat((donor_encoded, encoded[:, 2:3]), dim=1)
            wrong_actions = torch.cat((actions[donor_ids, :2], factual_actions[:, 2:3]), dim=1)
            for arm_name in ARM_NAMES:
                history_prediction = predict_from_shared_encoding(
                    arms[arm_name],
                    wrong_encoded,
                    wrong_actions,
                    target_indices,
                    candidate_blind=_arm_is_candidate_blind(arm_name),
                )
                wrong_history[arm_name].append(_energy(history_prediction.raw, target))

        if include_nine_way:
            per_arm: dict[str, list[torch.Tensor]] = {name: [] for name in ARM_NAMES}
            for candidate_id in range(ACTION_COUNT):
                intervention = factual_actions.clone()
                intervention[:, 2] = candidate_id
                for arm_name in ARM_NAMES:
                    prediction = predict_from_shared_encoding(
                        arms[arm_name],
                        encoded,
                        intervention,
                        target_indices,
                        candidate_blind=_arm_is_candidate_blind(arm_name),
                    )
                    per_arm[arm_name].append(_energy(prediction.raw, target))
            for arm_name in ARM_NAMES:
                grid = torch.stack(per_arm[arm_name], dim=1)
                candidates[arm_name].append(grid)
                if arm_name == "blind":
                    maximum_blind_spread = max(
                        maximum_blind_spread,
                        float((grid.max(dim=1).values - grid.min(dim=1).values).max()),
                    )

    for arm in arms.values():
        arm.train()
    return EvaluationVectors(
        role=role,
        row_indices=selected,
        factual_energy={name: torch.cat(factual[name]) for name in ARM_NAMES},
        persistence_energy=torch.cat(persistence) if persistence else None,
        wrong_history_energy={name: torch.cat(wrong_history[name]) for name in ARM_NAMES if wrong_history[name]},
        candidate_energy={name: torch.cat(candidates[name]) for name in ARM_NAMES if candidates[name]},
        prediction_tokens={name: torch.cat(prediction_tokens[name]) for name in ARM_NAMES if prediction_tokens[name]},
        target_tokens=torch.cat(targets) if targets else None,
        blind_candidate_max_spread=maximum_blind_spread if include_nine_way else None,
    )


def _resolve_binding_path(binding: Mapping[str, Any]) -> Path:
    value = binding.get("path")
    if type(value) is not str or not value:
        raise ThreeArmWorkerError("binding path is absent")
    selected = Path(value)
    return selected if selected.is_absolute() else REPO_ROOT / selected


def _validate_required_source_closure(
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = sorted(set(REQUIRED_SOURCE_PATHS) - set(source_bindings))
    if missing:
        raise ThreeArmWorkerError(f"authority source closure is incomplete: {missing}")
    for name, relative in REQUIRED_SOURCE_PATHS.items():
        binding = source_bindings[name]
        observed = _resolve_binding_path(binding).resolve(strict=True)
        expected = (REPO_ROOT / relative).resolve(strict=True)
        if observed != expected:
            raise ThreeArmWorkerError(f"authority binds a different source for {name}")
        _binding_matches(expected, binding, label=f"source {name}")


def _validate_runtime_identity(authority: Mapping[str, Any]) -> None:
    runtime = authority["runtime"]
    if dict(os.environ) != runtime["environment"]:
        raise ThreeArmWorkerError("effective worker environment differs from authority")
    invocation = Path(runtime["python_invocation_path"])
    if invocation.resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise ThreeArmWorkerError("worker Python differs from the authorized runtime")
    if torch.__version__ != EXPECTED_PYTORCH:
        raise ThreeArmWorkerError(f"unexpected PyTorch runtime: {torch.__version__}")
    if torch.version.hip != EXPECTED_HIP:
        raise ThreeArmWorkerError(f"unexpected HIP runtime: {torch.version.hip}")
    if np.__version__ != EXPECTED_NUMPY:
        raise ThreeArmWorkerError(f"unexpected NumPy runtime: {np.__version__}")
    if packer.PIL.__version__ != EXPECTED_PILLOW:
        raise ThreeArmWorkerError(f"unexpected Pillow runtime: {packer.PIL.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ThreeArmWorkerError("authority requires exactly one visible ROCm device")


def _validate_input_bindings(authority: Mapping[str, Any]) -> None:
    inputs = authority["input_bindings"]
    expected = {
        "predecessor_checkpoint": PREDECESSOR,
        "train_index": REPO_ROOT / h6.TRAIN_INDEX,
        "validation_index": REPO_ROOT / h6.VALIDATION_INDEX,
        "index_manifest": REPO_ROOT / h6.INDEX_ROOT / "manifest.json",
    }
    if set(inputs) != set(expected):
        raise ThreeArmWorkerError(
            "authority input bindings must contain exactly predecessor and corrected H6 indices"
        )
    for name, path in expected.items():
        binding = inputs[name]
        if _resolve_binding_path(binding).resolve(strict=True) != path.resolve(strict=True):
            raise ThreeArmWorkerError(f"authority input path changed for {name}")
        _binding_matches(path, binding, label=f"input {name}")
    predecessor = inputs["predecessor_checkpoint"]
    if (
        predecessor["byte_count"] != PREDECESSOR_BYTE_COUNT
        or predecessor["file_sha256"] != PREDECESSOR_SHA256
    ):
        raise ThreeArmWorkerError("predecessor binding changed")
    for role, name in (("train", "train_index"), ("val", "validation_index")):
        registered = h6.INDEX_BINDINGS[role]
        if (
            inputs[name]["byte_count"] != registered.byte_count
            or inputs[name]["file_sha256"] != registered.sha256
        ):
            raise ThreeArmWorkerError(f"corrected H6 {role} binding changed")


def _load_and_validate_reservation(
    *,
    authority_path: Path,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_root = Path(authority["output_root"])
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ThreeArmWorkerError("external supervisor did not reserve the attempt root")
    entries = sorted(path.name for path in attempt_root.iterdir())
    if entries != ["reservation.json"]:
        raise ThreeArmWorkerError(
            f"reserved attempt was not pristine at worker entry: {entries}"
        )
    reservation_path = attempt_root / "reservation.json"
    reservation_binding = supervisor_contract.file_binding(reservation_path)
    reservation = read_bound_json(
        reservation_path,
        expected_byte_count=reservation_binding["byte_count"],
        expected_sha256=reservation_binding["file_sha256"],
        label="attempt reservation",
    )
    nonce = reservation.get("supervisor_nonce")
    expected_worker_command = [
        authority["runtime"]["python_invocation_path"],
        str((REPO_ROOT / REQUIRED_SOURCE_PATHS["worker"]).resolve()),
        "--authority",
        str(authority_path.resolve()),
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
        "--expected-authority-sha256",
        str(authority_binding["file_sha256"]),
    ]
    expected_checker_template = [
        authority["runtime"]["python_invocation_path"],
        str((REPO_ROOT / REQUIRED_SOURCE_PATHS["checker"]).resolve()),
        "--manifest",
        str((attempt_root / "result.json").resolve()),
        "--expected-file-sha256",
        "<WORKER_RESULT_SHA256>",
        "--expected-byte-count",
        "<WORKER_RESULT_BYTE_COUNT>",
        "--output",
        str((attempt_root / "receipt_check.json").resolve()),
    ]
    expected_pairs = {
        "schema": supervisor_contract.RESERVATION_SCHEMA,
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "authority_binding": dict(authority_binding),
        "plan_binding": dict(plan_binding),
        "review_binding": authority["review_binding"],
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "preregistration_binding": authority["preregistration_binding"],
        "source_bindings": authority["source_bindings"],
        "runtime": authority["runtime"],
        "input_bindings": authority["input_bindings"],
        "predecessor_terminal_failure_binding": authority[
            "predecessor_terminal_failure_binding"
        ],
        "attempt": authority["attempt"],
        "caps": authority["caps"],
        "worker_binding": sources["worker"],
        "checker_binding": sources["checker"],
        "output_root": authority["output_root"],
        "execution": authority["execution"],
        "worker_command": expected_worker_command,
        "checker_command_template": expected_checker_template,
        "authorized_device_idle_preflight_passed": True,
        "maximum_attempts": 1,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    if (
        type(nonce) is not str
        or len(nonce) != 64
        or any(character not in _SHA256_CHARACTERS for character in nonce)
    ):
        raise ThreeArmWorkerError("reservation supervisor nonce is invalid")
    for key, expected in expected_pairs.items():
        if reservation.get(key) != expected:
            raise ThreeArmWorkerError(f"reservation field changed: {key}")
    if set(reservation) != set(expected_pairs) | {"supervisor_nonce"}:
        raise ThreeArmWorkerError("reservation keys changed")
    if supervisor_contract.file_binding(reservation_path) != reservation_binding:
        raise ThreeArmWorkerError("reservation changed while being validated")
    return reservation, reservation_binding


def load_authorized_contract(
    *,
    authority_path: Path,
    expected_byte_count: int,
    expected_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    authority, authority_binding, _plan, plan_binding, sources = (
        supervisor_contract.load_and_validate_authority(
            authority_path,
            expected_byte_count=expected_byte_count,
            expected_sha256=expected_sha256,
        )
    )
    _validate_required_source_closure(sources)
    _validate_runtime_identity(authority)
    _validate_input_bindings(authority)
    caps = authority["caps"]
    if (
        set(caps)
        != {
            "maximum_wall_seconds",
            "maximum_gpu_seconds",
            "maximum_training_updates",
        }
        or caps["maximum_training_updates"] != TRAINING_UPDATES
        or not 0 < float(caps["maximum_wall_seconds"]) <= MAXIMUM_WALL_SECONDS
        or not 0 < float(caps["maximum_gpu_seconds"]) <= MAXIMUM_GPU_SECONDS
    ):
        raise ThreeArmWorkerError("authority caps changed")
    reservation, reservation_binding = _load_and_validate_reservation(
        authority_path=authority_path,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        sources=sources,
    )
    return (
        authority,
        authority_binding,
        plan_binding,
        sources,
        reservation,
        reservation_binding,
    )


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return value


def _predecessor_model_state_is_valid(
    state: Any,
) -> bool:
    """Validate the exact spatial-V1 state dtypes before migration.

    Spatial V1 contains float32 parameter/encoder tensors plus one persistent
    scalar-long ema_update_count buffer. That accounting buffer is validated
    here but is intentionally rejected from temporal migration by the model's
    stricter inventory/migration contract.
    """

    if type(state) is not dict or not state:
        return False
    for name, value in state.items():
        if type(name) is not str or not isinstance(value, torch.Tensor):
            return False
        if value.layout != torch.strided:
            return False
        if name == "ema_update_count":
            if (
                value.dtype != torch.long
                or tuple(value.shape) != ()
                or int(value.detach().cpu().item()) != PREDECESSOR_UPDATE
            ):
                return False
            continue
        if (
            value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            return False
    return True


def load_predecessor_state(binding: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    path = _resolve_binding_path(binding)
    if path.is_symlink() or not path.is_file():
        raise ThreeArmWorkerError("predecessor checkpoint is unavailable or unsafe")
    raw = path.read_bytes()
    if (
        len(raw) != binding["byte_count"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
    ):
        raise ThreeArmWorkerError("predecessor checkpoint changed before loading")
    checkpoint = torch.load(
        io.BytesIO(raw), map_location="cpu", weights_only=True
    )
    del raw
    if type(checkpoint) is not dict or type(checkpoint.get("model_state_dict")) is not dict:
        raise ThreeArmWorkerError("predecessor checkpoint schema changed")
    state = checkpoint["model_state_dict"]
    if not _predecessor_model_state_is_valid(state):
        raise ThreeArmWorkerError("predecessor model state is invalid")
    return {name: value.detach().clone() for name, value in state.items()}


def build_frozen_substrate_and_arms(
    predecessor_state: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[
    temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    dict[str, ArmCore],
    dict[str, torch.optim.AdamW],
    dict[str, ArmPartition],
    dict[str, Any],
]:
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    substrate = temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1(
        predecessor_state
    ).to(device)
    substrate.eval()
    for parameter in substrate.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if int(substrate.ema_update_count.detach().cpu()) != 0:
        raise ThreeArmWorkerError("fresh temporal substrate has a nonzero EMA count")
    encoder_sha256 = module_state_sha256(substrate.encoder)
    target_sha256 = module_state_sha256(substrate.target_encoder)
    if encoder_sha256 != target_sha256:
        raise ThreeArmWorkerError("frozen online and target encoders are not exact copies")

    arms = {name: ArmCore(substrate).to(device) for name in ARM_NAMES}
    initial_hashes = {name: module_state_sha256(arm) for name, arm in arms.items()}
    if len(set(initial_hashes.values())) != 1:
        raise ThreeArmWorkerError("three arms did not receive identical initialization")
    parameter_ids = {
        name: {id(parameter) for parameter in arm.parameters()}
        for name, arm in arms.items()
    }
    if any(
        parameter_ids[left] & parameter_ids[right]
        for offset, left in enumerate(ARM_NAMES)
        for right in ARM_NAMES[offset + 1 :]
    ):
        raise ThreeArmWorkerError("arm parameters are not independently allocated")
    optimizers: dict[str, torch.optim.AdamW] = {}
    partitions: dict[str, ArmPartition] = {}
    for name, arm in arms.items():
        optimizer, partition = build_arm_optimizer(arm)
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        optimizers[name] = optimizer
        partitions[name] = partition
        arm.train()

    # A payload-free contract probe proves that the narrow conditioned helper
    # exactly reproduces the reviewed model route and that blind candidate IDs
    # are mathematically inert before any training input is used.
    probe_history = torch.zeros((2, 3, 256, 192), dtype=torch.float32, device=device)
    probe_actions = torch.tensor(((0, 1, 2), (8, 7, 6)), dtype=torch.long, device=device)
    probe_targets, _ = temporal_metrics.batched_mask_indices(
        "val", (0, 1), device=device
    )
    with torch.no_grad():
        reference = substrate.predict_from_encoded_history(
            probe_history, probe_actions, probe_targets
        ).raw_predicted_target_tokens
        reproduced = predict_from_shared_encoding(
            arms["conditioned"],
            probe_history,
            probe_actions,
            probe_targets,
            candidate_blind=False,
        ).raw
        helper_error = float((reference - reproduced).abs().max())
        blind_outputs = []
        for candidate_id in range(ACTION_COUNT):
            intervention = probe_actions.clone()
            intervention[:, 2] = candidate_id
            blind_outputs.append(
                predict_from_shared_encoding(
                    arms["blind"],
                    probe_history,
                    intervention,
                    probe_targets,
                    candidate_blind=True,
                ).raw
            )
        blind_spread = float(
            (torch.stack(blind_outputs).max(dim=0).values
             - torch.stack(blind_outputs).min(dim=0).values).abs().max()
        )
        probe_rgb = torch.zeros((2, 3, 112, 112), dtype=torch.float32, device=device)
        online_current = _gather_spatial_tokens(
            substrate.encode_online_full_frame(probe_rgb), probe_targets
        )
        target_current = substrate.encode_target(
            probe_rgb, probe_targets
        ).raw_target_tokens
        persistence_parity_error = float(
            (online_current - target_current).abs().max()
        )
    if (
        helper_error != 0.0
        or blind_spread != 0.0
        or persistence_parity_error != 0.0
    ):
        raise ThreeArmWorkerError(
            "head, blind-treatment, or online/target parity probe failed"
        )
    return substrate, arms, optimizers, partitions, {
        "encoder_sha256": encoder_sha256,
        "target_sha256": target_sha256,
        "initial_arm_state_sha256": initial_hashes,
        "pairwise_disjoint_arm_parameters": True,
        "conditioned_helper_max_abs_error": helper_error,
        "blind_candidate_max_abs_spread": blind_spread,
        "persistence_online_target_max_abs_error": persistence_parity_error,
        "ema_update_count": 0,
    }


def assert_frozen_substrate_unchanged(
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    *,
    encoder_sha256: str,
    target_sha256: str,
) -> None:
    if (
        substrate.training
        or substrate.encoder.training
        or substrate.target_encoder.training
        or module_state_sha256(substrate.encoder) != encoder_sha256
        or module_state_sha256(substrate.target_encoder) != target_sha256
        or int(substrate.ema_update_count.detach().cpu()) != 0
        or any(parameter.requires_grad or parameter.grad is not None for parameter in substrate.parameters())
    ):
        raise ThreeArmWorkerError("frozen shared substrate changed")


def snapshot_payload(
    *,
    arm_name: str,
    arm: ArmCore,
    optimizer: torch.optim.AdamW,
    update: int,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    substrate_receipt: Mapping[str, Any],
    schedule_audit: Mapping[str, Any],
    metric_vectors: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "status": "INERT_AUDIT_SNAPSHOT",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "arm": arm_name,
        "update": update,
        "authority_binding": dict(authority_binding),
        "plan_binding": dict(plan_binding),
        "substrate": dict(substrate_receipt),
        "schedule": dict(schedule_audit),
        "metric_vectors": _clone_cpu(metric_vectors),
        "arm_state_dict": _clone_cpu(arm.state_dict()),
        "optimizer_state_dict": _clone_cpu(optimizer.state_dict()),
    }


def create_fresh_pack(
    *,
    pack_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if pack_root.exists() or pack_root.is_symlink():
        raise ThreeArmWorkerError("fresh pack path is already occupied")
    command = [
        authority["runtime"]["python_invocation_path"],
        str((REPO_ROOT / REQUIRED_SOURCE_PATHS["packer"]).resolve()),
        "--workers",
        "16",
        "--verify",
        "24",
        "--out",
        str(pack_root.resolve()),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise ThreeArmWorkerError(
            f"fresh visible-frame packer exited {completed.returncode}"
        )
    manifest = pack_root / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ThreeArmWorkerError("fresh pack did not publish a manifest")
    return file_binding(manifest, relative_to=Path(authority["output_root"]))


def build_pack_artifact_bindings(
    *,
    pack_root: Path,
    attempt_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bind all six inert pack payloads directly in the terminal result."""

    selected_pack = Path(pack_root)
    selected_attempt = Path(attempt_root)
    if (
        selected_pack.is_symlink()
        or not selected_pack.is_dir()
        or selected_pack.resolve(strict=True)
        != (selected_attempt.resolve(strict=True) / "pack")
    ):
        raise ThreeArmWorkerError("pack root changed before terminal binding")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for role, artifacts in PACK_ARTIFACT_RELATIVE_PATHS.items():
        result[role] = {}
        for name, relative in artifacts.items():
            path = selected_attempt / Path(relative)
            result[role][name] = file_binding(path, relative_to=selected_attempt)
    return result


def _set_optimizer_learning_rates(
    optimizer: torch.optim.AdamW,
    *,
    fraction: float,
) -> dict[str, float]:
    expected = {
        "predictor": PREDICTOR_BASE_LR * LR_SCALE * fraction,
        "memory": MEMORY_BASE_LR * LR_SCALE * fraction,
    }
    for group in optimizer.param_groups:
        name = group.get("group_name")
        if name not in expected:
            raise ThreeArmWorkerError("optimizer group identity changed")
        group["lr"] = expected[name]
    return expected


def train_one_update(
    *,
    update: int,
    batch_rows_cpu: torch.Tensor,
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    arms: Mapping[str, ArmCore],
    optimizers: Mapping[str, torch.optim.AdamW],
    partitions: Mapping[str, ArmPartition],
    train_frames: torch.Tensor,
    train_actions: torch.Tensor,
    shuffled_candidate_actions: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    if batch_rows_cpu.shape != (BATCH_SIZE,) or batch_rows_cpu.dtype != torch.long:
        raise ThreeArmWorkerError("training batch row schedule changed")
    fraction = learning_rate_fraction(update)
    learning_rates = {
        name: _set_optimizer_learning_rates(optimizer, fraction=fraction)
        for name, optimizer in optimizers.items()
    }
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    weighted_loss = {name: 0.0 for name in ARM_NAMES}

    for start in range(0, BATCH_SIZE, MICROBATCH_SIZE):
        micro_cpu = batch_rows_cpu[start : start + MICROBATCH_SIZE]
        micro = micro_cpu.to(device=train_frames.device)
        normalized = scaled.to_float(train_frames[micro])
        target_indices, _ = temporal_metrics.batched_mask_indices(
            "train", micro_cpu.tolist(), device=train_frames.device
        )
        encoded, target = _encode_context_and_future(
            substrate, normalized, target_indices
        )
        factual_actions = train_actions[micro]
        for arm_name in ARM_NAMES:
            arm_actions = _candidate_action_tensor(
                factual_actions,
                arm_name=arm_name,
                row_indices=micro,
                shuffled_candidate_actions=shuffled_candidate_actions,
                training=True,
            )
            prediction = predict_from_shared_encoding(
                arms[arm_name],
                encoded,
                arm_actions,
                target_indices,
                candidate_blind=_arm_is_candidate_blind(arm_name),
            )
            loss = normalized_half_squared_jepa_loss_v1(prediction.raw, target)
            if not bool(torch.isfinite(loss)):
                raise ThreeArmWorkerError(f"nonfinite training loss in {arm_name}")
            (loss * (MICROBATCH_SIZE / BATCH_SIZE)).backward()
            weighted_loss[arm_name] += float(loss.detach()) * MICROBATCH_SIZE / BATCH_SIZE

    for arm_name in ARM_NAMES:
        partition = partitions[arm_name]
        if not any(parameter.grad is not None for parameter in partition.all):
            raise ThreeArmWorkerError(f"{arm_name} produced no gradient tensors")
        norm = torch.nn.utils.clip_grad_norm_(partition.all, GRADIENT_CLIP)
        if not bool(torch.isfinite(norm)):
            raise ThreeArmWorkerError(f"{arm_name} gradient norm is nonfinite")
        optimizers[arm_name].step()
        if any(
            not bool(torch.isfinite(parameter).all())
            for parameter in partition.all
        ):
            raise ThreeArmWorkerError(f"{arm_name} parameters became nonfinite")
        optimizers[arm_name].zero_grad(set_to_none=True)
    return weighted_loss, {
        "fraction": fraction,
        "predictor": learning_rates["conditioned"]["predictor"],
        "memory": learning_rates["conditioned"]["memory"],
    }


def _energy_summary(
    values: torch.Tensor,
    rows: Sequence[h6.H6V2Row],
    row_indices: Sequence[int],
) -> dict[str, Any]:
    energy = values.detach().to(device="cpu", dtype=torch.float64).numpy()
    if energy.shape != (len(row_indices),) or not np.isfinite(energy).all() or bool((energy <= 0.0).any()):
        raise ThreeArmWorkerError("energy vector is invalid")
    selected_rows = [rows[int(index)] for index in row_indices]
    family_means = []
    for family in experiment_metrics.REGISTERED_FAMILIES:
        positions = [
            position
            for position, row in enumerate(selected_rows)
            if row.family == family
        ]
        if not positions:
            raise ThreeArmWorkerError("energy panel lost a registered family")
        family_means.append(float(energy[positions].mean()))
    action_means = []
    for action in range(ACTION_COUNT):
        positions = [
            position
            for position, row in enumerate(selected_rows)
            if int(row.actions[2]) == action
        ]
        if not positions:
            raise ThreeArmWorkerError("energy panel lost a registered action")
        action_means.append(float(energy[positions].mean()))
    return {
        "mean": float(energy.mean()),
        "family_equal_mean": math.fsum(family_means) / len(family_means),
        "action_equal_mean": math.fsum(action_means) / len(action_means),
        "family_count": len(experiment_metrics.REGISTERED_FAMILIES),
        "action_count": ACTION_COUNT,
        "scene_count": len({row.scene_id for row in selected_rows}),
    }


def _action_identification_receipt(
    summary: experiment_metrics.ActionIdentificationSummary,
) -> dict[str, Any]:
    unique_count = summary.row_count - summary.exact_tie_row_count
    if unique_count != summary.unique_winner_count:
        raise ThreeArmWorkerError("nine-way unique-winner accounting changed")
    return {
        "bootstrap_algorithm": summary.bootstrap_algorithm,
        "bootstrap_interpretation": summary.bootstrap_interpretation,
        "bootstrap_seed": int(summary.bootstrap_seed),
        "bootstrap_replicates": int(summary.bootstrap_replicates),
        "bootstrap_lower_index": int(summary.bootstrap_lower_index),
        "family_action_supporting_scene_counts": {
            family: [int(value) for value in counts]
            for family, counts in summary.family_action_supporting_scene_counts.items()
        },
        "minimum_family_action_supporting_scene_count": int(
            summary.minimum_family_action_supporting_scene_count
        ),
        "balanced_accuracy": float(summary.scene_family_balanced_accuracy),
        "balanced_accuracy_one_sided_95_lower_bound": float(
            summary.balanced_accuracy_bootstrap_lower_95
        ),
        "balanced_chance": 1.0 / ACTION_COUNT,
        "exact_tie_count": int(summary.exact_tie_row_count),
        "exact_tie_rate": float(summary.exact_tie_rate),
        "unique_winner_count": int(unique_count),
        "unique_winner_accuracy": float(summary.unique_winner_accuracy),
        "hardest_wrong_action_margin": float(summary.hardest_action_margin),
        "hardest_wrong_action_margin_one_sided_95_lower_bound": float(
            summary.hardest_margin_bootstrap_lower_95
        ),
    }


@dataclass
class ValidationAnalysis:
    validation_by_arm: dict[str, dict[str, Any]]
    cross_arm: dict[str, experiment_metrics.PairedLogEnergyComparison]
    controls: dict[str, dict[str, experiment_metrics.PairedLogEnergyComparison]]
    action_identification: dict[str, experiment_metrics.ActionIdentificationSummary]
    rank_ratio_by_arm: dict[str, float]


def analyze_validation(
    vectors: EvaluationVectors,
    *,
    rows: Sequence[h6.H6V2Row],
) -> ValidationAnalysis:
    if (
        vectors.role != "val"
        or vectors.row_indices != tuple(range(EXPECTED_VALIDATION_ROWS))
        or vectors.persistence_energy is None
        or vectors.target_tokens is None
        or set(vectors.candidate_energy) != set(ARM_NAMES)
        or set(vectors.wrong_history_energy) != set(ARM_NAMES)
        or set(vectors.prediction_tokens) != set(ARM_NAMES)
    ):
        raise ThreeArmWorkerError("validation vector panel is incomplete")
    selected_rows = [rows[index] for index in vectors.row_indices]
    scene_ids = [row.scene_id for row in selected_rows]
    family_ids = [row.family for row in selected_rows]
    factual_actions = [int(row.actions[2]) for row in selected_rows]
    cross_arm = {
        control: experiment_metrics.paired_log_energy_comparison(
            vectors.factual_energy["conditioned"],
            vectors.factual_energy[control],
            scene_ids,
            family_ids,
            control_name=control,
        )
        for control in ("blind", "shuffled")
    }
    controls = {
        arm_name: {
            "persistence": experiment_metrics.paired_log_energy_comparison(
                vectors.factual_energy[arm_name],
                vectors.persistence_energy,
                scene_ids,
                family_ids,
                control_name="persistence",
            ),
            "wrong_history": experiment_metrics.paired_log_energy_comparison(
                vectors.factual_energy[arm_name],
                vectors.wrong_history_energy[arm_name],
                scene_ids,
                family_ids,
                control_name="wrong_history",
            ),
        }
        for arm_name in ARM_NAMES
    }
    action_identification = {
        arm_name: experiment_metrics.summarize_nine_way_action_identification(
            vectors.candidate_energy[arm_name],
            factual_actions,
            scene_ids,
            family_ids,
        )
        for arm_name in ARM_NAMES
    }
    target_rank, _target_variance = scaled.effective_rank(vectors.target_tokens)
    validation: dict[str, dict[str, Any]] = {}
    rank_ratios: dict[str, float] = {}
    cross_receipt = {
        "conditioned_vs_blind_log_energy_advantage": float(
            cross_arm["blind"].macro_log_advantage
        ),
        "conditioned_vs_blind_one_sided_95_lower_bound": float(
            cross_arm["blind"].bootstrap_lower_95
        ),
        "conditioned_vs_shuffled_log_energy_advantage": float(
            cross_arm["shuffled"].macro_log_advantage
        ),
        "conditioned_vs_shuffled_one_sided_95_lower_bound": float(
            cross_arm["shuffled"].bootstrap_lower_95
        ),
        "scene_cluster_count": int(cross_arm["blind"].scene_count),
    }
    for arm_name in ARM_NAMES:
        prediction_rank, _prediction_variance = scaled.effective_rank(
            vectors.prediction_tokens[arm_name]
        )
        ratio = prediction_rank / target_rank if target_rank > 0.0 else 0.0
        rank_ratios[arm_name] = ratio
        validation[arm_name] = {
            "row_count": EXPECTED_VALIDATION_ROWS,
            "factual_energy": _energy_summary(
                vectors.factual_energy[arm_name], rows, vectors.row_indices
            ),
            "cross_arm": dict(cross_receipt),
            "controls": {
                "persistence_log_energy_advantage": float(
                    controls[arm_name]["persistence"].macro_log_advantage
                ),
                "persistence_one_sided_95_lower_bound": float(
                    controls[arm_name]["persistence"].bootstrap_lower_95
                ),
                "wrong_history_log_energy_advantage": float(
                    controls[arm_name]["wrong_history"].macro_log_advantage
                ),
                "wrong_history_one_sided_95_lower_bound": float(
                    controls[arm_name]["wrong_history"].bootstrap_lower_95
                ),
            },
            "action_identification": _action_identification_receipt(
                action_identification[arm_name]
            ),
            "representation": {
                "prediction_effective_rank": float(prediction_rank),
                "target_effective_rank": float(target_rank),
                "prediction_to_target_rank_ratio": float(ratio),
            },
        }
    return ValidationAnalysis(
        validation_by_arm=validation,
        cross_arm=cross_arm,
        controls=controls,
        action_identification=action_identification,
        rank_ratio_by_arm=rank_ratios,
    )


def analyze_training_fit(
    vectors: EvaluationVectors,
    *,
    rows: Sequence[h6.H6V2Row],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, experiment_metrics.FamilyEqualLogEnergyAdvantage],
]:
    if vectors.role != "train" or vectors.row_indices != tuple(range(EXPECTED_TRAIN_ROWS)):
        raise ThreeArmWorkerError("training-fit vector panel is incomplete")
    family_ids = [row.family for row in rows]
    comparisons = {
        control: experiment_metrics.family_equal_paired_log_energy_advantage(
            vectors.factual_energy["conditioned"],
            vectors.factual_energy[control],
            family_ids,
            control_name=control,
        )
        for control in ("blind", "shuffled")
    }
    return {
        arm_name: {
            "row_count": EXPECTED_TRAIN_ROWS,
            "family_count": len(experiment_metrics.REGISTERED_FAMILIES),
            "factual_mean_energy": float(
                vectors.factual_energy[arm_name].mean()
            ),
            "conditioned_vs_blind_family_equal_log_energy_advantage": float(
                comparisons["blind"].macro_log_advantage
            ),
            "conditioned_vs_shuffled_family_equal_log_energy_advantage": float(
                comparisons["shuffled"].macro_log_advantage
            ),
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
        for arm_name in ARM_NAMES
    }, comparisons


def build_overlap_audit(
    train_rows: Sequence[h6.H6V2Row],
    val_rows: Sequence[h6.H6V2Row],
) -> dict[str, Any]:
    diagnostic = experiment_metrics.audit_h6_metadata_overlap(
        tuple(train_rows) + tuple(val_rows)
    )
    if diagnostic.get("passed") is not True:
        raise ThreeArmWorkerError("existing-pool overlap audit did not pass")
    return diagnostic


def build_shuffle_audit(
    derangement: experiment_metrics.CandidateActionDerangement,
    train_rows: Sequence[h6.H6V2Row],
) -> dict[str, Any]:
    donor_positions = tuple(int(value) for value in derangement.donor_positions)
    factual = tuple(int(value) for value in derangement.factual_candidate_action_ids)
    shuffled = tuple(int(value) for value in derangement.deranged_candidate_action_ids)
    if len(donor_positions) != EXPECTED_TRAIN_ROWS:
        raise ThreeArmWorkerError("candidate-action derangement row count changed")
    same_scene = sum(
        train_rows[row].scene_id == train_rows[donor].scene_id
        for row, donor in enumerate(donor_positions)
    )
    cross_family = sum(
        train_rows[row].family != train_rows[donor].family
        for row, donor in enumerate(donor_positions)
    )
    marginals_preserved = all(
            sorted(
                factual[index]
                for index, row in enumerate(train_rows)
                if row.family == family
            )
            == sorted(
                shuffled[index]
                for index, row in enumerate(train_rows)
                if row.family == family
            )
            for family in experiment_metrics.REGISTERED_FAMILIES
        )
    if not (
        sorted(donor_positions) == list(range(len(donor_positions)))
        and sum(row == donor for row, donor in enumerate(donor_positions)) == 0
        and same_scene == 0
        and sum(left == right for left, right in zip(factual, shuffled, strict=True)) == 0
        and cross_family == 0
        and marginals_preserved
    ):
        raise ThreeArmWorkerError("candidate-action derangement audit failed")
    return dict(derangement.audit)


def measurement_payload(
    *,
    arm_name: str,
    update: int,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    substrate_receipt: Mapping[str, Any],
    validation: Mapping[str, Any],
    training: Mapping[str, Any] | None,
    loss: float | None,
    learning_rate: Mapping[str, float],
) -> dict[str, Any]:
    if arm_name not in ARM_NAMES or update not in OBSERVATION_UPDATES:
        raise ThreeArmWorkerError("measurement arm/update changed")
    if update == 0:
        expected_fraction = 0.0
        if loss is not None:
            raise ThreeArmWorkerError("update-zero loss must be null")
    else:
        expected_fraction = learning_rate_fraction(update)
        if loss is None or not math.isfinite(loss) or loss < 0.0:
            raise ThreeArmWorkerError("measurement loss is invalid")
    return {
        "schema": MEASUREMENT_SCHEMA,
        "status": "COMPLETE",
        "arm": arm_name,
        "update": update,
        "authority_binding": dict(authority_binding),
        "plan_binding": dict(plan_binding),
        "encoder_sha256": substrate_receipt["encoder_sha256"],
        "target_sha256": substrate_receipt["target_sha256"],
        "panel": {
            "kind": "scene_disjoint_factual_validation",
            "row_count": EXPECTED_VALIDATION_ROWS,
            "row_indices_sha256": canonical_sha256(
                list(range(EXPECTED_VALIDATION_ROWS))
            ),
        },
        "validation": dict(validation),
        "training": None if training is None else dict(training),
        "optimization": {
            "completed_updates": update,
            "optimizer_steps": update,
            "loss": loss,
            "learning_rate_fraction": expected_fraction,
            "predictor_learning_rate": float(learning_rate["predictor"]),
            "memory_learning_rate": float(learning_rate["memory"]),
            "warmup_updates": WARMUP_UPDATES,
            "schedule_horizon_updates": COSINE_SCHEDULE_UPDATES,
        },
        "integrity": {
            "candidate_blind_treatment_exact": True,
            "shuffled_derangement_exact": True,
            "factual_evaluation_exact": True,
            "frozen_substrate_exact": True,
            "no_gradient_during_evaluation": True,
            "finite": True,
        },
    }


def _snapshot_metric_vectors(
    vectors: EvaluationVectors,
    *,
    arm_name: str,
    training_vectors: EvaluationVectors | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "validation_row_indices": list(vectors.row_indices),
        "validation_factual_energy": vectors.factual_energy[arm_name],
        "validation_persistence_energy": vectors.persistence_energy,
        "validation_wrong_history_energy": vectors.wrong_history_energy[arm_name],
        "validation_candidate_energy": vectors.candidate_energy[arm_name],
        "prediction_tokens": vectors.prediction_tokens[arm_name],
    }
    if arm_name == "conditioned":
        result["target_tokens"] = vectors.target_tokens
    if training_vectors is not None:
        result.update(
            {
                "training_row_indices": list(training_vectors.row_indices),
                "training_factual_energy": training_vectors.factual_energy[arm_name],
            }
        )
    return result


def exact_accounting() -> dict[str, Any]:
    return {
        "bound_h6_rows": 18_048,
        "initial_rgb_leaf_opens": 72_192,
        "verification_rgb_leaf_reopens": 192,
        "total_rgb_leaf_opens": 72_384,
        "forbidden_future_rgb_leaf_opens": 0,
        "packed_frame_bytes": 2_716_729_344,
        "training_schedule_row_presentations": 179_200,
        "sequence_presentations_per_arm": 179_200,
        "total_arm_head_sequence_presentations": 537_600,
        "shared_online_context_frame_encodings": 537_600,
        "shared_future_target_frame_encodings": 179_200,
        "actual_training_frame_encodings": 716_800,
        "optimizer_steps_per_arm": 700,
        "total_optimizer_steps": 2_100,
        "target_ema_steps": 0,
        "validation_row_panels_per_arm": 16_384,
        "shared_validation_frame_encodings": 65_536,
        "nine_way_arm_candidate_row_queries": 442_368,
        "validation_backward_calls": 0,
        "validation_optimizer_steps": 0,
        "train_fit_rows": 16_000,
        "train_fit_shared_frame_encodings": 64_000,
        "train_fit_arm_factual_row_queries": 48_000,
        "train_fit_backward_calls": 0,
        "train_fit_optimizer_steps": 0,
        "total_shared_frame_encodings": 846_336,
        "measurement_receipts": 24,
        "snapshot_bindings": 24,
        "sealed_open_count": 0,
        "heldout_open_count": 0,
        "network_access_count": 0,
        "training_consumed_pack_only": True,
    }


def build_joint_decision(
    *,
    metric_decision: experiment_metrics.ThreeArmDecision,
    substrate_receipt: Mapping[str, Any],
    training_by_arm: Mapping[str, Mapping[str, Any]],
    validation_analyses: Mapping[int, ValidationAnalysis],
) -> dict[str, Any]:
    tail = []
    for update in (500, 600, 700):
        validation = validation_analyses[update].validation_by_arm["conditioned"]
        cross = validation["cross_arm"]
        tail.append(
            {
                "update": update,
                "conditioned_vs_blind_log_energy_advantage": cross[
                    "conditioned_vs_blind_log_energy_advantage"
                ],
                "conditioned_vs_blind_one_sided_95_lower_bound": cross[
                    "conditioned_vs_blind_one_sided_95_lower_bound"
                ],
                "conditioned_vs_shuffled_log_energy_advantage": cross[
                    "conditioned_vs_shuffled_log_energy_advantage"
                ],
                "conditioned_vs_shuffled_one_sided_95_lower_bound": cross[
                    "conditioned_vs_shuffled_one_sided_95_lower_bound"
                ],
                "prediction_to_target_rank_ratio": validation["representation"][
                    "prediction_to_target_rank_ratio"
                ],
            }
        )
    final = validation_analyses[700].validation_by_arm["conditioned"]
    result = {
        "status": metric_decision.status,
        "citable_as_scientific_evidence": False,
        "scientific_claim_authorized": False,
        "treatment": {
            "conditioned_action_gains": [1, 1, 1],
            "blind_action_gains": [1, 1, 0],
            "shuffled_action_gains": [1, 1, 1],
            "blind_preserves_factual_history": True,
            "shuffled_changes_only_training_candidate": True,
            "shuffled_validation_uses_factual_candidate": True,
            "requested_executed_equivalence_claimed": False,
        },
        "schedule": {
            "seed": TRAINING_SEED,
            "updates": TRAINING_UPDATES,
            "sequence_batch": BATCH_SIZE,
            "microbatch": MICROBATCH_SIZE,
            "train_rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "warmup_updates": WARMUP_UPDATES,
            "schedule_horizon_updates": COSINE_SCHEDULE_UPDATES,
            "observation_updates": list(OBSERVATION_UPDATES),
            "early_stopping": False,
            "checkpoint_selection": False,
        },
        "frozen_substrate": {
            "encoder_initial_sha256": substrate_receipt["encoder_sha256"],
            "encoder_final_sha256": substrate_receipt["encoder_sha256"],
            "target_initial_sha256": substrate_receipt["target_sha256"],
            "target_final_sha256": substrate_receipt["target_sha256"],
            "requires_grad": False,
            "evaluation_mode": True,
            "gradient_tensor_count": 0,
            "ema_update_count": 0,
        },
        "evidence": {
            "train_fit_update_700": {
                "conditioned_vs_blind_family_equal_log_energy_advantage": (
                    training_by_arm["conditioned"][
                        "conditioned_vs_blind_family_equal_log_energy_advantage"
                    ]
                ),
                "conditioned_vs_shuffled_family_equal_log_energy_advantage": (
                    training_by_arm["conditioned"][
                        "conditioned_vs_shuffled_family_equal_log_energy_advantage"
                    ]
                ),
            },
            "validation_tail": tail,
            "conditioned_update_700": {
                "balanced_accuracy_one_sided_95_lower_bound": final[
                    "action_identification"
                ]["balanced_accuracy_one_sided_95_lower_bound"],
                "hardest_wrong_action_margin_one_sided_95_lower_bound": final[
                    "action_identification"
                ]["hardest_wrong_action_margin_one_sided_95_lower_bound"],
                "persistence_one_sided_95_lower_bound": final["controls"][
                    "persistence_one_sided_95_lower_bound"
                ],
                "wrong_history_one_sided_95_lower_bound": final["controls"][
                    "wrong_history_one_sided_95_lower_bound"
                ],
            },
        },
        "gate_precedence": [
            "INCONCLUSIVE_CONTRACT_FAILURE",
            "LOCALIZE_TRAIN_FIT_FAILURE",
            "LOCALIZE_GENERALIZATION_OR_CONFOUNDING",
            "LOCALIZE_ACTION_ALIGNMENT_FAILURE",
            "LOCALIZE_PREDICTOR_NOT_USEFUL",
            "PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY",
        ],
    }
    return result


def _observed_runtime(
    *,
    gpu_elapsed: float,
    wall_elapsed: float,
    output_inventory: Sequence[str],
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "device_name": torch.cuda.get_device_name(0),
        "device_arch": str(getattr(properties, "gcnArchName", "")),
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
        "numpy_version": np.__version__,
        "pillow_version": packer.PIL.__version__,
        "python_version": sys.version.split()[0],
        "gpu_phase_elapsed_seconds": gpu_elapsed,
        "wall_elapsed_seconds": wall_elapsed,
        "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "output_inventory": list(output_inventory),
    }


def execute_authorized_experiment(
    *,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> int:
    wall_started = time.monotonic()
    (
        authority,
        authority_binding,
        plan_binding,
        sources,
        reservation,
        reservation_binding,
    ) = load_authorized_contract(
        authority_path=authority_path,
        expected_byte_count=expected_authority_byte_count,
        expected_sha256=expected_authority_sha256,
    )
    attempt_root = Path(authority["output_root"])
    pack_root = attempt_root / "pack"
    arms_root = attempt_root / "arms"
    arms_root.mkdir()
    for arm_name in ARM_NAMES:
        (arms_root / arm_name / "measurements").mkdir(parents=True)
        (arms_root / arm_name / "snapshots").mkdir()

    train_rows, train_index_audit = h6.load_bound_index(REPO_ROOT, role="train")
    val_rows, val_index_audit = h6.load_bound_index(REPO_ROOT, role="val")
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(val_rows) != EXPECTED_VALIDATION_ROWS:
        raise ThreeArmWorkerError("bound H6 role row counts changed")
    if (
        train_index_audit["file_sha256"]
        != authority["input_bindings"]["train_index"]["file_sha256"]
        or val_index_audit["file_sha256"]
        != authority["input_bindings"]["validation_index"]["file_sha256"]
    ):
        raise ThreeArmWorkerError("loaded H6 index differs from authority")

    overlap_audit = build_overlap_audit(train_rows, val_rows)
    derangement = experiment_metrics.build_candidate_action_derangement(train_rows)
    shuffle_audit = build_shuffle_audit(derangement, train_rows)
    overlap_binding = write_immutable_json(
        attempt_root / "overlap_audit.json",
        {
            "schema": OVERLAP_AUDIT_SCHEMA,
            "status": "PASS",
            "passed": True,
            "authority_binding": dict(authority_binding),
            "plan_binding": dict(plan_binding),
            "audit": overlap_audit,
        },
        attempt_root=attempt_root,
    )
    shuffle_binding = write_immutable_json(
        attempt_root / "shuffle_audit.json",
        {
            "schema": SHUFFLE_AUDIT_SCHEMA,
            "status": "PASS",
            "passed": True,
            "authority_binding": dict(authority_binding),
            "plan_binding": dict(plan_binding),
            "audit": shuffle_audit,
        },
        attempt_root=attempt_root,
    )
    schedule, schedule_audit = build_bound_training_schedule()
    pack_binding = create_fresh_pack(pack_root=pack_root, authority=authority)

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    gpu_started = time.monotonic()

    predecessor_state = load_predecessor_state(
        authority["input_bindings"]["predecessor_checkpoint"]
    )
    substrate, arms, optimizers, partitions, substrate_receipt = (
        build_frozen_substrate_and_arms(predecessor_state, device=device)
    )
    del predecessor_state
    train_frames, train_actions, train_pack_binding = scaled.load_pack(
        pack_root, "train", device
    )
    val_frames, val_actions, val_pack_binding = scaled.load_pack(
        pack_root, "val", device
    )
    pack_bindings = {"train": train_pack_binding, "val": val_pack_binding}
    shuffled_candidate_actions = torch.tensor(
        derangement.deranged_candidate_action_ids,
        dtype=torch.long,
        device=device,
    )
    val_donors = temporal_metrics.build_wrong_history_donor_indices(
        _to_temporal_metrics_rows(val_rows)
    )

    measurement_bindings: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ARM_NAMES
    }
    snapshot_bindings: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ARM_NAMES
    }
    validation_analyses: dict[int, ValidationAnalysis] = {}
    training_by_arm: dict[str, dict[str, Any]] | None = None
    training_comparisons: dict[
        str, experiment_metrics.FamilyEqualLogEnergyAdvantage
    ] | None = None
    losses: dict[str, float | None] = {name: None for name in ARM_NAMES}
    learning_rate = {"fraction": 0.0, "predictor": 0.0, "memory": 0.0}
    maximum_blind_spread = 0.0

    def observe(update: int) -> None:
        nonlocal training_by_arm, training_comparisons, maximum_blind_spread
        assert_frozen_substrate_unchanged(
            substrate,
            encoder_sha256=substrate_receipt["encoder_sha256"],
            target_sha256=substrate_receipt["target_sha256"],
        )
        if any(parameter.grad is not None for arm in arms.values() for parameter in arm.parameters()):
            raise ThreeArmWorkerError("evaluation began with residual gradient tensors")
        train_vectors: EvaluationVectors | None = None
        if update == TRAINING_UPDATES:
            train_vectors = evaluate_panel(
                substrate=substrate,
                arms=arms,
                frames=train_frames,
                actions=train_actions,
                role="train",
                row_indices=range(EXPECTED_TRAIN_ROWS),
                wrong_history_donors=None,
                include_controls=False,
                include_nine_way=False,
                include_rank_tokens=False,
            )
            training_by_arm, training_comparisons = analyze_training_fit(
                train_vectors, rows=train_rows
            )
        validation_vectors = evaluate_panel(
            substrate=substrate,
            arms=arms,
            frames=val_frames,
            actions=val_actions,
            role="val",
            row_indices=range(EXPECTED_VALIDATION_ROWS),
            wrong_history_donors=val_donors,
            include_controls=True,
            include_nine_way=True,
            include_rank_tokens=True,
        )
        if validation_vectors.blind_candidate_max_spread is None:
            raise ThreeArmWorkerError("blind candidate spread audit is absent")
        maximum_blind_spread = max(
            maximum_blind_spread,
            validation_vectors.blind_candidate_max_spread,
        )
        if maximum_blind_spread != 0.0:
            raise ThreeArmWorkerError("blind candidate action was not exactly inert")
        analysis = analyze_validation(validation_vectors, rows=val_rows)
        validation_analyses[update] = analysis
        for arm_name in ARM_NAMES:
            snapshot = save_immutable_snapshot(
                arms_root / arm_name / "snapshots" / f"update_{update:06d}.pt",
                snapshot_payload(
                    arm_name=arm_name,
                    arm=arms[arm_name],
                    optimizer=optimizers[arm_name],
                    update=update,
                    authority_binding=authority_binding,
                    plan_binding=plan_binding,
                    substrate_receipt=substrate_receipt,
                    schedule_audit=schedule_audit,
                    metric_vectors=_snapshot_metric_vectors(
                        validation_vectors,
                        arm_name=arm_name,
                        training_vectors=train_vectors,
                    ),
                ),
                attempt_root=attempt_root,
            )
            snapshot_bindings[arm_name].append(snapshot)
            measurement = measurement_payload(
                arm_name=arm_name,
                update=update,
                authority_binding=authority_binding,
                plan_binding=plan_binding,
                substrate_receipt=substrate_receipt,
                validation=analysis.validation_by_arm[arm_name],
                training=(
                    None
                    if training_by_arm is None
                    else training_by_arm[arm_name]
                ),
                loss=losses[arm_name],
                learning_rate=learning_rate,
            )
            binding = write_immutable_json(
                arms_root
                / arm_name
                / "measurements"
                / f"update_{update:06d}.json",
                measurement,
                attempt_root=attempt_root,
            )
            measurement_bindings[arm_name].append(binding)
        conditioned = analysis.validation_by_arm["conditioned"]
        print(
            json.dumps(
                {
                    "update": update,
                    "conditioned_energy": conditioned["factual_energy"]["mean"],
                    "c_vs_blind": conditioned["cross_arm"][
                        "conditioned_vs_blind_log_energy_advantage"
                    ],
                    "c_vs_shuffled": conditioned["cross_arm"][
                        "conditioned_vs_shuffled_log_energy_advantage"
                    ],
                    "nine_way_ba": conditioned["action_identification"][
                        "balanced_accuracy"
                    ],
                    "rank_ratio": conditioned["representation"][
                        "prediction_to_target_rank_ratio"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del validation_vectors, train_vectors
        torch.cuda.empty_cache()

    observe(0)
    for update in range(1, TRAINING_UPDATES + 1):
        update_losses, learning_rate = train_one_update(
            update=update,
            batch_rows_cpu=schedule[update - 1],
            substrate=substrate,
            arms=arms,
            optimizers=optimizers,
            partitions=partitions,
            train_frames=train_frames,
            train_actions=train_actions,
            shuffled_candidate_actions=shuffled_candidate_actions,
        )
        losses = dict(update_losses)
        assert_frozen_substrate_unchanged(
            substrate,
            encoder_sha256=substrate_receipt["encoder_sha256"],
            target_sha256=substrate_receipt["target_sha256"],
        )
        if time.monotonic() - gpu_started > float(authority["caps"]["maximum_gpu_seconds"]):
            raise TimeoutError("authorized GPU-phase ceiling exceeded")
        if update in OBSERVATION_UPDATES:
            observe(update)

    if training_by_arm is None or training_comparisons is None:
        raise ThreeArmWorkerError("update-700 training-fit panel is absent")
    final_analysis = validation_analyses[700]
    contract_checks = {
        "authority_exact": True,
        "reservation_exact": True,
        "source_closure_exact": True,
        "runtime_exact": True,
        "input_bindings_exact": True,
        "fresh_pack_exact": True,
        "schedule_exact": True,
        "derangement_exact": True,
        "shared_encoding_exact": True,
        "parameter_partition_exact": True,
        "optimizer_exact": True,
        "frozen_substrate_exact": True,
        "output_immutability_exact": True,
        "accounting_exact": True,
        "finiteness_exact": True,
        "validation_no_gradient_exact": True,
    }
    metric_decision = experiment_metrics.localize_three_arm_decision(
        train_point_advantages={
            control: comparison.macro_log_advantage
            for control, comparison in training_comparisons.items()
        },
        validation_tail_point_advantages={
            update: {
                control: validation_analyses[update].cross_arm[
                    control
                ].macro_log_advantage
                for control in ("blind", "shuffled")
            }
            for update in (500, 600, 700)
        },
        validation_comparisons=final_analysis.cross_arm,
        action_identification=final_analysis.action_identification["conditioned"],
        persistence_comparison=final_analysis.controls["conditioned"]["persistence"],
        wrong_history_comparison=final_analysis.controls["conditioned"]["wrong_history"],
        rank_ratio_by_update={
            update: validation_analyses[update].rank_ratio_by_arm["conditioned"]
            for update in (500, 600, 700)
        },
        encoder_identity_exact=True,
        target_identity_exact=True,
        contract_checks=contract_checks,
    )
    joint_decision = build_joint_decision(
        metric_decision=metric_decision,
        substrate_receipt=substrate_receipt,
        training_by_arm=training_by_arm,
        validation_analyses=validation_analyses,
    )

    scaled._assert_pack_bindings_unchanged(pack_root, pack_bindings)
    if file_binding(pack_root / "manifest.json", relative_to=attempt_root) != pack_binding:
        raise ThreeArmWorkerError("pack manifest changed during experiment")
    pack_artifact_bindings = build_pack_artifact_bindings(
        pack_root=pack_root,
        attempt_root=attempt_root,
    )
    _validate_required_source_closure(sources)
    _validate_input_bindings(authority)
    supervisor_contract._reverify_contract(authority)
    reservation_path = attempt_root / "reservation.json"
    if (
        supervisor_contract.file_binding(reservation_path) != reservation_binding
        or read_bound_json(
            reservation_path,
            expected_byte_count=reservation_binding["byte_count"],
            expected_sha256=reservation_binding["file_sha256"],
            label="terminal reservation",
        )
        != reservation
    ):
        raise ThreeArmWorkerError("attempt reservation changed")
    assert_frozen_substrate_unchanged(
        substrate,
        encoder_sha256=substrate_receipt["encoder_sha256"],
        target_sha256=substrate_receipt["target_sha256"],
    )

    torch.cuda.synchronize(device)
    gpu_elapsed = time.monotonic() - gpu_started
    wall_elapsed = time.monotonic() - wall_started
    if gpu_elapsed > float(authority["caps"]["maximum_gpu_seconds"]):
        raise TimeoutError("authorized GPU-phase ceiling exceeded at terminal sync")
    if wall_elapsed > float(authority["caps"]["maximum_wall_seconds"]):
        raise TimeoutError("authorized worker wall ceiling exceeded")
    # This is the worker-produced artifact inventory.  reservation.json was
    # created and bound by the external supervisor before this process began;
    # its binding lives under result.attempt.reservation rather than being
    # misreported as a worker output.  result.json cannot inventory itself
    # before its exclusive terminal write.
    output_inventory = sorted(
        [
            "pack/manifest.json",
            "overlap_audit.json",
            "shuffle_audit.json",
            *(
                relative
                for artifacts in PACK_ARTIFACT_RELATIVE_PATHS.values()
                for relative in artifacts.values()
            ),
        ]
        + [
            f"arms/{arm_name}/measurements/update_{update:06d}.json"
            for arm_name in ARM_NAMES
            for update in OBSERVATION_UPDATES
        ]
        + [
            f"arms/{arm_name}/snapshots/update_{update:06d}.pt"
            for arm_name in ARM_NAMES
            for update in OBSERVATION_UPDATES
        ]
    )
    result_attempt = {
        **authority["attempt"],
        "reservation": {
            "binding": dict(reservation_binding),
            "supervisor_nonce": reservation["supervisor_nonce"],
            "status": "RESERVED_ATTEMPT_CONSUMED",
            "maximum_attempts": 1,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        },
    }
    result = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE_PENDING_TERMINAL_REVIEW",
        "authority_binding": dict(authority_binding),
        "plan_binding": dict(plan_binding),
        "review_binding": authority["review_binding"],
        "source_commit": authority["source_commit"],
        "attempt": result_attempt,
        "caps": authority["caps"],
        "runtime": {
            "authorized": authority["runtime"],
            "observed": _observed_runtime(
                gpu_elapsed=gpu_elapsed,
                wall_elapsed=wall_elapsed,
                output_inventory=output_inventory,
            ),
        },
        "input_bindings": authority["input_bindings"],
        "predecessor_terminal_failure_binding": authority[
            "predecessor_terminal_failure_binding"
        ],
        "pack_binding": pack_binding,
        "pack_artifact_bindings": pack_artifact_bindings,
        "overlap_audit_binding": overlap_binding,
        "shuffle_audit_binding": shuffle_binding,
        "arms": {
            name: {
                "status": "COMPLETE",
                "measurement_bindings": measurement_bindings[name],
            }
            for name in ARM_NAMES
        },
        "joint_decision": joint_decision,
        "accounting": exact_accounting(),
        "forbidden_access": {
            "sealed_material_opened": False,
            "heldout_material_opened": False,
            "network_access_used": False,
            "validation_used_for_gradient_updates": False,
            "existing_pool_modified": False,
        },
        "checkpoint_bindings": snapshot_bindings,
    }
    write_immutable_json(
        attempt_root / "result.json", result, attempt_root=attempt_root
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_PENDING_TERMINAL_REVIEW",
                "joint_decision": metric_decision.status,
                "gpu_minutes": gpu_elapsed / 60.0,
                "wall_minutes": wall_elapsed / 60.0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    parser.add_argument("--expected-authority-sha256", required=True)
    args = parser.parse_args()
    authority_path = Path(args.authority).resolve()
    try:
        return execute_authorized_experiment(
            authority_path=authority_path,
            expected_authority_byte_count=args.expected_authority_byte_count,
            expected_authority_sha256=args.expected_authority_sha256,
        )
    except BaseException as error:
        # A reservation consumes the attempt.  If it exists, publish one
        # immutable source-diagnostic receipt without pretending to produce a
        # scientific result or authorizing a retry/resume.
        attempt_root = supervisor_contract.ATTEMPT_ROOT
        failure_path = attempt_root / "failure.json"
        if (
            attempt_root.is_dir()
            and not attempt_root.is_symlink()
            and not failure_path.exists()
            and not failure_path.is_symlink()
            and not (attempt_root / "result.json").exists()
        ):
            try:
                write_immutable_json(
                    failure_path,
                    {
                        "schema": FAILURE_SCHEMA,
                        "status": "ATTEMPT_CONSUMED_WORKER_FAILURE",
                        "authorizes_retry_or_resume": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    attempt_root=attempt_root,
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
