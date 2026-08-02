#!/usr/bin/env python3
"""Fresh multi-seed development comparison for planning-oriented Go2 JEPAs.

The runner compares a 2x2 panel of independently optimized heads over one
frozen spatial encoder and one fixed H6 pack:

``masked_plain``
    The current registered 64-token one-step objective.

``masked_delta``
    The same head plus an action-from-latent-displacement auxiliary objective.

``full_plain`` and ``full_delta``
    Complete 256-token counterparts that isolate full-grid supervision from
    the displacement auxiliary.  Their outputs are structurally re-entrant and
    can therefore be tested in blind rollouts if they survive matched-branch
    evaluation.

The displacement decoder is trained first, exclusively on true latent
transitions, and is then frozen.  Predictor training cannot therefore co-adapt
the decoder into an action watermark.  Direct matched branches remain the only
causal adjudicator.

This is development evidence only.  It opens no sealed or held-out role and it
does not select checkpoints from evaluation metrics.  The fixed terminal update
is always reported; matched simulator branches remain the primary adjudicator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as temporal_metrics,
)
from lewm.models.go2_world_model_progression_v1 import (  # noqa: E402
    SpatialLatentDisplacementActionDecoderV1,
    normalized_spatial_energy_v1,
    predict_dynamic_spatial_tokens_v1,
)
from lewm.models import (  # noqa: E402
    rgb_recurrent_patch_memory_temporal_jepa_v1 as temporal_model,
)
from scripts import dev_train_temporal_jepa_scaled as scaled  # noqa: E402
from scripts import execute_go2_world_model_existing_pool_three_arm_v1 as historical  # noqa: E402


SCHEMA = "dev_go2_world_model_progression_v1"
SNAPSHOT_SCHEMA = "lewm_go2_world_model_progression_v1_snapshot_v1"
ARM_NAMES = ("masked_plain", "masked_delta", "full_plain", "full_delta")
DELTA_ARM_NAMES = frozenset({"masked_delta", "full_delta"})
FULL_ARM_NAMES = frozenset({"full_plain", "full_delta"})
DEFAULT_PACK_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3"
    / "attempt_v1/pack"
)
DEFAULT_OUTPUT_PARENT = REPO_ROOT / ".generated/dev/world_model_progression_v1"
DEFAULT_SEEDS = (2026080201, 2026080202, 2026080203)
ACTION_AUXILIARY_WEIGHT = 0.1
DECODER_BASE_LR = 1.0e-4
ACTION_COUNT = 9
DEFAULT_DECODER_PRETRAIN_UPDATES = 300
DEFAULT_DECODER_ANCHOR_LOWER_BOUND = 1.0 / ACTION_COUNT
DECODER_ANCHOR_BOOTSTRAP_RESAMPLES = 2_000
DECODER_ANCHOR_BOOTSTRAP_SEED = 20260802
FULL_TOKEN_COUNT = 256
FEATURE_DIM = 192


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_binding(path: Path) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"source is a symlink: {raw}")
    selected = raw.resolve(strict=True)
    if not selected.is_file():
        raise ValueError(f"source is not a regular non-symlink file: {selected}")
    return {
        "path": selected.relative_to(REPO_ROOT).as_posix(),
        "byte_count": selected.stat().st_size,
        "sha256": _sha256_file(selected),
    }


def _require_development_output(path: Path) -> Path:
    selected = path.resolve(strict=False)
    parent = DEFAULT_OUTPUT_PARENT.resolve(strict=False)
    if selected.parent != parent or selected.name in {"", ".", ".."}:
        raise ValueError(f"output must be one fresh child of {parent}")
    parts = set(selected.parts)
    if "sealed" in parts or any(part.startswith("sealed_") for part in parts):
        raise ValueError("protected output path rejected")
    if selected.exists() or selected.is_symlink():
        raise FileExistsError(f"refusing to overwrite output {selected}")
    selected.mkdir(parents=True, exist_ok=False)
    return selected


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _paired_dropout_seed(seed: int, update: int, microbatch: int) -> int:
    payload = f"{SCHEMA}/{seed}/{update}/{microbatch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _load_predecessor() -> dict[str, torch.Tensor]:
    return historical.load_predecessor_state(
        {
            "path": str(historical.PREDECESSOR),
            "byte_count": historical.PREDECESSOR_BYTE_COUNT,
            "file_sha256": historical.PREDECESSOR_SHA256,
        }
    )


def _load_pack_scene_ids(
    pack_root: Path,
    *,
    role: str,
    role_binding: Mapping[str, Any],
) -> tuple[str, ...]:
    metadata = role_binding.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "path",
        "byte_count",
        "sha256",
    }:
        raise ValueError(f"{role} pack metadata binding changed")
    path = pack_root / str(metadata["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} pack metadata is not a regular file")
    raw = path.read_bytes()
    if (
        len(raw) != int(metadata["byte_count"])
        or hashlib.sha256(raw).hexdigest() != metadata["sha256"]
    ):
        raise ValueError(f"{role} pack metadata binding disagrees")
    value = json.loads(raw)
    if not isinstance(value, Mapping) or set(value) != {"scene_ids", "families"}:
        raise ValueError(f"{role} pack metadata fields changed")
    scene_ids = value["scene_ids"]
    families = value["families"]
    expected_rows = int(role_binding["frames"]["shape"][0])
    if (
        not isinstance(scene_ids, list)
        or not isinstance(families, list)
        or len(scene_ids) != expected_rows
        or len(families) != expected_rows
        or any(not isinstance(item, str) or not item for item in scene_ids)
        or any(not isinstance(item, str) or not item for item in families)
    ):
        raise ValueError(f"{role} pack metadata rows changed")
    return tuple(scene_ids)


def _build_substrate_and_arms(
    predecessor: Mapping[str, torch.Tensor],
    *,
    seed: int,
    device: torch.device,
) -> tuple[
    temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    dict[str, historical.ArmCore],
    SpatialLatentDisplacementActionDecoderV1,
    dict[str, torch.optim.AdamW],
    dict[str, tuple[nn.Parameter, ...]],
    dict[str, Any],
]:
    _set_seed(seed)
    substrate = temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1(predecessor).to(device)
    substrate.eval()
    substrate.requires_grad_(False)
    for parameter in substrate.parameters():
        parameter.grad = None

    template = historical.ArmCore(substrate).to(device)
    template_state = copy.deepcopy(template.state_dict())
    arms: dict[str, historical.ArmCore] = {}
    for name in ARM_NAMES:
        arm = historical.ArmCore(substrate).to(device)
        arm.load_state_dict(template_state, strict=True)
        arm.train()
        arms[name] = arm

    _set_seed(seed + 17)
    decoder = SpatialLatentDisplacementActionDecoderV1().to(device)
    decoder.train()

    optimizers: dict[str, torch.optim.AdamW] = {}
    trainable: dict[str, tuple[nn.Parameter, ...]] = {}
    for name, arm in arms.items():
        partition = historical.partition_arm_parameters(arm)
        groups: list[dict[str, Any]] = [
            {
                "group_name": "predictor",
                "params": list(partition.predictor),
                "lr": historical.PREDICTOR_BASE_LR * historical.LR_SCALE,
            },
            {
                "group_name": "memory",
                "params": list(partition.memory),
                "lr": historical.MEMORY_BASE_LR * historical.LR_SCALE,
            },
        ]
        parameters = list(partition.all)
        if len({id(value) for value in parameters}) != len(parameters):
            raise RuntimeError(f"{name} trainable parameter identity repeats")
        optimizers[name] = torch.optim.AdamW(
            groups,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=historical.WEIGHT_DECAY,
            amsgrad=False,
        )
        trainable[name] = tuple(parameters)

    indices, _ = temporal_metrics.batched_mask_indices("val", (0, 1), device=device)
    probe_history = torch.zeros((2, 3, FULL_TOKEN_COUNT, FEATURE_DIM), device=device)
    probe_actions = torch.tensor(((0, 1, 2), (8, 7, 6)), dtype=torch.long, device=device)
    for arm in arms.values():
        arm.eval()
    with torch.no_grad():
        registered = historical.predict_from_shared_encoding(
            arms["masked_plain"],
            probe_history,
            probe_actions,
            indices,
            candidate_blind=False,
        ).raw
        dynamic = predict_dynamic_spatial_tokens_v1(
            arms["masked_plain"],
            probe_history,
            probe_actions,
            indices,
        ).raw
    parity_error = float((registered - dynamic).abs().max())
    if parity_error != 0.0:
        raise RuntimeError("dynamic 64-token route diverged from the registered predictor")
    for arm in arms.values():
        arm.train()

    initial_hashes = {
        name: historical.module_state_sha256(arm) for name, arm in arms.items()
    }
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("comparison arms did not receive identical core initialization")
    all_parameter_ids = {
        name: {id(parameter) for parameter in parameters}
        for name, parameters in trainable.items()
    }
    for left_index, left in enumerate(ARM_NAMES):
        for right in ARM_NAMES[left_index + 1 :]:
            if all_parameter_ids[left] & all_parameter_ids[right]:
                raise RuntimeError("comparison arms share a trainable parameter")
    return substrate, arms, decoder, optimizers, trainable, {
        "seed": seed,
        "core_initial_sha256": initial_hashes,
        "dynamic_registered_parity_max_abs_error": parity_error,
        "encoder_sha256": historical.module_state_sha256(substrate.encoder),
        "target_encoder_sha256": historical.module_state_sha256(substrate.target_encoder),
        "decoder_initial_sha256": historical.module_state_sha256(decoder),
    }


def _set_learning_rates(optimizer: torch.optim.AdamW, update: int) -> None:
    fraction = historical.learning_rate_fraction(update)
    expected = {
        "predictor": historical.PREDICTOR_BASE_LR * historical.LR_SCALE,
        "memory": historical.MEMORY_BASE_LR * historical.LR_SCALE,
    }
    for group in optimizer.param_groups:
        name = group.get("group_name")
        if name not in expected:
            raise RuntimeError("optimizer group identity changed")
        group["lr"] = expected[name] * fraction


def _next_batch(
    *,
    generator: torch.Generator,
    order: torch.Tensor,
    cursor: int,
    row_count: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if cursor + batch_size <= row_count:
        return order[cursor : cursor + batch_size], order, cursor + batch_size
    tail = order[cursor:]
    fresh = torch.randperm(row_count, generator=generator)
    needed = batch_size - len(tail)
    return torch.cat((tail, fresh[:needed])), fresh, needed


def _encode_batch(
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    frames: torch.Tensor,
    row_ids_cpu: torch.Tensor,
    *,
    role: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = frames.shape[0]
    target_indices, _ = temporal_metrics.batched_mask_indices(
        role,
        row_ids_cpu.tolist(),
        device=frames.device,
    )
    with torch.no_grad():
        encoded = substrate.encode_online_full_frame(
            frames[:, :3].reshape(batch * 3, 3, 112, 112)
        ).reshape(batch, 3, FULL_TOKEN_COUNT, FEATURE_DIM)
        target_full = substrate.encode_target_full_frame(frames[:, 3])
        target_masked = target_full.gather(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, FEATURE_DIM),
        )
    return encoded.detach(), target_full.detach(), target_masked.detach(), target_indices


def _arm_prediction(
    name: str,
    arm: historical.ArmCore,
    encoded: torch.Tensor,
    actions: torch.Tensor,
    target_indices: torch.Tensor,
):
    if name in FULL_ARM_NAMES:
        full_indices = torch.arange(
            FULL_TOKEN_COUNT, dtype=torch.long, device=encoded.device
        ).unsqueeze(0).expand(encoded.shape[0], -1)
        return predict_dynamic_spatial_tokens_v1(
            arm,
            encoded,
            actions,
            full_indices,
        ), full_indices
    return (
        predict_dynamic_spatial_tokens_v1(
            arm,
            encoded,
            actions,
            target_indices,
        ),
        target_indices,
    )


def _pretrain_true_delta_decoder(
    *,
    seed: int,
    updates: int,
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    decoder: SpatialLatentDisplacementActionDecoderV1,
    train_frames: torch.Tensor,
    train_actions: torch.Tensor,
    batch_size: int,
    microbatch_size: int,
    trace_every: int,
) -> list[dict[str, float]]:
    """Anchor one decoder on true transitions before predictor optimization.

    Masked and complete-grid panels contribute equally.  After this phase the
    decoder parameters are frozen permanently; gradients through the frozen
    decoder may still reach a predicted future during the comparison phase.
    """

    if updates < 1:
        raise ValueError("decoder pretraining requires at least one update")
    decoder.train()
    decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=DECODER_BASE_LR,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=historical.WEIGHT_DECAY,
        amsgrad=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 29)
    order = torch.randperm(len(train_frames), generator=generator)
    cursor = 0
    trace: list[dict[str, float]] = []
    for update in range(1, updates + 1):
        rows_cpu, order, cursor = _next_batch(
            generator=generator,
            order=order,
            cursor=cursor,
            row_count=len(train_frames),
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        total_masked = 0.0
        total_full = 0.0
        for start in range(0, batch_size, microbatch_size):
            micro_cpu = rows_cpu[start : start + microbatch_size]
            micro = micro_cpu.to(device=train_frames.device)
            frames = scaled.to_float(train_frames[micro])
            encoded, target_full, target_masked, target_indices = _encode_batch(
                substrate, frames, micro_cpu, role="train"
            )
            labels = train_actions[micro, -1]
            current_full = encoded[:, -1]
            current_masked = current_full.gather(
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, FEATURE_DIM),
            )
            full_indices = torch.arange(
                FULL_TOKEN_COUNT, dtype=torch.long, device=frames.device
            ).unsqueeze(0).expand(len(micro_cpu), -1)
            masked_ce = F.cross_entropy(
                decoder(current_masked, target_masked, target_indices), labels
            )
            full_ce = F.cross_entropy(
                decoder(current_full, target_full, full_indices), labels
            )
            loss = 0.5 * (masked_ce + full_ce)
            scale = len(micro_cpu) / batch_size
            (loss * scale).backward()
            total_masked += float(masked_ce.detach()) * scale
            total_full += float(full_ce.detach()) * scale
        norm = torch.nn.utils.clip_grad_norm_(
            tuple(decoder.parameters()), historical.GRADIENT_CLIP
        )
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("decoder pretraining gradient became nonfinite")
        optimizer.step()
        if update == 1 or update % trace_every == 0 or update == updates:
            trace.append(
                {
                    "update": float(update),
                    "masked_true_delta_cross_entropy": total_masked,
                    "full_true_delta_cross_entropy": total_full,
                    "pre_clip_gradient_norm": float(norm),
                }
            )
    optimizer.zero_grad(set_to_none=True)
    decoder.eval()
    decoder.requires_grad_(False)
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise RuntimeError("frozen decoder retained a gradient")
    return trace


def _train_update(
    *,
    seed: int,
    update: int,
    batch_rows_cpu: torch.Tensor,
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    arms: Mapping[str, historical.ArmCore],
    decoder: SpatialLatentDisplacementActionDecoderV1,
    optimizers: Mapping[str, torch.optim.AdamW],
    trainable: Mapping[str, tuple[nn.Parameter, ...]],
    train_frames: torch.Tensor,
    train_actions: torch.Tensor,
    batch_size: int,
    microbatch_size: int,
) -> dict[str, dict[str, float]]:
    if batch_rows_cpu.shape != (batch_size,):
        raise ValueError("training batch shape changed")
    for optimizer in optimizers.values():
        _set_learning_rates(optimizer, update)
        optimizer.zero_grad(set_to_none=True)
    totals = {
        name: {"total": 0.0, "jepa": 0.0, "predicted_ce": 0.0}
        for name in ARM_NAMES
    }

    for start in range(0, batch_size, microbatch_size):
        micro_cpu = batch_rows_cpu[start : start + microbatch_size]
        micro = micro_cpu.to(device=train_frames.device)
        frames = scaled.to_float(train_frames[micro])
        encoded, target_full, target_masked, target_indices = _encode_batch(
            substrate, frames, micro_cpu, role="train"
        )
        actions = train_actions[micro]
        labels = actions[:, -1]
        scale = len(micro_cpu) / batch_size

        for name in ARM_NAMES:
            if name in {"masked_plain", "masked_delta"}:
                dropout_seed = _paired_dropout_seed(seed, update, start)
            else:
                dropout_seed = _paired_dropout_seed(seed + 10_000, update, start)
            _set_seed(dropout_seed)
            prediction, prediction_indices = _arm_prediction(
                name, arms[name], encoded, actions, target_indices
            )
            if name in FULL_ARM_NAMES:
                target = target_full
                current = encoded[:, -1]
            else:
                target = target_masked
                current = encoded[:, -1].gather(
                    1,
                    target_indices.unsqueeze(-1).expand(-1, -1, FEATURE_DIM),
                )
            jepa_loss = normalized_spatial_energy_v1(prediction.raw, target).mean()
            predicted_ce = torch.zeros((), device=frames.device)
            total = jepa_loss
            if name in DELTA_ARM_NAMES:
                predicted_logits = decoder(
                    current, prediction.raw, prediction_indices
                )
                predicted_ce = F.cross_entropy(predicted_logits, labels)
                total = total + ACTION_AUXILIARY_WEIGHT * predicted_ce
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"{name} loss became nonfinite")
            (total * scale).backward()
            totals[name]["total"] += float(total.detach()) * scale
            totals[name]["jepa"] += float(jepa_loss.detach()) * scale
            totals[name]["predicted_ce"] += float(predicted_ce.detach()) * scale

    for name in ARM_NAMES:
        parameters = trainable[name]
        if not any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError(f"{name} produced no gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, historical.GRADIENT_CLIP)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError(f"{name} gradient norm became nonfinite")
        totals[name]["pre_clip_core_gradient_norm"] = float(norm)
        optimizers[name].step()
        optimizers[name].zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise RuntimeError("frozen decoder accumulated parameter gradients")
    return totals


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for action in range(ACTION_COUNT):
        selected = labels == action
        if not selected.any():
            raise ValueError("evaluation lost an action class")
        recalls.append(float((predictions[selected] == action).mean()))
    return float(np.mean(recalls))


def _scene_clustered_balanced_accuracy_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    scene_ids: Sequence[str],
    *,
    resamples: int = DECODER_ANCHOR_BOOTSTRAP_RESAMPLES,
    seed: int = DECODER_ANCHOR_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap balanced accuracy by resampling whole scene clusters."""

    if (
        labels.shape != predictions.shape
        or labels.ndim != 1
        or len(scene_ids) != len(labels)
        or resamples < 1
    ):
        raise ValueError("balanced-accuracy interval inputs are invalid")
    scenes = sorted(set(scene_ids))
    if not scenes or any(not isinstance(scene, str) or not scene for scene in scenes):
        raise ValueError("balanced-accuracy scene clusters are invalid")
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    correct = np.zeros((len(scenes), ACTION_COUNT), dtype=np.int64)
    total = np.zeros((len(scenes), ACTION_COUNT), dtype=np.int64)
    for row, (label, prediction) in enumerate(zip(labels, predictions, strict=True)):
        if not 0 <= int(label) < ACTION_COUNT:
            raise ValueError("balanced-accuracy label left the action vocabulary")
        cluster = scene_index[scene_ids[row]]
        total[cluster, int(label)] += 1
        correct[cluster, int(label)] += int(prediction == label)
    if bool((total.sum(axis=0) == 0).any()):
        raise ValueError("balanced-accuracy interval lost an action class")
    rng = np.random.default_rng(seed)
    sampled_scenes = rng.integers(
        0, len(scenes), size=(resamples, len(scenes))
    )
    sampled_correct = correct[sampled_scenes].sum(axis=1)
    sampled_total = total[sampled_scenes].sum(axis=1)
    valid = (sampled_total > 0).all(axis=1)
    if int(valid.sum()) < max(100, int(0.95 * resamples)):
        raise ValueError("too many scene bootstrap draws lost a rare action")
    draws = (sampled_correct[valid] / sampled_total[valid]).mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return {
        "point": _balanced_accuracy(labels, predictions),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "requested_resamples": resamples,
        "valid_resamples": int(valid.sum()),
        "seed": seed,
        "scene_clusters": len(scenes),
    }


@torch.no_grad()
def _evaluate(
    *,
    substrate: temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1,
    arms: Mapping[str, historical.ArmCore],
    decoder: SpatialLatentDisplacementActionDecoderV1,
    frames: torch.Tensor,
    actions: torch.Tensor,
    scene_ids: Sequence[str],
    batch_size: int,
    role: str,
) -> dict[str, dict[str, Any]]:
    if len(scene_ids) != len(frames):
        raise ValueError("evaluation scene identities do not align with frames")
    for arm in arms.values():
        arm.eval()
    decoder.eval()
    result: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {
            "candidate_energy": [],
            "persistence_energy": [],
            "full_grid_candidate_energy": [],
            "full_grid_persistence_energy": [],
            "true_decoder_logits": [],
            "predicted_decoder_logits": [],
            "labels": [],
        }
        for name in ARM_NAMES
    }

    for start in range(0, len(frames), batch_size):
        end = min(len(frames), start + batch_size)
        row_ids_cpu = torch.arange(start, end, dtype=torch.long)
        rows = row_ids_cpu.to(device=frames.device)
        normalized = scaled.to_float(frames[rows])
        encoded, target_full, target_masked, target_indices = _encode_batch(
            substrate, normalized, row_ids_cpu, role=role
        )
        factual_actions = actions[rows]
        labels = factual_actions[:, -1]
        current_masked = encoded[:, -1].gather(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, FEATURE_DIM),
        )
        persistence = normalized_spatial_energy_v1(
            current_masked, target_masked
        ).cpu()
        persistence_full = normalized_spatial_energy_v1(
            encoded[:, -1], target_full
        ).cpu()

        for name in ARM_NAMES:
            candidates = []
            full_grid_candidates = []
            factual_prediction = None
            prediction_indices = None
            for candidate in range(ACTION_COUNT):
                intervention = factual_actions.clone()
                intervention[:, -1] = candidate
                prediction, selected_indices = _arm_prediction(
                    name, arms[name], encoded, intervention, target_indices
                )
                if name in FULL_ARM_NAMES:
                    # The registered 64-token mask is the common comparison
                    # surface across all arms.  Keep complete-grid fidelity as
                    # a separate diagnostic instead of silently comparing a
                    # 256-token energy to the other arms' 64-token energy.
                    selected_prediction = prediction.raw.gather(
                        1,
                        target_indices.unsqueeze(-1).expand(-1, -1, FEATURE_DIM),
                    )
                    selected_target = target_masked
                    full_grid_candidates.append(
                        normalized_spatial_energy_v1(
                            prediction.raw, target_full
                        ).cpu()
                    )
                else:
                    selected_prediction = prediction.raw
                    selected_target = target_masked
                candidates.append(
                    normalized_spatial_energy_v1(
                        selected_prediction, selected_target
                    ).cpu()
                )
                factual_rows = labels == candidate
                if bool(factual_rows.any()):
                    if factual_prediction is None:
                        factual_prediction = torch.empty_like(prediction.raw)
                    factual_prediction[factual_rows] = prediction.raw[factual_rows]
                prediction_indices = selected_indices
            result[name]["candidate_energy"].append(torch.stack(candidates, dim=1))
            result[name]["persistence_energy"].append(persistence)
            if full_grid_candidates:
                result[name]["full_grid_candidate_energy"].append(
                    torch.stack(full_grid_candidates, dim=1)
                )
                result[name]["full_grid_persistence_energy"].append(persistence_full)
            result[name]["labels"].append(labels.cpu())
            if name in DELTA_ARM_NAMES:
                if factual_prediction is None or prediction_indices is None:
                    raise RuntimeError("factual prediction assembly failed")
                current = encoded[:, -1] if name in FULL_ARM_NAMES else current_masked
                target = target_full if name in FULL_ARM_NAMES else target_masked
                result[name]["true_decoder_logits"].append(
                    decoder(current, target, prediction_indices).cpu()
                )
                result[name]["predicted_decoder_logits"].append(
                    decoder(current, factual_prediction, prediction_indices).cpu()
                )

    summaries: dict[str, dict[str, Any]] = {}
    for name in ARM_NAMES:
        candidate = torch.cat(result[name]["candidate_energy"]).numpy().astype(np.float64)
        persistence_values = torch.cat(result[name]["persistence_energy"]).numpy().astype(np.float64)
        labels = torch.cat(result[name]["labels"]).numpy().astype(np.int64)
        factual = candidate[np.arange(len(labels)), labels]
        wrong = candidate.copy()
        wrong[np.arange(len(labels)), labels] = math.inf
        hardest_margin = wrong.min(axis=1) - factual
        predictions = candidate.argmin(axis=1)
        summary: dict[str, Any] = {
            "row_count": int(len(labels)),
            "factual_energy_mean": float(factual.mean()),
            "persistence_energy_mean": float(persistence_values.mean()),
            "persistence_advantage_mean": float((persistence_values - factual).mean()),
            "hardest_wrong_action_margin_mean": float(hardest_margin.mean()),
            "hardest_wrong_action_margin_q05": float(np.quantile(hardest_margin, 0.05)),
            "nine_way_action_balanced_accuracy": _balanced_accuracy(labels, predictions),
            "candidate_energy_spread_mean": float(candidate.std(axis=1).mean()),
            "per_action": {},
        }
        for action in range(ACTION_COUNT):
            selected = labels == action
            summary["per_action"][str(action)] = {
                "rows": int(selected.sum()),
                "hardest_margin_mean": float(hardest_margin[selected].mean()),
                "persistence_advantage_mean": float(
                    (persistence_values[selected] - factual[selected]).mean()
                ),
                "recall": float((predictions[selected] == action).mean()),
            }
        if name in DELTA_ARM_NAMES:
            true_logits = torch.cat(result[name]["true_decoder_logits"]).numpy()
            predicted_logits = torch.cat(result[name]["predicted_decoder_logits"]).numpy()
            true_predictions = true_logits.argmax(axis=1)
            predicted_predictions = predicted_logits.argmax(axis=1)
            summary["true_delta_decoder_balanced_accuracy"] = _balanced_accuracy(
                labels, true_predictions
            )
            summary["predicted_delta_decoder_balanced_accuracy"] = _balanced_accuracy(
                labels, predicted_predictions
            )
            summary["true_delta_decoder_balanced_accuracy_interval"] = (
                _scene_clustered_balanced_accuracy_interval(
                    labels, true_predictions, scene_ids
                )
            )
            summary["predicted_delta_decoder_balanced_accuracy_interval"] = (
                _scene_clustered_balanced_accuracy_interval(
                    labels,
                    predicted_predictions,
                    scene_ids,
                    seed=DECODER_ANCHOR_BOOTSTRAP_SEED + 1,
                )
            )
        if result[name]["full_grid_candidate_energy"]:
            full_candidate = (
                torch.cat(result[name]["full_grid_candidate_energy"])
                .numpy()
                .astype(np.float64)
            )
            full_persistence = (
                torch.cat(result[name]["full_grid_persistence_energy"])
                .numpy()
                .astype(np.float64)
            )
            full_factual = full_candidate[np.arange(len(labels)), labels]
            full_wrong = full_candidate.copy()
            full_wrong[np.arange(len(labels)), labels] = math.inf
            full_margin = full_wrong.min(axis=1) - full_factual
            summary["full_grid_diagnostics"] = {
                "factual_energy_mean": float(full_factual.mean()),
                "persistence_energy_mean": float(full_persistence.mean()),
                "persistence_advantage_mean": float(
                    (full_persistence - full_factual).mean()
                ),
                "hardest_wrong_action_margin_mean": float(full_margin.mean()),
                "hardest_wrong_action_margin_q05": float(
                    np.quantile(full_margin, 0.05)
                ),
                "nine_way_action_balanced_accuracy": _balanced_accuracy(
                    labels, full_candidate.argmin(axis=1)
                ),
            }
        summaries[name] = summary
    for arm in arms.values():
        arm.train()
    decoder.eval()
    return summaries


def _snapshot(
    *,
    path: Path,
    name: str,
    seed: int,
    update: int,
    arm: historical.ArmCore,
    decoder: SpatialLatentDisplacementActionDecoderV1 | None,
    metrics: Mapping[str, Any],
) -> None:
    arm_state = {
        key: value.detach().cpu().clone() for key, value in arm.state_dict().items()
    }
    decoder_state = (
        {
            key: value.detach().cpu().clone()
            for key, value in decoder.state_dict().items()
        }
        if decoder is not None
        else None
    )
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "COMPLETE",
        "development_only": True,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "arm": name,
        "seed": seed,
        "update": update,
        "full_grid_training": name in FULL_ARM_NAMES,
        "action_auxiliary_weight": ACTION_AUXILIARY_WEIGHT if decoder is not None else 0.0,
        "metrics": dict(metrics),
        "arm_state_dict": arm_state,
        "decoder_state_dict": decoder_state,
    }
    torch.save(payload, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("the bound ROCm device is unavailable")
    if args.updates < 1 or args.batch_size < 1 or args.microbatch_size < 1:
        raise ValueError("update and batch sizes must be positive")
    if args.batch_size % args.microbatch_size != 0:
        raise ValueError("batch size must be divisible by microbatch size")
    if args.decoder_pretrain_updates < 1 or args.decoder_trace_every < 1:
        raise ValueError("decoder pretraining and trace intervals must be positive")
    if not 0.0 <= args.minimum_decoder_anchor_lower_bound <= 1.0:
        raise ValueError("decoder anchor accuracy must lie in [0, 1]")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    source_paths = (
        Path(__file__),
        REPO_ROOT / "lewm/models/go2_world_model_progression_v1.py",
        Path(historical.__file__),
        Path(scaled.__file__),
        Path(temporal_model.__file__),
        Path(temporal_metrics.__file__),
    )
    source_bindings = [_source_binding(path) for path in source_paths]
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    predecessor = _load_predecessor()
    train_frames, train_actions, train_binding = scaled.load_pack(
        args.pack_root, "train", device
    )
    val_frames, val_actions, val_binding = scaled.load_pack(
        args.pack_root, "val", device
    )
    val_scene_ids = _load_pack_scene_ids(
        args.pack_root, role="val", role_binding=val_binding
    )
    if args.batch_size > len(train_frames):
        raise ValueError("batch size exceeds training rows")
    if args.eval_rows < 0 or args.eval_rows > len(val_frames):
        raise ValueError("eval rows must be zero or no greater than validation rows")
    if args.eval_rows:
        val_frames = val_frames[: args.eval_rows]
        val_actions = val_actions[: args.eval_rows]
        val_scene_ids = val_scene_ids[: args.eval_rows]
    output = _require_development_output(args.output)

    started = time.monotonic()
    seed_results: dict[str, Any] = {}
    for seed in args.seeds:
        seed_started = time.monotonic()
        seed_root = output / f"seed_{seed}"
        seed_root.mkdir()
        substrate, arms, decoder, optimizers, trainable, build_receipt = (
            _build_substrate_and_arms(predecessor, seed=seed, device=device)
        )
        pretraining_trace = _pretrain_true_delta_decoder(
            seed=seed,
            updates=args.decoder_pretrain_updates,
            substrate=substrate,
            decoder=decoder,
            train_frames=train_frames,
            train_actions=train_actions,
            batch_size=args.batch_size,
            microbatch_size=args.microbatch_size,
            trace_every=args.decoder_trace_every,
        )
        build_receipt["decoder_frozen_sha256"] = historical.module_state_sha256(
            decoder
        )
        update_zero = _evaluate(
            substrate=substrate,
            arms=arms,
            decoder=decoder,
            frames=val_frames,
            actions=val_actions,
            scene_ids=val_scene_ids,
            batch_size=args.eval_batch_size,
            role="val",
        )
        decoder_anchor = {
            "masked": update_zero["masked_delta"][
                "true_delta_decoder_balanced_accuracy_interval"
            ],
            "full": update_zero["full_delta"][
                "true_delta_decoder_balanced_accuracy_interval"
            ],
        }
        if min(value["lower_95"] for value in decoder_anchor.values()) <= (
            args.minimum_decoder_anchor_lower_bound
        ):
            raise RuntimeError(
                "true-delta decoder failed the predeclared lower-bound anchor: "
                f"{decoder_anchor} <= {args.minimum_decoder_anchor_lower_bound}"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(len(train_frames), generator=generator)
        cursor = 0
        trace = []
        final_losses: Mapping[str, Any] | None = None
        for update in range(1, args.updates + 1):
            batch_rows, order, cursor = _next_batch(
                generator=generator,
                order=order,
                cursor=cursor,
                row_count=len(train_frames),
                batch_size=args.batch_size,
            )
            final_losses = _train_update(
                seed=seed,
                update=update,
                batch_rows_cpu=batch_rows,
                substrate=substrate,
                arms=arms,
                decoder=decoder,
                optimizers=optimizers,
                trainable=trainable,
                train_frames=train_frames,
                train_actions=train_actions,
                batch_size=args.batch_size,
                microbatch_size=args.microbatch_size,
            )
            if update == 1 or update % args.trace_every == 0 or update == args.updates:
                trace.append({"update": update, "losses": final_losses})
        final_metrics = _evaluate(
            substrate=substrate,
            arms=arms,
            decoder=decoder,
            frames=val_frames,
            actions=val_actions,
            scene_ids=val_scene_ids,
            batch_size=args.eval_batch_size,
            role="val",
        )
        terminal_decoder_hash = historical.module_state_sha256(decoder)
        if terminal_decoder_hash != build_receipt["decoder_frozen_sha256"]:
            raise RuntimeError("frozen true-delta decoder changed during predictor training")
        if not args.skip_snapshots:
            for name in ARM_NAMES:
                _snapshot(
                    path=seed_root / f"{name}_update_{args.updates:06d}.pt",
                    name=name,
                    seed=seed,
                    update=args.updates,
                    arm=arms[name],
                    decoder=decoder if name in DELTA_ARM_NAMES else None,
                    metrics=final_metrics[name],
                )
        seed_results[str(seed)] = {
            "build": build_receipt,
            "decoder_pretraining_trace": pretraining_trace,
            "decoder_anchor_balanced_accuracy": decoder_anchor,
            "update_zero": update_zero,
            "terminal": final_metrics,
            "terminal_losses": final_losses,
            "training_trace": trace,
            "terminal_core_sha256": {
                name: historical.module_state_sha256(arm)
                for name, arm in arms.items()
            },
            "terminal_decoder_sha256": terminal_decoder_hash,
            "wall_seconds": time.monotonic() - seed_started,
        }
        del substrate, arms, decoder, optimizers, trainable
        torch.cuda.empty_cache()

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_COMPARISON",
        "citable_as_scientific_evidence": False,
        "protected_material_opened": False,
        "configuration": {
            "arms": list(ARM_NAMES),
            "seeds": list(args.seeds),
            "updates": args.updates,
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "action_auxiliary_weight": ACTION_AUXILIARY_WEIGHT,
            "decoder_pretrain_updates": args.decoder_pretrain_updates,
            "minimum_decoder_anchor_lower_bound": (
                args.minimum_decoder_anchor_lower_bound
            ),
            "decoder_anchor_bootstrap_resamples": (
                DECODER_ANCHOR_BOOTSTRAP_RESAMPLES
            ),
            "decoder_frozen_during_predictor_training": True,
            "strict_deterministic_algorithms": True,
            "checkpoint_selection": "fixed_terminal_update_only",
            "evaluation_rows": len(val_frames),
            "snapshots_written": not args.skip_snapshots,
        },
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "wall_seconds": time.monotonic() - started,
        },
        "inputs": {
            "predecessor": {
                "path": str(historical.PREDECESSOR),
                "byte_count": historical.PREDECESSOR_BYTE_COUNT,
                "sha256": historical.PREDECESSOR_SHA256,
            },
            "pack_root": str(args.pack_root.resolve()),
            "train": train_binding,
            "val": val_binding,
        },
        "source_bindings": source_bindings,
        "seed_results": seed_results,
    }
    if [_source_binding(path) for path in source_paths] != source_bindings:
        raise RuntimeError("progression source closure changed during execution")
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--updates", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--decoder-pretrain-updates",
        type=int,
        default=DEFAULT_DECODER_PRETRAIN_UPDATES,
    )
    parser.add_argument("--decoder-trace-every", type=int, default=50)
    parser.add_argument(
        "--minimum-decoder-anchor-lower-bound",
        type=float,
        default=DEFAULT_DECODER_ANCHOR_LOWER_BOUND,
    )
    parser.add_argument(
        "--eval-rows",
        type=int,
        default=0,
        help="development smoke only: evaluate a validation prefix; zero uses all rows",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="development smoke only: do not write terminal model snapshots",
    )
    parser.add_argument("--trace-every", type=int, default=100)
    args = parser.parse_args()
    result = run(args)
    terminal = {
        seed: {
            arm: metrics["nine_way_action_balanced_accuracy"]
            for arm, metrics in payload["terminal"].items()
        }
        for seed, payload in result["seed_results"].items()
    }
    print(json.dumps({"status": result["status"], "terminal_balanced_accuracy": terminal}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
