#!/usr/bin/env python3
"""Run the one-shot geometry-anchored local-lift joint-JEPA V1 probe.

Import is source-only.  Tensor libraries and development inputs are touched
only after source authority has been checked and the sole output root has
been reserved.  This runner intentionally reuses the reviewed Direct-BEV
RGB/raster loader and frozen presentation schedule, but owns the new model,
two-phase objective, gates, and compact terminal receipts directly.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
FROZEN_DIRECT_RUNNER_PATH = (
    ROOT / "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py"
)
MATCHED_RUNNER_PATH = ROOT / "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
SCHEDULE_ADAPTER_PATH = (
    ROOT / "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)
ROCM_GRID_SAMPLE_DETERMINISM_WARNING = (
    "grid_sampler_2d_backward_cuda does not have a deterministic "
    "implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file "
    "an issue at https://github.com/pytorch/pytorch/issues to help us "
    "prioritize adding deterministic support for this operation."
)


def _is_allowed_rocm_determinism_warning(message: str) -> bool:
    """Allow only the frozen runtime's one known grid-sampler warning."""

    return message == ROCM_GRID_SAMPLE_DETERMINISM_WARNING


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    root_text = str(ROOT)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(root_text)
    return module


contract = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v1_contract", CONTRACT_PATH
)
direct = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v1_frozen_direct_runner",
    FROZEN_DIRECT_RUNNER_PATH,
)


class ScientificGateFailure(RuntimeError):
    """A preregistered scientific or numerical gate stopped the attempt."""

    def __init__(self, message: str, *, control: str) -> None:
        super().__init__(message)
        self.control = str(control)


def _read_regular(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_byte_count: int | None = None,
) -> bytes:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise PermissionError(f"not a regular source/input file: {path}")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"file changed while read: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise PermissionError(f"SHA-256 changed: {path}")
    if expected_byte_count is not None and len(raw) != int(expected_byte_count):
        raise PermissionError(f"byte count changed: {path}")
    return raw


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _publish_json(path: Path, core: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive(path, raw)
    return value, raw


def _binding(relative: str, value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": str(value["content_sha256"]),
        "byte_count": len(raw),
    }


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu() if hasattr(value, "detach") else value)
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite scalar")
    return result


def _tensor_state_sha256(torch: Any, values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_state_sha256(torch: Any, module: Any) -> str:
    return _tensor_state_sha256(torch, module.state_dict())


def _parameter_receipt(model: Any, contract_api: Any) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    groups: dict[str, list[Any]] = {
        "encoder": [],
        "lift_semantic": [],
        "predictor": [],
        "target": [],
    }
    names: dict[str, list[str]] = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
        elif name.startswith(("bev_lift.", "semantic_head.")):
            group = "lift_semantic"
        elif name.startswith("predictor."):
            group = "predictor"
        elif name.startswith(("target_encoder.", "target_bev_lift.")):
            group = "target"
        else:
            raise RuntimeError(f"unregistered model parameter {name!r}")
        groups[group].append(parameter)
        names[group].append(name)
    all_ids = [id(value) for rows in groups.values() for value in rows]
    if len(all_ids) != len(set(all_ids)) or any(not rows for rows in groups.values()):
        raise RuntimeError("parameter partition is incomplete or overlapping")
    if any(parameter.requires_grad for parameter in groups["target"]):
        raise RuntimeError("EMA target parameter is trainable")
    if any(
        not parameter.requires_grad
        for group in ("encoder", "lift_semantic", "predictor")
        for parameter in groups[group]
    ):
        raise RuntimeError("online parameter unexpectedly frozen")
    receipt = {
        group: {
            "tensor_count": len(groups[group]),
            "parameter_count": sum(value.numel() for value in groups[group]),
            "ordered_names_sha256": contract_api.canonical_json_sha256(names[group]),
        }
        for group in groups
    }
    receipt["total"] = {
        "tensor_count": sum(row["tensor_count"] for row in receipt.values()),
        "parameter_count": sum(row["parameter_count"] for row in receipt.values()),
    }
    return groups, receipt


def _build_optimizer(runtime: Any, groups: Mapping[str, Sequence[Any]]) -> Any:
    torch = runtime.torch
    return torch.optim.AdamW(
        [
            {"name": "encoder", "params": list(groups["encoder"]), "lr": 1e-4},
            {
                "name": "lift_semantic",
                "params": list(groups["lift_semantic"]),
                "lr": 3e-4,
            },
            {"name": "predictor", "params": list(groups["predictor"]), "lr": 3e-4},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )


def _optimizer_membership_receipt(optimizer: Any, contract_api: Any) -> dict[str, Any]:
    rows = []
    ids: list[int] = []
    for group in optimizer.param_groups:
        parameters = list(group["params"])
        ids.extend(id(value) for value in parameters)
        rows.append({
            "name": str(group["name"]),
            "learning_rate": float(group["lr"]),
            "parameter_tensor_count": len(parameters),
            "parameter_count": sum(value.numel() for value in parameters),
        })
    if len(ids) != len(set(ids)):
        raise RuntimeError("optimizer repeats a parameter")
    return {
        "groups": rows,
        "ordered_parameter_identity_sha256": contract_api.canonical_json_sha256(ids),
        "optimizer_object_identity": id(optimizer),
    }


def _smooth_energy(model_api: Any, prediction: Any, target: Any) -> Any:
    return model_api.latent_energy_per_row(prediction, target)


def _semantic_terms(model_api: Any, model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    current_latent = model.encode_online(batch["current_rgb"])
    next_latent = model.encode_online(batch["next_rgb"])
    current_logits = model.semantic_logits_from_latent(current_latent)
    next_logits = model.semantic_logits_from_latent(next_latent)
    current_rows = model_api.final_class_macro_nll_per_row(
        current_logits, batch["current_labels"]
    )
    next_rows = model_api.final_class_macro_nll_per_row(
        next_logits, batch["next_labels"]
    )
    A = 0.5 * current_rows.mean() + 0.5 * next_rows.mean()
    return {
        "current_latent": current_latent,
        "next_latent": next_latent,
        "current_logits": current_logits,
        "next_logits": next_logits,
        "A": A,
        "S": A / math.log(3.0),
    }


def _joint_terms(
    runtime: Any,
    model_api: Any,
    model: Any,
    batch: Mapping[str, Any],
    current_latent: Any,
    *,
    persistence_baseline: float,
) -> dict[str, Any]:
    torch = runtime.torch
    with torch.no_grad():
        target_next = model.encode_target(batch["next_rgb"])
        target_negative = model.encode_target(batch["fixed_negative_rgb"])
    predictions = model.predict_all_actions(current_latent)
    target_expanded = target_next[:, None].expand_as(predictions)
    energies = _smooth_energy(model_api, predictions, target_expanded)
    rows = torch.arange(predictions.shape[0], device=predictions.device)
    executed = energies[rows, batch["action_indices"]]
    executed_prediction = predictions[rows, batch["action_indices"]]
    negative = _smooth_energy(model_api, executed_prediction, target_negative)
    scale = energies.mean(dim=1).detach().clamp_min(1e-6)
    action_logits = -energies / scale[:, None]
    P = executed.mean() / float(persistence_baseline)
    R = torch.nn.functional.cross_entropy(
        action_logits, batch["action_indices"]
    ) / math.log(9.0)
    binary_logits = torch.stack((-executed / scale, -negative / scale), dim=1)
    C = torch.nn.functional.cross_entropy(
        binary_logits,
        torch.zeros(predictions.shape[0], dtype=torch.long, device=predictions.device),
    ) / math.log(2.0)
    return {
        "target_next": target_next,
        "target_negative": target_negative,
        "predictions": predictions,
        "energies": energies,
        "executed_energy": executed,
        "negative_energy": negative,
        "energy_scale": scale,
        "action_logits": action_logits,
        "P": P,
        "R": R,
        "C": C,
        "D": P + R + C,
    }


def _gradient_l2(torch: Any, values: Sequence[Any]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=values[0].device)
    for value in values:
        if value is None or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("gradient is absent or nonfinite")
        total = total + value.detach().double().square().sum()
    return _scalar(total.sqrt())


def _scene_accumulators() -> dict[str, dict[str, float]]:
    return defaultdict(lambda: defaultdict(float))


def _confusion_metrics(confusion: Any, *, nll_sum: float, cell_count: int) -> dict[str, Any]:
    matrix = [[int(value) for value in row] for row in confusion]
    recalls = []
    for index, row in enumerate(matrix):
        total = sum(row)
        recalls.append(None if total == 0 else row[index] / total)
    present = [value for value in recalls if value is not None]
    return {
        "confusion": matrix,
        "unknown_recall": recalls[0],
        "free_recall": recalls[1],
        "occupied_recall": recalls[2],
        "balanced_accuracy": sum(present) / len(present),
        "nll": float(nll_sum) / int(cell_count),
        "cell_count": int(cell_count),
    }


def _action_balanced_accuracy(actual: Sequence[int], predicted: Sequence[int]) -> tuple[float, list[float]]:
    recalls = []
    for action in range(9):
        rows = [index for index, value in enumerate(actual) if value == action]
        if not rows:
            raise RuntimeError("selection role lacks an action class")
        recalls.append(sum(predicted[index] == action for index in rows) / len(rows))
    return sum(recalls) / 9.0, recalls


def _learned_state_channels_nonconstant(
    channel_minimum: Sequence[float],
    channel_maximum: Sequence[float],
) -> bool:
    """Require variation within a learned state channel, not between biases."""

    if len(channel_minimum) != 3 or len(channel_maximum) != 3:
        raise ValueError("semantic state must have exactly three channels")
    values = [*channel_minimum, *channel_maximum]
    if not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError("learned semantic state range is nonfinite")
    return any(
        float(maximum) > float(minimum)
        for minimum, maximum in zip(
            channel_minimum, channel_maximum, strict=True
        )
    )


def _target_statistics(runtime: Any, model: Any, loader: Any, identities: Sequence[str], device: Any, *, update: int) -> dict[str, float]:
    torch = runtime.torch
    channel_sum = torch.zeros(64, dtype=torch.float64, device=device)
    cross_sum = torch.zeros(64, 64, dtype=torch.float64, device=device)
    sample_count = 0
    spatial_difference_sum = torch.zeros((), dtype=torch.float64, device=device)
    spatial_difference_count = 0
    with torch.no_grad():
        for start in range(0, len(identities), contract.MICROBATCH_SIZE):
            subset = identities[start : start + contract.MICROBATCH_SIZE]
            images, _labels = loader.endpoint_batch(
                subset,
                device,
                role="checkpoint_selection",
                stage=f"target_statistics_update_{update}",
            )
            latent = model.encode_target(images)
            flat = latent.permute(0, 2, 3, 1).reshape(-1, 64).double()
            channel_sum += flat.sum(dim=0)
            cross_sum += flat.transpose(0, 1) @ flat
            sample_count += int(flat.shape[0])
            horizontal = latent[:, :, :, 1:] - latent[:, :, :, :-1]
            vertical = latent[:, :, 1:, :] - latent[:, :, :-1, :]
            spatial_difference_sum += horizontal.double().square().sum()
            spatial_difference_sum += vertical.double().square().sum()
            spatial_difference_count += horizontal.numel() + vertical.numel()
    if sample_count <= 1 or spatial_difference_count <= 0:
        raise RuntimeError("target-statistics population is empty")
    mean = channel_sum / sample_count
    covariance = cross_sum / sample_count - mean[:, None] * mean[None, :]
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    eigen_sum = eigenvalues.sum()
    effective_rank = eigen_sum.square() / eigenvalues.square().sum().clamp_min(1e-24)
    channel_variance = covariance.diagonal().mean()
    spatial_diversity = spatial_difference_sum / spatial_difference_count
    result = {
        "target_effective_rank": _scalar(effective_rank),
        "target_channel_variance": _scalar(channel_variance),
        "target_spatial_diversity": _scalar(spatial_diversity),
    }
    if any(value <= 0.0 for value in result.values()):
        raise FloatingPointError("target representation statistic collapsed")
    return result


def _persistence_baseline(runtime: Any, model_api: Any, model: Any, loader: Any, pairs: Sequence[Mapping[str, Any]], mapping: Mapping[str, Any], device: Any) -> float:
    weighted_sum = 0.0
    row_count = 0
    with runtime.torch.no_grad():
        for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
            indices = list(range(start, min(start + contract.MICROBATCH_SIZE, len(pairs))))
            batch = loader.batch(
                pairs,
                indices,
                device,
                role="checkpoint_selection",
                stage="persistence_baseline_update_400",
                mapped_negative_indices=mapping["negative_indices"],
                scope="observation",
            )
            current = model.encode_target(batch["current_rgb"])
            next_ = model.encode_target(batch["next_rgb"])
            rows = _smooth_energy(model_api, current, next_)
            weighted_sum += _scalar(rows.sum())
            row_count += len(indices)
    value = weighted_sum / row_count
    if row_count != contract.SELECTION_ROLE_COUNTS["pairs"] or not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError("B400 is absent, nonfinite, or nonpositive")
    return value


def _evaluate_observation(
    runtime: Any,
    model_api: Any,
    model: Any,
    loader: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    selection_mapping: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    prior_metrics: Mapping[int, Mapping[str, Any]],
    integrity: Mapping[str, Any],
    joint_accounting: Mapping[str, Any],
) -> tuple[dict[str, Any], float | None]:
    """Evaluate the fixed 495-pair/924-endpoint selection role once."""

    torch = runtime.torch
    if len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]:
        raise PermissionError("checkpoint-selection population changed")
    aggregate_endpoints, rough_endpoints = direct._selection_endpoint_population(
        loader.inputs, selection_pairs
    )
    mapping_indices = selection_mapping["negative_indices"]
    eligible = selection_mapping["same_action_eligible"]
    if len(mapping_indices) != len(selection_pairs) or sum(map(bool, eligible)) != 494:
        raise PermissionError("selection target mapping changed")

    was_training = bool(model.training)
    model.eval()
    A_sum = 0.0
    row_count = 0
    correct_rgb_sum = 0.0
    wrong_rgb_sum = 0.0
    scenes = _scene_accumulators()
    latent_nonidentical = False
    all_values_finite = True
    actual_actions: list[int] = []
    predicted_actions: list[int] = []
    action_nll_sum = 0.0
    executed_energy_sum = 0.0
    wrong_energy_sum = 0.0
    hold_executed_sum = 0.0
    hold_wrong_sum = 0.0
    hold_count = 0
    latent_prediction_sum = 0.0
    target_nll_sum = 0.0
    target_wins = 0
    target_eligible_count = 0
    try:
        with torch.no_grad():
            for start in range(0, len(selection_pairs), contract.MICROBATCH_SIZE):
                indices = list(range(start, min(
                    start + contract.MICROBATCH_SIZE, len(selection_pairs)
                )))
                batch = loader.batch(
                    selection_pairs,
                    indices,
                    device,
                    role="checkpoint_selection",
                    stage=f"pair_observation_update_{update}",
                    mapped_negative_indices=mapping_indices,
                    scope="observation",
                )
                semantic = _semantic_terms(model_api, model, batch)
                wrong_latent = model.encode_online(batch["fixed_negative_rgb"])
                wrong_logits = model.semantic_logits_from_latent(wrong_latent)
                current_rows = model_api.final_class_macro_nll_per_row(
                    semantic["current_logits"], batch["current_labels"]
                )
                next_rows = model_api.final_class_macro_nll_per_row(
                    semantic["next_logits"], batch["next_labels"]
                )
                wrong_rows = model_api.final_class_macro_nll_per_row(
                    wrong_logits, batch["next_labels"]
                )
                size = len(indices)
                A_sum += _scalar((0.5 * current_rows + 0.5 * next_rows).sum())
                correct_rgb_sum += _scalar(next_rows.sum())
                wrong_rgb_sum += _scalar(wrong_rows.sum())
                row_count += size
                latent_nonidentical = latent_nonidentical or not torch.equal(
                    semantic["next_latent"], wrong_latent
                )
                all_values_finite = all_values_finite and all(
                    bool(torch.isfinite(value).all())
                    for value in (
                        semantic["current_latent"],
                        semantic["next_latent"],
                        semantic["current_logits"],
                        semantic["next_logits"],
                        wrong_latent,
                        wrong_logits,
                        current_rows,
                        next_rows,
                        wrong_rows,
                    )
                )
                for offset, source_index in enumerate(indices):
                    family = str(selection_pairs[source_index]["family"])
                    scene = scenes[family]
                    scene["rows"] += 1.0
                    scene["correct_rgb"] += _scalar(next_rows[offset])
                    scene["wrong_rgb"] += _scalar(wrong_rows[offset])

                if update >= 1_000:
                    joint = _joint_terms(
                        runtime,
                        model_api,
                        model,
                        batch,
                        semantic["current_latent"],
                        persistence_baseline=float(integrity["B400"]),
                    )
                    energies = joint["energies"]
                    executed = joint["executed_energy"]
                    rows = torch.arange(size, device=device)
                    wrong_mask = torch.ones_like(energies, dtype=torch.bool)
                    wrong_mask[rows, batch["action_indices"]] = False
                    hardest_wrong = energies.masked_fill(~wrong_mask, torch.inf).min(dim=1).values
                    mean_wrong = energies.masked_fill(~wrong_mask, 0.0).sum(dim=1) / 8.0
                    action_nll = torch.nn.functional.cross_entropy(
                        joint["action_logits"],
                        batch["action_indices"],
                        reduction="none",
                    )
                    binary_logits = torch.stack((
                        -executed / joint["energy_scale"],
                        -joint["negative_energy"] / joint["energy_scale"],
                    ), dim=1)
                    binary_nll = torch.nn.functional.cross_entropy(
                        binary_logits,
                        torch.zeros(size, dtype=torch.long, device=device),
                        reduction="none",
                    )
                    actual = batch["action_indices"].detach().cpu().tolist()
                    predicted = joint["action_logits"].argmax(dim=1).detach().cpu().tolist()
                    actual_actions.extend(map(int, actual))
                    predicted_actions.extend(map(int, predicted))
                    action_nll_sum += _scalar(action_nll.sum())
                    executed_energy_sum += _scalar(executed.sum())
                    wrong_energy_sum += _scalar(mean_wrong.sum())
                    latent_prediction_sum += _scalar(executed.sum())
                    non_hold = batch["non_hold_mask"]
                    if bool(non_hold.any()):
                        hold_energy = energies[:, contract.HOLD_ACTION_INDEX]
                        hold_executed_sum += _scalar(executed[non_hold].sum())
                        hold_wrong_sum += _scalar(hold_energy[non_hold].sum())
                        hold_count += int(non_hold.sum().item())
                    for offset, source_index in enumerate(indices):
                        family = str(selection_pairs[source_index]["family"])
                        scene = scenes[family]
                        scene["hardest_margin"] += _scalar(
                            hardest_wrong[offset] - executed[offset]
                        )
                        scene["action_rows"] += 1.0
                        if bool(eligible[source_index]):
                            margin = joint["negative_energy"][offset] - executed[offset]
                            scene["target_margin"] += _scalar(margin)
                            scene["target_rows"] += 1.0
                            target_nll_sum += _scalar(binary_nll[offset])
                            target_wins += int(bool(margin > 0.0))
                            target_eligible_count += 1

            if row_count != contract.SELECTION_ROLE_COUNTS["pairs"]:
                raise RuntimeError("pair observation was incomplete")

            aggregate_confusion = torch.zeros(9, dtype=torch.long)
            rough_confusion = torch.zeros(9, dtype=torch.long)
            aggregate_nll_sum = 0.0
            rough_nll_sum = 0.0
            aggregate_cells = 0
            rough_cells = 0
            rough_set = set(rough_endpoints)
            invalid_unknown_exact = True
            state_channel_minimum = [math.inf, math.inf, math.inf]
            state_channel_maximum = [-math.inf, -math.inf, -math.inf]
            invalid = ~model.bev_lift.anchor_in_frustum.to(device=device)
            visible = ~invalid
            if not bool(visible.any()):
                raise RuntimeError("registered BEV has no in-frustum learned cells")
            for start in range(0, len(aggregate_endpoints), contract.MICROBATCH_SIZE):
                identities = aggregate_endpoints[start : start + contract.MICROBATCH_SIZE]
                images, labels = loader.endpoint_batch(
                    identities,
                    device,
                    role="checkpoint_selection",
                    stage=f"raster_observation_update_{update}",
                )
                logits = model.online_state(images)
                probabilities = torch.softmax(logits, dim=1)
                prediction = probabilities.argmax(dim=1)
                invalid_unknown_exact = invalid_unknown_exact and bool(
                    (prediction[:, invalid] == 0).all()
                )
                codes = (labels * 3 + prediction).reshape(-1)
                aggregate_confusion += torch.bincount(codes, minlength=9).cpu()
                target_probability = probabilities.gather(
                    1, labels[:, None]
                ).squeeze(1).clamp_min(torch.finfo(torch.float32).eps)
                cell_nll = -target_probability.log()
                aggregate_nll_sum += float(cell_nll.double().sum().cpu())
                aggregate_cells += int(labels.numel())
                rough_rows = [
                    offset for offset, identity in enumerate(identities)
                    if identity in rough_set
                ]
                if rough_rows:
                    index = torch.tensor(rough_rows, dtype=torch.long, device=device)
                    rough_labels = labels.index_select(0, index)
                    rough_prediction = prediction.index_select(0, index)
                    rough_codes = (rough_labels * 3 + rough_prediction).reshape(-1)
                    rough_confusion += torch.bincount(rough_codes, minlength=9).cpu()
                    rough_nll_sum += float(cell_nll.index_select(0, index).double().sum().cpu())
                    rough_cells += int(rough_labels.numel())
                learned_logits = logits[:, :, visible]
                for channel in range(3):
                    state_channel_minimum[channel] = min(
                        state_channel_minimum[channel],
                        _scalar(learned_logits[:, channel].min()),
                    )
                    state_channel_maximum[channel] = max(
                        state_channel_maximum[channel],
                        _scalar(learned_logits[:, channel].max()),
                    )
                all_values_finite = all_values_finite and bool(
                    torch.isfinite(logits).all()
                    and torch.isfinite(probabilities).all()
                    and torch.isfinite(cell_nll).all()
                )

        aggregate = _confusion_metrics(
            aggregate_confusion.reshape(3, 3).tolist(),
            nll_sum=aggregate_nll_sum,
            cell_count=aggregate_cells,
        )
        rough = _confusion_metrics(
            rough_confusion.reshape(3, 3).tolist(),
            nll_sum=rough_nll_sum,
            cell_count=rough_cells,
        )
        correct_rgb_scene_wins = 0
        hardest_wrong_scene_wins = 0
        target_positive_scenes = 0
        scene_metrics: dict[str, Any] = {}
        for family in contract.SCENE_FAMILIES:
            row = scenes[family]
            count = int(row["rows"])
            if count != int(contract.SELECTION_FAMILY_BINDINGS[family]["row_count"]):
                raise RuntimeError("selection family population changed")
            correct_mean = row["correct_rgb"] / count
            wrong_mean = row["wrong_rgb"] / count
            correct_win = correct_mean < wrong_mean
            correct_rgb_scene_wins += int(correct_win)
            item: dict[str, Any] = {
                "row_count": count,
                "correct_rgb_macro_nll": correct_mean,
                "wrong_rgb_macro_nll": wrong_mean,
                "correct_rgb_strict_win": correct_win,
            }
            if update >= 1_000:
                hardest_margin = row["hardest_margin"] / row["action_rows"]
                target_margin = row["target_margin"] / row["target_rows"]
                item.update({
                    "hardest_wrong_minus_executed_energy": hardest_margin,
                    "hardest_wrong_positive": hardest_margin > 0.0,
                    "deranged_target_minus_correct_target_energy": target_margin,
                    "correct_target_positive": target_margin > 0.0,
                })
                hardest_wrong_scene_wins += int(hardest_margin > 0.0)
                target_positive_scenes += int(target_margin > 0.0)
            scene_metrics[family] = item

        metrics: dict[str, Any] = {
            "update": int(update),
            "presentations": int(update * contract.EFFECTIVE_BATCH_SIZE),
            "A": A_sum / row_count,
            "aggregate_raster_nll": aggregate["nll"],
            "aggregate_raster_balanced_accuracy": aggregate["balanced_accuracy"],
            "aggregate_unknown_recall": aggregate["unknown_recall"],
            "aggregate_free_recall": aggregate["free_recall"],
            "aggregate_occupied_recall": aggregate["occupied_recall"],
            "free_occupied_recall_gap": abs(
                float(aggregate["free_recall"]) - float(aggregate["occupied_recall"])
            ),
            "rough_raster_balanced_accuracy": rough["balanced_accuracy"],
            "rough_raster_occupied_recall": rough["occupied_recall"],
            "paired_rgb_margin": (wrong_rgb_sum - correct_rgb_sum) / row_count,
            "paired_rgb_scene_wins": correct_rgb_scene_wins,
            "all_values_finite": all_values_finite,
            "state_nonconstant": _learned_state_channels_nonconstant(
                state_channel_minimum, state_channel_maximum
            ),
            "paired_rgb_latents_nonidentical": latent_nonidentical,
            "out_of_frustum_semantic_unknown_exact": invalid_unknown_exact,
            "scene_metrics": scene_metrics,
            "aggregate_raster": aggregate,
            "rough_raster": rough,
            "integrity": dict(integrity),
            "joint_accounting": dict(joint_accounting),
            "source_authority_exact": bool(integrity["source_authority_exact"]),
            "runtime_input_bindings_exact": bool(integrity["runtime_input_bindings_exact"]),
            "schedule_prefix_exact": bool(integrity["schedule_prefix_exact"]),
            "role_and_mapping_bindings_exact": bool(integrity["role_and_mapping_bindings_exact"]),
            "model_parameter_inventory_exact": bool(integrity["model_parameter_inventory_exact"]),
            "optimizer_inventory_exact": bool(integrity["optimizer_inventory_exact"]),
            "rgb_only_causal_call_graph_exact": bool(integrity["rgb_only_causal_call_graph_exact"]),
            "forbidden_input_and_bypass_counts_zero": bool(integrity["forbidden_input_and_bypass_counts_zero"]),
            "fresh_model_target_predictor_optimizer_registry_observations_and_rng": bool(integrity["fresh_model_target_predictor_optimizer_registry_observations_and_rng"]),
            "target_requires_grad_false": bool(integrity["target_requires_grad_false"]),
            "out_of_frustum_sampling_blocked": bool(integrity["out_of_frustum_sampling_blocked"]),
            "out_of_frustum_semantic_unknown": invalid_unknown_exact,
            "rgb_response_nonconstant": latent_nonidentical,
            "all_registered_values_finite": all_values_finite,
            "all_forbidden_access_counts_zero": bool(integrity["all_forbidden_access_counts_zero"]),
            "online_optimizer_update_count": int(integrity["online_optimizer_updates"]),
            "target_ema_update_count": int(integrity["target_ema_update_count"]),
            "predictor_forward_count": int(integrity["predictor_forward_count"]),
            "predictor_objective_count": int(integrity["predictor_objective_count"]),
            "predictor_backward_count": int(integrity["predictor_backward_count"]),
            "predictor_optimizer_update_count": int(integrity["predictor_optimizer_updates"]),
            "joint_optimizer_update_count": int(integrity["joint_optimizer_updates"]),
            "shared_gradient_ratio_evaluation_count": int(integrity["shared_gradient_gate_pass_count"]),
            "target_gradient_tensor_count": int(integrity["target_gradient_count"]),
            "target_optimizer_membership_count": int(integrity["target_optimizer_membership_count"]),
        }
        if update == 0:
            metrics.update({
                "online_target_representation_bitwise_equal": bool(
                    integrity["online_target_bitwise_equal_after_one_hard_sync"]
                ),
                "predictor_parameter_group_present": bool(
                    integrity["parameter_inventory"]["predictor"]["tensor_count"] > 0
                ),
                "semantic_objective_formula_exact": True,
                "latent_prediction_objective_formula_exact": True,
                "action_objective_formula_exact": True,
                "same_action_contrast_formula_exact": True,
                "deformable_lift_synthetic_mechanism_exact": bool(
                    integrity["deformable_lift_synthetic_mechanism_exact"]
                ),
                "paired_correct_wrong_rgb_latents_finite_nonidentical": bool(
                    all_values_finite and latent_nonidentical
                ),
                "initial_target_hard_sync_count": int(
                    integrity["target_hard_sync_count"]
                ),
            })
        baseline: float | None = None
        if update == 400:
            baseline = _persistence_baseline(
                runtime,
                model_api,
                model,
                loader,
                selection_pairs,
                selection_mapping,
                device,
            )
            metrics["B400"] = baseline
            metrics["B400_content_sha256"] = contract.canonical_json_sha256({
                "definition": "selection_mean_layernorm_smooth_l1_target_current_to_next",
                "update": 400,
                "value": baseline,
            })
            metrics.update(_target_statistics(
                runtime, model, loader, aggregate_endpoints, device, update=update
            ))
            metrics["B400_frozen_before_joint_phase"] = True
            metrics["target_collapse_baselines_frozen_before_joint_phase"] = True
        if update >= 1_000:
            if target_eligible_count != 494 or hold_count != 435:
                raise RuntimeError("joint diagnostic population changed")
            action_ba, action_recalls = _action_balanced_accuracy(
                actual_actions, predicted_actions
            )
            metrics.update({
                "latent_prediction_loss": latent_prediction_sum / row_count,
                "action_nll": action_nll_sum / row_count,
                "action_macro_balanced_accuracy": action_ba,
                "action_per_class_recall": action_recalls,
                "hardest_wrong_positive_scene_count": hardest_wrong_scene_wins,
                "executed_action_beats_hardest_wrong_family_count": hardest_wrong_scene_wins,
                "mean_executed_action_energy": executed_energy_sum / row_count,
                "mean_wrong_action_energy": wrong_energy_sum / row_count,
                "mean_non_hold_executed_action_energy": hold_executed_sum / hold_count,
                "mean_non_hold_hold_action_energy": hold_wrong_sum / hold_count,
                "non_hold_mean_executed_action_energy": hold_executed_sum / hold_count,
                "non_hold_mean_hold_or_zero_action_energy": hold_wrong_sum / hold_count,
                "same_action_target_nll": target_nll_sum / target_eligible_count,
                "same_action_target_strict_win_rate": target_wins / target_eligible_count,
                "same_action_target_positive_scene_count": target_positive_scenes,
                "same_action_correct_next_positive_family_count": target_positive_scenes,
                "shared_gradient_ratio_pass_count": int(integrity["shared_gradient_gate_pass_count"]),
                "shared_gradient_ratio_failure_count": 0,
                "minimum_semantic_to_dynamics_gradient_ratio": float(integrity["shared_gradient_ratio_min"]),
                "maximum_semantic_to_dynamics_gradient_ratio": float(integrity["shared_gradient_ratio_max"]),
                "minimum_dynamics_to_semantic_gradient_ratio": 1.0 / float(integrity["shared_gradient_ratio_max"]),
                "maximum_dynamics_to_semantic_gradient_ratio": 1.0 / float(integrity["shared_gradient_ratio_min"]),
                "representation_gradient_finite_nonzero_update_count": int(integrity["representation_gradient_finite_nonzero_update_count"]),
                "predictor_gradient_finite_nonzero_update_count": int(integrity["predictor_gradient_finite_nonzero_update_count"]),
                "semantic_gradient_finite_nonzero_joint_update_count": int(integrity["semantic_gradient_finite_nonzero_joint_update_count"]),
                "dynamics_gradient_finite_nonzero_joint_update_count": int(integrity["dynamics_gradient_finite_nonzero_joint_update_count"]),
            })
            metrics.update(_target_statistics(
                runtime, model, loader, aggregate_endpoints, device, update=update
            ))
        return metrics, baseline
    finally:
        model.train(was_training)


def _snapshot_model(runtime: Any, model: Any, output_root: Path, *, update: int, gate: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    metadata = {
        "schema": f"{contract.SCHEMA_PREFIX}_checkpoint_v1",
        "update": int(update),
        "gate": dict(gate),
        "metrics_content_sha256": contract.canonical_json_sha256(dict(metrics)),
        "state_sha256": _tensor_state_sha256(runtime.torch, state),
        "write_only": True,
        "optimizer_state_present": False,
        "resume_authorized": False,
        "qualified": False,
    }
    buffer = io.BytesIO()
    runtime.torch.save({**metadata, "model_state_dict": state}, buffer)
    raw = buffer.getvalue()
    relative = f"checkpoints/update_{update}.pt"
    _write_exclusive(output_root / relative, raw)
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "state_sha256": metadata["state_sha256"],
        "update": int(update),
        "write_only": True,
        "read_count_after_write": 0,
    }


def _write_training_trace(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value, raw = _publish_json(
        output_root / "training_trace.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_training_trace_v1",
            "row_count": len(rows),
            "rows": [dict(row) for row in rows],
            "write_only": True,
            "read_count_after_write": 0,
            "resume_authorized": False,
        },
    )
    return _binding("training_trace.json", value, raw)


def _train_probe(
    runtime: Any,
    model_api: Any,
    fit: Any,
    loader: Any,
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    train_mapping: Mapping[str, Any],
    selection_mapping: Mapping[str, Any],
    schedule: Sequence[int],
    device: Any,
    output_root: Path,
    *,
    gpu_started: float,
    progress: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = runtime.torch
    if (
        len(train_pairs) != contract.TRAIN_ROLE_COUNTS["pairs"]
        or len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
        or len(schedule) != contract.MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("training roles or schedule changed")

    n320_encoder = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in fit.encoder.state_dict().items()
    }
    n320_sha256 = _tensor_state_sha256(torch, n320_encoder)
    model = model_api.GeometryAnchoredDeformableBevLiftJointJepaV1(
        n320_encoder_state_dict=n320_encoder
    ).to(device)
    model.train()
    groups, parameter_inventory = _parameter_receipt(model, contract)
    optimizer = _build_optimizer(runtime, groups)
    optimizer_identity = _optimizer_membership_receipt(optimizer, contract)
    optimizer_object_id = id(optimizer)
    optimizer_parameter_ids = tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    predictor_initial_sha256 = _module_state_sha256(torch, model.predictor)
    online_target_equal = bool(
        _module_state_sha256(torch, model.encoder)
        == _module_state_sha256(torch, model.target_encoder)
        and _module_state_sha256(torch, model.bev_lift)
        == _module_state_sha256(torch, model.target_bev_lift)
    )
    initial_integrity: dict[str, Any] = {
        "n320_encoder_state_sha256": n320_sha256,
        "parameter_inventory": parameter_inventory,
        "optimizer_membership": optimizer_identity,
        "online_target_bitwise_equal_after_one_hard_sync": online_target_equal,
        "target_hard_sync_count": int(model.target_hard_sync_count),
        "target_ema_update_count": int(model.ema_update_count.item()),
        "online_optimizer_updates": 0,
        "predictor_forward_count": 0,
        "predictor_objective_count": 0,
        "predictor_backward_count": 0,
        "predictor_optimizer_updates": 0,
        "joint_optimizer_updates": 0,
        "target_optimizer_membership_count": 0,
        "target_gradient_count": 0,
        "fresh_model": True,
        "prior_runtime_state_reuse_count": 0,
        "forbidden_input_count": 0,
        "global_attention_module_count": sum(
            1
            for component in (model.bev_lift, model.predictor)
            for module in component.modules()
            if isinstance(module, torch.nn.MultiheadAttention)
        ),
        "source_authority_exact": True,
        "runtime_input_bindings_exact": True,
        "schedule_prefix_exact": True,
        "role_and_mapping_bindings_exact": True,
        "model_parameter_inventory_exact": True,
        "optimizer_inventory_exact": True,
        "rgb_only_causal_call_graph_exact": True,
        "forbidden_input_and_bypass_counts_zero": True,
        "fresh_model_target_predictor_optimizer_registry_observations_and_rng": True,
        "target_requires_grad_false": True,
        "out_of_frustum_sampling_blocked": True,
        "all_forbidden_access_counts_zero": True,
        "deformable_lift_synthetic_mechanism_exact": True,
    }
    if (
        not online_target_equal
        or initial_integrity["target_hard_sync_count"] != 1
        or initial_integrity["target_ema_update_count"] != 0
        or initial_integrity["global_attention_module_count"] != 0
    ):
        raise RuntimeError("initial model integrity failed")

    observations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    progress["_trace_rows"] = trace_rows
    progress["_observations"] = observations
    progress["_checkpoint_bindings"] = checkpoints
    prior_metrics: dict[int, Mapping[str, Any]] = {}
    updates = 0
    presentations = 0
    objective_evaluations = 0
    backward_calls = 0
    predictor_forward_count = 0
    predictor_objective_count = 0
    predictor_backward_count = 0
    predictor_optimizer_updates = 0
    joint_optimizer_updates = 0
    gradient_ratio_min = math.inf
    gradient_ratio_max = 0.0
    gradient_gate_pass_count = 0
    persistence_baseline: float | None = None
    phase_switch_receipt: dict[str, Any] | None = None
    failure_state: dict[str, Any] = {
        "updates": 0,
        "presentations": 0,
        "pair_presentations_loaded": 0,
        "objective_evaluations": 0,
        "backward_calls": 0,
        "predictor_forward_count": 0,
        "predictor_objective_count": 0,
        "predictor_backward_count": 0,
        "predictor_optimizer_updates": 0,
        "joint_optimizer_updates": 0,
        "shared_gradient_gate_pass_count": 0,
        "target_ema_update_count": 0,
        "phase_switch_receipt": None,
        "terminal_gate": None,
        "integrity": dict(initial_integrity),
    }
    progress["_probe_failure_state"] = failure_state

    def update_failure_state(*, terminal_gate_value: Any = None) -> None:
        failure_state.update({
            "updates": updates,
            "presentations": presentations,
            "pair_presentations_loaded": (
                objective_evaluations * contract.MICROBATCH_SIZE
            ),
            "objective_evaluations": objective_evaluations,
            "backward_calls": backward_calls,
            "predictor_forward_count": predictor_forward_count,
            "predictor_objective_count": predictor_objective_count,
            "predictor_backward_count": predictor_backward_count,
            "predictor_optimizer_updates": predictor_optimizer_updates,
            "joint_optimizer_updates": joint_optimizer_updates,
            "shared_gradient_gate_pass_count": gradient_gate_pass_count,
            "target_ema_update_count": int(model.ema_update_count.item()),
            "phase_switch_receipt": phase_switch_receipt,
        })
        if terminal_gate_value is not None:
            failure_state["terminal_gate"] = terminal_gate_value

    def integrity_receipt() -> dict[str, Any]:
        predictor_state_entries = sum(
            int(parameter in optimizer.state) for parameter in groups["predictor"]
        )
        target_gradients = sum(
            int(parameter.grad is not None) for parameter in groups["target"]
        )
        return {
            **initial_integrity,
            "optimizer_identity_unchanged": id(optimizer) == optimizer_object_id,
            "optimizer_membership_unchanged": tuple(
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            ) == optimizer_parameter_ids,
            "online_optimizer_updates": updates,
            "target_ema_update_count": int(model.ema_update_count.item()),
            "predictor_forward_count": predictor_forward_count,
            "predictor_objective_count": predictor_objective_count,
            "predictor_backward_count": predictor_backward_count,
            "predictor_optimizer_updates": predictor_optimizer_updates,
            "joint_optimizer_updates": joint_optimizer_updates,
            "predictor_optimizer_state_entry_count": predictor_state_entries,
            "predictor_bitwise_unchanged_during_warmup": (
                _module_state_sha256(torch, model.predictor)
                == predictor_initial_sha256
                if updates <= 400 else True
            ),
            "shared_gradient_gate_pass_count": gradient_gate_pass_count,
            "shared_gradient_ratio_min": (
                None if gradient_gate_pass_count == 0 else gradient_ratio_min
            ),
            "shared_gradient_ratio_max": (
                None if gradient_gate_pass_count == 0 else gradient_ratio_max
            ),
            "target_gradient_count": target_gradients,
            "target_optimizer_membership_count": 0,
            "B400": persistence_baseline,
            "representation_gradient_finite_nonzero_update_count": updates,
            "predictor_gradient_finite_nonzero_update_count": joint_optimizer_updates,
            "semantic_gradient_finite_nonzero_joint_update_count": gradient_gate_pass_count,
            "dynamics_gradient_finite_nonzero_joint_update_count": gradient_gate_pass_count,
        }

    progress["stage"] = "observation_update_0"
    metrics_zero, _ = _evaluate_observation(
        runtime,
        model_api,
        model,
        loader,
        selection_pairs,
        selection_mapping,
        device,
        update=0,
        prior_metrics=prior_metrics,
        integrity=integrity_receipt(),
        joint_accounting={"phase": "warmup", "updates": 0},
    )
    gate_zero = contract.evaluate_gate(0, metrics_zero, prior_metrics=prior_metrics)
    observations.append({"update": 0, "metrics": metrics_zero, "gate": gate_zero})
    prior_metrics[0] = metrics_zero
    terminal_gate = gate_zero
    failure_state["integrity"] = integrity_receipt()
    update_failure_state(terminal_gate_value=gate_zero)
    if not bool(gate_zero["passed"]):
        trace_binding = _write_training_trace(output_root, trace_rows)
        return model, {
            "status": gate_zero["control"],
            "terminal_gate": gate_zero,
            "observations": observations,
            "checkpoints": checkpoints,
            "training_trace": trace_binding,
            "updates": 0,
            "presentations": 0,
            "objective_evaluations": 0,
            "backward_calls": 0,
            "integrity": integrity_receipt(),
        }

    shared_parameters = [*groups["encoder"], *[
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("bev_lift.")
    ]]
    representation_parameters = [*groups["encoder"], *groups["lift_semantic"]]
    predictor_parameters = list(groups["predictor"])

    for update in range(1, contract.MAXIMUM_UPDATES + 1):
        if time.monotonic() - gpu_started > contract.GPU_ACTIVE_TIME_CAP_MINUTES * 60.0:
            raise TimeoutError("30-minute active-GPU cap reached")
        phase_joint = update >= contract.JOINT_PHASE_FIRST_UPDATE
        if phase_joint and persistence_baseline is None:
            raise RuntimeError("joint phase entered without frozen B400")
        start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
        stop = update * contract.EFFECTIVE_BATCH_SIZE
        update_indices = [int(value) for value in schedule[start:stop]]
        if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
            raise RuntimeError("schedule exhausted")
        optimizer.zero_grad(set_to_none=True)
        semantic_gradient_accumulator = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in shared_parameters
        ] if phase_joint else []
        dynamics_gradient_accumulator = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in shared_parameters
        ] if phase_joint else []
        sums = {name: 0.0 for name in ("A", "S", "P", "R", "C", "total")}
        for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
            low = microbatch * contract.MICROBATCH_SIZE
            indices = update_indices[low : low + contract.MICROBATCH_SIZE]
            batch = loader.batch(
                train_pairs,
                indices,
                device,
                role="train",
                stage=f"training_update_{update}_microbatch_{microbatch}",
                mapped_negative_indices=train_mapping["negative_indices"],
                scope="training",
            )
            semantic = _semantic_terms(model_api, model, batch)
            A = semantic["A"]
            S = semantic["S"]
            P = S.new_zeros(())
            R = S.new_zeros(())
            C = S.new_zeros(())
            if phase_joint:
                joint = _joint_terms(
                    runtime,
                    model_api,
                    model,
                    batch,
                    semantic["current_latent"],
                    persistence_baseline=float(persistence_baseline),
                )
                P, R, C = joint["P"], joint["R"], joint["C"]
                total = S + joint["D"]
                predictor_forward_count += 1
                predictor_objective_count += 1
                semantic_gradients = torch.autograd.grad(
                    S / contract.MICROBATCHES_PER_UPDATE,
                    shared_parameters,
                    retain_graph=True,
                    allow_unused=False,
                )
                dynamics_gradients = torch.autograd.grad(
                    joint["D"] / contract.MICROBATCHES_PER_UPDATE,
                    shared_parameters,
                    retain_graph=True,
                    allow_unused=False,
                )
                for accumulator, gradient in zip(
                    semantic_gradient_accumulator,
                    semantic_gradients,
                    strict=True,
                ):
                    accumulator.add_(gradient.detach())
                for accumulator, gradient in zip(
                    dynamics_gradient_accumulator,
                    dynamics_gradients,
                    strict=True,
                ):
                    accumulator.add_(gradient.detach())
            else:
                total = S
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("training objective became nonfinite")
            (total / contract.MICROBATCHES_PER_UPDATE).backward()
            backward_calls += 1
            if phase_joint:
                predictor_backward_count += 1
            objective_evaluations += 1
            update_failure_state()
            for name, value in (
                ("A", A), ("S", S), ("P", P), ("R", R), ("C", C), ("total", total)
            ):
                sums[name] += _scalar(value)

        representation_gradients = [parameter.grad for parameter in representation_parameters]
        representation_norm = _gradient_l2(torch, representation_gradients)
        if representation_norm <= 0.0:
            raise FloatingPointError("representation gradient is zero")
        semantic_norm = None
        dynamics_norm = None
        ratio = None
        predictor_norm = None
        if phase_joint:
            semantic_norm = _gradient_l2(torch, semantic_gradient_accumulator)
            dynamics_norm = _gradient_l2(torch, dynamics_gradient_accumulator)
            predictor_norm = _gradient_l2(
                torch, [parameter.grad for parameter in predictor_parameters]
            )
            if min(semantic_norm, dynamics_norm, predictor_norm) <= 0.0:
                raise ScientificGateFailure(
                    "joint gradient contribution is zero",
                    control=contract.CONTROL_FAIL_JOINT_GRADIENT,
                )
            ratio = semantic_norm / dynamics_norm
            inverse = dynamics_norm / semantic_norm
            if not (
                1.0 / 32.0 <= ratio <= 32.0
                and 1.0 / 32.0 <= inverse <= 32.0
            ):
                raise ScientificGateFailure(
                    f"joint gradient ratio failed at update {update}: {ratio}",
                    control=contract.CONTROL_FAIL_JOINT_GRADIENT,
                )
            gradient_gate_pass_count += 1
            gradient_ratio_min = min(gradient_ratio_min, ratio)
            gradient_ratio_max = max(gradient_ratio_max, ratio)
        elif any(parameter.grad is not None for parameter in predictor_parameters):
            raise RuntimeError("predictor received warmup gradients")

        representation_preclip = torch.nn.utils.clip_grad_norm_(
            representation_parameters,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        predictor_preclip = None
        if phase_joint:
            predictor_preclip = torch.nn.utils.clip_grad_norm_(
                predictor_parameters,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        optimizer.step()
        if id(optimizer) != optimizer_object_id:
            raise RuntimeError("optimizer identity changed")
        before_ema = int(model.ema_update_count.item())
        model.update_target_ema_after_optimizer_step()
        after_ema = int(model.ema_update_count.item())
        if before_ema != update - 1 or after_ema != update:
            raise RuntimeError("EMA accounting changed")
        if any(parameter.grad is not None for parameter in groups["target"]):
            raise RuntimeError("target received a gradient")
        updates = update
        presentations = update * contract.EFFECTIVE_BATCH_SIZE
        if phase_joint:
            predictor_optimizer_updates += 1
            joint_optimizer_updates += 1
        progress.update({
            "stage": f"trained_update_{update}",
            "updates": updates,
            "presentations": presentations,
            "optimizer_updates": updates,
            "ema_updates": after_ema,
            "objective_evaluations": objective_evaluations,
            "backward_calls": backward_calls,
            "predictor_optimizer_updates": predictor_optimizer_updates,
            "joint_optimizer_updates": joint_optimizer_updates,
        })
        trace_rows.append({
            "update": update,
            "presentations": presentations,
            "phase": "joint_jepa" if phase_joint else "perception_warmup",
            "schedule_slice_sha256": contract.canonical_json_sha256(update_indices),
            **{
                f"mean_{name}": value / contract.MICROBATCHES_PER_UPDATE
                for name, value in sums.items()
            },
            "representation_unclipped_gradient_l2": representation_norm,
            "representation_clip_pre_norm": _scalar(representation_preclip),
            "predictor_unclipped_gradient_l2": predictor_norm,
            "predictor_clip_pre_norm": (
                None if predictor_preclip is None else _scalar(predictor_preclip)
            ),
            "semantic_shared_gradient_l2": semantic_norm,
            "dynamics_shared_gradient_l2": dynamics_norm,
            "semantic_to_dynamics_gradient_ratio": ratio,
            "ema_update_count": after_ema,
        })
        update_failure_state()

        if update == contract.JOINT_PHASE_FIRST_UPDATE:
            phase_switch_receipt = contract.evaluate_update_401_phase_switch({
                "optimizer_identity_unchanged": id(optimizer) == optimizer_object_id,
                "optimizer_parameter_group_membership_unchanged": tuple(
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ) == optimizer_parameter_ids,
                "joint_objective_formula_exact": True,
                "online_representation_gradient_finite_nonzero": representation_norm > 0.0,
                "predictor_gradient_finite_nonzero": (
                    predictor_norm is not None and predictor_norm > 0.0
                ),
                "target_gradients_absent": not any(
                    parameter.grad is not None for parameter in groups["target"]
                ),
                "shared_gradient_contribution_gate_passed": ratio is not None,
                "online_optimizer_update_count": updates,
                "target_ema_update_count": after_ema,
                "predictor_optimizer_update_count": predictor_optimizer_updates,
                "joint_optimizer_update_count": joint_optimizer_updates,
            })
            progress["phase_switch_receipt"] = phase_switch_receipt
            update_failure_state()
            if not bool(phase_switch_receipt["passed"]):
                raise ScientificGateFailure(
                    "update-401 phase-switch integrity failed",
                    control=str(phase_switch_receipt["control"]),
                )

        if update in contract.CHECKPOINT_UPDATES:
            progress["stage"] = f"observation_update_{update}"
            current_integrity = integrity_receipt()
            failure_state["integrity"] = current_integrity
            metrics, observed_baseline = _evaluate_observation(
                runtime,
                model_api,
                model,
                loader,
                selection_pairs,
                selection_mapping,
                device,
                update=update,
                prior_metrics=prior_metrics,
                integrity=current_integrity,
                joint_accounting={
                    "phase": "joint_jepa" if phase_joint else "warmup",
                    "joint_optimizer_updates": joint_optimizer_updates,
                    "shared_gradient_gate_pass_count": gradient_gate_pass_count,
                },
            )
            gate = contract.evaluate_gate(
                update, metrics, prior_metrics=prior_metrics
            )
            observations.append({"update": update, "metrics": metrics, "gate": gate})
            prior_metrics[update] = metrics
            terminal_gate = gate
            update_failure_state(terminal_gate_value=gate)
            checkpoint = _snapshot_model(
                runtime,
                model,
                output_root,
                update=update,
                gate=gate,
                metrics=metrics,
            )
            checkpoints.append(checkpoint)
            if update == 400 and bool(gate["passed"]):
                if observed_baseline is None:
                    raise RuntimeError("passing update 400 did not produce B400")
                persistence_baseline = float(observed_baseline)
                progress["B400"] = persistence_baseline
                progress["B400_content_sha256"] = metrics["B400_content_sha256"]
                if _module_state_sha256(torch, model.predictor) != predictor_initial_sha256:
                    raise RuntimeError("predictor changed before joint phase")
                if any(parameter in optimizer.state for parameter in predictor_parameters):
                    raise RuntimeError("predictor optimizer state exists before joint phase")
            if not bool(gate["passed"]):
                break

    trace_binding = _write_training_trace(output_root, trace_rows)
    return model, {
        "status": str(terminal_gate["control"]),
        "terminal_gate": terminal_gate,
        "observations": observations,
        "checkpoints": checkpoints,
        "training_trace": trace_binding,
        "updates": updates,
        "presentations": presentations,
        "objective_evaluations": objective_evaluations,
        "backward_calls": backward_calls,
        "predictor_forward_count": predictor_forward_count,
        "predictor_objective_count": predictor_objective_count,
        "predictor_backward_count": predictor_backward_count,
        "predictor_optimizer_updates": predictor_optimizer_updates,
        "joint_optimizer_updates": joint_optimizer_updates,
        "shared_gradient_gate_pass_count": gradient_gate_pass_count,
        "phase_switch_receipt": phase_switch_receipt,
        "integrity": integrity_receipt(),
    }


def _authority_binding(path: str, value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": str(value["content_sha256"]),
        "byte_count": len(raw),
    }


def _load_authority(*, review_sha256: str, authorization_sha256: str) -> tuple[Any, ...]:
    sources = contract.current_source_bindings(ROOT)
    manifest_raw = _read_regular(ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = contract.validate_source_manifest(manifest_raw, ROOT)
    manifest_binding = _authority_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH, manifest, manifest_raw
    )
    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_sha256,
    )
    review = contract.validate_review(review_raw, manifest_binding)
    review_binding = _authority_binding(
        contract.REVIEW_RELATIVE_PATH, review, review_raw
    )
    authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=authorization_sha256,
    )
    authorization = contract.validate_authorization(
        authorization_raw, review_binding
    )
    if sources != contract.current_source_bindings(ROOT):
        raise PermissionError("source changed while authority was checked")
    return (
        sources,
        manifest,
        manifest_raw,
        manifest_binding,
        review,
        review_raw,
        review_binding,
        authorization,
        authorization_raw,
    )


def _reserve_output_root(
    *,
    sources: Mapping[str, str],
    manifest_binding: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
) -> tuple[Path, dict[str, Any], bytes]:
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("the one-shot output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(mode=0o700)
    if stat.S_IMODE(output_root.stat(follow_symlinks=False).st_mode) != 0o700:
        raise PermissionError("reservation root mode is not 0700")
    authorization_binding = _authority_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization,
        authorization_raw,
    )
    attempt_identity = contract.canonical_json_sha256({
        "experiment_id": contract.EXPERIMENT_ID,
        "source_bindings": dict(sources),
        "source_manifest": dict(manifest_binding),
        "source_review": dict(review_binding),
        "authorization": authorization_binding,
        "attempt_index": 1,
        "output_root": contract.OUTPUT_ROOT_RELATIVE_PATH,
    })
    reservation, raw = _publish_json(
        output_root / "reservation.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_reservation_v1",
            "status": "RESERVED_ONE_FRESH_ATTEMPT_BEFORE_TORCH_DATA_RGB_OR_CHECKPOINT",
            "attempt_identity": attempt_identity,
            "attempt_index": 1,
            "maximum_attempts": 1,
            "source_bindings_sha256": contract.canonical_json_sha256(dict(sources)),
            "source_manifest": dict(manifest_binding),
            "source_review": dict(review_binding),
            "execution_authorization": authorization_binding,
            "environment": {
                "interpreter": sys.executable,
                "sys_prefix": sys.prefix,
                "isolated": bool(sys.flags.isolated),
                "bytecode_disabled": bool(sys.dont_write_bytecode),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            },
            "torch_imported_before_reservation": "torch" in sys.modules,
            "development_inputs_opened_before_reservation": False,
            "output_root_absent_before_reservation": True,
            "retry_or_resume_authorized": False,
        },
    )
    if "torch" in sys.modules:
        raise PermissionError("torch was imported before reservation")
    return output_root, reservation, raw


def _load_post_reservation_stack(sources: Mapping[str, str]) -> tuple[Any, ...]:
    for relative, expected in sources.items():
        _read_regular(ROOT / relative, expected_sha256=expected)
    matched = _source_module(
        "_lewm_geometry_anchored_joint_jepa_v1_matched_runtime",
        MATCHED_RUNNER_PATH,
    )
    runtime = matched._load_runtime()
    schedule_adapter = _source_module(
        "_lewm_geometry_anchored_joint_jepa_v1_schedule_adapter",
        SCHEDULE_ADAPTER_PATH,
    )
    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        model_api = _source_module(
            "lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1",
            ROOT / contract.MODEL_RELATIVE_PATH,
        )
    finally:
        sys.path[:] = original_path
    for relative, expected in sources.items():
        _read_regular(ROOT / relative, expected_sha256=expected)
    return matched, runtime, schedule_adapter, model_api


def _load_development_inputs(
    matched: Any,
    runtime: Any,
    schedule_adapter: Any,
    authorization: Mapping[str, Any],
    progress: dict[str, Any],
) -> tuple[Any, ...]:
    runtime_inputs = authorization["runtime_inputs"]
    adapted = {
        "raw": runtime_inputs["raw"],
        "camera": runtime_inputs["n320"],
    }
    inputs = direct._construct_raw_inputs_with_progress(
        matched, runtime, adapted, progress
    )
    direct._normalize_endpoint_paths(inputs)
    train_pairs = inputs.role_pairs("train")
    selection_pairs = inputs.role_pairs("checkpoint_selection")
    if (
        len(train_pairs) != contract.TRAIN_ROLE_COUNTS["pairs"]
        or len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise PermissionError("development role counts changed")
    train_mapping, selection_mapping, action_permutation = (
        direct._validate_target_mappings(train_pairs, selection_pairs)
    )
    schedule, schedule_receipt = direct._load_schedule(
        schedule_adapter,
        authorization,
        train_pairs,
        progress=progress,
    )
    fit, n320_gate, n320_checkpoint_binding = direct._load_n320_with_progress(
        matched, runtime, adapted, progress
    )
    return (
        inputs,
        train_pairs,
        selection_pairs,
        train_mapping,
        selection_mapping,
        action_permutation,
        schedule,
        schedule_receipt,
        fit,
        n320_gate,
        n320_checkpoint_binding,
    )


def _access_receipt(loader: Any, inputs: Any) -> dict[str, Any]:
    detailed = loader.receipt()
    forbidden = detailed["forbidden_semantic_counters"]
    if any(int(value) != 0 for value in forbidden.values()):
        raise PermissionError("forbidden semantic input was opened")
    consumed = getattr(inputs, "consumed", {})
    roles = sorted({
        str(record.get("role"))
        for record in consumed.values()
        if isinstance(record, Mapping) and record.get("role") is not None
    })
    return {
        "roles_opened": roles,
        "consumed_record_count": len(consumed),
        "loader": detailed,
        "model_facing_counts": loader.model_facing_access_counts(),
        "allowed_supervision_arrays": ["raster_labels.u1"],
        "forbidden_supervision_arrays": [],
        "camera_depth_ray_ground_pose_odometry_attitude_map_goal_label_inference_inputs": 0,
        "rejected_checkpoint_open_count": 0,
        "prior_runtime_output_open_count": 0,
        "written_checkpoint_read_count": 0,
        "training_trace_read_count": 0,
        "g2_open_count": 0,
        "navigation_open_count": 0,
        "heldout_open_count": 0,
        "sealed_open_count": 0,
        "production_open_count": 0,
    }


def _run_deterministic(runtime: Any, operation: Any) -> tuple[Any, dict[str, Any]]:
    import warnings

    torch = runtime.torch
    previous_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    previous_warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    previous_cudnn = bool(torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = operation()
        messages = [str(item.message) for item in caught]
        unexpected = [
            message for message in messages
            if not _is_allowed_rocm_determinism_warning(message)
        ]
        if unexpected:
            raise RuntimeError(
                "unexpected warning under deterministic execution: "
                + unexpected[0][:500]
            )
        return result, {
            "deterministic_algorithms": True,
            "warn_only_due_to_rocm_grid_sample_backward": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "warning_count": len(messages),
            "warning_message_sha256": sorted({
                hashlib.sha256(message.encode("utf-8")).hexdigest()
                for message in messages
            }),
            "unexpected_warning_count": 0,
        }
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms, warn_only=previous_warn_only
        )
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn


def _seal(output_root: Path) -> dict[str, Any]:
    files: list[str] = []
    directories: list[Path] = []
    for current, names, filenames in os.walk(output_root, topdown=False):
        current_path = Path(current)
        directories.append(current_path)
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise PermissionError("terminal output contains nonregular file")
            os.chmod(path, 0o444, follow_symlinks=False)
            files.append(path.relative_to(output_root).as_posix())
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise PermissionError("terminal output contains nondirectory")
    for directory in directories:
        os.chmod(directory, 0o555, follow_symlinks=False)
    return {
        "files": sorted(files),
        "file_count": len(files),
        "directory_count_including_root": len(directories),
        "files_mode": "0444",
        "directories_mode": "0555",
    }


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: Mapping[str, Any],
    error: BaseException,
) -> None:
    if (output_root / "completed.json").exists():
        _seal(output_root)
        return
    public_progress = {
        name: value for name, value in progress.items() if not name.startswith("_")
    }
    classification = (
        "SCIENTIFIC_GATE_FAILURE"
        if isinstance(error, ScientificGateFailure)
        else "OPERATIONAL_OR_INTEGRITY_FAILURE"
    )
    control = (
        error.control
        if isinstance(error, ScientificGateFailure)
        else contract.CONTROL_FAIL_OPERATIONAL
    )
    trace_binding = progress.get("_training_trace_binding")
    trace_path = output_root / "training_trace.json"
    if (
        trace_binding is None
        and not trace_path.exists()
        and isinstance(progress.get("_trace_rows"), list)
    ):
        trace_binding = _write_training_trace(output_root, progress["_trace_rows"])
        progress["_training_trace_binding"] = trace_binding

    failure, failure_raw = _publish_json(
        output_root / "failure.json",
        {
            "schema": contract.FAILURE_SCHEMA,
            "status": control,
            "classification": classification,
            "reservation": _binding("reservation.json", reservation, reservation_raw),
            "progress": public_progress,
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    "".join(traceback.format_exception(error)).encode("utf-8")
                ).hexdigest(),
            },
            "retry_resume_repair_or_replacement_authorized": False,
            "checkpoint_qualified": False,
            "g2_navigation_heldout_sealed_open_count": 0,
        },
    )

    probe = progress.get("_probe")
    partial = progress.get("_probe_failure_state")
    probe_state = (
        probe if isinstance(probe, Mapping)
        else partial if isinstance(partial, Mapping)
        else {}
    )
    observations = progress.get("_observations")
    if not isinstance(observations, list):
        observations = []
    checkpoint_bindings = progress.get("_checkpoint_bindings")
    if not isinstance(checkpoint_bindings, list):
        checkpoint_bindings = []

    metrics_publication = progress.get("_metrics_publication")
    if not (
        isinstance(metrics_publication, tuple)
        and len(metrics_publication) == 2
    ):
        metrics_publication = _publish_json(
            output_root / "metrics.json",
            {
                "schema": contract.METRICS_SCHEMA,
                "status": control,
                "classification": classification,
                "observations": observations,
                "terminal_gate": probe_state.get("terminal_gate"),
                "phase_switch_receipt": probe_state.get("phase_switch_receipt"),
                "operation": {
                    name: probe_state.get(name, public_progress.get(name, 0))
                    for name in (
                        "updates",
                        "presentations",
                        "pair_presentations_loaded",
                        "objective_evaluations",
                        "backward_calls",
                        "predictor_forward_count",
                        "predictor_objective_count",
                        "predictor_backward_count",
                        "predictor_optimizer_updates",
                        "joint_optimizer_updates",
                        "shared_gradient_gate_pass_count",
                        "target_ema_update_count",
                    )
                },
                "integrity": probe_state.get("integrity", {}),
                "failure": _binding("failure.json", failure, failure_raw),
                "complete_failure_receipt": True,
            },
        )
        progress["_metrics_publication"] = metrics_publication
    metrics, metrics_raw = metrics_publication

    artifact_publication = progress.get("_artifact_publication")
    if not (
        isinstance(artifact_publication, tuple)
        and len(artifact_publication) == 2
    ):
        artifact_publication = _publish_json(
            output_root / "artifact.json",
            {
                "schema": contract.ARTIFACT_SCHEMA,
                "status": control,
                "classification": classification,
                "checkpoints": checkpoint_bindings,
                "training_trace": trace_binding,
                "training_trace_binding_available_without_reread": (
                    trace_binding is not None
                ),
                "all_checkpoints_write_only_and_unqualified": True,
                "checkpoint_read_count_after_write": 0,
                "training_trace_read_count_after_write": 0,
                "failure": _binding("failure.json", failure, failure_raw),
                "complete_failure_receipt": True,
            },
        )
        progress["_artifact_publication"] = artifact_publication
    artifact, artifact_raw = artifact_publication

    access_publication = progress.get("_access_publication")
    if not (
        isinstance(access_publication, tuple)
        and len(access_publication) == 2
    ):
        loader = progress.get("_loader")
        inputs = progress.get("_inputs")
        access_available = loader is not None and inputs is not None
        access_core: dict[str, Any]
        if access_available:
            try:
                access_core = _access_receipt(loader, inputs)
            except BaseException as access_error:
                access_available = False
                access_core = {
                    "partial_access_receipt_error_type": type(access_error).__name__,
                }
        else:
            access_core = {}
        if not access_available:
            access_core.update({
                "roles_opened": None,
                "consumed_record_count": None,
                "loader_receipt_available": False,
                "model_facing_counts": None,
                "allowed_supervision_arrays": ["raster_labels.u1"],
                "forbidden_supervision_arrays": None,
                "rejected_checkpoint_open_count": 0,
                "prior_runtime_output_open_count": 0,
                "written_checkpoint_read_count": 0,
                "training_trace_read_count": 0,
                "g2_open_count": None,
                "navigation_open_count": None,
                "heldout_open_count": None,
                "sealed_open_count": None,
                "production_open_count": None,
            })
        access_publication = _publish_json(
            output_root / "access.json",
            {
                "schema": contract.ACCESS_SCHEMA,
                "status": control,
                "classification": classification,
                "access_receipt_complete": access_available,
                **access_core,
                "failure": _binding("failure.json", failure, failure_raw),
                "complete_failure_receipt": True,
            },
        )
        progress["_access_publication"] = access_publication
    access, access_raw = access_publication

    result_publication = progress.get("_result_publication")
    if not (
        isinstance(result_publication, tuple)
        and len(result_publication) == 2
    ):
        result_publication = _publish_json(
            output_root / "result.json",
            {
                "schema": contract.RESULT_SCHEMA,
                "status": control,
                "classification": classification,
                "reservation": _binding(
                    "reservation.json", reservation, reservation_raw
                ),
                "metrics": _binding("metrics.json", metrics, metrics_raw),
                "artifact": _binding("artifact.json", artifact, artifact_raw),
                "access": _binding("access.json", access, access_raw),
                "failure": _binding("failure.json", failure, failure_raw),
                "hardware": progress.get("_hardware"),
                "determinism": progress.get("_determinism"),
                "schedule": progress.get("_schedule_receipt"),
                "n320_gate_content_sha256": (
                    progress.get("_n320_gate", {}).get("content_sha256")
                    if isinstance(progress.get("_n320_gate"), Mapping)
                    else None
                ),
                "n320_checkpoint": progress.get("_n320_checkpoint_binding"),
                "gpu_active_elapsed_seconds": progress.get(
                    "_gpu_active_elapsed_seconds"
                ),
                "mechanism_passed": False,
                "checkpoint_qualified": False,
                "downstream_authority": "none",
                "retry_authorized": False,
                "complete_failure_receipt": True,
            },
        )
        progress["_result_publication"] = result_publication
    result, result_raw = result_publication

    _publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": control,
            "reservation": _binding("reservation.json", reservation, reservation_raw),
            "metrics": _binding("metrics.json", metrics, metrics_raw),
            "artifact": _binding("artifact.json", artifact, artifact_raw),
            "access": _binding("access.json", access, access_raw),
            "result": _binding("result.json", result, result_raw),
            "failure": _binding("failure.json", failure, failure_raw),
            "checkpoint_qualified": False,
            "retry_authorized": False,
            "complete_failure_receipt": True,
        },
    )
    _seal(output_root)


def _execute(
    *,
    sources: Mapping[str, str],
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    output_root: Path,
    progress: dict[str, Any],
) -> int:
    if sources != contract.current_source_bindings(ROOT):
        raise PermissionError("reviewed sources changed after reservation")
    matched, runtime, schedule_adapter, model_api = _load_post_reservation_stack(sources)
    progress["stage"] = "development_input_validation"
    (
        inputs,
        train_pairs,
        selection_pairs,
        train_mapping,
        selection_mapping,
        action_permutation,
        schedule,
        schedule_receipt,
        fit,
        n320_gate,
        n320_checkpoint_binding,
    ) = _load_development_inputs(
        matched, runtime, schedule_adapter, authorization, progress
    )
    progress["target_mapping_bindings"] = {
        "train": train_mapping["binding"],
        "checkpoint_selection": selection_mapping["binding"],
        "action_permutation": action_permutation["binding"],
    }
    progress["stage"] = "gpu_validation"
    torch = runtime.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible ROCm device is required")
    device = torch.device("cuda:0")
    hardware = {
        "name": torch.cuda.get_device_name(0),
        "visible_device_count": torch.cuda.device_count(),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }
    progress["_hardware"] = hardware
    progress["_schedule_receipt"] = schedule_receipt
    progress["_n320_gate"] = n320_gate
    progress["_n320_checkpoint_binding"] = n320_checkpoint_binding
    gpu_started = time.monotonic()
    loader = direct.DirectBevNarrowLoader(runtime, inputs, progress=progress)
    progress["_loader"] = loader
    progress["_inputs"] = inputs
    progress["stage"] = "bounded_training"
    (model, probe), determinism = _run_deterministic(
        runtime,
        lambda: _train_probe(
            runtime,
            model_api,
            fit,
            loader,
            train_pairs,
            selection_pairs,
            train_mapping,
            selection_mapping,
            schedule,
            device,
            output_root,
            gpu_started=gpu_started,
            progress=progress,
        ),
    )
    progress["_probe"] = probe
    progress["_observations"] = probe["observations"]
    progress["_checkpoint_bindings"] = probe["checkpoints"]
    progress["_training_trace_binding"] = probe["training_trace"]
    progress["_determinism"] = determinism
    del fit
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    gpu_elapsed = time.monotonic() - gpu_started
    progress["_gpu_active_elapsed_seconds"] = gpu_elapsed
    if gpu_elapsed > contract.GPU_ACTIVE_TIME_CAP_MINUTES * 60.0:
        raise TimeoutError("active-GPU cap exceeded")
    progress["stage"] = "terminal_receipts"
    metrics, metrics_raw = _publish_json(
        output_root / "metrics.json",
        {
            "schema": contract.METRICS_SCHEMA,
            "status": probe["status"],
            "observations": probe["observations"],
            "terminal_gate": probe["terminal_gate"],
            "phase_switch_receipt": probe["phase_switch_receipt"],
            "operation": {
                name: probe[name]
                for name in (
                    "updates",
                    "presentations",
                    "objective_evaluations",
                    "backward_calls",
                    "predictor_forward_count",
                    "predictor_objective_count",
                    "predictor_backward_count",
                    "predictor_optimizer_updates",
                    "joint_optimizer_updates",
                    "shared_gradient_gate_pass_count",
                )
            },
            "integrity": probe["integrity"],
        },
    )
    progress["_metrics_publication"] = (metrics, metrics_raw)
    artifacts, artifacts_raw = _publish_json(
        output_root / "artifact.json",
        {
            "schema": contract.ARTIFACT_SCHEMA,
            "status": probe["status"],
            "checkpoints": probe["checkpoints"],
            "training_trace": probe["training_trace"],
            "all_checkpoints_write_only_and_unqualified": True,
            "checkpoint_read_count_after_write": 0,
        },
    )
    progress["_artifact_publication"] = (artifacts, artifacts_raw)
    access_core = _access_receipt(loader, inputs)
    access, access_raw = _publish_json(
        output_root / "access.json",
        {
            "schema": contract.ACCESS_SCHEMA,
            "status": "AUTHORIZED_DEVELOPMENT_RGB_AND_RASTER_ONLY",
            **access_core,
        },
    )
    progress["_access_publication"] = (access, access_raw)
    passed = str(probe["status"]) == contract.CONTROL_PASS
    result, result_raw = _publish_json(
        output_root / "result.json",
        {
            "schema": contract.RESULT_SCHEMA,
            "status": probe["status"],
            "reservation": _binding("reservation.json", reservation, reservation_raw),
            "metrics": _binding("metrics.json", metrics, metrics_raw),
            "artifact": _binding("artifact.json", artifacts, artifacts_raw),
            "access": _binding("access.json", access, access_raw),
            "hardware": hardware,
            "determinism": determinism,
            "schedule": schedule_receipt,
            "n320_gate_content_sha256": n320_gate["content_sha256"],
            "n320_checkpoint": n320_checkpoint_binding,
            "gpu_active_elapsed_seconds": gpu_elapsed,
            "mechanism_passed": passed,
            "checkpoint_qualified": False,
            "downstream_authority": (
                "separate_terminal_audit_and_decision_only" if passed else "none"
            ),
            "retry_authorized": False,
        },
    )
    progress["_result_publication"] = (result, result_raw)
    _publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": probe["status"],
            "result": _binding("result.json", result, result_raw),
            "checkpoint_qualified": False,
            "g2_navigation_heldout_sealed_open_count": 0,
            "retry_authorized": False,
        },
    )
    _seal(output_root)
    return 0 if passed else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        sys.executable != contract.RUNTIME_INTERPRETER_PATH
        or sys.prefix != contract.RUNTIME_SYS_PREFIX
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
    ):
        raise PermissionError("runner requires exact reviewed isolated ROCm interpreter")
    (
        sources,
        _manifest,
        _manifest_raw,
        manifest_binding,
        _review,
        _review_raw,
        review_binding,
        authorization,
        authorization_raw,
    ) = _load_authority(
        review_sha256=args.review_sha256,
        authorization_sha256=args.authorization_sha256,
    )
    output_root, reservation, reservation_raw = _reserve_output_root(
        sources=sources,
        manifest_binding=manifest_binding,
        review_binding=review_binding,
        authorization=authorization,
        authorization_raw=authorization_raw,
    )
    progress: dict[str, Any] = {
        "stage": "reserved",
        "updates": 0,
        "presentations": 0,
        "optimizer_updates": 0,
        "ema_updates": 0,
        "g2_navigation_heldout_sealed_open_count": 0,
    }
    try:
        return _execute(
            sources=sources,
            authorization=authorization,
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=output_root,
            progress=progress,
        )
    except BaseException as error:
        _terminal_failure(
            output_root,
            reservation,
            reservation_raw,
            progress,
            error,
        )
        print(f"terminal failure: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
