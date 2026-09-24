#!/usr/bin/env python3
"""Run the V11 masked current-to-next pair-tubelet JEPA proxy.

Importing this module is source-only.  Torch, PIL, NumPy, generated inputs,
RGB payloads, and the N320 initialization checkpoint remain unreachable until
the frozen authority has been validated and the fresh one-shot root reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_MASKED_CURRENT_NEXT_PAIR_TUBELET_JEPA_V11_PREFLIGHT_JSON"
)


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_go2_rgb_masked_pair_tubelet_v11_runner_contract",
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py",
)
_V10 = _source_only_module(
    "_lewm_go2_rgb_masked_pair_tubelet_v11_frozen_v10_runner",
    ROOT / "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py",
)

# All shared helpers resolve their contract through the frozen module global.
_V10.contract = contract
_V10.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY

ScientificGateFailure = _V10.ScientificGateFailure
RGBOnlyLoader = _V10.RGBOnlyLoader

_binding = _V10._binding
_check_gpu_time = _V10._check_gpu_time
_construct_raw_inputs_with_progress = _V10._construct_raw_inputs_with_progress
_effective_rank = _V10._effective_rank
_empty_mapped_negative_io_receipt = _V10._empty_mapped_negative_io_receipt
_load_authority_pre_reservation = _V10._load_authority_pre_reservation
_load_n320_with_progress = _V10._load_n320_with_progress
_load_schedule = _V10._load_schedule
_normalize_endpoint_paths = _V10._normalize_endpoint_paths
_publish_json = _V10._publish_json
_read_regular = _V10._read_regular
_reserve = _V10._reserve
_run_preflight_after_reservation = _V10._run_preflight_after_reservation
_run_with_rng_preserved = _V10._run_with_rng_preserved
_scalar = _V10._scalar
_scene_derangement = _V10._scene_derangement
_seal_terminal_with_repair = _V10._seal_terminal_with_repair
_snapshot_model = _V10._snapshot_model
_source_authority_receipt = _V10._source_authority_receipt
_state_sha = _V10._state_sha
_terminal_authority_rehash = _V10._terminal_authority_rehash
_terminal_failure = _V10._terminal_failure
_terminal_inventory = _V10._terminal_inventory
_terminal_runtime_rehash = _V10._terminal_runtime_rehash
_write_exclusive = _V10._write_exclusive
_register_output_semantic_metadata = (
    _V10._register_output_semantic_metadata
)


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, Any, Any, Any]:
    """First Torch-capable import point, after reservation and preflight."""
    required = (
        contract.MATCHED_V1_RUNNER_RELATIVE_PATH,
        contract.SCHEDULE_ADAPTER_RELATIVE_PATH,
        contract.V11_MODEL_RELATIVE_PATH,
    )
    if any(
        relative not in sources
        or not contract.is_sha256(sources[relative])
        for relative in required
    ):
        raise PermissionError("reviewed V11 runtime source is incomplete")
    # Rehash the complete reviewed source closure immediately before the first
    # runtime import.  This avoids importing V10's now-unused Phase-B stack.
    for relative, expected_sha256 in sources.items():
        _read_regular(ROOT / relative, expected_sha256=expected_sha256)
    matched_path = ROOT / contract.MATCHED_V1_RUNNER_RELATIVE_PATH
    matched = _V10._load_source_module(
        "_lewm_masked_pair_tubelet_v11_matched_runtime",
        matched_path,
    )
    runtime = matched._load_runtime()
    schedule_path = ROOT / contract.SCHEDULE_ADAPTER_RELATIVE_PATH
    schedule_adapter = _V10._load_source_module(
        "_lewm_masked_pair_tubelet_v11_schedule_adapter",
        schedule_path,
    )
    relative = contract.V11_MODEL_RELATIVE_PATH
    source = ROOT / relative
    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        model_api = _source_only_module(
            "lewm.models.rgb_masked_current_next_pair_tubelet_jepa_v11",
            source,
        )
    finally:
        sys.path[:] = original_path
    observed = Path(model_api.__file__)
    if (
        observed.is_symlink()
        or source.is_symlink()
        or observed.resolve() != source.resolve()
    ):
        raise PermissionError("imported V11 model source changed")
    _read_regular(
        matched_path,
        expected_sha256=sources[contract.MATCHED_V1_RUNNER_RELATIVE_PATH],
    )
    _read_regular(
        schedule_path,
        expected_sha256=sources[contract.SCHEDULE_ADAPTER_RELATIVE_PATH],
    )
    _read_regular(source, expected_sha256=sources[relative])
    return matched, runtime, schedule_adapter, model_api


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise TypeError(f"V11 result lacks {name}")
        return value[name]
    if not hasattr(value, name):
        raise TypeError(f"V11 result lacks {name}")
    return getattr(value, name)


def _phase_a_parameter_partition(model: Any) -> dict[str, Any]:
    """Fail closed on the exact online/EMA parameter ownership boundary."""
    encoder: list[tuple[str, Any]] = []
    other: list[tuple[str, Any]] = []
    frozen: list[tuple[str, Any]] = []
    unexpected: list[str] = []
    encoder_prefixes = tuple(contract.PHASE_A_ENCODER_PARAMETER_PREFIXES)
    other_prefixes = tuple(contract.PHASE_A_AUXILIARY_PARAMETER_PREFIXES)
    frozen_prefixes = tuple(contract.PHASE_A_FROZEN_PARAMETER_PREFIXES)
    for name, parameter in model.named_parameters():
        if name in contract.PHASE_A_EXACT_FROZEN_PARAMETER_NAMES:
            frozen.append((name, parameter))
        elif name.startswith(encoder_prefixes):
            encoder.append((name, parameter))
        elif name.startswith(other_prefixes):
            other.append((name, parameter))
        elif name.startswith(frozen_prefixes):
            frozen.append((name, parameter))
        elif parameter.numel() > 0:
            unexpected.append(name)
    if unexpected:
        raise PermissionError(
            f"V11 Phase-A parameter partition changed: {unexpected[:4]}"
        )
    if not encoder or not other or not frozen:
        raise PermissionError("V11 Phase-A parameter partition is empty")
    if any(parameter.requires_grad for _, parameter in frozen):
        raise PermissionError("V11 frozen parameter became trainable")
    if any(not parameter.requires_grad for _, parameter in (*encoder, *other)):
        raise PermissionError("V11 online parameter became frozen")
    receipt = {
        "encoder_parameter_count": sum(p.numel() for _, p in encoder),
        "encoder_tensor_count": len(encoder),
        "auxiliary_parameter_count": sum(p.numel() for _, p in other),
        "auxiliary_tensor_count": len(other),
        "frozen_parameter_count": sum(p.numel() for _, p in frozen),
        "frozen_tensor_count": len(frozen),
        "encoder_names_sha256": contract.canonical_json_sha256(
            [name for name, _ in encoder]
        ),
        "auxiliary_names_sha256": contract.canonical_json_sha256(
            [name for name, _ in other]
        ),
        "frozen_names_sha256": contract.canonical_json_sha256(
            [name for name, _ in frozen]
        ),
        "target_inventory_optimizer_excluded": True,
        "unused_cls_token_frozen": True,
    }
    expected = getattr(contract, "PHASE_A_PARAMETER_PARTITION", None)
    if expected is not None and receipt != dict(expected):
        raise PermissionError("V11 parameter counts or names changed")
    return {
        "encoder": [parameter for _, parameter in encoder],
        "other": [parameter for _, parameter in other],
        "frozen": [parameter for _, parameter in frozen],
        "receipt": receipt,
    }


def _phase_a_model(
    runtime: Any,
    model_api: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Create V11 once, with the N320 encoder loaded before new draws."""
    torch = runtime.torch
    torch.manual_seed(contract.BASE_INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(contract.BASE_INITIALIZATION_SEED)
    n320_encoder = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in fit.encoder.state_dict().items()
    }
    n320_sha = _state_sha(runtime, n320_encoder)
    config = model_api.MaskedPairTubeletJepaV11Config(
        **contract.v11_model_config()
    )
    model = model_api.MaskedCurrentNextPairTubeletJepaV11(
        n320_encoder_state_dict=n320_encoder,
        config=config,
    )
    if int(model.ema_update_count.detach().cpu().item()) != 0:
        raise RuntimeError("V11 target EMA count did not start at zero")
    if int(torch.count_nonzero(model.online_action_embedding.weight).item()):
        raise RuntimeError("V11 action embedding was not exactly zero")
    if _state_sha(runtime, model.encoder) != n320_sha:
        raise RuntimeError("V11 online N320 encoder migration changed")
    target_encoder = getattr(model, "target_encoder", None)
    if target_encoder is None or _state_sha(runtime, target_encoder) != n320_sha:
        raise RuntimeError("V11 target N320 encoder hard sync changed")
    named_parameters = dict(model.named_parameters())
    ema_inventory = tuple(model.ema_inventory_exact())
    if (
        not ema_inventory
        or ema_inventory != tuple(contract.TARGET_EMA_PARAMETER_PAIRS)
        or any(
            online not in named_parameters
            or target not in named_parameters
            or not torch.equal(
                named_parameters[online], named_parameters[target]
            )
            or named_parameters[target].requires_grad
            for online, target in ema_inventory
        )
    ):
        raise RuntimeError("V11 target EMA hard-sync inventory changed")
    model = model.to(device)
    model.train()
    for name, module in model.named_modules():
        if name.startswith("target_"):
            module.eval()
    partition = _phase_a_parameter_partition(model)
    receipt = {
        "schema": f"{contract.SCHEMA_PREFIX}_phase_a_initialization_v1",
        "seed": contract.BASE_INITIALIZATION_SEED,
        "model_config": contract.v11_model_config(),
        "n320_online_encoder_state_sha256": n320_sha,
        "n320_ema_encoder_state_sha256": n320_sha,
        "online_and_ema_encoder_exactly_equal": True,
        "n320_loaded_before_registered_new_parameter_draws": True,
        "new_parameter_draw_order": list(
            getattr(
                contract,
                "V11_NEW_PARAMETER_DRAW_ORDER",
                (
                    "online_future_mask_token",
                    "online_future_temporal_embedding",
                    "online_action_embedding.weight_exact_zero",
                    "online_future_projector.weight",
                    "online_future_projector.bias_exact_zero",
                ),
            )
        ),
        "action_embedding_exact_zero_at_update_zero": True,
        "target_inventory_hard_sync_count": 1,
        "target_ema_parameter_pair_count": len(ema_inventory),
        "target_ema_inventory_exactly_equal_at_update_zero": True,
        "target_ema_update_count_at_update_zero": 0,
        "target_ema_momentum": getattr(
            contract, "TARGET_EMA_MOMENTUM", 0.996
        ),
        "n320_evidence_head_copy_count": 0,
        "prior_runtime_output_open_count": 0,
        "rejected_checkpoint_open_count": 0,
        "complete_initial_state_sha256": _state_sha(runtime, model),
        "partition": partition["receipt"],
    }
    return model, partition, receipt


def _phase_a_gate_references(
    runtime: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    target_mapping: Mapping[str, Any],
    device: Any,
) -> dict[str, float]:
    """Compute the immutable float32 equal-logit references exactly once."""
    torch = runtime.torch
    same_action = torch.tensor(
        target_mapping["same_action_eligible"],
        dtype=torch.bool,
        device=device,
    )
    executed = torch.tensor(
        [
            contract.ACTION_VOCABULARY.index(str(row["primitive"]))
            for row in selection_pairs
        ],
        dtype=torch.long,
        device=device,
    )
    if (
        len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
        or int(same_action.sum().item())
        != contract.SELECTION_SAME_ACTION_PAIR_COUNT
    ):
        raise PermissionError("V11 gate-reference population changed")
    with torch.no_grad():
        action_reference = torch.nn.functional.cross_entropy(
            torch.zeros(
                (len(selection_pairs), len(contract.ACTION_VOCABULARY)),
                dtype=torch.float32,
                device=device,
            ),
            executed,
        )
        two_target_reference = torch.nn.functional.cross_entropy(
            torch.zeros(
                (contract.SELECTION_SAME_ACTION_PAIR_COUNT, 2),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                contract.SELECTION_SAME_ACTION_PAIR_COUNT,
                dtype=torch.long,
                device=device,
            ),
        )
    if (
        action_reference.dtype != torch.float32
        or two_target_reference.dtype != torch.float32
        or not bool(torch.isfinite(action_reference).item())
        or not bool(torch.isfinite(two_target_reference).item())
    ):
        raise FloatingPointError("V11 gate-reference computation changed")
    return {
        "action_equal_logit_reference": float(action_reference),
        "two_target_equal_logit_reference": float(two_target_reference),
    }


def _phase_a_loss(
    runtime: Any,
    model_api: Any,
    model: Any,
    current_rgb: Any,
    next_rgb: Any,
    deranged_next_rgb: Any,
    action: Any,
    non_hold: Any,
) -> dict[str, Any]:
    """Run the registered current+mask online path and detached target path."""
    torch = runtime.torch
    if (
        current_rgb.ndim != 4
        or next_rgb.shape != current_rgb.shape
        or deranged_next_rgb.shape != current_rgb.shape
        or action.shape != (current_rgb.shape[0], 9)
        or non_hold.shape != (current_rgb.shape[0],)
    ):
        raise ValueError("V11 Phase-A retrieval batch shape changed")
    if (
        action.dtype != torch.float32
        or not torch.equal(
            action.sum(dim=1), torch.ones_like(action[:, 0])
        )
        or not bool(((action == 0.0) | (action == 1.0)).all().item())
    ):
        raise ValueError("V11 action rows are not exact one-hot vectors")
    executed = action.argmax(dim=1)
    if not torch.equal(
        non_hold,
        executed != contract.HOLD_ACTION_INDEX,
    ):
        raise ValueError("V11 non-hold mask changed")

    # This signature has no next/target argument.  Target tensors are created
    # only afterward by the detached EMA branch.
    all_actions = model.predict_all_actions(current_rgb)
    targets = model.build_fixed_current_targets(
        current_rgb,
        next_rgb,
        deranged_next_rgb,
    )
    predictions = _field(all_actions, "normalized_projected_future")
    candidate_indices = _field(all_actions, "action_indices")
    target_values = tuple(
        _field(targets, name)
        for name in ("correct_next", "deranged_next", "no_change_current")
    )
    if (
        tuple(predictions.shape)
        != (current_rgb.shape[0], 9, 256, 192)
        or not torch.equal(
            candidate_indices,
            torch.arange(9, dtype=torch.long, device=current_rgb.device),
        )
        or any(
            tuple(value.shape) != (current_rgb.shape[0], 256, 192)
            or value.requires_grad
            for value in target_values
        )
    ):
        raise PermissionError("V11 action or target population changed")
    objective = model_api.masked_pair_tubelet_objective_v11(
        all_actions,
        targets,
        executed,
    )
    return {
        "loss": _field(objective, "total"),
        "masked_future_jepa_loss": _field(
            objective, "masked_future_jepa"
        ),
        "action_retrieval_loss": _field(
            objective, "action_retrieval"
        ),
        "target_retrieval_loss": _field(
            objective, "target_retrieval"
        ),
        "whitening_variance_loss": _field(
            objective, "whitening_variance"
        ),
        "whitening_covariance_loss": _field(
            objective, "whitening_covariance"
        ),
        "action_retrieval_energies": _field(
            objective, "action_energies"
        ),
        "action_retrieval_logits": _field(objective, "action_logits"),
        "action_retrieval_nll_per_row": _field(
            objective, "action_nll_per_row"
        ),
        "target_retrieval_energies": _field(
            objective, "target_energies"
        ),
        "target_retrieval_logits": _field(objective, "target_logits"),
        "target_retrieval_nll_per_row": _field(
            objective, "target_nll_per_row"
        ),
        "target_candidate_mask": _field(
            objective, "target_candidate_mask"
        ),
        "all_action_predictions": _field(
            all_actions, "normalized_projected_future"
        ),
        "shared_current_patch_tokens": _field(
            all_actions, "shared_current_patch_tokens"
        ),
        "target_correct_next": _field(targets, "correct_next"),
        "target_deranged_next": _field(targets, "deranged_next"),
        "target_no_change_current": _field(targets, "no_change_current"),
        "executed_action_indices": executed,
    }


def _verify_update_zero_action_symmetry_batch(
    torch: Any,
    all_predictions: Any,
) -> tuple[int, int]:
    if (
        all_predictions.ndim != 4
        or all_predictions.shape[0] < 1
        or all_predictions.shape[1] != len(contract.ACTION_VOCABULARY)
    ):
        raise PermissionError("V11 update-zero action population changed")
    comparison_count = 0
    for left in range(9):
        for right in range(left + 1, 9):
            comparison_count += 1
            if not torch.equal(
                all_predictions[:, left], all_predictions[:, right]
            ):
                raise PermissionError(
                    "V11 update-zero action predictions are not bitwise equal"
                )
    return int(all_predictions.shape[0]), comparison_count


def _update_zero_action_symmetry_receipt(
    *, row_count: int, comparison_count: int | None
) -> dict[str, Any]:
    if row_count != contract.SELECTION_ROLE_COUNTS["pairs"]:
        raise PermissionError("V11 update-zero action row count changed")
    if comparison_count != 36:
        raise RuntimeError("V11 update-zero action pair count changed")
    return {
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": comparison_count,
        "all_action_prediction_row_count": row_count,
    }


def _positive_ratio(
    torch: Any,
    numerator: Any,
    denominator: Any,
    name: str,
) -> float:
    numerator_mean = numerator.float().mean()
    denominator_mean = denominator.float().mean()
    if (
        not bool(torch.isfinite(numerator_mean).item())
        or not bool(torch.isfinite(denominator_mean).item())
        or not bool((denominator_mean > 0).item())
    ):
        raise FloatingPointError(f"{name} denominator is not positive finite")
    return float(numerator_mean / denominator_mean)


def _population_health(torch: Any, tokens: Any) -> dict[str, float]:
    """V10 whitening diagnostics on one normalized token population."""
    values = tokens.float()
    centered_samples = values - values.mean(dim=0, keepdim=True)
    flattened = centered_samples.reshape(-1, centered_samples.shape[-1])
    rms_square = flattened.square().mean().detach()
    whitened = flattened / torch.sqrt(rms_square + 1e-4)
    covariance = (
        whitened.T @ whitened / max(1, whitened.shape[0] - 1)
    )
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    return {
        "effective_rank": _effective_rank(torch, values),
        "cross_sample_variance": float(
            values.var(dim=0, unbiased=False).mean()
        ),
        "off_diagonal_covariance": float(
            off_diagonal.square().sum() / float(values.shape[-1])
        ),
        "spatial_diversity": float(
            centered_samples.var(dim=1, unbiased=False).mean()
        ),
    }


def _target_parameters(model: Any) -> list[tuple[str, Any]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(tuple(contract.PHASE_A_FROZEN_PARAMETER_PREFIXES))
    ]


def _run_phase_a_with_strict_determinism(
    runtime: Any,
    operation: Any,
) -> tuple[Any, dict[str, Any]]:
    """V11 admits no warning-only nondeterministic operation."""
    torch = runtime.torch
    previous_enabled = bool(torch.are_deterministic_algorithms_enabled())
    previous_warn_only = bool(
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        result = operation()
    finally:
        torch.use_deterministic_algorithms(True, warn_only=False)
    return result, {
        "strict_deterministic_algorithms_enabled_during_phase_a": True,
        "warn_only_enabled_during_phase_a": False,
        "previous_deterministic_algorithms_enabled": previous_enabled,
        "previous_warn_only_enabled": previous_warn_only,
        "strict_deterministic_algorithms_restored": True,
        "unexpected_warning_count": 0,
    }


def _phase_a_diagnostics(
    runtime: Any,
    model_api: Any,
    model: Any,
    loader: RGBOnlyLoader,
    pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    update: int,
    expected_ema_update_count: int,
    target_mapping: Mapping[str, Any],
    action_permutation: Mapping[str, Any],
    gate_references: Mapping[str, float],
) -> dict[str, Any]:
    """Observe all 495 frozen selection pairs without mutating model or RNG."""
    torch = runtime.torch
    before_state = _state_sha(runtime, model)
    was_training = bool(model.training)
    negative_indices = tuple(target_mapping["negative_indices"])
    same_action_eligible = tuple(target_mapping["same_action_eligible"])
    permuted_action_indices = tuple(
        action_permutation["control_action_indices"]
    )

    def observe() -> dict[str, Any]:
        model.eval()
        current_rows: list[Any] = []
        prediction_rows: list[Any] = []
        correct_target_rows: list[Any] = []
        deranged_target_rows: list[Any] = []
        current_target_rows: list[Any] = []
        action_index_rows: list[Any] = []
        non_hold_rows: list[Any] = []
        action_energy_rows: list[Any] = []
        action_nll_rows: list[Any] = []
        target_energy_rows: list[Any] = []
        target_nll_rows: list[Any] = []
        target_mask_rows: list[Any] = []
        update_zero_row_count = 0
        update_zero_pair_count: int | None = None
        with torch.no_grad():
            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                indices = list(
                    range(
                        start,
                        min(start + contract.MICROBATCH_SIZE, len(pairs)),
                    )
                )
                current, next_rgb, deranged_next, action, non_hold = (
                    loader.batch(
                        pairs,
                        indices,
                        device,
                        role="checkpoint_selection",
                        stage=f"phase_a_diagnostic_update_{update}",
                        mapped_negative_indices=negative_indices,
                        mapped_negative_scope="observation",
                    )
                )
                loss = _phase_a_loss(
                    runtime,
                    model_api,
                    model,
                    current,
                    next_rgb,
                    deranged_next,
                    action,
                    non_hold,
                )
                all_predictions = loss["all_action_predictions"]
                if update == 0:
                    rows, comparisons = (
                        _verify_update_zero_action_symmetry_batch(
                            torch, all_predictions
                        )
                    )
                    update_zero_row_count += rows
                    if update_zero_pair_count is None:
                        update_zero_pair_count = comparisons
                    elif update_zero_pair_count != comparisons:
                        raise RuntimeError(
                            "V11 update-zero action comparisons changed"
                        )
                executed = loss["executed_action_indices"]
                batch_rows = torch.arange(
                    len(indices), dtype=torch.long, device=device
                )
                current_rows.append(current.detach().cpu())
                prediction_rows.append(
                    all_predictions[batch_rows, executed].detach().cpu()
                )
                correct_target_rows.append(
                    loss["target_correct_next"].detach().cpu()
                )
                deranged_target_rows.append(
                    loss["target_deranged_next"].detach().cpu()
                )
                current_target_rows.append(
                    loss["target_no_change_current"].detach().cpu()
                )
                action_index_rows.append(executed.detach().cpu())
                non_hold_rows.append(non_hold.detach().cpu())
                action_energy_rows.append(
                    loss["action_retrieval_energies"].detach().cpu()
                )
                action_nll_rows.append(
                    loss["action_retrieval_nll_per_row"].detach().cpu()
                )
                target_energy_rows.append(
                    loss["target_retrieval_energies"].detach().cpu()
                )
                target_nll_rows.append(
                    loss["target_retrieval_nll_per_row"].detach().cpu()
                )
                target_mask_rows.append(
                    loss["target_candidate_mask"].detach().cpu()
                )

        current_population = torch.cat(current_rows).float()
        prediction = torch.cat(prediction_rows).float()
        target = torch.cat(correct_target_rows).float()
        deranged_target = torch.cat(deranged_target_rows).float()
        current_target = torch.cat(current_target_rows).float()
        requested_indices = torch.cat(action_index_rows).long()
        non_hold = torch.cat(non_hold_rows).bool()
        action_energies = torch.cat(action_energy_rows).float()
        action_nll = torch.cat(action_nll_rows).float()
        target_energies = torch.cat(target_energy_rows).float()
        target_nll = torch.cat(target_nll_rows).float()
        target_mask = torch.cat(target_mask_rows).bool()
        same_action = torch.tensor(same_action_eligible, dtype=torch.bool)
        permuted_indices = torch.tensor(
            permuted_action_indices, dtype=torch.long
        )
        rows = torch.arange(len(pairs), dtype=torch.long)
        expected_shape = (len(pairs), 256, 192)
        if (
            len(pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
            or len(negative_indices) != len(pairs)
            or len(permuted_action_indices) != len(pairs)
            or tuple(prediction.shape) != expected_shape
            or tuple(target.shape) != expected_shape
            or tuple(deranged_target.shape) != expected_shape
            or tuple(current_target.shape) != expected_shape
            or tuple(action_energies.shape) != (len(pairs), 9)
            or tuple(action_nll.shape) != (len(pairs),)
            or tuple(target_energies.shape) != (len(pairs), 3)
            or tuple(target_nll.shape) != (len(pairs),)
            or tuple(target_mask.shape) != (len(pairs), 3)
            or int(non_hold.sum())
            != contract.SELECTION_NON_HOLD_PAIR_COUNT
            or int(same_action.sum())
            != contract.SELECTION_SAME_ACTION_PAIR_COUNT
            or not torch.equal(
                non_hold,
                requested_indices != contract.HOLD_ACTION_INDEX,
            )
            or bool((permuted_indices < 0).any())
            or bool((permuted_indices >= 9).any())
            or bool((permuted_indices == requested_indices).any())
        ):
            raise PermissionError("V11 Phase-A selection population changed")
        expected_mask = torch.ones_like(target_mask)
        expected_mask[~non_hold, 2] = False
        if not torch.equal(target_mask, expected_mask):
            raise PermissionError("V11 target candidate mask changed")

        action_symmetry = (
            _update_zero_action_symmetry_receipt(
                row_count=update_zero_row_count,
                comparison_count=update_zero_pair_count,
            )
            if update == 0
            else None
        )

        # The shuffled-current control preserves the original row's action and
        # target while replacing current RGB with a scene-local derangement.
        current_mapping = torch.tensor(
            _scene_derangement(
                pairs, endpoint_key="current_endpoint_sha256"
            ),
            dtype=torch.long,
        )
        next_mapping = torch.tensor(
            _scene_derangement(pairs, endpoint_key="next_endpoint_sha256"),
            dtype=torch.long,
        )
        shuffled_current_predictions: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                stop = min(start + contract.MICROBATCH_SIZE, len(pairs))
                all_shuffled = model.predict_all_actions(
                    current_population[current_mapping[start:stop]].to(device)
                )
                values = _field(
                    all_shuffled, "normalized_projected_future"
                )
                local_rows = torch.arange(
                    stop - start, dtype=torch.long, device=device
                )
                shuffled_current_predictions.append(
                    values[
                        local_rows,
                        requested_indices[start:stop].to(device),
                    ]
                    .detach()
                    .cpu()
                )
        shuffled_current = torch.cat(shuffled_current_predictions).float()
        shuffled_next = target[next_mapping]
        mean_target = target.mean(dim=0, keepdim=True).expand_as(target)

        def row_mse(left: Any, right: Any) -> Any:
            return (left.float() - right.float()).square().mean(dim=(1, 2))

        true_mse = row_mse(prediction, target)
        shuffled_next_mse = row_mse(prediction, shuffled_next)
        shuffled_current_mse = row_mse(shuffled_current, target)
        mean_target_mse = row_mse(prediction, mean_target)
        executed_energy = action_energies[rows, requested_indices]
        cyclic_indices = (requested_indices + 1) % 9
        cyclic_energy = action_energies[rows, cyclic_indices]
        wrong_mask = torch.ones_like(action_energies, dtype=torch.bool)
        wrong_mask[rows, requested_indices] = False
        hardest_wrong_energy = action_energies.masked_fill(
            ~wrong_mask, float("inf")
        ).min(dim=1).values
        hold_energy = action_energies[non_hold, contract.HOLD_ACTION_INDEX]
        permuted_energy = action_energies[rows, permuted_indices]
        correct_energy = target_energies[:, 0]
        deranged_energy = target_energies[:, 1]
        current_energy = target_energies[non_hold, 2]
        same_correct = correct_energy[same_action]
        same_deranged = deranged_energy[same_action]
        same_two_logits = -torch.stack(
            (same_correct, same_deranged), dim=1
        )
        same_two_nll_rows = torch.nn.functional.cross_entropy(
            same_two_logits,
            torch.zeros(
                contract.SELECTION_SAME_ACTION_PAIR_COUNT,
                dtype=torch.long,
            ),
            reduction="none",
        )
        strict_win_count = int((same_correct < same_deranged).sum().item())

        action_predictions = action_energies.argmin(dim=1)
        per_action: dict[str, dict[str, int | float]] = {}
        recalls: list[float] = []
        for action_index, action_name in enumerate(
            contract.ACTION_VOCABULARY
        ):
            mask = requested_indices == action_index
            count = int(mask.sum().item())
            if count < 1:
                raise PermissionError(
                    f"V11 action population is empty: {action_name}"
                )
            recall = (
                int((action_predictions[mask] == action_index).sum().item())
                / float(count)
            )
            recalls.append(recall)
            per_action[action_name] = {
                "row_count": count,
                "mean_nll": float(action_nll[mask].mean()),
                "recall": recall,
            }

        masked_target_energies = target_energies.masked_fill(
            ~target_mask, float("inf")
        )
        target_predictions = masked_target_energies.argmin(dim=1)
        target_top1_count = int((target_predictions == 0).sum().item())
        per_family: dict[str, dict[str, Any]] = {}
        family_scene_ids: set[str] = set()
        for family in contract.SCENE_FAMILIES:
            family_mask = torch.tensor(
                [row["family"] == family for row in pairs],
                dtype=torch.bool,
            )
            family_same = family_mask & same_action
            family_non_hold = family_mask & non_hold
            scenes = {
                str(row["scene_id"])
                for row in pairs
                if row["family"] == family
            }
            if (
                int(family_mask.sum()) < 1
                or int(family_same.sum()) < 1
                or int(family_non_hold.sum()) < 1
                or len(scenes) != 1
            ):
                raise PermissionError(
                    f"V11 family population is empty: {family}"
                )
            scene_id = next(iter(scenes))
            if (
                scene_id
                != contract.SELECTION_FAMILY_BINDINGS[family]["scene_id"]
            ):
                raise PermissionError(f"V11 family scene changed: {family}")
            family_scene_ids.add(scene_id)
            per_family[family] = {
                "scene_id": scene_id,
                "row_count": int(family_mask.sum()),
                "same_action_row_count": int(family_same.sum()),
                "non_hold_row_count": int(family_non_hold.sum()),
                "deranged_minus_correct_energy": float(
                    (
                        deranged_energy[family_same]
                        - correct_energy[family_same]
                    ).mean()
                ),
                "current_target_minus_correct_energy": float(
                    (
                        target_energies[family_non_hold, 2]
                        - correct_energy[family_non_hold]
                    ).mean()
                ),
                "cyclic_wrong_minus_executed_energy": float(
                    (
                        cyclic_energy[family_mask]
                        - executed_energy[family_mask]
                    ).mean()
                ),
                "hardest_wrong_minus_executed_energy": float(
                    (
                        hardest_wrong_energy[family_mask]
                        - executed_energy[family_mask]
                    ).mean()
                ),
                "hold_minus_non_hold_executed_energy": float(
                    (
                        action_energies[
                            family_non_hold, contract.HOLD_ACTION_INDEX
                        ]
                        - executed_energy[family_non_hold]
                    ).mean()
                ),
                "permuted_minus_executed_energy": float(
                    (
                        permuted_energy[family_mask]
                        - executed_energy[family_mask]
                    ).mean()
                ),
                "hold_action_rows_match_non_hold_rows": True,
            }
        if len(family_scene_ids) != len(contract.SCENE_FAMILIES):
            raise PermissionError("V11 family-to-scene binding changed")

        def positive_family_count(field: str) -> int:
            return sum(
                int(float(value[field]) > 0.0)
                for value in per_family.values()
            )

        finite_tensors = (
            prediction,
            target,
            deranged_target,
            current_target,
            action_energies,
            action_nll,
            target_energies,
            target_nll,
            same_two_nll_rows,
            shuffled_current,
            true_mse,
            shuffled_next_mse,
            shuffled_current_mse,
            mean_target_mse,
        )
        factorized_retrieval = {
            "all_values_finite": bool(
                all(torch.isfinite(value).all().item() for value in finite_tensors)
            ),
            "energy_values_within_closed_zero_four": bool(
                ((action_energies >= 0.0) & (action_energies <= 4.0))
                .all()
                .item()
                and ((target_energies[target_mask] >= 0.0)
                     & (target_energies[target_mask] <= 4.0)).all().item()
            ),
            "target_candidate_order_and_counts_exact": bool(
                torch.equal(target_mask, expected_mask)
            ),
            "same_action_target_mapping_exact": bool(
                target_mapping["binding"]
                == contract.TARGET_MAPPING_BINDINGS["checkpoint_selection"]
            ),
            "selection_action_permutation_exact": bool(
                action_permutation["binding"]
                == contract.SELECTION_ACTION_PERMUTATION_BINDING
            ),
            "reference_values_immutable": True,
            "action_equal_logit_reference": float(
                gate_references["action_equal_logit_reference"]
            ),
            "two_target_equal_logit_reference": float(
                gate_references["two_target_equal_logit_reference"]
            ),
            "action_retrieval_nll": float(action_nll.mean()),
            "action_retrieval_top1_accuracy": float(
                (action_predictions == requested_indices).float().mean()
            ),
            "per_executed_action_action_retrieval": per_action,
            "action_retrieval_macro_balanced_accuracy": (
                sum(recalls) / float(len(recalls))
            ),
            "target_retrieval_nll": float(target_nll.mean()),
            "same_action_target_retrieval_nll": float(
                target_nll[same_action].mean()
            ),
            "hold_target_retrieval_nll": float(target_nll[~non_hold].mean()),
            "non_hold_target_retrieval_nll": float(
                target_nll[non_hold].mean()
            ),
            "same_action_two_target_nll": float(same_two_nll_rows.mean()),
            "target_retrieval_top1_count": target_top1_count,
            "target_retrieval_top1_accuracy": (
                target_top1_count / float(len(pairs))
            ),
            "same_action_strict_win_count": strict_win_count,
            "same_action_strict_win_rate": (
                strict_win_count
                / float(contract.SELECTION_SAME_ACTION_PAIR_COUNT)
            ),
            "same_action_correct_energy": float(same_correct.mean()),
            "same_action_deranged_energy": float(same_deranged.mean()),
            "same_action_correct_to_deranged_ratio": _positive_ratio(
                torch,
                same_correct,
                same_deranged,
                "same-action correct/deranged energy",
            ),
            "non_hold_correct_energy": float(correct_energy[non_hold].mean()),
            "non_hold_current_target_energy": float(current_energy.mean()),
            "non_hold_correct_to_current_ratio": _positive_ratio(
                torch,
                correct_energy[non_hold],
                current_energy,
                "non-hold correct/current energy",
            ),
            "executed_action_energy": float(executed_energy.mean()),
            "cyclic_wrong_action_energy": float(cyclic_energy.mean()),
            "hardest_wrong_action_energy": float(hardest_wrong_energy.mean()),
            "permuted_action_energy": float(permuted_energy.mean()),
            "non_hold_executed_action_energy": float(
                executed_energy[non_hold].mean()
            ),
            "non_hold_hold_action_energy": float(hold_energy.mean()),
            "executed_to_cyclic_ratio": _positive_ratio(
                torch, executed_energy, cyclic_energy, "executed/cyclic energy"
            ),
            "executed_to_hardest_wrong_ratio": _positive_ratio(
                torch,
                executed_energy,
                hardest_wrong_energy,
                "executed/hardest-wrong energy",
            ),
            "executed_to_permuted_ratio": _positive_ratio(
                torch,
                executed_energy,
                permuted_energy,
                "executed/permuted energy",
            ),
            "non_hold_executed_to_hold_ratio": _positive_ratio(
                torch,
                executed_energy[non_hold],
                hold_energy,
                "non-hold executed/hold energy",
            ),
            "all_row_count": len(pairs),
            "same_action_row_count": int(same_action.sum()),
            "fallback_row_count": int((~same_action).sum()),
            "hold_row_count": int((~non_hold).sum()),
            "non_hold_row_count": int(non_hold.sum()),
            "target_candidate_count": int(target_mask.sum()),
            "action_candidate_count": len(contract.ACTION_VOCABULARY),
            "all_wrong_action_candidate_count": len(pairs) * 8,
            "selection_target_mapping_sha256": str(
                target_mapping["binding"]["mapping_sha256"]
            ),
            "selection_action_permutation_sha256": str(
                action_permutation["binding"]["mapping_sha256"]
            ),
            "per_family": per_family,
            "deranged_positive_family_margin_count": positive_family_count(
                "deranged_minus_correct_energy"
            ),
            "current_target_positive_family_margin_count": (
                positive_family_count("current_target_minus_correct_energy")
            ),
            "cyclic_positive_family_margin_count": positive_family_count(
                "cyclic_wrong_minus_executed_energy"
            ),
            "hold_positive_family_margin_count": positive_family_count(
                "hold_minus_non_hold_executed_energy"
            ),
            "permuted_positive_family_margin_count": positive_family_count(
                "permuted_minus_executed_energy"
            ),
        }
        if set(factorized_retrieval) != set(
            contract.FACTORIZED_RETRIEVAL_OBSERVATION_FIELDS
        ):
            raise RuntimeError("V11 factorized retrieval fields changed")

        predicted_health = _population_health(torch, prediction)
        target_health = _population_health(torch, target)
        target_parameters = _target_parameters(model)
        target_gradient_free = bool(target_parameters) and all(
            parameter.grad is None and not parameter.requires_grad
            for _, parameter in target_parameters
        )
        metric = {
            "all_values_finite": bool(
                factorized_retrieval["all_values_finite"]
            ),
            "ema_target_gradient_free": target_gradient_free,
            "pair_count": len(pairs),
            "scene_family_count": len(contract.SCENE_FAMILIES),
            "non_hold_pair_count": int(non_hold.sum()),
            "masked_future_jepa_loss": float(true_mse.mean()),
            "normalized_projected_future_effective_rank": predicted_health[
                "effective_rank"
            ],
            "normalized_projected_future_cross_sample_variance": (
                predicted_health["cross_sample_variance"]
            ),
            "normalized_projected_future_off_diagonal_covariance": (
                predicted_health["off_diagonal_covariance"]
            ),
            "normalized_projected_future_spatial_diversity": (
                predicted_health["spatial_diversity"]
            ),
            "detached_target_future_effective_rank": target_health[
                "effective_rank"
            ],
            "detached_target_future_cross_sample_variance": target_health[
                "cross_sample_variance"
            ],
            "detached_target_future_off_diagonal_covariance": target_health[
                "off_diagonal_covariance"
            ],
            "detached_target_future_spatial_diversity": target_health[
                "spatial_diversity"
            ],
            "true_pair_mse": float(true_mse.mean()),
            "shuffled_next_mse": float(shuffled_next_mse.mean()),
            "shuffled_current_mse": float(shuffled_current_mse.mean()),
            "mean_target_mse": float(mean_target_mse.mean()),
            "factorized_retrieval": factorized_retrieval,
        }
        if set(metric) != set(contract.PHASE_A_METRIC_FIELDS):
            raise RuntimeError("V11 Phase-A metric fields changed")

        ema_update_count = int(
            model.ema_update_count.detach().cpu().item()
        )
        inventory_exact = bool(target_parameters) and all(
            name.startswith(
                tuple(contract.PHASE_A_FROZEN_PARAMETER_PREFIXES)
            )
            for name, _ in target_parameters
        )
        if hasattr(model, "ema_inventory_exact"):
            observed_inventory = tuple(model.ema_inventory_exact())
            expected_inventory = tuple(
                getattr(
                    contract,
                    "TARGET_EMA_PARAMETER_PAIRS",
                    observed_inventory,
                )
            )
            inventory_exact = (
                inventory_exact
                and bool(observed_inventory)
                and observed_inventory == expected_inventory
            )
        integrity = {
            "rng_state_preserved": True,
            "state_mutation_count": 0,
            "future_leakage_prohibition_passed": True,
            "target_path_nonvacuity_passed": bool(
                torch.count_nonzero(target - deranged_target).item()
                and torch.count_nonzero(target - current_target).item()
            ),
            "online_target_autograd_separation_passed": (
                target_gradient_free
                and not target.requires_grad
                and not deranged_target.requires_grad
                and not current_target.requires_grad
            ),
            "ema_inventory_exact": inventory_exact,
            "ema_update_count": ema_update_count,
            "expected_ema_update_count": expected_ema_update_count,
            "normalized_population_exact": True,
            "all_nine_candidates_exact": True,
            "observation_row_count": len(pairs),
        }
        if set(integrity) != set(contract.PHASE_A_OBSERVATION_INTEGRITY_FIELDS):
            raise RuntimeError("V11 observation integrity fields changed")
        return {
            "metric": metric,
            "integrity": integrity,
            "action_indexed_symmetry": action_symmetry,
        }

    try:
        observed = _run_with_rng_preserved(runtime, observe)
    finally:
        if was_training:
            model.train()
            for name, module in model.named_modules():
                if name.startswith("target_"):
                    module.eval()
    if _state_sha(runtime, model) != before_state:
        raise RuntimeError("V11 Phase-A diagnostics mutated model state")
    if (
        observed["integrity"]["ema_update_count"]
        != expected_ema_update_count
    ):
        raise RuntimeError("V11 EMA update count changed at observation")
    return {
        "update": update,
        "role": "checkpoint_selection",
        "metric": observed["metric"],
        "integrity": observed["integrity"],
        "action_indexed_symmetry": observed["action_indexed_symmetry"],
        "mapped_negative_io": loader.mapped_negative_io_receipt(),
        "model_state_sha256_before_and_after": before_state,
        "rng_state_preserved": True,
        "state_mutation_count": 0,
    }


def _phase_a_train(
    runtime: Any,
    model_api: Any,
    fit: Any,
    loader: RGBOnlyLoader,
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    train_target_mapping: Mapping[str, Any],
    selection_target_mapping: Mapping[str, Any],
    selection_action_permutation: Mapping[str, Any],
    schedule: Sequence[int],
    device: Any,
    output_root: Path,
    *,
    gpu_started: float,
    progress: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = runtime.torch
    model, partition, initialization = _phase_a_model(
        runtime, model_api, fit, device
    )
    gate_references = _phase_a_gate_references(
        runtime,
        selection_pairs,
        selection_target_mapping,
        device,
    )
    initialization["factorized_retrieval_gate_references"] = {
        **gate_references,
        "computed_once_after_reservation_before_update_zero": True,
        "action_reference_shape": [495, 9],
        "two_target_reference_shape": [494, 2],
        "dtype": "float32",
    }
    initialization["target_mappings"] = {
        "train": dict(train_target_mapping["binding"]),
        "checkpoint_selection": dict(
            selection_target_mapping["binding"]
        ),
        "selection_action_permutation": dict(
            selection_action_permutation["binding"]
        ),
    }
    trainable = [*partition["encoder"], *partition["other"]]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(partition["encoder"]),
                "lr": 1e-4,
                "group_name": "encoder",
            },
            {
                "params": list(partition["other"]),
                "lr": 3e-4,
                "group_name": "auxiliary",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
    )
    torch.cuda.manual_seed_all(contract.BASE_INITIALIZATION_SEED)
    diagnostics = [
        _phase_a_diagnostics(
            runtime,
            model_api,
            model,
            loader,
            selection_pairs,
            device,
            update=0,
            expected_ema_update_count=0,
            target_mapping=selection_target_mapping,
            action_permutation=selection_action_permutation,
            gate_references=gate_references,
        )
    ]
    update0_health = {
        **dict(diagnostics[0]["metric"]),
        **dict(diagnostics[0]["action_indexed_symmetry"]),
    }
    if set(update0_health) != set(contract.PHASE_A_UPDATE0_FIELDS):
        raise RuntimeError("V11 Phase-A update-zero fields changed")

    update0_gate = contract.evaluate_phase_a_update_zero(
        diagnostics[0]["metric"],
        update0_health,
        diagnostics[0]["integrity"],
    )

    trace: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    continuation_gates: list[dict[str, Any]] = [update0_gate]
    early_failure: dict[str, Any] | None = (
        None if update0_gate["passed"] else update0_gate
    )
    ema_update_count = 0
    loss_names = (
        "loss",
        "masked_future_jepa_loss",
        "action_retrieval_loss",
        "target_retrieval_loss",
        "whitening_variance_loss",
        "whitening_covariance_loss",
    )
    training_updates = (
        range(1, contract.PHASE_A_MAXIMUM_UPDATE + 1)
        if early_failure is None
        else ()
    )
    for update in training_updates:
        _check_gpu_time(
            gpu_started,
            maximum_minutes=contract.PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES,
            stage="V11 Phase A",
        )
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, Any] = {}
        start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
        update_indices = list(
            schedule[start : start + contract.EFFECTIVE_BATCH_SIZE]
        )
        if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
            raise PermissionError("V11 Phase-A schedule ended early")
        for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
            low = microbatch * contract.MICROBATCH_SIZE
            indices = update_indices[low : low + contract.MICROBATCH_SIZE]
            current, next_rgb, deranged_next, action, non_hold = loader.batch(
                train_pairs,
                indices,
                device,
                role="train",
                stage="phase_a_gradient",
                mapped_negative_indices=train_target_mapping[
                    "negative_indices"
                ],
                mapped_negative_scope="training",
            )
            progress["phase_a_pair_loads"] = int(
                progress.get("phase_a_pair_loads", 0)
            ) + len(indices)
            loss = _phase_a_loss(
                runtime,
                model_api,
                model,
                current,
                next_rgb,
                deranged_next,
                action,
                non_hold,
            )
            progress["phase_a_objective_evaluations"] = int(
                progress.get("phase_a_objective_evaluations", 0)
            ) + 1
            if not bool(torch.isfinite(loss["loss"]).item()):
                raise FloatingPointError("V11 Phase-A objective became nonfinite")
            (
                loss["loss"] / contract.MICROBATCHES_PER_UPDATE
            ).backward()
            progress["phase_a_backward_calls"] = int(
                progress.get("phase_a_backward_calls", 0)
            ) + 1
            for name in loss_names:
                contribution = (
                    loss[name].detach()
                    / contract.MICROBATCHES_PER_UPDATE
                )
                sums[name] = sums.get(
                    name, contribution.new_zeros(())
                ) + contribution
        if any(parameter.grad is not None for parameter in partition["frozen"]):
            raise RuntimeError("V11 frozen parameter acquired a gradient")
        gradient_before = torch.nn.utils.clip_grad_norm_(
            trainable, max_norm=1.0
        )
        if not bool(torch.isfinite(gradient_before).item()):
            raise FloatingPointError("V11 Phase-A gradient became nonfinite")
        squared_gradients = [
            parameter.grad.detach().float().square().sum()
            for parameter in trainable
            if parameter.grad is not None
        ]
        if not squared_gradients:
            raise RuntimeError("V11 Phase-A had no online gradients")
        gradient_after = _scalar(torch.stack(squared_gradients).sum().sqrt())
        if gradient_after > 1.00001:
            raise RuntimeError("V11 Phase-A global clip norm changed")
        optimizer.step()
        progress["phase_a_optimizer_updates"] = int(
            progress.get("phase_a_optimizer_updates", 0)
        ) + 1
        model.update_target_ema()
        progress["phase_a_ema_updates"] = int(
            progress.get("phase_a_ema_updates", 0)
        ) + 1
        ema_update_count += 1
        if (
            int(model.ema_update_count.detach().cpu().item())
            != ema_update_count
        ):
            raise RuntimeError("V11 target EMA update count changed")
        progress["phase_a_updates"] = update
        progress["phase_a_presentations"] = (
            update * contract.EFFECTIVE_BATCH_SIZE
        )
        trace.append({
            "schema": f"{contract.SCHEMA_PREFIX}_phase_a_trace_row_v1",
            "update": update,
            "presentation_indices_sha256": (
                contract.canonical_json_sha256(update_indices)
            ),
            "encoder_learning_rate": 1e-4,
            "auxiliary_learning_rate": 3e-4,
            "microbatch_count": contract.MICROBATCHES_PER_UPDATE,
            "pair_presentations": update * contract.EFFECTIVE_BATCH_SIZE,
            "backward_count": update * contract.MICROBATCHES_PER_UPDATE,
            "optimizer_step_count": update,
            "ema_update_count": ema_update_count,
            "global_clip_count": update,
            "gradient_norm_before_clip": _scalar(gradient_before),
            "gradient_norm_after_clip": gradient_after,
            "losses": {
                name: float(sums[name].detach().cpu()) for name in loss_names
            },
        })
        if update in contract.CHECKPOINT_UPDATES:
            diagnostic = _phase_a_diagnostics(
                runtime,
                model_api,
                model,
                loader,
                selection_pairs,
                device,
                update=update,
                expected_ema_update_count=ema_update_count,
                target_mapping=selection_target_mapping,
                action_permutation=selection_action_permutation,
                gate_references=gate_references,
            )
            diagnostics.append(diagnostic)
            if update in {100, 400}:
                previous_metric = (
                    None if update == 100 else diagnostics[-2]["metric"]
                )
                continuation = contract.evaluate_phase_a_continuation(
                    update,
                    diagnostic["metric"],
                    update0_health,
                    diagnostic["integrity"],
                    previous_metric,
                )
                continuation_gates.append(continuation)
                if not continuation["passed"]:
                    early_failure = continuation
            snapshots.append(
                _snapshot_model(
                    runtime,
                    model,
                    output_root,
                    phase="phase_a",
                    update=update,
                    metadata={
                        "initialization": initialization,
                        "partition": partition["receipt"],
                    },
                )
            )
        _check_gpu_time(
            gpu_started,
            maximum_minutes=contract.PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES,
            stage="V11 Phase A",
        )
        if early_failure is not None:
            break

    if ema_update_count != len(trace):
        raise RuntimeError("V11 Phase-A EMA count differs from updates")
    if early_failure is not None:
        terminal_gate = early_failure
    else:
        if ema_update_count != contract.PHASE_A_MAXIMUM_UPDATE:
            raise RuntimeError("V11 terminal update count changed")
        terminal_observation = diagnostics[-1]
        terminal_gate = contract.evaluate_phase_a(
            terminal_observation["metric"],
            update0_health,
            terminal_observation["integrity"],
            diagnostics[-2]["metric"],
        )
    phase_status = str(terminal_gate["control"])
    trace_raw = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in trace
    )
    trace_path = output_root / "phase_a/training_trace.jsonl"
    trace_content_sha256 = contract.canonical_json_sha256(trace)
    _write_exclusive(trace_path, trace_raw)
    _register_output_semantic_metadata(
        trace_path,
        content_sha256=trace_content_sha256,
        row_count=len(trace),
    )
    metrics, metrics_raw = _publish_json(
        output_root / "phase_a/metrics.json",
        {
            "schema": contract.PHASE_A_METRICS_SCHEMA,
            "status": phase_status,
            "observations": diagnostics,
            "update0_health": update0_health,
            "continuation_gates": continuation_gates,
            "terminal_gate": terminal_gate,
            "selection_evaluation_updates": [
                observation["update"] for observation in diagnostics
            ],
            "observer_rerun_count": 0,
            "rng_state_preserved_at_every_observation": True,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    artifact = {
        "schema": contract.PHASE_A_ARTIFACT_SCHEMA,
        "status": phase_status,
        "initialization": initialization,
        "partition": partition["receipt"],
        "snapshots": snapshots,
        "metrics": _binding("phase_a/metrics.json", metrics, metrics_raw),
        "training_trace": {
            "path": "phase_a/training_trace.jsonl",
            "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
            "byte_count": len(trace_raw),
            "row_count": len(trace),
            "content_sha256": trace_content_sha256,
        },
        "updates": len(trace),
        "presentations": len(trace) * contract.EFFECTIVE_BATCH_SIZE,
        "ema_update_count": ema_update_count,
        "selection_observation_count": len(diagnostics),
        "target_mapping_bindings": {
            "train": dict(train_target_mapping["binding"]),
            "checkpoint_selection": dict(
                selection_target_mapping["binding"]
            ),
            "selection_action_permutation": dict(
                selection_action_permutation["binding"]
            ),
        },
        "terminal_online_model_state_sha256": _state_sha(runtime, model),
        "target_state_gradient_count": 0,
        "frozen_state_gradient_count": 0,
        "camera_supervision_array_open_count": (
            loader.supervision_array_open_count
        ),
        "general_raw_v13_frame_loader_call_count": (
            loader.general_frame_loader_call_count
        ),
        "mapped_negative_io": loader.mapped_negative_io_receipt(),
        "gate": terminal_gate,
        "phase_b_entered": False,
        "pass_authorizes": (
            contract.DOWNSTREAM_DENIALS["pass_authorizes"]
            if terminal_gate["passed"]
            else "nothing"
        ),
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    return model, artifact


def _execute_after_reservation(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    output_root: Path,
    progress: dict[str, Any],
) -> int:
    """Execute exactly one capped V11 proxy and terminate without Phase B."""
    progress["stage"] = "post_reservation_source_authority_rehash"
    if contract.current_source_bindings(ROOT) != dict(sources):
        raise PermissionError("V11 reviewed source changed across reservation")
    progress["stage"] = "post_reservation_preflight_validation"
    preflight = _run_preflight_after_reservation(
        launcher_source_sha256=sources[contract.LAUNCHER_RELATIVE_PATH],
        expected_source_authority=_source_authority_receipt(
            review=review,
            review_raw=review_raw,
            authorization=authorization,
            authorization_raw=authorization_raw,
            sources=sources,
        ),
    )
    progress["preflight_validated"] = True

    progress["stage"] = "post_preflight_source_authority_rehash"
    if contract.current_source_bindings(ROOT) != dict(sources):
        raise PermissionError("V11 reviewed source changed after preflight")
    source_manifest_raw = _read_regular(
        ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH,
        expected_sha256=sources[contract.SOURCE_MANIFEST_RELATIVE_PATH],
    )
    observed_review = contract.validate_review(
        contract.parse_canonical_json(review_raw, name="source review rehash"),
        expected_sources=sources,
        source_manifest_raw=source_manifest_raw,
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    observed_authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            authorization_raw, name="execution authorization rehash"
        ),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    if (
        observed_review != dict(review)
        or observed_authorization != dict(authorization)
    ):
        raise PermissionError("V11 authority changed across reservation")

    progress["stage"] = "deferred_runtime_import"
    matched, runtime, schedule_adapter, model_api = (
        _load_post_reservation_stack(sources)
    )
    progress["_runtime"] = runtime
    runtime_authority = authorization["runtime_inputs"]
    adapted_authorization = {
        "raw": runtime_authority["raw"],
        "camera": runtime_authority["camera"],
    }
    progress["stage"] = "raw_authority_and_index_validation"
    inputs = _construct_raw_inputs_with_progress(
        matched,
        runtime,
        adapted_authorization,
        progress,
    )
    _normalize_endpoint_paths(inputs)
    trainer = matched.Trainer(runtime, inputs, output_root, reservation)
    train_pairs = inputs.role_pairs("train")
    selection_pairs = inputs.role_pairs("checkpoint_selection")
    if (
        len(train_pairs) != contract.TRAIN_ROLE_COUNTS["pairs"]
        or len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise PermissionError("V11 development role population changed")
    progress["target_mapping_bindings"] = {}
    train_target_mapping = contract.validate_same_action_target_mapping(
        train_pairs, role="train"
    )
    progress["target_mapping_bindings"]["train"] = dict(
        train_target_mapping["binding"]
    )
    selection_target_mapping = contract.validate_same_action_target_mapping(
        selection_pairs, role="checkpoint_selection"
    )
    progress["target_mapping_bindings"]["checkpoint_selection"] = dict(
        selection_target_mapping["binding"]
    )
    selection_action_permutation = (
        contract.validate_selection_action_permutation(selection_pairs)
    )
    progress["target_mapping_bindings"][
        "selection_action_permutation"
    ] = dict(selection_action_permutation["binding"])
    schedule, schedule_receipt = _load_schedule(
        schedule_adapter,
        authorization,
        train_pairs,
        progress=progress,
    )

    progress["stage"] = "reserved_runtime_device_validation"
    gpu_started = time.monotonic()
    device, hardware = trainer.device()
    if (
        hardware["visible_device_count"] != 1
        or "r9700" not in hardware["name"].casefold().replace(" ", "")
        or hardware["name"] != preflight["visible_device_name"]
        or hardware["total_memory_bytes"] != preflight["total_memory_bytes"]
    ):
        raise PermissionError("V11 runtime GPU differs from preflight")
    progress["gpu_active_started"] = True

    progress["stage"] = "n320_initialization_checkpoint_load"
    fit, gate, camera_binding = _load_n320_with_progress(
        matched,
        runtime,
        adapted_authorization,
        progress,
    )
    progress["n320_checkpoint_loaded"] = True
    progress["stage"] = "phase_a"
    loader = RGBOnlyLoader(runtime, inputs)
    progress["_loader"] = loader
    progress["phase_a_determinism_scope_entered"] = True
    (phase_a_model, phase_a), determinism_receipt = (
        _run_phase_a_with_strict_determinism(
            runtime,
            lambda fit_model=fit: _phase_a_train(
                runtime,
                model_api,
                fit_model,
                loader,
                train_pairs,
                selection_pairs,
                train_target_mapping,
                selection_target_mapping,
                selection_action_permutation,
                schedule,
                device,
                output_root,
                gpu_started=gpu_started,
                progress=progress,
            ),
        )
    )
    progress["phase_a_determinism_restored"] = True
    phase_a["determinism"] = determinism_receipt
    progress["phase_a_updates"] = phase_a["updates"]
    progress["phase_a_presentations"] = phase_a["presentations"]
    progress["phase_a_passed"] = bool(phase_a["gate"]["passed"])
    phase_a_value, phase_a_raw = _publish_json(
        output_root / "phase_a/artifact.json", phase_a
    )

    # A V11 PASS is only a proxy result.  Phase B and every qualification role
    # are deliberately absent from this executor.
    progress["phase_b_entered"] = False
    phase_a_model.to("cpu")
    del phase_a_model
    del fit
    loader.cache.clear()
    runtime.torch.cuda.empty_cache()

    progress["stage"] = "terminal_input_rehash"
    consumed = inputs.rehash_consumed()
    consumed_roles = {
        role for record in consumed["records"] for role in record["roles"]
    }
    required_model_facing_roles = {"checkpoint_selection"}
    if phase_a["presentations"] > 0:
        required_model_facing_roles.add("train")
    permitted_roles = {
        "authority",
        "index",
        "train",
        "checkpoint_selection",
    }
    if (
        "probability_calibration" in consumed_roles
        or not consumed_roles.issubset(permitted_roles)
        or not required_model_facing_roles.issubset(consumed_roles)
        or contract.current_source_bindings(ROOT) != dict(sources)
    ):
        raise PermissionError("V11 consumed an unauthorized role or source")
    runtime_rehash = _terminal_runtime_rehash(authorization)
    access_zero_counters = {
        "probability_calibration_open_count": 0,
        "prior_runtime_output_open_count": 0,
        "rejected_checkpoint_open_count": 0,
        "phase_a_camera_supervision_array_open_count": (
            loader.supervision_array_open_count
        ),
        "phase_a_general_raw_loader_call_count": (
            loader.general_frame_loader_call_count
        ),
        "g2_open_count": 0,
        "navigation_open_count": 0,
        "heldout_open_count": 0,
        "sealed_open_count": 0,
        "production_input_open_count": 0,
        "deployment_input_open_count": 0,
        "observer_rerun_count": 0,
    }
    if (
        tuple(access_zero_counters) != tuple(contract.ACCESS_ZERO_COUNTER_FIELDS)
        or any(access_zero_counters.values())
    ):
        raise PermissionError("V11 forbidden runtime access counter changed")
    mapped_negative_io = loader.mapped_negative_io_receipt()
    if (
        mapped_negative_io["by_scope"]["training"][
            "endpoint_request_count"
        ]
        != phase_a["presentations"]
        or mapped_negative_io["by_scope"]["observation"][
            "endpoint_request_count"
        ]
        != (
            phase_a["selection_observation_count"]
            * contract.SELECTION_ROLE_COUNTS["pairs"]
        )
        or any(
            row["endpoint_request_count"]
            != row["cache_hit_count"] + row["cache_miss_count"]
            or row["cache_miss_count"]
            != row["physical_read_attempt_count"]
            or row["physical_read_attempt_count"]
            != row["physical_read_success_count"]
            for row in mapped_negative_io["by_scope"].values()
        )
    ):
        raise PermissionError("V11 mapped-negative accounting changed")
    authority_rehash = _terminal_authority_rehash(
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
    )
    access, access_raw = _publish_json(
        output_root / "access.json",
        {
            "schema": contract.ACCESS_SCHEMA,
            "status": "ALL_CONSUMED_DEVELOPMENT_INPUTS_REHASHED",
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "roles_opened": sorted(consumed_roles),
            "model_facing_roles_opened": sorted(
                required_model_facing_roles
            ),
            "phase_a": {
                "dedicated_rgb_only_loader": True,
                "general_raw_v13_frame_loader_call_count": (
                    loader.general_frame_loader_call_count
                ),
                "camera_supervision_array_open_count": (
                    loader.supervision_array_open_count
                ),
                "mapped_negative_io": mapped_negative_io,
                "target_mapping_bindings": dict(
                    phase_a["target_mapping_bindings"]
                ),
            },
            "phase_b_entered": False,
            "consumed": consumed,
            "fixed_runtime_input_rehash": runtime_rehash,
            "source_authority_rehash": authority_rehash,
            "schedule": schedule_receipt,
            "n320": {
                "gate_content_sha256": gate["content_sha256"],
                "checkpoint": camera_binding,
                "initialization_only": True,
                "encoder_only": True,
            },
            "reviewed_sources": {
                "count": len(sources),
                "bindings": dict(sources),
                "all_rehashed": True,
            },
            **access_zero_counters,
            "all_consumed_inputs_rehashed": True,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )

    passed = bool(phase_a["gate"]["passed"])
    status = str(phase_a["gate"]["control"])
    progress["stage"] = "result_publication"
    result, result_raw = _publish_json(
        output_root / "result.json",
        {
            "schema": contract.RESULT_SCHEMA,
            "status": status,
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "access": _binding("access.json", access, access_raw),
            "phase_a": _binding(
                "phase_a/artifact.json", phase_a_value, phase_a_raw
            ),
            "phase_b": None,
            "phase_b_entered": False,
            "terminal_control": phase_a["gate"],
            "operation_counts": {
                "optimizer_updates": phase_a["updates"],
                "pair_presentations": phase_a["presentations"],
                "phase_a_ema_updates": phase_a["ema_update_count"],
                "phase_a_mapped_negative_io": mapped_negative_io,
                "phase_b_jepa_objectives": 0,
                "observer_reruns": 0,
            },
            "gpu_active_elapsed_seconds": time.monotonic() - gpu_started,
            "checkpoint_qualified": False,
            "pass_authorizes": (
                contract.DOWNSTREAM_DENIALS["pass_authorizes"]
                if passed
                else "nothing"
            ),
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    inventory = _terminal_inventory(output_root)
    progress["stage"] = "completion_publication"
    _publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": "TERMINAL_PASS" if passed else status,
            "attempt_identity": reservation["attempt_identity"],
            "result": _binding("result.json", result, result_raw),
            "phase_b_entered": False,
            "exact_precompletion_files": inventory["files"],
            "exact_precompletion_file_bindings": inventory["file_bindings"],
            "partial_evidence_bindings": inventory[
                "partial_evidence_bindings"
            ],
            "exact_terminal_files": sorted(
                [*inventory["files"], "completed.json"]
            ),
            "exact_terminal_directories_including_root": inventory[
                "directories_including_root"
            ],
            "all_inputs_rehashed": True,
            "all_terminal_files_sealed_read_only": True,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    progress["completion_published"] = True
    progress["stage"] = "terminal_sealing"
    _seal_terminal_with_repair(output_root)
    return 0 if passed else 2


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    review, review_raw, authorization, authorization_raw, sources = (
        _load_authority_pre_reservation(
            review_file_sha256,
            authorization_file_sha256,
        )
    )
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation, reservation_raw = _reserve(
        output_root,
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
    )
    progress: dict[str, Any] = {
        "stage": "reserved",
        "preflight_validated": False,
        "gpu_active_started": False,
        "schedule_open_attempted": False,
        "schedule_open_succeeded": False,
        "schedule_validated": False,
        "raw_inputs_constructed": False,
        "n320_load_entered": False,
        "n320_gate_open_attempted": False,
        "n320_gate_open_succeeded": False,
        "n320_checkpoint_open_attempted": False,
        "n320_checkpoint_open_succeeded": False,
        "n320_checkpoint_loaded": False,
        "phase_a_determinism_scope_entered": False,
        "phase_a_determinism_restored": False,
        "phase_a_updates": 0,
        "phase_a_optimizer_updates": 0,
        "phase_a_ema_updates": 0,
        "phase_a_presentations": 0,
        "phase_a_pair_loads": 0,
        "phase_a_objective_evaluations": 0,
        "phase_a_backward_calls": 0,
        "phase_a_passed": False,
        "phase_b_entered": False,
        "phase_b_determinism_scope_entered": False,
        "phase_b_determinism_restored": False,
        "phase_b_updates": 0,
        "phase_b_optimizer_updates": 0,
        "phase_b_presentations": 0,
        "phase_b_pair_loads": 0,
        "phase_b_camera_objectives": 0,
        "phase_b_backward_calls": 0,
        "phase_b_passed": False,
        "completion_published": False,
    }
    try:
        return _execute_after_reservation(
            review=review,
            review_raw=review_raw,
            authorization=authorization,
            authorization_raw=authorization_raw,
            sources=sources,
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=output_root,
            progress=progress,
        )
    except BaseException as error:
        try:
            _terminal_failure(
                output_root,
                reservation,
                reservation_raw,
                authorization=authorization,
                error=error,
                progress=progress,
            )
        except BaseException as receipt_error:
            raise RuntimeError(
                "V11 failed and terminal failure publication also failed"
            ) from receipt_error
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("execution requires --run")
    for name in ("review_sha256", "authorization_sha256"):
        if not contract.is_sha256(getattr(args, name)):
            parser.error(
                f"--{name.replace('_', '-')} must be an exact SHA-256"
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
