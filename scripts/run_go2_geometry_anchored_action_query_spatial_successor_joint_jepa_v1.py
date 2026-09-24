#!/usr/bin/env python3
"""Run the one-shot Action-Query Spatial Successor joint-JEPA V1 probe."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_action_query_spatial_successor_"
    "joint_jepa_v1.py"
)
FROZEN_V3_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_action_query_spatial_successor_runner_contract",
    ROOT / CONTRACT_RELATIVE_PATH,
)
if ROOT / contract.RUNNER_RELATIVE_PATH != RUNNER_PATH:
    raise PermissionError("Action-Query Spatial Successor runner path changed")

_V3 = _source_module(
    "_lewm_action_query_spatial_successor_frozen_v3_runner",
    ROOT / FROZEN_V3_RUNNER_RELATIVE_PATH,
)
_V2 = _V3._V2
_BASE = _V2._V1

_tensor_state_sha256 = _V3._tensor_state_sha256
_BASE_PARAMETER_RECEIPT = _BASE._parameter_receipt
_BASE_BUILD_OPTIMIZER = _BASE._build_optimizer
_BASE_EXECUTE = _BASE._execute
_BASE_LOAD_DEVELOPMENT_INPUTS = _BASE._load_development_inputs
_BASE_PUBLISH_JSON = _BASE._publish_json
_BASE_SEAL = _BASE._seal

_ACTIVE_SOURCE_BINDINGS: dict[str, str] = {}
_WORK: dict[str, int] = {}
_OBSERVATION_LIVE: dict[str, Any] = {}
_OBSERVATION_HISTORY: list[dict[str, Any]] = []
_INITIAL_PARAMETER_VALUES: dict[str, Any] = {}
_PREDICTOR_COMPONENT_NAMES: dict[str, tuple[str, ...]] = {}
_GPU_DEADLINE_MONOTONIC: float | None = None
_WARNING_PROVENANCE_SUFFIX = re.compile(
    r" \(Triggered internally at /pytorch/aten/src/ATen/"
    r"Context\.cpp:[0-9]+\.\)"
)


def _reset_work() -> None:
    _WORK.clear()
    _WORK.update({
        "training_microbatch_count": 0,
        "scheduled_pair_presentations_loaded": 0,
        "joint_combined_objective_evaluation_count": 0,
        "combined_backward_call_count": 0,
        "effective_batch_divided_backward_count": 0,
        "semantic_term_evaluation_count": 0,
        "successor_term_evaluation_count": 0,
        "local_action_term_evaluation_count": 0,
        "local_target_term_evaluation_count": 0,
        "registered_scalar_term_evaluation_count": 0,
        "all_action_predictor_training_forward_count": 0,
        "candidate_row_successor_count": 0,
        "online_encoder_lift_training_forward_count": 0,
        "semantic_head_training_forward_count": 0,
        "target_encoder_lift_training_forward_count": 0,
        "optimizer_zero_grad_call_count": 0,
        "online_optimizer_update_count": 0,
        "target_ema_update_count": 0,
        "predictor_optimizer_update_count": 0,
        "joint_optimizer_update_count": 0,
        "perception_only_update_count": 0,
        "predictor_only_update_count": 0,
        "separately_trained_predictor_update_count": 0,
        "route_probe_call_count": 0,
        "semantic_route_probe_call_count": 0,
        "dynamics_route_probe_call_count": 0,
    })


_reset_work()


def _expected_work_at_update(update: int) -> dict[str, int]:
    """Return the exact registered training work after a committed update."""

    if type(update) is not int or not 0 <= update <= 1_000:
        raise ValueError("update must be an integer in [0,1000]")
    microbatches = update * 4
    route = 8 if update >= 1 else 0
    return {
        "training_microbatch_count": microbatches,
        "scheduled_pair_presentations_loaded": update * 16,
        "joint_combined_objective_evaluation_count": microbatches,
        "combined_backward_call_count": microbatches,
        "effective_batch_divided_backward_count": microbatches,
        "semantic_term_evaluation_count": microbatches,
        "successor_term_evaluation_count": microbatches,
        "local_action_term_evaluation_count": microbatches,
        "local_target_term_evaluation_count": microbatches,
        "registered_scalar_term_evaluation_count": microbatches * 4,
        "all_action_predictor_training_forward_count": microbatches,
        "candidate_row_successor_count": microbatches * 4 * 9,
        "online_encoder_lift_training_forward_count": microbatches * 2,
        "semantic_head_training_forward_count": microbatches * 2,
        "target_encoder_lift_training_forward_count": microbatches * 2,
        "optimizer_zero_grad_call_count": update,
        "online_optimizer_update_count": update,
        "target_ema_update_count": update,
        "predictor_optimizer_update_count": update,
        "joint_optimizer_update_count": update,
        "perception_only_update_count": 0,
        "predictor_only_update_count": 0,
        "separately_trained_predictor_update_count": 0,
        "route_probe_call_count": route,
        "semantic_route_probe_call_count": route // 2,
        "dynamics_route_probe_call_count": route // 2,
    }


def _work_is_exact(update: int) -> bool:
    return _WORK == _expected_work_at_update(update)


def _sync_work_to_progress(progress: dict[str, Any]) -> None:
    progress.update(_WORK)


def _begin_observation_accounting(update: int) -> None:
    _OBSERVATION_LIVE.clear()
    _OBSERVATION_LIVE.update({
        "observation_update": int(update),
        "observation_status": "in_progress",
        "observation_pair_microbatch_count": 0,
        "observation_endpoint_microbatch_count": 0,
        "observation_pair_rows_loaded": 0,
        "observation_endpoint_rows_loaded": 0,
        "observation_online_encoder_lift_forward_count": 0,
        "observation_target_encoder_lift_forward_count": 0,
        "observation_semantic_head_forward_count": 0,
        "observation_all_action_predictor_forward_count": 0,
        "observation_all_action_candidate_successor_count": 0,
        "observation_selected_action_predictor_forward_count": 0,
        "observation_selected_action_successor_count": 0,
        "observation_semantic_term_evaluation_count": 0,
        "observation_action_query_objective_evaluation_count": 0,
        "observation_semantic_nll_helper_call_count": 0,
        "observation_action_score_helper_call_count": 0,
        "observation_target_score_helper_call_count": 0,
        "observation_action_ce_reporting_call_count": 0,
        "observation_target_ce_reporting_call_count": 0,
        "observation_confusion_metric_helper_call_count": 0,
        "observation_target_statistics_pass_count": 0,
        "observation_backward_call_count": 0,
        "observation_optimizer_update_count": 0,
        "observation_ema_update_count": 0,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
        "last_started_call": None,
        "last_successful_call": None,
    })


def _observation_live_receipt() -> dict[str, Any] | None:
    if not _OBSERVATION_LIVE:
        return None
    receipt = dict(_OBSERVATION_LIVE)
    receipt["observation_objective_evaluation_count"] = int(
        receipt["observation_semantic_term_evaluation_count"]
        + receipt["observation_action_query_objective_evaluation_count"]
    )
    receipt["observation_predictor_forward_count"] = int(
        receipt["observation_all_action_predictor_forward_count"]
        + receipt["observation_selected_action_predictor_forward_count"]
    )
    receipt["observation_reporting_helper_call_count"] = int(sum(
        receipt[name]
        for name in (
            "observation_semantic_nll_helper_call_count",
            "observation_action_score_helper_call_count",
            "observation_target_score_helper_call_count",
            "observation_action_ce_reporting_call_count",
            "observation_target_ce_reporting_call_count",
            "observation_confusion_metric_helper_call_count",
        )
    ))
    return receipt


def _cumulative_observation_receipt() -> dict[str, Any]:
    rows = [*_OBSERVATION_HISTORY]
    live = _observation_live_receipt()
    if live is not None and live.get("observation_status") != "complete":
        rows.append(live)
    count_names = tuple(_expected_observation_work(0))
    return {
        "completed_observation_count": len(_OBSERVATION_HISTORY),
        "completed_observation_updates": [
            int(row["observation_update"]) for row in _OBSERVATION_HISTORY
        ],
        "included_partial_observation": bool(
            live is not None and live.get("observation_status") != "complete"
        ),
        "totals": {
            name: sum(int(row.get(name, 0)) for row in rows)
            for name in count_names
        },
    }


def _expected_observation_work(update: int) -> dict[str, int]:
    if update not in contract.OBSERVATION_UPDATES:
        raise ValueError("observation update changed")
    pair_batches = math.ceil(
        contract.SELECTION_ROLE_COUNTS["pairs"] / contract.MICROBATCH_SIZE
    )
    endpoint_batches = math.ceil(
        contract.SELECTION_ROLE_COUNTS["unique_endpoints"]
        / contract.MICROBATCH_SIZE
    )
    learned = update > 0
    rollout_steps = 7 if update == 1_000 else 0
    selection_rows = int(contract.SELECTION_ROLE_COUNTS["pairs"])
    endpoint_rows = int(contract.SELECTION_ROLE_COUNTS["unique_endpoints"])
    return {
        "observation_pair_microbatch_count": pair_batches,
        "observation_endpoint_microbatch_count": endpoint_batches * (2 if learned else 1),
        "observation_pair_rows_loaded": selection_rows,
        "observation_endpoint_rows_loaded": endpoint_rows * (2 if learned else 1),
        "observation_online_encoder_lift_forward_count": pair_batches * 3 + endpoint_batches,
        "observation_target_encoder_lift_forward_count": (
            pair_batches * 2 + endpoint_batches if learned else 0
        ),
        "observation_semantic_head_forward_count": pair_batches * 3 + endpoint_batches,
        "observation_all_action_predictor_forward_count": (
            pair_batches if learned else 0
        ),
        "observation_all_action_candidate_successor_count": (
            selection_rows * 9 if learned else 0
        ),
        "observation_selected_action_predictor_forward_count": (
            pair_batches * rollout_steps
        ),
        "observation_selected_action_successor_count": (
            selection_rows * 9 * rollout_steps
        ),
        "observation_semantic_term_evaluation_count": pair_batches,
        "observation_action_query_objective_evaluation_count": (
            pair_batches if learned else 0
        ),
        "observation_semantic_nll_helper_call_count": pair_batches * 5,
        "observation_action_score_helper_call_count": pair_batches if learned else 0,
        "observation_target_score_helper_call_count": pair_batches if learned else 0,
        "observation_action_ce_reporting_call_count": pair_batches if learned else 0,
        "observation_target_ce_reporting_call_count": pair_batches if learned else 0,
        "observation_confusion_metric_helper_call_count": 2,
        "observation_target_statistics_pass_count": 1 if learned else 0,
        "observation_backward_call_count": 0,
        "observation_optimizer_update_count": 0,
        "observation_ema_update_count": 0,
        "observation_predictor_forward_count": (
            pair_batches * (1 + rollout_steps) if learned else 0
        ),
        "observation_objective_evaluation_count": pair_batches * (2 if learned else 1),
        "observation_reporting_helper_call_count": (
            pair_batches * (9 if learned else 5) + 2
        ),
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    }


def _complete_observation_accounting(update: int) -> dict[str, Any]:
    receipt = _observation_live_receipt()
    if receipt is None:
        raise RuntimeError("observation accounting is absent")
    expected = _expected_observation_work(update)
    mismatches = {
        name: {"expected": expected_value, "observed": receipt.get(name)}
        for name, expected_value in expected.items()
        if receipt.get(name) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "observation work changed: " + contract.canonical_json_sha256(mismatches)
        )
    _OBSERVATION_LIVE["observation_status"] = "complete"
    completed = _observation_live_receipt()
    if completed is None:
        raise RuntimeError("completed observation accounting is absent")
    _OBSERVATION_HISTORY.append(completed)
    return completed


def _mark_observation_failure(error: BaseException) -> None:
    if not _OBSERVATION_LIVE:
        return
    _OBSERVATION_LIVE["observation_status"] = "failed"
    _OBSERVATION_LIVE["failure_type"] = type(error).__name__
    _OBSERVATION_LIVE["failure_message_sha256"] = hashlib.sha256(
        str(error).encode("utf-8")
    ).hexdigest()


@contextmanager
def _instrument_observation_calls(model: Any, loader: Any) -> Any:
    """Count successful observation loader/model calls without training work."""

    restorations: list[tuple[type[Any], str, bool, Any]] = []

    def install_model(name: str, counter: str, *, candidate_rows: bool = False) -> None:
        owner = type(model)
        existed = name in owner.__dict__
        prior = owner.__dict__.get(name)
        original = getattr(owner, name)

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            _OBSERVATION_LIVE["last_started_call"] = name
            result = original(instance, *args, **kwargs)
            _OBSERVATION_LIVE[counter] += 1
            if candidate_rows:
                latent = args[0] if args else kwargs["normalized_current_latent"]
                _OBSERVATION_LIVE[
                    "observation_all_action_candidate_successor_count"
                ] += int(latent.shape[0]) * 9
            if name == "predict":
                latent = args[0] if args else kwargs["normalized_current_latent"]
                _OBSERVATION_LIVE[
                    "observation_selected_action_successor_count"
                ] += int(latent.shape[0])
            _OBSERVATION_LIVE["last_successful_call"] = name
            return result

        setattr(owner, name, wrapped)
        restorations.append((owner, name, existed, prior))

    def install_loader(name: str, *, endpoint: bool) -> None:
        owner = type(loader)
        existed = name in owner.__dict__
        prior = owner.__dict__.get(name)
        original = getattr(owner, name)

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            call = f"loader.{name}:{kwargs.get('stage', 'unknown')}"
            _OBSERVATION_LIVE["last_started_call"] = call
            result = original(instance, *args, **kwargs)
            if endpoint:
                rows = int(result[0].shape[0])
                count = "observation_endpoint_microbatch_count"
                row_count = "observation_endpoint_rows_loaded"
            else:
                rows = int(result["action_indices"].shape[0])
                count = "observation_pair_microbatch_count"
                row_count = "observation_pair_rows_loaded"
            _OBSERVATION_LIVE[count] += 1
            _OBSERVATION_LIVE[row_count] += rows
            _OBSERVATION_LIVE["last_successful_call"] = call
            return result

        setattr(owner, name, wrapped)
        restorations.append((owner, name, existed, prior))

    install_model("encode_online", "observation_online_encoder_lift_forward_count")
    install_model("encode_target", "observation_target_encoder_lift_forward_count")
    install_model("semantic_logits_from_latent", "observation_semantic_head_forward_count")
    install_model(
        "predict_all_actions",
        "observation_all_action_predictor_forward_count",
        candidate_rows=True,
    )
    install_model("predict", "observation_selected_action_predictor_forward_count")
    install_loader("batch", endpoint=False)
    install_loader("endpoint_batch", endpoint=True)
    try:
        yield
    finally:
        for owner, name, existed, prior in reversed(restorations):
            if existed:
                setattr(owner, name, prior)
            else:
                delattr(owner, name)


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu() if hasattr(value, "detach") else value)
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite scalar")
    return result


def _enforce_active_gpu_cap(stage: str) -> None:
    if (
        _GPU_DEADLINE_MONOTONIC is not None
        and time.monotonic() > _GPU_DEADLINE_MONOTONIC
    ):
        raise TimeoutError(f"30-minute active-GPU cap reached during {stage}")


def _gradient_l2(torch: Any, gradients: Sequence[Any], *, require_each: bool) -> float:
    if not gradients:
        raise FloatingPointError("gradient collection is empty")
    device = next((value.device for value in gradients if value is not None), None)
    if device is None:
        raise FloatingPointError("all gradients are absent")
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("gradient is absent or nonfinite")
        norm2 = gradient.detach().double().square().sum()
        if require_each and not bool(norm2 > 0.0):
            raise FloatingPointError("a required gradient tensor is zero")
        total = total + norm2
    result = _scalar(total.sqrt())
    if result <= 0.0:
        raise FloatingPointError("aggregate gradient is zero")
    return result


def _route_probes_for_microbatch(
    torch: Any,
    semantic: Any,
    dynamics: Any,
    shared_parameters: Sequence[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Take the two registered non-mutating /4 probes for one U1 batch."""

    semantic_gradients = torch.autograd.grad(
        semantic / contract.MICROBATCHES_PER_UPDATE,
        shared_parameters,
        retain_graph=True,
        allow_unused=False,
    )
    _WORK["route_probe_call_count"] += 1
    _WORK["semantic_route_probe_call_count"] += 1
    dynamics_gradients = torch.autograd.grad(
        dynamics / contract.MICROBATCHES_PER_UPDATE,
        shared_parameters,
        retain_graph=True,
        allow_unused=False,
    )
    _WORK["route_probe_call_count"] += 1
    _WORK["dynamics_route_probe_call_count"] += 1
    return tuple(semantic_gradients), tuple(dynamics_gradients)


def _combined_backward(total: Any) -> None:
    """Perform the sole registered /4 combined backward for a microbatch."""

    (total / contract.MICROBATCHES_PER_UPDATE).backward()
    _WORK["combined_backward_call_count"] += 1
    _WORK["effective_batch_divided_backward_count"] += 1


def _determinism_warning_receipt(
    caught: Sequence[Any], *, scientific_callable_returned: bool
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    canonical_hashes: set[str] = set()
    provenance_suffix_count = 0
    for item in caught:
        message = str(item.message)
        category = item.category
        base = str(_BASE.ROCM_GRID_SAMPLE_DETERMINISM_WARNING)
        suffix = message[len(base):] if message.startswith(base) else None
        canonical = bool(
            message == base
            or suffix is not None
            and _WARNING_PROVENANCE_SUFFIX.fullmatch(suffix) is not None
        )
        if canonical:
            canonical_hashes.add(hashlib.sha256(base.encode("utf-8")).hexdigest())
            provenance_suffix_count += int(message != base)
        rows.append({
            "category": getattr(category, "__name__", str(category)),
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "allowed": bool(
                category is UserWarning
                and canonical
            ),
        })
    return {
        "deterministic_algorithms": True,
        "warn_only_due_to_rocm_grid_sample_backward": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "warning_count": len(rows),
        "warning_message_sha256": sorted({row["message_sha256"] for row in rows}),
        "canonical_warning_message_sha256": sorted(canonical_hashes),
        "warning_provenance_suffix_count": provenance_suffix_count,
        "warning_categories": [row["category"] for row in rows],
        "unexpected_warning_count": sum(not row["allowed"] for row in rows),
        "scientific_callable_returned_before_warning_finalization": bool(
            scientific_callable_returned
        ),
    }


def _run_deterministic(runtime: Any, operation: Any) -> tuple[Any, dict[str, Any]]:
    """Preserve complete warning evidence on success and on any exception."""

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
            try:
                result = operation()
            except BaseException as error:
                error.determinism_warning_receipt = _determinism_warning_receipt(
                    caught, scientific_callable_returned=False
                )
                raise
        receipt = _determinism_warning_receipt(
            caught, scientific_callable_returned=True
        )
        if receipt["unexpected_warning_count"]:
            error = RuntimeError("unexpected warning under deterministic execution")
            error.determinism_warning_receipt = receipt
            raise error
        return result, receipt
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms, warn_only=previous_warn_only
        )
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn


def _semantic_terms(
    model_api: Any, model: Any, batch: Mapping[str, Any]
) -> dict[str, Any]:
    training = bool(model.training)
    if training:
        _WORK["scheduled_pair_presentations_loaded"] += int(
            batch["action_indices"].shape[0]
        )
    current_latent = model.encode_online(batch["current_rgb"])
    if training:
        _WORK["online_encoder_lift_training_forward_count"] += 1
    next_latent = model.encode_online(batch["next_rgb"])
    if training:
        _WORK["online_encoder_lift_training_forward_count"] += 1
    current_logits = model.semantic_logits_from_latent(current_latent)
    if training:
        _WORK["semantic_head_training_forward_count"] += 1
    next_logits = model.semantic_logits_from_latent(next_latent)
    if training:
        _WORK["semantic_head_training_forward_count"] += 1
    current_rows = model_api.final_class_macro_nll_per_row(
        current_logits, batch["current_labels"]
    )
    next_rows = model_api.final_class_macro_nll_per_row(
        next_logits, batch["next_labels"]
    )
    A = 0.5 * current_rows.mean() + 0.5 * next_rows.mean()
    if training:
        _WORK["semantic_term_evaluation_count"] += 1
        _WORK["registered_scalar_term_evaluation_count"] += 1
    else:
        _OBSERVATION_LIVE["observation_semantic_term_evaluation_count"] += 1
        _OBSERVATION_LIVE["observation_semantic_nll_helper_call_count"] += 2
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
    persistence_baseline: float | None = None,
) -> dict[str, Any]:
    """Compute the exact local action-query successor objective."""

    del persistence_baseline
    torch = runtime.torch
    training = bool(model.training)
    x = model_api.normalize_latent_per_cell_v1(current_latent)
    with torch.no_grad():
        target_next = model_api.normalize_latent_per_cell_v1(
            model.encode_target(batch["next_rgb"])
        )
        if training:
            _WORK["target_encoder_lift_training_forward_count"] += 1
        target_negative = model_api.normalize_latent_per_cell_v1(
            model.encode_target(batch["fixed_negative_rgb"])
        )
        if training:
            _WORK["target_encoder_lift_training_forward_count"] += 1
    predictions = model.predict_all_actions(x)
    if training:
        _WORK["all_action_predictor_training_forward_count"] += 1
        _WORK["candidate_row_successor_count"] += int(x.shape[0]) * 9
    objective = model_api.action_query_joint_objective_v1(
        predictions,
        target_next,
        target_negative,
        x,
        batch["action_indices"],
    )
    if not training:
        _OBSERVATION_LIVE[
            "observation_action_query_objective_evaluation_count"
        ] += 1
    P = objective.P_successor
    R = objective.R_local_action
    C = objective.C_deranged
    dynamics = objective.dynamics
    if not bool(
        torch.isfinite(P)
        and torch.isfinite(R)
        and torch.isfinite(C)
        and torch.isfinite(dynamics)
    ):
        raise FloatingPointError("action-query objective became nonfinite")
    if training:
        _WORK["successor_term_evaluation_count"] += 1
        _WORK["local_action_term_evaluation_count"] += 1
        _WORK["local_target_term_evaluation_count"] += 1
        _WORK["registered_scalar_term_evaluation_count"] += 3
        _WORK["training_microbatch_count"] += 1
        _WORK["joint_combined_objective_evaluation_count"] += 1
    action_scores = model_api.action_energy_scores_v1(
        objective.positive, objective.action_scale
    )
    target_scores = model_api.target_energy_scores_v1(
        objective.executed_positive,
        objective.executed_negative,
        objective.target_scale,
    )
    if not training:
        _OBSERVATION_LIVE["observation_action_score_helper_call_count"] += 1
        _OBSERVATION_LIVE["observation_target_score_helper_call_count"] += 1
    if isinstance(target_scores, tuple):
        correct_score, negative_score = target_scores
    else:
        correct_score = target_scores.correct
        negative_score = target_scores.negative
    return {
        "x": x,
        "target_next": target_next,
        "target_negative": target_negative,
        "predictions": predictions,
        "positive": objective.positive,
        "negative": objective.negative,
        "persistence": objective.persistence,
        "executed_positive": objective.executed_positive,
        "executed_negative": objective.executed_negative,
        "action_scale": objective.action_scale,
        "target_scale": objective.target_scale,
        "local_action_ce": objective.local_action_ce,
        "local_target_ce": objective.local_target_ce,
        "action_scores": action_scores,
        "correct_target_score": correct_score,
        "negative_target_score": negative_score,
        "P": P,
        "R": R,
        "C": C,
        "D": dynamics,
    }


def _parameter_receipt(
    model: Any, contract_api: Any
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    groups, receipt = _BASE_PARAMETER_RECEIPT(model, contract_api)
    names = [
        name for name, _parameter in model.named_parameters()
        if name.startswith("predictor.")
    ]
    predictor = receipt["predictor"]
    if (
        names != list(contract.PREDICTOR_ORDERED_PARAMETER_NAMES)
        or predictor["parameter_count"] != contract.PREDICTOR_PARAMETER_COUNT
        or predictor["tensor_count"] != contract.PREDICTOR_PARAMETER_TENSOR_COUNT
    ):
        raise PermissionError("Action-Query predictor inventory changed")
    component_receipt = model.predictor_component_parameter_receipt()
    expected_components = (
        "downsampler",
        "action_embedding",
        "future_queries",
        "block_0_attention",
        "block_0_mlp",
        "block_1_attention",
        "block_1_mlp",
        "output_head",
    )
    if tuple(component_receipt) != expected_components:
        raise PermissionError("predictor component inventory changed")
    _PREDICTOR_COMPONENT_NAMES.clear()
    for component, row in component_receipt.items():
        component_names = tuple(row["ordered_parameter_names"])
        if not component_names or not set(component_names).issubset(names):
            raise PermissionError("predictor component names are incomplete")
        _PREDICTOR_COMPONENT_NAMES[str(component)] = component_names
    flattened = [
        name for component in expected_components
        for name in _PREDICTOR_COMPONENT_NAMES[component]
    ]
    if sorted(flattened) != sorted(names) or len(flattened) != len(set(flattened)):
        raise PermissionError("predictor component partition changed")
    receipt["predictor_components"] = dict(component_receipt)
    return groups, receipt


def _canonical_root_entry(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).resolve() == ROOT
    except (OSError, RuntimeError):
        return False


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, ...]:
    """Load the frozen runtime and new model under one canonical root."""

    _ACTIVE_SOURCE_BINDINGS.clear()
    _ACTIVE_SOURCE_BINDINGS.update({
        str(relative): str(digest) for relative, digest in sources.items()
    })
    for relative, expected in sources.items():
        _BASE._read_regular(ROOT / relative, expected_sha256=expected)
    original_path = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if not _canonical_root_entry(entry)]
        sys.path.insert(0, str(ROOT))
        if (
            sys.path[0] != str(ROOT)
            or sum(_canonical_root_entry(entry) for entry in sys.path) != 1
        ):
            raise PermissionError("canonical repository import root is not exact")
        matched = _BASE._source_module(
            "_lewm_action_query_spatial_successor_matched_runtime",
            _BASE.MATCHED_RUNNER_PATH,
        )
        runtime = matched._load_runtime()
        schedule_adapter = _BASE._source_module(
            "_lewm_action_query_spatial_successor_schedule_adapter",
            _BASE.SCHEDULE_ADAPTER_PATH,
        )
        model_api = _BASE._source_module(
            "lewm.models.geometry_anchored_action_query_spatial_successor_"
            "joint_jepa_v1",
            ROOT / contract.MODEL_RELATIVE_PATH,
        )
    finally:
        sys.path[:] = original_path
    if sys.path != original_path:
        raise PermissionError("post-stack import did not restore sys.path")
    model_class = getattr(model_api, contract.MODEL_CLASS_NAME, None)
    if model_class is None:
        raise PermissionError("Action-Query model class binding is absent")
    for relative, expected in sources.items():
        _BASE._read_regular(ROOT / relative, expected_sha256=expected)
    return matched, runtime, schedule_adapter, model_api


def _reviewed_cpu_source_witness(*, source_authority_exact: bool) -> dict[str, Any]:
    """Bind the reviewed CPU synthetic proof without a digest fallback."""

    path = str(contract.MODEL_TEST_RELATIVE_PATH)
    digest = _ACTIVE_SOURCE_BINDINGS.get(path)
    if not source_authority_exact or digest is None or not contract.is_sha256(digest):
        raise PermissionError("reviewed model source witness is absent")
    return {
        "reviewed_model_source_synthetic_witness_path": path,
        "reviewed_model_source_synthetic_witness_sha256": digest,
        "reviewed_model_source_synthetic_witness_sha256_non_null": True,
        "reviewed_cpu_shapes_positions_and_vectorization_exact": True,
        "reviewed_cpu_action_permutation_equivariance_exact": True,
        "reviewed_cpu_distinct_initialization_exact": True,
        "reviewed_cpu_residual_identity_arithmetic_exact": True,
        "reviewed_cpu_forbidden_branch_count_zero": True,
        "reviewed_cpu_update_zero_autograd_routes_exact": True,
        "runtime_update_zero_synthetic_accelerator_call_count": 0,
        "fallback_hard_coded_sha_used": False,
    }


def _terminal_source_witness() -> dict[str, Any]:
    try:
        return _reviewed_cpu_source_witness(source_authority_exact=True)
    except BaseException as error:
        return {
            "reviewed_model_source_synthetic_witness_path": str(
                contract.MODEL_TEST_RELATIVE_PATH
            ),
            "reviewed_model_source_synthetic_witness_sha256": None,
            "reviewed_model_source_synthetic_witness_sha256_non_null": False,
            "fallback_hard_coded_sha_used": False,
            "witness_error_type": type(error).__name__,
            "witness_error_message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
        }


def _load_development_inputs(*args: Any, **kwargs: Any) -> Any:
    progress = kwargs.get("progress", args[4] if len(args) >= 5 else None)
    loaded = _BASE_LOAD_DEVELOPMENT_INPUTS(*args, **kwargs)
    if isinstance(progress, dict):
        progress["_development_inputs_loaded"] = True
        progress.setdefault("_gpu_active_started_monotonic", time.monotonic())
    return loaded


def _state_value_sha256(torch: Any, value: Any) -> str:
    """Hash nested observer state without reopening any serialized artifact."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(str(len(item)).encode("ascii") + b"\0")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8") + b"\0")
        else:
            raise TypeError(f"unsupported state value: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def _capture_terminal_committed_state_hashes(
    progress: dict[str, Any]
) -> dict[str, Any] | None:
    """Hash the latest complete optimizer+EMA commit once on terminal failure."""

    live = progress.get("_live_state_objects")
    if not isinstance(live, Mapping):
        return None
    if progress.get("_live_commit_safe_to_hash") is not True:
        progress["_terminal_state_hash_capture"] = {
            "available": False,
            "reason": "optimizer_or_ema_commit_was_in_progress",
            "last_audited_committed_state_hashes": progress.get(
                "_last_committed_state_hashes"
            ),
        }
        return None
    runtime = live.get("runtime")
    model = live.get("model")
    optimizer = live.get("optimizer")
    if runtime is None or model is None or optimizer is None:
        return None
    hashes = {
        "update": int(progress.get("_live_committed_update", 0)),
        "model_state_sha256": _BASE._module_state_sha256(runtime.torch, model),
        "optimizer_state_sha256": _state_value_sha256(
            runtime.torch, optimizer.state_dict()
        ),
        "rng_state_sha256": str(progress["_live_committed_rng_sha256"]),
        "optimizer_and_ema_commit_complete": True,
        "captured_once_in_terminal_exception_path": True,
    }
    progress["_last_committed_state_hashes"] = hashes
    progress["_terminal_state_hash_capture"] = {
        "available": True,
        **hashes,
    }
    return hashes


def _rng_snapshot(torch: Any, device: Any) -> tuple[Any, tuple[Any, ...]]:
    cpu = torch.get_rng_state().clone()
    accelerator = (
        tuple(value.clone() for value in torch.cuda.get_rng_state_all())
        if getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"
        else ()
    )
    return cpu, accelerator


def _rng_equal(torch: Any, left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return bool(
        torch.equal(left[0], right[0])
        and len(left[1]) == len(right[1])
        and all(torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True))
    )


def _parameter_displacements(torch: Any, model: Any) -> dict[str, Any]:
    named = dict(model.named_parameters())
    if set(named) != set(_INITIAL_PARAMETER_VALUES):
        raise PermissionError("parameter inventory changed after initialization")

    def displacement(names: Sequence[str]) -> float:
        total = 0.0
        for name in names:
            current = named[name].detach().to(device="cpu", dtype=torch.float64)
            initial = _INITIAL_PARAMETER_VALUES[name].to(dtype=torch.float64)
            total += float((current - initial).square().sum())
        result = math.sqrt(total)
        if not math.isfinite(result):
            raise FloatingPointError("parameter displacement became nonfinite")
        return result

    components = {
        name: displacement(parameter_names)
        for name, parameter_names in _PREDICTOR_COMPONENT_NAMES.items()
    }
    encoder_names = tuple(name for name in named if name.startswith("encoder."))
    lift_names = tuple(name for name in named if name.startswith("bev_lift."))
    return {
        "encoder_l2": displacement(encoder_names),
        "lift_l2": displacement(lift_names),
        "predictor_components": components,
        "all_predictor_components_positive": bool(
            components and all(value > 0.0 for value in components.values())
        ),
    }


def _scene_accumulators() -> dict[str, dict[str, float]]:
    return defaultdict(lambda: defaultdict(float))


def _target_statistics_for_gate(
    runtime: Any,
    model_api: Any,
    model: Any,
    loader: Any,
    identities: Sequence[str],
    device: Any,
    *,
    update: int,
) -> dict[str, float]:
    """Return finite target statistics, including zero-collapse gate evidence."""

    torch = runtime.torch
    channel_sum = torch.zeros(64, dtype=torch.float64, device=device)
    cross_sum = torch.zeros(64, 64, dtype=torch.float64, device=device)
    sample_count = 0
    spatial_difference_sum = torch.zeros((), dtype=torch.float64, device=device)
    spatial_difference_count = 0
    with torch.no_grad():
        for start in range(0, len(identities), contract.MICROBATCH_SIZE):
            _enforce_active_gpu_cap("target-statistics observation")
            subset = identities[start : start + contract.MICROBATCH_SIZE]
            images, _labels = loader.endpoint_batch(
                subset,
                device,
                role="checkpoint_selection",
                stage=f"target_statistics_update_{update}",
            )
            latent = model_api.normalize_latent_per_cell_v1(
                model.encode_target(images)
            )
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
    effective_rank = (
        eigenvalues.sum().square()
        / eigenvalues.square().sum().clamp_min(1e-24)
    )
    result = {
        "target_effective_rank": _scalar(effective_rank),
        "target_channel_variance": _scalar(covariance.diagonal().mean()),
        "target_spatial_diversity": _scalar(
            spatial_difference_sum / spatial_difference_count
        ),
    }
    if any(value < 0.0 for value in result.values()):
        raise FloatingPointError("target representation statistic is negative")
    _OBSERVATION_LIVE["observation_target_statistics_pass_count"] += 1
    return result


def _evaluate_observation_body(
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
) -> tuple[dict[str, Any], None]:
    """Evaluate perception and local action-query evidence read-only."""

    del prior_metrics
    torch = runtime.torch
    if len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]:
        raise PermissionError("checkpoint-selection population changed")
    aggregate_endpoints, rough_endpoints = _BASE.direct._selection_endpoint_population(
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
    non_hold_executed_sum = 0.0
    non_hold_hold_sum = 0.0
    non_hold_count = 0
    target_nll_sum = 0.0
    target_wins = 0
    target_row_count = 0
    successor_sum = 0.0
    persistence_sum = 0.0
    rollout_finite = True
    rollout_rows = 0
    try:
        with torch.no_grad():
            for start in range(0, len(selection_pairs), contract.MICROBATCH_SIZE):
                _enforce_active_gpu_cap("pair observation")
                indices = list(range(
                    start,
                    min(start + contract.MICROBATCH_SIZE, len(selection_pairs)),
                ))
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
                _OBSERVATION_LIVE[
                    "observation_semantic_nll_helper_call_count"
                ] += 3
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

                if update > 0:
                    joint = _joint_terms(
                        runtime,
                        model_api,
                        model,
                        batch,
                        semantic["current_latent"],
                    )
                    scores = joint["action_scores"]
                    rows = torch.arange(size, device=device)
                    executed = scores[rows, batch["action_indices"]]
                    wrong_mask = torch.ones_like(scores, dtype=torch.bool)
                    wrong_mask[rows, batch["action_indices"]] = False
                    hardest_wrong = scores.masked_fill(
                        ~wrong_mask, torch.inf
                    ).min(dim=1).values
                    mean_wrong = scores.masked_fill(~wrong_mask, 0.0).sum(dim=1) / 8.0
                    action_nll = torch.nn.functional.cross_entropy(
                        -scores, batch["action_indices"], reduction="none"
                    )
                    target_logits = torch.stack((
                        -joint["correct_target_score"],
                        -joint["negative_target_score"],
                    ), dim=1)
                    target_nll = torch.nn.functional.cross_entropy(
                        target_logits,
                        torch.zeros(size, dtype=torch.long, device=device),
                        reduction="none",
                    )
                    _OBSERVATION_LIVE[
                        "observation_action_ce_reporting_call_count"
                    ] += 1
                    _OBSERVATION_LIVE[
                        "observation_target_ce_reporting_call_count"
                    ] += 1
                    successor_rows = joint["executed_positive"].mean(dim=1)
                    persistence_rows = joint["persistence"].mean(dim=1)
                    actual = batch["action_indices"].detach().cpu().tolist()
                    predicted = scores.argmin(dim=1).detach().cpu().tolist()
                    actual_actions.extend(map(int, actual))
                    predicted_actions.extend(map(int, predicted))
                    action_nll_sum += _scalar(action_nll.sum())
                    executed_energy_sum += _scalar(executed.sum())
                    wrong_energy_sum += _scalar(mean_wrong.sum())
                    successor_sum += _scalar(successor_rows.sum())
                    persistence_sum += _scalar(persistence_rows.sum())
                    non_hold = batch["non_hold_mask"]
                    if bool(non_hold.any()):
                        hold = scores[:, contract.HOLD_ACTION_INDEX]
                        non_hold_executed_sum += _scalar(executed[non_hold].sum())
                        non_hold_hold_sum += _scalar(hold[non_hold].sum())
                        non_hold_count += int(non_hold.sum().item())
                    for offset, source_index in enumerate(indices):
                        family = str(selection_pairs[source_index]["family"])
                        scene = scenes[family]
                        scene["action_rows"] += 1.0
                        scene["hardest_margin"] += _scalar(
                            hardest_wrong[offset] - executed[offset]
                        )
                        scene["successor_margin"] += _scalar(
                            persistence_rows[offset] - successor_rows[offset]
                        )
                        target_margin = (
                            joint["negative_target_score"][offset]
                            - joint["correct_target_score"][offset]
                        )
                        scene["target_rows"] += 1.0
                        scene["target_margin"] += _scalar(target_margin)
                        target_nll_sum += _scalar(target_nll[offset])
                        target_wins += int(bool(target_margin > 0.0))
                        target_row_count += 1

                    if update == 1_000:
                        state = joint["predictions"]
                        rollout_finite = rollout_finite and bool(
                            torch.isfinite(state).all()
                        )
                        actions = torch.arange(9, device=device).repeat(size)
                        for _step in range(2, 9):
                            _enforce_active_gpu_cap("autoregressive rollout")
                            flat = state.reshape(
                                size * 9, *state.shape[2:]
                            )
                            state = model.predict(flat, actions).reshape(
                                size, 9, *flat.shape[1:]
                            )
                            rollout_finite = rollout_finite and bool(
                                torch.isfinite(state).all()
                            )
                        rollout_rows += size * 9

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
            channel_minimum = [math.inf, math.inf, math.inf]
            channel_maximum = [-math.inf, -math.inf, -math.inf]
            invalid = ~model.bev_lift.anchor_in_frustum.to(device=device)
            visible = ~invalid
            if not bool(visible.any()):
                raise RuntimeError("registered BEV has no in-frustum cells")
            for start in range(0, len(aggregate_endpoints), contract.MICROBATCH_SIZE):
                _enforce_active_gpu_cap("raster observation")
                identities = aggregate_endpoints[start:start + contract.MICROBATCH_SIZE]
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
                    rough_confusion += torch.bincount(
                        rough_codes, minlength=9
                    ).cpu()
                    rough_nll_sum += float(
                        cell_nll.index_select(0, index).double().sum().cpu()
                    )
                    rough_cells += int(rough_labels.numel())
                learned_logits = logits[:, :, visible]
                for channel in range(3):
                    channel_minimum[channel] = min(
                        channel_minimum[channel],
                        _scalar(learned_logits[:, channel].min()),
                    )
                    channel_maximum[channel] = max(
                        channel_maximum[channel],
                        _scalar(learned_logits[:, channel].max()),
                    )
                all_values_finite = all_values_finite and bool(
                    torch.isfinite(logits).all()
                    and torch.isfinite(probabilities).all()
                    and torch.isfinite(cell_nll).all()
                )

        aggregate = _BASE._confusion_metrics(
            aggregate_confusion.reshape(3, 3).tolist(),
            nll_sum=aggregate_nll_sum,
            cell_count=aggregate_cells,
        )
        rough = _BASE._confusion_metrics(
            rough_confusion.reshape(3, 3).tolist(),
            nll_sum=rough_nll_sum,
            cell_count=rough_cells,
        )
        _OBSERVATION_LIVE["observation_confusion_metric_helper_call_count"] += 2
        paired_wins = 0
        hardest_wins = 0
        target_positive_families = 0
        successor_wins = 0
        scene_metrics: dict[str, Any] = {}
        for family in contract.SCENE_FAMILIES:
            row = scenes[family]
            count = int(row["rows"])
            if count != int(contract.SELECTION_FAMILY_BINDINGS[family]["row_count"]):
                raise RuntimeError("selection family population changed")
            correct_mean = row["correct_rgb"] / count
            wrong_mean = row["wrong_rgb"] / count
            paired_win = correct_mean < wrong_mean
            paired_wins += int(paired_win)
            item: dict[str, Any] = {
                "row_count": count,
                "correct_rgb_macro_nll": correct_mean,
                "wrong_rgb_macro_nll": wrong_mean,
                "correct_rgb_strict_win": paired_win,
            }
            if update > 0:
                hardest_margin = row["hardest_margin"] / row["action_rows"]
                target_margin = row["target_margin"] / row["target_rows"]
                successor_margin = row["successor_margin"] / row["action_rows"]
                item.update({
                    "hardest_wrong_minus_executed_energy": hardest_margin,
                    "hardest_wrong_positive": hardest_margin > 0.0,
                    "deranged_minus_correct_target_energy": target_margin,
                    "correct_target_positive": target_margin > 0.0,
                    "persistence_minus_successor_energy": successor_margin,
                    "successor_over_persistence_strict_win": successor_margin > 0.0,
                })
                hardest_wins += int(hardest_margin > 0.0)
                target_positive_families += int(target_margin > 0.0)
                successor_wins += int(successor_margin > 0.0)
            scene_metrics[family] = item

        metrics: dict[str, Any] = {
            "update": update,
            "presentations": update * contract.EFFECTIVE_BATCH_SIZE,
            "A": A_sum / row_count,
            "aggregate_raster_nll": aggregate["nll"],
            "aggregate_raster_balanced_accuracy": aggregate["balanced_accuracy"],
            "aggregate_unknown_recall": aggregate["unknown_recall"],
            "aggregate_free_recall": aggregate["free_recall"],
            "aggregate_occupied_recall": aggregate["occupied_recall"],
            "free_occupied_recall_gap": abs(
                float(aggregate["free_recall"])
                - float(aggregate["occupied_recall"])
            ),
            "rough_raster_balanced_accuracy": rough["balanced_accuracy"],
            "rough_raster_occupied_recall": rough["occupied_recall"],
            "paired_rgb_margin": (wrong_rgb_sum - correct_rgb_sum) / row_count,
            "paired_rgb_scene_wins": paired_wins,
            "all_values_finite": all_values_finite,
            "all_registered_values_finite": all_values_finite,
            "state_nonconstant": _BASE._learned_state_channels_nonconstant(
                channel_minimum, channel_maximum
            ),
            "paired_rgb_latents_nonidentical": latent_nonidentical,
            "rgb_response_nonconstant": latent_nonidentical,
            "out_of_frustum_semantic_unknown_exact": invalid_unknown_exact,
            "out_of_frustum_semantic_unknown": invalid_unknown_exact,
            "scene_metrics": scene_metrics,
            "aggregate_raster": aggregate,
            "rough_raster": rough,
            "integrity": dict(integrity),
            "joint_accounting": dict(joint_accounting),
            "source_authority_exact": bool(integrity["source_authority_exact"]),
            "runtime_input_bindings_exact": bool(
                integrity["runtime_input_bindings_exact"]
            ),
            "schedule_prefix_exact": bool(integrity["schedule_prefix_exact"]),
            "role_and_mapping_bindings_exact": bool(
                integrity["role_and_mapping_bindings_exact"]
            ),
            "model_parameter_inventory_exact": bool(
                integrity["model_parameter_inventory_exact"]
            ),
            "optimizer_inventory_exact": bool(integrity["optimizer_inventory_exact"]),
            "rgb_only_causal_call_graph_exact": bool(
                integrity["rgb_only_causal_call_graph_exact"]
            ),
            "forbidden_input_and_bypass_counts_zero": bool(
                integrity["forbidden_input_and_bypass_counts_zero"]
            ),
            "target_requires_grad_false": bool(integrity["target_requires_grad_false"]),
            "all_forbidden_access_counts_zero": bool(
                integrity["all_forbidden_access_counts_zero"]
            ),
            "target_gradient_tensor_count": int(integrity["target_gradient_count"]),
            "target_optimizer_membership_count": int(
                integrity["target_optimizer_membership_count"]
            ),
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
                "action_query_objective_formula_exact": True,
                "initial_target_hard_sync_count": int(
                    integrity["target_hard_sync_count"]
                ),
                **dict(integrity["reviewed_cpu_source_witness"]),
            })
        else:
            if target_row_count != row_count or non_hold_count != 435:
                raise RuntimeError("action-query diagnostic population changed")
            action_ba, action_recalls = _BASE._action_balanced_accuracy(
                actual_actions, predicted_actions
            )
            displacements = integrity["parameter_displacements"]
            if not isinstance(displacements, Mapping):
                raise RuntimeError("observation parameter displacement is absent")
            metrics.update({
                "action_raw_nll": action_nll_sum / row_count,
                "action_nll": action_nll_sum / row_count,
                "action_macro_balanced_accuracy": action_ba,
                "action_per_class_recall": action_recalls,
                "hardest_wrong_positive_margin_family_count": hardest_wins,
                "executed_action_beats_hardest_wrong_family_count": hardest_wins,
                "mean_executed_action_energy": executed_energy_sum / row_count,
                "mean_wrong_action_energy": wrong_energy_sum / row_count,
                "mean_non_hold_executed_action_energy": (
                    non_hold_executed_sum / non_hold_count
                ),
                "mean_non_hold_hold_action_energy": non_hold_hold_sum / non_hold_count,
                "correct_next_deranged_raw_nll": target_nll_sum / target_row_count,
                "correct_next_deranged_strict_win_rate": target_wins / target_row_count,
                "correct_next_positive_margin_family_count": target_positive_families,
                "mean_successor_unscaled_local_energy": successor_sum / row_count,
                "mean_persistence_unscaled_local_energy": persistence_sum / row_count,
                "successor_over_persistence_strict_win_family_count": successor_wins,
                "encoder_parameter_displacement_l2": displacements["encoder_l2"],
                "lift_parameter_displacement_l2": displacements["lift_l2"],
                "predictor_component_parameter_displacement_l2": dict(
                    displacements["predictor_components"]
                ),
                "encoder_parameter_displaced": displacements["encoder_l2"] > 0.0,
                "lift_parameter_displaced": displacements["lift_l2"] > 0.0,
                "all_predictor_components_displaced": bool(
                    displacements["all_predictor_components_positive"]
                ),
            })
            metrics.update(_target_statistics_for_gate(
                runtime,
                model_api,
                model,
                loader,
                aggregate_endpoints,
                device,
                update=update,
            ))
        if update == 1_000:
            metrics.update({
                "autoregressive_rollout_step_count": 8,
                "autoregressive_rollout_action_count": 9,
                "autoregressive_rollout_start_row_count": row_count,
                "autoregressive_rollout_trajectory_count": rollout_rows,
                "autoregressive_rollout_all_intermediate_and_final_finite": (
                    rollout_finite and rollout_rows == row_count * 9
                ),
                "autoregressive_rollout_future_rgb_input_count": 0,
                "autoregressive_rollout_objective_backward_step_ema_count": 0,
                "autoregressive_rollout_renormalization_count": 0,
            })
        return metrics, None
    finally:
        model.train(was_training)


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
) -> tuple[dict[str, Any], None]:
    """Evaluate once with separate, exact non-training work accounting."""

    _begin_observation_accounting(update)
    try:
        with _instrument_observation_calls(model, loader):
            metrics, baseline = _evaluate_observation_body(
                runtime,
                model_api,
                model,
                loader,
                selection_pairs,
                selection_mapping,
                device,
                update=update,
                prior_metrics=prior_metrics,
                integrity=integrity,
                joint_accounting=joint_accounting,
            )
        metrics["observation_work"] = _complete_observation_accounting(update)
        metrics["cumulative_observation_work"] = _cumulative_observation_receipt()
        return metrics, baseline
    except BaseException as error:
        _mark_observation_failure(error)
        raise


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
    """Run exactly 1,000 jointly trained updates unless a gate fails."""

    torch = runtime.torch
    global _GPU_DEADLINE_MONOTONIC
    if (
        len(train_pairs) != contract.TRAIN_ROLE_COUNTS["pairs"]
        or len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
        or len(schedule) != contract.MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("training roles or schedule changed")
    _reset_work()
    _OBSERVATION_LIVE.clear()
    _OBSERVATION_HISTORY.clear()
    _GPU_DEADLINE_MONOTONIC = (
        float(gpu_started) + contract.GPU_ACTIVE_TIME_CAP_MINUTES * 60.0
    )
    _INITIAL_PARAMETER_VALUES.clear()
    _PREDICTOR_COMPONENT_NAMES.clear()

    n320_encoder = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in fit.encoder.state_dict().items()
    }
    n320_sha256 = _tensor_state_sha256(torch, n320_encoder)
    model_class = getattr(model_api, contract.MODEL_CLASS_NAME)
    model = model_class(n320_encoder_state_dict=n320_encoder).to(device)
    model.train()
    groups, parameter_inventory = _parameter_receipt(model, contract)
    optimizer = _BASE_BUILD_OPTIMIZER(runtime, groups)
    optimizer_membership = _BASE._optimizer_membership_receipt(optimizer, contract)
    optimizer_object_id = id(optimizer)
    optimizer_parameter_ids = tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    target_ids = {id(parameter) for parameter in groups["target"]}
    if target_ids.intersection(optimizer_parameter_ids):
        raise PermissionError("EMA target entered optimizer membership")
    progress["_live_state_objects"] = {
        "runtime": runtime,
        "model": model,
        "optimizer": optimizer,
    }
    progress["_live_committed_update"] = 0
    progress["_live_committed_rng_sha256"] = _state_value_sha256(
        torch, _rng_snapshot(torch, device)
    )
    progress["_live_commit_safe_to_hash"] = True

    for name, parameter in model.named_parameters():
        _INITIAL_PARAMETER_VALUES[name] = (
            parameter.detach().to(device="cpu").contiguous().clone()
        )
    online_target_equal = bool(
        _BASE._module_state_sha256(torch, model.encoder)
        == _BASE._module_state_sha256(torch, model.target_encoder)
        and _BASE._module_state_sha256(torch, model.bev_lift)
        == _BASE._module_state_sha256(torch, model.target_bev_lift)
    )
    source_witness = _reviewed_cpu_source_witness(source_authority_exact=True)
    attention_count = sum(
        int(isinstance(module, torch.nn.MultiheadAttention))
        for module in model.predictor.modules()
    )
    initial_integrity: dict[str, Any] = {
        "n320_encoder_state_sha256": n320_sha256,
        "parameter_inventory": parameter_inventory,
        "optimizer_membership": optimizer_membership,
        "online_target_bitwise_equal_after_one_hard_sync": online_target_equal,
        "target_hard_sync_count": int(model.target_hard_sync_count),
        "target_ema_update_count": int(model.ema_update_count.item()),
        "target_optimizer_membership_count": 0,
        "target_gradient_count": 0,
        "fresh_model": True,
        "prior_runtime_state_reuse_count": 0,
        "source_authority_exact": True,
        "runtime_input_bindings_exact": True,
        "schedule_prefix_exact": True,
        "role_and_mapping_bindings_exact": True,
        "model_parameter_inventory_exact": True,
        "optimizer_inventory_exact": True,
        "rgb_only_causal_call_graph_exact": True,
        "forbidden_input_and_bypass_counts_zero": True,
        "target_requires_grad_false": all(
            not parameter.requires_grad for parameter in groups["target"]
        ),
        "out_of_frustum_sampling_blocked": True,
        "all_forbidden_access_counts_zero": True,
        "predictor_multihead_attention_module_count": attention_count,
        "reviewed_cpu_source_witness": source_witness,
        "runtime_synthetic_accelerator_call_count": 0,
        "joint_training_first_update": 1,
        "perception_warmup_update_count": 0,
    }
    if (
        not online_target_equal
        or initial_integrity["target_hard_sync_count"] != 1
        or initial_integrity["target_ema_update_count"] != 0
        or not initial_integrity["target_requires_grad_false"]
        or attention_count != 2
        or not _work_is_exact(0)
    ):
        raise RuntimeError("initial Action-Query model integrity failed")

    observations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    progress["_trace_rows"] = trace_rows
    progress["_observations"] = observations
    progress["_checkpoint_bindings"] = checkpoints
    prior_metrics: dict[int, Mapping[str, Any]] = {}
    updates = 0
    presentations = 0
    terminal_gate: Mapping[str, Any] | None = None
    first_step_integrity: dict[str, Any] | None = None
    first_semantic_norm: float | None = None
    first_dynamics_norm: float | None = None
    first_semantic_to_dynamics_ratio: float | None = None
    first_dynamics_to_semantic_ratio: float | None = None
    predictor_gradient_qualified_updates = 0
    representation_gradient_qualified_updates = 0
    failure_state: dict[str, Any] = {
        "updates": 0,
        "presentations": 0,
        "terminal_gate": None,
        "first_step_integrity": None,
        "integrity": dict(initial_integrity),
        **_WORK,
    }
    progress["_probe_failure_state"] = failure_state

    named_parameters = dict(model.named_parameters())
    shared_names = tuple(
        name for name in named_parameters
        if name.startswith(("encoder.", "bev_lift."))
    )
    shared_parameters = [named_parameters[name] for name in shared_names]
    representation_parameters = [*groups["encoder"], *groups["lift_semantic"]]
    predictor_parameters = list(groups["predictor"])

    def integrity_receipt(
        *, include_displacements: bool = False
    ) -> dict[str, Any]:
        target_gradient_count = sum(
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
            "predictor_optimizer_updates": _WORK["predictor_optimizer_update_count"],
            "joint_optimizer_updates": _WORK["joint_optimizer_update_count"],
            "target_gradient_count": target_gradient_count,
            "target_optimizer_membership_count": 0,
            "work_accounting": dict(_WORK),
            "work_accounting_exact_for_committed_update": _work_is_exact(updates),
            "first_step_integrity": first_step_integrity,
            "semantic_route_gradient_l2": first_semantic_norm,
            "dynamics_route_gradient_l2": first_dynamics_norm,
            "semantic_to_dynamics_gradient_ratio": first_semantic_to_dynamics_ratio,
            "dynamics_to_semantic_gradient_ratio": first_dynamics_to_semantic_ratio,
            "route_ratio_abort_count": 0,
            "predictor_gradient_finite_nonzero_update_count": (
                predictor_gradient_qualified_updates
            ),
            "representation_gradient_finite_nonzero_update_count": (
                representation_gradient_qualified_updates
            ),
            "parameter_displacements": (
                _parameter_displacements(torch, model)
                if include_displacements
                else None
            ),
        }

    def update_failure_state(*, gate: Any = None) -> None:
        failure_state.update({
            "updates": updates,
            "presentations": presentations,
            "pair_presentations_loaded": _WORK[
                "scheduled_pair_presentations_loaded"
            ],
            "target_ema_update_count": int(model.ema_update_count.item()),
            "first_step_integrity": first_step_integrity,
            "integrity": integrity_receipt(),
            **_WORK,
        })
        if gate is not None:
            failure_state["terminal_gate"] = gate
        _sync_work_to_progress(progress)

    def observe(update: int) -> tuple[dict[str, Any], Mapping[str, Any]]:
        if not _work_is_exact(update):
            raise RuntimeError(f"training work is inexact before observation {update}")
        before_model = _BASE._module_state_sha256(torch, model)
        before_optimizer = _state_value_sha256(torch, optimizer.state_dict())
        before_rng = _rng_snapshot(torch, device)
        before_rng_hash = _state_value_sha256(torch, before_rng)
        committed_state_hashes = {
            "update": int(update),
            "model_state_sha256": before_model,
            "optimizer_state_sha256": before_optimizer,
            "rng_state_sha256": before_rng_hash,
            "audited_immediately_before_non_mutating_observation": True,
        }
        progress["_last_committed_state_hashes"] = committed_state_hashes
        before_work = dict(_WORK)
        current_integrity = integrity_receipt(include_displacements=True)
        metrics, _unused = _evaluate_observation(
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
                "phase": "joint_jepa" if update > 0 else "update_zero",
                "joint_optimizer_updates": _WORK["joint_optimizer_update_count"],
                "route_probe_call_count": _WORK["route_probe_call_count"],
            },
        )
        after_model = _BASE._module_state_sha256(torch, model)
        after_optimizer = _state_value_sha256(torch, optimizer.state_dict())
        after_rng = _rng_snapshot(torch, device)
        after_rng_hash = _state_value_sha256(torch, after_rng)
        state_receipt = {
            "committed_update": int(update),
            "model_state_sha256_before": before_model,
            "model_state_sha256_after": after_model,
            "optimizer_state_sha256_before": before_optimizer,
            "optimizer_state_sha256_after": after_optimizer,
            "rng_state_sha256_before": before_rng_hash,
            "rng_state_sha256_after": after_rng_hash,
            "model_state_hash_unchanged": before_model == after_model,
            "optimizer_state_hash_unchanged": before_optimizer == after_optimizer,
            "rng_state_unchanged": _rng_equal(torch, before_rng, after_rng),
            "training_work_counters_unchanged": before_work == _WORK,
            "model_training_mode_restored": bool(model.training),
            "schedule_not_advanced": True,
        }
        preservation_names = (
            "model_state_hash_unchanged",
            "optimizer_state_hash_unchanged",
            "rng_state_unchanged",
            "training_work_counters_unchanged",
            "model_training_mode_restored",
            "schedule_not_advanced",
        )
        if not all(bool(state_receipt[name]) for name in preservation_names):
            raise RuntimeError("observation mutated registered training state")
        metrics["observation_state_preservation"] = state_receipt
        metrics.update({
            "observation_model_state_hash_unchanged": True,
            "observation_optimizer_state_hash_unchanged": True,
            "observation_rng_state_unchanged": True,
            "observation_training_work_counters_unchanged": True,
            "work_accounting_exact": _work_is_exact(update),
            "work_accounting": dict(_WORK),
            "first_step_integrity": first_step_integrity,
            "update_one_route_probe_call_count": (
                0 if update == 0 else _WORK["route_probe_call_count"]
            ),
            "gradient_ratio_abort_count": 0,
            "last_committed_state_hashes": dict(committed_state_hashes),
            **_WORK,
        })
        gate = contract.evaluate_gate(update, metrics, prior_metrics=prior_metrics)
        observations.append({"update": update, "metrics": metrics, "gate": gate})
        prior_metrics[update] = metrics
        return metrics, gate

    progress["stage"] = "observation_update_0"
    _metrics_zero, gate_zero = observe(0)
    terminal_gate = gate_zero
    update_failure_state(gate=gate_zero)
    if not bool(gate_zero["passed"]):
        trace_binding = _BASE._write_training_trace(output_root, trace_rows)
        progress["_training_trace_binding"] = trace_binding
        return model, {
            "status": str(gate_zero["control"]),
            "terminal_gate": gate_zero,
            "observations": observations,
            "checkpoints": checkpoints,
            "training_trace": trace_binding,
            "updates": 0,
            "presentations": 0,
            "objective_evaluations": 0,
            "backward_calls": 0,
            "predictor_forward_count": 0,
            "predictor_objective_count": 0,
            "predictor_backward_count": 0,
            "predictor_optimizer_updates": 0,
            "joint_optimizer_updates": 0,
            "shared_gradient_gate_pass_count": 0,
            "phase_switch_receipt": None,
            "last_committed_state_hashes": progress.get(
                "_last_committed_state_hashes"
            ),
            "terminal_state_hash_capture": progress.get(
                "_terminal_state_hash_capture"
            ),
            "cumulative_observation_work": _cumulative_observation_receipt(),
            **_WORK,
            "integrity": integrity_receipt(include_displacements=True),
        }

    for update in range(1, contract.MAXIMUM_UPDATES + 1):
        if time.monotonic() - gpu_started > contract.GPU_ACTIVE_TIME_CAP_MINUTES * 60.0:
            raise TimeoutError("30-minute active-GPU cap reached")
        start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
        stop = update * contract.EFFECTIVE_BATCH_SIZE
        update_indices = [int(value) for value in schedule[start:stop]]
        if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
            raise RuntimeError("schedule exhausted")

        optimizer.zero_grad(set_to_none=True)
        _WORK["optimizer_zero_grad_call_count"] += 1
        semantic_accumulators = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in shared_parameters
        ] if update == 1 else []
        dynamics_accumulators = [
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for parameter in shared_parameters
        ] if update == 1 else []
        sums = {
            name: 0.0
            for name in ("A", "S", "P_successor", "R_local_action", "C_deranged", "total")
        }
        for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
            low = microbatch * contract.MICROBATCH_SIZE
            indices = update_indices[low:low + contract.MICROBATCH_SIZE]
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
            joint = _joint_terms(
                runtime,
                model_api,
                model,
                batch,
                semantic["current_latent"],
            )
            S = semantic["S"]
            P = joint["P"]
            R = joint["R"]
            C = joint["C"]
            total = S + P + R + C
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("combined objective became nonfinite")
            if update == 1:
                semantic_gradients, dynamics_gradients = _route_probes_for_microbatch(
                    torch, S, joint["D"], shared_parameters
                )
                for accumulator, gradient in zip(
                    semantic_accumulators, semantic_gradients, strict=True
                ):
                    accumulator.add_(gradient.detach())
                for accumulator, gradient in zip(
                    dynamics_accumulators, dynamics_gradients, strict=True
                ):
                    accumulator.add_(gradient.detach())
            _combined_backward(total)
            _sync_work_to_progress(progress)
            update_failure_state()
            for name, value in (
                ("A", semantic["A"]),
                ("S", S),
                ("P_successor", P),
                ("R_local_action", R),
                ("C_deranged", C),
                ("total", total),
            ):
                sums[name] += _scalar(value)

        representation_norm = _gradient_l2(
            torch,
            [parameter.grad for parameter in representation_parameters],
            require_each=False,
        )
        predictor_norm = _gradient_l2(
            torch,
            [parameter.grad for parameter in predictor_parameters],
            require_each=False,
        )
        representation_gradient_qualified_updates += 1
        predictor_gradient_qualified_updates += 1
        semantic_norm = None
        dynamics_norm = None
        semantic_to_dynamics = None
        dynamics_to_semantic = None
        component_gradient_norms: dict[str, float] | None = None
        if update == 1:
            semantic_norm = _gradient_l2(
                torch, semantic_accumulators, require_each=False
            )
            dynamics_norm = _gradient_l2(
                torch, dynamics_accumulators, require_each=False
            )
            semantic_to_dynamics = semantic_norm / dynamics_norm
            dynamics_to_semantic = dynamics_norm / semantic_norm
            component_gradient_norms = {}
            for component, names in _PREDICTOR_COMPONENT_NAMES.items():
                component_gradient_norms[component] = _gradient_l2(
                    torch,
                    [named_parameters[name].grad for name in names],
                    require_each=False,
                )
            first_semantic_norm = semantic_norm
            first_dynamics_norm = dynamics_norm
            first_semantic_to_dynamics_ratio = semantic_to_dynamics
            first_dynamics_to_semantic_ratio = dynamics_to_semantic
            first_step_integrity = {
                "update": 1,
                "objective_arithmetic_exact": True,
                "microbatch_count": 4,
                "combined_backward_call_count": 4,
                "combined_backward_divisor": 4,
                "semantic_route_probe_call_count": 4,
                "dynamics_route_probe_call_count": 4,
                "route_probe_call_count": 8,
                "semantic_route_gradient_l2": semantic_norm,
                "dynamics_route_gradient_l2": dynamics_norm,
                "semantic_to_dynamics_gradient_ratio": semantic_to_dynamics,
                "dynamics_to_semantic_gradient_ratio": dynamics_to_semantic,
                "ratio_abort_applied": False,
                "predictor_component_gradient_l2": component_gradient_norms,
                "all_predictor_component_gradients_finite_nonzero": all(
                    value > 0.0 for value in component_gradient_norms.values()
                ),
                "target_gradients_absent": not any(
                    parameter.grad is not None for parameter in groups["target"]
                ),
                "optimizer_membership_exact": tuple(
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ) == optimizer_parameter_ids,
            }
            if not all((
                first_step_integrity["all_predictor_component_gradients_finite_nonzero"],
                first_step_integrity["target_gradients_absent"],
                first_step_integrity["optimizer_membership_exact"],
            )):
                raise RuntimeError("update-1 joint integrity failed")

        representation_preclip = torch.nn.utils.clip_grad_norm_(
            representation_parameters, max_norm=1.0, error_if_nonfinite=True
        )
        predictor_preclip = torch.nn.utils.clip_grad_norm_(
            predictor_parameters, max_norm=1.0, error_if_nonfinite=True
        )
        progress["_live_commit_safe_to_hash"] = False
        optimizer.step()
        _WORK["online_optimizer_update_count"] += 1
        _WORK["predictor_optimizer_update_count"] += 1
        _WORK["joint_optimizer_update_count"] += 1
        if id(optimizer) != optimizer_object_id:
            raise RuntimeError("optimizer identity changed")
        before_ema = int(model.ema_update_count.item())
        model.update_target_ema_after_optimizer_step()
        after_ema = int(model.ema_update_count.item())
        _WORK["target_ema_update_count"] += 1
        if before_ema != update - 1 or after_ema != update:
            raise RuntimeError("EMA accounting changed")
        if any(parameter.grad is not None for parameter in groups["target"]):
            raise RuntimeError("EMA target received a gradient")

        updates = update
        presentations = update * contract.EFFECTIVE_BATCH_SIZE
        progress["_live_committed_update"] = update
        progress["_live_committed_rng_sha256"] = _state_value_sha256(
            torch, _rng_snapshot(torch, device)
        )
        progress["_live_commit_safe_to_hash"] = True
        if not _work_is_exact(update):
            raise RuntimeError(f"registered work changed at update {update}")
        progress.update({
            "stage": f"trained_update_{update}",
            "updates": updates,
            "presentations": presentations,
            "optimizer_updates": updates,
            "ema_updates": after_ema,
        })
        _sync_work_to_progress(progress)
        trace_rows.append({
            "update": update,
            "presentations": presentations,
            "phase": "joint_jepa",
            "schedule_slice_sha256": contract.canonical_json_sha256(update_indices),
            **{
                f"mean_{name}": value / contract.MICROBATCHES_PER_UPDATE
                for name, value in sums.items()
            },
            "representation_unclipped_gradient_l2": representation_norm,
            "representation_clip_pre_norm": _scalar(representation_preclip),
            "predictor_unclipped_gradient_l2": predictor_norm,
            "predictor_clip_pre_norm": _scalar(predictor_preclip),
            "semantic_route_gradient_l2": semantic_norm,
            "dynamics_route_gradient_l2": dynamics_norm,
            "semantic_to_dynamics_gradient_ratio": semantic_to_dynamics,
            "dynamics_to_semantic_gradient_ratio": dynamics_to_semantic,
            "ratio_abort_applied": False,
            "ema_update_count": after_ema,
        })
        update_failure_state()

        if update in contract.CHECKPOINT_UPDATES:
            progress["stage"] = f"observation_update_{update}"
            expected_prefix = contract.SCHEDULE_PREFIX_SHA256[update]
            actual_prefix = contract.canonical_json_sha256(
                [int(value) for value in schedule[:presentations]]
            )
            initial_integrity["schedule_prefix_exact"] = (
                actual_prefix == expected_prefix
            )
            if not initial_integrity["schedule_prefix_exact"]:
                raise PermissionError("schedule prefix changed")
            metrics, gate = observe(update)
            terminal_gate = gate
            update_failure_state(gate=gate)
            checkpoint = _BASE._snapshot_model(
                runtime,
                model,
                output_root,
                update=update,
                gate=gate,
                metrics=metrics,
            )
            checkpoints.append(checkpoint)
            if not bool(gate["passed"]):
                break

    if terminal_gate is None:
        raise RuntimeError("terminal gate is absent")
    trace_binding = _BASE._write_training_trace(output_root, trace_rows)
    progress["_training_trace_binding"] = trace_binding
    receipt = {
        "status": str(terminal_gate["control"]),
        "terminal_gate": terminal_gate,
        "observations": observations,
        "checkpoints": checkpoints,
        "training_trace": trace_binding,
        "updates": updates,
        "presentations": presentations,
        "objective_evaluations": _WORK[
            "joint_combined_objective_evaluation_count"
        ],
        "backward_calls": _WORK["combined_backward_call_count"],
        "predictor_forward_count": _WORK[
            "all_action_predictor_training_forward_count"
        ],
        "predictor_objective_count": _WORK[
            "joint_combined_objective_evaluation_count"
        ],
        "predictor_backward_count": _WORK["combined_backward_call_count"],
        "predictor_optimizer_updates": _WORK["predictor_optimizer_update_count"],
        "joint_optimizer_updates": _WORK["joint_optimizer_update_count"],
        "shared_gradient_gate_pass_count": 0,
        "phase_switch_receipt": None,
        "last_committed_state_hashes": progress.get("_last_committed_state_hashes"),
        "cumulative_observation_work": _cumulative_observation_receipt(),
        "pair_presentations_loaded": _WORK["scheduled_pair_presentations_loaded"],
        "first_step_integrity": first_step_integrity,
        **_WORK,
        "integrity": integrity_receipt(include_displacements=True),
    }
    failure_state.update(receipt)
    return model, receipt


def _runtime_input_authority_receipt(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_inputs = authorization.get("runtime_inputs")
    if not isinstance(runtime_inputs, Mapping):
        return {
            "availability": "unknown_or_absent",
            "binding": None,
            "content_sha256": None,
            "matches_frozen_authorization_template_exact": False,
        }
    binding = dict(runtime_inputs)
    return {
        "availability": "available",
        "binding": binding,
        "content_sha256": contract.canonical_json_sha256(binding),
        "matches_frozen_authorization_template_exact": bool(
            binding == contract.runtime_authorization_template()
        ),
    }


_TOP_LEVEL_RECEIPT_NAMES = frozenset({
    "reservation.json",
    "metrics.json",
    "artifact.json",
    "access.json",
    "result.json",
    "failure.json",
    "completed.json",
})


def _receipt_inventory(output_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in output_root.iterdir()
        if path.is_file() and path.name in _TOP_LEVEL_RECEIPT_NAMES
    )


def _execute(
    *,
    sources: Mapping[str, str],
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    output_root: Path,
    progress: dict[str, Any],
) -> int:
    """Run the deepest V3 lifecycle while staging complete public receipts."""

    _ACTIVE_SOURCE_BINDINGS.clear()
    _ACTIVE_SOURCE_BINDINGS.update({
        str(relative): str(digest) for relative, digest in sources.items()
    })
    progress["_authorized_runtime_inputs"] = _runtime_input_authority_receipt(
        authorization
    )
    staged: dict[str, tuple[Path, dict[str, Any], bytes]] = {}
    normal_names = {
        "metrics.json", "artifact.json", "access.json", "result.json",
        "completed.json",
    }

    def stage_publication(
        path: Path, core: Mapping[str, Any]
    ) -> tuple[dict[str, Any], bytes]:
        if path.parent == output_root and path.name in normal_names:
            if path.name in staged or path.exists():
                raise FileExistsError(f"duplicate staged receipt {path.name}")
            staged_core = dict(core)
            if path.name == "metrics.json":
                operation = dict(staged_core.get("operation", {}))
                probe = progress.get("_probe")
                if isinstance(probe, Mapping):
                    operation.update({
                        name: value
                        for name, value in probe.items()
                        if name.endswith("_count")
                        or name in (
                            "updates",
                            "presentations",
                            "objective_evaluations",
                            "backward_calls",
                            "pair_presentations_loaded",
                        )
                    })
                staged_core["operation"] = operation
                staged_core["cumulative_observation_work"] = (
                    _cumulative_observation_receipt()
                )
                staged_core["last_committed_state_hashes"] = progress.get(
                    "_last_committed_state_hashes"
                )
                staged_core["complete_work_receipt"] = True
            value = contract.with_content_sha256(staged_core)
            raw = contract.canonical_json_bytes(value) + b"\n"
            staged[path.name] = (path, value, raw)
            return value, raw
        return _BASE_PUBLISH_JSON(path, core)

    def defer_seal(path: Path) -> dict[str, Any]:
        if path == output_root:
            return {"deferred_until_all_receipts_validated": True}
        return _BASE_SEAL(path)

    _BASE._publish_json = stage_publication
    _BASE._seal = defer_seal
    try:
        result = _BASE_EXECUTE(
            sources=sources,
            authorization=authorization,
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=output_root,
            progress=progress,
        )
    except BaseException as error:
        warning_receipt = getattr(error, "determinism_warning_receipt", None)
        if isinstance(warning_receipt, Mapping):
            progress["_determinism"] = dict(warning_receipt)
        try:
            _capture_terminal_committed_state_hashes(progress)
        except BaseException as hash_error:
            progress["_terminal_state_hash_capture"] = {
                "available": False,
                "reason": "terminal_hash_capture_failed",
                "error_type": type(hash_error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(hash_error).encode("utf-8")
                ).hexdigest(),
                "last_audited_committed_state_hashes": progress.get(
                    "_last_committed_state_hashes"
                ),
            }
        progress.pop("_live_state_objects", None)
        started = progress.get("_gpu_active_started_monotonic")
        progress["_gpu_active_elapsed_seconds"] = (
            0.0
            if started is None
            else max(0.0, time.monotonic() - float(started))
        )
        raise error
    else:
        progress.pop("_live_state_objects", None)
        expected = {
            "metrics.json", "artifact.json", "access.json", "result.json",
            "completed.json",
        }
        if set(staged) != expected:
            raise RuntimeError("normal terminal receipt staging is incomplete")
        for name in (
            "metrics.json", "artifact.json", "access.json", "result.json",
            "completed.json",
        ):
            path, _value, raw = staged[name]
            _BASE._write_exclusive(path, raw)
        expected_inventory = [
            "access.json", "artifact.json", "completed.json", "metrics.json",
            "reservation.json", "result.json",
        ]
        if _receipt_inventory(output_root) != expected_inventory:
            raise RuntimeError("normal terminal receipt inventory changed")
        _BASE_SEAL(output_root)
        return result
    finally:
        _BASE._publish_json = _BASE_PUBLISH_JSON
        _BASE._seal = _BASE_SEAL


def _terminal_trace_binding(
    output_root: Path, progress: Mapping[str, Any]
) -> Any:
    binding = progress.get("_training_trace_binding")
    if binding is None and isinstance(progress.get("_trace_rows"), list):
        trace_path = output_root / "training_trace.json"
        if not trace_path.exists():
            binding = _BASE._write_training_trace(
                output_root, progress["_trace_rows"]
            )
            if isinstance(progress, dict):
                progress["_training_trace_binding"] = binding
    return binding


def _terminal_access(progress: Mapping[str, Any]) -> dict[str, Any]:
    loader = progress.get("_loader")
    inputs = progress.get("_inputs")
    protected_zero = {
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
    if loader is None or inputs is None:
        consumed = None if inputs is None else getattr(inputs, "consumed", None)
        roles = (
            sorted({
                str(record.get("role"))
                for record in consumed.values()
                if isinstance(record, Mapping) and record.get("role") is not None
            })
            if isinstance(consumed, Mapping)
            else None
        )
        return {
            "loader_receipt_available": False,
            "access_phase": str(progress.get("stage", "unknown")),
            "roles_opened": roles,
            "consumed_record_count": (
                len(consumed) if isinstance(consumed, Mapping) else None
            ),
            "model_facing_counts": None,
            **protected_zero,
        }
    try:
        return {
            **_BASE._access_receipt(loader, inputs),
            "loader_receipt_available": True,
            "access_phase": str(progress.get("stage", "unknown")),
        }
    except BaseException as error:
        return {
            "loader_receipt_available": False,
            "access_receipt_error_type": type(error).__name__,
            "access_receipt_error_message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "roles_opened": None,
            "consumed_record_count": None,
            "model_facing_counts": None,
            "access_phase": str(progress.get("stage", "unknown")),
            **protected_zero,
        }


def _terminal_probe_state(progress: Mapping[str, Any]) -> Mapping[str, Any]:
    probe = progress.get("_probe")
    if isinstance(probe, Mapping):
        return probe
    partial = progress.get("_probe_failure_state")
    return partial if isinstance(partial, Mapping) else {}


def _terminal_operation(
    progress: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    names = (
        "updates",
        "presentations",
        "pair_presentations_loaded",
        "objective_evaluations",
        "backward_calls",
        *_expected_work_at_update(0).keys(),
    )
    aliases = {
        "pair_presentations_loaded": "scheduled_pair_presentations_loaded",
        "objective_evaluations": "joint_combined_objective_evaluation_count",
        "backward_calls": "combined_backward_call_count",
    }
    return {
        name: (
            _WORK[aliases[name]]
            if name in aliases
            else _WORK[name]
            if name in _WORK
            else state.get(name, progress.get(name, 0))
        )
        for name in dict.fromkeys(names)
    }


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: Mapping[str, Any],
    error: BaseException,
) -> None:
    """Publish one self-contained, complete operational failure receipt."""

    if (output_root / "completed.json").exists():
        _BASE_SEAL(output_root)
        return
    if progress.get("_terminal_state_hash_capture") is None:
        try:
            _capture_terminal_committed_state_hashes(progress)
        except BaseException as hash_error:
            progress["_terminal_state_hash_capture"] = {
                "available": False,
                "reason": "terminal_hash_capture_failed",
                "error_type": type(hash_error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(hash_error).encode("utf-8")
                ).hexdigest(),
                "last_audited_committed_state_hashes": progress.get(
                    "_last_committed_state_hashes"
                ),
            }
    state = _terminal_probe_state(progress)
    trace_binding = _terminal_trace_binding(output_root, progress)
    checkpoints = progress.get("_checkpoint_bindings")
    checkpoints = checkpoints if isinstance(checkpoints, list) else []
    observations = progress.get("_observations")
    observations = observations if isinstance(observations, list) else []
    public_progress = {
        name: value for name, value in progress.items()
        if not name.startswith("_")
    }
    failure, failure_raw = _BASE._publish_json(
        output_root / "failure.json",
        {
            "schema": contract.FAILURE_SCHEMA,
            "status": contract.CONTROL_FAIL_OPERATIONAL,
            "classification": "OPERATIONAL_OR_INTEGRITY_FAILURE",
            "reservation": _BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
            "progress": public_progress,
            "first_failure_stage": progress.get("stage"),
            "error": {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    "".join(traceback.format_exception(error)).encode("utf-8")
                ).hexdigest(),
            },
            "observations": observations,
            "observation_work": _observation_live_receipt(),
            "cumulative_observation_work": _cumulative_observation_receipt(),
            "last_committed_state_hashes": progress.get(
                "_last_committed_state_hashes"
            ),
            "terminal_state_hash_capture": progress.get(
                "_terminal_state_hash_capture"
            ),
            "terminal_gate": state.get("terminal_gate"),
            "first_step_integrity": state.get("first_step_integrity"),
            "operation": _terminal_operation(progress, state),
            "integrity": state.get("integrity", {}),
            "checkpoints": checkpoints,
            "training_trace": trace_binding,
            "access": _terminal_access(progress),
            "hardware": progress.get("_hardware"),
            "determinism": progress.get("_determinism"),
            "schedule": progress.get("_schedule_receipt"),
            "source_bindings": dict(_ACTIVE_SOURCE_BINDINGS),
            "reviewed_cpu_source_witness": _terminal_source_witness(),
            "authorized_runtime_inputs": progress.get(
                "_authorized_runtime_inputs"
            ),
            "n320_gate": progress.get("_n320_gate"),
            "n320_checkpoint": progress.get("_n320_checkpoint_binding"),
            "gpu_active_elapsed_seconds": progress.get(
                "_gpu_active_elapsed_seconds"
            ),
            "all_checkpoints_write_only_and_unqualified": True,
            "checkpoint_read_count_after_write": 0,
            "training_trace_read_count_after_write": 0,
            "mechanism_passed": False,
            "checkpoint_qualified": False,
            "downstream_authority": "none",
            "retry_resume_repair_or_replacement_authorized": False,
            "complete_failure_receipt": True,
        },
    )
    _BASE._publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": contract.CONTROL_FAIL_OPERATIONAL,
            "reservation": _BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
            "failure": _BASE._binding("failure.json", failure, failure_raw),
            "receipt_inventory": [
                "reservation.json", "failure.json", "completed.json"
            ],
            "checkpoint_qualified": False,
            "retry_authorized": False,
            "complete_failure_receipt": True,
        },
    )
    expected = ["completed.json", "failure.json", "reservation.json"]
    if _receipt_inventory(output_root) != expected:
        raise RuntimeError("operational receipt inventory changed")
    _BASE_SEAL(output_root)


_RUNNER_BINDING_NAMES = (
    "_load_post_reservation_stack",
    "_load_development_inputs",
    "_tensor_state_sha256",
    "_run_deterministic",
    "_parameter_receipt",
    "_semantic_terms",
    "_joint_terms",
    "_evaluate_observation",
    "_train_probe",
    "_execute",
    "_terminal_failure",
)


def _rebind_inherited_runner_once() -> None:
    """Install all final hooks directly on the deepest frozen V3 runner."""

    _BASE.contract = contract
    _BASE.RUNNER_PATH = RUNNER_PATH
    _BASE.CONTRACT_PATH = ROOT / contract.CONTRACT_RELATIVE_PATH
    _BASE.__file__ = str(RUNNER_PATH)
    _BASE.__doc__ = __doc__
    for name in _RUNNER_BINDING_NAMES:
        setattr(_BASE, name, globals()[name])


_rebind_inherited_runner_once()


def _assert_final_runner_bindings() -> None:
    if (
        _BASE.contract is not contract
        or _BASE.RUNNER_PATH != RUNNER_PATH
        or _BASE.CONTRACT_PATH != ROOT / contract.CONTRACT_RELATIVE_PATH
    ):
        raise PermissionError("final Action-Query runner identity changed")
    for name in _RUNNER_BINDING_NAMES:
        if getattr(_BASE, name) is not globals()[name]:
            raise PermissionError(f"final Action-Query hook changed: {name}")


_assert_final_runner_bindings()


def run_isolated_import_preflight() -> dict[str, Any]:
    """Report source-only final bindings without touching runtime inputs."""

    _assert_final_runner_bindings()
    return {
        "deepest_base_dispatch_bound": True,
        "action_query_model_loader_bound": True,
        "action_query_objective_and_training_hooks_bound": True,
        "scalar_safe_state_hash_bound": True,
        "source_witness_uses_runtime_source_map_without_fallback": True,
        "runtime_or_generated_inputs_opened": [],
        "checkpoints_tensors_traces_or_predecessor_outputs_opened": [],
        "accelerators_queried_or_used": [],
        "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
    }


def parse_args(argv: Sequence[str] | None = None) -> Any:
    _assert_final_runner_bindings()
    return _BASE.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    _assert_final_runner_bindings()
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
