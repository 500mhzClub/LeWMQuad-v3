#!/usr/bin/env python3
"""Staged controller for the V18 causal delay-line memory JEPA V1."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
from typing import Any, Mapping, Sequence

from lewm.benchmarks import go2_v18_delay_line_memory_metrics_v1 as metrics
from scripts import execute_go2_rgb_memory_role_factorized_joint_jepa_v1 as v5
from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
    as physical_executor,
)
from scripts import launch_go2_rgb_memory_role_factorized_joint_jepa_v1 as v5_launcher


SCHEMA_PREFIX = (
    "lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4"
)
PREREGISTRATION_PATH = (
    "docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4_preregistration_2026-07-31.md"
)
PREREGISTRATION_COMMIT = "1d81275ab86e98bc909f7716afbc42d963e99131"
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4_source_manifest_2026-07-31.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4_source_review_2026-07-31.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4_clean_export_certification_"
    "2026-07-31.json"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4_execution_authorization_"
    "2026-07-31.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4/attempt_v1"
)
CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-v18-spatial-token-delay-line-physical-comparison-alias-state-integrity-replacement-v4-source"
)
MODEL_CLASS_NAME = "V18SpatialTokenDelayLineCausalConvolutionJointJepaV1"
MODEL_MODULE_NAME = (
    "lewm.models.v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1"
)
EVALUATION_MODULE_NAME = (
    "scripts.evaluate_go2_v18_spatial_token_delay_line_causal_convolution_"
    "joint_jepa_v1"
)

MAXIMUM_UPDATES = 1_000
STAGE_A_UPDATES = 500
MEMORY_PRESENTATIONS_PER_UPDATE = 16
PHYSICAL_PRESENTATIONS_PER_UPDATE = 8
PRESENTATIONS_PER_UPDATE = 24
MAXIMUM_MEMORY_PRESENTATIONS = 16_000
MAXIMUM_PHYSICAL_PRESENTATIONS = 8_000
MAXIMUM_PRESENTATIONS = 24_000
OBSERVATION_UPDATES = (0, 100, 250, 500, 750, 1_000)
SNAPSHOT_UPDATES = (250, 500, 750, 1_000)
PHYSICAL_OBSERVATION_ALIAS = {
    update: 400 for update in OBSERVATION_UPDATES
}
PLACE_OBSERVATION_ALIAS = {0: 0, 100: 100, 250: 400, 500: 400, 750: 400, 1_000: 400}

# Exact inherited runtime API consumed by the V13 composer.
RUNTIME_INPUT_BINDING_NAMES = v5.RUNTIME_INPUT_BINDING_NAMES
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = v5.CHECKPOINT_SCHEDULE_PREFIX_SHA256
REGISTERED_FAMILIES = v5.REGISTERED_FAMILIES
SCOPES = v5.SCOPES
V12_GATE_CHECK_NAMES = v5.V12_GATE_CHECK_NAMES
CONTROL_NAMES = v5.CONTROL_NAMES
MATCHED_UPDATE400_THRESHOLDS = v5.MATCHED_UPDATE400_THRESHOLDS
EXPECTED_RUNTIME_FINGERPRINT = v5.EXPECTED_RUNTIME_FINGERPRINT
flatten_physical_metrics_v13 = v5.flatten_physical_metrics_v13
registered_wrong_rgb_mapping_v13 = v5.registered_wrong_rgb_mapping_v13
_canonical_json_bytes = v5._canonical_json_bytes
_write_immutable_json_v13 = v5._write_immutable_json_v13


def _content_bound(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    value.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    return value


def validate_content_bound_v1(value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("content_sha256")) is not str:
        raise TypeError("delay-line content-bound value must be an exact object")
    core = dict(value)
    observed = core.pop("content_sha256")
    if observed != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise RuntimeError("delay-line content binding changed")
    return dict(value)


validate_content_bound_v13 = validate_content_bound_v1


def _binding(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"delay-line {name} binding is absent")
    result = dict(value)
    if (
        not {"path", "file_sha256", "byte_count"}.issubset(result)
        or type(result["path"]) is not str
        or type(result["file_sha256"]) is not str
        or len(result["file_sha256"]) != 64
        or type(result["byte_count"]) is not int
        or result["byte_count"] <= 0
    ):
        raise TypeError(f"delay-line {name} binding changed")
    return result


def validate_future_execution_prerequisites_v1(authority: Any) -> dict[str, Any]:
    value = validate_content_bound_v1(authority)
    required = {
        "schema": f"{SCHEMA_PREFIX}_execution_authority_v1",
        "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_STAGED_ONE_SHOT",
        "scientific_payload_authorized": True,
        "one_shot": True,
        "maximum_updates": MAXIMUM_UPDATES,
        "stage_a_updates": STAGE_A_UPDATES,
        "maximum_memory_presentations": MAXIMUM_MEMORY_PRESENTATIONS,
        "maximum_physical_presentations": MAXIMUM_PHYSICAL_PRESENTATIONS,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "retry_authorized": False,
        "scientific_resume_authorized": False,
        "infrastructure_recovery_authorized": True,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "runtime_data_root": "/home/andrewknowles/Workspace/LeWMQuad-v3",
    }
    if any(value.get(name) != expected for name, expected in required.items()):
        raise PermissionError("delay-line authority identity or cap changed")
    if value.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise PermissionError("delay-line authority is not bound to preregistration")
    if value.get("selectors") != {
        "executor_module": __name__,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "evaluation_module": EVALUATION_MODULE_NAME,
    }:
        raise PermissionError("delay-line runtime selectors changed")
    runtime_inputs = value.get("runtime_inputs")
    if type(runtime_inputs) is not dict or set(runtime_inputs) != set(
        RUNTIME_INPUT_BINDING_NAMES
    ):
        raise PermissionError("delay-line runtime input registry changed")
    for name in RUNTIME_INPUT_BINDING_NAMES:
        _binding(runtime_inputs[name], name=name)
    _binding(value.get("clean_export_certification"), name="certification")
    commit = value.get("pinned_source_and_review_commit")
    if type(commit) is not str or len(commit) != 40:
        raise PermissionError("delay-line source-and-review commit is malformed")
    return value


validate_future_execution_prerequisites_v13 = validate_future_execution_prerequisites_v1


def validate_bound_sources_v1(_repository_root: Path) -> dict[str, Any]:
    raise PermissionError("delay-line source validation requires the frozen launcher")


validate_bound_sources_v13 = validate_bound_sources_v1


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"{name} must be a regular non-symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{name} is not strict JSON") from error
    if type(value) is not dict or raw != _canonical_json_bytes(value) + b"\n":
        raise PermissionError(f"{name} must be canonical JSON")
    return value


def _reservation_core(authority: Mapping[str, Any], created_utc: str) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_attempt_reservation_v1",
        "status": "RESERVED_STAGED_ONE_SHOT",
        "created_utc": created_utc,
        "authority_sha256": hashlib.sha256(_canonical_json_bytes(authority)).hexdigest(),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "infrastructure_recovery_authorized": True,
        "attempt_consumed": True,
    }


def validate_attempt_reservation_v1(value: Any) -> dict[str, Any]:
    result = validate_content_bound_v1(value)
    if (
        result.get("schema") != f"{SCHEMA_PREFIX}_attempt_reservation_v1"
        or result.get("status") != "RESERVED_STAGED_ONE_SHOT"
        or result.get("output_root") != OUTPUT_ROOT_RELATIVE_PATH
        or result.get("maximum_updates") != MAXIMUM_UPDATES
        or result.get("maximum_presentations") != MAXIMUM_PRESENTATIONS
        or result.get("attempt_consumed") is not True
    ):
        raise PermissionError("delay-line attempt reservation changed")
    return result


def _latest_snapshot(output_root: Path) -> tuple[int, Path, dict[str, Any]] | None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for update in SNAPSHOT_UPDATES:
        metadata_path = output_root / f"snapshots/update_{update}.binding.json"
        if not metadata_path.exists() and not metadata_path.is_symlink():
            continue
        metadata = validate_content_bound_v1(
            _read_json(metadata_path, name=f"update-{update} snapshot binding")
        )
        state_binding = _binding(metadata.get("state"), name="snapshot state")
        state_path = output_root / state_binding["path"]
        if (
            metadata.get("update") != update
            or state_binding["path"] != f"snapshots/update_{update}.pt"
            or state_path.is_symlink()
            or not state_path.is_file()
        ):
            raise PermissionError("delay-line snapshot identity changed")
        info = os.stat(state_path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size != state_binding["byte_count"]:
            raise PermissionError("delay-line snapshot file changed")
        raw = state_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != state_binding["file_sha256"]:
            raise PermissionError("delay-line snapshot hash changed")
        candidates.append((update, state_path, metadata))
    return max(candidates, default=None, key=lambda value: value[0])


def reserve_or_recover_attempt_v1(
    repository_root: Path,
    authority: Mapping[str, Any],
    *,
    created_utc: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    validated = validate_future_execution_prerequisites_v1(dict(authority))
    root = Path(repository_root).resolve(strict=True)
    output = root / OUTPUT_ROOT_RELATIVE_PATH
    reservation_path = output / "reservation.json"
    if not output.exists() and not output.is_symlink():
        os.mkdir(output, 0o700)
        reservation = _write_immutable_json_v13(
            reservation_path, _reservation_core(validated, created_utc)
        )
        return validate_attempt_reservation_v1(reservation), None
    if output.is_symlink() or not output.is_dir():
        raise PermissionError("delay-line output root changed type")
    if any((output / name).exists() for name in ("success.json", "failure.json")):
        raise PermissionError("delay-line one-shot attempt already terminated")
    reservation = validate_attempt_reservation_v1(
        _read_json(reservation_path, name="attempt reservation")
    )
    if reservation["authority_sha256"] != hashlib.sha256(
        _canonical_json_bytes(validated)
    ).hexdigest():
        raise PermissionError("delay-line recovery authority changed")
    recovery_path = output / "recovery.json"
    if recovery_path.exists() or recovery_path.is_symlink():
        raise PermissionError("delay-line infrastructure recovery already consumed")
    snapshot = _latest_snapshot(output)
    if snapshot is None:
        raise PermissionError("delay-line recovery has no complete exact snapshot")
    update, state_path, metadata = snapshot
    recovery = _write_immutable_json_v13(
        recovery_path,
        {
            "schema": f"{SCHEMA_PREFIX}_infrastructure_recovery_v1",
            "status": "AUTHORIZED_LATEST_EXACT_SNAPSHOT_RECOVERY",
            "created_utc": created_utc,
            "update": update,
            "snapshot_metadata_content_sha256": metadata["content_sha256"],
            "scientific_retry": False,
            "cap_changed": False,
        },
    )
    return reservation, {
        "receipt": recovery,
        "update": update,
        "state_path": str(state_path),
        "metadata": metadata,
    }


def terminalize_failure_v1(
    output_root: Path,
    reservation: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
    created_utc: str,
) -> dict[str, Any]:
    validate_attempt_reservation_v1(dict(reservation))
    path = Path(output_root) / "failure.json"
    if path.exists() or path.is_symlink():
        return _read_json(path, name="terminal failure")
    return _write_immutable_json_v13(
        path,
        {
            "schema": f"{SCHEMA_PREFIX}_terminal_infrastructure_failure_v1",
            "status": "FAIL_TERMINAL_INFRASTRUCTURE",
            "created_utc": created_utc,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempt_consumed": True,
            "retry_authorized": False,
            "scientific_resume_authorized": False,
            "infrastructure_recovery_available": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        },
    )


terminalize_failure_v13 = terminalize_failure_v1


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"{name} must be a dataclass or mapping")
    return result


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    return value


def _finite_values(value: Any) -> bool:
    if is_dataclass(value):
        return _finite_values(asdict(value))
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return not isinstance(value, float) or math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_values(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_values(item) for item in value)
    return True


def validate_update_integrity_v1(runtime: Any, model: Any, result: Any, *, update: int) -> dict[str, Any]:
    accounting = _mapping(result.accounting, name="training accounting")
    expected = {
        "updates": update,
        "presentations": update * PRESENTATIONS_PER_UPDATE,
        "physical_presentations": update * PHYSICAL_PRESENTATIONS_PER_UPDATE,
        "memory_presentations": update * MEMORY_PRESENTATIONS_PER_UPDATE,
        "physical_microbatch_graphs": update * 2,
        "memory_microbatch_graphs": update * 8,
        "optimizer_steps": update,
        "ema_steps": update,
    }
    checks = {
        "accounting_exact": all(accounting.get(name) == value for name, value in expected.items()),
        "one_optimizer_step": result.optimizer_steps_this_update == 1,
        "one_ema_step": result.ema_steps_this_update == 1,
        "model_ema_matches_update": int(model.ema_update_count.item()) == update,
        "target_gradients_zero": result.target_gradient_tensor_count == 0,
        "losses_finite": _finite_values(result.mean_losses),
        "diagnostics_finite": _finite_values(result.memory_diagnostics),
        "gradient_routes_present": bool(result.gradient_routes),
    }
    target_parameters = [
        parameter for module in model.target_modules() for parameter in module.parameters()
    ]
    checks["target_modules_frozen_and_gradless"] = all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in target_parameters
    )
    checks["model_state_finite"] = all(
        not parameter.is_floating_point()
        or bool(runtime.torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )
    return {
        "schema": f"{SCHEMA_PREFIX}_update_integrity_v1",
        "update": update,
        "accounting": accounting,
        "mean_losses": _jsonable(_mapping(result.mean_losses, name="mean losses")),
        "gradient_routes": _jsonable(
            _mapping(result.gradient_routes, name="gradient routes")
        ),
        "memory_diagnostics": _jsonable(
            _mapping(result.memory_diagnostics, name="memory diagnostics")
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_update0_gate_v1(
    observation: metrics.ObservationMetrics,
    receipt: Mapping[str, Any],
) -> metrics.GateDecision:
    if observation.update != 0:
        raise ValueError("delay-line update-zero gate requires update 0")
    temporal_integrity = receipt.get("temporal", {}).get("integrity", {})
    temporal = observation.temporal
    numeric_tolerance = 1.0e-5
    checks = {
        "integrity_pass": observation.safeguards.integrity_pass,
        "target_finite_nonzero": (
            temporal_integrity.get("checks", {}).get("target_state_finite") is True
            and temporal_integrity.get("checks", {}).get(
                "target_state_nonzero_scale"
            )
            is True
        ),
        "online_finite_nonzero": (
            temporal_integrity.get("checks", {}).get("online_state_finite") is True
            and temporal_integrity.get("checks", {}).get(
                "online_state_nonzero_scale"
            )
            is True
        ),
        "memory_finite_nonzero": (
            temporal_integrity.get("checks", {}).get("memory_state_finite") is True
            and temporal_integrity.get("checks", {}).get(
                "memory_state_nonzero_scale"
            )
            is True
        ),
        "prediction_level_persistence_identity": (
            temporal_integrity.get("checks", {}).get(
                "update_zero_controls_equal_persistence"
            )
            is True
        ),
        "score_is_one_within_tolerance": all(
            abs(value - 1.0) <= numeric_tolerance for value in temporal.score.macro
        ),
        "persistence_lift_is_zero_within_tolerance": all(
            abs(value) <= numeric_tolerance
            for value in temporal.persistence_lift.macro
        ),
        "action_lift_is_zero_within_tolerance": all(
            abs(value) <= numeric_tolerance for value in temporal.action_lift.macro
        ),
        "history_lift_is_zero_within_tolerance": all(
            abs(value) <= numeric_tolerance for value in temporal.history_lift.macro
        ),
        "place_at_least_2x_chance": (
            observation.substrate.place_chance_multiple >= 2.0
        ),
        "place_above_chance_in_six_scenes": (
            observation.substrate.place_scene_count_above_chance >= 6
        ),
        "target_place_rank_at_least_2": (
            observation.substrate.target_place_rank >= 2.0
        ),
    }
    passed = all(checks.values())
    return metrics.GateDecision(
        update=0,
        status=(
            "PASS_UPDATE0_INTEGRITY"
            if passed
            else "FAIL_UPDATE0_INTEGRITY_TERMINAL"
        ),
        action="CONTINUE" if passed else "STOP_TERMINAL",
        passed=passed,
        checks=checks,
        failed_checks=tuple(name for name, value in checks.items() if not value),
        observed={
            "maximum_prediction_persistence_delta": temporal_integrity.get(
                "update_zero_max_control_prediction_delta"
            ),
            "place_chance_multiple": observation.substrate.place_chance_multiple,
            "place_scene_count_above_chance": (
                observation.substrate.place_scene_count_above_chance
            ),
            "target_place_rank": observation.substrate.target_place_rank,
            "absolute_noncollapse_enforced": temporal_integrity.get(
                "absolute_noncollapse_enforced"
            ),
            "target_noncollapsed_diagnostic": observation.safeguards.target_noncollapsed,
            "online_noncollapsed_diagnostic": observation.safeguards.online_noncollapsed,
            "memory_noncollapsed_diagnostic": observation.memory_state.noncollapsed,
        },
    )


def _immediate_integrity_failure_decision(
    *,
    update: int,
    status: str,
    checks: Mapping[str, bool],
    observed: Mapping[str, float | int | bool | None],
) -> metrics.GateDecision:
    failed = tuple(name for name, passed in checks.items() if not passed)
    if not failed:
        raise ValueError("immediate integrity failure requires a failed check")
    return metrics.GateDecision(
        update=update,
        status=status,
        action="STOP_TERMINAL",
        passed=False,
        checks=dict(checks),
        failed_checks=failed,
        observed=dict(observed),
    )


def evaluate_observation_integrity_v1(
    observation: metrics.ObservationMetrics,
) -> metrics.GateDecision | None:
    checks = {
        "integrity_pass": observation.safeguards.integrity_pass,
        "gradient_accounting_pass": observation.safeguards.gradient_accounting_pass,
    }
    if observation.update >= 250:
        checks.update(
            {
                "target_noncollapsed": observation.safeguards.target_noncollapsed,
                "online_noncollapsed": observation.safeguards.online_noncollapsed,
                "memory_noncollapsed": observation.memory_state.noncollapsed,
            }
        )
    if all(checks.values()):
        return None
    return _immediate_integrity_failure_decision(
        update=observation.update,
        status=f"STOP_UPDATE{observation.update}_INTEGRITY_OR_COLLAPSE",
        checks=checks,
        observed={
            "memory_participation_rank_ratio": (
                observation.memory_state.participation_rank_ratio
            ),
            "memory_near_zero_fraction": observation.memory_state.near_zero_fraction,
        },
    )


def _physical_batches(runtime: Any, update: int) -> tuple[Mapping[str, Any], ...]:
    start = (update - 1) * PHYSICAL_PRESENTATIONS_PER_UPDATE
    indices = tuple(runtime.schedule[start : start + PHYSICAL_PRESENTATIONS_PER_UPDATE])
    if len(indices) != PHYSICAL_PRESENTATIONS_PER_UPDATE:
        raise RuntimeError("delay-line physical schedule exhausted")
    builder = v5_launcher._BASE_LAUNCHER._build_one_microbatch_v13
    return tuple(
        builder(
            runtime=runtime,
            indices=indices[offset : offset + 4],
            stage=f"delay_line_train_update_{update:04d}_physical_{offset // 4}",
        )
        for offset in (0, 4)
    )


def _place_metrics(role_observation: Mapping[str, Any]) -> dict[str, float | int]:
    evaluation = __import__(v5.EVALUATION_MODULE_NAME, fromlist=["*"])
    return evaluation._place_gate_metrics(role_observation["place"])


def _physical_control_count(controls: Mapping[str, Any]) -> int:
    evaluation = __import__(v5.EVALUATION_MODULE_NAME, fromlist=["*"])
    flattened = evaluation._flatten_controls(controls)
    return sum(value is True for value in flattened.values())


def _physical_observation_v1(
    runtime: Any,
    model: Any,
    *,
    outer_update: int,
    integrity_pass: bool,
) -> Mapping[str, Any]:
    """Release V19's inner alias receipt after one complete V18 observation."""

    alias = PHYSICAL_OBSERVATION_ALIAS[outer_update]
    missing = object()
    before = getattr(runtime, "causal_comparisons_v19", missing)
    if before is not missing and (type(before) is not dict or before):
        raise RuntimeError(
            "V18 physical comparison alias cache is malformed or nonempty before observation"
        )
    result = physical_executor._observation_v13(
        runtime,
        model,
        update=alias,
        integrity_pass=integrity_pass,
    )
    after = getattr(runtime, "causal_comparisons_v19", None)
    if type(after) is not dict or set(after) != {alias}:
        raise RuntimeError(
            "V18 physical comparison alias cache has missing or unexpected keys"
        )
    captured = after[alias]
    if type(captured) is not dict or tuple(captured) != tuple(CONTROL_NAMES):
        raise RuntimeError(
            "V18 physical comparison alias cache control structure changed"
        )
    del after[alias]
    if after:
        raise RuntimeError("V18 physical comparison alias cache cleanup is incomplete")
    return result


def _observation(
    *,
    runtime: Any,
    model: Any,
    h6_runtime: Any,
    role_runtime: Any,
    update: int,
    training_integrity_pass: bool,
    update0_place_rank: float | None,
) -> tuple[dict[str, Any], metrics.ObservationMetrics, float]:
    physical_alias = PHYSICAL_OBSERVATION_ALIAS[update]
    place_alias = PLACE_OBSERVATION_ALIAS[update]
    physical = _physical_observation_v1(
        runtime,
        model,
        outer_update=update,
        integrity_pass=training_integrity_pass,
    )
    roles = role_runtime.evaluate_role_metrics(
        model, update=place_alias, device=runtime.device
    )
    temporal = h6_runtime.evaluate_temporal_metrics(model, update, runtime.device)
    place = _place_metrics(roles)
    baseline_rank = (
        float(place["target_place_key_effective_rank"])
        if update0_place_rank is None
        else update0_place_rank
    )
    current_rank = float(place["target_place_key_effective_rank"])
    rank_retention = current_rank / baseline_rank if baseline_rank > 0.0 else 0.0
    physical_summary = physical["physical"]
    physical_count = int(physical_summary["passed_margin_count"])
    control_count = _physical_control_count(physical["controls"])
    temporal_integrity = dict(temporal.integrity)
    integrity_pass = bool(
        training_integrity_pass
        and physical.get("integrity_pass") is True
        and roles.get("integrity", {}).get("passed") is True
        and temporal_integrity.get("passed") is True
    )
    perception_pass = bool(
        float(place["chance_multiple"]) >= 2.0
        and int(place["scene_count_above_chance"]) >= 6
        and current_rank >= 2.0
        and rank_retention >= 0.80
        and physical_count >= 60
        and control_count == 12
    )
    observation_metrics = metrics.ObservationMetrics(
        update=update,
        temporal=temporal.temporal,
        memory_state=temporal.memory_rank,
        safeguards=metrics.RuntimeSafeguards(
            integrity_pass=integrity_pass,
            perception_safeguards_pass=perception_pass,
            gradient_accounting_pass=training_integrity_pass,
            target_noncollapsed=temporal.target_rank.noncollapsed,
            online_noncollapsed=temporal.online_rank.noncollapsed,
        ),
        substrate=metrics.SubstrateMetrics(
            place_chance_multiple=float(place["chance_multiple"]),
            place_scene_count_above_chance=int(place["scene_count_above_chance"]),
            target_place_rank=current_rank,
            target_place_rank_retention=rank_retention,
            physical_passed_margin_count=physical_count,
            physical_causal_control_pass_count=control_count,
        ),
    )
    energy_means = {
        name: tuple(
            float(value)
            for value in getattr(temporal.energies, name).mean(dim=0).tolist()
        )
        for name in (
            "real",
            "persistence",
            "wrong_action",
            "reset",
            "reverse",
            "shuffle",
        )
    }
    temporal_receipt = {
        "update": temporal.update,
        "temporal": asdict(temporal.temporal),
        "target_rank": asdict(temporal.target_rank),
        "online_rank": asdict(temporal.online_rank),
        "memory_rank": asdict(temporal.memory_rank),
        "energy_means_by_horizon": energy_means,
        "hold_diagnostics": asdict(temporal.hold_diagnostics),
        "access_receipt": dict(temporal.access_receipt),
        "memory_receipt": dict(temporal.memory_receipt),
        "integrity": temporal_integrity,
    }
    receipt = {
        "schema": f"{SCHEMA_PREFIX}_observation_v1",
        "update": update,
        "physical_evaluator_alias_update": physical_alias,
        "place_evaluator_alias_update": place_alias,
        "temporal": temporal_receipt,
        "physical": physical,
        "roles": roles,
        "gate_metrics": asdict(observation_metrics),
        "integrity_pass": integrity_pass,
        "perception_safeguards_pass": perception_pass,
        "state_mutation_count": 0,
        "probability_calibration_opened": False,
        "navigation_executed": False,
        "held_out_or_sealed_opened": False,
    }
    return receipt, observation_metrics, baseline_rank


def _publish_json(publisher: Any, relative: str, core: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = publisher.publish_json(relative, dict(core))
    if type(result) is not dict or set(result) != {"value", "binding"}:
        raise RuntimeError("delay-line publisher JSON result changed")
    return validate_content_bound_v1(result["value"]), dict(result["binding"])


class ExactRecoveryReplayPublisherV1:
    """Allow recovery to reuse only byte-identical write-once artifacts."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.output_root = Path(delegate.output_root)

    def _path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
            raise PermissionError("delay-line recovery artifact path escaped")
        return self.output_root.joinpath(*path.parts)

    def publish_json(
        self, relative: str, core: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        try:
            return self.delegate.publish_json(relative, core)
        except FileExistsError:
            path = self._path(relative)
            observed = validate_content_bound_v1(
                _read_json(path, name=f"replayed {relative}")
            )
            expected = _content_bound(core)
            if observed != expected:
                raise PermissionError(
                    f"delay-line recovery JSON replay changed: {relative}"
                )
            raw = _canonical_json_bytes(observed) + b"\n"
            return {
                "value": observed,
                "binding": {
                    "path": relative,
                    "file_sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw),
                    "content_sha256": observed["content_sha256"],
                },
            }

    def publish_bytes(self, relative: str, raw: bytes) -> Mapping[str, Any]:
        try:
            return self.delegate.publish_bytes(relative, raw)
        except FileExistsError:
            path = self._path(relative)
            if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                raise PermissionError(
                    f"delay-line recovery byte replay changed: {relative}"
                )
            return {
                "path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }


def _content_bound_json_artifact_binding_v1(
    relative: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    validated = validate_content_bound_v1(dict(value))
    raw = _canonical_json_bytes(validated) + b"\n"
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "content_sha256": validated["content_sha256"],
    }


def _nonnegative_integer_mapping_v1(
    value: Any, *, expected_keys: set[str], name: str
) -> dict[str, int]:
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or any(type(item) is not int or item < 0 for item in value.values())
    ):
        raise PermissionError(f"delay-line recovery {name} changed")
    return {str(key): int(item) for key, item in value.items()}


def _capture_exact_access_state_v1(
    runtime: Any, h6_runtime: Any, role_runtime: Any
) -> dict[str, Any]:
    h6_loader = h6_runtime._require_loader()
    local_loader = role_runtime._local_loader
    return {
        "schema": f"{SCHEMA_PREFIX}_exact_access_state_v1",
        "physical_consumed": _jsonable(runtime.raw_inputs.consumed),
        "h6_access": dict(h6_loader._access),
        "role_local_consumed": _jsonable(local_loader._consumed),
        "role_local_access": {
            "tensor_requests": local_loader._tensor_requests,
            "open_attempts": local_loader._open_attempts,
            "open_successes": local_loader._open_successes,
            "decode_successes": local_loader._decode_successes,
            "byte_count": local_loader._byte_count,
        },
        "role_place_access": dict(role_runtime._place_reference_counts),
        "role_place_loader_calls": role_runtime._place_loader_calls,
        "role_place_loaded_row_keys": [
            [role, index]
            for role, index in sorted(role_runtime._place_loaded_row_keys)
        ],
    }


def _restore_exact_access_state_v1(
    runtime: Any,
    h6_runtime: Any,
    role_runtime: Any,
    value: Any,
) -> None:
    expected_keys = {
        "schema",
        "physical_consumed",
        "h6_access",
        "role_local_consumed",
        "role_local_access",
        "role_place_access",
        "role_place_loader_calls",
        "role_place_loaded_row_keys",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema") != f"{SCHEMA_PREFIX}_exact_access_state_v1"
    ):
        raise PermissionError("delay-line recovery access-state schema changed")

    physical = value["physical_consumed"]
    if type(physical) is not dict:
        raise PermissionError("delay-line recovery physical ledger changed")
    physical_copy: dict[str, dict[str, Any]] = {}
    record_keys = {
        "path",
        "file_sha256",
        "byte_count",
        "kind",
        "roles",
        "arms",
        "stages",
    }
    for path, record in physical.items():
        if (
            type(path) is not str
            or type(record) is not dict
            or set(record) != record_keys
            or record.get("path") != path
            or type(record.get("file_sha256")) is not str
            or len(record["file_sha256"]) != 64
            or type(record.get("byte_count")) is not int
            or record["byte_count"] <= 0
            or type(record.get("kind")) is not str
            or any(
                type(record.get(name)) is not list
                or not record[name]
                or any(type(item) is not str for item in record[name])
                or len(record[name]) != len(set(record[name]))
                for name in ("roles", "arms", "stages")
            )
        ):
            raise PermissionError("delay-line recovery physical record changed")
        physical_copy[path] = {
            **record,
            "roles": list(record["roles"]),
            "arms": list(record["arms"]),
            "stages": list(record["stages"]),
        }
    for path, current in runtime.raw_inputs.consumed.items():
        restored = physical_copy.get(path)
        if (
            restored is None
            or any(
                current.get(name) != restored.get(name)
                for name in ("path", "file_sha256", "byte_count", "kind")
            )
            or any(
                not set(current.get(name, ())).issubset(restored[name])
                for name in ("roles", "arms", "stages")
            )
        ):
            raise PermissionError("delay-line recovery physical baseline changed")
    runtime.raw_inputs.consumed = physical_copy
    runtime._access_consumed_count = -1
    runtime._access_opened_roles = ()

    h6_loader = h6_runtime._require_loader()
    h6_access = _nonnegative_integer_mapping_v1(
        value["h6_access"],
        expected_keys=set(h6_loader._access),
        name="H6 access counters",
    )
    h6_loader._access.clear()
    h6_loader._access.update(h6_access)

    local_loader = role_runtime._local_loader
    local_consumed = value["role_local_consumed"]
    if type(local_consumed) is not dict:
        raise PermissionError("delay-line recovery role-local ledger changed")
    local_record_keys = {
        "path",
        "file_sha256",
        "byte_count",
        "role",
        "row_index",
        "leaf",
    }
    local_copy: dict[str, dict[str, Any]] = {}
    for path, record in local_consumed.items():
        if (
            type(path) is not str
            or type(record) is not dict
            or set(record) != local_record_keys
            or record.get("path") != path
            or type(record.get("file_sha256")) is not str
            or len(record["file_sha256"]) != 64
            or type(record.get("byte_count")) is not int
            or record["byte_count"] <= 0
            or type(record.get("role")) is not str
            or type(record.get("row_index")) is not int
            or type(record.get("leaf")) is not str
        ):
            raise PermissionError("delay-line recovery role-local record changed")
        local_copy[path] = dict(record)
    local_access = _nonnegative_integer_mapping_v1(
        value["role_local_access"],
        expected_keys={
            "tensor_requests",
            "open_attempts",
            "open_successes",
            "decode_successes",
            "byte_count",
        },
        name="role-local access counters",
    )
    local_loader._consumed = local_copy
    local_loader._tensor_requests = local_access["tensor_requests"]
    local_loader._open_attempts = local_access["open_attempts"]
    local_loader._open_successes = local_access["open_successes"]
    local_loader._decode_successes = local_access["decode_successes"]
    local_loader._byte_count = local_access["byte_count"]

    place_access = _nonnegative_integer_mapping_v1(
        value["role_place_access"],
        expected_keys={"attempt", "sha256_verified", "success", "failure"},
        name="role-place access counters",
    )
    if place_access["attempt"] != place_access["success"] + place_access["failure"]:
        raise PermissionError("delay-line recovery role-place accounting changed")
    loader_calls = value["role_place_loader_calls"]
    row_keys = value["role_place_loaded_row_keys"]
    if (
        type(loader_calls) is not int
        or loader_calls < 0
        or type(row_keys) is not list
        or any(
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or (item[0], item[1]) not in role_runtime._place_rows
            for item in row_keys
        )
        or len(row_keys) != len({(item[0], item[1]) for item in row_keys})
    ):
        raise PermissionError("delay-line recovery role-place rows changed")
    role_runtime._place_reference_counts = place_access
    role_runtime._place_loader_calls = loader_calls
    role_runtime._place_loaded_row_keys = {
        (role, index) for role, index in row_keys
    }


def _serialize_snapshot(
    runtime: Any,
    model: Any,
    optimizer: Any,
    *,
    h6_runtime: Any,
    role_runtime: Any,
    authority: Mapping[str, Any],
    update: int,
    accounting: Any,
    observations: Mapping[int, metrics.ObservationMetrics],
    observation_receipts: Mapping[int, Mapping[str, Any]],
    trace: Sequence[Mapping[str, Any]],
    metric_bindings: Sequence[Mapping[str, Any]],
    snapshot_records: Sequence[Mapping[str, Any]],
    update0_place_rank: float,
    pending_decision: metrics.GateDecision | None,
) -> tuple[bytes, dict[str, Any]]:
    state = {
        "schema": f"{SCHEMA_PREFIX}_exact_recovery_state_v1",
        "update": update,
        "model_state_dict": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "accounting": accounting,
        "observations": dict(observations),
        "observation_receipts": {key: dict(value) for key, value in observation_receipts.items()},
        "trace": [dict(value) for value in trace],
        "metric_bindings": [dict(value) for value in metric_bindings],
        "snapshot_records": [dict(value) for value in snapshot_records],
        "update0_place_rank": update0_place_rank,
        "pending_decision": pending_decision,
        "access_state": _capture_exact_access_state_v1(
            runtime, h6_runtime, role_runtime
        ),
        "rng": {
            "torch_cpu": runtime.torch.random.get_rng_state().clone(),
            "visible_gpu": tuple(value.clone() for value in runtime.torch.cuda.get_rng_state_all()),
            "python": random.getstate(),
            "numpy": runtime.np.random.get_state(),
        },
        "schedule_cursor": {
            "completed_update": update,
            "memory_presentations": update * MEMORY_PRESENTATIONS_PER_UPDATE,
            "physical_presentations": update * PHYSICAL_PRESENTATIONS_PER_UPDATE,
            "total_presentations": update * PRESENTATIONS_PER_UPDATE,
        },
        "authority_sha256": hashlib.sha256(_canonical_json_bytes(authority)).hexdigest(),
    }
    buffer = io.BytesIO()
    runtime.torch.save(state, buffer)
    raw = buffer.getvalue()
    return raw, {
        "schema": f"{SCHEMA_PREFIX}_exact_recovery_binding_v1",
        "update": update,
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "schedule_cursor": state["schedule_cursor"],
        "same_attempt_infrastructure_recovery_only": True,
        "scientific_retry_authorized": False,
    }


def _restore_snapshot(runtime: Any, model: Any, optimizer: Any, recovery: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(recovery["state_path"]))
    try:
        state = runtime.torch.load(path, map_location=runtime.device, weights_only=False)
    except TypeError:
        state = runtime.torch.load(path, map_location=runtime.device)
    if (
        type(state) is not dict
        or state.get("schema") != f"{SCHEMA_PREFIX}_exact_recovery_state_v1"
        or state.get("update") != recovery.get("update")
        or state.get("authority_sha256")
        != hashlib.sha256(_canonical_json_bytes(authority)).hexdigest()
    ):
        raise PermissionError("delay-line recovery state identity changed")
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    runtime.torch.random.set_rng_state(state["rng"]["torch_cpu"])
    runtime.torch.cuda.set_rng_state_all(list(state["rng"]["visible_gpu"]))
    random.setstate(state["rng"]["python"])
    runtime.np.random.set_state(state["rng"]["numpy"])
    return state


def _terminal_common(
    *,
    terminal_update: int,
    decision: metrics.GateDecision,
    accounting: Any,
    metric_bindings: Sequence[Mapping[str, Any]],
    snapshot_records: Sequence[Mapping[str, Any]],
    trace_binding: Mapping[str, Any],
    access_binding: Mapping[str, Any],
    access_content_sha256: str,
    recovered: bool,
) -> dict[str, Any]:
    return {
        "terminal_update": terminal_update,
        "decision": asdict(decision),
        "accounting": _mapping(accounting, name="terminal accounting"),
        "metrics": [dict(value) for value in metric_bindings],
        "snapshots": [dict(value) for value in snapshot_records],
        "trace": dict(trace_binding),
        "terminal_access": dict(access_binding),
        "terminal_access_content_sha256": access_content_sha256,
        "attempt_consumed": True,
        "infrastructure_recovery_used": recovered,
        "retry_authorized": False,
        "scientific_resume_authorized": False,
        "probability_calibration_opened": False,
        "navigation_executed": False,
        "held_out_or_sealed_opened": False,
    }


def run_future_authorized_engine_v1(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    recovery: Mapping[str, Any] | None,
    runtime: Any,
    publisher: Any,
) -> dict[str, Any]:
    validated_authority = validate_future_execution_prerequisites_v1(dict(authority))
    validated_reservation = validate_attempt_reservation_v1(dict(reservation))
    if validated_reservation["authority_sha256"] != hashlib.sha256(
        _canonical_json_bytes(validated_authority)
    ).hexdigest():
        raise PermissionError("delay-line reservation does not bind authority")
    evaluation = __import__(EVALUATION_MODULE_NAME, fromlist=["*"])
    physical_train_scenes = tuple(sorted({str(pair["scene_id"]) for pair in runtime.pairs["train"]}))
    physical_selection_scenes = tuple(
        sorted({str(pair["scene_id"]) for pair in runtime.pairs["checkpoint_selection"]})
    )
    role_runtime = v5.load_memory_role_runtime_v1(
        runtime.runtime_data_root,
        runtime_inputs=validated_authority["runtime_inputs"],
        physical_train_scene_ids=physical_train_scenes,
        physical_selection_scene_ids=physical_selection_scenes,
    )
    h6_runtime = evaluation.load_delay_line_h6_runtime_v1(runtime.runtime_data_root)
    model: Any = None
    optimizer: Any = None
    accounting: Any = None
    observations: dict[int, metrics.ObservationMetrics] = {}
    observation_receipts: dict[int, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []
    update0_place_rank: float | None = None
    pending_decision: metrics.GateDecision | None = None
    start_update = 0
    integrity_pass = True
    stage = "initialize"
    try:
        if recovery is not None:
            publisher = ExactRecoveryReplayPublisherV1(publisher)
        model, optimizer, initialization = runtime.initialize_model_v13()
        if recovery is not None:
            stage = "restore_latest_exact_snapshot"
            restored = _restore_snapshot(runtime, model, optimizer, recovery, validated_authority)
            _restore_exact_access_state_v1(
                runtime,
                h6_runtime,
                role_runtime,
                restored.get("access_state"),
            )
            start_update = int(restored["update"])
            accounting = restored["accounting"]
            observations = dict(restored["observations"])
            observation_receipts = dict(restored["observation_receipts"])
            trace = list(restored["trace"])
            metric_bindings = list(restored["metric_bindings"])
            snapshot_records = list(restored["snapshot_records"])
            if not any(record.get("update") == start_update for record in snapshot_records):
                recovery_metadata = dict(recovery["metadata"])
                snapshot_records.append(
                    {
                        "update": start_update,
                        "state": dict(recovery_metadata["state"]),
                        "metadata": _content_bound_json_artifact_binding_v1(
                            f"snapshots/update_{start_update}.binding.json",
                            recovery_metadata,
                        ),
                        "metadata_content_sha256": recovery_metadata["content_sha256"],
                        "same_attempt_infrastructure_recovery_only": True,
                    }
                )
            update0_place_rank = float(restored["update0_place_rank"])
            pending_decision = restored["pending_decision"]
            integrity_pass = all(
                observation.safeguards.integrity_pass for observation in observations.values()
            )
            trace.append(
                {
                    "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                    "event": "infrastructure_recovery",
                    "update": start_update,
                    "scientific_retry": False,
                }
            )
        else:
            structural = physical_executor._derive_initial_structural_integrity_v13(runtime, model)
            integrity_pass = structural.get("passed") is True
            trace.append(
                {
                    "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                    "event": "initialized",
                    "update": 0,
                    "initialization": dict(initialization),
                    "initial_structural_integrity": structural,
                    "h6_preflight": h6_runtime.preflight_receipt(),
                    "role_preflight": role_runtime.preflight_receipt(),
                }
            )

        def observe(update: int) -> None:
            nonlocal integrity_pass, update0_place_rank
            receipt, gate_value, update0_place_rank = _observation(
                runtime=runtime,
                model=model,
                h6_runtime=h6_runtime,
                role_runtime=role_runtime,
                update=update,
                training_integrity_pass=integrity_pass,
                update0_place_rank=update0_place_rank,
            )
            observations[update] = gate_value
            observation_receipts[update] = receipt
            integrity_pass = (
                integrity_pass and gate_value.safeguards.integrity_pass
            )
            _, binding = _publish_json(publisher, f"metrics/update_{update}.json", receipt)
            metric_bindings.append(binding)

        if recovery is None:
            stage = "observe_update_0"
            accounting = runtime.training_module.JointTrainingAccountingV1()
            observe(0)
            update0_decision = evaluate_update0_gate_v1(
                observations[0], observation_receipts[0]
            )
            trace.append(
                {
                    "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                    "event": "update0_integrity_gate",
                    "update": 0,
                    "decision": asdict(update0_decision),
                }
            )
            pending_decision = None if update0_decision.passed else update0_decision
            start_update = 0

        if pending_decision is None or pending_decision.passed:
            for update in range(start_update + 1, MAXIMUM_UPDATES + 1):
                stage = f"train_update_{update}"
                physical_batches = _physical_batches(runtime, update)
                memory_batches = h6_runtime.build_train_microbatches(update, runtime.device)
                result = runtime.training_module.joint_training_update_v1(
                    model,
                    optimizer,
                    physical_batches,
                    memory_batches,
                    accounting=accounting,
                )
                accounting = result.accounting
                integrity = validate_update_integrity_v1(runtime, model, result, update=update)
                integrity_pass = integrity_pass and integrity["passed"]
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "optimizer_ema_update",
                        **integrity,
                    }
                )
                if not integrity["passed"]:
                    pending_decision = _immediate_integrity_failure_decision(
                        update=update,
                        status=f"STOP_UPDATE{update}_TRAINING_INTEGRITY",
                        checks=integrity["checks"],
                        observed={"training_update": update},
                    )
                    trace.append(
                        {
                            "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                            "event": "registered_gate",
                            "update": update,
                            "decision": asdict(pending_decision),
                        }
                    )
                    break
                if update not in OBSERVATION_UPDATES:
                    continue
                stage = f"observe_update_{update}"
                observe(update)
                pending_decision = evaluate_observation_integrity_v1(
                    observations[update]
                )
                if pending_decision is not None:
                    pass
                elif update == 250:
                    pending_decision = metrics.update250_futility_decision(
                        observations[100], observations[250]
                    )
                elif update == 500:
                    pending_decision = metrics.update500_continuation_decision(observations[500])
                elif update == 1_000:
                    pending_decision = metrics.terminal_qualification_decision(
                        [observations[500], observations[750], observations[1_000]]
                    )
                else:
                    pending_decision = None
                if pending_decision is not None:
                    trace.append(
                        {
                            "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                            "event": "registered_gate",
                            "update": update,
                            "decision": asdict(pending_decision),
                        }
                    )
                if update in SNAPSHOT_UPDATES:
                    if accounting is None or update0_place_rank is None:
                        raise RuntimeError("delay-line snapshot state is incomplete")
                    raw, metadata = _serialize_snapshot(
                        runtime,
                        model,
                        optimizer,
                        h6_runtime=h6_runtime,
                        role_runtime=role_runtime,
                        authority=validated_authority,
                        update=update,
                        accounting=accounting,
                        observations=observations,
                        observation_receipts=observation_receipts,
                        trace=trace,
                        metric_bindings=metric_bindings,
                        snapshot_records=snapshot_records,
                        update0_place_rank=update0_place_rank,
                        pending_decision=pending_decision,
                    )
                    state_binding = dict(
                        publisher.publish_bytes(f"snapshots/update_{update}.pt", raw)
                    )
                    metadata_value, metadata_binding = _publish_json(
                        publisher,
                        f"snapshots/update_{update}.binding.json",
                        {**metadata, "state": state_binding},
                    )
                    snapshot_records.append(
                        {
                            "update": update,
                            "state": state_binding,
                            "metadata": metadata_binding,
                            "metadata_content_sha256": metadata_value["content_sha256"],
                            "same_attempt_infrastructure_recovery_only": True,
                        }
                    )
                if pending_decision is not None and not pending_decision.passed:
                    break

        if accounting is None or pending_decision is None:
            raise RuntimeError("delay-line controller ended without a registered decision")
        terminal_update = pending_decision.update
        stage = f"terminalize_update_{terminal_update}"
        trace_raw = b"".join(
            _canonical_json_bytes(_content_bound(row)) + b"\n" for row in trace
        )
        trace_binding = publisher.publish_bytes("trace.jsonl", trace_raw)
        terminal_access = {
            "schema": f"{SCHEMA_PREFIX}_terminal_access_receipt_v1",
            "physical": runtime.terminal_access_receipt_v13(),
            "h6": h6_runtime.terminal_access_receipt(),
            "roles": role_runtime.terminal_access_receipt(),
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        }
        access_value, access_binding = _publish_json(
            publisher, "receipts/terminal_access.json", terminal_access
        )
        common = _terminal_common(
            terminal_update=terminal_update,
            decision=pending_decision,
            accounting=accounting,
            metric_bindings=metric_bindings,
            snapshot_records=snapshot_records,
            trace_binding=trace_binding,
            access_binding=access_binding,
            access_content_sha256=access_value["content_sha256"],
            recovered=recovery is not None,
        )
        if not pending_decision.passed:
            value, _ = _publish_json(
                publisher,
                "failure.json",
                {
                    "schema": f"{SCHEMA_PREFIX}_scientific_failure_v1",
                    "status": pending_decision.status,
                    **common,
                    "checkpoint_selected": False,
                },
            )
            return value
        selected = pending_decision.selected_update
        selected_snapshot = next(
            (record for record in snapshot_records if record["update"] == selected), None
        )
        if selected_snapshot is None:
            raise RuntimeError("selected delay-line observation has no complete snapshot")
        value, _ = _publish_json(
            publisher,
            "success.json",
            {
                "schema": f"{SCHEMA_PREFIX}_success_v1",
                "status": pending_decision.status,
                **common,
                "checkpoint_selected": True,
                "selected_checkpoint": selected_snapshot,
                "navigation_authorized": False,
                "held_out_authorized": False,
            },
        )
        return value
    except BaseException as error:
        output_root = Path(publisher.output_root)
        interruption_path = output_root / "interruption.json"
        if (
            recovery is None
            and not stage.startswith("terminalize_update_")
            and not interruption_path.exists()
            and not interruption_path.is_symlink()
            and _latest_snapshot(output_root) is not None
        ):
            value, _ = _publish_json(
                publisher,
                "interruption.json",
                {
                    "schema": f"{SCHEMA_PREFIX}_recoverable_infrastructure_interruption_v1",
                    "status": "INTERRUPTED_LATEST_EXACT_SNAPSHOT_RECOVERY_AVAILABLE",
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "attempt_consumed": True,
                    "scientific_retry": False,
                    "cap_changed": False,
                    "infrastructure_recovery_available": True,
                    "navigation_executed": False,
                    "held_out_or_sealed_opened": False,
                },
            )
            return value
        return terminalize_failure_v1(
            output_root,
            validated_reservation,
            stage=stage,
            error=error,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
    finally:
        h6_runtime.close()
        role_runtime.close()


run_future_authorized_engine_v13 = run_future_authorized_engine_v1


__all__ = [
    "AUTHORITY_RELATIVE_PATH",
    "CERTIFIED_SOURCE_ROOT",
    "EVALUATION_MODULE_NAME",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MODEL_CLASS_NAME",
    "MODEL_MODULE_NAME",
    "OBSERVATION_UPDATES",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "RUNTIME_INPUT_BINDING_NAMES",
    "SNAPSHOT_UPDATES",
    "TRAINING_MODULE_NAME",
    "reserve_or_recover_attempt_v1",
    "run_future_authorized_engine_v1",
    "terminalize_failure_v1",
    "ExactRecoveryReplayPublisherV1",
    "_capture_exact_access_state_v1",
    "_content_bound_json_artifact_binding_v1",
    "_restore_exact_access_state_v1",
    "evaluate_observation_integrity_v1",
    "evaluate_update0_gate_v1",
    "validate_content_bound_v1",
    "validate_future_execution_prerequisites_v1",
    "validate_update_integrity_v1",
]
