#!/usr/bin/env python3
"""One-shot controller for the preregistered V27 joint JEPA probe.

The proven V13/V26 physical evaluator and raw-data runtime remain authoritative
for the labelled half of each update.  This controller adds only the H6 plan
route, its validation controls, and the V27 terminal gate.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import (
    execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26 as v26,
)
from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
    as physical_executor,
)


SCHEMA_PREFIX = (
    "lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27"
)
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "4e0d1a10412e0992c69886a628c5b29c7d16b624"
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27_source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27_clean_export_certification_2026-07-30.json"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27_execution_authorization_2026-07-30.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27/attempt_v1"
)
CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-v27-explicit-plan-successor-source"
)
MODEL_CLASS_NAME = (
    "GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_object_space_explicit_plan_discounted_successor_state_"
    "joint_jepa_v27"
)
EVALUATION_MODULE_NAME = (
    "scripts.evaluate_go2_rgb_object_space_explicit_plan_discounted_successor_"
    "state_joint_jepa_v27"
)

MAXIMUM_UPDATES = 400
MAXIMUM_PRESENTATIONS = 12_800
PHYSICAL_PRESENTATIONS_PER_UPDATE = 16
PLAN_PRESENTATIONS_PER_UPDATE = 16
PRESENTATIONS_PER_UPDATE = 32
OBSERVATION_UPDATES = (0, 100, 400)
TERMINAL_UPDATES = (400,)
PLAN_METRIC_NAMES = (
    "persistence_advantage",
    "wrong_plan_advantage",
    "tail_advantage",
    "wrong_scene_advantage",
    "mean_prior_advantage",
)

# Exact inherited physical runtime API consumed by the V13 composer.
RUNTIME_INPUT_BINDING_NAMES = v26.RUNTIME_INPUT_BINDING_NAMES
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = v26.CHECKPOINT_SCHEDULE_PREFIX_SHA256
REGISTERED_FAMILIES = v26.REGISTERED_FAMILIES
SCOPES = v26.SCOPES
V12_GATE_CHECK_NAMES = v26.V12_GATE_CHECK_NAMES
CONTROL_NAMES = v26.CONTROL_NAMES
MATCHED_UPDATE400_THRESHOLDS = v26.MATCHED_UPDATE400_THRESHOLDS
EXPECTED_RUNTIME_FINGERPRINT = v26.EXPECTED_RUNTIME_FINGERPRINT
flatten_physical_metrics_v13 = v26.flatten_physical_metrics_v26
registered_wrong_rgb_mapping_v13 = v26.registered_wrong_rgb_mapping_v26
_canonical_json_bytes = v26._canonical_json_bytes
_write_immutable_json_v13 = v26._write_immutable_json_v13


def _content_bound(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    value.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    return value


def validate_content_bound_v27(value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("content_sha256")) is not str:
        raise TypeError("V27 content-bound value must be an exact object")
    observed = value["content_sha256"]
    core = dict(value)
    core.pop("content_sha256")
    expected = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    if observed != expected:
        raise RuntimeError("V27 content binding changed")
    return dict(value)


def _binding(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V27 {name} binding is absent")
    result = dict(value)
    if (
        set(result) != {"path", "file_sha256", "byte_count"}
        or type(result["path"]) is not str
        or type(result["file_sha256"]) is not str
        or len(result["file_sha256"]) != 64
        or type(result["byte_count"]) is not int
        or result["byte_count"] <= 0
    ):
        raise TypeError(f"V27 {name} binding changed")
    return result


def validate_future_execution_prerequisites_v27(
    authority: Any,
) -> dict[str, Any]:
    value = validate_content_bound_v27(authority)
    required = {
        "schema": f"{SCHEMA_PREFIX}_future_execution_authority_v1",
        "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
        "scientific_payload_authorized": True,
        "one_shot": True,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "retry_authorized": False,
        "resume_authorized": False,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    }
    if any(value.get(name) != expected for name, expected in required.items()):
        raise PermissionError("V27 authority identity or terminal bounds changed")
    if value.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise PermissionError("V27 authority is not bound to the final preregistration")
    selectors = value.get("selectors")
    if selectors != {
        "executor_module": __name__,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "evaluation_module": EVALUATION_MODULE_NAME,
    }:
        raise PermissionError("V27 runtime selectors changed")
    if value.get("runtime_data_root") != str(Path(__file__).resolve().parents[1]):
        # A clean export deliberately points back to the immutable data worktree.
        expected_data_root = "/home/andrewknowles/Workspace/LeWMQuad-v3"
        if value.get("runtime_data_root") != expected_data_root:
            raise PermissionError("V27 runtime-data root changed")
    certification = value.get("clean_export_certification")
    if not isinstance(certification, Mapping) or set(certification) != {
        "path",
        "file_sha256",
        "byte_count",
        "content_sha256",
    }:
        raise PermissionError("V27 authority lacks its clean-export certification")
    if certification.get("path") != CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH:
        raise PermissionError("V27 certification path changed")
    if not isinstance(value.get("runtime_inputs"), Mapping) or any(
        name not in value["runtime_inputs"] for name in RUNTIME_INPUT_BINDING_NAMES
    ):
        raise PermissionError("V27 inherited runtime bindings are incomplete")
    for role in ("h6_train_index", "h6_validation_index"):
        _binding(value["runtime_inputs"].get(role), name=role)
    if value.get("rgb_root_relative_path") != (
        ".generated/datagen_full/render_textured_v03"
    ):
        raise PermissionError("V27 H6 RGB root changed")
    return value


validate_content_bound_v13 = validate_content_bound_v27
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v27
)


def _read_certification(repository_root: Path) -> dict[str, Any]:
    path = repository_root / CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise PermissionError("V27 clean-export certification is absent")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError("V27 clean-export certification is invalid") from error
    value = validate_content_bound_v27(value)
    if (
        value.get("schema") != f"{SCHEMA_PREFIX}_clean_export_certification_v1"
        or value.get("status") != "PASS_CLEAN_EXPORT_CERTIFIED"
        or value.get("passed") is not True
        or value.get("certified_source_root") != str(repository_root)
    ):
        raise PermissionError("V27 clean-export certification identity changed")
    return value


def validate_bound_sources_v27(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    certification = _read_certification(root)
    bindings = certification.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise PermissionError("V27 certified source inventory is absent")
    canonical = hashlib.sha256(_canonical_json_bytes(bindings)).hexdigest()
    if canonical != certification.get("bindings_sha256"):
        raise PermissionError("V27 source inventory binding changed")
    validated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_binding in bindings:
        binding = _binding(raw_binding, name="certified source")
        relative = Path(binding["path"])
        path = root / relative
        lexical = relative.as_posix().lower()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] not in {"docs", "lewm", "scripts", "lewm_worlds"}
            or any(token in lexical for token in ("heldout", "held_out", "sealed"))
            or lexical.startswith(".generated/")
            or binding["path"] in seen_paths
            or path.is_symlink()
            or not path.is_file()
        ):
            raise PermissionError("V27 certified source path escaped or changed type")
        payload = path.read_bytes()
        if (
            len(payload) != binding["byte_count"]
            or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"V27 certified source changed: {relative}")
        seen_paths.add(binding["path"])
        validated.append(binding)
    return {
        "validated_path_count": len(validated),
        "bindings_sha256": canonical,
        "certification_content_sha256": certification["content_sha256"],
        "passed": True,
    }


validate_bound_sources_v13 = validate_bound_sources_v27


def reserve_attempt_v27(
    repository_root: Path,
    authority: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    validated = validate_future_execution_prerequisites_v27(dict(authority))
    output = root / OUTPUT_ROOT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    core = {
        "schema": f"{SCHEMA_PREFIX}_attempt_reservation_v1",
        "status": "RESERVED_ONE_SHOT",
        "created_utc": created_utc,
        "authority_sha256": hashlib.sha256(
            _canonical_json_bytes(validated)
        ).hexdigest(),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "attempt": 1,
    }
    reservation = _content_bound(core)
    path = output / "reservation.json"
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(reservation) + b"\n")
    return reservation


def validate_attempt_reservation_v27(value: Any) -> dict[str, Any]:
    reservation = validate_content_bound_v27(value)
    if (
        reservation.get("schema") != f"{SCHEMA_PREFIX}_attempt_reservation_v1"
        or reservation.get("status") != "RESERVED_ONE_SHOT"
        or reservation.get("output_root") != OUTPUT_ROOT_RELATIVE_PATH
        or reservation.get("maximum_updates") != MAXIMUM_UPDATES
        or reservation.get("maximum_presentations") != MAXIMUM_PRESENTATIONS
        or reservation.get("attempt") != 1
    ):
        raise PermissionError("V27 reservation identity changed")
    return reservation


reserve_attempt_v13 = reserve_attempt_v27
validate_attempt_reservation_v13 = validate_attempt_reservation_v27


def terminalize_failure_v27(
    output_root: Path,
    reservation: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
    created_utc: str,
) -> dict[str, Any]:
    validate_attempt_reservation_v27(dict(reservation))
    core = {
        "schema": f"{SCHEMA_PREFIX}_exception_failure_v1",
        "status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME",
        "stage": stage,
        "created_utc": created_utc,
        "exception_type": type(error).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(error).encode("utf-8")
        ).hexdigest(),
        "attempt_consumed": True,
        "checkpoint_published": False,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    return _write_immutable_json_v13(Path(output_root) / "failure.json", core)


terminalize_failure_v13 = terminalize_failure_v27


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"V27 {name} must be a dataclass or mapping")
    return result


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"V27 {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"V27 {name} is nonfinite")
    return result


def validate_update_integrity_v27(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
) -> dict[str, Any]:
    accounting = _mapping(result.accounting, name="accounting")
    expected = {
        "updates": update,
        "presentations": 32 * update,
        "physical_presentations": 16 * update,
        "plan_presentations": 16 * update,
        "physical_microbatch_graphs": 4 * update,
        "plan_microbatch_graphs": 4 * update,
        "autograd_grad_calls": 16 * update,
        "optimizer_steps": update,
        "ema_steps": update,
    }
    if accounting != expected:
        raise RuntimeError("V27 update accounting changed")
    route_names = (
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "predictor_core_protected_survival_output",
        "explicit_plan_discounted_successor_state",
    )
    routes = {
        name: _mapping(value, name=f"route {name}")
        for name, value in result.gradient_routes.items()
    }
    if tuple(routes) != route_names:
        raise RuntimeError("V27 gradient route order or membership changed")
    for name, route in routes.items():
        preclip = _finite(route.get("preclip_l2"), name=f"{name} preclip L2")
        scale = _finite(route.get("applied_scale"), name=f"{name} scale")
        if route.get("absent_tensor_gradient_count") != 0 or preclip < 0.0:
            raise RuntimeError(f"V27 route {name} has an absent or invalid gradient")
        if name != "representation" and not preclip > 0.0:
            raise RuntimeError(f"V27 required route {name} is zero")
        if not 0.0 < scale <= 1.0:
            raise RuntimeError(f"V27 route {name} clipping scale changed")
    losses = {
        name: _finite(value, name=f"loss {name}")
        for name, value in result.mean_losses.items()
    }
    if not {"C", "N27", "J24", "P27", "L27"}.issubset(losses):
        raise RuntimeError("V27 required loss receipt is incomplete")
    diagnostics = _mapping(result.plan_diagnostics, name="plan diagnostics")
    energies = diagnostics.get("energy_per_row")
    if (
        diagnostics.get("mechanism")
        != "explicit_plan_discounted_successor_state"
        or diagnostics.get("gamma") != 0.9
        or diagnostics.get("p25_evaluation_count") != 0
        or not isinstance(energies, (tuple, list))
        or len(energies) != 16
        or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in energies)
    ):
        raise RuntimeError("V27 plan diagnostics changed")
    if (
        result.target_gradient_tensor_count != 0
        or result.optimizer_steps_this_update != 1
        or result.ema_steps_this_update != 1
        or int(model.ema_update_count.item()) != update
        or any(
            parameter.grad is not None
            for module in model.target_modules()
            for parameter in module.parameters()
        )
    ):
        raise RuntimeError("V27 target, optimizer, or EMA integrity failed")
    for value in model.state_dict().values():
        if value.is_floating_point() and not bool(runtime.torch.isfinite(value).all()):
            raise FloatingPointError("V27 model state became nonfinite")
    return {
        "schema": f"{SCHEMA_PREFIX}_update_integrity_v1",
        "update": update,
        "accounting": accounting,
        "gradient_routes": routes,
        "mean_losses": losses,
        "plan_diagnostics": diagnostics,
        "target_gradient_tensor_count": 0,
        "p25_evaluation_count": 0,
        "passed": True,
    }


def evaluate_gate_v27(
    *,
    update100_physical: Mapping[str, Any],
    update400_physical: Mapping[str, Any],
    update400_controls: Mapping[str, Mapping[str, bool]],
    plan_metrics: Mapping[str, Any],
    integrity_pass: bool,
) -> dict[str, Any]:
    physical = v26.evaluate_update400_gate_v26(
        update100_physical,
        update400_physical,
        update400_controls,
        integrity_pass=integrity_pass,
        matched_update400_thresholds=MATCHED_UPDATE400_THRESHOLDS,
    )
    advantages = plan_metrics.get("advantages")
    if not isinstance(advantages, Mapping) or tuple(advantages) != PLAN_METRIC_NAMES:
        raise RuntimeError("V27 registered plan metrics changed")

    def positive(name: str, *, minimum_mean: float = 0.0) -> bool:
        metric = advantages[name]
        mean = _finite(metric.get("equal_family_mean"), name=f"{name} mean")
        mean_pass = mean > 0.0 if minimum_mean == 0.0 else mean >= minimum_mean
        return bool(
            mean_pass
            and _finite(metric.get("bootstrap_lower_95"), name=f"{name} lower")
            > 0.0
            and metric.get("positive_family_count", -1) >= 6
        )

    checks = {
        "integrity_pass": bool(integrity_pass),
        "all_registered_values_finite": plan_metrics.get(
            "all_registered_values_finite"
        )
        is True,
        "correct_ratio_strictly_below_0_90": _finite(
            plan_metrics.get("correct_ratio"), name="correct ratio"
        )
        < 0.90,
        "persistence_advantage_pass": positive("persistence_advantage"),
        "wrong_plan_advantage_pass": positive("wrong_plan_advantage"),
        "tail_advantage_at_least_0_05_and_robust": positive(
            "tail_advantage", minimum_mean=0.05
        ),
        "exact_plan_wrong_scene_advantage_pass": positive(
            "wrong_scene_advantage"
        ),
        "leave_one_scene_mean_prior_advantage_pass": positive(
            "mean_prior_advantage"
        ),
        "physical_update400_gate_pass": physical.get("passed") is True,
    }
    passed = all(checks.values())
    return {
        "schema": f"{SCHEMA_PREFIX}_update400_gate_v1",
        "update": 400,
        "checks": checks,
        "physical": physical,
        "plan": dict(plan_metrics),
        "passed": passed,
        "action": (
            "PASS_PUBLISH_BOUNDED_DEVELOPMENT_SCALE_SEED"
            if passed
            else "FAIL_TERMINAL_NO_RETRY_NO_RESUME"
        ),
    }


def _publish_json(
    publisher: Any, relative_path: str, core: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = publisher.publish_json(relative_path, dict(core))
    if type(result) is not dict or set(result) != {"value", "binding"}:
        raise RuntimeError("V27 publisher JSON result changed")
    return validate_content_bound_v27(result["value"]), dict(result["binding"])


def _serialize_checkpoint_v27(
    runtime: Any,
    model: Any,
    optimizer: Any,
    accounting: Any,
    authority: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    state = {
        "schema": f"{SCHEMA_PREFIX}_development_scale_seed_v1",
        "update": 400,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "model_state_dict": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "accounting": _mapping(accounting, name="checkpoint accounting"),
        "rng": {
            "torch_cpu": runtime.torch.random.get_rng_state().clone(),
            "visible_gpu": tuple(
                value.clone() for value in runtime.torch.cuda.get_rng_state_all()
            ),
        },
        "authority_sha256": hashlib.sha256(
            _canonical_json_bytes(authority)
        ).hexdigest(),
        "resume_authorized": False,
    }
    buffer = io.BytesIO()
    runtime.torch.save(state, buffer)
    raw = buffer.getvalue()
    return raw, {
        "schema": f"{SCHEMA_PREFIX}_development_scale_seed_binding_v1",
        "update": 400,
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "resume_authorized": False,
        "navigation_authorized": False,
        "held_out_authorized": False,
    }


def run_future_authorized_engine_v27(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
) -> dict[str, Any]:
    """Execute exactly update 0, 400 mixed updates, and the terminal gate."""

    validated_authority = validate_future_execution_prerequisites_v27(
        dict(authority)
    )
    validated_reservation = validate_attempt_reservation_v27(dict(reservation))
    if validated_reservation["authority_sha256"] != hashlib.sha256(
        _canonical_json_bytes(validated_authority)
    ).hexdigest():
        raise PermissionError("V27 reservation does not bind supplied authority")

    evaluation = __import__(EVALUATION_MODULE_NAME, fromlist=["*"])
    h6_runtime = evaluation.load_v27_h6_runtime(runtime.runtime_data_root)
    trace: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    accounting: Any = None
    model: Any = None
    optimizer: Any = None
    terminal_published = False
    stage = "initialize"
    try:
        model, optimizer, initialization = runtime.initialize_model_v13()
        initial_structural = physical_executor._derive_initial_structural_integrity_v13(
            runtime, model
        )
        trace.append(
            {
                "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                "event": "initialized",
                "update": 0,
                "initialization": dict(initialization),
                "initial_structural_integrity": initial_structural,
                "h6_preflight": h6_runtime.preflight_receipt(),
            }
        )
        observations: dict[int, dict[str, Any]] = {}
        integrity_pass = initial_structural.get("passed") is True
        for update in range(0, MAXIMUM_UPDATES + 1):
            if update:
                stage = f"train_update_{update}"
                start = (update - 1) * PHYSICAL_PRESENTATIONS_PER_UPDATE
                physical_indices = tuple(
                    runtime.schedule[
                        start : start + PHYSICAL_PRESENTATIONS_PER_UPDATE
                    ]
                )
                physical_batches = runtime.build_microbatches_v13(
                    physical_indices, update=update
                )
                plan_batches = h6_runtime.build_train_microbatches(
                    update, runtime.device
                )
                result = runtime.training_module.joint_training_update_v27(
                    model,
                    optimizer,
                    physical_batches,
                    plan_batches,
                    accounting=accounting,
                )
                accounting = result.accounting
                integrity = validate_update_integrity_v27(
                    runtime, model, result, update=update
                )
                integrity_pass = integrity_pass and integrity["passed"]
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "optimizer_ema_update",
                        **integrity,
                    }
                )
            if update not in OBSERVATION_UPDATES:
                continue
            stage = f"observe_update_{update}"
            physical = physical_executor._observation_v13(
                runtime,
                model,
                update=update,
                integrity_pass=integrity_pass,
            )
            plan = h6_runtime.evaluate_plan_metrics(
                model, update, runtime.device
            )
            observation = {
                "schema": f"{SCHEMA_PREFIX}_observation_v1",
                "update": update,
                "physical": physical,
                "plan": plan,
                "integrity_pass": bool(
                    integrity_pass
                    and physical.get("integrity_pass") is True
                    and plan.get("integrity", {}).get("passed") is True
                ),
                "state_mutation_count": 0,
                "probability_calibration_opened": False,
                "navigation_executed": False,
                "held_out_or_sealed_opened": False,
            }
            observations[update] = observation
            _, binding = _publish_json(
                publisher, f"metrics/update_{update}.json", observation
            )
            metric_bindings.append(binding)

        if accounting is None or set(observations) != set(OBSERVATION_UPDATES):
            raise RuntimeError("V27 controller did not complete its exact schedule")
        stage = "classify_update400"
        gate = evaluate_gate_v27(
            update100_physical=observations[100]["physical"]["physical"],
            update400_physical=observations[400]["physical"]["physical"],
            update400_controls=observations[400]["physical"]["controls"],
            plan_metrics=observations[400]["plan"],
            integrity_pass=all(
                observation["integrity_pass"]
                for observation in observations.values()
            ),
        )
        trace.append(
            {
                "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                "event": "update400_terminal_gate",
                "update": 400,
                "decision": gate,
            }
        )
        trace_raw = b"".join(
            _canonical_json_bytes(_content_bound(row)) + b"\n" for row in trace
        )
        trace_binding = publisher.publish_bytes("trace.jsonl", trace_raw)
        h6_access = h6_runtime.terminal_access_receipt()
        physical_access = runtime.terminal_access_receipt_v13()
        access_value, access_binding = _publish_json(
            publisher,
            "receipts/terminal_access.json",
            {
                "schema": f"{SCHEMA_PREFIX}_terminal_access_receipt_v1",
                "physical": physical_access,
                "h6": h6_access,
                "probability_calibration_opened": False,
                "navigation_executed": False,
                "held_out_or_sealed_opened": False,
            },
        )
        common = {
            "terminal_update": 400,
            "decision": gate,
            "accounting": _mapping(accounting, name="terminal accounting"),
            "metrics": metric_bindings,
            "trace": trace_binding,
            "terminal_access": access_binding,
            "terminal_access_content_sha256": access_value["content_sha256"],
            "attempt_consumed": True,
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        }
        if not gate["passed"]:
            value, _ = _publish_json(
                publisher,
                "failure.json",
                {
                    "schema": f"{SCHEMA_PREFIX}_scientific_failure_v1",
                    "status": "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL",
                    **common,
                    "checkpoint_published": False,
                    "retry_authorized": False,
                    "resume_authorized": False,
                },
            )
            terminal_published = True
            return value

        stage = "publish_pass_checkpoint"
        checkpoint_raw, checkpoint_core = _serialize_checkpoint_v27(
            runtime, model, optimizer, accounting, validated_authority
        )
        checkpoint_binding = publisher.publish_bytes(
            "checkpoint_update_400.pt", checkpoint_raw
        )
        checkpoint_value, checkpoint_metadata_binding = _publish_json(
            publisher,
            "checkpoint_update_400.binding.json",
            {**checkpoint_core, "checkpoint": checkpoint_binding},
        )
        value, _ = _publish_json(
            publisher,
            "success.json",
            {
                "schema": f"{SCHEMA_PREFIX}_success_v1",
                "status": "PASS_DEVELOPMENT_UPDATE400_TERMINAL",
                **common,
                "checkpoint_published": True,
                "checkpoint": checkpoint_binding,
                "checkpoint_metadata": checkpoint_metadata_binding,
                "checkpoint_metadata_content_sha256": checkpoint_value[
                    "content_sha256"
                ],
                "scale_resume_requires_separate_authority": True,
                "resume_authorized": False,
                "navigation_authorized": False,
                "held_out_authorized": False,
            },
        )
        terminal_published = True
        return value
    except BaseException as error:
        if terminal_published:
            raise
        try:
            trace_raw = b"".join(
                _canonical_json_bytes(_content_bound(row)) + b"\n"
                for row in trace
            )
            trace_binding = publisher.publish_bytes("trace.jsonl", trace_raw)
            value, _ = _publish_json(
                publisher,
                "failure.json",
                {
                    "schema": f"{SCHEMA_PREFIX}_exception_failure_v1",
                    "status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME",
                    "stage": stage,
                    "exception_type": type(error).__name__,
                    "exception_message_sha256": hashlib.sha256(
                        str(error).encode("utf-8")
                    ).hexdigest(),
                    "trace": trace_binding,
                    "attempt_consumed": True,
                    "checkpoint_published": False,
                    "retry_authorized": False,
                    "resume_authorized": False,
                },
            )
            return value
        except BaseException:
            raise error
    finally:
        h6_runtime.close()


run_future_authorized_engine_v13 = run_future_authorized_engine_v27


__all__ = [
    "AUTHORITY_RELATIVE_PATH",
    "CERTIFIED_SOURCE_ROOT",
    "CHECKPOINT_SCHEDULE_PREFIX_SHA256",
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH",
    "CONTROL_NAMES",
    "EVALUATION_MODULE_NAME",
    "MATCHED_UPDATE400_THRESHOLDS",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MODEL_CLASS_NAME",
    "MODEL_MODULE_NAME",
    "OBSERVATION_UPDATES",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "REGISTERED_FAMILIES",
    "RUNTIME_INPUT_BINDING_NAMES",
    "SCHEMA_PREFIX",
    "SCOPES",
    "TERMINAL_UPDATES",
    "TRAINING_MODULE_NAME",
    "V12_GATE_CHECK_NAMES",
    "evaluate_gate_v27",
    "reserve_attempt_v27",
    "run_future_authorized_engine_v27",
    "terminalize_failure_v27",
    "validate_bound_sources_v27",
    "validate_content_bound_v27",
    "validate_future_execution_prerequisites_v27",
    "validate_update_integrity_v27",
]
