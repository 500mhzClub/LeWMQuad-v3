#!/usr/bin/env python3
"""V15-only extended-horizon lifecycle over the frozen V14/V13 engine.

The caller supplies its privately adapted engine module explicitly.  This
module contains no discovery, input loading, authority selection, or public
execution entrypoint; the certified one-shot launcher retains those duties.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import io
from typing import Any, Mapping


FINAL_UPDATE_V15 = 2_000
OBSERVATION_UPDATES_V15 = (0, 100, 400, 1_000, 1_400, 2_000)
TERMINAL_UPDATES_V15 = (400, 1_400, 2_000)

UPDATE1400_THRESHOLDS_V15 = {
    "passed_margin_count_minimum": 99,
    "total_shortfall_strictly_less_than": 38.1,
    "rough_depth_p95_m_strictly_less_than": 1.304,
    "rough_pixel_balanced_accuracy_strictly_greater_than":
        0.8198594673963917,
    "rough_ground_balanced_accuracy_strictly_greater_than":
        0.647134926562893,
}


def evaluate_update1400_gate_v15(
    v12_gate: Mapping[str, Any],
    physical_summary: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
    engine: Any,
) -> dict[str, Any]:
    """Evaluate the preregistered update-1,400 feasibility stop."""

    if type(integrity_pass) is not bool:
        raise TypeError("update-1400 structural-integrity decision must be Boolean")
    if (
        not isinstance(v12_gate, Mapping)
        or type(v12_gate.get("passed")) is not bool
        or type(v12_gate.get("checks")) is not dict
        or tuple(v12_gate["checks"]) != engine.V12_GATE_CHECK_NAMES
        or any(type(value) is not bool for value in v12_gate["checks"].values())
        or v12_gate["passed"] != all(v12_gate["checks"].values())
    ):
        raise ValueError("inherited V12 24-check gate changed or is inconsistent")

    if type(controls) is not dict or set(controls) != set(engine.CONTROL_NAMES):
        raise ValueError("update-1400 causal control set changed")
    control_checks: dict[str, bool] = {}
    for name in engine.CONTROL_NAMES:
        row = controls[name]
        if type(row) is not dict or set(row) != set(engine.CONTROL_CHECK_NAMES):
            raise ValueError(f"update-1400 causal control schema changed: {name}")
        for check in engine.CONTROL_CHECK_NAMES:
            if type(row[check]) is not bool:
                raise TypeError("update-1400 causal control decisions must be Boolean")
            control_checks[f"{name}:{check}"] = row[check]

    physical = engine._validate_physical_summary(physical_summary)
    rough = physical["rough_motion"]
    checks = {
        "structural_integrity_pass": integrity_pass,
        "inherited_v12_full_arm_24_of_24": v12_gate["passed"],
        "all_twelve_causal_control_checks_true": all(control_checks.values()),
        "passed_physical_margin_count_at_least_99": (
            physical["passed_margin_count"]
            >= UPDATE1400_THRESHOLDS_V15["passed_margin_count_minimum"]
        ),
        "total_physical_shortfall_strictly_below_38_1": (
            physical["total_shortfall"]
            < UPDATE1400_THRESHOLDS_V15[
                "total_shortfall_strictly_less_than"
            ]
        ),
        "rough_depth_p95_strictly_below_1_304_m": (
            rough["depth_p95_m"]
            < UPDATE1400_THRESHOLDS_V15[
                "rough_depth_p95_m_strictly_less_than"
            ]
        ),
        "rough_pixel_balanced_accuracy_strictly_above_threshold": (
            rough["pixel_balanced_accuracy"]
            > UPDATE1400_THRESHOLDS_V15[
                "rough_pixel_balanced_accuracy_strictly_greater_than"
            ]
        ),
        "rough_ground_balanced_accuracy_strictly_above_threshold": (
            rough["ground_balanced_accuracy"]
            > UPDATE1400_THRESHOLDS_V15[
                "rough_ground_balanced_accuracy_strictly_greater_than"
            ]
        ),
    }
    passed = all(checks.values())
    return {
        "schema": f"{engine.SCHEMA_PREFIX}_update1400_feasibility_gate_v1",
        "update": 1_400,
        "thresholds": dict(UPDATE1400_THRESHOLDS_V15),
        "checks": checks,
        "causal_control_checks": control_checks,
        "passed": passed,
        "action": (
            "CONTINUE_TO_UPDATE_2000"
            if passed
            else "FAIL_TERMINAL_NO_RETRY_NO_RESUME"
        ),
        "next_update": 2_000 if passed else None,
        "checkpoint_authorized": False,
        "physical_adapter_preregistration_eligible": False,
        "probability_calibration_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "held_out_authorized": False,
        "retry_authorized": False,
        "resume_authorized": False,
    }


def _serialize_development_checkpoint_v15(
    runtime: Any,
    model: Any,
    authority: Mapping[str, Any],
    *,
    engine: Any,
) -> tuple[bytes, dict[str, Any]]:
    """Serialize the sole eligible update-2,000 development checkpoint."""

    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    }
    state_manifest = []
    for name, value in state.items():
        raw_tensor = value.numpy().tobytes(order="C")
        state_manifest.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "numel": int(value.numel()),
                "tensor_sha256": hashlib.sha256(raw_tensor).hexdigest(),
            }
        )
    groups = model.trainable_parameter_groups_v13()
    online_counts = {
        name: sum(parameter.numel() for _, parameter in group)
        for name, group in zip(
            ("shared", "representation", "predictor"), groups, strict=True
        )
    }
    config = model.config
    if not is_dataclass(config):
        raise RuntimeError("V15 model config is not the frozen dataclass")
    metadata = {
        "schema": f"{engine.SCHEMA_PREFIX}_development_checkpoint_binding_v1",
        "update": FINAL_UPDATE_V15,
        "model_module": type(model).__module__,
        "model_class": type(model).__name__,
        "model_config": asdict(config),
        "preregistration_commit": engine.PREREGISTRATION_COMMIT,
        "frozen_source_and_review_commit": authority[
            "frozen_source_and_review_commit"
        ],
        "recursive_source_closure_manifest_sha256": authority[
            "recursive_source_closure_manifest_sha256"
        ],
        "execution_binding_commit": authority["execution_binding_commit"],
        "authority_sha256": hashlib.sha256(
            engine._canonical_json_bytes(authority)
        ).hexdigest(),
        "state_manifest": state_manifest,
        "state_manifest_sha256": engine._canonical_value_sha256(state_manifest),
        "state_key_count": len(state),
        "online_parameter_counts": online_counts,
        "target_parameter_count": sum(
            parameter.numel()
            for module in model.target_modules()
            for parameter in module.parameters()
        ),
        "promotable_state": "update_2000_only",
        "development_pass_required": True,
        "probability_calibration_used": False,
    }
    payload = {
        "schema": f"{engine.SCHEMA_PREFIX}_development_checkpoint_v1",
        "update": FINAL_UPDATE_V15,
        "model_state_dict": state,
        "metadata": metadata,
        "probability_calibration_used": False,
    }
    stream = io.BytesIO()
    runtime.torch.save(payload, stream)
    raw = stream.getvalue()
    if not raw:
        raise RuntimeError("V15 development checkpoint serialization is empty")
    return raw, metadata


def _validate_lifecycle_configuration_v15(engine: Any) -> None:
    expected_paths = {
        "DEVELOPMENT_CHECKPOINT_RELATIVE_PATH": "checkpoints/update_2000.pt",
        "DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH": (
            "checkpoints/update_2000.binding.json"
        ),
    }
    if (
        engine.MAXIMUM_UPDATES != FINAL_UPDATE_V15
        or engine.MAXIMUM_PRESENTATIONS != 32_000
        or tuple(engine.OBSERVATION_UPDATES) != OBSERVATION_UPDATES_V15
        or tuple(engine.TERMINAL_UPDATES) != TERMINAL_UPDATES_V15
        or any(
            getattr(engine, name, None) != value
            for name, value in expected_paths.items()
        )
    ):
        raise PermissionError("V15 extended-horizon lifecycle configuration changed")


def _validate_terminal_inventory_v15(
    trace: list[dict[str, Any]],
    metric_bindings: list[dict[str, Any]],
    *,
    terminal_update: int,
    engine: Any,
) -> None:
    expected_trace_counts = {400: 402, 1_400: 1_403, 2_000: 2_004}
    expected_observations = {
        update for update in OBSERVATION_UPDATES_V15 if update <= terminal_update
    }
    observed_metric_paths = {
        binding.get("path") if isinstance(binding, Mapping) else None
        for binding in metric_bindings
    }
    expected_metric_paths = {
        engine.METRIC_RELATIVE_PATHS[update]
        for update in expected_observations
    }
    if (
        len(trace) != expected_trace_counts[terminal_update]
        or len(metric_bindings) != len(expected_observations)
        or observed_metric_paths != expected_metric_paths
    ):
        raise RuntimeError("V15 terminal trace or metric inventory changed")


def run_future_authorized_engine_v15(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
    engine: Any,
) -> dict[str, Any]:
    """Run the single continuous V15 trajectory after one-shot reservation."""

    stage = "validate_post_reservation_authority"
    trace: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    trace_binding: dict[str, Any] | None = None
    terminal_access_binding: dict[str, Any] | None = None
    terminal_access_content_sha256: str | None = None
    terminal_published = False
    validated_authority: dict[str, Any] | None = None

    def publish_trace() -> dict[str, Any]:
        nonlocal trace_binding
        if trace_binding is None:
            raw = b"".join(
                engine._canonical_json_bytes(engine._content_bound(row)) + b"\n"
                for row in trace
            )
            trace_binding = engine._publisher_bytes_v13(
                publisher, engine.TRACE_RELATIVE_PATH, raw
            )
        return trace_binding

    def publish_terminal_access(receipt: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal terminal_access_binding, terminal_access_content_sha256
        if terminal_access_binding is None:
            value, terminal_access_binding = engine._publisher_json_v13(
                publisher,
                engine.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH,
                {
                    "schema": f"{engine.SCHEMA_PREFIX}_terminal_access_receipt_v1",
                    "receipt": dict(receipt),
                },
            )
            terminal_access_content_sha256 = value["content_sha256"]
        return terminal_access_binding

    try:
        _validate_lifecycle_configuration_v15(engine)
        validated_authority = engine.validate_future_execution_prerequisites_v13(
            authority
        )
        validated_reservation = engine.validate_attempt_reservation_v13(
            reservation
        )
        if validated_reservation["authority_sha256"] != hashlib.sha256(
            engine._canonical_json_bytes(validated_authority)
        ).hexdigest():
            raise PermissionError("V15 reservation does not bind supplied authority")

        stage = "validate_deferred_runtime_and_schedule"
        schedule_receipt = engine.validate_schedule_v13(
            runtime.schedule,
            train_pair_count=int(runtime.train_pair_count),
        )
        stage = "initialize_n320_v15_model_optimizer"
        model, optimizer, initialization = runtime.initialize_model_v13()
        initialization_receipt = engine._validate_initialization_v13(
            runtime, model, initialization
        )
        initial_structural = engine._derive_initial_structural_integrity_v13(
            runtime, model
        )
        access = engine._validate_access_receipt_v13(runtime.access_receipt_v13())
        trace.append(
            {
                "schema": f"{engine.SCHEMA_PREFIX}_trace_row_v1",
                "event": "initialized",
                "update": 0,
                "initialization": initialization_receipt,
                "structural_integrity": initial_structural,
                "schedule": schedule_receipt,
                "access_receipt_sha256": engine._canonical_value_sha256(access),
            }
        )

        stage = "observe_update_0"
        observations: dict[int, dict[str, Any]] = {}
        observations[0] = engine._observation_v13(
            runtime,
            model,
            update=0,
            integrity_pass=bool(initial_structural["passed"]),
        )
        _, binding = engine._publisher_json_v13(
            publisher, engine.METRIC_RELATIVE_PATHS[0], observations[0]
        )
        metric_bindings.append(binding)

        accounting: Any = None
        structural_pass = bool(observations[0]["integrity_pass"])
        terminal_update: int | None = None
        scientific_decision: dict[str, Any] | None = None
        for update in range(1, FINAL_UPDATE_V15 + 1):
            stage = f"train_update_{update}"
            start = (update - 1) * engine.PRESENTATIONS_PER_UPDATE
            indices = list(
                runtime.schedule[start : start + engine.PRESENTATIONS_PER_UPDATE]
            )
            if len(indices) != engine.PRESENTATIONS_PER_UPDATE:
                raise PermissionError("V15 frozen repeated schedule ended early")
            microbatches = runtime.build_microbatches_v13(indices, update=update)
            engine._validate_microbatches_for_engine_v13(
                runtime, model, microbatches
            )
            result = runtime.training_module.joint_training_update_v13(
                model,
                optimizer,
                microbatches,
                accounting=accounting,
            )
            accounting = result.accounting
            integrity = engine._validate_update_integrity_v13(
                runtime,
                model,
                result,
                update=update,
                access_receipt=runtime.access_receipt_v13(),
            )
            structural_pass = structural_pass and bool(integrity["passed"])
            trace.append(
                {
                    "schema": f"{engine.SCHEMA_PREFIX}_trace_row_v1",
                    "event": "optimizer_ema_update",
                    **integrity,
                }
            )

            if update not in OBSERVATION_UPDATES_V15:
                continue
            stage = f"observe_update_{update}"
            observations[update] = engine._observation_v13(
                runtime,
                model,
                update=update,
                integrity_pass=structural_pass,
            )
            structural_pass = bool(observations[update]["integrity_pass"])
            _, binding = engine._publisher_json_v13(
                publisher,
                engine.METRIC_RELATIVE_PATHS[update],
                observations[update],
            )
            metric_bindings.append(binding)

            if update == 400:
                scientific_decision = engine.evaluate_update400_gate_v13(
                    observations[100]["physical"],
                    observations[400]["physical"],
                    observations[400]["controls"],
                    integrity_pass=structural_pass,
                    matched_update400_thresholds=(
                        engine.MATCHED_UPDATE400_THRESHOLDS
                    ),
                )
                trace.append(
                    {
                        "schema": f"{engine.SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update400_control",
                        "update": 400,
                        "decision": scientific_decision,
                    }
                )
                if not scientific_decision["passed"]:
                    terminal_update = 400
                    break
            elif update == 1_400:
                scientific_decision = evaluate_update1400_gate_v15(
                    observations[1_400]["v12_gate"],
                    observations[1_400]["physical"],
                    observations[1_400]["controls"],
                    integrity_pass=structural_pass,
                    engine=engine,
                )
                trace.append(
                    {
                        "schema": f"{engine.SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update1400_feasibility_gate",
                        "update": 1_400,
                        "decision": scientific_decision,
                    }
                )
                if not scientific_decision["passed"]:
                    terminal_update = 1_400
                    break
            elif update == FINAL_UPDATE_V15:
                scientific_decision = dict(
                    engine.evaluate_final_gate_v13(
                        observations[FINAL_UPDATE_V15]["v12_gate"],
                        observations[FINAL_UPDATE_V15]["physical"],
                        integrity_pass=structural_pass,
                    )
                )
                scientific_decision["update"] = FINAL_UPDATE_V15
                terminal_update = FINAL_UPDATE_V15
                trace.append(
                    {
                        "schema": f"{engine.SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update2000_final_gate",
                        "update": FINAL_UPDATE_V15,
                        "decision": scientific_decision,
                    }
                )

        if (
            terminal_update not in TERMINAL_UPDATES_V15
            or scientific_decision is None
        ):
            raise RuntimeError("V15 engine did not reach a preregistered terminal update")
        terminal_accounting = engine.validate_terminal_accounting_v13(
            accounting, terminal_update=terminal_update
        )
        _validate_terminal_inventory_v15(
            trace,
            metric_bindings,
            terminal_update=terminal_update,
            engine=engine,
        )
        terminal_access_reader = getattr(
            runtime, "terminal_access_receipt_v13", None
        )
        final_access = engine._validate_access_receipt_v13(
            (
                terminal_access_reader()
                if callable(terminal_access_reader)
                else runtime.access_receipt_v13()
            ),
            terminal=True,
        )
        if (
            final_access["runtime_data_root"]
            != validated_authority["runtime_data_root"]
        ):
            raise PermissionError(
                "V15 terminal rehash used a different authority-bound runtime data root"
            )
        if (
            final_access["source_root"]
            != validated_authority["certified_source_root"]
        ):
            raise PermissionError(
                "V15 terminal rehash used a different certified source root"
            )
        if final_access["runtime_fingerprint"] != validated_authority["runtime"]:
            raise PermissionError("V15 terminal rehash used a different runtime stack")
        final_access_artifact = publish_terminal_access(final_access)
        trace_record = publish_trace()

        if not scientific_decision["passed"]:
            stage = "publish_terminal_scientific_failure"
            if callable(getattr(runtime, "close_v13", None)):
                runtime.close_v13()
            failure_core = {
                "schema": f"{engine.SCHEMA_PREFIX}_scientific_failure_v1",
                "status": f"FAIL_SCIENTIFIC_UPDATE{terminal_update}_GATE_TERMINAL",
                "terminal_update": terminal_update,
                "decision": scientific_decision,
                "accounting": terminal_accounting,
                "metrics": metric_bindings,
                "trace": trace_record,
                "access_receipt_sha256": engine._canonical_value_sha256(final_access),
                "terminal_access_receipt": final_access_artifact,
                "terminal_access_receipt_content_sha256": (
                    terminal_access_content_sha256
                ),
                "checkpoint_published": False,
                "probability_calibration_opened": False,
                "attempt_consumed": True,
                "retry_authorized": False,
                "resume_authorized": False,
            }
            value, _ = engine._publisher_json_v13(
                publisher, engine.SCIENTIFIC_FAILURE_RELATIVE_PATH, failure_core
            )
            terminal_published = True
            return value

        stage = "publish_pass2000_development_checkpoint"
        checkpoint_raw, checkpoint_core = _serialize_development_checkpoint_v15(
            runtime, model, validated_authority, engine=engine
        )
        checkpoint_binding = engine._publisher_bytes_v13(
            publisher,
            engine.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH,
            checkpoint_raw,
        )
        checkpoint_core["checkpoint"] = checkpoint_binding
        checkpoint_value, checkpoint_metadata_binding = engine._publisher_json_v13(
            publisher,
            engine.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH,
            checkpoint_core,
        )
        if callable(getattr(runtime, "close_v13", None)):
            runtime.close_v13()
        stage = "publish_terminal_success"
        success_core = {
            "schema": f"{engine.SCHEMA_PREFIX}_success_v1",
            "status": "PASS_DEVELOPMENT_UPDATE2000_TERMINAL",
            "terminal_update": FINAL_UPDATE_V15,
            "decision": scientific_decision,
            "accounting": terminal_accounting,
            "metrics": metric_bindings,
            "trace": trace_record,
            "checkpoint": checkpoint_binding,
            "checkpoint_metadata": checkpoint_metadata_binding,
            "checkpoint_metadata_content_sha256": checkpoint_value[
                "content_sha256"
            ],
            "access_receipt_sha256": engine._canonical_value_sha256(final_access),
            "terminal_access_receipt": final_access_artifact,
            "terminal_access_receipt_content_sha256": (
                terminal_access_content_sha256
            ),
            "physical_adapter_preregistration_eligible": True,
            "probability_calibration_authorized": False,
            "probability_calibration_opened": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "held_out_authorized": False,
            "attempt_consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
        }
        value, _ = engine._publisher_json_v13(
            publisher, engine.SUCCESS_RELATIVE_PATH, success_core
        )
        terminal_published = True
        return value
    except BaseException as error:
        if terminal_published:
            raise
        try:
            if callable(getattr(runtime, "close_v13", None)):
                runtime.close_v13()
            terminal_reader = getattr(runtime, "terminal_access_receipt_v13", None)
            exception_access = engine._validate_access_receipt_v13(
                (
                    terminal_reader()
                    if callable(terminal_reader)
                    else runtime.access_receipt_v13()
                ),
                terminal=True,
            )
            if validated_authority is not None and (
                exception_access["runtime_data_root"]
                != validated_authority["runtime_data_root"]
                or exception_access["source_root"]
                != validated_authority["certified_source_root"]
                or exception_access["runtime_fingerprint"]
                != validated_authority["runtime"]
            ):
                raise PermissionError(
                    "V15 exception access receipt used an unbound source or data root"
                )
            exception_access_artifact = publish_terminal_access(exception_access)
            trace_record = publish_trace()
            failure_core = {
                "schema": f"{engine.SCHEMA_PREFIX}_exception_failure_v1",
                "status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME",
                "stage": stage,
                "exception_type": type(error).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "trace": trace_record,
                "access_receipt_sha256": engine._canonical_value_sha256(
                    exception_access
                ),
                "terminal_access_receipt": exception_access_artifact,
                "terminal_access_receipt_content_sha256": (
                    terminal_access_content_sha256
                ),
                "checkpoint_published": False,
                "probability_calibration_opened": False,
                "attempt_consumed": True,
                "retry_authorized": False,
                "resume_authorized": False,
            }
            value, _ = engine._publisher_json_v13(
                publisher, engine.SCIENTIFIC_FAILURE_RELATIVE_PATH, failure_core
            )
            return value
        except BaseException:
            raise error
