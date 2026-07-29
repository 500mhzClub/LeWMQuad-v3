#!/usr/bin/env python3
"""V16-only bounded lifecycle with full-state recovery milestones."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from scripts import (
    go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_recovery
    as _recovery,
)


EXTENSION_THRESHOLDS_V16 = {
    "inherited_check_count_minimum": 23,
    "passed_margin_count_minimum": 89,
    "total_shortfall_strictly_less_than": 41.41604892978589,
    "rough_depth_p95_m_strictly_less_than": 1.45,
    "rough_ground_balanced_accuracy_strictly_greater_than": 0.647134926562893,
    "rough_pixel_balanced_accuracy_strictly_greater_than": 0.8198594673963917,
}

UPDATE400_THRESHOLDS_V16 = {
    "passed_margin_count_minimum": 72,
    "total_shortfall_strictly_less_than": 68.96964862816927,
    "rough_depth_p95_m_strictly_less_than": 1.8582415819168085,
}


def _control_checks_v16(
    controls: Mapping[str, Mapping[str, bool]], engine: Any
) -> dict[str, bool]:
    if type(controls) is not dict or set(controls) != set(engine.CONTROL_NAMES):
        raise ValueError("V16 causal-control set changed")
    result: dict[str, bool] = {}
    for name in engine.CONTROL_NAMES:
        row = controls[name]
        if type(row) is not dict or set(row) != set(engine.CONTROL_CHECK_NAMES):
            raise ValueError(f"V16 causal-control schema changed: {name}")
        for check in engine.CONTROL_CHECK_NAMES:
            value = row[check]
            if type(value) is not bool:
                raise TypeError("V16 causal-control decisions must be Boolean")
            result[f"{name}:{check}"] = value
    return result


def evaluate_update400_gate_v16(
    physical_summary: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
    engine: Any,
) -> dict[str, Any]:
    """Apply exactly the frozen V16 update-400 continuation conjunction."""

    if type(integrity_pass) is not bool:
        raise TypeError("V16 structural-integrity decision must be Boolean")
    physical = engine._validate_physical_summary(physical_summary)
    control_checks = _control_checks_v16(controls, engine)
    rough = physical["rough_motion"]
    checks = {
        "structural_integrity_pass": integrity_pass,
        "all_twelve_causal_control_checks_true": all(control_checks.values()),
        "passed_physical_margin_count_at_least_72": (
            physical["passed_margin_count"]
            >= UPDATE400_THRESHOLDS_V16["passed_margin_count_minimum"]
        ),
        "total_physical_shortfall_strictly_below_v14_v15_update400": (
            physical["total_shortfall"]
            < UPDATE400_THRESHOLDS_V16[
                "total_shortfall_strictly_less_than"
            ]
        ),
        "rough_depth_p95_strictly_below_v14_v15_update400": (
            rough["depth_p95_m"]
            < UPDATE400_THRESHOLDS_V16[
                "rough_depth_p95_m_strictly_less_than"
            ]
        ),
    }
    passed = all(checks.values())
    return {
        "schema": f"{engine.SCHEMA_PREFIX}_update400_gate_v1",
        "update": 400,
        "checks": checks,
        "causal_control_checks": control_checks,
        "passed": passed,
        "observed": {
            "passed_physical_margin_count": physical["passed_margin_count"],
            "total_physical_shortfall": physical["total_shortfall"],
            "rough_depth_p95_m": rough["depth_p95_m"],
        },
        "thresholds": dict(UPDATE400_THRESHOLDS_V16),
        "action": (
            "CONTINUE_TO_UPDATE_1000"
            if passed
            else "FAIL_TERMINAL_NO_RETRY_NO_RESUME"
        ),
        "next_update": 1_000 if passed else None,
        "retry_authorized": False,
        "resume_authorized": False,
    }


def evaluate_extension_eligibility_v16(
    v12_gate: Mapping[str, Any],
    physical_summary: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
    engine: Any,
) -> dict[str, Any]:
    """Evaluate the fixed update-1,000 scale-only-if-promising rule."""

    if type(integrity_pass) is not bool:
        raise TypeError("V16 structural-integrity decision must be Boolean")
    if (
        not isinstance(v12_gate, Mapping)
        or type(v12_gate.get("checks")) is not dict
        or tuple(v12_gate["checks"]) != engine.V12_GATE_CHECK_NAMES
        or any(type(value) is not bool for value in v12_gate["checks"].values())
        or type(v12_gate.get("passed")) is not bool
        or v12_gate["passed"] != all(v12_gate["checks"].values())
    ):
        raise ValueError("V16 inherited 24-check gate changed")
    control_checks = _control_checks_v16(controls, engine)

    physical = engine._validate_physical_summary(physical_summary)
    rough = physical["rough_motion"]
    inherited_count = sum(v12_gate["checks"].values())
    checks = {
        "structural_integrity_pass": integrity_pass,
        "all_twelve_causal_control_checks_true": all(control_checks.values()),
        "inherited_v12_checks_at_least_23_of_24": (
            inherited_count
            >= EXTENSION_THRESHOLDS_V16["inherited_check_count_minimum"]
        ),
        "passed_physical_margin_count_at_least_89": (
            physical["passed_margin_count"]
            >= EXTENSION_THRESHOLDS_V16["passed_margin_count_minimum"]
        ),
        "total_physical_shortfall_strictly_below_v14_update1000": (
            physical["total_shortfall"]
            < EXTENSION_THRESHOLDS_V16["total_shortfall_strictly_less_than"]
        ),
        "rough_depth_p95_strictly_below_1_45_m": (
            rough["depth_p95_m"]
            < EXTENSION_THRESHOLDS_V16[
                "rough_depth_p95_m_strictly_less_than"
            ]
        ),
        "rough_ground_balanced_accuracy_above_final_floor": (
            rough["ground_balanced_accuracy"]
            > EXTENSION_THRESHOLDS_V16[
                "rough_ground_balanced_accuracy_strictly_greater_than"
            ]
        ),
        "rough_pixel_balanced_accuracy_above_final_floor": (
            rough["pixel_balanced_accuracy"]
            > EXTENSION_THRESHOLDS_V16[
                "rough_pixel_balanced_accuracy_strictly_greater_than"
            ]
        ),
    }
    passed = all(checks.values())
    return {
        "schema": f"{engine.SCHEMA_PREFIX}_extension_eligibility_v1",
        "update": 1_000,
        "checks": checks,
        "passed": passed,
        "observed": {
            "inherited_v12_passed_check_count": inherited_count,
            "passed_physical_margin_count": physical["passed_margin_count"],
            "total_physical_shortfall": physical["total_shortfall"],
            "rough_depth_p95_m": rough["depth_p95_m"],
            "rough_ground_balanced_accuracy": rough[
                "ground_balanced_accuracy"
            ],
            "rough_pixel_balanced_accuracy": rough[
                "pixel_balanced_accuracy"
            ],
        },
        "thresholds": dict(EXTENSION_THRESHOLDS_V16),
        "action": (
            "ELIGIBLE_FOR_SEPARATELY_AUTHORIZED_SCIENCE_IDENTICAL_CONTINUATION"
            if passed
            else "TERMINATE_V16_NO_CONTINUATION"
        ),
        "automatic_execution_authorized": False,
    }


def _publish_recovery_v16(
    *,
    runtime: Any,
    model: Any,
    optimizer: Any,
    accounting: Any,
    update: int,
    schedule_receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    trace: list[dict[str, Any]],
    metric_bindings: list[dict[str, Any]],
    publisher: Any,
    engine: Any,
) -> dict[str, Any]:
    consumed = getattr(getattr(runtime, "raw_inputs", None), "consumed", None)
    if type(consumed) is not dict or not consumed:
        raise PermissionError("V16 consumed-input ledger is absent")
    access = engine._validate_access_receipt_v13(runtime.access_receipt_v13())
    raw, binding = _recovery.serialize_recovery_checkpoint_v16(
        runtime.torch,
        model,
        optimizer,
        accounting,
        update=update,
        schedule=runtime.schedule,
        schedule_receipt=schedule_receipt,
        authority=authority,
        trace=trace,
        metric_bindings=metric_bindings,
        access_receipt=access,
        consumed_inputs=consumed,
    )

    def publish_bytes(relative_path: str, value: bytes) -> Mapping[str, Any]:
        return engine._publisher_bytes_v13(publisher, relative_path, value)

    result = _recovery.publish_recovery_checkpoint_v16(
        raw,
        binding,
        publish_bytes,
    )
    if result["value"]["update"] != update:
        raise RuntimeError("V16 recovery publisher returned another update")
    return result


def run_future_authorized_engine_v16(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
    engine: Any,
) -> dict[str, Any]:
    """Run one fresh 1,000-update V16 falsification after reservation."""

    stage = "validate_post_reservation_authority"
    trace: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    recovery_bindings: list[dict[str, Any]] = []
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
        if (
            engine.MAXIMUM_UPDATES != 1_000
            or engine.MAXIMUM_PRESENTATIONS != 16_000
            or tuple(engine.OBSERVATION_UPDATES) != (0, 100, 400, 1_000)
            or tuple(engine.TERMINAL_UPDATES) != (400, 1_000)
        ):
            raise PermissionError("V16 bounded lifecycle configuration changed")
        validated_authority = engine.validate_future_execution_prerequisites_v13(
            authority
        )
        validated_reservation = engine.validate_attempt_reservation_v13(reservation)
        if validated_reservation["authority_sha256"] != hashlib.sha256(
            engine._canonical_json_bytes(validated_authority)
        ).hexdigest():
            raise PermissionError("V16 reservation does not bind supplied authority")

        stage = "validate_deferred_runtime_and_schedule"
        schedule_receipt = engine.validate_schedule_v13(
            runtime.schedule,
            train_pair_count=int(runtime.train_pair_count),
        )
        stage = "initialize_n320_v16_model_optimizer"
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
        extension_decision: dict[str, Any] | None = None
        for update in range(1, 1_001):
            stage = f"train_update_{update}"
            start = (update - 1) * engine.PRESENTATIONS_PER_UPDATE
            indices = list(
                runtime.schedule[start : start + engine.PRESENTATIONS_PER_UPDATE]
            )
            if len(indices) != engine.PRESENTATIONS_PER_UPDATE:
                raise PermissionError("V16 frozen schedule ended early")
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

            if update not in (100, 400, 1_000):
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
                scientific_decision = evaluate_update400_gate_v16(
                    observations[400]["physical"],
                    observations[400]["controls"],
                    integrity_pass=structural_pass,
                    engine=engine,
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
                stage = "publish_recovery_update_400"
                recovery_bindings.append(
                    _publish_recovery_v16(
                        runtime=runtime,
                        model=model,
                        optimizer=optimizer,
                        accounting=accounting,
                        update=400,
                        schedule_receipt=schedule_receipt,
                        authority=validated_authority,
                        trace=trace,
                        metric_bindings=metric_bindings,
                        publisher=publisher,
                        engine=engine,
                    )
                )
            else:
                scientific_decision = engine.evaluate_final_gate_v13(
                    observations[1_000]["v12_gate"],
                    observations[1_000]["physical"],
                    integrity_pass=structural_pass,
                )
                extension_decision = evaluate_extension_eligibility_v16(
                    observations[1_000]["v12_gate"],
                    observations[1_000]["physical"],
                    observations[1_000]["controls"],
                    integrity_pass=structural_pass,
                    engine=engine,
                )
                terminal_update = 1_000
                trace.append(
                    {
                        "schema": f"{engine.SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update1000_final_gate",
                        "update": 1_000,
                        "decision": scientific_decision,
                        "extension_eligibility": extension_decision,
                    }
                )
                if scientific_decision["passed"] or extension_decision["passed"]:
                    stage = "publish_recovery_update_1000"
                    recovery_bindings.append(
                        _publish_recovery_v16(
                            runtime=runtime,
                            model=model,
                            optimizer=optimizer,
                            accounting=accounting,
                            update=1_000,
                            schedule_receipt=schedule_receipt,
                            authority=validated_authority,
                            trace=trace,
                            metric_bindings=metric_bindings,
                            publisher=publisher,
                            engine=engine,
                        )
                    )

        if terminal_update not in (400, 1_000) or scientific_decision is None:
            raise RuntimeError("V16 did not reach a preregistered terminal update")
        terminal_accounting = engine.validate_terminal_accounting_v13(
            accounting, terminal_update=terminal_update
        )
        expected_trace_count = 402 if terminal_update == 400 else 1_003
        if len(trace) != expected_trace_count:
            raise RuntimeError("V16 terminal trace inventory changed")
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
            or final_access["source_root"]
            != validated_authority["certified_source_root"]
            or final_access["runtime_fingerprint"]
            != validated_authority["runtime"]
        ):
            raise PermissionError("V16 terminal rehash used an unbound runtime")
        final_access_artifact = publish_terminal_access(final_access)
        trace_record = publish_trace()

        if not scientific_decision["passed"]:
            stage = "publish_terminal_scientific_failure"
            if callable(getattr(runtime, "close_v13", None)):
                runtime.close_v13()
            continuation_eligible = bool(
                terminal_update == 1_000
                and extension_decision is not None
                and extension_decision["passed"]
            )
            failure_core = {
                "schema": f"{engine.SCHEMA_PREFIX}_scientific_failure_v1",
                "status": (
                    "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"
                    if terminal_update == 400
                    else (
                        "FAIL_SCIENTIFIC_UPDATE1000_GATE_CONTINUATION_ELIGIBLE"
                        if continuation_eligible
                        else "FAIL_SCIENTIFIC_UPDATE1000_GATE_TERMINAL"
                    )
                ),
                "terminal_update": terminal_update,
                "decision": scientific_decision,
                "extension_eligibility": extension_decision,
                "accounting": terminal_accounting,
                "metrics": metric_bindings,
                "trace": trace_record,
                "recovery_checkpoints": recovery_bindings,
                "science_identical_continuation_eligible": continuation_eligible,
                "continuation_requires_separate_exact_authority": True,
                "access_receipt_sha256": engine._canonical_value_sha256(
                    final_access
                ),
                "terminal_access_receipt": final_access_artifact,
                "terminal_access_receipt_content_sha256": (
                    terminal_access_content_sha256
                ),
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

        stage = "publish_qualified_update1000_model"
        checkpoint_raw, checkpoint_core = engine._serialize_development_checkpoint_v13(
            runtime, model, validated_authority
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
            "status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL",
            "terminal_update": 1_000,
            "decision": scientific_decision,
            "extension_eligibility": extension_decision,
            "accounting": terminal_accounting,
            "metrics": metric_bindings,
            "trace": trace_record,
            "recovery_checkpoints": recovery_bindings,
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
                    "V16 exception receipt used an unbound source or data root"
                )
            exception_access_artifact = publish_terminal_access(exception_access)
            trace_record = publish_trace()
            failure_core = {
                "schema": f"{engine.SCHEMA_PREFIX}_exception_failure_v1",
                "status": "FAIL_EXCEPTION_TERMINAL_REVIEW_RECOVERY_MILESTONE",
                "stage": stage,
                "exception_type": type(error).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "trace": trace_record,
                "recovery_checkpoints": recovery_bindings,
                "recovery_requires_separate_exact_authority_and_review": True,
                "access_receipt_sha256": engine._canonical_value_sha256(
                    exception_access
                ),
                "terminal_access_receipt": exception_access_artifact,
                "terminal_access_receipt_content_sha256": (
                    terminal_access_content_sha256
                ),
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


__all__ = [
    "EXTENSION_THRESHOLDS_V16",
    "UPDATE400_THRESHOLDS_V16",
    "evaluate_extension_eligibility_v16",
    "evaluate_update400_gate_v16",
    "run_future_authorized_engine_v16",
]
