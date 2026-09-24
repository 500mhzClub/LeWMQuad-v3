from __future__ import annotations

import difflib
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
FROZEN_EXECUTOR = (
    ROOT
    / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "matched_no_persistence.py"
)
REPLACEMENT_EXECUTOR = (
    ROOT
    / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "matched_no_persistence_integrity_replacement.py"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_fixture(
    module: Any,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    accounting = {
        "updates": 1_000,
        "presentations": 16_000,
        "microbatch_graphs": 4_000,
        "backward_calls": 4_000,
        "optimizer_steps": 1_000,
        "ema_steps": 1_000,
        "predictor_forwards": 4_000,
        "predictor_objectives": 4_000,
    }
    components = dict(module.FIRST_UPDATE_COMPONENT_MEANS)
    losses = {
        **components,
        "L_full_diagnostic": sum(components.values()),
        "L_backward": sum(
            components[name] for name in ("S", "U", "R", "O")
        ),
    }
    trace = [
        {
            "update": update,
            "presentations": update * 16,
            "losses": dict(losses),
            "gradient_l2": {
                "encoder": 1.0,
                "lift_semantic": 1.0,
                "predictor": 1.0,
            },
        }
        for update in range(1, 1_001)
    ]
    diagnostics = {
        "gradient_groups": ["encoder", "lift_semantic", "predictor"],
        "first_update_component_witness": {
            "expected": components,
            "observed": components,
            "exact_match": True,
            "checked_after_backward_calls": 4,
            "checked_before_optimizer_step": True,
        },
    }
    return accounting, trace, diagnostics


def test_frozen_source_hashes_are_exact() -> None:
    expected = {
        "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "matched_no_persistence.py": (
            "90b66a5e4bdc7e6634db57d6852d9b3c5a187581d67a80ce81bf95fb371c34ab"
        ),
        "lewm/tests/test_run_go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "matched_no_persistence.py": (
            "1cb39173c8fa389abe38897ea0409b927ed7717deaa4516412d89ce0d405f647"
        ),
        "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "matched_no_persistence.py": (
            "f1e6a74c070d2db018cad120e4dcbc764f5432e4ebff1d88f179db079ad09cfd"
        ),
        "lewm/tests/test_execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "matched_no_persistence.py": (
            "684a52056fc45cbf6d04e0c9a1ff963e0add0048138acec4d38c9859809f5e69"
        ),
        "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v2_"
        "occupied_safety_aux.py": (
            "6f76dd5b098ff360a3ada5bbb18f74a13342f3a5212e871da6db8f5f3a5bb1bf"
        ),
        "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v3_"
        "half_occupied_safety_aux.py": (
            "7cab73752593b12b638b55710714ff956a2441e92df2fe775902472a7b69a8cb"
        ),
        "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v3_"
        "half_occupied_safety_aux.py": (
            "164e2baf53f2a882ef18eabeee99ae4b2c27a7d8d799543c798f24a49782b182"
        ),
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v4_"
        "residual_local_semantic_decoder.py": (
            "1c5a26f02a856d9a84903063c53bf23095142d86885787556b09388c508711ef"
        ),
        "lewm/tests/test_geometry_anchored_swept_progress_survival_joint_jepa_v4_"
        "residual_local_semantic_decoder.py": (
            "05e2783eeeffbe231b9e1128aae4695d5a6f695ea566ca64f0336bbf730763b2"
        ),
    }
    assert {
        relative: _sha256(ROOT / relative) for relative in expected
    } == expected


def test_replacement_science_bindings_equal_frozen_executor() -> None:
    frozen = _load("_test_integrity_frozen_executor", FROZEN_EXECUTOR)
    replacement = _load(
        "_test_integrity_replacement_executor", REPLACEMENT_EXECUTOR
    )
    for name in (
        "LABEL_ROOT_RELATIVE_PATH",
        "LABEL_MANIFEST_NAME",
        "LABEL_MANIFEST_CONTENT_SHA256",
        "LABEL_MANIFEST_FILE_SHA256",
        "LABEL_MANIFEST_BYTE_COUNT",
        "REQUIRED_GPU_NAME",
        "REQUIRED_GPU_MEMORY_BYTES",
        "ACTION_ORDER",
        "ROLE_FILES",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "CONSTRUCTOR_INITIALIZATION_SEED",
        "SEMANTIC_DECODER_INITIALIZATION_SEED",
        "EXPERIMENT_SEED",
        "BOOTSTRAP_SEED",
        "CONTROL_NAMES",
        "ALL_ARM_NAMES",
        "REGISTERED_FAMILIES",
        "GATE_THRESHOLDS",
        "PROGRESS_SEGMENT_M",
        "AUXILIARY_OBJECTIVE",
        "FULL_V4_SOURCE_COMMIT",
        "FULL_V4_TERMINAL_RECEIPT_COMMIT",
        "FULL_V4_RESULT_FILE_SHA256",
        "FULL_V4_RESULT_CONTENT_SHA256",
        "FULL_V4_TRACE_FILE_SHA256",
        "FULL_V4_TRACE_CONTENT_SHA256",
        "FULL_V4_REFERENCE_FAMILY_UTILITY_SHA256",
        "FULL_V4_REFERENCE_FAMILY_UTILITY",
        "BOOTSTRAP_REPLICATES",
        "BOOTSTRAP_LOWER_INDEX",
        "FIRST_UPDATE_COMPONENT_MEANS",
        "TRACE_LOSS_KEYS",
        "PERSISTENCE_TREATMENT",
    ):
        assert getattr(replacement, name) == getattr(frozen, name)
    for name in (
        "scientific_metrics_control",
        "semantic_metrics_control",
        "paired_control_comparison_control",
        "evaluate_gate_control",
    ):
        assert getattr(replacement, name) is getattr(frozen, name)
    assert replacement.persistence_treatment_receipt_v1() == (
        frozen.persistence_treatment_receipt_v1()
    )
    assert replacement.full_v4_reference_family_utility_receipt_v1() == (
        frozen.full_v4_reference_family_utility_receipt_v1()
    )


def test_operational_identity_and_integrity_receipt_are_exact_and_detached() -> None:
    module = _load("_test_integrity_receipt", REPLACEMENT_EXECUTOR)
    assert module.OUTPUT_RELATIVE_PATH == (
        ".generated/"
        "go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence/"
        "attempt_v2_integrity_replacement"
    )
    schemas = {
        module.CHECKPOINT_SCHEMA,
        module.TRACE_SCHEMA,
        module.RESULT_SCHEMA,
        module.FAILURE_SCHEMA,
    }
    assert len(schemas) == 4
    assert all("integrity_replacement" in schema for schema in schemas)
    receipt = module.integrity_replacement_receipt_v1()
    assert receipt["science_changed"] is False
    assert receipt["sole_integrity_delta"] == (
        "redundant_terminal_loss_identity_predicate_matches_frozen_core"
    )
    assert receipt["terminal_loss_identity_predicates"] == {
        "failed_attempt_executor": {
            "function": "absolute_error_lte",
            "rel_tol": None,
            "abs_tol": 1.0e-6,
        },
        "replacement_and_frozen_core": {
            "function": "math.isclose",
            "rel_tol": 2.0e-6,
            "abs_tol": 2.0e-6,
        },
        "identities": [
            "L_full_diagnostic=S+P_diagnostic+U+R+O",
            "L_backward=S+U+R+O",
        ],
    }
    assert receipt["replacement_preregistration_commit"] == (
        "d5c25a3b11181aba29a2c96e9954c09c19b8f1ad"
    )
    assert receipt["failed_attempt_source_commit"] == (
        "4d55f6b68ac4edfa8aef93fdb3b2e4c7666f09e2"
    )
    assert receipt["failed_attempt_failure_document_commit"] == (
        "8f6b187b52f8d7a47d33392e7ccaa242cb55e072"
    )
    assert receipt["failed_attempt_failure_file_sha256"] == (
        "b2a99cf0b88c918c80690620f5f9f7ee5c891fb60cde581eabe7118d3f89c6d8"
    )
    assert receipt["failed_attempt_failure_content_sha256"] == (
        "86ce444bba577a3744606480fb08803b67ced42e02b86cae5c22c88802d685b9"
    )
    assert receipt["expected_initial_state_digest"] == (
        "181b7cd4eef301a4986a9182940d0819b236ccf28876e471f5c30a62838112fd"
    )
    assert receipt["expected_empty_optimizer_digest"] == (
        "f45a9c253820a4bdab542e34ef07b8975bb799b7cdce2751ba781d905a386d2d"
    )
    assert receipt["failed_attempt_runtime_artifact_opened"] is False
    assert receipt["full_v4_runtime_artifact_reopened"] is False
    assert receipt["retry_resume_or_second_replacement_authorized"] is False
    receipt["science_changed"] = True
    assert module.INTEGRITY_REPLACEMENT["science_changed"] is False


def test_expected_initial_and_optimizer_receipts_fail_closed() -> None:
    module = _load("_test_integrity_initial_receipts", REPLACEMENT_EXECUTOR)
    module._validate_expected_initial_state_receipt_v1(
        {
            "canonical_state_entries_sha256": (
                "181b7cd4eef301a4986a9182940d0819b236ccf28876e471f5c30a62838112fd"
            )
        }
    )
    module._validate_expected_optimizer_receipt_v1(
        {
            "canonical_json_sha256": (
                "f45a9c253820a4bdab542e34ef07b8975bb799b7cdce2751ba781d905a386d2d"
            )
        }
    )
    with pytest.raises(RuntimeError, match="initial-state digest"):
        module._validate_expected_initial_state_receipt_v1(
            {"canonical_state_entries_sha256": "0" * 64}
        )
    with pytest.raises(RuntimeError, match="optimizer receipt digest"):
        module._validate_expected_optimizer_receipt_v1(
            {"canonical_json_sha256": "0" * 64}
        )


def test_exact_math_isclose_near_and_far_behavior() -> None:
    module = _load("_test_integrity_tolerance", REPLACEMENT_EXECUTOR)
    module._validate_terminal_loss_identity_v1(
        update=1,
        identity="absolute-near",
        observed=1.9e-6,
        expected=0.0,
    )
    module._validate_terminal_loss_identity_v1(
        update=1,
        identity="relative-near",
        observed=10.00001,
        expected=10.0,
    )
    with pytest.raises(module.TerminalLossIdentityMismatchV1) as caught:
        module._validate_terminal_loss_identity_v1(
            update=17,
            identity="absolute-far",
            observed=2.1e-6,
            expected=0.0,
        )
    error = caught.value
    assert error.update == 17
    assert error.identity == "absolute-far"
    assert error.observed == 2.1e-6
    assert error.expected == 0.0
    assert error.absolute_error == 2.1e-6
    assert module.terminal_loss_identity_failure_receipt_v1(error) == {
        "schema": "lewm_terminal_loss_identity_mismatch_v1",
        "update": 17,
        "identity": "absolute-far",
        "observed": 2.1e-6,
        "expected": 0.0,
        "absolute_error": 2.1e-6,
        "predicate": {
            "function": "math.isclose",
            "rel_tol": 2.0e-6,
            "abs_tol": 2.0e-6,
        },
    }


def test_terminal_validator_fixes_only_old_tolerance_and_checks_both_sums() -> None:
    frozen = _load("_test_integrity_old_terminal", FROZEN_EXECUTOR)
    module = _load("_test_integrity_new_terminal", REPLACEMENT_EXECUTOR)
    accounting, trace, diagnostics = _terminal_fixture(module)
    trace[499]["losses"]["L_full_diagnostic"] += 5.0e-6
    trace[499]["losses"]["L_backward"] += 5.0e-6
    module._validate_terminal_training_receipt_v1(
        accounting, trace, diagnostics
    )
    with pytest.raises(RuntimeError, match="loss identity"):
        frozen._validate_terminal_training_receipt_v1(
            accounting, trace, diagnostics
        )

    accounting, trace, diagnostics = _terminal_fixture(module)
    trace[616]["losses"]["L_full_diagnostic"] += 2.0e-5
    with pytest.raises(module.TerminalLossIdentityMismatchV1) as full:
        module._validate_terminal_training_receipt_v1(
            accounting, trace, diagnostics
        )
    assert full.value.update == 617
    assert full.value.identity == "L_full_diagnostic=S+P_diagnostic+U+R+O"

    accounting, trace, diagnostics = _terminal_fixture(module)
    trace[616]["losses"]["L_backward"] += 2.0e-5
    with pytest.raises(module.TerminalLossIdentityMismatchV1) as backward:
        module._validate_terminal_training_receipt_v1(
            accounting, trace, diagnostics
        )
    assert backward.value.update == 617
    assert backward.value.identity == "L_backward=S+U+R+O"


def test_structured_first_update_failure_receipt_uses_fields() -> None:
    module = _load("_test_integrity_first_update_failure", REPLACEMENT_EXECUTOR)
    error = SimpleNamespace(
        expected={"S": 1.0},
        observed={"S": 2.0},
        mismatch={"S": {"expected": 1.0, "observed": 2.0}},
        pre_step_operation_counts={
            "presentations_consumed": 16,
            "microbatch_graphs_completed": 4,
            "backward_calls_completed": 4,
            "optimizer_steps_completed": 0,
            "ema_steps_completed": 0,
            "predictor_forwards_completed": 4,
            "predictor_objectives_evaluated": 4,
        },
    )
    receipt = module.first_update_component_failure_receipt_v1(error)
    assert receipt == {
        "schema": "lewm_first_update_component_witness_mismatch_v1",
        "expected": {"S": 1.0},
        "observed": {"S": 2.0},
        "mismatch": {"S": {"expected": 1.0, "observed": 2.0}},
        "pre_step_operation_counts": {
            "presentations_consumed": 16,
            "microbatch_graphs_completed": 4,
            "backward_calls_completed": 4,
            "optimizer_steps_completed": 0,
            "ema_steps_completed": 0,
            "predictor_forwards_completed": 4,
            "predictor_objectives_evaluated": 4,
        },
    }
    module._v1._canonical_json_bytes(receipt)
    assert module.first_update_component_failure_receipt_v1(RuntimeError()) is None


def test_output_is_write_once_and_source_has_no_prior_artifact_access(
    tmp_path: Path,
) -> None:
    module = _load("_test_integrity_source", REPLACEMENT_EXECUTOR)
    output = module._fresh_output_root_v4_matched_no_persistence(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="integrity replacement"):
        module._fresh_output_root_v4_matched_no_persistence(tmp_path)

    source = REPLACEMENT_EXECUTOR.read_text()
    assert source.count(
        "training_control.run_fixed_training_v4_matched_no_persistence("
    ) == 1
    assert source.count(
        "model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4("
    ) == 2
    assert source.count('"integrity_replacement":') == 4
    for forbidden in (
        "attempt_v1",
        "torch.load(",
        "execute_v4(",
        "run_fixed_training_v3(",
        ".read_bytes(",
        ".read_text(",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v1/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v2_"
        "occupied_safety_aux/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v3_"
        "half_occupied_safety_aux/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v4_"
        "residual_local_semantic_decoder/",
    ):
        assert forbidden not in source

    first_model = source.index(
        "model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4("
    )
    second_model = source.index(
        "model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(",
        first_model + 1,
    )
    state_receipt = source.index(
        "_reconstructed_initialization_receipt_v1(", second_model
    )
    state_digest = source.index(
        "_validate_expected_initial_state_receipt_v1(", state_receipt
    )
    device_move = source.index('model = model.to(context["device"])', state_digest)
    optimizer = source.index("training_v1.build_frozen_optimizer_v1(", device_move)
    optimizer_receipt = source.index("_optimizer_receipt_v1(", optimizer)
    optimizer_digest = source.index(
        "_validate_expected_optimizer_receipt_v1(", optimizer_receipt
    )
    training = source.index(
        "training_control.run_fixed_training_v4_matched_no_persistence(",
        optimizer_digest,
    )
    checkpoint_write = source.index(
        'output / "checkpoint_update_1000.pt"', training
    )
    trace_write = source.index(
        'output / "training_trace.json"', checkpoint_write
    )
    evaluation = source.index("_v1.score_role_v1(", trace_write)
    assert first_model < second_model < state_receipt < state_digest < device_move
    assert device_move < optimizer < optimizer_receipt < optimizer_digest
    assert optimizer_digest < training < checkpoint_write < trace_write < evaluation
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"heldout_or_sealed_opened": False' in source
    assert '"retry_or_resume_authorized": False' in source
    assert '"replacement_or_warm_start_authorized": False' in source


def test_exact_source_delta_is_mechanically_frozen() -> None:
    frozen = FROZEN_EXECUTOR.read_text().splitlines()
    replacement = REPLACEMENT_EXECUTOR.read_text().splitlines()
    delta = "\n".join(
        difflib.unified_diff(
            frozen,
            replacement,
            fromfile="frozen_attempt_executor",
            tofile="integrity_replacement_executor",
            lineterm="",
        )
    ).encode("utf-8")
    assert hashlib.sha256(delta).hexdigest() == (
        "2fa62ea8a4b70077be6ae10e62c3e23612528a7d0b077ff30ada4a6802e8c261"
    )
