#!/usr/bin/env python3
"""Run the one-shot matched no-persistence integrity replacement.

This is a science-identical replacement for a terminal receipt-adapter
failure.  The frozen control science is copied unchanged; only the redundant
terminal loss-identity predicate and fresh operational identity differ.  No
prior experiment runtime artifact is named, opened, loaded, or reused.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_v4 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "residual_local_semantic_decoder"
)
_v3 = _v4._v3
_v1 = _v4._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/"
    "go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence/"
    "attempt_v2_integrity_replacement"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_"
    "persistence_integrity_replacement_checkpoint_v1"
)
TRACE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_"
    "persistence_integrity_replacement_trace_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_"
    "persistence_integrity_replacement_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_"
    "persistence_integrity_replacement_failure_v1"
)

LABEL_ROOT_RELATIVE_PATH = _v4.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v4.LABEL_MANIFEST_NAME
LABEL_MANIFEST_CONTENT_SHA256 = _v4.LABEL_MANIFEST_CONTENT_SHA256
LABEL_MANIFEST_FILE_SHA256 = _v4.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v4.LABEL_MANIFEST_BYTE_COUNT
REQUIRED_GPU_NAME = _v4.REQUIRED_GPU_NAME
REQUIRED_GPU_MEMORY_BYTES = _v4.REQUIRED_GPU_MEMORY_BYTES
ACTION_ORDER = _v4.ACTION_ORDER
ROLE_FILES = _v4.ROLE_FILES
MICROBATCH_SIZE = _v4.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v4.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v4.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v4.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v4.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v4.CONSTRUCTOR_INITIALIZATION_SEED
SEMANTIC_DECODER_INITIALIZATION_SEED = _v4.SEMANTIC_DECODER_INITIALIZATION_SEED
EXPERIMENT_SEED = _v4.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v4.BOOTSTRAP_SEED
CONTROL_NAMES = _v4.CONTROL_NAMES
ALL_ARM_NAMES = _v4.ALL_ARM_NAMES
REGISTERED_FAMILIES = _v4.REGISTERED_FAMILIES
GATE_THRESHOLDS = _v4.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v4.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v4.AUXILIARY_OBJECTIVE)

PREREGISTRATION_COMMIT = "d5c25a3b11181aba29a2c96e9954c09c19b8f1ad"
ORIGINAL_PREREGISTRATION_COMMIT = (
    "3dd4ca0680347f0a7f35d42d387781ecf53b1685"
)
PRE_RUNTIME_CLARIFICATION_COMMIT = (
    "8cd4486ff8fc5e82dbfb745da1ed8d4b3a4101b1"
)
FAILED_ATTEMPT_SOURCE_COMMIT = "4d55f6b68ac4edfa8aef93fdb3b2e4c7666f09e2"
FAILED_ATTEMPT_EXECUTION_BINDING_COMMIT = (
    "49d281480db196187b20c34f4cb5a61beede264a"
)
FAILED_ATTEMPT_FAILURE_DOCUMENT_COMMIT = (
    "8f6b187b52f8d7a47d33392e7ccaa242cb55e072"
)
FAILED_ATTEMPT_FAILURE_FILE_SHA256 = (
    "b2a99cf0b88c918c80690620f5f9f7ee5c891fb60cde581eabe7118d3f89c6d8"
)
FAILED_ATTEMPT_FAILURE_CONTENT_SHA256 = (
    "86ce444bba577a3744606480fb08803b67ced42e02b86cae5c22c88802d685b9"
)
FAILED_ATTEMPT_EXECUTOR_SHA256 = (
    "f1e6a74c070d2db018cad120e4dcbc764f5432e4ebff1d88f179db079ad09cfd"
)
FAILED_ATTEMPT_EXECUTOR_TEST_SHA256 = (
    "684a52056fc45cbf6d04e0c9a1ff963e0add0048138acec4d38c9859809f5e69"
)
NO_PERSISTENCE_TRAINING_CORE_SHA256 = (
    "90b66a5e4bdc7e6634db57d6852d9b3c5a187581d67a80ce81bf95fb371c34ab"
)
NO_PERSISTENCE_TRAINING_CORE_TEST_SHA256 = (
    "1cb39173c8fa389abe38897ea0409b927ed7717deaa4516412d89ce0d405f647"
)
EXPECTED_INITIAL_STATE_DIGEST = (
    "181b7cd4eef301a4986a9182940d0819b236ccf28876e471f5c30a62838112fd"
)
EXPECTED_EMPTY_OPTIMIZER_DIGEST = (
    "f45a9c253820a4bdab542e34ef07b8975bb799b7cdce2751ba781d905a386d2d"
)
TERMINAL_LOSS_IDENTITY_REL_TOL = 2.0e-6
TERMINAL_LOSS_IDENTITY_ABS_TOL = 2.0e-6
FULL_V4_SOURCE_COMMIT = "aaa47a138d0eeb78aa20d9524e67f813f7a74a41"
FULL_V4_TERMINAL_RECEIPT_COMMIT = "8b3a8063b087c81030189deadc6c5f6e1c7d44c3"
FULL_V4_RESULT_FILE_SHA256 = (
    "bf93c96cf020553be74d51847c6876e345cd6cc391b05cec186e36b20ca15aa4"
)
FULL_V4_RESULT_CONTENT_SHA256 = (
    "27ecf4895dfea01a1e5bb4f6f13f3add6a182a8dfa4b9f8651204bd1e6222ad8"
)
FULL_V4_TRACE_FILE_SHA256 = (
    "2ad16afd722ada26439c4ebfb2993330ec3abe1cbe4a75ced496a7c2a2b1580b"
)
FULL_V4_TRACE_CONTENT_SHA256 = (
    "bb027f8af94f352aac3ca1291a84285e25df431ca90682660afc7e1b476d4c12"
)
FULL_V4_REFERENCE_FAMILY_UTILITY_SHA256 = (
    "8ba8d6126e922f6a36038304e3444d0d21ee69350fef4acd3828265754810e1e"
)
FULL_V4_REFERENCE_FAMILY_UTILITY = {
    "schema": "lewm_v4_full_reference_family_utility_v1",
    "family_order": list(REGISTERED_FAMILIES),
    "normalized_chosen_prefix_utility": [
        0.8896189747752248,
        0.9384050589932943,
        0.8938629676334595,
        0.8772593292124542,
        0.8934829059829059,
        0.9430145611963794,
        0.922340425531915,
        0.9229020111832612,
    ],
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_INDEX = 249
FIRST_UPDATE_COMPONENT_MEANS = {
    "S": 1.313827022910118,
    "P_diagnostic": 1.0,
    "U": 0.9792981296777725,
    "R": 1.0,
    "O": 1.026371382176876,
}
TRACE_LOSS_KEYS = (
    "S",
    "P_diagnostic",
    "U",
    "R",
    "O",
    "L_full_diagnostic",
    "L_backward",
)
PERSISTENCE_TREATMENT = {
    "schema": "lewm_v4_matched_no_persistence_backward_membership_v1",
    "full_v4_backward_scalar": "S + P + U + R + O",
    "control_backward_scalar": "S + U + R + O",
    "persistence_diagnostic_computed": True,
    "persistence_backward_coefficient": 0.0,
    "persistence_detached": False,
    "sole_treatment_delta": "P_absent_from_backward_membership",
}
INTEGRITY_REPLACEMENT = {
    "schema": (
        "lewm_v4_matched_no_persistence_integrity_replacement_binding_v1"
    ),
    "science_changed": False,
    "sole_integrity_delta": (
        "redundant_terminal_loss_identity_predicate_matches_frozen_core"
    ),
    "terminal_loss_identity_predicates": {
        "failed_attempt_executor": {
            "function": "absolute_error_lte",
            "rel_tol": None,
            "abs_tol": 1.0e-6,
        },
        "replacement_and_frozen_core": {
            "function": "math.isclose",
            "rel_tol": TERMINAL_LOSS_IDENTITY_REL_TOL,
            "abs_tol": TERMINAL_LOSS_IDENTITY_ABS_TOL,
        },
        "identities": [
            "L_full_diagnostic=S+P_diagnostic+U+R+O",
            "L_backward=S+U+R+O",
        ],
    },
    "replacement_preregistration_commit": PREREGISTRATION_COMMIT,
    "original_preregistration_commit": ORIGINAL_PREREGISTRATION_COMMIT,
    "pre_runtime_clarification_commit": PRE_RUNTIME_CLARIFICATION_COMMIT,
    "failed_attempt_source_commit": FAILED_ATTEMPT_SOURCE_COMMIT,
    "failed_attempt_execution_binding_commit": (
        FAILED_ATTEMPT_EXECUTION_BINDING_COMMIT
    ),
    "failed_attempt_failure_document_commit": (
        FAILED_ATTEMPT_FAILURE_DOCUMENT_COMMIT
    ),
    "failed_attempt_failure_file_sha256": FAILED_ATTEMPT_FAILURE_FILE_SHA256,
    "failed_attempt_failure_content_sha256": (
        FAILED_ATTEMPT_FAILURE_CONTENT_SHA256
    ),
    "failed_attempt_executor_sha256": FAILED_ATTEMPT_EXECUTOR_SHA256,
    "failed_attempt_executor_test_sha256": FAILED_ATTEMPT_EXECUTOR_TEST_SHA256,
    "no_persistence_training_core_sha256": NO_PERSISTENCE_TRAINING_CORE_SHA256,
    "no_persistence_training_core_test_sha256": (
        NO_PERSISTENCE_TRAINING_CORE_TEST_SHA256
    ),
    "expected_initial_state_digest": EXPECTED_INITIAL_STATE_DIGEST,
    "expected_empty_optimizer_digest": EXPECTED_EMPTY_OPTIMIZER_DIGEST,
    "failed_attempt_runtime_artifact_opened": False,
    "full_v4_runtime_artifact_reopened": False,
    "retry_resume_or_second_replacement_authorized": False,
}

scientific_metrics_control = _v4.scientific_metrics_v4
semantic_metrics_control = _v4.semantic_metrics_v4
paired_control_comparison_control = _v4.paired_control_comparison_v4
evaluate_gate_control = _v4.evaluate_gate_v4


def persistence_treatment_receipt_v1() -> dict[str, Any]:
    return copy.deepcopy(PERSISTENCE_TREATMENT)


def integrity_replacement_receipt_v1() -> dict[str, Any]:
    return copy.deepcopy(INTEGRITY_REPLACEMENT)


def full_v4_reference_family_utility_receipt_v1() -> dict[str, Any]:
    payload = copy.deepcopy(FULL_V4_REFERENCE_FAMILY_UTILITY)
    observed = _v1._canonical_json_sha256(payload)
    if observed != FULL_V4_REFERENCE_FAMILY_UTILITY_SHA256:
        raise RuntimeError("frozen full-V4 family reference digest changed")
    return {
        "payload": payload,
        "canonical_json_sha256": observed,
        "source_commit": FULL_V4_SOURCE_COMMIT,
        "terminal_receipt_commit": FULL_V4_TERMINAL_RECEIPT_COMMIT,
        "reviewed_result_file_sha256": FULL_V4_RESULT_FILE_SHA256,
        "reviewed_result_content_sha256": FULL_V4_RESULT_CONTENT_SHA256,
        "reviewed_training_trace_file_sha256": FULL_V4_TRACE_FILE_SHA256,
        "reviewed_training_trace_content_sha256": FULL_V4_TRACE_CONTENT_SHA256,
        "runtime_artifact_reopened": False,
    }


def _fresh_output_root_v4_matched_no_persistence(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "fresh matched-no-persistence integrity replacement already exists"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _tensor_bytes_v1(tensor: Any, *, torch: Any) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    return value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _canonical_tensor_state_receipt_v1(model: Any, *, torch: Any) -> Mapping[str, Any]:
    entries = []
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": [int(value) for value in tensor.shape],
                "tensor_byte_sha256": hashlib.sha256(
                    _tensor_bytes_v1(tensor, torch=torch)
                ).hexdigest(),
            }
        )
    return {
        "schema": "lewm_v4_canonical_initial_tensor_state_entries_v1",
        "entries": entries,
        "canonical_json_sha256": _v1._canonical_json_sha256(entries),
    }


def _reconstructed_initialization_receipt_v1(
    control_model: Any,
    witness_model: Any,
    *,
    torch: Any,
) -> Mapping[str, Any]:
    control = _canonical_tensor_state_receipt_v1(control_model, torch=torch)
    witness = _canonical_tensor_state_receipt_v1(witness_model, torch=torch)
    payloads_equal = control["entries"] == witness["entries"]
    digests_equal = (
        control["canonical_json_sha256"] == witness["canonical_json_sha256"]
    )
    counters = []
    for ordinal, model in enumerate((control_model, witness_model), start=1):
        hard_sync = int(model.target_hard_sync_count.item())
        ema = int(model.ema_update_count.item())
        if hard_sync != 1 or ema != 0:
            raise RuntimeError("fresh V4 target/EMA counters changed")
        counters.append(
            {
                "reconstruction": ordinal,
                "target_hard_sync_count": hard_sync,
                "ema_update_count": ema,
            }
        )
    if not payloads_equal or not digests_equal:
        raise RuntimeError("two fresh V4 reconstructions are not tensor-identical")
    return {
        "schema": "lewm_v4_reconstructed_initialization_witness_v1",
        "initialization_source": "exact_n320_encoder_only",
        "reconstruction_count": 2,
        "selected_control_reconstruction": 1,
        "state_entry_count": len(control["entries"]),
        "canonical_state_entries": control["entries"],
        "canonical_state_entries_sha256": control["canonical_json_sha256"],
        "reconstruction_digests": [
            control["canonical_json_sha256"],
            witness["canonical_json_sha256"],
        ],
        "payloads_equal": payloads_equal,
        "digests_equal": digests_equal,
        "counters": counters,
    }


def _optimizer_receipt_v1(
    optimizer: Any,
    partition: Any,
) -> Mapping[str, Any]:
    if optimizer.state:
        raise RuntimeError("fresh optimizer state is not empty")
    groups = []
    for group in optimizer.param_groups:
        name = str(group.get("name"))
        if name not in ("encoder", "lift_semantic", "predictor"):
            raise RuntimeError("optimizer parameter-group name changed")
        parameters = tuple(group["params"])
        parameter_names = tuple(partition.names[name])
        expected_parameters = tuple(getattr(partition, name))
        if (
            len(parameter_names) != len(parameters)
            or tuple(map(id, parameters)) != tuple(map(id, expected_parameters))
        ):
            raise RuntimeError("optimizer parameter inventory changed")
        inventory = [
            {
                "name": parameter_name,
                "dtype": str(parameter.dtype),
                "shape": [int(value) for value in parameter.shape],
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
            }
            for parameter_name, parameter in zip(
                parameter_names, parameters, strict=True
            )
        ]
        groups.append(
            {
                "name": name,
                "parameters": inventory,
                "parameter_count": len(inventory),
                "trainable_scalar_count": sum(row["numel"] for row in inventory),
                "hyperparameters": {
                    "lr": float(group["lr"]),
                    "betas": [float(value) for value in group["betas"]],
                    "eps": float(group["eps"]),
                    "weight_decay": float(group["weight_decay"]),
                    "amsgrad": bool(group["amsgrad"]),
                    "maximize": bool(group["maximize"]),
                    "foreach": group["foreach"],
                    "capturable": bool(group["capturable"]),
                    "differentiable": bool(group["differentiable"]),
                    "fused": group["fused"],
                },
            }
        )
    payload = {
        "schema": "lewm_v4_matched_no_persistence_optimizer_inventory_v1",
        "optimizer_type": optimizer.__class__.__name__,
        "state_entry_count": 0,
        "state_empty": True,
        "parameter_groups": groups,
        "target_parameter_names": list(partition.names["target"]),
        "target_excluded_from_optimizer": True,
        "all_online_parameters_in_exactly_one_group": True,
    }
    return {
        **payload,
        "canonical_json_sha256": _v1._canonical_json_sha256(payload),
    }


def _validate_expected_initial_state_receipt_v1(
    receipt: Mapping[str, Any],
) -> None:
    if (
        receipt.get("canonical_state_entries_sha256")
        != EXPECTED_INITIAL_STATE_DIGEST
    ):
        raise RuntimeError("fresh reconstructed initial-state digest changed")


def _validate_expected_optimizer_receipt_v1(
    receipt: Mapping[str, Any],
) -> None:
    if receipt.get("canonical_json_sha256") != EXPECTED_EMPTY_OPTIMIZER_DIGEST:
        raise RuntimeError("fresh empty optimizer receipt digest changed")


def _validate_training_core_v1(
    training_v1: Any,
    training_v3: Any,
    training_control: Any,
) -> None:
    _v4._validate_training_core_v4(training_v1, training_v3)
    for name in (
        "ACTION_ORDER",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
    ):
        if getattr(training_control, name, None) != getattr(training_v1, name):
            raise PermissionError(f"matched control changed frozen {name}")
    if not callable(
        getattr(
            training_control,
            "run_fixed_training_v4_matched_no_persistence",
            None,
        )
    ):
        raise PermissionError("matched no-persistence training entrypoint is absent")
    if dict(
        getattr(training_control, "FIRST_UPDATE_COMPONENT_MEANS", {})
    ) != FIRST_UPDATE_COMPONENT_MEANS:
        raise PermissionError("matched control first-update witness changed")


class TerminalLossIdentityMismatchV1(RuntimeError):
    """Structured evidence from the redundant terminal trace validator."""

    def __init__(
        self,
        *,
        update: int,
        identity: str,
        observed: float,
        expected: float,
    ) -> None:
        self.update = int(update)
        self.identity = str(identity)
        self.observed = float(observed)
        self.expected = float(expected)
        self.absolute_error = abs(self.observed - self.expected)
        self.rel_tol = TERMINAL_LOSS_IDENTITY_REL_TOL
        self.abs_tol = TERMINAL_LOSS_IDENTITY_ABS_TOL
        super().__init__(
            f"terminal loss identity {self.identity!r} changed at update "
            f"{self.update}: observed={self.observed!r}, "
            f"expected={self.expected!r}, "
            f"absolute_error={self.absolute_error!r}"
        )


def _validate_terminal_loss_identity_v1(
    *,
    update: int,
    identity: str,
    observed: float,
    expected: float,
) -> None:
    if not math.isclose(
        observed,
        expected,
        rel_tol=TERMINAL_LOSS_IDENTITY_REL_TOL,
        abs_tol=TERMINAL_LOSS_IDENTITY_ABS_TOL,
    ):
        raise TerminalLossIdentityMismatchV1(
            update=update,
            identity=identity,
            observed=observed,
            expected=expected,
        )


def terminal_loss_identity_failure_receipt_v1(
    error: BaseException,
) -> Mapping[str, Any] | None:
    if not isinstance(error, TerminalLossIdentityMismatchV1):
        return None
    return {
        "schema": "lewm_terminal_loss_identity_mismatch_v1",
        "update": error.update,
        "identity": error.identity,
        "observed": error.observed,
        "expected": error.expected,
        "absolute_error": error.absolute_error,
        "predicate": {
            "function": "math.isclose",
            "rel_tol": error.rel_tol,
            "abs_tol": error.abs_tol,
        },
    }


def first_update_component_failure_receipt_v1(
    error: BaseException,
) -> Mapping[str, Any] | None:
    required = (
        "expected",
        "observed",
        "mismatch",
        "pre_step_operation_counts",
    )
    if not all(hasattr(error, name) for name in required):
        return None
    return {
        "schema": "lewm_first_update_component_witness_mismatch_v1",
        "expected": dict(error.expected),
        "observed": dict(error.observed),
        "mismatch": {
            name: dict(fields) for name, fields in error.mismatch.items()
        },
        "pre_step_operation_counts": dict(error.pre_step_operation_counts),
    }


def _validate_terminal_training_receipt_v1(
    accounting: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> None:
    expected_accounting = {
        "updates": 1_000,
        "presentations": 16_000,
        "microbatch_graphs": 4_000,
        "backward_calls": 4_000,
        "optimizer_steps": 1_000,
        "ema_steps": 1_000,
        "predictor_forwards": 4_000,
        "predictor_objectives": 4_000,
    }
    if dict(accounting) != expected_accounting:
        raise RuntimeError("matched control terminal accounting changed")
    if len(trace) != MAXIMUM_UPDATES:
        raise RuntimeError("matched control trace length changed")
    for index, row in enumerate(trace, start=1):
        if (
            row.get("update") != index
            or row.get("presentations") != index * PRESENTATIONS_PER_UPDATE
            or tuple(row.get("losses", {})) != TRACE_LOSS_KEYS
            or set(row.get("gradient_l2", {}))
            != {"encoder", "lift_semantic", "predictor"}
        ):
            raise RuntimeError("matched control trace schema or order changed")
        losses = row["losses"]
        if not all(math.isfinite(float(value)) for value in losses.values()):
            raise FloatingPointError("matched control trace contains nonfinite loss")
        full_observed = float(losses["L_full_diagnostic"])
        full_expected = sum(
            float(losses[name])
            for name in ("S", "P_diagnostic", "U", "R", "O")
        )
        _validate_terminal_loss_identity_v1(
            update=index,
            identity="L_full_diagnostic=S+P_diagnostic+U+R+O",
            observed=full_observed,
            expected=full_expected,
        )
        backward_observed = float(losses["L_backward"])
        backward_expected = sum(
            float(losses[name]) for name in ("S", "U", "R", "O")
        )
        _validate_terminal_loss_identity_v1(
            update=index,
            identity="L_backward=S+U+R+O",
            observed=backward_observed,
            expected=backward_expected,
        )
    if tuple(diagnostics.get("gradient_groups", ())) != (
        "encoder",
        "lift_semantic",
        "predictor",
    ):
        raise RuntimeError("matched control diagnostic gradient groups changed")
    witness = diagnostics.get("first_update_component_witness")
    if not isinstance(witness, Mapping) or (
        dict(witness.get("expected", {})) != FIRST_UPDATE_COMPONENT_MEANS
        or dict(witness.get("observed", {})) != FIRST_UPDATE_COMPONENT_MEANS
        or witness.get("exact_match") is not True
        or witness.get("checked_after_backward_calls") != 4
        or witness.get("checked_before_optimizer_step") is not True
    ):
        raise RuntimeError("matched control first-update component witness changed")


def v4_minus_control_treatment_comparison_v1(
    control_selection_metrics: Mapping[str, Any],
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    np: Any,
) -> Mapping[str, Any]:
    reference_receipt = full_v4_reference_family_utility_receipt_v1()
    reference = reference_receipt["payload"]
    family_order = tuple(reference["family_order"])
    if family_order != tuple(REGISTERED_FAMILIES):
        raise RuntimeError("frozen treatment family order changed")
    observed_families = control_selection_metrics.get("families", {})
    if set(observed_families) != set(family_order):
        raise RuntimeError("control selection family inventory changed")
    control_values = np.asarray(
        [
            observed_families[family]["normalized_chosen_prefix_utility"]
            for family in family_order
        ],
        dtype=np.float64,
    )
    reference_values = np.asarray(
        reference["normalized_chosen_prefix_utility"], dtype=np.float64
    )
    if (
        control_values.shape != (8,)
        or reference_values.shape != (8,)
        or not bool(np.isfinite(control_values).all())
        or not bool(np.isfinite(reference_values).all())
    ):
        raise FloatingPointError("treatment family utility vector is incomplete")
    deltas = reference_values - control_values
    if not bool(np.isfinite(deltas).all()):
        raise FloatingPointError("treatment family delta vector is nonfinite")

    scene_families: dict[str, set[str]] = {}
    for scene, family in zip(scene_ids, family_ids, strict=True):
        scene_families.setdefault(str(scene), set()).add(str(family))
    if len(scene_families) != 8 or any(
        len(families) != 1 for families in scene_families.values()
    ):
        raise RuntimeError("treatment comparison requires eight one-family scenes")
    family_to_scene = {
        next(iter(families)): scene for scene, families in scene_families.items()
    }
    if set(family_to_scene) != set(family_order):
        raise RuntimeError("selection scene/family matching changed")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 8, size=(BOOTSTRAP_REPLICATES, 8))
    replicates = np.sort(deltas[draws].mean(axis=1))
    mean_delta = float(deltas.mean())
    lower = float(replicates[BOOTSTRAP_LOWER_INDEX])
    positive_family_count = int((deltas > 0.0).sum())
    checks = {
        "strictly_positive_equal_scene_mean": mean_delta > 0.0,
        "strictly_positive_bootstrap_lower_95": lower > 0.0,
        "at_least_six_positive_families": positive_family_count >= 6,
    }
    passed = all(checks.values())
    per_family = dict(zip(family_order, map(float, deltas), strict=True))
    scene_order = [family_to_scene[family] for family in family_order]
    return {
        "schema": "lewm_v4_minus_matched_no_persistence_treatment_predicate_v1",
        "valid": True,
        "passed": passed,
        "status": (
            "POSITIVE_PERSISTENCE_TREATMENT"
            if passed
            else "NEGATIVE_PERSISTENCE_TREATMENT"
        ),
        "checks": checks,
        "full_v4_reference": reference_receipt,
        "family_order": list(family_order),
        "scene_order_aligned_to_family_order": scene_order,
        "control_normalized_chosen_prefix_utility": list(map(float, control_values)),
        "full_v4_minus_control_delta_vector": list(map(float, deltas)),
        "equal_scene_mean_delta": mean_delta,
        "bootstrap": {
            "algorithm": "paired_control_comparison_v1",
            "dtype": "float64",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "draw_shape": [BOOTSTRAP_REPLICATES, 8],
            "lower_95_sorted_zero_based_index": BOOTSTRAP_LOWER_INDEX,
            "lower_95": lower,
        },
        "positive_family_count": positive_family_count,
        "per_family_delta": per_family,
        "per_scene_delta": {
            family_to_scene[family]: per_family[family] for family in family_order
        },
        "allowed_positive_conclusion": (
            "P improved development selection utility under this fixed "
            "deterministic training schedule."
            if passed
            else None
        ),
        "negative_result_interpretation": (
            None
            if passed
            else (
                "Negative evidence for benefit from P under this fixed schedule; "
                "no rerun or replacement is authorized."
            )
        ),
    }


def execute_v4_matched_no_persistence(
    *, repository_root: Path = ROOT
) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v4_matched_no_persistence(repository_root)
    stage = "reserved_fresh_output"
    initial_state_receipt: Mapping[str, Any] | None = None
    initial_decoder_receipt: Mapping[str, Any] | None = None
    optimizer_receipt: Mapping[str, Any] | None = None
    access_receipt: Mapping[str, Any] | None = None
    context: Mapping[str, Any] | None = None
    accounting: Mapping[str, Any] | None = None
    training_diagnostics: Mapping[str, Any] | None = None
    checkpoint_binding: Mapping[str, Any] | None = None
    trace_binding: Mapping[str, Any] | None = None
    try:
        labels_api = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_labels_v1"
        )
        manifest, rows_by_role = _v1.load_label_bundle_v1(
            repository_root, labels_api=labels_api
        )
        context = _v1._prepare_runtime_v1(repository_root, manifest, labels_api)
        stage = "runtime_preflight"
        torch, np = context["torch"], context["np"]
        preflight = labels_api.summarize_preflight_v1(
            rows_by_role, context["schedule"]
        )
        if preflight != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")

        training_v1 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
        )
        training_v3 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_"
            "half_occupied_safety_aux"
        )
        training_control = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v4_"
            "matched_no_persistence"
        )
        _validate_training_core_v1(training_v1, training_v3, training_control)
        frozen = {
            role: training_v1.freeze_role_labels_v1(rows, role=role, np=np)
            for role, rows in rows_by_role.items()
        }
        informative = {
            role: np.asarray(
                [group[0]["informative_state"] for group in labels.state_groups],
                dtype=np.bool_,
            )
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_"
            "residual_local_semantic_decoder"
        )
        _v4._validate_model_api_v4(model_api)
        parent_model_api = importlib.import_module(
            "lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1"
        )
        survival_scoring = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1"
        )
        metrics_api = importlib.import_module(
            "lewm.benchmarks.go2_post_action_projective_support_metrics_v1"
        )
        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        n320_state = {
            name: value.detach().cpu().float().contiguous().clone()
            for name, value in context["fit"].encoder.state_dict().items()
        }
        masks = survival_scoring.build_swept_progress_masks_v1()
        current_frame_persistence_masks = (
            survival_scoring.build_current_frame_swept_progress_masks_v1()
        )
        stage = "reconstruct_fresh_v4_twice"
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_state, masks
        )
        witness_model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_state, masks
        )
        initial_state_receipt = _reconstructed_initialization_receipt_v1(
            model, witness_model, torch=torch
        )
        _validate_expected_initial_state_receipt_v1(initial_state_receipt)
        del witness_model

        model = model.to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_decoder_receipt = _v4._initial_decoder_receipt_v4(
            model,
            partition,
            torch=torch,
            inherited_semantic_method=(
                parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.
                semantic_logits_from_latent
            ),
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        training_v1.validate_optimizer_v1(optimizer, partition)
        optimizer_receipt = _optimizer_receipt_v1(optimizer, partition)
        _validate_expected_optimizer_receipt_v1(optimizer_receipt)
        if not any(
            name.startswith("predictor.swept_progress_head.")
            for name in partition.names["predictor"]
        ):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        stage = "fixed_training_1000_updates"
        accounting_state, trace, training_diagnostics = (
            training_control.run_fixed_training_v4_matched_no_persistence(
                model,
                optimizer,
                context["loader"],
                pairs["train"],
                frozen["train"],
                context["schedule"],
                context["device"],
            )
        )
        accounting = dict(accounting_state.__dict__)
        _validate_terminal_training_receipt_v1(
            accounting, trace, training_diagnostics
        )
        model.eval()
        model.requires_grad_(False)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "integrity_replacement": integrity_replacement_receipt_v1(),
                "development_only": True,
                "resume_authorized": False,
                "qualified": False,
                "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
                "semantic_decoder_initialization_seed": (
                    SEMANTIC_DECODER_INITIALIZATION_SEED
                ),
                "experiment_seed": EXPERIMENT_SEED,
                "initialization_source": "exact_n320_encoder_only",
                "predecessor_experiment_checkpoint_read": False,
                "persistence_treatment": persistence_treatment_receipt_v1(),
                "auxiliary_objective": dict(AUXILIARY_OBJECTIVE),
                "reconstructed_initialization": initial_state_receipt,
                "initial_semantic_decoder": initial_decoder_receipt,
                "initial_optimizer": optimizer_receipt,
                "accounting": accounting,
                "training_diagnostics": training_diagnostics,
                "first_update_component_witness": training_diagnostics[
                    "first_update_component_witness"
                ],
                "model_state_dict": state,
            },
            checkpoint_buffer,
        )
        checkpoint_binding = _v1._atomic_write_v1(
            output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue()
        )
        _, trace_binding = _v1._write_json_v1(
            output / "training_trace.json",
            {
                "schema": TRACE_SCHEMA,
                "integrity_replacement": integrity_replacement_receipt_v1(),
                "status": "COMPLETE",
                "persistence_treatment": persistence_treatment_receipt_v1(),
                "auxiliary_objective": dict(AUXILIARY_OBJECTIVE),
                "reconstructed_initialization": initial_state_receipt,
                "initial_semantic_decoder": initial_decoder_receipt,
                "initial_optimizer": optimizer_receipt,
                "accounting": accounting,
                "diagnostics": training_diagnostics,
                "first_update_component_witness": training_diagnostics[
                    "first_update_component_witness"
                ],
                "rows": list(trace),
            },
        )

        stage = "terminal_evaluation"
        action_prior_m = (
            frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64)
            * PROGRESS_SEGMENT_M
        )
        scored = {
            role: _v1.score_role_v1(
                model,
                context["loader"],
                pairs[role],
                frozen[role],
                action_prior_m,
                context["device"],
                torch=torch,
                np=np,
                training_core=training_v1,
                current_frame_persistence_masks=current_frame_persistence_masks,
                metrics_api=metrics_api,
            )
            for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_control(
                    scored[role]["scores_m"][arm],
                    frozen[role].prefix_lengths,
                    informative[role],
                    frozen[role].scene_ids,
                    frozen[role].family_ids,
                    np=np,
                )
                for arm in ALL_ARM_NAMES
            }
            for role in scored
        }
        selection_semantic = semantic_metrics_control(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"],
            np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_control(
                selection_scores["full"],
                selection_scores[name],
                selection_labels.prefix_lengths,
                informative["checkpoint_selection"],
                selection_labels.scene_ids,
                selection_labels.family_ids,
                np=np,
            )
            for name in CONTROL_NAMES
        }
        gate = evaluate_gate_control(
            role_metrics["checkpoint_selection"], selection_semantic, comparisons
        )
        treatment = v4_minus_control_treatment_comparison_v1(
            role_metrics["checkpoint_selection"]["full"],
            selection_labels.scene_ids,
            selection_labels.family_ids,
            np=np,
        )
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(
                current_frame_persistence_masks
            ),
        }
        status = (
            "COMPLETE_POSITIVE_PERSISTENCE_TREATMENT"
            if treatment["passed"]
            else "COMPLETE_NEGATIVE_PERSISTENCE_TREATMENT"
        )
        result, _ = _v1._write_json_v1(
            output / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "integrity_replacement": integrity_replacement_receipt_v1(),
                "status": status,
                "absolute_control_gate": gate,
                "treatment_predicate": treatment,
                "caps": {
                    "updates": MAXIMUM_UPDATES,
                    "presentations": MAXIMUM_PRESENTATIONS,
                },
                "seeds": {
                    "inherited_fresh_component_constructor": (
                        CONSTRUCTOR_INITIALIZATION_SEED
                    ),
                    "semantic_decoder": SEMANTIC_DECODER_INITIALIZATION_SEED,
                    "experiment_and_stochastic_execution": EXPERIMENT_SEED,
                    "bootstrap": BOOTSTRAP_SEED,
                },
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "pre_runtime_clarification_commit": (
                    PRE_RUNTIME_CLARIFICATION_COMMIT
                ),
                "label_manifest": {
                    "path": f"{LABEL_ROOT_RELATIVE_PATH}/{LABEL_MANIFEST_NAME}",
                    "file_sha256": LABEL_MANIFEST_FILE_SHA256,
                    "content_sha256": manifest["content_sha256"],
                    "byte_count": LABEL_MANIFEST_BYTE_COUNT,
                    "role_files": manifest["files"],
                },
                "n320": {
                    "gate_content_sha256": context["n320_gate"]["content_sha256"],
                    "checkpoint": context["n320_checkpoint"],
                    "encoder_only_initialization": True,
                    "predecessor_experiment_checkpoint_read": False,
                },
                "hardware": context["hardware"],
                "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
                "masks": mask_receipts,
                "scientific_treatment": {
                    "only_change": "persistence_absent_from_backward_membership",
                    "persistence": persistence_treatment_receipt_v1(),
                    "model_changed": False,
                    "data_changed": False,
                    "optimizer_rules_changed": False,
                    "schedule_changed": False,
                    "evaluation_changed": False,
                    "forwards_or_ema_changed": False,
                },
                "initialization": {
                    "reconstructed_state": initial_state_receipt,
                    "semantic_decoder": initial_decoder_receipt,
                    "optimizer": optimizer_receipt,
                },
                "training": {
                    "accounting": accounting,
                    "diagnostics": training_diagnostics,
                    "first_update_component_witness": training_diagnostics[
                        "first_update_component_witness"
                    ],
                    "joint_from_update_one": True,
                    "separate_head_or_predictor_training": False,
                    "checkpoint": checkpoint_binding,
                    "trace": trace_binding,
                },
                "action_prior_mean_progress_m": action_prior_m.tolist(),
                "roles": role_metrics,
                "selection_semantic": selection_semantic,
                "selection_control_comparisons": comparisons,
                "wrong_rgb_mapping_sha256": {
                    role: scored[role]["wrong_rgb_mapping_sha256"] for role in scored
                },
                "determinism": {
                    "algorithms_enabled": bool(
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "warn_only": True,
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                },
                "access": access_receipt,
                "interpretation": {
                    "development_selection_diagnostic_only": True,
                    "allowed_positive_conclusion": treatment[
                        "allowed_positive_conclusion"
                    ],
                    "negative_result_interpretation": treatment[
                        "negative_result_interpretation"
                    ],
                    "generalization_claimed": False,
                    "navigation_claimed": False,
                    "seed_robustness_claimed": False,
                },
                "authority": {
                    "development_only": True,
                    "diagnostic_only": True,
                    "g2_navigation_final_evaluation_opened": False,
                    "heldout_or_sealed_opened": False,
                    "checkpoint_qualified": False,
                    "promotion_performed": False,
                    "retry_or_resume_authorized": False,
                    "replacement_or_warm_start_authorized": False,
                },
            },
        )
        return result
    except Exception as error:
        if access_receipt is None and context is not None:
            try:
                access_receipt = _v1._access_receipt_v1(context)
            except Exception:
                pass
        if not (output / "result.json").exists() and not (
            output / "failure.json"
        ).exists():
            try:
                _v1._write_json_v1(
                    output / "failure.json",
                    {
                        "schema": FAILURE_SCHEMA,
                        "integrity_replacement": (
                            integrity_replacement_receipt_v1()
                        ),
                        "status": "FAILED_NO_RETRY_OR_RESUME",
                        "failure_stage": stage,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "terminal_loss_identity_failure": (
                            terminal_loss_identity_failure_receipt_v1(error)
                        ),
                        "first_update_component_failure": (
                            first_update_component_failure_receipt_v1(error)
                        ),
                        "preregistration_commit": PREREGISTRATION_COMMIT,
                        "pre_runtime_clarification_commit": (
                            PRE_RUNTIME_CLARIFICATION_COMMIT
                        ),
                        "caps": {
                            "updates": MAXIMUM_UPDATES,
                            "presentations": MAXIMUM_PRESENTATIONS,
                        },
                        "seeds": {
                            "constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                            "semantic_decoder": SEMANTIC_DECODER_INITIALIZATION_SEED,
                            "execution": EXPERIMENT_SEED,
                            "bootstrap": BOOTSTRAP_SEED,
                        },
                        "persistence_treatment": persistence_treatment_receipt_v1(),
                        "full_v4_reference": (
                            full_v4_reference_family_utility_receipt_v1()
                        ),
                        "reconstructed_initialization": initial_state_receipt,
                        "initial_semantic_decoder": initial_decoder_receipt,
                        "initial_optimizer": optimizer_receipt,
                        "terminal_training_if_completed": {
                            "accounting": accounting,
                            "diagnostics": training_diagnostics,
                            "checkpoint": checkpoint_binding,
                            "trace": trace_binding,
                        },
                        "access_receipt_if_completed": access_receipt,
                        "predecessor_experiment_checkpoint_read": False,
                        "authority": {
                            "development_only": True,
                            "diagnostic_only": True,
                            "g2_navigation_final_evaluation_opened": False,
                            "heldout_or_sealed_opened": False,
                            "checkpoint_qualified": False,
                            "promotion_performed": False,
                            "retry_or_resume_authorized": False,
                            "replacement_or_warm_start_authorized": False,
                        },
                    },
                )
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = execute_v4_matched_no_persistence(repository_root=args.repository_root)
    print(
        _v1._canonical_json_bytes(
            {
                "status": result["status"],
                "result": f"{OUTPUT_RELATIVE_PATH}/result.json",
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
