"""Source-only contract for the science-identical joint-JEPA V3 hash fix.

V3 reuses the complete frozen V2 science, model, import-lifetime correction and
failure-receipt machinery.  Its sole implementation delta is flattening each
detached CPU-contiguous tensor before the write-only ``torch.uint8`` state-hash
view so that zero-dimensional buffers are supported.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V2_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)


def _source_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_v2 = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v3_hash_frozen_v2_contract",
    FROZEN_V2_CONTRACT_RELATIVE_PATH,
)
for _name in _v2.__all__:
    globals()[_name] = getattr(_v2, _name)

# Private source-only helpers used by inherited launchers and validators.
_read_regular_source = _v2._read_regular_source
_source_freeze_commit = _v2._source_freeze_commit
_validate_artifact_binding = _v2._validate_artifact_binding


IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/v3_contract_impl",
    "/root/v3_runner_tests_impl",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement"
)
EXPERIMENT_ID = (
    "geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement"
)

FROZEN_V2_SOURCE_COMMIT = "1ff023d306f63d0651639f699038d538f8f6336d"
FROZEN_V2_RUNNER_RELATIVE_PATH = _v2.RUNNER_RELATIVE_PATH
FROZEN_V2_LAUNCHER_RELATIVE_PATH = _v2.LAUNCHER_RELATIVE_PATH
FROZEN_V2_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _v2.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)
FROZEN_V2_MODEL_RELATIVE_PATH = _v2.MODEL_RELATIVE_PATH
FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH = _v2.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V2_SOURCE_MANIFEST_COMMIT = FROZEN_V2_SOURCE_COMMIT
FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256 = (
    "270f64c520e7a0193f73a81e7a4cad9c62db162ddbcfbb840302e79c15ba004e"
)
FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "90b0f6feca7d987d6a547f201d55784c52e5ea2646dc339f9d4ff63f81cb2d0a"
)
FROZEN_V2_SOURCE_BINDINGS_SHA256 = (
    "8c017968de1eb2b1077d970f407917631db1c167774b7385552ad6a0e0020403"
)
FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT = 25_368
FROZEN_V2_SOURCE_COUNT = 79

FROZEN_V2_REVIEW_RELATIVE_PATH = _v2.REVIEW_RELATIVE_PATH
FROZEN_V2_REVIEW_COMMIT = "fb73acb0acd19ab29a6826ddbec393a6d2913f80"
FROZEN_V2_REVIEW_FILE_SHA256 = (
    "d9855b4454d45fd10f6947ba1dfb66c37549e1270681d479533a2976f2bc17a3"
)
FROZEN_V2_REVIEW_CONTENT_SHA256 = (
    "fa2bcee13e1abf0cdf985ba17586e3cd16b4ccd2cfd91e2a95d051a061bacdb5"
)
FROZEN_V2_REVIEW_BYTE_COUNT = 32_359

FROZEN_V2_AUTHORIZATION_RELATIVE_PATH = _v2.AUTHORIZATION_RELATIVE_PATH
FROZEN_V2_AUTHORIZATION_COMMIT = "7fd4a0718c45c1b088780732d42e1e9756c092e6"
FROZEN_V2_AUTHORIZATION_FILE_SHA256 = (
    "ca77161bceace3b20a7642ccb9b71cd382396a5a71c6592ea3c52d32e00fe908"
)
FROZEN_V2_AUTHORIZATION_CONTENT_SHA256 = (
    "027e4a65be73301c41b55bcdcd422705c0474a6a017d6b2b7728d525cd45d740"
)
FROZEN_V2_AUTHORIZATION_BYTE_COUNT = 26_610

V2_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement_terminal_audit_2026-07-27.json"
)
V2_TERMINAL_AUDIT_COMMIT = "20b13fe3100d96e8d17b65da49261d1388d5015f"
V2_TERMINAL_AUDIT_FILE_SHA256 = (
    "184ba6c10e2c37fec12608bf56ba97fc345ab180c3f001989588a630dde9bb5e"
)
V2_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "5cfff834da4b0c0667ebc6e282abdba651dfa58a766db0174d00d32c2510ea51"
)
V2_TERMINAL_AUDIT_BYTE_COUNT = 8_922
V2_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_ZERO_PRESENTATION_SCALAR_TENSOR_STATE_HASH_OPERATIONAL_"
    "FAILURE_V2_CONSUMED_CLOSED_NO_RETRY"
)
V2_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_ZERO_PRESENTATION_SCALAR_TENSOR_STATE_HASH_OPERATIONAL_INTEGRITY_"
    "FAILURE_NOT_A_SCIENTIFIC_OR_MECHANISM_RESULT"
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement_preregistration_"
    "2026-07-27.md"
)
PREREGISTRATION_COMMIT = "549085000059ba3b2c5c48b1cf60a284f0398462"
PREREGISTRATION_FILE_SHA256 = (
    "02ad62c3da94a2da907ebadf14bf7a7a1fce326bb78eb9cb7f921ac3c82efb92"
)
PREREGISTRATION_BYTE_COUNT = 6_708

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement_source_manifest_"
    "2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement_source_review_"
    "2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement_execution_authorization_"
    "2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v2.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != FROZEN_V2_SOURCE_COUNT or len(SOURCE_PATHS) != 84:
    raise PermissionError("V3 must be the frozen 79-source V2 closure plus five files")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
# Runtime receipt schema labels deliberately remain the frozen V2 labels.  The
# preregistered sole delta does not rewrite the complete receipt machinery.
if any(
    globals()[name] != getattr(_v2, name)
    for name in (
        "RESERVATION_SCHEMA", "METRICS_SCHEMA", "ARTIFACT_SCHEMA",
        "ACCESS_SCHEMA", "RESULT_SCHEMA", "COMPLETION_SCHEMA", "FAILURE_SCHEMA",
    )
):
    raise PermissionError("V3 runtime receipt schema labels must equal frozen V2")
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_"
    "V3_SCALAR_TENSOR_STATE_HASH_INTEGRITY_REPLACEMENT"
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_GEOMETRY_ANCHORED_DEFORMABLE_BEV_LIFT_JOINT_JEPA_V3_SCALAR_"
    "TENSOR_STATE_HASH_INTEGRITY_REPLACEMENT_OPERATIONAL_FAILURE_NO_RETRY"
)
CONTROL_FAIL_OPERATIONAL = OPERATIONAL_FAILURE_STATUS

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement/attempt_v1"
)
EXECUTION_AUTHORITY = {
    **_v2.EXECUTION_AUTHORITY,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v2_retry_authorized": False,
    "v2_resume_or_state_reuse_authorized": False,
    "science_identical_scalar_tensor_state_hash_integrity_replacement_only": True,
}
SOURCE_ONLY_AUTHORITY = {
    **_v2.SOURCE_ONLY_AUTHORITY,
    "isolated_cpu_scalar_and_full_model_state_hash_preflight_authorized": True,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)

FROZEN_V2_SCIENCE_CONTRACT_SHA256 = (
    "f839076bf7f9db9e9f211703323436f4b607cca21e2e60fb228e4d174c699fa3"
)
FROZEN_V2_SCIENCE_COMPONENT_SHA256 = {
    "model": "595d91a6fc9ae985378ff480780bf7ad5a9beeb3c7f35ab012c010bb74162f39",
    "objective": "93c73c1f1a91de70699f634821159d4d544431b45faa469202016fa0b9fd7ba8",
    "optimizer": "2bb70f943838b656540b3dac3b6e0f30bb384547180270274abfc5077e264b34",
    "schedule": "bc0ad45c06171cff7533fbfcb054e5afecf6086de0a58060c35cb5ca0256c2e3",
    "gate_thresholds": (
        "0c485c0bccb88873c0ff76a1061a315420b6c27c4865b259d3b4c6f374862bd0"
    ),
}
INTEGRITY_DELTA = {
    "scope": "write_only_tensor_state_hash_scalar_rank_adapter_only",
    "v2_failure": (
        "RuntimeError: self.dim() cannot be 0 to view Float as Byte "
        "(different element sizes)"
    ),
    "frozen_v2_byte_view": "tensor.view(torch.uint8)",
    "v3_byte_view": "tensor.reshape(-1).view(torch.uint8)",
    "detach_cpu_and_contiguous_steps_preserved": True,
    "scalar_value_and_byte_order_preserved": True,
    "all_non_scalar_digest_contributions_equal_frozen_v2": True,
    "v2_runtime_import_root_lifetime_fix_preserved": True,
    "v2_runtime_receipt_schema_labels_preserved": True,
    "digest_used_by_model_loss_gradient_gate_or_optimizer": False,
    "model_data_seed_schedule_losses_thresholds_initialization_or_caps_changed": False,
}
STATE_HASH_PREFLIGHT_REQUIREMENTS = {
    "device": "cpu_only",
    "synthetic_encoder_state_only": True,
    "scalar_float_integer_and_boolean_buffers_hash_without_error": True,
    "mixed_scalar_and_non_scalar_mapping_matches_independent_raw_byte_reference": True,
    "all_non_scalar_mapping_digest_identical_under_frozen_v2_and_v3": True,
    "fresh_unchanged_v1_model_constructed": True,
    "complete_model_state_groups_hashed": [
        "predictor",
        "online_encoder",
        "target_encoder",
        "online_bev_lift",
        "target_bev_lift",
        "full_state",
    ],
    "persistent_scalar_camera_and_counter_buffers_included": True,
    "scalar_counter_change_changes_full_state_digest": True,
    "online_target_initial_equality_preserved": True,
    "all_initialized_tensor_values_and_bytes_unchanged": True,
    "caller_cpu_rng_state_restored_exactly": True,
    "caller_sys_path_restored_exactly": True,
    "generated_inputs_opened": [],
    "n320_checkpoint_or_runtime_tensors_opened": [],
    "runtime_outputs_or_traces_opened": [],
    "accelerator_queries_or_work_performed": [],
    "navigation_heldout_sealed_or_rejected_material_opened": [],
}
SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v2_manifest_and_all_79_sources_rehashed": True,
    "v2_review_and_consumed_authorization_revalidated": True,
    "v2_terminal_audit_exact_and_zero_scientific_presentations": True,
    "v2_permanently_closed_and_v3_is_not_a_retry_or_resume": True,
    "reshape_before_uint8_view_is_the_only_implementation_delta": True,
    "exact_v1_model_reused_without_new_model_file": True,
    "v2_runtime_import_root_lifetime_fix_preserved": True,
    "v2_runtime_receipt_schema_labels_preserved": True,
    "model_objective_optimizer_schedule_and_gate_hashes_equal_v1_v2": True,
    "data_seed_initialization_losses_thresholds_and_caps_equal_v1_v2": True,
    "scalar_and_mixed_mapping_cpu_hash_preflight_passed": True,
    "all_non_scalar_v2_v3_digest_identity_preflight_passed": True,
    "complete_fresh_model_cpu_state_hash_preflight_passed": True,
    "scalar_counter_digest_sensitivity_preflight_passed": True,
    "cpu_rng_and_sys_path_restored_exactly_by_preflight": True,
    "distinct_absent_before_reservation_v3_output_root": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def v2_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V2_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V2_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V2_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V2_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V2_TERMINAL_AUDIT_BYTE_COUNT,
    }


def frozen_v2_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V2_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT,
        "source_bindings_sha256": FROZEN_V2_SOURCE_BINDINGS_SHA256,
        "source_count": FROZEN_V2_SOURCE_COUNT,
    }


def frozen_v2_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V2_REVIEW_COMMIT,
        "file_sha256": FROZEN_V2_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V2_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V2_REVIEW_BYTE_COUNT,
    }


def frozen_v2_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V2_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V2_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V2_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V2_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V2_AUTHORIZATION_BYTE_COUNT,
    }


def validate_frozen_v2_source_and_authority(
    root: Path = ROOT,
) -> dict[str, str]:
    manifest_raw = _read_regular_source(
        root / FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH
    )
    if (
        len(manifest_raw) != FROZEN_V2_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(manifest_raw).hexdigest()
        != FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V2 source manifest raw identity changed")
    manifest = _v2.validate_source_manifest(manifest_raw, root)
    if (
        manifest.get("status") != "PASS_SOURCE_CLOSURE"
        or manifest.get("content_sha256")
        != FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("source_bindings_sha256")
        != FROZEN_V2_SOURCE_BINDINGS_SHA256
        or manifest.get("source_count") != FROZEN_V2_SOURCE_COUNT
    ):
        raise PermissionError("frozen V2 source manifest conclusion changed")
    manifest_binding = _v2.artifact_binding(
        FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=FROZEN_V2_SOURCE_MANIFEST_CONTENT_SHA256,
    )

    review_raw = _read_regular_source(root / FROZEN_V2_REVIEW_RELATIVE_PATH)
    if (
        len(review_raw) != FROZEN_V2_REVIEW_BYTE_COUNT
        or hashlib.sha256(review_raw).hexdigest() != FROZEN_V2_REVIEW_FILE_SHA256
    ):
        raise PermissionError("frozen V2 review raw identity changed")
    review = _v2.validate_review(review_raw, manifest_binding, root=root)
    if review.get("content_sha256") != FROZEN_V2_REVIEW_CONTENT_SHA256:
        raise PermissionError("frozen V2 review conclusion changed")
    review_binding = _v2.artifact_binding(
        FROZEN_V2_REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=FROZEN_V2_REVIEW_CONTENT_SHA256,
    )

    authorization_raw = _read_regular_source(
        root / FROZEN_V2_AUTHORIZATION_RELATIVE_PATH
    )
    if (
        len(authorization_raw) != FROZEN_V2_AUTHORIZATION_BYTE_COUNT
        or hashlib.sha256(authorization_raw).hexdigest()
        != FROZEN_V2_AUTHORIZATION_FILE_SHA256
    ):
        raise PermissionError("frozen V2 authorization raw identity changed")
    authorization = _v2.validate_authorization(
        authorization_raw, review_binding, root=root
    )
    if (
        authorization.get("content_sha256")
        != FROZEN_V2_AUTHORIZATION_CONTENT_SHA256
    ):
        raise PermissionError("frozen V2 authorization conclusion changed")
    return {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }


def model_config() -> dict[str, Any]:
    return _v2.model_config()


def objective_contract() -> dict[str, Any]:
    return _v2.objective_contract()


def optimizer_contract() -> dict[str, Any]:
    return _v2.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v2.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v2.runtime_authorization_template()


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v2.science_contract())
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["governing_documents"] = {
        **value["governing_documents"],
        "v3_preregistration": preregistration_binding(),
        "v2_source_manifest": frozen_v2_source_manifest_binding(),
        "v2_source_review": frozen_v2_review_binding(),
        "v2_consumed_authorization": frozen_v2_authorization_binding(),
        "v2_terminal_audit": v2_terminal_audit_binding(),
    }
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "integrity_replacement_of": _v2.EXPERIMENT_ID,
        "v2_retry_or_resume": False,
    }
    value["integrity_replacement"] = {
        **INTEGRITY_DELTA,
        "frozen_v2_science_contract_sha256": FROZEN_V2_SCIENCE_CONTRACT_SHA256,
        "frozen_v2_science_component_sha256": dict(
            FROZEN_V2_SCIENCE_COMPONENT_SHA256
        ),
    }
    return value


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = _v2.validate_governing_documents(root)
    validate_frozen_v2_source_and_authority(root)

    prereg_raw = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(prereg_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(prereg_raw).hexdigest() != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("V3 preregistration changed")

    audit_raw = _read_regular_source(root / V2_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(audit_raw) != V2_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest()
        != V2_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V2 terminal audit raw identity changed")
    audit = _v2.parse_canonical_json(audit_raw, name="V2 terminal audit")
    zero = audit.get("zero_scientific_work", {})
    runtime = audit.get("runtime_boundary", {})
    if (
        audit.get("content_sha256") != V2_TERMINAL_AUDIT_CONTENT_SHA256
        or audit.get("status") != V2_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V2_TERMINAL_AUDIT_CLASSIFICATION
        or any(
            zero.get(name) != 0
            for name in (
                "updates", "presentations", "objective_evaluations",
                "backward_calls", "optimizer_updates", "ema_updates",
                "pair_presentations_loaded",
            )
        )
        or runtime.get("training_pair_presentations_loaded") != 0
        or runtime.get("rgb_or_raster_row_requests") != 0
        or runtime.get("rgb_or_raster_physical_read_attempts") != 0
        or audit.get("scientific_conclusion", {}).get(
            "scientific_evidence_produced"
        ) is not False
        or audit.get("closure", {}).get("v2_closed") is not True
        or audit.get("closure", {}).get(
            "v2_retry_resume_repair_or_same_root_reuse_authorized"
        ) is not False
    ):
        raise PermissionError("V2 terminal audit conclusion changed")

    result.update({
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
        FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH:
            FROZEN_V2_SOURCE_MANIFEST_FILE_SHA256,
        FROZEN_V2_REVIEW_RELATIVE_PATH: FROZEN_V2_REVIEW_FILE_SHA256,
        FROZEN_V2_AUTHORIZATION_RELATIVE_PATH:
            FROZEN_V2_AUTHORIZATION_FILE_SHA256,
        V2_TERMINAL_AUDIT_RELATIVE_PATH: V2_TERMINAL_AUDIT_FILE_SHA256,
    })
    return result


def validate_source_manifest(raw: bytes, root: Path = ROOT) -> dict[str, Any]:
    value = _v2.parse_canonical_json(raw, name="V3 source manifest")
    fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources") != list(SOURCE_PATHS)
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or paths != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_count") != len(SOURCE_PATHS)
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("V3 source manifest contract changed")
    for relative, binding in zip(SOURCE_PATHS, bindings, strict=True):
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256", "byte_count"}
            or binding.get("path") != relative
            or safe_relative_source_path(relative) != relative
            or not is_sha256(binding.get("file_sha256"))
            or type(binding.get("byte_count")) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V3 source binding changed")
        payload = _read_regular_source(root / relative)
        if (
            len(payload) != binding["byte_count"]
            or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"manifest-bound V3 source changed: {relative}")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    result = {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def validate_review(
    raw: bytes,
    manifest_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = _v2.parse_canonical_json(raw, name="V3 source review")
    expected_manifest = _validate_artifact_binding(
        dict(manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH
    )
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    if expected_manifest != artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    ):
        raise PermissionError("V3 review manifest binding changed")
    expected_sources = current_source_bindings(root)
    fields = {
        "schema", "status", "implementation_authors", "reviewer",
        "source_freeze_commit", "reviewed_sources", "source_manifest",
        "preregistration", "v2_source_manifest", "v2_source_review",
        "v2_consumed_authorization", "v2_terminal_audit", "science_contract",
        "cpu_state_hash_preflight", "source_only_checks", "scientific_checks",
        "findings", "authority", "content_sha256",
    }
    reviewer = value.get("reviewer")
    _source_freeze_commit(
        value.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != REVIEW_SCHEMA
        or value.get("status")
        != "PASS_SOURCE_AND_SCIENCE_IDENTICAL_SCALAR_TENSOR_STATE_HASH_FIX"
        or value.get("implementation_authors") != list(IMPLEMENTATION_AUTHORS)
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in IMPLEMENTATION_AUTHORS
        or value.get("reviewed_sources") != expected_sources
        or value.get("source_manifest") != expected_manifest
        or value.get("preregistration") != preregistration_binding()
        or value.get("v2_source_manifest")
        != frozen_v2_source_manifest_binding()
        or value.get("v2_source_review") != frozen_v2_review_binding()
        or value.get("v2_consumed_authorization")
        != frozen_v2_authorization_binding()
        or value.get("v2_terminal_audit") != v2_terminal_audit_binding()
        or value.get("science_contract") != science_contract()
        or value.get("cpu_state_hash_preflight")
        != STATE_HASH_PREFLIGHT_REQUIREMENTS
        or value.get("source_only_checks") != {
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "accelerators_queried_or_used": [],
            "navigation_heldout_sealed_or_rejected_material_opened": [],
        }
        or value.get("scientific_checks") != SCIENTIFIC_REVIEW_CHECKS
        or value.get("findings") != []
        or value.get("authority") != REVIEW_AUTHORITY
    ):
        raise PermissionError("V3 source review did not pass exact hash-fix scope")
    return dict(value)


def validate_authorization(
    raw: bytes,
    review_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = _v2.parse_canonical_json(raw, name="V3 execution authorization")
    expected_review = _validate_artifact_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_raw = _read_regular_source(root / REVIEW_RELATIVE_PATH)
    if expected_review != artifact_binding(
        REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=_v2.parse_canonical_json(
            review_raw, name="V3 source review"
        )["content_sha256"],
    ):
        raise PermissionError("V3 authorization review binding changed")
    manifest_raw = _read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw, root)
    manifest_binding = artifact_binding(
        SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    )
    review = validate_review(review_raw, manifest_binding, root=root)
    fields = {
        "schema", "status", "authorizer", "source_freeze_commit",
        "independent_source_review", "preregistration", "v2_terminal_audit",
        "runtime_inputs", "experiment", "authority", "content_sha256",
    }
    authorizer = value.get("authorizer")
    source_commit = _source_freeze_commit(
        review.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != AUTHORIZATION_SCHEMA
        or value.get("status") != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {*IMPLEMENTATION_AUTHORS, review["reviewer"]}
        or value.get("source_freeze_commit") != source_commit
        or value.get("independent_source_review") != expected_review
        or value.get("preregistration") != preregistration_binding()
        or value.get("v2_terminal_audit") != v2_terminal_audit_binding()
        or value.get("runtime_inputs") != runtime_authorization_template()
        or value.get("experiment") != science_contract()
        or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("V3 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v2.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity", "current_source_bindings",
    "frozen_v2_authorization_binding", "frozen_v2_review_binding",
    "frozen_v2_source_manifest_binding", "model_config",
    "objective_contract", "optimizer_contract", "preregistration_binding",
    "runtime_authorization_template", "science_contract",
    "v2_terminal_audit_binding", "validate_authorization",
    "validate_frozen_v2_source_and_authority", "validate_governing_documents",
    "validate_review", "validate_source_manifest",
})
