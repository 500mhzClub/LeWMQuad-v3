"""Source-only contract for the event-delta V2 delegation fix.

V2 reuses the complete frozen V1 science and model.  Its sole implementation
delta is the final control transfer after the complete V1 event-delta runtime
rebind: execution goes directly to the frozen base runner and launcher so no
predecessor ``main`` can replace the reviewed event-delta hooks.

Importing this module opens only repository source and public governing
documents.  It opens no generated input, dataset row, RGB, raster, checkpoint,
tensor, runtime output, trace, accelerator, navigation, held-out, sealed,
rejected, or production material.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V1_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
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


_v1 = _source_module(
    "_lewm_two_mode_event_delta_v2_delegation_frozen_v1_contract",
    FROZEN_V1_CONTRACT_RELATIVE_PATH,
)
for _name in _v1.__all__:
    globals()[_name] = getattr(_v1, _name)

# Private source-only helpers reused by the inherited launcher and validators.
_read_regular_source = _v1._read_regular_source
_source_freeze_commit = _v1._source_freeze_commit
_validate_artifact_binding = _v1._validate_artifact_binding


IMPLEMENTATION_AUTHORS = (
    "/root",
    "/root/event_delta_v2_contract_impl",
    "/root/event_delta_v2_runner_impl",
)
SCHEMA_PREFIX = (
    "lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement"
)
EXPERIMENT_ID = (
    "geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement"
)

FROZEN_V1_SOURCE_COMMIT = "c414231d6d0e0d0cbf9282aec16944d4d4b7cfca"
FROZEN_V1_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
FROZEN_V1_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "source_closure.py"
)
FROZEN_V1_MODEL_RELATIVE_PATH = _v1.MODEL_RELATIVE_PATH
FROZEN_V1_MODEL_TEST_RELATIVE_PATH = _v1.MODEL_TEST_RELATIVE_PATH
FROZEN_V1_MODEL_TEST_FILE_SHA256 = (
    "09170a2cceb297df65bfd6c3bf6f4f3aedda077777c8f837095cbde3a53198d6"
)
FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH = _v1.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V1_SOURCE_MANIFEST_COMMIT = FROZEN_V1_SOURCE_COMMIT
FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256 = (
    "f87aa717fd118f3fb6e0a0e169dd0f4aec812f5a305cf95eb5b809e0c6c13e50"
)
FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "db5c7fdab152f75a3bafd7c94ba555bac5c5441e44fbb1ddb7ddb439ae74aa70"
)
FROZEN_V1_SOURCE_BINDINGS_SHA256 = (
    "d7f6d4302c6e5ab6ff1ce24089ba8c7b20df80dda92dcaea3897ccb200315f8b"
)
FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT = 33_275
FROZEN_V1_SOURCE_COUNT = 98

FROZEN_V1_REVIEW_RELATIVE_PATH = _v1.REVIEW_RELATIVE_PATH
FROZEN_V1_REVIEW_COMMIT = "60dea0ae159db279643e5dafbd5c5aa4701f436b"
FROZEN_V1_REVIEW_FILE_SHA256 = (
    "c22857709ff8eb6128e7957a45eb2ab6e1dae697dc9b7be1afa8b67ab3811177"
)
FROZEN_V1_REVIEW_CONTENT_SHA256 = (
    "c4bc70ccea0bb90c1d79f942c9a627c41c3de333635467b60614ff7028de1a4e"
)
FROZEN_V1_REVIEW_BYTE_COUNT = 50_356

FROZEN_V1_AUTHORIZATION_RELATIVE_PATH = _v1.AUTHORIZATION_RELATIVE_PATH
FROZEN_V1_AUTHORIZATION_COMMIT = (
    "9b5a5594c7bb2f7fd79f56dab83649b0eaca16b6"
)
FROZEN_V1_AUTHORIZATION_FILE_SHA256 = (
    "1eb762cbac646553bb3bda481478032a08b2a3dbec03eb3598a9991b4a800eba"
)
FROZEN_V1_AUTHORIZATION_CONTENT_SHA256 = (
    "520411f30753f9b4781d0bdfa09cdcd170b1cb5528a7b0d4718305832bcfb4b1"
)
FROZEN_V1_AUTHORIZATION_BYTE_COUNT = 38_836

V1_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v1_"
    "terminal_audit_2026-07-27.json"
)
V1_TERMINAL_AUDIT_COMMIT = "2f88edb653a93c5b9a98cfa8792a73fe4900fc9f"
V1_TERMINAL_AUDIT_FILE_SHA256 = (
    "38417a41b0483cbba318fffb9460a14d021c07141525a4f19f2a2748e9398495"
)
V1_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "594db612aa4561688343f76c1c6f8579ac307f5f5289d72c58cef6ac20a41111"
)
V1_TERMINAL_AUDIT_BYTE_COUNT = 15_588
V1_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_ZERO_PRESENTATION_TERMINAL_SOURCE_WITNESS_BINDING_DEFECT_"
    "ATTEMPT_CONSUMED_MECHANISM_UNTESTED_NO_RETRY"
)
V1_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_RECEIPT_CHAIN_ZERO_PRESENTATION_TERMINAL_STRUCTURAL_SOURCE_"
    "WITNESS_BINDING_FAILURE_MECHANISM_NOT_TESTED"
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement_preregistration_2026-07-27.md"
)
PREREGISTRATION_COMMIT = "bff17ce09e564157586a64c612329bd279dc8db0"
PREREGISTRATION_FILE_SHA256 = (
    "07e10107464df2d070a57a49766c5011ba330fb0aab236e9899999fccf9b98a8"
)
PREREGISTRATION_BYTE_COUNT = 11_858

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement.py"
)

# The reviewed model witness remains the exact frozen V1 model test.  Only
# generic V2 wrapper/checker aliases point at the additive combined test.
MODEL_TEST_RELATIVE_PATH = FROZEN_V1_MODEL_TEST_RELATIVE_PATH
CONTRACT_RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement_source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement_source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement_execution_authorization_"
    "2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v1.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if (
    len(REUSED_SOURCE_PATHS) != FROZEN_V1_SOURCE_COUNT
    or len(ADDITIVE_SOURCE_PATHS) != 5
    or len(SOURCE_PATHS) != 103
):
    raise PermissionError("V2 must be the frozen 98-source V1 closure plus five files")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_GEOMETRY_ANCHORED_TWO_MODE_EVENT_DELTA_JOINT_JEPA_"
    "V2_RUNTIME_DELEGATION_INTEGRITY_REPLACEMENT"
)
OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_GEOMETRY_ANCHORED_TWO_MODE_EVENT_DELTA_JOINT_JEPA_V2_RUNTIME_"
    "DELEGATION_INTEGRITY_REPLACEMENT_OPERATIONAL_FAILURE_NO_RETRY"
)
CONTROL_FAIL_OPERATIONAL = OPERATIONAL_FAILURE_STATUS

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement/attempt_v1"
)
EXECUTION_AUTHORITY = {
    **copy.deepcopy(_v1.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v1_retry_authorized": False,
    "science_identical_runtime_delegation_integrity_replacement_only": True,
}
SOURCE_ONLY_AUTHORITY = {
    **copy.deepcopy(_v1.SOURCE_ONLY_AUTHORITY),
    "isolated_cpu_runtime_delegation_preflight_authorized": True,
}
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)

FROZEN_V1_SCIENCE_CONTRACT_SHA256 = (
    "26c095f0b330e6e43952814e6a3b910f15b72a906d1c2f3d931a70c959ae6974"
)
FROZEN_V1_SCIENCE_COMPONENT_SHA256 = {
    "model": "4c84691d76eaf2c3b5eee345bb3b1c9cf8dd747e9512fc91c9d6f74b37337b03",
    "objective": "85017d1618e75970a2e70e1ace6f6930650aa5b351c60855753bcdceaa3515d4",
    "optimizer": "2bb70f943838b656540b3dac3b6e0f30bb384547180270274abfc5077e264b34",
    "schedule": "bc0ad45c06171cff7533fbfcb054e5afecf6086de0a58060c35cb5ca0256c2e3",
    "gate_thresholds": (
        "97fa8bb4b2740e68cadf90974ab80ff33419a854b07a16a258e2f49c3f177036"
    ),
    "work_accounting": (
        "013837055e693ae754324d7c9b8b098d47efed5f569505cf0f58fca8b432e359"
    ),
    "warning_policy": (
        "01a958d0de33a399453c7262d07f6328aabb3bbeaa83cfa045f52cdd03b6a67b"
    ),
    "runtime_input_template": (
        "393563699929bbfd7ca4d9c97c2c63b8a2583bfcc093f61ca0926cb63d24924b"
    ),
}
FROZEN_V1_SCIENTIFIC_IDENTITY_SHA256 = {
    "science_contract": FROZEN_V1_SCIENCE_CONTRACT_SHA256,
    **FROZEN_V1_SCIENCE_COMPONENT_SHA256,
}
if len(FROZEN_V1_SCIENTIFIC_IDENTITY_SHA256) != 9:
    raise PermissionError("event-delta V1 scientific identity must contain nine hashes")

INTEGRITY_DELTA = {
    "scope": "final_runner_and_launcher_control_transfer_only",
    "v1_failure": "reviewed_model_source_synthetic_witness_sha256_was_null",
    "load_frozen_v1_event_runner_before_v2_contract_rebind": True,
    "complete_v1_event_hooks_rebound_to_v2_identity": True,
    "runner_transfers_directly_to_frozen_base_main": True,
    "launcher_transfers_directly_to_already_rebound_frozen_base_launcher": True,
    "predecessor_main_after_final_v2_rebind_called": False,
    "reviewed_source_map_populated_before_update_zero": True,
    "fallback_hard_coded_witness_sha_permitted": False,
    "v1_source_modified": False,
    "model_data_seed_schedule_losses_thresholds_initialization_or_caps_changed": False,
    "receipt_structure_inventory_semantics_or_terminal_lifecycle_changed": False,
}

RUNTIME_DELEGATION_PREFLIGHT_REQUIREMENTS = {
    "isolated_import": {
        "interpreter_arguments": ["-I", "-B"],
        "runner_launcher_and_checker_imported": True,
        "torch_imported": False,
        "numpy_imported": False,
    },
    "final_runner_boundary": {
        "contract": CONTRACT_RELATIVE_PATH,
        "execute": f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_execute",
        "terminal_failure": (
            f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_terminal_failure"
        ),
        "load_post_reservation_stack": (
            f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_load_post_reservation_stack"
        ),
        "parameter_receipt": (
            f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_parameter_receipt"
        ),
        "evaluate_observation": (
            f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_evaluate_observation"
        ),
        "train_probe": f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_train_probe",
        "normal_terminal_lifecycle": (
            f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_execute"
        ),
        "operational_terminal_lifecycle": (
            f"{FROZEN_V1_RUNNER_RELATIVE_PATH}::_terminal_failure"
        ),
    },
    "control_transfer": {
        "runner_reaches_frozen_base_main_directly": True,
        "launcher_reaches_rebound_frozen_base_launcher_directly": True,
        "predecessor_main_or_later_rebind_called": False,
    },
    "reviewed_source_witness": {
        "source_map_installed_before_inherited_execute_body": True,
        "path": FROZEN_V1_MODEL_TEST_RELATIVE_PATH,
        "file_sha256": FROZEN_V1_MODEL_TEST_FILE_SHA256,
        "runtime_value_non_null_and_exact": True,
        "fallback_hard_coded_sha_used": False,
    },
    "scientific_identity": copy.deepcopy(FROZEN_V1_SCIENTIFIC_IDENTITY_SHA256),
    "source_closure": {
        "frozen_v1_source_count": FROZEN_V1_SOURCE_COUNT,
        "additive_v2_source_count": 5,
        "total_source_count": 103,
        "frozen_v1_sources_unchanged": True,
    },
    "source_only_access": {
        "generated_inputs_or_runtime_rows_opened": [],
        "checkpoints_tensors_traces_or_predecessor_runtime_outputs_opened": [],
        "accelerators_queried_or_used": [],
        "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
    },
}
# Stable aliases let runner, reviewer, and authorization code reuse one exact
# requirements object without reproducing the delegation contract.
DELEGATION_PREFLIGHT_REQUIREMENTS = RUNTIME_DELEGATION_PREFLIGHT_REQUIREMENTS
IMPORT_PREFLIGHT_REQUIREMENTS = RUNTIME_DELEGATION_PREFLIGHT_REQUIREMENTS

SOURCE_ONLY_REVIEW_CHECKS = {
    "generated_inputs_or_runtime_rows_opened": [],
    "checkpoints_tensors_traces_or_predecessor_runtime_outputs_opened": [],
    "accelerators_queried_or_used": [],
    "navigation_g2_heldout_sealed_or_rejected_material_opened": [],
}
SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v1_manifest_and_all_98_sources_rehashed": True,
    "v1_review_and_consumed_authorization_revalidated": True,
    (
        "v1_terminal_audit_exact_zero_updates_zero_presentations_and_"
        "mechanism_untested"
    ): True,
    "v1_permanently_closed_and_v2_is_not_a_retry_or_resume": True,
    "final_runtime_delegation_is_the_only_implementation_delta": True,
    "exact_v1_model_and_model_test_witness_reused_without_new_model_file": True,
    "all_nine_v1_scientific_identity_hashes_exact": True,
    "data_seed_initialization_losses_thresholds_and_caps_equal_v1": True,
    (
        "normal_and_operational_receipt_structures_inventories_and_"
        "semantics_preserved"
    ): True,
    "distinct_absent_before_reservation_v2_output_root": True,
    "isolated_delegation_preflight_passed_without_runtime_inputs": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def v1_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V1_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V1_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V1_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V1_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V1_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V1_TERMINAL_AUDIT_STATUS,
        "classification": V1_TERMINAL_AUDIT_CLASSIFICATION,
    }


def frozen_v1_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V1_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT,
        "source_bindings_sha256": FROZEN_V1_SOURCE_BINDINGS_SHA256,
        "source_count": FROZEN_V1_SOURCE_COUNT,
    }


def frozen_v1_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V1_REVIEW_COMMIT,
        "file_sha256": FROZEN_V1_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V1_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V1_REVIEW_BYTE_COUNT,
    }


def frozen_v1_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V1_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V1_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V1_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V1_AUTHORIZATION_BYTE_COUNT,
    }


def delegation_preflight_requirements() -> dict[str, Any]:
    return copy.deepcopy(RUNTIME_DELEGATION_PREFLIGHT_REQUIREMENTS)


def frozen_v1_scientific_identity_sha256() -> dict[str, str]:
    """Recompute and return the exact nine frozen V1 science hashes."""

    actual = {
        "science_contract": _v1.canonical_json_sha256(_v1.science_contract()),
        "model": _v1.canonical_json_sha256(_v1.model_config()),
        "objective": _v1.canonical_json_sha256(_v1.objective_contract()),
        "optimizer": _v1.canonical_json_sha256(_v1.optimizer_contract()),
        "schedule": _v1.canonical_json_sha256(_v1.build_schedule_identity()),
        "gate_thresholds": _v1.canonical_json_sha256(_v1.GATE_THRESHOLDS),
        "work_accounting": _v1.canonical_json_sha256(
            _v1.WORK_ACCOUNTING_CONTRACT
        ),
        "warning_policy": _v1.canonical_json_sha256(_v1.WARNING_POLICY),
        "runtime_input_template": _v1.canonical_json_sha256(
            _v1.runtime_authorization_template()
        ),
    }
    if actual != FROZEN_V1_SCIENTIFIC_IDENTITY_SHA256:
        raise PermissionError("frozen V1 scientific identity changed")
    return actual


def validate_frozen_v1_source_and_authority(
    root: Path = ROOT,
) -> dict[str, str]:
    """Revalidate the frozen V1 source, review, and consumed authority."""

    manifest_raw = _read_regular_source(
        root / FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH
    )
    if (
        len(manifest_raw) != FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(manifest_raw).hexdigest()
        != FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V1 source manifest raw identity changed")
    manifest = _v1.validate_source_manifest(manifest_raw, root)
    if (
        manifest.get("status") != "PASS_SOURCE_CLOSURE"
        or manifest.get("content_sha256")
        != FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("source_bindings_sha256")
        != FROZEN_V1_SOURCE_BINDINGS_SHA256
        or manifest.get("source_count") != FROZEN_V1_SOURCE_COUNT
    ):
        raise PermissionError("frozen V1 source manifest conclusion changed")
    manifest_binding = _v1.artifact_binding(
        FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256,
    )

    review_raw = _read_regular_source(root / FROZEN_V1_REVIEW_RELATIVE_PATH)
    if (
        len(review_raw) != FROZEN_V1_REVIEW_BYTE_COUNT
        or hashlib.sha256(review_raw).hexdigest() != FROZEN_V1_REVIEW_FILE_SHA256
    ):
        raise PermissionError("frozen V1 review raw identity changed")
    review = _v1.validate_review(review_raw, manifest_binding, root=root)
    if review.get("content_sha256") != FROZEN_V1_REVIEW_CONTENT_SHA256:
        raise PermissionError("frozen V1 review conclusion changed")
    review_binding = _v1.artifact_binding(
        FROZEN_V1_REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=FROZEN_V1_REVIEW_CONTENT_SHA256,
    )

    authorization_raw = _read_regular_source(
        root / FROZEN_V1_AUTHORIZATION_RELATIVE_PATH
    )
    if (
        len(authorization_raw) != FROZEN_V1_AUTHORIZATION_BYTE_COUNT
        or hashlib.sha256(authorization_raw).hexdigest()
        != FROZEN_V1_AUTHORIZATION_FILE_SHA256
    ):
        raise PermissionError("frozen V1 authorization raw identity changed")
    authorization = _v1.validate_authorization(
        authorization_raw, review_binding, root=root
    )
    if (
        authorization.get("content_sha256")
        != FROZEN_V1_AUTHORIZATION_CONTENT_SHA256
    ):
        raise PermissionError("frozen V1 authorization conclusion changed")
    frozen_v1_scientific_identity_sha256()
    source_bindings = {
        binding["path"]: binding["file_sha256"]
        for binding in manifest["source_bindings"]
    }
    if (
        source_bindings.get(FROZEN_V1_MODEL_TEST_RELATIVE_PATH)
        != FROZEN_V1_MODEL_TEST_FILE_SHA256
    ):
        raise PermissionError("frozen V1 reviewed model-test witness changed")
    return source_bindings


def model_config() -> dict[str, Any]:
    return _v1.model_config()


def objective_contract() -> dict[str, Any]:
    return _v1.objective_contract()


def optimizer_contract() -> dict[str, Any]:
    return _v1.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v1.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v1.runtime_authorization_template()


def science_contract() -> dict[str, Any]:
    """Return frozen V1 science with only V2 governance identities changed."""

    frozen_hashes = frozen_v1_scientific_identity_sha256()
    value = copy.deepcopy(_v1.science_contract())
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["governing_documents"] = {
        **value["governing_documents"],
        "v2_preregistration": preregistration_binding(),
        "v1_source_manifest": frozen_v1_source_manifest_binding(),
        "v1_source_review": frozen_v1_review_binding(),
        "v1_consumed_authorization": frozen_v1_authorization_binding(),
        "v1_terminal_audit": v1_terminal_audit_binding(),
    }
    value["receipts"] = copy.deepcopy(value["receipts"])
    value["receipts"]["schemas"] = {
        "reservation": RESERVATION_SCHEMA,
        "metrics": METRICS_SCHEMA,
        "artifact": ARTIFACT_SCHEMA,
        "access": ACCESS_SCHEMA,
        "result": RESULT_SCHEMA,
        "completion": COMPLETION_SCHEMA,
        "failure": FAILURE_SCHEMA,
    }
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "integrity_replacement_of": _v1.EXPERIMENT_ID,
        "v1_retry": False,
    }
    value["integrity_replacement"] = {
        **copy.deepcopy(INTEGRITY_DELTA),
        "frozen_v1_science_contract_sha256": FROZEN_V1_SCIENCE_CONTRACT_SHA256,
        "frozen_v1_science_component_sha256": copy.deepcopy(
            FROZEN_V1_SCIENCE_COMPONENT_SHA256
        ),
        "frozen_v1_scientific_identity_sha256": frozen_hashes,
    }
    return value


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    """Validate only public committed governing and V1 identity documents."""

    result = _v1.validate_governing_documents(root)
    validate_frozen_v1_source_and_authority(root)

    prereg_raw = _read_regular_source(root / PREREGISTRATION_RELATIVE_PATH)
    if (
        len(prereg_raw) != PREREGISTRATION_BYTE_COUNT
        or hashlib.sha256(prereg_raw).hexdigest() != PREREGISTRATION_FILE_SHA256
    ):
        raise PermissionError("V2 preregistration changed")

    audit_raw = _read_regular_source(root / V1_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(audit_raw) != V1_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest()
        != V1_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V1 terminal audit raw identity changed")
    audit = _v1.parse_canonical_json(audit_raw, name="V1 terminal audit")
    attempt = audit.get("attempt", {})
    runtime = audit.get("runtime_and_access", {})
    conclusion = audit.get("scientific_conclusion", {})
    closure = audit.get("closure", {})
    receipt = audit.get("receipt_validation", {})
    failure_receipt = audit.get("failure_receipt", {})
    if (
        audit.get("content_sha256") != V1_TERMINAL_AUDIT_CONTENT_SHA256
        or audit.get("status") != V1_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V1_TERMINAL_AUDIT_CLASSIFICATION
        or attempt.get("attempt_index") != 1
        or attempt.get("maximum_attempts") != 1
        or attempt.get("retry_or_resume_authorized") is not False
        or any(
            runtime.get(name) != 0
            for name in (
                "updates",
                "presentations",
                "pair_presentations_loaded",
                "objective_evaluations",
                "backward_calls",
                "online_optimizer_updates",
                "target_ema_updates",
                "joint_optimizer_updates",
                "predictor_forward_count",
                "predictor_objective_count",
                "predictor_backward_count",
                "predictor_optimizer_updates",
                "shared_gradient_gate_passes",
            )
        )
        or runtime.get("all_forbidden_semantic_counts_zero") is not True
        or runtime.get("g2_navigation_heldout_sealed_open_count") != 0
        or runtime.get("rejected_checkpoint_open_count") != 0
        or runtime.get("prior_runtime_output_open_count") != 0
        or runtime.get("training_trace_read_count") != 0
        or runtime.get("written_checkpoint_read_count") != 0
        or conclusion.get("mechanism_was_tested") is not False
        or conclusion.get("mechanism_scientifically_falsified") is not False
        or conclusion.get("mechanism_passed") is not False
        or conclusion.get("scientific_mechanism_evidence_produced") is not False
        or conclusion.get("phase_switch_reached") is not False
        or conclusion.get("first_failed_gate_update") != 0
        or closure.get("attempt_consumed") is not True
        or closure.get("mechanism_tested") is not False
        or closure.get("mechanism_scientifically_falsified") is not False
        or closure.get("predictor_or_joint_training_started") is not False
        or closure.get("family_closure_due_to_scientific_evidence") is not False
        or closure.get(
            "retry_resume_repair_same_root_reuse_alternate_seed_or_second_"
            "attempt_authorized"
        ) is not False
        or receipt.get("normal_top_level_receipt_inventory_exact") is not True
        or receipt.get("all_cross_receipt_bindings_exact") is not True
        or receipt.get("all_terminal_scientific_statuses_cross_receipt_exact")
        is not True
        or receipt.get(
            "canonical_ascii_finite_duplicate_safe_json_with_single_"
            "trailing_newline"
        )
        is not True
        or receipt.get("attempt_root_mode") != "0555"
        or failure_receipt.get("exists") is not False
    ):
        raise PermissionError(
            "V1 terminal audit no-work, receipt, or mechanism conclusion changed"
        )

    result.update({
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
        FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH:
            FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256,
        FROZEN_V1_REVIEW_RELATIVE_PATH: FROZEN_V1_REVIEW_FILE_SHA256,
        FROZEN_V1_AUTHORIZATION_RELATIVE_PATH:
            FROZEN_V1_AUTHORIZATION_FILE_SHA256,
        V1_TERMINAL_AUDIT_RELATIVE_PATH: V1_TERMINAL_AUDIT_FILE_SHA256,
    })
    return result


def validate_source_manifest(
    raw: bytes, root: Path = ROOT
) -> dict[str, Any]:
    value = _v1.parse_canonical_json(raw, name="V2 source manifest")
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
        or value.get("source_count") != 103
        or value.get("source_bindings_sha256")
        != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("V2 source manifest contract changed")
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
            raise PermissionError("V2 source binding changed")
        payload = _read_regular_source(root / relative)
        if (
            len(payload) != binding["byte_count"]
            or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"manifest-bound V2 source changed: {relative}")
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
    value = _v1.parse_canonical_json(raw, name="V2 source review")
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
        raise PermissionError("V2 review manifest binding changed")
    expected_sources = current_source_bindings(root)
    fields = {
        "schema", "status", "implementation_authors", "reviewer",
        "source_freeze_commit", "reviewed_sources", "source_manifest",
        "preregistration", "v1_source_manifest", "v1_source_review",
        "v1_consumed_authorization", "v1_terminal_audit", "science_contract",
        "runtime_delegation_preflight", "source_only_checks",
        "scientific_checks", "findings", "authority", "content_sha256",
    }
    reviewer = value.get("reviewer")
    _source_freeze_commit(
        value.get("source_freeze_commit"), name="review.source_freeze_commit"
    )
    if (
        set(value) != fields
        or value.get("schema") != REVIEW_SCHEMA
        or value.get("status")
        != "PASS_SOURCE_AND_SCIENCE_IDENTICAL_RUNTIME_DELEGATION_FIX"
        or value.get("implementation_authors") != list(IMPLEMENTATION_AUTHORS)
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in IMPLEMENTATION_AUTHORS
        or value.get("reviewed_sources") != expected_sources
        or value.get("source_manifest") != expected_manifest
        or value.get("preregistration") != preregistration_binding()
        or value.get("v1_source_manifest")
        != frozen_v1_source_manifest_binding()
        or value.get("v1_source_review") != frozen_v1_review_binding()
        or value.get("v1_consumed_authorization")
        != frozen_v1_authorization_binding()
        or value.get("v1_terminal_audit") != v1_terminal_audit_binding()
        or value.get("science_contract") != science_contract()
        or value.get("runtime_delegation_preflight")
        != RUNTIME_DELEGATION_PREFLIGHT_REQUIREMENTS
        or value.get("source_only_checks") != SOURCE_ONLY_REVIEW_CHECKS
        or value.get("scientific_checks") != SCIENTIFIC_REVIEW_CHECKS
        or value.get("findings") != []
        or value.get("authority") != REVIEW_AUTHORITY
    ):
        raise PermissionError(
            "V2 source review did not pass exact runtime-delegation scope"
        )
    return dict(value)


def validate_authorization(
    raw: bytes,
    review_binding: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = _v1.parse_canonical_json(raw, name="V2 execution authorization")
    expected_review = _validate_artifact_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    review_raw = _read_regular_source(root / REVIEW_RELATIVE_PATH)
    if expected_review != artifact_binding(
        REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=_v1.parse_canonical_json(
            review_raw, name="V2 source review"
        )["content_sha256"],
    ):
        raise PermissionError("V2 authorization review binding changed")
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
        "independent_source_review", "preregistration", "v1_terminal_audit",
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
        or value.get("v1_terminal_audit") != v1_terminal_audit_binding()
        or value.get("runtime_inputs") != runtime_authorization_template()
        or value.get("experiment") != science_contract()
        or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("V2 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v1.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity", "current_source_bindings",
    "delegation_preflight_requirements", "frozen_v1_authorization_binding",
    "frozen_v1_review_binding", "frozen_v1_scientific_identity_sha256",
    "frozen_v1_source_manifest_binding", "model_config",
    "objective_contract", "optimizer_contract", "preregistration_binding",
    "runtime_authorization_template", "science_contract",
    "v1_terminal_audit_binding", "validate_authorization",
    "validate_frozen_v1_source_and_authority", "validate_governing_documents",
    "validate_review", "validate_source_manifest",
})
