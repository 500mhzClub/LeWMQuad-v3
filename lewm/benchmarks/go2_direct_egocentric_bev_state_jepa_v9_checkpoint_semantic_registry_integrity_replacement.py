"""Source-only contract for the science-identical Direct BEV V9 retry.

V9 reuses the complete frozen V8 perception experiment.  Its additive runner
adapts only the lexical checkpoint-semantic registry and preserves complete
failure evidence.  Importing this module reads source and committed governance
documents only; it grants no runtime, navigation, or held-out authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V8_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py"
)


def _source_only_module(name: str, relative_path: str) -> Any:
    source = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only contract {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V8 = _source_only_module(
    "_lewm_direct_bev_v9_frozen_v8_contract",
    FROZEN_V8_CONTRACT_RELATIVE_PATH,
)
_FROZEN_V8_SCIENCE_CONTRACT = _V8.science_contract()

for _name in _V8.__all__:
    globals()[_name] = getattr(_V8, _name)

canonical_json_bytes = _V8.canonical_json_bytes
canonical_json_sha256 = _V8.canonical_json_sha256
is_sha256 = _V8.is_sha256
with_content_sha256 = _V8.with_content_sha256
parse_canonical_json = _V8.parse_canonical_json
artifact_binding = _V8.artifact_binding
validate_binding = _V8.validate_binding


IMPLEMENTATION_AUTHOR = "/root/v9_contract_implementation"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement_v1"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
MODEL_RELATIVE_PATH = _V8.MODEL_RELATIVE_PATH
MODEL_TEST_RELATIVE_PATH = _V8.MODEL_TEST_RELATIVE_PATH

FROZEN_V8_RUNNER_RELATIVE_PATH = _V8.RUNNER_RELATIVE_PATH
FROZEN_V8_LAUNCHER_RELATIVE_PATH = _V8.LAUNCHER_RELATIVE_PATH
FROZEN_V8_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V8.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement_"
    "preregistration_2026-07-27.json"
)
PREREGISTRATION_COMMIT = "ef37e850c5f7b03c31b24fb86680f5f4cb8bd95f"
PREREGISTRATION_FILE_SHA256 = (
    "b8139b980df66d5c6d5fd1f73d0656a18fb78b9bd9b9428342f7175fa6baa603"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "9cbe395fe4b192517c0977c4c3842af06645842d1914c86f5cc216c591e10cc7"
)
PREREGISTRATION_BYTE_COUNT = 13_511
PREREGISTRATION_STATUS = (
    "PREREGISTERED_ONE_FRESH_SCIENCE_IDENTICAL_V9_CHECKPOINT_SEMANTIC_"
    "REGISTRY_INTEGRITY_REPLACEMENT_PENDING_SOURCE_FREEZE_INDEPENDENT_"
    "REVIEW_AND_SEPARATE_MACHINE_AUTHORIZATION"
)

FROZEN_V8_SOURCE_MANIFEST_RELATIVE_PATH = _V8.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V8_SOURCE_MANIFEST_COMMIT = (
    "ae179bb65192eced558104ab53822162163d83d2"
)
FROZEN_V8_SOURCE_MANIFEST_FILE_SHA256 = (
    "3553309facdd9da0eef1d4e28442ef087d98677de89bd6eeec3f70eb3811c2f2"
)
FROZEN_V8_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "90d45fc2c6b4b8207de5ac0eed4181762940a2fa85eb0cfca98691efa4927cd5"
)
FROZEN_V8_SOURCE_MANIFEST_BYTE_COUNT = 42_411
FROZEN_V8_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V8_SOURCE_COUNT = 122

FROZEN_V8_REVIEW_RELATIVE_PATH = _V8.REVIEW_RELATIVE_PATH
FROZEN_V8_REVIEW_COMMIT = "d9c3d66e606d62023d2b4d8006764991803db67a"
FROZEN_V8_REVIEW_FILE_SHA256 = (
    "74f047fb6c75fd42eff75e09a6b3c0cc4009f1522f1213bdfc797dfdffe82ca5"
)
FROZEN_V8_REVIEW_CONTENT_SHA256 = (
    "f916c7bad3d2ad46a379b08d8cd0423c4165e6b458aedf2ee23ef1e6a68d2dfa"
)
FROZEN_V8_REVIEW_BYTE_COUNT = 67_864
FROZEN_V8_REVIEW_STATUS = (
    "PASS_SOURCE_AND_LEARNED_BEV_QUERY_PROTOTYPE_SCIENCE"
)

FROZEN_V8_AUTHORIZATION_RELATIVE_PATH = _V8.AUTHORIZATION_RELATIVE_PATH
FROZEN_V8_AUTHORIZATION_COMMIT = (
    "49e60fab48475492bbceea5d2b05db79eb1e63a4"
)
FROZEN_V8_AUTHORIZATION_FILE_SHA256 = (
    "7dcb8a99fe241fc095e943e3fc99654565e1df9c30fa7729422b6a38be49728b"
)
FROZEN_V8_AUTHORIZATION_CONTENT_SHA256 = (
    "d7d29a3198155d5234d479100ba6d9fc6d02d036ce1772b33ea5cb8b515b0dc4"
)
FROZEN_V8_AUTHORIZATION_BYTE_COUNT = 57_483
FROZEN_V8_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V8_LEARNED_BEV_QUERY_PROTOTYPE_"
    "PERCEPTION_FALSIFICATION"
)

V8_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder_terminal_audit_2026-07-26.json"
)
V8_TERMINAL_AUDIT_COMMIT = "5ade8a1808141e453a392f0adb7c5e90f0029830"
V8_TERMINAL_AUDIT_FILE_SHA256 = (
    "1cf43adc2362cb8a48b81d9010167e95ad3ac67e4e403ced3c6ddc63eba8e8db"
)
V8_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "22bbdb2a9d6ca76732776591679fc8beb6525d809407c8df2c10368ee4abe0ad"
)
V8_TERMINAL_AUDIT_BYTE_COUNT = 17_156
V8_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_UPDATE_50_CHECKPOINT_REGISTRY_OPERATIONAL_FAILURE_"
    "CLOSES_V8_NO_RETRY"
)
V8_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_ONE_SHOT_OPERATIONAL_INTEGRITY_FAILURE_AFTER_PASSED_UPDATE_ZERO_"
    "AND_UPDATE_50_HEALTH_GATES_BEFORE_UPDATE_100_NO_TERMINAL_PERCEPTION_"
    "RESULT_V8_PERMANENTLY_CLOSED_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement_"
    "source_manifest_2026-07-27.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement_"
    "source_review_2026-07-27.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement_"
    "execution_authorization_2026-07-27.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V8.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
if len(REUSED_SOURCE_PATHS) != 122 or len(SOURCE_PATHS) != 127:
    raise RuntimeError("V9 recursive source cardinality changed")
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V8_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V8_REVIEW_RELATIVE_PATH,
    FROZEN_V8_AUTHORIZATION_RELATIVE_PATH,
    V8_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v9/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v9_"
    "checkpoint_semantic_registry_integrity_replacement_v1"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V9_"
    "CHECKPOINT_SEMANTIC_REGISTRY_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
)

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

REVIEW_STATUS = (
    "PASS_SOURCE_SCIENCE_IDENTITY_CHECKPOINT_REGISTRY_ADAPTER_AND_COMPLETE_"
    "FAILURE_RECEIPTS"
)
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V9_CHECKPOINT_SEMANTIC_REGISTRY_"
    "INTEGRITY_REPLACEMENT"
)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **dict(_V8.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "one_fresh_v8_learned_bev_query_prototype_perception_attempt_only": False,
    "v8_retry_resume_repair_recovery_or_extension_authorized": False,
    "v8_checkpoint_tensor_trace_receipt_parameter_optimizer_or_rng_reuse_"
    "authorized": False,
    "one_fresh_v9_checkpoint_semantic_registry_integrity_replacement_only": True,
    "science_identical_to_frozen_v8": True,
    "predictor_training_or_evaluation_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "sealed_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
    "deployment_authorized": False,
}

# Reassert every scientific cap and schedule identity inherited from V8.
MAXIMUM_ATTEMPTS = 1
ATTEMPT_INDEX = 1
MAXIMUM_UPDATES = 250
MAXIMUM_PRESENTATIONS = 4_000
GPU_ACTIVE_TIME_CAP_MINUTES = 30
EFFECTIVE_BATCH_SIZE = 16
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
CHECKPOINT_UPDATES = (50, 100, 250)
SNAPSHOT_UPDATES = CHECKPOINT_UPDATES
OBSERVATION_UPDATES = (0, *CHECKPOINT_UPDATES)
SCHEDULE_PREFIX_SHA256 = dict(_V8.SCHEDULE_PREFIX_SHA256)
if (
    _V8.MODEL_RELATIVE_PATH != MODEL_RELATIVE_PATH
    or _V8.MAXIMUM_ATTEMPTS != MAXIMUM_ATTEMPTS
    or _V8.ATTEMPT_INDEX != ATTEMPT_INDEX
    or _V8.MAXIMUM_UPDATES != MAXIMUM_UPDATES
    or _V8.MAXIMUM_PRESENTATIONS != MAXIMUM_PRESENTATIONS
    or _V8.GPU_ACTIVE_TIME_CAP_MINUTES != GPU_ACTIVE_TIME_CAP_MINUTES
    or _V8.EFFECTIVE_BATCH_SIZE != EFFECTIVE_BATCH_SIZE
    or _V8.MICROBATCH_SIZE != MICROBATCH_SIZE
    or _V8.MICROBATCHES_PER_UPDATE != MICROBATCHES_PER_UPDATE
    or tuple(_V8.CHECKPOINT_UPDATES) != CHECKPOINT_UPDATES
    or tuple(_V8.OBSERVATION_UPDATES) != OBSERVATION_UPDATES
    or dict(_V8.SCHEDULE_PREFIX_SHA256) != SCHEDULE_PREFIX_SHA256
):
    raise PermissionError("frozen V8 model, schedule, or cap identity changed")

FROZEN_V8_SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(
    _FROZEN_V8_SCIENCE_CONTRACT
)
if FROZEN_V8_SCIENCE_CONTRACT_SHA256 != (
    "bacb31b0eb2070821bbd37862e6f3b9a39d7ecb0ab14ed8d758894c36f06f728"
):
    raise PermissionError("frozen V8 science contract identity changed")

INTEGRITY_REPLACEMENT_DELTA = {
    "science_changed": False,
    "scientific_delta_count": 0,
    "frozen_v8_source_modified": False,
    "model_data_seed_schedule_initialization_objective_optimizer_ema_gates_"
    "thresholds_or_caps_changed": False,
    "mechanical_seams": [
        "exception_safe_lexical_checkpoint_registry_contract_adapter",
        "identity_preserving_completed_observation_receipt_capture",
        "complete_failure_progress_receipt_delegation",
    ],
    "sole_behavioral_delta": (
        "temporarily_bind_the_lexical_base_checkpoint_registry_to_the_active_"
        "V9_contract_only_around_the_frozen_V8_snapshot_then_restore_the_exact_"
        "prior_object_in_finally"
    ),
    "complete_failure_receipt_delta": (
        "persist_every_completed_observation_gate_metric_and_available_"
        "accounting_determinism_source_authority_access_and_input_binding_"
        "without_fabrication"
    ),
    "v8_output_root": _V8.OUTPUT_ROOT_RELATIVE_PATH,
    "v9_output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v8_runtime_output_open_or_reuse_authorized": False,
    "v8_retry_resume_repair_or_recovery_authorized": False,
    "v9_maximum_attempts": 1,
}

INTEGRITY_REVIEW_CHECKS = {
    "frozen_v8_manifest_and_all_122_sources_rehashed": True,
    "frozen_v8_review_authorization_and_terminal_audit_exact": True,
    "v8_permanently_closed_and_runtime_reuse_forbidden": True,
    "v9_preregistration_exact": True,
    "v9_adds_no_model_data_loss_optimizer_schedule_gate_or_threshold_code": True,
    "v9_science_normalizes_exactly_to_frozen_v8": True,
    "scientific_delta_count_is_exactly_zero": True,
    "wrapper_topology_and_all_nonmechanical_v8_seam_identities_exact": True,
    "lexical_registry_rebind_is_narrow_and_finally_restored_by_identity": True,
    "lexical_registry_prior_object_restored_after_success_and_exception": True,
    "u50_u100_u250_registry_and_schedule_prefix_semantics_exact": True,
    "completed_observations_enter_failure_progress_before_snapshot": True,
    "completed_observation_state_is_empty_before_each_fresh_process_run": True,
    "completed_observation_state_is_an_exact_ordered_observation_prefix": True,
    "failure_context_is_truthful_complete_and_never_synthesized": True,
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v8_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V8_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V8_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V8_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V8_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V8_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V8_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V8_SOURCE_COUNT,
    }


def frozen_v8_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V8_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V8_REVIEW_COMMIT,
        "file_sha256": FROZEN_V8_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V8_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V8_REVIEW_BYTE_COUNT,
        "status": FROZEN_V8_REVIEW_STATUS,
    }


def frozen_v8_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V8_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V8_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V8_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V8_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V8_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V8_AUTHORIZATION_STATUS,
    }


def v8_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V8_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V8_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V8_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V8_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V8_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V8_TERMINAL_AUDIT_STATUS,
        "classification": V8_TERMINAL_AUDIT_CLASSIFICATION,
    }


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
        "status": PREREGISTRATION_STATUS,
    }


def frozen_v8_science_contract() -> dict[str, Any]:
    return deepcopy(_FROZEN_V8_SCIENCE_CONTRACT)


def runtime_authorization_template() -> dict[str, Any]:
    value = deepcopy(_V8.runtime_authorization_template())
    value["experiment_scope"] = {
        **value["experiment_scope"],
        "one_fresh_attempt": True,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "attempt_index": ATTEMPT_INDEX,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "maximum_active_gpu_minutes": GPU_ACTIVE_TIME_CAP_MINUTES,
        "fresh_initialization_required": True,
        "perception_only": True,
        "predictor_forward_or_training": False,
        "prior_runtime_or_checkpoint_reuse": False,
        "v8_runtime_output_reuse": False,
        "v8_retry_resume_repair_or_recovery": False,
        "output_root_must_be_absent_before_reservation": True,
        "reservation_consumes_the_sole_attempt": True,
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt": (
            False
        ),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    }
    return value


def science_contract() -> dict[str, Any]:
    """Return the byte/canonical-identical frozen V8 scientific contract."""

    return frozen_v8_science_contract()


def normalize_v9_operational_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or dict(value) != science_contract():
        raise PermissionError("V9 experiment differs from its exact contract")
    return frozen_v8_science_contract()


def science_identity_receipt() -> dict[str, Any]:
    value = science_contract()
    normalized = normalize_v9_operational_identity(value)
    return {
        "frozen_v8_science_contract_sha256": (
            FROZEN_V8_SCIENCE_CONTRACT_SHA256
        ),
        "v9_science_contract_sha256": canonical_json_sha256(value),
        "normalized_v9_science_contract_sha256": canonical_json_sha256(
            normalized
        ),
        "normalized_exactly_equals_frozen_v8": (
            normalized == _FROZEN_V8_SCIENCE_CONTRACT
        ),
        "scientific_delta_count": 0,
        "mechanical_seams": list(INTEGRITY_REPLACEMENT_DELTA["mechanical_seams"]),
        "sole_behavioral_delta": INTEGRITY_REPLACEMENT_DELTA[
            "sole_behavioral_delta"
        ],
    }


def _read_bound_json(
    relative_path: str,
    *,
    file_sha256: str,
    content_sha256: str,
    byte_count: int,
    status: str,
    classification: str | None = None,
) -> dict[str, Any]:
    read = _V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
    raw = read(ROOT / relative_path)
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != file_sha256:
        raise PermissionError(f"governing document changed: {relative_path}")
    value = json.loads(raw)
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        declared != content_sha256
        or canonical_json_sha256(core) != content_sha256
        or value.get("status") != status
        or (
            classification is not None
            and value.get("classification") != classification
        )
    ):
        raise PermissionError(f"governing conclusion changed: {relative_path}")
    return dict(value)


def validate_frozen_v8_source_closure(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("V9 frozen V8 closure must use repository root")
    manifest = _read_bound_json(
        FROZEN_V8_SOURCE_MANIFEST_RELATIVE_PATH,
        file_sha256=FROZEN_V8_SOURCE_MANIFEST_FILE_SHA256,
        content_sha256=FROZEN_V8_SOURCE_MANIFEST_CONTENT_SHA256,
        byte_count=FROZEN_V8_SOURCE_MANIFEST_BYTE_COUNT,
        status=FROZEN_V8_SOURCE_MANIFEST_STATUS,
    )
    if manifest.get("source_count") != FROZEN_V8_SOURCE_COUNT:
        raise PermissionError("frozen V8 source count changed")
    current = _V8.current_source_bindings(root)
    if current.get(FROZEN_V8_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V8_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V8 source closure changed")
    return current


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    current = validate_frozen_v8_source_closure(root)
    review = _read_bound_json(
        FROZEN_V8_REVIEW_RELATIVE_PATH,
        file_sha256=FROZEN_V8_REVIEW_FILE_SHA256,
        content_sha256=FROZEN_V8_REVIEW_CONTENT_SHA256,
        byte_count=FROZEN_V8_REVIEW_BYTE_COUNT,
        status=FROZEN_V8_REVIEW_STATUS,
    )
    authorization = _read_bound_json(
        FROZEN_V8_AUTHORIZATION_RELATIVE_PATH,
        file_sha256=FROZEN_V8_AUTHORIZATION_FILE_SHA256,
        content_sha256=FROZEN_V8_AUTHORIZATION_CONTENT_SHA256,
        byte_count=FROZEN_V8_AUTHORIZATION_BYTE_COUNT,
        status=FROZEN_V8_AUTHORIZATION_STATUS,
    )
    _V8.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _V8.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    _read_bound_json(
        V8_TERMINAL_AUDIT_RELATIVE_PATH,
        file_sha256=V8_TERMINAL_AUDIT_FILE_SHA256,
        content_sha256=V8_TERMINAL_AUDIT_CONTENT_SHA256,
        byte_count=V8_TERMINAL_AUDIT_BYTE_COUNT,
        status=V8_TERMINAL_AUDIT_STATUS,
        classification=V8_TERMINAL_AUDIT_CLASSIFICATION,
    )
    _read_bound_json(
        PREREGISTRATION_RELATIVE_PATH,
        file_sha256=PREREGISTRATION_FILE_SHA256,
        content_sha256=PREREGISTRATION_CONTENT_SHA256,
        byte_count=PREREGISTRATION_BYTE_COUNT,
        status=PREREGISTRATION_STATUS,
    )
    current.update({
        FROZEN_V8_SOURCE_MANIFEST_RELATIVE_PATH: (
            FROZEN_V8_SOURCE_MANIFEST_FILE_SHA256
        ),
        FROZEN_V8_REVIEW_RELATIVE_PATH: FROZEN_V8_REVIEW_FILE_SHA256,
        FROZEN_V8_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V8_AUTHORIZATION_FILE_SHA256
        ),
        V8_TERMINAL_AUDIT_RELATIVE_PATH: V8_TERMINAL_AUDIT_FILE_SHA256,
        PREREGISTRATION_RELATIVE_PATH: PREREGISTRATION_FILE_SHA256,
    })
    return current


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="V9 source manifest")
    fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    core = dict(value)
    declared = core.pop("content_sha256", None)
    bindings = value.get("source_bindings")
    if (
        set(value) != fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources")
        != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or value.get("source_paths") != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_count") != len(SOURCE_PATHS)
        or value.get("source_bindings_sha256")
        != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V9 source manifest contract changed")
    safe = _V8._V7._V6._v5._v4._v3._v2._v1.safe_relative_source_path
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V9 source binding fields changed")
        relative = safe(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V9 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V9 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(
                f"manifest-bound V9 source changed: {binding['path']}"
            )
        result[binding["path"]] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(
    source_manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source_manifest_binding is None:
        read = _V8._V7._V6._v5._v4._v3._v2._v1._read_regular_source
        raw = read(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        manifest = validate_source_manifest(raw)
        source_manifest_binding = artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=str(manifest["content_sha256"]),
        )
    return validate_binding(
        dict(source_manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH
    )


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    source_manifest_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer",
        "reviewed_sources", "source_manifest", "frozen_v8_source_manifest",
        "frozen_v8_source_review", "frozen_v8_execution_authorization",
        "v8_terminal_audit", "v9_preregistration", "science_contract",
        "science_identity", "source_only_checks", "integrity_checks",
        "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V9 source review fields changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    reviewer = value["reviewer"]
    required = set(SOURCE_PATHS) | set(SOURCE_REVIEW_ADDITIONAL_PATHS)
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != REVIEW_STATUS
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or not required.issubset(expected_sources)
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"]
        != _manifest_binding_or_read(source_manifest_binding)
        or value["frozen_v8_source_manifest"]
        != frozen_v8_source_manifest_binding()
        or value["frozen_v8_source_review"] != frozen_v8_review_binding()
        or value["frozen_v8_execution_authorization"]
        != frozen_v8_authorization_binding()
        or value["v8_terminal_audit"] != v8_terminal_audit_binding()
        or value["v9_preregistration"] != preregistration_binding()
        or value["science_contract"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "cpu_synthetic_torch_tests_permitted": True,
            "generated_inputs_opened": [],
            "checkpoints_tensors_traces_or_runtime_outputs_opened": [],
            "gpu_state_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["integrity_checks"] != INTEGRITY_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V9 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v8_source_manifest", "frozen_v8_source_review",
        "frozen_v8_execution_authorization", "v8_terminal_audit",
        "v9_preregistration", "runtime_inputs", "experiment",
        "science_identity", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V9 execution authorization fields changed")
    expected_review = validate_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
    core = dict(value)
    declared = core.pop("content_sha256", None)
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review
        or value["frozen_v8_source_manifest"]
        != frozen_v8_source_manifest_binding()
        or value["frozen_v8_source_review"] != frozen_v8_review_binding()
        or value["frozen_v8_execution_authorization"]
        != frozen_v8_authorization_binding()
        or value["v8_terminal_audit"] != v8_terminal_audit_binding()
        or value["v9_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V9 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V8.__all__,
    *(name for name in globals() if name.isupper()),
    "current_source_bindings",
    "frozen_v8_authorization_binding",
    "frozen_v8_review_binding",
    "frozen_v8_science_contract",
    "frozen_v8_source_manifest_binding",
    "normalize_v9_operational_identity",
    "preregistration_binding",
    "runtime_authorization_template",
    "science_contract",
    "science_identity_receipt",
    "v8_terminal_audit_binding",
    "validate_authorization",
    "validate_frozen_v8_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
