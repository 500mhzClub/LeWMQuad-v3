"""Source-only contract for the Direct BEV V4 hook-integrity successor.

V4 source-loads the frozen V3 contract and changes no scientific mechanism.
Its sole implementation delta is the update-zero call-count witness described
by the committed amendment: the inherited probe hooks the exact once-per-all-
actions ``residual_head`` instead of the bypassed outer predictor module.

Importing this module performs source reads only.  It grants no execution,
generated-input, checkpoint, tensor, GPU, navigation, held-out, or sealed
authority.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V3_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
_V3_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v4_hook_integrity_frozen_v3_contract",
    ROOT / FROZEN_V3_CONTRACT_RELATIVE_PATH,
)
if _V3_SPEC is None or _V3_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V3 source-only contract")
_v3 = importlib.util.module_from_spec(_V3_SPEC)
sys.modules[_V3_SPEC.name] = _v3
_V3_SPEC.loader.exec_module(_v3)

for _name in _v3.__all__:
    globals()[_name] = getattr(_v3, _name)
with_content_sha256 = _v3.with_content_sha256


IMPLEMENTATION_AUTHOR = "/root/plan_efficiency"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity"
)

FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH = _v3.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V3_SOURCE_MANIFEST_COMMIT = (
    "551bff22aaf69a369f887b9984cccb97e2ffd90a"
)
FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256 = (
    "5669974b4a4b5410bc06ad6f0d3c0038b418755b024be18bfa0b8ece4aa01cb3"
)
FROZEN_V3_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "bbe08609e842d1465a3983aa6f3965b5d0cd37511d28193aef50a795bbd0811f"
)
FROZEN_V3_SOURCE_MANIFEST_BYTE_COUNT = 26_713
FROZEN_V3_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V3_SOURCE_COUNT = 83

FROZEN_V3_REVIEW_RELATIVE_PATH = _v3.REVIEW_RELATIVE_PATH
FROZEN_V3_REVIEW_COMMIT = "38cb19326d5c01f55e8d5f596cc45d327354f9d1"
FROZEN_V3_REVIEW_FILE_SHA256 = (
    "29f7512f992c89bbe1fce1e3dc4df62d1d83da9794d9cbafdae25fe5d0845418"
)
FROZEN_V3_REVIEW_CONTENT_SHA256 = (
    "9deaa7450b2cdf5822b340be683e9933720fc8ef4e2834180718ce7969c19e30"
)
FROZEN_V3_REVIEW_BYTE_COUNT = 42_157
FROZEN_V3_REVIEW_STATUS = "PASS_SOURCE_AND_PREDICTOR_ONLY_SCIENCE"

FROZEN_V3_AUTHORIZATION_RELATIVE_PATH = _v3.AUTHORIZATION_RELATIVE_PATH
FROZEN_V3_AUTHORIZATION_COMMIT = (
    "0a1888406aaa27de4ddf276c37204066c154e36f"
)
FROZEN_V3_AUTHORIZATION_FILE_SHA256 = (
    "dd09784bbdfebbf577634b8cd014adcce06a891cc3c939486b2b5b4307bd7f5b"
)
FROZEN_V3_AUTHORIZATION_CONTENT_SHA256 = (
    "25976dfcd1203717dec8e99dc9054724d185a48000858cf3ffcc9ffb09f6d32d"
)
FROZEN_V3_AUTHORIZATION_BYTE_COUNT = 37_345
FROZEN_V3_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V3_FILM_UNET_PROBE"
)

V3_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor_terminal_audit_2026-07-26.json"
)
V3_TERMINAL_AUDIT_COMMIT = "2496bfac12c3841c2ead46cb582bc1a25a9ce2b2"
V3_TERMINAL_AUDIT_FILE_SHA256 = (
    "c298a56fe3f4c7ab9d7a02447f6dfdd16ad28c0909b6cd67d6c2b0900bd1f324"
)
V3_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "ee7caf34174bdab3fbaf8765950140cc09eb618dda59bfc21dc285062e64d203"
)
V3_TERMINAL_AUDIT_BYTE_COUNT = 7_684
V3_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_UPDATE_ZERO_INTEGRITY_INSTRUMENTATION_FAILURE_CLOSES_V3_"
    "NO_RETRY"
)
V3_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_UPDATE_ZERO_INTEGRITY_HARNESS_FAILURE_OUTER_PREDICTOR_FORWARD_"
    "HOOK_INCOMPATIBLE_WITH_DIRECT_ALL_ACTIONS_METHOD_ZERO_SCIENTIFIC_WORK_"
    "V3_PERMANENTLY_CLOSED"
)
FROZEN_V3_INITIAL_MODEL_STATE_SHA256 = (
    "84748bc66f0639b9dae1c81880f5c0fa756f4c4d9e75d0ffddac1310c7d05d0a"
)

INTEGRITY_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_amendment_2026-07-26.md"
)
INTEGRITY_AMENDMENT_COMMIT = "beb26e475a8939c1e23c35249c04917060522381"
INTEGRITY_AMENDMENT_FILE_SHA256 = (
    "87c3c43cf44e71e95e4c9ee5315d721799613f95e528278743be7f9043f071b2"
)
INTEGRITY_AMENDMENT_BYTE_COUNT = 6_465

FROZEN_V3_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _v3.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)
FROZEN_V3_LAUNCHER_RELATIVE_PATH = _v3.LAUNCHER_RELATIVE_PATH
FROZEN_V3_MODEL_RELATIVE_PATH = _v3.MODEL_RELATIVE_PATH

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)
MODEL_RELATIVE_PATH = FROZEN_V3_MODEL_RELATIVE_PATH
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_runner.py"
)
LAUNCHER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_launch_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_source_closure.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity_execution_authorization_2026-07-26.json"
)

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    LAUNCHER_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v3.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V3_REVIEW_RELATIVE_PATH,
    FROZEN_V3_AUTHORIZATION_RELATIVE_PATH,
    V3_TERMINAL_AUDIT_RELATIVE_PATH,
    INTEGRITY_AMENDMENT_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v4_"
    "residual_head_hook_integrity"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V4_"
    "RESIDUAL_HEAD_HOOK_INTEGRITY_PREFLIGHT_JSON"
)

RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
AUTHORIZATION_STATUS = "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V4_HOOK_INTEGRITY_PROBE"

PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **_v3.EXECUTION_AUTHORITY,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v3_retry_authorized": False,
    "v3_checkpoint_or_runtime_output_reuse_authorized": False,
    "science_identical_hook_integrity_replacement_only": True,
}

# All runner-facing science remains the frozen V3 object/value.  Explicit
# aliases make accidental V4 drift mechanically visible to callers and tests.
PREDICTOR_CONFIG = _v3.PREDICTOR_CONFIG
MODEL_PARAMETER_INVENTORY = _v3.MODEL_PARAMETER_INVENTORY
GATE_THRESHOLDS = _v3.GATE_THRESHOLDS
GATE_CONTROLS = _v3.GATE_CONTROLS
CONTROL_UPDATE_ZERO_FAIL = _v3.CONTROL_UPDATE_ZERO_FAIL
CONTROL_UPDATE_100_FAIL = _v3.CONTROL_UPDATE_100_FAIL
CONTROL_CONTINUE_UPDATE_100 = _v3.CONTROL_CONTINUE_UPDATE_100
CONTROL_UPDATE_400_FAIL = _v3.CONTROL_UPDATE_400_FAIL
CONTROL_CONTINUE_UPDATE_400 = _v3.CONTROL_CONTINUE_UPDATE_400
CONTROL_UPDATE_1000_FAIL = _v3.CONTROL_UPDATE_1000_FAIL
CONTROL_PASS = _v3.CONTROL_PASS
FAILURE_CONTROLS = _v3.FAILURE_CONTROLS

INTEGRITY_DELTA = {
    "scope": "update_zero_predictor_call_count_hook_witness_only",
    "frozen_probe_hook_registration_expression": "model.predictor",
    "v3_hook_target": "real_model.predictor",
    "v4_hook_target": "real_model.predictor.residual_head",
    "v3_training_call": (
        "real_model.predictor.predict_all_actions(current_state_logits)"
    ),
    "expected_outer_predictor_forward_hook_count": 0,
    "expected_residual_head_forward_hook_count": 1,
    "expected_training_objective_call_counts": {
        "online_state_stack": 3,
        "predictor": 1,
        "target_state_stack": 3,
    },
    "residual_head_topology": {
        "registered_module": True,
        "type": "Conv2d",
        "in_channels": 16,
        "out_channels": 3,
        "kernel_size": [3, 3],
        "padding": [1, 1],
        "bias": True,
    },
    "view_lifetime": "update_zero_gradient_probe_only_then_discarded",
    "non_predictor_view_attributes_object_identical": True,
    "model_output_gradient_parameter_buffer_rng_or_state_changed": False,
    "scientific_objective_call_changed": False,
    "model_data_seed_initialization_schedule_losses_optimizer_ema_changed": False,
    "gates_thresholds_accounting_receipts_or_caps_changed": False,
    "frozen_v3_initial_model_state_sha256": (
        FROZEN_V3_INITIAL_MODEL_STATE_SHA256
    ),
    "fresh_v4_state_must_match_frozen_v3_initial_state": True,
    "v3_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    "v3_permanently_closed": True,
}

SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v3_manifest_and_all_83_sources_rehashed": True,
    "frozen_v3_review_and_execution_authorization_exact": True,
    "v3_terminal_audit_exact_and_zero_scientific_work": True,
    "v3_permanently_closed_and_v4_is_not_a_retry": True,
    "residual_head_hook_witness_is_the_only_implementation_delta": True,
    "model_data_seed_initialization_schedule_losses_optimizer_ema_exact": True,
    "observations_metrics_gates_thresholds_accounting_receipts_and_caps_exact": True,
    "fresh_initial_state_must_match_frozen_v3": True,
    "v3_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    "distinct_absent_before_reservation_v4_output_root": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v3_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V3_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V3_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V3_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V3_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V3_SOURCE_COUNT,
    }


def frozen_v3_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V3_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V3_REVIEW_COMMIT,
        "file_sha256": FROZEN_V3_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V3_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V3_REVIEW_BYTE_COUNT,
        "status": FROZEN_V3_REVIEW_STATUS,
    }


def frozen_v3_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V3_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V3_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V3_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V3_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V3_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V3_AUTHORIZATION_STATUS,
    }


def v3_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V3_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V3_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V3_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V3_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V3_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V3_TERMINAL_AUDIT_STATUS,
        "classification": V3_TERMINAL_AUDIT_CLASSIFICATION,
    }


def integrity_amendment_binding() -> dict[str, Any]:
    return {
        "path": INTEGRITY_AMENDMENT_RELATIVE_PATH,
        "commit": INTEGRITY_AMENDMENT_COMMIT,
        "file_sha256": INTEGRITY_AMENDMENT_FILE_SHA256,
        "byte_count": INTEGRITY_AMENDMENT_BYTE_COUNT,
    }


def validate_frozen_v3_source_closure(root: Path = ROOT) -> dict[str, str]:
    read = _v3._v2._v1._read_regular_source
    raw = read(root / FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH)
    if (
        len(raw) != FROZEN_V3_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(raw).hexdigest()
        != FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V3 source manifest raw identity changed")
    manifest = _v3.validate_source_manifest(raw)
    if (
        manifest.get("content_sha256")
        != FROZEN_V3_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("status") != FROZEN_V3_SOURCE_MANIFEST_STATUS
        or manifest.get("source_count") != FROZEN_V3_SOURCE_COUNT
        or manifest.get("source_paths") != list(REUSED_SOURCE_PATHS)
    ):
        raise PermissionError("frozen V3 source manifest conclusion changed")
    current = _v3.current_source_bindings(root)
    if current.get(FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH) != (
        FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("current V3 source manifest changed")
    for binding in manifest["source_bindings"]:
        if current.get(binding["path"]) != binding["file_sha256"]:
            raise PermissionError(f"current V3 source changed: {binding['path']}")
    return current


def model_config() -> dict[str, Any]:
    return _v3.model_config()


def objective_contract() -> dict[str, Any]:
    return _v3.objective_contract()


def optimizer_contract() -> dict[str, Any]:
    return _v3.optimizer_contract()


def build_schedule_identity() -> dict[str, Any]:
    return _v3.build_schedule_identity()


def runtime_authorization_template() -> dict[str, Any]:
    return _v3.runtime_authorization_template()


def evaluate_gate(
    update: int,
    metrics: Mapping[str, Any],
    *,
    update_zero: Mapping[str, Any] | None = None,
    prior_gates_passed: bool = True,
) -> dict[str, Any]:
    return _v3.evaluate_gate(
        update,
        metrics,
        update_zero=update_zero,
        prior_gates_passed=prior_gates_passed,
    )


def science_contract() -> dict[str, Any]:
    value = _v3.science_contract()
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v3_source_manifest": frozen_v3_source_manifest_binding(),
        "frozen_v3_source_review": frozen_v3_review_binding(),
        "frozen_v3_execution_authorization": (
            frozen_v3_authorization_binding()
        ),
        "v3_terminal_audit": v3_terminal_audit_binding(),
        "v4_integrity_amendment": integrity_amendment_binding(),
    }
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "integrity_successor_of": _v3.EXPERIMENT_ID,
        "v3_retry": False,
        "v3_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    }
    value["integrity_replacement"] = dict(INTEGRITY_DELTA)
    value["authority"] = {
        **value["authority"],
        "v4_execution_authorized_by_source_contract": False,
        "v3_checkpoint_or_runtime_output_reuse_authorized": False,
    }
    value["scientific_checks"] = dict(SCIENTIFIC_REVIEW_CHECKS)
    return value


def _read_and_validate_frozen_v3_review_and_authorization(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    read = _v3._v2._v1._read_regular_source
    review_raw = read(root / FROZEN_V3_REVIEW_RELATIVE_PATH)
    authorization_raw = read(root / FROZEN_V3_AUTHORIZATION_RELATIVE_PATH)
    if (
        len(review_raw) != FROZEN_V3_REVIEW_BYTE_COUNT
        or hashlib.sha256(review_raw).hexdigest() != FROZEN_V3_REVIEW_FILE_SHA256
    ):
        raise PermissionError("frozen V3 source review raw identity changed")
    if (
        len(authorization_raw) != FROZEN_V3_AUTHORIZATION_BYTE_COUNT
        or hashlib.sha256(authorization_raw).hexdigest()
        != FROZEN_V3_AUTHORIZATION_FILE_SHA256
    ):
        raise PermissionError("frozen V3 authorization raw identity changed")
    review = _v3.parse_canonical_json(review_raw, name="frozen V3 source review")
    authorization = _v3.parse_canonical_json(
        authorization_raw, name="frozen V3 authorization"
    )
    if (
        review.get("content_sha256") != FROZEN_V3_REVIEW_CONTENT_SHA256
        or review.get("status") != FROZEN_V3_REVIEW_STATUS
        or authorization.get("content_sha256")
        != FROZEN_V3_AUTHORIZATION_CONTENT_SHA256
        or authorization.get("status") != FROZEN_V3_AUTHORIZATION_STATUS
    ):
        raise PermissionError("frozen V3 review or authorization conclusion changed")
    _v3.validate_review(
        review,
        expected_sources=review["reviewed_sources"],
        source_manifest_binding=review["source_manifest"],
    )
    _v3.validate_authorization(
        authorization,
        review_binding=authorization["independent_source_review"],
        reviewer=review["reviewer"],
    )
    return dict(review), dict(authorization)


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = validate_frozen_v3_source_closure(root)
    _read_and_validate_frozen_v3_review_and_authorization(root)
    read = _v3._v2._v1._read_regular_source
    audit_raw = read(root / V3_TERMINAL_AUDIT_RELATIVE_PATH)
    amendment = read(root / INTEGRITY_AMENDMENT_RELATIVE_PATH)
    if (
        len(audit_raw) != V3_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit_raw).hexdigest() != V3_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V3 terminal audit raw identity changed")
    if (
        len(amendment) != INTEGRITY_AMENDMENT_BYTE_COUNT
        or hashlib.sha256(amendment).hexdigest()
        != INTEGRITY_AMENDMENT_FILE_SHA256
    ):
        raise PermissionError("V4 integrity amendment changed")
    audit = json.loads(audit_raw)
    core = dict(audit)
    declared = core.pop("content_sha256", None)
    accounting = audit.get("execution_accounting", {})
    diagnosis = audit.get("instrumentation_diagnosis", {})
    consequence = audit.get("scientific_consequence", {})
    state = audit.get("state_identity", {})
    zero_work = (
        "updates", "presentations", "objective_evaluations", "backward_calls",
        "optimizer_updates", "ema_updates",
    )
    if (
        declared != V3_TERMINAL_AUDIT_CONTENT_SHA256
        or canonical_json_sha256(core) != declared
        or audit.get("status") != V3_TERMINAL_AUDIT_STATUS
        or audit.get("classification") != V3_TERMINAL_AUDIT_CLASSIFICATION
        or any(accounting.get(name) != 0 for name in zero_work)
        or diagnosis.get("outer_predictor_forward_hook_expected_count") != 0
        or diagnosis.get("residual_head_forward_hook_expected_count") != 1
        or diagnosis.get("failure_is_call_count_witness_incompatibility_not_model_or_gradient_failure") is not True
        or consequence.get("v3_permanently_closed") is not True
        or consequence.get("v3_retry_resume_repair_or_checkpoint_reuse_authorized") is not False
        or consequence.get("v4_may_reuse_v3_checkpoint_tensor_trace_or_runtime_output") is not False
        or consequence.get("v4_must_construct_fresh_state") is not True
        or state.get("initial_model_state_sha256")
        != FROZEN_V3_INITIAL_MODEL_STATE_SHA256
        or state.get("terminal_model_state_sha256")
        != FROZEN_V3_INITIAL_MODEL_STATE_SHA256
        or state.get("initial_equals_terminal") is not True
    ):
        raise PermissionError("V3 terminal audit conclusion changed")
    result.update({
        FROZEN_V3_REVIEW_RELATIVE_PATH: FROZEN_V3_REVIEW_FILE_SHA256,
        FROZEN_V3_AUTHORIZATION_RELATIVE_PATH: (
            FROZEN_V3_AUTHORIZATION_FILE_SHA256
        ),
        V3_TERMINAL_AUDIT_RELATIVE_PATH: V3_TERMINAL_AUDIT_FILE_SHA256,
        INTEGRITY_AMENDMENT_RELATIVE_PATH: INTEGRITY_AMENDMENT_FILE_SHA256,
    })
    return result


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = _v3.parse_canonical_json(raw, name="V4 source manifest")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    expected_fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    paths = value.get("source_paths")
    bindings = value.get("source_bindings")
    if (
        set(value) != expected_fields
        or value.get("schema") != SOURCE_MANIFEST_SCHEMA
        or value.get("status") != "PASS_SOURCE_CLOSURE"
        or value.get("entrypoints") != list(SOURCE_MANIFEST_ENTRYPOINTS)
        or value.get("forced_dynamic_sources")
        != list(SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES)
        or value.get("excluded_runtime_categories")
        != list(PROHIBITED_RUNTIME_CATEGORIES)
        or paths != list(SOURCE_PATHS)
        or type(bindings) is not list
        or len(bindings) != len(SOURCE_PATHS)
        or value.get("source_count") != 91
        or value.get("source_bindings_sha256") != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V4 source manifest contract changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V4 source binding fields changed")
        relative = _v3._v2._v1.safe_relative_source_path(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V4 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V4 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound V4 source changed: {binding['path']}")
        result[binding["path"]] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(manifest_raw).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(
    source_manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source_manifest_binding is None:
        raw = _v3._v2._v1._read_regular_source(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        value = validate_source_manifest(raw)
        source_manifest_binding = artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=str(value["content_sha256"]),
        )
    return validate_binding(dict(source_manifest_binding), path=SOURCE_MANIFEST_RELATIVE_PATH)


def validate_review(
    value: object,
    *,
    expected_sources: Mapping[str, str],
    source_manifest_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer",
        "reviewed_sources", "source_manifest", "frozen_v3_source_manifest",
        "frozen_v3_source_review", "frozen_v3_execution_authorization",
        "v3_terminal_audit", "v4_integrity_amendment", "science_contract",
        "source_only_checks", "scientific_checks", "findings", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V4 source review fields changed")
    manifest_binding = _manifest_binding_or_read(source_manifest_binding)
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    required_reviewed = set(SOURCE_PATHS) | set(SOURCE_REVIEW_ADDITIONAL_PATHS)
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS_SOURCE_AND_SCIENCE_IDENTICAL_HOOK_INTEGRITY"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or not required_reviewed.issubset(expected_sources)
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"] != manifest_binding
        or expected_sources.get(SOURCE_MANIFEST_RELATIVE_PATH)
        != manifest_binding["file_sha256"]
        or expected_sources.get(FROZEN_V3_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V3_SOURCE_MANIFEST_FILE_SHA256
        or expected_sources.get(FROZEN_V3_REVIEW_RELATIVE_PATH)
        != FROZEN_V3_REVIEW_FILE_SHA256
        or expected_sources.get(FROZEN_V3_AUTHORIZATION_RELATIVE_PATH)
        != FROZEN_V3_AUTHORIZATION_FILE_SHA256
        or expected_sources.get(V3_TERMINAL_AUDIT_RELATIVE_PATH)
        != V3_TERMINAL_AUDIT_FILE_SHA256
        or expected_sources.get(INTEGRITY_AMENDMENT_RELATIVE_PATH)
        != INTEGRITY_AMENDMENT_FILE_SHA256
        or value["frozen_v3_source_manifest"] != frozen_v3_source_manifest_binding()
        or value["frozen_v3_source_review"] != frozen_v3_review_binding()
        or value["frozen_v3_execution_authorization"]
        != frozen_v3_authorization_binding()
        or value["v3_terminal_audit"] != v3_terminal_audit_binding()
        or value["v4_integrity_amendment"] != integrity_amendment_binding()
        or value["science_contract"] != science_contract()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "cpu_synthetic_torch_tests_permitted": True,
            "generated_inputs_opened": [],
            "checkpoints_tensors_traces_or_runtime_outputs_opened": [],
            "gpu_state_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["scientific_checks"] != SCIENTIFIC_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V4 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v3_source_manifest", "frozen_v3_source_review",
        "frozen_v3_execution_authorization", "v3_terminal_audit",
        "v4_integrity_amendment", "runtime_inputs", "experiment", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V4 execution authorization fields changed")
    expected_review = validate_binding(dict(review_binding), path=REVIEW_RELATIVE_PATH)
    core = dict(value)
    declared = core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != AUTHORIZATION_STATUS
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_source_review"] != expected_review
        or value["frozen_v3_source_manifest"] != frozen_v3_source_manifest_binding()
        or value["frozen_v3_source_review"] != frozen_v3_review_binding()
        or value["frozen_v3_execution_authorization"]
        != frozen_v3_authorization_binding()
        or value["v3_terminal_audit"] != v3_terminal_audit_binding()
        or value["v4_integrity_amendment"] != integrity_amendment_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V4 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v3.__all__,
    *(name for name in globals() if name.isupper()),
    "build_schedule_identity",
    "current_source_bindings",
    "evaluate_gate",
    "frozen_v3_authorization_binding",
    "frozen_v3_review_binding",
    "frozen_v3_source_manifest_binding",
    "integrity_amendment_binding",
    "model_config",
    "objective_contract",
    "optimizer_contract",
    "runtime_authorization_template",
    "science_contract",
    "v3_terminal_audit_binding",
    "validate_authorization",
    "validate_frozen_v3_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
    "with_content_sha256",
})
