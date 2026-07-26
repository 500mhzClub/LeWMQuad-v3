"""Source-only contract for the science-identical Direct BEV V7 recovery.

V7 reuses the complete frozen V6 experiment.  Its sole behavioral change is
in the additive runner wrapper: after installing the reviewed V6 seams, the
wrapper delegates directly to the deepest V1 entry points so intermediate
successor wrappers cannot reinstall an older initializer.  Importing this
module reads source and committed governance documents only.
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
FROZEN_V6_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
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


_V6 = _source_only_module(
    "_lewm_direct_bev_v7_frozen_v6_contract",
    FROZEN_V6_CONTRACT_RELATIVE_PATH,
)
_FROZEN_V6_SCIENCE_CONTRACT = _V6.science_contract()

for _name in _V6.__all__:
    globals()[_name] = getattr(_V6, _name)

canonical_json_bytes = _V6.canonical_json_bytes
canonical_json_sha256 = _V6.canonical_json_sha256
is_sha256 = _V6.is_sha256
with_content_sha256 = _V6.with_content_sha256
parse_canonical_json = _V6.parse_canonical_json
artifact_binding = _V6.artifact_binding
validate_binding = _V6.validate_binding


IMPLEMENTATION_AUTHOR = "/root/v7_minimal_implementation"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement"
)
EXPERIMENT_ID = (
    "go2_rgb_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement_source_closure.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
RUNNER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
LAUNCHER_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
SOURCE_CLOSURE_TEST_RELATIVE_PATH = TEST_RELATIVE_PATH
MODEL_RELATIVE_PATH = _V6.MODEL_RELATIVE_PATH

FROZEN_V6_RUNNER_RELATIVE_PATH = _V6.RUNNER_RELATIVE_PATH
FROZEN_V6_LAUNCHER_RELATIVE_PATH = _V6.LAUNCHER_RELATIVE_PATH
FROZEN_V6_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    _V6.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement_preregistration_2026-07-26.json"
)
PREREGISTRATION_COMMIT = "50e5f49aebcde5973902eb42766159d80e9a3e66"
PREREGISTRATION_FILE_SHA256 = (
    "4cfb40ddaf8d5d0bbca22038c15633dcb5e89a8661260fee5791ef5103e64635"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "3b81cdb18a29f005e82f7b59463436f8bf037f3e6924a5b5af7a5e66247c4152"
)
PREREGISTRATION_BYTE_COUNT = 17_693

FROZEN_V6_SOURCE_MANIFEST_RELATIVE_PATH = _V6.SOURCE_MANIFEST_RELATIVE_PATH
FROZEN_V6_SOURCE_MANIFEST_COMMIT = (
    "d8da3404d4382b0c1b427504ad73e0a6a8dcb462"
)
FROZEN_V6_SOURCE_MANIFEST_FILE_SHA256 = (
    "2d1fda7c437156877f9a3d89a8f7b276c92faaa4ba372e1ca31ba0dca5136861"
)
FROZEN_V6_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "aef47b3ae970d38d9a7851a924b4562629704bd68d7abe21761465ef52278a01"
)
FROZEN_V6_SOURCE_MANIFEST_BYTE_COUNT = 38_109
FROZEN_V6_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V6_SOURCE_COUNT = 111

FROZEN_V6_REVIEW_RELATIVE_PATH = _V6.REVIEW_RELATIVE_PATH
FROZEN_V6_REVIEW_COMMIT = "6c96135af925f87a63b3bd36af37f38d367a06ed"
FROZEN_V6_REVIEW_FILE_SHA256 = (
    "88d2f07da9bdb8a7284069690de6cc65e09c4e3abf6752207390cd4e17ff750b"
)
FROZEN_V6_REVIEW_CONTENT_SHA256 = (
    "ea4b8ea5479e753002fa7bb9bdf35a1424b31a272a6560cd309854ceca3993d6"
)
FROZEN_V6_REVIEW_BYTE_COUNT = 57_240
FROZEN_V6_REVIEW_STATUS = (
    "PASS_SOURCE_AND_PHASE_SEPARATED_FROZEN_STATE_PREDICTION_SCIENCE"
)

FROZEN_V6_AUTHORIZATION_RELATIVE_PATH = _V6.AUTHORIZATION_RELATIVE_PATH
FROZEN_V6_AUTHORIZATION_COMMIT = (
    "6daeab8b10fcc7a63e26c86f75e51dfc8c3bd4dc"
)
FROZEN_V6_AUTHORIZATION_FILE_SHA256 = (
    "00bb82292ef52b5005ac8bb18ad582f14c0266786a40637abc560b2fb98904a1"
)
FROZEN_V6_AUTHORIZATION_CONTENT_SHA256 = (
    "79e12a84a5e2a2f142dd6b31eab7d34cbfd4cd1bbbad294825895350211070c5"
)
FROZEN_V6_AUTHORIZATION_BYTE_COUNT = 45_278
FROZEN_V6_AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V6_PHASE_SEPARATED_FROZEN_STATE_"
    "PREDICTION_PROBE"
)

V6_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_terminal_audit_2026-07-26.json"
)
V6_TERMINAL_AUDIT_COMMIT = "690c24602de3345dd73e879ef46a095406665f0c"
V6_TERMINAL_AUDIT_FILE_SHA256 = (
    "6e0e28dda42eda83308d4b25a30ff6d2757e4f24cc7d200a596a243e7113b849"
)
V6_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "87ffbef5cfaee479549a33ccb1163528cc4ccc416efcb21b5b8804bce04d253c"
)
V6_TERMINAL_AUDIT_BYTE_COUNT = 7_789
V6_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_OPERATIONAL_REBIND_FAILURE_AT_ZERO_SCIENTIFIC_"
    "WORK_CLOSES_V6_NO_RETRY"
)

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement_execution_authorization_2026-07-26.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted({
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
}))
REUSED_SOURCE_PATHS = tuple(sorted(set(_V6.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V6_SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V6_REVIEW_RELATIVE_PATH,
    FROZEN_V6_AUTHORIZATION_RELATIVE_PATH,
    V6_TERMINAL_AUDIT_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v7/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v7_"
    "runner_integrity_replacement"
)
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V7_"
    "RUNNER_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
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

REVIEW_STATUS = "PASS_SOURCE_SCIENCE_IDENTITY_AND_RUNNER_INTEGRITY"
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V7_RUNNER_INTEGRITY_REPLACEMENT"
)
PRESENT_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    **dict(_V6.EXECUTION_AUTHORITY),
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v6_retry_resume_or_repair_authorized": False,
    "v6_checkpoint_tensor_trace_receipt_or_runtime_output_reuse_authorized": (
        False
    ),
    "science_identical_v7_runner_integrity_replacement_only": True,
}

FROZEN_V6_SCIENCE_CONTRACT_SHA256 = canonical_json_sha256(
    _FROZEN_V6_SCIENCE_CONTRACT
)
FROZEN_V6_AUTHORIZATION_EXPERIMENT_SHA256 = (
    "bb2fa98619ad848832638ee13bc871663abb99d89dd0f1d2758ae7c0ade3708e"
)
if FROZEN_V6_SCIENCE_CONTRACT_SHA256 != (
    FROZEN_V6_AUTHORIZATION_EXPERIMENT_SHA256
):
    raise PermissionError("frozen V6 science contract identity changed")

INTEGRITY_REPLACEMENT_DELTA = {
    "science_changed": False,
    "sole_behavioral_delta": (
        "install_complete_frozen_v6_seams_then_delegate_directly_to_deepest_"
        "v1_entrypoints"
    ),
    "model_data_seed_schedule_initialization_objectives_optimizer_gates_"
    "thresholds_or_caps_changed": False,
    "v6_runtime_output_open_or_reuse_authorized": False,
    "v6_output_root": _V6.OUTPUT_ROOT_RELATIVE_PATH,
    "v7_output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v6_retry_resume_or_repair_authorized": False,
    "v7_maximum_attempts": 1,
}

INTEGRITY_REVIEW_CHECKS = {
    "frozen_v6_manifest_and_all_111_sources_rehashed": True,
    "frozen_v6_review_authorization_and_terminal_audit_exact": True,
    "v6_closed_at_zero_scientific_work_and_runtime_reuse_forbidden": True,
    "v7_preregistration_exact": True,
    "v7_adds_no_model_loss_optimizer_schedule_gate_or_data_code": True,
    "complete_v6_seam_table_installed_and_asserted": True,
    "parse_args_run_parent_and_main_delegate_only_to_deepest_v1_leaf": True,
    "intermediate_successor_entrypoints_are_never_called": True,
    "initializer_to_optimizer_v6_model_witness_handoff_exact": True,
    "v7_science_normalizes_exactly_to_frozen_v6": True,
    "one_fresh_attempt_caps_and_downstream_denials_exact": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def frozen_v6_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V6_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V6_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V6_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V6_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V6_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V6_SOURCE_MANIFEST_STATUS,
        "source_count": FROZEN_V6_SOURCE_COUNT,
    }


def frozen_v6_review_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V6_REVIEW_RELATIVE_PATH,
        "commit": FROZEN_V6_REVIEW_COMMIT,
        "file_sha256": FROZEN_V6_REVIEW_FILE_SHA256,
        "content_sha256": FROZEN_V6_REVIEW_CONTENT_SHA256,
        "byte_count": FROZEN_V6_REVIEW_BYTE_COUNT,
        "status": FROZEN_V6_REVIEW_STATUS,
    }


def frozen_v6_authorization_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V6_AUTHORIZATION_RELATIVE_PATH,
        "commit": FROZEN_V6_AUTHORIZATION_COMMIT,
        "file_sha256": FROZEN_V6_AUTHORIZATION_FILE_SHA256,
        "content_sha256": FROZEN_V6_AUTHORIZATION_CONTENT_SHA256,
        "byte_count": FROZEN_V6_AUTHORIZATION_BYTE_COUNT,
        "status": FROZEN_V6_AUTHORIZATION_STATUS,
    }


def v6_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V6_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V6_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V6_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V6_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V6_TERMINAL_AUDIT_BYTE_COUNT,
        "status": V6_TERMINAL_AUDIT_STATUS,
    }


def preregistration_binding() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "content_sha256": PREREGISTRATION_CONTENT_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def frozen_v6_science_contract() -> dict[str, Any]:
    return deepcopy(_FROZEN_V6_SCIENCE_CONTRACT)


def science_contract() -> dict[str, Any]:
    """Return frozen V6 science with only V7 custody identities added."""

    value = frozen_v6_science_contract()
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v6_source_manifest": frozen_v6_source_manifest_binding(),
        "frozen_v6_source_review": frozen_v6_review_binding(),
        "frozen_v6_execution_authorization": (
            frozen_v6_authorization_binding()
        ),
        "v6_terminal_audit": v6_terminal_audit_binding(),
        "v7_preregistration": preregistration_binding(),
    }
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "integrity_replacement_of": _V6.EXPERIMENT_ID,
        "v6_retry_resume_or_repair": False,
        "v6_checkpoint_tensor_trace_or_runtime_output_reuse": False,
    }
    value["integrity_replacement"] = dict(INTEGRITY_REPLACEMENT_DELTA)
    value["authority"] = {
        **value["authority"],
        "v7_execution_authorized_by_source_contract": False,
        "v6_checkpoint_tensor_trace_or_runtime_output_reuse_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "promotion_authorized": False,
    }
    return value


def normalize_v7_operational_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that a complete V7 experiment has no undeclared difference."""

    if type(value) is not dict or dict(value) != science_contract():
        raise PermissionError("V7 experiment differs from its exact contract")
    return frozen_v6_science_contract()


def science_identity_receipt() -> dict[str, Any]:
    value = science_contract()
    normalized = normalize_v7_operational_identity(value)
    return {
        "frozen_v6_science_contract_sha256": (
            FROZEN_V6_SCIENCE_CONTRACT_SHA256
        ),
        "v7_science_contract_sha256": canonical_json_sha256(value),
        "normalized_v7_science_contract_sha256": canonical_json_sha256(
            normalized
        ),
        "normalized_exactly_equals_frozen_v6": normalized
        == _FROZEN_V6_SCIENCE_CONTRACT,
        "scientific_delta_count": 0,
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
    status: str | None = None,
) -> bytes:
    read = _V6._v5._v4._v3._v2._v1._read_regular_source
    raw = read(ROOT / relative_path)
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != file_sha256:
        raise PermissionError(f"governing document changed: {relative_path}")
    value = _V6._v5._v4._v3._v2._v1._parse_pretty_content_bound_json(
        raw,
        name=relative_path,
        expected_content_sha256=content_sha256,
    )
    if status is not None and value.get("status") != status:
        raise PermissionError(f"governing status changed: {relative_path}")
    return raw


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    if root.resolve() != ROOT.resolve():
        raise PermissionError("V7 governing documents must use repository root")
    rows = (
        (
            FROZEN_V6_SOURCE_MANIFEST_RELATIVE_PATH,
            FROZEN_V6_SOURCE_MANIFEST_FILE_SHA256,
            FROZEN_V6_SOURCE_MANIFEST_CONTENT_SHA256,
            FROZEN_V6_SOURCE_MANIFEST_BYTE_COUNT,
            FROZEN_V6_SOURCE_MANIFEST_STATUS,
        ),
        (
            FROZEN_V6_REVIEW_RELATIVE_PATH,
            FROZEN_V6_REVIEW_FILE_SHA256,
            FROZEN_V6_REVIEW_CONTENT_SHA256,
            FROZEN_V6_REVIEW_BYTE_COUNT,
            FROZEN_V6_REVIEW_STATUS,
        ),
        (
            FROZEN_V6_AUTHORIZATION_RELATIVE_PATH,
            FROZEN_V6_AUTHORIZATION_FILE_SHA256,
            FROZEN_V6_AUTHORIZATION_CONTENT_SHA256,
            FROZEN_V6_AUTHORIZATION_BYTE_COUNT,
            FROZEN_V6_AUTHORIZATION_STATUS,
        ),
        (
            V6_TERMINAL_AUDIT_RELATIVE_PATH,
            V6_TERMINAL_AUDIT_FILE_SHA256,
            V6_TERMINAL_AUDIT_CONTENT_SHA256,
            V6_TERMINAL_AUDIT_BYTE_COUNT,
            V6_TERMINAL_AUDIT_STATUS,
        ),
        (
            PREREGISTRATION_RELATIVE_PATH,
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_CONTENT_SHA256,
            PREREGISTRATION_BYTE_COUNT,
            None,
        ),
    )
    result: dict[str, str] = {}
    for path, raw_sha, content_sha, count, status in rows:
        _read_bound_json(
            path,
            file_sha256=raw_sha,
            content_sha256=content_sha,
            byte_count=count,
            status=status,
        )
        result[path] = raw_sha
    return result


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = parse_canonical_json(raw, name="V7 source manifest")
    expected_fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count", "generated_input_open_count",
        "checkpoint_or_tensor_open_count", "sealed_or_heldout_open_count",
        "whole_tree_export_authorized", "authority", "content_sha256",
    }
    core = dict(value)
    declared = core.pop("content_sha256", None)
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
        raise PermissionError("V7 source manifest contract changed")
    normalized: list[str] = []
    safe = _V6._v5._v4._v3._v2._v1.safe_relative_source_path
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V7 source binding fields changed")
        relative = safe(binding["path"])
        if (
            not is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V7 source binding identity changed")
        normalized.append(relative)
    if normalized != list(SOURCE_PATHS):
        raise PermissionError("V7 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    read = _V6._v5._v4._v3._v2._v1._read_regular_source
    manifest_raw = read(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        payload = read(root / binding["path"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound V7 source changed: {binding['path']}")
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
        read = _V6._v5._v4._v3._v2._v1._read_regular_source
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
        "reviewed_sources", "source_manifest", "frozen_v6_source_manifest",
        "frozen_v6_source_review", "frozen_v6_execution_authorization",
        "v6_terminal_audit", "v7_preregistration", "science_contract",
        "science_identity", "source_only_checks", "integrity_checks",
        "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V7 source review fields changed")
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
        or value["frozen_v6_source_manifest"]
        != frozen_v6_source_manifest_binding()
        or value["frozen_v6_source_review"] != frozen_v6_review_binding()
        or value["frozen_v6_execution_authorization"]
        != frozen_v6_authorization_binding()
        or value["v6_terminal_audit"] != v6_terminal_audit_binding()
        or value["v7_preregistration"] != preregistration_binding()
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
        raise PermissionError("V7 source review did not pass exact scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "frozen_v6_source_manifest", "frozen_v6_source_review",
        "frozen_v6_execution_authorization", "v6_terminal_audit",
        "v7_preregistration", "runtime_inputs", "experiment",
        "science_identity", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V7 execution authorization fields changed")
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
        or value["frozen_v6_source_manifest"]
        != frozen_v6_source_manifest_binding()
        or value["frozen_v6_source_review"] != frozen_v6_review_binding()
        or value["frozen_v6_execution_authorization"]
        != frozen_v6_authorization_binding()
        or value["v6_terminal_audit"] != v6_terminal_audit_binding()
        or value["v7_preregistration"] != preregistration_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["science_identity"] != science_identity_receipt()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V7 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_V6.__all__,
    *(name for name in globals() if name.isupper()),
    "current_source_bindings",
    "frozen_v6_authorization_binding",
    "frozen_v6_review_binding",
    "frozen_v6_science_contract",
    "frozen_v6_source_manifest_binding",
    "normalize_v7_operational_identity",
    "preregistration_binding",
    "science_contract",
    "science_identity_receipt",
    "v6_terminal_audit_binding",
    "validate_authorization",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
})
