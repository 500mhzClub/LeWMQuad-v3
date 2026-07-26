"""Source-only governance contract for Direct BEV V2 integrity replacement.

V2 inherits every V1 scientific constant and changes only the fresh-module
CPU seeding implementation plus versioned governance/output identities.
Importing this module source-loads only the frozen V1 contract; it performs no
generated-input, runtime-artifact, protected-material, tensor, or device I/O.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
_V1_CONTRACT_SOURCE = (
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py"
)
_V1_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v2_integrity_frozen_v1_contract",
    _V1_CONTRACT_SOURCE,
)
if _V1_SPEC is None or _V1_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V1 source-only contract")
_v1 = importlib.util.module_from_spec(_V1_SPEC)
sys.modules[_V1_SPEC.name] = _v1
_V1_SPEC.loader.exec_module(_v1)


# Re-export the complete reviewed V1 runner-facing API first.  Versioned V2
# identities and governance functions below deliberately replace selected
# names; all numerical science constants remain the same objects/values.
for _name in _v1.__all__:
    globals()[_name] = getattr(_v1, _name)

# The frozen V1 contract intentionally omits this source-only document helper
# from its public export list, while its manifest/review builders use it.
with_content_sha256 = _v1.with_content_sha256


IMPLEMENTATION_AUTHOR = "/root/plan_efficiency"
SCHEMA_PREFIX = "lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity"
EXPERIMENT_ID = "go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity"

FROZEN_V1_SOURCE_COMMIT = "51ce1480ab2cfdcf9df7e984c7be6e58890811af"
FROZEN_V1_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py"
)
FROZEN_V1_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py"
)
FROZEN_V1_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v1.py"
)
FROZEN_V1_MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v1.py"
)
FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v1_source_closure.py"
)
FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "source_manifest_2026-07-26.json"
)
FROZEN_V1_SOURCE_MANIFEST_COMMIT = FROZEN_V1_SOURCE_COMMIT
FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256 = (
    "a2fc88e015e51f2d17263fc9f00cd26bb21964bc2fe1cb046f828b33805e07c7"
)
FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "e41e6cee37d5e2a69dafdb07e3219dbbf7cf484c16e5b99bd0e2975b8fba94d9"
)
FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT = 19_505
FROZEN_V1_SOURCE_MANIFEST_STATUS = "PASS_SOURCE_CLOSURE"
FROZEN_V1_SCIENCE_CONTRACT_SHA256 = (
    "9cd985135ad9fef2ce324d4389b01d95f800d186724729626cbe895e1d8bdfb9"
)
FROZEN_V1_SCIENCE_COMPONENT_SHA256 = {
    "model": "90a2d725ce3694235d5aca2ecdda8b9e0d38df0f11e29a4d571dd0e7c2d76c5b",
    "objective": "a3e48bb32f35d5d66c21572c9aa5fe5b5673833be768759a7875f5c07630e19a",
    "optimizer": "af379468031d4dc7c7bf26ec9e0a0d30ca29fc16a9b34c44e688994e41372715",
    "schedule": "f156cc0274590be295bac0607790b61e2ed6aed9528a236bfb157cd5dd4beba2",
    "gate_thresholds": (
        "18a62d22d0bd2b1b7b93e469d6a9d4954d517b7fcbe2961c64d7a675bc53f1b0"
    ),
}
FROZEN_V1_SOURCE_BINDINGS = {
    FROZEN_V1_CONTRACT_RELATIVE_PATH: {
        "path": FROZEN_V1_CONTRACT_RELATIVE_PATH,
        "file_sha256": "79e66a4ca5bd814030f374413e4ac0a2edda2552d0614ec23b54b6b0e52ff1b6",
        "byte_count": 80_541,
        "commit": FROZEN_V1_SOURCE_COMMIT,
    },
    FROZEN_V1_MODEL_RELATIVE_PATH: {
        "path": FROZEN_V1_MODEL_RELATIVE_PATH,
        "file_sha256": "e39c8cb485e33ef891f5d1f29e0d513443715597e0973ab6af807bc15c45b930",
        "byte_count": 31_957,
        "commit": FROZEN_V1_SOURCE_COMMIT,
    },
    FROZEN_V1_RUNNER_RELATIVE_PATH: {
        "path": FROZEN_V1_RUNNER_RELATIVE_PATH,
        "file_sha256": "33617086a5481f2fa0bf8ae6993110c40bf8db85f066d1d6e874dde12fb07000",
        "byte_count": 113_094,
        "commit": FROZEN_V1_SOURCE_COMMIT,
    },
    FROZEN_V1_LAUNCHER_RELATIVE_PATH: {
        "path": FROZEN_V1_LAUNCHER_RELATIVE_PATH,
        "file_sha256": "e8cf683dd0eafc9d26f1c65f47af70dc35634676049bcaa3f5bed6e1a49f4654",
        "byte_count": 2_089,
        "commit": FROZEN_V1_SOURCE_COMMIT,
    },
    FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH: {
        "path": FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        "file_sha256": "806a90620e6d1cfaaa3f53df594885811ef99ddad776e125412ef0430c0d9fb9",
        "byte_count": 4_466,
        "commit": FROZEN_V1_SOURCE_COMMIT,
    },
}

INTEGRITY_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity_"
    "amendment_2026-07-26.md"
)
INTEGRITY_AMENDMENT_FILE_SHA256 = (
    "ff06e8834a96cab616a8a8c5ed7589fb73de202166ad278253258a55ad688509"
)
INTEGRITY_AMENDMENT_BYTE_COUNT = 4_248
INTEGRITY_AMENDMENT_COMMIT = (
    "0221d4ddd5e266a9c715d8ccb788107c0671f6ee"
)

V1_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v1_"
    "terminal_audit_2026-07-26.json"
)
V1_TERMINAL_AUDIT_COMMIT = "ae94021d44711bf9ba5fbb1386b4f8caf2617dac"
V1_TERMINAL_AUDIT_FILE_SHA256 = (
    "f928c11a2e52349145701b25a21f8b1b987ee80a365aaa2c3858d3cf650220c4"
)
V1_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "2974d914f9cde1ae93c34d76d07b1740d8c5ac17beb3b5f4922500bd242df956"
)
V1_TERMINAL_AUDIT_BYTE_COUNT = 9_291
V1_TERMINAL_AUDIT_STATUS = (
    "PASS_VALID_TERMINAL_OPERATIONAL_INTEGRITY_FAILURE_AT_ZERO_SCIENTIFIC_"
    "WORK_CLOSES_V1_NO_RETRY"
)
V1_TERMINAL_AUDIT_CLASSIFICATION = (
    "VALID_ONE_SHOT_OPERATIONAL_RNG_PRESERVATION_FAILURE_BEFORE_UPDATE_ZERO_"
    "OBSERVATION_NO_SCIENTIFIC_RESULT_V1_PERMANENTLY_CLOSED"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/direct_egocentric_bev_state_jepa_v2_integrity.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v2_integrity_"
    "contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v2_integrity_runner.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_direct_egocentric_bev_state_jepa_v2_integrity_"
    "source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_direct_egocentric_bev_state_jepa_v2_integrity_"
    "source_closure.py"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity_"
    "source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity_"
    "source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity_"
    "execution_authorization_2026-07-26.json"
)

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(set(_v1.SOURCE_PATHS)))
SOURCE_PATHS = tuple(sorted(set((*REUSED_SOURCE_PATHS, *ADDITIVE_SOURCE_PATHS))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
    INTEGRITY_AMENDMENT_RELATIVE_PATH,
    V1_TERMINAL_AUDIT_RELATIVE_PATH,
    _v1.PREREGISTRATION_RELATIVE_PATH,
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_direct_egocentric_bev_state_jepa_probe_v2_integrity"
)

RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_metrics_v1"
ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
AUTHORIZATION_STATUS = "AUTHORIZED_ONE_EXACT_DIRECT_BEV_V2_INTEGRITY_PROBE"

EXECUTION_AUTHORITY = {
    **_v1.EXECUTION_AUTHORITY,
    "output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "v1_retry_authorized": False,
    "science_identical_rng_integrity_replacement_only": True,
}

INTEGRITY_DELTA = {
    "scope": "fresh_module_cpu_default_generator_seeding_only",
    "v1_call": "torch.random.manual_seed(20260712)",
    "v2_call": "torch.random.default_generator.manual_seed(20260712)",
    "caller_cpu_rng_preserved": True,
    "every_device_rng_preserved": True,
    "parameter_draw_order_changed": False,
    "initialized_parameter_bytes_changed": False,
    "architecture_data_objective_optimizer_schedule_gate_or_cap_changed": False,
}

SCIENTIFIC_REVIEW_CHECKS = {
    "frozen_v1_manifest_and_every_bound_source_rehashed": True,
    "v1_terminal_audit_exact_and_zero_scientific_work": True,
    "v1_permanently_closed_and_v2_is_not_a_retry": True,
    "cpu_default_generator_is_the_only_implementation_delta": True,
    "v1_v2_initialized_state_dict_bitwise_equal": True,
    "caller_cpu_and_every_device_rng_preserved": True,
    "model_data_seed_draw_order_losses_optimizer_schedule_exact": True,
    "observations_metrics_gates_thresholds_and_caps_exact": True,
    "custody_failure_receipts_one_shot_and_downstream_denials_exact": True,
    "distinct_absent_before_reservation_v2_output_root": True,
    "no_runtime_or_protected_material_opened_by_source_work": True,
}


def integrity_amendment_binding() -> dict[str, Any]:
    return {
        "path": INTEGRITY_AMENDMENT_RELATIVE_PATH,
        "commit": INTEGRITY_AMENDMENT_COMMIT,
        "file_sha256": INTEGRITY_AMENDMENT_FILE_SHA256,
        "byte_count": INTEGRITY_AMENDMENT_BYTE_COUNT,
    }


def v1_terminal_audit_binding() -> dict[str, Any]:
    return {
        "path": V1_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": V1_TERMINAL_AUDIT_COMMIT,
        "file_sha256": V1_TERMINAL_AUDIT_FILE_SHA256,
        "content_sha256": V1_TERMINAL_AUDIT_CONTENT_SHA256,
        "byte_count": V1_TERMINAL_AUDIT_BYTE_COUNT,
    }


def frozen_v1_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": FROZEN_V1_SOURCE_MANIFEST_COMMIT,
        "file_sha256": FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256,
        "byte_count": FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT,
        "status": FROZEN_V1_SOURCE_MANIFEST_STATUS,
    }


def validate_frozen_v1_source_closure(
    root: Path = ROOT,
) -> dict[str, str]:
    raw = _v1._read_regular_source(
        root / FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH
    )
    if (
        len(raw) != FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(raw).hexdigest()
        != FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen V1 source manifest raw identity changed")
    manifest = _v1.validate_source_manifest(raw)
    if (
        manifest.get("content_sha256")
        != FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("status") != FROZEN_V1_SOURCE_MANIFEST_STATUS
    ):
        raise PermissionError("frozen V1 source manifest conclusion changed")

    manifest_bindings = {
        binding["path"]: binding for binding in manifest["source_bindings"]
    }
    for relative, expected in FROZEN_V1_SOURCE_BINDINGS.items():
        if expected.get("commit") != FROZEN_V1_SOURCE_COMMIT or (
            manifest_bindings.get(relative)
            != {
                "path": relative,
                "file_sha256": expected["file_sha256"],
                "byte_count": expected["byte_count"],
            }
        ):
            raise PermissionError(f"frozen V1 binding changed: {relative}")

    current = _v1.current_source_bindings(root)
    if (
        current.get(FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("current V1 source manifest changed")
    for binding in manifest["source_bindings"]:
        if current.get(binding["path"]) != binding["file_sha256"]:
            raise PermissionError(
                f"current V1 source changed: {binding['path']}"
            )
    return current


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
    value = _v1.science_contract()
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["governing_documents"] = {
        **value["governing_documents"],
        "frozen_v1_source_manifest": frozen_v1_source_manifest_binding(),
        "v1_terminal_audit": v1_terminal_audit_binding(),
        "v2_integrity_amendment": integrity_amendment_binding(),
    }
    value["lifecycle"] = {
        **value["lifecycle"],
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "integrity_replacement_of": _v1.EXPERIMENT_ID,
        "v1_retry": False,
    }
    value["integrity_replacement"] = {
        **INTEGRITY_DELTA,
        "frozen_v1_science_contract_sha256": (
            FROZEN_V1_SCIENCE_CONTRACT_SHA256
        ),
        "frozen_v1_science_component_sha256": dict(
            FROZEN_V1_SCIENCE_COMPONENT_SHA256
        ),
    }
    return value


def validate_governing_documents(root: Path = ROOT) -> dict[str, str]:
    result = _v1.validate_governing_documents(root)
    validate_frozen_v1_source_closure(root)
    amendment = _v1._read_regular_source(root / INTEGRITY_AMENDMENT_RELATIVE_PATH)
    audit = _v1._read_regular_source(root / V1_TERMINAL_AUDIT_RELATIVE_PATH)
    if (
        len(amendment) != INTEGRITY_AMENDMENT_BYTE_COUNT
        or hashlib.sha256(amendment).hexdigest()
        != INTEGRITY_AMENDMENT_FILE_SHA256
    ):
        raise PermissionError("V2 integrity amendment changed")
    if (
        len(audit) != V1_TERMINAL_AUDIT_BYTE_COUNT
        or hashlib.sha256(audit).hexdigest() != V1_TERMINAL_AUDIT_FILE_SHA256
    ):
        raise PermissionError("V1 terminal audit raw identity changed")
    audit_value = _v1.parse_canonical_json(audit, name="V1 terminal audit")
    if (
        audit_value.get("content_sha256") != V1_TERMINAL_AUDIT_CONTENT_SHA256
        or audit_value.get("status") != V1_TERMINAL_AUDIT_STATUS
        or audit_value.get("classification") != V1_TERMINAL_AUDIT_CLASSIFICATION
        or audit_value.get("execution_accounting", {}).get("scientific_work_count")
        != 0
        or audit_value.get("scientific_consequence", {}).get(
            "valid_scientific_result_produced"
        ) is not False
    ):
        raise PermissionError("V1 terminal audit conclusion changed")
    result[INTEGRITY_AMENDMENT_RELATIVE_PATH] = (
        INTEGRITY_AMENDMENT_FILE_SHA256
    )
    result[FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH] = (
        FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
    )
    result[V1_TERMINAL_AUDIT_RELATIVE_PATH] = V1_TERMINAL_AUDIT_FILE_SHA256
    return result


def validate_source_manifest(raw: bytes) -> dict[str, Any]:
    value = _v1.parse_canonical_json(raw, name="V2 source manifest")
    expected_fields = {
        "schema", "status", "entrypoints", "forced_dynamic_sources",
        "excluded_runtime_categories", "source_paths", "source_bindings",
        "source_bindings_sha256", "source_count",
        "generated_input_open_count", "checkpoint_or_tensor_open_count",
        "sealed_or_heldout_open_count", "whole_tree_export_authorized",
        "authority", "content_sha256",
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
        or type(paths) is not list
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
        or not set(SOURCE_PATHS).issubset(paths)
        or type(bindings) is not list
        or len(bindings) != len(paths)
        or value.get("source_count") != len(paths)
        or value.get("source_bindings_sha256")
        != canonical_json_sha256(bindings)
        or value.get("generated_input_open_count") != 0
        or value.get("checkpoint_or_tensor_open_count") != 0
        or value.get("sealed_or_heldout_open_count") != 0
        or value.get("whole_tree_export_authorized") is not False
        or value.get("authority") != SOURCE_ONLY_AUTHORITY
    ):
        raise PermissionError("V2 source manifest contract changed")
    normalized: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "path", "file_sha256", "byte_count"
        }:
            raise PermissionError("V2 source binding fields changed")
        relative = _v1.safe_relative_source_path(binding["path"])
        if (
            not _v1.is_sha256(binding["file_sha256"])
            or type(binding["byte_count"]) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError("V2 source binding identity changed")
        normalized.append(relative)
    if normalized != paths:
        raise PermissionError("V2 source binding order changed")
    return dict(value)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    manifest_raw = _v1._read_regular_source(root / SOURCE_MANIFEST_RELATIVE_PATH)
    manifest = validate_source_manifest(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        relative = binding["path"]
        payload = _v1._read_regular_source(root / relative)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != binding["file_sha256"] or len(payload) != binding["byte_count"]:
            raise PermissionError(f"manifest-bound V2 source changed: {relative}")
        result[relative] = digest
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    result.update(validate_governing_documents(root))
    return result


def _manifest_binding_or_read(
    source_manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source_manifest_binding is None:
        raw = _v1._read_regular_source(ROOT / SOURCE_MANIFEST_RELATIVE_PATH)
        value = validate_source_manifest(raw)
        source_manifest_binding = _v1.artifact_binding(
            SOURCE_MANIFEST_RELATIVE_PATH,
            raw,
            content_sha256=str(value["content_sha256"]),
        )
    return _v1.validate_binding(
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
        "reviewed_sources", "source_manifest", "integrity_amendment",
        "frozen_v1_source_manifest", "v1_terminal_audit",
        "science_contract", "source_only_checks",
        "scientific_checks", "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V2 source review fields changed")
    manifest_binding = _manifest_binding_or_read(source_manifest_binding)
    core = dict(value)
    declared = core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS_SOURCE_AND_SCIENCE_IDENTICAL_INTEGRITY"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["source_manifest"] != manifest_binding
        or expected_sources.get(SOURCE_MANIFEST_RELATIVE_PATH)
        != manifest_binding["file_sha256"]
        or expected_sources.get(INTEGRITY_AMENDMENT_RELATIVE_PATH)
        != INTEGRITY_AMENDMENT_FILE_SHA256
        or expected_sources.get(FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH)
        != FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
        or expected_sources.get(V1_TERMINAL_AUDIT_RELATIVE_PATH)
        != V1_TERMINAL_AUDIT_FILE_SHA256
        or value["integrity_amendment"] != integrity_amendment_binding()
        or value["frozen_v1_source_manifest"]
        != frozen_v1_source_manifest_binding()
        or value["v1_terminal_audit"] != v1_terminal_audit_binding()
        or value["science_contract"] != science_contract()
        or value["source_only_checks"] != {
            "stdlib_only_contract_import": True,
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "sealed_or_heldout_opened": [],
        }
        or value["scientific_checks"] != SCIENTIFIC_REVIEW_CHECKS
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V2 source review did not pass exact integrity scope")
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_source_review",
        "integrity_amendment", "v1_terminal_audit", "runtime_inputs",
        "experiment", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("V2 execution authorization fields changed")
    expected_review = validate_binding(
        dict(review_binding), path=REVIEW_RELATIVE_PATH
    )
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
        or value["integrity_amendment"] != integrity_amendment_binding()
        or value["v1_terminal_audit"] != v1_terminal_audit_binding()
        or value["runtime_inputs"] != runtime_authorization_template()
        or value["experiment"] != science_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("V2 execution authorization changed")
    return dict(value)


__all__ = sorted({
    *_v1.__all__,
    *(name for name in globals() if name.isupper()),
    "current_source_bindings",
    "frozen_v1_source_manifest_binding",
    "integrity_amendment_binding",
    "model_config",
    "objective_contract",
    "optimizer_contract",
    "runtime_authorization_template",
    "science_contract",
    "v1_terminal_audit_binding",
    "validate_authorization",
    "validate_frozen_v1_source_closure",
    "validate_governing_documents",
    "validate_review",
    "validate_source_manifest",
    "with_content_sha256",
})
