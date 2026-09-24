"""Source-only V10R integrity replacement for retrieval JEPA V10.

V10R changes no science.  It reuses the exact frozen V10 model, data,
initialization, objective, optimizer, schedule, gates, caps, and lifecycle.
The sole integrity delta permits eight binary32 epsilons when comparing the
update-zero action NLL with its immutable equal-logit reference.  Importing
this module reads source files only; it does not inspect generated inputs,
runtime outputs, checkpoints, RGB, or accelerator state.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
V10_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py"
)


def _load_source_module(relative_path: str, module_name: str) -> Any:
    source = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only contract {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V10 = _load_source_module(
    V10_CONTRACT_RELATIVE_PATH,
    "_lewm_v10r_frozen_v10_contract",
)

# Capture science before replacing any operational identity.  The captured
# value deliberately retains the V10 science schema and V10 lifecycle fields.
_FROZEN_V10_SCIENCE_CONTRACT = _V10.science_contract()
_BASE_V10_NORMALIZE_PHASE_A_INPUTS = _V10._normalize_phase_a_inputs
_BASE_V10_READ_REGULAR_SOURCE = _V10._read_regular_source
_BASE_V10_VALIDATE_SOURCE_MANIFEST = _V10.validate_source_manifest
V10_RUNNER_RELATIVE_PATH = _V10.RUNNER_RELATIVE_PATH
V10_LAUNCHER_RELATIVE_PATH = _V10.LAUNCHER_RELATIVE_PATH
V10_OUTPUT_ROOT_RELATIVE_PATH = _V10.OUTPUT_ROOT_RELATIVE_PATH

for _name in _V10.__all__:
    if _name.isupper():
        globals()[_name] = getattr(_V10, _name)

canonical_json_bytes = _V10.canonical_json_bytes
canonical_json_sha256 = _V10.canonical_json_sha256
is_sha256 = _V10.is_sha256
with_content_sha256 = _V10.with_content_sha256
parse_canonical_json = _V10.parse_canonical_json
safe_relative_path = _V10.safe_relative_path
artifact_binding = _V10.artifact_binding
validate_binding = _V10.validate_binding


IMPLEMENTATION_AUTHOR = "/root/v10_update0_repro"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/"
    "run_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/"
    "launch_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/"
    "test_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py"
)
RUNNER_TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/"
    "check_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/"
    "test_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_source_closure.py"
)
TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_preregistration_2026-07-26.json"
)
PREREGISTRATION_COMMIT = (
    "bdf30305645efbcde56c7e52711e2ded7bf728fb"
)
PREREGISTRATION_FILE_SHA256 = (
    "38e3f4d9378d4974f77b4a10b069a704b6722caea31bd97f237f0eac00f2308a"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "4100001b5217091bea6b917057eb33cb9331b77c47dd24468c036d5535e8d97e"
)
PREREGISTRATION_BYTE_COUNT = 16_613
PREREGISTRATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_preregistration_independent_review_"
    "2026-07-26.json"
)
PREREGISTRATION_REVIEW_COMMIT = (
    "5d532e814c73c7c8238a59cf853e9cef4975c541"
)
PREREGISTRATION_REVIEW_FILE_SHA256 = (
    "606138757d9292ef3c8a75f16c1e8abb34da5fa11d84db60362d126b68cf2acf"
)
PREREGISTRATION_REVIEW_CONTENT_SHA256 = (
    "8f1316b203734fdf844cc04819e8b370510258da3d6680381667228f22037763"
)
PREREGISTRATION_REVIEW_BYTE_COUNT = 13_235

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_execution_authorization_2026-07-26.json"
)
V10_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_v10_"
    "terminal_audit_2026-07-26.json"
)
PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = V10_TERMINAL_AUDIT_RELATIVE_PATH
PRIOR_TERMINAL_AUDIT_COMMIT = (
    "b590e50af272ae046618819eed4b88f1cd7a0cab"
)
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "e33030f59c1d36aecf61d98750213daa89d7aeb8ee0daf83ff92812ca31ce4e5"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "9ab2aec125e2d8ced8f35da7dab6c2d2794035d33c3888c28f8584b6e7070eb4"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 5_999

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(dict.fromkeys((
    V10_CONTRACT_RELATIVE_PATH,
    V10_RUNNER_RELATIVE_PATH,
    V10_LAUNCHER_RELATIVE_PATH,
    *_V10.SOURCE_PATHS,
))))
SOURCE_PATHS = tuple(sorted(dict.fromkeys((
    *ADDITIVE_SOURCE_PATHS,
    *REUSED_SOURCE_PATHS,
))))
SOURCE_MANIFEST_ENTRYPOINTS = (
    LAUNCHER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = tuple(dict.fromkeys((
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_REVIEW_RELATIVE_PATH,
    V10_TERMINAL_AUDIT_RELATIVE_PATH,
    *_V10.SOURCE_REVIEW_ADDITIONAL_PATHS,
)))

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_action_conditioned_next_target_retrieval_jepa_probe_"
    "v10r_integrity_replacement"
)
PROHIBITED_RUNTIME_OUTPUT_ROOTS = (V10_OUTPUT_ROOT_RELATIVE_PATH,)

SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
PHASE_A_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_metrics_v1"
PHASE_A_ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_artifact_v1"
PHASE_B_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_phase_b_metrics_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

OPERATIONAL_FAILURE_STATUS = (
    "TERMINAL_V10R_INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
)
RESERVATION_PUBLICATION_FAILURE_STATUS = (
    "TERMINAL_V10R_RESERVATION_PUBLICATION_FAILURE_NO_RETRY"
)

EXECUTION_AUTHORITY = {
    **dict(_V10.EXECUTION_AUTHORITY),
    "generated_mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
}

SCIENTIFIC_REVIEW_CHECKS = dict(_V10.SCIENTIFIC_REVIEW_CHECKS)
del SCIENTIFIC_REVIEW_CHECKS[
    "final_single_frame_v5_family_closure_exact"
]
SCIENTIFIC_REVIEW_CHECKS.update({
    "v10r_contract_normalizes_to_frozen_v10_at_only_four_operational_"
    "identity_leaves_exact": True,
    "sole_eight_float32_epsilon_integrity_adapter_exact": True,
    "one_v10r_only_limited_supersession_exact": True,
    "no_further_integrity_replacement_authorized": True,
})

FLOAT32_EPSILON = 2.0**-23
UPDATE_ZERO_ACTION_NLL_ABS_TOLERANCE = 8.0 * FLOAT32_EPSILON
INTEGRITY_REPLACEMENT_DELTA = {
    "science_changed": False,
    "v10_runtime_output_open_authorized": False,
    "v10_output_root": V10_OUTPUT_ROOT_RELATIVE_PATH,
    "v10r_output_root": OUTPUT_ROOT_RELATIVE_PATH,
    "comparison": "update_zero_action_nll_to_equal_logit_reference",
    "v10_absolute_tolerance": 1e-7,
    "v10r_absolute_tolerance": UPDATE_ZERO_ACTION_NLL_ABS_TOLERANCE,
    "bitwise_action_prediction_check_changed": False,
    "exact_action_ratio_checks_changed": False,
    "exact_per_family_action_margin_checks_changed": False,
    "model_data_seed_schedule_losses_thresholds_initialization_or_cap_changed":
        False,
    "retry_resume_second_seed_or_schedule_extension_authorized": False,
}

V10R_OPERATIONAL_IDENTITY_LEAVES = (
    "/schema",
    "/lifecycle/output_root",
    "/lifecycle/operational_failure/failure_status",
    "/lifecycle/operational_failure/reservation_publication_failure_status",
)


def science_contract() -> dict[str, Any]:
    """Return V10 science with only truthful V10R operational identities."""

    value = deepcopy(_FROZEN_V10_SCIENCE_CONTRACT)
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    lifecycle = value["lifecycle"]
    lifecycle["output_root"] = OUTPUT_ROOT_RELATIVE_PATH
    failure = lifecycle["operational_failure"]
    failure["failure_status"] = OPERATIONAL_FAILURE_STATUS
    failure["reservation_publication_failure_status"] = (
        RESERVATION_PUBLICATION_FAILURE_STATUS
    )
    return value


def frozen_v10_science_contract() -> dict[str, Any]:
    """Return the immutable V10 experiment used as the science witness."""

    return deepcopy(_FROZEN_V10_SCIENCE_CONTRACT)


def normalize_v10r_operational_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Map the four registered V10R custody leaves back to the V10 witness."""

    if type(value) is not dict:
        raise TypeError("V10R science contract must be a dictionary")
    expected = science_contract()
    normalized = deepcopy(value)
    try:
        if (
            normalized["schema"] != expected["schema"]
            or normalized["lifecycle"]["output_root"]
            != expected["lifecycle"]["output_root"]
            or normalized["lifecycle"]["operational_failure"]
            ["failure_status"]
            != expected["lifecycle"]["operational_failure"]
            ["failure_status"]
            or normalized["lifecycle"]["operational_failure"]
            ["reservation_publication_failure_status"]
            != expected["lifecycle"]["operational_failure"]
            ["reservation_publication_failure_status"]
        ):
            raise PermissionError("V10R operational identity changed")
    except (KeyError, TypeError) as error:
        raise PermissionError("V10R operational identity changed") from error

    frozen = _FROZEN_V10_SCIENCE_CONTRACT
    normalized["schema"] = frozen["schema"]
    normalized["lifecycle"]["output_root"] = frozen["lifecycle"][
        "output_root"
    ]
    normalized["lifecycle"]["operational_failure"]["failure_status"] = (
        frozen["lifecycle"]["operational_failure"]["failure_status"]
    )
    normalized["lifecycle"]["operational_failure"][
        "reservation_publication_failure_status"
    ] = frozen["lifecycle"]["operational_failure"][
        "reservation_publication_failure_status"
    ]
    return normalized


def preregistration_review_binding() -> dict[str, Any]:
    """Return the exact committed V10R preregistration-review binding."""

    return {
        "path": PREREGISTRATION_REVIEW_RELATIVE_PATH,
        "commit": PREREGISTRATION_REVIEW_COMMIT,
        "file_sha256": PREREGISTRATION_REVIEW_FILE_SHA256,
        "content_sha256": PREREGISTRATION_REVIEW_CONTENT_SHA256,
        "byte_count": PREREGISTRATION_REVIEW_BYTE_COUNT,
    }


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    """Rehash source plus the exact V10R governing documents.

    V10's helper required its prior audit to use canonical one-line JSON.
    The separately committed V10 terminal audit is exact pretty JSON, so V10R
    verifies its immutable raw and declared-content hashes without changing
    or canonicalizing those historical bytes.
    """

    manifest_raw = _BASE_V10_READ_REGULAR_SOURCE(
        root / SOURCE_MANIFEST_RELATIVE_PATH
    )
    manifest = _BASE_V10_VALIDATE_SOURCE_MANIFEST(manifest_raw)
    result: dict[str, str] = {}
    for binding in manifest["source_bindings"]:
        relative = binding["path"]
        raw = _BASE_V10_READ_REGULAR_SOURCE(root / relative)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            len(raw) != binding["byte_count"]
            or digest != binding["file_sha256"]
        ):
            raise PermissionError(
                f"manifest-bound source changed: {relative}"
            )
        result[relative] = digest

    documents = (
        (
            PREREGISTRATION_RELATIVE_PATH,
            PREREGISTRATION_BYTE_COUNT,
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_CONTENT_SHA256,
            "V10R preregistration",
        ),
        (
            V10_TERMINAL_AUDIT_RELATIVE_PATH,
            PRIOR_TERMINAL_AUDIT_BYTE_COUNT,
            PRIOR_TERMINAL_AUDIT_FILE_SHA256,
            PRIOR_TERMINAL_AUDIT_CONTENT_SHA256,
            "V10 terminal audit",
        ),
        (
            PREREGISTRATION_REVIEW_RELATIVE_PATH,
            PREREGISTRATION_REVIEW_BYTE_COUNT,
            PREREGISTRATION_REVIEW_FILE_SHA256,
            PREREGISTRATION_REVIEW_CONTENT_SHA256,
            "V10R preregistration independent review",
        ),
    )
    result[SOURCE_MANIFEST_RELATIVE_PATH] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    for relative, byte_count, file_sha256, content_sha256, name in documents:
        raw = _BASE_V10_READ_REGULAR_SOURCE(root / relative)
        digest = hashlib.sha256(raw).hexdigest()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PermissionError(f"{name} changed") from error
        if (
            len(raw) != byte_count
            or digest != file_sha256
            or type(value) is not dict
            or value.get("content_sha256") != content_sha256
        ):
            raise PermissionError(f"{name} changed")
        result[relative] = digest
    return result


def _replacement_tolerance_applies(
    update0_metrics: Mapping[str, Any],
) -> bool:
    """Recognize only the registered finite float32 reduction discrepancy."""

    if type(update0_metrics) is not dict:
        return False
    retrieval = update0_metrics.get("factorized_retrieval")
    if type(retrieval) is not dict:
        return False
    observed = retrieval.get("action_retrieval_nll")
    reference = retrieval.get("action_equal_logit_reference")
    if (
        type(observed) not in {int, float}
        or type(reference) not in {int, float}
    ):
        return False
    observed_float = float(observed)
    reference_float = float(reference)
    return (
        math.isfinite(observed_float)
        and math.isfinite(reference_float)
        and math.isclose(
            observed_float,
            reference_float,
            rel_tol=0.0,
            abs_tol=UPDATE_ZERO_ACTION_NLL_ABS_TOLERANCE,
        )
    )


def _normalize_phase_a_inputs(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only V10R's wider update-zero float32 integrity tolerance."""

    if not _replacement_tolerance_applies(update0_metrics):
        return _BASE_V10_NORMALIZE_PHASE_A_INPUTS(
            metrics,
            update0_metrics,
            observation_integrity,
        )

    adapted_update0 = deepcopy(update0_metrics)
    retrieval = adapted_update0["factorized_retrieval"]
    observed_nll = float(retrieval["action_retrieval_nll"])
    retrieval["action_retrieval_nll"] = retrieval[
        "action_equal_logit_reference"
    ]
    normalized = _BASE_V10_NORMALIZE_PHASE_A_INPUTS(
        metrics,
        adapted_update0,
        observation_integrity,
    )
    # The adapter changes validation only.  Preserve the observed metric in
    # normalized evidence rather than replacing it with the frozen reference.
    normalized["update0_factorized_retrieval"][
        "action_retrieval_nll"
    ] = observed_nll
    return normalized


# Inherited functions resolve constants and private helpers in their defining
# module.  Rebind the frozen V10 module once so every receipt uses V10R's new
# operational identity while the executable contract normalizes exactly to
# frozen V10 at the four registered custody leaves.
for _name, _value in tuple(globals().items()):
    if _name.isupper():
        setattr(_V10, _name, _value)
_V10.science_contract = science_contract
_V10._normalize_phase_a_inputs = _normalize_phase_a_inputs
_V10.current_source_bindings = current_source_bindings
_V10.preregistration_review_binding = preregistration_review_binding

_OVERRIDDEN_PUBLIC = {
    "current_source_bindings",
    "frozen_v10_science_contract",
    "normalize_v10r_operational_identity",
    "preregistration_review_binding",
    "science_contract",
}
for _name in _V10.__all__:
    if _name not in _OVERRIDDEN_PUBLIC and not _name.isupper():
        globals()[_name] = getattr(_V10, _name)

__all__ = [name for name in globals() if name.isupper()] + sorted(
    _OVERRIDDEN_PUBLIC
    | {
        name
        for name in _V10.__all__
        if not name.isupper()
    }
)
