"""Source-only contract for the overlap-tokenization V2 integrity replacement.

V2 changes no science.  It reuses the exact V1 model, experiment contract,
runner, and custody lifecycle, while correcting the two raw-authority byte
counts that were missing from the V1 runtime-binding table.  Importing this
module reads source files only; it does not inspect generated inputs, runtime
outputs, checkpoints, RGB, or accelerator state.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V1_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_overlapping_tokenization_v1.py"
)


def _load_source_module(relative_path: str, module_name: str) -> Any:
    source = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only contract {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V1 = _load_source_module(
    V1_CONTRACT_RELATIVE_PATH,
    "_lewm_overlap_v2_frozen_v1_contract",
)

# Capture science before replacing any operational identity.  V2 returns this
# exact value, including the V1 science schema and model runtime version.
_FROZEN_V1_SCIENCE_CONTRACT = _V1.science_contract()
_BASE_V1_VALIDATE_RUNTIME_INPUTS = _V1.validate_runtime_inputs
V1_RUNNER_RELATIVE_PATH = _V1.RUNNER_RELATIVE_PATH
V1_LAUNCHER_RELATIVE_PATH = _V1.LAUNCHER_RELATIVE_PATH

for _name in _V1.__all__:
    if _name.isupper():
        globals()[_name] = getattr(_V1, _name)

canonical_json_bytes = _V1.canonical_json_bytes
canonical_json_sha256 = _V1.canonical_json_sha256
is_sha256 = _V1.is_sha256
with_content_sha256 = _V1.with_content_sha256
parse_canonical_json = _V1.parse_canonical_json
safe_relative_path = _V1.safe_relative_path
artifact_binding = _V1.artifact_binding
validate_binding = _V1.validate_binding


CONTRACT_AUTHOR = "/root/overlap_v2_minimal_impl"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_overlapping_tokenization_v2_integrity_replacement"
)

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_rgb_overlapping_tokenization_v2_integrity_replacement.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_overlapping_tokenization_v2_"
    "integrity_replacement.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_overlapping_tokenization_v2_"
    "integrity_replacement.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_overlapping_tokenization_v2_"
    "integrity_replacement.py"
)
TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v2_integrity_"
    "replacement_preregistration_2026-07-25.json"
)
PREREGISTRATION_COMMIT = (
    "311d06fc0ccfd79c347d4f89edc0f5f9c9654ff9"
)
PREREGISTRATION_FILE_SHA256 = (
    "eff10e7113d3fd3821064e526ff9a724b6acb664906d15e317177410228daf54"
)
PREREGISTRATION_BYTE_COUNT = 7_606
PREREGISTRATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v2_integrity_"
    "replacement_preregistration_independent_review_2026-07-25.json"
)
PREREGISTRATION_REVIEW_FILE_SHA256 = (
    "5707a881d03da6745a490e209f81c6437c98397175ff1c2c2059040a43951d47"
)
PREREGISTRATION_REVIEW_CONTENT_SHA256 = (
    "ba43b9c31764d7c00cec9b1493c3a844a57c66808aaf192873312e1c7812ac48"
)
PREREGISTRATION_REVIEW_BYTE_COUNT = 8_640
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v2_integrity_"
    "replacement_source_manifest_2026-07-25.json"
)
SOURCE_MANIFEST_SCHEMA = (
    f"{SCHEMA_PREFIX}_source_manifest_v1"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v2_integrity_"
    "replacement_source_review_2026-07-25.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v2_integrity_"
    "replacement_execution_authorization_2026-07-25.json"
)
V1_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v1_"
    "terminal_audit_2026-07-25.json"
)
V1_AUTHORIZATION_RELATIVE_PATH = _V1.AUTHORIZATION_RELATIVE_PATH

ADDITIVE_SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
)
REUSED_SOURCE_PATHS = tuple(dict.fromkeys((
    V1_CONTRACT_RELATIVE_PATH,
    V1_RUNNER_RELATIVE_PATH,
    V1_LAUNCHER_RELATIVE_PATH,
    *_V1.SOURCE_PATHS,
)))
SOURCE_PATHS = tuple(dict.fromkeys((
    *ADDITIVE_SOURCE_PATHS,
    *REUSED_SOURCE_PATHS,
)))
SOURCE_MANIFEST_ENTRYPOINTS = (
    LAUNCHER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = tuple(dict.fromkeys((
    *ADDITIVE_SOURCE_PATHS,
    *_V1.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES,
)))
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_REVIEW_RELATIVE_PATH,
    V1_TERMINAL_AUDIT_RELATIVE_PATH,
    V1_AUTHORIZATION_RELATIVE_PATH,
    _V1.SOURCE_MANIFEST_RELATIVE_PATH,
)

V1_OUTPUT_ROOT_RELATIVE_PATH = _V1.OUTPUT_ROOT_RELATIVE_PATH
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_overlapping_tokenization_probe_v2_integrity_replacement"
)
PROHIBITED_RUNTIME_OUTPUT_ROOTS = tuple(dict.fromkeys((
    *_V1.PROHIBITED_RUNTIME_OUTPUT_ROOTS,
    V1_OUTPUT_ROOT_RELATIVE_PATH,
)))

# The sole integrity fix.  Indexing this table is mandatory for authorization
# construction; there is deliberately no fallback byte count.
RAW_MANIFEST_BYTE_COUNT = 311_598
RAW_AUDIT_BYTE_COUNT = 26_975
RUNTIME_BYTE_COUNTS = {
    **dict(_V1.RUNTIME_BYTE_COUNTS),
    RAW_MANIFEST_RELATIVE_PATH: RAW_MANIFEST_BYTE_COUNT,
    RAW_AUDIT_RELATIVE_PATH: RAW_AUDIT_BYTE_COUNT,
}

REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_snapshot_v1"
METRIC_SIDECAR_SCHEMA = f"{SCHEMA_PREFIX}_metric_sidecar_v1"
CHECKPOINT_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_checkpoint_metrics_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
PRE_LEDGER_FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_pre_ledger_failure_v1"
CONTRACT_INVALID_LEDGER_FAILURE_SCHEMA = (
    f"{SCHEMA_PREFIX}_contract_invalid_ledger_failure_v1"
)
PARTIAL_ACCESS_RECORD_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_record_v1"
PARTIAL_ACCESS_LEDGER_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_ledger_v1"

NORMAL_FAILURE_STATUS = (
    "TERMINAL_OVERLAPPING_TOKENIZATION_V2_INTEGRITY_REPLACEMENT_"
    "OPERATIONAL_OR_INTEGRITY_FAILURE_NO_RETRY"
)
PRE_LEDGER_FAILURE_STATUS = (
    "TERMINAL_OVERLAPPING_TOKENIZATION_V2_INTEGRITY_REPLACEMENT_"
    "POST_RESERVATION_PRE_LEDGER_FAILURE_NO_RETRY"
)
CONTRACT_INVALID_LEDGER_FAILURE_STATUS = (
    "TERMINAL_OVERLAPPING_TOKENIZATION_V2_INTEGRITY_REPLACEMENT_"
    "CONTRACT_INVALID_ACCESS_LEDGER_NO_RETRY"
)

EXECUTION_AUTHORITY = {
    **dict(_V1.EXECUTION_AUTHORITY),
    "generated_mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
}

INTEGRITY_REPLACEMENT_DELTA = {
    "science_changed": False,
    "v1_runtime_output_open_authorized": False,
    "raw_manifest_byte_count": RAW_MANIFEST_BYTE_COUNT,
    "raw_audit_byte_count": RAW_AUDIT_BYTE_COUNT,
    "missing_authorized_byte_count_fallback": None,
}


def science_contract() -> dict[str, Any]:
    """Return the frozen V1 science contract without version mutation."""

    return deepcopy(_FROZEN_V1_SCIENCE_CONTRACT)


def validate_runtime_inputs(value: object) -> dict[str, Any]:
    """Require the exact corrected raw byte counts, then run full V1 checks."""

    if type(value) is not dict:
        raise PermissionError("runtime input groups changed")
    try:
        raw = value["raw"]
        manifest = raw["manifest"]
        audit = raw["audit"]
    except (KeyError, TypeError):
        raise PermissionError("raw runtime bindings changed") from None
    expected = (
        (
            manifest,
            RAW_MANIFEST_RELATIVE_PATH,
            RAW_MANIFEST_BYTE_COUNT,
        ),
        (audit, RAW_AUDIT_RELATIVE_PATH, RAW_AUDIT_BYTE_COUNT),
    )
    for binding, path, byte_count in expected:
        try:
            checked = validate_binding(binding, path=path)
        except (TypeError, ValueError) as error:
            raise PermissionError(f"runtime binding changed: {path}") from error
        if checked["byte_count"] != byte_count:
            raise PermissionError(f"runtime binding changed: {path}")
    return _BASE_V1_VALIDATE_RUNTIME_INPUTS(value)


# Inherited validators resolve constants in their defining modules.  Rebind
# each private layer once so receipts and authorization use only the V2
# operational identity while science_contract() remains the captured V1 value.
for _target in (_V1, _V1._MOTION, _V1._MOTION._TEMPORAL):
    for _name, _value in tuple(globals().items()):
        if _name.isupper():
            setattr(_target, _name, _value)
    _target.science_contract = science_contract
    _target.validate_runtime_inputs = validate_runtime_inputs

_OVERRIDDEN_PUBLIC = {"science_contract", "validate_runtime_inputs"}
for _name in _V1.__all__:
    if _name not in _OVERRIDDEN_PUBLIC and not _name.isupper():
        globals()[_name] = getattr(_V1, _name)

__all__ = [name for name in globals() if name.isupper()] + sorted(
    _OVERRIDDEN_PUBLIC
    | {
        name
        for name in _V1.__all__
        if not name.isupper()
    }
)
