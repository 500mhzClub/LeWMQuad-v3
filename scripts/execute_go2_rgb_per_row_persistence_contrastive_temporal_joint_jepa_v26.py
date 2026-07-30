#!/usr/bin/env python3
"""Denied-by-default V26 schema-integrity adapter over frozen V25.

V26 changes only lifecycle identity/evidence and verifies the repaired training
schema aliases.  The frozen V25 executor still owns every scientific,
accounting, gate, terminalization, and write-only recovery operation.  This
source shell grants no execution authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V25_EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_"
    "jepa_v25.py"
)
V25_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "43231c689547b66de83f3cafbfac270455a7a234"
)
V25_EXECUTOR_FILE_SHA256 = (
    "31e7e802220b8c5aee71fe25398352050618eb060c5c071d267b86a9281225ab"
)
V25_EXECUTOR_BYTE_COUNT = 53_074
V25_PUBLIC_MODULE_NAME = (
    "scripts.execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_"
    "jepa_v25"
)
PRIVATE_V25_MODULE_NAME = f"{__name__}.__private_v25_executor"
_PUBLIC_V25_WAS_LOADED_BEFORE_ADAPTER = V25_PUBLIC_MODULE_NAME in sys.modules

SCHEMA_PREFIX = (
    "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26"
)
PREREGISTRATION_COMMIT = "0c277fd7350931a7993d5affc2d1d4633ffed916"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "97061601af2922622673d7e4f8b4c1a6625edcdf899abd647373c28daa192a18"
)
PREREGISTRATION_BYTE_COUNT = 7_999
V25_TERMINAL_FAILURE_RESULT_COMMIT = (
    "26c8fd902319c06d4dbf25cab36a63ec2df44081"
)
V25_TERMINAL_FAILURE_RESULT_PATH = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_terminal_failure_result_2026-07-30.json"
)
V25_TERMINAL_FAILURE_RESULT_FILE_SHA256 = (
    "5c8d6d80ce24c60900c49f6cf49979c3001024666a2156d945e526b396dd1596"
)
V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT = 10_380
V25_TERMINAL_FAILURE_RESULT_CONTENT_SHA256 = (
    "59423f03ca153ca481d71ea4e88aaa625128ece4a15eb8b6253ae4f009272929"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26/"
    "attempt_v1"
)


def _load_private_v25_executor() -> ModuleType:
    if V25_EXECUTOR_PATH.is_symlink() or not V25_EXECUTOR_PATH.is_file():
        raise FileNotFoundError("frozen V25 executor source is absent or not regular")
    source = V25_EXECUTOR_PATH.read_bytes()
    if (
        len(source) != V25_EXECUTOR_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != V25_EXECUTOR_FILE_SHA256
    ):
        raise RuntimeError("frozen V25 executor source binding changed")
    if PRIVATE_V25_MODULE_NAME in sys.modules:
        raise RuntimeError("private V25 executor module name is already occupied")
    module = ModuleType(PRIVATE_V25_MODULE_NAME)
    module.__file__ = str(V25_EXECUTOR_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_V25_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(V25_EXECUTOR_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_V25_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_V25_MODULE_NAME)
    return module


_v25 = _load_private_v25_executor()
if (
    not _v25.SCHEMA_PREFIX.endswith("joint_jepa_v25")
    or _v25.PREREGISTRATION_COMMIT
    != "f00e20df3b429f9242516ac38f67fea587e04b22"
    or _v25.MAXIMUM_UPDATES != 1_000
    or _v25.MAXIMUM_PRESENTATIONS != 16_000
    or _v25.CURRENT_EXECUTION_AUTHORIZED is not False
):
    raise RuntimeError("frozen V25 executor defaults changed")

# Preserve the complete inherited compatibility surface.  Functions retain
# their exact frozen V25 implementations; only the lifecycle globals below
# are retargeted in this private adapter instance.
for _name, _value in vars(_v25).items():
    if _name not in {
        "ROOT",
        "SCHEMA_PREFIX",
        "PREREGISTRATION_COMMIT",
        "PREREGISTRATION_PATH",
        "PREREGISTRATION_FILE_SHA256",
        "PREREGISTRATION_BYTE_COUNT",
        "OUTPUT_ROOT_RELATIVE_PATH",
        "BOUND_PARENT_SOURCES",
        "CURRENT_EXECUTION_AUTHORIZED",
        "CURRENT_EXECUTION_DENIAL",
        "main",
    }:
        globals().setdefault(_name, _value)

_engine = _v25._engine
_bound_parent_sources = dict(_v25.BOUND_PARENT_SOURCES)
_bound_parent_sources.update(
    {
        PREREGISTRATION_PATH: (
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_BYTE_COUNT,
        ),
        V25_TERMINAL_FAILURE_RESULT_PATH: (
            V25_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
    }
)
for _module in (
    _v25,
    _v25._v24,
    _v25._v24._v23._base,
    _v25._v24._v23,
    _engine,
):
    _module.SCHEMA_PREFIX = SCHEMA_PREFIX
    _module.PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
    _module.PREREGISTRATION_PATH = PREREGISTRATION_PATH
    _module.OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH
    _module.BOUND_PARENT_SOURCES = _bound_parent_sources
_v25._bound_parent_sources = _bound_parent_sources
_engine.CURRENT_EXECUTION_AUTHORIZED = False
_engine.CURRENT_EXECUTION_DENIAL = (
    "V26 schema-integrity successor execution is denied until recursive source "
    "closure, independent review, narrow certification, and one-shot authority"
)
_v25.CURRENT_EXECUTION_AUTHORIZED = False
_v25.CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL

BOUND_PARENT_SOURCES = _engine.BOUND_PARENT_SOURCES
CURRENT_EXECUTION_AUTHORIZED = _engine.CURRENT_EXECUTION_AUTHORIZED
CURRENT_EXECUTION_DENIAL = _engine.CURRENT_EXECUTION_DENIAL
TRAINING_REQUIRED_BATCH_KEYS_V25 = tuple(_v25.TRAINING_REQUIRED_BATCH_KEYS_V25)
TRAINING_REQUIRED_BATCH_KEYS_V26 = TRAINING_REQUIRED_BATCH_KEYS_V25
TRAINING_REQUIRED_BATCH_KEYS = TRAINING_REQUIRED_BATCH_KEYS_V26


def validate_bound_sources_v26(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Delegate exact V25 binding checks over the V26 evidence closure."""

    selected = BOUND_PARENT_SOURCES if bindings is None else bindings
    return _v25.validate_bound_sources_v25(repository_root, selected)


def validate_training_api_v26(module: Any) -> dict[str, Any]:
    """Validate V25 science plus the sole V26 schema-alias correction."""

    receipt = dict(_v25.validate_training_api_v25(module))
    frozen = getattr(module, "_v25", None)
    if frozen is None:
        raise RuntimeError("V26 training private frozen V25 module is absent")
    full = getattr(frozen, "_validate_microbatches_v25", None)
    projected_v21 = getattr(frozen._v24, "_validate_microbatches_v21", None)
    projected_v23 = getattr(frozen._v24, "_validate_microbatches_v23", None)
    expected = {
        "_validate_microbatches_v13": full,
        "_validate_microbatches_v21": projected_v21,
        "_validate_microbatches_v23": projected_v23,
        "_validate_microbatches_v24": full,
        "_validate_microbatches_v25": full,
        "_validate_microbatches_v26": full,
    }
    if (
        not all(callable(value) for value in expected.values())
        or any(getattr(module, name, None) is not value for name, value in expected.items())
    ):
        raise RuntimeError("V26 training schema compatibility aliases changed")
    return {
        **receipt,
        "required_batch_key_count_v26": len(TRAINING_REQUIRED_BATCH_KEYS_V26),
        "science_identical_to_v25": True,
        "schema_integrity_alias_correction_only": True,
        "v21_projected_validator_restored": True,
        "v23_projected_validator_restored": True,
    }


validate_model_api_v26 = _v25.validate_model_api_v25
validate_microbatches_for_engine_v26 = _v25.validate_microbatches_for_engine_v25
validate_update_integrity_v26 = _v25.validate_update_integrity_v25
observation_v26 = _v25.observation_v25
validate_terminal_accounting_v26 = _v25.validate_terminal_accounting_v25
run_future_authorized_engine_v26 = _v25.run_future_authorized_engine_v25
validate_content_bound_v26 = _v25.validate_content_bound_v25
validate_future_execution_prerequisites_v26 = (
    _v25.validate_future_execution_prerequisites_v25
)
execution_denial_receipt_v26 = _v25.execution_denial_receipt_v25
reserve_attempt_v26 = _v25.reserve_attempt_v25
terminalize_failure_v26 = _v25.terminalize_failure_v25
flatten_physical_metrics_v26 = _v25.flatten_physical_metrics_v25
registered_wrong_rgb_mapping_v26 = _v25.registered_wrong_rgb_mapping_v25
evaluate_update400_gate_v26 = _v25.evaluate_update400_gate_v25
evaluate_final_gate_v26 = _v25.evaluate_final_gate_v25
validate_schedule_v26 = _v25.validate_schedule_v25
validate_attempt_reservation_v26 = _v25.validate_attempt_reservation_v25
execute_v26 = _v25.execute_v25

_engine.validate_bound_sources_v13 = validate_bound_sources_v26
_engine.validate_training_api_v13 = validate_training_api_v26

# Compatibility names consumed by the unchanged inherited launcher/runtime.
validate_bound_sources_v13 = validate_bound_sources_v26
validate_training_api_v13 = validate_training_api_v26
execute_v13 = execute_v26


def private_adapter_receipt_v26() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_private_v25_executor_adapter_v1",
        "base_executor": str(V25_EXECUTOR_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": (
            V25_FROZEN_SOURCE_AND_REVIEW_COMMIT
        ),
        "base_executor_file_sha256": V25_EXECUTOR_FILE_SHA256,
        "base_executor_byte_count": V25_EXECUTOR_BYTE_COUNT,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v25_terminal_failure_result_commit": (
            V25_TERMINAL_FAILURE_RESULT_COMMIT
        ),
        "v25_terminal_failure_result_content_sha256": (
            V25_TERMINAL_FAILURE_RESULT_CONTENT_SHA256
        ),
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "scientific_behavior_delegated_exactly_to_v25": True,
        "v25_recovery_writer_delegated_exactly": True,
        "schema_integrity_alias_correction_owned_by_training_wrapper": True,
        "model_data_seed_schedule_losses_thresholds_initialization_unchanged": True,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "retry_authorized": False,
        "resume_authorized": False,
        "public_v25_was_loaded_before_adapter": _PUBLIC_V25_WAS_LOADED_BEFORE_ADAPTER,
        "public_v25_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_V25_MODULE_NAME in sys.modules,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V26 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v26(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
