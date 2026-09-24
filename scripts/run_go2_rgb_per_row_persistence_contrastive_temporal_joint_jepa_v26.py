#!/usr/bin/env python3
"""Source-only V26 schema-integrity adapter over frozen V25 science.

V26 privately loads the exact independently reviewed V25 training source and
delegates every scientific operation to it.  Only this module's compatibility
validator aliases distinguish full V25 batches from the inherited V21 and V23
projected schemas.  Import performs no experiment or accelerator I/O.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "43231c689547b66de83f3cafbfac270455a7a234"
)
BASE_TRAINING_FILE_SHA256 = (
    "063c397c6b4f274b5331c659256631203b143823cb1ed34f6167294ebec91046"
)
BASE_TRAINING_BYTE_COUNT = 37_737
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v25_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_COMMIT_V26 = "0c277fd7350931a7993d5affc2d1d4633ffed916"
PREREGISTRATION_PATH_V26 = (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_preregistration_2026-07-30.md"
)
PREREGISTRATION_FILE_SHA256_V26 = (
    "97061601af2922622673d7e4f8b4c1a6625edcdf899abd647373c28daa192a18"
)
PREREGISTRATION_BYTE_COUNT_V26 = 7_999
SCHEMA_INTEGRITY_CORRECTION_V26 = (
    "restore_inherited_v21_and_v23_projected_schema_validators"
)


def _load_private_base_training_v26() -> ModuleType:
    if BASE_TRAINING_PATH.is_symlink() or not BASE_TRAINING_PATH.is_file():
        raise FileNotFoundError("frozen V25 training source is absent or not regular")
    source = BASE_TRAINING_PATH.read_bytes()
    if (
        len(source) != BASE_TRAINING_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_TRAINING_FILE_SHA256
    ):
        raise RuntimeError("frozen V25 training source binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V25 training module name is already occupied")
    module = ModuleType(PRIVATE_BASE_MODULE_NAME)
    module.__file__ = str(BASE_TRAINING_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(BASE_TRAINING_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_BASE_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_BASE_MODULE_NAME)
    return module


_v25 = _load_private_base_training_v26()
if (
    _v25.PREREGISTRATION_COMMIT_V25
    != "f00e20df3b429f9242516ac38f67fea587e04b22"
    or _v25.MAXIMUM_UPDATES != 1_000
    or _v25.MAXIMUM_PRESENTATIONS != 16_000
    or _v25.PRESENTATIONS_PER_UPDATE != 16
):
    raise RuntimeError("frozen V25 scientific identity or cap changed")

# Re-export the complete inherited lifecycle-facing surface.  Functions retain
# the frozen private V25 globals in which the scientific implementation lives.
for _name in _v25.__all__:
    globals()[_name] = getattr(_v25, _name)

REQUIRED_BATCH_KEYS_V26 = _v25.REQUIRED_BATCH_KEYS_V25
PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V26 = (
    _v25.PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25
)
PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V26 = (
    _v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
)

# V26 names are identity aliases: no wrapper can alter arguments, tensor work,
# accounting, optimizer behavior, EMA ordering, or return values.
JointTrainingAccountingV26 = _v25.JointTrainingAccountingV25
JointUpdateResultV26 = _v25.JointUpdateResultV25
PerRowPersistenceContrastiveTemporalTermsV26 = (
    _v25.PerRowPersistenceContrastiveTemporalTermsV25
)
joint_training_update_v26 = _v25.joint_training_update_v25
validate_accounting_v26 = _v25.validate_accounting_v25
per_row_persistence_contrastive_temporal_loss_v26 = (
    _v25.per_row_persistence_contrastive_temporal_loss_v25
)
predictor_core_protected_survival_objective_v26 = (
    _v25.predictor_core_protected_survival_objective_v25
)
predictor_core_protected_survival_parameter_subset_v26 = (
    _v25.predictor_core_protected_survival_parameter_subset_v25
)
partition_parameters_v26 = _v25.partition_parameters_v25
build_frozen_optimizer_v26 = _v25.build_frozen_optimizer_v25
validate_optimizer_v26 = _v25.validate_optimizer_v25

# Sole functional correction.  These aliases are local to V26; the frozen V25
# module and all function globals inside it remain unchanged.
_validate_microbatches_v13 = _v25._validate_microbatches_v25
_validate_microbatches_v21 = _v25._v24._validate_microbatches_v21
_validate_microbatches_v23 = _v25._v24._validate_microbatches_v23
_validate_microbatches_v24 = _v25._validate_microbatches_v25
_validate_microbatches_v25 = _v25._validate_microbatches_v25
_validate_microbatches_v26 = _v25._validate_microbatches_v25


def private_training_adapter_receipt_v26() -> dict[str, Any]:
    return {
        "schema": (
            "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
            "v26_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": (
            BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT
        ),
        "base_training_file_sha256": BASE_TRAINING_FILE_SHA256,
        "base_training_byte_count": BASE_TRAINING_BYTE_COUNT,
        "public_base_was_loaded_before_adapter": (
            _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT_V26,
        "preregistration_path": PREREGISTRATION_PATH_V26,
        "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256_V26,
        "preregistration_byte_count": PREREGISTRATION_BYTE_COUNT_V26,
        "scientific_source_delegated_by_identity": True,
        "schema_integrity_correction": SCHEMA_INTEGRITY_CORRECTION_V26,
        "v25_private_module_mutated": False,
        "recovery_reader_added": False,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


__all__ = tuple(
    dict.fromkeys(
        (
            *_v25.__all__,
            "JointTrainingAccountingV26",
            "JointUpdateResultV26",
            "PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V26",
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V26",
            "PREREGISTRATION_BYTE_COUNT_V26",
            "PREREGISTRATION_COMMIT_V26",
            "PREREGISTRATION_FILE_SHA256_V26",
            "PREREGISTRATION_PATH_V26",
            "PerRowPersistenceContrastiveTemporalTermsV26",
            "REQUIRED_BATCH_KEYS_V26",
            "SCHEMA_INTEGRITY_CORRECTION_V26",
            "build_frozen_optimizer_v26",
            "joint_training_update_v26",
            "partition_parameters_v26",
            "per_row_persistence_contrastive_temporal_loss_v26",
            "predictor_core_protected_survival_objective_v26",
            "predictor_core_protected_survival_parameter_subset_v26",
            "private_training_adapter_receipt_v26",
            "validate_accounting_v26",
            "validate_optimizer_v26",
        )
    )
)
