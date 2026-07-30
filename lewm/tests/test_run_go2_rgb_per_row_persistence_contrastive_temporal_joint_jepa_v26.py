from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26.py"
)


def _load_module():
    name = "_v26_training_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_private_adapter_binds_exact_reviewed_v25_source() -> None:
    module = _load_module()
    receipt = module.private_training_adapter_receipt_v26()
    assert receipt == {
        "schema": (
            "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
            "v26_training_adapter_v1"
        ),
        "base_training": (
            "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_"
            "joint_jepa_v25.py"
        ),
        "base_frozen_source_and_review_commit": (
            "43231c689547b66de83f3cafbfac270455a7a234"
        ),
        "base_training_file_sha256": (
            "063c397c6b4f274b5331c659256631203b143823cb1ed34f6167294ebec91046"
        ),
        "base_training_byte_count": 37_737,
        "public_base_was_loaded_before_adapter": False,
        "public_base_loaded_by_adapter": False,
        "private_module_registered": False,
        "preregistration_commit": (
            "0c277fd7350931a7993d5affc2d1d4633ffed916"
        ),
        "preregistration_path": (
            "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_"
            "joint_jepa_v26_preregistration_2026-07-30.md"
        ),
        "preregistration_file_sha256": (
            "97061601af2922622673d7e4f8b4c1a6625edcdf899abd647373c28daa192a18"
        ),
        "preregistration_byte_count": 7_999,
        "scientific_source_delegated_by_identity": True,
        "schema_integrity_correction": (
            "restore_inherited_v21_and_v23_projected_schema_validators"
        ),
        "v25_private_module_mutated": False,
        "recovery_reader_added": False,
        "maximum_updates": 1_000,
        "maximum_presentations": 16_000,
    }


def test_every_inherited_export_and_v26_scientific_name_delegates_by_identity() -> None:
    module = _load_module()
    for name in module._v25.__all__:
        assert getattr(module, name) is getattr(module._v25, name), name

    delegates = {
        "JointTrainingAccountingV26": "JointTrainingAccountingV25",
        "JointUpdateResultV26": "JointUpdateResultV25",
        "PerRowPersistenceContrastiveTemporalTermsV26": (
            "PerRowPersistenceContrastiveTemporalTermsV25"
        ),
        "joint_training_update_v26": "joint_training_update_v25",
        "validate_accounting_v26": "validate_accounting_v25",
        "per_row_persistence_contrastive_temporal_loss_v26": (
            "per_row_persistence_contrastive_temporal_loss_v25"
        ),
        "predictor_core_protected_survival_objective_v26": (
            "predictor_core_protected_survival_objective_v25"
        ),
        "predictor_core_protected_survival_parameter_subset_v26": (
            "predictor_core_protected_survival_parameter_subset_v25"
        ),
        "partition_parameters_v26": "partition_parameters_v25",
        "build_frozen_optimizer_v26": "build_frozen_optimizer_v25",
        "validate_optimizer_v26": "validate_optimizer_v25",
    }
    for v26_name, v25_name in delegates.items():
        assert getattr(module, v26_name) is getattr(module._v25, v25_name)

    assert module.REQUIRED_BATCH_KEYS_V26 is module._v25.REQUIRED_BATCH_KEYS_V25
    assert (
        module.PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V26
        is module._v25.PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25
    )
    assert (
        module.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V26
        is module._v25.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
    )


def test_exact_full_and_projected_schema_aliases_leave_v25_untouched() -> None:
    module = _load_module()
    full = module._v25._validate_microbatches_v25
    projected_v21 = module._v25._v24._validate_microbatches_v21
    projected_v23 = module._v25._v24._validate_microbatches_v23

    assert module._validate_microbatches_v13 is full
    assert module._validate_microbatches_v21 is projected_v21
    assert module._validate_microbatches_v23 is projected_v23
    assert module._validate_microbatches_v24 is full
    assert module._validate_microbatches_v25 is full
    assert module._validate_microbatches_v26 is full
    assert projected_v21 is not full
    assert projected_v23 is not full

    # The consumed V25 source remains frozen, including its two faulty aliases.
    assert module._v25._validate_microbatches_v21 is full
    assert module._v25._validate_microbatches_v23 is full


def test_full_v26_schema_validator_delegates_to_frozen_v24_boundary(
    monkeypatch,
) -> None:
    module = _load_module()
    marker = object()
    batches = tuple(
        {name: marker for name in module.REQUIRED_BATCH_KEYS_V26}
        for _ in range(module.MICROBATCHES_PER_UPDATE)
    )
    calls = []
    monkeypatch.setattr(
        module._v25._v24,
        "_validate_microbatches_v24",
        lambda torch, values: calls.append((torch, values)),
    )
    torch_marker = object()
    module._validate_microbatches_v26(torch_marker, batches)
    assert calls == [(torch_marker, batches)]


def test_v26_training_source_adds_no_recovery_reader_or_resume_path() -> None:
    module = _load_module()
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "torch.load(",
        "load_state_dict(",
        "pickle.load(",
        "resume_from",
        "_read_recovery",
    ):
        assert forbidden not in source
    assert not any(
        "resume" in name.lower() or "recover" in name.lower()
        for name in module.__all__
    )
