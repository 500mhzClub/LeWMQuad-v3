from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
V13_TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
V15_TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


canonical = _load("_test_v15_extended_horizon_canonical_training", V13_TRAINING_PATH)
training = _load("_test_v15_extended_horizon_training", V15_TRAINING_PATH)


def _accounting(update: int) -> object:
    return training.JointTrainingAccountingV13(
        updates=update,
        presentations=16 * update,
        microbatch_graphs=4 * update,
        backward_calls=8 * update,
        camera_route_grad_calls=4 * update,
        joint_route_grad_calls=4 * update,
        camera_frame_objectives=32 * update,
        optimizer_steps=update,
        ema_steps=update,
        predictor_forwards=4 * update,
        predictor_objectives=4 * update,
    )


def test_private_adapter_changes_only_accounting_caps() -> None:
    assert training._training is not canonical
    assert training.PRIVATE_BASE_MODULE_NAME not in sys.modules
    assert canonical.MAXIMUM_UPDATES == 1_000
    assert canonical.MAXIMUM_PRESENTATIONS == 16_000
    assert training.MAXIMUM_UPDATES == 2_000
    assert training.MAXIMUM_PRESENTATIONS == 32_000
    assert training.MICROBATCH_SIZE == canonical.MICROBATCH_SIZE == 4
    assert training.MICROBATCHES_PER_UPDATE == canonical.MICROBATCHES_PER_UPDATE == 4
    assert training.PRESENTATIONS_PER_UPDATE == canonical.PRESENTATIONS_PER_UPDATE == 16
    for name in (
        "ENCODER_LEARNING_RATE",
        "OTHER_ONLINE_LEARNING_RATE",
        "ADAMW_BETAS",
        "ADAMW_EPSILON",
        "ADAMW_WEIGHT_DECAY",
        "REQUIRED_BATCH_KEYS",
    ):
        assert getattr(training, name) == getattr(canonical, name)
    for name in (
        "partition_parameters_v13",
        "build_frozen_optimizer_v13",
        "validate_optimizer_v13",
        "joint_training_update_v13",
        "validate_accounting_v13",
    ):
        assert getattr(training, name).__code__.co_code == getattr(
            canonical, name
        ).__code__.co_code
        assert getattr(training, name).__globals__ is training._training.__dict__


def test_v15_accounting_accepts_2000_and_refuses_an_update_beyond_it() -> None:
    terminal = _accounting(2_000)
    training.validate_accounting_v13(terminal)
    with pytest.raises(PermissionError, match="cap leaves no complete update"):
        training._training._validate_update_capacity_v13(terminal)

    penultimate = _accounting(1_999)
    training._training._validate_update_capacity_v13(penultimate)
    assert training._training._advance_accounting_v13(penultimate) == terminal


def test_source_only_adapter_receipt_binds_the_single_change() -> None:
    receipt = training.private_training_adapter_receipt_v15()
    assert type(receipt["public_base_was_loaded_before_adapter"]) is bool
    receipt.pop("public_base_was_loaded_before_adapter")
    assert receipt == {
        "schema": (
            "lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
            "private_training_adapter_v1"
        ),
        "base_training": (
            "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
            "camera_evidence_bottleneck.py"
        ),
        "public_base_loaded_by_adapter": False,
        "private_module_registered": False,
        "maximum_updates": 2_000,
        "maximum_presentations": 32_000,
        "presentations_per_update": 16,
        "scientific_change": "terminal_accounting_caps_only",
    }
