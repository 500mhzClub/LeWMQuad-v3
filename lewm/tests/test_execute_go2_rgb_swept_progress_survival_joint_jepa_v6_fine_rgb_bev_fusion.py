from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion.py"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v6_inherits_v4_data_schedule_losses_gate_and_caps() -> None:
    module = _load("_test_v6_executor_bindings")
    for name in (
        "LABEL_ROOT_RELATIVE_PATH", "LABEL_MANIFEST_NAME",
        "LABEL_MANIFEST_CONTENT_SHA256", "LABEL_MANIFEST_FILE_SHA256",
        "LABEL_MANIFEST_BYTE_COUNT", "REQUIRED_GPU_NAME", "REQUIRED_GPU_MEMORY_BYTES",
        "ACTION_ORDER", "ROLE_FILES", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "CONSTRUCTOR_INITIALIZATION_SEED", "SEMANTIC_DECODER_INITIALIZATION_SEED",
        "EXPERIMENT_SEED", "BOOTSTRAP_SEED", "CONTROL_NAMES", "ALL_ARM_NAMES",
        "GATE_THRESHOLDS",
    ):
        assert getattr(module, name) == getattr(module._v4, name)
    assert module.evaluate_gate_v6 is module._v4.evaluate_gate_v4
    assert module.FINE_RGB_BRANCH_INITIALIZATION_SEED == 20_260_714
    assert module.FINE_RGB_BRANCH_ADDED_PARAMETER_COUNT == 12_256


def test_architecture_receipt_is_exact_and_detached() -> None:
    module = _load("_test_v6_executor_architecture")
    receipt = module.fine_rgb_architecture_receipt_v6()
    assert receipt["added_trainable_parameter_count"] == 12_256
    assert receipt["branch"][0] == {
        "type": "Conv2d", "in_channels": 3, "out_channels": 32,
        "kernel_size": 3, "stride": 1, "padding": 1, "bias": True,
    }
    assert receipt["branch"][-1]["initialization"] == "exact_zero"
    assert receipt["sampling"]["align_corners"] is False
    receipt["branch"][0]["out_channels"] = 1
    assert module.FINE_RGB_ARCHITECTURE["branch"][0]["out_channels"] == 32


def test_output_is_write_once_and_calibration_remains_separate(tmp_path: Path) -> None:
    module = _load("_test_v6_executor_output")
    output = module._fresh_output_root_v6(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="fine-RGB-BEV-fusion"):
        module._fresh_output_root_v6(tmp_path)
    passed = module._physical_calibration_stage_v6(True)
    failed = module._physical_calibration_stage_v6(False)
    assert passed["status"] == "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    assert failed["status"] == "CLOSED_FULL_ARM_GATE_FAILED"
    assert passed["physical_calibration_run_in_this_attempt"] is False


def test_source_executes_v6_once_and_emits_truthful_scope_receipts() -> None:
    source = ENTRYPOINT.read_text()
    assert source.count("training_v6.run_fixed_training_v6(") == 1
    assert "training_v3.run_fixed_training_v3(" not in source
    assert "execute_v4(" not in source
    assert "torch.load(" not in source
    assert "near_field_hazard" not in source
    assert "GeometryAnchoredSweptProgressSurvivalJointJepaV6(" in source
    assert '"initialization_source": "exact_n320_encoder_only"' in source
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"only_change": "fine_rgb_bev_fusion_branch"' in source
    assert '"model_changed": True' in source
    assert '"losses_changed": False' in source
    assert '"heldout_or_sealed_opened": False' in source
    assert '"retry_or_resume_authorized": False' in source
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    assert checkpoint_write < evaluation
    assert trace_write < evaluation
