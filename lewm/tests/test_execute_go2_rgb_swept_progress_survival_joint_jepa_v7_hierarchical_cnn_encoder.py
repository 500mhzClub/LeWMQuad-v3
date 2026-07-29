from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder.py"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v7_inherits_v4_data_schedule_losses_gate_and_caps() -> None:
    module = _load("_test_v7_executor_bindings")
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
    assert module.evaluate_gate_v7 is module._v4.evaluate_gate_v4
    assert module.PREREGISTRATION_COMMIT == "34c4a33e2fa25926b3127e0c893755757426cfd4"
    assert module.HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED == 20_260_715
    assert module.HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT == 1_994_880


def test_architecture_receipt_is_exact_and_detached() -> None:
    module = _load("_test_v7_executor_architecture")
    receipt = module.hierarchical_cnn_architecture_receipt_v7()
    assert receipt["trainable_parameter_count"] == 1_994_880
    assert receipt["stem"]["conv"] == [3, 48, 5, 2, 2]
    assert [stage["width"] for stage in receipt["stages"]] == [48, 96, 192]
    assert receipt["spatial_adapter"] == {
        "type": "bilinear_interpolation", "size": [16, 16], "align_corners": False,
    }
    assert receipt["tokens"]["output_shape"] == [257, 192]
    receipt["stem"]["conv"][1] = 1
    assert module.HIERARCHICAL_CNN_ARCHITECTURE["stem"]["conv"][1] == 48


def test_output_is_write_once_and_calibration_remains_separate(tmp_path: Path) -> None:
    module = _load("_test_v7_executor_output")
    output = module._fresh_output_root_v7(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="hierarchical-CNN"):
        module._fresh_output_root_v7(tmp_path)
    passed = module._physical_calibration_stage_v7(True)
    failed = module._physical_calibration_stage_v7(False)
    assert passed["status"] == "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    assert failed["status"] == "CLOSED_FULL_ARM_GATE_FAILED"
    assert passed["physical_calibration_run_in_this_attempt"] is False


def test_source_executes_v7_once_and_emits_truthful_scope_receipts() -> None:
    source = ENTRYPOINT.read_text()
    assert source.count("training_v7.run_fixed_training_v7(") == 1
    assert "execute_v4(" not in source
    assert "torch.load(" not in source
    assert "GeometryAnchoredSweptProgressSurvivalJointJepaV7(" in source
    assert '"initialization_source": "fresh_hierarchical_cnn_plus_inherited_v4_nonencoder_components"' in source
    assert '"n320_encoder_parameter_initialization_used_by_v7_cnn": False' in source
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"only_change": "wholesale_hierarchical_cnn_encoder_replacement"' in source
    assert '"inherited_nonencoder_components_unchanged": True' in source
    assert '"losses_changed": False' in source
    assert '"heldout_or_sealed_opened": False' in source
    assert '"retry_or_resume_authorized": False' in source
    assert '"checkpoint_access_status": checkpoint_access' in source
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    assert checkpoint_write < evaluation
    assert trace_write < evaluation
