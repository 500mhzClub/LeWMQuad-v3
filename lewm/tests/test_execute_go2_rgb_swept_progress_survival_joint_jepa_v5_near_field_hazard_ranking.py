from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    ROOT
    / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_hazard_ranking.py"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v5_inherits_v4_model_data_schedule_gate_and_caps() -> None:
    module = _load("_test_v5_executor_bindings")
    predecessor = module._v4
    for name in (
        "LABEL_ROOT_RELATIVE_PATH",
        "LABEL_MANIFEST_NAME",
        "LABEL_MANIFEST_CONTENT_SHA256",
        "LABEL_MANIFEST_FILE_SHA256",
        "LABEL_MANIFEST_BYTE_COUNT",
        "REQUIRED_GPU_NAME",
        "REQUIRED_GPU_MEMORY_BYTES",
        "ACTION_ORDER",
        "ROLE_FILES",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "CONSTRUCTOR_INITIALIZATION_SEED",
        "SEMANTIC_DECODER_INITIALIZATION_SEED",
        "EXPERIMENT_SEED",
        "BOOTSTRAP_SEED",
        "CONTROL_NAMES",
        "ALL_ARM_NAMES",
        "GATE_THRESHOLDS",
    ):
        assert getattr(module, name) == getattr(predecessor, name)
    assert module.evaluate_gate_v5 is predecessor.evaluate_gate_v4
    assert module.OUTPUT_RELATIVE_PATH.endswith(
        "v5_near_field_hazard_ranking/attempt_v1"
    )


def test_v5_hazard_objective_receipt_is_exact_and_detached() -> None:
    module = _load("_test_v5_executor_objective")
    receipt = module.hazard_ranking_objective_receipt_v5()
    assert receipt["coefficient"] == 1.0
    assert receipt["new_parameter_count"] == 0
    assert receipt["joint_from_update_one"] is True
    assert receipt["raster_centers_m"] == {
        "forward": {"start": -0.95, "stop": 5.35, "count": 64},
        "left": {"start": -3.15, "stop": 3.15, "count": 64},
    }
    assert receipt["near_field_range_m"] == 2.0
    assert receipt["near_field_cell_count"] == 1_016
    assert receipt["pair_set"] == "complete_cartesian_per_raster_row"
    assert receipt["normalization"] == math.log(2.0)
    assert receipt["sampling_or_mining"] is False
    receipt["raster_centers_m"]["forward"]["count"] = 1
    assert module.HAZARD_RANKING_OBJECTIVE["raster_centers_m"]["forward"][
        "count"
    ] == 64


def test_v5_training_core_guard_freezes_inheritance_and_h_contract() -> None:
    module = _load("_test_v5_executor_core_guard")

    class V1:
        ACTION_ORDER = module.ACTION_ORDER
        MICROBATCH_SIZE = module.MICROBATCH_SIZE
        MICROBATCHES_PER_UPDATE = module.MICROBATCHES_PER_UPDATE
        PRESENTATIONS_PER_UPDATE = module.PRESENTATIONS_PER_UPDATE
        MAXIMUM_UPDATES = module.MAXIMUM_UPDATES
        MAXIMUM_PRESENTATIONS = module.MAXIMUM_PRESENTATIONS

    class V3(V1):
        OCCUPIED_CLASS_INDEX = 2
        OCCUPIED_SAFETY_AUX_COEFFICIENT = 0.5
        OCCUPIED_SAFETY_AUX_NORMALIZATION = math.log(2.0)

        @staticmethod
        def run_fixed_training_v3() -> None:
            return None

    class V5(V3):
        RASTER_SIZE = 64
        FORWARD_MIN_M = -0.95
        FORWARD_MAX_M = 5.35
        LEFT_MIN_M = -3.15
        LEFT_MAX_M = 3.15
        NEAR_FIELD_RANGE_M = 2.0
        NEAR_FIELD_CELL_COUNT = 1_016
        HAZARD_RANKING_COEFFICIENT = 1.0
        HAZARD_RANKING_NORMALIZATION = math.log(2.0)

        @staticmethod
        def near_field_hazard_ranking_loss_v5() -> None:
            return None

        @staticmethod
        def run_fixed_training_v5() -> None:
            return None

    module._validate_training_core_v5(V1, V3, V5)
    V5.NEAR_FIELD_CELL_COUNT = 1_015
    with pytest.raises(PermissionError, match="hazard-ranking constants"):
        module._validate_training_core_v5(V1, V3, V5)


def _complete_trace(module: Any) -> list[dict[str, Any]]:
    microbatch = {
        "H": 0.25,
        "hazard_active": True,
        "hazard_current_eligible_row_count": 1,
        "hazard_next_eligible_row_count": 0,
        "hazard_current_ranked_pair_count": 2,
        "hazard_next_ranked_pair_count": 0,
    }
    return [
        {
            "update": update,
            "presentations": update * module.PRESENTATIONS_PER_UPDATE,
            "losses": {
                "S": 0.1,
                "P": 0.2,
                "U": 0.3,
                "R": 0.4,
                "O": 0.5,
                "H": 0.25,
                "L": 1.75,
            },
            "gradient_l2": {
                "encoder": 1.0,
                "lift_semantic": 1.0,
                "predictor": 1.0,
            },
            "hazard_ranking_activity": {
                "microbatches": [dict(microbatch) for _ in range(4)]
            },
        }
        for update in range(1, module.MAXIMUM_UPDATES + 1)
    ]


def test_hazard_activity_receipt_covers_every_microbatch_and_window() -> None:
    module = _load("_test_v5_executor_activity")
    trace = _complete_trace(module)
    receipt = module._hazard_training_receipt_v5(trace)
    assert receipt["update_count"] == 1_000
    assert receipt["microbatch_count"] == 4_000
    assert receipt["active_microbatch_count"] == 4_000
    assert receipt["inactive_microbatch_count"] == 0
    assert receipt["current_eligible_row_count"] == 4_000
    assert receipt["next_eligible_row_count"] == 0
    assert receipt["current_ranked_pair_count"] == 8_000
    assert receipt["next_ranked_pair_count"] == 0
    assert receipt["hazard_loss_microbatch_mean"] == 0.25
    assert len(receipt["windows_100_updates"]) == 10
    assert all(
        window["active_microbatch_count"] == 400
        for window in receipt["windows_100_updates"]
    )

    trace[0]["hazard_ranking_activity"]["microbatches"][0][
        "hazard_active"
    ] = False
    with pytest.raises(RuntimeError, match="inactive hazard receipt"):
        module._hazard_training_receipt_v5(trace)


def test_output_is_write_once_and_status_only_stages_calibration(tmp_path: Path) -> None:
    module = _load("_test_v5_executor_output")
    output = module._fresh_output_root_v5(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="near-field-hazard-ranking"):
        module._fresh_output_root_v5(tmp_path)
    schemas = {
        module.CHECKPOINT_SCHEMA,
        module.TRACE_SCHEMA,
        module.RESULT_SCHEMA,
        module.FAILURE_SCHEMA,
    }
    assert len(schemas) == 4
    assert all("v5_near_field_hazard_ranking" in value for value in schemas)
    passed = module._physical_calibration_stage_v5(True)
    failed = module._physical_calibration_stage_v5(False)
    assert passed["status"] == "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    assert failed["status"] == "CLOSED_FULL_ARM_GATE_FAILED"
    assert passed["physical_calibration_run_in_this_attempt"] is False
    assert passed["physical_gate_passed"] is False


def test_source_calls_v5_once_and_emits_truthful_v5_receipts() -> None:
    source = ENTRYPOINT.read_text()
    assert source.count("training_v5.run_fixed_training_v5(") == 1
    assert "training_v3.run_fixed_training_v3(" not in source
    assert "execute_v4(" not in source
    assert "torch.load(" not in source
    for forbidden in (
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v3_",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v4_",
    ):
        assert forbidden not in source
    assert "model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(" in source
    assert '"initialization_source": "exact_n320_encoder_only"' in source
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"only_change": "near_field_hazard_ranking_loss"' in source
    assert '"model_changed": False' in source
    assert '"losses_changed": True' in source
    assert '"physical_calibration_run_in_this_attempt": False' in source
    assert '"retry_or_resume_authorized": False' in source
    assert '"heldout_or_sealed_opened": False' in source
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    assert checkpoint_write < evaluation
    assert trace_write < evaluation
