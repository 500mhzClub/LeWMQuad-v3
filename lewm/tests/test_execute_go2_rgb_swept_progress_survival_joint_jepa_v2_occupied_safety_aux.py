from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    ROOT
    / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux.py"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_reuses_every_frozen_v1_execution_binding() -> None:
    module = _load("_test_occupied_safety_aux_execute_bindings")
    v1 = module._v1
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
        "EXPERIMENT_SEED",
        "BOOTSTRAP_SEED",
        "CONTROL_NAMES",
        "ALL_ARM_NAMES",
        "GATE_THRESHOLDS",
    ):
        assert getattr(module, name) == getattr(v1, name)
    assert module.OUTPUT_RELATIVE_PATH == (
        ".generated/"
        "go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux/attempt_v1"
    )


def test_auxiliary_receipt_is_the_single_exact_scientific_delta() -> None:
    module = _load("_test_occupied_safety_aux_execute_receipt")
    receipt = module.auxiliary_objective_receipt_v2()
    assert receipt == {
        "name": "occupied_vs_rest_safety",
        "coefficient": 1.0,
        "logit_definition": (
            "occupied_semantic_logit_minus_logsumexp_free_and_unknown_semantic_logits"
        ),
        "row_balancing": (
            "per_raster_row_equal_average_of_present_occupied_and_rest_target_classes"
        ),
        "current_next_aggregation": "equal_average",
        "normalization": "binary_cross_entropy_with_logits_divided_by_log_2",
        "new_trainable_parameters": False,
    }
    receipt["coefficient"] = 99.0
    assert module.AUXILIARY_OBJECTIVE["coefficient"] == 1.0


def test_unchanged_metrics_and_gate_are_direct_v1_helper_reuse() -> None:
    module = _load("_test_occupied_safety_aux_execute_metrics")
    assert module.scientific_metrics_v2 is module._v1.scientific_metrics_v1
    assert module.semantic_metrics_v2 is module._v1.semantic_metrics_v1
    assert module.paired_control_comparison_v2 is module._v1.paired_control_comparison_v1
    assert module.evaluate_gate_v2 is module._v1.evaluate_gate_v1

    target_row = np.asarray([15, 0, 2, 4, 6, 8, 0, 10, 12], dtype=np.int64)
    target = np.stack([target_row for _ in range(8)])
    full = target.astype(np.float64) * module.PROGRESS_SEGMENT_M
    informative = np.ones(8, dtype=np.bool_)
    scenes = [f"scene_{index}" for index in range(8)]
    families = list(module.REGISTERED_FAMILIES)
    metrics = module.scientific_metrics_v2(
        full, target, informative, scenes, families, np=np
    )
    assert metrics["overall"]["normalized_chosen_prefix_utility"] == 1.0
    assert metrics["overall"]["selected_zero_prefix_rate"] == 0.0
    assert metrics["overall"]["unequal_pair_concordance"] == 1.0


def test_output_root_is_write_once_and_v2_schemas_are_distinct(tmp_path: Path) -> None:
    module = _load("_test_occupied_safety_aux_execute_output")
    output = module._fresh_output_root_v2(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="fresh occupied-safety-aux"):
        module._fresh_output_root_v2(tmp_path)

    schemas = {
        module.CHECKPOINT_SCHEMA,
        module.TRACE_SCHEMA,
        module.RESULT_SCHEMA,
        module.FAILURE_SCHEMA,
    }
    assert len(schemas) == 4
    assert all("v2_occupied_safety_aux" in schema for schema in schemas)


def test_source_has_one_v2_training_call_and_no_predecessor_artifact_read() -> None:
    source = ENTRYPOINT.read_text()
    assert source.count("training_v2.run_fixed_training_v2(") == 1
    assert "torch.load(" not in source
    assert (
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v1/attempt_v1"
        not in source
    )
    assert '"initialization_source": "exact_n320_encoder_only"' in source
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"resume_authorized": False' in source
    assert '"checkpoint_qualified": False' in source
    assert '"retry_or_resume_authorized": False' in source
    assert '"status": "STAGED_ONLY_IF_FULL_ARM_PASSES"' in source
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    assert checkpoint_write < evaluation
    assert trace_write < evaluation


def test_training_core_guard_rejects_schedule_or_cap_drift() -> None:
    module = _load("_test_occupied_safety_aux_execute_core_guard")

    class Baseline:
        ACTION_ORDER = module.ACTION_ORDER
        MICROBATCH_SIZE = module.MICROBATCH_SIZE
        MICROBATCHES_PER_UPDATE = module.MICROBATCHES_PER_UPDATE
        PRESENTATIONS_PER_UPDATE = module.PRESENTATIONS_PER_UPDATE
        MAXIMUM_UPDATES = module.MAXIMUM_UPDATES
        MAXIMUM_PRESENTATIONS = module.MAXIMUM_PRESENTATIONS

    class Candidate(Baseline):
        OCCUPIED_CLASS_INDEX = 2
        OCCUPIED_SAFETY_AUX_COEFFICIENT = 1.0
        OCCUPIED_SAFETY_AUX_NORMALIZATION = module.math.log(2.0)

        @staticmethod
        def run_fixed_training_v2() -> None:
            return None

    module._validate_training_core_v2(Baseline, Candidate)
    Candidate.MAXIMUM_PRESENTATIONS += 1
    with pytest.raises(PermissionError, match="MAXIMUM_PRESENTATIONS"):
        module._validate_training_core_v2(Baseline, Candidate)


def test_actual_v2_core_matches_executor_contract_without_runtime_access() -> None:
    module = _load("_test_occupied_safety_aux_execute_actual_core")
    training_v1 = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
    )
    training_v2 = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux"
    )
    module._validate_training_core_v2(training_v1, training_v2)
    assert training_v2.OCCUPIED_SAFETY_AUX_COEFFICIENT == (
        module.AUXILIARY_OBJECTIVE["coefficient"]
    )
