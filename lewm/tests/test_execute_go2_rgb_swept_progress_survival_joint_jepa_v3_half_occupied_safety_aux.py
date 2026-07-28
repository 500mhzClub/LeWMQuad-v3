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
    / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux.py"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v3_reuses_every_frozen_predecessor_execution_binding() -> None:
    module = _load("_test_half_occupied_safety_execute_bindings")
    predecessor = module._v2
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
        assert getattr(module, name) == getattr(predecessor, name)
    assert module.OUTPUT_RELATIVE_PATH == (
        ".generated/"
        "go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux/"
        "attempt_v1"
    )


def test_auxiliary_receipt_changes_only_coefficient_to_one_half() -> None:
    module = _load("_test_half_occupied_safety_execute_receipt")
    expected = dict(module._v2.AUXILIARY_OBJECTIVE)
    expected["coefficient"] = 0.5
    assert module.auxiliary_objective_receipt_v3() == expected
    assert expected["logit_definition"] == (
        "occupied_semantic_logit_minus_logsumexp_free_and_unknown_semantic_logits"
    )
    assert expected["row_balancing"] == (
        "per_raster_row_equal_average_of_present_occupied_and_rest_target_classes"
    )
    assert expected["current_next_aggregation"] == "equal_average"
    assert expected["normalization"] == (
        "binary_cross_entropy_with_logits_divided_by_log_2"
    )
    receipt = module.auxiliary_objective_receipt_v3()
    receipt["coefficient"] = 99.0
    assert module.AUXILIARY_OBJECTIVE["coefficient"] == 0.5


def test_unchanged_metrics_and_gate_are_direct_helper_reuse() -> None:
    module = _load("_test_half_occupied_safety_execute_metrics")
    assert module.scientific_metrics_v3 is module._v2.scientific_metrics_v2
    assert module.semantic_metrics_v3 is module._v2.semantic_metrics_v2
    assert module.paired_control_comparison_v3 is module._v2.paired_control_comparison_v2
    assert module.evaluate_gate_v3 is module._v2.evaluate_gate_v2

    target_row = np.asarray([15, 0, 2, 4, 6, 8, 0, 10, 12], dtype=np.int64)
    target = np.stack([target_row for _ in range(8)])
    full = target.astype(np.float64) * module.PROGRESS_SEGMENT_M
    informative = np.ones(8, dtype=np.bool_)
    scenes = [f"scene_{index}" for index in range(8)]
    families = list(module.REGISTERED_FAMILIES)
    metrics = module.scientific_metrics_v3(
        full, target, informative, scenes, families, np=np
    )
    assert metrics["overall"]["normalized_chosen_prefix_utility"] == 1.0
    assert metrics["overall"]["selected_zero_prefix_rate"] == 0.0
    assert metrics["overall"]["unequal_pair_concordance"] == 1.0


def test_output_root_is_write_once_and_all_schemas_are_distinct(tmp_path: Path) -> None:
    module = _load("_test_half_occupied_safety_execute_output")
    output = module._fresh_output_root_v3(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="fresh half-occupied-safety"):
        module._fresh_output_root_v3(tmp_path)

    schemas = {
        module.CHECKPOINT_SCHEMA,
        module.TRACE_SCHEMA,
        module.RESULT_SCHEMA,
        module.FAILURE_SCHEMA,
    }
    assert len(schemas) == 4
    assert all("v3_half_occupied_safety_aux" in schema for schema in schemas)


def test_source_calls_v3_once_and_never_names_rejected_artifact_paths() -> None:
    source = ENTRYPOINT.read_text()
    assert source.count("training_v3.run_fixed_training_v3(") == 1
    assert "torch.load(" not in source
    for forbidden in (
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v1/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux/",
    ):
        assert forbidden not in source
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


def test_training_core_guard_rejects_cap_or_coefficient_drift() -> None:
    module = _load("_test_half_occupied_safety_execute_core_guard")

    class Baseline:
        ACTION_ORDER = module.ACTION_ORDER
        MICROBATCH_SIZE = module.MICROBATCH_SIZE
        MICROBATCHES_PER_UPDATE = module.MICROBATCHES_PER_UPDATE
        PRESENTATIONS_PER_UPDATE = module.PRESENTATIONS_PER_UPDATE
        MAXIMUM_UPDATES = module.MAXIMUM_UPDATES
        MAXIMUM_PRESENTATIONS = module.MAXIMUM_PRESENTATIONS

    class Candidate(Baseline):
        OCCUPIED_CLASS_INDEX = 2
        OCCUPIED_SAFETY_AUX_COEFFICIENT = 0.5
        OCCUPIED_SAFETY_AUX_NORMALIZATION = module.math.log(2.0)

        @staticmethod
        def run_fixed_training_v3() -> None:
            return None

    module._validate_training_core_v3(Baseline, Candidate)
    Candidate.OCCUPIED_SAFETY_AUX_COEFFICIENT = 1.0
    with pytest.raises(PermissionError, match="half occupied-safety"):
        module._validate_training_core_v3(Baseline, Candidate)


def test_actual_v3_core_matches_executor_contract_without_runtime_access() -> None:
    module = _load("_test_half_occupied_safety_execute_actual_core")
    training_v1 = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
    )
    training_v3 = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux"
    )
    module._validate_training_core_v3(training_v1, training_v3)
    assert training_v3.OCCUPIED_SAFETY_AUX_COEFFICIENT == (
        module.AUXILIARY_OBJECTIVE["coefficient"]
    )
