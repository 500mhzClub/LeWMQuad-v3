from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    ROOT / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v1.py"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_population(module: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    target_row = np.asarray([15, 0, 2, 4, 6, 8, 0, 10, 12], dtype=np.int64)
    target = np.stack([target_row for _ in range(8)])
    full = target.astype(np.float64) * module.PROGRESS_SEGMENT_M
    control = np.zeros_like(full)
    control[:, 1] = module.PROGRESS_HORIZON_M
    informative = np.ones(8, dtype=np.bool_)
    scenes = [f"scene_{index}" for index in range(8)]
    families = list(module.REGISTERED_FAMILIES)
    return full, control, target, informative, scenes, families


def test_perfect_progress_metrics_and_paired_bootstrap_are_exact() -> None:
    module = _load("_test_swept_progress_execute_metrics")
    full, control, target, informative, scenes, families = _synthetic_population(module)
    metrics = module.scientific_metrics_v1(
        full, target, informative, scenes, families, np=np
    )
    assert metrics["expected_progress_mae_m"] == 0.0
    assert metrics["overall"]["normalized_chosen_prefix_utility"] == 1.0
    assert metrics["overall"]["selected_zero_prefix_rate"] == 0.0
    assert metrics["overall"]["unequal_pair_concordance"] == 1.0
    assert metrics["progress_calibration"]["weighted_absolute_gap_m"] == 0.0

    comparison = module.paired_control_comparison_v1(
        full,
        control,
        target,
        informative,
        scenes,
        families,
        np=np,
    )
    assert comparison["equal_scene_mean_delta"] == 1.0
    assert comparison["bootstrap_lower_95"] == 1.0
    assert comparison["positive_family_count"] == 8
    assert comparison["bootstrap_seed"] == 20_260_728
    assert comparison["bootstrap_replicates"] == 10_000


def test_compact_gate_passes_exact_floors_but_requires_positive_control_delta() -> None:
    module = _load("_test_swept_progress_execute_gate")
    full, _, target, informative, scenes, families = _synthetic_population(module)
    full_metrics = module.scientific_metrics_v1(
        full, target, informative, scenes, families, np=np
    )
    selection = {name: full_metrics for name in module.ALL_ARM_NAMES}
    semantic = {
        "balanced_accuracy": 0.80,
        "free_recall": 0.85,
        "occupied_recall": 0.70,
        "unknown_recall": 0.90,
        "rough_occupied_recall": 0.65,
    }
    positive = {
        "equal_scene_mean_delta": 0.01,
        "bootstrap_lower_95": 0.001,
        "positive_family_count": 6,
    }
    comparisons = {name: dict(positive) for name in module.CONTROL_NAMES}
    passed = module.evaluate_gate_v1(selection, semantic, comparisons)
    assert passed["passed"] is True
    assert passed["status"] == "PASS_FULL_ARM"

    comparisons["wrong_rgb"]["bootstrap_lower_95"] = 0.0
    failed = module.evaluate_gate_v1(selection, semantic, comparisons)
    assert failed["passed"] is False
    assert "wrong_rgb:positive_bootstrap_lower_95" in failed["failed_checks"]


def test_bound_read_rejects_changed_bytes_and_atomic_output_is_write_once(
    tmp_path: Path,
) -> None:
    module = _load("_test_swept_progress_execute_io")
    source = tmp_path / "synthetic.jsonl"
    raw = b"{}\n"
    source.write_bytes(raw)
    binding = {
        "path": source.name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    assert module._read_bound_file(source, binding) == raw
    source.write_bytes(b"[]\n")
    with pytest.raises(PermissionError, match="bound input changed"):
        module._read_bound_file(source, binding)

    output = tmp_path / "result.json"
    first = module._atomic_write_v1(output, b"first\n")
    assert first["byte_count"] == 6
    assert output.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError, match="write-once"):
        module._atomic_write_v1(output, b"second\n")


def test_execution_caps_seeds_and_staged_no_jepa_boundary_are_fixed() -> None:
    module = _load("_test_swept_progress_execute_constants")
    assert (
        module.MAXIMUM_UPDATES,
        module.MAXIMUM_PRESENTATIONS,
        module.MICROBATCH_SIZE,
        module.MICROBATCHES_PER_UPDATE,
    ) == (1_000, 16_000, 4, 4)
    assert module.CONSTRUCTOR_INITIALIZATION_SEED == 20_260_712
    assert module.EXPERIMENT_SEED == 20_260_728
    assert module.LABEL_MANIFEST_CONTENT_SHA256 == (
        "6e0ea572612cdf94cb6dd91dffb90e50c828053617f69b42307161c958700c03"
    )
    assert module.LABEL_MANIFEST_FILE_SHA256 == (
        "edc0df8c796f97d3f91c8c3796e9795a4355dceac79770b91de382132fe8e1d3"
    )
    assert module.LABEL_MANIFEST_BYTE_COUNT == 5_914
    assert module.REQUIRED_GPU_NAME == "AMD Radeon AI PRO R9700"
    assert module.REQUIRED_GPU_MEMORY_BYTES == 34_208_743_424
    assert module.CONTROL_NAMES == (
        "coordinate_matched_persistence",
        "shuffled_action",
        "wrong_rgb",
        "train_action_mean_prior",
    )
    source = ENTRYPOINT.read_text()
    assert '"status": "STAGED_ONLY_IF_FULL_ARM_PASSES"' in source
    assert '"jepa_treatment_effect_claimed": False' in source
