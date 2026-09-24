from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as frozen
from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
)
from scripts import finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as finalizer


ROOT = Path(__file__).resolve().parents[2]


def test_finalizer_uses_unchanged_frozen_n5_thresholds() -> None:
    assert asdict(frozen.FIT_THRESHOLDS[5]) == {
        "pixel_hit_balanced_accuracy_min": 0.99,
        "pixel_hit_depth_median_error_m_max": 0.06,
        "pixel_hit_depth_p95_error_m_max": 0.15,
        "ground_overall_balanced_accuracy_min": 0.99,
        "ground_distance_balanced_accuracy_min": 0.97,
        "ground_family_balanced_accuracy_min": 0.97,
        "raster_nll_max": 0.06,
        "raster_balanced_accuracy_min": 0.99,
        "raster_class_recall_min": 0.97,
        "wrong_pixel_balanced_accuracy_drop_min": 0.08,
        "wrong_depth_median_error_increase_m_min": 0.08,
        "wrong_depth_p95_error_increase_m_min": 0.12,
        "wrong_ground_balanced_accuracy_drop_min": 0.08,
        "wrong_raster_nll_increase_min": 0.08,
        "wrong_raster_balanced_accuracy_drop_min": 0.08,
    }


def test_finalizer_revalidates_receipt_and_never_authorizes_execution_directly() -> None:
    source = (ROOT / policy.FINALIZER_RELATIVE_PATH).read_text()
    validate_body = source[
        source.index("def _validate_metric_receipt") : source.index("def run")
    ]
    run_body = source[source.index("def run") : source.index("def _isolated_child")]
    assert "policy.validate_evaluation_structure" in validate_body
    assert "frozen._validated_metric_evaluation" in validate_body
    assert "frozen._gate_stage" in validate_body
    assert 'receipt.get("numeric_gate") != numeric' in validate_body
    assert run_body.index("policy.verify_authority") < run_body.index(
        "_validate_attempt_bundle"
    )
    assert '"later_rung_design_review_authorized": passes' in run_body
    for forbidden in (
        '"n16_execution_authorized": True',
        '"second_seed_authorized": True',
        '"holdout_authorized": True',
        '"g2_authorized": True',
        '"runtime_authorized": True',
        '"promotion_authorized": True',
    ):
        assert forbidden not in run_body


def test_finalizer_is_stdlib_only_until_review_preflight() -> None:
    source = (ROOT / policy.FINALIZER_RELATIVE_PATH).read_text()
    tree = ast.parse(source)
    top_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torch" not in top_imports
    assert "numpy" not in top_imports
    assert "PIL" not in top_imports


def test_finalizer_cli_requires_all_exact_bindings() -> None:
    args = finalizer.parse_args(
        [
            "--source-review", "review.json",
            "--source-review-sha256", "0" * 64,
            "--reservation", f"reservation.json:{'1' * 64}",
            "--result", f"result.json:{'2' * 64}",
            "--checkpoint", f"checkpoint.pt:{'3' * 64}",
            "--completion", f"completed.json:{'4' * 64}",
            "--metric-verification", f"metric.json:{'5' * 64}",
        ]
    )
    assert args.metric_verification.endswith("5" * 64)
    with pytest.raises(SystemExit):
        finalizer.parse_args([])
