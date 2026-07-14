from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
)
from scripts import verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as verifier


ROOT = Path(__file__).resolve().parents[2]


def _synthetic_control(*, wrong: bool) -> dict[str, object]:
    mapping = [1, 2, 3, 4, 0] if wrong else [0, 1, 2, 3, 4]
    components = {
        "ordered_first_hit_nll": 0.8,
        "target_bin_offset_smooth_l1": 0.02,
        "ground_clear_distance_state_balanced_bce": 0.04,
        "derived_raster_hierarchical_bce": 0.2,
    }
    return {
        "control": (
            "wrong_rgb_with_target_calibration" if wrong else "matched_rgb"
        ),
        "wrong_rgb_degenerate_singleton": False,
        "image_index_mapping": mapping,
        "image_mapping_sha256": policy.canonical_json_sha256(mapping),
        "losses": {
            **components,
            "total": 0.25 * sum(
                components[name] for name in policy.LOSS_COMPONENTS
            ),
        },
        "metrics": {},
    }


def test_structural_arithmetic_is_unchanged_and_cannot_be_repaired() -> None:
    evaluation = {
        "matched_rgb": _synthetic_control(wrong=False),
        "wrong_rgb_with_target_calibration": _synthetic_control(wrong=True),
    }
    policy.validate_evaluation_structure(evaluation)
    broken = copy.deepcopy(evaluation)
    broken["matched_rgb"]["losses"]["total"] += 2e-9  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="losses are inconsistent"):
        policy.validate_evaluation_structure(broken)


def test_verifier_recomputes_inference_and_compares_exact_result() -> None:
    source = (ROOT / policy.VERIFIER_RELATIVE_PATH).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "recompute_evaluation" in functions
    compute_source = ast.get_source_segment(source, functions["recompute_evaluation"])
    receipt_source = ast.get_source_segment(source, functions["_compute_receipt"])
    assert compute_source is not None and "for index in range(5)" in compute_source
    assert receipt_source is not None
    assert 'if evaluation != result["evaluation"]' in receipt_source
    assert "gate._validated_metric_evaluation" in receipt_source
    assert "gate._gate_stage" in receipt_source
    assert '"metric_repair_applied": False' in receipt_source
    assert '"threshold_weakened": False' in receipt_source
    assert '"result_metrics_reused": False' in receipt_source


def test_verifier_preflights_before_attempt_or_heavy_import() -> None:
    source = (ROOT / policy.VERIFIER_RELATIVE_PATH).read_text()
    run_body = source[source.index("def run") : source.index("def _isolated_child")]
    assert run_body.index("policy.verify_authority") < run_body.index(
        "_validate_attempt_bundle"
    )
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
    assert not ({"torch", "numpy", "PIL"} & top_imports)


def test_verifier_cli_has_only_caller_hash_bindings() -> None:
    args = verifier.parse_args(
        [
            "--source-review", "review.json",
            "--source-review-sha256", "0" * 64,
            "--reservation", f"reservation.json:{'1' * 64}",
            "--result", f"result.json:{'2' * 64}",
            "--checkpoint", f"checkpoint.pt:{'3' * 64}",
            "--completion", f"completed.json:{'4' * 64}",
        ]
    )
    assert args.result.endswith("2" * 64)
    with pytest.raises(SystemExit):
        verifier.parse_args([])
