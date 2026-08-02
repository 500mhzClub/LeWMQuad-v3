from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = (
    ROOT / "scripts/analyze_go2_world_model_counterfactual_calibration_v1.py"
)


def _load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_calibration_analyzer_v1", ANALYZER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(marker: str) -> dict[str, object]:
    return {
        "path": f"/synthetic/{marker}.json",
        "file_sha256": f"{len(marker) % 16:x}" * 64,
        "byte_count": len(marker) + 1,
    }


def _collection(*, collapse_last_repeat_action: bool = False) -> dict[str, object]:
    states = []
    frame_receipts = {}
    for group_index in range(16):
        state_id = f"state-{group_index}"
        branches = []
        for action in range(9):
            branches.append({
                "action_id": action,
                "clipped": False,
                "physical_fell": False,
                "physical_tipped": False,
                "physical_target_progress_m": action * 0.1,
                "physical_path_length_m": 1.0 + action * 0.01,
                "endpoint_state": {
                    "base_pos_world": [action * 0.1, 0.0, 0.3],
                    "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            })
        repeat_action = group_index % 9
        if collapse_last_repeat_action and group_index == 8:
            repeat_action = 0
        sentinel = copy.deepcopy(branches[repeat_action])
        sentinel["action_id"] = repeat_action
        states.append({
            "state": {"scene_id": f"scene-{group_index // 2}"},
            "document": {"state": {"state_index_in_scene": group_index % 2}},
            "branches": [*branches, sentinel],
        })
        for context_index in range(3):
            identity = f"{state_id}:context:{context_index}"
            frame_receipts[identity] = {
                "frame_identity": identity,
                "byte_count": 100 + context_index,
            }
        for target_index in range(10):
            identity = f"{state_id}:candidate:{target_index}"
            frame_receipts[identity] = {
                "frame_identity": identity,
                "byte_count": 200 + target_index,
            }
    scene_metrics = [
        {
            "physics_build_wall_seconds": 0.1,
            "render_scene_build_wall_seconds": 0.1,
            "common_prefix_step_wall_seconds": 0.2,
            "branch_step_wall_seconds": 0.1,
            "native_render_wall_seconds": 0.2,
            "camera_quality_resize_wall_seconds": 0.1,
            "png_encode_write_hash_wall_seconds": 0.1,
            "post_lockstep_receipt_wall_seconds": 0.2,
            "scene_total_wall_seconds": 1.0,
        }
        for _ in range(8)
    ]
    return {
        "purpose": "sizing_calibration_only",
        "counts": {
            "scenes": 8,
            "states": 16,
            "roles": {"calibration": 16},
            "candidate_branches": 144,
            "sentinel_branches": 16,
            "total_branches": 160,
        },
        "states": states,
        "frame_receipts": frame_receipts,
        "document": {
            "attempt_id": "calibration-attempt-v1",
            "scene_metrics": scene_metrics,
            "collection_wall_seconds": 8.5,
        },
    }


def test_analyzer_derives_tolerances_and_all_action_repeat_coverage() -> None:
    analyzer = _load_analyzer()
    receipt = analyzer.derive_calibration_receipt_v1(
        _collection(),
        collection_binding=_binding("collection"),
        analyzer_binding=_binding("analyzer"),
        checker_binding=_binding("checker"),
        joiner_binding=_binding("joiner"),
    )
    assert receipt["decision"] == "FREEZE_PILOT_CONTRACT"
    assert receipt["calibration_contract"]["progress_tolerance_m"] == 1e-6
    assert receipt["calibration_contract"]["path_length_tolerance_m"] == 1e-6
    assert receipt["repeatability_analysis"]["repeated_action_ids"] == [
        index % 9 for index in range(16)
    ]
    assert receipt["repeatability_analysis"]["all_requested_primitives_covered"] is True
    assert receipt["repeatability_analysis"]["interpretation"] == (
        "deterministic_replay_gate_not_empirical_noise_estimate"
    )
    assert receipt["calibration_contract"]["tolerance_derivation"][
        "empirical_noise_scale_estimated"
    ] is False
    assert receipt["visual_validation"]["visual_domain_fidelity_claimed"] is False
    assert receipt["resource_measurements"]["stored_rgb_png"]["total_frames"] == 208
    assert receipt["resource_measurements"]["outcome_counts"][
        "camera_invalid_frames"
    ] == 0
    analyzer.validate_calibration_receipt_v1(
        receipt, verify_external_bindings=False
    )


def test_analyzer_rejects_repeat_panel_that_misses_a_primitive() -> None:
    analyzer = _load_analyzer()
    with pytest.raises(analyzer.CalibrationAnalysisError, match="all nine"):
        analyzer.derive_calibration_receipt_v1(
            _collection(collapse_last_repeat_action=True),
            collection_binding=_binding("collection"),
            analyzer_binding=_binding("analyzer"),
            checker_binding=_binding("checker"),
            joiner_binding=_binding("joiner"),
        )


def test_analyzer_rejects_nonexact_repeat_as_contract_drift() -> None:
    analyzer = _load_analyzer()
    collection = _collection()
    collection["states"][0]["branches"][-1]["physical_target_progress_m"] += 0.01
    with pytest.raises(analyzer.CalibrationAnalysisError, match="exact deterministic"):
        analyzer.derive_calibration_receipt_v1(
            collection,
            collection_binding=_binding("collection"),
            analyzer_binding=_binding("analyzer"),
            checker_binding=_binding("checker"),
            joiner_binding=_binding("joiner"),
        )
