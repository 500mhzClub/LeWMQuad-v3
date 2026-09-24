from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_go2_world_model_existing_pool_support_v1 as audit


def _physical_row(
    *,
    index: int,
    scene: str,
    family: str,
    history: tuple[int, int],
    action: int,
    offset: float,
) -> dict:
    requested_vx = 0.1 * (action - 4)
    requested_yaw = 0.05 * (action - 4)
    return {
        "role": "train",
        "index": index,
        "scene_id": scene,
        "family": family,
        "history_actions": list(history),
        "candidate_action": action,
        "state_vector": [offset, offset * 0.5, float(history[0]), float(history[1])],
        "current": {
            "base_z_m": 0.3 + offset,
            "roll_rad": 0.01,
            "pitch_rad": -0.02,
            "twist_linear_x_mps": requested_vx,
            "twist_linear_y_mps": 0.0,
            "twist_linear_z_mps": 0.0,
            "twist_angular_x_radps": 0.0,
            "twist_angular_y_radps": 0.0,
            "twist_angular_z_radps": requested_yaw,
            "joint_position_l2": 3.0,
            "joint_velocity_l2": 2.0,
        },
        "history_egomotion": {
            "d01_forward_m": 0.01,
            "d01_lateral_m": 0.0,
            "d01_yaw_rad": 0.0,
            "d12_forward_m": 0.01,
            "d12_lateral_m": 0.0,
            "d12_yaw_rad": 0.0,
        },
        "candidate_outcome": {
            "forward_m": requested_vx * 0.5,
            "lateral_m": 0.0,
            "yaw_rad": requested_yaw * 0.5,
            "planar_m": abs(requested_vx * 0.5),
            "vertical_m": 0.0,
        },
        "requested": {
            "vx_tape": [requested_vx] * 5,
            "vy_tape": [0.0] * 5,
            "yaw_rate_tape": [requested_yaw] * 5,
            "mean_vx_mps": requested_vx,
            "mean_vy_mps": 0.0,
            "mean_yaw_rate_radps": requested_yaw,
        },
        "realized": {
            "mean_twist_linear_x_mps": requested_vx * 0.8,
            "mean_twist_linear_y_mps": 0.0,
            "mean_twist_angular_z_radps": requested_yaw * 0.75,
        },
        "execution_or_clipping_metadata_keys": [],
        "joint_effort_available": False,
    }


def test_body_delta_is_egocentric_and_wraps_yaw() -> None:
    observed = audit.body_delta(
        (1.0, 2.0, 0.3, math.pi / 2.0),
        (1.0, 3.0, 0.4, -math.pi + math.pi / 2.0 + 0.1),
    )
    assert observed[0] == pytest.approx(1.0)
    assert observed[1] == pytest.approx(0.0, abs=1e-12)
    assert observed[2] == pytest.approx(-math.pi + 0.1)
    assert observed[3] == pytest.approx(1.0)
    assert observed[4] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "path",
    (
        Path("x/sealed/y"),
        Path("x/sealed_v4/y"),
        Path("x/sealed_test.json"),
        Path("x/heldout/y"),
        Path("x/raw/y"),
        Path("x/labels/y"),
    ),
)
def test_protected_and_out_of_scope_paths_fail_closed(path: Path) -> None:
    with pytest.raises(audit.PoolSupportAuditError):
        audit._reject_protected_path(path, label="test")


def test_summarize_index_distinguishes_exact_from_local_support() -> None:
    rows = []
    for index in range(18):
        scene = f"family_{index // 9}"
        action = index % 9
        rows.append(
            SimpleNamespace(
                scene_id=scene,
                family="family",
                rgb=(f"{scene}/0", f"{scene}/1", f"{scene}/{index}", f"{scene}/3"),
                actions=(index % 3, (index + 1) % 3, action, 0, 0, 0),
            )
        )
    summary = audit.summarize_index({"train": rows, "val": rows})
    assert summary["train"]["row_count"] == 18
    assert summary["train"]["scene_count"] == 2
    assert summary["train"]["full_nine_action_support_scene_count"] == 2
    assert summary["train"]["exact_preaction_histories_with_multiple_candidate_actions"] == 0
    assert summary["cross_role_scene_overlap_count"] == 2


def test_summarize_census_preserves_full_pool_action_counts() -> None:
    action_counts = {f"p2:{name}": index + 1 for index, name in enumerate(audit.PRIMITIVES)}
    census = {
        "totals": {
            "byte_count": 1,
            "packed_h6": 2,
            "primitive_transitions": 3,
            "row_count": 4,
            "sliding_h6": 5,
            "source_count": 6,
        },
        "by_role_family": {
            "train": {"family": {"action_position_counts": action_counts}},
            "val": {"family": {"action_position_counts": action_counts}},
        },
    }
    summary = audit.summarize_census(census)
    assert summary["totals"]["sliding_h6"] == 5
    assert summary["sliding_h6_candidate_position_p2_action_counts"]["train"]["hold"] == 7


def test_neighborhood_support_reports_action_overlap() -> None:
    rows = []
    for scene_index in range(20):
        for action in range(9):
            rows.append(
                _physical_row(
                    index=len(rows),
                    scene=f"scene_{scene_index}",
                    family="family",
                    history=(0, 1),
                    action=action,
                    offset=scene_index * 1e-3 + action * 1e-5,
                )
            )
    summary = audit.neighborhood_support(
        rows,
        rows,
        k=9,
        exclude_same_scene=True,
    )
    assert summary["all"]["eligible_query_fraction"] == 1.0
    assert summary["history"]["eligible_query_fraction"] == 1.0
    assert summary["family_history"]["mean_unique_local_action_count"] >= 5.0
    assert summary["all"]["zero_local_factual_action_support_fraction"] < 0.5
    action = summary["all"]["per_action_factual_support"]["arc_left"]
    assert action["total_query_count"] == 20
    assert action["eligible_query_fraction"] == 1.0
    assert 0.0 <= action["k_neighbor_and_factual_action_support_query_fraction"] <= 1.0


def test_physical_summary_does_not_call_realized_twist_executed_command() -> None:
    rows = [
        _physical_row(
            index=action,
            scene=f"scene_{action}",
            family="family",
            history=(0, 1),
            action=action,
            offset=action * 0.01,
        )
        for action in range(9)
    ]
    summary = audit.summarize_physical(rows)
    executed = summary["executed_or_clipping_metadata"]
    assert executed["matching_key_row_count"] == 0
    assert executed["exact_requested_vs_executed_command_comparison_available"] is False
    assert "not the controller's" in executed["interpretation"]
    assert summary["requested_command"]["unique_tape_count_by_action"]["hold"] == 1
    assert summary["motion_density_by_action"]["arc_left"]["row_count"] == 1
