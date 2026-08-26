import json

import numpy as np
import pytest

from lewm.safety import body_centric_range_coverage_metrics_v1 as BASE
from lewm.safety import body_centric_range_coverage_reporting_v1 as REPORTING
from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as M
from lewm.safety import (
    minimum_multi_origin_body_range_coverage_qualification_v1_contract as CONTRACT,
)


def _region_masks() -> dict[str, np.ndarray]:
    link = np.arange(M.PROTECTED_LINKS)
    return {
        "TRUNK": link == 0,
        "FRONT_LIMBS": (link >= 1) & (link <= 6),
        "REAR_LIMBS": link >= 7,
        "HIPS_AND_THIGHS": np.isin(link, [1, 2, 4, 5, 7, 8, 10, 11]),
        "CALVES": np.isin(link, [3, 6, 9, 12]),
    }


def _score(layout_id: str, values: tuple[float, ...]) -> dict:
    fields = (
        "min_body_region_support",
        "p5_transition_support",
        "rear_limb_support",
        "calf_support",
        "overall_mean_support",
        "self_occlusion_fraction",
    )
    return {
        "schema": "minimum_multi_origin_training_layout_support_v1",
        "role": "training",
        "label_free": True,
        "layout_id": layout_id,
        **dict(zip(fields, values, strict=True)),
    }


def _next(action_index: int, contact: bool = False) -> dict:
    return {
        "action_index": action_index,
        "controller": "route",
        "oracle_contact": contact,
        "predicted_contact": contact,
    }


def _state(state_id: str, family: str) -> dict:
    return {
        "state_id": state_id,
        "family": family,
        "current_actions": [
            {
                "action_index": 0,
                "controller": "route",
                "oracle_contact": False,
                "predicted_contact": False,
                "h3_progress_m": 0.1,
                "h3_heading_improvement_rad": 0.0,
                "decision_progress_m": 0.1,
                "next_actions": [_next(0)],
            }
        ],
    }


def test_predecessor_contact_calibration_two_ply_and_gate_are_exact_aliases():
    assert M.contact_predictions is BASE.contact_predictions
    assert M.enumerate_threshold_frontier is BASE.enumerate_threshold_frontier
    assert M.reduce_two_ply_state is BASE.reduce_two_ply_state
    assert M.reduce_two_ply_states is BASE.reduce_two_ply_states
    assert M.evaluate_true_future_gate is REPORTING.evaluate_true_future_gate


def test_public_ids_and_classifications_match_the_executable_contract():
    assert M.DUAL_LAYOUT_IDS == CONTRACT.PAIR_LAYOUT_IDS
    assert M.TRIPLE_LAYOUT_IDS == CONTRACT.THREE_LAYOUT_IDS
    assert M.REGION_IDS == CONTRACT.BODY_REGION_IDS
    assert M.CONDITION_IDS == CONTRACT.CONDITION_IDS
    assert M.PRIMARY_CLASSIFICATIONS == CONTRACT.PRIMARY_CLASSIFICATIONS
    assert M.ERROR_CLASSES == CONTRACT.COVERAGE_ERROR_CLASSES
    assert M.BENCHMARK_CLASSIFICATIONS == CONTRACT.COMPUTE_CLASSIFICATIONS


def test_origin_union_and_clearance_fusion_are_conservative_and_exact():
    support = {
        "head": np.asarray([[True, False], [False, False]]),
        "rear": np.asarray([[False, True], [False, False]]),
    }
    clearance = {
        "head": np.asarray([[0.3, 99.0], [99.0, 99.0]]),
        "rear": np.asarray([[99.0, 0.2], [99.0, 99.0]]),
    }
    assert M.union_origin_support(support, ["head", "rear"]).tolist() == [
        [True, True],
        [False, False],
    ]
    fused = M.fuse_origin_clearance(
        origin_clearance_m=clearance,
        origin_support=support,
        origin_ids=["head", "rear"],
    )
    assert fused["minimum_clearance_m"][0].tolist() == [0.3, 0.2]
    assert fused["unsupported"][1].tolist() == [True, True]
    assert np.isposinf(fused["minimum_clearance_m"][1]).all()


def test_float32_per_link_reduction_delegates_ties_and_unsupported_fail_closed():
    shape = (M.PHYSICS_STEPS, M.PROTECTED_LINKS)
    clearance = np.full(shape, 0.3, np.float32)
    clearance[10, 4] = np.float32(0.2)
    result = M.reduce_prematerialized_per_link_contact(
        clearance, np.ones(shape, bool), float(clearance[10, 4])
    )
    assert result["minimum_clearance_m"] == pytest.approx(0.2)
    assert result["predicted_contact"]  # exact threshold tie
    support = np.ones(shape, bool)
    support[-1, -1] = False
    conservative = M.reduce_prematerialized_per_link_contact(
        np.full(shape, 100.0, np.float32), support, 0.0
    )
    assert conservative["unsupported"] and conservative["predicted_contact"]
    with pytest.raises(M.MultiOriginMetricsError, match="float32"):
        M.reduce_prematerialized_per_link_contact(
            clearance.astype(np.float64), np.ones(shape, bool), 0.2
        )


def test_training_layout_summary_uses_complete_witness_and_nominal_denominators():
    support = np.zeros((2, M.PHYSICS_STEPS, M.PROTECTED_LINKS), bool)
    support[1] = True
    nominal = np.ones_like(support)
    occluded = np.zeros_like(support)
    occluded[0, :5] = True  # 5 * 13 of 2 * 50 * 13 nominal queries.
    result = M.summarize_training_layout_support(
        layout_id=M.DUAL_LAYOUT_IDS[0],
        support=support,
        nominal_fov=nominal,
        self_occluded=occluded,
        region_link_masks=_region_masks(),
        transition_ids=["train-b", "train-a"],
    )
    assert result["role"] == "training" and result["label_free"]
    assert result["witness_queries"] == 2 * 50 * 13
    assert result["overall_mean_support"] == 0.5
    assert result["min_body_region_support"] == 0.5
    assert result["p5_transition_support"] == pytest.approx(0.05)
    assert result["self_occlusion_fraction"] == pytest.approx(0.05)
    assert set(result["body_region_support"]) == set(M.REGION_IDS)


def test_training_layout_summary_rejects_incomplete_or_nonphysical_masks():
    shape = (1, M.PHYSICS_STEPS, M.PROTECTED_LINKS)
    masks = _region_masks()
    masks.pop("CALVES")
    with pytest.raises(M.MultiOriginMetricsError, match="exactly"):
        M.summarize_training_layout_support(
            layout_id=M.DUAL_LAYOUT_IDS[0],
            support=np.ones(shape, bool),
            nominal_fov=np.ones(shape, bool),
            self_occluded=np.zeros(shape, bool),
            region_link_masks=masks,
            transition_ids=["train"],
        )
    occluded = np.zeros(shape, bool)
    occluded[0, 0, 0] = True
    with pytest.raises(M.MultiOriginMetricsError, match="inside nominal FOV"):
        M.summarize_training_layout_support(
            layout_id=M.DUAL_LAYOUT_IDS[0],
            support=np.ones(shape, bool),
            nominal_fov=np.zeros(shape, bool),
            self_occluded=occluded,
            region_link_masks=_region_masks(),
            transition_ids=["train"],
        )


def test_layout_selection_obeys_all_six_criteria_then_fixed_id_order():
    tied = [_score(layout_id, (0.5, 0.4, 0.6, 0.6, 0.7, 0.2)) for layout_id in M.DUAL_LAYOUT_IDS]
    selected = M.select_training_layout(tied[::-1], expected_layout_ids=M.DUAL_LAYOUT_IDS)
    assert selected["selected_layout_id"] == M.DUAL_LAYOUT_IDS[0]

    # A smaller first criterion cannot be rescued by every later criterion.
    scores = [
        _score(M.DUAL_LAYOUT_IDS[0], (0.50, 1.0, 1.0, 1.0, 1.0, 0.0)),
        _score(M.DUAL_LAYOUT_IDS[1], (0.51, 0.0, 0.0, 0.0, 0.0, 1.0)),
        _score(M.DUAL_LAYOUT_IDS[2], (0.49, 1.0, 1.0, 1.0, 1.0, 0.0)),
    ]
    assert M.select_training_layout(
        scores, expected_layout_ids=M.DUAL_LAYOUT_IDS
    )["selected_layout_id"] == M.DUAL_LAYOUT_IDS[1]


def test_layout_selector_rejects_nontraining_or_incomplete_candidates():
    rows = [_score(layout_id, (0.5,) * 6) for layout_id in M.DUAL_LAYOUT_IDS]
    rows[0]["role"] = "development-held-out"
    with pytest.raises(M.MultiOriginMetricsError, match="training"):
        M.select_training_layout(rows, expected_layout_ids=M.DUAL_LAYOUT_IDS)
    with pytest.raises(M.MultiOriginMetricsError, match="candidate set"):
        M.select_training_layout(rows[1:], expected_layout_ids=M.DUAL_LAYOUT_IDS)


def test_layout_selector_accepts_streaming_aggregate_aliases_but_no_outcomes():
    rows = [
        {
            "layout_id": layout_id,
            "mount_ids": ["HEAD_STOCK", "REAR_TOP_TRUNK"],
            "region_support": {region: 0.5 for region in M.REGION_IDS},
            "minimum_body_region_support": 0.5,
            "transition_support_p05": 0.4,
            "rear_limb_support": 0.6,
            "calf_support": 0.6,
            "overall_mean_support": 0.7,
            "self_occluded_fraction": 0.2,
            "minimum_family_support": 0.3,
            "per_family_support": {"family-a": 0.3},
            "transition_count": 10,
        }
        for layout_id in M.DUAL_LAYOUT_IDS
    ]
    result = M.select_training_layout(rows, expected_layout_ids=M.DUAL_LAYOUT_IDS)
    assert result["layout_id"] == result["selected_layout_id"] == M.DUAL_LAYOUT_IDS[0]
    rows[0]["oracle_contact_recall"] = 1.0
    with pytest.raises(M.MultiOriginMetricsError, match="forbidden"):
        M.select_training_layout(rows, expected_layout_ids=M.DUAL_LAYOUT_IDS)


def _dual(realistic: bool, spherical: bool = False, fov: bool = False) -> dict[str, bool]:
    return {
        "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND": spherical,
        "DUAL_DENSE_L2_FOV_UPPER_BOUND": fov,
        "DUAL_REALISTIC_L2_SCAN": realistic,
    }


def _triple(realistic: bool, spherical: bool = False, fov: bool = False) -> dict[str, bool]:
    return {
        "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND": spherical,
        "THREE_DENSE_L2_FOV_UPPER_BOUND": fov,
        "THREE_REALISTIC_L2_SCAN": realistic,
    }


def test_conditional_classification_stops_after_dual_realistic_signal():
    result = M.classify_multi_origin_conditions(_dual(True))
    assert result["primary_classification"] == "DUAL_ORIGIN_REALISTIC_RANGE_SIGNAL"
    assert result["executed_condition_ids"] == list(M.DUAL_CONDITION_IDS)
    assert result["replanning_interface_classification"] == "REPLANNING_INTERFACE_UNRESOLVED"


def test_condition_classification_accepts_both_reused_regressions_together():
    result = M.classify_multi_origin_conditions(
        {
            "REALISTIC_PLATFORM_SCAN": False,
            "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT": False,
            **_dual(True),
        }
    )
    assert result["executed_condition_ids"][:2] == list(M.REGRESSION_CONDITION_IDS)
    with pytest.raises(M.MultiOriginMetricsError, match="together"):
        M.classify_multi_origin_conditions(
            {"REALISTIC_PLATFORM_SCAN": False, **_dual(True)}
        )


def test_conditional_classification_priorities_and_all_four_diagnostic():
    triple_signal = M.classify_multi_origin_conditions(
        {**_dual(False, spherical=True), **_triple(True, spherical=True)}
    )
    assert triple_signal["primary_classification"] == "THREE_ORIGIN_REALISTIC_RANGE_SIGNAL"

    dual_dense = M.classify_multi_origin_conditions(
        {
            **_dual(False, fov=True),
            **_triple(False, spherical=False, fov=False),
            M.ALL_FOUR_CONDITION_ID: False,
        }
    )
    assert dual_dense["primary_classification"] == (
        "DUAL_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK"
    )

    no_go = M.classify_multi_origin_conditions(
        {
            **_dual(False),
            **_triple(False),
            M.ALL_FOUR_CONDITION_ID: True,
        }
    )
    assert no_go["primary_classification"] == (
        "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO"
    )
    assert no_go["secondary_classifications"] == [
        "FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED"
    ]

    triple_fov_signal = M.classify_multi_origin_conditions(
        {
            **_dual(False),
            **_triple(False, spherical=False, fov=True),
            M.ALL_FOUR_CONDITION_ID: True,
        }
    )
    assert triple_fov_signal["primary_classification"] == (
        "THREE_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK"
    )
    assert "FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED" not in (
        triple_fov_signal["secondary_classifications"]
    )


def test_conditional_classification_rejects_skipped_or_spurious_conditions():
    with pytest.raises(M.MultiOriginMetricsError, match="must not execute"):
        M.classify_multi_origin_conditions({**_dual(True), **_triple(False)})
    with pytest.raises(M.MultiOriginMetricsError, match="required"):
        M.classify_multi_origin_conditions({**_dual(False), **_triple(False)})
    with pytest.raises(M.MultiOriginMetricsError, match="forbidden"):
        M.classify_multi_origin_conditions(
            {
                **_dual(False),
                **_triple(False, spherical=True),
                M.ALL_FOUR_CONDITION_ID: False,
            }
        )


def test_error_attribution_is_complete_deterministic_and_fail_closed():
    one = M.classify_multi_origin_error({"scan_timing_limitation": True})
    assert one["error_class"] == "SCAN_TIMING_LIMITATION"
    ambiguous = M.classify_multi_origin_error(
        {"scan_timing_limitation": True, "robot_self_occlusion": True}
    )
    assert ambiguous["error_class"] == "UNRESOLVED" and ambiguous["ambiguous"]
    result = M.classify_multi_origin_errors(
        [{"scan_timing_limitation": True}, {}, {"near_blind_region": True}]
    )
    assert result["complete"]
    assert result["counts"]["SCAN_TIMING_LIMITATION"] == 1
    assert result["counts"]["NEAR_BLIND_REGION"] == 1
    assert result["counts"]["UNRESOLVED"] == 1


def test_secondary_classifications_are_supported_and_frozen_ordered():
    conditional = {
        "secondary_classifications": [
            "FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED"
        ]
    }
    labels = M.derive_secondary_classifications(
        conditional_classification=conditional,
        planning_time_observability_limitation=True,
        coverage_error_counts={
            "ROBOT_SELF_OCCLUSION": 2,
            "SCAN_TIMING_LIMITATION": 1,
        },
        assumed_sensor_contract=True,
    )
    assert labels == [
        "FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED",
        "PLANNING_TIME_MULTI_ORIGIN_OBSERVABILITY_LIMITATION",
        "ROBOT_BODY_SELF_OCCLUSION",
        "SCAN_DENSITY_OR_TIMING_LIMITATION",
        "ASSUMED_SENSOR_CONTRACT",
        "REPLANNING_INTERFACE_UNRESOLVED",
    ]


@pytest.mark.parametrize(
    ("p99", "maximum", "expected"),
    [
        (50.0, 80.0, "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL"),
        (50.0001, 80.0, "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY"),
        (80.0, 100.0, "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY"),
        (80.0001, 100.0, "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO"),
        (50.0, 100.0001, "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO"),
    ],
)
def test_complete_set_benchmark_classification_uses_unrounded_boundaries(
    p99: float, maximum: float, expected: str
):
    assert M.classify_complete_set_benchmark(
        {"p99_ms": p99, "max_ms": maximum}
    ) == expected


def test_benchmark_reports_each_state_complete_set_and_replanning_boundary():
    states = [_state("state-b", "family-b"), _state("state-a", "family-a")]
    result = M.benchmark_complete_and_per_state_decisions(
        states,
        warmups=0,
        iterations=2,
        required_families=["family-a", "family-b"],
        enforce_contract_minimums=False,
    )
    assert result["device"] == "CPU"
    assert result["iterations"] == 2
    assert set(result["per_state"]) == {"state-a", "state-b"}
    assert result["classification"] in M.BENCHMARK_CLASSIFICATIONS
    assert result["complete_set"] == result["pooled_complete_state_samples"]
    assert result["classification"] == result["pooled_complete_state_samples"]["classification"]
    assert not result["all_states_aggregate_wall_time_classified"]
    assert not result["contract_minimums_met"]
    assert result["replanning_interface_classification"] == (
        "REPLANNING_INTERFACE_UNRESOLVED"
    )
    assert all("p99_ms" in row and "classification" in row for row in result["per_state"].values())


def test_canonical_reporting_is_byte_identical_and_rejects_nonfinite_values():
    first = M.deterministic_report(z=np.asarray([2, 1]), a={"b": np.int64(3)})
    second = M.deterministic_report(a={"b": 3}, z=[2, 1])
    assert M.canonical_json_bytes(first) == M.canonical_json_bytes(second)
    assert first["content_sha256"] == second["content_sha256"]
    parsed = json.loads(M.canonical_json_bytes(first))
    assert parsed["schema"].endswith("_v1")
    with pytest.raises(M.MultiOriginMetricsError, match="NaN"):
        M.canonical_json_bytes({"bad": np.nan})


def test_fixture_receipt_is_byte_identical_and_complete():
    first = M.run_fixtures()
    second = M.run_fixtures()
    assert first["pass"] and all(first["fixtures"].values())
    assert M.canonical_json_bytes(first) == M.canonical_json_bytes(second)
