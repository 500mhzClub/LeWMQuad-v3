from __future__ import annotations

import copy

import numpy as np
import pytest

from lewm.safety import body_centric_range_coverage_analysis_v1 as A


def _link_rows(contact: bool, clearance: float, *, supported: bool = True) -> dict:
    return {
        "base": {
            "clearance_m": [clearance, clearance + 0.05],
            "oracle_contact": contact,
            "body_region": "TRUNK",
            "support": [supported, supported],
            "nominal_fov": [True, True],
            "direct_visibility": [supported, supported],
            "point_support_count": [1 if supported else 0, 1 if supported else 0],
        },
        "FL_calf": {
            "clearance_m": [1.0, 0.9],
            "oracle_contact": False,
            "body_region": "CALF",
            "support": [True, False],
            "nominal_fov": [True, False],
            "direct_visibility": [True, False],
            "point_support_count": [1, 0],
        },
    }


def _state_records(
    state_id: str,
    role: str,
    *,
    contact: bool,
    clearance: float,
    family: str = "family-a",
    supported: bool = True,
) -> list[dict]:
    common = {
        "state_id": state_id,
        "role": role,
        "family": family,
        "controller": "route",
        "applied_action": [0.2, 0.0, 0.0],
        "action_index": 0,
        "frozen_contact": contact,
        "clearance_m": clearance,
        "unsupported": not supported,
        "per_link": _link_rows(contact, clearance, supported=supported),
    }
    current = {
        **common,
        "identity": f"{state_id}:current:0",
        "level": "current",
        "current_action_index": -1,
        "h3_progress_m": 0.25,
        "h3_heading_improvement_rad": 0.0,
        "decision_progress_m": 0.25,
    }
    successor = {
        **common,
        "identity": f"{state_id}:successor:0:0",
        "level": "successor",
        "current_action_index": 0,
    }
    return [current, successor]


def _complete_records() -> list[dict]:
    return [
        *_state_records("cal-safe", "calibration", contact=False, clearance=0.4),
        *_state_records("cal-contact", "calibration", contact=True, clearance=0.1),
        *_state_records("held-safe", "heldout", contact=False, clearance=0.5),
        *_state_records("held-contact", "heldout", contact=True, clearance=0.05),
    ]


def test_build_and_inject_predictions_preserve_frozen_two_ply_structure() -> None:
    records = _complete_records()[:4]
    states = A.build_two_ply_states(records)
    assert [state["state_id"] for state in states] == ["cal-safe", "cal-contact"]
    assert states[0]["current_actions"][0]["next_actions"][0]["oracle_contact"] is False

    predicted = A.inject_contact_predictions(states, records, 0.1)
    assert predicted[0]["current_actions"][0]["predicted_contact"] is False
    assert predicted[0]["current_actions"][0]["next_actions"][0]["predicted_contact"] is False
    assert predicted[1]["current_actions"][0]["predicted_contact"] is True
    assert "predicted_contact" not in states[0]["current_actions"][0]


def test_calibration_uses_exact_frontier_and_frozen_lexicographic_decision() -> None:
    records = _complete_records()
    result = A.calibrate_threshold(records)
    assert result["selected_threshold_m"] == pytest.approx(0.1)
    assert result["selected"]["contact"]["recall"] == 1.0
    assert result["selected"]["contact"]["negative_retention"] == 1.0
    decision = result["selected"]["decision"]
    assert decision["states_retaining_admitted_action"] == 1
    assert decision["correct_abstentions"] == 1


def test_condition_mode_computes_contact_safe_counts_viability_and_groups() -> None:
    result = A.summarize_condition_mode(_complete_records(), None, 0.1)
    assert result["current_contact"]["auc"] == 1.0
    assert result["successor_contact"]["average_precision"] == 1.0
    assert result["combined_contact"]["recall"] == 1.0
    assert result["combined_contact"]["negative_retention"] == 1.0
    assert result["safe_action_count"]["exact_count_accuracy"] == 1.0
    assert result["safe_action_count"]["zero_vs_nonzero_accuracy"] == 1.0
    assert result["viability"]["oracle_viable_states"] == 1
    assert result["viability"]["states_retaining_admitted_action"] == 1
    assert result["viability"]["oracle_nonviable_states"] == 1
    assert result["viability"]["correct_abstentions"] == 1
    assert result["viability"]["selected_immediate_contacts"] == 0
    assert result["viability"]["selected_oracle_nonviable_successors"] == 0
    assert result["per_family"]["family-a"]["combined_contact"]["auc"] == 1.0
    assert result["per_family"]["family-a"]["states_retaining_admitted_action"] == 1
    assert result["per_family"]["family-a"]["coverage"]["protected_links"] == 2
    assert set(result["per_family"]["family-a"]["per_link"]) == {"base", "FL_calf"}


def test_per_link_contact_and_coverage_reduce_physics_step_vectors() -> None:
    result = A.per_link_and_coverage_summaries(
        [row for row in _complete_records() if row["role"] == "heldout"],
        0.1,
    )
    assert set(result["per_link"]) == {"base", "FL_calf"}
    assert result["per_link"]["base"]["combined_contact"]["auc"] == 1.0
    assert result["per_link"]["base"]["coverage"]["support_fraction"] == 1.0
    assert result["per_link"]["FL_calf"]["coverage"]["support_fraction"] == 0.5
    assert result["per_link"]["FL_calf"]["coverage"][
        "mean_unsupported_swept_volume_fraction"
    ] == 0.5
    assert result["per_region"]["TRUNK"]["combined_contact"]["recall"] == 1.0
    assert result["coverage"]["available"] is True
    assert result["coverage"]["protected_links"] == 2
    assert result["coverage"]["transition_link_rows"] == 8


def test_end_to_end_analysis_and_condition_mode_mapping_are_deterministic() -> None:
    records = _complete_records()
    first = A.analyze_condition_mode(records)
    second = A.analyze_condition_mode(copy.deepcopy(records))
    assert first == second
    assert first["selected_threshold_m"] == pytest.approx(0.1)
    assert first["evaluation"]["safe_action_count"]["false_nonzero_rate"] == 0.0

    nested = A.analyze_condition_modes(
        {
            ("condition-a", "causal"): records,
            ("condition-a", "future"): copy.deepcopy(records),
        }
    )
    assert set(nested) == {"condition-a"}
    assert set(nested["condition-a"]) == {"causal", "future"}
    assert (
        nested["condition-a"]["causal"]["selected_threshold_m"]
        == nested["condition-a"]["future"]["selected_threshold_m"]
    )


def test_unsupported_or_nonfinite_evidence_is_conservative_contact_risk() -> None:
    records = _complete_records()
    for row in records:
        if row["state_id"] == "held-safe":
            row["clearance_m"] = np.inf
            row["unsupported"] = True
    result = A.summarize_condition_mode(records, None, 0.1)
    safe = next(
        row for row in result["per_state_decisions"] if row["state_id"] == "held-safe"
    )
    assert safe["selected"] is None
    assert safe["false_abstention"] is True


def test_duplicate_transition_identity_and_incomplete_state_binding_fail_closed() -> None:
    records = _complete_records()[:4]
    duplicated = [*records, copy.deepcopy(records[0])]
    with pytest.raises(A.AnalysisError, match="duplicate transition identity"):
        A.validate_transition_records(duplicated)

    states = A.build_two_ply_states(records)
    with pytest.raises(A.AnalysisError, match="not bound"):
        A.inject_contact_predictions(states[:1], records, 0.1)


def test_vector_per_link_form_is_supported_without_mapping_rewrite() -> None:
    rows = _state_records("vector", "heldout", contact=False, clearance=0.5)
    for row in rows:
        row.pop("per_link")
        row.update(
            {
                "link_names": ["base", "calf"],
                "per_link_clearance_m": [0.5, 0.8],
                "per_link_unsupported": [False, True],
                "per_link_contact": [False, False],
                "per_link_support_fraction": [1.0, 0.25],
                "per_link_nominal_fov_fraction": [1.0, 0.5],
                "per_link_direct_visibility_fraction": [1.0, 0.25],
                "body_region_by_link": {"base": "TRUNK", "calf": "CALF"},
            }
        )
    summary = A.per_link_and_coverage_summaries(rows, 0.1)
    assert summary["per_link"]["calf"]["coverage"]["support_fraction"] == 0.25
    assert summary["per_link"]["calf"]["coverage"][
        "mean_unsupported_swept_volume_fraction"
    ] == 0.75


def test_materialised_arrays_join_to_frozen_identity_rows_without_io() -> None:
    identities = _state_records("array", "heldout", contact=False, clearance=9.0)
    for row in identities:
        row.pop("clearance_m")
        row.pop("unsupported")
        row.pop("per_link")
    records = A.transition_records_from_arrays(
        identities,
        clearance_m=np.asarray([0.4, 0.3]),
        unsupported=np.asarray([False, True]),
        link_names=["base", "calf"],
        per_link_clearance_m=np.asarray(
            [
                [[0.4, 0.8], [0.3, 0.7]],
                [[0.3, 0.7], [0.2, 0.6]],
            ]
        ),
        per_link_support=np.asarray(
            [
                [[True, True], [True, False]],
                [[True, False], [False, False]],
            ]
        ),
        per_link_contact=np.zeros((2, 2, 2), dtype=bool),
        per_link_nominal_fov=np.ones((2, 2, 2), dtype=bool),
        per_link_direct_visibility=np.ones((2, 2, 2), dtype=bool),
        per_link_point_support_count=np.ones((2, 2, 2), dtype=np.int16),
        body_region_by_link={"base": "TRUNK", "calf": "CALF"},
    )
    assert records[0]["clearance_m"] == 0.4
    assert records[1]["unsupported"] is True
    np.testing.assert_array_equal(records[0]["per_link"]["base"]["clearance_m"], [0.4, 0.3])
    assert records[1]["per_link"]["calf"]["unsupported_swept_volume_fraction"] == 1.0
    assert records[0]["per_link"]["base"]["body_region"] == "TRUNK"


def test_array_join_requires_explicit_decision_shapes() -> None:
    identities = _state_records("bad-array", "heldout", contact=False, clearance=1.0)
    with pytest.raises(A.AnalysisError, match="clearance_m must have shape"):
        A.transition_records_from_arrays(
            identities,
            clearance_m=np.zeros((2, 1)),
            unsupported=np.zeros(2, dtype=bool),
        )
