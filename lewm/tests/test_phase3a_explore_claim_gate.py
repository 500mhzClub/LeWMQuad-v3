from __future__ import annotations

from scripts.check_jepa_phase3a_explore_claim_gate import evaluate_gate


def _phase(
    *,
    source_states: int,
    primitive_match_rate: float,
    primitive_regret: float,
    sequence_regret: float,
    marker_rate: float,
    claim_rate: float,
    top5_claim_rate: float,
) -> dict:
    return {
        "source_states": source_states,
        "primitive_match_rate": primitive_match_rate,
        "mean_target_utility_regret": primitive_regret,
        "mean_selected_sequence_target_utility_regret": sequence_regret,
        "selected_future_goal_marker_seen_rate": marker_rate,
        "selected_goal_claimed_rate": claim_rate,
        "topk_claimed_rate": {"1": claim_rate, "3": top5_claim_rate, "5": top5_claim_rate},
    }


def _delta(left: dict, right: dict) -> dict:
    return {
        "source_states": left["source_states"],
        "primitive_match_rate_delta": (
            left["primitive_match_rate"] - right["primitive_match_rate"]
        ),
        "mean_target_utility_regret_delta": (
            left["mean_target_utility_regret"] - right["mean_target_utility_regret"]
        ),
        "selected_future_goal_marker_seen_rate_delta": (
            left["selected_future_goal_marker_seen_rate"]
            - right["selected_future_goal_marker_seen_rate"]
        ),
        "selected_goal_claimed_rate_delta": (
            left["selected_goal_claimed_rate"] - right["selected_goal_claimed_rate"]
        ),
        "sequence_match_rate_delta": 0.0,
        "mean_selected_sequence_regret_delta": (
            left["mean_selected_sequence_target_utility_regret"]
            - right["mean_selected_sequence_target_utility_regret"]
        ),
    }


def test_explore_claim_gate_reports_non_privileged_egocentric_baseline() -> None:
    memory_phases = {
        "explore_unseen": _phase(
            source_states=2,
            primitive_match_rate=1.0,
            primitive_regret=0.0,
            sequence_regret=0.0,
            marker_rate=0.0,
            claim_rate=0.0,
            top5_claim_rate=0.0,
        ),
        "discover_visible_marker": _phase(
            source_states=2,
            primitive_match_rate=1.0,
            primitive_regret=0.0,
            sequence_regret=0.0,
            marker_rate=1.0,
            claim_rate=0.0,
            top5_claim_rate=0.0,
        ),
        "claim_after_marker_seen": _phase(
            source_states=4,
            primitive_match_rate=0.5,
            primitive_regret=0.5,
            sequence_regret=5.0,
            marker_rate=0.0,
            claim_rate=0.25,
            top5_claim_rate=0.75,
        ),
    }
    no_memory_phases = {
        "explore_unseen": _phase(
            source_states=2,
            primitive_match_rate=1.0,
            primitive_regret=0.0,
            sequence_regret=0.0,
            marker_rate=0.0,
            claim_rate=0.0,
            top5_claim_rate=0.0,
        ),
        "discover_visible_marker": _phase(
            source_states=2,
            primitive_match_rate=0.5,
            primitive_regret=0.5,
            sequence_regret=0.5,
            marker_rate=0.5,
            claim_rate=0.0,
            top5_claim_rate=0.0,
        ),
        "claim_after_marker_seen": _phase(
            source_states=4,
            primitive_match_rate=0.25,
            primitive_regret=1.0,
            sequence_regret=6.0,
            marker_rate=0.0,
            claim_rate=0.0,
            top5_claim_rate=0.5,
        ),
    }
    egocentric_phases = {
        **memory_phases,
        "claim_after_marker_seen": _phase(
            source_states=4,
            primitive_match_rate=1.0,
            primitive_regret=0.0,
            sequence_regret=0.5,
            marker_rate=0.0,
            claim_rate=1.0,
            top5_claim_rate=1.0,
        ),
    }
    report = {
        "summaries": {
            "memory": {"phases": memory_phases},
            "no_memory": {"phases": no_memory_phases},
            "egocentric_marker_memory": {"phases": egocentric_phases},
        },
        "score_summaries": {
            "memory": {
                "online_marker_memory_score": {"phases": egocentric_phases},
            },
        },
        "comparisons": {
            "memory_minus_no_memory": {
                phase: _delta(memory_phases[phase], no_memory_phases[phase])
                for phase in memory_phases
            },
        },
        "memory_training_aggregate": {
            "primitive_match_rate": 0.75,
            "mean_target_utility_regret": 0.5,
            "mean_selected_sequence_target_utility_regret": 5.0,
        },
        "no_memory_training_aggregate": {
            "primitive_match_rate": 0.50,
            "mean_target_utility_regret": 1.0,
            "mean_selected_sequence_target_utility_regret": 6.0,
        },
        "validation_audit": {
            "rows": 8,
            "current_goal_beacon_counts": {"False": 8},
            "history_goal_beacon_counts": {"False": 8},
        },
    }

    gate = evaluate_gate(
        report,
        min_explore_sources=2,
        min_discover_sources=2,
        min_claim_sources=4,
    )

    assert not gate["passed"]
    assert "claim_after_marker_seen_selection_rate_below_threshold" in gate[
        "failure_reasons"
    ]
    assert gate["non_privileged_egocentric_marker_memory"]["passed"]
    assert gate["model_online_marker_memory_score"]["passed"]
    assert (
        gate["observed"]["egocentric_marker_memory_claim_after_marker_seen"][
            "selected_goal_claimed_rate"
        ]
        == 1.0
    )


def test_explore_claim_gate_allows_claim_only_spatial_memory_report() -> None:
    empty_phase = _phase(
        source_states=0,
        primitive_match_rate=0.0,
        primitive_regret=0.0,
        sequence_regret=0.0,
        marker_rate=0.0,
        claim_rate=0.0,
        top5_claim_rate=0.0,
    )
    memory_phases = {
        "explore_unseen": empty_phase,
        "discover_visible_marker": empty_phase,
        "claim_after_marker_seen": _phase(
            source_states=12,
            primitive_match_rate=0.33,
            primitive_regret=0.0,
            sequence_regret=0.57,
            marker_rate=1.0,
            claim_rate=0.92,
            top5_claim_rate=0.92,
        ),
    }
    no_memory_phases = {
        "explore_unseen": empty_phase,
        "discover_visible_marker": empty_phase,
        "claim_after_marker_seen": _phase(
            source_states=12,
            primitive_match_rate=0.58,
            primitive_regret=0.0,
            sequence_regret=5.83,
            marker_rate=1.0,
            claim_rate=0.25,
            top5_claim_rate=0.25,
        ),
    }
    report = {
        "summaries": {
            "memory": {"phases": memory_phases},
            "no_memory": {"phases": no_memory_phases},
            "egocentric_marker_memory": {"phases": memory_phases},
        },
        "score_summaries": {
            "memory": {
                "spatial_marker_memory_score": {"phases": memory_phases},
                "online_marker_memory_score": {"phases": memory_phases},
            },
        },
        "comparisons": {
            "memory_minus_no_memory": {
                phase: _delta(memory_phases[phase], no_memory_phases[phase])
                for phase in memory_phases
            },
        },
        "memory_training_aggregate": {
            "primitive_match_rate": 0.33,
            "mean_target_utility_regret": 0.0,
            "mean_selected_sequence_target_utility_regret": 0.57,
        },
        "no_memory_training_aggregate": {
            "primitive_match_rate": 0.58,
            "mean_target_utility_regret": 0.0,
            "mean_selected_sequence_target_utility_regret": 5.83,
        },
        "validation_audit": {
            "rows": 3072,
            "current_goal_beacon_counts": {"False": 3072},
            "history_goal_beacon_counts": {"False": 3072},
        },
    }

    gate = evaluate_gate(
        report,
        min_explore_sources=0,
        min_discover_sources=0,
        min_claim_sources=4,
    )

    assert gate["passed"]
    assert gate["model_spatial_marker_memory_score"]["passed"]
    assert not gate["thresholds"]["memory_must_beat_no_memory_aggregate"]
    assert (
        gate["observed"]["spatial_marker_memory_score_claim_after_marker_seen"][
            "selected_goal_claimed_rate"
        ]
        == 0.92
    )
