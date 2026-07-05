from __future__ import annotations

from lewm.benchmarks.phase2d_gate import phase2d_smoke_gate_report


def test_phase2d_smoke_gate_accepts_explicit_passing_final_gate() -> None:
    gate = phase2d_smoke_gate_report(
        {
            "final_validation_gate": {
                "stability_pass": True,
                "hard_negative_action_advantage": 0.12,
                "zero_action_advantage": 0.11,
                "one_step_rollout_persistence_ratio": 0.8,
            },
            "checkpoint_selection_permitted": True,
        }
    )

    assert gate["passed"]
    assert gate["failure_reasons"] == []


def test_phase2d_smoke_gate_rejects_collapsed_legacy_report() -> None:
    gate = phase2d_smoke_gate_report(
        {
            "history": [
                {
                    "validation_interface_diagnostic": {
                        "checkpoint_rule_record": {
                            "stability_pass": False,
                            "hard_negative_action_advantage": -80.0,
                            "one_step_rollout_persistence_ratio": 247.0,
                        },
                        "checkpoint_selection_permitted": True,
                    }
                }
            ]
        }
    )

    assert not gate["passed"]
    assert "stability_failed" in gate["failure_reasons"]
    assert "hard_negative_action_advantage_below_threshold" in gate[
        "failure_reasons"
    ]
    assert "missing_zero_action_advantage" in gate["failure_reasons"]
    assert "persistence_ratio_not_below_threshold" in gate["failure_reasons"]
