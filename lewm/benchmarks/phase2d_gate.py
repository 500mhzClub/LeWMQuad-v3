"""Machine-checkable Phase 2D launch and checkpoint gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .phase2d_training import (
    REGISTERED_ACTION_ADVANTAGE_THRESHOLD,
    REGISTERED_PERSISTENCE_RATIO_THRESHOLD,
)


def _latest_validation_diagnostic(report: Mapping) -> Mapping | None:
    final_gate = report.get("final_validation_gate")
    if isinstance(final_gate, Mapping):
        return {
            "checkpoint_rule_record": final_gate,
            "checkpoint_selection_permitted": report.get(
                "checkpoint_selection_permitted"
            ),
            "source": "final_validation_gate",
        }
    history = report.get("history", [])
    if not isinstance(history, list):
        return None
    for record in reversed(history):
        if not isinstance(record, Mapping):
            continue
        diagnostic = record.get("validation_interface_diagnostic")
        if isinstance(diagnostic, Mapping):
            return {**diagnostic, "source": "history.validation_interface_diagnostic"}
    return None


def _float_field(rule: Mapping, key: str) -> float | None:
    value = rule.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def phase2d_smoke_gate_report(report: Mapping) -> dict:
    """Return a strict pre-full-training gate report for one smoke run."""

    diagnostic = _latest_validation_diagnostic(report)
    if diagnostic is None:
        return {
            "schema": "jepa_phase2d_smoke_gate_report_v0",
            "passed": False,
            "failure_reasons": ["missing_validation_diagnostic"],
        }
    rule = diagnostic.get("checkpoint_rule_record")
    if not isinstance(rule, Mapping):
        return {
            "schema": "jepa_phase2d_smoke_gate_report_v0",
            "passed": False,
            "source": diagnostic.get("source"),
            "failure_reasons": ["missing_checkpoint_rule_record"],
        }

    reasons = []
    stability_pass = bool(rule.get("stability_pass", False))
    if not stability_pass:
        reasons.append("stability_failed")

    hard_negative_advantage = _float_field(rule, "hard_negative_action_advantage")
    if hard_negative_advantage is None:
        reasons.append("missing_hard_negative_action_advantage")
        hard_negative_pass = False
    else:
        hard_negative_pass = (
            hard_negative_advantage >= REGISTERED_ACTION_ADVANTAGE_THRESHOLD
        )
        if not hard_negative_pass:
            reasons.append("hard_negative_action_advantage_below_threshold")

    zero_action_advantage = _float_field(rule, "zero_action_advantage")
    if zero_action_advantage is None:
        reasons.append("missing_zero_action_advantage")
        zero_action_pass = False
    else:
        zero_action_pass = (
            zero_action_advantage >= REGISTERED_ACTION_ADVANTAGE_THRESHOLD
        )
        if not zero_action_pass:
            reasons.append("zero_action_advantage_below_threshold")

    persistence_ratio = _float_field(rule, "one_step_rollout_persistence_ratio")
    if persistence_ratio is None:
        reasons.append("missing_one_step_rollout_persistence_ratio")
        persistence_pass = False
    else:
        persistence_pass = persistence_ratio < REGISTERED_PERSISTENCE_RATIO_THRESHOLD
        if not persistence_pass:
            reasons.append("persistence_ratio_not_below_threshold")

    gate_pass = (
        stability_pass
        and hard_negative_pass
        and zero_action_pass
        and persistence_pass
    )
    reported_selection = bool(diagnostic.get("checkpoint_selection_permitted", False))
    if gate_pass and not reported_selection:
        reasons.append("checkpoint_selection_not_permitted_by_report")

    passed = gate_pass and reported_selection
    return {
        "schema": "jepa_phase2d_smoke_gate_report_v0",
        "passed": passed,
        "source": diagnostic.get("source"),
        "failure_reasons": reasons,
        "thresholds": {
            "action_advantage": REGISTERED_ACTION_ADVANTAGE_THRESHOLD,
            "persistence_ratio": REGISTERED_PERSISTENCE_RATIO_THRESHOLD,
        },
        "observed": {
            "stability_pass": stability_pass,
            "hard_negative_action_advantage": hard_negative_advantage,
            "zero_action_advantage": zero_action_advantage,
            "one_step_rollout_persistence_ratio": persistence_ratio,
            "checkpoint_selection_permitted": reported_selection,
        },
    }


def phase2d_smoke_gate_report_from_path(path: Path) -> dict:
    report = json.loads(path.read_text())
    gate = phase2d_smoke_gate_report(report)
    gate["report_path"] = str(path.resolve())
    return gate
