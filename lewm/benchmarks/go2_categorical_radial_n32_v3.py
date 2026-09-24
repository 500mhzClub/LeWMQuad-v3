"""Pure identity and decision contract for the N32 token-width diagnostic."""
from __future__ import annotations

from typing import Any, Mapping

from lewm.benchmarks.go2_categorical_radial_n32 import HOLDOUT_PANELS


EXECUTION_BINDING_SHA256 = (
    "a9898d349d82f65ce35443192b555aac4386136032c8fe70c115eda5a788a5ad"
)
RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_v3_result_v1"
SMOKE_RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_v3_smoke_result_v1"
STAGE_SCHEMA = "lewm_go2_categorical_radial_n32_v3_stage_v1"
TWO_SEED_RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_v3_two_seed_result_v1"
PER_SEED_DECISION_SCHEMA = "lewm_go2_categorical_radial_n32_v3_seed_decision_v1"
STAGE_NAME = "token_width_32_exposure_matched_v3_cosine"


def per_seed_decision(
    stage: Mapping[str, Any],
    holdouts: Mapping[str, Mapping[str, Any]] | None,
    *,
    authoritative: bool = True,
) -> dict[str, Any]:
    """Adjudicate one V3 seed without granting a two-seed license."""

    if not isinstance(authoritative, bool):
        raise TypeError("authoritative must be a bool")
    terminal = stage.get("terminal_fit_gate")
    if not isinstance(terminal, Mapping) or not isinstance(
        terminal.get("passes"), bool
    ):
        raise ValueError("N32 V3 stage lacks a strict terminal fit decision")
    fit_passes = bool(terminal["passes"])
    if not authoritative:
        if holdouts not in (None, {}):
            raise ValueError("N32 V3 smoke must never contain holdouts")
        holdout_passes = None
        favorable = False
        classification = "non_authoritative_smoke"
    elif not fit_passes:
        if holdouts not in (None, {}):
            raise ValueError("N32 V3 holdouts are forbidden after a fit failure")
        holdout_passes = None
        favorable = False
        classification = "fit_gate_failed"
    else:
        if not isinstance(holdouts, Mapping) or set(holdouts) != set(
            HOLDOUT_PANELS
        ):
            raise ValueError("both N32 V3 holdouts are mandatory after a fit pass")
        holdout_passes = {}
        for panel in HOLDOUT_PANELS:
            passes = holdouts[panel].get("passes")
            if not isinstance(passes, bool):
                raise ValueError(f"N32 V3 holdout lacks a strict decision: {panel}")
            holdout_passes[panel] = passes
        favorable = all(holdout_passes.values())
        classification = (
            "favorable" if favorable else "fit_pass_holdout_gate_failed"
        )
    return {
        "schema": PER_SEED_DECISION_SCHEMA,
        "token_width_32_fit_passes": fit_passes,
        "qualifying_optimizer_stage": (
            STAGE_NAME if authoritative and fit_passes else None
        ),
        "holdout_passes": holdout_passes,
        "classification": classification,
        "favorable": favorable,
        "aggregation_eligible": authoritative,
        "shared_jepa_full_train_candidate_licensed": False,
        "runtime_ready": False,
        "g2_licensed": False,
        "g3_licensed": False,
        "promotion_licensed": False,
    }


__all__ = [
    "EXECUTION_BINDING_SHA256",
    "PER_SEED_DECISION_SCHEMA",
    "RESULT_SCHEMA",
    "SMOKE_RESULT_SCHEMA",
    "STAGE_NAME",
    "STAGE_SCHEMA",
    "TWO_SEED_RESULT_SCHEMA",
    "per_seed_decision",
]
