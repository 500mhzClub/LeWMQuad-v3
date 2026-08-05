"""Pure identity and decision contract for the exposure-matched N32 V2 run."""
from __future__ import annotations

from typing import Any, Mapping

from lewm.benchmarks.go2_categorical_radial_n32 import HOLDOUT_PANELS


EXECUTION_BINDING_SHA256 = (
    "4164ec011910cb2d1d2fbea5beaad81eb13ea6b506e063ebf13a66a41e14fb6f"
)
RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_v2_result_v1"
SMOKE_RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_v2_smoke_result_v1"
STAGE_SCHEMA = "lewm_go2_categorical_radial_n32_v2_stage_v1"
TWO_SEED_RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_v2_two_seed_result_v1"
PER_SEED_DECISION_SCHEMA = (
    "lewm_go2_categorical_radial_n32_v2_seed_decision_v1"
)
STAGE_NAME = "exposure_matched_v3_cosine"


def per_seed_decision(
    stage: Mapping[str, Any],
    holdouts: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Adjudicate the sole V2 stage without granting a two-seed license."""

    terminal = stage.get("terminal_fit_gate")
    if not isinstance(terminal, Mapping) or not isinstance(
        terminal.get("passes"), bool
    ):
        raise ValueError("V2 stage lacks a strict terminal fit decision")
    fit_passes = bool(terminal["passes"])
    if not fit_passes:
        if holdouts not in (None, {}):
            raise ValueError("V2 holdouts are forbidden after a fit failure")
        holdout_passes = None
        favorable = False
        classification = "fit_gate_failed"
    else:
        if not isinstance(holdouts, Mapping) or set(holdouts) != set(
            HOLDOUT_PANELS
        ):
            raise ValueError("both V2 holdouts are mandatory after a fit pass")
        holdout_passes = {}
        for panel in HOLDOUT_PANELS:
            passes = holdouts[panel].get("passes")
            if not isinstance(passes, bool):
                raise ValueError(f"V2 holdout lacks a strict decision: {panel}")
            holdout_passes[panel] = passes
        favorable = all(holdout_passes.values())
        classification = (
            "favorable" if favorable else "fit_pass_holdout_gate_failed"
        )
    return {
        "schema": PER_SEED_DECISION_SCHEMA,
        "exposure_matched_v3_cosine_fit_passes": fit_passes,
        "qualifying_optimizer_stage": STAGE_NAME if fit_passes else None,
        "holdout_passes": holdout_passes,
        "classification": classification,
        "favorable": favorable,
        "aggregation_eligible": True,
        "categorical_radial_full_train_candidate_licensed": False,
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
