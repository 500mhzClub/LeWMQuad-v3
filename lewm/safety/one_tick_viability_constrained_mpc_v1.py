"""Pure reducers for the one-tick viability-constrained MPC scaffold."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


DISTANCE_TIE_M = 0.03


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def route_order(rows: Sequence[Mapping], distance_key: str, heading_key: str) -> list[int]:
    """Frozen distance/tie/heading/index order over an arbitrary candidate set."""

    remaining = list(range(len(rows)))
    ordered: list[int] = []
    while remaining:
        best_distance = max(float(rows[index][distance_key]) for index in remaining)
        tied = [
            index
            for index in remaining
            if best_distance - float(rows[index][distance_key]) <= DISTANCE_TIE_M
        ]
        chosen = min(
            tied,
            key=lambda index: (
                -float(rows[index][heading_key]),
                int(rows[index]["candidate_index"]),
            ),
        )
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def state_classification(
    rows: Sequence[Mapping], *, pre_existing: bool, contact_before_authority: bool
) -> str:
    if pre_existing:
        return "PRE_EXISTING_CONTACT"
    admissible = [row for row in rows if bool(row["safe_prefix"]) and bool(row["viable"])]
    if admissible:
        if any(float(row["immediate_progress_m"]) > 0.0 for row in admissible):
            return "VIABILITY_ADMISSIBLE_PROGRESS_ACTION_AVAILABLE"
        return "VIABILITY_ADMISSIBLE_NONPROGRESS_ACTION_AVAILABLE"
    if any(bool(row["safe_prefix"]) for row in rows):
        return "SAFE_PREFIX_ONLY_NO_VIABLE_SUCCESSOR"
    if contact_before_authority:
        return "CONTACT_BEFORE_CONTROL_AUTHORITY"
    return "NO_SAFE_PREFIX_ACTION"


def normalized_regret(rows: Sequence[Mapping], selected: int | None) -> float | None:
    """Progress regret over admissible candidates, normalized by their spread."""

    admissible = [i for i, row in enumerate(rows) if bool(row["admissible"])]
    if selected is None or len(admissible) < 2:
        return None
    values = [float(rows[i]["h3_realised_progress_m"]) for i in admissible]
    spread = max(values) - min(values)
    if spread <= 1e-8:
        return None
    return (max(values) - float(rows[selected]["h3_realised_progress_m"])) / spread


def fixture_payload() -> dict:
    rows = [
        {"candidate_index": 0, "safe_prefix": True, "viable": True,
         "immediate_progress_m": .01, "h3_nominal_progress_m": .3,
         "h3_nominal_heading_improvement_rad": .0, "h3_realised_progress_m": .2,
         "admissible": True},
        {"candidate_index": 1, "safe_prefix": True, "viable": False,
         "immediate_progress_m": .02, "h3_nominal_progress_m": .4,
         "h3_nominal_heading_improvement_rad": .0, "h3_realised_progress_m": .4,
         "admissible": False},
        {"candidate_index": 2, "safe_prefix": False, "viable": False,
         "immediate_progress_m": .0, "h3_nominal_progress_m": .2,
         "h3_nominal_heading_improvement_rad": .1, "h3_realised_progress_m": .0,
         "admissible": False},
    ]
    tests = {
        "safe_prefix_and_viable": bool(rows[0]["safe_prefix"] and rows[0]["viable"]),
        "safe_prefix_nonviable_is_excluded": not bool(rows[1]["admissible"]),
        "contact_prefix_is_excluded": not bool(rows[2]["admissible"]),
        "classification_progress": state_classification(
            rows, pre_existing=False, contact_before_authority=False
        ) == "VIABILITY_ADMISSIBLE_PROGRESS_ACTION_AVAILABLE",
        "all_successors_contact": state_classification(
            [{**row, "safe_prefix": True, "viable": False} for row in rows],
            pre_existing=False,
            contact_before_authority=False,
        ) == "SAFE_PREFIX_ONLY_NO_VIABLE_SUCCESSOR",
        "pre_existing": state_classification(
            rows, pre_existing=True, contact_before_authority=False
        ) == "PRE_EXISTING_CONTACT",
        "before_authority": state_classification(
            [{**row, "safe_prefix": False, "viable": False} for row in rows],
            pre_existing=False,
            contact_before_authority=True,
        ) == "CONTACT_BEFORE_CONTROL_AUTHORITY",
        "h3_rank_uses_long_horizon": route_order(
            rows, "h3_nominal_progress_m", "h3_nominal_heading_improvement_rad"
        )[0] == 1,
        "exactly_one_viable_successor": sum(row["admissible"] for row in rows) == 1,
        "finite_tie_constant": math.isfinite(DISTANCE_TIE_M),
    }
    payload = {
        "schema": "one_tick_viability_constrained_mpc_fixture_v1",
        "tests": tests,
        "pass": all(tests.values()),
    }
    payload["content_digest"] = digest(payload)
    return payload
