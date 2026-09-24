"""Pure reducers for the multi-cycle oracle viability-envelope experiment."""
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


def route_order(rows: Sequence[Mapping]) -> list[int]:
    """Frozen H3 distance/tie/heading/index ordering."""

    remaining = list(range(len(rows)))
    ordered: list[int] = []
    while remaining:
        best_distance = max(float(rows[index]["h3_progress_m"]) for index in remaining)
        tied = [
            index
            for index in remaining
            if best_distance - float(rows[index]["h3_progress_m"]) <= DISTANCE_TIE_M
        ]
        chosen = min(
            tied,
            key=lambda index: (
                -float(rows[index]["h3_heading_improvement_rad"]),
                int(rows[index]["candidate_index"]),
            ),
        )
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def stable_predecessor_depth(rows: Sequence[Mapping], required: int = 3) -> int | None:
    """Return the earliest-in-time depth of a stable forward viability run.

    Depth increases backward in time.  A rollout beginning at depth ``d``
    therefore advances through ``d, d - 1, ...`` toward the historical
    failure.  Return the deepest boundary of the demonstrated run so that the
    rollout includes the complete stable envelope.
    """

    viable = {int(row["depth"]): bool(row["viability_admissible_count"]) for row in rows}
    for earliest in sorted(viable):
        if earliest >= required and all(
            viable.get(earliest - offset, False) for offset in range(required)
        ):
            return earliest
    return None


def intervention_class(
    rows: Sequence[Mapping], *, contact_already_unavoidable: bool = False
) -> str:
    if contact_already_unavoidable:
        return "CONTACT_ALREADY_UNAVOIDABLE"
    stable = stable_predecessor_depth(rows)
    if stable is None:
        if len(rows) >= 10 and not any(int(row["viability_admissible_count"]) for row in rows):
            return "PERSISTENT_CANDIDATE_BANK_VIABILITY_FAILURE"
        return "UNRESOLVED"
    # The stable envelope begins at ``stable`` (deepest/earliest boundary),
    # while the closest boundary in that same demonstrated run is the minimum
    # intervention lead time.
    lead = stable - 3 + 1
    if lead == 1:
        return "ONE_CYCLE_EARLIER_INTERVENTION_SUFFICIENT"
    if lead <= 3:
        return "TWO_TO_THREE_CYCLE_INTERVENTION_REQUIRED"
    return "FOUR_TO_TEN_CYCLE_INTERVENTION_REQUIRED"


def normalized_regret(rows: Sequence[Mapping], selected_index: int | None) -> float | None:
    admissible = [row for row in rows if bool(row["admissible"])]
    if selected_index is None or len(admissible) < 2:
        return None
    values = [float(row["h3_progress_m"]) for row in admissible]
    spread = max(values) - min(values)
    if spread <= 1e-8:
        return None
    selected = next(row for row in rows if int(row["candidate_index"]) == selected_index)
    return (max(values) - float(selected["h3_progress_m"])) / spread


def fixture_payload() -> dict:
    rows = [
        {"candidate_index": 0, "h3_progress_m": 0.20, "h3_heading_improvement_rad": 0.0,
         "admissible": True},
        {"candidate_index": 1, "h3_progress_m": 0.21, "h3_heading_improvement_rad": 0.2,
         "admissible": True},
        {"candidate_index": 2, "h3_progress_m": 0.10, "h3_heading_improvement_rad": 0.0,
         "admissible": False},
    ]
    stable = [
        {"depth": 1, "viability_admissible_count": 2},
        {"depth": 2, "viability_admissible_count": 1},
        {"depth": 3, "viability_admissible_count": 4},
    ]
    persistent = [{"depth": depth, "viability_admissible_count": 0} for depth in range(1, 11)]
    tests = {
        "h3_tie_uses_heading": route_order(rows)[0] == 1,
        "stable_three_cycle_envelope": stable_predecessor_depth(stable) == 3,
        "one_cycle_class": intervention_class(stable) == "ONE_CYCLE_EARLIER_INTERVENTION_SUFFICIENT",
        "persistent_bank_failure": intervention_class(persistent)
        == "PERSISTENT_CANDIDATE_BANK_VIABILITY_FAILURE",
        "unavoidable_is_distinct": intervention_class(
            persistent, contact_already_unavoidable=True
        ) == "CONTACT_ALREADY_UNAVOIDABLE",
        "finite_tie": math.isfinite(DISTANCE_TIE_M),
    }
    payload = {
        "schema": "multi_cycle_viability_envelope_fixture_v1",
        "tests": tests,
        "pass": all(tests.values()),
    }
    payload["content_digest"] = digest(payload)
    return payload
