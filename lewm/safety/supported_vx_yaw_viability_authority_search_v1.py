"""Pure contracts and reducers for the supported vx/yaw viability search."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


MAX_REVERSE_M_S = 0.20
MAX_YAW_RAD_S = 0.45
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def requested_grid() -> list[dict]:
    """Return the prospectively frozen 21-command grid.

    The matched-fraction reverse arcs cover every prescribed reverse and yaw
    level while respecting the explicit cap of 25 requested combinations.
    """

    rows = [{"search_index": 0, "name": "zero", "family": "ZERO",
             "fraction": 0.0, "requested_vx_vy_wz": [0.0, 0.0, 0.0]}]
    for fraction in FRACTIONS:
        rows.append({"search_index": len(rows), "name": f"reverse_{fraction:.2f}",
                     "family": "PURE_REVERSE_RETREAT", "fraction": fraction,
                     "requested_vx_vy_wz": [-MAX_REVERSE_M_S * fraction, 0.0, 0.0]})
    for fraction in FRACTIONS:
        for direction, sign in (("left", 1.0), ("right", -1.0)):
            rows.append({"search_index": len(rows), "name": f"turn_{direction}_{fraction:.2f}",
                         "family": "MIRRORED_IN_PLACE_ESCAPE_TURN", "direction": direction,
                         "fraction": fraction,
                         "requested_vx_vy_wz": [0.0, 0.0, sign * MAX_YAW_RAD_S * fraction]})
    for fraction in FRACTIONS:
        for direction, sign in (("left", 1.0), ("right", -1.0)):
            rows.append({"search_index": len(rows), "name": f"reverse_arc_{direction}_{fraction:.2f}",
                         "family": "MIRRORED_REVERSE_ARC_RETREAT", "direction": direction,
                         "fraction": fraction,
                         "requested_vx_vy_wz": [-MAX_REVERSE_M_S * fraction, 0.0,
                                                  sign * MAX_YAW_RAD_S * fraction]})
    if len(rows) != 21:
        raise AssertionError(len(rows))
    return rows


def command_key(command: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(np.float32(value)) for value in command)


def deduplicate_applied(rows: Sequence[Mapping], historical: Sequence[Mapping]) -> dict:
    historical_keys = {command_key(row["target_command"]): int(row["candidate_index"])
                       for row in historical}
    groups: dict[tuple[float, float, float], list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[command_key(row["applied_vx_vy_wz"])].append(row)
    unique = []
    for key, group in sorted(groups.items(), key=lambda item: min(int(r["search_index"]) for r in item[1])):
        first = min(group, key=lambda row: int(row["search_index"]))
        unique.append({
            "representative_search_index": int(first["search_index"]),
            "applied_vx_vy_wz": list(key),
            "search_indices": [int(row["search_index"]) for row in group],
            "duplicate_within_grid_count": len(group) - 1,
            "historical_duplicate_candidate": historical_keys.get(key),
            "genuinely_new": key not in historical_keys,
        })
    return {
        "requested_count": len(rows),
        "unique_applied_count": len(unique),
        "duplicates_within_grid": sum(row["duplicate_within_grid_count"] for row in unique),
        "duplicates_of_historical": sum(row["historical_duplicate_candidate"] is not None for row in unique),
        "genuinely_new_applied": sum(row["genuinely_new"] for row in unique),
        "unique": unique,
    }


def residual_classification(rows: Sequence[Mapping], *, boundary_contact: bool = False) -> str:
    if boundary_contact:
        return "PRE_EXISTING_CONTACT"
    viable = [row for row in rows if row.get("viability_admissible")]
    if viable:
        families = {str(row["family"]) for row in viable}
        if "PURE_REVERSE_RETREAT" in families:
            return "PURE_REVERSE_RECOVERS_VIABILITY"
        if "MIRRORED_REVERSE_ARC_RETREAT" in families:
            return "REVERSE_ARC_RECOVERS_VIABILITY"
        if "MIRRORED_IN_PLACE_ESCAPE_TURN" in families:
            return "IN_PLACE_TURN_RECOVERS_VIABILITY"
    if any(row.get("safe_prefix") for row in rows):
        return "SUPPORTED_COMMAND_RECOVERS_PREFIX_ONLY"
    contact_steps = [row.get("outcome", {}).get("first_contact_step") for row in rows]
    if contact_steps and all(step is not None and int(step) < 10 for step in contact_steps):
        # One low-level policy interval is 10 physics steps (20 ms).  Before
        # that interval completes no newly requested command can produce a
        # full controller-response update.
        return "CONTACT_BEFORE_SUPPORTED_CONTROL_AUTHORITY"
    return "NO_SUPPORTED_VX_YAW_VIABLE_ACTION"


def fixture_payload() -> dict:
    grid = requested_grid()
    tests = {
        "grid_within_cap": len(grid) <= 25,
        "vy_always_zero": all(row["requested_vx_vy_wz"][1] == 0.0 for row in grid),
        "no_forward": all(row["requested_vx_vy_wz"][0] <= 0.0 for row in grid),
        "mirrored_turns": sum(row["family"] == "MIRRORED_IN_PLACE_ESCAPE_TURN" for row in grid) == 8,
        "mirrored_arcs": sum(row["family"] == "MIRRORED_REVERSE_ARC_RETREAT" for row in grid) == 8,
        "pure_reverse": sum(row["family"] == "PURE_REVERSE_RETREAT" for row in grid) == 4,
    }
    result = {"schema": "supported_vx_yaw_viability_authority_search_fixture_v1",
              "tests": tests, "pass": all(tests.values())}
    result["content_digest"] = digest(result)
    return result
