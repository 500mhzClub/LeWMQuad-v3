"""Prospective contact-hazard ontology for Go2 search/inspection development.

This module deliberately does not reinterpret the historical binary contact
target.  It classifies newly instrumented *events*, and leaves an event
unresolved whenever the evidence needed to establish material consequence or
recoverability is absent.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "lewm_contact_hazard_ontology_v1"
EVENT_GAP_PHYSICS_STEPS = 2
PHYSICS_DT_S = 0.002
DISALLOWED_CONTACT_FORCE_FLOOR_N = 1e-3

# These are conservative *development screening* assumptions, not injury or
# robot-damage limits.  They can establish a low-energy class only when object
# consequence metadata is independently complete.  They never establish that
# contact with a person, fragile object, or safety-critical asset is acceptable.
LOW_ENERGY_SCREEN = {
    "maximum_duration_s": 0.040,
    "maximum_relative_normal_speed_m_s": 0.20,
    "maximum_integrated_normal_impulse_n_s": 0.50,
    "maximum_penetration_m": 0.002,
    "maximum_repeated_contact_count": 1,
}

CATEGORIES = (
    "MATERIAL_HAZARDOUS_CONTACT",
    "RECOVERABLE_LOW_SEVERITY_CONTACT",
    "SEVERITY_UNRESOLVED",
)
ANNOTATIONS = (
    "STABILITY_HAZARD",
    "DAMAGE_RELEVANT",
    "HUMAN_OR_FRAGILE_OBJECT_RELEVANT",
    "STUCK_OR_ENTRAPMENT_CONSEQUENCE",
    "TASK_PROGRESS_CONSEQUENCE",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def side_from_body_point(point_body_m: Sequence[float] | None) -> str:
    """Assign one of the frozen octants, with an underside exception."""
    if point_body_m is None or len(point_body_m) != 3:
        return "unresolved"
    x, y, z = (float(v) for v in point_body_m)
    if not all(math.isfinite(v) for v in (x, y, z)):
        return "unresolved"
    if z < -0.18:
        return "underside"
    angle = math.degrees(math.atan2(y, x))
    octants = (
        (22.5, "front"), (67.5, "front-left"), (112.5, "left"),
        (157.5, "rear-left"), (202.5, "rear"),
        (247.5, "rear-right"), (292.5, "right"),
        (337.5, "front-right"), (360.0, "front"),
    )
    wrapped = angle % 360.0
    for boundary, name in octants:
        if wrapped < boundary:
            return name
    return "front"


def is_disallowed_contact(
    *, robot_link_id: int, environment_link_id: int,
    foot_link_ids: Iterable[int], ground_link_ids: Iterable[int],
    self_contact: bool = False, force_magnitude_n: float | None = None,
) -> bool:
    """Apply the unchanged historical exclusion and force-floor contract."""
    if self_contact:
        return False
    if force_magnitude_n is not None and force_magnitude_n <= DISALLOWED_CONTACT_FORCE_FLOOR_N:
        return False
    return not (
        int(robot_link_id) in set(int(v) for v in foot_link_ids)
        and int(environment_link_id) in set(int(v) for v in ground_link_ids)
    )


def group_contact_points(
    points: Sequence[Mapping[str, Any]], *, gap_steps: int = EVENT_GAP_PHYSICS_STEPS,
) -> list[list[dict[str, Any]]]:
    """Group by branch/link/object with gaps no larger than ``gap_steps``."""
    if gap_steps < 0:
        raise ValueError("gap_steps must be non-negative")
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in points:
        row = dict(raw)
        key = (
            str(row["branch_id"]), int(row["robot_link_id"]),
            str(row["environment_object_id"]),
        )
        grouped[key].append(row)
    events: list[list[dict[str, Any]]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: (
            int(row["physics_step"]), int(row.get("contact_point_index", 0))))
        current: list[dict[str, Any]] = []
        last_step: int | None = None
        for row in rows:
            step = int(row["physics_step"])
            if current and last_step is not None and step - last_step > gap_steps + 1:
                events.append(current)
                current = []
            current.append(row)
            last_step = step
        if current:
            events.append(current)
    return sorted(events, key=lambda rows: (
        str(rows[0]["branch_id"]), int(rows[0]["physics_step"]),
        int(rows[0]["robot_link_id"]), str(rows[0]["environment_object_id"])))


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    result = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def reduce_event(points: Sequence[Mapping[str, Any]], *, event_index: int) -> dict[str, Any]:
    if not points:
        raise ValueError("event requires at least one contact point")
    rows = sorted((dict(row) for row in points), key=lambda row: (
        int(row["physics_step"]), int(row.get("contact_point_index", 0))))
    first, last = rows[0], rows[-1]
    steps = sorted({int(row["physics_step"]) for row in rows})
    force = _finite_values(rows, "normal_force_n")
    impulse = _finite_values(rows, "normal_impulse_increment_n_s")
    tangential_impulse = _finite_values(rows, "tangential_impulse_increment_n_s")
    normal_speed = _finite_values(rows, "relative_normal_speed_m_s")
    tangential_speed = _finite_values(rows, "relative_tangential_speed_m_s")
    penetration = _finite_values(rows, "penetration_m")
    embodied_fields = (
        "body_linear_velocity_change_m_s", "body_angular_velocity_change_rad_s",
        "projected_gravity_change", "joint_velocity_change_rad_s",
        "joint_acceleration_peak_rad_s2", "actuator_torque_peak_nm",
        "support_contact_force_peak_n",
    )
    sides = sorted({str(row.get("side_of_robot", "unresolved")) for row in rows})
    event = {
        "schema": "lewm_contact_hazard_event_v1",
        "event_id": f"{first['branch_id']}:event-{event_index:03d}",
        "branch_id": str(first["branch_id"]),
        "state_id": str(first["state_id"]),
        "candidate_index": int(first["candidate_index"]),
        "family": str(first["family"]),
        "robot_link_id": int(first["robot_link_id"]),
        "robot_link_name": first.get("robot_link_name"),
        "environment_link_id": int(first["environment_link_id"]),
        "environment_link_name": first.get("environment_link_name"),
        "environment_object_id": str(first["environment_object_id"]),
        "environment_object_class": first.get("environment_object_class"),
        "environment_properties": first.get("environment_properties"),
        "start_physics_step": steps[0],
        "end_physics_step": steps[-1],
        "start_tick": int(first["tick"]),
        "end_tick": int(last["tick"]),
        "duration_s": (steps[-1] - steps[0] + 1) * PHYSICS_DT_S,
        "physics_steps_with_contact": len(steps),
        "raw_contact_point_count": len(rows),
        "maximum_simultaneous_contact_points": max(
            int(row.get("simultaneous_contact_points", 1)) for row in rows),
        "side_of_robot": sides[0] if len(sides) == 1 else "unresolved",
        "side_observations": sides,
        "peak_normal_force_n": max(force) if force else None,
        "integrated_normal_impulse_n_s": sum(impulse) if impulse else None,
        "integrated_tangential_impulse_n_s": sum(tangential_impulse) if tangential_impulse else None,
        "peak_relative_normal_speed_m_s": max(normal_speed) if normal_speed else None,
        "peak_relative_tangential_speed_m_s": max(tangential_speed) if tangential_speed else None,
        "maximum_penetration_m": max(penetration) if penetration else None,
        "peak_embodied_response": {
            field: (max(values) if (values := _finite_values(rows, field)) else None)
            for field in embodied_fields
        },
        "contact_points_world_m": [row.get("contact_point_world_m") for row in rows],
        "contact_points_body_m": [row.get("contact_point_body_m") for row in rows],
        "contact_normals_world": [row.get("contact_normal_world") for row in rows],
        "loss_of_stability": any(bool(row.get("loss_of_stability", False)) for row in rows),
        "fall": any(bool(row.get("fall", False)) for row in rows),
        "controller_saturation": None,
        "subsequent_stuck": bool(first.get("branch_stuck", False)),
        "route_progress_m": first.get("route_progress_m"),
        "task_progress_consequence": bool(
            isinstance(first.get("route_progress_m"), (int, float))
            and float(first["route_progress_m"]) <= 0.0),
        "repeated_contact_count": 1,
        "time_since_previous_contact_s": None,
        "recovery_time_s": None,
        "pose_deviation_from_nominal_m": None,
        "force_semantics": "Genesis solver contact force; impulse is force integrated at 0.002 s",
    }
    event["event_digest"] = digest(event)
    return event


def annotate_repetition(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in events:
        by_branch[str(raw["branch_id"])].append(dict(raw))
    result: list[dict[str, Any]] = []
    for branch_id in sorted(by_branch):
        rows = sorted(by_branch[branch_id], key=lambda row: int(row["start_physics_step"]))
        previous_end: int | None = None
        for index, row in enumerate(rows, start=1):
            row.pop("event_digest", None)
            row["repeated_contact_count"] = index
            row["time_since_previous_contact_s"] = (
                None if previous_end is None else
                (int(row["start_physics_step"]) - previous_end - 1) * PHYSICS_DT_S)
            row["event_digest"] = digest(row)
            result.append(row)
            previous_end = int(row["end_physics_step"])
    return result


def classify_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Classify exactly once; missing consequence evidence forces unresolved."""
    props = dict(event.get("environment_properties") or {})
    human_or_fragile = bool(
        props.get("human_or_person_proxy") is True
        or props.get("fragility_category") not in (None, "none", "non_fragile")
        or props.get("safety_critical") is True)
    stability = bool(event.get("loss_of_stability") or event.get("fall"))
    stuck = bool(event.get("subsequent_stuck"))
    progress = bool(event.get("task_progress_consequence"))
    damage = props.get("damage_observed") is True
    prohibited = props.get("prohibited_contact") is True
    annotations = {
        "STABILITY_HAZARD": stability,
        "DAMAGE_RELEVANT": bool(damage or human_or_fragile or prohibited),
        "HUMAN_OR_FRAGILE_OBJECT_RELEVANT": human_or_fragile,
        "STUCK_OR_ENTRAPMENT_CONSEQUENCE": stuck,
        "TASK_PROGRESS_CONSEQUENCE": progress,
    }
    reasons: list[str] = []
    if human_or_fragile:
        reasons.append("human_fragile_or_safety_critical_object")
    if stability:
        reasons.append("loss_of_stability_or_fall")
    if damage:
        reasons.append("recorded_damage")
    if prohibited:
        reasons.append("prospectively_prohibited_object_contact")
    if reasons:
        category = "MATERIAL_HAZARDOUS_CONTACT"
    else:
        required_object = (
            props.get("fragility_category") is not None
            and props.get("safety_critical") is not None
            and props.get("human_or_person_proxy") is not None
            and props.get("damage_observed") is not None)
        numeric = {
            "duration_s": event.get("duration_s"),
            "relative_normal_speed_m_s": event.get("peak_relative_normal_speed_m_s"),
            "integrated_normal_impulse_n_s": event.get("integrated_normal_impulse_n_s"),
            "penetration_m": event.get("maximum_penetration_m"),
            "repeated_contact_count": event.get("repeated_contact_count"),
        }
        evidence_complete = required_object and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) for value in numeric.values())
        low = evidence_complete and (
            float(numeric["duration_s"]) <= LOW_ENERGY_SCREEN["maximum_duration_s"]
            and float(numeric["relative_normal_speed_m_s"])
                <= LOW_ENERGY_SCREEN["maximum_relative_normal_speed_m_s"]
            and float(numeric["integrated_normal_impulse_n_s"])
                <= LOW_ENERGY_SCREEN["maximum_integrated_normal_impulse_n_s"]
            and float(numeric["penetration_m"]) <= LOW_ENERGY_SCREEN["maximum_penetration_m"]
            and int(numeric["repeated_contact_count"])
                <= LOW_ENERGY_SCREEN["maximum_repeated_contact_count"]
            and not stuck and not progress)
        if low:
            category = "RECOVERABLE_LOW_SEVERITY_CONTACT"
            reasons.append("complete_evidence_below_conservative_low_energy_screen")
        else:
            category = "SEVERITY_UNRESOLVED"
            if not required_object:
                reasons.append("object_consequence_metadata_incomplete")
            if not evidence_complete:
                reasons.append("physical_severity_evidence_incomplete")
            if evidence_complete and not low:
                reasons.append("not_below_low_energy_screen_without_material_hazard_limit")
    result = {
        "category": category,
        "annotations": annotations,
        "rationale": reasons,
        "ontology_schema": SCHEMA,
        "prospective_only": True,
    }
    result["classification_digest"] = digest(result)
    return result


def branch_labels(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = [str(row["classification"]["category"]) for row in events]
    severity_rank = {
        "RECOVERABLE_LOW_SEVERITY_CONTACT": 1,
        "SEVERITY_UNRESOLVED": 2,
        "MATERIAL_HAZARDOUS_CONTACT": 3,
    }
    maximum = max(categories, key=lambda value: severity_rank[value]) if categories else "NO_DISALLOWED_CONTACT"
    result = {
        "any_material_hazardous_contact": "MATERIAL_HAZARDOUS_CONTACT" in categories,
        "any_recoverable_low_severity_contact": "RECOVERABLE_LOW_SEVERITY_CONTACT" in categories,
        "any_severity_unresolved": "SEVERITY_UNRESOLVED" in categories,
        "maximum_event_severity": maximum,
        "contact_event_count": len(events),
        "cumulative_impulse_n_s": sum(
            float(row["integrated_normal_impulse_n_s"])
            for row in events if row.get("integrated_normal_impulse_n_s") is not None),
        "contact_followed_by_stuck": any(bool(row.get("subsequent_stuck")) for row in events),
        "no_disallowed_contact": not events,
    }
    result["label_digest"] = digest(result)
    return result
