from lewm.safety.contact_hazard_ontology_v1 import (
    branch_labels, classify_event, digest, group_contact_points,
    is_disallowed_contact, reduce_event, side_from_body_point,
)


def _point(step, *, link=4, obj="wall", force=10.0, branch="s:00", side=(0.2, 0.0, 0.0)):
    return {
        "branch_id": branch, "state_id": "s", "candidate_index": 0,
        "family": "large_enclosed_maze", "physics_step": step,
        "tick": step // 50 + 1, "contact_point_index": 0,
        "robot_link_id": link, "robot_link_name": "trunk",
        "environment_link_id": 100, "environment_link_name": obj,
        "environment_object_id": obj, "environment_object_class": "wall",
        "environment_properties": {
            "fragility_category": "non_fragile", "safety_critical": False,
            "human_or_person_proxy": False, "damage_observed": False,
            "prohibited_contact": False,
        },
        "normal_force_n": force, "normal_impulse_increment_n_s": force * 0.002,
        "relative_normal_speed_m_s": 0.1, "relative_tangential_speed_m_s": 0.02,
        "penetration_m": 0.001, "side_of_robot": side_from_body_point(side),
        "contact_point_world_m": [0, 0, 0], "contact_point_body_m": list(side),
        "contact_normal_world": [1, 0, 0], "simultaneous_contact_points": 1,
        "loss_of_stability": False, "fall": False, "branch_stuck": False,
        "route_progress_m": 0.2,
    }


def test_exclusion_and_body_region_contract():
    assert not is_disallowed_contact(robot_link_id=12, environment_link_id=1,
                                     foot_link_ids={12}, ground_link_ids={1})
    assert not is_disallowed_contact(robot_link_id=3, environment_link_id=4,
                                     foot_link_ids={12}, ground_link_ids={1}, self_contact=True)
    assert is_disallowed_contact(robot_link_id=3, environment_link_id=9,
                                 foot_link_ids={12}, ground_link_ids={1})
    assert side_from_body_point((1, 0, 0)) == "front"
    assert side_from_body_point((1, 1, 0)) == "front-left"
    assert side_from_body_point((0, 0, -0.3)) == "underside"


def test_event_grouping_gap_and_repeat_are_deterministic():
    points = [_point(1), _point(2), _point(5), _point(9)]
    events = group_contact_points(points)
    assert [len(event) for event in events] == [3, 1]
    assert digest(events) == digest(group_contact_points(list(reversed(points))))


def test_low_energy_brush_and_missing_evidence():
    event = reduce_event([_point(1)], event_index=0)
    event["classification"] = classify_event(event)
    assert event["classification"]["category"] == "RECOVERABLE_LOW_SEVERITY_CONTACT"
    missing = dict(event)
    missing["environment_properties"] = {"fragility_category": None}
    assert classify_event(missing)["category"] == "SEVERITY_UNRESOLVED"


def test_hazard_fixtures_and_branch_reduction():
    cases = []
    for kind in ("persistent", "high_speed", "repeated", "stuck", "instability", "fragile"):
        event = reduce_event([_point(i) for i in range(1, 31)] if kind == "persistent" else [_point(1)], event_index=len(cases))
        if kind == "high_speed":
            event["peak_relative_normal_speed_m_s"] = 0.8
        elif kind == "repeated":
            event["repeated_contact_count"] = 2
        elif kind == "stuck":
            event["subsequent_stuck"] = True
        elif kind == "instability":
            event["loss_of_stability"] = True
        elif kind == "fragile":
            event["environment_properties"] = dict(event["environment_properties"], fragility_category="fragile")
        event["classification"] = classify_event(event)
        cases.append(event)
    assert cases[4]["classification"]["category"] == "MATERIAL_HAZARDOUS_CONTACT"
    assert cases[5]["classification"]["category"] == "MATERIAL_HAZARDOUS_CONTACT"
    assert all(row["classification"]["category"] == "SEVERITY_UNRESOLVED" for row in cases[:4])
    reduced = branch_labels(cases)
    assert reduced["any_material_hazardous_contact"]
    assert reduced["maximum_event_severity"] == "MATERIAL_HAZARDOUS_CONTACT"


def test_threshold_tie_and_byte_identical_serialisation():
    point = _point(1, force=250.0)
    first = reduce_event([point], event_index=0)
    second = reduce_event([point], event_index=0)
    assert first == second
    assert digest(first) == digest(second)
