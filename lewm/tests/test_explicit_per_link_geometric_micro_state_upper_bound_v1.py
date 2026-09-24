import numpy as np

from scripts import materialize_explicit_per_link_geometric_micro_state_upper_bound_v1 as M
from scripts import evaluate_explicit_per_link_geometric_micro_state_upper_bound_v1 as E


def test_unique_deployable_actions_use_controller_and_applied_command():
    rows = [
        {"controller": "route", "applied_action": [0.0, 0.0, 0.0], "action_index": 0},
        {"controller": "route", "applied_action": [0.0, 0.0, 0.0], "action_index": 1},
        {"controller": "lateral", "applied_action": [0.0, 0.0, 0.0], "action_index": 2},
    ]
    assert [row["action_index"] for row in M.unique_rows(rows)] == [0, 2]


def test_scene_and_point_clearance_detect_crossing():
    contract = [{"link_index": 0, "link_name": "base", "kind": "sphere", "data": [0.05]}]
    geom = np.zeros((50, 1, 7), np.float32)
    geom[:, 0, 0] = np.linspace(0.5, 0.05, 50)
    geom[:, 0, 2] = 0.5
    geom[:, 0, 3] = 1.0
    boxes = (
        np.asarray([[0.0, 0.0, 0.5]]),
        np.asarray([[0.1, 0.5, 0.5]]),
        np.asarray([0.0]),
        np.asarray(["wall"]),
    )
    cloud = np.asarray([[0.1, 0.0, 0.5], [0.1, 0.05, 0.5]])
    assert M.geom_clearance_steps(geom, contract, boxes).min() < 0
    assert M.point_clearance_steps(geom, contract, cloud).min() < 0


def test_contact_metrics_keep_frozen_target_distinct_from_query():
    frozen = np.asarray([False, True, True, False])
    exact_query = np.asarray([False, True, False, True], float)
    result = E.contact_metrics(frozen, exact_query, None, True)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 1
