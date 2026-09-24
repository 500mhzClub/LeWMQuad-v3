"""Frozen-source regression witness; no simulator or experiment artifacts opened."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path('/home/andrewknowles/Workspace/LeWMQuad-v3')
sys.path.insert(0, str(ROOT))
from scripts.run_physical_graph_edge_handoff_qualification_v1 import _GenesisPhysicalSession
from lewm.safety.contact_hazard_ontology_v1 import is_disallowed_contact


def frozen_detector(contact_rows):
    stub = SimpleNamespace(
        ctx=SimpleNamespace(
            build=SimpleNamespace(robot=SimpleNamespace(get_contacts=lambda **_: contact_rows)),
            runner=SimpleNamespace(_as_np=lambda value: value),
        ),
        _contact_topology={'robot': {1, 2}, 'support': {2}, 'ground': {0}},
    )
    return _GenesisPhysicalSession._disallowed_contact(stub)


@pytest.mark.parametrize('wall_first', [True, False])
def test_frozen_batched_detector_falsely_counts_zero_force_wall_contact(wall_first):
    # Link 1 is body, 2 is foot, 0 is floor, 10 is wall.
    # Both geometric contact entries belong to the robot and are API-valid.
    # Only the allowed foot/floor pair has nonzero force.
    pairs = [(1, 10, [0., 0., 0.]), (2, 0, [0., 0., 100.])]
    if not wall_first:
        pairs.reverse()
    rows = {
        'link_a': np.array([[pair[0] for pair in pairs]]),
        'link_b': np.array([[pair[1] for pair in pairs]]),
        'force_a': np.array([[pair[2] for pair in pairs]]),
        'force_b': -np.array([[pair[2] for pair in pairs]]),
        'valid_mask': np.array([[True, True]]),
    }
    assert not any(is_disallowed_contact(
        robot_link_id=a, environment_link_id=b,
        foot_link_ids={2}, ground_link_ids={0}, self_contact=False,
        force_magnitude_n=float(np.linalg.norm(force)),
    ) for a, b, force in pairs)
    # This assertion intentionally documents the existing bug, not desired behavior.
    assert frozen_detector(rows) is True
    # Exactly the same contacts supplied with unbatched API shape are classified correctly.
    unbatched = {key: value[0] for key, value in rows.items() if key != 'valid_mask'}
    assert frozen_detector(unbatched) is False


def test_frozen_detector_still_identifies_a_real_nonzero_wall_contact():
    rows = {
        'link_a': np.array([[1]]), 'link_b': np.array([[10]]),
        'force_a': np.array([[[0., 2., 0.]]]),
        'force_b': np.array([[[0., -2., 0.]]]),
        'valid_mask': np.array([[True]]),
    }
    assert frozen_detector(rows) is True
