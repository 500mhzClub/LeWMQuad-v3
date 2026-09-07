import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_primitive_obstacle_memory_development import observed_stream
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.diagnose_go2_plane_memory_startup_development import local_geometry, cause_chain


def test_local_geometry_does_not_require_or_fabricate_historical_translation():
    p, d, _, _ = next(observed_stream(1))
    row = local_geometry(ArticulatedCollisionGeometry(URDF), p, d, np.array([0., 0., 1.]))
    assert row['hypothesis_cell_rc'] is not None
    assert not row['historical_translation_used'] and not row['contact_permitted'] and not row['navigation_qualified']
    assert not row['floor_covered_primitives']
    assert max(row['foot_centre_optical_depths_m']) < .2


@pytest.mark.parametrize('fault', ['clock', 'validity'])
def test_local_geometry_requires_current_measured_posture(fault):
    p, d, _, _ = next(observed_stream(1))
    q = p['sensor_state']['sensed']['joints']
    if fault == 'clock': q['measured_ns'][-1] -= 1
    else: q['valid'][-1, 0] = False
    with pytest.raises(SensorContractError):
        local_geometry(ArticulatedCollisionGeometry(URDF), p, d, np.array([0., 0., 1.]))


def test_failure_report_preserves_root_cause():
    try:
        try: raise ValueError('velocity unavailable')
        except ValueError as error: raise SensorContractError('stop') from error
    except SensorContractError as error:
        assert cause_chain(error) == ['stop', 'velocity unavailable']
