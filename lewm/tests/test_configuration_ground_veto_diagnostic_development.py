import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_observed_setup_configuration_development import ready
from scripts.diagnose_go2_configuration_ground_veto_development import trace_configuration


@pytest.mark.parametrize('offset', [(1., 0., 0.), (2.2, 0., 0.), (-1.3, 0., 0.)])
def test_reproduces_every_original_veto_without_changing_decision(offset):
    owner, now = ready(3.)
    result = trace_configuration(owner, np.asarray(offset), now_ns=now, through_ns=now+100_000_000)
    assert result['all_veto_witnesses_reproduced'] and result['original_query_unchanged']
    assert result['compiled_reference_all_fields_exact']
    assert not result['ground_support_permission'] and not result['navigation_action_permitted']
    current = [w for w in result['witnesses'] if w['measured_ns'] == now]
    assert len(current) == 27 and all(w['relative_point_allowance_m'] == 0 for w in current)
    if offset[0] == 2.2:
        base = [w for w in result['witnesses'] if w['shape_id'] == 'base:0']
        assert any(w['original_obstacle_veto'] and w['plane_masked_conflict'] for w in base)
    if offset[0] < 0:
        assert not result['original_query']['all_primitives_conditionally_nonfloor_clear'] or all(
            r['supplied_clearance_used'] for r in result['original_query']['primitives'])


def test_stale_diagnostic_fails_before_reading_observation_internals():
    owner, now = ready()
    with pytest.raises(SensorContractError):
        trace_configuration(owner, np.array([1.,0.,0.]), now_ns=now-1, through_ns=now)
