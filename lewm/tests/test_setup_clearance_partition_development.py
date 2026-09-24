import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.setup_clearance_partition_development import partition_setup_clearance, current_body_setup_partition
from lewm.tests.test_startup_observation_turn_development import components
from lewm.tests.test_continuous_startup_handoff_development import model, frames


def query(low, high, radius=0., **kwargs):
    _, settings = components()
    args = dict(identity=(0, 0, 0), now_ns=1_600_000_000, through_ns=2_000_000_000, observed_conflict=False)
    args.update(kwargs)
    return partition_setup_clearance(settings['region_prior'], low, high, radius, **args)


def volume(box): return float(np.prod(np.asarray(box[1])-box[0]))


def inside(points, box):
    return ((points >= box[0]) & (points <= box[1])).all(axis=1)


def test_straddling_box_keeps_residual_unknown_not_filled_hull():
    row = query([.8, -.1, -.1], [1.2, .1, .1])
    assert len(row['setup_covered_boxes']) == len(row['requires_sensor_evidence_boxes']) == 1
    assert not row['entire_query_conditionally_setup_nonfloor_clear']
    assert not row['residual_is_known_clear'] and not row['navigation_action_permitted']
    assert row['requires_sensor_evidence_boxes'][0][0][0] < 1.


def test_random_box_partition_covers_query_exactly_without_volume_overlap_or_missing_space():
    rng = np.random.default_rng(609071)
    for _ in range(100):
        low = rng.uniform(-2., 1., 3); high = low+rng.uniform(.01, 3., 3)
        row = query(low, high, .02)
        all_boxes = row['setup_covered_boxes']+row['requires_sensor_evidence_boxes']
        assert len(row['setup_covered_boxes']) <= 1 and len(row['requires_sensor_evidence_boxes']) <= 6
        assert sum(volume(b) for b in all_boxes) == pytest.approx(volume(row['whole_query_for_observed_veto']), abs=1e-12)
        points = rng.uniform(low-.02, high+.02, (100, 3))
        assert np.logical_or.reduce([inside(points, b) for b in all_boxes]).all()
        for box in all_boxes:
            assert np.all(np.asarray(box[0]) >= low-.02) and np.all(np.asarray(box[1]) <= high+.02)
        for box in row['setup_covered_boxes']:
            assert np.all(np.asarray(box[0]) > -1.) and np.all(np.asarray(box[1]) < 1.)


@pytest.mark.parametrize('case', ['conflict', 'expiry', 'before_anchor', 'horizon_expiry', 'outside'])
def test_expiry_or_conflict_never_preserves_supplied_clearance(case):
    kwargs = {}; low, high = [-.1]*3, [.1]*3
    if case == 'conflict': kwargs['observed_conflict'] = True
    if case == 'expiry': kwargs.update(now_ns=3_700_000_000, through_ns=3_700_000_000)
    if case == 'before_anchor': kwargs.update(now_ns=1_500_000_000, through_ns=1_600_000_000)
    if case == 'horizon_expiry': kwargs['through_ns'] = 3_600_000_001
    if case == 'outside': low, high = [2.]*3, [3.]*3
    row = query(low, high, **kwargs)
    assert not row['setup_covered_boxes']
    assert row['requires_sensor_evidence_boxes'] == [row['whole_query_for_observed_veto']]


@pytest.mark.parametrize('low,high', [([0.]*3, [0.]*3), ([0.,0.,0.], [0.,.5,.5]), ([1.]*3, [1.]*3)])
def test_degenerate_boxes_and_exact_boundary_points_are_not_lost(low, high):
    row = query(low, high)
    assert row['setup_covered_boxes'] or row['requires_sensor_evidence_boxes']
    if low == [1.]*3: assert not row['entire_query_conditionally_setup_nonfloor_clear']


@pytest.mark.parametrize('fault', ['identity', 'bool_identity', 'negative_radius', 'nan', 'inverted', 'bad_horizon', 'bool_radius'])
def test_malformed_query_is_rejected(fault):
    kwargs = {}; low, high, radius = [-.1]*3, [.1]*3, 0.
    if fault == 'identity': kwargs['identity'] = (0,0,1)
    if fault == 'bool_identity': kwargs['identity'] = (False,0,0)
    if fault == 'negative_radius': radius = -.1
    if fault == 'nan': low[0] = float('nan')
    if fault == 'inverted': low[0] = .2
    if fault == 'bad_horizon': kwargs['through_ns'] = 1_500_000_000
    if fault == 'bool_radius': radius = True
    with pytest.raises(SensorContractError): query(low, high, radius, **kwargs)


def test_live_current_body_partition_retains_source_roles_and_cannot_approve_motion():
    owner = model()
    for p,d,f,now in frames(8): owner.observe(p,d,f,now_ns=now)
    row = current_body_setup_partition(owner, now_ns=now)
    assert len(row['per_shape']) == 27 and len(row['setup_conditionally_covered_shapes']) == 27
    assert not row['shapes_requiring_additional_observation']
    assert not row['observed_current_posture']['conditional_clearance']
    assert not row['ground_support_permission'] and not row['future_gait_qualified'] and not row['navigation_action_permitted']
    assert row['current_measured_posture_only'] and row['point_expansion_m'] >= .04
    with pytest.raises(SensorContractError): current_body_setup_partition(owner, now_ns=now-1)


def test_continuing_observer_does_not_renew_expired_setup_evidence():
    owner = model()
    for p,d,f,now in frames(23):
        row = owner.observe(p,d,f,now_ns=now)
        assert not row['terminal']
    assert not owner.navigation_snapshot(now_ns=now)['supplied_region_active']
    partition = current_body_setup_partition(owner, now_ns=now)
    assert not partition['setup_conditionally_covered_shapes']
    assert len(partition['shapes_requiring_additional_observation']) == 27
    assert not partition['navigation_action_permitted']


def test_observed_veto_inside_starting_cube_removes_that_shapes_supplied_clearance():
    owner = model()
    for p,d,f,now in frames(8): owner.observe(p,d,f,now_ns=now)
    owner._evidence['non_floor_conflict'][0] = True
    partition = current_body_setup_partition(owner, now_ns=now)
    assert partition['per_shape'][0]['partition']['observed_conflict']
    assert len(partition['setup_conditionally_covered_shapes']) == 26
    assert len(partition['shapes_requiring_additional_observation']) == 1
