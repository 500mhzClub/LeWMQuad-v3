"""Synthetic late-frame witnesses; no continuous raw visual history claim."""
import ast
from copy import deepcopy
from functools import partial
import inspect

import numpy as np
import pytest

from lewm import extended_return_budget_transport_development as new
from lewm.tests import test_measured_floor_transport_development as fixtures
from lewm.tests.test_continuous_pulse_execution_development import visual


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(fixtures.fixture, 'visual', partial(visual, origin=1_500_000_000))


def late_inputs(frame):
    _, anchor, raw, fit, clouds, _ = fixtures.transported()
    shift = frame-2
    def shifted(value, key=None):
        if isinstance(value, dict): return {k:shifted(v, k) for k,v in value.items()}
        if isinstance(value, list): return [shifted(v) for v in value]
        if type(value) is int and key in ('frame', 'previous_frame', 'current_frame', 'reference_frame'):
            return value+shift
        if type(value) is int and key is not None and key.endswith('_ns'):
            return value+shift*100_000_000
        return deepcopy(value)
    revised_anchor = shifted(anchor)
    # The independently defined initial floor reference remains frame zero.
    revised_anchor['floor_registration']['reference'] = deepcopy(anchor['floor_registration']['reference'])
    return revised_anchor, shifted(raw), fit, clouds, 1_500_000_000+frame*100_000_000


def produce(frame):
    anchor, raw, fit, clouds, now = late_inputs(frame)
    result = new.transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
        auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])
    return result, anchor, raw, fit, clouds, now


@pytest.mark.parametrize('frame', [2, 4095, 4096, 8013])
def test_complete_transport_reconstruction_at_old_and_new_boundaries(frame):
    result, anchor, raw, fit, clouds, now = produce(frame)
    position, rotation, pose = new.current_measured_floor_pose(result, identity=(0, 0, 0), now_ns=now)
    assert pose['frame'] == frame and result['floor_transport']['correction']['anchor_frame'] == frame-1
    assert result['floor_transport']['anchor'] == anchor
    assert result['original_visual_evidence'] == raw
    np.testing.assert_allclose(position, [.02, 0., .02], atol=1e-6, rtol=0)
    expected_rotation = (np.asarray(anchor['current_pose']['rotation_initial_body_from_current_body'])
        @ np.asarray(anchor['original_visual_evidence']['current_pose']['rotation_initial_body_from_current_body']).T
        @ np.asarray(raw['current_pose']['rotation_initial_body_from_current_body']))
    np.testing.assert_array_equal(rotation, expected_rotation)
    assert not result['navigation_qualified'] and not result['native_pose_used']
    assert not result['historical_map_rewritten'] and not pose['current_floor_registration_used']
    assert result['floor_transport']['correction']['position_error_bound'] is None
    if frame < 4096:
        old = new.original.transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
            auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])
        assert result == old
    else:
        with pytest.raises(ValueError, match='earlier admitted floor anchor'):
            new.original.transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
                auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])


@pytest.mark.parametrize('frame', [8014, 8192])
def test_exceeding_complete_observation_population_is_rejected(frame):
    with pytest.raises(ValueError, match='earlier admitted floor anchor'): produce(frame)


@pytest.mark.parametrize('fault', ['pose', 'anchor_pose', 'raw', 'plane', 'residual', 'count',
    'auxiliary_hash', 'clock', 'identity', 'claim', 'correction', 'future_anchor', 'reference'])
def test_late_transport_keeps_original_witness_rejections(fault):
    result, _, _, _, _, now = produce(4096)
    witness = result['floor_transport']
    if fault == 'pose': result['current_pose']['position_initial_body_m'][0] += .001
    elif fault == 'anchor_pose': witness['anchor']['current_pose']['position_initial_body_m'][0] += .001
    elif fault == 'raw': result['original_visual_evidence']['current_pose']['position_initial_body_m'][0] += .001
    elif fault == 'plane': witness['unavailable_current_plane']['available'] = True
    elif fault == 'residual': witness['camera_residuals'][1]['maximum_residual_m'] = .00301
    elif fault == 'count': witness['camera_residuals'][1]['count'] -= 1
    elif fault == 'auxiliary_hash': witness['auxiliary_depth_sha256'] = 'a'*64
    elif fault == 'clock': result['decision_ns'] += 1
    elif fault == 'identity': result['identity'] = (0, False, 0)
    elif fault == 'claim': result['navigation_qualified'] = True
    elif fault == 'correction': witness['correction']['correction_magnitude_m'] = 0.
    elif fault == 'future_anchor': witness['anchor']['decision_ns'] = now+100_000_000
    elif fault == 'reference': witness['anchor']['floor_registration']['reference']['frame'] = 4094
    with pytest.raises((ValueError, KeyError)):
        new.current_measured_floor_pose(result, identity=(0, 0, 0), now_ns=now)


def test_late_geometric_conflict_does_not_become_missingness_agreement():
    anchor, raw, _, clouds, now = late_inputs(8013)
    clouds = (clouds[0], clouds[1]+[0., 0., .01])
    fit = fixtures.fit_joint_plane(*clouds, [0., 0., 1.])
    assert fit['reason'] == 'insufficient_combined_two_axis_extent'
    with pytest.raises(ValueError, match='conflicts'):
        new.transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
            auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])


def test_registration_preserves_available_missing_and_reacquired_plane_receipts():
    original = new.TiledDensityFloorRegistration()
    revised = new.ExtendedReturnBudgetFloorRegistration()
    previous = None
    for frame, narrow in enumerate((False, True, True, False)):
        policy, primary, auxiliary, raw, now, _ = fixtures.item(frame, previous, narrow=narrow)
        old = original.observe(policy, primary, auxiliary, raw, now_ns=now)
        actual = revised.observe(policy, primary, auxiliary, raw, now_ns=now)
        assert old == actual and vars(original) == vars(revised)
        previous = raw
    with pytest.raises(ValueError): revised.observe(policy, primary, auxiliary, raw, now_ns=now)
    assert revised.failed


def test_composition_changes_only_the_declared_frame_ceiling():
    old = ast.parse(inspect.getsource(new.original.composition))
    actual = ast.parse(inspect.getsource(new.composition))
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.Constant(value=4096) if node.id == 'MAX_OBSERVATIONS' else node
    assert ast.dump(Normalize().visit(actual)) == ast.dump(old)
