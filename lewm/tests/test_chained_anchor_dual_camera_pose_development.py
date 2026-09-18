"""Retained-anchor recovery, original conflict vetoes and exact fallback accounting."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose, BRIDGE_FIELDS
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.multi_reference_rgbd_pose_development import Reference
from lewm.tests.test_chained_corner_flow_association_development import scene
from lewm.direct_corner_flow_association_development import tracked_points
from lewm.joint_rgbd_rigid_pose_development import register
from lewm import joint_temporal_anchor_continuity_development as primary
from lewm import dual_camera_anchor_pose_development as dual


def empty(*args):
    return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2))


def prepared(monkeypatch):
    frames = scene(3)
    model = ChainedAnchorDualCameraPose()
    views = [dict(primary=f[2], auxiliary=f[2]) for f in frames]
    start = 1_500_000_000
    for i in range(2):
        model.frame = i
        model._cache_images(views[i], start+i*100_000_000)
    R, t, _, _ = register(*tracked_points(frames[0][2], frames[1][2])[0],
                          gyro_rotation=np.eye(3), mode='joint', frame=1)
    anchor = Reference(0, start, views[0], np.eye(3), np.eye(3), np.zeros(3))
    previous = Reference(1, start+100_000_000, views[1], R, np.eye(3), t)
    model.references = [anchor]
    model.previous = previous
    model.last_R, model.last_p, model.frame = R.copy(), t.copy(), 2
    monkeypatch.setattr(primary, 'matched_points', empty)
    monkeypatch.setattr(dual, 'matched_points', empty)
    return model, views[2], start+200_000_000


def test_actual_chained_fit_replaces_bridge_with_anchor_without_pose_composition(monkeypatch):
    model, current, now = prepared(monkeypatch)
    references, previous = list(model.references), model.previous
    result = model._measure(current, np.eye(3), now)
    candidate, alternative, bridge = result
    assert not alternative and not bridge
    assert candidate['reference'] is references[0]
    assert candidate['registration']['inliers'] >= 12
    assert candidate['registration']['chained_corner_flow_association']['intervals'] == 2
    assert model.last_continuity['status'] == 'ANCHOR_MEASUREMENT'
    assert model.bridge_frames == model.total_bridge_frames == 0
    assert model.bridge_path_m == 0.
    receipt = model.last_chained_anchor_fallback
    assert receipt['accepted'] and receipt['original_bridge_available']
    assert receipt['original_qualified_measurements_checked'] > 0
    assert receipt['pose_increments_composed'] is False and receipt['bridge_budget_unchanged']
    assert model.references == references and model.previous is previous and not model._chain_mode


def test_missing_chain_restores_original_bridge_and_trace_exactly(monkeypatch):
    model, current, now = prepared(monkeypatch)
    del model._image_history[0]
    result = model._measure(current, np.eye(3), now)
    assert result[2] is True and model.bridge_frames == model.total_bridge_frames == 1
    assert model.last_continuity['status'] == 'MEASURED_INCREMENT_BRIDGE'
    assert model.last_direct_flow_fallback['accepted']
    assert model.last_chained_anchor_fallback['original_result_restored']
    assert not model.last_chained_anchor_fallback['accepted'] and not model._chain_mode


def test_no_chain_keeps_original_exhausted_bridge_terminal(monkeypatch):
    model, current, now = prepared(monkeypatch)
    model._image_history.clear()
    model.bridge_frames = 10
    with pytest.raises(SensorContractError, match='bounded measured bridge exhausted'):
        model._measure(current, np.eye(3), now)
    assert model.bridge_frames == 10 and model.total_bridge_frames == 0
    assert model.last_continuity['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    assert model.last_chained_anchor_fallback['original_result_restored']


def test_original_anchor_success_does_not_enter_chain_search(monkeypatch):
    model = ChainedAnchorDualCameraPose()
    monkeypatch.setattr(model, '_cache_images', lambda *args: None)
    result = (object(), False, False)
    monkeypatch.setattr(DirectFlowDualCameraAnchorPose, '_measure', lambda *args: result)
    assert model._measure({}, np.eye(3), 1) is result
    assert model.last_chained_anchor_fallback is None


@pytest.mark.parametrize('status', ['ANCHOR_CONFLICT_OR_INVALID', 'ANCHOR_INCREMENT_CONFLICT',
    'CROSS_CAMERA_MEASUREMENT_CONFLICT', 'DIRECT_FLOW_ORIGINAL_MEASUREMENT_CONFLICT', 'VALIDATING_CURRENT_INPUT'])
def test_original_conflict_or_invalidity_never_enters_chain_search(monkeypatch, status):
    model = ChainedAnchorDualCameraPose()
    monkeypatch.setattr(model, '_cache_images', lambda *args: None)
    calls = []
    def measure(*args):
        calls.append(model._chain_mode)
        model.last_continuity = dict(status=status)
        raise SensorContractError('original qualified conflict')
    monkeypatch.setattr(DirectFlowDualCameraAnchorPose, '_measure', measure)
    with pytest.raises(SensorContractError, match='original qualified conflict'):
        model._measure({}, np.eye(3), 1)
    assert calls == [False] and model.last_chained_anchor_fallback is None


@pytest.mark.parametrize('conflict', ['position', 'rotation', None])
def test_original_measurements_veto_new_anchor(monkeypatch, conflict):
    model = ChainedAnchorDualCameraPose()
    model.frame = 3
    monkeypatch.setattr(model, '_cache_images', lambda *args: None)
    witness = dict(position_initial_body_m=[.03 if conflict == 'position' else 0., 0., 0.],
        composed_rotation_initial_body_from_current_body=np.eye(3).tolist())
    if conflict == 'rotation':
        c, s = np.cos(.11), np.sin(.11)
        witness['composed_rotation_initial_body_from_current_body'] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    original = (object(), False, True)
    candidate = dict(p=np.zeros(3), R=np.eye(3), reference=SimpleNamespace(frame=0),
                     registration={'chained_corner_flow_association': {}})
    calls = []
    def measure(*args):
        calls.append(model._chain_mode)
        model.last_camera_selection = {}
        if not model._chain_mode:
            model.bridge_frames += 1
            model.total_bridge_frames += 1
            model.bridge_path_m += .01
            model.last_continuity = dict(status='MEASURED_INCREMENT_BRIDGE',
                                         rotation_measurement_witnesses=[deepcopy(witness)])
            return original
        assert [getattr(model, k) for k in BRIDGE_FIELDS] == [0, 0, 0.]
        model.last_continuity = dict(status='ANCHOR_MEASUREMENT')
        return candidate, False, False
    monkeypatch.setattr(DirectFlowDualCameraAnchorPose, '_measure', measure)
    if conflict:
        with pytest.raises(SensorContractError, match='qualified original measurement'):
            model._measure({}, np.eye(3), 1)
        assert model.last_continuity['status'] == 'CHAINED_ANCHOR_ORIGINAL_MEASUREMENT_CONFLICT'
        assert not model.last_chained_anchor_fallback['accepted']
    else:
        assert model._measure({}, np.eye(3), 1)[0] is candidate
        assert model.last_chained_anchor_fallback['accepted']
    assert calls == [False, True] and not model._chain_mode
    assert model.last_chained_anchor_fallback['original_qualified_measurements_checked'] == 1


def test_new_qualified_conflict_is_terminal_instead_of_restoring_old_bridge(monkeypatch):
    model = ChainedAnchorDualCameraPose()
    monkeypatch.setattr(model, '_cache_images', lambda *args: None)
    def measure(*args):
        model.last_camera_selection = {}
        model.last_continuity = dict(status='ANCHOR_INCREMENT_CONFLICT' if model._chain_mode else 'MEASURED_INCREMENT_BRIDGE')
        if model._chain_mode: raise SensorContractError('new qualified conflict')
        model.bridge_frames += 1
        return object(), False, True
    monkeypatch.setattr(DirectFlowDualCameraAnchorPose, '_measure', measure)
    with pytest.raises(SensorContractError, match='new qualified conflict'):
        model._measure({}, np.eye(3), 1)
    assert model.last_continuity['status'] == 'ANCHOR_INCREMENT_CONFLICT'
    assert 'original_result_restored' not in model.last_chained_anchor_fallback
    assert not model._chain_mode


def test_history_is_owned_readonly_bounded_and_clock_checked():
    model = ChainedAnchorDualCameraPose()
    frame = scene(2)[0][2]
    views = dict(primary=frame, auxiliary=frame)
    for i in range(36):
        model.frame = i
        model._cache_images(views, 1_500_000_000+i*100_000_000)
    assert tuple(model._image_history) == tuple(range(3, 36))
    cached = model._image_history[35][1]['primary']
    expected = cached.gray.copy()
    frame.gray.fill(0)
    np.testing.assert_array_equal(cached.gray, expected)
    assert not cached.gray.flags.writeable and not cached.depth['valid'].flags.writeable
    with pytest.raises(SensorContractError, match='same measured clock'):
        model._cache_images(views, 1)


def test_public_motion_contract_latches_invalid_sensor_failure():
    from lewm.chained_anchor_visual_motion_development import ChainedAnchorVisualMotion
    motion = ChainedAnchorVisualMotion()
    result = motion.observe({}, {}, {}, auxiliary_rgb={}, auxiliary_depth={}, now_ns=1)
    assert result['status'] == 'VISUAL_TERMINAL_FAILURE' and result['current_pose'] is None
    repeated = motion.observe({}, {}, {}, auxiliary_rgb={}, auxiliary_depth={}, now_ns=2)
    assert repeated['terminal_failure'] == result['terminal_failure']
    assert motion.model._image_history == {}
