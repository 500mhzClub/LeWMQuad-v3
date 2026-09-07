"""Independent verifier math and complete synthetic saved-row challenges."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import scripts.verify_go2_temporal_anchor_continuity_v1 as checker
import scripts.replay_go2_temporal_anchor_continuity_v1 as replay
from lewm.tests.test_temporal_anchor_continuity_replay_development import synthetic, bridge_factory


@pytest.mark.parametrize('bad,expected', [
    (True, 1), (3., 3), ({'x': 1, 'extra': 0}, {'x': 1}),
    ([1], [1, 2]), (float('nan'), 0.), (float('inf'), 0.), (.1, .2),
])
def test_comparison_rejects_bad_types_structure_and_nonfinite_values(bad, expected):
    with pytest.raises(AssertionError):checker.same(bad, expected)


def test_statistics_and_increment_errors_have_explicit_empty_denominators():
    assert checker.statistics([]) == dict(count=0, mean=None, median=None, p95=None, maximum=None)
    assert checker.statistics([0., 1., 2.]) == dict(count=3, mean=1., median=1., p95=1.9, maximum=2.)
    R = np.eye(3);a = checker.pose_errors(np.zeros(3), R, np.zeros(3), R, None)
    assert a == dict(position_m=0., orientation_rad=0., incremental_position_m=None, incremental_orientation_rad=None)
    measured = Rotation.from_rotvec([0., 0., .2]).as_matrix()
    truth = Rotation.from_rotvec([0., 0., .1]).as_matrix()
    b = checker.pose_errors(np.array([.02, 0., 0.]), measured, np.array([.01, 0., 0.]), truth,
        (np.zeros(3), R, np.zeros(3), R))
    assert b['position_m'] == b['incremental_position_m'] == pytest.approx(.01)
    assert b['orientation_rad'] == b['incremental_orientation_rad'] == pytest.approx(.1)


def bind_checker(monkeypatch):
    for name in ('OUTPUT', 'PREDECESSOR', 'COHORTS', 'TRIALS', 'read_npz'):
        monkeypatch.setattr(checker, name, getattr(replay, name))


def test_independent_checker_accepts_real_synthetic_bridge_output(monkeypatch, synthetic):
    monkeypatch.setitem(replay.MODELS, 'temporal_anchor', bridge_factory({2}))
    replay.run();bind_checker(monkeypatch)
    result = replay.read_json(replay.OUTPUT, 'result.json')
    verified = checker.verify_stream('inner', 'synthetic', result)
    assert verified['frames'] == 3 and verified['availability']['both'] == 3
    assert verified['bridge_spans'][0]['frames'] == 1
    assert verified['bridge_spans'][0]['outcome'] == 'END_OF_RECORDING'


@pytest.mark.parametrize('fault', ['row_error', 'clock', 'reference', 'bridge_span', 'bridge_denominator', 'extra_row', 'original'])
def test_independent_saved_row_or_summary_mutations_are_detected(monkeypatch, synthetic, fault):
    monkeypatch.setitem(replay.MODELS, 'temporal_anchor', bridge_factory({2}))
    replay.run();bind_checker(monkeypatch)
    result = deepcopy(replay.read_json(replay.OUTPUT, 'result.json'))
    if fault in ('bridge_span', 'bridge_denominator'):
        if fault == 'bridge_span':result['reports']['inner_synthetic']['continuity']['bridge_spans'][0]['outcome'] = 'ANCHOR_REJOINED'
        else:result['evaluations']['inner_synthetic']['bridged_frame_errors']['position_m']['count'] = 2
    else:
        name = 'inner_synthetic_evaluation.jsonl' if fault in ('row_error', 'clock', 'extra_row') else 'inner_synthetic_estimates.jsonl'
        path = replay.OUTPUT/name;rows = [json.loads(x) for x in path.read_text().splitlines()]
        if fault == 'row_error':rows[2]['errors']['temporal_anchor']['position_m'] += .01
        if fault == 'clock':rows[2]['measured_ns'] += 1
        if fault == 'reference':rows[2]['arms']['temporal_anchor']['pose']['reference_frame'] = 0
        if fault == 'original':rows[1]['arms']['original']['pose']['position_initial_body_m'][0] += .01
        if fault == 'extra_row':rows.append(rows[-1])
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(AssertionError):checker.verify_stream('inner', 'synthetic', result)


def test_missing_completed_result_cannot_read_native(monkeypatch, synthetic):
    bind_checker(monkeypatch)
    def forbidden(*args):raise AssertionError('native coordinates must remain unread')
    monkeypatch.setattr(checker, 'read_npz', forbidden)
    with pytest.raises(ValueError):checker.run('0'*64)


def test_direct_stream_helper_also_requires_a_bound_complete_sensor_phase(monkeypatch, synthetic):
    bind_checker(monkeypatch)
    def forbidden(*args):raise AssertionError('native coordinates must remain unread')
    monkeypatch.setattr(checker, 'read_npz', forbidden)
    result = dict(status='TEMPORAL_ANCHOR_PAIRED_DEVELOPMENT_REPLAY_COMPLETE',
        output_sha256={'all_sensor_estimates_complete.json': '0'*64})
    with pytest.raises(ValueError):checker.verify_stream('inner', 'synthetic', result)
