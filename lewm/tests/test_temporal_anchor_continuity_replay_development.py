"""Synthetic paired-population, sensor/native separation and failure tests."""
from copy import deepcopy
import json
import numpy as np
import pytest
import scripts.navigation_artifact_root_development as authority
import scripts.replay_go2_registration_conditioning_v1 as store_module
import scripts.replay_go2_temporal_anchor_continuity_v1 as replay
from lewm.tests.test_rgbd_correspondence_motion_development import texture, packets


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, 'BASE', tmp_path)
    monkeypatch.setattr(store_module, 'BASE', tmp_path)
    monkeypatch.setattr(replay, 'BASE', tmp_path)
    monkeypatch.setattr(replay, 'OUTPUT', tmp_path / 'go2_gyro_test_attempt_001')
    monkeypatch.setattr(replay, 'PREDECESSOR', tmp_path / 'go2_original_test_attempt_001')
    replay.PREDECESSOR.mkdir()
    monkeypatch.setattr(replay, 'COHORTS', {'inner': dict(root=tmp_path / 'input')})
    monkeypatch.setattr(replay, 'TRIALS', ('synthetic',))
    samples = list(packets([texture()] * 3)); model = replay.MultiReferenceVisualLedMotion(); rows = []
    for i, (p, d, f, now) in enumerate(samples):
        r = model.observe(p, d, f, now_ns=now)
        rows.append(dict(frame=i, measured_ns=now, pose=replay.serial(r['current_pose']),
            selection=replay.serial(r['reference_selection']), failure=replay.serial(r['terminal_failure'])))
    witness = replay.PREDECESSOR / 'inner_synthetic_original_estimates.jsonl'
    witness.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    class Reader:
        def __init__(self, path): self.frames = list(range(len(samples)))
        def packet(self, i): return deepcopy(samples[i])
    monkeypatch.setattr(replay, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(replay, 'preflight', lambda: {'synthetic': True})
    monkeypatch.setattr(replay, 'verify', lambda launch: None)
    monkeypatch.setattr(replay, 'frozen_comparison', lambda reports: {'synthetic': True})
    raw = np.zeros((900, 7)); raw[:, 6] = 1.; times = np.arange(1, 901) * .002
    for i, sample in enumerate(samples): times[749 + 50 * i] = sample[-1] / 1e9
    reads = []
    def native(directory, name):
        assert (replay.OUTPUT / 'all_sensor_estimates_complete.json').is_file()
        reads.append(name)
        return dict(base_pose_world=raw.copy(), timestamp_s=times.copy())
    monkeypatch.setattr(replay, 'read_npz', native)
    return samples, witness, reads


def test_complete_paired_stream_precedes_native_and_scores_orientation(synthetic):
    replay.run(); result = replay.read_json(replay.OUTPUT, 'result.json')
    assert result['status'] == 'TEMPORAL_ANCHOR_PAIRED_DEVELOPMENT_REPLAY_COMPLETE'
    assert synthetic[2] == ['physics_trace.npz']
    r = result['reports']['inner_synthetic']; e = result['evaluations']['inner_synthetic']
    assert r['original_witness_frames_checked'] == r['frames'] == 3
    assert e['availability'] == dict(both=3, original_only=0, temporal_anchor_only=0, neither=0)
    for arm in replay.MODELS:
        assert e['arms'][arm]['paired_position_m']['count'] == 3
        assert e['arms'][arm]['orientation_rad']['maximum'] < 1e-8
    assert not result['correspondence_checks_changed'] and not result['rigid_registration_gates_changed']
    assert not result['physical_recovery'] and not result['independent_validation']
    authority.verify_artifacts(replay.OUTPUT, result['output_sha256'])


def test_original_witness_mismatch_stops_before_native_and_keeps_partial_rows(synthetic):
    _, witness, reads = synthetic
    rows = [json.loads(x) for x in witness.read_text().splitlines()]
    rows[1]['pose']['position_initial_body_m'][0] = 9.
    witness.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='exact original'): replay.run()
    assert not reads and not (replay.OUTPUT / 'result.json').exists()
    assert len((replay.OUTPUT / 'inner_synthetic_estimates.jsonl').read_text().splitlines()) == 1
    assert not replay.read_json(replay.OUTPUT, 'failure.json')['retry_performed']


def test_candidate_missingness_keeps_original_and_common_denominators(monkeypatch, synthetic):
    original = replay.TemporalAnchorVisualLedMotion
    class Stops(original):
        def observe(self, *args, **kwargs):
            row = super().observe(*args, **kwargs)
            if self.model.frame >= 1: row['current_pose'] = None
            return row
    monkeypatch.setitem(replay.MODELS, 'temporal_anchor', Stops)
    replay.run(); r = replay.read_json(replay.OUTPUT, 'result.json'); e = r['evaluations']['inner_synthetic']
    assert e['availability'] == dict(both=1, original_only=2, temporal_anchor_only=0, neither=0)
    assert e['arms']['original']['position_m']['count'] == 3
    assert e['arms']['original']['paired_position_m']['count'] == 1
    assert e['arms']['temporal_anchor']['paired_position_m']['count'] == 1
    assert r['reports']['inner_synthetic']['arms']['temporal_anchor']['first_failure'] == 1


@pytest.mark.parametrize('fault', ['native_clock', 'native_exception', 'changed_estimates', 'incomplete_phase', 'source', 'budget'])
def test_faults_retain_failure_without_claiming_completion(monkeypatch, synthetic, fault):
    if fault in ('native_clock', 'native_exception'):
        native = replay.read_npz
        def broken(*a):
            raw = native(*a)
            if fault == 'native_exception': raise ValueError('synthetic missing native data')
            raw['timestamp_s'][799] += .002
            return raw
        monkeypatch.setattr(replay, 'read_npz', broken)
    elif fault in ('changed_estimates', 'incomplete_phase'):
        evaluate = replay.evaluate_trial
        def broken(c, t, report, store):
            if fault == 'changed_estimates':
                path = store.output / report['estimates_file']; path.write_text(path.read_text() + '\n')
            else:
                marker = 'all_sensor_estimates_complete.json'
                phase = replay.read_json(store.output, marker); phase['reports'] = {}
                path = store.output / marker; path.write_text(json.dumps(phase)); store.hashes[marker] = replay.digest(path)
            return evaluate(c, t, report, store)
        monkeypatch.setattr(replay, 'evaluate_trial', broken)
    elif fault == 'source':
        monkeypatch.setattr(replay, 'verify', lambda x: (_ for _ in ()).throw(ValueError('synthetic source change')))
    else: monkeypatch.setattr(store_module, 'BUDGET', 30)
    with pytest.raises(ValueError): replay.run()
    assert (replay.OUTPUT / 'failure.json').exists() and not (replay.OUTPUT / 'result.json').exists()
    if fault not in ('native_clock', 'native_exception'): assert not synthetic[2]


def test_evaluator_without_sensor_marker_cannot_read_truth(synthetic):
    replay.create_output(replay.OUTPUT); store = replay.Store(replay.OUTPUT)
    with pytest.raises(ValueError, match='sensor phase'):
        replay.evaluate_trial('inner', 'synthetic', {}, store)
    assert not synthetic[2]


@pytest.mark.parametrize('extra', [False, True])
def test_witness_population_cannot_be_shortened_or_extended(synthetic, extra):
    _, witness, reads = synthetic; rows = witness.read_text().splitlines()
    witness.write_text('\n'.join(rows + [rows[-1]] if extra else rows[:-1]) + '\n')
    with pytest.raises(ValueError, match='witness'): replay.run()
    assert not reads and not (replay.OUTPUT / 'result.json').exists()

@pytest.mark.parametrize('fault', [None, 'missing', 'frames', 'original_failure'])
def test_frozen_comparison_requires_the_same_complete_original_population(fault):
    original = dict(frames=3, available=3, first_failure=None)
    reports = {'x': dict(frames=3, arms={'original': original})}
    old = dict(status='MEASURED_POSE_PAIRED_DEVELOPMENT_REPLAY_COMPLETE',
        reports=deepcopy(reports), evaluations={'x': {'synthetic': True}})
    if fault == 'missing': old['evaluations'] = {}
    if fault == 'frames': old['reports']['x']['frames'] = 4
    if fault == 'original_failure': old['reports']['x']['arms']['original']['available'] = 2
    if fault:
        with pytest.raises(ValueError): replay.compare_frozen(reports, old)
    else:
        row = replay.compare_frozen(reports, old)
        assert not row['measured_pose_baseline_rerun'] and not row['cross_run_timing_comparison_qualified']


def bridge_factory(missing):
    class MissingAnchor(replay.TemporalAnchorVisualLedMotion):
        def __init__(self):
            super().__init__(); candidate = self.model._candidate
            def injected(ref, current, G):
                if self.model.frame in missing and ref.frame != self.model.previous.frame:
                    from lewm.causal_sensor_state import SensorContractError
                    raise SensorContractError('synthetic retained-anchor rejection')
                return candidate(ref, current, G)
            self.model._candidate = injected
    return MissingAnchor


def actual_rows(count, missing):
    model = bridge_factory(missing)(); rows = []
    for i, (p, d, f, now) in enumerate(packets([texture()] * count)):
        r = model.observe(p, d, f, now_ns=now)
        rows.append(dict(frame=i, measured_ns=now, arms={'temporal_anchor': dict(
            pose=replay.serial(r['current_pose']), selection=replay.serial(r['reference_selection']),
            continuity=replay.serial(r['continuity_evidence']))}))
    return rows


@pytest.mark.parametrize('ending', ['rejoin', 'recording_end', 'terminal'])
def test_complete_bridge_spans_distinguish_rejoin_truncation_and_failure(ending):
    count = {'rejoin': 5, 'recording_end': 4, 'terminal': 14}[ending]
    missing = {2, 3} if ending == 'rejoin' else set(range(2, count))
    audit = replay.ContinuityAudit()
    for row in actual_rows(count, missing):audit.observe(row)
    summary = audit.summary();span, = summary['bridge_spans']
    assert span['start_frame'] == 2
    assert span['frames'] == (10 if ending == 'terminal' else 2)
    assert span['outcome'] == {'rejoin': 'ANCHOR_REJOINED', 'recording_end': 'END_OF_RECORDING',
        'terminal': 'TERMINAL_FAILURE'}[ending]
    assert summary['unavailable_frames'] == (2 if ending == 'terminal' else 0)
    if ending == 'rejoin':assert span['rejoin_increment_available'] and span['following_frame'] == 4
    if ending == 'terminal':assert summary['first_failure']['frame'] == 12
    assert audit.summary() == summary  # Summary calls cannot close/mutate a live span.


@pytest.mark.parametrize('fault', ['clock', 'frame', 'prior', 'promotion', 'reference', 'selection',
    'count', 'total', 'path', 'position', 'reset', 'mode'])
def test_sensor_bridge_provenance_faults_fail_closed(fault):
    rows = actual_rows(3, {2}); audit = replay.ContinuityAudit()
    for row in rows[:2]:audit.observe(row)
    row = deepcopy(rows[2]);arm = row['arms']['temporal_anchor'];e = arm['continuity'];p = arm['pose']
    if fault == 'clock':row['measured_ns'] += 1
    if fault == 'frame':row['frame'] += 1
    if fault == 'prior':e['previous_measured_ns'] -= 100_000_000
    if fault == 'promotion':p['promoted_keyframe'] = True
    if fault == 'reference':p['reference_frame'] = 0
    if fault == 'selection':arm['selection']['selected_reference_retained_anchor'] = True
    if fault == 'count':e['bridge_frames'] = 11
    if fault == 'total':e['total_bridge_frames'] = 2
    if fault == 'path':e['bridge_path_m'] += .01
    if fault == 'position':e['incremental_position_initial_body_m'][0] += .01
    if fault == 'reset':p['global_history_reset'] = True
    if fault == 'mode':e['status'] = 'UNKNOWN'
    with pytest.raises((ValueError, AssertionError)):audit.observe(row)


def test_rejoin_cannot_erase_bridge_span_or_resume_after_terminal():
    rows = actual_rows(5, {2, 3});audit = replay.ContinuityAudit()
    for row in rows[:4]:audit.observe(row)
    bad = deepcopy(rows[4]);bad['arms']['temporal_anchor']['continuity']['preceding_bridge_frames'] = 0
    with pytest.raises(ValueError, match='history'):audit.observe(bad)
    audit = replay.ContinuityAudit();audit.observe(rows[0])
    missing = deepcopy(rows[1]);missing['arms']['temporal_anchor']['pose'] = None;audit.observe(missing)
    with pytest.raises(ValueError, match='terminal'):audit.observe(rows[2])


def test_native_bridge_errors_and_increment_denominators_are_separate(monkeypatch, synthetic):
    monkeypatch.setitem(replay.MODELS, 'temporal_anchor', bridge_factory({2}))
    native = replay.read_npz
    def shifted(*args):
        raw = native(*args)
        raw['base_pose_world'][849, 0] = .01
        raw['base_pose_world'][849, 3:] = [0., 0., np.sin(.01), np.cos(.01)]
        return raw
    monkeypatch.setattr(replay, 'read_npz', shifted)
    replay.run();r = replay.read_json(replay.OUTPUT, 'result.json');e = r['evaluations']['inner_synthetic']
    assert e['continuity'] == r['reports']['inner_synthetic']['continuity']
    assert e['continuity']['total_bridged_frames'] == 1
    for arm in replay.MODELS:
        assert e['arms'][arm]['position_m']['count'] == 3
        assert e['arms'][arm]['incremental_position_m']['count'] == 2
    for name, value in [('position_m', .01), ('incremental_position_m', .01),
                        ('orientation_rad', .02), ('incremental_orientation_rad', .02)]:
        assert e['bridged_frame_errors'][name]['count'] == 1
        assert e['bridged_frame_errors'][name]['mean'] == pytest.approx(value, abs=1e-10)


def test_corrupt_bridge_sensor_summary_cannot_be_published_as_evaluation(monkeypatch, synthetic):
    evaluate = replay.evaluate_trial
    def corrupt(c, t, report, store):
        bad = deepcopy(report);bad['continuity']['total_bridged_frames'] = 99
        return evaluate(c, t, bad, store)
    monkeypatch.setattr(replay, 'evaluate_trial', corrupt)
    with pytest.raises(ValueError, match='bound sensor report'):replay.run()
    assert not synthetic[2] and not (replay.OUTPUT / 'result.json').exists()
