"""Synthetic paired-population, sensor/native separation and failure tests."""
from copy import deepcopy
import json
import numpy as np
import pytest
import scripts.navigation_artifact_root_development as authority
import scripts.replay_go2_registration_conditioning_v1 as store_module
import scripts.replay_go2_measured_pose_correspondence_v1 as replay
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
    assert result['status'] == 'MEASURED_POSE_PAIRED_DEVELOPMENT_REPLAY_COMPLETE'
    assert synthetic[2] == ['physics_trace.npz']
    r = result['reports']['inner_synthetic']; e = result['evaluations']['inner_synthetic']
    assert r['original_witness_frames_checked'] == r['frames'] == 3
    assert e['availability'] == dict(both=3, original_only=0, measured_pose_only=0, neither=0)
    for arm in replay.MODELS:
        assert e['arms'][arm]['paired_position_m']['count'] == 3
        assert e['arms'][arm]['orientation_rad']['maximum'] < 1e-8
    assert result['correspondence_checks_changed'] and not result['rigid_registration_gates_changed']
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
    original = replay.MeasuredPoseSeededVisualLedMotion
    class Stops(original):
        def observe(self, *args, **kwargs):
            row = super().observe(*args, **kwargs)
            if self.model.frame >= 1: row['current_pose'] = None
            return row
    monkeypatch.setitem(replay.MODELS, 'measured_pose', Stops)
    replay.run(); r = replay.read_json(replay.OUTPUT, 'result.json'); e = r['evaluations']['inner_synthetic']
    assert e['availability'] == dict(both=1, original_only=2, measured_pose_only=0, neither=0)
    assert e['arms']['original']['position_m']['count'] == 3
    assert e['arms']['original']['paired_position_m']['count'] == 1
    assert e['arms']['measured_pose']['paired_position_m']['count'] == 1
    assert r['reports']['inner_synthetic']['arms']['measured_pose']['first_failure'] == 1


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
    old = dict(status='GYRO_SEEDED_PAIRED_DEVELOPMENT_REPLAY_COMPLETE',
        reports=deepcopy(reports), evaluations={'x': {'synthetic': True}})
    if fault == 'missing': old['evaluations'] = {}
    if fault == 'frames': old['reports']['x']['frames'] = 4
    if fault == 'original_failure': old['reports']['x']['arms']['original']['available'] = 2
    if fault:
        with pytest.raises(ValueError): replay.compare_frozen(reports, old)
    else:
        row = replay.compare_frozen(reports, old)
        assert not row['rotation_only_rerun'] and not row['cross_run_timing_comparison_qualified']
