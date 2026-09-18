"""Boundary checks prevent a diagnostic from claiming an unexecuted future."""
from copy import deepcopy
import pytest
from scripts.replay_go2_contact_anchored_direct_flow_observer_prefix_v1 import compare


def evidence(frame=0):
    return dict(decision_ns=1_500_000_000+frame*100_000_000, status='CURRENT_VISUAL_POSE',
                current_pose={'position': [0., 0., 0.]}, identity=(0,0,0))


def test_original_receipt_mismatch_rejected_even_when_candidate_matches_recording():
    recorded = evidence(); original = deepcopy(recorded); original['current_pose']['position'][0] = .01
    with pytest.raises(ValueError, match='did not reproduce'): compare(recorded, original, recorded, frame=0)


def test_changed_pose_stops_immediately_without_mutating_receipts():
    original = evidence(); candidate = deepcopy(original)
    candidate['current_pose']['position'][0] = .01
    candidate['direct_corner_flow_fallback'] = {'accepted': True}
    before = deepcopy(candidate)
    check = compare(original, original, candidate, frame=0)
    assert check['stop'] and check['stop_reason'] == 'FIRST_CHANGED_OBSERVER_EVIDENCE'
    assert candidate == before


def test_unexplained_change_cannot_be_metadata_normalized():
    original = evidence(); candidate = deepcopy(original); candidate['new_unreviewed_flag'] = True
    with pytest.raises(ValueError, match='unexplained'): compare(original, original, candidate, frame=0)


@pytest.mark.parametrize('which', ['original', 'candidate'])
def test_either_terminal_stops(which):
    original = evidence(); candidate = deepcopy(original)
    target = original if which == 'original' else candidate
    target['status'] = 'VISUAL_TERMINAL_FAILURE'; target['current_pose'] = None
    candidate['direct_corner_flow_fallback'] = {'accepted': False}
    assert compare(original, original, candidate, frame=0)['stop']


def test_json_tuple_normalization_retains_all_other_fields():
    original = evidence(); recorded = deepcopy(original); recorded['identity'] = [0,0,0]
    check = compare(recorded, original, original, frame=0)
    assert check['candidate_original_fields_exact'] and not check['stop']


@pytest.mark.parametrize('which', [0, 1, 2])
def test_each_receipt_clock_is_checked(which):
    values = [evidence() for _ in range(3)]; values[which]['decision_ns'] += 1
    with pytest.raises(ValueError, match='clock'): compare(*values, frame=0)


def test_fixed_end_is_terminal_for_replay_even_with_unchanged_pose():
    value = evidence(561)
    check = compare(value, value, value, frame=561)
    assert check['stop'] and check['stop_reason'] == 'FIXED_PREFIX_LIMIT'


def test_comparator_preserves_original_code_and_only_changes_fixed_limit():
    from scripts import replay_go2_contact_anchored_direct_flow_observer_prefix_v1 as run
    assert run.compare.__code__ is run.original.compare.__code__
    for key, value in run.original.compare.__globals__.items():
        if key != 'MAX_FRAMES': assert run.compare.__globals__[key] is value
    assert run.MAX_FRAMES == 562


@pytest.mark.parametrize('fault', [None, 'mutated_input', 'invalid_current_pose'])
def test_actual_loop_stops_before_any_following_raw_observation(tmp_path, monkeypatch, fault):
    from contextlib import contextmanager
    from scripts import replay_go2_contact_anchored_direct_flow_observer_prefix_v1 as run
    consumed = []; observed = []; saved = []; validations = []
    monkeypatch.setattr(run, 'OUTPUT', tmp_path); (tmp_path/run.NAME).write_bytes(b'')
    def rows(directory):
        for frame in range(5):
            consumed.append(frame)
            value = evidence(frame) | dict(terminal_failure=None)
            yield dict(tick=frame, observation_index=frame,
                decision=dict(original_visual_evidence=value, requested_command=[0.,0.,0.]))
    monkeypatch.setattr(run, 'read_rows', rows)
    monkeypatch.setattr(run, 'read_json', lambda *a:[{}]*5)
    monkeypatch.setattr(run, 'public_acquisition', lambda x:x)
    monkeypatch.setattr(run, 'packet', lambda *a, **k:({},{}))
    monkeypatch.setattr(run, 'fingerprint', lambda x:repr(x))
    class Reader:
        def __init__(self, directory): pass
        def packet(self, frame): return {'frame':frame}, {}, {}, 1_500_000_000+frame*100_000_000
    monkeypatch.setattr(run, 'IntentReturnRGBDReplay', Reader)
    class Baseline:
        def __init__(self, **k): pass
        def observe(self, policy, *a, **k):
            return evidence(policy['frame']) | dict(terminal_failure=None)
    class Candidate(Baseline):
        def observe(self, policy, *a, **k):
            frame = policy['frame']; observed.append(frame); result = super().observe(policy, *a, **k)
            if frame == 2:
                result['direct_corner_flow_fallback'] = {'accepted':True}
                result['current_pose']['position'][0] = .01
                if fault == 'mutated_input': policy['changed'] = True
            return result
    monkeypatch.setattr(run, 'DualCameraVisualMotion', Baseline)
    monkeypatch.setattr(run, 'DirectFlowDualCameraVisualMotion', Candidate)
    def validate(e, *a, **k):
        validations.append(e['decision_ns'])
        if fault == 'invalid_current_pose' and e.get('direct_corner_flow_fallback'):
            raise ValueError('current pose not admissible')
    monkeypatch.setattr(run, 'current_dual_camera_pose', validate)
    @contextmanager
    def writer(root): yield saved.append
    monkeypatch.setattr(run, 'writer', writer)
    if fault:
        with pytest.raises(ValueError): run.replay(tmp_path)
    else:
        report = run.replay(tmp_path)
        assert report['frames'] == 3 and report['candidate_exact_original_frames'] == 2
        assert report['boundary']['stop_reason'] == 'FIRST_CHANGED_OBSERVER_EVIDENCE'
        assert len(saved) == 3 and len(validations) == 6
    assert consumed == observed == [0,1,2]


def test_live_cpu_timeout_does_not_open_result_or_start_observer(monkeypatch):
    from scripts import replay_go2_contact_anchored_direct_flow_observer_prefix_v1 as run
    state = {'ticks':0}; events = []
    monkeypatch.setattr(run, 'owner_live', lambda owner:True)
    monkeypatch.setattr(run, 'WAIT_SECONDS', 30)
    monkeypatch.setattr(run.Path, 'read_text', lambda *a, **k:run.BOOT)
    monkeypatch.setattr(run, 'digest', lambda *a:pytest.fail('no completion read for live process'))
    monkeypatch.setattr(run, 'DualCameraVisualMotion', lambda *a, **k:pytest.fail('no live CPU overlap'))
    def sleep(seconds): assert seconds == 30; state['ticks'] += 1
    with pytest.raises(ValueError, match='expired'):
        run.wait_for_cpu({}, lambda *a, **k:events.append(a), sleep=sleep, clock=lambda:state['ticks']*30)
    assert state['ticks'] == 1 and len(events) == 1


@pytest.mark.parametrize('fault', [None, 'failure', 'source', 'status', 'boundary'])
def test_cpu_slot_requires_original_completed_replay(tmp_path, monkeypatch, fault):
    from scripts import replay_go2_contact_anchored_direct_flow_observer_prefix_v1 as run
    root = tmp_path; spec = run.previous.prefix.LAUNCH_SHA
    result = dict(status='SUSTAINED_HOLD_REORIENTATION_RAW_PREFIX_V1_COMPLETE', source_sha256={'src':'sha'},
        artifact_sha256={'launch.json':spec, 'raw.gz':'raw-sha'},
        report=dict(frames=407, first_changed_command_frame=406, raw_model_forecast_comparisons=404,
            original_requested_command=[0.,0.,0.], candidate_requested_command=[0.,0.,.45]))
    launch = {'source_sha256':{'src':'sha'}}
    if fault == 'failure': (root/'failure.json').write_text('{}')
    if fault == 'source': launch['source_sha256']['src'] = 'changed'
    if fault == 'status': result['status'] = 'RUNNING'
    if fault == 'boundary': result['report']['first_changed_command_frame'] = 405
    monkeypatch.setattr(run.previous.replay, 'OUTPUT', root)
    monkeypatch.setattr(run, 'owner_live', lambda owner:False)
    monkeypatch.setattr(run.Path, 'read_text', lambda *a, **k:run.BOOT)
    monkeypatch.setattr(run, 'digest', lambda *a:'result-sha')
    monkeypatch.setattr(run, 'read_json', lambda root, name:result if name == 'result.json' else launch)
    monkeypatch.setattr(run, 'verify_artifacts', lambda *a:None)
    monkeypatch.setattr(run, 'verify', lambda *a:None)
    if fault:
        with pytest.raises(ValueError): run.wait_for_cpu({'src':'sha'}, lambda *a, **k:None)
    else:
        receipt = run.wait_for_cpu({'src':'sha'}, lambda *a, **k:None)
        assert receipt['original_cpu_owner_ended'] is True
        assert receipt['original_cpu_artifact_sha256'] == result['artifact_sha256'] | {'result.json':'result-sha'}
