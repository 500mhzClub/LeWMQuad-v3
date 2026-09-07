"""Coverage is tested independently of rendered success and training outcomes."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.independent_layout_collection_development import schedule
from lewm.pulse_timed_observation_pairing_development import pulse_window
from lewm.recorded_pulse_native_targets_development import RecordedPulseNativeTargets
from lewm.terminal_event_coverage_development import classify_coverage, population_coverage, CONTACT_FIELDS
from lewm.tests.test_independent_pulse_context_development import trace_fixture
from scripts.audit_go2_independent_pulse_context_pilot_v1 import target_row


def fixture(kind='schedule', stop_sample=1648):
    raw, tape, rows, result = trace_fixture(1)
    planned = schedule(1, 'recent_forward')
    raw['requested_command'] = raw['requested_command'].astype(np.float64)
    for i, expected in enumerate(planned):
        tape[i].update(expected)
        raw['requested_command'][750 + i * 50:800 + i * 50] = expected['requested_command']
    stop = None
    if kind != 'schedule':
        n = stop_sample + 1
        raw = {k: v[:n].copy() for k, v in raw.items()}
        tape = [t for t in tape if t['pre_sample_index'] < stop_sample]
        if tape:
            tape[-1]['post_sample_index'] = stop_sample
            tape[-1]['completed'] = kind == 'infrastructure'
        stop = ('DISALLOWED_CONTACT' if kind == 'contact' else 'BODY_STABILITY_LIMIT'
                if kind == 'stability' else None)
        if kind == 'contact':
            raw['physics_contact'][-1] = 1
        result.update(physical_stop=stop, acquisition_stop='STORAGE_RESERVE_STOP' if kind == 'infrastructure' else None,
                      schedule_terminal=None)
    n = len(raw['timestamp_s'])
    result.update(physics_samples=n, setup_checked=True, command_ticks=len(tape),
                  completed_ticks=sum(t['completed'] for t in tape), departure_present=len(tape) > 8)
    count = len(tape) + int(kind == 'schedule')
    if kind == 'setup':
        result.update(setup_admitted=False, physical_stop='FRESH_MISSION_INITIAL_SETUP_REJECTED')
        count = 1
    frames = [dict(decision_ns=1_500_000_000 + i * 100_000_000,
                   image_ns=1_500_000_000 + i * 100_000_000) for i in range(count)]
    cameras = [dict(physical_sample_index=749 + 50 * i,
                    timestamp_s=float(raw['timestamp_s'][749 + 50 * i])) for i in range(count)]
    result['rgbd_frames'] = count
    contacts = {k: np.zeros((n, 3) if k in ('force_a', 'force_b', 'position') else (n,)) for k in CONTACT_FIELDS}
    contacts.update(frame_offsets=np.arange(n + 1), frame_timestamp_s=raw['timestamp_s'].copy())
    events = [dict(sample_index=i, timestamp_s=float(raw['timestamp_s'][i]), phase=int(raw['phase'][i]),
                   disallowed_contacts=[{'synthetic': True}] if raw['physics_contact'][i] else []) for i in range(n)]
    report = {k: result[k] for k in ('physics_samples', 'setup_checked', 'setup_admitted', 'departure_present', 'acquisition_stop')}
    report.update(recorded_sensor_reconstruction_pass=True, paired_frames=count,
                  schedule_complete=kind == 'schedule', physical_stop_message=result['physical_stop'],
                  physical_stop=dict(sample_index=n - 1, reason=stop) if stop else None,
                  physical_visibility_pass=False)
    prefix = dict(status='COMPLETE_PREFIX', native_samples=1150, frames=9, sha256={'synthetic': 'not-a-raw-audit'})
    spec = dict(action_index=1, history_kind='recent_forward', command=[.2, 0., 0.], pulse_ticks=5)
    window = labels = None
    if result['departure_present']:
        window = dict(condition='synthetic', action_index=1) | pulse_window(frames, tape,
            departure_tick=8, departure_ns=2_300_000_000, command=(.2, 0., 0.), pulse_ticks=5)
        labels = target_row(window, RecordedPulseNativeTargets(raw).labels(window))
    return dict(spec=spec, raw=raw, contacts=contacts, events=events, cameras=cameras, tape=tape,
                result=result, report=report, prefix=prefix, window=window, labels=labels)


@pytest.mark.parametrize('kind,sample,classification', [
    ('schedule', 1648, 'SCHEDULE_RECORDED'), ('contact', 1648, 'PHYSICAL_TERMINAL_RECORDED'),
    ('contact', 1649, 'PHYSICAL_TERMINAL_RECORDED'), ('stability', 1648, 'PHYSICAL_TERMINAL_RECORDED'),
    ('stability', 999, 'PRE_DEPARTURE_PHYSICAL_TERMINAL'), ('setup', 749, 'SETUP_FAILED'),
    ('infrastructure', 1649, 'INFRASTRUCTURE_TRUNCATED')])
def test_separate_recording_completion_action_observation_and_task_success(kind, sample, classification):
    data = fixture(kind, sample)
    result = classify_coverage(**data)
    assert result['classification'] == classification
    assert result['command_schedule_complete'] == (kind == 'schedule')
    assert result['candidate_acquisition_complete'] == (classification in ('SCHEDULE_RECORDED', 'PHYSICAL_TERMINAL_RECORDED'))
    assert result['strict_visibility_pass'] is False  # Accounting never rescues failed sensing.
    assert result['training_eligibility_granted'] is result['navigation_qualified'] is False


@pytest.mark.parametrize('fault', ['missing_terminal', 'contact_offsets', 'contact_field', 'contact_clock',
    'contact_attribution', 'missing_capture', 'extra_capture', 'camera_clock', 'changed_request',
    'changed_tape', 'post_stop', 'terminal_index', 'wrong_reason', 'multiple_causes', 'no_prefix',
    'no_labels', 'censored_negative', 'censored_motion', 'future_image', 'hidden_endpoint', 'wrong_contact_time'])
def test_corruption_is_not_a_recorded_terminal_event(fault):
    d = fixture('contact')
    if fault == 'missing_terminal': d['raw'] = {k: v[:-1] for k, v in d['raw'].items()}
    elif fault == 'contact_offsets': d['contacts']['frame_offsets'] = d['contacts']['frame_offsets'][:-1]
    elif fault == 'contact_field': d['contacts']['force_a'] = d['contacts']['force_a'][:-1]
    elif fault == 'contact_clock': d['contacts']['frame_timestamp_s'][-1] += .002
    elif fault == 'contact_attribution': d['events'][-1]['disallowed_contacts'] = []
    elif fault == 'missing_capture': d['cameras'].pop(9)
    elif fault == 'extra_capture': d['cameras'].append(dict(physical_sample_index=1648, timestamp_s=3.298))
    elif fault == 'camera_clock': d['cameras'][-1]['timestamp_s'] += .002
    elif fault == 'changed_request': d['raw']['requested_command'][1150, 0] += 1e-9
    elif fault == 'changed_tape': d['tape'][8]['requested_command'] = [0., 0., 0.]
    elif fault == 'post_stop':
        d['raw']['physics_contact'][-2] = 1
        d['events'][-2]['disallowed_contacts'] = [{'synthetic': True}]
    elif fault == 'terminal_index': d['report']['physical_stop']['sample_index'] -= 1
    elif fault == 'wrong_reason': d['result']['physical_stop'] = 'BODY_STABILITY_LIMIT'
    elif fault == 'multiple_causes': d['result']['acquisition_stop'] = 'STORAGE_RESERVE_STOP'
    elif fault == 'no_prefix': d['prefix']['status'] = 'MISSING_DEPARTURE_PREFIX'
    elif fault == 'no_labels': d['labels'] = None
    elif fault == 'censored_negative': d['labels']['targets'][5]['contact'] = 0.
    elif fault == 'censored_motion': d['labels']['targets'][5]['motion'] = [0., 0., 0.]
    elif fault == 'future_image': d['window']['targets'][5]['future_valid'] = True
    elif fault == 'hidden_endpoint': d['window']['targets'][0]['command_prefix_executed'] = False
    elif fault == 'wrong_contact_time': d['labels']['accounting']['first_matched_contact_ns'] += 2_000_000
    with pytest.raises(ValueError): classify_coverage(**d)


def test_noncontact_stop_censors_safety_instead_of_creating_negative_labels():
    d = fixture('stability')
    assert classify_coverage(**d)['contact_event_count'] == 0
    assert d['labels']['targets'][5]['contact'] is None
    d['labels']['targets'][5].update(contact_valid=True, contact=0.)
    with pytest.raises(ValueError): classify_coverage(**d)


def test_population_keeps_missing_failed_and_collision_cases_in_full_denominator():
    evidence = {k: classify_coverage(**fixture(kind)) for k, kind in [('a', 'schedule'), ('b', 'contact')]}
    result = population_coverage(['a', 'b', 'c', 'd', 'e'], evidence, invalid=['c'], infrastructure=['d'])
    assert result['planned_cases'] == 5 and result['candidate_acquisition_complete'] == 2
    assert result['command_schedule_complete'] == 1 and result['counts']['UNATTEMPTED'] == 1
    for kwargs in [dict(invalid=['b']), dict(invalid=['x']), dict(invalid=['c', 'c']), dict(invalid=['c'], infrastructure=['c'])]:
        with pytest.raises(ValueError): population_coverage(['a', 'b', 'c'], evidence, **kwargs)
    with pytest.raises(ValueError): population_coverage(['a', 'a'], {})


def test_wrapper_cannot_bypass_full_raw_auditor(monkeypatch):
    import scripts.terminal_event_collection_audit_development as mod
    calls = []
    def fail(*args):
        calls.append('raw'); raise ValueError('missing native contact attribution')
    monkeypatch.setattr(mod, 'audit_condition', fail)
    monkeypatch.setattr(mod, 'classify_coverage', lambda *args: calls.append('coverage'))
    with pytest.raises(ValueError): mod.audit_terminal_event_condition(None, {}, {}, 'definition')
    assert calls == ['raw']
