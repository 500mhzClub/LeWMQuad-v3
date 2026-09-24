"""Full replay admission and fresh physics prefix, excluding changed-command outcomes."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.partial_floor_height_prefix_development import compare_step
from lewm.tests.test_partial_floor_height_prefix_development import rows as decisions
from scripts import partial_floor_height_maze01_native_prefix_development as prefix


def evidence():
    saved = []; original = []
    for i in range(prefix.FRAMES):
        old, new = decisions(i)
        if i == prefix.CHANGED: new['evidence']['partial_floor_height'] = {'fixture':True}
        check = compare_step(old, new, old['requested_command'], frame=i)
        saved.append(dict(tick=i, decision=new, comparison=check,
            original_requested_command=old['requested_command'], public_input_arrays_unchanged=True))
        original.append(dict(tick=i, observation_index=i, pre_sample_index=749+50*i, decision=old))
    report = dict(case=prefix.CASE[0], layout_index=1, frames=505, maximum_frames=505, boundary_frame=504,
        exact_original_decisions=504, raw_model_forecast_comparisons=501,
        original_actual_commands_before_intervention_exact=True, prior_commands_compared=504,
        final_requested_command=[0., 0., .45], final_terminal=None, final_failure=None,
        full_controller_recovered_at_boundary=True, stopped_at_original_failed_observation=True,
        following_recorded_observations_consumed=False, public_input_arrays_unchanged=True,
        raw_visual_tracker_unchanged=True, model_state_sha256=MODEL_STATE, model_state_unchanged=True,
        unexecuted_outcomes_inferred=False, native_execution=False, navigation_verified=False,
        boundary_comparison=deepcopy(saved[-1]['comparison']),
        partial_height_receipt=deepcopy(saved[-1]['decision']['evidence']['partial_floor_height']))
    return dict(status='PARTIAL_FLOOR_HEIGHT_PREFIX_V1_COMPLETE', model_loaded=True,
        model_training=False, native_execution=False, shadow_replay_only=True, report=report), saved, original


@pytest.mark.parametrize('fault', [None, 'short', 'extra', 'model', 'following', 'recovery',
    'prior_request', 'state', 'forecast_count', 'height', 'early_recovery', 'failed', 'raw_tracker', 'hold'])
def test_completed_changed_replay_required_before_native_execution(monkeypatch, fault):
    result, saved, original = evidence()
    if fault == 'short': saved.pop()
    elif fault == 'extra': saved.append(deepcopy(saved[-1]))
    elif fault == 'model': result['report']['model_state_sha256'] = '0'*64
    elif fault == 'following': result['report']['following_recorded_observations_consumed'] = True
    elif fault == 'recovery': result['report']['full_controller_recovered_at_boundary'] = False
    elif fault == 'prior_request': saved[5]['original_requested_command'] = [1., 0., 0.]
    elif fault == 'state': saved[3]['decision']['mission_receipt']['changed'] = True
    elif fault == 'forecast_count': saved[3]['comparison']['raw_model_forecasts_compared'] = False
    elif fault == 'height': result['report']['partial_height_receipt'] = {}
    elif fault == 'early_recovery': saved[503]['comparison']['controller_recovered'] = True
    elif fault == 'failed': result['status'] = 'TERMINAL_PARTIAL_FLOOR_HEIGHT_PREFIX_FAILURE'
    elif fault == 'raw_tracker': saved[504]['decision']['original_visual_evidence']['changed'] = True
    elif fault == 'hold':
        d = saved[-1]['decision']; d['new_selection']['action'] = 'hold'; d['requested_command'] = [0., 0., 0.]
        saved[-1]['comparison'] = compare_step(original[-1]['decision'], d, [0., 0., 0.], frame=504)
        result['report'].update(final_requested_command=[0., 0., 0.], boundary_comparison=saved[-1]['comparison'])
    monkeypatch.setattr(prefix, 'read_rows', lambda p:iter(saved if p is None else original))
    if fault is None:
        assert prefix.admit_prefix(None, result) == result['report']|dict(prior_requested_command=[0., 0., 0.])
    else:
        with pytest.raises(ValueError): prefix.admit_prefix(None, result)


def native_fixture(monkeypatch, tmp_path):
    result, bound, original = evidence(); old = tmp_path/'old'; new = tmp_path/'new'; saved = tmp_path/'saved'
    data = dict(timestamp_s=np.arange(26000)*.002, base_pose_world=np.zeros((26000, 7)))
    for p in (old, new, saved): p.mkdir()
    for p in (old, new): np.savez(p/'physics_trace.npz', **data)
    newrows = [deepcopy(r)|dict(observation_index=i, pre_sample_index=749+50*i) for i,r in enumerate(bound)]
    rows = {old:original, new:newrows, saved:bound}
    tapes = {p:[dict(requested_command=r['decision']['requested_command'], completed=True) for r in rows[p]] for p in (old, new)}
    reads = []; packets = []; changes = {}
    def read_rows(p):
        for i,r in enumerate(rows[p]): reads.append((p,i)); yield r
        pytest.fail('read beyond505-observation intervention')
    monkeypatch.setattr(prefix, 'read_rows', read_rows)
    monkeypatch.setattr(prefix, 'read_json', lambda p,n:tapes[p] if n == 'command_tape.json' else [{}]*505)
    def reader(p):
        def packet(i):
            packets.append((p,i))
            return {'frame':i, 'value':changes.get((p,i),0)}, {}, {}, 1_500_000_000+i*100_000_000
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix, 'IntentReturnRGBDReplay', reader)
    monkeypatch.setattr(prefix, 'public_acquisition', lambda r:r)
    monkeypatch.setattr(prefix, 'packet', lambda *a, **k:({}, {}))
    return result['report']|dict(prior_requested_command=[0., 0., 0.]), old, new, saved, rows, tapes, reads, packets, changes, data


@pytest.mark.parametrize('fault', [None, 'physical', 'public', 'original_failure', 'decision', 'prior_command',
    'new_command', 'short', 'new_future', 'partial_new_command'])
def test_exact_native_past_and_no_borrowed_intervention_outcomes(monkeypatch, tmp_path, fault):
    report, old, new, saved, rows, tapes, reads, packets, changes, data = native_fixture(monkeypatch, tmp_path)
    if fault == 'physical': data['base_pose_world'][25949,0] = .1; np.savez(new/'physics_trace.npz', **data)
    elif fault == 'new_future': data['base_pose_world'][25950:,0] = 100.; np.savez(new/'physics_trace.npz', **data)
    elif fault == 'public': changes[(new,504)] = 1
    elif fault == 'original_failure': rows[old][504]['decision']['failure'] = 'unrelated'
    elif fault == 'decision': rows[new][504]['decision']['new_selection']['prediction'] = [[2.]]
    elif fault == 'prior_command': tapes[new][503]['requested_command'] = [.2,0.,0.]
    elif fault == 'new_command': tapes[new][504]['requested_command'] = [0.,0.,0.]
    elif fault == 'short': np.savez(new/'physics_trace.npz', **{k:v[:25949] for k,v in data.items()})
    elif fault == 'partial_new_command': tapes[new][504]['completed'] = False
    if fault in (None, 'new_future', 'partial_new_command'):
        r = prefix.compare(old, new, saved, report)
        assert r['physical_prefix_samples'] == 25950 and r['common_prefix_frames'] == 505
        assert r['raw_model_forecast_comparisons'] == 501 and r['complete_candidate_decisions_match_prospective_prefix']
        assert r['candidate_intervention_command_completed'] == (fault != 'partial_new_command')
        assert len(reads) == 1515 and len(packets) == 1010 and not r['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError): prefix.compare(old, new, saved, report)
