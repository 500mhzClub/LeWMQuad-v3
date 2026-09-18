from copy import deepcopy
import json
import numpy as np
import pytest
from lewm.tests.test_dual_camera_native_preparation_development import prefix
from scripts.maze_decision_stream_development import writer
from scripts.dual_camera_intervention_witness_development import admit_intervention, camera_execution
from scripts import dual_camera_native_prefix_comparison_development as native


def decision(tick, auxiliary=False):
    pose = dict(frame=tick)
    return dict(terminal=None, failure=None, requested_command=[0., 0., .45 if auxiliary else 0.],
        evidence=dict(current_pose=pose), original_visual_evidence=dict(current_pose=pose,
            camera_selection_current=True, terminal_failure=None,
            camera_selection=dict(auxiliary_attempted=auxiliary,
                selected_camera='auxiliary' if auxiliary else 'primary')))


def intervention_fixture():
    report = prefix() | dict(frames=3, exact_primary_decision_frames=2,
        first_auxiliary_intervention_frame=2, final_requested_command=[0., 0., .45])
    rows = [dict(tick=i, decision=decision(i, i==2)) for i in range(3)]
    rows[-1]['first_auxiliary_intervention'] = True
    evidence = dict(frame=2, first_auxiliary_attempt=True, following_recorded_observations_consumed=False,
        original=decision(2), candidate=deepcopy(rows[-1]['decision']))
    return report, rows, evidence


@pytest.mark.parametrize('fault', [None, 'short', 'later', 'earlier', 'command', 'summary', 'stale', 'old_frame'])
def test_saved_intervention_admission_rejects_incomplete_or_different_evidence(tmp_path, fault):
    report, rows, evidence = intervention_fixture()
    if fault=='short': rows.pop()
    if fault=='later': rows.append(dict(tick=3, decision=decision(3)))
    if fault=='earlier': rows[1]['decision']['original_visual_evidence']['camera_selection']['auxiliary_attempted']=True
    if fault=='command': rows[2]['decision']['requested_command']=[.2, 0., 0.]
    if fault=='summary': report['final_requested_command']=[.2, 0., 0.]
    if fault=='stale': evidence['candidate']['evidence']['current_pose']=None
    if fault=='old_frame': evidence['candidate']['evidence']['current_pose']['frame']=1
    with writer(tmp_path) as append:
        for row in rows: append(row)
    (tmp_path/'intervention.json').write_text(json.dumps(evidence))
    if fault is None:
        result = admit_intervention(tmp_path, report)
        assert result['frames']==3 and result['saved_intervention_matches_summary_and_stream']
    else:
        with pytest.raises(ValueError): admit_intervention(tmp_path, report)


def test_camera_readout_does_not_count_latched_failure_snapshots_as_new_attempts():
    rows = [dict(tick=i, decision=decision(i, i in (1, 3, 4))) for i in range(5)]
    for i in (3, 4):
        raw=rows[i]['decision']['original_visual_evidence']
        raw.update(current_pose=None, camera_selection_current=False, terminal_failure=dict(decision_ns=1800000000))
        raw['camera_selection']['selected_camera']=None
    result=camera_execution(rows)
    assert result['auxiliary_attempt_frames']==[1, 3]
    assert result['auxiliary_selected_frames']==[1]
    assert result['primary_selected_frames']==[0, 2]
    assert result['retained_auxiliary_snapshot_frames']==[4]


@pytest.mark.parametrize('fault', [None, 'physics_at_observation', 'prior_command', 'intervention_command',
    'public_packet', 'complete_decision', 'short_stream'])
def test_native_prefix_includes_intervention_observation_but_allows_new_subsequent_physics(tmp_path, monkeypatch, fault):
    report, bound_rows, _ = intervention_fixture()
    paths=[tmp_path/n for n in ('prior', 'current', 'prefix')]
    for p in paths: p.mkdir()
    prior, current, bound=paths
    old_rows=[dict(tick=i, decision=decision(i)) for i in range(3)]
    new_rows=deepcopy(bound_rows)
    rows={prior:old_rows, current:new_rows, bound:bound_rows}
    arrays={p:np.zeros((900, 3)) for p in (prior, current)}
    arrays[current][850:]=10.  # New command may change subsequent state, after sample849.
    if fault=='physics_at_observation': arrays[current][849, 0]=1.
    for p, a in arrays.items(): np.savez(p/'physics_trace.npz', base_pose_world=a)
    tapes={p:[dict(requested_command=deepcopy(r['decision']['requested_command'])) for r in rows[p]]
        for p in (prior, current)}
    if fault=='prior_command': tapes[current][1]['requested_command']=[.2, 0., 0.]
    if fault=='intervention_command': tapes[current][2]['requested_command']=[.2, 0., 0.]
    if fault=='complete_decision': new_rows[2]['decision']['unreviewed_change']=True
    if fault=='short_stream': new_rows.pop()
    monkeypatch.setattr(native, 'read_rows', lambda p:iter(rows[p]))
    monkeypatch.setattr(native, 'read_json', lambda p,n:tapes[p] if n=='command_tape.json' else [{}, {}, {}])
    class Reader:
        def __init__(self, directory): self.directory=directory
        def packet(self, i):
            altered=(fault=='public_packet' and self.directory==current and i==2)
            return dict(frame=i, changed=altered), dict(depth=i), dict(gyro=i), i
    monkeypatch.setattr(native, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(native, 'public_acquisition', lambda r:r)
    monkeypatch.setattr(native, 'packet', lambda directory,i,p,a,now_ns:(dict(rgb=i), dict(auxiliary=i)))
    calls=[]
    def comparison(old, new, *args, **kwargs):
        assert old==new
        calls.append(old)
    monkeypatch.setattr(native, 'compare_json_primary_decision', comparison)
    if fault is None:
        result=native.compare(prior,current,bound,report)
        assert result['common_prefix_frames']==3 and len(calls)==2
        assert result['original_intervention_command']!=result['candidate_intervention_command']
    else:
        with pytest.raises(ValueError): native.compare(prior,current,bound,report)
