from copy import deepcopy
import numpy as np
import pytest
from scripts.probe_go2_maze_renderer_witness_v1 import admit_native, CASE
from scripts.novel_maze_round_trip_command_audit_development import audit_commands
from scripts import renderer_witness_maze_probe_comparison_development as comparison


@pytest.mark.parametrize('fault',[None,'partial','raw','prefix','decisions','case'])
def test_probe_requires_completed_current_native_raw_evidence(fault):
    record=dict(case=CASE[0],status='DUAL_CAMERA_SETTLED_MAZE_COLLECTED_AND_RAW_AUDITED',
        prefix_comparison=dict(physical_and_public_prefix_exact=True,complete_candidate_decisions_match_prospective_prefix=True))
    result=dict(status='DUAL_CAMERA_SETTLED_MAZE_PILOT_V1_COMPLETE',conditions=[record])
    report={k:True for k in ('raw_sensor_reconstruction_pass','additional_auxiliary_rgb_reconstructed',
        'raw_model_command_replay_pass','raw_command_audit_pass','model_state_unchanged')}
    if fault=='partial':result['status']='RUNNING'
    if fault=='raw':report['raw_model_command_replay_pass']=False
    if fault=='prefix':record['prefix_comparison']['physical_and_public_prefix_exact']=False
    if fault=='decisions':record['prefix_comparison']['complete_candidate_decisions_match_prospective_prefix']=False
    if fault=='case':record['case']='different'
    if fault is None:admit_native(result,report)
    else:
        with pytest.raises(ValueError):admit_native(result,report)


def test_declared_short_probe_passes_original_raw_command_audit_but_is_not_a_terminal_navigation_run():
    raw={k:np.zeros((900,3),np.float64) for k in ('requested_command','applied_command','post_slew_applied_command')}
    raw.update(timestamp_s=np.arange(900)*.002,phase=np.r_[np.zeros(750,dtype=int),np.ones(150,dtype=int)])
    tape=[dict(tick=i,requested_command=[0.,0.,0.],phase=1,role='causal_history_warmup',
        pre_sample_index=749+50*i,post_sample_index=799+50*i,completed=True) for i in range(3)]
    rows=[dict(tick=i,decision=dict(requested_command=[0.,0.,0.],terminal=None)) for i in range(3)]
    result=dict(command_ticks=3,decisions=3,completed_ticks=3,terminal_zero_ticks=0,
        physical_stop=None,acquisition_stop='DECLARED_THREE_OBSERVATION_PROBE_BOUNDARY',schedule_terminal=None)
    audit_commands(raw,tape,rows,result)
    with pytest.raises(AssertionError):audit_commands(raw,tape,rows,result|dict(acquisition_stop=None))


@pytest.mark.parametrize('fault',[None,'before_boundary','after_boundary','extra_current_physics',
    'public','decision','raw','commands'])
def test_full_three_command_prefix_includes_last_step_and_excludes_old_future(tmp_path,monkeypatch,fault):
    prior,current=[tmp_path/n for n in ('prior','current')]
    for p in (prior,current):p.mkdir()
    arrays={prior:np.zeros((1000,3)),current:np.zeros((901 if fault=='extra_current_physics' else 900,3))}
    if fault=='before_boundary':arrays[current][899,0]=1.
    if fault=='after_boundary':arrays[prior][900:,0]=1.
    for p,a in arrays.items():
        np.savez(p/'physics_trace.npz',base_pose_world=a)
        for i in range(3):
            depth=np.ones((2,2),np.float32)
            if fault=='raw' and p==current and i==1:depth[0,0]=2.
            np.savez(p/f'native_depth_{i:04d}.npz',optical_depth_m=depth)
            np.savez(p/f'auxiliary_depth_{i:04d}.npz',native_optical_depth_m=np.ones((2,2),np.float32),
                diagnostic_segmentation=np.ones((2,2),np.int32))
    rows={p:[dict(tick=i,decision=dict(requested_command=[0.,0.,0.],terminal=None)) for i in range(3)] for p in (prior,current)}
    if fault=='decision':rows[current][1]['decision']['extra_field']=True
    tapes={p:[dict(requested_command=[0.,0.,0.],completed=True) for i in range(3)] for p in (prior,current)}
    if fault=='commands':tapes[current][2]['requested_command']=[.2,0.,0.]
    monkeypatch.setattr(comparison,'read_json',lambda p,n:tapes[p] if n=='command_tape.json' else [{}, {}, {}])
    monkeypatch.setattr(comparison,'read_rows',lambda p:iter(rows[p]))
    class Reader:
        def __init__(self,p):self.path=p
        def packet(self,i):return dict(frame=i,changed=fault=='public' and self.path==current and i==1),{}, {},i
    monkeypatch.setattr(comparison,'IntentReturnRGBDReplay',Reader)
    monkeypatch.setattr(comparison,'public_acquisition',lambda a:a)
    monkeypatch.setattr(comparison,'packet',lambda *args,**kwargs:({},{}))
    monkeypatch.setattr(comparison,'renderer_audit',lambda p:dict(test_scope='separate witness audit tested independently'))
    if fault in (None,'after_boundary'):
        result=comparison.compare(prior,current)
        assert result['frames']==3 and result['physics_samples']==900 and result['actual_zero_commands_exact']
    else:
        with pytest.raises(ValueError):comparison.compare(prior,current)
