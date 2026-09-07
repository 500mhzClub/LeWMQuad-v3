import numpy as np
import pytest

from lewm.moving_prefix_evidence_development import suffix_targets
from lewm.tests.test_moving_prefix_evidence_development import fixture
from scripts.audit_go2_moving_prefix_counterfactual_development_v1 import audit_commands,audit_targets


def target_example():
    raw,frames=fixture()
    row={'branchable':True,'prefix_terminal_sample_index':249,'suffix_terminal_sample_index':1749,
        'suffix_window':suffix_targets(raw,249,1749,frames)}
    return raw,row,frames


def test_independent_six_horizon_target_audit():
    assert audit_targets(*target_example())=={'motion':6,'contact':6,'positive':0}


@pytest.mark.parametrize('fault',['motion','future','mask','contact','release','history'])
def test_independent_target_audit_rejects_corruption(fault):
    raw,row,frames=target_example(); window=row['suffix_window']; t=window['targets'][0]
    if fault=='motion': t['delta_xy_yaw_current_body'][0]+=.1
    if fault=='future': t['future_observation_index']+=1
    if fault=='mask': t['motion_valid']=False
    if fault=='contact': t['contact_by_horizon']=True
    if fault=='release': window['targets'][6].update(in_plan=True,contact_valid=True,contact_by_horizon=False)
    if fault=='history': window['history_observation_indices'][0]+=1
    with pytest.raises(ValueError): audit_targets(raw,row,frames)


def test_contact_absorption_and_noncontact_censoring():
    raw,row,frames=target_example(); row['suffix_terminal_sample_index']=499
    raw['physics_contact'][499]=True; row['suffix_window']=suffix_targets(raw,249,499,frames)
    assert audit_targets(raw,row,frames)=={'motion':0,'contact':6,'positive':6}
    raw['physics_contact'][499]=False; row['suffix_window']=suffix_targets(raw,249,499,frames)
    assert audit_targets(raw,row,frames)=={'motion':1,'contact':1,'positive':0}


def command_example():
    teacher=749; count=3000; times=.002*np.arange(1,count+1)
    raw={'timestamp_s':times,'edge_index':np.zeros(count,dtype=int),'phase':np.zeros(count,dtype=int),
        'requested_command':np.zeros((count,3)),'applied_command':np.zeros((count,3),dtype=np.float32),
        'physics_contact':np.zeros(count,dtype=bool)}
    spec={'prefix_command':[.3,0,0],'future_command':[-.2,0,0]}; tape=[]
    for tick in range(45):
        a=teacher+50*tick; b=a+50
        stage='moving_prefix' if tick<10 else 'suffix' if tick<40 else 'release'
        command=spec['prefix_command'] if tick<10 else spec['future_command'] if tick<40 else [0.,0.,0.]
        raw['edge_index'][a+1:b+1]=1 if tick<10 else 2
        raw['phase'][a+1:b+1]=1 if tick<40 else 2
        raw['requested_command'][a+1:b+1]=command
        prior=raw['applied_command'][a]
        raw['applied_command'][a+1:b+1]=prior+np.clip(np.asarray(command,dtype=np.float32)-prior,[-.25,0,-.35],[.25,0,.35])
        tape.append({'tick':tick,'pre_sample_index':a,'post_sample_index':b,'timestamp_s':times[a],
            'stage':stage,'requested_command':command.copy()})
    row={'teacher_terminal_sample_index':teacher,'prefix_terminal_sample_index':1249,
        'suffix_terminal_sample_index':2749,'teacher_available':True,'branchable':True,'stop_reason':None}
    return raw,row,spec,tape


def test_full_command_sequence_and_native_boundary_stop():
    raw,row,spec,tape=command_example(); audit_commands(raw,row,spec,tape)
    tape=tape[:12]; end=tape[-1]['post_sample_index']; raw={k:v[:end+1].copy() for k,v in raw.items()}
    raw['physics_contact'][-1]=True; row.update(stop_reason='DISALLOWED_CONTACT',suffix_terminal_sample_index=end)
    audit_commands(raw,row,spec,tape)


@pytest.mark.parametrize('fault',['gap','stage','action','slew','phase','release_endpoint','lost_tick'])
def test_command_audit_rejects_corruption(fault):
    raw,row,spec,tape=command_example()
    if fault=='gap': tape[10]['pre_sample_index']+=1
    if fault=='stage': tape[10]['stage']='release'
    if fault=='action': tape[10]['requested_command']=[0.,0.,0.]
    if fault=='slew': raw['applied_command'][1250,0]=-.2
    if fault=='phase': raw['phase'][1250]=2
    if fault=='release_endpoint': row['suffix_terminal_sample_index']=2999
    if fault=='lost_tick': tape=tape[:-1]
    with pytest.raises(ValueError): audit_commands(raw,row,spec,tape)


def test_no_suffix_after_unavailable_moving_prefix():
    raw,row,spec,tape=command_example(); tape=tape[:3]; end=tape[-1]['post_sample_index']
    raw={k:v[:end+1].copy() for k,v in raw.items()}; raw['physics_contact'][-1]=True
    row.update(stop_reason='DISALLOWED_CONTACT',branchable=False,prefix_terminal_sample_index=end,suffix_terminal_sample_index=end)
    audit_commands(raw,row,spec,tape)
    row['suffix_window']=None
    assert audit_targets(raw,row,[])=={'motion':0,'contact':0,'positive':0}
