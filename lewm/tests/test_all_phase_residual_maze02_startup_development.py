from copy import deepcopy
import numpy as np
import pytest
from scripts.all_phase_residual_maze02_startup_development import admit_startup


def fixture():
    trace={'position':np.zeros((950,3)), 'time':np.arange(950,dtype=float)}
    tape=[dict(tick=i,completed=True,requested_command=[0.,0.,0.],
        pre_sample_index=749+50*i,post_sample_index=799+50*i) for i in range(3)]
    rows=[dict(tick=i,observation_index=i,pre_sample_index=749+50*i,
        decision=dict(requested_command=[0.,0.,0.],terminal=None)) for i in range(4)]
    return [[deepcopy(trace),deepcopy(trace)],[deepcopy(tape),deepcopy(tape)],
        [deepcopy(rows),deepcopy(rows)],[['frame0','frame1','frame2','frame3']]*2]


def test_changed_model_command_and_terminal_are_scientific_outcomes():
    args=fixture();args[2][1][3]['decision'].update(requested_command=[.16,0.,.45],terminal='FAILURE')
    report=admit_startup(*args)
    assert report['physical_and_public_startup_exact']
    assert report['candidate_first_model_command']==[.16,0.,.45]
    assert report['candidate_first_model_terminal']=='FAILURE'
    assert report['later_physical_outcomes_compared'] is False


@pytest.mark.parametrize('fault',['physics','short_physics','public','short_public','short_rows',
    'index','warmup_not_completed','warmup_command','warmup_endpoint','decision_command'])
def test_startup_evidence_faults_reject(fault):
    args=fixture()
    if fault=='physics':args[0][1]['position'][899,0]=.001
    if fault=='short_physics':args[0][1]['time']=args[0][1]['time'][:899]
    if fault=='public':args[3][1]=['different']*4
    if fault=='short_public':args[3][1]=['frame0']
    if fault=='short_rows':args[2][1].pop()
    if fault=='index':args[2][1][3]['pre_sample_index']+=1
    if fault=='warmup_not_completed':args[1][1][2]['completed']=False
    if fault=='warmup_command':args[1][1][0]['requested_command']=[.2,0.,0.]
    if fault=='warmup_endpoint':args[1][1][2]['post_sample_index']+=1
    if fault=='decision_command':args[2][1][2]['decision']['requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError):admit_startup(*args)


def test_postintervention_physics_is_never_used_to_claim_startup_difference():
    args=fixture();args[0][1]['position'][900:]=99
    assert admit_startup(*args)['physical_and_public_startup_exact']
