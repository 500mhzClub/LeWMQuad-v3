from copy import deepcopy
import numpy as np
import pytest
from scripts import settled_boundary_native_prefix_comparison_development as module
from lewm.tests.test_settled_boundary_prefix_comparison_development import pair as decision_pair


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    old,new,bound=[tmp_path/n for n in ('old','new','prefix')]
    for p in (old,new,bound):p.mkdir()
    for p in (old,new):np.savez(p/'physics_trace.npz',timestamp_s=np.arange(850)*.002)
    report=dict(frames=2,first_mission_behavior_difference=1,final_requested_command=[0.,0.,0.])
    tapes={p:[dict(requested_command=[0.,0.,0.]) for _ in range(2)] for p in (old,new)}
    monkeypatch.setattr(module,'read_json',lambda p,n:tapes[p] if n=='command_tape.json' else [dict(frame=i) for i in range(2)])
    class Reader:
        def __init__(self,directory):pass
        def packet(self,i):return dict(frame=i),{},{},i
    monkeypatch.setattr(module,'IntentReturnRGBDReplay',Reader)
    monkeypatch.setattr(module,'public_acquisition',lambda r:r)
    monkeypatch.setattr(module,'packet',lambda *a,**k:{})
    rows={old:[],new:[]}
    for i in range(2):
        a,b=decision_pair()
        if i==1:
            a['mission_receipt'].update(phase='RETURN',active_goal_initial_body_xy_m=[0.,0.],
                phase_transition='OUTBOUND_TO_RETURN',arrivals=[{'frame':1}],arrival_confirmed_this_frame=True)
            a['goal_initial_body_xy_m']=[0.,0.]
        for p,d in ((old,a),(new,b)):rows[p].append(dict(tick=i,decision=d))
    rows[bound]=deepcopy(rows[new])
    monkeypatch.setattr(module,'read_rows',lambda p:iter(deepcopy(rows[p])))
    return old,new,bound,report,rows,tapes


def test_exact_physics_public_and_decisions_before_mission_change(fixture):
    old,new,bound,report,_,_=fixture
    values=np.arange(850)*.002;values[800:]+=.5
    np.savez(new/'physics_trace.npz',timestamp_s=values)
    r=module.compare(old,new,bound,report)
    assert r['physical_and_public_prefix_exact'] and r['all_prefix_requested_commands_exact']
    assert r['first_changed_command_in_compared_prefix'] is None


@pytest.mark.parametrize('fault',['physics','public','map','pose','forecast','command','last_command',
    'clock','missing','earlier_phase','terminal','prospective'])
def test_mismatch_rejects_even_when_prospective_candidate_is_changed_to_match(fixture,fault,monkeypatch):
    old,new,bound,report,rows,tapes=fixture;d=rows[new][1]['decision']
    if fault=='physics':np.savez(new/'physics_trace.npz',timestamp_s=np.arange(850)*.003)
    elif fault=='public':monkeypatch.setattr(module,'packet',lambda p,*a,**k:str(p))
    elif fault=='map':d['memory_receipt']['cells']+=1
    elif fault=='pose':d['evidence']['pose']+=.01
    elif fault=='forecast':d['new_selection']['prediction'][0]+=.1
    elif fault=='command':tapes[new][0]['requested_command'][0]=.2
    elif fault=='last_command':tapes[new][1]['requested_command'][2]=.45
    elif fault=='clock':rows[new][1]['tick']=3
    elif fault=='missing':rows[bound].pop()
    elif fault=='earlier_phase':rows[old][0]['decision']['mission_receipt']['phase']='RETURN'
    elif fault=='terminal':d['terminal']='stopped'
    else:rows[bound][1]['decision']['evidence']['pose']=.1
    if fault in ('map','pose','forecast','terminal'):rows[bound]=deepcopy(rows[new])
    with pytest.raises(ValueError):module.compare(old,new,bound,report)
