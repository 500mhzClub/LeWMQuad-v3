"""Synthetic action, progress, censoring and raw execution contract checks."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lewm.geometry_progress_pilot_development import (ACTIONS, APPEARANCES, GEOMETRIES, TRIALS,
    assignments, specification, pack, geometry, candidate_commands, timed_candidate, schedule,
    decision, progress_outcome, panel_informativeness)
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.tests.test_independent_pulse_context_development import policy
from scripts import audit_go2_geometry_progress_pilot_v1 as audit
from scripts import run_go2_geometry_progress_pilot_v1 as run


def test_balanced_opaque_assignment_and_action_free_scene_constructor():
    cells=assignments()
    assert len(cells)==24 and tuple(cells)==TRIALS
    assert {(x['geometry'],x['appearance_seed'],x['action']) for x in cells.values()}=={
        (g,s,a) for g in GEOMETRIES for s in APPEARANCES for a in ACTIONS}
    for t in TRIALS:
        spec=specification(t);p=pack(spec)
        assert not {'action','action_index','command','goal'} & set(spec)
        assert spec['data_role']=='train' and len(p.static_objects)==5
        assert p.robot.spawn_xyz_m==(0.,0.,.375)
        assert all(a not in t for a in ACTIONS)
    left,right=[geometry(g)['wall_boxes'][0] for g in GEOMETRIES]
    assert left['centre_xyz'][1]==-right['centre_xyz'][1]
    bad=specification(TRIALS[0]);bad['friction_mu']=.8
    with pytest.raises(ValueError):pack(bad)


@pytest.mark.parametrize('action',ACTIONS)
def test_known_four_second_prefix_roundtrips_existing_model_interface(action):
    commands=np.asarray(candidate_commands(action));blocks,valid=timed_candidate(action)
    active,offsets=validate_timed_plan(blocks[None],valid[None],1)
    assert active.all() and offsets.tolist()==[[500_000_000*i for i in range(1,9)]]
    np.testing.assert_allclose((blocks.reshape(40,3)*torch.tensor([.3,1.,.5])).numpy(),commands,atol=1e-7)
    assert np.array_equal(commands[-10:],np.zeros((10,3)))
    assert len(schedule(action))==43 and decision(action,43,policy(43))['terminal']
    assert decision(action,3,policy(3))['requested_command']==commands[0].tolist()


@pytest.mark.parametrize('fault',['clock','privilege','future','identity','tick','action'])
def test_decision_rejects_invalid_or_privileged_packets(fault):
    p=policy(3);tick=3;action='left_arc'
    if fault=='clock':tick=4
    if fault=='privilege':p['native_pose']=[0]*7
    if fault=='future':p['sensor_state']['sensed']['gyro']['available_ns'][-1]+=1
    if fault=='identity':p['sensor_state']['identity']=(0,1,0)
    if fault=='tick':tick=True
    if fault=='action':action='adaptive_arc'
    with pytest.raises(ValueError):decision(action,tick,p)


def outcome(xy,**kw):
    return progress_outcome(xy,**(dict(complete=True,disallowed_contact=False,
        physical_stop=None,acquisition_stop=None)|kw))


def test_progress_requires_actual_distance_reduction_and_complete_contact_free_horizon():
    assert outcome([.25,.10])['successful_progress']
    assert not outcome([0,0])['successful_progress']
    assert not outcome([0,.5])['successful_progress']
    assert not outcome([.3,0],disallowed_contact=True)['successful_progress']
    assert not outcome([.3,0],complete=False,physical_stop='DISALLOWED_CONTACT')['successful_progress']
    assert outcome(None,complete=False,acquisition_stop='MISSING_PACKET')['progress_m'] is None
    with pytest.raises(ValueError):outcome(None)
    with pytest.raises(ValueError):outcome([.3,0],physical_stop='SPEED')


def panel():
    return [dict(**c,outcome=dict(successful_progress=(c['action']==('left_arc' if c['geometry']=='left_open' else 'right_arc'))))
            for c in assignments().values()]


def test_gate_requires_all_strata_reversal_and_rejects_constant_controls_or_subsets():
    rows=panel();assert panel_informativeness(rows)['informative_for_next_dataset']
    for r in rows:
        if r['action']=='forward':r['outcome']['successful_progress']=True
    assert not panel_informativeness(rows)['informative_for_next_dataset']
    rows=panel();next(r for r in rows if r['action']=='hold')['outcome']['successful_progress']=True
    assert not panel_informativeness(rows)['informative_for_next_dataset']
    with pytest.raises(ValueError):panel_informativeness(rows[:-1])
    with pytest.raises(ValueError):panel_informativeness(rows[:-1]+[rows[0]])


def trace(action='left_arc'):
    planned=schedule(action);n=750+50*len(planned)
    raw=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=np.tile([0.,0.,.35,0.,0.,0.,1.],(n,1)),
        requested_command=np.zeros((n,3),np.float32),applied_command=np.zeros((n,3),np.float32),
        phase=np.zeros(n,np.uint8),physics_contact=np.zeros(n,np.uint8))
    tape=[];rows=[]
    for i,r in enumerate(planned):
        lo=749+50*i;hi=lo+50
        raw['requested_command'][lo+1:hi+1]=r['requested_command']
        raw['applied_command'][lo+1:hi+1]=raw['applied_command'][lo]+np.clip(
            np.asarray(r['requested_command'],np.float32)-raw['applied_command'][lo],[-.25,0,-.35],[.25,0,.35])
        raw['phase'][lo+1:hi+1]=r['phase']
        tape.append(r|dict(tick=i,pre_sample_index=lo,post_sample_index=hi,completed=True))
        rows.append(dict(tick=i,decision=r|dict(terminal=False)))
    rows.append(dict(tick=len(planned),decision=dict(terminal=True)))
    result=dict(command_ticks=len(tape),decisions=len(rows),completed_ticks=len(tape),departure_present=True,
        physical_stop=None,acquisition_stop=None,setup_admitted=True,schedule_terminal='FIXED_CONTEXT_PULSE_COMPLETE')
    frames=[dict(physical_sample_index=749+50*i) for i in range(44)]
    return raw,tape,rows,result,frames


def test_raw_command_replay_checks_arc_rate_limit_actual_prefix_and_missing_commands():
    raw,tape,rows,result,_=trace();audit.audit_commands(raw,tape,rows,result,'left_arc')
    assert raw['applied_command'][900,2]==np.float32(.35)
    bad=deepcopy(raw);bad['applied_command'][900,2]=.45
    with pytest.raises(AssertionError):audit.audit_commands(bad,tape,rows,result,'left_arc')
    bad=deepcopy(raw);bad['requested_command'][1700,0]=0
    with pytest.raises(AssertionError):audit.audit_commands(bad,tape,rows,result,'left_arc')


@pytest.mark.parametrize('contact',[False,True])
def test_interrupted_event_targets_keep_all_slots_without_fabricating_future(contact):
    raw,_,_,_,frames=trace();raw={k:v[:1201] for k,v in raw.items()}
    raw['physics_contact'][-1]=int(contact)
    labels=audit.native_horizons(raw,[f for f in frames if f['physical_sample_index']<1201])
    first,*later=labels['targets'];assert first['motion_valid'] and first['future_image_valid']
    assert all(not t['motion_valid'] and not t['future_image_valid'] and t['motion'] is None for t in later)
    assert all(t['contact_valid']==contact and t['contact']==(1. if contact else None) for t in later)


def test_image_missing_does_not_censor_actual_contact_free_native_motion():
    raw,_,_,_,_=trace();labels=audit.native_horizons(raw,[])
    assert all(t['motion_valid'] and t['contact_valid'] and not t['future_image_valid'] for t in labels['targets'])
    assert labels['targets'][-1]['offset_ns']==4_000_000_000


def test_recording_chain_and_native_guard_reuse():
    from scripts.geometry_progress_session_development import GeometryProgressSession
    from scripts.geometry_progress_physical_init_development import GeometryProgressPhysicalInit
    from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession
    from scripts import audit_go2_independent_pulse_context_pilot_v1 as old
    assert GeometryProgressSession.__mro__.index(AttributedSession)<GeometryProgressSession.__mro__.index(GeometryProgressPhysicalInit)
    assert audit.audit_setup is old.audit_setup and audit.audit_stops is old.audit_stops
    assert audit.audit_sensors is old.audit_sensors
    def method(path,name):
        s=Path(path).read_text().replace('action_index','action')
        return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))
    assert method(audit.__file__,'audit_commands')==method(old.__file__,'audit_commands')


def test_unequal_camera_prefix_is_recorded_without_sibling_exclusion():
    rows=[]
    for i,c in enumerate(assignments().values()):
        rows.append(dict(trial=str(i),**c,prefix=dict(complete=True,sha256={'native/pose':'same','packet/0':str(i)})))
    r=audit.compare_prefixes(rows)
    assert len(r)==24 and sum(x['exact_equal'] for x in r)==4
    assert all(not x['used_as_exclusion'] and not x['same_observation_counterfactual_claim'] for x in r)


def test_collection_retains_stops_and_refuses_retry(monkeypatch,tmp_path):
    output=tmp_path/'attempt';calls=[]
    monkeypatch.setattr(run,'OUTPUT',output)
    monkeypatch.setattr(run,'preflight',lambda:dict(source_sha256={run.PROTOCOL:'0'*64}))
    monkeypatch.setattr(run,'create_output',lambda p:p.mkdir())
    monkeypatch.setattr(run,'verify',lambda p:None);monkeypatch.setattr(run,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(run.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(run,'artifacts',lambda c,r:[])
    def collect(c,definition):
        calls.append(c);return dict(physical_stop='DISALLOWED_CONTACT')
    monkeypatch.setattr(run,'collect',collect)
    run.main();assert calls==list(TRIALS)
    with pytest.raises(ValueError,match='exclusive'):run.main()
