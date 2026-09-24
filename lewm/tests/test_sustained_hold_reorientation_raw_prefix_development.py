"""Negative wiring tests; actual raw-controller reconstruction remains required."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.sustained_hold_reorientation_prefix_development import compare_step,identity
from scripts import replay_go2_sustained_hold_reorientation_maze02_prefix_v1 as runner


def pair(frame):
    selection=None if frame<3 else dict(action='hold',action_index=0,requested_command=[0.,0.,0.],
        prediction=['same forecast'],surface_checks=['same evidence'],utility_m=.1)
    old=dict(tick=frame,controller='hold_reorientation_controller_v1',hold_reorientation_enabled=True,
        terminal=None,failure=None,new_selection=selection,requested_command=[0.,0.,0.],
        selected_action=None if frame<3 else 'hold',evidence=dict(observed_xy=[1.,2.]))
    if frame==405:
        old['selected_action']='left_turn';old['requested_command']=[0.,0.,.45]
        old['new_selection'].update(action='left_turn',action_index=4,requested_command=[0.,0.,.45],
            hold_reorientation=dict(frame=frame))
    new=deepcopy(old);new.update(controller='sustained_hold_reorientation_controller_v1',sustained_hold_reorientation_enabled=True)
    if frame in (405,406):
        new['new_selection']['sustained_hold_reorientation']=dict(frame=frame,
            measured_ns=1_500_000_000+frame*100_000_000,starting=frame==405)
    if frame==406:
        new['selected_action']='left_turn';new['requested_command']=[0.,0.,.45]
        new['new_selection'].update(action='left_turn',action_index=4,requested_command=[0.,0.,.45])
    return old,new


@pytest.mark.parametrize('frame',[0,3,404,405,406])
def test_complete_comparison_allows_only_declared_metadata_and_final_request(frame):
    old,new=pair(frame)
    r=compare_step(old,new,old['requested_command'],frame=frame,expected_selection_sha256=identity(new['new_selection']))
    assert r['requested_command_changed'] is (frame==406)
    assert r['normalized_complete_decision_exact']


@pytest.mark.parametrize('fault',['forecast','surface','pose','early_command','missing_receipt','stale_receipt','expected_sha'])
def test_comparator_rejects_evidence_or_boundary_changes(fault):
    old,new=pair(406);expected=identity(new['new_selection'])
    if fault=='forecast':new['new_selection']['prediction']=['changed']
    if fault=='surface':new['new_selection']['surface_checks']=['changed']
    if fault=='pose':new['evidence']['observed_xy']=[9.,9.]
    if fault=='early_command':old,new=pair(405);new['requested_command']=[0.,0.,-.45]
    if fault=='missing_receipt':new['new_selection'].pop('sustained_hold_reorientation')
    if fault=='stale_receipt':new['new_selection']['sustained_hold_reorientation']['frame']=405
    if fault=='expected_sha':expected='0'*64
    with pytest.raises(ValueError):compare_step(old,new,old['requested_command'],frame=old['tick'],expected_selection_sha256=expected)


def fixture(monkeypatch,tmp_path,fault=None):
    pairs=[pair(i) for i in range(407)];rows=[dict(tick=i,decision=a) for i,(a,b) in enumerate(pairs)]
    comparisons=[dict(original_row_sha256=runner.saved.identity(r),candidate_selection_sha256=identity(pairs[i][1]['new_selection'])) for i,r in enumerate(rows)]
    expected=dict(comparisons=comparisons,first_boundary=dict(original_requested_command=[0.,0.,0.],candidate_requested_command=[0.,0.,.45]))
    tape=[dict(tick=i,completed=True,pre_sample_index=749+50*i,post_sample_index=799+50*i,
        requested_command=a['requested_command']) for i,(a,b) in enumerate(pairs)]
    if fault=='incomplete':tape[-1]['completed']=False
    if fault=='clock':tape[-1]['post_sample_index']+=1
    if fault=='saved_row':comparisons[3]['original_row_sha256']='bad'
    if fault=='saved_candidate':comparisons[-1]['candidate_selection_sha256']='bad'
    seen=[]
    def stream(directory):
        yield from deepcopy(rows)
        raise AssertionError('observation after first unexecuted request consumed')
    class Reader:
        def __init__(self,directory):pass
        def packet(self,frame):
            if frame>=407:raise AssertionError('post-intervention raw packet consumed')
            seen.append(frame);return dict(frame=frame),{},{},1_500_000_000+frame*100_000_000
    class Controller:
        def __init__(self,*args,candidate=False,**kwargs):
            self.candidate=candidate;self.residual=SimpleNamespace(pending=None)
            self.mapper=SimpleNamespace(floor={(0,0)},occupied=set());self.memory={'observed':['same']}
        def observe(self,p,*args,**kwargs):
            frame=p['frame'];r=deepcopy(pairs[frame][int(self.candidate)])
            if frame==3:
                if fault=='raw_original' and not self.candidate:r['evidence']['observed_xy']=[9.,9.]
                if self.candidate:
                    if fault=='contact':self.memory={'changed':True}
                    if fault=='map':self.mapper.floor.add((1,1))
                    if fault=='mutation':p['changed']=True
                    if fault=='pending':self.residual.pending={'changed':True}
                    if fault=='early_command':r['requested_command']=[0.,0.,.45]
            return r
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    monkeypatch.setattr(runner,'read_rows',stream);monkeypatch.setattr(runner,'IntentReturnRGBDReplay',Reader)
    monkeypatch.setattr(runner,'packet',lambda *a,**k:({},{}));monkeypatch.setattr(runner,'public_acquisition',lambda x:x)
    monkeypatch.setattr(runner,'read_json',lambda root,name:deepcopy(tape) if name=='command_tape.json' else [{}]*407 if name=='auxiliary_camera_audit.json' else {})
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda x:object())
    monkeypatch.setattr(runner,'HoldReorientationController',Controller)
    monkeypatch.setattr(runner,'SustainedHoldReorientationController',lambda *a,**k:Controller(*a,candidate=True,**k))
    monkeypatch.setattr(runner.original,'assigned_model',lambda launch:SimpleNamespace(state_dict=lambda:{},parameters=lambda:[]))
    monkeypatch.setattr(runner,'state_digest',lambda x:runner.MODEL_SHA)
    return seen,expected


def test_complete_raw_wiring_reads_exactly_407_observations_and_never_following_outcome(monkeypatch,tmp_path):
    seen,expected=fixture(monkeypatch,tmp_path);r=runner.replay(expected)
    assert seen==list(range(407)) and r['raw_model_forecast_comparisons']==404
    assert r['first_changed_command_frame']==406 and not r['changed_command_executed']


@pytest.mark.parametrize('fault',['incomplete','clock','saved_row','saved_candidate','raw_original','contact','map','mutation','pending','early_command'])
def test_raw_loop_rejects_changed_inputs_and_state(monkeypatch,tmp_path,fault):
    _,expected=fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):runner.replay(expected)


@pytest.mark.parametrize('memory,storage',[(31,100),(64,40)])
def test_resource_shortfall_rejected(memory,storage):
    with pytest.raises(ValueError):runner.resources_for(dict(memory_available_bytes=memory*1024**3,artifact_free_bytes=storage*1024**3))
