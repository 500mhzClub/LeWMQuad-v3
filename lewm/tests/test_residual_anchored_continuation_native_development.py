"""Physical prefix ends before changed action outcomes; original audit retained."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.residual_anchored_continuation_prefix_development import compare_step
from lewm.tests import test_residual_anchored_continuation_development as component
from lewm.tests.test_residual_anchored_continuation_prefix_development import decisions
from scripts import residual_anchored_continuation_native_prefix_development as prefix


def shift_clock(value, delta):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key in ('tick','frame','available_tick','pending_forecast_tick') and type(item) is int:
                result[key]=item+delta
            elif key=='measured_ns' and type(item) is int: result[key]=item+delta*100_000_000
            elif key in ('residual_source_ticks','residual_available_ticks'):
                result[key]=[i+delta for i in item]
            else: result[key]=shift_clock(item,delta)
        return result
    if isinstance(value,list):return [shift_clock(v,delta) for v in value]
    return value


def boundary():
    old,_=decisions()
    original,receipt,mapper=component.fixture()
    prediction=np.asarray(original['prediction']).copy()
    prediction[1,:,4],prediction[2,:,4]=prediction[2,:,4].copy(),prediction[1,:,4].copy()
    selection=component.score_candidates(prediction,goal_body_xy_m=[1.,0.],contact_penalty_m=1.2)
    selection.update(prediction=prediction.tolist(),first_prediction_horizon_ns=100_000_000,
        target_offsets_ns=list(range(100_000_000,800_000_001,100_000_000)),mode='WAYPOINT',
        phase_allowed_actions=list(component.ACTIONS),phase_admissible_candidates=6,
        intermediate_target_is_mission_goal=False,view_budget_exhausted=False)
    selection=component.filter_selection(selection,mapper.surface,object(),now_ns=component.NOW,persistent=True)
    selection=component.constrain(selection,mapper.surface.position,np.eye(3),mapper.occupied)
    selection=component.plan(selection,mapper.surface.position,np.eye(3),mapper.occupied)
    selection=component.score_waypoint_execution(selection,receipt)
    assert selection['action']=='hold'
    changed=component.apply(selection,receipt,mapper)
    assert changed['action']=='left_arc'
    old.update(new_selection=selection,causal_residual_receipt=receipt)
    new=deepcopy(old)
    new.update(controller='residual_anchored_continuation_controller_v1',
        residual_hold_feasibility_enabled=True,residual_anchored_continuation_enabled=True,
        new_selection=changed,requested_command=changed['requested_command'],selected_action=changed['action'])
    return shift_clock(old,170),shift_clock(new,170)


def evidence():
    old_boundary,new_boundary=boundary();saved=[];original=[]
    for i in range(prefix.FRAMES):
        old=shift_clock(old_boundary,i-prefix.CHANGED)
        if i==prefix.CHANGED:new=deepcopy(new_boundary)
        else:
            # Prior rows are equality fixtures; raw native replay owns their history validity.
            new=deepcopy(old)
            new.update(controller='residual_anchored_continuation_controller_v1',
                residual_hold_feasibility_enabled=True,residual_anchored_continuation_enabled=True)
        if i<3:old['new_selection']=new['new_selection']=None
        check=compare_step(old,new,old['requested_command'],frame=i)
        saved.append(dict(tick=i,decision=new,comparison=check,
            original_requested_command=old['requested_command'],public_input_arrays_unchanged=True))
        original.append(dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old))
    report=dict(case='full_jepa_residual_first_interval_maze_02',frames=181,maximum_frames=3004,
        first_requested_command_difference=180,hold_reconsideration_interventions=1,
        anchored_continuation_interventions=1,final_requested_command=[.16,0.,.45],
        prior_requested_command=[0.,0.,0.],final_terminal=None,raw_model_forecast_comparisons=178,
        complete_original_selection_preserved=True,unchanged_observed_mission_and_residual_state_exact=True,
        original_actual_commands_before_intervention_exact=True,stopped_at_first_command_or_terminal_difference=True,
        following_recorded_observations_consumed=False,public_input_arrays_unchanged=True,
        model_state_sha256=MODEL_STATE,model_state_unchanged=True,unexecuted_outcomes_inferred=False,
        native_execution=False,navigation_verified=False,changed_selection=deepcopy(saved[-1]['decision']['new_selection']))
    return dict(status='RESIDUAL_ANCHORED_CONTINUATION_PREFIX_V1_COMPLETE',model_loaded=True,
        model_training=False,native_execution=False,shadow_replay_only=True,report=report),saved,original


@pytest.mark.parametrize('fault',[None,'short','extra','model','frames','following','prior_request',
    'state','forecast_count','missing_bank','selection','early_change','old_state','hold'])
def test_full_positive_prefix_reconstructed_before_native_admission(monkeypatch,fault):
    result,rows,original=evidence()
    if fault=='short':rows.pop()
    elif fault=='extra':rows.append(deepcopy(rows[-1]))
    elif fault=='model':result['report']['model_state_sha256']='0'*64
    elif fault=='frames':result['report']['frames']=180
    elif fault=='following':result['report']['following_recorded_observations_consumed']=True
    elif fault=='prior_request':rows[5]['original_requested_command']=[1.,0.,0.]
    elif fault=='state':rows[3]['decision']['memory_receipt']['changed']=True
    elif fault=='forecast_count':rows[3]['comparison']['raw_model_forecasts_compared']=False
    elif fault=='missing_bank':
        original[3]['decision']['new_selection']=rows[3]['decision']['new_selection']=None
        rows[3]['comparison']=compare_step(original[3]['decision'],rows[3]['decision'],
            original[3]['decision']['requested_command'],frame=3)
    elif fault=='selection':result['report']['changed_selection']={}
    elif fault=='early_change':rows[179]['comparison']['anchored_continuation_changed_action']=True
    elif fault=='old_state':original[5]['decision']['mission_receipt']['changed']=True
    elif fault=='hold':result['report']['final_requested_command']=[0.,0.,0.]
    monkeypatch.setattr(prefix,'read_rows',lambda path:iter(rows if path is None else original))
    if fault is None:assert prefix.admit_prefix(None,result)==result['report']
    else:
        with pytest.raises(ValueError):prefix.admit_prefix(None,result)


def native_fixture(monkeypatch,tmp_path):
    result,bound,original=evidence();old=tmp_path/'old';new=tmp_path/'new';saved=tmp_path/'saved'
    data=dict(timestamp_s=np.arange(9800)*.002,base_pose_world=np.zeros((9800,7)))
    for path in (old,new,saved):path.mkdir()
    for path in (old,new):np.savez(path/'physics_trace.npz',**data)
    newrows=[deepcopy(row)|dict(observation_index=i,pre_sample_index=749+50*i) for i,row in enumerate(bound)]
    rows={old:original,new:newrows,saved:bound}
    tapes={path:[dict(requested_command=row['decision']['requested_command'],completed=True) for row in rows[path]] for path in (old,new)}
    reads=[];packets=[];changes={}
    def read_rows(path):
        for i,row in enumerate(rows[path]):reads.append((path,i));yield row
        pytest.fail('read past fixed181-observation boundary')
    monkeypatch.setattr(prefix,'read_rows',read_rows)
    monkeypatch.setattr(prefix,'read_json',lambda path,name:tapes[path] if name=='command_tape.json' else [{}]*181)
    def reader(path):
        def packet(i):
            packets.append((path,i))
            return {'frame':i,'value':changes.get((path,i),0)},{},{},1_500_000_000+i*100_000_000
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix,'IntentReturnRGBDReplay',reader)
    monkeypatch.setattr(prefix,'public_acquisition',lambda r:r)
    monkeypatch.setattr(prefix,'packet',lambda *a,**k:({},{}))
    return result['report'],old,new,saved,rows,tapes,reads,packets,changes,data


@pytest.mark.parametrize('fault',[None,'physical','public','state','decision','prior_command',
    'new_command','short','new_future','partial_new_command'])
def test_actual_shared_prefix_excludes_changed_command_outcomes(monkeypatch,tmp_path,fault):
    report,old,new,saved,rows,tapes,reads,packets,changes,data=native_fixture(monkeypatch,tmp_path)
    if fault=='physical':data['base_pose_world'][9749,0]=.1;np.savez(new/'physics_trace.npz',**data)
    elif fault=='new_future':data['base_pose_world'][9750:,0]=100.;np.savez(new/'physics_trace.npz',**data)
    elif fault=='public':changes[(new,180)]=1
    elif fault=='state':rows[old][180]['decision']['memory_receipt']={'changed':True}
    elif fault=='decision':rows[new][180]['decision']['new_selection']['prediction']=[[2.]]
    elif fault=='prior_command':tapes[new][179]['requested_command']=[.2,0.,0.]
    elif fault=='new_command':tapes[new][180]['requested_command']=[0.,0.,0.]
    elif fault=='short':np.savez(new/'physics_trace.npz',**{k:v[:9749] for k,v in data.items()})
    elif fault=='partial_new_command':tapes[new][180]['completed']=False
    if fault in (None,'new_future','partial_new_command'):
        result=prefix.compare(old,new,saved,report)
        assert result['physical_prefix_samples']==9750 and result['common_prefix_frames']==181
        assert result['raw_model_forecast_comparisons']==178
        assert result['candidate_intervention_command_completed']==(fault!='partial_new_command')
        assert len(reads)==543 and len(packets)==362 and not result['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError):prefix.compare(old,new,saved,report)


def test_original_collect_and_audit_calculations_are_preserved():
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='ResidualAnchoredContinuationController':node.id='ResidualFirstIntervalController'
            return node
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value=node.value.replace('RESIDUAL_ANCHORED_CONTINUATION_','RESIDUAL_FIRST_INTERVAL_')
            return node
        def visit_Call(self,node):
            node.keywords=[k for k in node.keywords if k.arg not in ('residual_hold_feasibility_enabled','residual_anchored_continuation_enabled')]
            return self.generic_visit(node)
        def visit_Assert(self,node):
            if ast.unparse(node.test) in ("result['residual_hold_feasibility_enabled'] is True", "result['residual_anchored_continuation_enabled'] is True"):return None
            return self.generic_visit(node)
    def function(path,name):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    for kind,names in [('episode',('collect','artifacts')),('audit',('audit',))]:
        for name in names:
            old=function('scripts/residual_first_interval_maze_'+kind+'_development.py',name)
            new=function('scripts/residual_anchored_continuation_maze_'+kind+'_development.py',name)
            assert ast.dump(old)==ast.dump(Normalize().visit(new)),(kind,name)
