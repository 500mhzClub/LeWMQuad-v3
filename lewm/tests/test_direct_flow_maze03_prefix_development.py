"""Same tracking intervention rules at the independently recorded maze3 failure."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
from lewm.tests.test_direct_flow_prefix_development import rows, boundary as maze1_boundary
from lewm import direct_flow_maze03_prefix_development as comparison
from scripts import replay_go2_direct_flow_maze03_prefix_v1 as runner


def boundary():
    old,new=maze1_boundary(); frame=264; now=1_500_000_000+frame*100_000_000
    old['tick']=frame-1; new['tick']=frame
    for d in (old,new): d['original_visual_evidence']['decision_ns']=now
    new['original_visual_evidence']['current_pose']['frame']=frame
    new['original_visual_evidence']['direct_corner_flow_fallback'].update(frame=frame,measured_ns=now)
    new['new_selection']['action']='left_turn'; new['requested_command']=[0.,0.,.45]
    return old,new


def test_comparator_and_replay_preserve_original_rules_at_new_fixed_boundary():
    path=Path('lewm/direct_flow_prefix_development.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='7660f2175042e704e3a224e7840d427688c12659e6086a5f32f720a51a4d708b'
    old=ast.parse(path.read_text()); new=ast.parse(Path('lewm/direct_flow_maze03_prefix_development.py').read_text())
    for node in new.body:
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='BOUNDARY_FRAME':
            assert node.value.value==264; node.value.value=214
    assert ast.dump(old)==ast.dump(new)
    def function(path):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='replay')
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str): n.value=n.value.replace('DIRECT_FLOW_MAZE03_PREFIX','DIRECT_FLOW_MAZE01_PREFIX')
            return n
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='public_mission':
                assert n.args[0].value==3; n.args[0].value=1
            n.keywords=[k for k in n.keywords if k.arg!='layout_index']
            return self.generic_visit(n)
    assert ast.dump(function('scripts/replay_go2_direct_flow_maze01_prefix_v2.py'))==ast.dump(
        Normalize().visit(function('scripts/replay_go2_direct_flow_maze03_prefix_v1.py')))
    assert runner.CASE[0:2]==('full_jepa_novel_maze_03',3)
    assert runner.MAX_FRAMES==265 and runner.BOUNDARY_FRAME==264


@pytest.mark.parametrize('fault',[None,'earlier_state','failure_witness','stale_pose','later_observation'])
def test_fixed_boundary_cannot_borrow_different_state_or_following_observation(fault):
    old,new=boundary(); frame=264
    if fault=='earlier_state':
        frame=263; old,new=rows(frame); new['mission_receipt']={'changed':True}
    elif fault=='failure_witness': new['original_visual_evidence']['direct_corner_flow_fallback']['original_camera_selection']={}
    elif fault=='stale_pose': new['original_visual_evidence']['current_pose']['frame']=263
    elif fault=='later_observation': frame=265; old,new=rows(frame)
    if fault is None:
        result=comparison.compare_step(old,new,old['requested_command'],frame=frame)
        assert result['controller_recovered'] and result['stop']
    else:
        with pytest.raises(ValueError): comparison.compare_step(old,new,old['requested_command'],frame=frame)


@pytest.mark.parametrize('outcome',['recovered','negative','earlier_change','input_mutation','model_change','live_rejection'])
def test_full_prefix_scope_and_preserved_boundary_failure(monkeypatch,tmp_path,outcome):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path); weight={'hash':runner.MODEL_STATE}
    model=SimpleNamespace(state_dict=lambda:dict(weight),parameters=lambda:[])
    monkeypatch.setattr(runner,'load_assigned',lambda *a:(model,'jepa','full'))
    monkeypatch.setattr(runner,'state_digest',lambda d:d['hash'])
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda *a:SimpleNamespace(free=2**40))
    packets=[]; consumed=[]
    def decisions(i):
        old,new=boundary() if i==264 else rows(i)
        if i<3: old['new_selection']=new['new_selection']=None
        return old,new
    def original_rows(*a):
        for i in range(265):
            if outcome in ('earlier_change','input_mutation') and i>5:
                pytest.fail('consumed observation after altered current state')
            consumed.append(i)
            yield dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=decisions(i)[0])
        pytest.fail('consumed observation following the changed failure boundary')
    monkeypatch.setattr(runner,'read_rows',original_rows)
    def packet_at(i):
        packets.append(i); return {'frame':i},{},{},1_500_000_000+i*100_000_000
    monkeypatch.setattr(runner,'IntentReturnRGBDReplay',lambda *a:SimpleNamespace(frames=[None]*275,packet=packet_at))
    tape=[dict(requested_command=[0.,0.,0.],completed=True) for _ in range(274)]
    monkeypatch.setattr(runner,'read_json',lambda p,n:tape if n=='command_tape.json' else [{}]*275)
    monkeypatch.setattr(runner,'packet',lambda *a,**k:({},{}))
    monkeypatch.setattr(runner,'public_acquisition',lambda a:a)
    def validate(*a,**k):
        if outcome=='live_rejection' and a[2]['boundary_reached']: raise ValueError('synthetic live contract rejection')
    monkeypatch.setattr(runner,'validate_live',validate)
    class Controller:
        def observe(self,policy,*a,**k):
            i=policy['frame']; old,new=decisions(i)
            if i==5:
                if outcome=='earlier_change': new['mission_receipt']={'changed':True}
                elif outcome=='input_mutation': policy['changed']=True
                elif outcome=='model_change': weight['hash']='changed'
            if i==264 and outcome=='negative':
                new.update(tick=263,terminal=old['terminal'],failure='fallback rejected',requested_command=[0.,0.,0.],
                    new_selection=None,evidence=None)
                new['original_visual_evidence'].update(status='VISUAL_TERMINAL_FAILURE',current_pose=None)
                new['original_visual_evidence']['direct_corner_flow_fallback']['accepted']=False
            return new
    monkeypatch.setattr(runner,'DirectFlowFloorTransportController',lambda *a,**k:Controller())
    if outcome in ('recovered','negative'):
        result=runner.replay({'correction_admission':{}})
        assert result['frames']==265 and result['exact_original_decisions']==264
        assert result['raw_model_forecast_comparisons']==261 and result['prior_commands_compared']==264
        assert result['full_controller_recovered_at_boundary']==(outcome=='recovered')
        assert not result['following_recorded_observations_consumed'] and packets==consumed==list(range(265))
    else:
        with pytest.raises(ValueError): runner.replay({'correction_admission':{}})
        if outcome=='live_rejection':
            from scripts.maze_decision_stream_development import read_rows
            saved=list(read_rows(tmp_path))
            assert saved[-1]['tick']==264 and 'synthetic live contract rejection' in saved[-1]['comparison_failure']
