"""Paired timing scope, complete equality and stopping before changed outcomes."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import benchmark_go2_scoped_support_cache_v1 as runner


def row(frame):
    return dict(tick=frame,observation_index=frame,pre_sample_index=749+50*frame,
        decision=dict(terminal='DONE' if frame==5 else None,requested_command=[0.,0.,0.],
            memory_receipt={'frame':frame}))


@pytest.mark.parametrize('frame',[0,1])
@pytest.mark.parametrize('fault',[None,'state','command','mutation'])
def test_pair_checks_complete_decision_inputs_and_alternates_order(frame,fault):
    calls=[]
    class Controller:
        def __init__(self,index): self.index=index
        def observe(self,policy,*args,**kwargs):
            calls.append(self.index); result=deepcopy(row(frame)['decision'])
            if self.index==1:
                if fault=='state': result['memory_receipt']['frame']=99
                elif fault=='command': result['requested_command']=[.2,0.,0.]
                elif fault=='mutation': policy['changed']=True
            return result
    inputs=({}, {}, {}, {}, {}, 1_500_000_000+frame*100_000_000)
    times=iter([0,10_000_000,11_000_000,31_000_000])
    if fault is None:
        result=runner.paired_step([Controller(0),Controller(1)],inputs,row(frame),frame=frame,clock=lambda:next(times))
        order=[0,1] if frame==0 else [1,0]
        assert calls==order and result['order']==[runner.LABELS[i] for i in order]
        assert result['controller_wall_ms'][runner.LABELS[order[0]]]==10.
        assert result['controller_wall_ms'][runner.LABELS[order[1]]]==20.
    else:
        with pytest.raises(ValueError):
            runner.paired_step([Controller(0),Controller(1)],inputs,row(frame),frame=frame,clock=lambda:next(times))
        if frame==1: assert calls==[1]  # A mismatch ends the pair immediately.


def test_timing_comparison_separates_warmup_terminal_and_order_groups():
    rows=[dict(frame=i,warmup=i<3,terminal='DONE' if i==5 else None,
        order=list(runner.LABELS[::1 if i%2==0 else -1]),
        controller_wall_ms={'single_pass_receipt_copied':200.+i,'support_cached_single_pass':150.+i}) for i in range(6)]
    result=runner.aggregate(rows)
    assert result['active']['observations']==2 and result['active']['paired_median_reduction_ms']==50.
    assert result['warmup_observations']==3 and result['terminal_observations']==1
    assert all(r['observations']==1 for r in result['by_first_controller'].values())
    assert not result['acquisition_and_receipt_io_timed'] and not result['real_time_qualified']


@pytest.mark.parametrize('mismatch',[False,True])
def test_complete_replay_or_immediate_stop_without_following_observation(monkeypatch,tmp_path,mismatch):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path); monkeypatch.setattr(runner,'FRAMES',6)
    models=[]; observed=[]; reads=[]
    def load(*a):
        models.append(SimpleNamespace(state_dict=lambda:{'hash':runner.MODEL_STATE},parameters=lambda:[]))
        return models[-1],'jepa','full'
    monkeypatch.setattr(runner,'load_assigned',load)
    monkeypatch.setattr(runner,'state_digest',lambda d:d['hash'])
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    class Controller:
        def __init__(self,model,*a,**k): self.index=len(models)-1
        def observe(self,policy,*a,**k):
            frame=policy['frame']; observed.append((self.index,frame)); result=deepcopy(row(frame)['decision'])
            if mismatch and self.index==1 and frame==2: result['memory_receipt']['frame']=7
            return result
    monkeypatch.setattr(runner,'SinglePassReceiptCopiedController',Controller)
    monkeypatch.setattr(runner,'SupportCachedSinglePassController',Controller)
    monkeypatch.setattr(runner,'IntentReturnRGBDReplay',lambda *a:SimpleNamespace(frames=[None]*6,
        packet=lambda i:({'frame':i},{},{},1_500_000_000+i*100_000_000)))
    tape=[dict(requested_command=[0.,0.,0.],completed=True) for _ in range(5)]
    monkeypatch.setattr(runner,'read_json',lambda p,n:tape if n=='command_tape.json' else [{}]*6)
    monkeypatch.setattr(runner,'packet',lambda *a,**k:({},{}))
    monkeypatch.setattr(runner,'public_acquisition',lambda p:p)
    monkeypatch.setattr(runner,'hardware',lambda:dict(artifact_free_bytes=2**40))
    def rows(*a):
        for i in range(6):
            reads.append(i)
            if mismatch and i>2: pytest.fail('followed a changed candidate with another observation')
            yield row(i)
        pytest.fail('read beyond the fixed original episode')
    monkeypatch.setattr(runner,'read_rows',rows)
    if mismatch:
        with pytest.raises(ValueError,match='complete support_cached_single_pass decision'): runner.replay({'correction_admission':{}})
        assert reads==[0,1,2]
    else:
        result=runner.replay({'correction_admission':{}})
        assert result['frames']==6 and result['both_model_states_unchanged']
        assert result['paired_timing']['active']['observations']==2 and len(observed)==12
    assert len(models)==2 and models[0] is not models[1]


@pytest.mark.parametrize('fault', [None, 'source_conflict', 'ancestor', 'phase'])
def test_current_benchmark_and_phase_proofs_use_their_original_verifiers(monkeypatch, fault):
    roots = list(runner.PREDECESSORS); calls = []
    monkeypatch.setattr(runner, 'verify_artifacts', lambda *a:None)
    def read(root, name):
        i = roots.index(root.name)
        if name == 'result.json':
            return dict(artifact_sha256={}, source_sha256={'shared':'b'*64 if fault == 'source_conflict' and i else 'a'*64})
        return {'proof':roots[i]}
    monkeypatch.setattr(runner, 'read_json', read)
    def current(launch):
        assert launch == {'proof':roots[0]}; calls.append('current')
    def ancestors():
        calls.append('ancestors')
        if fault == 'ancestor': raise ValueError('ancestor authentication failure')
    def phase(launch):
        assert launch == {'proof':roots[1]}; calls.append('phase')
        if fault == 'phase': raise ValueError('phase authentication failure')
    monkeypatch.setattr(runner, 'verify_inputs', current)
    monkeypatch.setattr(runner, 'verify_optimization_ancestry', ancestors)
    monkeypatch.setattr(runner, 'verify_phase', phase)
    if fault:
        with pytest.raises(ValueError): runner.verify_predecessors()
    else: assert runner.verify_predecessors() == {'shared':'a'*64}
    assert calls == (['current', 'ancestors'] if fault == 'ancestor' else ['current', 'ancestors', 'phase'])

