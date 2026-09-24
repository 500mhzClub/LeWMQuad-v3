"""Recorded-prefix orchestration with synthetic models, packets and controllers."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from scripts import extended_return_budget_prefix_replay_development as job
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('weight',torch.zeros(1));self.eval()
    def forward(self):return self.weight


def setup(tmp_path,monkeypatch,fault=None):
    source=tmp_path/'source';output=tmp_path/'output';source.mkdir();output.mkdir()
    case=('synthetic',2,'no_rgb','direct');directory=source/case[0];directory.mkdir()
    monkeypatch.setattr(job,'native',NS(OUTPUT=source,CASE=case))
    sha=job.state_digest(Model().state_dict());monkeypatch.setattr(job,'MODEL_SHA',sha)
    models=[];controllers=[];consumed=[]
    def model():
        m=models[0] if models and fault=='shared_model' else Model()
        models.append(m);return m
    monkeypatch.setattr(job,'assigned_model',model)
    monkeypatch.setattr(job,'geometry_factory',lambda path:{'urdf':'synthetic'})
    monkeypatch.setattr(job.comparison,'BOUNDARY',4)
    monkeypatch.setattr(job.comparison,'MAX_PREFIX_OBSERVATIONS',5)
    monkeypatch.setattr(job,'FIXED_STATE_FRAMES',(0,3))
    # Existing performance normalization has separate actual-controller tests;
    # this fixture's recorded baseline is already its unchanged synthetic row.
    monkeypatch.setattr(job.comparison,'to_recorded',deepcopy)
    monkeypatch.setattr(job.comparison,'observed_state',lambda c:dict(frame=c.frame,hidden=c.hidden))
    def decision(frame):
        return dict(tick=frame,controller=job.comparison.BASELINE,requested_command=[0.,0.,0.],
            terminal='MISSION_TICK_BUDGET_EXHAUSTED' if frame>=4 else None,
            shared_navigation_budget_ticks=4000,mission_receipt={'global_navigation_ticks':4000},
            new_selection={'prediction':[1.]} if frame==3 else None)
    references=[];tape=[]
    for frame in range(4014):
        row,command=endpoint(frame,decision(frame));references.append(row)
        if frame<4013:tape.append(command)
    class Controller:
        def __init__(self,model,geometry,*,navigation_ticks,**kwargs):
            self.model=model;self.geometry=geometry;self.arm=len(controllers)
            self.frame=-1;self.hidden=0
            assert navigation_ticks==(4000 if self.arm==0 else 8000)
            controllers.append(self)
        def observe(self,p,d,f,**kwargs):
            frame=p['frame'];self.frame=frame;result=decision(frame)
            forward=frame==3 or self.arm==1 and frame==4
            if forward and not (self.arm==1 and fault=='calls' and frame==3):self.model()
            if self.arm==1:
                result.update(controller=job.comparison.candidate.CONTROLLER,extended_return_budget_enabled=True,
                    shared_navigation_budget_ticks=8000,mission_receipt={'global_navigation_ticks':8000})
                if frame==4:
                    result.update(terminal=None,requested_command=[.2,0.,0.],new_selection={'prediction':[1.]})
                if fault=='early_intervention' and frame==2:result['unknown_evidence']=True
                if frame==3:
                    if fault=='input':p['changed']=True
                    elif fault=='state':self.hidden=1
                    elif fault=='model':self.model.weight.add_(1)
                    elif fault=='geometry':self.geometry['changed']=True
            return result
    monkeypatch.setattr(job.comparison.candidate,'MeasuredPlaneChainedSinglePassController',Controller)
    monkeypatch.setattr(job.comparison.candidate,'ExtendedReturnBudgetChainedController',Controller)
    original_read=job.run.read_json
    def read(root,name):
        if root==directory and name=='command_tape.json':return deepcopy(tape)
        return original_read(root,name)
    monkeypatch.setattr(job.run,'read_json',read)
    original_rows=job.pipeline.read_rows
    def rows(root):
        if root==directory:
            for row in references:yield deepcopy(row)
        else:yield from original_rows(root)
    monkeypatch.setattr(job.pipeline,'read_rows',rows)
    def packets(root,count):
        assert root==directory and count==4014
        limit=3 if fault=='early_intervention' else 5
        for frame in range(limit):
            consumed.append(frame)
            yield ({'frame':frame},{'depth':np.array([frame])},{},{},{}),1_500_000_000+frame*100_000_000
        raise AssertionError('decoded a public packet following the intervention')
    monkeypatch.setattr(job,'packets',packets)
    admission=dict(frames=4014,native_case=case[0],model_state_sha256=sha,native_owner_ended=True,
        original_worker_ended=True,complete_raw_audit_verified=True,original_physical_prefix_reconstructed=True,
        original_schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED',learned_result_sha256='a'*64)
    return admission,consumed,models,output


def test_complete_prefix_stops_before_next_packet_and_saved_output_reconstructs(tmp_path,monkeypatch):
    admission,consumed,models,output=setup(tmp_path,monkeypatch)
    report=job.replay(admission,output=output)
    assert consumed==list(range(5)) and report['frames']==5
    assert report['budget_only_preboundary_decisions_supported']
    assert report['actual_model_forward_calls']==[1,2]
    assert [r['frame'] for r in report['observed_state_checks']]==[-1,0,3,4]
    assert report['observed_state_checks'][-1]['equality_required'] is False
    assert all(not m._forward_hooks for m in models)
    assert not report['native_execution'] and not report['physical_prefix_verified']
    job.check_output(report,admission,output=output)
    assert consumed==list(range(5))*2
    before=(output/job.pipeline.stream.NAME).read_bytes()
    with pytest.raises(ValueError,match='exclusive'):job.replay(admission,output=output)
    assert (output/job.pipeline.stream.NAME).read_bytes()==before and len(models)==2


def test_early_decision_intervention_is_preserved_as_negative_evidence(tmp_path,monkeypatch):
    admission,consumed,models,output=setup(tmp_path,monkeypatch,'early_intervention')
    report=job.replay(admission,output=output)
    assert consumed==[0,1,2] and not report['budget_only_preboundary_decisions_supported']
    assert [r['frame'] for r in report['observed_state_checks']]==[-1,0,2]
    job.check_output(report,admission,output=output)
    assert consumed==[0,1,2]*2 and all(not m._forward_hooks for m in models)


@pytest.mark.parametrize('fault',['input','calls','state','model','geometry','shared_model'])
def test_changed_execution_evidence_rejects_and_removes_hooks(tmp_path,monkeypatch,fault):
    admission,consumed,models,output=setup(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):job.replay(admission,output=output)
    expected=[] if fault=='shared_model' else list(range(5)) if fault in ('model','geometry') else list(range(4))
    assert consumed==expected and all(not m._forward_hooks for m in models)
    assert not (output/'result.json').exists()


@pytest.mark.parametrize('fault',['missing','extra','packet','clock','endpoint','recorded','decision','comparison',
    'state','initial_state','identity','resource','report'])
def test_checker_rejects_changed_saved_population_and_evidence(tmp_path,monkeypatch,fault):
    admission,consumed,models,output=setup(tmp_path,monkeypatch)
    report=job.replay(admission,output=output)
    path=output/job.pipeline.stream.NAME
    with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
    if fault=='missing':rows.pop()
    elif fault=='extra':rows.append(deepcopy(rows[-1])|{'tick':5})
    elif fault=='packet':rows[3]['public_packet_sha256']='b'*64
    elif fault=='clock':rows[3]['observation_now_ns']+=1
    elif fault=='endpoint':rows[3]['pre_sample_index']+=1
    elif fault=='recorded':rows[3]['recorded_decision_sha256']='b'*64
    elif fault=='decision':rows[3]['decision']['new_evidence']=True
    elif fault=='comparison':rows[3]['comparison']['stop']=True
    elif fault in ('state','initial_state'):
        p=output/'state_checks.json';states=json.loads(p.read_text())
        if fault=='state':states.pop()
        else:states[0]['candidate_sha256']='b'*64
        p.write_text(json.dumps(states))
    elif fault=='identity':
        p=output/'identities.json';ids=json.loads(p.read_text());ids['final_model_sha256'][1]='b'*64
        p.write_text(json.dumps(ids))
    elif fault=='resource':(output/'resource_monitor.jsonl').write_text('')
    else:report['actual_model_forward_calls'][1]=0
    with gzip.open(path,'wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError):job.check_output(report,admission,output=output)
    if fault=='extra':assert consumed==list(range(5))*2


def test_output_and_runtime_resource_bounds_stop_without_next_packet(tmp_path,monkeypatch):
    admission,consumed,models,output=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(job,'MAX_OUTPUT_BYTES',1)
    with pytest.raises(ValueError,match='2 GiB'):job.replay(admission,output=output)
    assert consumed==[0] and all(not m._forward_hooks for m in models)


@pytest.mark.parametrize('field,value',[('frames',4013),('native_owner_ended',False),
    ('model_state_sha256','b'*64),('original_schedule_terminal','other')])
def test_wrong_admission_rejected_before_models_or_inputs(tmp_path,monkeypatch,field,value):
    admission,consumed,models,output=setup(tmp_path,monkeypatch)
    admission[field]=value
    with pytest.raises(ValueError):job.replay(admission,output=output)
    assert not consumed and not models
