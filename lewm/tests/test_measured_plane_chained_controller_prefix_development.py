"""First changed command, output reconstruction and predecessor ordering."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from scripts import replay_go2_measured_plane_chained_controller_prefix_v1 as job
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def decision(frame):
    selection = dict(action='hold', prediction=[[[0.,0.,0.,1.,-30.]]*8]*6,
        head='direct',input_variant='no_rgb',target_offsets_ns=[100_000_000*i for i in range(1,9)],
        model_prediction_corrected=True,translation_bias_training_only=True,translation_bias_xy_m=[.001,.002])
    return dict(controller=job.comparison.ORIGINAL,tick=frame,terminal=None,failure=None,
        measured_plane_constrained_estimator=True,mission_receipt=dict(frame=frame,hold_required=False),
        requested_command=[0.,0.,0.],new_selection=selection if frame>=3 else None,
        original_visual_evidence={'pose':frame},evidence={'floor':frame})


def candidate(original):
    return deepcopy(original)|dict(controller=job.comparison.CANDIDATE,
        chained_anchor_reacquisition_enabled=True,direct_corner_flow_missingness_fallback_enabled=True)


@pytest.mark.parametrize('mode',['command','failure','post_forecast_failure','mission_hold','same'])
def test_comparison_preserves_actual_stop_and_forecast_semantics(mode):
    old=decision(3);new=candidate(old);calls=[1,1]
    if mode=='command':new['new_selection']['action']='forward';new['requested_command']=[.2,0.,0.]
    elif mode in ('failure','post_forecast_failure'):
        new.update(terminal='SENSOR_OR_MODEL_FAILURE',failure='preserved failure',new_selection=None)
        calls[1]=int(mode=='post_forecast_failure')
    elif mode=='mission_hold':new['new_selection']=None;new['mission_receipt']['hold_required']=True;calls[1]=0
    row=job.comparison.compare(old,new,deepcopy(old),frame=3,maximum_frames=6,model_calls=calls)
    assert row['stop']==(mode in ('command','failure','post_forecast_failure'))
    assert row['original_forecast_compared']==(mode in ('command','same'))
    assert row['following_changed_command_outcome_consumed'] is False


@pytest.mark.parametrize('fault',['original','prediction','command','scope','calls','missing_forecast','clock'])
def test_unexplained_changes_rejected(fault):
    old=decision(3);actual=deepcopy(old);new=candidate(old);calls=[1,1]
    if fault=='original':old['evidence']['floor']='changed'
    elif fault=='prediction':new['new_selection']['prediction']=[]
    elif fault=='command':new['requested_command']=[.2,0.,0.]
    elif fault=='scope':new['chained_anchor_reacquisition_enabled']=False
    elif fault=='calls':calls[1]=0
    elif fault=='missing_forecast':new['new_selection']=None;calls[1]=0
    elif fault=='clock':new['tick']=4
    with pytest.raises(ValueError):job.comparison.compare(old,new,actual,frame=3,maximum_frames=6,model_calls=calls)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('weight',torch.zeros(1));self.eval()
    def forward(self):return self.weight


def setup(tmp_path,monkeypatch,fault=None):
    source=tmp_path/'source';source.mkdir();output=tmp_path/'output';output.mkdir()
    directory=source/job.original.CASE[0];directory.mkdir()
    monkeypatch.setattr(job.native,'OUTPUT',source);monkeypatch.setattr(job,'OUTPUT',output)
    sha=job.original.state_digest(Model().state_dict());monkeypatch.setattr(job.inputs.job,'MODEL_SHA',sha)
    models=[]
    def model():
        value=Model();models.append(value);return value
    monkeypatch.setattr(job.original,'assigned_model',model)
    monkeypatch.setattr(job.original,'ArticulatedCollisionGeometry',lambda path:'geometry')
    references=[];tape=[]
    for frame in range(6):
        value=decision(frame)
        if frame==5:value.update(terminal='SENSOR_OR_MODEL_FAILURE',failure='original failure',new_selection=None)
        row,command=endpoint(frame,value);references.append(row)
        if frame<5:tape.append(command)
    controllers=[]
    class Controller:
        def __init__(self,model,geometry,**kwargs):
            self.model=model;self.arm=len(controllers);controllers.append(self)
        def observe(self,p,d,f,**kwargs):
            frame=p['frame']
            if frame in (3,4) and not (self.arm==1 and fault=='calls'):self.model()
            result=deepcopy(references[frame]['decision'])
            if self.arm==0:
                if fault=='original' and frame==3:result['evidence']['floor']='changed'
                return result
            result=candidate(result)
            if frame==3:
                if fault=='input':p['changed']=True
                if fault=='prediction':result['new_selection']['prediction']=[]
                if fault=='model':self.model.weight.add_(1)
            if frame==4 and fault!='matched_terminal':
                if fault=='candidate_failure':
                    result.update(terminal='SENSOR_OR_MODEL_FAILURE',failure='candidate failure',new_selection=None)
                else:
                    result['new_selection']['action']='forward';result['requested_command']=[.2,0.,0.]
            return result
    monkeypatch.setattr(job,'MeasuredPlaneResidualController',Controller)
    monkeypatch.setattr(job,'MeasuredPlaneChainedAnchorController',Controller)
    def read_json(root,name):
        if root==directory and name=='command_tape.json':return deepcopy(tape)
        return json.loads((root/name).read_text())
    monkeypatch.setattr(job.run,'read_json',read_json)
    original_reader=job.run.pipeline.read_rows
    def rows(root):
        if root==directory:yield from deepcopy(references)
        else:yield from original_reader(root)
    monkeypatch.setattr(job.run.pipeline,'read_rows',rows)
    consumed=[]
    def packets(root,count):
        assert root==directory and count==6
        for frame in range(count):
            consumed.append(frame)
            yield ({'frame':frame},{'image':np.array([frame])},{},{},{}),1_500_000_000+frame*100_000_000
    monkeypatch.setattr(job.full,'packets',packets)
    return dict(frames=6),consumed,models,output


@pytest.mark.parametrize('mode',[None,'candidate_failure','matched_terminal'])
def test_replay_stops_at_boundary_and_rechecks_only_consumed_packets(tmp_path,monkeypatch,mode):
    admission,consumed,models,output=setup(tmp_path,monkeypatch,mode)
    report=job.replay(admission)
    expected=list(range(6 if mode=='matched_terminal' else 5))
    assert consumed==expected
    assert report['frames']==len(expected) and report['actual_model_forward_calls']==[2,2]
    assert all(not model._forward_hooks for model in models)
    assert report['boundary_comparison']['stop'] is True
    job.check_output(report,admission)
    assert consumed==expected*2
    assert (output/'resource_monitor.jsonl').is_file()


@pytest.mark.parametrize('fault',['original','input','calls','prediction','model'])
def test_integrity_failure_stops_and_releases_hooks(tmp_path,monkeypatch,fault):
    admission,consumed,models,output=setup(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError):job.replay(admission)
    assert consumed==list(range(5 if fault=='model' else 4))
    assert all(not model._forward_hooks for model in models)
    assert not (output/'result.json').exists()


@pytest.mark.parametrize('fault',['missing','extra','packet','original','clock','comparison','report','calls'])
def test_closed_output_forgery_rejected(tmp_path,monkeypatch,fault):
    admission,_,_,output=setup(tmp_path,monkeypatch)
    report=job.replay(admission);rows=list(job.run.pipeline.read_rows(output))
    if fault=='missing':rows.pop()
    elif fault=='extra':rows.append(deepcopy(rows[-1])|dict(tick=len(rows)))
    elif fault=='packet':rows[3]['public_packet_sha256']='changed'
    elif fault=='original':rows[3]['original']['evidence']['floor']='changed'
    elif fault=='clock':rows[3]['observation_now_ns']+=1
    elif fault=='comparison':rows[3]['comparison']['stop']=True
    elif fault=='report':report['frames']-=1
    elif fault=='calls':rows[3]['actual_model_forward_calls'][1]=0
    (output/'context_decisions.jsonl.gz').unlink() # This test's temporary stream only.
    with job.run.pipeline.writer(output) as append:
        for row in rows:append(row)
    with pytest.raises(ValueError):job.check_output(report,admission)


def test_live_reserved_cpu_owner_prevents_output_or_native_admission(monkeypatch):
    monkeypatch.setattr(job,'cpu_launch',lambda:{})
    monkeypatch.setattr(job.run,'owner_live',lambda owner:owner==job.CPU_OWNER)
    with pytest.raises(ValueError,match='finish and end first'):job.cpu_slot('native','cpu',{})


def cpu_fixture(monkeypatch,tmp_path,fault=None):
    root=tmp_path/'waiter';root.mkdir();child=tmp_path/'replay';child.mkdir()
    monkeypatch.setattr(job.timing_waiter,'OUTPUT',root);monkeypatch.setattr(job.full,'OUTPUT',child)
    sources={'bound-source':'hash'};launch=dict(source_sha256=sources,boot_id='boot')
    report=dict(native_result_sha256='native',replay_result_sha256='replay-result',frames=3124,
        complete_actual_native_history=True,original_owner_ended=True,
        complete_rows_states_and_public_packets_reconstructed=True,original_raw_artifacts_reauthenticated=True)
    result=dict(status='MEASURED_PLANE_FULL_HISTORY_TIMING_WAIT_V1_COMPLETE',source_sha256=sources,
        artifact_sha256={name:job.CPU_LAUNCH_SHA if name=='launch.json' else 'hash'
            for name in ('launch.json','events.jsonl','replay_stdout.log','completion.json')},report=report)
    child_launch=dict(owner={'pid':123},boot_id='boot',source_sha256=sources,
        input_admission={'learned_result_sha256':'native'})
    replay=dict(status='MEASURED_PLANE_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE',source_sha256=sources,
        complete_output_and_public_packets_rechecked=True,original_raw_inputs_reauthenticated_before_and_after=True,
        report=dict(frames=3124),artifact_sha256={name:'hash' for name in (
            'launch.json','comparison.jsonl','state_checks.json','resource_monitor.jsonl','report.json')})
    if fault=='native':report['native_result_sha256']='substitute'
    elif fault=='sources':result['source_sha256']={'changed':'hash'}
    elif fault=='incomplete':report['complete_rows_states_and_public_packets_reconstructed']=False
    elif fault=='child_manifest':replay['artifact_sha256'].pop('comparison.jsonl')
    elif fault=='waiter_failure':(root/'failure.json').write_text('{}')
    elif fault=='child_failure':(child/'failure.json').write_text('{}')
    documents={(root,'result.json'):result,(root,'completion.json'):deepcopy(report),
        (child,'launch.json'):child_launch,(child,'result.json'):replay,(child,'report.json'):deepcopy(replay['report'])}
    if fault=='saved_report':documents[(child,'report.json')]['frames']=4
    monkeypatch.setattr(job,'cpu_launch',lambda:launch)
    monkeypatch.setattr(job.run,'owner_live',lambda owner:fault=='child_live' and owner==child_launch['owner'])
    monkeypatch.setattr(job.run,'read_json',lambda root,name:deepcopy(documents[(root,name)]))
    monkeypatch.setattr(job.run,'verify_artifacts',lambda *args:None)
    return sources


def test_completed_cpu_ordering_does_not_require_positive_timing_result(monkeypatch,tmp_path):
    sources=cpu_fixture(monkeypatch,tmp_path)
    receipt=job.cpu_slot('native','cpu-result',sources)
    assert receipt['preceding_waiter_and_replay_owners_ended'] is True
    assert receipt['timing_improvement_required'] is False
    assert receipt['preceding_replay_science_reexecuted'] is False


@pytest.mark.parametrize('fault',['native','sources','incomplete','child_manifest','waiter_failure',
    'child_failure','saved_report','child_live'])
def test_incomplete_or_changed_cpu_predecessor_is_not_bypassed(monkeypatch,tmp_path,fault):
    sources=cpu_fixture(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):job.cpu_slot('native','cpu-result',sources)


def test_candidate_source_preflight_does_not_admit_or_execute(monkeypatch,tmp_path):
    monkeypatch.setattr(job,'OUTPUT',tmp_path/'not-created')
    monkeypatch.setattr(job.run,'ENV',{})
    monkeypatch.setattr(job.run.cv2.ocl,'useOpenCL',lambda:False)
    monkeypatch.setattr(job.run,'validate_root',lambda *args,**kwargs:None)
    monkeypatch.setattr(job,'prepared_sources',lambda:{'prepared':'source'})
    monkeypatch.setattr(job.full,'resources',lambda:{'capacity':'synthetic'})
    def forbidden(*args,**kwargs):pytest.fail('source preflight reached execution admission')
    monkeypatch.setattr(job,'admit',forbidden);monkeypatch.setattr(job,'replay',forbidden)
    job.main(source_only=True)
    assert not job.OUTPUT.exists()
