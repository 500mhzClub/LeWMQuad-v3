"""Physical prefix joining uses synthetic archives and a real replay receipt."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pytest

from scripts import extended_return_budget_native_prefix_development as prefix
from lewm.tests import test_extended_return_budget_prefix_replay_development as runner_test
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def setup(tmp_path,monkeypatch):
    admission,_,_,replay_root=runner_test.setup(tmp_path,monkeypatch)
    report=runner_test.job.replay(admission,output=replay_root)
    (replay_root/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(prefix.replay,'OUTPUT',replay_root)
    monkeypatch.setattr(prefix.run,'read_json',lambda root,name:json.loads((root/name).read_text()))
    prior=tmp_path/'source'/'synthetic';current=tmp_path/'current'/'synthetic_long';current.mkdir(parents=True)
    with gzip.open(replay_root/'context_decisions.jsonl.gz','rt') as f:saved=[json.loads(line) for line in f]
    bound=prefix.boundary(report);count=bound['frames'];n=bound['physics_samples']+50
    for directory,arm in ((prior,'baseline'),(current,'decision')):
        raw=dict(timestamp_s=np.arange(1,n+1)*.002,base_pose_world=np.zeros((n,7)),
            base_twist_world=np.zeros((n,6)),joint_position=np.zeros((n,12)),joint_velocity=np.zeros((n,12)),
            requested_command=np.zeros((n,3)),applied_command=np.zeros((n,3)),post_slew_applied_command=np.zeros((n,3)),
            physics_contact=np.zeros(n,np.uint8),phase=np.zeros(n,np.uint8),edge_index=np.zeros(n,np.uint8))
        raw['base_pose_world'][:,6]=1.
        if directory==current:
            raw['base_pose_world'][bound['physics_samples']:,0]=1. # Outcomes after the intervention may differ.
            raw['requested_command'][bound['physics_samples']:]=report['boundary']['candidate_requested_command']
        np.savez_compressed(directory/'physics_trace.npz',**raw)
        tape=[]
        with gzip.open(directory/'context_decisions.jsonl.gz','wt') as f:
            for frame,row in enumerate(saved):
                observation,command=endpoint(frame,deepcopy(row[arm]));tape.append(command)
                f.write(json.dumps(observation)+'\n')
        (directory/'command_tape.json').write_text(json.dumps(tape))
        # Population bindings only: public reconstruction is replaced below.
        names={'policy_observations.json','policy_histories.npz','depth_observations.json',
            'fast_gyro_histories.npz','auxiliary_camera_audit.json'}
        names|={f'{kind}_{frame:04d}.{suffix}' for frame in range(count)
            for kind,suffix in (('rgb','png'),('depth','npz'),('auxiliary_rgb','png'),('auxiliary_depth','npz'))}
        for name in names:(directory/name).write_bytes(b'synthetic bound packet input')
    checks=[];decoded=[]
    def verify(root,bindings):
        checks.append(root)
        for name,sha in bindings.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError('changed bound input')
    monkeypatch.setattr(prefix.run,'verify_artifacts',verify)
    monkeypatch.setattr(prefix,'artifact_path',lambda root,name:root/name)
    def packets(directory,frames,*,longer):
        assert frames==count and longer is (directory==current)
        for frame in range(frames):
            decoded.append((directory,frame))
            yield saved[frame]['public_packet_sha256'],saved[frame]['observation_now_ns']
    monkeypatch.setattr(prefix,'public_packets',packets)
    def bindings(directory):
        return {directory.name+'/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    ids=[bindings(prior),bindings(current),{name:hashlib.sha256((replay_root/name).read_bytes()).hexdigest()
        for name in ('context_decisions.jsonl.gz','report.json')}]
    return prior,current,report,ids,checks,decoded


def compare(values):
    prior,current,report,ids,_,_=values
    return prefix.compare(prior,current,report,prior_bindings=ids[0],current_bindings=ids[1],replay_bindings=ids[2])


def test_actual_boundary_requests_and_complete_preintervention_physics_reproduce(tmp_path,monkeypatch):
    values=setup(tmp_path,monkeypatch);prior,current,report,ids,checks,decoded=values
    result=compare(values)
    assert result['common_prefix_frames']==5 and result['physical_prefix_samples']==950
    assert result['physical_and_public_prefix_exact'] and result['candidate_intervention_command_completed']
    assert result['boundary_command_samples_present']==50 and result['all_supplied_artifacts_rehashed_before_and_after']
    assert not result['following_physical_outcomes_compared'] and not result['navigation_verified']
    assert len(checks)==6 and decoded==[(p,i) for i in range(5) for p in (prior,current)]


@pytest.mark.parametrize('fault',['physics','schema','boundary_request','command_record','decision','unbound',
    'packet','clock','report','extra_replay_row','after_hash'])
def test_changed_physics_packets_decisions_commands_or_bindings_reject(tmp_path,monkeypatch,fault):
    values=setup(tmp_path,monkeypatch);prior,current,report,ids,checks,decoded=values
    def rebind(directory,path,index):ids[index][directory.name+'/'+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if fault in ('physics','schema','boundary_request'):
        path=current/'physics_trace.npz'
        with np.load(path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
        if fault=='physics':raw['joint_position'][949,0]=.01
        elif fault=='schema':del raw['edge_index']
        else:raw['requested_command'][999,0]=0.
        np.savez_compressed(path,**raw);rebind(current,path,1)
    elif fault=='command_record':
        path=current/'command_tape.json';tape=json.loads(path.read_text());tape[2]['unexpected_phase']=3
        path.write_text(json.dumps(tape));rebind(current,path,1)
    elif fault=='decision':
        path=current/'context_decisions.jsonl.gz'
        with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
        rows[2]['decision']['new_evidence']=True
        with gzip.open(path,'wt') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
        rebind(current,path,1)
    elif fault=='unbound':del ids[1][current.name+'/auxiliary_depth_0004.npz']
    elif fault in ('packet','clock'):
        original=prefix.public_packets
        def changed(directory,count,*,longer):
            for frame,(sha,now) in enumerate(original(directory,count,longer=longer)):
                if longer and frame==4:
                    if fault=='packet':sha='b'*64
                    else:now+=1
                yield sha,now
        monkeypatch.setattr(prefix,'public_packets',changed)
    elif fault=='report':report['boundary']['candidate_requested_command']=[0.,0.,0.]
    elif fault=='extra_replay_row':
        path=prefix.replay.OUTPUT/'context_decisions.jsonl.gz'
        with gzip.open(path,'at') as f:f.write(json.dumps({'tick':5})+'\n')
        ids[2][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        original=prefix.run.verify_artifacts
        def changed(root,bindings):
            if len(checks)==5:raise ValueError('changed final bound bytes')
            return original(root,bindings)
        monkeypatch.setattr(prefix.run,'verify_artifacts',changed)
    with pytest.raises((ValueError,AssertionError)):compare(values)


@pytest.mark.parametrize('fault',['negative','early','model','already_executed','qualified'])
def test_ineligible_prospective_receipt_cannot_define_a_native_boundary(tmp_path,monkeypatch,fault):
    _,_,report,_,_,_=setup(tmp_path,monkeypatch)
    if fault=='negative':report['budget_only_preboundary_decisions_supported']=False
    elif fault=='early':report['frames']-=1
    elif fault=='model':report['identities']['final_model_sha256'][1]='b'*64
    elif fault=='already_executed':report['changed_command_executed']=True
    else:report['navigation_qualified']=True
    with pytest.raises(ValueError):prefix.boundary(report)


def test_early_native_failures_are_accounted_without_claiming_a_physical_prefix(tmp_path,monkeypatch):
    _,_,report,_,_,_=setup(tmp_path,monkeypatch)
    full=dict(decisions=5,command_ticks=5,completed_ticks=5,physics_samples=1000)
    assert prefix.prefix_availability(full,report)['boundary_interval_available_by_collection_counts']
    for changed,reason in [({'decisions':4,'command_ticks':4,'completed_ticks':4},'INTERVENTION_OBSERVATION_NOT_REACHED'),
        ({'command_ticks':4,'completed_ticks':4},'INTERVENTION_COMMAND_NOT_ISSUED'),
        ({'completed_ticks':4},'INTERVENTION_COMMAND_INCOMPLETE'),({'physics_samples':999},'INSUFFICIENT_PHYSICAL_SAMPLES')]:
        result=prefix.prefix_availability(full|changed,report)
        assert result['unavailable_reason']==reason and not result['actual_physical_prefix_reconstructed']
