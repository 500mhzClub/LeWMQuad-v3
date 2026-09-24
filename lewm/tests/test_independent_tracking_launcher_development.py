"""Source-only launcher tests; workers and native audits are synthetic fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import run_go2_independent_tracking_challenge_v1 as mod
from scripts import navigation_artifact_root_development as custody
from scripts import independent_tracking_evaluation_development as evaluation
from scripts import independent_tracking_cohort_development as base
from lewm.tests.test_independent_tracking_cohort_development import episode
from lewm.tests.test_independent_pulse_parallel_science_development import complete as complete_parallel_study
from lewm.tests.test_independent_pulse_parallel_study_development import small


def save(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base.encode(value))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_exact_worker_cli_and_full_output_envelope():
    r=mod.resource_contract()
    assert r['streams']==192 and r['metadata_files']==48
    assert r['worst_case_bound_bytes']/1024**3==51.8828125
    assert r['worst_case_bound_bytes']<r['total_bytes']==52*1024**3
    assert r['combined_challenge_and_supervision_bound_bytes']==r['worst_case_bound_bytes']+96*1024**2
    assert r['combined_challenge_and_supervision_bound_bytes']<r['total_bytes']
    assert not r['memory_scope']['workload_fit_proved']
    assert r['episode_allocation']['all_reservations_bytes']<=5*1024**3
    assert not r['episode_allocation']['native_recording_bound_proved']
    command=mod.command(mod.TRIALS[0],'a'*64,'b'*64)
    assert command==[str(mod.PYTHON),mod.SOURCE,'--worker-trial',mod.TRIALS[0],
        '--launch-sha256','a'*64,'--request-sha256','b'*64]
    with pytest.raises(ValueError):mod.command('../trial','a'*64,'b'*64)


@pytest.fixture
def review(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    monkeypatch.setattr(custody,'BASE',tmp_path)
    probe_roots={m:tmp_path/f'go2_resource_{m}_fixture_attempt_001' for m in ('fit','overflow')}
    monkeypatch.setattr(mod.memory,'PROBE_OUTPUTS',probe_roots)
    sources={n:save(tmp_path,n,{'synthetic':'recorder'}) for n in mod.REVIEWED_RECORDERS}
    proof_sources={n:sources[n] for n in ('scripts/tracking_kernel_scope_development.py',
        'scripts/independent_tracking_memory_supervision_development.py')}
    freeze={};terminals={};fit_sha=None
    for mode,root in probe_roots.items():
        root.mkdir()
        definition=dict(mode=mode,unit=mod.memory.kernel.PROBE_UNITS[mode],
            profile=mod.memory.kernel.profile(mod.memory.kernel.PROBE_UNITS[mode]),source_sha256=proof_sources)
        identity=hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        freeze[mode+'_definition_sha256']=identity
        request_sha=save(root,'request.json',dict(definition=definition,definition_sha256=identity,
            prior_fit_terminal_sha256=fit_sha))
        log_sha=save(root,'unit.log',dict(synthetic='not actual service evidence'))
        status='TINY_KEEPER_FIT_VERIFIED' if mode=='fit' else 'TINY_KEEPER_GROUP_OOM_EVIDENCE_VERIFIED'
        sha=save(root,'terminal.json',dict(status=status,definition_sha256=identity,
            child_handle_terminal=True,log_complete=True,log_omitted_bytes=0,
            native_execution=False,native_workload_fit_proved=False,navigation_qualified=False,
            goal_achieved=False,retry_performed=False,output_sha256={'request.json':request_sha,'unit.log':log_sha}))
        terminals[str(root/'terminal.json')]=sha
        if mode=='fit':fit_sha=sha
    evidence={n:save(tmp_path,n,freeze if n==mod.RESOURCE_EVIDENCE_DOCS[2] else {'synthetic':'proof document'})
        for n in mod.RESOURCE_EVIDENCE_DOCS}
    row=dict(status='SOURCE_RECORDING_BOUNDS_AND_SCOPED_FAILURE_EVIDENCE_VERIFIED',maximum_physics_samples=mod.MAX_PHYSICS_SAMPLES,
        maximum_rgbd_frames=mod.MAX_FRAMES,
        maximum_deferred_serialized_bytes=mod.episode_resource_contract()['static_recording_serialization_ceiling_bytes'],
        maximum_combined_challenge_and_supervision_bytes=mod.resource_contract()['combined_challenge_and_supervision_bound_bytes'],
        memory_scope=mod.memory.contract(),workload_fit_proved=False,
        native_contact_source_sha256={'synthetic':'native binding'},
        source_sha256=sources,evidence='synthetic validation fixture, not a resource proof',
        evidence_document_sha256=evidence,tiny_probe_terminal_sha256=terminals)
    monkeypatch.setattr(mod,'native_source_bindings',lambda:{'synthetic':'native binding'})
    save(tmp_path,mod.RESOURCE_REVIEW,row)
    return tmp_path,row


def test_missing_review_cannot_be_replaced_by_a_default_or_initialize_native(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    def forbidden():raise AssertionError('no dependency or native initialization without reviewed bounds')
    monkeypatch.setattr(mod.learning,'definition',forbidden)
    with pytest.raises(ValueError,match='not complete'):mod.definition()


@pytest.mark.parametrize('fault',[None,'status','frames','samples','deferred','memory','source','evidence',
    'fit_claim','native_binding','old_review','evidence_doc','proof_terminal','fit_link','proof_promotion',
    'misstated_arithmetic'])
def test_resource_review_is_exact_bounded_and_source_bound(review,fault):
    root,row=review
    if fault=='status':row['status']='PENDING'
    elif fault=='frames':row['maximum_rgbd_frames']-=1
    elif fault=='samples':row['maximum_physics_samples']-=1
    elif fault=='deferred':row['maximum_deferred_serialized_bytes']=mod.DEFERRED_BYTES+1
    elif fault=='misstated_arithmetic':row['maximum_deferred_serialized_bytes']=1024
    elif fault=='memory':row['memory_scope']['memory_max_bytes']+=1
    elif fault=='source':row['source_sha256'][mod.REVIEWED_RECORDERS[0]]='0'*64
    elif fault=='evidence':row['evidence']=''
    elif fault=='fit_claim':row['workload_fit_proved']=True
    elif fault=='native_binding':row['native_contact_source_sha256']={}
    elif fault=='old_review':
        row['status']='SOURCE_AND_RECORDING_BOUNDS_VERIFIED';row['maximum_worker_retained_bytes']=2048
    elif fault=='evidence_doc':save(root,mod.RESOURCE_EVIDENCE_DOCS[0],{'synthetic':'changed proof'})
    elif fault=='proof_terminal':save(mod.memory.PROBE_OUTPUTS['fit'],'terminal.json',{'synthetic':'changed terminal'})
    elif fault in ('fit_link','proof_promotion'):
        proof=mod.memory.PROBE_OUTPUTS['overflow']
        terminal=json.loads((proof/'terminal.json').read_text())
        if fault=='fit_link':
            request=json.loads((proof/'request.json').read_text());request['prior_fit_terminal_sha256']='0'*64
            terminal['output_sha256']['request.json']=save(proof,'request.json',request)
        else:terminal['native_workload_fit_proved']=True
        row['tiny_probe_terminal_sha256'][str(proof/'terminal.json')]=save(proof,'terminal.json',terminal)
    save(root,mod.RESOURCE_REVIEW,row)
    if fault:
        with pytest.raises(ValueError):mod.native_resource_review()
    else:assert mod.native_resource_review()==row


def test_owned_scoped_process_detection_does_not_return_unrelated_command_lines(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    entries=[]
    for pid,args in [(101,['python',mod.LEARNING_JOBS[0]]),
                     (102,['python',str(tmp_path/mod.LEARNING_JOBS[-1])]),
                     (103,['python','unrelated.py','private_argument'])]:
        p=tmp_path/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(v.encode() for v in args)+b'\0')
        (p/'cwd').symlink_to(tmp_path,target_is_directory=True)
        entries.append(NS(name=str(pid),path=str(p)))
    original=mod.os.scandir
    monkeypatch.setattr(mod.os,'scandir',lambda p: entries if p=='/proc' else original(p))
    assert mod.active_learning_jobs()==[dict(pid=101,script=mod.LEARNING_JOBS[0]),dict(pid=102,script=mod.LEARNING_JOBS[-1])]
    with pytest.raises(ValueError,match='running'):mod.scheduling_check()


@pytest.fixture
def completed(tmp_path,monkeypatch):
    monkeypatch.setattr(custody,'BASE',tmp_path)
    study=tmp_path/'go2_learning_fixture_attempt_001';study.mkdir()
    sequence=tmp_path/'go2_sequence_fixture_attempt_001';sequence.mkdir()
    monkeypatch.setattr(mod.study,'OUTPUT',study);monkeypatch.setattr(mod.learning,'SEQUENCE',sequence)
    monkeypatch.setattr(mod,'verify_ordered_launch',lambda _:None)
    monkeypatch.setattr(mod.learning,'identity',lambda _:mod.LEARNING_DEFINITION)
    launch_sha=save(sequence,'launch.json',{'synthetic':True})
    monkeypatch.setattr(mod.learning,'SEQUENCE_LAUNCH',launch_sha)
    terminal=dict(status='ALL12_FIXED_RGB_BODY_BATCH_RECEIPTS_VERIFIED',completed_batches=list(mod.learning.BATCHES),
        planned_layouts=12,planned_episodes=1440,output_sha256={})
    sequence_sha=save(sequence,'result.json',terminal)
    d={'synthetic':'source definition'}
    study_launch=save(study,'launch.json',dict(definition=d,definition_sha256=mod.LEARNING_DEFINITION,
        sequence_receipt={'launch.json':launch_sha,'result.json':sequence_sha}))
    fits=[f'seed_{seed}_{v}_{c}' for seed in mod.learning.SEEDS for v in mod.learning.VARIANTS for c in mod.learning.CONDITIONS]
    r=dict(status='MATCHED_DEVELOPMENT_COMPARISON_COMPLETE',execution_revision='parallel.v1',completed_fits=fits,fits=36,optimizer_updates=43200,
        output_sha256={'launch.json':study_launch},final_evaluation=False,checkpoint_selection_performed=False,
        navigation_qualified=False,hardware_qualified=False,goal_achieved=False)
    # Lightweight delegation fixture; the separate integration test below uses
    # the real reader against all generated synthetic worker artifacts.
    def authenticate(sha,definition_sha):
        assert definition_sha==mod.LEARNING_DEFINITION
        mod.verify_artifacts(study,{'result.json':sha})
        result=base.read(study,'result.json');mod.verify_artifacts(study,result['output_sha256'])
        base.require(not (study/'failure.json').exists(),'failed study')
        return result,base.read(study,'launch.json')
    monkeypatch.setattr(mod.study_reader,'authenticate',authenticate)
    return study,r


@pytest.mark.parametrize('fault',[None,'partial_fits','updates','failure_file','promoted','changed_launch','sequential_revision'])
def test_only_complete_parallel_development_study_releases_scheduling_gate(completed,fault):
    study,r=completed
    if fault=='partial_fits':r['completed_fits'].pop();r['fits']=35
    elif fault=='updates':r['optimizer_updates']-=1
    elif fault=='failure_file':save(study,'failure.json',{'synthetic':'failure'})
    elif fault=='promoted':r['navigation_qualified']=True
    elif fault=='changed_launch':save(study,'launch.json',{'synthetic':'corrupted'})
    elif fault=='sequential_revision':r['execution_revision']='sequential.v1'
    sha=save(study,'result.json',r)
    if fault:
        with pytest.raises(ValueError):mod.completed_learning(sha)
    else:
        row=mod.completed_learning(sha)
        assert row['completed_fits']==36 and row['optimizer_updates']==43200
        assert row['study_execution_revision']=='parallel.v1' and row['study_output_root']==str(study)


def test_real_parallel_reader_integration_authenticates_complete_generated_synthetic_artifacts(complete_parallel_study,monkeypatch):
    result_sha,definition_sha=complete_parallel_study
    monkeypatch.setattr(mod,'LEARNING_DEFINITION',definition_sha)
    row=mod.completed_learning(result_sha)
    assert row['completed_fits']==36 and row['optimizer_updates']==36*mod.learning.UPDATES
    assert row['study_definition_sha256']==definition_sha and row['study_result_sha256']==result_sha
    assert row['study_output_root']==str(mod.study.OUTPUT)
    save(mod.study.OUTPUT,'seed_2026091101_full_direct_failure.json',{'synthetic':'failure'})
    with pytest.raises(ValueError,match='failed'):mod.completed_learning(result_sha)


def test_original_collection_failure_rejects_before_reading_study(completed,monkeypatch):
    save(mod.learning.SEQUENCE,'failure.json',{'synthetic':True})
    def forbidden(*args):raise AssertionError('no study access after known collection failure')
    monkeypatch.setattr(mod.study_reader,'authenticate',forbidden)
    with pytest.raises(ValueError,match='collector'):mod.completed_learning('a'*64)


def test_both_original_and_parallel_learning_jobs_are_excluded_from_native_overlap():
    assert mod.learning.SOURCE in mod.LEARNING_JOBS and mod.study.SOURCE in mod.LEARNING_JOBS
    assert len(mod.LEARNING_JOBS)==len(set(mod.LEARNING_JOBS))


@pytest.fixture
def launchable(tmp_path,monkeypatch):
    monkeypatch.setattr(custody,'BASE',tmp_path)
    output=tmp_path/'go2_launcher_fixture_attempt_001'
    monkeypatch.setattr(mod,'OUTPUT',output);monkeypatch.setattr(mod,'BASE',tmp_path)
    monkeypatch.setattr(mod,'runtime_check',lambda:None)
    monkeypatch.setattr(mod.memory,'own_scope',lambda:dict(synthetic=True))
    monkeypatch.setattr(mod.memory,'admit_request',lambda *a:dict(request_sha256='c'*64))
    monkeypatch.setattr(mod,'verify_ordered_launch',lambda _:None)
    d=dict(source_sha256={mod.PROTOCOL:'a'*64},synthetic=True)
    requested=[];native=[];comparison=[]
    def run(cmd,**kwargs):
        trial=cmd[cmd.index('--worker-trial')+1];requested.append(trial)
        launch_sha=cmd[cmd.index('--launch-sha256')+1];request_sha=cmd[cmd.index('--request-sha256')+1]
        r,receipt=episode(output,trial,frames=0)
        mod._save_worker_receipt(output,trial,dict(trial=trial,launch_sha256=launch_sha,
            request_sha256=request_sha,result=r,receipt=receipt))
        return NS(returncode=0)
    monkeypatch.setattr(mod.subprocess,'run',run)
    def audit(out,trial,result,protocol):
        assert len(base.read(out,'stress_sensor_phase_complete.json')['reports'])==8
        native.append(trial)
        return dict(timestamp_s=np.array([.002]),base_pose_world=np.array([[0.,0.,.375,0.,0.,0.,1.]])),dict(
            coverage=dict(intended_motion_covered=False),sensors=dict(depth_checks=[]),synthetic=True)
    monkeypatch.setattr(evaluation,'_raw_audit',audit)
    def compare(out,c,b,s,r):
        assert base.read(out,'result.json')['status']=='EIGHT_TRIAL_BASE_AND_FIXED_STRESS_RAW_AUDIT_AND_SCORING_COMPLETE'
        comparison.append(r)
        return dict(status='SYNTHETIC_UNAVAILABLE_PREFIXES',all_six_predecessor_nonidentity_checks_pass=False,
                    independent_observations_verified=False,navigation_qualified=False)
    monkeypatch.setattr(mod,'compare_new_population',compare)
    return output,d,requested,native,comparison,run


def test_full_orchestration_keeps_order_phase_separation_and_bound_final_artifacts(launchable):
    output,d,requested,native,comparison,_=launchable
    result=mod.execute(d,dict(study_result_sha256='b'*64),dict(request_sha256='c'*64))
    assert requested==native==list(mod.TRIALS) and len(comparison)==1
    assert result['status']=='NATIVE_TRACKING_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE'
    assert not result['full_challenge_pass'] and not result['goal_achieved']
    assert not result['independent_result_verification_complete']
    custody.verify_artifacts(output,result['output_sha256'])
    # The 240-path allowance includes failure.json (absent on success) and
    # challenge_result.json (cannot bind itself). Neither is a missing output.
    assert len(result['output_sha256'])==238
    assert 'failure.json' not in result['output_sha256']
    assert 'challenge_result.json' not in result['output_sha256']
    assert {p.name for p in output.iterdir() if p.is_file()} == (
        set(result['output_sha256']) | {'challenge_result.json'})
    for i,trial in enumerate(mod.TRIALS):
        req=base.read(output,trial+'_worker_request.json')
        expected=None if not i else mod.digest(output/(mod.TRIALS[i-1]+'_receipt.json'))
        assert req['previous_receipt_sha256']==expected
    with pytest.raises(ValueError):mod.execute(d,dict(study_result_sha256='b'*64),dict(request_sha256='c'*64))


@pytest.mark.parametrize('fault',['worker_exit','missing_report','wrong_report'])
def test_native_worker_failure_keeps_prior_evidence_and_never_starts_later_trial(launchable,monkeypatch,fault):
    output,d,requested,native,comparison,original=launchable
    def failed(cmd,**kwargs):
        trial=cmd[cmd.index('--worker-trial')+1]
        if trial!=mod.TRIALS[1]:return original(cmd,**kwargs)
        requested.append(trial)
        if fault=='wrong_report':mod._save_worker_receipt(output,trial,dict(trial=trial,launch_sha256='0'*64,request_sha256='0'*64))
        return NS(returncode=1 if fault=='worker_exit' else 0)
    monkeypatch.setattr(mod.subprocess,'run',failed)
    with pytest.raises(ValueError):mod.execute(d,dict(study_result_sha256='b'*64),dict(request_sha256='c'*64))
    assert requested==list(mod.TRIALS[:2]) and native==comparison==[]
    assert (output/(mod.TRIALS[0]+'_receipt.json')).is_file()
    assert not (output/'collection_complete.json').exists() and not (output/'challenge_result.json').exists()
    assert base.read(output,'failure.json')['stage']=='native_launcher:'+mod.TRIALS[1]


def test_preflight_rejects_occupied_scheduler_before_any_output_or_definition(launchable,monkeypatch):
    output,_,_,_,_,_=launchable
    def busy():raise ValueError('learning jobs running')
    def forbidden(*a):raise AssertionError('no definition or native allocation before scheduling')
    monkeypatch.setattr(mod,'runtime_check',busy);monkeypatch.setattr(mod,'definition',forbidden)
    with pytest.raises(ValueError,match='running'):mod.preflight('a'*64,'b'*64)
    assert not output.exists()


def test_worker_cannot_create_episode_without_matching_bound_request(launchable,monkeypatch):
    output,d,_,_,_,_=launchable;output.mkdir()
    monkeypatch.setattr(mod,'definition',lambda:d);monkeypatch.setattr(mod,'completed_learning',lambda _:None)
    launch_sha=save(output,'launch.json',dict(definition=d,definition_sha256=mod.learning.identity(d),
        completed_learning=dict(study_result_sha256='b'*64),memory_supervision=dict(request_sha256='c'*64)))
    trial=mod.TRIALS[0]
    request=dict(trial=trial,launch_sha256=launch_sha,specification=mod.specification(trial),
        protocol_sha256='a'*64,previous_receipt_sha256='incorrect')
    request_sha=save(output,trial+'_worker_request.json',request)
    def forbidden(*a):raise AssertionError('no native collection before request admission')
    monkeypatch.setattr(mod,'collect_episode',forbidden)
    with pytest.raises(ValueError,match='request'):mod.worker(trial,launch_sha,request_sha)
    assert not (output/trial).exists()
