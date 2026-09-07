"""One complete native tracking challenge, after the parallel matched study.

Draft launcher: an exact source-bound native resource review is REQUIRED and
is not created by this program. No retry/resume or controller adoption. Each
native tape runs in a fresh process; all sensor replays precede native scoring.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from lewm.independent_tracking_challenge_development import TRIALS,specification,MAX_FRAMES,MAX_PHYSICS_SAMPLES
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts.independent_tracking_predecessor_comparison_development import compare_new_population
from scripts.independent_tracking_artifacts_development import (
    EpisodeStore,EPISODE_BYTES,DEFERRED_BYTES,episode_resource_contract)
from scripts.independent_tracking_collection_development import collect_episode
from scripts import run_go2_independent_pulse_matched_study_v1 as learning
from scripts import run_go2_independent_pulse_parallel_study_v1 as study
from scripts import read_go2_independent_pulse_parallel_science_v1 as study_reader
from scripts.startup_source_inventory_development import discover_sources,allowed_relative
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify_ordered_source
from scripts.independent_tracking_native_contact_guard_development import native_source_bindings
from scripts import independent_tracking_memory_supervision_development as memory
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts,artifact_path

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.generated/venvs/genesis_rocm_0_4_6_v1/bin/python'
SOURCE='scripts/run_go2_independent_tracking_challenge_v1.py'
TEST='lewm/tests/test_independent_tracking_launcher_development.py'
PROTOCOL='docs/go2_independent_tracking_challenge_v1_2026-09-07.md'
RESOURCE_REVIEW='docs/go2_independent_tracking_native_resource_review_v1_2026-09-07.json'
OUTPUT=BASE/'go2_independent_tracking_challenge_v1_attempt_001'
LEARNING_DEFINITION='8d8c3456054a284aa83031ea417d8c433beddbcc04a8b47d3164f120bc0ae5d8'
ENVIRONMENT=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONPATH='.:lewm_genesis:lewm_worlds',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',
    PYOPENGL_PLATFORM='egl',EGL_DEVICE_ID='2')
REVIEWED_RECORDERS=('scripts/independent_tracking_collection_development.py',
    'scripts/independent_tracking_session_development.py','scripts/independent_tracking_snapshot_development.py',
    'scripts/independent_tracking_artifacts_development.py',
    'scripts/independent_tracking_native_contact_guard_development.py',
    'lewm/independent_tracking_recording_budget_development.py',
    'scripts/independent_tracking_memory_supervision_development.py',
    'scripts/supervise_go2_independent_tracking_challenge_v1.py',
    'scripts/tracking_kernel_scope_development.py',SOURCE)
RESOURCE_EVIDENCE_DOCS=(
    'docs/go2_independent_tracking_recording_bounds_and_memory_scope_result_2026-09-07.md',
    'docs/go2_independent_tracking_native_contact_guard_result_2026-09-07.md',
    'docs/go2_tracking_keeper_memory_probe_v1_frozen_definitions_2026-09-07.json',
    'docs/go2_tracking_keeper_memory_probe_v1_result_2026-09-07.md')
LEARNING_JOBS=('scripts/run_go2_independent_rgb_body_remaining_stages_v1.py',
    'scripts/run_go2_independent_rgb_body_stage_v1.py','scripts/run_go2_independent_rgb_body_collection_v1.py',
    'scripts/audit_go2_independent_rgb_body_collection_v1.py',learning.SOURCE,study.SOURCE)
EXTRA={'predecessor_comparison.json','challenge_result.json'} | {
    t+s for t in TRIALS for s in ('_worker_request.json','_worker_exit.json','_worker_receipt.json')}


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_ordered_launch(d):
    verify_ordered_source(d)
    if d.get('schema')=='independent_tracking_complete_native_challenge.v1':
        base.require(d.get('native_contact_source_sha256')==native_source_bindings(),
                     'exact reviewed native contact implementation bindings required')


def resource_contract():
    r=stress.resource_contract();extra=len(EXTRA)
    bound=r['worst_case_bound_bytes']+extra*base.MAX_METADATA
    combined=bound+memory.TOTAL_BYTES
    base.require(combined<=stress.TOTAL_BYTES,'challenge plus outside evidence must fit declared total cap')
    return r|dict(metadata_files=r['metadata_files']+extra,worst_case_bound_bytes=bound,
        episode_allocation=episode_resource_contract(),
        native_worker_processes=len(TRIALS),simultaneous_native_workers=1,
        minimum_available_memory_bytes=16*1024**3,memory_scope=memory.contract(),
        combined_challenge_and_supervision_bound_bytes=combined)


class ChallengeStore(stress.StressCohortStore):
    def __init__(self,output):
        super().__init__(output)
        base.require(not any((self.output/n).exists() or (self.output/n).is_symlink() for n in EXTRA),
                     'fresh launcher outputs only; no retry/resume')
        self.allowed |= EXTRA
        r=resource_contract()
        base.require(len(self.allowed)==r['streams']+r['metadata_files'],'exact full launcher roster required')


def native_resource_review():
    path=ROOT/allowed_relative(RESOURCE_REVIEW)
    base.require(path.is_file() and path.resolve()==path,
        'native recording/memory resource review is not complete; no native launch')
    base.require(path.stat().st_size<=base.MAX_METADATA,'bounded resource review required')
    review=json.loads(path.read_text())
    base.require(review['status']=='SOURCE_RECORDING_BOUNDS_AND_SCOPED_FAILURE_EVIDENCE_VERIFIED'
        and review['maximum_physics_samples']==MAX_PHYSICS_SAMPLES
        and review['maximum_rgbd_frames']==MAX_FRAMES
        and type(review['maximum_deferred_serialized_bytes']) is int
        and review['maximum_deferred_serialized_bytes']==episode_resource_contract()['static_recording_serialization_ceiling_bytes']
        and review['maximum_deferred_serialized_bytes']<=DEFERRED_BYTES
        and review.get('maximum_combined_challenge_and_supervision_bytes')==resource_contract()['combined_challenge_and_supervision_bound_bytes']
        and review.get('memory_scope')==memory.contract()
        and review.get('native_contact_source_sha256')==native_source_bindings()
        and review.get('workload_fit_proved') is False,
        'review must establish recording ceilings and scoped failure evidence, not invent a workload-fit proof')
    base.require(set(review['source_sha256'])==set(REVIEWED_RECORDERS), 'exact reviewed recorder source set')
    for name,sha in review['source_sha256'].items():
        base.require(digest(ROOT/allowed_relative(name))==sha,'native resource review recorder source changed')
    base.require(type(review['evidence']) is str and bool(review['evidence'].strip()), 'resource proof evidence required')
    verify_resource_evidence(review)
    return review


def verify_resource_evidence(review):
    """Reauthenticate the narrow proof chain, not a new probe or OOM experiment."""
    docs=review['evidence_document_sha256']
    base.require(type(docs) is dict and set(docs)==set(RESOURCE_EVIDENCE_DOCS), 'exact resource evidence documents')
    for name,sha in docs.items():
        path=ROOT/allowed_relative(name)
        base.require(path.resolve()==path and path.is_file() and digest(path)==sha,
            'resource evidence document changed: '+name)
    freeze=json.loads((ROOT/RESOURCE_EVIDENCE_DOCS[2]).read_text())
    terminals=review['tiny_probe_terminal_sha256']
    base.require(type(terminals) is dict and set(terminals)=={
        str(root/'terminal.json') for root in memory.PROBE_OUTPUTS.values()}, 'exact two completed tiny proof roots')
    common=None;fit_sha=None
    for mode in ('fit','overflow'):
        output=memory.PROBE_OUTPUTS[mode]
        sha=terminals[str(output/'terminal.json')]
        verify_artifacts(output,{'terminal.json':sha})
        result=base.read(output,'terminal.json')
        expected='TINY_KEEPER_FIT_VERIFIED' if mode=='fit' else 'TINY_KEEPER_GROUP_OOM_EVIDENCE_VERIFIED'
        base.require(result['status']==expected and result['child_handle_terminal'] is True
            and result['log_complete'] is True and result['log_omitted_bytes']==0
            and all(result[k] is False for k in ('native_execution','native_workload_fit_proved',
                'navigation_qualified','goal_achieved','retry_performed')),
            'complete limited tiny proof, never native fit or broader qualification')
        base.require(set(result['output_sha256'])=={'request.json','unit.log'}, 'exact tiny proof artifacts')
        verify_artifacts(output,result['output_sha256'])
        request=base.read(output,'request.json');definition=request['definition']
        identity=hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        base.require(identity==request['definition_sha256']==result['definition_sha256']==freeze[mode+'_definition_sha256']
            and definition['mode']==mode and definition['unit']==memory.kernel.PROBE_UNITS[mode]
            and definition['profile']==memory.kernel.profile(definition['unit']), 'same preregistered tiny source/config proof')
        sources=definition['source_sha256']
        for name,h in sources.items():
            path=ROOT/allowed_relative(name)
            base.require(path.resolve()==path and path.is_file() and digest(path)==h, 'tiny proof source changed: '+name)
        for name in ('scripts/tracking_kernel_scope_development.py','scripts/independent_tracking_memory_supervision_development.py'):
            base.require(sources.get(name)==review['source_sha256'][name], 'native scope must use the tested shared implementation')
        if mode=='fit':common=sources;fit_sha=sha
        else:
            base.require(sources==common and request['prior_fit_terminal_sha256']==fit_sha,
                'same-source completed fit must precede the single overflow proof')


def definition():
    """Read-only freeze candidate; no native initialization, checkpoint or fit."""
    review=native_resource_review()
    old=study.definition()
    base.require(learning.identity(old)==LEARNING_DEFINITION,'parallel matched-study definition changed')
    d={k:deepcopy(old[k]) for k in ('input_sha256','native_sha256','native_scene_sha256',
        'native_geometry_sha256','opencv_version','opencv_binary_sha256','rules')}
    d.update(source_sha256=discover_sources((SOURCE,TEST,PROTOCOL,RESOURCE_REVIEW,
        'scripts/supervise_go2_independent_tracking_challenge_v1.py',
        'lewm/tests/test_independent_tracking_memory_supervision_development.py',
        'docs/go2_independent_tracking_actual_predecessor_witnesses_2026-09-07.json'),old['source_sha256']),
        schema='independent_tracking_complete_native_challenge.v1',output_root=str(OUTPUT),
        native_contact_source_sha256=native_source_bindings(),
        trials=list(TRIALS),specifications={t:specification(t) for t in TRIALS},
        environment=ENVIRONMENT,python=str(PYTHON),resources=resource_contract(),
        native_resource_review=review,learning_definition_sha256=LEARNING_DEFINITION,
        learning_execution_revision='parallel.v1',learning_output_root=str(study.OUTPUT),
        scientific_scope='fixed observation challenge; no tracker/controller adoption',
        retry_performed=False,navigation_qualified=False,goal_achieved=False)
    verify_ordered_launch(d)
    return d


def active_learning_jobs():
    """Check only owned /proc command metadata; never print other command lines."""
    active=[]
    for entry in os.scandir('/proc'):
        if not entry.name.isdecimal():continue
        path=Path(entry.path)
        try:
            if path.stat().st_uid!=os.getuid() or int(entry.name)==os.getpid():continue
            argv=(path/'cmdline').read_bytes().split(b'\0')
            args=[x.decode(errors='replace') for x in argv if x]
            for script in LEARNING_JOBS:
                if str(ROOT/script) in args or script in args and (path/'cwd').resolve()==ROOT:
                    active.append(dict(pid=int(entry.name),script=script));break
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return active


def scheduling_check():
    base.require(not active_learning_jobs(),'original learning collection/study still running; do not displace it')


def available_memory():
    with Path('/proc/meminfo').open() as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise ValueError('actual available-memory witness required')


def completed_learning(result_sha256):
    # Reuse the distinct reader's complete artifact/job/ledger/source gate,
    # without aggregating scores, loading checkpoints or rerunning training.
    base.require(not (learning.SEQUENCE/'failure.json').exists()
        and not (learning.SEQUENCE/'failure.json').is_symlink(), 'failed original collector cannot release tracking')
    result,launch=study_reader.authenticate(result_sha256,LEARNING_DEFINITION)
    expected=[f'seed_{seed}_{v}_{c}' for seed in learning.SEEDS for v in learning.VARIANTS for c in learning.CONDITIONS]
    base.require(result['status']=='MATCHED_DEVELOPMENT_COMPARISON_COMPLETE'
        and result['execution_revision']=='parallel.v1' and result['completed_fits']==expected
        and result['fits']==36 and result['optimizer_updates']==36*learning.UPDATES,
        'complete parallel36-fit study required, not a partial cohort or failed attempt')
    base.require(all(result[k] is False for k in ('final_evaluation','checkpoint_selection_performed',
        'navigation_qualified','hardware_qualified','goal_achieved')), 'unchanged development-only learning claims required')
    return dict(study_result_sha256=result_sha256,study_launch_sha256=result['output_sha256']['launch.json'],
                sequence_receipt=launch['sequence_receipt'],completed_fits=36,
                optimizer_updates=result['optimizer_updates'],study_execution_revision='parallel.v1',
                study_definition_sha256=LEARNING_DEFINITION,study_output_root=str(study.OUTPUT))


def runtime_check():
    base.require(Path.cwd()==ROOT and str(sys.executable)==str(PYTHON)
        and all(os.environ.get(k)==v for k,v in ENVIRONMENT.items()),'exact reviewed CPU environment required')
    base.require(available_memory()>=resource_contract()['minimum_available_memory_bytes'],
                 'available-memory launch reserve exhausted')
    scheduling_check()


def preflight(definition_sha256,study_result_sha256):
    runtime_check();completed=completed_learning(study_result_sha256);d=definition()
    base.require(learning.identity(d)==definition_sha256,'exact reviewed challenge source/config definition required')
    validate_root(OUTPUT,must_exist=False)
    base.require(not OUTPUT.exists() and not OUTPUT.is_symlink(),'exclusive challenge attempt; no retry/resume')
    r=resource_contract()
    base.require(shutil.disk_usage(BASE).free>=r['reserve_bytes']+r['total_bytes'],'whole challenge storage plus reserve required')
    return d,completed


def command(trial,launch_sha256,request_sha256):
    base.require(trial in TRIALS,'fixed worker trial required')
    return [str(PYTHON),SOURCE,'--worker-trial',trial,'--launch-sha256',launch_sha256,'--request-sha256',request_sha256]


def _save_worker_receipt(output,trial,value):
    """One exact child-owned root file; no generic writer or metadata resume."""
    validate_root(output);base.require(trial in TRIALS,'exact worker receipt trial')
    path=output/(trial+'_worker_receipt.json');payload=base.encode(value)
    base.require(path.resolve()==path and not path.exists() and len(payload)<=base.MAX_METADATA,
                 'exclusive bounded worker receipt required')
    base.require(shutil.disk_usage(output).free>=base.RESERVE_BYTES+len(payload),'worker receipt storage reserve')
    with path.open('xb') as stream:
        if stream.write(payload)!=len(payload):raise OSError('short worker receipt write')
        stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def worker(trial,launch_sha256,request_sha256):
    memory.own_scope();runtime_check();validate_root(OUTPUT)
    request_name=trial+'_worker_request.json'
    verify_artifacts(OUTPUT,{'launch.json':launch_sha256,request_name:request_sha256})
    launch=base.read(OUTPUT,'launch.json');request=base.read(OUTPUT,request_name);d=definition()
    base.require(launch['definition']==d and launch['definition_sha256']==learning.identity(d), 'frozen worker source/config required')
    memory.admit_request(launch['memory_supervision']['request_sha256'],learning.identity(d),
        launch['completed_learning']['study_result_sha256'])
    completed_learning(launch['completed_learning']['study_result_sha256'])
    index=TRIALS.index(trial)
    prior=None if not index else digest(artifact_path(OUTPUT,TRIALS[index-1]+'_receipt.json'))
    base.require(request==dict(trial=trial,launch_sha256=launch_sha256,specification=specification(trial),
        protocol_sha256=d['source_sha256'][PROTOCOL],previous_receipt_sha256=prior), 'exact ordered worker request required')
    for previous in TRIALS[:index]:
        row=base.read(OUTPUT,previous+'_receipt.json');base.verify_episode(OUTPUT,previous,row['result'],row['receipt'])
    result,receipt=collect_episode(EpisodeStore(OUTPUT,trial),specification(trial),d['source_sha256'][PROTOCOL])
    verify_ordered_launch(d)
    _save_worker_receipt(OUTPUT,trial,dict(trial=trial,launch_sha256=launch_sha256,
        request_sha256=request_sha256,result=result,receipt=receipt))
    return 0 if result['status']=='TRACKING_TAPE_REQUIRES_RAW_AUDIT' else 1


def _adopt_worker_receipt(store,trial):
    name=trial+'_worker_receipt.json';path=artifact_path(store.output,name)
    base.require(name not in store.hashes and path.stat().st_size<=base.MAX_METADATA,'new bounded worker receipt')
    store.hashes[name]=digest(path);store.sizes[name]=path.stat().st_size;store.check(0)
    return base.read(store.output,name)


def execute(d,completed,supervision):
    """Called only after preflight; tests inject workers, never native simulation."""
    memory.own_scope()
    admitted=memory.admit_request(supervision['request_sha256'],learning.identity(d),completed['study_result_sha256'])
    base.require(admitted==supervision,'same freshly admitted scoped parent required')
    create_output(OUTPUT);store=ChallengeStore(OUTPUT);active=None
    try:
        launch_sha=store.save('launch.json',dict(definition=d,definition_sha256=learning.identity(d),
            completed_learning=completed,memory_supervision=supervision))
        for index,trial in enumerate(TRIALS):
            active=trial;memory.own_scope();runtime_check();verify_ordered_launch(d);store.check(EPISODE_BYTES)
            request=dict(trial=trial,launch_sha256=launch_sha,specification=specification(trial),
                protocol_sha256=d['source_sha256'][PROTOCOL],
                previous_receipt_sha256=None if not index else store.hashes[TRIALS[index-1]+'_receipt.json'])
            request_sha=store.save(trial+'_worker_request.json',request)
            print('INDEPENDENT_TRACKING_WORKER_START',trial,flush=True)
            process=subprocess.run(command(trial,launch_sha,request_sha),cwd=ROOT,env=os.environ|ENVIRONMENT,check=False)
            store.save(trial+'_worker_exit.json',dict(trial=trial,returncode=process.returncode))
            report=None
            if (store.output/(trial+'_worker_receipt.json')).exists():report=_adopt_worker_receipt(store,trial)
            base.require(process.returncode==0 and report is not None,'terminal native worker failure; no retry')
            base.require(report['trial']==trial and report['launch_sha256']==launch_sha
                and report['request_sha256']==request_sha,'exact worker receipt authority required')
            store.admit_episode(trial,report['result'],report['receipt'])
            print('INDEPENDENT_TRACKING_WORKER_ADMITTED',trial,flush=True)
        active=None;collection_sha=store.complete_collection()
        memory.own_scope()
        base_sha,_=base.replay_population(store,collection_sha)
        memory.own_scope()
        stress_sha,_=stress.replay_stress_population(store,collection_sha,base_sha)
        memory.own_scope()
        stress.evaluate_complete_population(store,collection_sha,base_sha,stress_sha,d['source_sha256'][PROTOCOL])
        result_sha=store.hashes['result.json']
        comparison=compare_new_population(store.output,collection_sha,base_sha,stress_sha,result_sha)
        comparison_sha=store.save('predecessor_comparison.json',comparison)
        memory.own_scope()
        verify_ordered_launch(d);verify_artifacts(store.output,store.hashes)
        terminal=dict(status='NATIVE_TRACKING_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE',
            definition_sha256=learning.identity(d),result_sha256=result_sha,comparison_sha256=comparison_sha,
            output_sha256=dict(store.hashes),resource_contract=resource_contract(),
            independent_result_verification_complete=False,full_challenge_pass=False,
            navigation_qualified=False,real_time_qualified=False,goal_achieved=False)
        store.save('challenge_result.json',terminal)
        return terminal
    except BaseException as error:
        base.record_phase_failure(store,'native_launcher:'+str(active),error)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--definition-sha256');parser.add_argument('--study-result-sha256')
    parser.add_argument('--worker-trial',choices=TRIALS);parser.add_argument('--launch-sha256');parser.add_argument('--request-sha256')
    parser.add_argument('--scoped-parent',action='store_true');parser.add_argument('--supervisor-request-sha256')
    args=parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    if args.worker_trial:
        base.require(args.launch_sha256 and args.request_sha256 and not args.definition_sha256
            and not args.study_result_sha256 and not args.scoped_parent
            and not args.supervisor_request_sha256,'exclusive exact worker CLI')
        return worker(args.worker_trial,args.launch_sha256,args.request_sha256)
    base.require(args.definition_sha256 and args.study_result_sha256 and not args.launch_sha256
        and not args.request_sha256 and args.scoped_parent and args.supervisor_request_sha256,
        'outside supervisor required; direct unbounded parent execution forbidden')
    memory.own_scope()
    d,completed=preflight(args.definition_sha256,args.study_result_sha256)
    supervision=memory.admit_request(args.supervisor_request_sha256,args.definition_sha256,args.study_result_sha256)
    execute(d,completed,supervision);return 0


if __name__=='__main__':raise SystemExit(main())
