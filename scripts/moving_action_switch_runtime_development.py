"""Fresh processes and strict batch gates for 144 moving-action branches."""
import contextlib
import json
import multiprocessing
from pathlib import Path
import resource
import shutil
import time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import cv2
import numpy as np
from lewm.moving_action_switch_family_development import TRIALS,assignments,specification,branch_specification,CANONICAL
from scripts.moving_action_switch_episode_development import collect,artifacts,RESERVE
from scripts.moving_action_switch_audit_development import audit_condition
from scripts.navigation_artifact_root_development import BASE,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify_sources
from scripts.geometry_progress_family_runtime_development import preflight as geometry_preflight
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

RATIONALE=BASE/'go2_executed_commitment_six_model_errors_v1_attempt_001'
RATIONALE_IDS={'launch.json':'c4de58bf409aaaaa0e6574b51912d3ddb95b9e1378cb9f5eb5f4e0f76a45b91e',
    'result.json':'6d3ce4bf199e9ca464c24d7a0b8e7ba19c8392198d58ac0f0152ff6b6a20065e'}
RATIONALE_REPORT='docs/go2_executed_commitment_six_model_errors_result_2026-09-08.md'
RATIONALE_REPORT_SHA='f3d473698594ab2140889d5a7dbdc8ab86528658ed21e80c8fc9038bf923cf03'


def verify(launch):
    verify_sources(launch)
    if launch.get('rationale_sha256')!=RATIONALE_IDS:raise ValueError('fixed diagnostic identity required')
    verify_artifacts(RATIONALE,RATIONALE_IDS)
    if digest(ROOT/RATIONALE_REPORT)!=RATIONALE_REPORT_SHA:raise ValueError('unchanged diagnostic report required')


def preflight(*,output,protocol,seed_paths,planned_trials,workers,storage_bytes):
    if workers not in (1,4) or not planned_trials or len(set(planned_trials))!=len(planned_trials) or not set(planned_trials)<=set(TRIALS):
        raise ValueError('bounded explicit unique moving-action worker plan required')
    verify_artifacts(RATIONALE,RATIONALE_IDS)
    if digest(ROOT/RATIONALE_REPORT)!=RATIONALE_REPORT_SHA:raise ValueError('frozen diagnostic report required')
    launch=geometry_preflight(output=output,protocol=protocol,seed_paths=(*seed_paths,RATIONALE_REPORT),
        planned_trials=list(CANONICAL.values()),workers=workers,storage_bytes=storage_bytes)
    launch.update(planned_trials=list(planned_trials),conditions={t:specification(t) for t in planned_trials},
        branch_conditions={t:branch_specification(t) for t in planned_trials},randomized_assignment=assignments(),
        rationale_sha256=RATIONALE_IDS,cohort_contract=dict(expected_cells=144,train_cells=72,geometry_transfer_cells=72,
            prefix_actions=6,suffix_actions=6,branch_tick=13,complete_commands=63,complete_frames=64,
            complete_physics_samples=3900,prefix_match='exact physical and public sensor arrays through branch',
            roles_share_no_cluster=True,independent_maze_evaluation_layouts=0),
        model_training=False,benchmark_outputs_are_training_data=False)
    verify(launch);return launch


def admissible_record(row):
    return bool(row['status']=='MOVING_ACTION_SWITCH_COLLECTED_AND_AUDITED'
        and not row['hard_measurement_failed_frames'] and row['strict_physical_visibility_pass']
        and row['collection']['acquisition_stop'] is None)


def signature(directory,result):
    """Array/pixel identities independent of ZIP timestamps and wall timing."""
    values={}
    for name in artifacts('',result):
        if name.endswith('.npz'):
            with np.load(directory/name,allow_pickle=False) as z:
                for key in z.files:values[name+'/'+key]=fingerprint(z[key])
    for i,c in enumerate(read_json(directory,'camera_audit.json')):values[f'rgb/{i}']=c['rgb_sha256']
    return values


def run_episode(request):
    output=validate_root(Path(request['output']));trial=request['trial'];start=time.perf_counter()
    if trial not in TRIALS or type(request['record_signature']) is not bool:raise ValueError('exact family worker request required')
    cv2.setNumThreads(1)
    terminal=dict(trial=trial,status='MOVING_ACTION_SWITCH_WORKER_FAILED',failure=None)
    # Each worker owns one explicitly assigned episode and three root-level
    # receipts. Existing claims fail before a constructor or old output access.
    with (output/(trial+'_worker.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            verify_artifacts(output,{'launch.json':request['launch_sha256']})
            launch=read_json(output,'launch.json')
            if (launch['output_root']!=str(output) or trial not in launch['planned_trials']
                    or launch['conditions'][trial]!=specification(trial)
                    or launch['branch_conditions'][trial]!=branch_specification(trial)):
                raise ValueError('worker must match exact prospective launch assignment')
            verify(launch)
            if shutil.disk_usage(BASE.parent).free<RESERVE+256*1024**2:raise ValueError('next episode storage reserve')
            collection=collect(trial,launch['source_sha256'][launch['protocol']],output=output)
            names=[trial+'/'+n for n in artifacts(trial,collection)]
            absent=[n for n in names if not (output/n).is_file()]
            if absent:raise ValueError('missing expected episode artifacts: '+repr(absent))
            bindings={n:digest(output/n) for n in names};verify_artifacts(output,bindings)
            report=audit_condition(trial,collection,launch['source_sha256'][launch['protocol']],input_root=output)
            report_name=trial+'_audit.json';write_json(output/report_name,report);bindings[report_name]=digest(output/report_name)
            if request['record_signature']:
                name=trial+'_signature.json';write_json(output/name,signature(output/trial,collection));bindings[name]=digest(output/name)
            verify(launch);verify_artifacts(output,bindings)
            terminal.update(status='MOVING_ACTION_SWITCH_COLLECTED_AND_AUDITED',collection=collection,
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                artifact_sha256=bindings,outcome=report['outcome'])
        except Exception as error:
            import traceback
            traceback.print_exc();terminal['failure']=repr(error)
        finally:
            terminal.update(wall_s=time.perf_counter()-start,maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    terminal['worker_log_sha256']=digest(output/(trial+'_worker.log'))
    write_json(output/(trial+'_worker_terminal.json'),terminal)
    return terminal


def run_phase(output,planned_trials,*,workers,record_signature):
    validate_root(output)
    if workers not in (1,4) or not planned_trials or len(set(planned_trials))!=len(planned_trials) or not set(planned_trials)<=set(TRIALS):
        raise ValueError('bounded unique phase required')
    launch_sha=digest(output/'launch.json');records={};start=time.perf_counter();last_monitor=0.
    with (output/'resource_monitor.jsonl').open('x') as monitor:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as executor:
            # Bounded batches ensure a failed worker cannot silently launch all
            # remaining episodes. Running siblings finish and retain receipts.
            for start_index in range(0,len(planned_trials),workers):
                batch=planned_trials[start_index:start_index+workers]
                pending={executor.submit(run_episode,dict(output=str(output),trial=t,launch_sha256=launch_sha,
                    record_signature=record_signature)):t for t in batch}
                while pending:
                    done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                    if time.perf_counter()-last_monitor>=15:
                        h=hardware();h['elapsed_s']=time.perf_counter()-start
                        monitor.write(json.dumps(h)+'\n');monitor.flush();last_monitor=time.perf_counter()
                    for future in done:
                        trial=pending.pop(future)
                        try:row=future.result()
                        except Exception as error:row=dict(trial=trial,status='MOVING_ACTION_SWITCH_WORKER_PROCESS_FAILURE',failure=repr(error))
                        records[trial]=row
                        print('FAMILY_WORKER',output.name,trial,row['status'],row.get('wall_s'),row.get('hard_measurement_failed_frames'),flush=True)
                if any(not admissible_record(records[t]) for t in batch):break
    return dict(wall_s=time.perf_counter()-start,workers=workers,records=records,
        completed_trials=[t for t in planned_trials if t in records],unlaunched_trials=[t for t in planned_trials if t not in records],
        all_workers_completed=bool(len(records)==len(planned_trials) and all(r['status']=='MOVING_ACTION_SWITCH_COLLECTED_AND_AUDITED' for r in records.values())),
        all_measurement_gates_pass=bool(len(records)==len(planned_trials) and all(admissible_record(r) for r in records.values())))

