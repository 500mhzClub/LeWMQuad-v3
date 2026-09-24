"""Receipt-bound fresh processes for prospective family episodes and raw audits."""
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
from lewm.geometry_progress_layout_family_development import TRIALS,assignments,specification,cohort_contract
from scripts.geometry_progress_family_episode_development import collect,artifacts,RESERVE
from scripts.geometry_progress_family_audit_development import audit_condition
from scripts.navigation_artifact_root_development import BASE,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

PILOT=BASE/'go2_geometry_progress_height_union_v1_attempt_001'
PILOT_IDS={'launch.json':'321eecd390175d99fd8da4535c401102e8101013e2a04fff037356da34a2f50d',
    'result.json':'46be655f7af866b024367945089907d11ddc0f06f8fef9766565d69ea2c4bb54',
    'near_field_audit.json':'1cabe4b09590d5c409ba107ece18df17672e686f2f107feff304e30ea7182112'}


def preflight(*,output,protocol,seed_paths,planned_trials,workers,storage_bytes):
    validate_root(output,must_exist=False)
    if output.exists() or output.is_symlink():raise ValueError('exclusive fresh family root required')
    if workers not in (1,4) or not planned_trials or len(set(planned_trials))!=len(planned_trials) or not set(planned_trials)<=set(TRIALS):
        raise ValueError('bounded explicit unique family worker plan required')
    verify_artifacts(PILOT,PILOT_IDS);old=read_json(PILOT,'launch.json');verify(old)
    audit=read_json(PILOT,'near_field_audit.json');collection=read_json(PILOT,'result.json')
    if audit['audited_episodes']!=24 or not audit['gate']['prediction_design_and_measurement_gate_pass']:
        raise ValueError('complete passing prospective pilot required')
    verify_artifacts(PILOT,collection['artifact_sha256']|audit['output_sha256'])
    sources=discover_sources((protocol,*seed_paths),old['source_sha256'])
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules')}
    resources=hardware()
    if resources['memory_available_bytes']<32*1024**3:raise ValueError('32GiB available RAM required')
    if resources['artifact_free_bytes']<RESERVE+storage_bytes:raise ValueError('planned storage plus40GiB reserve required')
    launch.update(source_sha256=sources,protocol=protocol,pilot_sha256=PILOT_IDS,output_root=str(output),
        planned_trials=list(planned_trials),conditions={c:specification(c) for c in planned_trials},
        randomized_assignment=assignments(),cohort_contract=cohort_contract(),hardware=resources,
        native_scene_workers=workers,maximum_tasks_per_process=1,opencv_threads=1,blas_threads=1,
        planned_storage_bytes=storage_bytes,minimum_free_bytes=RESERVE,model_training=False,
        physics_paused_during_compute=True,real_time_qualified=False,
        full_inherited_input_verification='before and after every episode; before and after cohort',
        navigation_qualified=False,goal_achieved=False)
    verify(launch);return launch


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
    terminal=dict(trial=trial,status='FAMILY_WORKER_FAILED',failure=None)
    # Each worker owns one explicitly assigned episode and three root-level
    # receipts. Existing claims fail before a constructor or old output access.
    with (output/(trial+'_worker.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            verify_artifacts(output,{'launch.json':request['launch_sha256']})
            launch=read_json(output,'launch.json')
            if (launch['output_root']!=str(output) or trial not in launch['planned_trials']
                    or launch['conditions'][trial]!=specification(trial)):
                raise ValueError('worker must match exact prospective launch assignment')
            verify(launch)
            if shutil.disk_usage(BASE.parent).free<RESERVE+200*1024**2:raise ValueError('next episode storage reserve')
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
            terminal.update(status='FAMILY_EPISODE_COLLECTED_AND_AUDITED',collection=collection,
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
                        except Exception as error:row=dict(trial=trial,status='FAMILY_WORKER_PROCESS_FAILURE',failure=repr(error))
                        records[trial]=row
                        print('FAMILY_WORKER',output.name,trial,row['status'],row.get('wall_s'),row.get('hard_measurement_failed_frames'),flush=True)
                if any(records[t]['status']!='FAMILY_EPISODE_COLLECTED_AND_AUDITED' for t in batch):break
    return dict(wall_s=time.perf_counter()-start,workers=workers,records=records,
        completed_trials=[t for t in planned_trials if t in records],unlaunched_trials=[t for t in planned_trials if t not in records],
        all_workers_completed=bool(len(records)==len(planned_trials) and all(r['status']=='FAMILY_EPISODE_COLLECTED_AND_AUDITED' for r in records.values())))
