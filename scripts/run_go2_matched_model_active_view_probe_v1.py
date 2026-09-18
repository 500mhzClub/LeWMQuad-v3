"""Twelve fresh native cases with matched trained model/head/input assignments."""
import argparse
import contextlib
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, wait
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.geometry_progress_family_runtime_development import preflight, verify
from scripts.matched_model_goal_episode_development import collect, artifacts
from scripts.matched_model_goal_audit_development import audit
from scripts.matched_family_model_admission_development import load_assigned
from scripts.family_transition_model_admission_development import admit
from scripts.run_go2_family_transition_fits_v1 import OUTPUT as FITS, ROSTER
from scripts.read_go2_active_view_six_model_transitions_v1 import OUTPUT as DIAGNOSTIC, FIT_SHA
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_matched_model_active_view_probe_v1_attempt_001'
DIAGNOSTIC_IDS={'launch.json':'f806c41d767e23185b8591709e82ad4cffecd218ab2a5a6965ec51a355fb621b',
    'result.json':'5d3cfc47b9d4a2cd0417b7987cb2b159df6a05a52888bb774185afb328e42657'}
PROTOCOL='docs/go2_matched_model_active_view_probe_v1_2026-09-08.md'
TRIALS=('family_episode_052','family_episode_039')
CASES=[(f'{v}_{c}_{t}',t,v,c,f'seed_2026091001_{v}_{c}') for v in ('full','no_rgb')
    for c in ('direct','supervised_rollout','jepa') for t in TRIALS]
random.Random(2026091101).shuffle(CASES)


def worker(case,launch_sha):
    name,trial,variant,condition,model_name=case
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    terminal=dict(case=name,trial=trial,variant=variant,condition=condition,model_name=model_name,
        status='MATCHED_MODEL_ACTIVE_VIEW_WORKER_FAILED',artifact_sha256={})
    started=time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT,{'launch.json':launch_sha})
            launch=read_json(OUTPUT,'launch.json');verify(launch)
            assert launch['planned_cases']==[list(c) for c in CASES] and tuple(case) in CASES
            assert launch['output_root']==str(OUTPUT) and digest(URDF)==launch['robot_urdf_sha256']
            model,c,v=load_assigned(launch,model_name);assert (c,v)==(condition,variant)
            before=state_digest(model.state_dict())
            result=collect(trial,launch['source_sha256'][PROTOCOL],output=OUTPUT,model=model,
                geometry=ArticulatedCollisionGeometry(URDF),persistent=True,episode_name=name,
                condition=condition,variant=variant)
            assert state_digest(model.state_dict())==before
            bindings={name+'/'+n:digest(OUTPUT/name/n) for n in artifacts(trial,result)}
            verify_artifacts(OUTPUT,bindings)
            replay_model,c,v=load_assigned(launch,model_name);assert (c,v)==(condition,variant)
            report=audit(trial,result,launch['source_sha256'][PROTOCOL],input_root=OUTPUT,
                model=replay_model,robot_geometry=ArticulatedCollisionGeometry(URDF),persistent=True,
                episode_name=name,condition=condition,variant=variant)
            audit_name=name+'_audit.json';write_json(OUTPUT/audit_name,report);bindings[audit_name]=digest(OUTPUT/audit_name)
            verify(launch);verify_artifacts(OUTPUT,bindings);verify_artifacts(FITS,launch['fit_artifact_sha256'])
            assert digest(URDF)==launch['robot_urdf_sha256']
            terminal.update(status='MATCHED_MODEL_ACTIVE_VIEW_COLLECTED_AND_RAW_AUDITED',
                artifact_sha256=bindings,collection=result,goal=report['goal'],selection_count=report['selection_count'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'])
        except Exception as exc:
            import traceback
            traceback.print_exc();terminal['failure']=repr(exc)
    terminal.update(wall_s=time.perf_counter()-started,worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'),terminal)
    return terminal


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive matched native attempt')
    verify_artifacts(DIAGNOSTIC,DIAGNOSTIC_IDS);diagnostic=read_json(DIAGNOSTIC,'result.json')
    assert diagnostic['status']=='ACTIVE_VIEW_SIX_MODEL_EXECUTED_TRANSITIONS_COMPLETE'
    verify_artifacts(DIAGNOSTIC,diagnostic['artifact_sha256'])
    _,receipt=admit(FIT_SHA);fitted=read_json(FITS,'result.json')
    fit_ids={'result.json':FIT_SHA}|fitted['artifact_sha256']
    snapshots={name:read_json(FITS,name+'_fit.json')['snapshot'] for name in ROSTER}
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,
        seed_paths=('scripts/run_go2_matched_model_active_view_probe_v1.py',
            'lewm/tests/test_matched_model_goal_probe_development.py',*diagnostic['source_sha256']),
        planned_trials=list(TRIALS),workers=1,storage_bytes=16*1024**3)
    for name,sha in diagnostic['source_sha256'].items():assert launch['source_sha256'].get(name)==sha,name
    ArticulatedCollisionGeometry(URDF)
    launch.update(all_six_admission=receipt,fit_artifact_sha256=fit_ids,snapshots=snapshots,
        diagnostic_sha256=DIAGNOSTIC_IDS,planned_cases=CASES,order_seed=2026091101,
        robot_urdf_path=str(URDF),robot_urdf_sha256=digest(URDF),
        experiment='matched six fitted predictors in active-view closed-loop control V1',
        data_scope='two known mirrored integration layouts; not independent mazes',
        concurrency_reason='fresh serial workers retain uncontended complete-loop timing; existing native workload measured',
        no_rgb_scope='predictor input only; shared visual observer and map retain RGB')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('MATCHED_MODEL_ACTIVE_VIEW_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    records=[];started=time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as executor:
                for case in CASES:
                    future=executor.submit(worker,case,digest(OUTPUT/'launch.json'))
                    while True:
                        monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started,case=case[0],**hardware()))+'\n');monitor.flush()
                        done,_=wait([future],timeout=15)
                        if done:break
                    record=future.result();records.append(record)
                    print('MATCHED_MODEL_ACTIVE_VIEW_TERMINAL',case[0],record['status'],record.get('goal'),flush=True)
                    if record['status']!='MATCHED_MODEL_ACTIVE_VIEW_COLLECTED_AND_RAW_AUDITED':
                        raise ValueError('infrastructure/raw audit failure; later cases unlaunched')
        bindings={n:h for r in records for n,h in r['artifact_sha256'].items()}
        for name in ('launch.json','resource_monitor.jsonl',*(c[0]+s for c in CASES for s in ('_worker.log','_worker_terminal.json'))):
            bindings[name]=digest(OUTPUT/name)
        verify(launch);verify_artifacts(OUTPUT,bindings);verify_artifacts(FITS,fit_ids)
        write_json(OUTPUT/'result.json',dict(status='MATCHED_MODEL_ACTIVE_VIEW_PROBE_COMPLETE',
            cases=CASES,conditions=records,artifact_sha256=bindings,source_sha256=launch['source_sha256'],
            wall_s=time.perf_counter()-started,snapshots=snapshots,
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and not r['hard_measurement_failed_frames'] for r in records),
            model_training=False,checkpoint_selection_performed=False,independent_maze_evaluation=False,
            navigation_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('MATCHED_MODEL_ACTIVE_VIEW_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MATCHED_MODEL_ACTIVE_VIEW_FAILURE',reason=repr(exc),
            completed_cases=[r['case'] for r in records],unlaunched_cases=[c[0] for c in CASES[len(records):]]))
        raise


if __name__=='__main__':main()
