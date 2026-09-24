"""Four fresh native cases with continuous nominal entry connectors."""
import argparse
import contextlib
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.geometry_progress_family_runtime_development import preflight, verify
from scripts.continuous_connector_goal_episode_development import collect, artifacts
from scripts.continuous_connector_goal_audit_development import audit
from scripts.matched_family_model_admission_development import load_assigned
from scripts.family_transition_model_admission_development import admit
from scripts.run_go2_family_transition_fits_v1 import OUTPUT as FITS, ROSTER
from scripts.read_go2_active_view_six_model_transitions_v1 import FIT_SHA
from scripts.run_go2_commitment_pose_goal_probe_v1 import OUTPUT as DIAGNOSTIC
from scripts.read_go2_observed_geometry_refinement_v1 import OUTPUT as REFINEMENT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_continuous_connector_goal_probe_v1_attempt_001'
READOUT=BASE/'go2_commitment_pose_goal_readout_v1_attempt_001'
DIAGNOSTIC_IDS={'launch.json':'4f1e6ce84d5ca60dc0f4ab7e9a6d8266fe6051e74709f923ed1ed34874ad5394'}
SCALING=BASE/'go2_geometry_progress_family_scaling_v1_attempt_001'
SCALING_IDS={'launch.json':'c31e78806a24b4250c99b5a9ffcd14ee624d3cab43aa1b288d8efc5907773b66',
    'result.json':'ed64dfb610062caea529f96db01d3ce5a3be2c621a5ca8bcd04822f5c1523969'}
PROTOCOL='docs/go2_continuous_connector_goal_probe_v1_2026-09-08.md'
REFINEMENT_SHA='aa2162f7bbe98213a5c2e16188961b183e6ed808695583f959770d2c06a2a804'
TRIALS=('family_episode_052','family_episode_039')
CASES=[(f'{v}_{c}_{t}',t,v,c,f'seed_2026091001_{v}_{c}') for v in ('full',)
    for c in ('direct','jepa') for t in TRIALS]
random.Random(2026091102).shuffle(CASES)


def worker(case,launch_sha):
    name,trial,variant,condition,model_name=case
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    terminal=dict(case=name,trial=trial,variant=variant,condition=condition,model_name=model_name,
        status='CONTINUOUS_CONNECTOR_GOAL_WORKER_FAILED',artifact_sha256={})
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
            terminal.update(status='CONTINUOUS_CONNECTOR_GOAL_COLLECTED_AND_RAW_AUDITED',
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior-result-sha256',required=True)
    parser.add_argument('--prior-readout-sha256',required=True)
    args=parser.parse_args();prior_ids=DIAGNOSTIC_IDS|{'result.json':args.prior_result_sha256}
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive commitment-pose native attempt')
    verify_artifacts(DIAGNOSTIC,prior_ids);diagnostic=read_json(DIAGNOSTIC,'result.json')
    assert diagnostic['status']=='COMMITMENT_POSE_GOAL_PROBE_COMPLETE' and len(diagnostic['conditions'])==4
    prior_bindings=prior_ids|diagnostic['artifact_sha256']
    verify_artifacts(READOUT,{'result.json':args.prior_readout_sha256});readout=read_json(READOUT,'result.json')
    assert readout['status']=='COMMITMENT_POSE_GOAL_READOUT_COMPLETE' and readout['probe_result_sha256']==args.prior_result_sha256
    readout_ids={'result.json':args.prior_readout_sha256,'launch.json':readout['launch_sha256']}
    verify_artifacts(READOUT,readout_ids);verify(read_json(READOUT,'launch.json'))
    verify_artifacts(SCALING,SCALING_IDS);scaling=read_json(SCALING,'result.json')
    assert scaling['status']=='GEOMETRY_PROGRESS_FAMILY_SCALING_COMPLETE' and scaling['selected_workers']==4
    verify(read_json(SCALING,'launch.json'))
    for root,ids in scaling['phase_result_sha256'].items():verify_artifacts(Path(root),ids)
    verify_artifacts(DIAGNOSTIC,prior_bindings)
    verify_artifacts(REFINEMENT,{'result.json':REFINEMENT_SHA})
    refinement=read_json(REFINEMENT,'result.json')
    assert refinement['status']=='OBSERVED_GEOMETRY_REFINEMENT_DIAGNOSTIC_COMPLETE'
    refinement_ids={'result.json':REFINEMENT_SHA}|refinement['artifact_sha256']
    verify_artifacts(REFINEMENT,refinement_ids);verify(read_json(REFINEMENT,'launch.json'))
    _,receipt=admit(FIT_SHA);fitted=read_json(FITS,'result.json')
    fit_ids={'result.json':FIT_SHA}|fitted['artifact_sha256']
    snapshots={name:read_json(FITS,name+'_fit.json')['snapshot'] for name in ROSTER}
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,
        seed_paths=('scripts/run_go2_continuous_connector_goal_probe_v1.py',
            'lewm/tests/test_continuous_connector_goal_probe_development.py',
            'docs/go2_observed_geometry_refinement_result_2026-09-08.md',*readout['source_sha256'],*refinement['source_sha256']),
        planned_trials=list(TRIALS),workers=4,storage_bytes=8*1024**3)
    for name,sha in diagnostic['source_sha256'].items():assert launch['source_sha256'].get(name)==sha,name
    for name,sha in readout['source_sha256'].items():assert launch['source_sha256'].get(name)==sha,name
    for name,sha in refinement['source_sha256'].items():assert launch['source_sha256'].get(name)==sha,name
    ArticulatedCollisionGeometry(URDF)
    launch.update(all_six_admission=receipt,fit_artifact_sha256=fit_ids,snapshots=snapshots,
        diagnostic_sha256=prior_ids,prior_artifact_sha256=prior_bindings,prior_readout_sha256=readout_ids,
        refinement_artifact_sha256=refinement_ids,
        scaling_sha256=SCALING_IDS,scaling_phase_result_sha256=scaling['phase_result_sha256'],
        planned_cases=CASES,order_seed=2026091102,
        robot_urdf_path=str(URDF),robot_urdf_sha256=digest(URDF),
        experiment='continuous nominal entry connector geometry V1',
        data_scope='two known mirrored integration layouts; not independent mazes',
        concurrency_reason='four fresh independent processes; prior native pipeline benchmark; concurrent timing is not uncontended timing',
        methods_selected_using_development_evidence=True,final_checkpoint_per_method_fixed=True)
    verify_artifacts(DIAGNOSTIC,prior_bindings);verify_artifacts(READOUT,readout_ids)
    verify_artifacts(REFINEMENT,refinement_ids)
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('CONTINUOUS_CONNECTOR_GOAL_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    records=[];started=time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as executor:
                pending={executor.submit(worker,case,digest(OUTPUT/'launch.json')):case for case in CASES}
                while pending:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started,
                        pending_cases=[c[0] for c in pending.values()],**hardware()))+'\n');monitor.flush()
                    done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                    for future in done:
                        case=pending.pop(future)
                        try:record=future.result()
                        except Exception as exc:
                            record=dict(case=case[0],status='CONTINUOUS_CONNECTOR_GOAL_PROCESS_FAILURE',failure=repr(exc))
                        records.append(record)
                        print('CONTINUOUS_CONNECTOR_GOAL_TERMINAL',case[0],record['status'],record.get('goal'),flush=True)
            if len(records)!=4 or any(r['status']!='CONTINUOUS_CONNECTOR_GOAL_COLLECTED_AND_RAW_AUDITED' for r in records):
                raise ValueError('infrastructure/raw audit failure; all launched siblings retained')
            records.sort(key=lambda r:[c[0] for c in CASES].index(r['case']))
        bindings={n:h for r in records for n,h in r['artifact_sha256'].items()}
        for name in ('launch.json','resource_monitor.jsonl',*(c[0]+s for c in CASES for s in ('_worker.log','_worker_terminal.json'))):
            bindings[name]=digest(OUTPUT/name)
        verify(launch);verify_artifacts(OUTPUT,bindings);verify_artifacts(FITS,fit_ids)
        verify_artifacts(DIAGNOSTIC,prior_bindings);verify_artifacts(READOUT,readout_ids)
        verify_artifacts(REFINEMENT,refinement_ids)
        verify_artifacts(SCALING,SCALING_IDS)
        for root,ids in scaling['phase_result_sha256'].items():verify_artifacts(Path(root),ids)
        write_json(OUTPUT/'result.json',dict(status='CONTINUOUS_CONNECTOR_GOAL_PROBE_COMPLETE',
            cases=CASES,conditions=records,artifact_sha256=bindings,source_sha256=launch['source_sha256'],
            wall_s=time.perf_counter()-started,snapshots=snapshots,
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and not r['hard_measurement_failed_frames'] for r in records),
            model_training=False,checkpoint_selection_performed=False,independent_maze_evaluation=False,
            navigation_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('CONTINUOUS_CONNECTOR_GOAL_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_CONTINUOUS_CONNECTOR_GOAL_FAILURE',reason=repr(exc),
            completed_cases=[r['case'] for r in records],
            missing_terminal_cases=[c[0] for c in CASES if c[0] not in [r['case'] for r in records]]))
        raise


if __name__=='__main__':main()
