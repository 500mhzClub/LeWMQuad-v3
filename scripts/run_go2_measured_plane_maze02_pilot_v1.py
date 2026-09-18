"""Fresh measured-plane navigation after the completed original native queue."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import numpy as np
import psutil
import torch

from scripts import measured_plane_native_inputs_development as inputs
from scripts import measured_plane_extended_maze_development as pipeline
from scripts import measured_plane_native_prefix_development as prefix
from scripts.supervised_commitment_contact_queue_gate_development import require_native_idle
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_readout_development import case_readout
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS, require_raw_audit

run,old = inputs.run,inputs.job.native
SOURCE = 'scripts/run_go2_measured_plane_maze02_pilot_v1.py'
PROTOCOL = 'docs/go2_measured_plane_maze02_pilot_v1_2026-09-11.md'
TESTS = ('lewm/tests/test_measured_plane_native_launcher_development.py',
    'lewm/tests/test_measured_plane_extended_maze_development.py',
    'lewm/tests/test_measured_plane_native_prefix_development.py')
OUTPUT = run.BASE/'go2_measured_plane_maze02_pilot_v1_attempt_001'
CASE = ('no_rgb_direct_measured_plane_maze_02',*old.CASE[1:])
WORKER_STATUS = 'MEASURED_PLANE_MAZE02_COLLECTED_AND_RAW_AUDITED'


def definition():
    return pipeline.definition() | dict(planned_case=list(CASE),output_root=str(OUTPUT),
        scene_specification=specification(2),public_mission=public_mission(2),
        model_state_sha256=inputs.job.MODEL_SHA,robot_urdf_sha256=run.digest(inputs.job.URDF),
        native_scene_workers=1,opencv_threads=1,blas_threads=1,maximum_tasks_per_process=1,
        planner_interface_adapter_enabled=True,native_execution=True,model_training=False,
        independent_layout_development_execution=False,reused_development_layout=True)


def resources():
    hw = run.hardware()
    if hw['memory_available_bytes'] < 32*1024**3 or hw['artifact_free_bytes'] < 55*1024**3:
        raise ValueError('32 GiB available RAM and 55 GiB artifact space required')
    return hw


def verify_inputs(launch):
    require_environment(launch)
    old.verify_ordered_launch(launch)
    if any(type(launch[k]) is not type(v) or launch[k] != v for k,v in definition().items()):
        raise ValueError('exact prospective measured-plane native definition required')
    admitted = inputs.admit(launch['input_admission']['queue']['waiter_result_sha256'],launch['source_sha256'])
    if admitted != launch['input_admission']: raise ValueError('complete input admission changed')


def require_environment(launch):
    if (any(run.os.environ.get(k) != v for k,v in run.ENV.items())
            or any(run.os.environ.get(k) != v for k,v in launch['renderer_environment'].items())):
        raise ValueError('original deterministic CPU and renderer environment required')


def assigned_model():
    model = old.assigned_model(run.read_json(old.OUTPUT,'launch.json'))
    if model.training or state_digest(model.state_dict()) != inputs.job.MODEL_SHA:
        raise ValueError('exact originally assigned corrected evaluation model required')
    return model


def prefix_result(result,report):
    if result['decisions'] < prefix.FRAMES:
        return dict(status='PROSPECTIVE_BOUNDARY_NOT_REACHED',decisions=result['decisions'],
            full_raw_audit_retained=True,actual_paired_execution_compared=False,
            navigation_verified=False,unexecuted_outcomes_inferred=False)
    return dict(status='COMPLETE_ACTUAL_PREFIX_COMPARISON',actual_paired_execution_compared=True,
        **prefix.compare(old.OUTPUT/old.CASE[0],OUTPUT/CASE[0],report))


def require_worker(record,audit):
    if (record['status'] != WORKER_STATUS or 'failure' in record
            or record['case'] != CASE[0] or record['layout_index'] != CASE[1]
            or record['variant'] != CASE[2] or record['condition'] != CASE[3] or record['model_name'] != CASE[4]
            or record['model_state_sha256'] != inputs.job.MODEL_SHA or record['model_state_unchanged'] is not True
            or record['measured_plane_constrained_estimator'] is not True
            or record['collection']['navigation_ticks'] != 4000
            or record['collection']['status'] != 'RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED'):
        raise ValueError('complete fixed measured-plane collection and unchanged model required')
    require_raw_audit(record,audit,learned=True)
    success = bool(audit['native_evaluation']['native_round_trip_candidate_pass']
        and audit['strict_physical_visibility_pass'] and not audit['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not success:
        raise ValueError('unchanged joint physical and sensing success criteria required')
    receipt = record['prefix_comparison']
    if record['collection']['decisions'] < prefix.FRAMES:
        expected = dict(status='PROSPECTIVE_BOUNDARY_NOT_REACHED',decisions=record['collection']['decisions'],
            full_raw_audit_retained=True,actual_paired_execution_compared=False,
            navigation_verified=False,unexecuted_outcomes_inferred=False)
        if receipt != expected or success: raise ValueError('retain early negative execution without a prefix claim')
    else:
        for key in ('actual_paired_execution_compared','physical_and_public_prefix_exact',
                'all_preintervention_requested_commands_exact','complete_candidate_decisions_match_prospective_prefix',
                'candidate_intervention_command_completed','intervention_command_changed'):
            if receipt[key] is not True: raise ValueError('full actual prospective intervention required: '+key)
        expected = dict(status='COMPLETE_ACTUAL_PREFIX_COMPARISON',common_prefix_frames=123,
            first_changed_command_frame=122,physical_prefix_samples=6850,original_forecasts_compared=120,
            original_intervention_command=[0.,0.,0.],candidate_intervention_command=[0.,0.,-.45],
            boundary_command_samples_present=50,following_physical_outcomes_compared=False,
            navigation_verified=False,unexecuted_outcomes_inferred=False)
        if any(type(receipt[k]) is not type(v) or receipt[k] != v for k,v in expected.items()):
            raise ValueError('exact completed changed-command boundary required')


def worker(launch_sha):
    name,index,variant,condition,model_name = CASE
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.perf_counter(); ids = {}
    record = dict(status='MEASURED_PLANE_MAZE02_WORKER_FAILED',case=name,layout_index=index,
        variant=variant,condition=condition,model_name=model_name,artifact_sha256={})
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            run.verify_artifacts(OUTPUT,{'launch.json':launch_sha}); launch=run.read_json(OUTPUT,'launch.json')
            verify_inputs(launch); resources()
            model=assigned_model(); before=state_digest(model.state_dict())
            result=pipeline.collect(index,launch['source_sha256'][PROTOCOL],output=OUTPUT,model=model,
                geometry=ArticulatedCollisionGeometry(inputs.job.URDF),episode_name=name,condition=condition,variant=variant)
            record['collection']=result
            if state_digest(model.state_dict()) != before: raise ValueError('collection changed model state')
            ids.update({name+'/'+n:run.digest(OUTPUT/name/n) for n in pipeline.artifacts(index,result)})
            record['artifact_sha256']=dict(ids); run.verify_artifacts(OUTPUT,ids)
            audit=pipeline.audit(index,result,launch['source_sha256'][PROTOCOL],input_root=OUTPUT,
                model=assigned_model(),robot_geometry=ArticulatedCollisionGeometry(inputs.job.URDF),
                episode_name=name,condition=condition,variant=variant)
            n=name+'_audit.json';run.write_json(OUTPUT/n,audit);ids[n]=run.digest(OUTPUT/n)
            record['artifact_sha256']=dict(ids)
            receipt=prefix_result(result,launch['input_admission']['prefix_report'])
            n=name+'_prefix_comparison.json';run.write_json(OUTPUT/n,receipt);ids[n]=run.digest(OUTPUT/n)
            record['artifact_sha256']=dict(ids)
            with np.load(OUTPUT/name/'physics_trace.npz',allow_pickle=False) as saved:
                readout=case_readout(audit,result,saved['physics_contact'])
            n=name+'_readout.json';run.write_json(OUTPUT/n,readout);ids[n]=run.digest(OUTPUT/n)
            record.update(status=WORKER_STATUS,artifact_sha256=dict(ids),prefix_comparison=receipt,readout=readout,
                model_state_sha256=before,model_state_unchanged=True,measured_plane_constrained_estimator=True,
                **{k:audit[k] for k in OUTCOME_KEYS})
            require_worker(record,audit);verify_inputs(launch);run.verify_artifacts(OUTPUT,ids)
        except Exception as error:
            import traceback
            traceback.print_exc();record.update(status='MEASURED_PLANE_MAZE02_WORKER_FAILED',failure=repr(error))
    record.update(wall_s=time.perf_counter()-start,maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=run.digest(OUTPUT/(name+'_worker.log')))
    run.write_json(OUTPUT/(name+'_worker_terminal.json'),record)
    return record


def main(queue_sha=None,source_only=False,preflight=False):
    if not __debug__: raise ValueError('assertions required')
    run.validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive native attempt; no retry or resume')
    sources=inputs.prepared_sources((SOURCE,PROTOCOL,*TESTS));hw=resources()
    if source_only:
        print('MEASURED_PLANE_NATIVE_SOURCE_PREFLIGHT',len(sources),json.dumps(hw),flush=True);return
    if not queue_sha: raise ValueError('actual completed original chained-anchor waiter SHA-256 required')
    admission=inputs.admit(queue_sha,sources)
    original=run.read_json(old.OUTPUT,'launch.json')
    launch={k:original[k] for k in ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules','renderer_environment')}
    launch.update(definition());launch.update(source_sha256=sources,protocol=PROTOCOL,input_admission=admission)
    verify_inputs(launch);assigned_model();launch['hardware']=resources()
    if preflight:
        print('MEASURED_PLANE_NATIVE_PREFLIGHT',len(sources),flush=True);return
    require_native_idle();run.create_output(OUTPUT)
    p=psutil.Process();launch.update(boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=p.pid,created=p.create_time(),command=p.cmdline()),automatic_retry=False)
    run.write_json(OUTPUT/'launch.json',launch);sha=run.digest(OUTPUT/'launch.json')
    print('MEASURED_PLANE_NATIVE_LAUNCHED',sha,flush=True);start=time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                future=pool.submit(worker,sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-start,**run.hardware()))+'\n');monitor.flush()
                    done,_=wait([future],timeout=15,return_when=FIRST_COMPLETED)
                    if done: record=future.result();break
        print('MEASURED_PLANE_NATIVE_TERMINAL',record['status'],record.get('verified_round_trip'),record.get('failure'),flush=True)
        if record['status'] != WORKER_STATUS: raise ValueError('native collection/audit failed: '+str(record.get('failure')))
        require_worker(record,run.read_json(OUTPUT,CASE[0]+'_audit.json'))
        ids=dict(record['artifact_sha256'])
        for n in ('launch.json','resource_monitor.jsonl',CASE[0]+'_worker.log',CASE[0]+'_worker_terminal.json'):
            ids[n]=run.digest(OUTPUT/n)
        verify_inputs(launch);run.verify_artifacts(OUTPUT,ids)
        run.write_json(OUTPUT/'result.json',dict(status='MEASURED_PLANE_MAZE02_PILOT_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,conditions=[record],wall_s=time.perf_counter()-start,
            controller_completion_sha256=prefix.COMPLETION_SHA,chained_wait_result_sha256=queue_sha,
            measured_round_trip_successes=int(record['verified_round_trip']),reused_layout_executions=1,
            new_independent_layout_executions=0,measured_plane_constrained_estimator=True,automatic_retry=False,
            model_training=False,navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('MEASURED_PLANE_NATIVE_COMPLETE',run.digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        run.write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MEASURED_PLANE_NATIVE_FAILURE',reason=repr(error),automatic_retry=False))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chained-wait-result-sha256')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--source-preflight-only',action='store_true');mode.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();main(args.chained_wait_result_sha256,args.source_preflight_only,args.preflight_only)
