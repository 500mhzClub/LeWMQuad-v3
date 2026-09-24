"""Fresh 8,000-step native trial after the completed prospective budget prefix."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED

import numpy as np
import psutil
import torch

from scripts import extended_return_budget_native_inputs_development as inputs
from scripts import extended_return_budget_native_result_development as results
from scripts import resource_guarded_extended_return_maze_development as pipeline
from scripts import extended_return_budget_worker_resources_development as worker_resources
from lewm.independent_reactive_floor_transport_study_development import MATCHED_KEYS,OUTCOME_KEYS

run=inputs.run
original=inputs.native.original
SOURCE='scripts/run_go2_extended_return_budget_maze02_v1.py'
PROTOCOL='docs/go2_extended_return_budget_maze02_v1_2026-09-12.md'
TESTS=(inputs.TEST,'lewm/tests/test_extended_return_budget_native_result_development.py',
    'lewm/tests/test_extended_return_budget_worker_resources_development.py',
    'lewm/tests/test_extended_return_budget_native_worker_development.py',
    'lewm/tests/test_extended_return_budget_native_launcher_development.py',
    'lewm/tests/test_extended_return_budget_native_prefix_development.py',
    'lewm/tests/test_extended_return_budget_resource_guard_development.py',
    'lewm/tests/test_extended_return_budget_resource_audit_development.py',
    'lewm/tests/test_extended_return_budget_maze_pipeline_development.py')
OUTPUT=run.BASE/'go2_extended_return_budget_maze02_v1_attempt_001'
CASE=('no_rgb_direct_extended_return_budget_maze_02',*inputs.native.CASE[1:])
WORKER_STATUS='EXTENDED_RETURN_BUDGET_MAZE02_COLLECTED_AND_RAW_AUDITED'
WORKER_FAILURE='EXTENDED_RETURN_BUDGET_MAZE02_WORKER_FAILED'
MAX_METADATA_BYTES=256*1024**2
assigned_model=inputs.native.assigned_model
state_digest=inputs.replay.pair.state_digest
geometry_factory=inputs.replay.pair.geometry_factory
URDF=inputs.replay.pair.URDF
cpu_idle=inputs.replay.cpu_idle


def bounded_json(path,value):
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if len(payload)>MAX_METADATA_BYTES:raise ValueError('bounded native metadata exceeded 256 MiB')
    with path.open('xb') as stream:stream.write(payload)


def definition():
    return pipeline.definition()|dict(planned_case=list(CASE),output_root=str(OUTPUT),
        scene_specification=original.specification(2),public_mission=original.public_mission(2),
        model_state_sha256=inputs.replay.pair.MODEL_SHA,robot_urdf_sha256=run.digest(URDF),
        native_scene_workers=1,opencv_threads=1,blas_threads=1,maximum_tasks_per_process=1,
        planner_interface_adapter_enabled=True,native_execution=True,model_training=False,
        independent_layout_development_execution=False,reused_development_layout=True,
        native_execution_protocol_frozen=True,completed_prospective_budget_prefix_required=True,
        single_pass_timing_change_adopted=True,worker_terminal_write_resource_observed=True)


def resources():
    hardware=run.hardware();pipeline.resources.admission(hardware);return hardware


def verify_inputs(launch):
    original.require_environment(launch);original.old.verify_ordered_launch(launch)
    expected=definition()
    if any(type(launch.get(k)) is not type(v) or launch[k]!=v for k,v in expected.items()):
        raise ValueError('exact prepared longer native definition required')
    baseline=inputs.old_inputs.native_launch()
    for key in MATCHED_KEYS:
        if key!='navigation_ticks' and (type(launch.get(key)) is not type(baseline[key]) or launch[key]!=baseline[key]):
            raise ValueError('same original scene, sensing, gait and CPU rules required: '+key)
    inputs.verify_bound(launch['input_admission'],launch['source_sha256'])


def worker(launch_sha):
    name,index,variant,condition,model_name=CASE
    run.cv2.setNumThreads(1);run.cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();ids={};failure=None
    record=dict(status=WORKER_FAILURE,case=name,layout_index=index,variant=variant,condition=condition,
        model_name=model_name,artifact_sha256={})
    envelope=worker_resources.WorkerEnvelope(OUTPUT,name)
    try:
        with (OUTPUT/(name+'_worker.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                envelope.check('start')
                run.verify_artifacts(OUTPUT,{'launch.json':launch_sha});launch=run.read_json(OUTPUT,'launch.json')
                verify_inputs(launch);envelope.check('inputs_verified')
                model=assigned_model();before=state_digest(model.state_dict())
                collection=pipeline.collect(index,launch['source_sha256'][PROTOCOL],output=OUTPUT,model=model,
                    geometry=geometry_factory(URDF),episode_name=name,condition=condition,variant=variant)
                record['collection']=collection
                if state_digest(model.state_dict())!=before or any(p.grad is not None for p in model.parameters()):
                    raise ValueError('collection changed the original assigned model')
                envelope.check('collection_persisted')
                ids.update({name+'/'+leaf:run.digest(OUTPUT/name/leaf) for leaf in pipeline.artifacts(index,collection)})
                for leaf in pipeline.resources.names(name,'collection'):ids[leaf]=run.digest(OUTPUT/leaf)
                record['artifact_sha256']=dict(ids);run.verify_artifacts(OUTPUT,ids)
                envelope.check('collection_artifacts_verified')
                audit=pipeline.audit(index,collection,launch['source_sha256'][PROTOCOL],input_root=OUTPUT,
                    model=assigned_model(),robot_geometry=geometry_factory(URDF),episode_name=name,condition=condition,variant=variant)
                bounded_json(OUTPUT/(name+'_audit.json'),audit)
                with np.load(OUTPUT/name/'physics_trace.npz',allow_pickle=False) as raw:
                    readout=original.case_readout(audit,collection,raw['physics_contact'])
                bounded_json(OUTPUT/(name+'_readout.json'),readout)
                resource_audit=results.resource_audit.check(OUTPUT,name,collection)
                bounded_json(OUTPUT/(name+'_resource_audit.json'),resource_audit)
                for leaf in pipeline.resources.names(name,'audit'):ids[leaf]=run.digest(OUTPUT/leaf)
                for suffix in ('_audit.json','_readout.json','_resource_audit.json'):ids[name+suffix]=run.digest(OUTPUT/(name+suffix))
                record.update(artifact_sha256=dict(ids),readout=readout,resource_audit=resource_audit)
                envelope.check('audit_written')
                receipt=results.prefix_result(collection,launch['input_admission'],output=OUTPUT,case=CASE,current_bindings=ids)
                bounded_json(OUTPUT/(name+'_prefix_comparison.json'),receipt)
                ids[name+'_prefix_comparison.json']=run.digest(OUTPUT/(name+'_prefix_comparison.json'))
                envelope.check('prefix_accounted')
                record.update(status=WORKER_STATUS,artifact_sha256=dict(ids),prefix_comparison=receipt,
                    model_state_sha256=before,model_state_unchanged=True,measured_plane_constrained_estimator=True,
                    extended_return_budget_enabled=True,single_pass_timing_change_adopted=True,sampled_resource_guards_enabled=True,
                    **{k:audit[k] for k in OUTCOME_KEYS})
                # This receipt was freshly reconstructed above. The parent independently
                # reconstructs it again after the worker ends and artifacts are closed.
                results.require_worker(record,audit,launch['input_admission'],output=OUTPUT,case=CASE,
                    worker_status=WORKER_STATUS,prefix_receipt=receipt)
                envelope.check('worker_validated')
                verify_inputs(launch);envelope.check('inputs_reverified')
                run.verify_artifacts(OUTPUT,ids);envelope.check('artifacts_verified')
            except Exception as error:
                import traceback
                traceback.print_exc();failure=error
                record.update(status=WORKER_FAILURE,failure=repr(error),artifact_sha256=dict(ids))
        record.update(wall_s=time.perf_counter()-started,maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            worker_log_sha256=run.digest(OUTPUT/(name+'_worker.log')))
        bounded_json(OUTPUT/(name+'_worker_terminal.json'),record)
        if failure is None:envelope.check('terminal_written')
        return record
    except BaseException as error:
        failure=error
        raise
    finally:envelope.finish(failure)


def main(prefix_sha=None,source_only=False,preflight=False):
    if not __debug__:raise ValueError('assertions required')
    run.validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive longer native attempt; no retry or resume')
    sources=inputs.prepared_sources((SOURCE,PROTOCOL,*TESTS))
    if source_only:
        print('EXTENDED_RETURN_NATIVE_SOURCE_PREFLIGHT',len(sources),json.dumps(run.hardware()),flush=True);return
    if not prefix_sha:raise ValueError('actual completed prospective prefix result SHA required')
    resources();cpu_idle();admission=inputs.admit(prefix_sha,sources)
    baseline=inputs.old_inputs.native_launch()
    launch={k:baseline[k] for k in ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules','renderer_environment')}
    launch.update(definition());launch.update(source_sha256=sources,protocol=PROTOCOL,input_admission=admission)
    verify_inputs(launch);assigned_model();launch['hardware']=resources();cpu_idle()
    if preflight:
        print('EXTENDED_RETURN_NATIVE_PREFLIGHT',len(sources),flush=True);return
    run.create_output(OUTPUT);process=psutil.Process()
    launch.update(boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid,created=process.create_time(),command=process.cmdline()),automatic_retry=False)
    bounded_json(OUTPUT/'launch.json',launch);launch_sha=run.digest(OUTPUT/'launch.json')
    print('EXTENDED_RETURN_NATIVE_LAUNCHED',launch_sha,flush=True);started=time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                future=pool.submit(worker,launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started,**run.hardware()))+'\n');monitor.flush()
                    done,_=wait([future],timeout=15,return_when=FIRST_COMPLETED)
                    if done:record=future.result();break
        if record['status']!=WORKER_STATUS:
            raise ValueError('longer native collection or audit failed: '+str(record.get('failure')))
        if run.canonical(run.read_json(OUTPUT,CASE[0]+'_worker_terminal.json'))!=run.canonical(record):
            raise ValueError('same complete persisted worker terminal required')
        whole_worker=worker_resources.check(OUTPUT,CASE[0])
        results.require_worker(record,run.read_json(OUTPUT,CASE[0]+'_audit.json'),admission,
            output=OUTPUT,case=CASE,worker_status=WORKER_STATUS)
        identities=dict(record['artifact_sha256'])
        for name in ('launch.json','resource_monitor.jsonl',CASE[0]+'_worker.log',CASE[0]+'_worker_terminal.json',
                *worker_resources.names(CASE[0])):
            identities[name]=run.digest(OUTPUT/name)
        verify_inputs(launch);run.verify_artifacts(OUTPUT,identities)
        final_resources=run.hardware()
        if (final_resources['memory_available_bytes']<pipeline.resources.MIN_AVAILABLE_RAM_BYTES
                or final_resources['artifact_free_bytes']<pipeline.resources.RETAINED_DISK_RESERVE_BYTES+MAX_METADATA_BYTES):
            raise pipeline.resources.ResourceLimitError('retain final result serialization headroom and disk reserve')
        result=dict(status='EXTENDED_RETURN_BUDGET_MAZE02_V1_COMPLETE',source_sha256=sources,
            artifact_sha256=identities,conditions=[record],worker_resource_check=whole_worker,
            final_parent_hardware=final_resources,maximum_result_metadata_bytes=MAX_METADATA_BYTES,
            controller_prefix_result_sha256=prefix_sha,original_native_result_sha256=inputs.replay.inputs.NATIVE_RESULT_SHA,
            measured_round_trip_successes=int(record['verified_round_trip']),reused_layout_executions=1,new_independent_layout_executions=0,
            navigation_ticks=8000,single_pass_timing_change_adopted=True,sampled_resource_guards_enabled=True,
            actual_physical_prefix_reconstructed_after_worker_end=record['prefix_comparison']['actual_paired_execution_compared'],
            complete_predecessor_inputs_reauthenticated=True,
            prior_controller_or_physics_execution_repeated=False,model_training=False,automatic_retry=False,
            navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False,
            wall_s=time.perf_counter()-started)
        bounded_json(OUTPUT/'result.json',result)
        print('EXTENDED_RETURN_NATIVE_COMPLETE',run.digest(OUTPUT/'result.json'),record['verified_round_trip'],flush=True)
    except BaseException as error:
        bounded_json(OUTPUT/'failure.json',dict(status='TERMINAL_EXTENDED_RETURN_NATIVE_FAILURE',reason=repr(error),
            automatic_retry=False,original_evidence_preserved=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--controller-prefix-result-sha256')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--source-preflight-only',action='store_true');mode.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();main(args.controller_prefix_result_sha256,args.source_preflight_only,args.preflight_only)
