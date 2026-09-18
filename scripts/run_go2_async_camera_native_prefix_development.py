"""Host-deadline native loop with asynchronous snapshot camera acquisition."""
from collections import deque,Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import contextlib
import gc
import hashlib
import json
import time

import cv2
import numpy as np
import psutil
import torch

from lewm.actuator_gain_development import configure_gains
from lewm.independent_round_trip_layouts_development import specification,public_mission
from lewm.paced_multirate_controller_development import AcquiredFrame
from lewm.process_mapped_runtime_development import (
    initialize_mapping,mapping_ready,initialize_pose_300,pose_ready,OverlappedObstacleRuntime)
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.all_phase_planner_model_admission_development import load_assigned
from scripts.navigation_artifact_root_development import BASE,validate_root
from scripts.paced_native_session_development import PacedNativeSession
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop

OUTPUT=BASE/'go2_paced_native_prefix_layout00_v1_attempt_001'
COUNT=61
LAYOUT_INDEX=0
EPOCH=1_500_000_000
CLOCK_MODE='wall'
PLANNING_DELAY_TICKS=2
MEASURED_RUNTIME_CLASS=None
OBSTACLE_INITIALIZER=None
OBSTACLE_READY=None
POSE_INITIALIZER=initialize_pose_300
MODEL_ASSIGNMENT='seed_2026091001_full_jepa'
DEFER_CYCLIC_GC=False


def write(name,value):
    with (OUTPUT/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def main():
    if CLOCK_MODE!='wall':raise ValueError('asynchronous camera probe requires actual host deadlines')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('preserve existing native attempt')
    if psutil.virtual_memory().available<32*1024**3:raise RuntimeError('short-run RAM headroom unavailable')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if MODEL_ASSIGNMENT=='reactive':
        model=None;condition='reactive';variant='full'
    else:
        admission=json.loads((BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'/'launch.json').read_text())['input_admission']['correction_admission']
        model,condition,variant=load_assigned(admission,MODEL_ASSIGNMENT)
    spec=specification(LAYOUT_INDEX);mission=public_mission(LAYOUT_INDEX)
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        __file__,'scripts/paced_native_session_development.py',
        'lewm/process_mapped_runtime_development.py','lewm/paced_multirate_controller_development.py',
        'lewm/fresh_obstacle_dispatch_development.py','lewm/delayed_action_planning_development.py',
        'lewm/feature_budget_300_tracker_development.py')}
    OUTPUT.mkdir();directory=OUTPUT/'native';directory.mkdir()
    owner=psutil.Process()
    write('launch.json',dict(owner=dict(pid=owner.pid,created=owner.create_time()),observations=COUNT,
        source_sha256=sources,public_mission=mission,layout_index=LAYOUT_INDEX,
        model_assignment=MODEL_ASSIGNMENT,camera_simulation_period_ns=100_000_000,
        command_service_simulation_period_ns=20_000_000,deadline_clock=CLOCK_MODE,
        planning_delay_ticks=PLANNING_DELAY_TICKS,
        measured_worker_and_acquisition_costs_charged_to_simulation=CLOCK_MODE=='measured_simulation',
        physics_waits_for_high_level_result=False,physics_paused_during_rendering=False,
        acquisition_retains_original_ideal_simulation_stamps=True,
        maximum_motion_dispatch_simulator_lag_ns=20_000_000,
        simulator_lag_is_not_retimed=True,shared_host=True,native_scene_workers=2,
        physical_execution=True,real_time_qualified=False,full_mission_implemented=False))
    print('PACED_NATIVE_PREFIX_LAUNCHED',str(OUTPUT),flush=True)
    controller=session=None;requests=[];acquisitions=[];published=[];stop=None;simulation_clock=None
    gc_events=[]
    gc_was_enabled=gc.isenabled()
    def gc_receipt(phase,info):
        gc_events.append(dict(wall_ns=time.perf_counter_ns(),phase=phase,**info))
    try:
        with (ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initialize_mapping) as mapping,
                (ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=POSE_INITIALIZER)
                    if POSE_INITIALIZER is not None else contextlib.nullcontext()) as pose,
                (ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=OBSTACLE_INITIALIZER)
                    if OBSTACLE_INITIALIZER is not None else contextlib.nullcontext()) as obstacle_process,
                (OUTPUT/'worker.log').open('x') as log,
                contextlib.redirect_stdout(log),contextlib.redirect_stderr(log)):
            assert mapping.submit(mapping_ready).result()
            if pose is not None:assert pose.submit(pose_ready).result()
            if obstacle_process is not None:assert obstacle_process.submit(OBSTACLE_READY).result()
            initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
            session=PacedNativeSession(spec,directory);session.install_contact_identity()
            gains=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
            write('actuator_identity.json',gains)
            session.settle_recorded()
            admit_context_setup(session,sources[__file__])
            if DEFER_CYCLIC_GC:
                gc.collect()
                gc.disable()
            start=time.perf_counter_ns()
            gc.callbacks.append(gc_receipt)
            def clock():return EPOCH+time.perf_counter_ns()-start
            runtime_type=OverlappedObstacleRuntime
            if CLOCK_MODE=='measured_simulation':
                from lewm.measured_latency_simulation_development import MeasuredLatencyClock,MeasuredLatencyRuntime
                clock=simulation_clock=MeasuredLatencyClock();runtime_type=MEASURED_RUNTIME_CLASS or MeasuredLatencyRuntime
            def sink(frame,raw,registered):
                published.append(dict(frame=frame,raw_pose=raw['current_pose'],registered_pose=registered['current_pose']))
            controller=runtime_type(model,goal_initial_xy=mission['goal_initial_body_xy_m'],
                condition=condition,variant=variant,clock_ns=clock,evidence_sink=sink,
                planning_delay_ticks=PLANNING_DELAY_TICKS,
                maximum_initial_dispatch_lateness_ns=0 if simulation_clock is not None else 1_000_000,
                **(dict(obstacle_executor=obstacle_process) if obstacle_process is not None else {}),
                mapping_executor=mapping,pose_executor=pose)
            history=deque(maxlen=4);pending=deque()
            if simulation_clock is not None:
                def advance_physics_clock(ns):
                    simulation_clock.advance(ns)
                    while pending and pending[0][0]<=ns:
                        _,packet=pending.popleft();controller.submit(packet)
                session.physics_clock_callback=advance_physics_clock
            def collect_camera():
                for packets, receipt in session.poll_sensor_packets():
                    p,d,f,a,rgb,measured=packets
                    history.append(p)
                    packet=AcquiredFrame(receipt['frame'],measured,p,d,f,rgb,a,tuple(history))
                    controller.submit(packet)
                    available=clock()
                    receipt['controller_submitted_wall_ns']=time.perf_counter_ns()
                    acquisitions.append(dict(frame=receipt['frame'],measured_ns=measured,
                        acquisition_started_ns=EPOCH+receipt['started_wall_ns']-start,
                        packet_available_ns=available,
                        measured_acquisition_wall_ns=receipt['received_wall_ns']-receipt['started_wall_ns'],
                        snapshot_submission_wall_ns=receipt['snapshot_submitted_wall_ns']-receipt['started_wall_ns'],
                        rendering_wall_ns=receipt['rendering_wall_ns'],asynchronous_snapshot_rendering=True))

            try:
                for tick in range(COUNT*5):
                    sim_ns=int(session.ctx.runner._sim_time_ns)
                    if simulation_clock is not None:simulation_clock.advance(sim_ns)
                    wait=(sim_ns-EPOCH-(time.perf_counter_ns()-start))/1e9
                    if wait>0:time.sleep(wait)
                    collect_camera()
                    if tick%5==0:
                        session.begin_sensor_packets()
                    while pending and pending[0][0]<=sim_ns:
                        _,packet=pending.popleft();controller.submit(packet)
                    now=clock();request_started=time.perf_counter_ns()
                    request=controller.request(now_ns=now)
                    request_finished=time.perf_counter_ns()
                    lag=EPOCH+time.perf_counter_ns()-start-sim_ns
                    # A slow simulator cannot turn host-time command windows
                    # into a different physical-time action sequence.
                    if simulation_clock is None and lag>20_000_000:
                        request=request|dict(requested_command=[0.,0.,0.],
                            reason='SIMULATOR_DISPATCH_LAG',underlying_reason=request['reason'])
                        with controller.lock:
                            for plan in controller.plans:
                                if plan.dispatch_ns<=now<plan.expires_ns:
                                    controller.rejected_windows[plan.observed_ns]='SIMULATOR_DISPATCH_LAG'
                    row=dict(request,simulator_ns=sim_ns,simulator_lag_ns=lag,
                        request_started_wall_ns=request_started,request_finished_wall_ns=request_finished,
                        pre_sample_index=len(session.samples)-1)
                    requests.append(row);session.phase=2
                    row['physical_service_started_wall_ns']=time.perf_counter_ns()
                    row['applied_command']=session.command_policy_step(request['requested_command'])
                    row['physical_service_completed_wall_ns']=time.perf_counter_ns()
                    row.update(post_sample_index=len(session.samples)-1,completed_wall_ns=clock())
                    if tick%500==0:
                        with controller.lock:progress=getattr(controller,'mission_latest',None)
                        print('NAVIGATION_PROGRESS',json.dumps(dict(camera_frame=tick//5,
                            simulation_s=(session.ctx.runner._sim_time_ns-EPOCH)/1e9,
                            observed_pose_frame=None if progress is None else progress.get('consumed_pose_frame',progress['frame']),
                            phase=None if progress is None else progress['phase'],
                            goal_distance_m=None if progress is None else progress['observed_goal_distance_m'])),flush=True)
                    if controller.faults:raise RuntimeError(str(controller.faults))
                    if getattr(controller,'mission_terminal',None) is not None:break
            except PhysicalStop as error:
                stop=str(error)
                raise
            if simulation_clock is not None:
                drain_deadline=time.perf_counter()+20.
                while True:
                    simulation_clock.advance(int(session.ctx.runner._sim_time_ns))
                    while pending and pending[0][0]<=simulation_clock.ns:
                        _,packet=pending.popleft();controller.submit(packet)
                    if controller.faults or not pending and not any(q.unfinished_tasks for q in controller.queues.values()):break
                    if time.perf_counter()>drain_deadline:
                        raise TimeoutError('pipeline drain exceeded 20 seconds while advancing zero-command physics')
                    if stop is not None:raise RuntimeError('physical stop prevents worker-drain physics')
                    session.phase=3;session.command_policy_step([0.,0.,0.])
            camera_drain_deadline=time.perf_counter()+20.
            while session.camera_pending:
                collect_camera()
                if time.perf_counter()>camera_drain_deadline:
                    raise TimeoutError('terminal camera drain exceeded 20 seconds')
                time.sleep(.001)
            controller.finish()
            if hasattr(controller,'depth_observer'):write('independent_depth_receipts.json',controller.depth_observer.receipts)
            if simulation_clock is not None:write('measured_latency_releases.json',simulation_clock.releases)
            write('stage_events.json',controller.events);write('planning.json',controller.planning)
            write('poses.json',published)
            if hasattr(controller,'mission_rows'):write('mission.json',controller.mission_rows)
            summary=dict(status='PACED_NATIVE_PREFIX_COMPLETE',physical_stop=stop,
                camera_frames=len(acquisitions),policy_steps=len(requests),physics_samples=len(session.samples),
                nonzero_requested_steps=sum(any(r['requested_command']) for r in requests),
                reasons=dict(Counter(r['reason'] for r in requests)),
                max_simulator_lag_ms=max(r['simulator_lag_ns']/1e6 for r in requests),
                acquisition_max_wall_ms=max((r['packet_available_ns']-r['acquisition_started_ns'])/1e6 for r in acquisitions),
                wall_s=(time.perf_counter_ns()-start)/1e9,simulation_s=(session.ctx.runner._sim_time_ns-EPOCH)/1e9,
                deadline_clock=CLOCK_MODE,planning_delay_ticks=PLANNING_DELAY_TICKS,
                plans_on_time=sum(p.get('on_time') is True for p in controller.planning),
                plans_late=sum(p.get('on_time') is False for p in controller.planning),
                disallowed_contact=any(bool(r['physics_contact']) for r in session.samples),
                physical_execution=True,shadow_requests_only=False,
                mission_terminal=getattr(controller,'mission_terminal',None),
                observed_arrivals=[] if not hasattr(controller,'mission') else controller.mission.arrivals,
                real_time_qualified=False,navigation_success_claimed=False,raw_sensor_audit_complete=False)
            write('result.json',summary)
        print('PACED_NATIVE_PREFIX_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as error:
        write('failure.json',dict(reason=repr(error),physical_stop=stop,
            acquired_frames=len(acquisitions),requested_steps=len(requests)))
        raise
    finally:
        if gc_receipt in gc.callbacks:gc.callbacks.remove(gc_receipt)
        if DEFER_CYCLIC_GC and gc_was_enabled:gc.enable()
        write('gc_timing_events.json',gc_events)
        if simulation_clock is not None:simulation_clock.close()
        if controller is not None:
            controller.stopped.set()
            for thread in controller.threads:thread.join(timeout=2.)
            for name,value in [('stage_events.json',controller.events),('planning.json',controller.planning),
                    ('poses.json',published),('pipeline_faults.json',controller.faults)]:
                if not (OUTPUT/name).exists():write(name,value)
            if hasattr(controller,'depth_observer') and not (OUTPUT/'independent_depth_receipts.json').exists():
                write('independent_depth_receipts.json',controller.depth_observer.receipts)
            if hasattr(controller,'mission_rows') and not (OUTPUT/'mission.json').exists():
                write('mission.json',controller.mission_rows)
            if hasattr(controller,'frontier_visits'):
                write('frontier_visits.json',dict(events=controller.frontier_visits.events,
                    pending=controller.frontier_visits.visit,excluded_cells=sorted(controller.frontier_visits.excluded)))
            if hasattr(controller,'initial_panorama'):
                write('initial_survey.json',controller.initial_panorama.state)
        write('requests.json',requests);write('acquisitions.json',acquisitions)
        if session is not None:
            # Preserve completed render results after a failure, without
            # submitting them to a stopped controller or claiming acceptance.
            drain_deadline=time.perf_counter()+20.
            while session.camera_pending:
                session.poll_sensor_packets()
                if time.perf_counter()>drain_deadline:break
                time.sleep(.001)
            write('asynchronous_camera_receipts.json',session.async_receipts)
            try:session.persist(directory);session.persist_observations(directory)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


if __name__=='__main__':main()
