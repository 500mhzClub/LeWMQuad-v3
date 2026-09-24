"""Sixty-second current learned-controller probe with asynchronous cameras."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_current_controller_wall_deadlines_development as previous
from scripts import run_go2_async_camera_native_prefix_development as asynchronous
from scripts import snapshot_camera_renderer_development as renderer
from scripts.asynchronous_camera_session_development import AsynchronousCameraSession

transfer=previous.transfer; cohort=previous.cohort
ROOT='go2_async_camera_wall_deadlines_layout01_600_v1_attempt_004'
NAVIGATION_TICKS=600


def finish(name,value):
    if name=='launch.json':
        value=value|dict(experiment='asynchronous_camera_host_deadline_probe_v1',
            tracker='LocalViewRevisitMotion', asynchronous_camera_acquisition=True,
            physics_paused_during_rendering=False, native_scene_workers=2,
            renderer_scene_never_steps_physics=True,
            native_configuration_only_enters_sensor_renderer=True,
            prior_attempt_preserved='go2_async_camera_wall_deadlines_layout01_600_v1_attempt_003',
            sensor_packet_assembly_in_renderer_worker=True,
            completed_camera_results_drained_before_new_snapshot=True,
            host_request_physics_and_gc_timings_recorded=True,
            owner_cyclic_gc_deferred_during_bounded_probe=True,
            python_reference_counting_unchanged=True,
            snapshot_pixel_equivalence='go2_snapshot_camera_renderer_check_v1_attempt_002',
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'scripts/run_go2_async_camera_native_prefix_development.py',
                    'scripts/asynchronous_camera_session_development.py',
                    'scripts/snapshot_camera_renderer_development.py')})
    RAW_WRITE(name,value)


def annotate(name,value):
    bind(previous.annotate,RAW_WRITE=bind(finish,RAW_WRITE=RAW_WRITE))(name,value)


def main():
    if sorted(os.sched_getaffinity(0))!=cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('original wall-probe CPU group required')
    base=cohort.stable.source.BASE
    check=json.loads((base/'go2_snapshot_camera_renderer_check_v1_attempt_002/result.json').read_text())
    if not all(c['rgb_equal'] and c['depth_equal'] for r in check['rows'] for c in r['cameras']):
        raise ValueError('snapshot pixels must match both original cameras')
    output=base/ROOT
    if output.exists():raise ValueError('preserve this timing probe')
    writer=bind(cohort.make_writer,annotate=annotate)(output,1,'supervised_rollout')
    holder={}
    cohort.stable.floor.configure()
    with (ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as registration,
          ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=renderer.initialize,initargs=(transfer.layouts.specification(1),str(base/(ROOT+'_renderer')))) as camera):
        assert registration.submit(registration_ready).result()
        renderer_identity=camera.submit(renderer.ready).result()

        def camera_session(*args,**kwargs):
            session=AsynchronousCameraSession(*args,renderer_executor=camera,
                noise_layout_index=1,noise_sigma_mm=2,**kwargs)
            original_step=session.command_policy_step
            def command_step(requested):
                if 'controller' in holder:
                    holder['controller'].record_physical_service(int(session.ctx.runner._sim_time_ns),requested)
                return original_step(requested)
            session.command_policy_step=command_step
            return session

        def runtime(*args,**kwargs):
            controller=previous.WallDeadlineRuntime(*args,motion_prediction_source='learned',
                registration_executor=registration,navigation_ticks=NAVIGATION_TICKS,arrival_radius_m=.02,**kwargs)
            holder['controller']=controller
            return controller

        run=bind(asynchronous.main,OUTPUT=output,COUNT=NAVIGATION_TICKS+1,LAYOUT_INDEX=1,
            specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',PacedNativeSession=camera_session,
            write=writer,CLOCK_MODE='wall',PLANNING_DELAY_TICKS=3,
            DEFER_CYCLIC_GC=True,
            POSE_INITIALIZER=transfer.stopping.previous.initialize_pose,
            OverlappedObstacleRuntime=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)
        try:
            run()
        finally:
            if output.exists():
                for name,value in [('renderer_identity.json',renderer_identity),
                        ('physical_command_service_receipts.json',holder['controller'].commitment_ledger.service_receipts if 'controller' in holder else [])]:
                    with (output/name).open('x') as f:json.dump(value,f,indent=2)


if __name__=='__main__':main()
