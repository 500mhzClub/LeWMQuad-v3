"""Full current-feedback control under the learned mission's host deadlines."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
import os
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_async_camera_wall_mission_development as learned

probe=learned.probe
previous=probe.previous
transfer=probe.transfer
cohort=probe.cohort
ROOT='go2_async_camera_wall_mission_reactive_layout01_4800_v1_attempt_001'


class WallReactiveRuntime(transfer.CurrentReactiveRuntime):
    _worker=PacedMultirateController._worker
    record_physical_service=previous.WallDeadlineRuntime.record_physical_service

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.commitment_ledger=previous.PhysicalServiceLedger()


def finish(name,value):
    if name=='launch.json':
        value=value|dict(experiment='asynchronous_wall_instantaneous_reactive_control_v1',
            comparison='complete_learned_predictive_vs_instantaneous_reactive_control',
            comparison_condition='reactive',actual_runtime_class='WallReactiveRuntime',
            reference_root_name=learned.ROOT,planned_conditions=['learned','reactive'],
            planned_layout_indices=[1],planned_layout_count=1,planned_native_assignments=2,
            fixed_dispatch_pairs=None,new_independent_development_layout=False,
            exposed_development_layout=True,layout_novelty_scope='exposed_completed_transfer_maze',
            navigation_tick_budget=4800,intended_camera_frames=4801,full_mission_implemented=True,
            timing_probe_not_full_navigation_trial=False,
            owner_cyclic_gc_deferred_during_bounded_probe=False,
            owner_cyclic_gc_deferred_during_bounded_mission=True,
            simulated_worker_release_waits_removed=True,
            physical_service_request_history=True,physical_history_records_post_host_veto_requests=True,
            host_clock_is_not_retimed=True,native_owners_during_probe=1,
            motion_prediction_source='none',forecast_xy_source='none',forecast_yaw_source='none',
            learned_yaw_retained=False,neural_xy_used_for_scoring=False,
            neural_outcomes_used_for_scoring=False,fully_model_free_controller=True,
            learned_model_used_for_action_selection=False,candidate_future_outcomes_evaluated=False,
            neural_inference_computed=False,planned_stopping_projection=False,
            predictive_clearance_and_stopping_projection_absent_in_reactive=True,
            reactive_recovery_rules_differ_from_predictive=True,
            isolated_predictive_scoring_effect_established=False,
            both_native_missions_run_alone=True,hardware_validated=False,
            extra_sources=value['extra_sources']|{
                __file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                previous.__file__:hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest()})
    RAW_WRITE(name,value)


def annotate(name,value):
    emit=bind(probe.finish,RAW_WRITE=bind(finish,RAW_WRITE=RAW_WRITE))
    bind(transfer.annotate,TREATMENT='reactive',RAW_WRITE=emit)(name,value)


def main():
    if sorted(os.sched_getaffinity(0))!=cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('same CPU group as learned wall-clock reference required')
    base=cohort.stable.source.BASE;output=base/ROOT
    if output.exists():raise ValueError('preserve this full reactive attempt')
    writer=bind(cohort.make_writer,annotate=annotate)(output,1,'reactive')
    holder={};cohort.stable.floor.configure()
    with (ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as registration,
          ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=probe.renderer.initialize,
            initargs=(transfer.layouts.specification(1),str(base/(ROOT+'_renderer')))) as camera):
        assert registration.submit(registration_ready).result()
        renderer_identity=camera.submit(probe.renderer.ready).result()
        def camera_session(*args,**kwargs):
            session=probe.AsynchronousCameraSession(*args,renderer_executor=camera,
                noise_layout_index=1,noise_sigma_mm=2,**kwargs)
            original=session.command_policy_step
            def step(requested):
                if 'controller' in holder:
                    holder['controller'].record_physical_service(int(session.ctx.runner._sim_time_ns),requested)
                return original(requested)
            session.command_policy_step=step
            return session
        def runtime(*args,**kwargs):
            controller=WallReactiveRuntime(*args,registration_executor=registration,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)
            holder['controller']=controller
            return controller
        try:
            bind(probe.asynchronous.main,OUTPUT=output,COUNT=4801,LAYOUT_INDEX=1,
                specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
                MODEL_ASSIGNMENT='reactive',PacedNativeSession=camera_session,write=writer,
                CLOCK_MODE='wall',PLANNING_DELAY_TICKS=3,DEFER_CYCLIC_GC=True,
                POSE_INITIALIZER=transfer.stopping.previous.initialize_pose,
                OverlappedObstacleRuntime=runtime,
                OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,OBSTACLE_READY=cohort.stable.obstacles_ready,
                initialize_mapping=cohort.learned.initialize_mapping)()
        finally:
            if output.exists():
                for name,value in [('renderer_identity.json',renderer_identity),
                        ('physical_command_service_receipts.json',holder['controller'].commitment_ledger.service_receipts
                            if 'controller' in holder else [])]:
                    with (output/name).open('x') as f:json.dump(value,f,indent=2)


if __name__=='__main__':main()
