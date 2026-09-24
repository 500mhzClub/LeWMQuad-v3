"""A single 60-second host-deadline probe of the current learned controller."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.process_registered_round_trip_development import registration_ready
from lewm.terminal_translation_pulse_development import PulseCommitmentLedger
from scripts import run_go2_stopping_projection_transfer_development as transfer

cohort = transfer.cohort
ROOT = 'go2_current_controller_wall_deadlines_layout01_600_v1_attempt_002'
REFERENCE = transfer.ROOT.format(index=1, condition='learned')


class PhysicalServiceLedger(PulseCommitmentLedger):
    """Keep host requests distinct from the physical command intervals served."""
    def __init__(self):
        super().__init__()
        self.pending_host_request = None
        self.service_receipts = []

    def record_request(self, now_ns, command):
        if self.pending_host_request is not None:
            raise ValueError('previous host request has no physical service receipt')
        self.pending_host_request = (now_ns, tuple(command))

    def record_service(self, simulator_ns, command):
        if self.pending_host_request is None:
            raise ValueError('physical service requires its actual host request')
        wall_ns, proposed = self.pending_host_request
        # No rounding, synthetic intervals, or changed host/sensor timestamps.
        super().record_request(simulator_ns, command)
        receipt = dict(simulator_ns=simulator_ns, host_request_ns=wall_ns,
            proposed_command=list(proposed), serviced_request=list(command),
            external_dispatch_changed_command=tuple(command)!=proposed)
        self.service_receipts.append(receipt)
        self.pending_host_request = None
        return receipt


class WallDeadlineRuntime(transfer.stopping.StoppingAwareRuntime):
    # Worker completion is already stamped by the actual host clock. Bypass
    # only the measured-simulation begin/end wrapper, which waits for physics.
    _worker = PacedMultirateController._worker

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commitment_ledger = PhysicalServiceLedger()

    def record_physical_service(self, simulator_ns, requested):
        with self.lock:
            receipt = self.commitment_ledger.record_service(simulator_ns, requested)
            if receipt['external_dispatch_changed_command']:
                now = receipt['host_request_ns']
                for plan in self.plans:
                    if plan.dispatch_ns <= now < plan.expires_ns:
                        self.rejected_windows[plan.observed_ns] = 'EXTERNAL_HOST_DISPATCH_VETO'
                        self.commitment_ledger.veto(plan, now)


def finish_wall(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='current_controller_host_deadline_probe_v1',
            comparison='same_learned_controller_with_actual_host_deadlines',
            comparison_condition='learned', reference_root_name=REFERENCE,
            planned_conditions=['learned'], planned_layout_indices=[1],
            planned_layout_count=1, planned_native_assignments=1,
            fixed_dispatch_pairs=[[[1, 'learned']]],
            new_independent_development_layout=False, exposed_development_layout=True,
            layout_novelty_scope='exposed_completed_transfer_maze',
            navigation_tick_budget=600, timing_probe_not_full_navigation_trial=True,
            intended_camera_frames=601, native_owners_during_probe=1,
            actual_runtime_class='WallDeadlineRuntime',
            current_controller_selection_and_perception_preserved=True,
            simulated_worker_release_waits_removed=True,
            physical_service_request_history=True,
            physical_history_records_post_host_veto_requests=True,
            prior_attempt_preserved='go2_current_controller_wall_deadlines_layout01_600_v1_attempt_001',
            host_clock_is_not_retimed=True, hardware_validated=False,
            extra_sources=value['extra_sources'] | {
                __file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(transfer.annotate, TREATMENT='learned',
        RAW_WRITE=bind(finish_wall, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('same CPU group as the measured layout-1 reference required')
    base = cohort.stable.source.BASE
    if not (base/'go2_stopping_projection_transfer_four_layout_summary_v1_attempt_001/result.json').is_file():
        raise ValueError('finish the fixed transfer comparison before the timing probe')
    output = base/ROOT
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve the single timing-probe outcome')
    writer = bind(cohort.make_writer, annotate=annotate)(output, 1, 'supervised_rollout')
    runtime_holder = {}

    def camera_session(*args, **kwargs):
        session = transfer.FreshCameraSession(*args, **kwargs)
        original_step = session.command_policy_step

        def command_step(requested):
            controller = runtime_holder.get('controller')
            if controller is not None:
                controller.record_physical_service(int(session.ctx.runner._sim_time_ns), requested)
            return original_step(requested)

        session.command_policy_step = command_step
        return session

    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            if kwargs.get('condition') != 'supervised_rollout':
                raise ValueError('same frozen supervised model required')
            controller = WallDeadlineRuntime(*args, motion_prediction_source='learned',
                registration_executor=executor, navigation_ticks=600,
                arrival_radius_m=.02, **kwargs)
            runtime_holder['controller'] = controller
            return controller

        # The base launcher selects this name in wall mode. Merely supplying
        # MEASURED_RUNTIME_CLASS would silently select its older controller.
        run = bind(cohort.stable.source.main, OUTPUT=output, COUNT=601, LAYOUT_INDEX=1,
            specification=transfer.layouts.specification,
            public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(camera_session,
                noise_layout_index=1, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='wall', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=transfer.stopping.previous.initialize_pose,
            OverlappedObstacleRuntime=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)
        try:
            run()
        finally:
            if 'controller' in runtime_holder:
                writer('physical_command_service_receipts.json',
                    runtime_holder['controller'].commitment_ledger.service_receipts)


if __name__ == '__main__':
    main()
