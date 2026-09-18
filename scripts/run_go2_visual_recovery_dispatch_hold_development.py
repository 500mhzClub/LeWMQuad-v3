"""One prospective exposed-layout JEPA trial of prompt recovery holds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from scripts import run_go2_persistent_visual_learning_comparison_development as previous
from lewm.visual_recovery_dispatch_hold_development import VisualRecoveryDispatchHoldRuntime
from lewm.planning_latency_stress_development import PlanningLatencyClock
from lewm.live_planning_profile_development import LivePlanningProfileRuntime
from lewm.live_planning_call_profile_development import LivePlanningCallProfileRuntime
from lewm.live_planning_stage_profile_development import LivePlanningStageProfileRuntime
from lewm.shared_candidate_history_development import SharedCandidateHistoryRuntime
from lewm.isolated_forecast_development import IsolatedForecastRuntime, initialize_forecast, forecast_ready
from scripts.nogil_drawing_session_development import NogilDrawingCameraSession

ROOT = 'go2_visual_recovery_dispatch_hold_jepa_noise_2mm_native_layout03_4800_v1_attempt_001'
REFERENCE = previous.ROOT.format(arm='jepa', index=3)
SOURCES = (__file__, 'lewm/visual_recovery_dispatch_hold_development.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--control', action='store_true', help='repeat the original controller without the new hold')
    parser.add_argument('--planning-extra-ms', type=int, choices=(0, 20), default=0)
    parser.add_argument('--arm', choices=('jepa', 'supervised_rollout'), default='jepa')
    parser.add_argument('--profile-planning', action='store_true')
    parser.add_argument('--profile-calls', action='store_true')
    parser.add_argument('--profile-stages', action='store_true')
    parser.add_argument('--shared-history', action='store_true')
    parser.add_argument('--isolated-forecast', action='store_true')
    parser.add_argument('--nogil-drawing', action='store_true')
    parser.add_argument('--parent-switch-ms', type=int, choices=(1, 5), default=5)
    args = parser.parse_args()
    if (args.planning_extra_ms and not args.control) or (args.arm != 'jepa' and not args.planning_extra_ms):
        raise ValueError('latency experiment uses the original controller; original hold study is JEPA only')
    study = previous.study
    root = ROOT if not args.control else 'go2_visual_recovery_original_control_jepa_noise_2mm_native_layout03_4800_v1_attempt_001'
    if args.planning_extra_ms:
        root = f'go2_planning_latency_plus20ms_{args.arm}_noise_2mm_native_layout03_4800_v1_attempt_001'
    runtime_class = previous.PersistentLocalVisualRuntime if args.control else VisualRecoveryDispatchHoldRuntime
    ticks = 4800
    if args.profile_planning:
        if not args.control or args.arm != 'jepa' or args.planning_extra_ms != 20:
            raise ValueError('fixed original JEPA plus20ms short live profile required')
        root = 'go2_live_planning_profile_jepa_plus20ms_layout03_800_v1_attempt_001'
        runtime_class = LivePlanningProfileRuntime
        ticks = 800
    if args.parent_switch_ms == 1:
        if not args.control or args.planning_extra_ms != 20 or args.profile_planning:
            raise ValueError('fixed full-mission plus20ms scheduling experiment required')
        root = f'go2_planning_switch1ms_plus20ms_{args.arm}_noise_2mm_native_layout03_4800_v1_attempt_001'
        runtime_class = LivePlanningProfileRuntime
    if args.profile_calls:
        if not args.profile_planning or args.parent_switch_ms != 5:
            raise ValueError('call profiling requires the original-switch short timing diagnostic')
        root = 'go2_live_planning_call_profile_jepa_plus20ms_layout03_800_v1_attempt_001'
        runtime_class = LivePlanningCallProfileRuntime
    if args.profile_stages:
        if not args.profile_planning or args.profile_calls or args.parent_switch_ms != 5:
            raise ValueError('direct stage profiling requires the original-switch short timing diagnostic')
        root = 'go2_live_planning_stage_profile_jepa_plus20ms_layout03_800_v1_attempt_001'
        runtime_class = LivePlanningStageProfileRuntime
    if args.shared_history:
        if not args.control or args.planning_extra_ms != 20 or args.parent_switch_ms != 5 or args.profile_planning:
            raise ValueError('shared-history experiment retains full missions, original switch and +20ms delay')
        root = f'go2_shared_history_plus20ms_{args.arm}_noise_2mm_native_layout03_4800_v1_attempt_001'
        runtime_class = SharedCandidateHistoryRuntime
    if args.isolated_forecast:
        if not args.shared_history:
            raise ValueError('process isolation retains the shared-history +20ms full-mission condition')
        root = f'go2_isolated_forecast_plus20ms_{args.arm}_noise_2mm_native_layout03_4800_v1_attempt_001'
        runtime_class = IsolatedForecastRuntime
    if args.nogil_drawing:
        if not args.control or args.planning_extra_ms != 20 or args.parent_switch_ms != 5 or args.profile_planning or args.shared_history:
            raise ValueError('drawing experiment retains original inference, switch, full budget and +20ms delay')
        root = f'go2_nogil_drawing_plus20ms_{args.arm}_noise_2mm_native_layout03_4800_v1_attempt_001'
        runtime_class = LivePlanningStageProfileRuntime
    output = study.BASE/root
    if args.evaluate:
        selected = SimpleNamespace(**(vars(study) | dict(
            ROOT=root, ASSIGNMENTS=((3, args.arm),), PLAN=previous.PLAN)))
        print(json.dumps(previous.bind(previous.evaluation.evaluate, study=selected)(1), indent=2))
        return
    if output.exists():
        raise ValueError('preserve the prospective attempt')
    if shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('same layout-3 CPU allocation required')
    before_switch = sys.getswitchinterval()
    if before_switch != .005:
        raise ValueError('original parent switch interval required before experiment')
    sys.setswitchinterval(args.parent_switch_ms/1000)
    reference_root = REFERENCE if args.arm == 'jepa' else 'go2_persistent_visual_transfer_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001'
    reference = json.loads((study.BASE/reference_root/'launch.json').read_text())
    resources = previous.hardware()
    runtimes = []

    def write(name, value):
        if name == 'launch.json':
            value = reference | value | dict(experiment='visual_recovery_dispatch_hold_v1',
                hardware=resources, reference_root_name=reference_root,
                comparison_condition='original_controller_repeat' if args.control else 'hold_pre_trigger_commands_on_recovery_publication',
                runtime_class=runtime_class.__name__,
                parent_python_switch_interval_before_s=before_switch,
                parent_python_switch_interval_s=sys.getswitchinterval(),
                spawned_worker_switch_interval_changed=False,
                controller_changed=not args.control, world_model_changed=False,
                fixed_sequential_assignments=[[3, args.arm]], planned_native_assignments=1,
                prospective_plan='docs/go2_visual_recovery_dispatch_hold_2026-09-16.md',
                extra_sources=previous.source_hashes() | {
                    p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES})
            value.pop('prospective_plan_sha256', None)
            if args.planning_extra_ms:
                value.update(experiment='planning_latency_plus20ms_v1',
                    comparison_condition='original_controller_plus20ms_planning_publication',
                    planning_extra_ns=args.planning_extra_ms*1_000_000,
                    prospective_plan='docs/go2_planning_latency_stress_2026-09-16.md',
                    extra_sources=value['extra_sources'] | {
                        'lewm/planning_latency_stress_development.py': hashlib.sha256(
                            Path('lewm/planning_latency_stress_development.py').read_bytes()).hexdigest()})
            if args.profile_planning or args.parent_switch_ms == 1 or args.shared_history or args.nogil_drawing:
                value.update(experiment='live_planning_profile_v1' if args.profile_planning else 'planning_switch1ms_plus20ms_v1', navigation_tick_budget=ticks,
                    prospective_plan='docs/go2_planning_cost_profile_2026-09-16.md',
                    short_timing_diagnostic=args.profile_planning, complete_navigation_trial=not args.profile_planning,
                    extra_sources=value['extra_sources'] | {
                        'lewm/live_planning_profile_development.py': hashlib.sha256(
                            Path('lewm/live_planning_profile_development.py').read_bytes()).hexdigest()})
            if args.profile_calls:
                value.update(experiment='live_planning_call_profile_v1',
                    sparse_cprofile_frames=list(range(300, 348, 4)),
                    profiling_can_change_deadlines=True,
                    extra_sources=value['extra_sources'] | {
                        'lewm/live_planning_call_profile_development.py': hashlib.sha256(
                            Path('lewm/live_planning_call_profile_development.py').read_bytes()).hexdigest()})
            if args.profile_stages or args.shared_history or args.nogil_drawing:
                value.update(experiment='live_planning_stage_profile_v1',
                    extra_sources=value['extra_sources'] | {
                        'lewm/live_planning_stage_profile_development.py': hashlib.sha256(
                            Path('lewm/live_planning_stage_profile_development.py').read_bytes()).hexdigest()})
            if args.shared_history:
                value.update(experiment='shared_candidate_history_plus20ms_v1',
                    inference_shared_history_encoding=True,
                    floating_point_batch_order_changed=True,
                    extra_sources=value['extra_sources'] | {
                        'lewm/shared_candidate_history_development.py': hashlib.sha256(
                            Path('lewm/shared_candidate_history_development.py').read_bytes()).hexdigest()})
            if args.isolated_forecast:
                value.update(experiment='isolated_forecast_shared_history_plus20ms_v1',
                    inference_worker=forecast_identity,
                    transfer_and_result_wait_charged_to_planning=True,
                    extra_sources=value['extra_sources'] | {
                        'lewm/isolated_forecast_development.py': hashlib.sha256(
                            Path('lewm/isolated_forecast_development.py').read_bytes()).hexdigest()})
            if args.nogil_drawing:
                value.update(experiment='nogil_camera_drawing_plus20ms_v1',
                    inference_shared_history_encoding=False,
                    extra_sources=value['extra_sources'] | {
                        p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                            'scripts/nogil_drawing_session_development.py',
                            'lewm_genesis/lewm_genesis/nogil_readback_development.py')})
        previous.bind(study.source.write, OUTPUT=output)(name, value)
        if name == 'requests.json' and runtimes:
            previous.bind(study.source.write, OUTPUT=output)(
                'visual_dispatch_events.json', getattr(runtimes[0], 'visual_dispatch_events', []))
            if args.planning_extra_ms:
                previous.bind(study.source.write, OUTPUT=output)(
                    'planning_latency_stress.json', runtimes[0].clock_ns.rows)
            if args.profile_planning or args.parent_switch_ms == 1 or args.shared_history or args.nogil_drawing:
                previous.bind(study.source.write, OUTPUT=output)(
                    'live_planning_profile.json', runtimes[0].plan_profile_rows)
            if args.profile_calls:
                runtimes[0].save_call_profiles(output)
            if args.shared_history:
                previous.bind(study.source.write, OUTPUT=output)(
                    'shared_history_treatment.json', runtimes[0].shared_history_receipt)
            if args.isolated_forecast:
                previous.bind(study.source.write, OUTPUT=output)(
                    'isolated_forecast_receipts.json', runtimes[0].forecast_receipts)

    study.cohort.stable.floor.configure()
    with ExitStack() as stack:
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration))
        assert executor.submit(previous.registration_ready).result()
        forecast_executor = forecast_identity = None
        if args.isolated_forecast:
            forecast_executor = stack.enter_context(ProcessPoolExecutor(max_workers=1,
                mp_context=get_context('spawn'), initializer=initialize_forecast, initargs=(args.arm,)))
            forecast_identity = forecast_executor.submit(forecast_ready).result(timeout=60.)

        def runtime(model, **kwargs):
            if args.isolated_forecast:
                kwargs.update(forecast_executor=forecast_executor, forecast_identity=forecast_identity)
            if args.planning_extra_ms:
                kwargs['clock_ns'] = PlanningLatencyClock(kwargs['clock_ns'], args.planning_extra_ms*1_000_000)
            result = runtime_class(model, prediction_source='neural',
                registration_executor=executor, navigation_ticks=ticks,
                arrival_radius_m=.02, **kwargs)
            runtimes.append(result)
            return result

        previous.bind(study.source.main, OUTPUT=output, COUNT=ticks+14, LAYOUT_INDEX=3,
            specification=previous.layouts.specification, public_mission=previous.layouts.public_mission,
            MODEL_ASSIGNMENT=args.arm, MODEL_LOADER=previous.bind(study.load_model, PLAN=previous.PLAN),
            PacedNativeSession=partial(NogilDrawingCameraSession if args.nogil_drawing else previous.FreshCameraSession,
                noise_layout_index=3, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(previous.initialize_pose, str(output)), MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()
