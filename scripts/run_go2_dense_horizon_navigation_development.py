"""Dense-model maze navigation with physics paused for high-level computation.

This is a synchronous simulation treatment, not real-time qualification.
The existing command cadence, observed map, arrival and backtracking logic stay
in use. Prospective layouts require an explicit full-mission assignment.
"""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import shutil
import time
import traceback

import cv2
import psutil
import torch

from lewm.actuator_gain_development import configure_gains
from lewm.eligible_floor_registration_development import bind
from lewm import dense_world_model_maze_layouts_development as prospective
from lewm.dense_horizon_navigation_development import DenseNativeContextRuntimeMixin, load_dense_navigation_model
from lewm.paced_multirate_controller_development import AcquiredFrame
from lewm.process_mapped_runtime_development import mapping_ready, pose_ready
from lewm.untimed_simulation_clock_development import UntimedSimulationClock
from lewm.sparse_corner_comparison_development import ReactiveCompletionRuntime
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts import run_go2_sparse_corner_replication_development as replication
from scripts.run_go2_sparse_corner_completion_development import CompletionRuntime, initialize_pose
from scripts.pulse_context_setup_development import admit_context_setup
from scripts import train_go2_horizon_dense_predictor_development as fit
from scripts.rgb_navigation_retention_development import RGBNavigationRetentionMixin

OUTPUT = fit.OUTPUT.parent/'go2_dense_horizon_untimed_exposed_maze_pilot_v1_attempt_002'
COUNT = 161
INDEX = 0
EPOCH = 1_500_000_000
previous = replication.previous


class DenseNavigationRuntime(DenseNativeContextRuntimeMixin, CompletionRuntime):
    pass


class DenseReactiveNavigationRuntime(DenseNativeContextRuntimeMixin, ReactiveCompletionRuntime):
    """Same forecast workload, with the established observed-waypoint selector."""
    pass


class RGBOnlyCameraSession(RGBNavigationRetentionMixin, replication.FreshCameraSession):
    pass


class ProspectivePhysicalInit(replication.FreshPhysicalInit):
    __init__ = bind(previous.native.study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=prospective.specification, pack=prospective.pack)


class ProspectiveCameraSession(previous.native.NogilDrawingMixin,
        previous.native.study.cohort.LiveDepthNoiseMixin,
        previous.native.study.cohort.CompactDepthRetentionMixin,
        previous.native.study.cohort.LzmaRawDepthPairedCameraSession, ProspectivePhysicalInit):
    pass


class ProspectiveRGBOnlySession(RGBNavigationRetentionMixin, ProspectiveCameraSession):
    pass


def write(name, value):
    with (OUTPUT/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def drain(controller, *, timeout=120.):
    deadline = time.monotonic()+timeout
    while any(q.unfinished_tasks for q in controller.queues.values()):
        if controller.faults:
            raise RuntimeError(str(controller.faults))
        if time.monotonic()>deadline:
            raise TimeoutError('untimed perception/planning drain exceeded host timeout')
        time.sleep(.005)
    if controller.faults:
        raise RuntimeError(str(controller.faults))


def main(*, full_mission=False, arm='action', depth_retention='full', readout_arm='original',
        prospective_layout=None):
    global OUTPUT, COUNT
    if arm not in ('action', 'no_future_action', 'command_history', 'reactive_feedback'):
        raise ValueError('explicit dense-model or matched comparison arm required')
    if depth_retention not in ('full','rgb_only'):
        raise ValueError('explicit recording treatment required')
    if readout_arm not in ('original', 'old_data', 'mixed_data', 'maze_view_old_data', 'maze_view_maze_data'):
        raise ValueError('explicit readout treatment required')
    independent = prospective_layout is not None
    if independent and (not full_mission or type(prospective_layout) is not int
            or not 0 <= prospective_layout < prospective.LAYOUT_COUNT):
        raise ValueError('prospective assignment requires a full mission on layout 0–3')
    index = prospective_layout if independent else INDEX
    layouts = prospective if independent else replication.layouts
    if full_mission:
        OUTPUT = fit.OUTPUT.parent/'go2_dense_horizon_untimed_exposed_maze_full_v1_attempt_001'
        COUNT = 4814
    if arm != 'action':
        kind = 'full' if full_mission else 'pilot'
        OUTPUT = fit.OUTPUT.parent/f'go2_dense_horizon_untimed_{arm}_exposed_maze_{kind}_v1_attempt_001'
    if readout_arm != 'original':
        kind = 'full' if full_mission else 'pilot'
        OUTPUT = fit.OUTPUT.parent/f'go2_dense_horizon_untimed_{arm}_{readout_arm}_readout_exposed_maze_{kind}_v1_attempt_001'
    if independent:
        OUTPUT = fit.OUTPUT.parent/f'go2_dense_world_model_maze_layout{index:02d}_{arm}_{readout_arm}_v1_attempt_001'
    readout_followup = readout_arm.startswith('maze_view_')
    if readout_followup:
        if not independent or arm != 'action':
            raise ValueError('maze-view readout follow-up requires an explicit cohort layout and action predictor')
        from scripts.navigation_artifact_root_development import BASE, validate_root
        OUTPUT = BASE/OUTPUT.name
        validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists():
        raise ValueError('preserve every existing native attempt')
    reserve_gib = (5 if depth_retention=='full' else 2) if full_mission else 1
    if shutil.disk_usage(OUTPUT.parent).free<reserve_gib*1024**3:
        raise RuntimeError('recording headroom unavailable')
    if psutil.virtual_memory().available<32*1024**3:
        raise RuntimeError('native scene RAM headroom unavailable')
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    model_arm = 'no_future_action' if arm=='no_future_action' else 'action'
    model = load_dense_navigation_model(model_arm, readout_arm=readout_arm)
    runtime_type = DenseReactiveNavigationRuntime if arm=='reactive_feedback' else DenseNavigationRuntime
    previous.warmup()
    previous.study.cohort.stable.floor.configure()
    spec = layouts.specification(index)
    mission = layouts.public_mission(index)
    inventory_path = Path('docs/go2_dense_world_model_maze_inventory_2026-09-18.json')
    if independent:
        inventory = json.loads(inventory_path.read_text())
        assert {k: v for k, v in inventory.items() if k != 'source_sha256'} == json.loads(
            json.dumps(prospective.build_inventory()))
    sources = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        __file__, 'lewm/dense_horizon_navigation_development.py',
        'lewm/untimed_simulation_clock_development.py', 'lewm/dense_native_observation_development.py',
        'lewm/live_planning_stage_profile_development.py','scripts/rgb_navigation_retention_development.py')}
    sources[layouts.__file__] = fit.digest(layouts.__file__)
    OUTPUT.mkdir()
    directory = OUTPUT/'native'
    directory.mkdir()
    owner = psutil.Process()
    write('launch.json', dict(owner=dict(pid=owner.pid, created=owner.create_time()),
        experiment='maze_view_readout_exposed_navigation_v1' if readout_followup else 'dense_world_model_prospective_maze_v1' if independent else 'dense_horizon_untimed_exposed_maze_v1', source_sha256=sources,
        public_mission=mission, layout_index=index, layout_source=layouts.__file__,
        exposed_development_layout=not independent or readout_followup,
        new_independent_development_layout=independent and not readout_followup,
        layout_inventory_sha256=fit.digest(inventory_path) if independent else None,
        model_assignment=arm, model_arm=model_arm, actual_runtime_class=runtime_type.__name__,
        predictor_sha256=fit.digest(fit.OUTPUT/f'{model_arm}_final.pt'),
        motion_readout=model.readout_identity,
        neural_forecasts_used_for_selection=arm in ('action', 'no_future_action'),
        predictive_motion_used_for_selection=arm!='reactive_feedback',
        neural_computation_retained_for_workload_control=arm in ('command_history','reactive_feedback'),
        reactive_comparison_is_controller_package=arm=='reactive_feedback',
        model_check=str(Path('docs/go2_dense_horizon_navigation_check_2026-09-18.json')),
        observations=COUNT, physics_waits_for_high_level_result=True,
        deadline_clock='untimed_simulation', worker_cost_charged_to_simulation=False,
        mapping_completed_before_same_frame_planning=True, camera_period_ns=100_000_000,
        command_period_ns=20_000_000, physics_period_ns=2_000_000, planning_delay_ticks=3,
        planning_commit_ticks=4, full_native_rgb=True, contact_prediction_available=False,
        depth_noise_sigma_mm=2, simulation_sensing='existing paired depth and RGB plus ideal body gyro',
        depth_retention=depth_retention, retention_changes_live_inputs=False,
        raw_depth_archives_saved=depth_retention=='full',
        rgb_physics_commands_and_perception_receipts_retained=True,
        real_time_qualified=False, hardware_validated=False, final_evaluation=False,
        full_mission=full_mission,
        purpose='matched readout intervention on exposed development layouts' if readout_followup else 'prospective-maze round-trip evaluation' if independent else 'exposed-maze round-trip evaluation' if full_mission else 'bounded integration and physical command execution',
        independent_navigation_result=False, navigation_budget_ticks=4800,
        gpu_name=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        cpu_affinity=owner.cpu_affinity(), available_ram_bytes=psutil.virtual_memory().available))
    print('DENSE_NATIVE_PILOT_LAUNCHED', OUTPUT, flush=True)
    controller = session = None
    clock = UntimedSimulationClock()
    requests, acquisitions, published = [], [], []
    started = time.monotonic()
    error = None
    try:
        with contextlib.ExitStack() as stack:
            def pool(initializer):
                return stack.enter_context(ProcessPoolExecutor(max_workers=1,
                    mp_context=get_context('spawn'), initializer=initializer))
            registration = pool(previous.study.previous.reference.previous.initialize_registration)
            mapping = pool(previous.initialize_mapping)
            pose = pool(partial(initialize_pose, str(OUTPUT)))
            obstacles = pool(previous.study.previous.reference.previous.initialize_obstacles)
            assert registration.submit(previous.native.baseline.registration_ready).result()
            assert mapping.submit(mapping_ready).result()
            assert pose.submit(pose_ready).result()
            assert obstacles.submit(previous.study.cohort.stable.obstacles_ready).result()
            log = stack.enter_context((OUTPUT/'worker.log').open('x'))
            stack.enter_context(contextlib.redirect_stdout(log))
            stack.enter_context(contextlib.redirect_stderr(log))
            initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
            session_type = replication.FreshCameraSession if depth_retention=='full' else RGBOnlyCameraSession
            if independent:
                session_type = ProspectiveCameraSession if depth_retention=='full' else ProspectiveRGBOnlySession
            session = session_type(spec, directory, noise_layout_index=index, noise_sigma_mm=2)
            session.install_contact_identity()
            write('actuator_identity.json', configure_gains(session.ctx.build.robot,
                session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint'))
            session.settle_recorded()
            admit_context_setup(session, sources[__file__])
            def sink(frame, raw, registered):
                published.append(dict(frame=frame, raw_pose=raw['current_pose'], registered_pose=registered['current_pose']))
            controller = runtime_type(model, goal_initial_xy=mission['goal_initial_body_xy_m'],
                condition='jepa', variant='full', clock_ns=clock, evidence_sink=sink,
                planning_delay_ticks=3, maximum_initial_dispatch_lateness_ns=0,
                prediction_source='command_history' if arm=='command_history' else 'neural', registration_executor=registration,
                navigation_ticks=4800, arrival_radius_m=.02,
                mapping_executor=mapping, pose_executor=pose, obstacle_executor=obstacles)
            session.physics_clock_callback = clock.advance
            history = deque(maxlen=4)
            for tick in range(COUNT*5):
                sim_ns = int(session.ctx.runner._sim_time_ns)
                clock.advance(sim_ns)
                if tick%5==0:
                    capture_start = time.perf_counter_ns()
                    policy, depth, fast, auxiliary_depth, auxiliary_rgb, measured = session.sensor_packets()
                    history.append(policy)
                    packet = AcquiredFrame(tick//5, measured, policy, depth, fast,
                        auxiliary_rgb, auxiliary_depth, tuple(history))
                    controller.submit(packet)
                    drain(controller)
                    acquisitions.append(dict(frame=tick//5, measured_ns=measured,
                        completed_sim_ns=clock(), acquisition_and_pipeline_wall_ns=time.perf_counter_ns()-capture_start))
                    if tick%100==0:
                        print('DENSE_NAVIGATION_PROGRESS', json.dumps(dict(frame=tick//5,
                            model_calls=len(model.receipts), mission=controller.mission_latest)), flush=True)
                request = controller.request(now_ns=clock())
                row = dict(request, simulator_ns=sim_ns, pre_sample_index=len(session.samples)-1)
                requests.append(row)
                session.phase = 2
                row['applied_command'] = session.command_policy_step(request['requested_command'])
                row['post_sample_index'] = len(session.samples)-1
                if controller.faults:
                    raise RuntimeError(str(controller.faults))
                if controller.mission_terminal is not None:
                    break
            controller.finish()
            write('result.json', dict(status='FULL_MISSION_RUN_COMPLETE' if full_mission else 'BOUNDED_PILOT_COMPLETE', camera_frames=len(acquisitions),
                policy_steps=len(requests), model_calls=len(model.receipts),
                nonzero_requested_steps=sum(any(r['requested_command']) for r in requests),
                reasons=dict(Counter(r['reason'] for r in requests)),
                disallowed_contact=any(bool(r['physics_contact']) for r in session.samples),
                mission_terminal=controller.mission_terminal, observed_arrivals=controller.mission.arrivals,
                wall_s=time.monotonic()-started, simulation_s=(session.ctx.runner._sim_time_ns-EPOCH)/1e9,
                physical_execution=True, real_time_qualified=False, navigation_success_claimed=False))
    except BaseException as exc:
        error = exc
        write('failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(),
            acquired_frames=len(acquisitions), requested_steps=len(requests)))
    finally:
        if controller is not None:
            controller.stopped.set()
            clock.close()
            for thread in controller.threads:
                thread.join(timeout=2.)
            for name, value in [('stage_events', controller.events), ('planning', controller.planning),
                    ('poses', published), ('pipeline_faults', controller.faults), ('mission', controller.mission_rows),
                    ('live_planning_profile', controller.plan_profile_rows),
                    ('visual_dispatch_events', controller.visual_dispatch_events)]:
                write(name+'.json', value)
            if hasattr(controller, 'depth_observer'):
                write('independent_depth_receipts.json', controller.depth_observer.receipts)
        write('worker_timing.json', clock.releases)
        write('dense_model_calls.json', model.receipts)
        write('requests.json', requests)
        write('acquisitions.json', acquisitions)
        if session is not None:
            try:
                session.persist(directory)
                session.persist_observations(directory)
            finally:
                session.ctx.build.scene.destroy()
        shutdown_genesis()
    if error is not None:
        raise error
    print('DENSE_NATIVE_PILOT_COMPLETE', OUTPUT, flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-mission', action='store_true')
    parser.add_argument('--arm', choices=('action','no_future_action','command_history','reactive_feedback'), default='action')
    parser.add_argument('--depth-retention', choices=('full','rgb_only'), default='full')
    parser.add_argument('--readout-arm', choices=('original','old_data','mixed_data','maze_view_old_data','maze_view_maze_data'), default='original')
    parser.add_argument('--prospective-layout', type=int, choices=range(4))
    args = parser.parse_args()
    main(full_mission=args.full_mission, arm=args.arm, depth_retention=args.depth_retention,
        readout_arm=args.readout_arm, prospective_layout=args.prospective_layout)
