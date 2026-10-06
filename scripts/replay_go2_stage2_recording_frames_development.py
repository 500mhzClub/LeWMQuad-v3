"""Regenerate the consumed RGB frames of the stage-2 recordings by verified replay (development; 5 October 2026).

Same method as the C3-v3 round's replays (scripts/replay_go2_c3v3_onpolicy_frames_development.py). It runs pass 1 of the
capability video renderer (render_go2_capability_v4_video_development.verify):
- it re-simulates the mission from its seed, recomputing C1's command model;
- it asserts every consumed-packet hash, selected action, C1 forecast, dispatch command and reason, applied command,
  native trace value and published pose against the log;
- it keeps the consumed primary RGB as ego_frames/.

`verify` here is a copy of the renderer's, with three documented differences that reproduce how the stage-2 recordings
ran (v12 pinned entry):
1. **Runtime.** It is composed as the pinned entry composed it: the run's development fixes (dev_run.json) plus the
   harness version's mixins outermost (launch_pin.json: reserve_exit_v2, ReserveExitMixin).
2. **Session.** owner.make_session is wrapped by the patches-v3 session with the run's recorded placement (launch_pin
   dynamics): the per-leg friction field, the tinted marker and the 0.40-m/s guard. The stall stop is not reinstalled.
3. **Missions ended by the stall stop.** The last request's command step was interrupted in the original (no
   post_sample_index), so that one step is not replayed. Its frame, decision and dispatch request are still checked.
   Traces and published poses are compared as prefixes, and mission rows and finish() are skipped. Round trips are
   checked in full, as before.
4. **Missions ended by a failure on their final frame** (fit14: "measured visual pose unavailable"). The frame was
   acquired one tick after the last logged request, and the failure stopped the mission before that tick's request. The
   loop over requests therefore ends one frame short. That final frame is acquired after the loop and its consumed-packet
   record is checked (it is not submitted to the controller). Published poses are compared as a prefix, and mission rows
   and finish() are skipped, as for the stall stop.

Contact-stopped missions are skipped: they contribute nothing to the training set.

Usage: replay_go2_stage2_recording_frames_development.py --cohort s2rec2 [--workers N] | --run ASSIGNMENT
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
from functools import partial
import importlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import yaml

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from lewm.navigation_capability_target_reference_development import install_task_cues, settled_task_cues
from lewm.navigation_capability_unused_workload_development import UnusedNeuralWorkload
from lewm.physical_execution_development import rotation_xyzw
from lewm_genesis.lewm_contract import SafetyLimits
from scripts import render_go2_capability_v4_video_development as renderer
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.render_go2_navigation_capability_pipeline_development import compare_poses
from scripts.run_go2_capability_completed_support_v4_gate_erratum_continuation_development import ENVIRONMENT
from scripts.run_go2_dev_mission_development import DevBudget

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
OUT = BASE/'stage2_recording_replays'
HARNESSES = {'reserve_exit_v2': 'lewm.dev_harness_reserve_exit_v2_development'}
source = owner.source


def runtime_mixin(source_root):
    from lewm.dev_harness_fixes_development import compose
    pin = json.loads((source_root/'launch_pin.json').read_text())
    dev = json.loads((source_root/'dev_run.json').read_text())
    version = importlib.import_module(HARNESSES[pin['harness']])
    mixins = tuple(version.mixins_for('C1'))
    assert [m.__name__ for m in mixins] == pin['mixins'], 'harness mixins differ from the launch pin'
    return compose(dev['fixes'], CompletedSupportRuntimeMixin, extra=mixins), pin


def install_patches(pin):
    from lewm.dev_dynamics_patches_v3_development import patch_session
    d = pin['dynamics']
    assert d['patches_version'] == 'v3' and d['placement_version'] == 'v2'
    owner.make_session = patch_session(owner.make_session, d['placement'], d['mu'], d['marked'])


def verify(source_root, root, budget, spec, packet, trace, requests, frames, mixin):
    """renderer.verify with the stage-2 differences listed in the module docstring."""
    receipts = json.loads((source_root/'model_calls.json').read_text())
    limits = SafetyLimits.from_manifest(yaml.safe_load((owner.REPO/'config/go2_platform_manifest.yaml').read_text()))
    model = UnusedNeuralWorkload('C1', limits)
    expected_calls = {r['observed_ns']: r for r in receipts}
    expected_plans = {r['frame']: r for r in json.loads((source_root/'planning.json').read_text()) if 'selection' in r}
    truncated = 'post_sample_index' not in requests[-1]
    final_frame = (not truncated and (source_root/'failure.json').exists()
                   and len(frames) == (len(requests)+4)//5+1)
    ego = root/'ego_frames'
    ego.mkdir()
    directory = root/'verification_native'
    directory.mkdir()
    clock = source.UntimedSimulationClock()
    controller = session = None
    published = []
    counts = dict(frames=0, decisions=0, dispatch=0)
    maximum = [0., 0.]
    try:
        with contextlib.ExitStack() as stack:
            def pool(initializer):
                return stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'), initializer=initializer))
            registration = pool(source.previous.study.previous.reference.previous.initialize_registration)
            mapping = pool(renderer.initialize_startup_mapping)
            pose = pool(partial(source.initialize_pose, str(root)))
            obstacles = pool(source.previous.study.previous.reference.previous.initialize_obstacles)
            assert registration.submit(source.previous.native.baseline.registration_ready).result()
            assert mapping.submit(source.mapping_ready).result()
            assert pose.submit(source.pose_ready).result()
            assert obstacles.submit(source.previous.study.cohort.stable.obstacles_ready).result()
            session = renderer.start_session(spec, directory, full_frames=True)
            for key, values in trace.items():
                np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]), values[:len(session.samples)])
            start = np.asarray(session.samples[-1]['base_pose_world'], dtype=float)
            cues = settled_task_cues(packet, start[:3], rotation_xyzw(start[3:]))
            if renderer.jsonable(cues) != json.loads((source_root/'public_task_cues.json').read_text()):
                raise ValueError('replayed settled task cues differ')

            def sink(frame, raw, registered):
                published.append(dict(frame=frame, raw_pose=raw['current_pose'], registered_pose=registered['current_pose'],
                    visual_support=registered.get('visual_support')))
            runtime = type('Stage2RecordingRuntime', (mixin, source.DenseNavigationRuntime), {})
            controller = runtime(model, goal_initial_xy=cues['goal_initial_body_xy_m'], condition='jepa', variant='full',
                clock_ns=clock, evidence_sink=sink, planning_delay_ticks=3, maximum_initial_dispatch_lateness_ns=0,
                prediction_source='command_history', registration_executor=registration,
                navigation_ticks=4800, arrival_radius_m=.02, mapping_executor=mapping, pose_executor=pose, obstacle_executor=obstacles)
            install_task_cues(controller, cues)
            session.physics_clock_callback = clock.advance
            history = deque(maxlen=4)
            for tick, expected in enumerate(requests):
                budget.check()
                now = int(session.ctx.runner._sim_time_ns)
                clock.advance(now)
                if now != expected['simulator_ns']:
                    raise ValueError(f'replay clock differs at tick {tick}')
                if tick % 5 == 0:
                    policy, depth, fast, auxiliary_depth, auxiliary_rgb, measured = session.sensor_packets()
                    history.append(policy)
                    camera = session.captured_pairs[-1]
                    record, logged = camera['consumed_hash_record'], frames[tick//5]
                    for key in ('pixel_sha256', 'live_depth_noise', 'consumed_packet_sha256', 'arrays', 'measured_ns', 'physical_sample_index'):
                        if record[key] != logged[key]:
                            raise ValueError(f'consumed packet record differs: frame {tick//5} {key}')
                    Image.fromarray(np.asarray(camera['images'][0][0])).save(ego/f'{tick//5:04d}.png')
                    camera['images'] = []
                    counts['frames'] += 1
                    acquired = source.AcquiredFrame(tick//5, measured, policy, depth, fast, auxiliary_rgb, auxiliary_depth, tuple(history))
                    controller.submit(acquired)
                    owner.drain(controller, model, session, budget)
                    actual_plan = next((r for r in reversed(controller.planning) if r['frame'] == acquired.frame and 'selection' in r), None)
                    expected_plan = expected_plans.get(acquired.frame)
                    if (actual_plan is None) != (expected_plan is None):
                        raise ValueError(f'decision availability differs at frame {acquired.frame}')
                    if expected_plan is not None:
                        if actual_plan['selection']['action'] != expected_plan['selection']['action']:
                            raise ValueError(f'selected action differs at frame {acquired.frame}')
                        np.testing.assert_array_equal(model.receipts[-1]['requested_commands'],
                                                      expected_calls[acquired.measured_ns]['requested_commands'])
                        np.testing.assert_array_equal(actual_plan['motion_correction']['applied_prediction_after_yaw_ablation'],
                                                      expected_plan['motion_correction']['applied_prediction_after_yaw_ablation'])
                        counts['decisions'] += 1
                request = controller.request(now_ns=clock())
                if request['requested_command'] != expected['requested_command'] or request['reason'] != expected['reason']:
                    raise ValueError(f'dispatch command/reason differs at tick {tick}')
                if truncated and tick == len(requests)-1:
                    break  # the original step was interrupted by the stall stop
                session.phase = 2
                applied = session.command_policy_step(request['requested_command'])
                np.testing.assert_array_equal(applied, expected['applied_command'])
                counts['dispatch'] += 1
                end = expected['post_sample_index']
                for key, values in trace.items():
                    np.testing.assert_array_equal(np.stack([r[key] for r in session.samples[-10:]]), values[end-9:end+1])
                position, yaw = compare_poses(np.stack([r['base_pose_world'] for r in session.samples[-10:]]),
                                              trace['base_pose_world'][end-9:end+1])
                maximum = [max(maximum[0], position), max(maximum[1], yaw)]
                if controller.mission_terminal is not None and tick != len(requests)-1:
                    raise ValueError('replay mission terminated early')
            if final_frame:
                clock.advance(int(session.ctx.runner._sim_time_ns))
                session.sensor_packets()
                camera = session.captured_pairs[-1]
                record, logged = camera['consumed_hash_record'], frames[-1]
                for key in ('pixel_sha256', 'live_depth_noise', 'consumed_packet_sha256', 'arrays', 'measured_ns', 'physical_sample_index'):
                    if record[key] != logged[key]:
                        raise ValueError(f'consumed packet record differs: final frame {key}')
                Image.fromarray(np.asarray(camera['images'][0][0])).save(ego/f'{len(frames)-1:04d}.png')
                camera['images'] = []
                counts['frames'] += 1
            if counts['decisions'] != len(expected_plans) or counts['frames'] != len(frames):
                raise ValueError('decision or frame count differs')
            logged_poses = json.loads((source_root/'poses.json').read_text())
            if truncated or final_frame:
                if renderer.jsonable(published) != logged_poses[:len(published)]:
                    raise ValueError('published poses differ (prefix)')
            else:
                if len(session.samples) != len(trace['timestamp_s']):
                    raise ValueError('replayed native trace length differs')
                controller.finish()
                if renderer.jsonable(published) != logged_poses:
                    raise ValueError('published poses differ')
                if renderer.jsonable(controller.mission_rows) != json.loads((source_root/'mission.json').read_text()):
                    raise ValueError('mission rows differ')
            return dict(passed=True, controller='C1', truncated_by_stall_stop=truncated, ended_by_final_frame_failure=final_frame,
                        bitwise_consumed_packet_records=counts['frames'], identical_selected_actions=counts['decisions'],
                        identical_dispatch_command_reason_steps=counts['dispatch'], identical_published_poses=len(published),
                        maximum_position_error_m=maximum[0], maximum_yaw_error_degrees=maximum[1], exact_native_trace_values=True)
    finally:
        if controller is not None:
            controller.stopped.set()
            for thread in controller.threads:
                thread.join(timeout=2.)
        clock.close()
        if session is not None:
            session.ctx.build.scene.destroy()


def replay(assignment):
    source_root = BASE/'runs'/assignment
    config = json.loads((source_root/'config.json').read_text())
    assert config['controller'] == 'C1'
    packet = json.loads((source_root/'episode.json').read_text())
    assert packet['role'] in ('stage2_fit', 'stage2_heldout')
    mixin, pin = runtime_mixin(source_root)
    install_patches(pin)
    output.install(BASE)
    root = OUT/assignment
    root.mkdir(parents=True, exist_ok=False)
    budget = DevBudget(BASE, json.loads(owner.PROTOCOL.read_text()))
    spec = json.loads((source_root/'specification.json').read_text())
    requests = json.loads((source_root/'requests.json').read_text())
    frames = json.loads((source_root/'native/in_memory_camera_observations.json').read_text())['frames']
    with np.load(source_root/'native/physics_trace.npz', allow_pickle=False) as a:
        trace = {k: a[k].copy() for k in a.files}
    started = time.monotonic()
    try:
        renderer.source.previous.warmup()
        renderer.source.previous.study.cohort.stable.floor.configure()
        renderer.source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        fidelity = verify(source_root, root, budget, spec, packet, trace, requests, frames, mixin)
        owner.save(root/'replay_verification.json', fidelity | dict(wall_s=time.monotonic()-started, script_sha256=owner.sha(__file__),
                                                                    renderer_sha256=owner.sha(renderer.__file__)))
    except BaseException as exc:
        owner.save(root/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        renderer.source.shutdown_genesis()


def child(assignment):
    with (OUT/f'{assignment}.log').open('x') as log:
        code = subprocess.run([sys.executable, '-m', 'scripts.replay_go2_stage2_recording_frames_development', '--run', assignment],
                              stdout=log, stderr=subprocess.STDOUT, env=os.environ | ENVIRONMENT).returncode
    return assignment, code


def main(cohort, workers):
    assert Path.cwd().resolve() == owner.REPO
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((BASE/'dev_cohorts'/cohort/'config.json').read_text())
    todo = []
    for _arm, _set, _maze, _ep, assignment in config['plan']:
        run = BASE/'runs'/assignment
        failure = run/'failure.json'
        contact = failure.exists() and 'DISALLOWED_CONTACT' in json.loads(failure.read_text()).get('reason', '')
        if (run/'planning.json').exists() and not contact and not (OUT/assignment).exists():
            todo.append(assignment)
    print(json.dumps(dict(replays=todo)), flush=True)
    with wall.job(BASE, f'stage-2 recording frame replays ({cohort})'), ProcessPoolExecutor(workers) as pool:
        for assignment, code in pool.map(child, todo):
            print(json.dumps(dict(assignment=assignment, exit=code)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohort')
    p.add_argument('--run')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    replay(a.run) if a.run else main(a.cohort, a.workers)
