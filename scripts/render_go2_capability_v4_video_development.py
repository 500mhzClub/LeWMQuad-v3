"""Capability video from a verified deterministic replay on the frozen V4 harness.

Pass 1 re-simulates the episode from its seed with the unchanged V4 session and
controller runtime. Every consumed packet record, decision, dispatch
command/reason, applied command, native trace value, published pose and mission
row must match the log; the consumed primary RGB is kept as the egocentric panel.
C1/C2 use their actual non-neural model; C0/C3/C4 use the recorded
prediction-slot outputs with identical candidate tapes asserted.
Pass 2 renders the chase camera in a separate commands-only session.
No simulator or controller code is changed. Intermediates stay on RecoveryStorage.
"""
import argparse
from bisect import bisect_right
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
from functools import partial
import json
import math
from multiprocessing import get_context
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

from lewm import decision_headroom_json_v42_development as output
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from lewm.navigation_capability_paired_floor_start_development import initialize_mapping as initialize_startup_mapping
from lewm.navigation_capability_target_reference_development import install_task_cues, settled_task_cues
from lewm.navigation_capability_unused_workload_development import UnusedNeuralWorkload
from lewm.physical_execution_development import rotation_xyzw
from lewm_genesis.lewm_contract import SafetyLimits
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.render_go2_navigation_capability_pipeline_development import RecordedPrediction, compare_poses

source = owner.source
NAMES = dict(C0='Oracle (diagnostic)', C1='Command history', C2='Reactive', C3='JEPA', C4='Supervised predictor')
ORDINARY = 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'


def jsonable(value):
    return json.loads(output.dumps(value))


def start_session(spec, directory, full_frames):
    session = owner.make_session(spec, directory, full_frames=full_frames)
    session.install_contact_identity()
    source.configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
    session.settle_recorded()
    source.admit_context_setup(session, owner.sha(owner.__file__))
    return session


def verify(source_root, root, budget, spec, packet, trace, requests, frames):
    config = json.loads((source_root/'config.json').read_text())
    arm = config['controller']
    receipts = json.loads((source_root/'model_calls.json').read_text())
    limits = SafetyLimits.from_manifest(yaml.safe_load((owner.REPO/'config/go2_platform_manifest.yaml').read_text()))
    model = UnusedNeuralWorkload(arm, limits) if arm in ('C1', 'C2') else RecordedPrediction(receipts, arm)
    expected_calls = {r['observed_ns']: r for r in receipts}
    expected_plans = {r['frame']: r for r in json.loads((source_root/'planning.json').read_text()) if 'selection' in r}
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
            mapping = pool(initialize_startup_mapping)
            pose = pool(partial(source.initialize_pose, str(root)))
            obstacles = pool(source.previous.study.previous.reference.previous.initialize_obstacles)
            assert registration.submit(source.previous.native.baseline.registration_ready).result()
            assert mapping.submit(source.mapping_ready).result()
            assert pose.submit(source.pose_ready).result()
            assert obstacles.submit(source.previous.study.cohort.stable.obstacles_ready).result()
            session = start_session(spec, directory, full_frames=True)
            for key, values in trace.items():
                np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]), values[:len(session.samples)])
            start = np.asarray(session.samples[-1]['base_pose_world'], dtype=float)
            cues = settled_task_cues(packet, start[:3], rotation_xyzw(start[3:]))
            if jsonable(cues) != json.loads((source_root/'public_task_cues.json').read_text()):
                raise ValueError('replayed settled task cues differ')

            def sink(frame, raw, registered):
                published.append(dict(frame=frame, raw_pose=raw['current_pose'], registered_pose=registered['current_pose'],
                    visual_support=registered.get('visual_support')))
            base_runtime = source.DenseReactiveNavigationRuntime if arm == 'C2' else source.DenseNavigationRuntime
            runtime = type('StartupRecoveredRuntime', (CompletedSupportRuntimeMixin, base_runtime), {})
            controller = runtime(model, goal_initial_xy=cues['goal_initial_body_xy_m'], condition='jepa', variant='full',
                clock_ns=clock, evidence_sink=sink, planning_delay_ticks=3, maximum_initial_dispatch_lateness_ns=0,
                prediction_source='command_history' if arm == 'C1' else 'neural', registration_executor=registration,
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
                        if arm in ('C1', 'C2'):
                            np.testing.assert_array_equal(model.receipts[-1]['requested_commands'],
                                expected_calls[acquired.measured_ns]['requested_commands'])
                        if arm == 'C1':
                            np.testing.assert_array_equal(actual_plan['motion_correction']['applied_prediction_after_yaw_ablation'],
                                expected_plan['motion_correction']['applied_prediction_after_yaw_ablation'])
                        counts['decisions'] += 1
                request = controller.request(now_ns=clock())
                if request['requested_command'] != expected['requested_command'] or request['reason'] != expected['reason']:
                    raise ValueError(f'dispatch command/reason differs at tick {tick}')
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
            if len(session.samples) != len(trace['timestamp_s']):
                raise ValueError('replayed native trace length differs')
            if counts['decisions'] != len(expected_plans) or counts['frames'] != len(frames):
                raise ValueError('decision or frame count differs')
            controller.finish()
            if jsonable(published) != json.loads((source_root/'poses.json').read_text()):
                raise ValueError('published poses differ')
            if jsonable(controller.mission_rows) != json.loads((source_root/'mission.json').read_text()):
                raise ValueError('mission rows differ')
            return dict(passed=True, controller=arm, bitwise_consumed_packet_records=counts['frames'],
                identical_selected_actions=counts['decisions'], identical_dispatch_command_reason_steps=counts['dispatch'],
                identical_published_poses=len(published), identical_mission_rows=len(controller.mission_rows),
                maximum_position_error_m=maximum[0], maximum_yaw_error_degrees=maximum[1], exact_native_trace_values=True,
                prediction_slot=('Actual non-neural model recomputed' if arm in ('C1', 'C2')
                    else 'Recorded prediction-slot outputs with identical candidate tapes asserted'),
                verification_from_commands_and_seed=True)
    finally:
        if controller is not None:
            controller.stopped.set()
            for thread in controller.threads:
                thread.join(timeout=2.)
        clock.close()
        if session is not None:
            session.ctx.build.scene.destroy()


class Canvas:
    def __init__(self, source_root, root, spec, packet, requests, evaluation, rate, pipeline_test):
        config = json.loads((source_root/'config.json').read_text())
        self.arm = config['controller']
        self.spec, self.packet, self.requests, self.root = spec, packet, requests, root
        plans = [r for r in json.loads((source_root/'planning.json').read_text()) if 'selection' in r]
        self.actions = {r['measured_ns']: r['selection']['action'] for r in plans}
        self.action_times = sorted(self.actions)
        self.mission = {r['frame']: r for r in json.loads((source_root/'mission.json').read_text())}
        self.arrivals = {r['phase']: r for r in evaluation['arrivals']}
        self.rate, self.pipeline_test = rate, pipeline_test
        self.epoch = requests[0]['simulator_ns']
        self.ego_frame, self.ego, self.path = -1, None, []
        self.map_base = self.draw_map()

    def point(self, xy):
        return (int(320+(xy[0]-.65)*58), int(180-(xy[1]-.65)*58))

    def draw_map(self):
        image = np.full((360, 640, 3), 235, np.uint8)
        for wall in self.spec['geometry']['wall_boxes']:
            c = np.array(wall['centre_xyz'][:2])
            h = np.array(wall['size_xyz'][:2])/2
            cv2.rectangle(image, self.point(c-h), self.point(c+h), (65, 65, 65), -1)
        cv2.circle(image, self.point(self.packet['home_se2_world'][:2]), 7, (50, 170, 70), -1)
        cv2.circle(image, self.point(self.packet['beacon_xy_world']), 7, (255, 170, 30), -1)
        for text, colour, row in (('home', (50, 170, 70), 0), ('beacon', (255, 170, 30), 1), ('outbound', (80, 160, 250), 2), ('return', (230, 70, 150), 3)):
            cv2.circle(image, (560, 20+row*22), 6, colour, -1)
            cv2.putText(image, text, (572, 25+row*22), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 40, 40), 1, cv2.LINE_AA)
        return image

    def arrival_text(self, phase, frame):
        row = self.arrivals.get(phase)
        if row is None or frame < row['frame']:
            return 'not yet'
        return 'reached (physically verified)' if row['passed'] else 'observed; rejected by physical check'

    def draw(self, chase, pose, elapsed):
        frame = min(int(elapsed*10+1e-8), max(self.mission))
        if frame != self.ego_frame:
            self.ego = np.asarray(Image.open(self.root/'ego_frames'/f'{frame:04d}.png').convert('RGB'))
            self.ego_frame = frame
        canvas = np.full((1080, 1920, 3), 20, np.uint8)
        canvas[:720, :960] = cv2.resize(self.ego, (960, 720))
        canvas[:720, 960:] = cv2.resize(chase, (960, 720))
        for text, x in (('EGOCENTRIC: RGB consumed by the controller (10 Hz, held)', 12), ('EXOCENTRIC: chase camera, separate replay pass', 972)):
            cv2.rectangle(canvas, (x-6, 8), (x+len(text)*11, 38), (0, 0, 0), -1)
            cv2.putText(canvas, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
        state = self.mission[frame]
        colour = (80, 160, 250) if state['phase'] == 'OUTBOUND' else (230, 70, 150)
        image = self.map_base.copy()
        self.path.append((self.point(pose[:2]), colour))
        for (a, _), (b, c) in zip(self.path, self.path[1:]):
            cv2.line(image, a, b, c, 2)
        Q = rotation_xyzw(pose[3:])
        cv2.circle(image, self.point(pose[:2]), 5, (220, 30, 30), -1)
        cv2.arrowedLine(image, self.point(pose[:2]), self.point(pose[:2]+Q[:2, 0]*.35), (220, 30, 30), 2, tipLength=.35)
        canvas[720:, :640] = image
        index = min(int(elapsed/.02), len(self.requests)-1)
        request = self.requests[index]
        position = bisect_right(self.action_times, request['now_ns'])-1
        action = self.actions[self.action_times[position]] if position >= 0 else 'awaiting first decision'
        substitution = '' if request['reason'] == ORDINARY else f' | dispatch: {request["reason"].lower()}'
        header = 'PIPELINE TEST (development episode) - not a capability video' if self.pipeline_test else 'Capability qualification (validation), not a paper result'
        lines = [f'{self.arm} {NAMES[self.arm]} | harness v4_completed_support (82b7b604)',
            f'Maze {self.packet["maze_id"]:02d}, episode {self.packet["episode_index"]} | mission time {elapsed:6.2f} s',
            f'Phase: {state["phase"]} | action: {action}',
            f'Stall: {"HOLD selected" if action == "hold" else "no"}{substitution}',
            f'Beacon: {self.arrival_text("OUTBOUND", frame)} | home: {self.arrival_text("RETURN", frame)}',
            self.rate, header]
        for row, line in enumerate(lines):
            cv2.putText(canvas, line, (670, 770+row*44), cv2.FONT_HERSHEY_SIMPLEX, .72, (240, 240, 240), 1, cv2.LINE_AA)
        return canvas


def chase_pass(root, budget, spec, trace, requests, canvas, stem):
    directory = root/'chase_native'
    directory.mkdir()
    session = start_session(spec, directory, full_frames=False)
    for key, values in trace.items():
        np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]), values[:len(session.samples)])
    duration = len(requests)*.02
    total = math.ceil(duration*30-1e-8)
    accelerated = duration > 120
    logs = []

    def encoder(path):
        log = (root/f'{path.stem}_ffmpeg.log').open('x')
        logs.append(log)
        return subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '30',
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(path)], stdin=subprocess.PIPE, stderr=log)
    normal = encoder(root/f'{stem}.provisional.mp4')
    fast = encoder(root/f'{stem}_4x.provisional.mp4') if accelerated else None
    sheet_frames = set(np.linspace(0, total-1, 12).astype(int).tolist())
    keyframes = []
    count = 0
    position = None
    maximum = [0., 0.]

    def render(pose):
        nonlocal count, position
        Q = rotation_xyzw(pose[3:])
        desired = pose[:3]-.6*Q[:, 0]+np.array([0., 0., 3.])
        position = desired if position is None else .2*desired+.8*position
        camera = session.ctx.build.camera
        camera.set_pose(pos=position, lookat=pose[:3], up=[0., 0., 1.])
        rendered = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
        rgb = np.asarray(session.ctx.runner._extract_rgb(rendered)).reshape(480, 640, 3)
        frame = canvas.draw(rgb, pose, count/30)
        normal.stdin.write(frame.tobytes())
        if fast is not None and count % 4 == 0:
            labelled = frame.copy()
            cv2.rectangle(labelled, (720, 640), (1200, 700), (0, 0, 160), -1)
            cv2.putText(labelled, '4x SIMULATED TIME', (748, 682), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            fast.stdin.write(labelled.tobytes())
        if count in sheet_frames:
            keyframes.append(cv2.resize(frame, (480, 270)))
        count += 1
    original = session._sample
    epoch = requests[0]['simulator_ns']

    def sampled(requested, applied, stamp):
        value = original(requested, applied, stamp)
        if count < total and stamp-epoch/1e9+1e-9 >= count/30:
            render(value['base_pose_world'])
        return value
    session._sample = sampled
    try:
        render(session.samples[-1]['base_pose_world'])
        for request in requests:
            budget.check()
            session.phase = 2
            applied = session.command_policy_step(request['requested_command'])
            np.testing.assert_array_equal(applied, request['applied_command'])
            end = request['post_sample_index']
            p, y = compare_poses(np.stack([r['base_pose_world'] for r in session.samples[-10:]]), trace['base_pose_world'][end-9:end+1])
            maximum = [max(maximum[0], p), max(maximum[1], y)]
        for process in (normal, fast):
            if process is not None:
                process.stdin.close()
                if process.wait():
                    raise RuntimeError('ffmpeg failed; see retained log')
        if count != total:
            raise ValueError('incorrect output frame count')
        sheet = np.zeros((1080, 1440, 3), np.uint8)
        for i, frame in enumerate(keyframes[:12]):
            sheet[i//3*270:(i//3+1)*270, i % 3*480:(i % 3+1)*480] = frame
        Image.fromarray(sheet).save(root/f'{stem}_contact_sheet.png')
        return dict(frames=count, accelerated_cut=accelerated, maximum_position_error_m=maximum[0], maximum_yaw_error_degrees=maximum[1],
            camera_offset_body_m=[-.6, 0., 3.], camera_smoothing_previous_weight=.8, native_render_time_quantization_ms=2,
            primary_camera_exposed_to_chase=False)
    finally:
        for process in (normal, fast):
            if process is not None and process.poll() is None:
                process.stdin.close()
                process.wait(timeout=30)
        for log in logs:
            log.close()
        session.ctx.build.scene.destroy()


def main(source_root, label, rate, pipeline_test):
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    assert source_root.resolve().is_relative_to((base/'runs').resolve())
    config = json.loads((source_root/'config.json').read_text())
    assert config['harness_sha256'] == owner.sha(owner.FREEZE)
    packet = json.loads((source_root/'episode.json').read_text())
    assert pipeline_test == (packet['role'] != 'validation')
    output.install(base)
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    root = base/'videos'/label
    root.mkdir(parents=True, exist_ok=False)
    budget = owner.Budget(base, protocol)
    budget.admit_persist(3*1024**3)
    spec = json.loads((source_root/'specification.json').read_text())
    requests = json.loads((source_root/'requests.json').read_text())
    frames = json.loads((source_root/'native/in_memory_camera_observations.json').read_text())['frames']
    evaluation = json.loads((source_root/'episode_evaluation.json').read_text())
    with np.load(source_root/'native/physics_trace.npz', allow_pickle=False) as a:
        trace = {k: a[k].copy() for k in a.files}
    stem = f'{config["controller"]}_{"dev" if pipeline_test else "val"}{packet["maze_id"]:02d}_ep{packet["episode_index"]}'
    owner.save(root/'config.json', dict(source=str(source_root), source_config_sha256=owner.sha(source_root/'config.json'),
        episode_sha256=owner.sha(source_root/'episode.json'), harness_sha256=owner.sha(owner.FREEZE), renderer_sha256=owner.sha(__file__),
        pipeline_test=pipeline_test, success_rate_label=rate, simulator_changed=False, controller_changed=False))
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    started = time.monotonic()
    try:
        source.previous.warmup()
        source.previous.study.cohort.stable.floor.configure()
        source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        fidelity = verify(source_root, root, budget, spec, packet, trace, requests, frames)
        owner.save(root/'replay_verification.json', fidelity)
        source.shutdown_genesis()
        source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        canvas = Canvas(source_root, root, spec, packet, requests, evaluation, rate, pipeline_test)
        chase = chase_pass(root, budget, spec, trace, requests, canvas, stem)
        outputs = {}
        for suffix in ('', '_4x'):
            provisional = root/f'{stem}{suffix}.provisional.mp4'
            if provisional.exists():
                provisional.rename(root/f'{stem}{suffix}.mp4')
                outputs[f'{stem}{suffix}.mp4'] = owner.sha(root/f'{stem}{suffix}.mp4')
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames',
            '-of', 'json', str(root/f'{stem}.mp4')], capture_output=True, text=True, check=True)
        owner.save(root/'metadata.json', dict(status='PIPELINE_TEST_REPLAY_VERIFIED' if pipeline_test else 'CAPABILITY_VIDEO_REPLAY_VERIFIED',
            controller=config['controller'], episode=packet['episode_id'], role=packet['role'], source=str(source_root),
            source_config_sha256=owner.sha(source_root/'config.json'), episode_sha256=owner.sha(source_root/'episode.json'),
            episode_evaluation_sha256=owner.sha(source_root/'episode_evaluation.json'), model_calls_sha256=owner.sha(source_root/'model_calls.json'),
            harness='v4_completed_support', harness_sha256=owner.sha(owner.FREEZE), model_bindings=protocol['controllers'].get(config['controller']),
            replay_verification=fidelity, replay_verification_sha256=owner.sha(root/'replay_verification.json'), chase=chase,
            round_trip_success=evaluation['round_trip_success'], success_rate_label=rate, outputs=outputs,
            contact_sheet_sha256=owner.sha(root/f'{stem}_contact_sheet.png'), ffprobe=json.loads(probe.stdout),
            renderer_sha256=owner.sha(__file__), wall_s=time.monotonic()-started, training_render_provenance='unverified'))
    except BaseException as exc:
        owner.save(root/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(), publish=False, automatic_retry=False))
        raise
    finally:
        source.shutdown_genesis()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--rate', required=True, help='Validation success-rate line shown in the HUD')
    p.add_argument('--pipeline-test', action='store_true')
    args = p.parse_args()
    main(args.source_root, args.label, args.rate, args.pipeline_test)
