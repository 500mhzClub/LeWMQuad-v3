"""V4 mission source collection. Requires separately approved V4 owner.

Audit-only fork of the qualified collector; live source/controller code unchanged.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
from functools import partial
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import time
import traceback

import cv2
import numpy as np
import psutil
import torch

from lewm.decision_headroom_packet_development import DecisionPacketCaptureMixin
from lewm.decision_headroom_snapshot_development import capture
from lewm.decision_headroom_v4_collection_development import MissionSampling
from lewm import decision_headroom_v4_layout_runtime_development as layouts
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_dense_horizon_navigation_development as source


CAPS = Path('docs/go2_decision_headroom_protocol_v4_2026-09-23.json')
ASSIGNMENTS = tuple((layout,arm) for layout in range(8) for arm in ('command_history','reactive_feedback','action'))
SOURCE_READOUT = 'maze_view_old_data'


class CapturedDenseRuntime(DecisionPacketCaptureMixin, source.DenseNavigationRuntime):
    pass


class CapturedReactiveRuntime(DecisionPacketCaptureMixin, source.DenseReactiveNavigationRuntime):
    pass


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def binary(path, value, budget):
    payload = pickle.dumps(value, protocol=4)
    budget.admit_write(len(payload))
    with path.open('xb') as stream:
        stream.write(payload)
    return dict(path=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def preserve_stop_trace(session, directory):
    """Use reserved closeout space for physical failure evidence, not images."""
    from scripts.run_go2_decision_headroom_branches_development import TRACE_FIELDS
    rows = session.samples
    arrays = {key:np.stack([row[key] for row in rows]) for key in TRACE_FIELDS} if rows else {}
    with (directory/'emergency_physics_trace.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    save(directory/'emergency_contact_events.json', session.contact_events)
    save(directory/'emergency_retention.json', dict(reason='PILOT_RESOURCE_STOP',
        physical_samples=len(rows), applied_commands_embedded_in_trace=True,
        original_source_command_records='../requests.json',
        emergency_dump_contains_images=False, source_recording_may_be_partial=True,
        incomplete_source_not_admitted_for_audit=True))


def require_stage_a_closed(root):
    stage_a = root.parent / 'go2_maze_view_readout_navigation_v1_attempt_001'
    if not (stage_a / 'result.json').is_file():
        raise RuntimeError('Stage A has not completed; do not launch pilot physics')
    owner = json.loads((stage_a / 'process.json').read_text())
    try:
        process = psutil.Process(owner['pid'])
        if abs(process.create_time() - owner['created']) < .01 and process.status() != psutil.STATUS_ZOMBIE:
            raise RuntimeError('Stage A coordinator is still live')
    except psutil.NoSuchProcess:
        pass
    for layout, arm in ((0, 'old_data'), (0, 'maze_data'), (2, 'maze_data'), (2, 'old_data')):
        if not (stage_a / f'layout{layout:02d}_{arm}_result.json').is_file():
            raise RuntimeError('Stage A assignment or physical reader is missing')


class AuditPhysicalInit(source.ProspectivePhysicalInit):
    __init__=bind(source.ProspectivePhysicalInit.__init__,specification=layouts.specification,pack=layouts.pack)

class AuditSession(source.RGBNavigationRetentionMixin,source.previous.native.NogilDrawingMixin,
        source.previous.native.study.cohort.LiveDepthNoiseMixin,
        source.previous.native.study.cohort.CompactDepthRetentionMixin,
        source.previous.native.study.cohort.LzmaRawDepthPairedCameraSession,AuditPhysicalInit):
    pass

def run(case):
    caps = json.loads(CAPS.read_text())['execution_caps']
    root = Path(caps['output_root'])
    require_stage_a_closed(root)
    admission = json.loads((root / 'pilot_execution_admission.json').read_text())
    if admission['caps_sha256'] != hashlib.sha256(CAPS.read_bytes()).hexdigest():
        raise RuntimeError('Phase 1 admission must bind the recorded caps')
    # The bounded pilot resource monitor is supplied by its single owner.
    # Do not execute this collector as an unmonitored standalone substitute.
    from scripts.run_go2_decision_headroom_pilot_development import PilotBudget
    budget = PilotBudget.attach(root, admission)
    budget.start_source(case)
    layout, arm = ASSIGNMENTS[case]
    output = root / f'source_{case:02d}'
    output.mkdir(exist_ok=False)
    directory = output / 'native'
    directory.mkdir()
    owner = psutil.Process()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(output / 'process.json', dict(pid=owner.pid, created=owner.create_time(), case=case,
        layout=layout, controller=arm, readout=SOURCE_READOUT, source_sha256=source_hash,
        source_collection_only=True, navigation_performance_trial=False))
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(4)
    controller = session = model = None
    requests, acquisitions, published, snapshots = [], [], [], []
    clock = source.UntimedSimulationClock()
    started = time.monotonic()
    error = None
    try:
        model = source.load_dense_navigation_model('action', readout_arm=SOURCE_READOUT)
        source.previous.warmup()
        source.previous.study.cohort.stable.floor.configure()
        spec = layouts.specification(layout)
        mission = layouts.public_mission(layout)
        save(output / 'specification.json', spec)
        save(output / 'model_identity.json', dict(readout=model.readout_identity,
            predictor_sha256=source.fit.digest(source.fit.OUTPUT / 'action_final.pt'),
            controller_class=source.DenseReactiveNavigationRuntime.__name__ if arm == 'reactive_feedback' else source.DenseNavigationRuntime.__name__,
            audit_adapter='DecisionPacketCaptureMixin', source_controller_code_unchanged=True))
        with contextlib.ExitStack() as stack:
            def pool(initializer):
                return stack.enter_context(ProcessPoolExecutor(max_workers=1,
                    mp_context=get_context('spawn'), initializer=initializer))
            registration = pool(source.previous.study.previous.reference.previous.initialize_registration)
            mapping = pool(source.previous.initialize_mapping)
            pose = pool(partial(source.initialize_pose, str(output)))
            obstacles = pool(source.previous.study.previous.reference.previous.initialize_obstacles)
            assert registration.submit(source.previous.native.baseline.registration_ready).result()
            assert mapping.submit(source.mapping_ready).result()
            assert pose.submit(source.pose_ready).result()
            assert obstacles.submit(source.previous.study.cohort.stable.obstacles_ready).result()
            log = stack.enter_context((output / 'worker.log').open('x'))
            stack.enter_context(contextlib.redirect_stdout(log))
            stack.enter_context(contextlib.redirect_stderr(log))
            source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
            budget.check('before_scene')
            session = AuditSession(spec, directory, noise_layout_index=layout % 4, noise_sigma_mm=2)
            session.install_contact_identity()
            save(output / 'actuator_identity.json', source.configure_gains(session.ctx.build.robot,
                session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint'))
            budget.reserve_source_physics(case, 1.5)
            session.settle_recorded()
            source.admit_context_setup(session, source_hash)
            def sink(frame, raw, registered):
                published.append(dict(frame=frame, raw_pose=raw['current_pose'], registered_pose=registered['current_pose']))
            runtime = CapturedReactiveRuntime if arm == 'reactive_feedback' else CapturedDenseRuntime
            controller = runtime(model, goal_initial_xy=mission['goal_initial_body_xy_m'],
                condition='jepa', variant='full', clock_ns=clock, evidence_sink=sink,
                planning_delay_ticks=3, maximum_initial_dispatch_lateness_ns=0,
                prediction_source='command_history' if arm == 'command_history' else 'neural',
                registration_executor=registration, navigation_ticks=4800, arrival_radius_m=.02,
                mapping_executor=mapping, pose_executor=pose, obstacle_executor=obstacles)
            controller.audit_snapshot_frames = tuple(range(12,4789,4))
            sampling=MissionSampling(case,output,budget)
            session.physics_clock_callback = clock.advance
            history = deque(maxlen=4)
            for tick in range(caps['collection_caps']['source_policy_steps_per_rollout']):
                budget.check('source_step')
                sim_ns = int(session.ctx.runner._sim_time_ns)
                clock.advance(sim_ns)
                if tick % 5 == 0:
                    policy, depth, fast, auxiliary_depth, auxiliary_rgb, measured = session.sensor_packets()
                    history.append(policy)
                    packet = source.AcquiredFrame(tick // 5, measured, policy, depth, fast,
                        auxiliary_rgb, auxiliary_depth, tuple(history))
                    controller.submit(packet)
                    source.drain(controller)
                    acquisitions.append(dict(frame=packet.frame, measured_ns=measured))
                    if packet.frame in controller.audit_snapshot_frames:
                        sampling.decision(controller,session,packet.frame)
                    sampling.camera(session)
                request = controller.request(now_ns=clock())
                row = dict(request, simulator_ns=sim_ns, pre_sample_index=len(session.samples)-1)
                requests.append(row)
                session.phase = 2
                budget.reserve_source_physics(case, .02)
                row['applied_command'] = session.command_policy_step(request['requested_command'])
                row['post_sample_index'] = len(session.samples)-1
                if controller.faults:
                    raise RuntimeError(str(controller.faults))
                if controller.mission_terminal is not None:
                    break
            controller.finish()
            snapshots=sampling.finish(arm,layout)
            save(output / 'result.json', dict(status='BOUNDED_AUDIT_SOURCE_COMPLETE',
                frames=len(acquisitions), policy_steps=len(requests), captured_states=len(snapshots),
                wall_s=time.monotonic()-started, navigation_performance_comparison=False,
                source_mission_records_retained_without_comparison=True))
    except BaseException as exc:
        error = exc
        save(output / 'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(),
            frames=len(acquisitions), policy_steps=len(requests), captured_states=len(snapshots)))
    finally:
        if 'sampling' in locals() and not (output/'sampling.json').exists():
            snapshots=sampling.finish(arm,layout)
        if controller is not None:
            controller.stopped.set()
            clock.close()
            for thread in controller.threads:
                thread.join(timeout=2.)
            for name, value in (('planning', controller.planning), ('poses', published),
                    ('pipeline_faults', controller.faults), ('mission', controller.mission_rows)):
                save(output / f'{name}.json', value)
        for name, value in (('requests', requests), ('acquisitions', acquisitions), ('snapshots', snapshots)):
            save(output / f'{name}.json', value)
        if model is not None:
            save(output / 'dense_model_calls.json', model.receipts)
        try:
            if session is not None:
                # Existing recording writes only RGB, physical/body traces and
                # receipts. Admit an upper bound before its parallel writers.
                budget.admit_write(256*1024**2)
                session.persist(directory)
                save(directory/'camera_terminal_identity.json',session._static_identity())
                budget.check('source_persisted',force=True)
                if error is None:
                    from scripts.run_go2_headroom_v4_branches_development import run as run_branches
                    run_branches(session,output,budget,model=model,case=case)
        except BaseException as exc:
            save(output / 'closeout_failure.json', dict(reason=repr(exc), traceback=traceback.format_exc()))
            if budget.stopped and session is not None:
                preserve_stop_trace(session, directory)
            if error is None:
                error = exc
        finally:
            if session is not None:
                session.ctx.build.scene.destroy()
            source.shutdown_genesis()
            budget.finish_source(case, error)
    if error is not None:
        raise error
    print('HEADROOM_SOURCE_COMPLETE', case, len(snapshots), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, choices=range(24), required=True)
    raise RuntimeError('Use the approval-bound V4 owner; no standalone source execution')
