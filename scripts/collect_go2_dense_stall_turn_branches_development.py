"""Three physical counterfactuals at one exposed stalled context; no fitting.

Replay the original 20-ms commands from the original initialization, then apply
the saved hold/left/right candidate tapes. This is a diagnostic, not navigation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import cv2
import numpy as np
import psutil
import torch

from lewm.actuator_gain_development import configure_gains
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts import run_go2_dense_horizon_navigation_development as native
from scripts.pulse_context_setup_development import admit_context_setup

BASE = native.fit.OUTPUT.parent
REFERENCE = BASE/'go2_dense_horizon_untimed_exposed_maze_full_v1_attempt_001'
BASELINE = BASE/'go2_dense_horizon_untimed_reactive_feedback_exposed_maze_full_v1_attempt_001'
OUTPUT = BASE/'go2_dense_stall_turn_branches_v1_attempt_001'
ACTIONS = ('hold', 'left_turn', 'right_turn')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare():
    report = read(REFERENCE/'dense_navigation_readout.json')
    start = report['decisions']['longest_consecutive_zero_request']['started_ns']
    plans = [p for p in read(REFERENCE/'planning.json') if 'selection' in p]
    plan = next(p for p in plans if p['measured_ns'] >= start+1_000_000_000)
    call = next(c for c in read(REFERENCE/'dense_model_calls.json') if c['observed_ns'] == plan['measured_ns'])
    assert plan['action']=='hold' and plan['selection']['scan_heading_error_rad'] is not None
    requests = [r for r in read(REFERENCE/'requests.json') if r['simulator_ns'] < plan['measured_ns']]
    assert len(requests)==plan['frame']*5
    assert all(not any(r['requested_command']) for r in requests[-50:])
    indices = [next(i for i,c in enumerate(plan['selection']['candidates']) if c['action']==a) for a in ACTIONS]
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT/'plan.json', dict(reference=str(REFERENCE), frame=plan['frame'],
        observed_ns=plan['measured_ns'], actions=ACTIONS, source_sha256=digest(__file__),
        selection='First saved planning frame with a full second of zero requested commands in the final sustained stall; post hoc exposed diagnostic.',
        reference_sha256={n:digest(REFERENCE/n) for n in ('launch.json','requests.json','planning.json','dense_model_calls.json','native/in_memory_camera_observations.json')},
        branches={a:dict(requested=call['requested_commands'][i], applied=call['applied_commands'][i],
            saved_predicted_motion=call['motion_xy_yaw'][i]) for a,i in zip(ACTIONS,indices,strict=True)},
        scan_heading_error_rad=plan['selection']['scan_heading_error_rad'],
        horizons_ms=[300,500,700,800], prefix_camera_pixel_equality_required=True,
        prefix_pose_maximum_absolute_error=1e-7, training=False, closed_loop_navigation=False,
        retention='RGB, commands, native physics and depth hashes; no depth arrays',
        no_candidate_model_or_controller_changes=True, no_fresh_maze_used=True))
    print('STALL_BRANCHES_PREPARED',plan['frame'],ACTIONS,flush=True)


def run(action):
    plan = read(OUTPUT/'plan.json')
    assert digest(__file__)==plan['source_sha256']
    assert all(digest(REFERENCE/n)==sha for n,sha in plan['reference_sha256'].items())
    # Do not compete with the fixed ongoing navigation assignment.
    owner = read(BASELINE/'launch.json')['owner']
    if psutil.pid_exists(owner['pid']):
        process = psutil.Process(owner['pid'])
        if process.create_time()==owner['created'] and process.is_running():
            raise RuntimeError('wait for the reactive navigation owner to exit')
    if shutil.disk_usage(OUTPUT).free < 512*1024**2:
        raise RuntimeError('insufficient room for bounded diagnostic recording')
    directory = OUTPUT/action
    directory.mkdir(exist_ok=False)
    save(directory/'launch.json',dict(pid=os.getpid(),action=action,plan_sha256=digest(OUTPUT/'plan.json')))
    requests = read(REFERENCE/'requests.json')[:plan['frame']*5]
    metadata = {r['frame']:r for r in read(REFERENCE/'native/in_memory_camera_observations.json')['frames']}
    with np.load(REFERENCE/'native/physics_trace.npz',allow_pickle=False) as archive:
        poses = archive['base_pose_world'].copy()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(4)
    native.previous.study.cohort.stable.floor.configure()
    spec = native.replication.layouts.specification(0)
    session = None; started = time.monotonic(); checks = []; executed = []
    try:
        initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        session = native.RGBOnlyCameraSession(spec,directory,noise_layout_index=0,noise_sigma_mm=2)
        session.install_contact_identity()
        gains = configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        save(directory/'actuator_identity.json',gains)
        assert gains['effective']==read(REFERENCE/'actuator_identity.json')['effective']
        session.settle_recorded(); admit_context_setup(session,plan['source_sha256'])
        final_tick = (plan['frame']+8)*5
        for tick in range(final_tick+1):
            frame = tick//5
            if tick%5==0:
                session.sensor_packets()
                if frame <= plan['frame']:
                    expected = metadata[frame]
                    actual_pose = np.asarray(session.samples[-1]['base_pose_world'])
                    error = float(np.max(np.abs(actual_pose-poses[expected['physical_sample_index']])))
                    assert error <= plan['prefix_pose_maximum_absolute_error'], (frame,error)
                    if frame in (plan['frame']-10,plan['frame']-5,plan['frame']):
                        rgb = session.captured_pairs[-1]['images'][0][0]
                        equal = hashlib.sha256(rgb.tobytes()).hexdigest()==expected['pixel_sha256']['primary']['rgb_sha256']
                        checks.append(dict(frame=frame,pose_max_abs_error=error,primary_rgb_exact=equal))
                        assert equal, ('camera context differs',frame)
                if frame%100==0:
                    print('STALL_BRANCH_PROGRESS',action,frame,flush=True)
            if tick==final_tick:
                break
            before = tick < plan['frame']*5
            offset = (tick-plan['frame']*5)//5
            requested = requests[tick]['requested_command'] if before else plan['branches'][action]['requested'][offset]
            session.phase = 2
            applied = session.command_policy_step(requested)
            expected = requests[tick]['applied_command'] if before else plan['branches'][action]['applied'][offset]
            np.testing.assert_allclose(applied,expected,rtol=0,atol=1e-7)
            executed.append(dict(tick=tick,requested=requested,applied=applied))
        save(directory/'result.json',dict(status='COMPLETE',action=action,context_checks=checks,
            camera_frames=len(session.captured_pairs),disallowed_contact=any(s['physics_contact'] for s in session.samples),
            replayed_prefix_steps=len(requests),branch_steps=40,wall_s=time.monotonic()-started,
            fixed_tape_diagnostic=True,closed_loop_navigation=False))
    except BaseException as error:
        save(directory/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc(),context_checks=checks))
        raise
    finally:
        save(directory/'executed_commands.json',executed)
        if session is not None:
            try:
                session.persist(directory); session.persist_observations(directory)
            finally:
                session.ctx.build.scene.destroy()
        shutdown_genesis()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--action',choices=ACTIONS)
    args = parser.parse_args()
    prepare() if args.prepare else run(args.action)
