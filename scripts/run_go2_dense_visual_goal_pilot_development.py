"""Prospective native image-goal pilot with fixed model and action-blind control.

Each action is selected from newly acquired RGB and applied-command history.
Recorded goal poses are evaluator-only and loaded after control terminates.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import cv2
import numpy as np
import torch

from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.dense_visual_goal_control_development import DenseVisualGoalControl
from lewm.geometry_progress_layout_family_development import specification
from lewm.physical_execution_development import rotation_xyzw
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.geometry_progress_family_session_development import GeometryProgressFamilySession
from scripts.pulse_context_setup_development import admit_context_setup
from scripts import evaluate_go2_frozen_vjepa_native_branches_development as base
from scripts import train_go2_frozen_vjepa_native_adaptation_development as fit

OUTPUT = fit.OUTPUT.parent/'go2_dense_visual_goal_pilot_v1_attempt_002'
PLAN = Path('docs/go2_dense_visual_goal_pilot_attempt_002_plan_2026-09-17.json')
GOAL_ROOT = base.data.ROOTS['family']
# Existing exposed transfer geometry, same appearance; goal is the opening-side
# arc image at frame 23 (two seconds after the old three-tick quiet prefix).
CASES = (('family_episode_026', 'family_episode_010', 'action'),
         ('family_episode_026', 'family_episode_010', 'no_future_action'),
         ('family_episode_003', 'family_episode_089', 'no_future_action'),
         ('family_episode_003', 'family_episode_089', 'action'))
SOURCE_FILES = ('lewm/dense_visual_goal_control_development.py', __file__)


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    terminal = json.loads((fit.OUTPUT/'result.json').read_text())
    assert terminal['status'] == 'COMPLETE'
    goals = {}
    for scene, goal, _ in CASES:
        live, recorded = specification(scene), specification(goal)
        assert live['geometry'] == recorded['geometry'] and live['appearance_seed'] == recorded['appearance_seed']
        assert live['data_role'] == recorded['data_role'] == 'geometry_transfer'
        image = GOAL_ROOT/goal/'rgb_0023.png'
        goals[goal] = dict(path=str(image), sha256=base.digest(image), frame=23)
    plan = dict(cases=CASES, goals=goals, source_sha256={p:base.digest(p) for p in SOURCE_FILES},
        predecessor='go2_dense_visual_goal_pilot_v1_attempt_001',
        correction='preserve canonical requested-command precision through strict session bounds; no model, goal or budget change',
        predictor_fit_sha256=base.digest(fit.OUTPUT/'result.json'), checkpoints=terminal['checkpoint_sha256'],
        warmup_ticks=10, decisions=20, commit_ticks=5, forecast_horizon_ms=500, final_drain_ticks=5,
        commands='six unchanged primitive starts, each sustained for five 100-ms ticks; actual limiter trajectory',
        cost='normalized dense predicted-feature MSE to supplied goal image', seed=2026091706,
        evaluation=dict(position_tolerance_m=.03, heading_tolerance_deg=5., sustained_camera_frames=3,
            goal_pose_available_to_controller=False, no_success_based_early_stop=True),
        precision='float32 encoder and predictor; CPU Genesis physics and gait',
        sensors_used_for_control=['RGB', 'past applied commands'], additional_sensors_recorded_not_used=True,
        physics_paused_during_compute=True, new_native_execution=True,
        resources=dict(available_ram_gib=72, output_free_gib=4.0, root_free_gib=1.2,
            gpu='Radeon AI PRO R9700', gpu_vram_gib=31.86, competing_compute_jobs=0),
        execution='up to two independent processes on disjoint four-core CPU groups; shared GPU; separate outputs',
        resource_limits='stop before next block if output-volume free space drops below 512 MiB',
        source_scope='development source only; no sealed material or historical experiment resumption',
        limitations=['two previously exposed transfer geometries, excluded from adaptation fit',
            'supplied image goal, local reaching only; no autonomous maze exploration or memory integration',
            'action-blind tie-randomization control is not a strong reactive visual-servo baseline',
            'simulation paused during inference, no real-time or hardware claim'])
    OUTPUT.mkdir()
    base.save(PLAN, plan)
    base.save(OUTPUT/'plan.json', plan)
    print('PREPARED', json.dumps(dict(cases=CASES, output=str(OUTPUT))), flush=True)


def evaluate(session, goal_directory, decisions, stop):
    # Called only after the last command. Never gives native goal/robot poses to
    # the controller, and never shortens a trial based on oracle success.
    cameras = json.loads((goal_directory/'camera_audit.json').read_text())
    with np.load(goal_directory/'physics_trace.npz', allow_pickle=False) as archive:
        goal = archive['base_pose_world'][cameras[23]['physical_sample_index']]
    goal_rotation = rotation_xyzw(goal[3:])
    rows = []
    for camera in session.image_audit:
        sample = session.samples[camera['physical_sample_index']]
        pose = sample['base_pose_world']
        relative = goal_rotation.T @ rotation_xyzw(pose[3:])
        xy = float(np.linalg.norm(pose[:2]-goal[:2]))
        yaw = float(abs(np.arctan2(relative[1, 0], relative[0, 0]))*180/np.pi)
        rows.append(dict(time_s=sample['timestamp_s'], xy_error_m=xy, yaw_error_deg=yaw,
                         within_goal=xy <= .03 and yaw <= 5.))
    active = rows[10:]
    sustained = any(all(v['within_goal'] for v in active[i:i+3]) for i in range(len(active)-2))
    contact = any(s['physics_contact'] for s in session.samples)
    return dict(status='COMPLETE', physical_stop=stop, planned_decisions=20, decisions=len(decisions),
        completed_budget=stop is None and len(decisions)==20 and len(rows)==116,
        goal_reached=bool(sustained and not contact and stop is None),
        disallowed_contact=bool(contact), sustained_goal_frames=3,
        initial_xy_error_m=rows[0]['xy_error_m'], initial_yaw_error_deg=rows[0]['yaw_error_deg'],
        final_xy_error_m=rows[-1]['xy_error_m'], final_yaw_error_deg=rows[-1]['yaw_error_deg'],
        minimum_xy_error_m=min(v['xy_error_m'] for v in active) if active else None,
        final_within_goal=rows[-1]['within_goal'], camera_goal_errors=rows,
        goal_pose_evaluator_only=goal.tolist(), continuous_native_execution=True,
        observations_reacquired_after_actions=True, full_maze_navigation=False,
        real_time_qualified=False, hardware_validated=False)


def run(case):
    from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
    plan = json.loads(PLAN.read_text())
    assert plan['cases'] == [list(c) for c in CASES]
    assert plan['source_sha256'] == {p:base.digest(p) for p in SOURCE_FILES}
    assert plan['predictor_fit_sha256'] == base.digest(fit.OUTPUT/'result.json')
    directory = OUTPUT/f'case_{case:02d}'
    directory.mkdir(exist_ok=False)
    scene_trial, goal_trial, arm = CASES[case]
    goal_image = GOAL_ROOT/goal_trial/'rgb_0023.png'
    assert base.digest(goal_image) == plan['goals'][goal_trial]['sha256']
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(4)
    torch.manual_seed(2026091706)
    started = time.monotonic()
    base.save(directory/'launch.json', dict(pid=os.getpid(), case=case, arm=arm,
        cpu_affinity=sorted(os.sched_getaffinity(0)), plan_sha256=base.digest(PLAN),
        free_output_bytes=shutil.disk_usage(OUTPUT).free))
    session = None
    decisions, observed = [], []
    stop = None
    try:
        controller = DenseVisualGoalControl(arm, goal_image)
        spec = specification(scene_trial)
        base.save(directory/'specification.json', spec)
        initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        session = GeometryProgressFamilySession(spec, directory)
        session.install_contact_identity()
        gains = configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
                                session.ctx.policy.env_cfg, 'checkpoint')
        base.save(directory/'actuator_identity.json', gains)
        try:
            session.settle_recorded(); session.capture_current()
            admit_context_setup(session, base.digest(PLAN))
            for tick in range(111):
                packet, _, _, now = session.sensor_packets()
                if tick % 5 == 0:
                    record = controller.observe(packet)
                    observed.append(record)
                    base.save(directory/f'observation_{tick:03d}.json', record)
                if tick == 110:
                    break
                if tick < 10:
                    requested = [0., 0., 0.]
                    session.phase = 1
                else:
                    if tick % 5 == 0:
                        if shutil.disk_usage(OUTPUT).free < 512*1024**2:
                            raise RuntimeError('output storage reserve reached')
                        selected = controller.choose(packet)
                        selected['tick'] = tick
                        decisions.append(selected)
                        base.save(directory/f'decision_{len(decisions):02d}.json', selected)
                        print('LIVE_VISUAL_DECISION', case, len(decisions), selected['action'],
                              'goal_mse', record['current_goal_mse'], flush=True)
                    requested = selected['requested_commands'][tick % 5]
                    session.phase = 2
                session.command_tick(requested)
            session.phase = 3
            for _ in range(5):
                session.command_tick([0., 0., 0.])
            session.capture_current()
        except PhysicalStop as error:
            stop = str(error)
            session.capture_current()
        terminal_gains = read_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist())
        assert terminal_gains == gains['effective']
        result = evaluate(session, GOAL_ROOT/goal_trial, decisions, stop)
        result.update(case=case, arm=arm, scene=scene_trial, goal_image_trial=goal_trial,
            wall_s=time.monotonic()-started, observed_forecast_windows=sum('previous_selected_forecast_mse' in r for r in observed),
            mean_planning_wall_s=float(np.mean([d['planning_wall_s'] for d in decisions])) if decisions else None,
            mean_encoding_wall_s=float(np.mean([r['observation_encoding_wall_s'] for r in observed])),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        base.save(directory/'result.json', result)
        print('DENSE_VISUAL_GOAL_PILOT_COMPLETE', json.dumps({k:v for k,v in result.items() if k!='camera_goal_errors'}), flush=True)
    except Exception as error:
        base.save(directory/'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        base.save(directory/'decisions.json', decisions)
        base.save(directory/'observed_costs.json', observed)
        if session is not None:
            try:
                session.persist(directory); session.persist_observations(directory)
                base.save(directory/'native_guard_rows.json', session.guard_rows)
            finally:
                session.ctx.build.scene.destroy()
        shutdown_genesis()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--case', type=int, choices=range(4))
    args = parser.parse_args()
    if args.prepare and args.case is None:
        prepare()
    elif not args.prepare and args.case is not None:
        run(args.case)
    else:
        parser.error('prepare or execute one fresh case')
