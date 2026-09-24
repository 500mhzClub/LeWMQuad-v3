"""Two hold counterfactuals for the first late turn in the learned-cost pilot.

Replay diagnosis only. Original forecasts and executed turns are retained;
the sole new outcome at each state is five native hold ticks.
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
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import run_go2_dense_metric_goal_pilot_development as live

pilot = live.previous
base = pilot.base
OUTPUT = live.OUTPUT.parent/'go2_dense_metric_late_turn_v1_attempt_001'
PLAN = Path('docs/go2_dense_metric_late_turn_plan_2026-09-17.json')
RESULT = Path('docs/go2_dense_metric_late_turn_result_2026-09-17.json')


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    records = []
    for case in (0, 3):
        directory = live.OUTPUT/f'case_{case:02d}'
        result = json.loads((directory/'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['completed_budget']
        decisions = json.loads((directory/'decisions.json').read_text())
        first_hold = next(d['tick'] for d in decisions if d['action'] == 'hold')
        selected = next(d for d in decisions if d['tick'] > first_hold and d['action'] != 'hold')
        departure = selected['tick']
        assert selected['action'] == 'right_turn'
        prefix = [[0., 0., 0.]]*10 + [c for d in decisions if d['tick'] < departure for c in d['requested_commands']]
        assert len(prefix) == departure
        assert result['camera_goal_errors'][departure]['within_goal']
        records.append(dict(case=case, source=str(directory), departure=departure,
            scene=pilot.CASES[case][0], goal_trial=pilot.CASES[case][1],
            prefix_commands=prefix, original_decision=selected,
            result_sha256=base.digest(directory/'result.json'),
            decisions_sha256=base.digest(directory/'decisions.json')))
    plan = dict(source_sha256=base.digest(__file__), sources=records,
        selection='first non-hold decision after first hold; both departures within original goal tolerances',
        metric_checkpoint_sha256=json.loads(live.PLAN.read_text())['metric_checkpoint_sha256'],
        new_branches=2, actions_compared=['hold', 'recorded right_turn'], suffix_ticks=5,
        no_training=True, post_hoc_diagnostic=True, new_navigation=False,
        resources=dict(output_free_bytes=shutil.disk_usage(OUTPUT.parent).free,
            available_ram_kib=next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))),
        execution='two CPU-native workers with four disjoint cores each; GPU scoring afterward',
        storage='full short native recordings; minimum free reserve 512 MiB',
        limitations=['two exposed post-hoc states', 'only hold versus executed turn; other actions not ranked'])
    OUTPUT.mkdir(); base.save(PLAN, plan); base.save(OUTPUT/'plan.json', plan)
    print('LATE_TURN_PREPARED', [(r['case'], r['departure']) for r in records], flush=True)


def branch(index):
    from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
    plan = json.loads(PLAN.read_text()); assert base.digest(__file__) == plan['source_sha256']
    source = plan['sources'][index]; original = Path(source['source']); departure = source['departure']
    assert base.digest(original/'decisions.json') == source['decisions_sha256']
    directory = OUTPUT/f'branch_{index:02d}'; directory.mkdir(exist_ok=False)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(4)
    torch.manual_seed(2026091706)
    base.save(directory/'launch.json', dict(pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        plan_sha256=base.digest(PLAN), case=source['case'], departure=departure))
    session = None; stop = None; matched = False; started = time.monotonic()
    try:
        spec = pilot.specification(source['scene']); base.save(directory/'specification.json', spec)
        pilot.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        session = pilot.GeometryProgressFamilySession(spec, directory); session.install_contact_identity()
        gains = pilot.configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
                                     session.ctx.policy.env_cfg, 'checkpoint')
        base.save(directory/'actuator_identity.json', gains)
        try:
            session.settle_recorded(); session.capture_current(); pilot.admit_context_setup(session, base.digest(PLAN))
            tape = source['prefix_commands'] + [[0., 0., 0.]]*5
            for tick in range(departure+6):
                session.sensor_packets()
                if tick == departure:
                    with np.load(original/'physics_trace.npz', allow_pickle=False) as archive:
                        for key in ('base_pose_world', 'joint_position', 'joint_velocity', 'applied_command'):
                            np.testing.assert_array_equal(np.stack([s[key] for s in session.samples]), archive[key][:len(session.samples)])
                    for frame in (departure-10, departure-5, departure):
                        assert (directory/f'rgb_{frame:04d}.png').read_bytes() == (original/f'rgb_{frame:04d}.png').read_bytes()
                    matched = True
                if tick == departure+5:
                    break
                if shutil.disk_usage(OUTPUT).free < 512*1024**2:
                    raise RuntimeError('output storage reserve reached')
                session.phase = 1 if tick < 10 else 2
                session.command_tick(tape[tick])
        except PhysicalStop as error:
            stop = str(error); session.capture_current()
        assert matched, 'original departure was not reproduced'
        assert pilot.read_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist()) == gains['effective']
        result = dict(status='COMPLETE', case=source['case'], departure=departure,
            prefix_native_and_context_rgb_exact=matched, physical_stop=stop,
            complete_500ms=stop is None and len(session.model_manifest) == departure+6,
            disallowed_contact=bool(any(s['physics_contact'] for s in session.samples)),
            endpoint_pose_evaluator_only=session.samples[-1]['base_pose_world'].tolist(),
            wall_s=time.monotonic()-started)
    except Exception as error:
        base.save(directory/'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if session is not None:
            try:
                session.persist(directory); session.persist_observations(directory)
                base.save(directory/'native_guard_rows.json', session.guard_rows)
            finally:
                session.ctx.build.scene.destroy()
        pilot.shutdown_genesis()
    base.save(directory/'result.json', result)
    print('LATE_TURN_BRANCH_COMPLETE', json.dumps(result), flush=True)


@torch.inference_mode()
def score():
    assert not RESULT.exists()
    plan = json.loads(PLAN.read_text()); torch.set_num_threads(4)
    assert base.digest(live.fitted.OUTPUT/'metric.pt') == plan['metric_checkpoint_sha256']
    model = live.fitted.load().cuda()
    encoder = base.encoders.VJepa21Arm(); encoder.build(torch.device('cuda:0'), torch.float32)
    def encode(path):
        return F.layer_norm(encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float(), (1024,))
    groups = []
    for index, source in enumerate(plan['sources']):
        directory = OUTPUT/f'branch_{index:02d}'; original = Path(source['source']); frame = source['departure']+5
        branch_result = json.loads((directory/'result.json').read_text())
        assert branch_result['status'] == 'COMPLETE' and branch_result['complete_500ms'] and not branch_result['disallowed_contact']
        assert base.digest(original/'result.json') == source['result_sha256']
        old = json.loads((original/'result.json').read_text())
        goal_pose = np.asarray(old['goal_pose_evaluator_only']); goal_rotation = pilot.rotation_xyzw(goal_pose[3:])
        goal = encode(pilot.GOAL_ROOT/source['goal_trial']/'rgb_0023.png')
        goal_embedding = model.embed(pool_tokens(goal))
        cameras = json.loads((original/'camera_audit.json').read_text())
        with np.load(original/'physics_trace.npz', allow_pickle=False) as a:
            executed_pose = a['base_pose_world'][cameras[frame]['physical_sample_index']]
        rows = []
        for action, root, pose, action_index in (
                ('hold', directory, np.asarray(branch_result['endpoint_pose_evaluator_only']), 0),
                ('right_turn', original, executed_pose, source['original_decision']['action_index'])):
            feature = encode(root/f'rgb_{frame:04d}.png'); embedding = model.embed(pool_tokens(feature))
            rotation = pilot.rotation_xyzw(pose[3:]); relative = goal_rotation.T @ rotation
            xy = float(np.linalg.norm(pose[:2]-goal_pose[:2])); yaw = float(abs(np.arctan2(relative[1,0], relative[0,0])))
            world_yaw_delta = np.arctan2(rotation[1,0],rotation[0,0])-np.arctan2(goal_rotation[1,0],goal_rotation[0,0])
            world_yaw_delta = np.arctan2(np.sin(world_yaw_delta), np.cos(world_yaw_delta))
            rows.append(dict(action=action, predicted_cost=source['original_decision']['costs'][action_index],
                actual_image_cost=float((embedding-goal_embedding).square().mean()),
                actual_raw_mse=float((feature-goal).square().mean()), xy_error_cm=100*xy,
                yaw_error_deg=float(np.rad2deg(yaw)), within_goal=bool(xy <= .03 and yaw <= np.deg2rad(5)),
                physical_cost=float((xy/.03)**2+(world_yaw_delta/np.deg2rad(5))**2)))
        groups.append(dict(case=source['case'], departure=source['departure'], rows=rows,
            predicted_choice=min(rows,key=lambda r:r['predicted_cost'])['action'],
            actual_image_choice=min(rows,key=lambda r:r['actual_image_cost'])['action'],
            physical_choice=min(rows,key=lambda r:r['physical_cost'])['action']))
    report = dict(status='COMPLETE', groups=groups, plan_sha256=base.digest(PLAN),
        prefix_native_and_context_rgb_exact=True, new_native_branches=2,
        model_changed=False, navigation_tested=False, post_hoc_diagnostic=True,
        scope='hold versus executed turn only; not exhaustive optimal action')
    base.save(OUTPUT/'result.json',report); base.save(RESULT,report)
    print('LATE_TURN_DIAGNOSIS_COMPLETE',json.dumps(groups),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true'); group.add_argument('--branch',type=int,choices=(0,1))
    group.add_argument('--score',action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    elif args.score: score()
    else: branch(args.branch)
