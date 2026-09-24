"""Matched native alternatives at the two live near-goal failures.

This replays a known causal prefix for diagnosis; it is not a navigation trial.
The predictor's costs were saved before any counterfactual was executed.
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

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from scripts import run_go2_dense_visual_goal_pilot_development as pilot

base = pilot.base
OUTPUT = pilot.OUTPUT.parent/'go2_dense_goal_overshoot_branches_v1_attempt_001'
PLAN = Path('docs/go2_dense_goal_overshoot_plan_2026-09-17.json')
RESULT = Path('docs/go2_dense_goal_overshoot_result_2026-09-17.json')
CASES = (0, 3)
DEPARTURE = 35


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    records = []
    for case in CASES:
        directory = pilot.OUTPUT/f'case_{case:02d}'
        result = json.loads((directory/'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['completed_budget'] and result['arm'] == 'action'
        decisions = json.loads((directory/'decisions.json').read_text())
        selected = next(d for d in decisions if d['tick'] == DEPARTURE)
        assert selected['action'] == 'forward'
        prefix = [[0., 0., 0.]]*10 + [c for d in decisions if d['tick'] < DEPARTURE for c in d['requested_commands']]
        assert len(prefix) == DEPARTURE
        records.append(dict(case=case, source=str(directory), scene=pilot.CASES[case][0],
            goal_trial=pilot.CASES[case][1], prefix_commands=prefix, original_decision=selected,
            original_result_sha256=base.digest(directory/'result.json'),
            original_decisions_sha256=base.digest(directory/'decisions.json')))
    plan = dict(source_sha256=base.digest(__file__), sources=records, departure_frame=DEPARTURE,
        actions=ACTIONS, suffix_ticks=5, branch_count=12, post_hoc_failure_diagnostic=True,
        no_training=True, no_new_navigation_claim=True,
        metrics=['original predicted goal cost', 'actual normalized future-feature goal cost',
            'actual XY and heading goal errors', 'native contact/stop'],
        matching='exact prefix native poses/joints/commands and context RGB; forward successor reproduces original run',
        resources=dict(available_ram_gib=72, output_free_gib=3.4, root_free_gib=1.2, competing_compute_jobs=0),
        execution='four CPU-native workers, four distinct physical cores each; encoder scoring afterward on one GPU',
        bounds='41 camera frames per branch; stop below 512 MiB output free space',
        limitations=['two post-hoc selected development states, not independent maze evaluations',
                    'single image goal; no policy or goal-cost changes during this diagnostic'])
    OUTPUT.mkdir(); base.save(PLAN, plan); base.save(OUTPUT/'plan.json', plan)
    print('OVERSHOOT_BRANCHES_PREPARED', len(records)*len(ACTIONS), flush=True)


def branch(index):
    from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
    plan = json.loads(PLAN.read_text())
    assert plan['source_sha256'] == base.digest(__file__)
    source = plan['sources'][index//6]; action = ACTIONS[index%6]
    original = Path(source['source'])
    assert base.digest(original/'decisions.json') == source['original_decisions_sha256']
    directory = OUTPUT/f'branch_{index:02d}'; directory.mkdir(exist_ok=False)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(4)
    torch.manual_seed(2026091706)
    base.save(directory/'launch.json', dict(pid=os.getpid(), index=index, case=source['case'], action=action,
        cpu_affinity=sorted(os.sched_getaffinity(0)), plan_sha256=base.digest(PLAN)))
    session = None; stop = None; matched = False; result = None; started = time.monotonic()
    try:
        spec = pilot.specification(source['scene'])
        base.save(directory/'specification.json', spec)
        pilot.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        session = pilot.GeometryProgressFamilySession(spec, directory)
        session.install_contact_identity()
        gains = pilot.configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
                                     session.ctx.policy.env_cfg, 'checkpoint')
        base.save(directory/'actuator_identity.json', gains)
        try:
            session.settle_recorded(); session.capture_current()
            pilot.admit_context_setup(session, base.digest(PLAN))
            tape = source['prefix_commands'] + [candidate_commands(action)[0]]*5
            for tick in range(DEPARTURE+6):
                session.sensor_packets()
                if tick == DEPARTURE:
                    with np.load(original/'physics_trace.npz', allow_pickle=False) as archive:
                        for key in ('base_pose_world', 'joint_position', 'joint_velocity', 'applied_command'):
                            np.testing.assert_array_equal(np.stack([s[key] for s in session.samples]),
                                                          archive[key][:len(session.samples)])
                    for frame in (25, 30, 35):
                        assert (directory/f'rgb_{frame:04d}.png').read_bytes() == (original/f'rgb_{frame:04d}.png').read_bytes()
                    matched = True
                    base.save(directory/'matched_departure.json', dict(native_prefix_exact=True, context_rgb_exact=True,
                        frame=tick, original_case=source['case'], samples=len(session.samples)))
                if tick == DEPARTURE+5:
                    break
                if shutil.disk_usage(OUTPUT).free < 512*1024**2:
                    raise RuntimeError('output storage reserve reached')
                session.phase = 1 if tick < 10 else 2
                session.command_tick(tape[tick])
        except PhysicalStop as error:
            stop = str(error); session.capture_current()
        if not matched:
            raise RuntimeError('original live departure not reproduced')
        complete = stop is None and len(session.model_manifest) == 41
        forward_matches = None
        if complete and action == 'forward':
            with np.load(original/'physics_trace.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(np.stack([s['base_pose_world'] for s in session.samples]),
                                              archive['base_pose_world'][:len(session.samples)])
            assert (directory/'rgb_0040.png').read_bytes() == (original/'rgb_0040.png').read_bytes()
            forward_matches = True
        assert pilot.read_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist()) == gains['effective']
        result = dict(status='COMPLETE', branch=index, case=source['case'], action=action,
            complete_500ms=complete, physical_stop=stop, matched_departure=True,
            original_forward_successor_exact=forward_matches, frames=len(session.model_manifest),
            disallowed_contact=bool(any(s['physics_contact'] for s in session.samples)),
            endpoint_pose_evaluator_only=session.samples[-1]['base_pose_world'].tolist(),
            wall_s=time.monotonic()-started, new_navigation=False)
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
    print('OVERSHOOT_BRANCH_COMPLETE', json.dumps(result), flush=True)


@torch.inference_mode()
def score():
    assert not RESULT.exists()
    plan = json.loads(PLAN.read_text())
    results = [json.loads((OUTPUT/f'branch_{i:02d}'/'result.json').read_text()) for i in range(12)]
    assert all(r['status'] == 'COMPLETE' and r['matched_departure'] for r in results)
    torch.set_num_threads(4)
    encoder = base.encoders.VJepa21Arm(); encoder.build(torch.device('cuda:0'), torch.float32)
    cache = {}

    def encode(path):
        key = base.digest(path)
        if key not in cache:
            value = encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float()
            cache[key] = F.layer_norm(value, (1024,))[0]
        return cache[key]

    groups = []
    for group, source in enumerate(plan['sources']):
        original = json.loads((Path(source['source'])/'result.json').read_text())
        goal_pose = np.asarray(original['goal_pose_evaluator_only'])
        rotation = pilot.rotation_xyzw(goal_pose[3:])
        goal_feature = encode(pilot.GOAL_ROOT/source['goal_trial']/'rgb_0023.png')
        rows = []
        for index in range(group*6, (group+1)*6):
            r = results[index]; pose = np.asarray(r['endpoint_pose_evaluator_only'])
            relative = rotation.T @ pilot.rotation_xyzw(pose[3:])
            cost = float((encode(OUTPUT/f'branch_{index:02d}'/'rgb_0040.png')-goal_feature).square().mean()) if r['complete_500ms'] else None
            rows.append(dict(action=r['action'], predicted_goal_mse=source['original_decision']['costs'][index%6],
                actual_goal_mse=cost, xy_error_cm=float(np.linalg.norm(pose[:2]-goal_pose[:2])*100),
                yaw_error_deg=float(abs(np.arctan2(relative[1,0], relative[0,0]))*180/np.pi),
                full_orientation_error_deg=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))*180/np.pi),
                height_error_cm=float(pose[2]-goal_pose[2])*100,
                complete_500ms=r['complete_500ms'], disallowed_contact=r['disallowed_contact']))
        valid = [r for r in rows if r['complete_500ms'] and not r['disallowed_contact']]
        groups.append(dict(case=source['case'], departure_frame=35, rows=rows,
            predicted_choice=ACTIONS[int(np.argmin(source['original_decision']['costs']))],
            actual_visual_best=min(valid,key=lambda r:r['actual_goal_mse'])['action'] if valid else None,
            actual_position_best=min(valid,key=lambda r:r['xy_error_cm'])['action'] if valid else None,
            incomplete_actions=[r['action'] for r in rows if not r['complete_500ms']]))
    report = dict(status='COMPLETE', groups=groups, plan_sha256=base.digest(PLAN),
        all_prefix_native_and_context_rgb_exact=True,
        original_forward_successors_exact=all(r['original_forward_successor_exact'] for r in results if r['action']=='forward'),
        unique_encoded_images=len(cache), navigation_tested=False, model_changed=False,
        post_hoc_failure_diagnostic=True, dense_cache_retained=False)
    base.save(OUTPUT/'result.json',report); base.save(RESULT,report)
    print('OVERSHOOT_DIAGNOSIS_COMPLETE', json.dumps(groups),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--branch', type=int, choices=range(12))
    group.add_argument('--score', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.score: score()
    else: branch(args.branch)
