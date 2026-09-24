"""Prospective four-task comparison; setup trajectories only provide goal RGB."""
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
from lewm.dense_visual_arrival_control_development import VisualArrivalControl
from lewm.direct_visual_feedback_control_development import DirectVisualFeedbackControl
from lewm.eligible_floor_registration_development import bind
from lewm.fresh_visual_goal_layouts_development import TASKS, specification, goal_commands
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.fresh_visual_goal_session_development import FreshVisualGoalSession
from scripts.rgb_only_goal_session_development import RGBOnlyGoalCapture
from scripts.geometry_progress_family_session_development import GeometryProgressFamilySession
from scripts.pulse_context_setup_development import admit_context_setup
from scripts import run_go2_direct_visual_feedback_pilot_development as previous

pilot = previous.pilot
OUTPUT = pilot.GOAL_ROOT.parent/'go2_fresh_visual_goal_comparison_v1_attempt_001'
GOALS = OUTPUT/'goals'
DESIGN = Path('docs/go2_fresh_visual_goal_comparison_design_2026-09-17.json')
PLAN = Path('docs/go2_fresh_visual_goal_comparison_plan_2026-09-17.json')
CASES = tuple((t[0], t[0], 'action') for t in TASKS for _ in range(2))
SOURCES = previous.SOURCES+('scripts/rgb_only_goal_session_development.py',
    'lewm/fresh_visual_goal_layouts_development.py', 'scripts/fresh_visual_goal_session_development.py', __file__)


def prepare():
    assert not OUTPUT.exists() and not DESIGN.exists()
    available = shutil.disk_usage(OUTPUT.parent).free
    assert available > (512+128)*1024**2, available
    plan = json.loads(previous.PLAN.read_text()) | dict(cases=CASES,
        source_sha256={p:pilot.base.digest(p) for p in SOURCES},
        task_definitions=[specification(t[0]) for t in TASKS],
        goal_setup_commands={t[0]:goal_commands(t[0]) for t in TASKS},
        controllers=['world_model' if i%2==0 else 'direct_feedback' for i in range(8)],
        goals={},case_directories=[str(OUTPUT/f'case_{i:02d}') for i in range(8)],
        intervention='new geometry and goal tasks; both prior controllers and checkpoints frozen',
        depth_recorded=False,full_maze_navigation=False,
        resources=dict(output_free_bytes=available,additional_allowance_bytes=128*1024**2,
            storage_reserve_bytes=512*1024**2,concurrency=2,cpu_groups=[[4,5,6,7],[8,9,10,11]]),
        output_volume='root volume, prospectively RGB-only recording',
        limitations=['four new parameter combinations in the same local obstruction family',
            'same appearance seed and physical dynamics as training; not new environment-type transfer',
            'task setup fixed tapes only supply goal photographs; evaluated controllers execute online',
            'both use previously exposed-development-designed arrival latch, with known false-positive failure',
            'no parameter tuning after fresh outcomes, no final benchmark or JEPA-training-isolation claim',
            'no complete maze, real-time or hardware evidence'])
    OUTPUT.mkdir();GOALS.mkdir();pilot.base.save(DESIGN,plan);pilot.base.save(OUTPUT/'design.json',plan)
    print('FRESH_DESIGN_PREPARED',str(OUTPUT),flush=True)


def check_design():
    plan=json.loads(DESIGN.read_text())
    assert plan['source_sha256']=={p:pilot.base.digest(p) for p in SOURCES}
    assert pilot.base.digest(pilot.fit.OUTPUT/'result.json')==plan['predictor_fit_sha256']
    assert pilot.base.digest(previous.previous.fitted.RESULT)==plan['arrival_readout_fit_sha256']
    return plan


class ReferenceCaptureSession(RGBOnlyGoalCapture, GeometryProgressFamilySession):
    pass


def collect(task=None, check=False):
    check_design()
    directory=OUTPUT/'capture_check' if check else GOALS/TASKS[task][0]
    directory.mkdir()
    spec=pilot.specification('family_episode_026') if check else specification(TASKS[task][0])
    session_type=ReferenceCaptureSession if check else FreshVisualGoalSession
    tape=[[0.,0.,0.]]*10 if check else goal_commands(TASKS[task][0])
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    session=None;started=time.monotonic()
    pilot.base.save(directory/'launch.json',dict(pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0))))
    pilot.base.save(directory/'specification.json',spec)
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=session_type(spec,directory);session.install_contact_identity()
        gains=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        pilot.base.save(directory/'actuator_identity.json',gains)
        session.settle_recorded();session.capture_current();admit_context_setup(session,pilot.base.digest(DESIGN))
        for requested in tape:
            if shutil.disk_usage(OUTPUT).free < 512*1024**2:raise RuntimeError('storage reserve reached')
            session.phase=1 if check else 2;session.command_tick(requested)
        session.capture_current()
        assert read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())==gains['effective']
        if check:
            reference=previous.previous.OUTPUT/'case_00'
            for i in range(11):
                assert (directory/f'rgb_{i:04d}.png').read_bytes()==(reference/f'rgb_{i:04d}.png').read_bytes()
        assert not any(s['physics_contact'] for s in session.samples)
        result=dict(status='COMPLETE',task=None if check else TASKS[task][0],
            frames=len(session.image_audit),goal_frame=None if check else 23,
            goal_rgb_sha256=None if check else pilot.base.digest(directory/'rgb_0023.png'),
            initial_11_rgb_exact_reference=check,wall_s=time.monotonic()-started,
            setup_only=True,closed_loop_navigation_evidence=False)
        pilot.base.save(directory/'result.json',result);print('GOAL_SETUP_COMPLETE',json.dumps(result),flush=True)
    except Exception as error:
        pilot.base.save(directory/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()));raise
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
                pilot.base.save(directory/'native_guard_rows.json',session.guard_rows)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


def ready():
    assert not PLAN.exists()
    plan=check_design()
    assert json.loads((OUTPUT/'capture_check/result.json').read_text())['initial_11_rgb_exact_reference']
    goals={}
    for task in TASKS:
        directory=GOALS/task[0];result=json.loads((directory/'result.json').read_text())
        assert result['status']=='COMPLETE' and result['frames']==24 and not (directory/'failure.json').exists()
        goals[task[0]]=dict(path=str(directory/'rgb_0023.png'),sha256=pilot.base.digest(directory/'rgb_0023.png'),frame=23)
    plan.update(goals=goals,design_sha256=pilot.base.digest(DESIGN))
    pilot.base.save(PLAN,plan);pilot.base.save(OUTPUT/'plan.json',plan)
    print('FRESH_COMPARISON_READY',flush=True)


def run(case):
    check_design()
    control=VisualArrivalControl if case%2==0 else DirectVisualFeedbackControl
    bind(pilot.run,OUTPUT=OUTPUT,PLAN=PLAN,SOURCE_FILES=SOURCES,CASES=CASES,
        GOAL_ROOT=GOALS,specification=specification,GeometryProgressFamilySession=FreshVisualGoalSession,
        DenseVisualGoalControl=control)(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--check-capture',action='store_true')
    group.add_argument('--goal',type=int,choices=range(4));group.add_argument('--ready',action='store_true')
    group.add_argument('--case',type=int,choices=range(8));a=p.parse_args()
    if a.prepare:prepare()
    elif a.check_capture:collect(check=True)
    elif a.goal is not None:collect(a.goal)
    elif a.ready:ready()
    else:run(a.case)
