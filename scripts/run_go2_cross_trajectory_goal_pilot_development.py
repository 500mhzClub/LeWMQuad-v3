"""Prospective exposed-task control after the fixed mixed-pair goal metric fit."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.cross_trajectory_goal_control_development import CrossTrajectoryGoalControl
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_fresh_visual_goal_comparison_development as previous
from scripts import train_go2_cross_trajectory_goal_metric_development as fitted

pilot=previous.pilot
OUTPUT=previous.OUTPUT.parent/'go2_cross_trajectory_goal_pilot_v1_attempt_001'
PLAN=Path('docs/go2_cross_trajectory_goal_pilot_plan_2026-09-17.json')
CASES=tuple(previous.CASES[i] for i in (0,2,4,6))
SOURCES=previous.SOURCES+('lewm/cross_trajectory_goal_control_development.py',__file__)


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    terminal=json.loads(fitted.RESULT.read_text());assert terminal['status']=='COMPLETE'
    evaluation=Path('docs/go2_cross_trajectory_goal_metric_evaluation_2026-09-17.json')
    evaluated=json.loads(evaluation.read_text());assert evaluated['status']=='COMPLETE'
    assert evaluated['fit_sha256']==pilot.base.digest(fitted.RESULT)
    free=shutil.disk_usage(OUTPUT.parent).free;assert free>(512+80)*1024**2,free
    plan=json.loads(previous.PLAN.read_text())|dict(cases=CASES,
        source_sha256={p:pilot.base.digest(p) for p in SOURCES},
        controllers=['world_model_cross_trajectory_metric']*4,predictor_called_by_case=[True]*4,
        case_directories=[str(OUTPUT/f'case_{i:02d}') for i in range(4)],
        original_cases=[0,2,4,6],reference_comparison=str(previous.OUTPUT),
        intervention='goal-metric pair coverage only: half within-recording, half matched cross-recording',
        metric_checkpoint_sha256=terminal['checkpoint_sha256'],metric_fit_sha256=pilot.base.digest(fitted.RESULT),
        metric_evaluation_sha256=pilot.base.digest(evaluation),
        cost='unchanged mean squared distance in learned 64-dimensional goal embedding',
        encoder_predictor_arrival_rule_unchanged=True,training_budget_matched=True,
        predecessor='within-only goal metric on the same four exposed tasks',
        resources=dict(output_free_bytes=free,reserve_bytes=512*1024**2,batch_allowance_bytes=80*1024**2,
            concurrency=2,cpu_groups=[[4,5,6,7],[8,9,10,11]]),
        limitations=['four exposed tasks informed the pairing diagnosis; not independent confirmation',
            'no change to known shared arrival-latch failure',
            'new goal-head fit, not new JEPA encoder/predictor training',
            'no full maze, real-time or hardware qualification'])
    OUTPUT.mkdir();pilot.base.save(PLAN,plan);pilot.base.save(OUTPUT/'plan.json',plan)
    print('CROSS_TRAJECTORY_GOAL_PILOT_PREPARED',str(OUTPUT),flush=True)


def run(case):
    previous.check_design()
    plan=json.loads(PLAN.read_text());assert pilot.base.digest(fitted.RESULT)==plan['metric_fit_sha256']
    bind(pilot.run,OUTPUT=OUTPUT,PLAN=PLAN,SOURCE_FILES=SOURCES,CASES=CASES,
        GOAL_ROOT=previous.GOALS,specification=previous.specification,
        GeometryProgressFamilySession=previous.FreshVisualGoalSession,
        DenseVisualGoalControl=CrossTrajectoryGoalControl)(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--case',type=int,choices=range(4));a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.case)
