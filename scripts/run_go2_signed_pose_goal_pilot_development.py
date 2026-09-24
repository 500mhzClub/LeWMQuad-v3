"""Four fixed exposed tasks, changing only the world-model goal cost."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.eligible_floor_registration_development import bind
from lewm.signed_pose_goal_control_development import SignedPoseGoalControl
from scripts import run_go2_fresh_visual_goal_comparison_development as previous

pilot=previous.pilot
OUTPUT=previous.OUTPUT.parent/'go2_signed_pose_goal_pilot_v1_attempt_001'
PLAN=Path('docs/go2_signed_pose_goal_pilot_plan_2026-09-17.json')
CASES=tuple(previous.CASES[i] for i in (0,2,4,6))
SOURCES=previous.SOURCES+('lewm/signed_pose_goal_control_development.py',__file__)


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    diagnostic=Path('docs/go2_signed_goal_cost_diagnostic_2026-09-17.json')
    assert json.loads(diagnostic.read_text())['status']=='COMPLETE'
    free=shutil.disk_usage(OUTPUT.parent).free;assert free>(512+80)*1024**2,free
    plan=json.loads(previous.PLAN.read_text())|dict(cases=CASES,
        source_sha256={p:pilot.base.digest(p) for p in SOURCES},
        controllers=['world_model_signed_pose_cost']*4,predictor_called_by_case=[True]*4,
        case_directories=[str(OUTPUT/f'case_{i:02d}') for i in range(4)],
        original_cases=[0,2,4,6],reference_comparison=str(previous.OUTPUT),
        intervention='replace learned embedding-distance cost with squared signed readout pose error; no other controller change',
        cost='sum squared signed goal displacement/heading in fixed 3-cm/5-degree units',
        cost_diagnostic_sha256=pilot.base.digest(diagnostic),
        no_training=True,arrival_readout_rule_unchanged=True,predictor_unchanged=True,
        resources=dict(output_free_bytes=free,reserve_bytes=512*1024**2,batch_allowance_bytes=80*1024**2,
            concurrency=2,cpu_groups=[[4,5,6,7],[8,9,10,11]]),
        limitations=['four previously exposed tasks; cost intervention informed by their diagnosis',
            'single fixed cost substitution, no retraining or threshold tuning',
            'readout trained on actual features; predicted-feature cost can still be inaccurate',
            'known shared arrival-latch false positives remain',
            'no independent maze, JEPA-training-isolation, real-time or hardware claim'])
    OUTPUT.mkdir();pilot.base.save(PLAN,plan);pilot.base.save(OUTPUT/'plan.json',plan)
    print('SIGNED_POSE_PILOT_PREPARED',str(OUTPUT),flush=True)


def run(case):
    previous.check_design()
    bind(pilot.run,OUTPUT=OUTPUT,PLAN=PLAN,SOURCE_FILES=SOURCES,CASES=CASES,
        GOAL_ROOT=previous.GOALS,specification=previous.specification,
        GeometryProgressFamilySession=previous.FreshVisualGoalSession,
        DenseVisualGoalControl=SignedPoseGoalControl)(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--case',type=int,choices=range(4));a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.case)
