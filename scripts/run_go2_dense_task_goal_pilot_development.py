"""Four fixed live trials: matched continuation versus geometry-aware predictor."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.dense_task_goal_control_development import TaskGoalControl
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_dense_metric_goal_pilot_development as previous
from scripts import train_go2_dense_task_predictor_development as fit

pilot = previous.previous
OUTPUT = fit.OUTPUT.parent/'go2_dense_task_goal_pilot_v1_attempt_001'
PLAN = Path('docs/go2_dense_task_goal_pilot_plan_2026-09-17.json')
CASES = (('family_episode_026','family_episode_010','dense_action'),
         ('family_episode_026','family_episode_010','metric_action'),
         ('family_episode_003','family_episode_089','metric_action'),
         ('family_episode_003','family_episode_089','dense_action'))
SOURCES = previous.SOURCES+('lewm/dense_task_goal_control_development.py',__file__)


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    terminal = json.loads(fit.RESULT.read_text()); assert terminal['status'] == 'COMPLETE'
    evaluation = Path('docs/go2_dense_task_predictor_evaluation_2026-09-17.json')
    assert json.loads(evaluation.read_text())['status'] == 'COMPLETE'
    assert shutil.disk_usage(OUTPUT.parent).free > 900*1024**2
    old = json.loads(previous.PLAN.read_text())
    plan = old|dict(cases=CASES,source_sha256={p:pilot.base.digest(p) for p in SOURCES},
        predecessor=str(previous.OUTPUT),intervention='continued dense-L1 versus geometry-aware predictor; same learned goal cost',
        continuation_fit_sha256=pilot.base.digest(fit.RESULT),evaluation_sha256=pilot.base.digest(evaluation),
        continuation_checkpoints=terminal['checkpoint_sha256'],
        encoder_and_predictors_unchanged=False,encoder_and_goal_metric_unchanged=True,
        goals_commands_timing_budget_stops_unchanged=True,stopping_rule_added=False,
        action_blind_reference=dict(root=str(previous.OUTPUT),cases=[1,2],new_trials=False,
            reason='action-blind predictor outputs one identical forecast for all candidates; fixed uniform-tie policy unchanged regardless of predictor weights'),
        resources=dict(output_free_bytes=shutil.disk_usage(OUTPUT.parent).free,
            gpu_vram_bytes=pilot.torch.cuda.get_device_properties(0).total_memory))
    OUTPUT.mkdir(); pilot.base.save(PLAN,plan); pilot.base.save(OUTPUT/'plan.json',plan)
    print('TASK_GOAL_PILOT_PREPARED',str(OUTPUT),flush=True)


def run(case):
    plan = json.loads(PLAN.read_text()); assert pilot.base.digest(fit.RESULT) == plan['continuation_fit_sha256']
    arm = CASES[case][2]
    assert pilot.base.digest(fit.OUTPUT/f'{arm}_latest.pt') == plan['continuation_checkpoints'][arm]
    bind(pilot.run,OUTPUT=OUTPUT,PLAN=PLAN,SOURCE_FILES=SOURCES,CASES=CASES,DenseVisualGoalControl=TaskGoalControl)(case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--case',type=int,choices=range(4)); args = parser.parse_args()
    if args.prepare and args.case is None: prepare()
    elif not args.prepare and args.case is not None: run(args.case)
    else: parser.error('prepare or run one fresh case')
