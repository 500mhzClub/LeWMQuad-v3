"""Prospective cost-only intervention; original native scene/task loop reused."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.dense_metric_goal_control_development import MetricGoalControl
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_dense_visual_goal_pilot_development as previous
from scripts import train_go2_dense_goal_metric_development as fitted

OUTPUT=previous.OUTPUT.parent/'go2_dense_metric_goal_pilot_v1_attempt_001'
PLAN=Path('docs/go2_dense_metric_goal_pilot_plan_2026-09-17.json')
SOURCES=previous.SOURCE_FILES+('lewm/dense_metric_goal_control_development.py',
    'lewm/dense_goal_metric_development.py',__file__)


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    fit=json.loads((fitted.OUTPUT/'result.json').read_text());assert fit['status']=='COMPLETE'
    evaluation=Path('docs/go2_dense_goal_metric_near_goal_result_2026-09-17.json')
    diagnosis=json.loads(evaluation.read_text());assert diagnosis['status']=='COMPLETE'
    assert diagnosis['goal_metric_fit_sha256']==previous.base.digest(fitted.OUTPUT/'result.json')
    old=json.loads(previous.PLAN.read_text())
    available_kib=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
    plan=old|dict(source_sha256={p:previous.base.digest(p) for p in SOURCES},
        predecessor=str(previous.OUTPUT),correction=None,
        intervention='replace raw dense-MSE goal cost with frozen training-only physical goal metric',
        cost='mean squared distance in learned 64-dimensional visual embedding',
        metric_checkpoint_sha256=fit['checkpoint_sha256'],metric_fit_sha256=previous.base.digest(fitted.OUTPUT/'result.json'),
        near_goal_diagnostic_sha256=previous.base.digest(evaluation),
        encoder_and_predictors_unchanged=True,goals_commands_timing_budget_stops_unchanged=True,
        stopping_rule_added=False,physical_pose_used_by_controller=False,
        raw_cost_reference=str(previous.OUTPUT),additional_goal_metric_supervision=True)
    plan['resources']=dict(available_ram_bytes=available_kib*1024,
        output_free_bytes=shutil.disk_usage(OUTPUT.parent).free,
        gpu_vram_bytes=previous.torch.cuda.get_device_properties(0).total_memory)
    OUTPUT.mkdir();previous.base.save(PLAN,plan);previous.base.save(OUTPUT/'plan.json',plan)
    print('METRIC_GOAL_PILOT_PREPARED',str(OUTPUT),flush=True)


def run(case):
    plan=json.loads(PLAN.read_text())
    assert previous.base.digest(fitted.OUTPUT/'metric.pt')==plan['metric_checkpoint_sha256']
    bind(previous.run,OUTPUT=OUTPUT,PLAN=PLAN,SOURCE_FILES=SOURCES,DenseVisualGoalControl=MetricGoalControl)(case)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--case',type=int,choices=range(4));args=parser.parse_args()
    if args.prepare and args.case is None:prepare()
    elif not args.prepare and args.case is not None:run(args.case)
    else:parser.error('prepare or execute one fresh case')
