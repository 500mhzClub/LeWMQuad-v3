"""Two local visual-arrival trials; same planner until a learned goal detection."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.dense_visual_arrival_control_development import VisualArrivalControl
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_dense_metric_goal_pilot_development as previous
from scripts import train_go2_direct_visual_goal_readout_development as fitted

pilot = previous.previous
OUTPUT = previous.OUTPUT.parent/'go2_dense_visual_arrival_pilot_v1_attempt_001'
PLAN = Path('docs/go2_dense_visual_arrival_pilot_plan_2026-09-17.json')
CASES = (pilot.CASES[0],pilot.CASES[3])
SOURCES = previous.SOURCES+('lewm/dense_visual_arrival_control_development.py',__file__)


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    terminal = json.loads(fitted.RESULT.read_text()); assert terminal['status']=='COMPLETE'
    evaluation = Path('docs/go2_direct_visual_goal_readout_evaluation_2026-09-17.json')
    assert json.loads(evaluation.read_text())['status']=='COMPLETE'
    observed = []
    for case in (0,3):
        root = previous.OUTPUT/f'case_{case:02d}'
        observed.append(sum(p.stat().st_size for p in root.iterdir() if p.is_file()
            and p.name!='sealed_test.json' and not p.name.startswith('sealed_')))
    required = 512*1024**2+int(2.5*max(observed))
    free = shutil.disk_usage(OUTPUT.parent).free
    assert free > required, (free,required)
    plan = json.loads(previous.PLAN.read_text())|dict(cases=CASES,
        source_sha256={p:pilot.base.digest(p) for p in SOURCES},
        predecessor=str(previous.OUTPUT),reference_cases=[0,3],
        arrival_readout_fit_sha256=pilot.base.digest(fitted.RESULT),arrival_readout_checkpoint_sha256=terminal['checkpoint_sha256'],
        arrival_evaluation_sha256=pilot.base.digest(evaluation),
        intervention='latch terminal hold on first learned observed-goal estimate within original 3 cm / 5 degree tolerances',
        stopping_rule_added=True,oracle_success_still_evaluator_only=True,
        no_budget_shortening=True,predictions_after_arrival=False,
        resources=dict(output_free_bytes=free,required_bytes=required,observed_reference_case_bytes=observed),
        output_volume='dedicated experiment volume; selected after checking measured output size and reserve',
        limitations=['two exposed development tasks; intervention motivated by their diagnosis',
            'one-observation arrival latch can produce false arrivals; native outcome remains independent',
            'success would isolate learned goal recognition, not improve the predictor or establish planning superiority',
            'no new reactive baseline, independent maze, real-time or hardware qualification'])
    OUTPUT.mkdir();pilot.base.save(PLAN,plan);pilot.base.save(OUTPUT/'plan.json',plan)
    print('VISUAL_ARRIVAL_PREPARED',str(OUTPUT),flush=True)


def run(case):
    plan=json.loads(PLAN.read_text());assert pilot.base.digest(fitted.RESULT)==plan['arrival_readout_fit_sha256']
    bind(pilot.run,OUTPUT=OUTPUT,PLAN=PLAN,SOURCE_FILES=SOURCES,CASES=CASES,DenseVisualGoalControl=VisualArrivalControl)(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case',type=int,choices=(0,1));a=p.parse_args()
    if a.prepare and a.case is None:prepare()
    elif not a.prepare and a.case is not None:run(a.case)
    else:p.error('prepare or execute one new case')
