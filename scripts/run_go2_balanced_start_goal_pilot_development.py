"""Fixed local control comparison following matched predictor continuation."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.balanced_start_goal_control_development import BalancedStartGoalControl
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_cross_trajectory_goal_pilot_development as previous
from scripts import train_go2_balanced_start_predictor_development as fit

pilot=previous.pilot
ROOTS=(fit.OUTPUT.parent/'go2_balanced_start_goal_pilot_v1_attempt_001',
       previous.OUTPUT.parent/'go2_balanced_start_goal_pilot_v1_attempt_001')
DESIGN=Path('docs/go2_balanced_start_goal_pilot_design_2026-09-17.json')
PLAN=Path('docs/go2_balanced_start_goal_pilot_plan_2026-09-17.json')
ARMS=('old_action','mixed_action','old_no_future_action')
CASES=tuple((scene,goal,arm) for scene,goal,_ in previous.CASES for arm in ARMS)
SOURCES=previous.SOURCES+('lewm/balanced_start_goal_control_development.py',__file__)


def root(case):
    return ROOTS[case%2]


def design():
    assert not DESIGN.exists()
    record=dict(cases=CASES,source_sha256={p:pilot.base.digest(p) for p in SOURCES},
                intervention='Training data coverage only within each continued predictor pair; same goal metric, arrival rule, candidate bank and timing.',
                no_selection_using_fit_or_evaluation_outcomes=True,
                fixed_criterion='Completed budget, no contact, final XY <=3cm and relative yaw <=5 degrees.',
                action_blind_policy='The action-blind predictor receives one zero future-action tensor and its forecast/cost is expanded identically across six candidates. Seeded uniform ties and observed-image arrival latch determine commands independently of predictor weights.',
                mixed_action_blind_new_trial=False,
                mixed_action_blind_reason='Same online policy as old_no_future_action by construction; evaluate both predictor weights offline, execute the shared policy once per task.',
                references_reused=['parent predictor with cross-trajectory goal metric: four retained trials',
                                   'direct visual feedback: four retained trials'],
                storage='Six cases on workspace volume, six on root volume; prospectively RGB only.',
                limitations=['four exposed local development tasks','one fit seed',
                             'unchanged known arrival-latch failure','not independent complete maze evaluation',
                             'coverage intervention, not JEPA encoder-objective isolation'])
    pilot.base.save(DESIGN,record);print('BALANCED_GOAL_DESIGN_FIXED',len(CASES),flush=True)


def prepare():
    record=json.loads(DESIGN.read_text())
    assert record['cases']==[list(c) for c in CASES]
    assert record['source_sha256']=={p:pilot.base.digest(p) for p in SOURCES}
    assert not PLAN.exists() and all(not p.exists() for p in ROOTS)
    terminal=json.loads(fit.RESULT.read_text());assert terminal['status']=='COMPLETE'
    evaluation=Path('docs/go2_balanced_start_predictor_evaluation_2026-09-17.json')
    evaluated=json.loads(evaluation.read_text());assert evaluated['status']=='COMPLETE'
    assert evaluated['fit_sha256']==pilot.base.digest(fit.RESULT)
    free=[shutil.disk_usage(p.parent).free for p in ROOTS]
    assert all(v>(512+96)*1024**2 for v in free),free
    plan=json.loads(previous.PLAN.read_text())|dict(cases=CASES,
        source_sha256=record['source_sha256'],design_sha256=pilot.base.digest(DESIGN),
        controllers=[c[2] for c in CASES],predictor_called_by_case=[True]*len(CASES),
        case_directories=[str(root(i)/f'case_{i:02d}') for i in range(len(CASES))],
        continuation_fit_sha256=pilot.base.digest(fit.RESULT),continuation_checkpoints=terminal['checkpoint_sha256'],
        continuation_evaluation_sha256=pilot.base.digest(evaluation),
        intervention=record['intervention'],encoder_predictor_arrival_rule_unchanged=False,
        encoder_goal_metric_arrival_rule_unchanged=True,training_budget_matched=True,
        resources=dict(volume_free_bytes=free,reserve_bytes=512*1024**2,
                       per_volume_allowance_bytes=96*1024**2,concurrency=2,cpu_groups=[[4,5,6,7],[8,9,10,11]]),
        limitations=record['limitations'],references_reused=record['references_reused'],
        action_blind_policy=record['action_blind_policy'],mixed_action_blind_new_trial=False)
    for path in ROOTS:path.mkdir()
    pilot.base.save(PLAN,plan)
    for path in ROOTS:pilot.base.save(path/'plan.json',plan)
    print('BALANCED_GOAL_PILOT_PREPARED',len(CASES),flush=True)


def run(case):
    previous.previous.check_design()
    plan=json.loads(PLAN.read_text())
    assert pilot.base.digest(DESIGN)==plan['design_sha256']
    assert pilot.base.digest(fit.RESULT)==plan['continuation_fit_sha256']
    arm=CASES[case][2]
    assert pilot.base.digest(fit.OUTPUT/f'{arm}_final.pt')==plan['continuation_checkpoints'][arm]
    bind(pilot.run,OUTPUT=root(case),PLAN=PLAN,SOURCE_FILES=SOURCES,CASES=CASES,
         GOAL_ROOT=previous.previous.GOALS,specification=previous.previous.specification,
         GeometryProgressFamilySession=previous.previous.FreshVisualGoalSession,
         DenseVisualGoalControl=BalancedStartGoalControl)(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--design',action='store_true');g.add_argument('--prepare',action='store_true')
    g.add_argument('--case',type=int,choices=range(len(CASES)));a=p.parse_args()
    if a.design:design()
    elif a.prepare:prepare()
    else:run(a.case)
