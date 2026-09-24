"""Exploratory classification of four retained Stage A logs; no model/physics calls."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
ACTIONS=('hold','forward','left_arc','right_arc','left_turn','right_turn')


def classify(row):
    s=row['selection'];checks={r['action']:r for r in s.get('memory_forecast_candidates',[])}
    utilities={r['action']:r['utility_m'] for r in s.get('scan_utilities',s.get('candidates',[]))}
    stopping=s.get('planned_stopping_projection',{})
    stop_checks={r['action']:r['projection_clear'] for r in stopping.get('candidates',[])}
    result=dict(frame=row['frame'],route_status=row.get('route_status'),category='insufficient_evidence',
        observation_action_space_exclusions=[a for a in ACTIONS[1:] if a not in utilities],
        motion_clearance_exclusions=[a for a in ACTIONS[1:] if a in utilities and a in checks and not checks[a].get('nominal_predicted_path_clear',False)],
        stopping_projection_exclusions=[a for a in ACTIONS[1:] if a in utilities and checks.get(a,{}).get('nominal_predicted_path_clear',False) and stop_checks.get(a) is False],
        clearance_modes={a:r.get('clearance_check_mode') for a,r in checks.items()},
        override_reason=None)
    if set(checks)!=set(ACTIONS) or not utilities or any(not math.isfinite(v) for v in utilities.values()):
        result['reason']='missing candidate gates or finite recorded utilities';return result
    eligible=[a for a in ACTIONS if a in utilities and checks[a].get('nominal_predicted_path_clear',False) and stop_checks.get(a,True)]
    moving=[a for a in eligible if a!='hold'];result['eligible_movement']=moving
    turn=s.get('clearance_turn',{})
    if stopping.get('changed') and stopping.get('before_action')!='hold' and s['action']=='hold':
        result.update(category='explicit_override',override_reason='PLANNED_STOPPING_PROJECTION')
    elif turn.get('active') and turn.get('latched_turn_forecast_clear') is False:
        result.update(category='explicit_override',override_reason='LATCHED_RECOVERY_TURN_BLOCKED')
    elif not moving:
        result['category']='no_eligible_movement'
    elif 'hold' in eligible and all(utilities[a]<=utilities['hold'] for a in moving):
        result.update(category='movement_lost_recorded_score_or_tie',
            exact_tie_with_movement=any(utilities[a]==utilities['hold'] for a in moving),
            tie_rule='first maximum in canonical candidate order; hold is first')
    else:
        result['reason']='logged eligible movement beats hold or hold eligibility missing; no explicit retained override explains it'
    result['no_eligible_movement_even_if_override']=not moving
    return result


def main():
    runs=[]
    for layout,head in ((0,'old_data'),(0,'maze_data'),(2,'maze_data'),(2,'old_data')):
        root=BASE/f'go2_dense_world_model_maze_layout{layout:02d}_action_maze_view_{head}_v1_attempt_001'
        planning=root/'planning.json';requests=root/'requests.json'
        plans=json.loads(planning.read_text());dispatch=json.loads(requests.read_text())
        selected=[p for p in plans if 'selection' in p]
        holds=[p for p in selected if p['action']=='hold']
        classified=[classify(p) for p in holds]
        zeros=[r for r in dispatch if not any(r['requested_command'])]
        summary=dict(layout=layout,head=head,selected_plans=len(selected),hold_plans=len(holds),
            categories=dict(Counter(r['category'] for r in classified)),
            explicit_override_reasons=dict(Counter(r['override_reason'] for r in classified if r['override_reason'])),
            exact_score_ties=sum(r.get('exact_tie_with_movement',False) for r in classified),
            holds_with_observation_action_space_restriction=sum(bool(r['observation_action_space_exclusions']) for r in classified),
            holds_with_motion_clearance_exclusion=sum(bool(r['motion_clearance_exclusions']) for r in classified),
            holds_with_stopping_projection_exclusion=sum(bool(r['stopping_projection_exclusions']) for r in classified),
            observation_only_excluded_candidate_count=sum(len(r['observation_action_space_exclusions']) for r in classified),
            motion_clearance_excluded_candidate_count=sum(len(r['motion_clearance_exclusions']) for r in classified),
            stopping_projection_excluded_candidate_count=sum(len(r['stopping_projection_exclusions']) for r in classified),
            zero_request_service_calls=len(zeros),zero_request_reason_counts=dict(Counter(r['reason'] for r in zeros)),
            non_selection_planning_rows=len(plans)-len(selected),
            inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (planning,requests)},rows=classified)
        assert sum(summary['categories'].values())==len(holds)
        runs.append(summary)
    report=dict(schema='stage_a_exploratory_hold_classification.v1',exploratory=True,runs=runs,
        classification_precedence=['explicit retained override','no eligible movement','movement loses logged score or exact tie','insufficient evidence'],
        eligibility='Recorded scan-action subset (or all candidate scores), nominal_predicted_path_clear including recorded recovery, and recorded stopping projection. No new model forecasts or clearance computations.',
        observation_only_scope='Scan/view action-space restrictions identifiable from logged scan_utilities; other observation-only exclusions not fully reconstructed.',
        motion_scope='Memory-path and stopping-projection gates depend on forecasts and observed geometry. Neither is physical acceptability.',
        dispatch_scope='20-ms zero-request calls are counted separately; they are not additional hold plans and are not summed with planner categories.',
        constraints=dict(model_calls=0,physics_steps=0,audit_regrets=0),
        limitations=['Different trajectories; no isolated causal readout effect.','Canonical tie logic checked against retained-source call path; no speculative override assignment.','Recorded utilities are not physical reference costs.'])
    path=Path('docs/go2_stage_a_holds_exploratory_2026-09-23.json')
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('rows','inputs')} for r in runs],indent=2))


if __name__=='__main__':main()
