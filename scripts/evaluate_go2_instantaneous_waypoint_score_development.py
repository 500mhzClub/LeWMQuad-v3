"""Check actual instantaneous ranking and independently evaluate navigation."""
import json
from types import SimpleNamespace
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.instantaneous_waypoint_score_development import instantaneous_scores
from scripts import evaluate_go2_multiseed_navigation_development as previous
from scripts import run_go2_instantaneous_waypoint_score_development as experiment


def evaluate(index,arm):
    study=SimpleNamespace(**(vars(previous.study)|dict(ROOT=experiment.ROOT)))
    result=bind(previous.evaluate,study=study)(index,arm)
    root=previous.previous.path(experiment.ROOT.format(index=index,arm=arm))
    plans=[p for p in previous.previous.read(root,'planning.json') if 'selection' in p]
    changed=[];overrides=[]
    for plan in plans:
        s=plan['selection'];e=s['instantaneous_ranking']
        rows=instantaneous_scores(s['waypoint_body_xy_m'],scan_error=s.get('scan_heading_error_rad'),
            pulse=s['terminal_translation_pulse']['enabled'])
        if e['rows']!=rows or e['predictions_used_for_main_utilities']:
            raise ValueError('actual main utilities differ from instantaneous objective')
        for candidate,row,forecast in zip(s['candidates'],rows,e['forecast_candidates']):
            if (candidate['action']!=row['action'] or candidate['utility_m']!=row['utility_m']
                    or candidate['position_contact_utility_m']!=row['position_utility_m']
                    or any(candidate[k]!=v for k,v in forecast.items()
                        if k not in ('utility_m','position_contact_utility_m'))):
                raise ValueError('ranking or retained forecast gate evidence differs')
        if 'scan_utilities' in s:
            expected=[dict(action=r['action'],utility_m=r['utility_m']) for r in rows if r['eligible_for_view']]
            if s['scan_utilities']!=expected:raise ValueError('view utilities differ')
        if e['forecast_ranked_action']!=e['instantaneous_ranked_action']:
            changed.append(dict(frame=plan['frame'],forecast=e['forecast_ranked_action'],
                instantaneous=e['instantaneous_ranked_action'],final=plan['action'],on_time=plan['on_time']))
        if plan['action']!=e['instantaneous_ranked_action']:overrides.append(plan['frame'])
    evidence=dict(selected_plans=len(plans),actual_instantaneous_main_utilities_verified=bool(plans),
        forecast_vs_instantaneous_preference_differences=len(changed),preference_differences=changed,
        final_action_differs_from_instantaneous_preference=len(overrides),override_frames=overrides,
        predictive_clearance_recovery_arrival_and_stopping_retained=True,
        full_online_rollout_ablation=False,counterfactual_navigation_claimed=False)
    previous.previous.save_or_read(root,'instantaneous_waypoint_score_treatment_v1.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('preference_differences','override_frames')}),flush=True)
    return result


if __name__=='__main__':
    study=SimpleNamespace(**(vars(previous.study)|dict(ARMS=experiment.ARMS)))
    bind(previous.main,study=study,evaluate=evaluate)()
