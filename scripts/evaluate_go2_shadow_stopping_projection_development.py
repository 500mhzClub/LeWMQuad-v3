"""Physical outcomes and actual unapplied stopping interventions."""
import json
from collections import Counter
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_multiseed_navigation_development as previous
from scripts import run_go2_shadow_stopping_projection_development as experiment


def evaluate(index, arm):
    study = SimpleNamespace(**(vars(previous.study) | dict(ROOT=experiment.ROOT)))
    result = bind(previous.evaluate, study=study)(index, arm)
    root = previous.previous.path(experiment.ROOT.format(index=index,arm=arm))
    selections = [p for p in previous.previous.read(root,'planning.json') if 'selection' in p]
    events = []
    for plan in selections:
        selection = plan['selection']
        check = selection['planned_stopping_projection']
        if (check['enforced'] or check['changed']
                or selection['action'] != check['before_action']
                or selection['action'] != check['after_action']):
            raise ValueError('planned stopping enforcement was not disabled as specified')
        if check['would_change_action']:
            events.append(dict(frame=plan['frame'],
                before_action=check['before_action'], shadow_after_action=check['shadow_after_action']))
    evidence = dict(selected_plans=len(selections), all_stopping_interventions_unapplied=bool(selections),
        would_change_action_count=len(events), events=events,
        full_online_rollout_ablation=False, counterfactual_navigation_outcome_claimed=False)
    previous.previous.save_or_read(root,'shadow_stopping_intervention_evaluation_v1.json',evidence)
    requests = previous.previous.read(root,'requests.json')
    dispatch = []
    for plan in selections:
        check = plan['selection']['planned_stopping_projection']
        if not check['would_change_action']: continue
        associated = [r for r in requests if r.get('command_observation_ns')==plan['measured_ns']]
        applied = [r for r in associated if any(r.get('applied_command',[0,0,0])[:2])]
        dispatch.append(dict(frame=plan['frame'],on_time=plan['on_time'],committed=plan['committed'],
            planned_action=plan['action'],shadow_action=check['shadow_after_action'],
            command_observation_ns=plan['measured_ns'],associated_dispatch_intervals=len(associated),
            applied_translation_intervals=len(applied),
            dispatch_reasons=dict(Counter(r['reason'] for r in associated)),
            minimum_observed_stopping_clearance_m=min((r['stopping_margin_connector']['minimum_observed_cell_distance_m']
                for r in applied if r.get('stopping_margin_connector',{}).get('minimum_observed_cell_distance_m') is not None),default=None)))
    previous.previous.save_or_read(root,'shadow_stopping_dispatch_exposure_v1.json',dict(events=dispatch,
        matching_scope='recorded command_observation_ns equals planning measured_ns',
        counterfactual_navigation_outcome_claimed=False))
    print(json.dumps(evidence),flush=True)
    return result


if __name__ == '__main__':
    study = SimpleNamespace(**(vars(previous.study) | dict(ARMS=experiment.ARMS)))
    bind(previous.main, study=study, evaluate=evaluate)()
