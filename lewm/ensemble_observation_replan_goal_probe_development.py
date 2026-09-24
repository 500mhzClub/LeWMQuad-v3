"""Three fixed predictors: mean utility subject to every member's constraints.

Finite model agreement is not a calibrated error bound or safety certificate.
Every member evaluates the same actual public context and prospective plans.
"""
from copy import deepcopy
import numpy as np
import torch
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.nominal_action_goal_probe_development import NominalActionWaypointSelector
from lewm.observation_replan_goal_probe_development import ObservationReplanGoalProbe

SEEDS=(2026091001,2026091401,2026091402)
MEMBER_KEYS=tuple('seed_'+str(seed) for seed in SEEDS)


def combine(selections):
    if len(selections)!=3:raise ValueError('all three fixed members required')
    first=selections[0]
    for s in selections:
        if any(s.get(k)!=first.get(k) for k in ('proposal','mode','view_budget_exhausted',
                'phase_allowed_actions','head','input_variant','goal_body_xy_m',
                'scan_target_map_yaw_rad','scan_index','scan_sign')):
            raise ValueError('same observed planning phase and candidate semantics required')
    common=dict(proposal=deepcopy(first['proposal']),mode=first['mode'],
        view_budget_exhausted=first['view_budget_exhausted'],member_selections=deepcopy(selections),
        ensemble_seeds=list(SEEDS),model_forecast_horizon_ns=500_000_000,
        finite_ensemble_is_calibrated_uncertainty=False,intermediate_motion_certified=False,
        original_surface_and_nominal_constraints_preserved=True)
    if first['view_budget_exhausted']:
        if any(s['action'] is not None for s in selections):raise ValueError('view exhaustion requires no action')
        return common|dict(action=None,action_index=None,requested_command=[0.,0.,0.],
            phase_admissible_candidates=0,member_predictions_present=False)
    for s in selections:
        if ([c['action'] for c in s['candidates']]!=list(ACTIONS)
                or len(s['surface_checks'])!=6 or len(s['nominal_action_checks'])!=6
                or [c['action'] for c in s['nominal_action_checks']]!=list(ACTIONS)):
            raise ValueError('complete ordered per-member candidate checks required')
    predictions=np.asarray([s['prediction'] for s in selections],float)
    utility=np.asarray([[c['utility_m'] for c in s['candidates']] for s in selections],float)
    if predictions.shape!=(3,6,8,5) or not np.isfinite(predictions).all() or not np.isfinite(utility).all():
        raise ValueError('finite complete forecasts and utilities required')
    feasible=[];checks=[]
    for i,action in enumerate(ACTIONS):
        member_pass=[action in s['phase_allowed_actions'] and not s['surface_checks'][i]['possible_intersection']
            and s['nominal_action_checks'][i]['nominal_disk_connector_clear'] for s in selections]
        accepted=all(member_pass)
        if accepted:feasible.append(i)
        spread=max(float(np.linalg.norm(predictions[a,i,0,:2]-predictions[b,i,0,:2]))
            for a in range(3) for b in range(a+1,3))
        checks.append(dict(action=action,member_feasible=member_pass,every_member_feasible=accepted,
            mean_utility_m=float(utility[:,i].mean()),maximum_pairwise_forecast_xy_difference_m=spread))
    chosen=max(feasible,key=lambda i:checks[i]['mean_utility_m']) if feasible else None
    return common|dict(action=None if chosen is None else ACTIONS[chosen],action_index=chosen,
        requested_command=[0.,0.,0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        phase_admissible_candidates=len(feasible),member_predictions_present=True,
        ensemble_candidate_checks=checks,selection_rule='maximum_mean_member_utility_subject_to_all_member_constraints',
        no_averaged_forecast_used_for_constraint_admission=True)


class EnsembleWaypointSelector:
    def __init__(self,*,condition,variant):
        self.condition=condition;self.variant=variant;self.mode='NEW'
        self.members=[NominalActionWaypointSelector(condition=condition,variant=variant) for _ in SEEDS]

    def choose(self,model,history,mapper,geometry,*,now_ns):
        if not isinstance(model,torch.nn.ModuleDict) or tuple(model.keys())!=MEMBER_KEYS:
            raise ValueError('all fixed-seed models in exact order required')
        selections=[selector.choose(model[key],history,mapper,geometry,now_ns=now_ns)
            for key,selector in zip(MEMBER_KEYS,self.members,strict=True)]
        result=combine(selections);self.mode=result['mode'];return result


class EnsembleObservationReplanGoalProbe(ObservationReplanGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.selector=EnsembleWaypointSelector(condition=condition,variant=variant)

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(
            controller='ensemble_observation_replan_goal_probe_v1',ensemble_seeds=list(SEEDS),
            finite_ensemble_is_calibrated_uncertainty=False)
