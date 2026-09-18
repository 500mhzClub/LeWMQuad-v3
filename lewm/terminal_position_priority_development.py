"""Prefer useful translation near an orientation-unconstrained mission point."""
from copy import deepcopy
import numpy as np
from lewm.clearance_preferred_route_development import ClearancePreferredTurnRecoveryRuntime
from lewm.clearance_turn_recovery_development import choose
from lewm.geometry_progress_pilot_development import ACTIONS

# One nominal 400 ms forward commitment (80 mm) plus the 20 mm observed
# arrival radius. This does not change either arrival or clearance limits.
TERMINAL_APPROACH_RADIUS_M=.10


def select_terminal_progress(selection):
    result=deepcopy(selection)
    if (result.get('clearance_turn') or {}).get('active'):return result
    candidates={r['action']:r for r in result['candidates']}
    baseline=max(candidates['hold']['position_contact_utility_m'],
        candidates[result['action']]['position_contact_utility_m'])
    eligible=[i for i,row in enumerate(result['memory_forecast_candidates'])
        if row['action'] in ('forward','left_arc','right_arc') and row['nominal_predicted_path_clear']
        and candidates[row['action']]['predicted_progress_during_commit_m']>0.
        and candidates[row['action']]['position_contact_utility_m']>baseline]
    original=result['action']
    if eligible:
        index=max(eligible,key=lambda i:candidates[ACTIONS[i]]['position_contact_utility_m'])
        choose(result,index)
    result['terminal_position_priority']=dict(original_action=original,selected_action=result['action'],
        changed=original!=result['action'],position_contact_baseline_m=baseline,
        original_selection_objective=result.get('selection_objective'),
        heading_guidance_retained_when_no_better_translation=True,
        prediction_and_clearance_checks_unchanged=True,final_heading_required=False)
    result['selection_objective']='terminal_position_contact_priority_with_heading_fallback'
    return result


class TerminalPositionPriorityMixin:
    def _route(self,*args,**kwargs):
        self.terminal_position_approach=False
        return super()._route(*args,**kwargs)

    def _route_target(self,route,snapshot,position):
        target=super()._route_target(route,snapshot,position)
        self.terminal_position_approach=bool(
            route['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
            and np.linalg.norm(target-np.asarray(route['goal_map_xy_m']))<1e-9
            and np.linalg.norm(target-np.asarray(position))<=TERMINAL_APPROACH_RADIUS_M)
        route['lookahead']['terminal_position_approach']=self.terminal_position_approach
        return target

    def _select_clear_prediction(self,*args,**kwargs):
        result=super()._select_clear_prediction(*args,**kwargs)
        return select_terminal_progress(result) if self.terminal_position_approach else result


class TerminalPositionPriorityRuntime(TerminalPositionPriorityMixin,ClearancePreferredTurnRecoveryRuntime):
    pass


class ProgressRejoiningTerminalRuntime(TerminalPositionPriorityRuntime):
    release_for_progress=True
