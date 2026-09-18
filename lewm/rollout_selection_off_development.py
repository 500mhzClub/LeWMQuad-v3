"""Current-clearance feedback without learned candidate-rollout decisions."""
import math
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.instantaneous_waypoint_score_development import instantaneous_scores
from lewm.fine_stored_obstacle_routing_development import cached_clearance


def select_current_clearance(goal, *, scan_error=None, pulse=False, clearance_m=None):
    if clearance_m is not None and (not math.isfinite(clearance_m) or clearance_m<0.):
        raise ValueError('finite nonnegative current observed clearance required')
    clear=clearance_m is None or clearance_m>.45+1e-12
    rows=instantaneous_scores(goal,scan_error=scan_error,pulse=pulse)
    for row in rows:
        row['eligible']=bool(clear and (scan_error is None or row['eligible_for_view']))
    eligible=[r for r in rows if r['eligible']]
    action=max(eligible,key=lambda r:r['utility_m'])['action'] if eligible else 'hold'
    return dict(action=action,action_index=ACTIONS.index(action),
        requested_command=candidate_commands(action)[0],candidates=rows,
        waypoint_body_xy_m=list(goal),scan_heading_error_rad=scan_error,
        current_stored_clearance_m=clearance_m,current_nominal_disk_clear=clear,
        selection_objective='instantaneous_waypoint_cost_with_current_clearance',
        learned_candidate_rollouts_used_for_selection=False,
        model_forecasts_computed_for_workload_control=True,
        model_output_validity_still_checked=True,
        predictive_clearance_recovery_arrival_and_stopping_retained=False,
        current_depth_dispatch_check_still_required=True,
        unknown_space_inferred_free=False,
        planned_stopping_projection=dict(enforced=False,computed=False,changed=False,
            dispatch_guards_unchanged=True),
        rollout_selection_off=dict(nominal_footprint_radius_m=.45,
            current_clearance_query_only=True,forecast_trajectory_clearance_used=False,
            predictive_recovery_used=False,predictive_arrival_override_used=False,
            geometric_view_planning_retained=True,actual_dispatch_projection_retained=True))


class RolloutSelectionOffMixin:
    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):
        # Do not call the forecast-dependent selection chain or read prediction.
        # Waypoint and scan error come from the observed-map route, not the model.
        clearance=cached_clearance(snapshot.fine_occupied).minimum(position[:2],position[:2])
        return select_current_clearance(selected['waypoint_body_xy_m'],
            scan_error=selected.get('scan_heading_error_rad'),
            pulse=bool(self.planning_translation_pulse),clearance_m=clearance)
