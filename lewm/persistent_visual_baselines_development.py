"""Existing action selectors with the fixed persistent visual controller."""
from lewm.continuous_reactive_selection_development import select_reactive
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.heading_first_terminal_reactive_development import heading_first_terminal
from lewm.instantaneous_waypoint_score_development import InstantaneousWaypointScoreMixin
from lewm.current_reserve_terminal_feedback_development import ReservedTerminalFeedbackMixin
from lewm.persistent_local_visual_recovery_development import PersistentLocalVisualRuntime


class InstantaneousPersistentRuntime(InstantaneousWaypointScoreMixin, PersistentLocalVisualRuntime):
    pass


class ComputedForecastsUnusedMixin:
    def _select_action(self, *args, **kwargs):
        selected, correction = super()._select_action(*args, **kwargs)
        selected['terminal_translation_pulse'].update(
            candidate_future_outcomes_evaluated=False,
            progress_score_includes_settling_tail=False)
        correction['neural_outcomes_used_for_action_selection'] = False
        correction['model_forecasts_computed_for_workload_control'] = True
        return selected, correction


class ReservedOffPersistentRuntime(ComputedForecastsUnusedMixin,
        ReservedTerminalFeedbackMixin, PersistentLocalVisualRuntime):
    pass


def select_reactive_terminal(goal, *, scan_error, clearance_m, pulse, arrival_radius_m):
    result = select_reactive(goal, scan_error=scan_error, current_clearance_m=clearance_m)
    result['terminal_translation_pulse'] = dict(enabled=pulse,
        selected_translation_pulse=pulse and any(result['requested_command'][:2]),
        translation_command_duration_ns=100_000_000, planning_cadence_ns=400_000_000,
        candidate_future_outcomes_evaluated=False)
    if pulse:
        result = heading_first_terminal(result, arrival_radius_m=arrival_radius_m)
    result['command_duration_ns'] = (100_000_000 if pulse and any(result['requested_command'][:2])
        else 400_000_000)
    result.update(model_forecasts_computed_for_workload_control=True,
        model_output_validity_still_checked=True, forecast_values_used_for_selection=False,
        planned_stopping_projection=dict(enforced=False, computed=False, changed=False,
            dispatch_guards_unchanged=True))
    return result


class ReactiveFeedbackMixin:
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        # The model still computes for workload control. No forecast value or
        # forecast-dependent selector determines this action.
        clearance = cached_clearance(snapshot.fine_occupied).minimum(position[:2], position[:2])
        return select_reactive_terminal(selected['waypoint_body_xy_m'],
            scan_error=selected.get('scan_heading_error_rad'), clearance_m=clearance,
            pulse=bool(self.planning_translation_pulse), arrival_radius_m=self.mission.arrival_radius_m)


class ReactivePersistentRuntime(ComputedForecastsUnusedMixin,
        ReactiveFeedbackMixin, PersistentLocalVisualRuntime):
    pass


RUNTIMES = dict(instantaneous=InstantaneousPersistentRuntime,
    reserved_off=ReservedOffPersistentRuntime, reactive_feedback=ReactivePersistentRuntime)
