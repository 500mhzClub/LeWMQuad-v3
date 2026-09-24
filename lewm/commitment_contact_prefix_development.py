"""Whole-decision comparison on the actual prefix, with a latched causal stop."""
from copy import deepcopy
from lewm.commitment_contact_score_development import score_commitment_contact
from lewm.geometry_progress_pilot_development import candidate_commands

MAX_FRAMES = 3004


class PrefixComparison:
    def __init__(self):
        self.next_frame = 0
        self.stopped = False
        self.first_command_difference = None

    def compare(self, original, candidate, actual_command, *, frame):
        if (self.stopped or type(frame) is not int or frame != self.next_frame or frame >= MAX_FRAMES
                or original['tick'] != frame or candidate['tick'] != frame
                or original['controller'] != 'measured_floor_transport_round_trip_controller_v1'
                or candidate['controller'] != 'commitment_contact_round_trip_controller_v1'
                or candidate['commitment_contact_policy_enabled'] is not True
                or original['requested_command'] != actual_command):
            raise ValueError('ordered original measured-floor prefix and declared contact successor required')
        for decision in (original, candidate):
            if (decision['model_condition'] != 'supervised_rollout' or decision['input_variant'] != 'full'
                    or decision['memory_variant'] != 'persistent'):
                raise ValueError('same supervised model arm, full input and persistent memory required')
        old = original['new_selection']; expected = score_commitment_contact(old)
        if candidate['new_selection'] != expected:
            raise ValueError('complete selection must equal only the declared contact-cost transformation')
        normalized = deepcopy(candidate)
        normalized.pop('commitment_contact_policy_enabled')
        normalized['controller'] = original['controller']; normalized['new_selection'] = deepcopy(old)
        changed = candidate['requested_command'] != actual_command
        if changed:
            if (original['terminal'] is not None or candidate['terminal'] is not None
                    or candidate['failure'] is not None or not expected
                    or expected['action'] is None or not old or old['action'] is None
                    or candidate['requested_command'] != candidate_commands(expected['action'])[0]
                    or candidate['selected_action'] != expected['action']):
                raise ValueError('changed command must be the feasible transformed action before either terminal')
            for key in ('requested_command', 'selected_action'):
                normalized[key] = deepcopy(original[key])
        if normalized != original:
            raise ValueError('complete observed, mission, map, residual and execution state must remain exact')
        stop = changed or original['terminal'] is not None or candidate['terminal'] is not None
        if changed: self.first_command_difference = frame
        self.next_frame += 1; self.stopped = stop
        return dict(requested_command_changed=changed, raw_model_forecasts_compared=bool(old and 'prediction' in old),
            complete_selection_equals_declared_transform=True, unchanged_observed_mission_and_residual_state_exact=True,
            raw_forecasts_and_all_original_constraints_exact=True, stop=stop,
            unexecuted_outcomes_inferred=False)
