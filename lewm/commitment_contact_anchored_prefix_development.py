"""Whole-decision checks for the fixed full-supervised contact-cost prefix."""
from copy import deepcopy
from lewm.commitment_contact_anchored_controller_development import (
    CONTROLLER, FLAG, ordinary_commitment_contact)
from lewm.geometry_progress_pilot_development import candidate_commands


class PrefixComparison:
    def __init__(self):
        self.next_frame = 0
        self.stopped = False
        self.first_command_difference = None

    def compare(self, original, candidate, actual_command, *, frame):
        if (self.stopped or type(frame) is not int or frame != self.next_frame or not 0 <= frame < 3004
                or original['tick'] != frame or candidate['tick'] != frame
                or original['controller'] != 'residual_anchored_continuation_controller_v1'
                or candidate['controller'] != CONTROLLER or candidate.get(FLAG) is not True
                or original['requested_command'] != actual_command):
            raise ValueError('ordered original anchored prefix and declared contact successor required')
        for decision in (original, candidate):
            if (decision['model_condition'] != 'supervised_rollout' or decision['input_variant'] != 'full'
                    or decision['memory_variant'] != 'persistent'):
                raise ValueError('fixed full-input persistent supervised prefix required')
        old = original['new_selection']; expected = ordinary_commitment_contact(old)
        if candidate['new_selection'] != expected:
            raise ValueError('complete selection must equal only the declared contact transformation')
        normalized = deepcopy(candidate)
        normalized.pop(FLAG)
        normalized['controller'] = original['controller']; normalized['new_selection'] = deepcopy(old)
        changed = candidate['requested_command'] != actual_command
        if changed:
            if (original['terminal'] is not None or candidate['terminal'] is not None
                    or candidate['failure'] is not None or not expected or expected['action'] is None
                    or not old or old['action'] is None
                    or candidate['requested_command'] != candidate_commands(expected['action'])[0]
                    or candidate['selected_action'] != expected['action']):
                raise ValueError('changed command must be the feasible transformed action before terminal')
            for key in ('requested_command', 'selected_action'):
                normalized[key] = deepcopy(original[key])
        if normalized != original:
            raise ValueError('complete observed mission map residual and execution state must remain exact')
        stop = changed or original['terminal'] is not None or candidate['terminal'] is not None
        if changed: self.first_command_difference = frame
        self.next_frame += 1; self.stopped = stop
        return dict(requested_command_changed=changed, stop=stop,
            raw_model_forecasts_compared=bool(old and 'prediction' in old),
            complete_selection_equals_declared_transform=True,
            unchanged_observed_mission_and_residual_state_exact=True,
            raw_forecasts_and_all_original_geometry_constraints_exact=True,
            ordinary_contact_cost_horizon_changed=bool(expected and expected.get('commitment_contact_scoring')),
            unexecuted_outcomes_inferred=False)
