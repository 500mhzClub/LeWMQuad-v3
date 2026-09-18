"""Prospective budget comparison; no input admission, replay or native launch."""
from copy import deepcopy

from lewm import extended_return_budget_controller_development as candidate
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMission
from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorVisualMotion
from lewm.measured_plane_chained_single_pass_controller_development import CONTROLLER as BASELINE
from scripts.measured_plane_chained_single_pass_comparison_development import normalize as to_recorded
from scripts.measured_plane_full_history_timing_development import observer_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

ORIGINAL_BUDGET = 4000
BOUNDARY = 4003
MAX_PREFIX_OBSERVATIONS = BOUNDARY+1
BUDGET_PATHS = ('shared_navigation_budget_ticks', 'mission_receipt.global_navigation_ticks')
TYPE_PATHS = (
    ('mapper', candidate.ExtendedReturnBudgetFloorMap, candidate.BodyProjectedTiledFloorMap),
    ('mapper.fields.surface', candidate.ExtendedReturnBudgetMemory, candidate.MeasuredFloorTransportMemory),
    ('mapper.fields.surface.fields.later_floor_evidence', candidate.ExtendedReturnBudgetLaterFloorEvidence, candidate.LaterFloorEvidence),
    ('memory', candidate.ExtendedReturnBudgetMemory, candidate.MeasuredFloorTransportMemory),
    ('memory.fields.later_floor_evidence', candidate.ExtendedReturnBudgetLaterFloorEvidence, candidate.LaterFloorEvidence),
    ('residual', candidate.ExtendedReturnBudgetResidual, candidate.MeasuredFloorTransportResidual),
    ('selector', candidate.ExtendedReturnBudgetSelector, candidate.receipts.ReceiptCopiedFootprintSelector),
    ('selector.fields.residual', candidate.ExtendedReturnBudgetResidual, candidate.MeasuredFloorTransportResidual),
    ('registration', candidate.ExtendedReturnBudgetFloorRegistration, candidate.TiledDensityFloorRegistration),
    ('mission', candidate.ExtendedReturnBudgetMeasuredMission, MeasuredFloorTransportMission))
STATE_BUDGET_PATHS = ('mission.fields.navigation_ticks',
    'mission.fields.last.global_navigation_ticks', 'mission_receipt.global_navigation_ticks')


def _exact_budget(value, expected):
    if type(value) is not int or value != expected:
        raise ValueError('exact declared integer mission budget required')


def normalize(decision):
    """Change two budget scalars and the new implementation declaration only."""
    if (type(decision) is not dict or decision.get('controller') != candidate.CONTROLLER
            or decision.get('extended_return_budget_enabled') is not True):
        raise ValueError('exact extended controller declaration required')
    _exact_budget(decision['shared_navigation_budget_ticks'], candidate.NAVIGATION_TICKS)
    _exact_budget(decision['mission_receipt']['global_navigation_ticks'], candidate.NAVIGATION_TICKS)
    result = deepcopy(decision)
    del result['extended_return_budget_enabled']
    result['controller'] = BASELINE
    result['shared_navigation_budget_ticks'] = ORIGINAL_BUDGET
    result['mission_receipt']['global_navigation_ticks'] = ORIGINAL_BUDGET
    return result


def observed_state(controller):
    """Keep all controller fields except separately authenticated model/geometry.

Only ten explicitly located type tags (seven classes, including aliases) and
three mission-budget scalars are normalized. Unknown fields remain visible.
"""
    extended = type(controller) is candidate.ExtendedReturnBudgetChainedController
    if (type(controller) not in (candidate.MeasuredPlaneChainedSinglePassController,
            candidate.ExtendedReturnBudgetChainedController)
            or type(controller.motion) is not MeasuredPlaneChainedAnchorVisualMotion
            or controller.memory is not controller.mapper.surface
            or controller.residual is not controller.selector.residual):
        raise ValueError('exact controllers, unchanged chained motion and original aliases required')
    expected_budget = candidate.NAVIGATION_TICKS if extended else ORIGINAL_BUDGET
    _exact_budget(controller.mission.navigation_ticks, expected_budget)
    tree = observer_state_tree({k:v for k,v in vars(controller).items() if k not in ('model', 'geometry')})
    for path, new_type, old_type in TYPE_PATHS:
        node = tree
        for key in path.split('.'): node = node[key]
        expected = new_type if extended else old_type
        if (set(node) != {'type', 'fields'}
                or node['type'] != expected.__module__+'.'+expected.__name__):
            raise ValueError('exact complete implementation state required at '+path)
        node['type'] = old_type.__module__+'.'+old_type.__name__
    tree['mission']['fields']['navigation_ticks'] = ORIGINAL_BUDGET
    for receipt in (tree['mission']['fields']['last'], tree['mission_receipt']):
        if receipt is not None:
            _exact_budget(receipt['global_navigation_ticks'], expected_budget)
            receipt['global_navigation_ticks'] = ORIGINAL_BUDGET
    return tree


class PrefixComparison:
    """Stop on any decision intervention; never admit a following observation."""
    def __init__(self):
        self.frames = 0
        self.stopped = False
        self.last = None
        self.forecasts = 0

    def observe(self, baseline, extended, recorded, *, model_calls):
        if self.stopped or self.frames >= MAX_PREFIX_OBSERVATIONS:
            raise ValueError('no observation after the prospective intervention or original boundary')
        frame = self.frames
        for decision in (baseline, extended, recorded):
            if type(decision.get('tick')) is not int or decision['tick'] != frame:
                raise ValueError('complete consecutive current decisions required')
        if baseline.get('controller') != BASELINE or 'extended_return_budget_enabled' in baseline:
            raise ValueError('exact unchanged single-pass baseline required')
        _exact_budget(baseline['shared_navigation_budget_ticks'], ORIGINAL_BUDGET)
        _exact_budget(baseline['mission_receipt']['global_navigation_ticks'], ORIGINAL_BUDGET)
        if fingerprint(to_recorded(baseline)) != fingerprint(recorded):
            raise ValueError('complete recorded native decision must reproduce')
        if (type(model_calls) is not list or len(model_calls) != 2
                or any(type(v) is not int or v not in (0, 1) for v in model_calls)):
            raise ValueError('two actual bounded model forward counts required')
        normalized = normalize(extended)
        exact = fingerprint(baseline) == fingerprint(normalized)
        forecasts = []
        for index, decision in enumerate((baseline, extended)):
            selection = decision['new_selection']
            forecast = bool(selection and 'prediction' in selection)
            if (forecast and model_calls[index] != 1
                    or decision['terminal'] is None and forecast != bool(model_calls[index])):
                raise ValueError('forecast evidence must match actual model calls')
            if decision['terminal'] is not None and decision['requested_command'] != [0., 0., 0.]:
                raise ValueError('terminal decision must retain zero command')
            forecasts.append(forecast)
        if exact and model_calls[0] != model_calls[1]:
            raise ValueError('equal decisions require equal actual model calls')
        changed_request = baseline['requested_command'] != extended['requested_command']
        changed_terminal = baseline['terminal'] != extended['terminal']
        terminal = baseline['terminal'] is not None or extended['terminal'] is not None
        reason = ('FIRST_NORMALIZED_DECISION_DIFFERENCE' if not exact else
            'MATCHED_TERMINAL' if terminal else 'ORIGINAL_BOUNDARY_REACHED' if frame == BOUNDARY else None)
        common_forecast = exact and all(forecasts)
        self.forecasts += int(common_forecast)
        self.frames += 1
        self.stopped = reason is not None
        self.last = dict(frame=frame, complete_recorded_decision_reproduced=True,
            complete_normalized_decision_exact=exact, actual_model_forward_calls=list(model_calls),
            complete_equal_forecast_compared=common_forecast, requested_command_changed=changed_request,
            terminal_changed=changed_terminal, original_terminal=baseline['terminal'],
            candidate_terminal=extended['terminal'], original_requested_command=deepcopy(baseline['requested_command']),
            candidate_requested_command=deepcopy(extended['requested_command']),
            stop=self.stopped, stop_reason=reason)
        return deepcopy(self.last)

    def report(self):
        if not self.stopped:
            raise ValueError('actual prospective stopping observation required')
        supported = (self.last['frame'] == BOUNDARY
            and self.last['original_terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
            and self.last['candidate_terminal'] is None)
        return dict(frames=self.frames, boundary=deepcopy(self.last),
            budget_only_preboundary_decisions_supported=supported,
            equal_preboundary_forecast_decisions=self.forecasts,
            normalized_budget_paths=list(BUDGET_PATHS),
            normalized_state_type_paths=[p+'.type' for p,_,_ in TYPE_PATHS],
            normalized_state_budget_paths=list(STATE_BUDGET_PATHS),
            state_checks_performed_by_this_comparator=False,
            public_packet_and_model_authentication_performed=False,
            following_intervention_observations_consumed=False,
            changed_command_executed=False, physical_prefix_verified=False,
            navigation_qualified=False, verified_round_trip=False, goal_achieved=False)
