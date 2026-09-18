"""Reconstruct the frozen predecessor selector using preserved original checks."""
from copy import deepcopy
import json
import numpy as np
from lewm.view_reentry_round_trip_controller_development import ViewReentrySelector
from lewm.causal_subtrajectory_learning_development import causal_history_tensors


class OriginalSurface:
    def __init__(self, surface, selection):
        self.surface = surface; self.checks = selection.get('surface_checks', [])
        self.prediction = selection.get('prediction', []); self.index = 0

    def __getattr__(self, key): return getattr(self.surface, key)

    def footprint(self, geometry, displacement_body_xy, yaw_rad, *, now_ns, persistent):
        if self.index >= len(self.checks) or now_ns != self.surface.last_ns or persistent is not True:
            raise ValueError('original ordered current footprint request required')
        dx, dy, sy, cy, _ = self.prediction[self.index][0]
        if displacement_body_xy != [dx, dy] or yaw_rad != float(np.arctan2(sy, cy)):
            raise ValueError('original first-horizon surface forecast required')
        result = deepcopy(self.checks[self.index]); self.index += 1
        return result


class OriginalMap:
    def __init__(self, mapper, selection):
        self.mapper = mapper; self.surface = OriginalSurface(mapper.surface, selection)

    def __getattr__(self, key): return getattr(self.mapper, key)


class OriginalSelectionReplay:
    def __init__(self, *, condition, variant, goal):
        self.selector = ViewReentrySelector(residual=None, condition=condition,
            variant=variant, goal_initial_body_xy_m=goal)

    def check(self, original, revised, controller, *, now_ns):
        for key in ('prediction', 'nominal_action_checks', 'nominal_path_checks', 'phase_allowed_actions',
                'proposal', 'waypoint_map_xy_m', 'goal_body_xy_m', 'causal_score_residual_receipt'):
            if key in original:
                assert revised[key] == original[key], ('unchanged observed route/forecast/path/score input', key)
        for old, new in zip(original.get('surface_checks', []), revised.get('surface_checks', []), strict=True):
            assert new['original_auxiliary_floor_contact_check'] == old, 'complete original surface witness'
            assert new['shapes'] == old['shapes'] and new['primary_possible_intersection'] == old['primary_possible_intersection']
            assert not new['non_foot_contacts_exempted'] and not new['non_floor_or_unknown_contacts_exempted']
        # Residual observation already consumed the prior actually executed
        # command. Clear only the current, not-yet-executed pending forecast in
        # this private diagnostic copy to reconstruct selection-time state.
        self.selector.residual = deepcopy(controller.residual)
        self.selector.residual.pending = None
        self.selector.set_goal(controller.mission.target())
        proxy = OriginalMap(controller.mapper, original)
        history = causal_history_tensors(list(controller.history), now_ns) if len(controller.history) == 4 else None
        result = self.selector.choose(controller.model, history, proxy, controller.geometry, now_ns=now_ns)
        assert json.loads(json.dumps(result)) == original, 'entire frozen predecessor selection must reconstruct'
        assert proxy.surface.index == len(proxy.surface.checks)
