"""Compute planned stopping intervention, retaining the preceding decision."""
from copy import deepcopy
from lewm.planned_stopping_projection_development import (
    PlannedStoppingProjectionMixin, avoid_blocked_translation, stopping_projection_checks)


def record_unapplied_stopping(selection, checks):
    shadow = avoid_blocked_translation(selection, checks)
    result = deepcopy(selection)
    result['planned_stopping_projection'] = shadow['planned_stopping_projection'] | dict(
        enforced=False, changed=False,
        would_change_action=shadow['planned_stopping_projection']['changed'],
        shadow_after_action=shadow['action'], after_action=selection['action'])
    return result


class ShadowStoppingProjectionMixin:
    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        # Skip only PlannedStoppingProjectionMixin. All downstream predicted
        # path, reserve, heading, arrival and recovery selection still executes.
        result = super(PlannedStoppingProjectionMixin, self)._select_clear_prediction(
            selected, prediction, snapshot, position, rotation)
        checks = stopping_projection_checks(prediction, snapshot.fine_occupied,
            position, rotation, pulse=bool(self.planning_translation_pulse))
        # An unapplied intervention must not reset the recovery-turn latch.
        return record_unapplied_stopping(result, checks)
