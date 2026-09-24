"""Commitment contact cost for ordinary anchored-controller waypoint choices."""
from lewm.commitment_contact_score_development import score_commitment_contact
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)

CONTROLLER = 'commitment_contact_anchored_continuation_controller_v1'
FLAG = 'ordinary_waypoint_commitment_contact_enabled'
RECOVERY_FIELDS = ('residual_first_interval_feasibility', 'residual_hold_feasibility',
    'residual_anchored_continuation')


def ordinary_commitment_contact(selection):
    if not selection or any(selection.get(name) is not None for name in RECOVERY_FIELDS):
        return selection
    return score_commitment_contact(selection)


class CommitmentContactAnchoredSelector(ResidualAnchoredContinuationSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        original = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        return ordinary_commitment_contact(original)


class CommitmentContactAnchoredController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = CommitmentContactAnchoredSelector(
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller': CONTROLLER, FLAG: True}
