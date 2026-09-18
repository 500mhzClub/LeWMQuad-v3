"""New native bootstrap composition; fixed goal/scoring/commit algorithm inherited."""
from lewm.learned_goal_probe_development import LearnedGoalProbe
from lewm.corner_support_joint_observer_development import CornerSupportVisualLedMotion


class FamilyTransitionGoalProbe(LearnedGoalProbe):
    def __init__(self, model):
        super().__init__(model)
        self.motion = CornerSupportVisualLedMotion(identity=(0, 0, 0))

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='family_transition_corner_goal_probe_v1',
            model_data_scope='family_transition_prediction_bootstrap_v1',
            observer='corner_upright_sift_support_v1')
