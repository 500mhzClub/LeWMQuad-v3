"""Frozen goal/arrival controller with a coverage-continuation predictor."""
from lewm.cross_trajectory_goal_control_development import CrossTrajectoryGoalControl
from scripts import train_go2_balanced_start_predictor_development as fit


class BalancedStartGoalControl(CrossTrajectoryGoalControl):
    def __init__(self, arm, goal_image, **kwargs):
        _, initial = fit.ARMS[arm]
        super().__init__(initial, goal_image, **kwargs)
        self.model = fit.load(arm).cuda()
        self.predictor_arm = arm

    def choose(self, packet):
        record = super().choose(packet)
        record['predictor_arm'] = self.predictor_arm
        return record
