"""Unchanged learned-cost planner using a continued dense predictor."""
from lewm.dense_metric_goal_control_development import MetricGoalControl
from scripts import train_go2_dense_task_predictor_development as fit


class TaskGoalControl(MetricGoalControl):
    def __init__(self, arm, goal_image, **kwargs):
        initial, _ = fit.ARMS[arm]
        super().__init__(initial, goal_image, **kwargs)
        self.model = fit.load(arm).cuda()
        self.predictor_arm = arm

    def choose(self, packet):
        record = super().choose(packet)
        record['predictor_arm'] = self.predictor_arm
        return record
