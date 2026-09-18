"""Prospective connector geometry change; measured maps and surface vetoes retained."""
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_visual_floor_map_development import JointVisualFloorMap
from lewm.continuous_connector_waypoint_development import propose
from lewm.commitment_pose_goal_probe_development import CommitmentPoseGoalProbe


class ContinuousConnectorFloorMap(JointVisualFloorMap):
    def waypoint(self,goal_initial_xy,*,now_ns):
        if self.failed:raise SensorContractError('floor map failure latched')
        self.surface._current(now_ns)
        goal=np.asarray(goal_initial_xy,float)
        if goal.shape!=(2,) or not np.isfinite(goal).all():raise SensorContractError('finite mission XY required')
        position=self.map_from_initial@self.surface.position
        target=self.map_from_initial@np.r_[goal,0.]
        return propose(self.floor,self.occupied,position[:2],target[:2])


class ContinuousConnectorGoalProbe(CommitmentPoseGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.mapper=ContinuousConnectorFloorMap(identity=(0,0,0))
        self.memory=self.mapper.surface

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='continuous_connector_goal_probe_v1')
