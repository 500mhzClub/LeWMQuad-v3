"""Same bounded reobservation controller with calibrated 45-degree depth mapping."""
from lewm.auxiliary_depth_reobserve_goal_probe_development import AuxiliaryDepthReobserveGoalProbe
from lewm.auxiliary_downward45_floor_map_development import AuxiliaryDownward45FloorMap
from lewm.auxiliary_downward45_depth_observation_development import CALIBRATION_ID


class AuxiliaryDownward45GoalProbe(AuxiliaryDepthReobserveGoalProbe):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mapper=AuxiliaryDownward45FloorMap(identity=(0,0,0));self.memory=self.mapper.surface

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(
            controller='auxiliary_downward45_goal_probe_v1',auxiliary_calibration_id=CALIBRATION_ID)
