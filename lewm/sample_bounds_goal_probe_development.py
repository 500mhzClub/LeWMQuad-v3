"""Prospective sample-bound geometry with unchanged nominal action constraints."""
from lewm.nominal_action_goal_probe_development import NominalActionGoalProbe
from lewm.continuous_connector_goal_probe_development import ContinuousConnectorFloorMap
from lewm.sample_bounds_surface_memory_development import SampleBoundsSurfaceMemory


class SampleBoundsFloorMap(ContinuousConnectorFloorMap):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity)
        self.surface=SampleBoundsSurfaceMemory(identity=identity)


class SampleBoundsGoalProbe(NominalActionGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.mapper=SampleBoundsFloorMap(identity=(0,0,0));self.memory=self.mapper.surface

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='sample_bounds_goal_probe_v1')
