"""Prospective measured-floor foot contacts with unchanged nominal planning."""
from copy import deepcopy
from lewm.causal_sensor_state import SensorContractError
from lewm.sample_bounds_goal_probe_development import SampleBoundsFloorMap,SampleBoundsGoalProbe
from lewm.measured_floor_contact_development import MeasuredFloorContactMemory


class MeasuredFloorContactMap(SampleBoundsFloorMap):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.surface=MeasuredFloorContactMemory(identity=identity)

    def observe(self,policy,depth,evidence,*,now_ns):
        try:
            receipt=super().observe(policy,depth,evidence,now_ns=now_ns)
            self.surface.classify_current(policy,depth,self.map_from_initial,self.floor_height,self.floor,now_ns=now_ns)
            return receipt
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as exc:
            self.failed=True;self.surface.failed=True
            raise SensorContractError('measured-floor contact map unavailable') from exc


class MeasuredFloorContactGoalProbe(SampleBoundsGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.mapper=MeasuredFloorContactMap(identity=(0,0,0));self.memory=self.mapper.surface

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='measured_floor_contact_goal_probe_v1',
            floor_partition_receipt=deepcopy(self.memory.classification_receipt) if self.memory_receipt is not None else None)
