"""Same learned controller with complete primary-plus-auxiliary observed mapping."""
from copy import deepcopy
from lewm.training_bias_goal_probe_development import TrainingBiasGoalProbe
from lewm.auxiliary_depth_floor_map_development import AuxiliaryDepthFloorMap


class AuxiliaryDepthGoalProbe(TrainingBiasGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.mapper=AuxiliaryDepthFloorMap(identity=(0,0,0));self.memory=self.mapper.surface

    def observe(self,policy,depth,fast,*,now_ns,auxiliary_depth=None):
        evidence=None;self.memory_receipt=None
        try:
            if self.terminal is None:
                evidence=self.motion.observe(policy,depth,fast,now_ns=now_ns)
                self.memory_receipt=self.mapper.observe(policy,depth,evidence,auxiliary_depth=auxiliary_depth,now_ns=now_ns)
            result=self.advance(policy,evidence,now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.terminal='SENSOR_OR_MODEL_FAILURE';self.failure=str(error)
            result=self._result([0.,0.,0.],None,None)
        return result|dict(evidence=evidence)

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='auxiliary_depth_goal_probe_v1',
            auxiliary_depth_input_required=True,auxiliary_depth_used_by_learned_model=False,
            auxiliary_floor_partition_receipt=deepcopy(self.memory.auxiliary_receipt) if self.memory_receipt is not None else None)
