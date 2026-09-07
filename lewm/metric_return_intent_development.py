"""One observed destination and approach heading across clipped local goals."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from lewm.room_return_pulse_development import local_target
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.causal_sensor_state import SensorContractError


@dataclass(frozen=True)
class ReturnIntent:
    stage: str
    target_xy: tuple
    travel_heading: float
    final_heading: float
    created_frame: int

    @classmethod
    def create(cls,stage,pose,target_xy):
        if stage not in ('RETURN_CORNER','TURN_HOME','RETURN_HOME'):raise SensorContractError('return stage required')
        target=np.asarray(target_xy,float);p=np.asarray(pose['position_initial_body_m'],float)
        R=proper(pose['rotation_initial_body_from_current_body'])
        if (target.shape!=(2,) or p.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite(p).all()
                or type(pose['frame']) is not int or pose['frame']<0):raise SensorContractError('finite observed destination and current pose required')
        delta=target-p[:2]
        # A centimetre-scale residual is not a stable approach-direction cue.
        travel=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>.06 else math.atan2(R[1,0],R[0,0])
        return cls(stage,tuple(target),travel,0. if stage=='RETURN_HOME' else travel,pose['frame'])

    def subgoal(self,pose):
        d,w,final=local_target(pose,self.target_xy,self.travel_heading)
        if final:d,w,final=local_target(pose,self.target_xy,self.final_heading)
        if self.stage=='TURN_HOME':d=[0.,0.];final=True
        return d,w,final

    def snapshot(self):return asdict(self)
