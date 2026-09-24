"""Persistent-intent mission with a consistent inner planning/settling region."""
from lewm.coupled_room_return_development import CoupledPulseExecution
from lewm.inner_goal_pulse_feedback_development import InnerGoalPulseServo
from lewm.intent_room_return_development import IntentRoomReturn
from lewm.causal_sensor_state import SensorContractError


class InnerGoalPulseExecution(CoupledPulseExecution):
    def begin(self,displacement_body_xy,yaw_delta_rad,*,now_ns):
        leg=super().begin(displacement_body_xy,yaw_delta_rad,now_ns=now_ns)
        try:
            self.active=InnerGoalPulseServo(self.active.goal,self.table)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.fail('INNER_GOAL_DISPATCH_REJECTED: '+str(error),now_ns=now_ns)
            raise SensorContractError('inner goal dispatch failure latched') from error
        return leg


class InnerGoalRoomReturn(IntentRoomReturn):
    def __init__(self,sign,table):
        super().__init__(sign,table)
        # Constructor-only replacement; no observation, pose or history reset.
        self.runtime.executor=InnerGoalPulseExecution(table)
