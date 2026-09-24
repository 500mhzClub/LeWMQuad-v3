"""The same continuous mission/waypoints with a distinct local controller."""
from lewm.continuous_pulse_execution_development import ContinuousPulseExecution
from lewm.coupled_pulse_feedback_development import CoupledPulseServo
from lewm.raw_pulse_runtime_development import RawPulseExecution
from lewm.visual_led_motion_development import VisualLedMotion
from lewm.room_return_pulse_development import RoomReturnPulse
from lewm.causal_sensor_state import SensorContractError
from lewm.coupled_pulse_rollout_development import PulseTable


class CoupledPulseExecution(ContinuousPulseExecution):
    def __init__(self,table,**kwargs):
        if not isinstance(table,PulseTable):raise SensorContractError('typed pulse model required')
        super().__init__(**kwargs);self.table=table

    def begin(self,displacement_body_xy,yaw_delta_rad,*,now_ns):
        leg=super().begin(displacement_body_xy,yaw_delta_rad,now_ns=now_ns)
        try:
            # Reuse transactional goal admission, but the old servo never steps.
            self.active=CoupledPulseServo(self.active.goal,self.table)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.fail('COUPLED_DISPATCH_REJECTED: '+str(error),now_ns=now_ns)
            raise SensorContractError('coupled dispatch failure latched') from error
        return leg


class RawCoupledPulseExecution(RawPulseExecution):
    def __init__(self,table,*,identity=(0,0,0),**budgets):
        self.motion=VisualLedMotion('gyro',identity=identity)
        self.executor=CoupledPulseExecution(table,identity=identity,**budgets)


class CoupledRoomReturn(RoomReturnPulse):
    def __init__(self,sign,table):
        super().__init__(sign)
        self.runtime=RawCoupledPulseExecution(table)
