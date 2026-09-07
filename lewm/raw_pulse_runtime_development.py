"""Raw sensor ownership for continuous motion and the episodic route bridge."""
from lewm.visual_led_motion_development import VisualLedMotion
from lewm.continuous_pulse_execution_development import ContinuousPulseExecution
from lewm.pulse_route_bridge_development import PulseRouteBridge


class RawPulseExecution:
    def __init__(self, *, identity=(0,0,0), **budgets):
        self.motion=VisualLedMotion('gyro',identity=identity)
        self.executor=ContinuousPulseExecution(identity=identity,**budgets)

    def observe(self, policy, depth, fast, *, now_ns):
        try:
            evidence=self.motion.observe(policy,depth,fast,now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            stamp=now_ns if type(now_ns) is int and now_ns>=0 else self.executor.last_ns or 0
            self.executor.fail('RAW_SENSOR_REJECTED: '+str(error),now_ns=stamp)
            evidence=None
        execution=self.executor.observe(evidence,now_ns=now_ns)
        return dict(evidence=evidence,execution=execution,requested_command=execution['requested_command'])


class RawPulseRouteRuntime:
    def __init__(self, *, identity=(0,0,0), **budgets):
        self.motion=VisualLedMotion('gyro',identity=identity)
        self.bridge=PulseRouteBridge(identity=identity,**budgets)

    def observe(self, policy, depth, fast, *, now_ns):
        try:
            evidence=self.motion.observe(policy,depth,fast,now_ns=now_ns)
            # Same gyro stream, not a second reference reconstructed from pose.
            attitude=self.motion.model.gyro._result()
            route=self.bridge.observe(policy,attitude,evidence,now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            stamp=self.bridge.executor.last_ns or 0
            if type(now_ns) is int and now_ns>=stamp:stamp=now_ns
            self.bridge._fail('RAW_SENSOR_REJECTED: '+str(error),stamp)
            evidence=None;route=self.bridge._result(stamp,None)
        return dict(evidence=evidence,route=route,requested_command=route['requested_command'])
