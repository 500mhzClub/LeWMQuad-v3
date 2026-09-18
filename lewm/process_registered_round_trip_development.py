"""Keep unchanged sequential registration off the simulator/planner Python thread."""
import cv2
import torch
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneFloorRegistration
from lewm.continuous_round_trip_runtime_development import ContinuousRoundTripRuntime
from lewm.delayed_action_planning_development import route_from_current_pose
from lewm.vectorized_connector_routing_development import propose
from lewm.eligible_floor_registration_development import bind

_registration=None


def initialize_registration():
    global _registration
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    _registration=SampledPlaneFloorRegistration()


def registration_ready():return _registration is not None


def register(policy,primary,auxiliary,raw,now_ns):
    return _registration.observe(policy,primary,auxiliary,raw,now_ns=now_ns)


class RegistrationProxy:
    def __init__(self,executor):self.executor=executor

    def observe(self,policy,primary,auxiliary,raw,*,now_ns):
        return self.executor.submit(register,policy,primary,auxiliary,raw,now_ns).result()


class ProcessRegisteredRoundTripRuntime(ContinuousRoundTripRuntime):
    _propose=staticmethod(propose)

    def _routing_proposer(self,snapshot):return self._propose

    def _route(self,snapshot,*args,**kwargs):
        return bind(route_from_current_pose,propose=self._routing_proposer(snapshot),
            current_measured_floor_pose=self._pose)(snapshot,*args,**kwargs)

    def __init__(self,*args,registration_executor,**kwargs):
        super().__init__(*args,**kwargs)
        self.registration=RegistrationProxy(registration_executor)
