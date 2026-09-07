"""Fixed fresh-motion stimulus and non-actuating, terminal shadow observer."""
from copy import deepcopy

from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_inertial_fusion_development import PointFusionHypotheses
from lewm.rgbd_inertial_ray_memory_development import RGBDInertialRayMemory
from lewm.setup_velocity_prior_development import SetupVelocityPrior
from lewm.setup_region_prior_development import SetupRegionPrior

POINT_HYPOTHESES=PointFusionHypotheses(.002,.01)


def schedule():
    segments=((6,(.08,0.,0.)),(4,(0.,0.,0.)),(6,(0.,0.,.3)),(4,(0.,0.,0.)),
              (6,(-.06,0.,0.)),(4,(0.,0.,0.)),(6,(0.,0.,-.3)),(4,(0.,0.,0.)),
              (6,(.10,0.,0.)),(4,(0.,0.,0.)))
    return [list(command) for count,command in segments for _ in range(count)]


def priors(definition_sha256):
    return (SetupVelocityPrior((0,0,0),1_500_000_000,(0.,0.,0.),.02,definition_sha256),
            SetupRegionPrior((0,0,0),1_500_000_000,7_500_000_000,(-1.25,)*3,(1.25,)*3,definition_sha256))


class ShadowObserver:
    def __init__(self, prior):
        self.model=RGBDInertialRayMemory(prior=prior,hypotheses=POINT_HYPOTHESES)
        self.failure=None; self.successes=0

    def observe(self, policy, depth, fast, *, now_ns):
        if self.failure is not None:
            return dict(status='NOT_REINVOKED_AFTER_SHADOW_FAILURE',measured_ns=now_ns,
                failure=deepcopy(self.failure),state=None,selects_command=False)
        try:
            state=self.model.observe(policy,depth,fast,now_ns=now_ns);self.successes+=1
            return dict(status='SHADOW_OBSERVATION_COMPLETE',measured_ns=now_ns,state=state,
                selects_command=False)
        except SensorContractError as error:
            reasons=[];cause=error
            while cause is not None:reasons.append(str(cause));cause=cause.__cause__
            self.failure=dict(measured_ns=now_ns,chain=reasons)
            return dict(status='TERMINAL_SHADOW_FAILURE',measured_ns=now_ns,state=None,
                failure=deepcopy(self.failure),selects_command=False)
