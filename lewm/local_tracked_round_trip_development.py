"""Run the smaller visual tracker locally to avoid camera-array process transfer."""
from lewm.feature_budget_150_tracker_development import FeatureBudget150VisualMotion
from lewm.process_registered_round_trip_development import ProcessRegisteredRoundTripRuntime


class LocalTrackedRoundTripRuntime(ProcessRegisteredRoundTripRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.motion=FeatureBudget150VisualMotion()

    def _track(self,packet):
        raw=self.motion.observe(packet.policy,packet.depth,packet.fast,now_ns=packet.measured_ns,
            auxiliary_rgb=packet.auxiliary_rgb,auxiliary_depth=packet.auxiliary_depth)
        if raw.get('current_pose') is None or raw.get('failure') is not None:
            raise ValueError('measured visual pose unavailable')
        self.clock_ns()
        self.queues['registration'].put_nowait((packet,raw))
