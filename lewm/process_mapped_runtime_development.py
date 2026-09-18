"""Run routing-map updates in a separate process from the image tracker."""
import cv2
import torch
from dataclasses import replace

from lewm.multirate_routing_map_development import MultirateRoutingMap
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.batched_patch_tracker_development import BatchedPatchVisualMotion

_mapper = None
_motion = None


def initialize_mapping():
    global _mapper
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    _mapper=MultirateRoutingMap()


def mapping_ready():
    return _mapper is not None


def mapping_update(packet,evidence):
    return _mapper.update(packet.policy,packet.depth,evidence,
        auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)


class ProcessMappedRuntime(PacedMultirateController):
    def __init__(self,*args,mapping_executor,**kwargs):
        self.mapping_executor=mapping_executor
        super().__init__(*args,**kwargs)

    def _map(self,item):
        packet,evidence=item
        snapshot=self.mapping_executor.submit(mapping_update,packet,evidence).result()
        with self.lock:self.latest_map=snapshot


def initialize_pose():
    global _motion
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    _motion=BatchedPatchVisualMotion()


def pose_ready():
    return _motion is not None


def initialize_pose_300():
    global _motion
    from lewm.feature_budget_300_tracker_development import FeatureBudget300VisualMotion
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    _motion=FeatureBudget300VisualMotion()


def pose_update(packet):
    return _motion.observe(packet.policy,packet.depth,packet.fast,now_ns=packet.measured_ns,
        auxiliary_rgb=packet.auxiliary_rgb,auxiliary_depth=packet.auxiliary_depth)


class ProcessSeparatedRuntime(ProcessMappedRuntime):
    def __init__(self,*args,pose_executor,**kwargs):
        self.pose_executor=pose_executor
        super().__init__(*args,**kwargs)

    def _track(self,packet):
        raw=self.pose_executor.submit(pose_update,replace(packet,history=())).result()
        if raw.get('current_pose') is None or raw.get('failure') is not None:
            raise ValueError('measured visual pose unavailable')
        self.queues['registration'].put_nowait((packet,raw))

    def _map(self,item):
        packet,evidence=item
        compact=replace(packet,history=(),fast={},auxiliary_rgb={})
        snapshot=self.mapping_executor.submit(mapping_update,compact,evidence).result()
        with self.lock:self.latest_map=snapshot


def pose_and_obstacles_update(packet):
    from lewm.fresh_obstacle_dispatch_development import observe_body_obstacles
    raw=pose_update(packet)
    if raw.get('current_pose') is None or raw.get('failure') is not None:
        raise ValueError('measured visual pose unavailable')
    obstacles=observe_body_obstacles(packet.policy,packet.depth,raw,
        auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)
    return raw,obstacles


class EarlyObstacleRuntime(ProcessSeparatedRuntime):
    def _track(self,packet):
        raw,obstacles=self.pose_executor.submit(
            pose_and_obstacles_update,replace(packet,history=())).result()
        with self.lock:self.latest_obstacles=obstacles
        self.queues['registration'].put_nowait((packet,raw))

    def _register(self,item):
        packet,raw=item
        evidence=self.registration.observe(packet.policy,packet.depth,packet.auxiliary_depth,raw,
            now_ns=packet.measured_ns)
        if self.evidence_sink is not None:self.evidence_sink(packet.frame,raw,evidence)
        # Slower registration must not overwrite a newer current-body observation.
        if packet.frame%4==0:
            self.queues['mapping'].put_nowait((packet,evidence))
            if packet.frame>=4:self.queues['planning'].put_nowait((packet,evidence))


class OverlappedObstacleRuntime(EarlyObstacleRuntime):
    # Keep point extraction off the serial tracker; reuse the registration worker.
    _track=ProcessSeparatedRuntime._track

    def _register(self,item):
        from lewm.fresh_obstacle_dispatch_development import observe_body_obstacles
        packet,raw=item;started=self.clock_ns()
        obstacles=observe_body_obstacles(packet.policy,packet.depth,raw,
            auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)
        with self.lock:self.latest_obstacles=obstacles
        self.events.append(dict(stage='obstacles',frame=packet.frame,measured_ns=packet.measured_ns,
            started_ns=started,completed_ns=self.clock_ns()))
        super()._register(item)
