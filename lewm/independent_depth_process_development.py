"""Stateful depth/gyro extraction separated from main-process Python work."""
from dataclasses import replace
import cv2
import torch
from lewm.independent_depth_obstacle_development import IndependentDepthObstacles
from lewm.independent_depth_runtime_development import IndependentDepthRuntime

_observer=None


def initialize_obstacles():
    global _observer
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    _observer=IndependentDepthObstacles()


def obstacles_ready():return _observer is not None


def observe(packet):
    current=_observer.observe(packet.policy,packet.depth,packet.fast,
        auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)
    return current,_observer.receipts[-1]


class IndependentDepthProcessRuntime(IndependentDepthRuntime):
    def __init__(self,*args,obstacle_executor,**kwargs):
        self.obstacle_executor=obstacle_executor
        super().__init__(*args,**kwargs)

    def _obstacles(self,packet):
        current,receipt=self.obstacle_executor.submit(observe,
            replace(packet,history=(),auxiliary_rgb={})).result()
        self.clock_ns()
        self.depth_observer.receipts.append(receipt)
        with self.lock:self.latest_obstacles=current
