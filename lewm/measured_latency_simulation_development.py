"""Release worker results no earlier than their measured service duration.

This charges host computation to simulation time. It is not host real-time
qualification: rendering may take longer than simulated sensor acquisition.
"""
from dataclasses import replace
from threading import Condition,local
import time

from lewm.process_mapped_runtime_development import ProcessSeparatedRuntime,pose_update,mapping_update
from lewm.fresh_obstacle_dispatch_development import observe_body_obstacles


class MeasuredLatencyClock:
    def __init__(self,initial_ns=1_500_000_000):
        self.ns=initial_ns;self.condition=Condition();self.local=local();self.closed=False
        self.releases=[]

    def advance(self,ns):
        with self.condition:
            if ns<self.ns:raise ValueError('simulation clock cannot reverse')
            self.ns=ns;self.condition.notify_all()

    def begin(self,stage):
        self.local.stage=stage;self.local.start_sim=self.ns
        self.local.start_wall=time.perf_counter_ns();self.local.waited=0

    def end(self):
        self.local.stage=None

    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()

    def __call__(self):
        if getattr(self.local,'stage',None) is None:return self.ns
        service=time.perf_counter_ns()-self.local.start_wall-self.local.waited
        ready=self.local.start_sim+service;wait_start=time.perf_counter_ns()
        with self.condition:
            while self.ns<ready and not self.closed:self.condition.wait(timeout=.05)
            if self.closed:raise RuntimeError('simulation clock closed before result release')
            released=self.ns
        self.local.waited+=time.perf_counter_ns()-wait_start
        self.releases.append(dict(stage=self.local.stage,start_sim_ns=self.local.start_sim,
            measured_service_ns=service,earliest_release_ns=ready,released_ns=released))
        return released


class MeasuredLatencyRuntime(ProcessSeparatedRuntime):
    def _worker(self,name,function):
        def measured(item):
            self.clock_ns.begin(name)
            try:function(item)
            finally:self.clock_ns.end()
        super()._worker(name,measured)

    def _track(self,packet):
        raw=self.pose_executor.submit(pose_update,replace(packet,history=())).result()
        if raw.get('current_pose') is None or raw.get('failure') is not None:
            raise ValueError('measured visual pose unavailable')
        self.clock_ns()
        self.queues['registration'].put_nowait((packet,raw))

    def _register(self,item):
        packet,raw=item
        obstacles=observe_body_obstacles(packet.policy,packet.depth,raw,
            auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)
        self.clock_ns()
        with self.lock:self.latest_obstacles=obstacles
        evidence=self.registration.observe(packet.policy,packet.depth,packet.auxiliary_depth,raw,
            now_ns=packet.measured_ns)
        self.clock_ns()
        if self.evidence_sink is not None:self.evidence_sink(packet.frame,raw,evidence)
        if packet.frame%4==0:
            self.queues['mapping'].put_nowait((packet,evidence))
            if packet.frame>=4:self.queues['planning'].put_nowait((packet,evidence))

    def _map(self,item):
        packet,evidence=item
        compact=replace(packet,history=(),fast={},auxiliary_rgb={})
        snapshot=self.mapping_executor.submit(mapping_update,compact,evidence).result()
        self.clock_ns()
        with self.lock:self.latest_map=snapshot
