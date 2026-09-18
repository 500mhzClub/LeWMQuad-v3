"""Low-level depth/gyro worker proceeds independently of visual pose tracking."""
from queue import Queue
from threading import Thread
from lewm.measured_latency_simulation_development import MeasuredLatencyRuntime
from lewm.independent_depth_obstacle_development import IndependentDepthObstacles


class IndependentDepthRuntime(MeasuredLatencyRuntime):
    def __init__(self,*args,**kwargs):
        self.depth_observer=IndependentDepthObstacles()
        super().__init__(*args,**kwargs)
        self.queues['obstacles']=Queue(maxsize=32)
        thread=Thread(target=self._worker,args=('obstacles',self._obstacles),daemon=True)
        self.threads.append(thread);thread.start()

    def submit(self,packet):
        super().submit(packet)
        self.queues['obstacles'].put_nowait(packet)

    def _obstacles(self,packet):
        current=self.depth_observer.observe(packet.policy,packet.depth,packet.fast,
            auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)
        self.clock_ns()
        with self.lock:self.latest_obstacles=current

    def _register(self,item):
        packet,raw=item
        evidence=self.registration.observe(packet.policy,packet.depth,packet.auxiliary_depth,raw,
            now_ns=packet.measured_ns)
        self.clock_ns()
        if self.evidence_sink is not None:self.evidence_sink(packet.frame,raw,evidence)
        if packet.frame%4==0:
            self.queues['mapping'].put_nowait((packet,evidence))
            if packet.frame>=4:self.queues['planning'].put_nowait((packet,evidence))
