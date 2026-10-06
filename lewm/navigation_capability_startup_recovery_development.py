"""Shared startup-only measured-floor recovery; no true geometry or pose inputs."""
from dataclasses import replace
import math
import numpy as np
from lewm import process_mapped_runtime_development as process
from lewm.projected_polygon_floor_coverage_development import (
    ProjectedPolygonFloorRoutingMap, initialize_mapping as previous_initialize_mapping)
from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.pipeline_age_dispatch_development import dispatch_request
from lewm.geometry_progress_pilot_development import candidate_commands

INITIAL_FLOOR_MISSING='initial measured floor unavailable'
WAIT_NS=1_000_000_000
MAX_STARTUP_NS=24_000_000_000
MAX_GYRO_TURN_RAD=2*math.pi
POST_RECOVERY_HOLD_NS=400_000_000


class RecoverableStartupMap(ProjectedPolygonFloorRoutingMap):
    startup_recovery_enabled=True


def initialize_mapping():
    previous_initialize_mapping()
    process._mapper=RecoverableStartupMap()


def mapping_update(packet,evidence):
    try:
        snapshot=process.mapping_update(packet,evidence)
        return snapshot,dict(status='READY',frame=packet.frame,
            floor_source=getattr(process._mapper,'initial_floor_source','primary'))
    except ValueError as error:
        mapper=process._mapper
        if str(error)!=INITIAL_FLOOR_MISSING or mapper.latest is not None or mapper.floor_height is not None:
            raise
        # Preserve the frame-zero gravity origin and all accepted sensor state.
        # Only the absent initial floor is recoverable; other faults still latch.
        mapper.failed=False
        return None,dict(status='WAITING_FOR_MEASURED_INITIAL_FLOOR',frame=packet.frame)


class StartupRecoveryRuntimeMixin:
    def __init__(self,*args,**kwargs):
        self.startup_missing=False
        self.startup_first_ns=None
        self.startup_last_gyro=None
        self.startup_gyro_turn=0.
        self.startup_gyro_ns=-1
        self.startup_recovered_ns=None
        self.startup_rows=[]
        super().__init__(*args,**kwargs)

    def _register(self,item):
        packet,raw=item
        if self.startup_recovered_ns is None:
            R=np.asarray(raw['current_pose']['gyro_rotation_initial_body_from_current_body'])
            yaw=math.atan2(R[1,0],R[0,0])
            with self.lock:
                if self.startup_last_gyro is not None:
                    delta=math.atan2(math.sin(yaw-self.startup_last_gyro),math.cos(yaw-self.startup_last_gyro))
                    self.startup_gyro_turn+=abs(delta)
                self.startup_last_gyro=yaw;self.startup_gyro_ns=packet.measured_ns
        return super()._register(item)

    def _map(self,item):
        packet,evidence=item
        compact=replace(packet,history=(),fast={},auxiliary_rgb={})
        snapshot,receipt=self.mapping_executor.submit(mapping_update,compact,evidence).result()
        self.clock_ns()  # Same publication hook as MeasuredLatencyRuntime._map.
        with self.lock:
            self.latest_map=snapshot
            if snapshot is None:
                self.startup_missing=True
                if self.startup_first_ns is None:self.startup_first_ns=packet.measured_ns
            elif self.startup_missing:
                self.startup_missing=False;self.startup_recovered_ns=packet.measured_ns
                receipt['recovered_after_ns']=packet.measured_ns-self.startup_first_ns
            if self.startup_first_ns is not None or packet.frame==0:
                self.startup_rows.append(receipt|dict(measured_ns=packet.measured_ns,
                    cumulative_absolute_gyro_turn_rad=self.startup_gyro_turn))

    def _plan(self,item):
        packet,_=item
        with self.lock:
            waiting=self.startup_missing or (self.startup_recovered_ns is not None and
                packet.measured_ns<self.startup_recovered_ns+POST_RECOVERY_HOLD_NS)
        if waiting:
            self.planning.append(dict(frame=packet.frame,measured_ns=packet.measured_ns,
                reason='STARTUP_MEASURED_FLOOR_RECOVERY'))
            return
        return super()._plan(item)

    def _command_gate(self,result,now_ns):
        result=super()._command_gate(result,now_ns)
        if not self.startup_missing:
            if self.startup_recovered_ns is not None and now_ns<self.startup_recovered_ns+POST_RECOVERY_HOLD_NS:
                return result|dict(requested_command=[0.,0.,0.],reason='STARTUP_RECOVERY_STOP_PREFIX')
            return result
        elapsed=now_ns-self.startup_first_ns
        common=dict(startup_floor_recovery=True,startup_elapsed_ns=elapsed,
            startup_cumulative_gyro_turn_rad=self.startup_gyro_turn)
        if elapsed>=MAX_STARTUP_NS or self.startup_gyro_turn>=MAX_GYRO_TURN_RAD:
            self.mission_terminal='INITIAL_FLOOR_RECOVERY_BUDGET_EXHAUSTED'
            return result|common|dict(requested_command=[0.,0.,0.],reason=self.mission_terminal)
        if elapsed<WAIT_NS or not 0<=now_ns-self.startup_gyro_ns<=100_000_000:
            return result|common|dict(requested_command=[0.,0.,0.],reason='STARTUP_WAIT_FOR_FLOOR_OR_FRESH_GYRO')
        # Existing candidate command, same current-depth disk/stopping guard.
        # Twenty-ms renewable requests are bounded by measured gyro travel and
        # the 24-s deadline. All requests enter the ordinary command ledger.
        plan=ScheduledCommand(self.startup_gyro_ns,now_ns,now_ns+20_000_000,
            tuple(candidate_commands('left_turn')[0]))
        guarded=dispatch_request(plan,self.latest_obstacles,now_ns=now_ns)
        return result|guarded|common|dict(startup_action='left_turn',
            reason='STARTUP_SCAN_'+guarded['reason'])
