"""Measured goal dwell and return with one persistent map and visual tracker."""
import numpy as np
from lewm.continuous_commitment_runtime_development import ContinuousCommitmentRuntime
from lewm.stop_conditioned_settling_development import StopConditionedSettlingMission


def preceding_requests(ledger,frame,now_ns):
    if frame==0:return (0.,0.,0.),[]  # original admitted settling precedes frame zero
    times=range(now_ns-100_000_000,now_ns,20_000_000)
    if any(ns not in ledger.requests for ns in times):
        raise ValueError('all five actual preceding policy requests required for arrival dwell')
    rows=[ledger.requests[ns] for ns in times]
    # The inherited mission only tests zero/nonzero. Pass an actual nonzero
    # request if any occurred; never replace a mixed interval with its last zero.
    return next((row for row in rows if any(row)),(0.,0.,0.)),rows


class ContinuousRoundTripRuntime(ContinuousCommitmentRuntime):
    def __init__(self,*args,navigation_ticks=1800,arrival_radius_m=.04,**kwargs):
        mission=dict(goal_initial_body_xy_m=list(kwargs['goal_initial_xy']),
            return_initial_body_xy_m=[0.,0.],require_return_after_goal=True)
        self.mission=StopConditionedSettlingMission(mission,navigation_ticks=navigation_ticks,
            arrival_radius_m=arrival_radius_m)
        self.mission_rows=[];self.mission_latest=None;self.mission_terminal=None
        self.frame_request_history={}
        self.mission_generation=0;self.planning_generation=None
        super().__init__(*args,**kwargs)

    def submit(self,packet):
        # Bind the preceding actuator interval while it is still recent.
        # Registration may finish after the live dispatch ledger expires it.
        with self.lock:
            self.frame_request_history[packet.frame]=preceding_requests(
                self.commitment_ledger,packet.frame,packet.measured_ns)
        super().submit(packet)

    def _register(self,item):
        packet,raw=item
        evidence=self.registration.observe(packet.policy,packet.depth,packet.auxiliary_depth,raw,
            now_ns=packet.measured_ns)
        # observe() has just validated this exact pose and its complete witness.
        # Consume that result without a second reconstruction on the serial path.
        pose=evidence['current_pose']
        p=np.asarray(pose['position_initial_body_m'],float)
        with self.lock:
            command,previous=self.frame_request_history.pop(packet.frame)
        receipt=self.mission.advance(p,frame=packet.frame,now_ns=packet.measured_ns,
            previous_requested_command=command)
        receipt=receipt|dict(preceding_20ms_requests=previous,consumed_pose_frame=pose['frame'])
        published_ns=self.clock_ns()
        receipt['published_ns']=published_ns
        with self.lock:
            self.mission_rows.append(receipt);self.mission_latest=receipt
            self.mission_terminal=receipt['terminal']
            self.goal=np.asarray(receipt['active_goal_initial_body_xy_m'],float)
            if receipt.get('phase_transition') is not None:
                self.mission_generation+=1;self.scan_target=None;self.scan_index=0
            if receipt['hold_required']:
                for plan in self.plans:
                    self.rejected_windows[plan.observed_ns]='MISSION_SETTLING_OR_TERMINAL'
                    self.commitment_ledger.veto(plan,published_ns)
        if self.evidence_sink is not None:self.evidence_sink(packet.frame,raw,evidence)
        if packet.frame%4==0:
            self.queues['mapping'].put_nowait((packet,evidence))
            if packet.frame>=4:self.queues['planning'].put_nowait((packet,evidence))

    def _plan(self,item):
        packet,_=item
        with self.lock:
            mission=self.mission_latest
            self.planning_generation=self.mission_generation
            if mission is None or mission['hold_required']:
                self.planning.append(dict(frame=packet.frame,reason='MISSION_HOLD',
                    measured_ns=packet.measured_ns));return
        super()._plan(item)

    def _store_plan(self,plan,completed,prefix):
        if (self.planning_generation!=self.mission_generation or self.mission_latest['hold_required']):
            self.planning[-1].update(committed=False,discard_reason='MISSION_CHANGED_DURING_PLANNING')
            return
        super()._store_plan(plan,completed,prefix)
        self.planning[-1].update(committed=True,mission_generation=self.mission_generation)

    def _command_gate(self,result,now_ns):
        mission=self.mission_latest
        if mission is None or mission['hold_required']:
            return result|dict(requested_command=[0.,0.,0.],reason='MISSION_SETTLING_OR_TERMINAL',
                mission_phase=None if mission is None else mission['phase'])
        return result|dict(mission_phase=mission['phase'])
