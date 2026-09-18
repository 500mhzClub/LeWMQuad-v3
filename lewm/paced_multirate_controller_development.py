"""Development runtime with independent sensing, mapping and command clocks.

Inputs are already-acquired paired packets. This runtime neither pauses physics
nor acquires sensors itself. The caller services request() on its actuator clock.
"""
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Event, Lock, Thread
import math
import time

import numpy as np
import torch

from lewm.batched_patch_tracker_development import BatchedPatchVisualMotion
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneFloorRegistration
from lewm.multirate_routing_map_development import MultirateRoutingMap
from lewm.delayed_action_planning_development import (
    delayed_candidate_inputs, score_delayed_predictions, route_from_current_pose, ScheduledCommand)
from lewm.fresh_obstacle_dispatch_development import observe_obstacles, dispatch_request
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.observation_horizon_input_ablation_development import transform_inputs
from lewm.extended_return_budget_transport_development import current_measured_floor_pose
from lewm.observed_floor_waypoint_development import centre
from lewm.geometry_progress_pilot_development import candidate_commands


@dataclass(frozen=True)
class AcquiredFrame:
    frame: int
    measured_ns: int
    policy: dict
    depth: dict
    fast: dict
    auxiliary_rgb: dict
    auxiliary_depth: dict
    history: tuple


class PacedMultirateController:
    def __init__(self, model, *, goal_initial_xy, condition='jepa', variant='full',
            clock_ns, evidence_sink=None, planning_delay_ticks=2, maximum_initial_dispatch_lateness_ns=1_000_000,
            commit_ticks=1):
        if type(planning_delay_ticks) is not int or planning_delay_ticks not in (2,3):
            raise ValueError('planning delay must fit the four-tick cycle')
        self.planning_delay_ticks=planning_delay_ticks
        if type(commit_ticks) is not int or commit_ticks not in (1,4):raise ValueError('one or four commitment ticks required')
        self.commit_ticks=commit_ticks
        self.maximum_initial_dispatch_lateness_ns=maximum_initial_dispatch_lateness_ns
        self.served_windows=set()
        self.model=model;self.goal=np.asarray(goal_initial_xy,float)
        self.condition=condition;self.variant=variant;self.clock_ns=clock_ns
        if condition not in ('direct','supervised_rollout','jepa','reactive') or variant not in ('full','no_rgb'):
            raise ValueError('explicit supported model treatment required')
        if condition=='reactive':
            if model is not None or variant!='full':raise ValueError('reactive arm requires no model and full sensing')
        elif model is None or any(module.training for module in model.modules()):
            raise ValueError('evaluation model required')
        if self.goal.shape!=(2,) or not np.isfinite(self.goal).all():raise ValueError('finite public goal required')
        self.motion=BatchedPatchVisualMotion();self.registration=SampledPlaneFloorRegistration()
        self.mapper=MultirateRoutingMap();self.evidence_sink=evidence_sink
        self.lock=Lock();self.stopped=Event();self.faults=[];self.events=[];self.planning=[]
        self.latest_map=None;self.latest_obstacles=None;self.plans=[]
        self.rejected_windows={}
        self.submitted=0;self.scan_target=None;self.scan_index=0
        self.queues={name:Queue(maxsize=size) for name,size in
            (('tracking',32),('registration',32),('mapping',2),('planning',2))}
        self.threads=[Thread(target=self._worker,args=(name,method),daemon=True)
            for name,method in (('tracking',self._track),('registration',self._register),
                ('mapping',self._map),('planning',self._plan))]
        for thread in self.threads:thread.start()

    def _worker(self, name, function):
        queue=self.queues[name]
        while not self.stopped.is_set():
            try:item=queue.get(timeout=.02)
            except Empty:continue
            started=self.clock_ns()
            try:function(item)
            except BaseException as error:
                with self.lock:self.faults.append(dict(stage=name,reason=repr(error)))
                self.stopped.set()
            finally:
                packet=item if isinstance(item,AcquiredFrame) else item[0]
                self.events.append(dict(stage=name,frame=packet.frame,measured_ns=packet.measured_ns,
                    started_ns=started,completed_ns=self.clock_ns()))
                queue.task_done()

    def submit(self, packet):
        if self.stopped.is_set():raise RuntimeError('runtime stopped')
        if (packet.frame!=self.submitted or packet.measured_ns!=1_500_000_000+packet.frame*100_000_000
                or self.clock_ns()<packet.measured_ns):
            raise ValueError('ordered actually available camera acquisitions required')
        self.queues['tracking'].put_nowait(packet);self.submitted+=1

    def _track(self, packet):
        raw=self.motion.observe(packet.policy,packet.depth,packet.fast,now_ns=packet.measured_ns,
            auxiliary_rgb=packet.auxiliary_rgb,auxiliary_depth=packet.auxiliary_depth)
        if raw.get('current_pose') is None or raw.get('failure') is not None:
            raise ValueError('measured visual pose unavailable')
        self.queues['registration'].put_nowait((packet,raw))

    def _register(self, item):
        packet,raw=item
        evidence=self.registration.observe(packet.policy,packet.depth,packet.auxiliary_depth,raw,
            now_ns=packet.measured_ns)
        if self.evidence_sink is not None:self.evidence_sink(packet.frame,raw,evidence)
        with self.lock:snapshot=self.latest_map
        if snapshot is not None:
            obstacles=observe_obstacles(packet.policy,packet.depth,evidence,snapshot,
                auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)
            with self.lock:self.latest_obstacles=obstacles
        if packet.frame%4==0:
            self.queues['mapping'].put_nowait((packet,evidence))
            if packet.frame>=4:self.queues['planning'].put_nowait((packet,evidence))

    def _map(self, item):
        packet,evidence=item
        snapshot=self.mapper.update(packet.policy,packet.depth,evidence,
            auxiliary_depth=packet.auxiliary_depth,measured_ns=packet.measured_ns)
        with self.lock:self.latest_map=snapshot

    def _route_target(self,route,snapshot,position):
        points=[centre(c) for c in route['route_cells']]
        return next((v for v in points if np.linalg.norm(v-position)>=.35),points[-1])

    def _correct_prediction(self,prediction,packet,evidence,prefix):
        return prediction,None

    @torch.inference_mode()
    def _plan(self, item):
        packet,evidence=item
        with self.lock:
            snapshot=self.latest_map
            planning_goal=self.goal.copy()
            prefix=self._prefix_commands(packet.measured_ns)
        if snapshot is None:
            self.planning.append(dict(frame=packet.frame,reason='NO_COMPLETED_MAP'));return
        if snapshot.measured_ns>packet.measured_ns:
            self.planning.append(dict(frame=packet.frame,reason='MAP_NEWER_THAN_PLANNING_OBSERVATION'));return
        route=self._route(snapshot,evidence,planning_goal,measured_ns=packet.measured_ns)
        p,R,_=self._pose(evidence,identity=(0,0,0),now_ns=packet.measured_ns)
        B=np.asarray(snapshot.map_from_initial);Q,q=B@R,B@p
        scan_error=None
        if 'view_heading_rad' in route:
            heading=math.atan2(Q[1,0],Q[0,0])
            scan_error=math.atan2(math.sin(route['view_heading_rad']-heading),math.cos(route['view_heading_rad']-heading))
            goal_body=np.zeros(2)
        elif route['route_cells']:
            target=self._route_target(route,snapshot,q[:2])
            goal_body=(Q.T@np.r_[target-q[:2],0.])[:2]
        else:
            offsets=(math.pi/4,math.pi/2,-math.pi/4,-math.pi/2,math.pi)
            heading=math.atan2(Q[1,0],Q[0,0])
            if self.scan_target is None:self.scan_target=offsets[self.scan_index]
            scan_error=math.atan2(math.sin(self.scan_target-heading),math.cos(self.scan_target-heading))
            if abs(scan_error)<=.1:
                self.scan_index+=1
                if self.scan_index>=len(offsets):
                    self.planning.append(dict(frame=packet.frame,reason='VIEW_BUDGET_EXHAUSTED'));return
                self.scan_target=offsets[self.scan_index]
                scan_error=math.atan2(math.sin(self.scan_target-heading),math.cos(self.scan_target-heading))
            goal_body=np.zeros(2)
        selected,correction=self._select_action(packet,evidence,prefix,goal_body,scan_error,snapshot,q,Q)
        completed=self.clock_ns()
        plan=ScheduledCommand.prepare(selected['action'],observed_ns=packet.measured_ns,completed_ns=completed,
            delay_ticks=self.planning_delay_ticks,commit_ticks=self.commit_ticks)
        record=dict(frame=packet.frame,measured_ns=packet.measured_ns,completed_ns=completed,
            map_frame=snapshot.frame,action=selected['action'],on_time=plan is not None,
            route_status=route['status'],selection=selected,committed_prefix=prefix)
        if 'frontier_visit' in route:record['frontier_visit']=route['frontier_visit']
        if 'lookahead' in route:record['lookahead']=route['lookahead']
        if 'initial_survey' in route:record['initial_survey']=route['initial_survey']
        if 'clearance_preferred_route' in route:record['clearance_preferred_route']=route['clearance_preferred_route']
        if 'fine_goal_route' in route:record['fine_goal_route']=route['fine_goal_route']
        if correction is not None:record['motion_correction']=correction
        self.planning.append(record)
        if plan is not None:
            with self.lock:self._store_plan(plan,completed,prefix)

    def _select_action(self,packet,evidence,prefix,goal_body,scan_error,snapshot,q,Q):
        history=causal_history_tensors(list(packet.history),packet.measured_ns)
        inputs=transform_inputs(delayed_candidate_inputs(history,prefix,
            delay_ticks=self.planning_delay_ticks,commit_ticks=self.commit_ticks),input_variant=self.variant)
        output=self.model(**inputs);head='direct_outcomes' if self.condition=='direct' else 'rollout_outcomes'
        expected=torch.arange(1,9).mul(100_000_000).expand(6,8)
        if not output['prediction_valid'].all() or not torch.equal(output['target_offsets_ns'],expected):
            raise ValueError('complete exact-horizon action forecasts required')
        prediction=output[head].cpu().numpy()
        prediction,correction=self._correct_prediction(prediction,packet,evidence,prefix)
        selected=self._score(prediction,goal_body,delay_ticks=self.planning_delay_ticks,commit_ticks=self.commit_ticks)
        if scan_error is not None:
            # Model-predicted yaw during the actual candidate interval, keeping
            # the same 1.2 contact coefficient and a fixed 0.35m angular scale.
            allowed=('hold','left_turn','right_turn');scores=[]
            for i,row in enumerate(selected['candidates']):
                if row['action'] not in allowed:continue
                begin=self.planning_delay_ticks-1;end=self.planning_delay_ticks+self.commit_ticks-1
                yaw=math.atan2(prediction[i,end,2],prediction[i,end,3])-math.atan2(prediction[i,begin,2],prediction[i,begin,3])
                remaining=math.atan2(math.sin(scan_error-yaw),math.cos(scan_error-yaw))
                scores.append((.35*(abs(scan_error)-abs(remaining))-1.2*row['predicted_contact_by_commit_end'],i))
            index=max(scores,key=lambda row:row[0])[1]
            selected['action']=selected['candidates'][index]['action'];selected['action_index']=index
            selected['requested_command']=candidate_commands(selected['action'])[0]
            selected.update(scan_heading_error_rad=scan_error,scan_angular_scale_m=.35,
                scan_utilities=[dict(action=selected['candidates'][i]['action'],utility_m=float(s)) for s,i in scores])
        selected=self._select_clear_prediction(selected,prediction,snapshot,q,Q)
        return selected,correction

    _score=staticmethod(score_delayed_predictions)

    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):return selected

    def _pose(self,*args,**kwargs):
        return current_measured_floor_pose(*args,**kwargs)

    def _route(self,*args,**kwargs):
        return route_from_current_pose(*args,**kwargs)

    def _prefix_commands(self,observed_ns):
        if any(p.expires_ns>observed_ns for p in self.plans):
            raise ValueError('existing committed action overlaps proposed zero prefix')
        return [[0.,0.,0.]]*self.planning_delay_ticks

    def _store_plan(self,plan,completed,prefix):
        self.plans.append(plan)

    def request(self, *, now_ns):
        with self.lock:
            if self.faults:return dict(requested_command=[0.,0.,0.],reason='PIPELINE_FAILURE')
            self.plans=[p for p in self.plans if p.expires_ns>now_ns]
            eligible=[p for p in self.plans if p.dispatch_ns<=now_ns]
            plan=eligible[-1] if eligible else None;obstacles=self.latest_obstacles
            self.rejected_windows={k:v for k,v in self.rejected_windows.items()
                if any(p.observed_ns==k for p in self.plans)}
            rejected=None if plan is None else self.rejected_windows.get(plan.observed_ns)
            self.served_windows.intersection_update(p.observed_ns for p in self.plans)
            first=plan is not None and plan.observed_ns not in self.served_windows
            if plan is not None:self.served_windows.add(plan.observed_ns)
        if rejected is not None:
            return dict(now_ns=now_ns,requested_command=[0.,0.,0.],reason='COMMAND_WINDOW_VETO_LATCHED',
                initial_veto=rejected,command_observation_ns=plan.observed_ns,command_expires_ns=plan.expires_ns)
        result=dispatch_request(plan,obstacles,now_ns=now_ns)
        if first and now_ns>plan.dispatch_ns+self.maximum_initial_dispatch_lateness_ns:
            result=result|dict(requested_command=[0.,0.,0.],reason='FIRST_DISPATCH_TOO_LATE')
        if plan is not None and result['reason']!='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED':
            with self.lock:self.rejected_windows[plan.observed_ns]=result['reason']
        return result

    def finish(self, timeout_s=20.):
        deadline=time.perf_counter()+timeout_s
        while not self.stopped.is_set() and any(q.unfinished_tasks for q in self.queues.values()):
            if time.perf_counter()>deadline:raise TimeoutError('pipeline drain deadline exceeded')
            time.sleep(.01)
        self.stopped.set()
        for thread in self.threads:thread.join(timeout=2.)
        if any(thread.is_alive() for thread in self.threads):raise RuntimeError('worker did not stop')
        if self.faults:raise RuntimeError(str(self.faults))
