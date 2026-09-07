"""Continuous bounded development action collection, not a navigation policy."""
from copy import deepcopy

import numpy as np

from lewm.articulated_trajectory_evidence_development import command_baseline_trajectory
from lewm.causal_sensor_state import SensorContractError
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.factored_configuration_evidence_development import query_factored_configuration
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior


def motion_priors(epoch, definition_sha256):
    if epoch != 1_500_000_000: raise SensorContractError('fixed new motion admission epoch required')
    return (SetupVelocityPrior((0,0,0),epoch,(0.,0.,0.),.02,definition_sha256),
            SetupRegionPrior((0,0,0),epoch,8_000_000_000,(-1.25,)*3,(1.25,)*3,definition_sha256))


def command_schedule(role):
    if role not in ('identification','validation'): raise ValueError('explicit A/B development role required')
    a = role=='identification'
    segments=((6,(.12 if a else .10,0.,0.)),(4,(0.,0.,0.)),
              (4,(0.,0.,.35 if a else -.35)),(4,(0.,0.,0.)),
              (6,(.15 if a else .08,0.,0.)),(4,(0.,0.,0.)))
    return [list(command) for count,command in segments for _ in range(count)]


class MotionState(ContinuousStartupHandoff):
    """Original startup/tail; explicit factored guard only after READY.

    Retain the predecessor's current-posture result as a labelled comparator.
    No pose reset, terminal-startup restart or ground-support invention.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.factored_guard=None

    def navigation_snapshot(self,*,now_ns):
        result=super().navigation_snapshot(now_ns=now_ns)
        result['predecessor_current_posture_comparator']=result.pop('observed_current_posture')
        result['factored_current_guard']=deepcopy(self.factored_guard)
        return result

    def observe(self,policy,depth,fast,*,now_ns):
        if self.status!='READY_FOR_NAVIGATION_CONSUMER':
            return super().observe(policy,depth,fast,now_ns=now_ns)
        try:
            if type(now_ns) is not int or now_ns!=self._last_ns+100_000_000:
                raise SensorContractError('consecutive motion observations required')
            relative=self._relative.observe(policy,depth,fast,now_ns=now_ns)
            self._memory.observe(policy,depth,relative,now_ns=now_ns)
            self._evidence=self._memory.query_current_primitives(now_ns=now_ns)
            self._last_ns=now_ns; self._count+=1; self._relative_row=deepcopy(relative)
            self.factored_guard=None
            query=query_factored_configuration(self,[0.,0.,0.],np.eye(3),self._memory._joints,0.,
                                               now_ns=now_ns,through_ns=now_ns)
            self.factored_guard={
                'nonfloor_conflict':[r['shape_id'] for r in query['primitives'] if r['nonfloor_conflict_sources']],
                'covered_penetration':[r['shape_id'] for r in query['primitives'] if r['ground']['observed_penetration_sources']],
                'incompatible_ground':[r['shape_id'] for r in query['primitives'] if r['ground']['incompatible_covered_plane_pairs']],
                'ground_support_permission':False}
            if any(self.factored_guard[k] for k in ('nonfloor_conflict','covered_penetration','incompatible_ground')):
                raise SensorContractError('factored current conflict: '+str(self.factored_guard))
            self._last_request=None
            self._last=dict(status=self.status,measured_ns=now_ns,terminal=False,requested_command=None,
                handoff_ready=True,startup_decision=None,consumed_observations=self._count,
                stopping_tail_observations=self._tail_frames,tail_start_ns=self._tail_start_ns,
                depth_rank=self._memory._rays.fusion['depth_rank'],
                combined_position_scale_m=self._memory._rays.fusion['position_error_scale_m'],
                factored_guard=deepcopy(self.factored_guard),navigation_qualified=False,
                contact_permitted=False,real_time_qualified=False)
            return deepcopy(self._last)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            return self._fail(now_ns,error)


class MotionIdentificationController:
    def __init__(self,owner):
        if not isinstance(owner,MotionState): raise TypeError('same continuous motion state required')
        self.owner=owner; self.index=0; self.status='ACTIVE'; self.last=None
        self.schedule=command_schedule('identification')

    def observe(self,policy,depth,fast,*,now_ns):
        if self.status!='ACTIVE': raise SensorContractError('terminal motion controller; no restart')
        state=self.owner.observe(policy,depth,fast,now_ns=now_ns)
        row=dict(measured_ns=now_ns,state=state,motion_index=self.index,phase=1,
            requested_command=[0.,0.,0.],terminal=False,status='ACTIVE',prediction=None,envelope=None,
            navigation_qualified=False,execution_model_validated=False)
        if state['terminal']:
            self.status='FAILED_OBSERVER'; row.update(status=self.status,terminal=True)
        elif not state['handoff_ready']:
            row['requested_command']=state['requested_command']
            row['phase']=2 if state['status']=='MEASURED_ZERO_TAIL' else 1
        else:
            fusion=self.owner._memory._rays.fusion
            centre=np.asarray(fusion['position_initial_body_m'])
            extent=self.owner._startup.radius+.04+.3*.4+fusion['position_error_scale_m']
            through=now_ns+400_000_000
            check=self.owner._region.query([centre-extent],[centre+extent],[0.],
                identity=self.owner._memory._rays.identity,now_ns=through,observed_conflict=[False])
            row['envelope']=dict(centre_initial_body_m=centre.tolist(),half_extent_m=float(extent),
                through_ns=through,setup_contains=bool(check['conditional_setup_non_floor_clearance'][0]))
            speed=float(np.linalg.norm(fusion['velocity_initial_body_m_s']))
            inferred=speed+fusion['initial_velocity_prior_transport']['velocity_radius_m_s']
            if not row['envelope']['setup_contains'] or inferred>.3:
                self.status='FAILED_MOTION_SETUP_OR_SPEED'; row.update(status=self.status,terminal=True)
            elif self.index==len(self.schedule):
                quiet=np.linalg.norm(np.asarray(fast['values'])[1:],axis=1).max()<=.1
                self.status='COMPLETE_IDENTIFICATION_SCHEDULE' if quiet and speed<=.05 else 'FAILED_FINAL_QUIET'
                row.update(status=self.status,terminal=True,phase=3)
            else:
                future=(self.schedule+[[0.,0.,0.]]*4)[self.index:self.index+4]
                row.update(phase=3,requested_command=list(future[0]),
                    prediction=command_baseline_trajectory(self.owner,policy,future,now_ns=now_ns))
                self.index+=1
        self.last=deepcopy(row)
        return row
