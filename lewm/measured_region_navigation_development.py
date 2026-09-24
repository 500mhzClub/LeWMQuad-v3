"""Development control using observed corner targets and sampled nominal volume.

This changes local decisions, but is not a learned policy or hardware safety
controller. It retains uncertain route memory and independent physical metrics.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.provisional_traversal_ledger_development import ProvisionalTraversalLedger
from lewm.observed_traversal_controller_development import ObservedTraversalController
from lewm.observed_turn_region_development import RayEvidenceMemory, nominal_turn_volume
from lewm.rgb_exit_candidates_development import observe_exit_candidates
from lewm.whole_task_navigation_development import WholeTaskNavigation


class MeasuredArrivalLedger(ProvisionalTraversalLedger):
    def finish(self, *, status, arrival=None):
        if self.record is None or self.record['status'] != 'PENDING': raise ValueError('pending measured attempt required')
        if status == 'ARRIVAL_CANDIDATE':
            if (not isinstance(arrival, dict) or arrival.get('schema') != 'sampled_nominal_turn_region_arrival.v1'
                    or arrival.get('qualified_arrival') is not False or arrival.get('place_identity') is not None
                    or arrival.get('quiet_body_proxy') is not True or arrival['decision_ns'] <= self.record['started_ns']
                    or not arrival['sampled_volume']['all_samples_supported']
                    or arrival['sampled_volume']['unknown_samples'] != 0):
                raise ValueError('explicit measured unqualified arrival evidence required')
        elif status != 'PHYSICAL_STOP' and not status.startswith('FAILED_'):
            raise ValueError('explicit measured terminal required')
        elif arrival is not None: raise ValueError('no arrival on failure')
        self.record.update(status=status, arrival=deepcopy(arrival))


class RegionObservation:
    def __init__(self, geometry):
        self.memory = RayEvidenceMemory(); self.geometry = geometry
        self.corners = []; self.relative = self.latest = None

    def observe(self, packet, depth, relative, *, now_ns):
        result = self.memory.observe(packet, depth, relative, now_ns=now_ns)
        self.relative = relative
        for c in result['corners']:
            if not c['exterior_side_corner']: continue
            point = self.memory.position+self.memory.rotation@np.array([*c['point_body_xy_m'], 0.])
            row = {'point_initial_body_m': point.tolist(), 'approach_initial_body': self.memory.rotation[:,0].tolist(),
                   'longitudinal_role': c['longitudinal_role'], 'observed_ns': now_ns, 'side': c['side']}
            same = [i for i, old in enumerate(self.corners)
                    if old['longitudinal_role'] == row['longitudinal_role'] and old['side'] == row['side']
                    and np.dot(old['approach_initial_body'], row['approach_initial_body']) > .98
                    and np.linalg.norm(np.array(old['point_initial_body_m'])-point) < .12]
            if same: self.corners[same[-1]] = row
            else: self.corners = [*self.corners[-63:], row]
        self.latest = result
        return result

    def volume(self, packet):
        joints=packet['sensor_state']['sensed']['joints']
        if not joints['valid'][:,:12].all(): raise SensorContractError('complete observed posture history required')
        return nominal_turn_volume(self.geometry, joints['values'][:,:12],
                                   self.memory.rotation.T@self.memory.up_initial)

    def target(self, packet, *, now_ns):
        volume = self.volume(packet); radius = volume['maximum_radius_m']
        candidates = []
        for corner in self.corners:
            if now_ns-corner['observed_ns'] > 30_000_000_000: continue
            if np.dot(corner['approach_initial_body'], self.memory.rotation[:,0]) < .95: continue
            point = self.memory.rotation.T@(np.asarray(corner['point_initial_body_m'])-self.memory.position)
            if not (.4 < point[0] < 4.5 and .3 < abs(point[1]) < 2.): continue
            sign = -1. if corner['longitudinal_role'] == 'far_boundary' else 1.
            x = float(point[0]+sign*(radius+.10))
            if x > .15:
                candidates.append({'target_body_m': [x,0.,0.], 'kind': 'OBSERVED_EXTERIOR_CORNER',
                    'corner': deepcopy(corner), 'nominal_radius_m': radius, 'wall_standoff_m': .10})
        if candidates: chosen = min(candidates, key=lambda c:c['target_body_m'][0])
        else:
            # A closed corridor end permits a measured observation/turn target;
            # absence of any visible end is not filled by a fixed travel length.
            front = [s for s in self.memory.latest_surface['surface_segments']
                     if s['normal_body_xy'][0] > .95 and s['offset_body_m'] > radius+.25
                     and min(p[1] for p in s['endpoints_body_xy_m']) <= .05
                     and max(p[1] for p in s['endpoints_body_xy_m']) >= -.05]
            if not front: return None
            wall = min(front, key=lambda s:s['offset_body_m'])
            chosen = {'target_body_m': [wall['offset_body_m']/wall['normal_body_xy'][0]-radius-.15,0.,0.],
                      'kind': 'OBSERVED_FRONT_WALL', 'nominal_radius_m': radius, 'wall_standoff_m': .15}
        chosen['target_initial_body_m'] = (self.memory.position+self.memory.rotation@chosen['target_body_m']).tolist()
        chosen['source_ns'] = now_ns
        return chosen

    def clearance(self, packet, *, now_ns):
        volume = self.volume(packet)
        result = self.memory.query(volume['points_body_m'], volume['ground_support_allowed'], now_ns=now_ns)
        return {'sample_points': len(volume['points_body_m']), 'unknown_samples': int(result['unknown_or_blocked'].sum()),
                'near_surface_samples': int(result['contradictory_or_near_surface'].sum()),
                'all_samples_supported': result['all_samples_supported'],
                'nominal_radius_m': volume['maximum_radius_m'], 'retained_views': result['retained_views'],
                'future_gait_qualified': False, 'continuous_volume_qualified': False}


class MeasuredRegionTraversal(ObservedTraversalController):
    def __init__(self, geometry, observations):
        super().__init__('directional_gait', geometry)
        self.observations = observations
        self.ledger = MeasuredArrivalLedger()
        self.target = None

    def _observe(self, packet, fast_packet, *, now_ns):
        self.tick += 1; self.clock = now_ns
        ground = self.ground.begin(packet, now_ns=now_ns) if self.tick == 0 else self.ground.step(packet, now_ns=now_ns)
        obs = self.observations; memory = obs.memory
        if memory.last_ns != now_ns: raise SensorContractError('current measured region state required')
        image_id = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()
        candidate = arrival = clearance = None
        command = [0.,0.,0.]; distance = None
        gyro = packet['sensor_state']['sensed']['gyro']; joints = packet['sensor_state']['sensed']['joints']
        quiet = bool(gyro['valid'][-5:].all() and joints['valid'][-5:,12:].all()
                     and np.max(np.linalg.norm(gyro['values'][-5:],axis=1)) <= .15
                     and np.sqrt(np.mean(joints['values'][-5:,12:]**2)) <= 1.)
        if self.tick < 3: self.status = 'WARMUP'
        elif self.tick == 3:
            proposals = observe_exit_candidates(packet,ground,now_ns=now_ns,observation_id=image_id)['candidate_rows']
            eligible = [c for c in proposals if abs(c['bearing_body_rad']) <= .35]
            self.target = obs.target(packet,now_ns=now_ns)
            if not eligible or self.target is None: self.status = 'FAILED_NO_MEASURED_TARGET'
            else:
                candidate = min(eligible,key=lambda c:(abs(c['bearing_body_rad']),-c['support_points']))
                self.ledger.begin(observation_id=image_id,decision_ns=now_ns,candidate=candidate)
                self.start_ns=now_ns; self.status='TRAVERSING'
        if self.status in ('TRAVERSING','BRAKING'):
            delta = memory.rotation.T@(np.asarray(self.target['target_initial_body_m'])-memory.position)
            distance = float(delta[0])
            motion = obs.relative['motion']
            speed = max(0.,float(motion['translation_previous_body_m'][0]/.1)) if motion is not None else 0.
            if self.status == 'TRAVERSING' and distance <= .5*speed+.03:
                self.status='BRAKING'; self.brake_ns=now_ns
            if now_ns-self.start_ns >= 30_000_000_000: self.status='FAILED_TIMEOUT'
            if self.status == 'BRAKING':
                if quiet:
                    if self.quiet_since is None: self.quiet_since=now_ns
                else: self.quiet_since=None
                if now_ns-self.brake_ns >= 500_000_000 and self.quiet_since is not None and now_ns-self.quiet_since >= 300_000_000:
                    clearance=obs.clearance(packet,now_ns=now_ns)
                    if distance < -.15: self.status='FAILED_TARGET_OVERSHOOT'
                    elif clearance['all_samples_supported'] and abs(distance) <= .20:
                        self.status='ARRIVAL_CANDIDATE'
                        arrival={'schema':'sampled_nominal_turn_region_arrival.v1','observation_id':image_id,
                            'decision_ns':now_ns,'target':deepcopy(self.target),'remaining_forward_m':distance,
                            'sampled_volume':clearance,'quiet_body_proxy':quiet,'qualified_arrival':False,'place_identity':None}
                    elif distance > .04:
                        # Bounded low-speed measured correction, not a larger
                        # fixed traversal distance or a relaxed clearance count.
                        self.status='TRAVERSING'; self.quiet_since=None
                    else: self.status='FAILED_UNOBSERVED_TURN_VOLUME'
                elif now_ns-self.brake_ns >= 2_000_000_000: self.status='FAILED_SETTLING'
            if self.status == 'TRAVERSING':
                bearing=math.atan2(delta[1],max(delta[0],.05))
                command=[.10 if distance < .25 else .20,0.,float(np.clip(1.5*bearing,-.25,.25))]
            if self.status not in ('TRAVERSING','BRAKING'):
                self.ledger.finish(status=self.status,arrival=arrival)
        terminal=self.status.startswith('FAILED_') or self.status=='ARRIVAL_CANDIDATE'
        return {'status':self.status,'decision_ns':now_ns,'tick':self.tick,'terminal':terminal,
            'requested_command':command,'selection':None,'selected_exit_candidate':candidate,
            'measured_target':deepcopy(self.target),'remaining_forward_m':distance,
            'quiet_body_proxy':quiet,'sampled_turn_volume':clearance,'ledger':self.ledger.snapshot(),
            'scope':'observed nominal observation region; no future-gait, trusted-place or hardware certificate'}


class MeasuredRegionNavigation(WholeTaskNavigation):
    def __init__(self, method, geometry, template=None, *, memory_arm):
        if method != 'measured_region' or template is not None: raise ValueError('first measured-control baseline only')
        super().__init__('fixed_forward',geometry,template,memory_arm=memory_arm)
        self.regions=RegionObservation(geometry)

    def _select(self, selected, now_ns):
        initial = self.stage == 'INITIAL_OBSERVE'
        super()._select(selected,now_ns)
        if initial:
            # The measured target is constructed along the current forward
            # approach. Do not yaw toward a floor-mask bearing before any
            # surrounding volume has been observed. Later acquired branches
            # still use the shared alignment controller and turn-volume gate.
            self.stage='HOLD_TRAVERSE'

    def observe_rgbd(self, packet, fast_packet, depth, relative, *, now_ns):
        try:
            observation=self.regions.observe(packet,depth,relative,now_ns=now_ns)
            decision=super().observe(packet,fast_packet,now_ns=now_ns)
            # Parent creates a fresh child at the end of the hold; replace it
            # before its first observation or command. All outer mission logic
            # remains shared, with an explicit different local controller.
            if self.stage == 'TRAVERSE' and self.child is not None and not isinstance(self.child,MeasuredRegionTraversal):
                if self.child.tick != -1: raise SensorContractError('cannot replace an active local attempt')
                self.child=MeasuredRegionTraversal(self.geometry,self.regions); self.children[-1]=self.child
            # Scan/alignment are development requests only when the current
            # sampled nominal yaw volume is observed; never equate this with
            # a certified future gait envelope.
            if abs(decision['requested_command'][2]) > 0 and decision['stage'] in ('SCAN','ALIGN'):
                clearance=self.regions.clearance(packet,now_ns=now_ns)
                decision['turn_volume']=clearance
                if not clearance['all_samples_supported']:
                    self._fail('FAILED_UNOBSERVED_TURN_VOLUME',now_ns)
                    decision.update(status=self.status,terminal=True,requested_command=[0.,0.,0.])
            decision['region_observation']=observation
            decision['local_controller']='measured_region_development_v1'
            if decision['status']=='HOME_CANDIDATE_ROUTE_HYPOTHESIS' and np.linalg.norm(self.regions.memory.position[:2]) > .25:
                self._fail('FAILED_HOME_ROUTE_MISMATCH',now_ns)
                decision.update(status=self.status,terminal=True,requested_command=[0.,0.,0.])
            return decision
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self._fail('FAILED_SENSOR',self.last_ns if self.last_ns is not None else 0)
            raise SensorContractError('measured-region input/control failure; apply zero') from error

    def _home_appearance(self, packet, attitude, now_ns):
        result=super()._home_appearance(packet,attitude,now_ns)
        distance=float(np.linalg.norm(self.regions.memory.position[:2]))
        result['measured_distance_from_initial_m']=distance
        result['provisional_match'] &= distance <= .25
        return result
