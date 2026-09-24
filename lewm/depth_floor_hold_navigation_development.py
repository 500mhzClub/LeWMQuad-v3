"""Current measured floor plane for active-hold scan/traversal initialization.

No zero-command history is fabricated. This changes the projection's source
from a freshly bootstrapped gravity/foot hypothesis to observed visual depth.
"""
from copy import deepcopy

import numpy as np

from lewm.causal_ground_plane_development import URDF_SHA256
from lewm.causal_sensor_state import SensorContractError
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm.observable_hold_navigation_development import ObservableHoldNavigation


class ObservedDepthFloor:
    def __init__(self,regions):
        self.regions=regions; self.last_ns=None; self.status='NEW'; self._state=None

    def _observe(self,packet,*,now_ns):
        try:
            validate_policy_packet(packet)
            if (self.status=='FAILED_SENSOR' or packet['sensor_state']['decision_ns']!=now_ns
                    or self.regions.memory.last_ns!=now_ns
                    or (self.last_ns is not None and now_ns-self.last_ns!=100_000_000)):
                raise SensorContractError('current consecutive measured floor required')
            floor=self.regions.floor_view
            if floor is None: raise SensorContractError('observed floor unavailable')
            normal=np.asarray(floor['floor_normal_body'],dtype=float)
            height=-float(floor['floor_offset_body_m'])
            if (normal.shape!=(3,) or not np.isfinite(normal).all()
                    or abs(np.linalg.norm(normal)-1.)>1e-8 or not .1<=height<=.6
                    or floor['floor_points']<100 or floor['maximum_plane_residual_m']>.01):
                raise SensorContractError('valid supported measured floor plane required')
            self._state={'decision_ns':now_ns,'up_current_body':normal.tolist(),
                'body_origin_height_m':height,'robot_geometry_sha256':URDF_SHA256,
                'ground_plane_qualified':False,'estimator_mode':'current_observed_depth_floor',
                'depth_floor_evidence':deepcopy(floor),
                'assumptions':['current observed approximately planar visual floor',
                    'fixed depth/body camera calibration; not a collision-ground certificate']}
            self.last_ns=now_ns; self.status='ACTIVE'
            return deepcopy(self._state)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.status='FAILED_SENSOR'
            raise SensorContractError('measured depth floor invalid; apply zero') from error

    def begin(self,packet,*,now_ns):
        if self.status!='NEW': raise SensorContractError('fresh measured floor adapter required')
        return self._observe(packet,now_ns=now_ns)

    def step(self,packet,*,now_ns):
        if self.status!='ACTIVE': raise SensorContractError('active measured floor adapter required')
        return self._observe(packet,now_ns=now_ns)


class DepthFloorHoldNavigation(ObservableHoldNavigation):
    def __init__(self,method,geometry,template=None,*,memory_arm):
        if method!='depth_floor_hold' or template is not None: raise ValueError('distinct depth-floor-hold method required')
        super().__init__('observable_hold',geometry,memory_arm=memory_arm)

    def observe_rgbd(self,packet,fast_packet,depth,relative,*,now_ns):
        try:
            row=super().observe_rgbd(packet,fast_packet,depth,relative,now_ns=now_ns)
            if self.stage=='SCAN' and self.scan.status=='NEW' and not isinstance(self.ground,ObservedDepthFloor):
                self.ground=ObservedDepthFloor(self.regions)
            if self.stage=='TRAVERSE' and self.child is not None and not isinstance(self.child.ground,ObservedDepthFloor):
                if self.child.tick!=-1: raise SensorContractError('cannot replace an active local ground observer')
                self.child.ground=ObservedDepthFloor(self.regions)
            row['local_controller']='depth_floor_hold_navigation_development_v1'
            return row
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self._fail('FAILED_SENSOR',self.last_ns if self.last_ns is not None else 0)
            raise SensorContractError('depth-floor-hold navigation failure; apply zero') from error
