"""Body-sensor/robot-geometry ground-plane hypothesis, not certified clearance.

Initial gravity direction assumes zero-command settling; it is not inferred from
simulator attitude. Subsequent attitude uses gyro integration. Height assumes
the lowest foot sphere touches a common flat support plane. Flight, slip, stairs,
uneven support, IMU bias and joint/calibration errors can violate these assumptions.
This is a candidate sensor fusion component and requires actual-trace evaluation.
"""
import hashlib
from pathlib import Path

import numpy as np

from lewm.causal_relative_orientation_development import CausalRelativeOrientation
from lewm.causal_sensor_state import SensorContractError,_ns
from lewm.simulated_body_observation_development import validate_policy_packet

URDF_SHA256='4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4'


def verify_robot_geometry(path):
    path=Path(path).absolute()
    if path.resolve()!=path or any(p in ('sealed','sealed_test.json') or p.startswith('sealed_') for p in path.parts):
        raise ValueError('explicit nonprotected non-symlink robot geometry required')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=URDF_SHA256: raise ValueError('robot geometry calibration changed')


def foot_sphere_centres_body(joint_position):
    q=np.asarray(joint_position,dtype=float)
    if q.shape!=(12,) or not np.isfinite(q).all(): raise ValueError('ordered12 finite joint angles required')
    # Exact source order: four hip, four thigh, four calf (FL,FR,RL,RR).
    result=[]
    for i in range(4):
        hip,thigh,calf=q[i],q[4+i],q[8+i]; side=1 if i in (0,2) else -1
        ch,sh=np.cos(hip),np.sin(hip); ct,st=np.cos(thigh),np.sin(thigh)
        ck,sk=np.cos(thigh+calf),np.sin(thigh+calf)
        rotation_hip=np.array([[1,0,0],[0,ch,-sh],[0,sh,ch]])
        # Two213-mm segments plus the foot collision sphere's -2-mm x offset
        # in the calf frame. Calflower visual links are not foot ancestors.
        leg=np.array([-.213*st-.213*sk-.002*ck,side*.0955,-.213*ct-.213*ck+.002*sk])
        result.append(np.array([.1934 if i<2 else -.1934,side*.0465,0])+rotation_hip@leg)
    return np.stack(result)


class CausalGroundPlane:
    def __init__(self):
        self.status='NEW'; self.orientation=CausalRelativeOrientation(); self._up_initial=None
        self._state=None; self.last_ns=None

    def _state_from(self,packet,rotation,now_ns):
        joints=packet['sensor_state']['sensed']['joints']
        if joints['measured_ns'][-1]!=now_ns or not joints['valid'][-1,:12].all():
            raise SensorContractError('current ordered joint positions required')
        feet=foot_sphere_centres_body(joints['values'][-1,:12]); up=np.asarray(rotation).T@self._up_initial
        heights=.022-feet@up; height=float(np.max(heights))
        if not .1<=height<=.6: raise SensorContractError('ground hypothesis outside robot kinematic envelope')
        return {'decision_ns':now_ns,'up_current_body':up.tolist(),'body_origin_height_m':height,
            'per_foot_support_height_m':heights.tolist(),'foot_sphere_centres_body_m':feet.tolist(),
            'robot_geometry_sha256':URDF_SHA256,'ground_plane_qualified':False,
            'assumptions':['initial zero-command gravity','gyro relative attitude','lowest foot sphere contacts flat support']}

    def begin(self,packet,*,now_ns):
        if self.status!='NEW': raise SensorContractError('fresh ground hypothesis required')
        try:
            validate_policy_packet(packet); now_ns=_ns(now_ns,'ground initialization clock')
            force=packet['sensor_state']['sensed']['specific_force']; command=packet['sensor_state']['control']['applied_command']
            gyro=packet['sensor_state']['sensed']['gyro']
            if not force['valid'].all() or not command['valid'].all() or np.any(np.abs(command['values'])>1e-8):
                raise SensorContractError('complete zero-command initial gravity history required')
            mean=np.asarray(force['values']).mean(0); magnitude=float(np.linalg.norm(mean))
            if not 8<=magnitude<=12 or np.sqrt(np.mean(np.asarray(gyro['values'])**2))>.5:
                raise SensorContractError('initial gravity assumption inconsistent with body sensing')
            self._up_initial=mean/magnitude
            rotation=self.orientation.begin(packet,now_ns=now_ns)['rotation_initial_body_from_current_body']
            state=self._state_from(packet,rotation,now_ns)
            self._state=state; self.last_ns=now_ns; self.status='ACTIVE'; return self.snapshot(now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.status='FAILED_SENSOR'; raise SensorContractError('ground initialization failed') from error

    def step(self,packet,*,now_ns):
        if self.status!='ACTIVE': raise SensorContractError('active ground hypothesis required')
        try:
            rotation=self.orientation.step(packet,now_ns=now_ns)['rotation_initial_body_from_current_body']
            state=self._state_from(packet,rotation,now_ns)
            self._state=state; self.last_ns=now_ns; return self.snapshot(now_ns=now_ns)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self.status='FAILED_SENSOR'; raise SensorContractError('ground update failed') from error

    def snapshot(self,*,now_ns):
        import copy
        now_ns=_ns(now_ns,'ground query clock')
        if self.status!='ACTIVE' or now_ns!=self.last_ns: raise SensorContractError('fresh active ground hypothesis required')
        return copy.deepcopy(self._state)

    def project_pixel(self,u,v,*,now_ns):
        state=self.snapshot(now_ns=now_ns)
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in (u,v)) or not (0<=u<640 and 0<=v<480):
            raise SensorContractError('native finite image coordinate required')
        focal=320/np.tan(np.deg2rad(78.323)/2)
        ray=np.array([1.,-(u+.5-320)/focal,-(v+.5-240)/focal])
        origin=np.array([.326,0,.043]); up=np.asarray(state['up_current_body']); denominator=float(up@ray)
        if denominator>=-1e-8: return None
        depth=-(state['body_origin_height_m']+up@origin)/denominator
        return (origin+depth*ray).tolist() if .05<=depth<=200 else None
