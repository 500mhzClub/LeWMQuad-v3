"""Ideal simulated body-origin sensing and a narrow causal policy boundary.

This is explicitly a simulator sensor model, not hardware calibration. Specific
force uses a causal backward velocity difference at 50 Hz, expressed in the
current body frame. No IMU lever arm, bias, noise or transport delay is modeled.
"""
import numpy as np
from numbers import Integral

from lewm.causal_sensor_state import CausalSensorBuffer, SensorSchema, SensorContractError, build_decision_packet
from lewm.physical_execution_development import rotation_xyzw

JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for joint in ('hip','thigh','calf') for leg in ('FL','FR','RL','RR'))
CALIBRATION = 'ideal-body-origin-50hz-v1'
CAMERA_CALIBRATION = 'go2-fixed-native-rgb-640x480-development-v1'
SCHEMAS = (
    SensorSchema('gyro',('wx','wy','wz'),('rad/s',)*3,20,400_000_000,CALIBRATION),
    SensorSchema('specific_force',('fx','fy','fz'),('m/s^2',)*3,20,400_000_000,CALIBRATION),
    SensorSchema('joints',tuple('q:'+n for n in JOINT_NAMES)+tuple('dq:'+n for n in JOINT_NAMES),
                 ('rad',)*12+('rad/s',)*12,20,400_000_000,CALIBRATION),
    SensorSchema('applied_command',('vx','vy','wz'),('m/s','m/s','rad/s'),15,1_500_000_000,
                 'applied-command-clock-v1',role='control'),
)


class IdealBodySensor:
    def __init__(self):
        self.previous_time = None
        self.previous_velocity = None

    def sample(self, *, measured_ns, quaternion_xyzw, velocity_world, angular_velocity_world,
               joint_position, joint_velocity, joint_names):
        if isinstance(measured_ns,bool) or not isinstance(measured_ns,int) or measured_ns<0:
            raise SensorContractError('invalid ideal sensor time')
        values=[np.asarray(value,dtype=float) for value in
                (velocity_world,angular_velocity_world,joint_position,joint_velocity)]
        if any(value.shape!=shape or not np.isfinite(value).all() for value,shape in
               zip(values,((3,),(3,),(12,),(12,)),strict=True)):
            raise SensorContractError('invalid simulator reference measurement')
        if tuple(joint_names)!=JOINT_NAMES:
            raise SensorContractError('joint identity mismatch; resolve names before sensor conversion')
        if self.previous_time is not None and measured_ns-self.previous_time!=20_000_000:
            raise SensorContractError('ideal sensor requires exact causal 20ms cadence')
        rotation=rotation_xyzw(quaternion_xyzw)
        velocity,angular,q,dq=values
        acceleration_valid=self.previous_time is not None
        force=rotation.T@((velocity-self.previous_velocity)/.02-np.array([0.,0.,-9.81])) if acceleration_valid else np.zeros(3)
        self.previous_time,self.previous_velocity=measured_ns,velocity.copy()
        return {'gyro':(rotation.T@angular,np.ones(3,dtype=bool)),
                'specific_force':(force,np.full(3,acceleration_valid,dtype=bool)),
                'joints':(np.concatenate([q,dq]),np.ones(24,dtype=bool))}


class BodyObservationBuffer:
    def __init__(self,identity):
        self.identity=identity
        self.buffer=CausalSensorBuffer(SCHEMAS)
        self.buffer.begin_episode(identity)

    def append_sensors(self,values,measured_ns):
        if set(values)!={'gyro','specific_force','joints'}:
            raise SensorContractError('unexpected sensed channel')
        for name,(value,valid) in values.items():
            self.buffer.append(name,value,valid,measured_ns=measured_ns,available_ns=measured_ns,
                identity=self.identity,calibration_id=CALIBRATION)

    def append_applied_command(self,command,measured_ns):
        self.buffer.append('applied_command',command,np.ones(3,dtype=bool),measured_ns=measured_ns,
            available_ns=measured_ns,identity=self.identity,calibration_id='applied-command-clock-v1')

    def packet(self,rgb,measured_ns):
        packet=build_decision_packet(self.buffer,rgb,image_ns=measured_ns,image_available_ns=measured_ns,
            decision_ns=measured_ns,identity=self.identity,camera_calibration_id=CAMERA_CALIBRATION,
            expected_calibration_id=CAMERA_CALIBRATION,expected_rgb_shape=(480,640,3),max_image_age_ns=0)
        validate_policy_packet(packet)
        return packet


def validate_policy_packet(packet):
    """Reject extra fields/modalities rather than merely ignoring privileged ones."""
    if set(packet)!={'image','sensor_state'} or set(packet['image'])!={'rgb','measured_ns','available_ns','calibration_id'}:
        raise SensorContractError('unexpected policy packet/image fields')
    state=packet['sensor_state']
    if set(state)!={'identity','image_ns','decision_ns','sensor_anchor','sensor_anchor_ns','sensed','control'}:
        raise SensorContractError('unexpected sensor-state fields')
    if set(state['sensed'])!={'gyro','specific_force','joints'} or set(state['control'])!={'applied_command'}:
        raise SensorContractError('undeclared policy modality')
    times=[packet['image'][key] for key in ('measured_ns','available_ns')]+[state[key] for key in ('image_ns','decision_ns','sensor_anchor_ns')]
    if any(isinstance(t,bool) or not isinstance(t,Integral) or t<0 for t in times):
        raise SensorContractError('policy clock must use nonnegative integer nanoseconds')
    for schema in SCHEMAS:
        row=state[schema.role][schema.name]
        if set(row)!={'values','valid','measured_ns','available_ns','channels','units','calibration_id'}:
            raise SensorContractError('unexpected sensor-history fields')
        if tuple(row['channels'])!=schema.channels or tuple(row['units'])!=schema.units or row['calibration_id']!=schema.calibration_id:
            raise SensorContractError('policy channel identity changed')
        shape=(schema.history_length,len(schema.channels))
        data,valid=np.asarray(row['values']),np.asarray(row['valid'])
        measured,available=np.asarray(row['measured_ns']),np.asarray(row['available_ns'])
        if data.shape!=shape or valid.shape!=shape or valid.dtype!=bool or not np.isfinite(data[valid]).all() or np.any(data[~valid]!=0):
            raise SensorContractError('invalid policy history values')
        if measured.shape!=(shape[0],) or available.shape!=measured.shape or measured.dtype.kind not in 'iu' or available.dtype.kind not in 'iu':
            raise SensorContractError('invalid policy history times')
        present=measured>=0
        if (np.any(measured[~present]!=-1) or np.any(available[~present]!=-1)
                or np.any(np.diff(measured[present])<=0) or np.any(valid[~present]) or np.any(available[present]<measured[present])
                or np.any(available[present]>state['decision_ns']) or np.any(measured[present]>state['sensor_anchor_ns'])
                or np.any(state['decision_ns']-measured[present]>schema.max_age_ns)):
            raise SensorContractError('noncausal or stale policy history')
    rgb=np.asarray(packet['image']['rgb'])
    if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or packet['image']['calibration_id']!=CAMERA_CALIBRATION:
        raise SensorContractError('policy RGB identity mismatch')
    if not packet['image']['measured_ns']<=packet['image']['available_ns']<=state['decision_ns']:
        raise SensorContractError('noncausal policy image')
    if (state['image_ns']!=packet['image']['measured_ns'] or state['sensor_anchor']!='decision'
            or state['sensor_anchor_ns']!=state['decision_ns']):
        raise SensorContractError('policy time anchor mismatch')
