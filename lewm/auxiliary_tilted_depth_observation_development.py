"""Separate causal auxiliary depth; no native pose or segmentation input."""
import numpy as np
from lewm.causal_sensor_state import SensorContractError,_ns,_identity
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm.causal_depth_observation_development import rgb_digest,FOCAL,INTRINSICS
from lewm.auxiliary_tilted_depth_geometry_development import body_from_optical

SCHEMA='causal_auxiliary_tilted_optical_depth_development.v1'
CALIBRATION_ID='go2_auxiliary_640x480_pitch30_mount035_000_008_depth02to5_v1'
FIELDS={'schema','calibration_id','identity','measured_ns','available_ns','decision_ns',
    'primary_rgb_sha256','depth_m','valid','representation','hardware_calibrated'}


def calibration_metadata():
    return dict(calibration_id=CALIBRATION_ID,resolution_wh=[640,480],intrinsics=[list(r) for r in INTRINSICS],
        body_from_optical=body_from_optical().tolist(),pixel_centres='column+0.5,row+0.5',
        representation='optical_axis_depth_m',minimum_depth_m=.2,maximum_depth_m=5.,
        hardware_calibrated=False,assumption='ideal_auxiliary_simulated_depth_zero_latency')


def from_native_depth(native_depth,policy,*,measured_ns,available_ns,now_ns):
    native=np.asarray(native_depth)
    if native.shape!=(480,640) or native.dtype!=np.float32:
        raise SensorContractError('auxiliary native optical metres require float32 HxW')
    valid=np.isfinite(native)&(native>=.2)&(native<=5.)
    result=dict(schema=SCHEMA,calibration_id=CALIBRATION_ID,identity=policy['sensor_state']['identity'],
        measured_ns=measured_ns,available_ns=available_ns,decision_ns=now_ns,
        primary_rgb_sha256=rgb_digest(policy),depth_m=np.where(valid,native,np.float32(0.)).copy(),
        valid=valid.copy(),representation='optical_axis_depth_m',hardware_calibrated=False)
    validate_depth(result,policy,now_ns=now_ns)
    return result


def validate_depth(depth,policy,*,now_ns):
    validate_policy_packet(policy);now=_ns(now_ns,'auxiliary decision')
    if not isinstance(depth,dict) or set(depth)!=FIELDS:
        raise SensorContractError('exact auxiliary depth fields required')
    if (depth['schema']!=SCHEMA or depth['calibration_id']!=CALIBRATION_ID
            or depth['representation']!='optical_axis_depth_m' or depth['hardware_calibrated'] is not False):
        raise SensorContractError('explicit auxiliary optical-depth calibration required')
    measured=_ns(depth['measured_ns'],'auxiliary measured');available=_ns(depth['available_ns'],'auxiliary available')
    if (_ns(depth['decision_ns'],'auxiliary decision')!=now or policy['sensor_state']['decision_ns']!=now
            or measured!=policy['image']['measured_ns'] or not measured<=available<=now or now-measured>100_000_000):
        raise SensorContractError('causal current primary/auxiliary acquisition clock required')
    if _identity(depth['identity'])!=policy['sensor_state']['identity'] or depth['primary_rgb_sha256']!=rgb_digest(policy):
        raise SensorContractError('auxiliary episode or primary frame binding mismatch')
    values,valid=depth['depth_m'],depth['valid']
    if (not isinstance(values,np.ndarray) or not isinstance(valid,np.ndarray) or values.shape!=(480,640)
            or values.dtype!=np.float32 or valid.shape!=values.shape or valid.dtype!=bool or not np.isfinite(values).all()
            or np.any(values[~valid]!=0.) or np.any(values[valid]<.2) or np.any(values[valid]>5.)):
        raise SensorContractError('bounded auxiliary metric depth and explicit unknown-ray mask required')


def body_points(depth,policy,*,now_ns,stride=4):
    validate_depth(depth,policy,now_ns=now_ns)
    if type(stride) is not int or stride<4 or 480%stride or 640%stride:
        raise ValueError('bounded auxiliary grid divisor of at least four required')
    rows=np.arange(stride//2,480,stride);columns=np.arange(stride//2,640,stride)
    u,v=np.meshgrid(columns+.5,rows+.5);z=depth['depth_m'][np.ix_(rows,columns)]
    valid=depth['valid'][np.ix_(rows,columns)];optical=np.stack((z*(u-320)/FOCAL,z*(v-240)/FOCAL,z),axis=-1)
    T=body_from_optical();points=optical@T[:3,:3].T+T[:3,3]
    return dict(points_body_m=np.where(valid[...,None],points,np.nan),valid=valid.copy(),rows=rows,columns=columns,
        measured_ns=depth['measured_ns'],calibration_id=CALIBRATION_ID,environment_clearance_qualified=False,
        scope='observed auxiliary surfaces only; missing rays remain unknown')
