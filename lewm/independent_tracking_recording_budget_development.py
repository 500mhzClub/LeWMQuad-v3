"""Fixed numeric recording layouts and allocation ceilings, no native execution.

These are byte/shape contracts, not contact reachability, measurement validity,
physical coverage or peak-memory proofs. Partial recordings may lack fields;
the shape validator is for complete recorded prefixes, never a repair routine.
"""
import math
import numpy as np

from lewm.independent_tracking_challenge_development import MAX_FRAMES,MAX_PHYSICS_SAMPLES
from lewm.simulated_body_observation_development import SCHEMAS

MAX_CONTACT_ROWS_PER_SAMPLE=750
MEMBER_OVERHEAD=65536


def _count(value,maximum):
    if type(value) is not int or not 0<=value<=maximum:
        raise ValueError('bounded integer recording count required')
    return value


def numeric_layouts(physics_samples,frames,contact_rows):
    n=_count(physics_samples,MAX_PHYSICS_SAMPLES)
    f=_count(frames,MAX_FRAMES)
    c=_count(contact_rows,n*MAX_CONTACT_ROWS_PER_SAMPLE)
    if not n and f:raise ValueError('frames without physics cannot form a recorded prefix')
    def spec(shape,dtype):return dict(shape=tuple(shape),dtype=np.dtype(dtype).str)
    physics={k:spec((n,)+shape,'float64') for k,shape in (
        ('timestamp_s',()),('base_pose_world',(7,)),('base_twist_world',(6,)),
        ('joint_position',(12,)),('joint_velocity',(12,)),('requested_command',(3,)),
        ('post_slew_applied_command',(3,)),('applied_command',(3,)))}
    physics.update({k:spec((n,),'uint8') for k in ('physics_contact','phase','edge_index')})
    contacts={k:spec((c,),'int32') for k in ('geom_a','geom_b','link_a','link_b')}
    contacts.update({k:spec((c,3),'float32') for k in ('force_a','force_b','position')})
    contacts.update(valid_mask=spec((c,),'bool'),frame_offsets=spec((n+1,),'int64'),
                    frame_timestamp_s=spec((n,),'float64'))
    slow={'measured_ns':spec((n//10,),'int64')}
    body={k:spec((f,),'int64') for k in ('image_ns','decision_ns')}
    for schema in SCHEMAS:
        channels=len(schema.channels);history=schema.history_length
        for field,dtype in (('values','float64'),('valid','bool')):
            if schema.role=='sensed':slow[schema.name+'_'+field]=spec((n//10,channels),dtype)
            body[schema.name+'_'+field]=spec((f,history,channels),dtype)
        for field in ('measured_ns','available_ns'):
            body[schema.name+'_'+field]=spec((f,history),'int64')
    fast={k:spec((n,),'int64') for k in ('measured_ns','available_ns')}
    fast.update(values=spec((n,3),'float64'),valid=spec((n,3),'bool'))
    history={k:spec((f,51),'int64') for k in ('measured_ns','available_ns')}
    history.update(values=spec((f,51,3),'float64'),valid=spec((f,51,3),'bool'))
    return {'physics_trace.npz':physics if n else {},'native_contacts.npz':contacts if n else {},
        'ideal_sensor_samples.npz':slow if n>=10 else {},'policy_histories.npz':body if f else {},
        'fast_gyro_samples.npz':fast if n else {},'fast_gyro_histories.npz':history if f else {}}


def reservation_from_nbytes(sizes):
    """Same conservative per-member bound used by the exclusive NPZ writer."""
    sizes=tuple(sizes)
    if any(type(n) is not int or n<0 for n in sizes):raise ValueError('nonnegative integer byte counts')
    return sum(n+n//10+MEMBER_OVERHEAD for n in sizes)+MEMBER_OVERHEAD


def numeric_envelope():
    layouts=numeric_layouts(MAX_PHYSICS_SAMPLES,MAX_FRAMES,
        MAX_PHYSICS_SAMPLES*MAX_CONTACT_ROWS_PER_SAMPLE)
    result={}
    for name,members in layouts.items():
        sizes=[math.prod(r['shape'])*np.dtype(r['dtype']).itemsize for r in members.values()]
        result[name]=dict(members=len(members),raw_array_bytes=sum(sizes),
                         serialization_ceiling_bytes=reservation_from_nbytes(sizes))
    return result


def validate_numeric_archive(name,arrays,*,physics_samples,frames,contact_rows):
    expected=numeric_layouts(physics_samples,frames,contact_rows)
    if name not in expected or not isinstance(arrays,dict) or set(arrays)!=set(expected[name]):
        raise ValueError('exact numeric archive members required')
    for key,spec in expected[name].items():
        value=arrays[key]
        if not isinstance(value,np.ndarray) or value.shape!=spec['shape'] or value.dtype.str!=spec['dtype']:
            raise ValueError('recorded array shape or dtype differs: '+name+':'+key)
    # This deliberately does not certify clocks, contacts, values or trajectories.
    return dict(numeric_layout_verified=True,measurement_validity_verified=False,
                native_recording_performed=False,navigation_qualified=False)
