"""Policy-only reader for explicitly selected causal capture V1 trials.

This reader never discovers trials or opens launch, physics, contact, geometry,
actuator, command-future, result or camera-world-transform artifacts.
"""
import json
from numbers import Integral
from pathlib import Path
import re

import numpy as np
from PIL import Image

from lewm.causal_sensor_state import SensorContractError
from lewm.simulated_body_observation_development import SCHEMAS,CAMERA_CALIBRATION,validate_policy_packet


def _protected(path):
    return any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in path.parts)


def _leaf(directory,name):
    if Path(name).name!=name or name in ('.','..') or _protected(Path(name)):
        raise SensorContractError('invalid policy artifact path')
    candidate=(directory/name).resolve()
    if _protected(candidate) or candidate.parent!=directory:
        raise SensorContractError('policy artifact escapes explicit trial directory')
    return candidate


def schema_metadata():
    return [{'name':s.name,'channels':list(s.channels),'units':list(s.units),'role':s.role,
        'calibration_id':s.calibration_id,'history_length':s.history_length,'max_age_ns':s.max_age_ns} for s in SCHEMAS]


def load_policy_observation(directory,index):
    directory=Path(directory).absolute()
    if _protected(directory):
        raise SensorContractError('protected policy input forbidden')
    directory=directory.resolve()
    if _protected(directory):
        raise SensorContractError('protected policy input forbidden')
    manifest=json.loads(_leaf(directory,'policy_observations.json').read_text())
    if set(manifest)!={'schema','camera_calibration_id','sensor_assumption','history_file','frames','sensor_schemas'}:
        raise SensorContractError('unexpected observation manifest fields')
    if (manifest['schema']!='causal_rgb_body_development.v1' or manifest['camera_calibration_id']!=CAMERA_CALIBRATION
            or manifest['sensor_assumption']!='ideal_simulated_body_origin_50hz_zero_latency'
            or manifest['sensor_schemas']!=schema_metadata() or manifest['history_file']!='policy_histories.npz'):
        raise SensorContractError('observation manifest identity mismatch')
    frames=manifest['frames']
    if not isinstance(frames,list) or not 1<=len(frames)<=46:
        raise SensorContractError('invalid V1 frame population')
    if isinstance(index,bool) or not isinstance(index,Integral) or not 0<=index<len(frames):
        raise SensorContractError('invalid observation index')
    for i,frame in enumerate(frames):
        if set(frame)!={'rgb_file','image_ns','decision_ns'} or frame['rgb_file']!=f'rgb_{i:04d}.png':
            raise SensorContractError('unexpected RGB record/path')
        if not re.fullmatch(r'rgb_[0-9]{4}\.png',frame['rgb_file']):
            raise SensorContractError('invalid RGB path')
    expected={'image_ns','decision_ns'} | {f'{s.name}_{f}' for s in SCHEMAS for f in ('values','valid','measured_ns','available_ns')}
    with np.load(_leaf(directory,manifest['history_file']),allow_pickle=False) as archive:
        if set(archive.files)!=expected:
            raise SensorContractError('unexpected policy tensor fields')
        arrays={key:archive[key] for key in archive.files}
    if any(value.shape[:1]!=(len(frames),) for value in arrays.values()):
        raise SensorContractError('incomplete policy history population')
    frame=frames[index]
    if arrays['image_ns'][index]!=frame['image_ns'] or arrays['decision_ns'][index]!=frame['decision_ns']:
        raise SensorContractError('image/history time mismatch')
    state={'identity':(0,0,0),'image_ns':frame['image_ns'],'decision_ns':frame['decision_ns'],
        'sensor_anchor':'decision','sensor_anchor_ns':frame['decision_ns'],'sensed':{},'control':{}}
    for schema in SCHEMAS:
        state[schema.role][schema.name]={**{field:arrays[f'{schema.name}_{field}'][index] for field in
            ('values','valid','measured_ns','available_ns')},'channels':schema.channels,'units':schema.units,
            'calibration_id':schema.calibration_id}
    with Image.open(_leaf(directory,frame['rgb_file'])) as image:
        rgb=np.array(image)
    packet={'image':{'rgb':rgb,'measured_ns':frame['image_ns'],'available_ns':frame['image_ns'],
        'calibration_id':CAMERA_CALIBRATION},'sensor_state':state}
    validate_policy_packet(packet)
    return packet
