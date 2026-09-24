"""Policy-only continuation reader with the already specified806-frame budget."""
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image

from lewm.causal_rgb_dataset_development import _leaf,_protected,schema_metadata
from lewm.causal_sensor_state import SensorContractError
from lewm.simulated_body_observation_development import SCHEMAS,CAMERA_CALIBRATION,validate_policy_packet


def load_continuation_observation(directory,index):
    directory=Path(directory).absolute()
    if _protected(directory) or _protected(directory.resolve()):
        raise SensorContractError('protected policy input forbidden')
    directory=directory.resolve()
    manifest=json.loads(_leaf(directory,'policy_observations.json').read_text())
    if set(manifest)!={'schema','camera_calibration_id','sensor_assumption','history_file','frames','sensor_schemas'}:
        raise SensorContractError('unexpected route observation fields')
    if (manifest['schema']!='causal_rgb_body_routes_development.v1' or manifest['camera_calibration_id']!=CAMERA_CALIBRATION
            or manifest['sensor_assumption']!='ideal_simulated_body_origin_50hz_zero_latency'
            or manifest['sensor_schemas']!=schema_metadata() or manifest['history_file']!='policy_histories.npz'):
        raise SensorContractError('route observation identity mismatch')
    frames=manifest['frames']
    if not isinstance(frames,list) or not 1<=len(frames)<=806:
        raise SensorContractError('invalid route observation population')
    if isinstance(index,bool) or not isinstance(index,Integral) or not 0<=index<len(frames):
        raise SensorContractError('invalid route observation index')
    for i,frame in enumerate(frames):
        if set(frame)!={'rgb_file','image_ns','decision_ns'} or frame['rgb_file']!=f'rgb_{i:04d}.png':
            raise SensorContractError('unexpected route RGB fields/path')
    expected={'image_ns','decision_ns'} | {f'{s.name}_{f}' for s in SCHEMAS for f in ('values','valid','measured_ns','available_ns')}
    with np.load(_leaf(directory,'policy_histories.npz'),allow_pickle=False) as archive:
        if set(archive.files)!=expected:
            raise SensorContractError('unexpected route policy tensors')
        arrays={key:archive[key] for key in archive.files}
    if any(v.shape[:1]!=(len(frames),) for v in arrays.values()):
        raise SensorContractError('incomplete route observation population')
    frame=frames[index]
    if arrays['image_ns'][index]!=frame['image_ns'] or arrays['decision_ns'][index]!=frame['decision_ns']:
        raise SensorContractError('route image/history time mismatch')
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

