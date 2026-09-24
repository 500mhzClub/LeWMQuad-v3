"""Explicit cached sensor replay; no native state, discovery or command future.

Manifest/tensor checks mirror the existing whole-task RGB-D and fast readers.
An offline caller must bind immutable input bytes before and after replay.
Returned packet arrays are copies, so a consumer cannot corrupt cached history.
"""
import json
from pathlib import Path
from numbers import Integral
import numpy as np
from PIL import Image
from lewm.causal_rgb_dataset_development import _leaf,_protected,schema_metadata
from lewm.causal_sensor_state import SensorContractError
from lewm.simulated_body_observation_development import SCHEMAS,CAMERA_CALIBRATION,validate_policy_packet
from lewm.causal_depth_observation_development import SCHEMA,calibration_metadata,validate_depth
from lewm.fast_gyro_development import SCHEMA_ID,SCHEMA as FAST_SCHEMA,CALIBRATION,validate_fast_packet


MAX_FRAMES=3600+1+10  # initial observation, maximum command ticks, terminal drain


def validate_frame_population(frames):
    if not isinstance(frames,list) or not 1<=len(frames)<=MAX_FRAMES:
        raise SensorContractError("bounded actual frame population required")


class IntentReturnRGBDReplay:
    def __init__(self,directory):
        directory=Path(directory).absolute()
        if _protected(directory) or _protected(directory.resolve()):raise SensorContractError('protected replay input forbidden')
        self.directory=directory.resolve()
        def read(name):return json.loads(_leaf(self.directory,name).read_text())
        manifest=read('policy_observations.json')
        if (set(manifest)!={'schema','camera_calibration_id','sensor_assumption','history_file','frames','sensor_schemas'}
                or manifest['schema']!='causal_rgb_body_routes_development.v1'
                or manifest['camera_calibration_id']!=CAMERA_CALIBRATION
                or manifest['sensor_assumption']!='ideal_simulated_body_origin_50hz_zero_latency'
                or manifest['sensor_schemas']!=schema_metadata() or manifest['history_file']!='policy_histories.npz'):
            raise SensorContractError('exact route sensor manifest required')
        self.frames=manifest['frames']
        validate_frame_population(self.frames)
        for i,f in enumerate(self.frames):
            if set(f)!={'rgb_file','image_ns','decision_ns'} or f['rgb_file']!=f'rgb_{i:04d}.png':raise SensorContractError('exact RGB frame path required')
        expected={'image_ns','decision_ns'}|{f'{s.name}_{f}' for s in SCHEMAS for f in ('values','valid','measured_ns','available_ns')}
        self.body=self._archive('policy_histories.npz',expected)
        dm=read('depth_observations.json')
        if (set(dm)!={'schema','calibration','frames'} or dm['schema']!=SCHEMA
                or dm['calibration']!=calibration_metadata() or not isinstance(dm['frames'],list)
                or len(dm['frames'])!=len(self.frames)):raise SensorContractError('exact paired depth manifest required')
        self.depth=dm['frames']
        for i,f in enumerate(self.depth):
            if (set(f)!={'depth_file','schema','calibration_id','identity','measured_ns','available_ns','decision_ns',
                         'rgb_sha256','representation','hardware_calibrated'} or f['depth_file']!=f'depth_{i:04d}.npz'):
                raise SensorContractError('exact depth frame fields/path required')
        self.fast=self._archive('fast_gyro_histories.npz',{'values','valid','measured_ns','available_ns'})
        if any(v.shape[:1]!=(len(self.frames),) for v in (*self.body.values(),*self.fast.values())):
            raise SensorContractError('complete body and fast frame populations required')

    def _archive(self,name,fields):
        with np.load(_leaf(self.directory,name),allow_pickle=False) as a:
            if set(a.files)!=fields:raise SensorContractError('exact replay tensor fields required')
            return {k:a[k] for k in a.files}

    def packet(self,index):
        if isinstance(index,bool) or not isinstance(index,Integral) or not 0<=index<len(self.frames):raise SensorContractError('actual frame index required')
        frame=self.frames[index];now=frame['decision_ns']
        if self.body['image_ns'][index]!=frame['image_ns'] or self.body['decision_ns'][index]!=now:raise SensorContractError('RGB/body clock mismatch')
        state=dict(identity=(0,0,0),image_ns=frame['image_ns'],decision_ns=now,sensor_anchor='decision',sensor_anchor_ns=now,sensed={},control={})
        for s in SCHEMAS:
            state[s.role][s.name]={**{f:self.body[f'{s.name}_{f}'][index].copy() for f in ('values','valid','measured_ns','available_ns')},
                                 'channels':s.channels,'units':s.units,'calibration_id':s.calibration_id}
        with Image.open(_leaf(self.directory,frame['rgb_file'])) as image:rgb=np.array(image)
        policy=dict(image=dict(rgb=rgb,measured_ns=frame['image_ns'],available_ns=frame['image_ns'],calibration_id=CAMERA_CALIBRATION),sensor_state=state)
        validate_policy_packet(policy)
        depth=dict(self.depth[index]);name=depth.pop('depth_file');depth['identity']=tuple(depth['identity'])
        depth.update(self._archive(name,{'depth_m','valid'}));validate_depth(depth,policy,now_ns=now)
        fast={k:v[index].copy() for k,v in self.fast.items()}|dict(schema=SCHEMA_ID,identity=(0,0,0),decision_ns=now,
                    calibration_id=CALIBRATION,channels=FAST_SCHEMA.channels,units=FAST_SCHEMA.units)
        validate_fast_packet(fast,policy,now_ns=now)
        return policy,depth,fast,now
