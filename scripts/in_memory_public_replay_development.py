"""Reconstruct public packets from saved paired-camera acquisitions; no native pose."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from lewm.causal_rgb_dataset_development import _protected
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.simulated_body_observation_development import SCHEMAS,CAMERA_CALIBRATION,validate_policy_packet
from lewm.causal_depth_observation_development import from_native_depth
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_depth
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.fast_gyro_development import SCHEMA_ID,SCHEMA as FAST_SCHEMA,CALIBRATION,validate_fast_packet
from scripts.raw_depth_archive_development import SCHEMA as RAW_DEPTH_SCHEMA, packet_digest


class PublicReplay:
    _archive=IntentReturnRGBDReplay._archive

    def __init__(self,directory):
        directory=Path(directory).absolute()
        if _protected(directory) or _protected(directory.resolve()):raise ValueError('protected input forbidden')
        self.directory=directory.resolve()
        retention = self.directory.parent/'depth_retention.json'
        if _protected(retention.resolve()): raise ValueError('protected retention record forbidden')
        if retention.exists() and json.loads(retention.read_text()).get('full_sensor_replay_available') is False:
            raise ValueError('Depth recording intentionally retired under the development retention policy; '
                'results and failure records remain. Use a retained recording or run a new experiment.')
        fields={'image_ns','decision_ns'}|{f'{s.name}_{k}' for s in SCHEMAS for k in ('values','valid','measured_ns','available_ns')}
        self.body=self._archive('policy_histories.npz',fields)
        self.fast=self._archive('fast_gyro_histories.npz',{'values','valid','measured_ns','available_ns'})
        self.depth_witnesses = None
        metadata_path = self.directory/'in_memory_camera_observations.json'
        if _protected(metadata_path.resolve()): raise ValueError('protected metadata forbidden')
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata['schema'] == RAW_DEPTH_SCHEMA:
                rows = metadata['frames']
                if [r['frame'] for r in rows] != list(range(len(self.body['decision_ns']))):
                    raise ValueError('complete ordered raw-depth frame witnesses required')
                self.depth_witnesses = [r['pixel_sha256'] for r in rows]
            elif metadata['schema'] != 'in_memory_paired_camera_development.v1':
                raise ValueError('recognized paired-camera archive schema required')

    def policy_packet(self,frame):
        now=int(self.body['decision_ns'][frame])
        if now!=1_500_000_000+frame*100_000_000 or self.body['image_ns'][frame]!=now:
            raise ValueError('actual consecutive paired-camera boundary required')
        state=dict(identity=(0,0,0),image_ns=now,decision_ns=now,sensor_anchor='decision',sensor_anchor_ns=now,sensed={},control={})
        for s in SCHEMAS:
            state[s.role][s.name]={**{k:self.body[f'{s.name}_{k}'][frame].copy() for k in ('values','valid','measured_ns','available_ns')},
                'channels':s.channels,'units':s.units,'calibration_id':s.calibration_id}
        with Image.open(self.directory/f'rgb_{frame:04d}.png') as image:rgb=np.array(image)
        p=dict(image=dict(rgb=rgb,measured_ns=now,available_ns=now,calibration_id=CAMERA_CALIBRATION),sensor_state=state)
        validate_policy_packet(p)
        return p

    def packet(self,frame):
        p=self.policy_packet(frame);now=p['sensor_state']['decision_ns']
        fields = {'native_optical_depth_m'} if self.depth_witnesses is not None else {'native_optical_depth_m','depth_m','valid'}
        d=self._archive(f'primary_depth_{frame:04d}.npz',fields)
        a=self._archive(f'auxiliary_depth_{frame:04d}.npz',fields)
        primary=from_native_depth(d['native_optical_depth_m'],p,measured_ns=now,available_ns=now,now_ns=now)
        auxiliary=auxiliary_depth(a['native_optical_depth_m'],p,measured_ns=now,available_ns=now,now_ns=now)
        if self.depth_witnesses is None and any(not np.array_equal(saved[k],packet[k]) for saved,packet in ((d,primary),(a,auxiliary)) for k in ('depth_m','valid')):
            raise ValueError('reconstructed depth differs from saved packet')
        with Image.open(self.directory/f'auxiliary_rgb_{frame:04d}.png') as image:rgb=np.array(image)
        if self.depth_witnesses is not None:
            for label, saved, packet, pixels in (('primary',d,primary,p['image']['rgb']),('auxiliary',a,auxiliary,rgb)):
                witness = self.depth_witnesses[frame][label]
                if (hashlib.sha256(saved['native_optical_depth_m'].tobytes()).hexdigest()!=witness['native_depth_sha256']
                        or hashlib.sha256(pixels.tobytes()).hexdigest()!=witness['rgb_sha256']
                        or packet_digest(packet)!=witness['derived_packet_sha256']):
                    raise ValueError('raw pixels or reconstructed public depth packet differ from capture')
        rgb=from_captured_rgb(rgb,auxiliary,p,measured_ns=now,available_ns=now,now_ns=now)
        fast={k:v[frame].copy() for k,v in self.fast.items()}|dict(schema=SCHEMA_ID,identity=(0,0,0),decision_ns=now,
            calibration_id=CALIBRATION,channels=FAST_SCHEMA.channels,units=FAST_SCHEMA.units)
        validate_fast_packet(fast,p,now_ns=now)
        return p,primary,fast,rgb,auxiliary,now
