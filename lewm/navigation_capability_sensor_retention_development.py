"""Hash consumed RGB-D packets; release only recording references.

Controllers continue to own their live packets/history. No packet is mutated.
Use full native RGB-D retention where pilot replay is unqualified.
"""
import hashlib
import numpy as np
from lewm.eligible_floor_registration_development import bind
from scripts.in_memory_paired_camera_session_development import InMemoryPairedCameraSession, write
from scripts.raw_depth_archive_development import packet_digest
from scripts.live_depth_noise_session_development import persist_pair as persist_full_pair, write_metadata as write_full_metadata


def array_identity(array):
    value=np.asarray(array)
    return dict(sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                shape=list(value.shape),dtype=value.dtype.str)


def record_consumed(row,packets):
    primary,depth,fast,auxiliary_depth,auxiliary_rgb,stamp=packets
    rgb=primary['image']['rgb']
    # The auxiliary packet's RGB payload is bound by its complete typed digest.
    # Raw per-camera RGB identities additionally preserve direct image checks.
    identities={label:dict(rgb=array_identity(image),native_depth=array_identity(native))
        for label,(image,native) in zip(('primary','auxiliary'),row['images'],strict=True)}
    assert identities['primary']['rgb']==array_identity(rgb)
    assert packet_digest(depth)==row['live_depth_noise']['primary_sha256']
    assert packet_digest(auxiliary_depth)==row['live_depth_noise']['auxiliary_sha256']
    return dict(frame=row['frame'],measured_ns=int(stamp),
        physical_sample_index=row['physical_sample_index'],transforms=row['transforms'],
        acquisition_wall_ms=row['acquisition_wall_ms'],pixel_sha256={label:dict(
            rgb_sha256=value['rgb']['sha256'],native_depth_sha256=value['native_depth']['sha256'],
            derived_packet_sha256=row['depth' if label=='primary' else 'auxiliary_depth'])
            for label,value in identities.items()},
        arrays=identities,live_depth_noise=row['live_depth_noise'],
        consumed_packet_sha256=dict(primary_rgb=packet_digest(primary['image']),
            primary_depth=packet_digest(depth),auxiliary_rgb=packet_digest(auxiliary_rgb),
            auxiliary_depth=packet_digest(auxiliary_depth)),
        recording='hashes_only',frames_retained=False)


def retained_metadata(row,directory,**kwargs):
    return row['consumed_hash_record']


def write_metadata(path,value):
    if path.name=='in_memory_camera_observations.json':
        value=dict(value,schema='navigation_capability_consumed_sensor_hashes.v1',
            native_depth_arrays_saved=False,live_noisy_depth_arrays_saved=False,
            rgb_arrays_saved=False,direct_depth_replay_available=False,
            requires_verified_command_replay=True,per_decision_snapshots_retained=False)
    write(path,value)


class SensorHashRetentionMixin:
    def sensor_packets(self):
        packets=super().sensor_packets()
        row=self.captured_pairs[-1]
        row['consumed_hash_record']=record_consumed(row,packets)
        row['images']=[]
        return packets

    persist_observations=bind(InMemoryPairedCameraSession.persist_observations,
        persist_camera_pair=retained_metadata,write=write_metadata)


def full_metadata(row,directory,**kwargs):
    retained=persist_full_pair(row,directory,**kwargs)
    return retained | dict(consumed_packet_sha256=row['consumed_hash_record']['consumed_packet_sha256'],
        arrays=row['consumed_hash_record']['arrays'],recording='full_frames',frames_retained=True,
        reason='Pilot sensor replay not qualified; native depth plus exact noise recipe and hashes retained')


class FullSensorRetentionMixin:
    def sensor_packets(self):
        packets=super().sensor_packets()
        row=self.captured_pairs[-1]
        row['consumed_hash_record']=record_consumed(row,packets)
        return packets

    persist_observations=bind(InMemoryPairedCameraSession.persist_observations,
        persist_camera_pair=full_metadata,write=write_full_metadata)
