"""Recording may release references, never alter live controller inputs."""
import copy
import numpy as np
import pytest
from lewm.navigation_capability_sensor_retention_development import SensorHashRetentionMixin, FullSensorRetentionMixin
from scripts.raw_depth_archive_development import packet_digest


@pytest.mark.parametrize('mixin,cleared', [(SensorHashRetentionMixin,True),(FullSensorRetentionMixin,False)])
def test_hash_retention_preserves_live_packets_and_detects_pixel_changes(mixin,cleared):
    rgb=np.arange(36,dtype=np.uint8).reshape(3,4,3)
    native=np.arange(12,dtype=np.float32).reshape(3,4)
    primary={'image':{'rgb':rgb,'measured_ns':100}}
    depth={'depth_m':native.copy(),'valid':np.ones((3,4),bool)}
    aux_depth=copy.deepcopy(depth);aux_rgb={'rgb':rgb.copy()}
    packets=(primary,depth,{},aux_depth,aux_rgb,100)
    row=dict(frame=0,measured_ns=100,physical_sample_index=10,transforms=[],
        acquisition_wall_ms=1.,images=[(rgb,native),(aux_rgb['rgb'],native.copy())],
        depth='a'*64,auxiliary_depth='b'*64,
        live_depth_noise=dict(primary_sha256=packet_digest(depth),auxiliary_sha256=packet_digest(aux_depth)))
    class Source:
        def sensor_packets(self):return packets
    class HashSession(mixin,Source):pass
    session=HashSession();session.captured_pairs=[row]
    before=packet_digest(packets)
    result=session.sensor_packets()
    assert result is packets
    assert before==packet_digest(result)
    assert (len(row['images'])==0)==cleared
    identity=copy.deepcopy(row['consumed_hash_record'])
    rgb[0,0,0]^=1
    assert packet_digest(primary['image'])!=identity['consumed_packet_sha256']['primary_rgb']
    depth['depth_m'][0,0]+=1
    assert packet_digest(depth)!=identity['consumed_packet_sha256']['primary_depth']
    assert identity==row['consumed_hash_record']
