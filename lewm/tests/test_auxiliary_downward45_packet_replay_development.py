import hashlib
import numpy as np
import pytest
from lewm.tests.test_causal_depth_observation_development import frame
from lewm.auxiliary_downward45_depth_geometry_development import CALIBRATION_ID
from scripts.auxiliary_downward45_packet_replay_development import packet,public_acquisition


def test_only_public_arrays_are_read_and_actual_capture_identity_is_required(tmp_path):
    policy,_,now=frame();native=np.full((480,640),2.,np.float32)
    np.savez(tmp_path/'auxiliary_depth_0000.npz',native_optical_depth_m=native,depth_m=native,
        valid=np.ones((480,640),bool),diagnostic_segmentation=np.array([{'privileged':'not a public array'}],dtype=object))
    row=dict(frame=0,measured_ns=now,native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        calibration_id=CALIBRATION_ID,world_from_optical='not public',diagnostic_segmentation='not public')
    public=public_acquisition(row);result=packet(tmp_path,0,policy,public,now_ns=now)
    assert 'world_from_optical' not in result and 'diagnostic_segmentation' not in result
    with pytest.raises(ValueError):packet(tmp_path,0,policy,row,now_ns=now)
    for key,value in (('frame',1),('measured_ns',now-100_000_000),('native_depth_sha256','0'*64)):
        with pytest.raises(ValueError):packet(tmp_path,0,policy,public|{key:value},now_ns=now)

