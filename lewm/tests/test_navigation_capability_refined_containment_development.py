import json
import numpy as np
import pytest
from lewm.navigation_capability_refined_containment_development import compare


def fixture(base, exposure=True):
    old=base/'runs/v0_task_c1_screen_C1_dev00_ep0_attempt001'
    new=base/'runs/corrected'
    frames=[dict(pixel_sha256=str(i),live_depth_noise={},consumed_packet_sha256=str(i),
        arrays={},measured_ns=i*10,physical_sample_index=i) for i in range(4)]
    for run in (old,new):
        (run/'native').mkdir(parents=True)
        (run/'episode.json').write_text(json.dumps({'episode_id':'00/0'}))
        (run/'requests.json').write_text(json.dumps([dict(simulator_ns=i*10,
            requested_command=[i,0,0],reason='original') for i in range(4)]))
        (run/'native/in_memory_camera_observations.json').write_text(json.dumps({'frames':frames}))
        np.savez(run/'native/physics_trace.npz',q=np.arange(4)[:,None])
    evidence=base/'grid_c3_bound_exposure/dev00/result.json';evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps(dict(status='PASS',native_arrays_exact_through_checked_prefix=True,
        consumed_sensor_hashes_bitwise=True,first_exposure={'frame':2} if exposure else None,
        full_source_checked=not exposure)))
    return new


def test_post_exposure_changes_allowed_but_reported(tmp_path):
    new=fixture(tmp_path)
    np.savez(new/'native/physics_trace.npz',q=np.array([0,1,2,99])[:,None])
    result=compare(tmp_path,new)
    assert result['passed'] and not result['entire_recording_identical']
    assert result['first_native_difference_sample_by_field']=={'q':3}


@pytest.mark.parametrize('kind',['native','requests','sensors'])
def test_pre_exposure_difference_is_a_stop(tmp_path,kind):
    new=fixture(tmp_path)
    if kind=='native':np.savez(new/'native/physics_trace.npz',q=np.array([0,99,2,3])[:,None])
    elif kind=='requests':
        p=new/'requests.json';data=json.loads(p.read_text());data[1]['reason']='different';p.write_text(json.dumps(data))
    else:
        p=new/'native/in_memory_camera_observations.json';data=json.loads(p.read_text())
        data['frames'][2]['consumed_packet_sha256']='different';p.write_text(json.dumps(data))
    assert not compare(tmp_path,new)['passed']


def test_without_exposure_whole_recording_must_match(tmp_path):
    new=fixture(tmp_path,False)
    assert compare(tmp_path,new)['entire_recording_identical']
    np.savez(new/'native/physics_trace.npz',q=np.array([0,1,2,99])[:,None])
    assert not compare(tmp_path,new)['passed']


def test_native_dtype_is_part_of_exact_reproduction(tmp_path):
    new=fixture(tmp_path)
    np.savez(new/'native/physics_trace.npz',q=np.arange(4,dtype=float)[:,None])
    r=compare(tmp_path,new)
    assert not r['passed'] and not r['entire_recording_identical']
