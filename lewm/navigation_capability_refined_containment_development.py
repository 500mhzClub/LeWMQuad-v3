"""Exact source-prefix containment, bounded by verified first legacy exposure."""
import json
import numpy as np
from lewm.navigation_capability_startup_containment_development import UNAFFECTED_STARTUP_IDS


def first_difference(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b)) if len(a)!=len(b) else None


def compare(base,new_run):
    episode=json.loads((new_run/'episode.json').read_text())['episode_id']
    if episode not in UNAFFECTED_STARTUP_IDS:return dict(episode=episode,required=False,passed=True,reason='Original pre-planning failure')
    maze=int(episode.split('/')[0]);old=base/f'runs/v0_task_c1_screen_C1_dev{maze:02d}_ep0_attempt001'
    evidence=base/f'grid_c3_bound_exposure/dev{maze:02d}/result.json'
    bound=json.loads(evidence.read_text())
    assert bound['status']=='PASS' and bound['native_arrays_exact_through_checked_prefix'] and bound['consumed_sensor_hashes_bitwise']
    exposure=bound['first_exposure'];frame=None if exposure is None else exposure['frame']
    if frame is None:assert bound['full_source_checked']
    old_frames=json.loads((old/'native/in_memory_camera_observations.json').read_text())['frames']
    new_frames=json.loads((new_run/'native/in_memory_camera_observations.json').read_text())['frames']
    cutoff=None if frame is None else int(old_frames[frame]['physical_sample_index'])
    cutoff_ns=None if frame is None else old_frames[frame]['measured_ns']
    failures=[];native_first={}
    with np.load(old/'native/physics_trace.npz',allow_pickle=False) as a,np.load(new_run/'native/physics_trace.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            count=len(a[key]) if cutoff is None else cutoff+1
            if a[key].dtype!=b[key].dtype or not np.array_equal(a[key][:count],b[key][:count]) or (cutoff is None and a[key].shape!=b[key].shape):failures.append('native/'+key)
            common=min(len(a[key]),len(b[key]));equal=(a[key][:common]==b[key][:common]).reshape(common,-1).all(1)
            indices=np.flatnonzero(~equal)
            if len(indices):native_first[key]=int(indices[0])
            elif len(a[key])!=len(b[key]):native_first[key]=common
    a=json.loads((old/'requests.json').read_text());b=json.loads((new_run/'requests.json').read_text())
    count=len(a) if cutoff_ns is None else sum(r['simulator_ns']<cutoff_ns for r in a)
    if a[:count]!=b[:count] or (cutoff_ns is None and len(a)!=len(b)):failures.append('requests')
    request_first=first_difference(a,b)
    commands_a=[r['requested_command'] for r in a];commands_b=[r['requested_command'] for r in b]
    command_first=first_difference(commands_a,commands_b)
    keys=('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index')
    a=[{k:r[k] for k in keys} for r in old_frames];b=[{k:r[k] for k in keys} for r in new_frames]
    count=len(a) if frame is None else frame+1
    if a[:count]!=b[:count] or (frame is None and len(a)!=len(b)):failures.append('consumed_sensor_hashes')
    sensor_first=first_difference(a,b)
    return dict(episode=episode,required=True,passed=not failures,first_legacy_exposure=exposure,
        exposure_evidence=str(evidence),prefix_failures=failures,
        prefix_semantics='Sensors inclusive through exposure frame; native states through that acquisition; requests before that acquisition. Full equality if no exposure.',
        first_native_difference_sample_by_field=native_first,first_request_difference_index=request_first,
        first_command_difference_policy_s=None if command_first is None else command_first*.02,
        first_sensor_difference_frame=sensor_first,
        entire_recording_identical=not(failures or native_first or request_first is not None or sensor_first is not None),
        disposition='STOP before further changes if prefix fails; post-exposure divergence requires explanation, not an exact-full-trajectory requirement')
