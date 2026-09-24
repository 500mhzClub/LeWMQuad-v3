"""Paired exact measured-point insertion on fixed saved maze observations."""
import hashlib
import json
import time
from itertools import islice
import numpy as np
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.batched_sample_bounds_development import BatchedMeasuredSampleBoundsIndex
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex
from lewm.causal_depth_observation_development import body_points
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.registered_json_pose_development import registered_json_pose
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import BASE, create_output, verify_artifacts, artifact_path
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

INPUT=BASE/'go2_later_floor_resolution_maze_pilot_v1_attempt_001'
OUTPUT=BASE/'go2_packed_owned_maze_bounds_benchmark_v1_attempt_001'
CASE='full_jepa_novel_maze_00'
CLASSES={'original':MeasuredSampleBoundsIndex,'batched':BatchedMeasuredSampleBoundsIndex,
    'packed_owned':PackedOwnedMeasuredSampleBoundsIndex}


def equal(a,b):
    assert list(a.cells)==list(b.cells) and a.cells==b.cells
    assert list(a.bounds)==list(b.bounds) and a.sample_counts==b.sample_counts and a.latest_frames==b.latest_frames
    assert all(a.bounds[k].tobytes()==b.bounds[k].tobytes() for k in a.bounds)


def backing_bytes(index):
    owners={}
    for array in index.bounds.values():
        owner=array
        while isinstance(owner.base,np.ndarray):owner=owner.base
        owners[id(owner)]=owner
    return sum(a.nbytes for a in owners.values())


def main():
    if not __debug__:raise ValueError('assertions required')
    import cv2
    cv2.setNumThreads(1)
    ids={'result.json':'3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755',
        'launch.json':'c3d035abcc69b3b42ecb160021203e7d6d685a176e860e5c3044afa2035cefa4',
        CASE+'/context_decisions.jsonl.gz':'23510c874165007e055cbd1fc49346df38fbad3b30b9d96ea68f22fd3f536aa9'}
    verify_artifacts(INPUT,ids)
    native=json.loads(artifact_path(INPUT,'result.json').read_text())
    assert native['status']=='LATER_FLOOR_RESOLUTION_MAZE_PILOT_COMPLETE'
    names=['policy_observations.json','policy_histories.npz','depth_observations.json','fast_gyro_histories.npz']
    names += [n for i in range(128) for n in (f'rgb_{i:04d}.png',f'depth_{i:04d}.npz')]
    ids |= {CASE+'/'+n:digest(artifact_path(INPUT,CASE+'/'+n)) for n in names}
    sources=discover_sources(('scripts/benchmark_go2_packed_owned_maze_bounds_v1.py',
        'docs/go2_packed_owned_maze_bounds_benchmark_v1_2026-09-09.md',
        'lewm/tests/test_packed_owned_sample_bounds_development.py'),native['source_sha256'])
    source_check(sources);resources=hardware()
    if resources['memory_available_bytes']<4*1024**3 or resources['artifact_free_bytes']<40*1024**3+64*1024**2:
        raise ValueError('bounded insertion benchmark resources unavailable')
    orders=[list(CLASSES),list(reversed(CLASSES))]
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',dict(input_sha256=ids,source_sha256=sources,
        hardware=resources,frames=128,implementation_orders=orders,cpu_processes=1,numerical_threads=1,
        minimum_available_ram_bytes=4*1024**3,output_allowance_bytes=64*1024**2,os_resource_limits_enforced=False,
        concurrency_reason='one bounded insertion benchmark beside the independent settling controller replay',
        native_execution=False,model_training=False,insertion_only_timing=True,isolated_hardware=False))
    print('PACKED_OWNED_MAZE_BOUNDS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        reader=IntentReturnRGBDReplay(INPUT/CASE);clouds=[]
        for i,row in enumerate(islice(read_rows(INPUT/CASE),128)):
            assert row['tick']==i
            policy,depth,_,now=reader.packet(i)
            p,R,pose=registered_json_pose(row['decision']['evidence'],now_ns=now)
            assert pose['frame']==i and pose['rgb_sha256']==depth['rgb_sha256']
            h=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()
            assert pose['depth_sha256']==h
            cloud=body_points(depth,policy,now_ns=now,stride=4)
            points=cloud['points_body_m'][cloud['valid']]@R.T+p
            witness=dict(frame=i,measured_ns=now,rgb_sha256=pose['rgb_sha256'],depth_sha256=h)
            points.setflags(write=False)
            clouds.append((points,witness,p))
        assert len(clouds)==128
        cloud_ids=[hashlib.sha256(p.tobytes()).hexdigest() for p,_,_ in clouds]
        runs=[]
        for repeat,order in enumerate(orders):
            indices={n:cls() for n,cls in CLASSES.items()};times={n:[] for n in CLASSES}
            for i,(points,witness,position) in enumerate(clouds):
                for name in order:
                    start=time.perf_counter_ns();indices[name].insert(points,witness)
                    times[name].append((time.perf_counter_ns()-start)/1e6)
                for name in order:
                    equal(indices['original'],indices[name])
                    for offset in ((.3,0.,-.3),(0.,.2,0.)):
                        center=position+offset
                        assert indices['original'].intersect(center-.04,center+.04)==indices[name].intersect(center-.04,center+.04)
                        assert indices['original'].intersect_sphere(center,.022)==indices[name].intersect_sphere(center,.022)
                assert all(v.flags.owndata and v.base is None for v in indices['packed_owned'].bounds.values())
            runs.append(dict(repeat=repeat,order=order,insertion_ms=times,
                retained_cells=len(indices['original'].cells),
                backing_storage_bytes={n:backing_bytes(v) for n,v in indices.items()},
                all_insertions_and_queries_exact=True,packed_bounds_independently_owned=True))
            print('PACKED_OWNED_MAZE_BOUNDS_REPEAT',repeat,flush=True)
        assert cloud_ids==[hashlib.sha256(p.tobytes()).hexdigest() for p,_,_ in clouds]
        verify_artifacts(INPUT,ids);source_check(sources)
        write_json(OUTPUT/'result.json',dict(status='PACKED_OWNED_MAZE_BOUNDS_BENCHMARK_COMPLETE',
            source_sha256=sources,artifact_sha256={'launch.json':digest(OUTPUT/'launch.json')},
            runs=runs,frames=128,cloud_sha256=cloud_ids,all_input_bytes_unchanged=True,
            insertion_only_timing=True,full_controller_replay=False,native_adoption=False,
            real_time_qualified=False,navigation_qualified=False,hardware_after=hardware()))
        print('PACKED_OWNED_MAZE_BOUNDS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()
