"""Exact recorded-cloud query comparison; no controller or native execution."""
import argparse
import hashlib
from itertools import islice
import json
import time
import cv2
import numpy as np
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.causal_depth_observation_development import body_points
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.registered_json_pose_development import registered_json_pose
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts,artifact_path
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware,source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.benchmark_go2_packed_owned_maze_bounds_v1 import INPUT,CASE,equal,OUTPUT as PRIOR

OUTPUT=BASE/'go2_single_pass_maze_queries_v1_attempt_001'
PROTOCOL='docs/go2_single_pass_maze_queries_v1_2026-09-09.md'
PRIOR_RESULT='e3f603f7baed193e23b53116fe14f46a94bc6a98ad4b5eebca0af604dbe9a872'
CLASSES={'original':MeasuredSampleBoundsIndex,'single_pass':SinglePassMeasuredSampleBoundsIndex}
REPETITIONS=16


def queries(points,position):
    centers=(position+(.3,0.,-.3),position+(0.,.2,0.),points[len(points)//2])
    return [(kind,center.copy(),size) for center in centers for kind,size in
        (('box',(.04,.04,.04)),('box',(.25,.12,.08)),('sphere',.022))]


def query(index,definition):
    kind,center,size=definition
    return index.intersect_sphere(center,size) if kind=='sphere' else index.intersect(center-size,center+size)


def verify_all(launch):
    source_check(launch['source_sha256']);verify_artifacts(INPUT,launch['input_sha256'])
    verify_artifacts(PRIOR,launch['prior_artifact_sha256'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive fresh query benchmark required')
    cv2.setNumThreads(1)
    verify_artifacts(PRIOR,{'result.json':PRIOR_RESULT});prior=read_json(PRIOR,'result.json')
    assert prior['status']=='PACKED_OWNED_MAZE_BOUNDS_BENCHMARK_COMPLETE' and prior['frames']==128
    prior_ids={'result.json':PRIOR_RESULT,**prior['artifact_sha256']};verify_artifacts(PRIOR,prior_ids)
    old=read_json(PRIOR,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/benchmark_go2_single_pass_maze_queries_v1.py',
        'lewm/tests/test_single_pass_sample_bounds_development.py'),prior['source_sha256'])
    resources=hardware()
    launch=old|dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,hardware=resources,
        prior_artifact_sha256=prior_ids,frames=128,implementation_orders=[list(CLASSES),list(reversed(CLASSES))],
        query_repetitions_per_frame=REPETITIONS,queries_per_frame=9,
        expected_timed_queries_per_implementation=128*9*REPETITIONS*2,
        query_only_timing=True,insertion_only_timing=False,recorded_clouds_reconstructed=True,
        actual_controller_query_distribution=False,query_receipts_complete=True,full_controller_replay=False,
        native_adoption=False,cpu_processes=1,numerical_threads=1,native_execution=False,model_training=False,
        minimum_available_ram_bytes=4*1024**3,output_allowance_bytes=64*1024**2,
        concurrency_reason='one bounded CPU query benchmark beside the existing single native maze scene')
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=4*1024**3
    storage_ok=resources['artifact_free_bytes']>=40*1024**3+64*1024**2
    if args.preflight_only:
        print('SINGLE_PASS_QUERY_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('bounded query benchmark resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('SINGLE_PASS_QUERY_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        reader=IntentReturnRGBDReplay(INPUT/CASE);clouds=[]
        for i,row in enumerate(islice(read_rows(INPUT/CASE),128)):
            assert row['tick']==i
            policy,depth,_,now=reader.packet(i)
            p,R,pose=registered_json_pose(row['decision']['evidence'],now_ns=now)
            h=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()
            assert pose['frame']==i and pose['rgb_sha256']==depth['rgb_sha256'] and pose['depth_sha256']==h
            cloud=body_points(depth,policy,now_ns=now,stride=4)
            points=cloud['points_body_m'][cloud['valid']]@R.T+p
            if not len(points):raise ValueError('fixed measured-point query requires a nonempty cloud')
            points.setflags(write=False)
            clouds.append((points,dict(frame=i,measured_ns=now,rgb_sha256=pose['rgb_sha256'],depth_sha256=h),p))
        cloud_ids=[hashlib.sha256(p.tobytes()).hexdigest() for p,_,_ in clouds]
        assert len(clouds)==128 and cloud_ids==prior['cloud_sha256']
        runs=[]
        for repeat,order in enumerate(launch['implementation_orders']):
            indices={n:cls() for n,cls in CLASSES.items()};times={n:[] for n in CLASSES};receipt_digest=hashlib.sha256()
            for frame,(points,witness,position) in enumerate(clouds):
                for index in indices.values():index.insert(points,witness)
                equal(indices['original'],indices['single_pass'])
                definitions=queries(points,position)
                reference=[query(indices['original'],q) for q in definitions]
                assert reference==[query(indices['single_pass'],q) for q in definitions]
                receipt_digest.update(json.dumps(reference,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
                receipt_digest.update(b'\n')
                for name in order:
                    start=time.perf_counter_ns()
                    answers=[[query(indices[name],q) for q in definitions] for _ in range(REPETITIONS)]
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    assert all(answer==reference for answer in answers)
                    times[name].append(elapsed)
                equal(indices['original'],indices['single_pass'])
                if frame%32==0:print('SINGLE_PASS_QUERY_FRAME',repeat,frame,flush=True)
            runs.append(dict(repeat=repeat,order=order,batch_query_ms=times,retained_cells=len(indices['original'].cells),
                median_paired_batch_speed_ratio=float(np.median(np.asarray(times['original'])/times['single_pass'])),
                query_receipt_sha256=receipt_digest.hexdigest(),every_timed_query_receipt_exact=True,
                insertions_and_index_state_exact=True))
        assert runs[0]['query_receipt_sha256']==runs[1]['query_receipt_sha256']
        assert cloud_ids==[hashlib.sha256(p.tobytes()).hexdigest() for p,_,_ in clouds]
        verify_all(launch)
        write_json(OUTPUT/'result.json',dict(status='SINGLE_PASS_MAZE_QUERY_BENCHMARK_COMPLETE',
            source_sha256=sources,artifact_sha256={'launch.json':digest(OUTPUT/'launch.json')},
            runs=runs,frames=128,timed_queries_per_implementation=128*9*REPETITIONS*2,
            cloud_sha256=cloud_ids,all_input_bytes_unchanged=True,wall_s=time.perf_counter()-started,
            hardware_after=hardware(),query_only_timing=True,actual_controller_query_distribution=False,
            full_controller_replay=False,native_adoption=False,real_time_qualified=False,
            navigation_qualified=False,goal_achieved=False))
        print('SINGLE_PASS_QUERY_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()
