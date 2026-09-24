"""Diagnose unchanged-radius connectors and original first-witness floor returns."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import numpy as np
from lewm.observed_geometry_refinement_development import nominal_connector,sampled_floor_patch
from lewm.causal_depth_observation_development import body_points
from lewm.joint_visual_floor_map_development import JointVisualFloorMap
from lewm.recorded_visual_evidence_identity_development import restore_identity
from lewm.observed_floor_waypoint_development import inflated_cells,segment_cells,centre
from lewm.joint_visual_surface_memory_development import VOXEL_M
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.read_go2_commitment_pose_route_loss_v2 import OUTPUT as ROUTE,INPUT,NATIVE_SHA,CASES
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_observed_geometry_refinement_v1_attempt_001'
PROTOCOL='docs/go2_observed_geometry_refinement_v1_2026-09-08.md'
ROUTE_SHA='a33b120bdd8b210ab54da3293e903ab07fdcbc006fa388a197127c5298ce1b84'


def replay(case,limit=None):
    prior=read_json(ROUTE,case+'.json');targets={}
    for selection in prior['selections']:
        for candidate in selection['candidates']:
            for hit in candidate['hits']:
                key=(hit['witness']['frame'],tuple(hit['first_cell']))
                if key not in targets:targets[key]=dict(witness=hit['witness'],occurrences=[])
                assert targets[key]['witness']==hit['witness']
                targets[key]['occurrences'].append(dict(tick=selection['tick'],action=candidate['action'],shape_id=hit['shape_id']))
    rows=read_json(INPUT/case,'context_decisions.json');reader=IntentReturnRGBDReplay(INPUT/case)
    mapper=JointVisualFloorMap(identity=(0,0,0));selections=[];returns=[];available=unavailable=0
    for row in rows[:limit]:
        tick=row['tick'];decision=row['decision'];recorded=decision['memory_receipt']
        if recorded is None:unavailable+=1;continue
        policy,depth,_,now=reader.packet(tick)
        evidence=restore_identity(decision['evidence'],policy['sensor_state']['identity'])
        receipt=mapper.observe(policy,depth,evidence,now_ns=now)
        assert receipt==recorded,('recorded map receipt',case,tick)
        available+=1;B=mapper.map_from_initial;p=B@mapper.surface.position
        frame_targets=[(key,value) for key,value in targets.items() if key[0]==receipt['frame']]
        if frame_targets:
            cloud=body_points(depth,policy,now_ns=now,stride=4)
            xyz=cloud['points_body_m']@mapper.surface.rotation.T+mapper.surface.position
            keys=np.floor(np.where(cloud['valid'][...,None],xyz,0.)/VOXEL_M).astype(np.int64)
            patch=sampled_floor_patch(depth['depth_m'],depth['valid'],B@mapper.surface.rotation,p,
                mapper.floor_height,cloud['rows'],cloud['columns'])
            for (frame,cell),target in frame_targets:
                assert mapper.surface.index.cells[cell]==target['witness']
                rr,cc=np.nonzero(cloud['valid']&(keys==cell).all(axis=2));assert len(rr)>0
                samples=[dict(pixel_row=int(cloud['rows'][r]),pixel_column=int(cloud['columns'][c]),
                    point_initial_m=xyz[r,c].tolist(),point_map_height_m=float(patch['point_map_height_m'][r,c]),
                    measured_floor_patch=bool(patch['measured_floor_patch'][r,c])) for r,c in zip(rr,cc,strict=True)]
                returns.append(dict(first_cell=list(cell),witness=target['witness'],occurrences=target['occurrences'],
                    floor_height_m=mapper.floor_height,samples=samples,
                    all_first_frame_returns_measured_floor_patches=all(s['measured_floor_patch'] for s in samples),
                    later_returns_classified=False,ground_support_approved=False))
        selection=decision['new_selection']
        if selection is None or 'prediction' not in selection:continue
        assert mapper.waypoint([1.2,0.],now_ns=now)==selection['proposal']
        blocked=inflated_cells(mapper.occupied);occupied=sorted(mapper.occupied)
        candidates=sorted((c for c in set(mapper.floor)-blocked if np.linalg.norm(centre(c)-p[:2])<=1.25),
            key=lambda c:(float(np.linalg.norm(centre(c)-p[:2])),c))
        start=nominal_connector(p[:2],p[:2],occupied);chosen=None;tested=0
        if start['nominal_disk_connector_clear']:
            for cell in candidates:
                tested+=1;check=nominal_connector(p[:2],centre(cell),occupied)
                if check['nominal_disk_connector_clear']:
                    chosen=dict(entry_cell=list(cell),entry_xy_m=centre(cell).tolist(),clearance=check);break
        selections.append(dict(tick=tick,original_proposal_status=selection['proposal']['status'],
            position_map_xy_m=p[:2].tolist(),original_start_inflated=bool(segment_cells(p[:2],p[:2])&blocked),
            start_clearance=start,candidates=len(candidates),connectors_tested=tested,
            first_continuous_connector=chosen))
    if limit is None:
        assert len(returns)==len(targets)
        assert available==prior['available_map_receipts_reconstructed'] and unavailable==prior['unavailable_map_receipts_retained']
        assert hashlib.sha256(json.dumps(sorted(mapper.floor.items())).encode()).hexdigest()==prior['final_floor_map_sha256']
        assert hashlib.sha256(json.dumps(sorted(mapper.occupied.items())).encode()).hexdigest()==prior['final_occupied_map_sha256']
    return dict(case=case,available=available,unavailable=unavailable,selections=selections,first_witness_returns=returns,
        controls_changed=False,surface_vetoes_changed=False,native_execution=False,ground_support_approved=False)


def execute(width,limit):
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures=[pool.submit(replay,c,limit) for c in CASES]
        return [f.result() for f in futures]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive diagnostic required')
    h=hardware();assert h['memory_available_bytes']>=8*1024**3 and h['artifact_free_bytes']>=41*1024**3
    verify_artifacts(ROUTE,{'result.json':ROUTE_SHA});prior=read_json(ROUTE,'result.json')
    assert prior['status']=='COMMITMENT_POSE_ROUTE_LOSS_DIAGNOSTIC_COMPLETE'
    route_ids={'result.json':ROUTE_SHA}|prior['artifact_sha256'];verify_artifacts(ROUTE,route_ids)
    verify_artifacts(INPUT,{'result.json':NATIVE_SHA});native=read_json(INPUT,'result.json')
    native_ids={'result.json':NATIVE_SHA}|native['artifact_sha256'];verify_artifacts(INPUT,native_ids)
    original=read_json(ROUTE,'launch.json');verify(original)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_observed_geometry_refinement_v1.py',
        'lewm/tests/test_observed_geometry_refinement_development.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        refinement_route_inputs_sha256=route_ids,refinement_native_inputs_sha256=native_ids,
        hardware=h,native_execution=False,maximum_output_bytes=1024**3)
    verify(launch)
    smoke=[replay(c,8) for c in CASES]
    launch['bounded_decode_preflight_sha256']=hashlib.sha256(json.dumps(smoke,sort_keys=True).encode()).hexdigest()
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('GEOMETRY_REFINEMENT_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        measurements=[];reference=None
        for width in (1,2):
            begin=time.perf_counter();value=execute(width,40)
            if reference is None:reference=value
            assert value==reference,'concurrent diagnostic changed'
            measurements.append(dict(workers=width,wall_s=time.perf_counter()-begin))
        width=min(measurements,key=lambda v:v['wall_s'])['workers']
        write_json(OUTPUT/'workload.json',dict(measurements=measurements,selected_workers=width,
            benchmark_frames_per_case=40,exact_reconstruction_equal=True))
        results=execute(width,None)
        for r in results:
            write_json(OUTPUT/(r['case']+'.json'),r)
            print('GEOMETRY_REFINEMENT_CASE',r['case'],len(r['selections']),len(r['first_witness_returns']),flush=True)
        verify(launch);verify_artifacts(ROUTE,route_ids);verify_artifacts(INPUT,native_ids)
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json','workload.json',*(c+'.json' for c in CASES))}
        verify_artifacts(OUTPUT,artifacts);assert sum((OUTPUT/n).stat().st_size for n in artifacts)<1024**3
        write_json(OUTPUT/'result.json',dict(status='OBSERVED_GEOMETRY_REFINEMENT_DIAGNOSTIC_COMPLETE',
            artifact_sha256=artifacts,source_sha256=sources,cases=CASES,wall_s=time.perf_counter()-started,
            hardware_after=hardware(),native_execution=False,controls_changed=False,surface_vetoes_changed=False,
            ground_support_approved=False,navigation_qualified=False,goal_achieved=False))
        print('GEOMETRY_REFINEMENT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_GEOMETRY_REFINEMENT_FAILURE',reason=repr(exc)));raise


if __name__=='__main__':main()
