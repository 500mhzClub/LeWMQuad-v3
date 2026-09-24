"""Reconstruct observed maps and explain route/footprint vetoes without changing them."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import numpy as np
from lewm.joint_visual_floor_map_development import JointVisualFloorMap
from lewm.observed_floor_waypoint_development import inflated_cells, segment_cells, centre
from lewm.joint_visual_surface_memory_development import VOXEL_M
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.run_go2_commitment_pose_goal_probe_v1 import OUTPUT as INPUT
from scripts.read_go2_commitment_pose_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_commitment_pose_route_loss_v1_attempt_001'
PROTOCOL='docs/go2_commitment_pose_route_loss_v1_2026-09-08.md'
NATIVE_SHA='485c7d2081cacc2a5176d72bc4d52a34fd795ea0f4c4bc8a73cb0a29091afd6e'
READOUT_SHA='e3626874caca2ace86e3f9c60ade48e720927fe3649ffbce32900ce8cd3d31f6'
CASES=('full_direct_family_episode_052','full_direct_family_episode_039')


def voxel_floor_interval(cell,B,height):
    corners=(np.asarray(cell)+np.array([(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]))*VOXEL_M
    heights=corners@B[2]
    return dict(map_height_interval_m=[float(heights.min()),float(heights.max())],
        measured_floor_height_m=height,floor_plane_crosses_voxel=bool(heights.min()<=height<=heights.max()),
        ground_surface_identity_established=False)


def replay(case,limit=None):
    rows=read_json(INPUT/case,'context_decisions.json');reader=IntentReturnRGBDReplay(INPUT/case)
    mapper=JointVisualFloorMap(identity=(0,0,0));geometry=ArticulatedCollisionGeometry(URDF)
    reports=[];available=unavailable=0
    for row in rows[:limit]:
        tick=row['tick'];decision=row['decision'];recorded=decision['memory_receipt']
        if recorded is None:
            unavailable+=1;continue
        policy,depth,_,now=reader.packet(tick)
        receipt=mapper.observe(policy,depth,decision['evidence'],now_ns=now)
        assert receipt==recorded,('recorded map receipt',case,tick)
        available+=1;selection=decision['new_selection']
        if selection is None or 'prediction' not in selection:continue
        proposal=mapper.waypoint([1.2,0.],now_ns=now);assert proposal==selection['proposal']
        B=mapper.map_from_initial;p=B@mapper.surface.position
        blocked=inflated_cells(mapper.occupied);start=segment_cells(p[:2],p[:2])
        start_blocked=sorted(start&blocked)
        possible=[c for c in set(mapper.floor)-blocked if np.linalg.norm(centre(c)-p[:2])<=1.25]
        checks=[]
        for candidate,prediction,original in zip(selection['candidates'],selection['prediction'],selection['surface_checks'],strict=True):
            dx,dy,sy,cy,_=prediction[0]
            check=mapper.surface.footprint(geometry,[dx,dy],float(np.arctan2(sy,cy)),now_ns=now,persistent=True)
            assert check==original,('recorded surface veto',case,tick,candidate['action'])
            hits=[v|voxel_floor_interval(v['first_cell'],B,mapper.floor_height)
                for v in check['shapes'] if v['intersecting_voxels']]
            checks.append(dict(action=candidate['action'],utility_m=candidate['utility_m'],
                possible_intersection=check['possible_intersection'],hits=hits))
        current=mapper.surface.footprint(geometry,[0.,0.],0.,now_ns=now,persistent=True)
        reports.append(dict(tick=tick,mode=selection['mode'],action=selection['action'],
            measured_map_xy_m=p[:2].tolist(),proposal=proposal,start_cells=sorted(start),
            start_cells_in_nominal_inflation=start_blocked,available_entry_candidates_within_1_25m=len(possible),
            every_closed_connector_intersects_inflation_at_start=bool(start_blocked),
            current_posture_surface_hits=[v|voxel_floor_interval(v['first_cell'],B,mapper.floor_height)
                for v in current['shapes'] if v['intersecting_voxels']],candidates=checks,
            disk_conflict_is_actual_robot_collision=False,foot_contact_permission_changed=False))
    return dict(case=case,actual_rows=len(rows) if limit is None else min(limit,len(rows)),
        available_map_receipts_reconstructed=available,unavailable_map_receipts_retained=unavailable,
        selections=reports,final_floor_cells=len(mapper.floor),final_occupied_cells=len(mapper.occupied),
        final_floor_map_sha256=hashlib.sha256(json.dumps(sorted(mapper.floor.items())).encode()).hexdigest(),
        final_occupied_map_sha256=hashlib.sha256(json.dumps(sorted(mapper.occupied.items())).encode()).hexdigest(),
        observer_reestimated=False,native_execution=False,controller_modified=False)


def execute(width,limit):
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures=[pool.submit(replay,c,limit) for c in CASES]
        return [f.result() for f in futures]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive route-loss diagnostic')
    h=hardware();assert h['memory_available_bytes']>=8*1024**3 and h['artifact_free_bytes']>=41*1024**3
    verify_artifacts(INPUT,{'result.json':NATIVE_SHA});native=read_json(INPUT,'result.json')
    assert native['status']=='COMMITMENT_POSE_GOAL_PROBE_COMPLETE'
    inputs={'result.json':NATIVE_SHA}|native['artifact_sha256'];verify_artifacts(INPUT,inputs)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    assert readout['status']=='COMMITMENT_POSE_GOAL_READOUT_COMPLETE' and readout['probe_result_sha256']==NATIVE_SHA
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']};verify_artifacts(READOUT,readout_ids)
    original=read_json(READOUT,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/read_go2_commitment_pose_route_loss_v1.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        route_loss_native_inputs_sha256=inputs,route_loss_readout_sha256=readout_ids,
        diagnostic_cases=CASES,hardware=h,native_execution=False,maximum_output_bytes=1024**3)
    verify(launch);assert digest(URDF)==launch['robot_urdf_sha256']
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('ROUTE_LOSS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        measurements=[];reference=None
        for width in (1,2):
            begin=time.perf_counter();value=execute(width,8)
            if reference is None:reference=value
            assert value==reference,'concurrent map arithmetic changed'
            measurements.append(dict(workers=width,wall_s=time.perf_counter()-begin))
        width=min(measurements,key=lambda v:v['wall_s'])['workers']
        write_json(OUTPUT/'workload.json',dict(measurements=measurements,selected_workers=width,
            benchmark_frames_per_case=8,exact_reconstruction_equal=True))
        results=execute(width,None)
        for r in results:
            write_json(OUTPUT/(r['case']+'.json'),r)
            print('ROUTE_LOSS_CASE',r['case'],r['available_map_receipts_reconstructed'],flush=True)
        verify(launch);verify_artifacts(INPUT,inputs);verify_artifacts(READOUT,readout_ids)
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json','workload.json',*(c+'.json' for c in CASES))}
        verify_artifacts(OUTPUT,artifacts);assert sum((OUTPUT/n).stat().st_size for n in artifacts)<1024**3
        write_json(OUTPUT/'result.json',dict(status='COMMITMENT_POSE_ROUTE_LOSS_DIAGNOSTIC_COMPLETE',
            artifact_sha256=artifacts,source_sha256=sources,cases=CASES,wall_s=time.perf_counter()-started,
            available_map_receipts_reconstructed=sum(r['available_map_receipts_reconstructed'] for r in results),
            actual_rows=sum(r['actual_rows'] for r in results),hardware_after=hardware(),
            commands_changed=False,surface_vetoes_changed=False,ground_support_approved=False,
            native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('ROUTE_LOSS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ROUTE_LOSS_DIAGNOSTIC_FAILURE',reason=repr(exc)));raise


if __name__=='__main__':main()
