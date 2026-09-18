"""All-return floor classification and conservative nominal foot coverage."""
import argparse
import hashlib
import json
import time
import numpy as np
from lewm.sample_bounds_goal_probe_development import SampleBoundsFloorMap
from lewm.measured_floor_partition_development import MeasuredFloorPartition,FOOT_IDS,foot_projection_coverage
from lewm.causal_depth_observation_development import body_points
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.recorded_visual_evidence_identity_development import restore_identity
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.run_go2_sample_bounds_goal_probe_v1 import OUTPUT as INPUT
from scripts.read_go2_sample_bounds_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_measured_floor_partition_v1_attempt_001'
PROTOCOL='docs/go2_measured_floor_partition_v1_2026-09-08.md'
NATIVE_SHA='76bcd4079e7271a2e8ecd6fde9c551cfc1a83918b84239f4449e138345a36dc4'
READOUT_SHA='2e2c89b3593d9d14662ffbd5fbd0ac39e5a9fd0716c1107c3a8d873407090b2f'
CASE='full_direct_family_episode_039'


def replay(limit=None):
    directory=INPUT/CASE;reader=IntentReturnRGBDReplay(directory);rows=read_json(directory,'context_decisions.json')
    mapper=SampleBoundsFloorMap(identity=(0,0,0));partition=MeasuredFloorPartition();geometry=ArticulatedCollisionGeometry(URDF)
    primitives={s['shape_id']:s for s in geometry._shapes}
    assert all(primitives[k]['kind']=='sphere' and float(primitives[k]['dimensions'][0])==.022 for k in FOOT_IDS)
    available=unavailable=0;selections=[];frame_counts=[]
    for row in rows[:limit]:
        d=row['decision'];recorded=d['memory_receipt'];tick=row['tick']
        if recorded is None:unavailable+=1;continue
        policy,depth,_,now=reader.packet(tick);evidence=restore_identity(d['evidence'],policy['sensor_state']['identity'])
        receipt=mapper.observe(policy,depth,evidence,now_ns=now);assert receipt==recorded
        memory=mapper.surface;B=mapper.map_from_initial;Rm=B@memory.rotation;pm=B@memory.position
        cloud=body_points(depth,policy,now_ns=now,stride=4)
        patch=sampled_floor_patch(depth['depth_m'],depth['valid'],Rm,pm,mapper.floor_height,cloud['rows'],cloud['columns'])
        points=cloud['points_body_m'][cloud['valid']]@memory.rotation.T+memory.position
        mask=patch['measured_floor_patch'][cloud['valid']]
        witness={k:receipt[k] for k in ('frame','measured_ns','rgb_sha256','depth_sha256')}
        partition.insert(points,mask,witness);available+=1
        assert partition.total_returns==sum(memory.index.sample_counts.values())
        assert set(partition.floor.cells)|set(partition.other.cells)==set(memory.index.cells)
        frame_counts.append(dict(tick=tick,returns=len(points),floor_returns=int(mask.sum()),other_returns=int((~mask).sum())))
        selection=d['new_selection']
        if selection is None or 'prediction' not in selection:continue
        candidates=[]
        for candidate,prediction,original in zip(selection['candidates'],selection['prediction'],selection['surface_checks'],strict=True):
            dx,dy,sy,cy,_=prediction[0];yaw=float(np.arctan2(sy,cy))
            check=memory.footprint(geometry,[dx,dy],yaw,now_ns=now,persistent=True);assert check==original
            c,s=np.cos(yaw),np.sin(yaw);R=memory.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            p=memory.position+memory.rotation@np.r_[dx,dy,0.]
            shapes=geometry.supports(memory.joints,R)['shapes'];feet=[];remaining=[]
            for shape,hit in zip(shapes,check['shapes'],strict=True):
                key=shape['shape_id']
                if key not in FOOT_IDS:
                    if hit['intersecting_voxels']:remaining.append(key)
                    continue
                center=p+R@np.asarray(shape['center_body_m']);radius=float(primitives[key]['dimensions'][0])
                ground=partition.floor.intersect_sphere(center,radius);other=partition.other.intersect_sphere(center,radius)
                assert max(ground['intersecting_voxels'],other['intersecting_voxels'])<=hit['intersecting_voxels']
                coverage=foot_projection_coverage((B@center)[:2],radius,mapper.floor)
                blocked=bool(other['intersecting_voxels'] or (hit['intersecting_voxels'] and not coverage['entire_nominal_projection_on_measured_floor']))
                if blocked:remaining.append(key)
                feet.append(dict(shape_id=key,original_intersections=hit['intersecting_voxels'],floor=ground,other_or_unknown=other,
                    projection=coverage,conditional_conflict=blocked))
            candidates.append(dict(action=candidate['action'],utility_m=candidate['utility_m'],
                original_possible=check['possible_intersection'],conditional_possible=bool(remaining),remaining_shapes=remaining,feet=feet))
        selections.append(dict(tick=tick,original_action=selection['action'],candidates=candidates))
    return dict(case=CASE,available=available,unavailable=unavailable,frame_counts=frame_counts,selections=selections,
        total_returns=partition.total_returns,floor_returns=partition.floor_returns,other_returns=partition.other_returns,
        floor_keys=len(partition.floor.cells),other_keys=len(partition.other.cells),mixed_keys=len(set(partition.floor.cells)&set(partition.other.cells)),
        original_commands_changed=False,contact_policy_applied=False,ground_support_approved=False,native_execution=False)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive partition diagnostic required')
    h=hardware();assert h['memory_available_bytes']>=8*1024**3 and h['artifact_free_bytes']>=41*1024**3
    verify_artifacts(INPUT,{'result.json':NATIVE_SHA});native=read_json(INPUT,'result.json')
    assert native['status']=='SAMPLE_BOUNDS_GOAL_PROBE_COMPLETE'
    native_ids={'result.json':NATIVE_SHA}|native['artifact_sha256'];verify_artifacts(INPUT,native_ids)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']};verify_artifacts(READOUT,readout_ids)
    original=read_json(READOUT,'launch.json');verify(original)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_measured_floor_partition_v1.py',
        'lewm/tests/test_measured_floor_partition_development.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),partition_native_sha256=native_ids,
        partition_readout_sha256=readout_ids,hardware=h,workers=1,maximum_output_bytes=1024**3,native_execution=False)
    verify(launch);smoke=replay(4)
    launch['bounded_decode_preflight_sha256']=hashlib.sha256(json.dumps(smoke,sort_keys=True).encode()).hexdigest()
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('FLOOR_PARTITION_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        result=replay();write_json(OUTPUT/'classification.json',result)
        verify(launch);verify_artifacts(INPUT,native_ids);verify_artifacts(READOUT,readout_ids)
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json','classification.json')};verify_artifacts(OUTPUT,artifacts)
        assert sum((OUTPUT/n).stat().st_size for n in artifacts)<1024**3
        write_json(OUTPUT/'result.json',dict(status='MEASURED_FLOOR_PARTITION_DIAGNOSTIC_COMPLETE',artifact_sha256=artifacts,
            source_sha256=sources,wall_s=time.perf_counter()-started,hardware_after=hardware(),
            original_commands_changed=False,contact_policy_applied=False,native_execution=False,ground_support_approved=False,navigation_qualified=False,goal_achieved=False))
        print('FLOOR_PARTITION_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FLOOR_PARTITION_FAILURE',reason=repr(exc)));raise


if __name__=='__main__':main()
