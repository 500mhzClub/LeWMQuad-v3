"""Compare original voxel vetoes with all-return bounds and exact foot spheres."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import numpy as np
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.joint_visual_surface_memory_development import JointVisualSurfaceMemory
from lewm.causal_depth_observation_development import body_points
from lewm.recorded_visual_evidence_identity_development import restore_identity
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_measured_sample_bounds_v1_attempt_001'
PROTOCOL='docs/go2_measured_sample_bounds_v1_2026-09-08.md'
READOUT=BASE/'go2_nominal_action_goal_readout_v1_attempt_001'
READOUT_SHA='c4f22e57ad82e3e50a03692fc288705c106b0c070df942f0d6c2f55a76931219'
INPUTS={
    'commitment_direct_039':dict(root=BASE/'go2_commitment_pose_goal_probe_v1_attempt_001',
        sha='485c7d2081cacc2a5176d72bc4d52a34fd795ea0f4c4bc8a73cb0a29091afd6e',status='COMMITMENT_POSE_GOAL_PROBE_COMPLETE'),
    'nominal_direct_039':dict(root=BASE/'go2_nominal_action_goal_probe_v1_attempt_001',
        sha='f9708a794b678bec4c61c5aba13dc3a3886ec836de7127f5440178cd5737288b',status='NOMINAL_ACTION_GOAL_PROBE_COMPLETE')}
CASE='full_direct_family_episode_039'


def compare(memory,index,geometry,displacement,yaw,now):
    old=memory.footprint(geometry,displacement,yaw,now_ns=now,persistent=True)
    c,s=np.cos(yaw),np.sin(yaw);R=memory.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    p=memory.position+memory.rotation@np.r_[displacement,0.]
    shapes=geometry.supports(memory.joints,R)['shapes'];primitives={s['shape_id']:s for s in geometry._shapes}
    reports=[]
    for shape,original in zip(shapes,old['shapes'],strict=True):
        assert shape['shape_id']==original['shape_id']
        low=np.asarray(shape['lower'])+p;high=np.asarray(shape['upper'])+p
        tight=index.intersect(low,high);assert tight['whole_voxel_intersections']==original['intersecting_voxels']
        primitive=primitives[shape['shape_id']]
        refined=index.intersect_sphere(p+R@np.asarray(shape['center_body_m']),float(primitive['dimensions'][0])) if primitive['kind']=='sphere' else tight
        assert refined['intersecting_voxels']<=tight['intersecting_voxels']
        if original['intersecting_voxels']:
            reports.append(dict(shape_id=shape['shape_id'],kind=primitive['kind'],original=original,
                predicted_shape_box_m=[low.tolist(),high.tolist()],tight_bounds=tight,primitive_refinement=refined))
    return old,dict(original_possible=old['possible_intersection'],
        tight_possible=any(r['tight_bounds']['intersecting_voxels'] for r in reports),
        primitive_possible=any(r['primitive_refinement']['intersecting_voxels'] for r in reports),hits=reports)


def replay(name,limit=None):
    directory=INPUTS[name]['root']/CASE;reader=IntentReturnRGBDReplay(directory)
    rows=read_json(directory,'context_decisions.json');memory=JointVisualSurfaceMemory(identity=(0,0,0))
    index=MeasuredSampleBoundsIndex();geometry=ArticulatedCollisionGeometry(URDF)
    selections=[];available=unavailable=points_total=0
    for row in rows[:limit]:
        d=row['decision'];recorded=d['memory_receipt'];tick=row['tick']
        if recorded is None:unavailable+=1;continue
        policy,depth,_,now=reader.packet(tick);evidence=restore_identity(d['evidence'],policy['sensor_state']['identity'])
        receipt=memory.observe(policy,depth,evidence,now_ns=now)
        for key in ('frame','measured_ns','rgb_sha256','depth_sha256'):assert receipt[key]==recorded[key]
        cloud=body_points(depth,policy,now_ns=now,stride=4)
        points=cloud['points_body_m'][cloud['valid']]@memory.rotation.T+memory.position
        witness={k:receipt[k] for k in ('frame','measured_ns','rgb_sha256','depth_sha256')}
        index.insert(points,witness);points_total+=len(points);available+=1
        assert index.cells==memory.index.cells and sum(index.sample_counts.values())==points_total
        selection=d['new_selection']
        if selection is None or 'prediction' not in selection:continue
        candidates=[]
        for candidate,prediction,check in zip(selection['candidates'],selection['prediction'],selection['surface_checks'],strict=True):
            dx,dy,sy,cy,_=prediction[0];original,refinement=compare(memory,index,geometry,[dx,dy],float(np.arctan2(sy,cy)),now)
            assert original==check,('original surface check',name,tick,candidate['action'])
            candidates.append(dict(action=candidate['action'],utility_m=candidate['utility_m'],**refinement))
        _,current=compare(memory,index,geometry,[0.,0.],0.,now)
        selections.append(dict(tick=tick,original_action=selection['action'],mode=selection['mode'],
            candidates=candidates,current_posture=current))
    return dict(case=name,available=available,unavailable=unavailable,points_enclosed=points_total,
        voxel_count=len(index.cells),selections=selections,
        first_witness_identity_sha256=hashlib.sha256(json.dumps(sorted(index.cells.items()),sort_keys=True).encode()).hexdigest(),
        final_bounds_sha256=hashlib.sha256(json.dumps([(k,index.bounds[k].tolist(),index.sample_counts[k]) for k in sorted(index.cells)]).encode()).hexdigest(),
        original_commands_changed=False,ground_support_approved=False,native_execution=False)


def execute(width,limit):
    with ThreadPoolExecutor(max_workers=width) as pool:
        jobs=[pool.submit(replay,name,limit) for name in INPUTS]
        return [f.result() for f in jobs]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive bounds diagnostic required')
    h=hardware();assert h['memory_available_bytes']>=8*1024**3 and h['artifact_free_bytes']>=41*1024**3
    bindings={}
    for name,spec in INPUTS.items():
        verify_artifacts(spec['root'],{'result.json':spec['sha']});r=read_json(spec['root'],'result.json');assert r['status']==spec['status']
        bindings[name]={'result.json':spec['sha']}|r['artifact_sha256'];verify_artifacts(spec['root'],bindings[name])
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});r=read_json(READOUT,'result.json')
    readout_ids={'result.json':READOUT_SHA,'launch.json':r['launch_sha256']};verify_artifacts(READOUT,readout_ids)
    original=read_json(READOUT,'launch.json');verify(original)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_measured_sample_bounds_v1.py',
        'lewm/tests/test_measured_sample_bounds_development.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        bounds_native_inputs_sha256=bindings,bounds_readout_sha256=readout_ids,
        hardware=h,native_execution=False,maximum_output_bytes=1024**3)
    verify(launch);assert digest(URDF)==launch['robot_urdf_sha256']
    smoke=execute(1,4)
    launch['bounded_decode_preflight_sha256']=hashlib.sha256(json.dumps(smoke,sort_keys=True).encode()).hexdigest()
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('SAMPLE_BOUNDS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        measurements=[];reference=None
        for width in (1,2):
            begin=time.perf_counter();value=execute(width,40)
            if reference is None:reference=value
            assert value==reference,'concurrent bounds arithmetic changed'
            measurements.append(dict(workers=width,wall_s=time.perf_counter()-begin))
        width=min(measurements,key=lambda r:r['wall_s'])['workers']
        write_json(OUTPUT/'workload.json',dict(measurements=measurements,selected_workers=width,benchmark_frames_per_case=40,exact_equal=True))
        results=execute(width,None)
        for result in results:
            write_json(OUTPUT/(result['case']+'.json'),result)
            print('SAMPLE_BOUNDS_CASE',result['case'],result['points_enclosed'],flush=True)
        verify(launch);verify_artifacts(READOUT,readout_ids)
        for name,spec in INPUTS.items():verify_artifacts(spec['root'],bindings[name])
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json','workload.json',*(name+'.json' for name in INPUTS))}
        verify_artifacts(OUTPUT,artifacts);assert sum((OUTPUT/n).stat().st_size for n in artifacts)<1024**3
        write_json(OUTPUT/'result.json',dict(status='MEASURED_SAMPLE_BOUNDS_DIAGNOSTIC_COMPLETE',artifact_sha256=artifacts,
            source_sha256=sources,wall_s=time.perf_counter()-started,hardware_after=hardware(),
            original_commands_changed=False,native_execution=False,ground_support_approved=False,navigation_qualified=False,goal_achieved=False))
        print('SAMPLE_BOUNDS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SAMPLE_BOUNDS_DIAGNOSTIC_FAILURE',reason=repr(exc)));raise


if __name__=='__main__':main()
