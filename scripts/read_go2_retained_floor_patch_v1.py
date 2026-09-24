"""Compare actual foot footprints against retained measured depth patches."""
import argparse
import hashlib
import json
import time
import numpy as np
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.measured_floor_contact_goal_probe_development import MeasuredFloorContactMap
from lewm.measured_floor_partition_development import FOOT_IDS
from lewm.recorded_visual_evidence_identity_development import restore_identity
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.run_go2_measured_floor_contact_goal_probe_v1 import OUTPUT as INPUT
from scripts.read_go2_measured_floor_contact_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_retained_floor_patch_v1_attempt_001'
PROTOCOL='docs/go2_retained_floor_patch_v1_2026-09-08.md'
NATIVE_SHA='607524c7b6cc86356bc0bdceac1dd310723c6012e1dc377fe80c890f5057f17e'
READOUT_SHA='473c1eec74d205672017efeaec0c742dedcf401f601dc37f0fba969d7b538c08'
CASE='full_direct_family_episode_039'


def replay(limit=None,batched=True):
    directory=INPUT/CASE;reader=IntentReturnRGBDReplay(directory);rows=read_json(directory,'context_decisions.json')
    mapper=MeasuredFloorContactMap(identity=(0,0,0));atlas=RetainedFloorPatches();geometry=ArticulatedCollisionGeometry(URDF)
    available=unavailable=0;selections=[]
    for row in rows[:limit]:
        d=row['decision'];recorded=d['memory_receipt'];tick=row['tick']
        if recorded is None:unavailable+=1;continue
        policy,depth,_,now=reader.packet(tick);evidence=restore_identity(d['evidence'],policy['sensor_state']['identity'])
        receipt=mapper.observe(policy,depth,evidence,now_ns=now);assert receipt==recorded
        memory=mapper.surface;B=mapper.map_from_initial
        assert memory.classification_receipt==d['floor_partition_receipt']
        witness={k:receipt[k] for k in ('frame','measured_ns','rgb_sha256','depth_sha256')}
        atlas.append(depth['depth_m'],depth['valid'],B@memory.rotation,B@memory.position,mapper.floor_height,witness);available+=1
        selection=d['new_selection']
        if selection is None or 'prediction' not in selection:continue
        checks=[];centres=[]
        for prediction,original in zip(selection['prediction'],selection['surface_checks'],strict=True):
            dx,dy,sy,cy,_=prediction[0];yaw=float(np.arctan2(sy,cy))
            check=memory.footprint(geometry,[dx,dy],yaw,now_ns=now,persistent=True);assert check==original
            checks.append(check);c,s=np.cos(yaw),np.sin(yaw)
            R=memory.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]]);p=memory.position+memory.rotation@np.r_[dx,dy,0.]
            shapes={s['shape_id']:s for s in geometry.supports(memory.joints,R)['shapes']}
            centres.extend((B@(p+R@np.asarray(shapes[k]['center_body_m'])))[:2].tolist() for k in FOOT_IDS)
        covered=atlas.coverage(centres) if batched else [atlas.coverage([c])[0] for c in centres]
        candidates=[]
        for i,(candidate,check) in enumerate(zip(selection['candidates'],checks,strict=True)):
            rules={f['shape_id']:f for f in check['foot_floor_contacts']};hits={h['shape_id']:h for h in check['shapes']}
            remaining=[k for k,h in hits.items() if k not in FOOT_IDS and h['intersecting_voxels']];feet=[]
            for j,key in enumerate(FOOT_IDS):
                old=rules[key];patch=covered[4*i+j];eligible=old['measured_floor_contact_rule_eligible'] or patch['complete_nominal_foot_patch']
                blocked=bool(old['other_or_unknown']['intersecting_voxels'] or (hits[key]['intersecting_voxels'] and not eligible))
                if blocked:remaining.append(key)
                feet.append(dict(shape_id=key,grid_covered=old['measured_floor_contact_rule_eligible'],patch=patch,
                    other_or_unknown_hits=old['other_or_unknown']['intersecting_voxels'],conditional_conflict=blocked))
            candidates.append(dict(action=candidate['action'],utility_m=candidate['utility_m'],original_possible=check['possible_intersection'],
                conditional_possible=bool(remaining),remaining_shapes=remaining,feet=feet))
        selections.append(dict(tick=tick,original_action=selection['action'],candidates=candidates))
    return dict(case=CASE,available=available,unavailable=unavailable,retained_prefix_bytes=sum(f['prefix'].nbytes for f in atlas.frames),
        selections=selections,commands_changed=False,native_execution=False,ground_support_approved=False)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive patch diagnostic required')
    h=hardware();assert h['memory_available_bytes']>=8*1024**3 and h['artifact_free_bytes']>=41*1024**3
    verify_artifacts(INPUT,{'result.json':NATIVE_SHA});native=read_json(INPUT,'result.json');assert native['status']=='MEASURED_FLOOR_CONTACT_GOAL_PROBE_COMPLETE'
    native_ids={'result.json':NATIVE_SHA}|native['artifact_sha256'];verify_artifacts(INPUT,native_ids)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']};verify_artifacts(READOUT,readout_ids)
    original=read_json(READOUT,'launch.json');verify(original)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_retained_floor_patch_v1.py',
        'lewm/tests/test_retained_floor_patch_development.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),patch_native_sha256=native_ids,
        patch_readout_sha256=readout_ids,hardware=h,workers=1,maximum_output_bytes=1024**3,native_execution=False)
    verify(launch);smoke=replay(4)
    launch['bounded_decode_preflight_sha256']=hashlib.sha256(json.dumps(smoke,sort_keys=True).encode()).hexdigest()
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('RETAINED_PATCH_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        measurements=[];reference=None
        for batch in (False,True):
            begin=time.perf_counter();r=replay(40,batched=batch)
            if reference is None:reference=r
            assert r==reference,'batched coverage changed'
            measurements.append(dict(batched=batch,wall_s=time.perf_counter()-begin))
        batch=min(measurements,key=lambda r:r['wall_s'])['batched']
        write_json(OUTPUT/'workload.json',dict(measurements=measurements,selected_batched=batch,exact_equal=True,frames=40))
        result=replay(batched=batch);write_json(OUTPUT/'coverage.json',result)
        verify(launch);verify_artifacts(INPUT,native_ids);verify_artifacts(READOUT,readout_ids)
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json','workload.json','coverage.json')};verify_artifacts(OUTPUT,artifacts)
        assert sum((OUTPUT/n).stat().st_size for n in artifacts)<1024**3
        write_json(OUTPUT/'result.json',dict(status='RETAINED_FLOOR_PATCH_DIAGNOSTIC_COMPLETE',artifact_sha256=artifacts,
            source_sha256=sources,wall_s=time.perf_counter()-started,hardware_after=hardware(),commands_changed=False,
            native_execution=False,ground_support_approved=False,navigation_qualified=False,goal_achieved=False))
        print('RETAINED_PATCH_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RETAINED_PATCH_FAILURE',reason=repr(exc)));raise


if __name__=='__main__':main()
