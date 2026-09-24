"""New fitting-only ideal load stream; evaluator terrain roles only after save."""
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.foot_load_sensor_development import FEET,LocalLoadHistory
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries
from lewm.physical_execution_development import rotation_xyzw
from lewm.simulated_foot_force_development import FIELDS,sample_ideal_foot_forces
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_ideal_foot_load_reconstruction_development_v1_attempt_001'
PREVIOUS=ROOT/'.generated/go2_startup_self_visible_camera_development_v1_attempt_001'
INPUT=ROOT/'.generated/go2_longer_observed_floor_motion_development_v1_attempt_001/fit'
PROTOCOL='docs/go2_ideal_foot_load_reconstruction_development_v1_2026-09-06.md'
IDENTITIES={'launch.json':'6a3c2fbdddb7ffd2db550e7e659db18ef9643c53fdaee09af467106e3ec46ec9',
 'result.json':'5df4a458504ef413b92e3a232c663ddccbda318636cd76b69b58fcd8f74ddc15',
 'sparse_ray_audit_launch.json':'e2473255cb50937e6ca3448e0e3a6ae8646576fb87d639a3317314e52f1bcef5',
 'sparse_ray_audit.json':'bc1df0b190107e246a21ceda56d09d0e59c49bf1b1bb86bd0ce6e6063fc6df42',
 'ray_disagreement_launch.json':'0134d3726f448c68fae0cfbaf95637a80d387ef11594a1f000b8f6987faf9330',
 'ray_disagreement.json':'71a4bfe1b6be79e45a4dc9fffb1a98bf253dadb73568ca879b8eac250b019e3a'}


def preflight():
    ids={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(ids)
    old=read_json(PREVIOUS,'launch.json'); verify(old); inherited=old['source_sha256']; inputs=old['input_sha256']|ids
    for name in ('sparse_ray_audit_launch.json','ray_disagreement_launch.json'):
        audit=read_json(PREVIOUS,name); verify_bindings(audit['source_sha256']|audit['input_sha256'])
        inherited|=audit['source_sha256']; inputs|=audit['input_sha256']
    sources=discover_sources((PROTOCOL,'scripts/reconstruct_go2_ideal_foot_load_development_v1.py',
        'lewm/tests/test_foot_load_sensor_development.py'),inherited)
    for name in ('physics_trace.npz','native_contacts.npz','startup_native_robot_geometry.json','floor_roles.json'):
        if str((INPUT/name).relative_to(ROOT)) not in inputs: raise ValueError('raw source must be inherited bound input')
    launch=old|dict(source_sha256=sources,input_sha256=inputs,scope='new ideal foot-force modality fitting-only offline reconstruction',
        protocol=PROTOCOL,load_threshold_n=5.,dwell_ns=20_000_000,max_age_ns=10_000_000,conditional_force_error_n=0.,
        ideal_transducer_not_vendor_sensor=True,physics_executed=False)
    verify(launch); return launch


def raw_inputs():
    raw=read_npz(INPUT,'physics_trace.npz'); contacts=read_npz(INPUT,'native_contacts.npz')
    n=len(raw['timestamp_s']); times=np.rint(raw['timestamp_s']*1e9).astype(np.int64); offsets=contacts['frame_offsets']
    if n!=30000 or not np.array_equal(times,np.arange(1,n+1)*2_000_000): raise ValueError('exact fitting tape and500Hz clocks required')
    if (offsets.shape!=(n+1,) or offsets.dtype.kind not in 'iu' or offsets[0]!=0 or np.any(np.diff(offsets)<0) or
            any(len(contacts[k])!=offsets[-1] for k in FIELDS)):
        raise ValueError('complete ordered contact acquisition required')
    np.testing.assert_array_equal(contacts['frame_timestamp_s'],raw['timestamp_s'])
    geometry=ArticulatedCollisionGeometry(URDF)
    feet=match_native_foot_geometries(read_json(INPUT,'startup_native_robot_geometry.json'),geometry,
        raw['joint_position'][749],raw['base_pose_world'][749])
    reverse={v:k for k,v in feet['native_foot_geom_to_shape'].items()}; ids=tuple(reverse[f+':0'] for f in FEET)
    return raw,contacts,times,geometry,ids,feet


def reconstruct():
    raw,contacts,times,geometry,ids,feet=raw_inputs(); n=len(times)
    values=np.zeros((n,4,3)); valid=np.zeros((n,4),bool); saturated=valid.copy(); dwell=valid.copy()
    statuses=np.empty((n,4),dtype='U24'); identity='fit-longer-motion-ideal-foot-vector-v1'
    model=LocalLoadHistory(acquisition_identity=identity)
    for i,stamp in enumerate(times):
        a,b=contacts['frame_offsets'][i:i+2]; packet={k:contacts[k][a:b] for k in FIELDS}
        links,_=geometry.transforms(raw['joint_position'][i]); R=rotation_xyzw(raw['base_pose_world'][i,3:])
        rotations=np.array([R@links[foot][:3,:3] for foot in FEET])
        sample=sample_ideal_foot_forces(packet,foot_geom_ids=ids,rotation_world_from_foot=rotations,
            acquisition_identity=identity,measured_ns=int(stamp),available_ns=int(stamp))
        result=model.observe(sample,now_ns=int(stamp),conditional_force_error_n=0.)
        values[i]=sample.force_foot_n; valid[i]=sample.valid; saturated[i]=sample.saturated
        statuses[i]=[r['status'] for r in result['feet']]; dwell[i]=[r['dwell_observed'] for r in result['feet']]
        if (i+1)%5000==0: print('IDEAL_FOOT_RECONSTRUCTION',i+1,flush=True)
    # No terrain-role file, native contact classification or old guard loaded above.
    with (OUTPUT/'sensor_predictions.npz').open('xb') as f:
        np.savez_compressed(f,force_foot_n=values,valid=valid,saturated=saturated,measured_ns=times,
            available_ns=times,status=statuses,dwell_observed=dwell)
    write_json(OUTPUT/'sensor_metadata.json',dict(kind='IDEAL_SIMULATED_THREE_AXIS_NET_FOOT_CONTACT_FORCE',
        acquisition_identity=identity,foot_order=FEET,axes='URDF_foot_link_local',units='simulated_newtons',
        availability='hypothetical_zero_latency_offline_reconstruction',hardware_equivalent=False,
        physical_error_calibrated=False,world_pose_returned=False,terrain_labels_returned=False,
        contact_or_support_permission_returned=False,conditional_force_error_n=0.,threshold_n=5.,dwell_ns=20_000_000))
    write_json(OUTPUT/'acquisition_foot_identity.json',feet)
    return raw,contacts,ids,values,statuses,dwell


def evaluate(contacts,ids,values,statuses,dwell):
    # The newly constructed sensor/prediction artifact already exists on disk.
    if not (OUTPUT/'sensor_predictions.npz').is_file(): raise ValueError('save sensor stream before evaluator role access')
    roles=read_json(INPUT,'floor_roles.json'); ground=set(roles['physical_ground_geom_ids'])
    counters=dict(ground_nonzero_foot_contact_sides=0,other_nonzero_foot_contact_sides=0,self_nonzero_foot_contact_sides=0)
    for side,other in [('a','b'),('b','a')]:
        g=contacts['geom_'+side]; h=contacts['geom_'+other]; f=contacts['force_'+side]
        active=contacts['valid_mask']&np.isin(g,ids)&(np.linalg.norm(f,axis=1)>0)
        counters['ground_nonzero_foot_contact_sides']+=int((active&np.isin(h,list(ground))).sum())
        counters['other_nonzero_foot_contact_sides']+=int((active&~np.isin(h,list(ground))).sum())
        counters['self_nonzero_foot_contact_sides']+=int((active&np.isin(h,ids)).sum())
    above=statuses=='ABOVE_LOAD_THRESHOLD'; count=above.sum(axis=1); stable=dwell.sum(axis=1)
    return dict(samples=len(values),startup_sample=749,
        startup_force_foot_n=values[749].tolist(),startup_resultant_n=np.linalg.norm(values[749],axis=1).tolist(),
        startup_above_threshold=above[749].tolist(),startup_dwell_observed=dwell[749].tolist(),
        sample_counts_by_loaded_feet={str(i):int((count==i).sum()) for i in range(5)},
        sample_counts_by_dwelled_feet={str(i):int((stable==i).sum()) for i in range(5)},
        per_foot_peak_resultant_n=np.linalg.norm(values,axis=2).max(axis=0).tolist(),**counters,
        ground_support_established=False,slip_validation=False,continuous_floor_established=False,
        prospective_footfalls_validated=False,physical_trial_executed=False,navigation_qualified=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive ideal foot-load reconstruction only')
    launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        raw,contacts,ids,values,statuses,dwell=reconstruct(); result=evaluate(contacts,ids,values,statuses,dwell)
        verify(launch); result|=dict(status='IDEAL_FOOT_LOAD_RECONSTRUCTION_COMPLETE',artifact_sha256={n:digest(OUTPUT/n)
            for n in ('sensor_predictions.npz','sensor_metadata.json','acquisition_foot_identity.json')})
        write_json(OUTPUT/'result.json',result); print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_IDEAL_FOOT_RECONSTRUCTION_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()
