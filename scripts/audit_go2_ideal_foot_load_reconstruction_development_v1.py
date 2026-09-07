"""Independent side-wise force accumulation and direct-window load audit."""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.foot_load_sensor_development import FEET
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.reconstruct_go2_ideal_foot_load_development_v1 import OUTPUT,INPUT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources

IDENTITIES={'launch.json':'8428c077fa4b4dec8c6192f65ff9e26e727c141a5b4be39473555141516d4535',
 'result.json':'c0f2de76355e43a231492743feb26e182773ea57baa90e85fb55a340f43a1717'}


def main():
    if (OUTPUT/'force_conservation_audit_launch.json').exists(): raise ValueError('exclusive one-shot force audit')
    bindings={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(bindings)
    launch=read_json(OUTPUT,'launch.json'); verify(launch); result=read_json(OUTPUT,'result.json')
    inputs=launch['input_sha256']|bindings|{str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources(('scripts/audit_go2_ideal_foot_load_reconstruction_development_v1.py',),launch['source_sha256'])
    verify_bindings(sources|inputs)
    write_json(OUTPUT/'force_conservation_audit_launch.json',dict(source_sha256=sources,input_sha256=inputs,
        maximum_coordinate_error_tolerance_n=1e-8,scope='all-frame numerical force bookkeeping, not physical calibration'))
    try:
        raw=read_npz(INPUT,'physics_trace.npz'); contacts=read_npz(INPUT,'native_contacts.npz')
        saved=read_npz(OUTPUT,'sensor_predictions.npz'); geom=ArticulatedCollisionGeometry(URDF)
        foot_identity=read_json(OUTPUT,'acquisition_foot_identity.json')['native_foot_geom_to_shape']
        index={int(k):FEET.index(v.split(':')[0]) for k,v in foot_identity.items()}
        n=len(raw['timestamp_s']); loads=np.zeros((n,4),bool); maxima=np.zeros(4); side_count=0
        times=np.arange(1,n+1)*2_000_000
        np.testing.assert_array_equal(saved['measured_ns'],times); np.testing.assert_array_equal(saved['available_ns'],times)
        assert saved['valid'].all() and not saved['saturated'].any()
        for i in range(n):
            world=np.zeros((4,3),np.longdouble); start,end=contacts['frame_offsets'][i:i+2]
            for c in range(start,end):
                if not contacts['valid_mask'][c]: continue
                for side in ('a','b'):
                    g=int(contacts['geom_'+side][c])
                    if g in index:
                        world[index[g]]+=contacts['force_'+side][c].astype(np.longdouble); side_count+=1
            # Different root-quaternion implementation; URDF joint kinematics shared.
            R=Rotation.from_quat(raw['base_pose_world'][i,3:]).as_matrix()
            links,_=geom.transforms(raw['joint_position'][i])
            for f,foot in enumerate(FEET):
                axes=(R@links[foot][:3,:3]).astype(np.longdouble)
                recovered=axes@saved['force_foot_n'][i,f].astype(np.longdouble)
                error=float(np.max(np.abs(recovered-world[f]))); maxima[f]=max(maxima[f],error)
                if error>1e-8: raise ValueError('force coordinate conservation failed')
                loads[i,f]=np.sqrt(np.sum(world[f]**2))>5
            if (i+1)%10000==0: print('FORCE_CONSERVATION_AUDIT',i+1,flush=True)
        np.testing.assert_array_equal(saved['status'],np.where(loads,'ABOVE_LOAD_THRESHOLD','BELOW_LOAD_THRESHOLD'))
        # Direct 11-sample windows, not the production state machine.
        dwell=np.zeros_like(loads)
        for i in range(10,n): dwell[i]=loads[i-10:i+1].all(axis=0)
        np.testing.assert_array_equal(saved['dwell_observed'],dwell)
        verify_bindings(sources|inputs)
        report=dict(status='FORCE_CONSERVATION_AND_DWELL_AUDIT_PASS',samples=n,foot_vectors=n*4,
            incident_foot_contact_sides=side_count,maximum_coordinate_error_per_foot_n=maxima.tolist(),
            direct_dwell_windows=(n-10)*4,all_statuses_and_dwell_exact=True,ground_roles_used=False,
            shared_urdf_joint_kinematics=True,physical_sensor_calibrated=False,navigation_qualified=False)
        write_json(OUTPUT/'force_conservation_audit.json',report); print(json.dumps(report),flush=True)
    except Exception as error:
        write_json(OUTPUT/'force_conservation_audit_failure.json',dict(status='TERMINAL_FORCE_AUDIT_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()
