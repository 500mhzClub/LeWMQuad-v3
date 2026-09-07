"""All-row numerical differentiation and separate quaternion-integration audit."""
import json
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.foot_load_sensor_development import FEET
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_causal_support_kinematics_development_v1 import OUTPUT,INPUT,PREVIOUS,MODES
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources

IDENTITIES={'launch.json':'88b6cdbf38cf20b6d2af3537474fe2e1e1f75b7221e7cbcb10d66b558b8f4132',
 'result.json':'957dbafe351941c4e78f8be12a07778381526e607d99bc1fa52a444d484763f1'}


def main():
    if (OUTPUT/'derivative_up_audit_launch.json').exists(): raise ValueError('exclusive derivative audit')
    ids={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(ids)
    launch=read_json(OUTPUT,'launch.json'); verify(launch); result=read_json(OUTPUT,'result.json')
    inputs=launch['input_sha256']|ids|{str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources(('scripts/audit_go2_causal_support_kinematics_development_v1.py',),launch['source_sha256'])
    verify_bindings(sources|inputs)
    write_json(OUTPUT/'derivative_up_audit_launch.json',dict(source_sha256=sources,input_sha256=inputs,
        directional_difference_step_s=1e-5,velocity_tolerance_m_s=2e-7,
        scope='numerical differentiation and quaternion implementation checks, not physical model validation'))
    try:
        rows=read_json(OUTPUT,'predictions.json')['rows']; body=read_npz(INPUT,'ideal_sensor_samples.npz')
        fast=read_npz(INPUT,'fast_gyro_samples.npz'); loads=read_npz(PREVIOUS,'sensor_predictions.npz')
        geom=ArticulatedCollisionGeometry(URDF); shapes={s['link']:s for s in geom._shapes if s['link'] in FEET}
        def pose(q):
            links,_=geom.transforms(q)
            return (np.array([(links[f]@shapes[f]['origin'])[:3,3] for f in FEET]),
                    np.array([links[f][:3,:3] for f in FEET]))
        orientation=Rotation.identity(); forces=[]; up_anchor=None; observations={r['sensor_sample']:r for r in rows}
        up_max=rotation_max=linear_max=angular_max=prediction_max=0.; checked=0
        for i in range(649,30000):
            if i>649: orientation=orientation*Rotation.from_rotvec((fast['values'][i-1]+fast['values'][i])*.001)
            R=orientation.as_matrix(); stamp=int(fast['measured_ns'][i]); b=(i+1)//10-1
            if stamp%20_000_000==0 and stamp<=1_500_000_000: forces.append(R@body['specific_force_values'][b])
            if stamp==1_500_000_000:
                assert len(forces)==11; up_anchor=np.mean(forces,axis=0); up_anchor/=np.linalg.norm(up_anchor)
            if i not in observations: continue
            row=observations[i]; up=R.T@up_anchor
            rotation_max=max(rotation_max,float(np.max(np.abs(R-row['rotation_gyro_anchor_from_body']))))
            up_max=max(up_max,float(np.max(np.abs(up-row['conditional_up_body']))))
            q,dq=body['joints_values'][b,:12],body['joints_values'][b,12:]; omega=body['gyro_values'][b]
            p,A=pose(q); h=1e-5; pplus,Aplus=pose(q+h*dq); pminus,Aminus=pose(q-h*dq)
            pdot=(pplus-pminus)/(2*h); W=((Aplus-Aminus)/(2*h))@A.transpose(0,2,1)
            angular=W[:,[2,0,1],[1,2,0]]
            expected_centre=-(np.cross(omega,p)+pdot)
            expected_rolling=expected_centre+.022*np.cross(omega+angular,up)
            actual_centre=np.array(row['modes']['stationary_centre']['per_foot_velocity_body_m_s'])
            actual_rolling=np.array(row['modes']['level_sphere_rolling']['per_foot_velocity_body_m_s'])
            error=max(float(np.max(np.abs(expected_centre-actual_centre))),float(np.max(np.abs(expected_rolling-actual_rolling))))
            prediction_max=max(prediction_max,error)
            if error>2e-7: raise ValueError('directional derivative prediction disagreement')
            selected=(loads['valid'][i-10:i+1]&~loads['saturated'][i-10:i+1]&
                (np.linalg.norm(loads['force_foot_n'][i-10:i+1],axis=2)>5)).all(axis=0)
            np.testing.assert_array_equal(selected,row['selected_loaded_feet'])
            np.testing.assert_allclose(p,row['foot_position_body_m'],atol=1e-12,rtol=0)
            for mode,expected in zip(MODES,(expected_centre,expected_rolling),strict=True):
                mean=row['modes'][mode]['consensus_velocity_body_m_s']
                if selected.sum()<2: assert mean is None
                else: np.testing.assert_allclose(expected[selected].mean(axis=0),mean,atol=2e-7,rtol=0)
            checked+=1
        if up_max>1e-10 or rotation_max>1e-10: raise ValueError('separate gyro/up reconstruction differs')
        verify_bindings(sources|inputs)
        report=dict(status='DERIVATIVE_AND_UP_AUDIT_PASS',rows=checked,per_foot_hypothesis_vectors=checked*8,
            maximum_directional_difference_prediction_error_m_s=prediction_max,maximum_rotation_coordinate_difference=rotation_max,
            maximum_up_coordinate_difference=up_max,all_load_selection_and_consensus_verified=True,
            shared_urdf_forward_kinematics=True,native_pose_or_contacts_loaded=False,physical_error_calibrated=False)
        write_json(OUTPUT/'derivative_up_audit.json',report); print(json.dumps(report),flush=True)
    except Exception as error:
        write_json(OUTPUT/'derivative_up_audit_failure.json',dict(status='TERMINAL_DERIVATIVE_AUDIT_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()
