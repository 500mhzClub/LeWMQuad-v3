"""Live all-load acquisition and frozen support predictions outside controller."""
import numpy as np

from lewm.foot_load_sensor_development import FEET
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries,nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.simulated_body_observation_development import CALIBRATION
from lewm.simulated_foot_force_development import FIELDS,sample_ideal_foot_forces
from lewm.support_kinematics_development import FootJacobians,CausalQuietUp,predict_support_motion
from lewm.longer_motion_collection_development import validate_command
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_session_development import RGBDSession
from scripts.rgbd_shadow_motion_session_development import AppearanceRGBDSession
from scripts.run_go2_contact_attributed_execution_development_v1 import array,PhysicalStop
from scripts.support_friction_physical_init_development import SupportFrictionPhysicalInit


class SupportFrictionSession(RGBDSession,SupportFrictionPhysicalInit):
    _build_contact_topology=AppearanceRGBDSession._build_contact_topology
    install_contact_identity=AppearanceRGBDSession.install_contact_identity
    capture_fixed_rgb=AppearanceRGBDSession.capture_fixed_rgb

    def __init__(self,*args,**kwargs):
        self.guard=None; self.guard_rows=[]; self.load_samples=[]; self.support_rows=[]; self.support_failure=None
        self.kinematics=FootJacobians(URDF); self.up_model=CausalQuietUp(); self.foot_ids=None
        super().__init__(*args,**kwargs)

    def install_sensor_identity(self):
        robot=self.ctx.build.robot; pos=array(robot.get_pos()).reshape(3); quat=array(robot.get_quat()).reshape(4)
        q=array(robot.get_dofs_position(self.ctx.runner._leg_dof_idx.tolist())).reshape(12)
        pose=np.r_[pos,quat[[1,2,3,0]]]
        self.foot_identity=match_native_foot_geometries(capture_native_robot_geometry(robot),self.kinematics.geometry,q,pose)
        reverse={v:k for k,v in self.foot_identity['native_foot_geom_to_shape'].items()}
        self.foot_ids=tuple(reverse[f+':0'] for f in FEET)

    def command_tick(self,requested): return super().command_tick(validate_command(requested))

    def _sample(self,requested,applied,timestamp_s):
        before=len(self.samples)
        try:
            row=super()._sample(requested,applied,timestamp_s)
            if self.guard is not None:
                packet={k:np.asarray(v)[0] for k,v in self.packets[-1].items()}
                nonfeet=nonfoot_ground_contact_indices(packet,**self.guard); speed=float(np.linalg.norm(row['base_twist_world'][:3]))
                inside=bool((np.abs(row['base_pose_world'][:2])<8).all())
                self.guard_rows.append(dict(sample_index=len(self.samples)-1,nonfoot_ground_contact_indices=nonfeet,
                    base_speed_m_s=speed,in_domain=inside,evaluator_only=True))
                if nonfeet or speed>.3 or not inside: raise PhysicalStop('SUPPORT_CHALLENGE_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP')
            return row
        finally:
            if len(self.samples)>before: self.capture_foot_sample(self.samples[-1])

    def capture_foot_sample(self,row):
        if self.foot_ids is None: raise ValueError('verified transducer identities before physics required')
        stamp=int(round(row['timestamp_s']*1e9)); links,_=self.kinematics.geometry.transforms(row['joint_position'])
        R=rotation_xyzw(row['base_pose_world'][3:]); axes=np.array([R@links[f][:3,:3] for f in FEET])
        packet={k:np.asarray(self.packets[-1][k])[0] for k in FIELDS}; identity=self.spec['ideal_foot_sensor_identity']
        sample=sample_ideal_foot_forces(packet,foot_geom_ids=self.foot_ids,rotation_world_from_foot=axes,
            acquisition_identity=identity,measured_ns=stamp,available_ns=stamp); self.load_samples.append(sample)
        # End of acquisition boundary: downstream code uses sensor rows only.
        if stamp<1_300_000_000 or self.support_failure is not None: return
        try:
            fast=self.fast_rows[-1]
            if int(fast['measured_ns'])!=stamp or not fast['valid'].all(): raise ValueError('current valid live gyro required')
            slow=self.sensor_rows[-1] if stamp%20_000_000==0 else None
            if slow is not None and (int(slow['measured_ns'])!=stamp or not all(slow[k].all() for k in ('specific_force_valid','joints_valid','gyro_valid'))):
                raise ValueError('co-timed valid live body samples required')
            up=self.up_model.update(stamp,fast['values'],slow['specific_force_values'] if slow is not None else None)
            if slow is None or up is None: return
            body=dict(identity=identity,calibration_id=CALIBRATION,measured_ns=stamp,available_ns=stamp,
                q=slow['joints_values'][:12],dq=slow['joints_values'][12:],gyro=slow['gyro_values'],
                specific_force=slow['specific_force_values'],valid=True)
            predicted=predict_support_motion(self.kinematics,body,self.load_samples[-11:],identity=identity,up_body=up)
            predicted['rotation_gyro_anchor_from_body']=self.up_model.Q.tolist(); self.support_rows.append(predicted)
        except (ValueError,TypeError,KeyError) as error:
            self.support_failure=dict(measured_ns=stamp,reason=repr(error),latched_no_restart=True)

    def persist_loads(self,output):
        with (output/'live_foot_sensor.npz').open('xb') as stream:
            np.savez_compressed(stream,measured_ns=[s.measured_ns for s in self.load_samples],
                available_ns=[s.available_ns for s in self.load_samples],force_foot_n=[s.force_foot_n for s in self.load_samples],
                valid=[s.valid for s in self.load_samples],saturated=[s.saturated for s in self.load_samples])
