"""Evaluator-only checks for the NEW static motion-calibration condition."""
from dataclasses import asdict
import json

import numpy as np

from lewm.action_motion_identification_development import motion_priors
from lewm.physical_execution_development import rotation_xyzw
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot
from scripts.audit_go2_startup_observation_turn_development_v1 import padded_body_inside_setup
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_observation_turn_session_development import StartupObservationSession, admit_setup


class MotionSession(StartupObservationSession):
    def __init__(self,*args,**kwargs):
        self.motion_setup=None; self.motion_region_rows=[]
        super().__init__(*args,**kwargs)

    def _sample(self,requested,applied,timestamp_s):
        row=super()._sample(requested,applied,timestamp_s)
        if self.motion_setup is not None:
            geometry,region,initial=self.motion_setup
            inside=padded_body_inside_setup(geometry,region,initial,row['base_pose_world'],row['joint_position'])
            self.motion_region_rows.append(dict(sample_index=len(self.samples)-1,inside=bool(inside)))
            if not inside: raise PhysicalStop('MOTION_PADDED_BODY_OUTSIDE_CHECKED_REGION')
        return row


def admit_motion_setup(session,definition_sha256):
    # Retain the original instantaneous setup check as an explicit witness;
    # independently check the NEW region, never silently extend its prior.
    geometry,_,_,original=admit_setup(session,definition_sha256)
    raw=session.samples[-1]; epoch=int(round(raw['timestamp_s']*1e9))
    velocity,region=motion_priors(epoch,definition_sha256)
    static=json.loads((session.output/'static_objects.json').read_text())
    pose=raw['base_pose_world']
    check=check_setup_snapshot(velocity,region,identity=(0,0,0),measured_ns=epoch,
        position_world_m=pose[:3],rotation_world_from_initial_body=rotation_xyzw(pose[3:]),
        velocity_world_m_s=raw['base_twist_world'][:3],native_static_boxes=static,
        expected_nonfloor_names=tuple(r['native_name'] for r in static),geometry=geometry,joint_position=raw['joint_position'])
    report=dict(new_region=asdict(region),check=check,inherited_initial_checks_sha256=original['checks_sha256'],
        static_scene_through_expiry_is_declared_condition=True,scope='new calibration-arena assumption; not sensor data or maze prior')
    write_json(session.output/'motion_setup_checks.json',report)
    if not check['velocity_and_nonfloor_setup_checks_pass']: raise PhysicalStop('MOTION_SETUP_REJECTED')
    admission=original | {'checks_sha256':digest(session.output/'motion_setup_checks.json')}
    write_json(session.output/'motion_admission.json',admission)
    session.motion_setup=(geometry,region,np.asarray(pose).copy())
    return geometry,velocity,region,admission
