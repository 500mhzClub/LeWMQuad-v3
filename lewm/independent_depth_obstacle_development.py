"""Current paired-depth obstacle evidence without visual pose or routing input.

Quiet initial specific force and the uninterrupted public gyro stream provide
only the candidate up direction. Current depth must independently fit the
strict common plane. This is an uncalibrated flat-floor hypothesis and a
sampled obstacle veto, not a future clearance/support certificate.
"""
import numpy as np

from lewm.camera_independent_gyro_development import CameraIndependentGyro
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,body_points
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.auxiliary_downward45_depth_observation_development import body_points as auxiliary_points
from lewm.sampled_plane_candidates_development import measured_candidates
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.fresh_obstacle_dispatch_development import CurrentObstacles,CELL_M


class IndependentDepthObstacles:
    def __init__(self):
        self.gyro=CameraIndependentGyro();self.initial_up=None;self.frames=0
        self.failed=False;self.receipts=[]

    def observe(self,policy,depth,fast,*,auxiliary_depth,measured_ns):
        if self.failed:raise ValueError('independent depth observer fault latched')
        try:return self._observe(policy,depth,fast,auxiliary_depth,measured_ns)
        except Exception:
            self.failed=True;raise

    def _observe(self,policy,depth,fast,auxiliary_depth,now):
        if now!=1_500_000_000+self.frames*100_000_000:
            raise ValueError('uninterrupted actual paired-depth stream required')
        gyro=self.gyro.observe(policy,fast,now_ns=now)
        if self.initial_up is None:
            force=policy['sensor_state']['sensed']['specific_force']
            command=policy['sensor_state']['control']['applied_command']
            up=force['values'].mean(0);magnitude=np.linalg.norm(up)
            if (not force['valid'].all() or not command['valid'].all()
                    or np.any(np.abs(command['values'])>1e-8) or not 8<=magnitude<=12):
                raise ValueError('quiet initial public gravity reference required')
            self.initial_up=up/magnitude
        up=np.asarray(gyro['rotation_initial_body_from_current_body']).T@self.initial_up
        candidates=[];clouds=[]
        for packet,E,project in ((depth,np.asarray(BODY_FROM_OPTICAL),body_points),
                (auxiliary_depth,body_from_optical(),auxiliary_points)):
            cloud=project(packet,policy,now_ns=now,stride=4)
            if packet['measured_ns']!=now:raise ValueError('current actual paired depth required')
            clouds.append(cloud['points_body_m'][cloud['valid']])
            candidates.append(measured_candidates(packet['depth_m'],packet['valid'],E,up)[0])
        plane=fit_joint_plane(*candidates,up)
        frame=self.frames;self.frames+=1
        self.receipts.append(dict(frame=frame,measured_ns=now,joint_plane=plane,
            gyro_intervals=gyro['samples_integrated'],initial_up_body=self.initial_up.tolist(),
            visual_pose_used=False,native_pose_used=False,gyro_bias_calibrated=False,
            current_plane_reused_from_history=False))
        if not plane['available'] or any(not len(p) for p in clouds):return None
        normal=np.asarray(plane['normal_body']);cells=set()
        for points in clouds:
            height=points@normal+plane['offset_body_m']
            above=points[(height>.03)&(height<.65)]
            cells.update(tuple(map(int,k)) for k in np.unique(np.floor(above[:,:2]/CELL_M).astype(int),axis=0))
        return CurrentObstacles(frame,now,(0.,0.,0.),tuple(map(tuple,np.eye(3))),
            frozenset(cells),tuple(map(len,clouds)),'current_body')
