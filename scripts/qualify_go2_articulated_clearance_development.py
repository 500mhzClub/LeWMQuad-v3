"""Bounded all-shape wall separation using existing URDF support functions.

Face-normal separation is a LOWER bound on positive Euclidean distance. An
overlapping projection is unresolved, not a collision proof. No dynamics or
unverified speed bound is invented to turn endpoints into a swept guarantee.
"""
import json
from pathlib import Path

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_collision_grouping_development import resolve_native_groups
from lewm.physical_execution_development import rotation_xyzw
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry


class ClearanceQualification:
    def __init__(self, session, spec, *, tolerance_m):
        self.model=ArticulatedCollisionGeometry(URDF)
        self.walls=spec['geometry']['wall_boxes']
        self.centres=np.array([w['centre_xyz'] for w in self.walls])
        self.halves=np.array([w['size_xyz'] for w in self.walls])/2
        axes=[]
        for wall in self.walls:
            c,s=np.cos(wall['yaw_rad']),np.sin(wall['yaw_rad'])
            axes.append([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
        self.axes=np.array(axes);self.normals=self.axes.reshape(-1,3)
        self.projected_centres=np.einsum('wij,wj->wi',self.axes,self.centres)
        native=capture_native_robot_geometry(session.ctx.build.robot)
        sample=session.samples[-1];pose=sample['base_pose_world'];Q=rotation_xyzw(pose[3:])
        transforms,unused=self.model.transforms(sample['joint_position'])
        groups=resolve_native_groups(URDF,self.model.supports(sample['joint_position'],np.eye(3))['shapes'],{r['link_name'] for r in native})
        used=set();bindings=[]
        for shape in self.model._shapes:
            T=transforms[shape['link']]@shape['origin'];p=pose[:3]+Q@T[:3,3]
            candidates=[r for r in native if r['geom_id'] not in used and r['link_name']==groups[shape['shape_id']] and r['geom_type'].lower()==shape['kind']]
            if not candidates:raise ValueError('native collision primitive identity missing: '+shape['shape_id'])
            found=min(candidates,key=lambda r:np.linalg.norm(np.array(r['position_world_m'])-p))
            position_error=float(np.linalg.norm(np.array(found['position_world_m'])-p))
            dimensions_error=float(np.max(np.abs(np.array(found['data'][:len(shape['dimensions'])])-shape['dimensions'])))
            nativeQ=rotation_xyzw(np.array(found['quaternion_world_wxyz'])[[1,2,3,0]])
            rotation_error=float(np.max(np.abs(nativeQ-Q@T[:3,:3])))
            if max(position_error,dimensions_error,rotation_error)>tolerance_m:
                raise ValueError(f'native/URDF geometry mismatch {shape["shape_id"]}: {position_error},{dimensions_error},{rotation_error}')
            used.add(found['geom_id']);bindings.append(dict(shape_id=shape['shape_id'],native=found,position_error_m=position_error,dimensions_error_m=dimensions_error,rotation_matrix_error=rotation_error))
        if len(used)!=len(native):raise ValueError('unmapped native robot collision primitive')
        self.binding=dict(primitive_count=len(bindings),bindings=bindings,all_feet_included=True,
            ground_excluded_from_wall_geometry=True,support_contact_definition_unchanged=True,
            internal_substeps=int(session.ctx.build.scene.substeps),physics_step_s=.002)
        if self.binding['internal_substeps']!=1:
            raise ValueError('2-ms recordings do not cover every internal simulator substep')

    def trace(self, arrays, contacts, *, budget):
        rows=[];disagreements=[]
        events={round(r['timestamp_s']*1e9):r['disallowed_contacts'] for r in contacts}
        for index,(pose,q,stamp) in enumerate(zip(arrays['base_pose_world'],arrays['joint_position'],arrays['timestamp_s'])):
            if index%100==0:budget.check('articulated_clearance')
            Q=rotation_xyzw(pose[3:])
            result=self.model.supports(q,self.normals@Q)
            translation=self.normals@pose[:3]
            low=np.array([s['lower'] for s in result['shapes']])+translation
            high=np.array([s['upper'] for s in result['shapes']])+translation
            low=low.reshape(len(result['shapes']),len(self.walls),3)
            high=high.reshape(low.shape)
            gap=np.maximum(low-(self.projected_centres+self.halves),self.projected_centres-self.halves-high).max(axis=2)
            pi,wi=np.unravel_index(np.argmin(gap),gap.shape)
            value=float(gap[pi,wi]);event=events.get(round(stamp*1e9),[])
            if event and value>1e-5:
                # Ground/self contacts may be disallowed independently of maze.
                disagreements.append(dict(sample=index,wall_lower_bound_m=value,contacts=event,
                    reason='Inspect identities: positive wall separation can coexist with disallowed non-wall contact'))
            rows.append(dict(sample=index,timestamp_s=float(stamp),wall_separation_lower_bound_m=value,
                nearest_bound_shape=result['shapes'][pi]['shape_id'],wall_id=self.walls[wi]['wall_id'],
                sampled_5mm_bound_pass=value>=.005,sampled_20mm_bound_pass=value>=.02,
                native_disallowed_contact=bool(arrays['physics_contact'][index]),
                attributed_disallowed_contacts=event))
        return dict(status='STEPWISE_BOUNDS_ONLY_CONTINUOUS_CLEARANCE_UNRESOLVED',samples=len(rows),
            rows=rows,minimum_stepwise_bound_m=min(r['wall_separation_lower_bound_m'] for r in rows),
            projection_overlap_means='unresolved; not a demonstrated intersection',
            cross_check_disagreements=disagreements,interval_count=max(0,len(rows)-1),
            unresolved_intervals=list(range(max(0,len(rows)-1))),
            unresolved_reason='No verified bound on continuous base angular/linear and joint point speeds between native steps. Endpoint velocities are not such a bound.',
            all_robot_primitives_including_feet=True,operating_swept_margin_qualified=False,
            actual_contact_identity_records_preserved=True,physics_or_models_executed=False)
