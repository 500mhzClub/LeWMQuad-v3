"""All-return bound memory and exact nominal sphere queries; no ground waiver."""
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_visual_surface_memory_development import JointVisualSurfaceMemory
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex


class SampleBoundsSurfaceMemory(JointVisualSurfaceMemory):
    def __init__(self,*,identity):
        super().__init__(identity=identity)
        self.index=MeasuredSampleBoundsIndex()

    def footprint(self,geometry,displacement_body_xy,yaw_rad,*,now_ns,persistent=True):
        if persistent is not True:raise SensorContractError('explicit persistent all-return bounds required')
        result=super().footprint(geometry,displacement_body_xy,yaw_rad,now_ns=now_ns,persistent=True)
        c,s=np.cos(yaw_rad),np.sin(yaw_rad)
        R=self.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        p=self.position+self.rotation@np.r_[displacement_body_xy,0.]
        shapes=geometry.supports(self.joints,R)['shapes'];primitives={s['shape_id']:s for s in geometry._shapes}
        refined=[]
        for shape,box in zip(shapes,result['shapes'],strict=True):
            primitive=primitives[shape['shape_id']]
            if primitive['kind']=='sphere':
                check=self.index.intersect_sphere(p+R@np.asarray(shape['center_body_m']),float(primitive['dimensions'][0]))
                if check['intersecting_voxels']>box['intersecting_voxels']:
                    raise SensorContractError('sphere query escaped its measured-bound box enclosure')
                refined.append(dict(shape_id=shape['shape_id'],**check))
            else:refined.append(box)
        return result|dict(shapes=refined,possible_intersection=any(r['intersecting_voxels'] for r in refined),
            sample_bounds_aabb_shapes=result['shapes'],sample_bounds_aabb_possible_intersection=result['possible_intersection'],
            whole_voxel_possible_intersection=any(r['whole_voxel_intersections'] for r in result['shapes']),
            measured_point_enclosures_only=True,all_returns_retained_in_enclosures=True,
            sensor_or_pose_uncertainty_envelope=False,ground_contact_waiver=False,
            exact_sphere_refinement=True,other_primitives_use_conservative_aabb=True)
