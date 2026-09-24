"""Explicit nominal foot/floor contact handling with all other evidence retained."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import body_points
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.measured_floor_partition_development import MeasuredFloorPartition,FOOT_IDS,foot_projection_coverage
from lewm.sample_bounds_surface_memory_development import SampleBoundsSurfaceMemory


def apply_floor_contact_rule(original,geometry,joints,R,p,B,floor_cells,partition):
    shapes=geometry.supports(joints,R)['shapes'];primitives={s['shape_id']:s for s in geometry._shapes}
    revised=[];contacts=[]
    for shape,hit in zip(shapes,original['shapes'],strict=True):
        key=shape['shape_id']
        if key!=hit['shape_id']:raise SensorContractError('ordered primitive evidence required')
        if key not in FOOT_IDS:revised.append(deepcopy(hit));continue
        primitive=primitives[key]
        if primitive['kind']!='sphere' or float(primitive['dimensions'][0])!=.022:
            raise SensorContractError('exact reviewed foot sphere required')
        center=p+R@np.asarray(shape['center_body_m']);radius=float(primitive['dimensions'][0])
        ground=partition.floor.intersect_sphere(center,radius);other=partition.other.intersect_sphere(center,radius)
        if max(ground['intersecting_voxels'],other['intersecting_voxels'])>hit['intersecting_voxels']:
            raise SensorContractError('typed query escaped its all-return enclosure')
        coverage=foot_projection_coverage((B@center)[:2],radius,floor_cells)
        eligible=coverage['entire_nominal_projection_on_measured_floor']
        revised.append(dict(shape_id=key,**other) if eligible else deepcopy(hit))
        contacts.append(dict(shape_id=key,floor=ground,other_or_unknown=other,projection=coverage,
            measured_floor_contact_rule_eligible=eligible,non_floor_or_unknown_contact_permitted=False))
    return original|dict(shapes=revised,possible_intersection=any(r['intersecting_voxels'] for r in revised),
        all_return_shapes=deepcopy(original['shapes']),all_return_possible_intersection=original['possible_intersection'],
        foot_floor_contacts=contacts,ground_contact_waiver=True,
        ground_contact_scope='four_foot_spheres_only_when_entire_nominal_projection_is_measured_floor',
        non_foot_contacts_exempted=False,non_floor_or_unknown_contacts_exempted=False,
        ground_support_approved=False,terrain_or_pose_uncertainty_certified=False)


class MeasuredFloorContactMemory(SampleBoundsSurfaceMemory):
    def __init__(self,*,identity):
        super().__init__(identity=identity)
        self.partition=MeasuredFloorPartition();self.classification_receipt=None
        self.classified_ns=None;self.floor_cells=None;self.map_from_initial=None

    def classify_current(self,policy,depth,B,floor_height,floor_cells,*,now_ns):
        self._current(now_ns)
        if self.classified_ns is not None and now_ns-self.classified_ns!=100_000_000:
            raise SensorContractError('uninterrupted per-return classification required')
        cloud=body_points(depth,policy,now_ns=now_ns,stride=4)
        patch=sampled_floor_patch(depth['depth_m'],depth['valid'],B@self.rotation,B@self.position,
            floor_height,cloud['rows'],cloud['columns'])
        points=cloud['points_body_m'][cloud['valid']]@self.rotation.T+self.position
        mask=patch['measured_floor_patch'][cloud['valid']]
        witness={k:self.route[-1][k] for k in ('frame','measured_ns','rgb_sha256','depth_sha256')}
        self.partition.insert(points,mask,witness)
        if self.partition.total_returns!=sum(self.index.sample_counts.values()):
            raise SensorContractError('every accumulated return must have exactly one classification')
        self.classified_ns=now_ns;self.map_from_initial=B.copy();self.floor_cells=floor_cells
        self.classification_receipt=dict(**witness,total_returns=self.partition.total_returns,
            floor_returns=self.partition.floor_returns,other_returns=self.partition.other_returns,
            current_floor_returns=int(mask.sum()),current_other_returns=int((~mask).sum()),
            floor_voxels=len(self.partition.floor.cells),other_voxels=len(self.partition.other.cells),
            first_witness_never_classifies_later_returns=True,ground_support_approved=False)

    def footprint(self,geometry,displacement_body_xy,yaw_rad,*,now_ns,persistent=True):
        if self.classified_ns!=now_ns:raise SensorContractError('current complete return classification required')
        original=super().footprint(geometry,displacement_body_xy,yaw_rad,now_ns=now_ns,persistent=persistent)
        c,s=np.cos(yaw_rad),np.sin(yaw_rad);R=self.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        p=self.position+self.rotation@np.r_[displacement_body_xy,0.]
        return apply_floor_contact_rule(original,geometry,self.joints,R,p,self.map_from_initial,self.floor_cells,self.partition)
