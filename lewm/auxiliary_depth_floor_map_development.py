"""Retain all auxiliary returns and measured floor, using the current observed pose."""
from copy import deepcopy
import hashlib
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.auxiliary_tilted_depth_observation_development import validate_depth,body_points,CALIBRATION_ID
from lewm.auxiliary_tilted_depth_geometry_development import reference_pose
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.measured_floor_partition_development import MeasuredFloorPartition,FOOT_IDS,foot_projection_coverage
from lewm.retained_floor_patch_development import RetainedFloorPatches
from lewm.retained_patch_contact_goal_probe_development import RetainedPatchContactMemory,RetainedPatchContactMap
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.joint_visual_floor_map_development import floor_coverage,GRID,CELL_M


class AuxiliaryDepthSurfaceMemory(RetainedPatchContactMemory):
    def __init__(self,*,identity):
        super().__init__(identity=identity)
        self.auxiliary_index=MeasuredSampleBoundsIndex();self.auxiliary_partition=MeasuredFloorPartition()
        self.auxiliary_patches=RetainedFloorPatches();self.auxiliary_ns=None;self.auxiliary_receipt=None

    def observe_auxiliary(self,policy,depth,B,floor_height,floor_cells,occupied,*,now_ns):
        self._current(now_ns);validate_depth(depth,policy,now_ns=now_ns)
        if (depth['measured_ns']!=now_ns or len(self.auxiliary_patches.frames)!=len(self.route)-1
                or self.classified_ns!=now_ns or (self.auxiliary_ns is not None and now_ns-self.auxiliary_ns!=100_000_000)):
            raise SensorContractError('one current auxiliary observation per admitted primary frame required')
        R=B@self.rotation;p=B@self.position;Q,q=reference_pose(R,p)
        witness=dict(frame=self.route[-1]['frame'],measured_ns=now_ns,calibration_id=CALIBRATION_ID,
            rgb_sha256=depth['primary_rgb_sha256'],
            depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest())
        cloud=body_points(depth,policy,now_ns=now_ns,stride=4)
        points=cloud['points_body_m'][cloud['valid']]@self.rotation.T+self.position
        patch=sampled_floor_patch(depth['depth_m'],depth['valid'],Q,q,floor_height,cloud['rows'],cloud['columns'])
        mask=patch['measured_floor_patch'][cloud['valid']]
        self.auxiliary_index.insert(points,witness);self.auxiliary_partition.insert(points,mask,witness)
        if self.auxiliary_partition.total_returns!=sum(self.auxiliary_index.sample_counts.values()):
            raise SensorContractError('every auxiliary return requires exactly one classification')
        self.auxiliary_patches.append(depth['depth_m'],depth['valid'],Q,q,floor_height,witness)
        coverage=floor_coverage(depth['depth_m'],depth['valid'],Q,q,floor_height)
        for cell in GRID[coverage['covered']]:floor_cells.setdefault(tuple(map(int,cell)),witness['frame'])
        mapped=points@B.T;above=mapped[(mapped[:,2]>floor_height+.03)&(mapped[:,2]<floor_height+.65)]
        keys=np.floor(above[:,:2]/CELL_M).astype(int)
        for cell in np.unique(keys,axis=0):
            if np.all(cell>=-100) and np.all(cell<100):occupied.setdefault(tuple(map(int,cell)),witness['frame'])
        self.auxiliary_ns=now_ns
        self.auxiliary_receipt=dict(**witness,current_returns=len(points),current_floor_returns=int(mask.sum()),
            current_other_returns=int((~mask).sum()),total_returns=self.auxiliary_partition.total_returns,
            floor_returns=self.auxiliary_partition.floor_returns,other_returns=self.auxiliary_partition.other_returns,
            retained_auxiliary_voxels=len(self.auxiliary_index.cells),
            current_observed_floor_cells=int(coverage['covered'].sum()),retained_patch_frames=len(self.auxiliary_patches.frames),
            native_pose_used=False,segmentation_used=False,all_sampled_returns_retained=True,
            unknown_rays_inferred_free=False,ground_support_approved=False)
        return deepcopy(self.auxiliary_receipt)

    def footprint(self,geometry,displacement_body_xy,yaw_rad,*,now_ns,persistent=True):
        if self.auxiliary_ns!=now_ns:raise SensorContractError('current auxiliary classification required for collision checks')
        original=super().footprint(geometry,displacement_body_xy,yaw_rad,now_ns=now_ns,persistent=persistent)
        c,s=np.cos(yaw_rad),np.sin(yaw_rad);R=self.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        p=self.position+self.rotation@np.r_[displacement_body_xy,0.]
        shapes=geometry.supports(self.joints,R)['shapes'];primitives={v['shape_id']:v for v in geometry._shapes}
        foot_centres={v['shape_id']:(self.map_from_initial@(p+R@np.asarray(v['center_body_m'])))[:2]
            for v in shapes if v['shape_id'] in FOOT_IDS}
        patches=dict(zip(foot_centres,self.auxiliary_patches.coverage(list(foot_centres.values())),strict=True))
        primary_patches=dict(zip(foot_centres,self.patches.coverage(list(foot_centres.values())),strict=True))
        primary={v['shape_id']:deepcopy(v) for v in original['shapes']}
        old_rules={v['shape_id']:v for v in original['foot_floor_contacts']};auxiliary=[];rules=[]
        for shape in shapes:
            key=shape['shape_id'];primitive=primitives[key]
            if primitive['kind']=='sphere':
                centre=p+R@np.asarray(shape['center_body_m']);radius=float(primitive['dimensions'][0])
                hit=self.auxiliary_index.intersect_sphere(centre,radius)
            else:hit=self.auxiliary_index.intersect(np.asarray(shape['lower'])+p,np.asarray(shape['upper'])+p)
            if key in FOOT_IDS:
                if primitive['kind']!='sphere' or radius!=.022:raise SensorContractError('exact reviewed nominal foot sphere required')
                other=self.auxiliary_partition.other.intersect_sphere(centre,radius)
                ground=self.auxiliary_partition.floor.intersect_sphere(centre,radius)
                if max(other['intersecting_voxels'],ground['intersecting_voxels'])>hit['intersecting_voxels']:
                    raise SensorContractError('auxiliary partition query escaped all-return enclosure')
                grid=foot_projection_coverage(foot_centres[key],radius,self.floor_cells)
                primary_eligible=old_rules[key]['measured_floor_contact_rule_eligible']
                eligible=(primary_eligible or grid['entire_nominal_projection_on_measured_floor']
                    or primary_patches[key]['complete_nominal_foot_patch'] or patches[key]['complete_nominal_foot_patch'])
                if eligible:primary[key]=dict(shape_id=key,**deepcopy(old_rules[key]['other_or_unknown']))
                selected=other if eligible else hit
                rules.append(dict(shape_id=key,auxiliary_all_returns=hit,auxiliary_floor=ground,auxiliary_other_or_unknown=other,
                    auxiliary_patch=patches[key],primary_patch=primary_patches[key],complete_grid_projection=grid,primary_coverage_eligible=primary_eligible,
                    measured_floor_contact_rule_eligible=eligible,nonfloor_or_unknown_contacts_exempted=False))
            else:selected=hit
            auxiliary.append(dict(shape_id=key,**selected))
        revised=[primary[v['shape_id']] for v in original['shapes']]
        return original|dict(shapes=revised,primary_shapes_before_auxiliary_coverage=original['shapes'],
            auxiliary_shapes=auxiliary,auxiliary_foot_floor_contacts=rules,
            primary_possible_intersection=any(v['intersecting_voxels'] for v in revised),
            auxiliary_possible_intersection=any(v['intersecting_voxels'] for v in auxiliary),
            possible_intersection=any(v['intersecting_voxels'] for v in revised+auxiliary),
            auxiliary_receipt=deepcopy(self.auxiliary_receipt),all_auxiliary_sampled_returns_retained=True,
            auxiliary_unknowns_inferred_free=False,auxiliary_observation_required=True,
            ground_contact_scope='four_foot_spheres_with_complete_measured_primary_or_auxiliary_floor_coverage')


class AuxiliaryDepthFloorMap(RetainedPatchContactMap):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.surface=AuxiliaryDepthSurfaceMemory(identity=identity)

    def observe(self,policy,depth,evidence,*,auxiliary_depth,now_ns):
        try:
            primary=super().observe(policy,depth,evidence,now_ns=now_ns)
            auxiliary=self.surface.observe_auxiliary(policy,auxiliary_depth,self.map_from_initial,
                self.floor_height,self.floor,self.occupied,now_ns=now_ns)
            return primary|dict(primary_receipt_before_auxiliary=primary,auxiliary_receipt=auxiliary,
                retained_observed_floor_cells=len(self.floor),retained_occupied_cells=len(self.occupied),
                auxiliary_observed_pose_registration=True,all_auxiliary_sampled_returns_retained=True)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.failed=True;self.surface.failed=True
            raise SensorContractError('complete primary and auxiliary observed map unavailable') from error
