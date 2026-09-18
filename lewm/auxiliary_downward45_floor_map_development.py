"""Same complete observed map and collision checks with explicit 45-degree depth."""
from copy import deepcopy
import hashlib
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.auxiliary_downward45_depth_observation_development import validate_depth,body_points,CALIBRATION_ID
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.auxiliary_depth_floor_map_development import AuxiliaryDepthSurfaceMemory,AuxiliaryDepthFloorMap
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.joint_visual_floor_map_development import floor_coverage,GRID,CELL_M


class AuxiliaryDownward45SurfaceMemory(AuxiliaryDepthSurfaceMemory):
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


class AuxiliaryDownward45FloorMap(AuxiliaryDepthFloorMap):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity)
        self.surface=AuxiliaryDownward45SurfaceMemory(identity=identity)

