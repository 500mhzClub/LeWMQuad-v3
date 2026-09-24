"""Same joint-floor map with explicit immutable per-observation floor-index reuse.

Each calculation is a separately named source derivative. No imported function
or active controller is patched; complete decision equality is required before
using this implementation in a fresh native attempt.
"""
from copy import deepcopy
import hashlib
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, body_points as primary_body_points
from lewm.auxiliary_downward45_depth_observation_development import (
    validate_depth as validate_auxiliary_depth, body_points as auxiliary_body_points, CALIBRATION_ID)
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.joint_visual_floor_map_development import GRID, CELL_M
from lewm.joint_floor_registered_controller_development import (
    JointFloorRegisteredSurfaceMemory, JointFloorRegisteredMap, JointFloorRegisteredRoundTripController)
from lewm.frame_cached_floor_geometry_development import FloorFrameGeometry

T = np.asarray(BODY_FROM_OPTICAL)


class FrameCachedFloorMemory(JointFloorRegisteredSurfaceMemory):
    frame_geometry = None

    def _classify_measured(self,policy,depth,B,floor_height,floor_cells,*,now_ns):
        self._current(now_ns)
        if self.classified_ns is not None and now_ns-self.classified_ns!=100_000_000:
            raise SensorContractError('uninterrupted per-return classification required')
        cloud=primary_body_points(depth,policy,now_ns=now_ns,stride=4)
        patch=self.frame_geometry.sampled_floor_patch(depth['depth_m'],depth['valid'],B@self.rotation,B@self.position,
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

    def _classify_retained(self,policy,depth,B,floor_height,floor_cells,*,now_ns):
        self._classify_measured(policy,depth,B,floor_height,floor_cells,now_ns=now_ns)
        witness={k:self.classification_receipt[k] for k in ('frame','measured_ns','rgb_sha256','depth_sha256')}
        self.frame_geometry.append_patch(self.patches,depth['depth_m'],depth['valid'],B@self.rotation,B@self.position,floor_height,witness)

    def classify_current(self, policy, depth, B, floor_height, floor_cells, *, now_ns):
        self._classify_retained(policy, depth, B, floor_height, floor_cells, now_ns=now_ns)
        plane = self.frame_geometry.primary_floor_plane(depth['depth_m'], depth['valid'], B@self.rotation,
            B@self.position, floor_height)
        self.primary_plane = plane | {k: self.classification_receipt[k] for k in
            ('frame', 'measured_ns', 'rgb_sha256', 'depth_sha256')}
        self.primary_plane_ns = now_ns

    def _observe_auxiliary_original(self,policy,depth,B,floor_height,floor_cells,occupied,*,now_ns):
        self._current(now_ns);validate_auxiliary_depth(depth,policy,now_ns=now_ns)
        if (depth['measured_ns']!=now_ns or len(self.auxiliary_patches.frames)!=len(self.route)-1
                or self.classified_ns!=now_ns or (self.auxiliary_ns is not None and now_ns-self.auxiliary_ns!=100_000_000)):
            raise SensorContractError('one current auxiliary observation per admitted primary frame required')
        R=B@self.rotation;p=B@self.position;Q,q=reference_pose(R,p)
        witness=dict(frame=self.route[-1]['frame'],measured_ns=now_ns,calibration_id=CALIBRATION_ID,
            rgb_sha256=depth['primary_rgb_sha256'],
            depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest())
        cloud=auxiliary_body_points(depth,policy,now_ns=now_ns,stride=4)
        points=cloud['points_body_m'][cloud['valid']]@self.rotation.T+self.position
        patch=self.frame_geometry.sampled_floor_patch(depth['depth_m'],depth['valid'],Q,q,floor_height,cloud['rows'],cloud['columns'])
        mask=patch['measured_floor_patch'][cloud['valid']]
        self.auxiliary_index.insert(points,witness);self.auxiliary_partition.insert(points,mask,witness)
        if self.auxiliary_partition.total_returns!=sum(self.auxiliary_index.sample_counts.values()):
            raise SensorContractError('every auxiliary return requires exactly one classification')
        self.frame_geometry.append_patch(self.auxiliary_patches,depth['depth_m'],depth['valid'],Q,q,floor_height,witness)
        coverage=self.frame_geometry.floor_coverage(depth['depth_m'],depth['valid'],Q,q,floor_height)
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

    def observe_auxiliary(self, policy, depth, B, floor_height, floor_cells, occupied, *, now_ns):
        if self.primary_plane_ns != now_ns:
            raise SensorContractError('current primary plane evidence required before auxiliary confirmation')
        original = self._observe_auxiliary_original(policy, depth, B, floor_height, floor_cells, occupied, now_ns=now_ns)
        cloud = auxiliary_body_points(depth, policy, now_ns=now_ns, stride=4)
        Q, q = reference_pose(B@self.rotation, B@self.position)
        mask, classification = self.frame_geometry.confirm_auxiliary_floor(depth['depth_m'], depth['valid'], Q, q,
            floor_height, cloud['rows'], cloud['columns'], self.primary_plane)
        if classification['original_floor_count'] != original['current_floor_returns']:
            raise SensorContractError('original auxiliary classification must reconstruct exactly')
        points = cloud['points_body_m'][cloud['valid']]@self.rotation.T+self.position
        witness = {k: original[k] for k in ('frame', 'measured_ns', 'calibration_id', 'rgb_sha256', 'depth_sha256')}
        partition = self.confirmed_auxiliary_partition
        partition.insert(points, mask[cloud['valid']], witness)
        if partition.total_returns != self.auxiliary_partition.total_returns:
            raise SensorContractError('every original auxiliary return requires one confirmed classification')
        self.confirmation_ns = now_ns
        self.confirmation_receipt = dict(**witness, primary_plane=deepcopy(self.primary_plane),
            classification=classification, total_returns=partition.total_returns,
            floor_returns=partition.floor_returns, other_returns=partition.other_returns,
            original_auxiliary_partition_retained=True, original_all_return_index_retained=True,
            pose_or_floor_grid_changed=False, classifications_use_only_contemporary_public_packets=True)
        # Keep self.auxiliary_receipt exactly original so predecessor footprint
        # receipts remain reproducible independently of the new partition.
        return original | dict(current_primary_floor_confirmation=deepcopy(self.confirmation_receipt))


class FrameCachedFloorMap(JointFloorRegisteredMap):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.surface = FrameCachedFloorMemory(identity=identity)
        self.frame_geometry = None
        self.last_cache_counts = None

    def observe(self, policy, depth, evidence, *, auxiliary_depth, now_ns):
        if self.frame_geometry is not None:
            raise SensorContractError('nested floor-map observation is unavailable')
        context = FloorFrameGeometry()
        self.frame_geometry = self.surface.frame_geometry = context
        try:
            return self._observe_both(policy, depth, evidence,
                auxiliary_depth=auxiliary_depth, now_ns=now_ns)
        finally:
            self.last_cache_counts = context.counts()
            context.close()
            self.frame_geometry = self.surface.frame_geometry = None

    def _observe_primary(self, policy, depth, evidence, *, now_ns):
        if self.failed: raise SensorContractError('floor map failure latched')
        try:
            receipt = self.surface.observe(policy, depth, evidence, now_ns=now_ns)
            if self.map_from_initial is None:
                force = policy['sensor_state']['sensed']['specific_force']
                commands = policy['sensor_state']['control']['applied_command']
                if not force['valid'].all() or not commands['valid'].all() or np.any(np.abs(commands['values']) > 1e-8):
                    raise SensorContractError('quiet initial public force history required')
                up = force['values'].mean(0); magnitude = np.linalg.norm(up)
                if not 8 <= magnitude <= 12: raise SensorContractError('initial gravity magnitude inconsistent')
                up = up/magnitude; forward = np.array([1., 0., 0.])-up*up[0]
                if np.linalg.norm(forward) < .8: raise SensorContractError('initial gravity/forward frame degenerate')
                forward /= np.linalg.norm(forward)
                self.map_from_initial = np.stack((forward, np.cross(up, forward), up))
            R = self.map_from_initial@self.surface.rotation
            p = self.map_from_initial@self.surface.position
            cloud = primary_body_points(depth, policy, now_ns=now_ns, stride=4)
            points = cloud['points_body_m'][cloud['valid']]@R.T+p
            if self.floor_height is None:
                up = R[2]; index = self.frame_geometry.index(depth['depth_m'], depth['valid'], up)
                rr, cc = np.nonzero(index['ground_cells'])
                if len(rr) < 100: raise SensorContractError('initial observed floor hypothesis unavailable')
                z = depth['depth_m'][rr, cc]
                optical = np.column_stack((z*(cc+.5-320)/FOCAL, z*(rr+.5-240)/FOCAL, z))
                xyz = (optical@T[:3, :3].T+T[:3, 3])@R.T+p
                self.floor_height = float(np.median(xyz[:, 2]))
            coverage = self.frame_geometry.floor_coverage(depth['depth_m'], depth['valid'], R, p, self.floor_height)
            for cell in GRID[coverage['covered']]: self.floor.setdefault(tuple(map(int, cell)), receipt['frame'])
            above = points[(points[:, 2] > self.floor_height+.03)&(points[:, 2] < self.floor_height+.65)]
            keys = np.floor(above[:, :2]/CELL_M).astype(int)
            for cell in np.unique(keys, axis=0):
                if np.all(cell >= -100) and np.all(cell < 100): self.occupied.setdefault(tuple(map(int, cell)), receipt['frame'])
            return dict(frame=receipt['frame'], measured_ns=now_ns,
                rgb_sha256=receipt['rgb_sha256'], depth_sha256=receipt['depth_sha256'],
                current_observed_floor_cells=int(coverage['covered'].sum()),
                retained_observed_floor_cells=len(self.floor), retained_occupied_cells=len(self.occupied),
                floor_height_map_m=self.floor_height, map_from_initial=self.map_from_initial.tolist(),
                static_flat_floor_hypothesis=True, uncertainty_calibrated=False,
                continuous_floor_or_volume_coverage=False, navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('observed floor map unavailable') from error

    def _observe_classified_primary(self,policy,depth,evidence,*,now_ns):
        try:
            receipt=self._observe_primary(policy,depth,evidence,now_ns=now_ns)
            self.surface.classify_current(policy,depth,self.map_from_initial,self.floor_height,self.floor,now_ns=now_ns)
            return receipt
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as exc:
            self.failed=True;self.surface.failed=True
            raise SensorContractError('measured-floor contact map unavailable') from exc

    def _observe_both(self,policy,depth,evidence,*,auxiliary_depth,now_ns):
        try:
            primary=self._observe_classified_primary(policy,depth,evidence,now_ns=now_ns)
            auxiliary=self.surface.observe_auxiliary(policy,auxiliary_depth,self.map_from_initial,
                self.floor_height,self.floor,self.occupied,now_ns=now_ns)
            return primary|dict(primary_receipt_before_auxiliary=primary,auxiliary_receipt=auxiliary,
                retained_observed_floor_cells=len(self.floor),retained_occupied_cells=len(self.occupied),
                auxiliary_observed_pose_registration=True,all_auxiliary_sampled_returns_retained=True)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.failed=True;self.surface.failed=True
            raise SensorContractError('complete primary and auxiliary observed map unavailable') from error


class FrameCachedJointFloorRoundTripController(JointFloorRegisteredRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = FrameCachedFloorMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface
