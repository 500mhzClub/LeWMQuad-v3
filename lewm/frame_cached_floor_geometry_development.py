"""Frozen floor calculations with an explicit per-observation index provider."""
from copy import deepcopy
import math
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.joint_visual_floor_map_development import GRID, CELL_M
from lewm.current_primary_floor_plane_development import ROWS, COLUMNS, measured_points
from lewm.frame_floor_index_cache_development import FrameFloorIndexCache

T = np.asarray(BODY_FROM_OPTICAL)


class FloorFrameGeometry(FrameFloorIndexCache):
    def floor_coverage(self, depth, valid, map_from_body, translation_map, floor_height, cells=GRID):
        """Require every image cell in each projected floor-square rectangle.
    
        The fixed measured plane proposes where to query; it cannot provide coverage.
        Every projected cell must independently pass the original ground mesh tests
        and have all four measured heights within 10 mm of that hypothesis.
        """
        R, p = np.asarray(map_from_body, float), np.asarray(translation_map, float)
        cells = np.asarray(cells)
        if (R.shape != (3, 3) or p.shape != (3,) or not np.isfinite([floor_height]).all()
                or not np.isfinite(R).all() or not np.isfinite(p).all()
                or not np.allclose(R.T@R, np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(R)-1) > 1e-8
                or cells.ndim != 2 or cells.shape[1:] != (2,) or cells.dtype.kind not in 'iu'
                or len(cells) > 40000 or np.any(cells < -100) or np.any(cells >= 100)):
            raise SensorContractError('bounded floor grid and proper observed map transform required')
        up_body = R[2]
        index = self.index(depth, valid, up_body)
        yy, xx = np.indices((480, 640))
        optical = np.stack((depth*(xx+.5-320)/FOCAL, depth*(yy+.5-240)/FOCAL, depth), axis=-1)
        body = optical@T[:3, :3].T+T[:3, 3]
        heights = body@up_body+p[2]
        near = np.abs(heights-floor_height) <= .01
        good = index['ground_cells'] & near[:-1, :-1] & near[:-1, 1:] & near[1:, :-1] & near[1:, 1:]
        prefix = np.zeros((480, 640), np.int64); prefix[1:, 1:] = (~good).cumsum(0).cumsum(1)
        xy = (cells[:, None, :]+np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))*CELL_M
        world = np.concatenate((xy, np.full((*xy.shape[:-1], 1), floor_height)), axis=-1)
        points = (world-p)@R
        camera = (points-T[:3, 3])@T[:3, :3]
        z = camera[..., 2]
        uv = camera[..., :2]/np.maximum(z[..., None], 1e-12)*FOCAL+[319.5, 239.5]
        lo, hi = uv.min(1)-1e-9, uv.max(1)+1e-9
        visible = ((z >= .2)&(z <= 5.)).all(1) & (lo >= 0).all(1) & (hi < [639, 479]).all(1)
        a, b = np.floor(np.clip(lo, -1, 640)).astype(int), np.floor(np.clip(hi, -1, 640)).astype(int)
        covered = np.zeros(len(cells), bool)
        ids = np.flatnonzero(visible)
        if len(ids):
            x0, y0 = a[ids].T; x1, y1 = (b[ids]+1).T
            covered[ids] = prefix[y1, x1]-prefix[y0, x1]-prefix[y1, x0]+prefix[y0, x0] == 0
        return dict(covered=covered, floor_candidate_pixels=int(good.sum()),
            projected_lower_xy=a, projected_upper_xy=b, ground_support_approved=False)

    def sampled_floor_patch(self, depth,valid,rotation_map_from_body,position_map,floor_height,rows,columns):
        """All four adjacent mesh quads plus all nine measured pixel heights required."""
        R=proper(rotation_map_from_body);p=np.asarray(position_map,float)
        rr,cc=np.asarray(rows),np.asarray(columns)
        if (p.shape!=(3,) or not np.isfinite(p).all() or not math.isfinite(floor_height)
                or rr.ndim!=1 or cc.ndim!=1 or rr.dtype.kind not in 'iu' or cc.dtype.kind not in 'iu'
                or np.any(rr<1) or np.any(rr>=479) or np.any(cc<1) or np.any(cc>=639)):
            raise ValueError('finite map pose/plane and interior integer sample pixels required')
        index=self.index(depth,valid,R[2]);T=np.asarray(BODY_FROM_OPTICAL)
        yy,xx=np.indices((480,640))
        optical=np.stack((depth*(xx+.5-320)/FOCAL,depth*(yy+.5-240)/FOCAL,depth),axis=-1)
        heights=(optical@T[:3,:3].T+T[:3,3])@R[2]+p[2]
        near=valid&(np.abs(heights-floor_height)<=.01)
        good=np.ones((len(rr),len(cc)),bool)
        for dr in (-1,0):
            for dc in (-1,0):good&=index['ground_cells'][np.ix_(rr+dr,cc+dc)]
        for dr in (-1,0,1):
            for dc in (-1,0,1):good&=near[np.ix_(rr+dr,cc+dc)]
        return dict(measured_floor_patch=good,point_map_height_m=heights[np.ix_(rr,cc)],
            plane_height_band_m=.01,ground_support_approved=False,continuous_coverage_established=False)

    def primary_floor_plane(self, depth, valid, rotation_map_from_body, position_map, floor_height):
        # Seed only from the predecessor's complete measured primary floor patches.
        patch = self.sampled_floor_patch(depth, valid, rotation_map_from_body, position_map,
            floor_height, ROWS, COLUMNS)
        points = measured_points(depth, rotation_map_from_body, position_map)
        seeds = points[np.ix_(ROWS, COLUMNS)][patch['measured_floor_patch']]
        receipt = dict(available=False, seed_count=len(seeds), original_seed_band_m=.01,
            maximum_fit_residual_m=.003, minimum_second_covariance_eigenvalue_m2=.05**2,
            minimum_up_alignment=.97, native_pose_used=False, floor_or_support_certified=False)
        if len(seeds) < 100: return receipt | dict(reason='insufficient_original_floor_patches')
        center = seeds.mean(0); delta = seeds-center
        values, vectors = np.linalg.eigh(delta.T@delta/len(seeds)); normal = vectors[:, 0]
        if normal[2] < 0: normal = -normal
        offset = -float(normal@center); errors = seeds@normal+offset
        receipt.update(normal_map=normal.tolist(), offset_m=offset,
            seed_xy_lower_m=seeds[:, :2].min(0).tolist(), seed_xy_upper_m=seeds[:, :2].max(0).tolist(),
            covariance_eigenvalues_m2=values.tolist(), maximum_absolute_seed_residual_m=float(np.abs(errors).max()))
        if values[1] < .05**2: return receipt | dict(reason='insufficient_two_axis_floor_extent')
        if normal[2] < .97: return receipt | dict(reason='plane_up_disagreement')
        if np.abs(errors).max() > .003: return receipt | dict(reason='primary_floor_not_one_coherent_plane')
        return receipt | dict(available=True, reason='current_primary_measured_floor_plane')

    def confirm_auxiliary_floor(self, depth, valid, rotation_map_from_reference, position_map,
            floor_height, rows, columns, plane):
        original = self.sampled_floor_patch(depth, valid, rotation_map_from_reference, position_map,
            floor_height, rows, columns)['measured_floor_patch']
        receipt = dict(original_floor_count=int(original.sum()), added_floor_count=0,
            measured_plane_band_m=.01, maximum_original_plane_height_difference_m=.03,
            primary_plane_extrapolation_requires_measured_auxiliary_patch=True,
            original_floor_classifications_preserved=True,
            original_ground_mesh_checks_preserved=True, unobserved_space_certified=False,
            ground_support_approved=False, non_foot_contact_exemption=False)
        if not plane['available']: return original.copy(), receipt
        R = proper(rotation_map_from_reference)
        points = measured_points(depth, R, position_map)
        normal = np.asarray(plane['normal_map'], float)
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or abs(np.linalg.norm(normal)-1.) > 1e-8 or normal[2] < .97
                or not np.isfinite(plane['offset_m'])):
            raise ValueError('finite admitted primary plane required')
        index = self.index(depth, valid, R[2])
        near = (valid & (np.abs(points@normal+plane['offset_m']) <= .01)
            & (np.abs(points[:, :, 2]-floor_height) <= .03))
        good = np.ones_like(original)
        for dr in (-1, 0):
            for dc in (-1, 0): good &= index['ground_cells'][np.ix_(rows+dr, columns+dc)]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1): good &= near[np.ix_(rows+dr, columns+dc)]
        revised = original | good
        receipt['added_floor_count'] = int((revised & ~original).sum())
        return revised, receipt

    def append_patch(self,history,depth,valid,rotation_map_from_body,position_map,floor_height,witness):
        R=proper(rotation_map_from_body);p=np.asarray(position_map,float)
        if (p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(floor_height)
                or type(witness['frame']) is not int or witness['frame']!=len(history.frames) or len(history.frames)>=4096
                or (history.frames and (floor_height!=history.frames[0]['floor_height'] or
                    witness['measured_ns']-history.frames[-1]['witness']['measured_ns']!=100_000_000))):
            raise ValueError('bounded uninterrupted patch history and fixed measured floor required')
        index=self.index(depth,valid,R[2]);yy,xx=np.indices((480,640))
        optical=np.stack((depth*(xx+.5-320)/FOCAL,depth*(yy+.5-240)/FOCAL,depth),axis=-1)
        heights=(optical@T[:3,:3].T+T[:3,3])@R[2]+p[2]
        near=np.abs(heights-floor_height)<=.01
        good=index['ground_cells']&near[:-1,:-1]&near[:-1,1:]&near[1:,:-1]&near[1:,1:]
        # At most 479*639 invalid pixels: exact int32 sums cannot overflow.
        prefix=np.zeros((480,640),np.int32);prefix[1:,1:]=(~good).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
        prefix.flags.writeable=False
        history.frames.append(dict(R=R.copy(),p=p.copy(),floor_height=float(floor_height),prefix=prefix,witness=deepcopy(witness)))
