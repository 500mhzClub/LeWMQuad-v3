"""Paired raw-depth floor candidates from a dominant height cluster.

An explicit development alternative to per-pixel mesh-normal selection. A
candidate cluster must contain at least a quarter of below-body observations;
this rejects diffuse wall-height populations, not every possible non-floor
surface. Existing downstream plane/pose acceptance remains necessary.
"""
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.floor_pose_registration_development import ROWS, COLUMNS, unit, proper

RESIDUAL_M = .003
MINIMUM_CLUSTER_FRACTION = .25
MAXIMUM_REFINEMENTS = 8


def select_clouds(clouds, up_body):
    up = unit(up_body)
    arrays = [np.asarray(p, float) for p in clouds]
    if len(arrays) != 2 or any(p.ndim != 2 or p.shape[1:] != (3,)
            or len(p) > 19200 or not np.isfinite(p).all() for p in arrays):
        raise ValueError('two bounded finite raw camera clouds required')
    points = np.concatenate(arrays); keep = np.zeros(len(points), bool)
    receipt = dict(raw_pool_count=len(points), height_slab_width_m=2*RESIDUAL_M,
        minimum_cluster_fraction=MINIMUM_CLUSTER_FRACTION,
        maximum_refinements=MAXIMUM_REFINEMENTS, refinement_steps=0,
        raw_points_selected_by_height=True, pool_outliers_excluded=True,
        floor_identity_established=False, native_pose_used=False)
    if len(points):
        height = points@up
        order = np.argsort(height, kind='stable'); h = height[order]
        right = np.searchsorted(h, h+2*RESIDUAL_M, side='right')
        left = int(np.argmax(right-np.arange(len(h))))
        keep[order[left:right[left]]] = True
    initial_count = int(keep.sum())
    receipt['initial_height_cluster_count'] = initial_count
    receipt['initial_height_cluster_fraction'] = initial_count/max(1,len(points))
    reason = 'insufficient_height_cluster'
    if initial_count >= 3 and initial_count >= MINIMUM_CLUSTER_FRACTION*len(points):
        for step in range(MAXIMUM_REFINEMENTS):
            selected = points[keep]; center = selected.mean(0); delta = selected-center
            values, vectors = np.linalg.eigh(delta.T@delta/len(selected))
            normal = vectors[:,0]
            if normal@up < 0: normal = -normal
            receipt['refinement_steps'] = step+1
            if normal@up < .97:
                reason = 'height_cluster_normal_disagrees_with_up'; keep[:] = False; break
            residual = np.abs(selected@normal-float(normal@center))
            if residual.max() <= RESIDUAL_M:
                reason = 'dominant_height_cluster_plane_inliers'; break
            ids = np.flatnonzero(keep); keep[ids[residual > RESIDUAL_M]] = False
            if keep.sum() < 3 or keep.sum() < MINIMUM_CLUSTER_FRACTION*len(points):
                reason = 'insufficient_cluster_after_residual_selection'; keep[:] = False; break
        else:
            reason = 'bounded_height_refinement_did_not_converge'; keep[:] = False
    else:
        keep[:] = False
    receipt.update(reason=reason, selected_count=int(keep.sum()), excluded_count=int((~keep).sum()))
    split = len(arrays[0])
    return (keep[:split], keep[split:]), receipt


class PairedHeightCandidates:
    """One consumer/observation's selection, shared across its two camera calls."""
    def __init__(self, primary, auxiliary):
        self.packets = (primary, auxiliary)
        self.mounts = (np.asarray(BODY_FROM_OPTICAL), body_from_optical())
        self._up = None; self.receipt = None; self._results = None

    def __call__(self, depth, valid, mount, up_body):
        indices = [i for i,p in enumerate(self.packets)
            if p['depth_m'] is depth and p['valid'] is valid and np.array_equal(mount,self.mounts[i])]
        if len(indices) != 1:
            raise ValueError('exact current paired camera input required')
        up = unit(up_body)
        if self._results is None:
            clouds = []; masks = []
            yy, xx = np.meshgrid(ROWS+.5, COLUMNS+.5, indexing='ij')
            for packet,E in zip(self.packets,self.mounts,strict=True):
                d,v = np.asarray(packet['depth_m']),np.asarray(packet['valid'])
                if (d.shape!=(480,640) or v.shape!=d.shape or v.dtype!=bool
                        or not np.isfinite(d).all() or np.any(d[~v]!=0)
                        or np.any((d[v]<.2)|(d[v]>5))):
                    raise ValueError('original finite depth and validity required')
                proper(E[:3,:3]);z=d[np.ix_(ROWS,COLUMNS)]
                optical=np.stack((z*(xx-320)/FOCAL,z*(yy-240)/FOCAL,z),axis=-1)
                xyz=optical@E[:3,:3].T+E[:3,3]
                good=xyz@up < -.15
                for dr in (-1,0,1):
                    for dc in (-1,0,1):good &= v[np.ix_(ROWS+dr,COLUMNS+dc)]
                masks.append(good);clouds.append(xyz[good])
            selected,self.receipt=select_clouds(clouds,up)
            self._results=[]
            for cloud,mask,keep in zip(clouds,masks,selected,strict=True):
                revised=np.zeros_like(mask);revised[mask]=keep
                self._results.append((cloud[keep],revised))
            self._up=up.copy()
        elif not np.array_equal(up,self._up):
            raise ValueError('one unchanged up direction per paired selection required')
        return self._results[indices[0]]
