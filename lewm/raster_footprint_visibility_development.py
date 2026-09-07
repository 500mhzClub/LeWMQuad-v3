"""Evaluator-only prospective pixel-footprint accounting, never a policy filter.

Preserves the original strict score. Boundary uncertainty is derived from every
box edge, including foreground silhouettes, independently of measured errors.
This is deliberately conservative for hidden/internal edges. It is not a native
raster-precision proof, a depth repair, or authorization to salvage old data.
"""
import itertools
import numpy as np
from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_first_surface_depth_development import expected_optical_depth, evaluate_visibility

PIXEL_HALF_DIAGONAL = np.sqrt(.5)


def projected_boundary_mask(boxes, transform, *, stride=8):
    # Reuse strict physical input validation before any projection.
    ref=expected_optical_depth(boxes,transform,stride=stride)
    if not 1 <= len(boxes) <= 128: raise ValueError('bounded nonempty box population required')
    T=np.asarray(transform,float)
    u,v=np.meshgrid(ref['columns']+.5,ref['rows']+.5)
    points=np.stack((u,v),axis=-1); minimum=np.full(u.shape,np.inf)
    signs=np.array(list(itertools.product((-1,1),repeat=3)))
    edge_indices=[(i,j) for i,j in itertools.combinations(range(8),2) if np.count_nonzero(signs[i]!=signs[j])==1]
    # Positive projection plane only. Do not use native near clipping to hide
    # an opaque occluder: the original strict check below retains those failures.
    positive_z=1e-9
    for box in boxes:
        angle=float(box['yaw_rad']); c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        world=(signs*np.asarray(box['size_xyz'])/2)@rotation.T+box['centre_xyz']
        optical=(world-T[:3,3])@T[:3,:3]
        for i,j in edge_indices:
            a,b=optical[i].copy(),optical[j].copy()
            if max(a[2],b[2]) <= positive_z: continue
            if a[2] < positive_z: a+=(positive_z-a[2])/(b[2]-a[2])*(b-a)
            if b[2] < positive_z: b+=(positive_z-b[2])/(a[2]-b[2])*(a-b)
            uv=np.stack((a,b)); uv=uv[:,:2]/uv[:,2,None]*FOCAL+[320,240]
            d=uv[1]-uv[0]; length2=d@d
            if length2 < 1e-24:
                distance=np.linalg.norm(points-uv[0],axis=-1)
            else:
                fraction=np.clip(np.sum((points-uv[0])*d,axis=-1)/length2,0,1)
                distance=np.linalg.norm(points-(uv[0]+fraction[...,None]*d),axis=-1)
            minimum=np.minimum(minimum,distance)
    # Circumscribed disk covers the whole one-pixel square. The small numerical
    # guard is fixed, not selected from recorded failures or renderer outcomes.
    return ref,minimum <= PIXEL_HALF_DIAGONAL+1e-9


def evaluate_footprint(native_depth, boxes, transform, *, render_near_m, stride=8):
    strict=evaluate_visibility(native_depth,boxes,transform,render_near_m=render_near_m,stride=stride)
    ref,boundary=projected_boundary_mask(boxes,transform,stride=stride)
    expected=ref['expected_depth_m']; measured=np.asarray(native_depth)[np.ix_(ref['rows'],ref['columns'])]
    domain=np.isfinite(expected)&(expected<4.98)&ref['surface_interior']
    stable=domain&~boundary; ambiguous=domain&boundary
    finite=np.isfinite(measured)
    residual=np.abs(measured-expected)
    bad=~finite|(residual>.001)
    values=residual[stable]
    return dict(schema='raster_footprint_visibility_diagnostic.v1',original_strict_score=strict,
        original_compared_rays=int(domain.sum()),stable_interior_rays=int(stable.sum()),
        boundary_ambiguous_rays=int(ambiguous.sum()),all_projected_boundary_rays=int(boundary.sum()),
        stable_interior_bad_rays=int((stable&bad).sum()),boundary_bad_rays=int((ambiguous&bad).sum()),
        stable_interior_metric_pass=bool(len(values)>=1000 and np.isfinite(values).all() and np.all(values<=.001)),
        stable_interior_maximum_error_m=float(values.max()) if len(values) and np.isfinite(values).all() else None,
        near_occlusion_failure=bool(strict['clipped_opaque_rays'] or strict['false_public_valid_near_rays']),
        boundary_pixels_certified=False,qualification_granted=False,policy_filter=False,
        scope='prospective evaluator-only coverage accounting; old strict failures unchanged')
