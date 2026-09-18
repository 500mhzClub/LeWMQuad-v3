"""Read-only witnesses for a complete foot square covered by retained patches."""
import numpy as np


def tile_squares(centre,*,radius,divisions):
    p=np.asarray(centre,dtype=float)
    if (p.shape!=(2,) or not np.isfinite(p).all() or np.max(np.abs(p))>4.8
            or isinstance(radius,bool) or not np.isfinite(radius) or not 0<radius<=.05
            or type(divisions) is not int or divisions not in (1,2,4,8)):
        raise ValueError('bounded square and fixed diagnostic subdivision required')
    lower=np.nextafter(p-radius,-np.inf);upper=np.nextafter(p+radius,np.inf)
    edges=[np.linspace(lower[i],upper[i],divisions+1) for i in range(2)]
    squares=[]
    for x in range(divisions):
        for y in range(divisions):
            lo=np.array([edges[0][x],edges[1][y]])
            hi=np.array([edges[0][x+1],edges[1][y+1]])
            mid=(lo+hi)/2
            r=np.nextafter(float(np.maximum(mid-lo,hi-mid).max()),np.inf)
            if not ((mid-r<=lo).all() and (mid+r>=hi).all()):
                raise ValueError('tile enclosure rounding failed')
            squares.append(dict(tile=[x,y],lower_xy_m=lo.tolist(),upper_xy_m=hi.tolist(),
                centre_xy_m=mid.tolist(),enclosing_radius_m=r))
    return squares


def coverage(patches,centre,*,radius=.022,divisions):
    tiles=tile_squares(centre,radius=radius,divisions=divisions)
    # One common outward radius permits the original bounded vectorized query.
    r=max(t['enclosing_radius_m'] for t in tiles)
    witnesses=patches.coverage([t['centre_xy_m'] for t in tiles],radius=r)
    if len(witnesses)!=len(tiles):raise ValueError('complete tile witness array required')
    records=[t|dict(queried_radius_m=r,coverage=w) for t,w in zip(tiles,witnesses,strict=True)]
    complete=all(t['coverage']['complete_nominal_foot_patch'] for t in records)
    return dict(centre_xy_m=np.asarray(centre,float).tolist(),original_radius_m=radius,
        divisions=divisions,tile_count=len(records),tiles=records,
        covered_tiles=sum(t['coverage']['complete_nominal_foot_patch'] for t in records),
        entire_original_square_covered=complete,footprint_shrunk=False,
        unobserved_area_inferred=False,ground_support_approved=False,
        pose_or_terrain_uncertainty_certified=False,navigation_qualified=False)
