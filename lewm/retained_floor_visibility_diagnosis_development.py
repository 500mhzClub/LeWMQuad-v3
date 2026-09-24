"""Explain fixed nominal foot tiles using the existing retained camera checks."""
from copy import deepcopy
import numpy as np
from lewm.retained_floor_patch_development import T,FOCAL
from lewm.partitioned_floor_patch_coverage_development import coverage


def explain(patches,centre):
    partition=coverage(patches,centre,divisions=8)
    tiles=partition['tiles'];xy=np.array([t['centre_xy_m'] for t in tiles]);radius=tiles[0]['queried_radius_m']
    views=[[] for _ in tiles]
    for frame in patches.frames:
        corners=xy[:,None,:]+radius*np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
        world=np.concatenate((corners,np.full((*corners.shape[:-1],1),frame['floor_height'])),axis=-1)
        points=(world-frame['p'])@frame['R'];camera=(points-T[:3,3])@T[:3,:3];z=camera[...,2]
        uv=camera[...,:2]/np.maximum(z[...,None],1e-12)*FOCAL+[319.5,239.5]
        lo=uv.min(1);hi=uv.max(1);margin=1e-9+64*np.finfo(float).eps*np.maximum(np.abs(lo),np.abs(hi))
        lo-=margin;hi+=margin
        visible=((z>=.2)&(z<=5.)).all(1)&(lo>=0).all(1)&(hi<[639,479]).all(1)
        a=np.floor(np.clip(lo,-1,640)).astype(int);b=np.floor(np.clip(hi,-1,640)).astype(int)
        for i in range(len(tiles)):
            row=dict(frame=frame['witness']['frame'],witness=deepcopy(frame['witness']),
                entire_tile_in_frustum=bool(visible[i]),
                near_or_behind=bool((z[i]<.2).any()),far=bool((z[i]>5.).any()),
                horizontal_outside=bool(lo[i,0]<0 or hi[i,0]>=639),
                vertical_outside=bool(lo[i,1]<0 or hi[i,1]>=479),
                optical_depth_range_m=[float(z[i].min()),float(z[i].max())],
                pixel_lower_xy=lo[i].tolist(),pixel_upper_xy=hi[i].tolist())
            if visible[i]:
                x0,y0=a[i];x1,y1=b[i]+1;prefix=frame['prefix']
                bad=int(prefix[y1,x1]-prefix[y0,x1]-prefix[y1,x0]+prefix[y0,x0])
                row.update(invalid_floor_quads=bad,projected_quads=int((x1-x0)*(y1-y0)),
                    pixel_rectangle=[a[i].tolist(),b[i].tolist()])
            views[i].append(row)
    explained=[]
    for tile,history in zip(tiles,views,strict=True):
        visible=[v for v in history if v['entire_tile_in_frustum']]
        good=[v for v in visible if v['invalid_floor_quads']==0]
        if bool(good)!=tile['coverage']['complete_nominal_foot_patch']:
            raise ValueError('visibility explanation changed original floor coverage')
        best=min(visible,key=lambda v:(v['invalid_floor_quads']/v['projected_quads'],v['invalid_floor_quads'],v['frame'])) if visible else None
        explained.append(dict(tile=tile['tile'],centre_xy_m=tile['centre_xy_m'],
            original_coverage=tile['coverage'],visible_frames=len(visible),passing_frames=len(good),
            failure_reason=None if good else 'VISIBLE_BUT_FLOOR_QUADS_REJECTED' if visible else 'NEVER_ENTIRELY_IN_FRUSTUM',
            best_visible_frame=best,current_frame=history[-1] if history else None,
            visible_frame_checks=visible))
    return dict(centre_xy_m=partition['centre_xy_m'],original_radius_m=.022,divisions=8,
        tile_count=len(tiles),covered_tiles=partition['covered_tiles'],tiles=explained,
        original_coverage_exact=True,occlusion_inferred=False,controller_changed=False,
        ground_support_approved=False,navigation_qualified=False)
