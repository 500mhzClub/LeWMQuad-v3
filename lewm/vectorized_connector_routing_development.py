"""Same closed grid supercover, with cell/box intersections evaluated in arrays."""
import numpy as np
from lewm import observed_floor_waypoint_development as original
from lewm.eligible_floor_registration_development import bind


def segment_cells(start,end):
    a,b=np.asarray(start,float),np.asarray(end,float)
    if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite([a,b]).all() or np.max(np.abs([a,b]))>5.:
        raise ValueError('bounded finite map connector required')
    low=np.floor(np.minimum(a,b)/original.CELL_M).astype(int)-1
    high=np.floor(np.maximum(a,b)/original.CELL_M).astype(int)
    x,y=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1),indexing='ij')
    cells=np.column_stack((x.ravel(),y.ravel()))
    lo=cells*original.CELL_M;hi=lo+original.CELL_M
    entry=np.zeros(len(cells));leave=np.ones(len(cells))
    for axis in range(2):
        delta=b[axis]-a[axis]
        if abs(delta)<1e-15:
            outside=(a[axis]<lo[:,axis]-1e-12)|(a[axis]>hi[:,axis]+1e-12)
            entry[outside]=1.;leave[outside]=0.
        else:
            u=(lo[:,axis]-a[axis])/delta;v=(hi[:,axis]-a[axis])/delta
            entry=np.maximum(entry,np.minimum(u,v))
            leave=np.minimum(leave,np.maximum(u,v))
    return {tuple(map(int,c)) for c in cells[entry<=leave+1e-12]}


propose=bind(original.propose,segment_cells=segment_cells)
