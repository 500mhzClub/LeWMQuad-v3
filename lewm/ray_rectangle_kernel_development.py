"""Exact CPU rectangle reductions; no fast-math, GPU, or persistent JIT cache."""
import numpy as np
from numba import njit


@njit(cache=False,nogil=True,boundscheck=True)
def reduce_rectangles(depth,valid,height,ground,query_height,radii,low,high,lefts,rights,tops,bottoms,tiles):
    count=len(radii)
    free=np.zeros(count,np.bool_); supported=np.zeros(count,np.bool_); conflict=np.zeros(count,np.bool_)
    visited=np.zeros(count,np.int64); area=np.zeros(count,np.int64)
    # query_height is NaN for roles that cannot claim ground support.
    for i in range(count):
        if high[i]+.04<.2: continue
        left,right,top,bottom=lefts[i],rights[i],tops[i],bottoms[i]
        complete=low[i]>=.2 and high[i]<=5. and left>=1 and right<=638 and top>=1 and bottom<=478
        left=max(0,left); right=min(639,right); top=max(0,top); bottom=min(479,bottom)
        if right<left or bottom<top: continue
        area[i]=(right-left+1)*(bottom-top+1)
        can_free=complete; can_support=complete and np.isfinite(query_height[i]); near=False
        min_depth=np.inf; min_height=np.inf; max_height=-np.inf; done=False
        for ty in range(top//8,bottom//8+1):
            for tx in range(left//8,right//8+1):
                y0=max(top,ty*8); y1=min(bottom,ty*8+7)
                x0=max(left,tx*8); x1=min(right,tx*8+7)
                entire=y0==ty*8 and y1==ty*8+7 and x0==tx*8 and x1==tx*8+7
                scan=True
                if entire:
                    dmin=tiles[0,ty,tx]; dmax=tiles[1,ty,tx]
                    if tiles[2,ty,tx]==0:
                        can_free=False; can_support=False
                    if dmin<min_depth: min_depth=dmin
                    if can_support:
                        if tiles[3,ty,tx]==0: can_support=False
                        else:
                            if tiles[4,ty,tx]<min_height: min_height=tiles[4,ty,tx]
                            if tiles[5,ty,tx]>max_height: max_height=tiles[5,ty,tx]
                    # Extrema are actual valid values. If either falls in
                    # the conflict interval, existence is proved; if both
                    # straddle it, inspect pixels instead of assuming a hit.
                    if ((dmin>=low[i]-.04 and dmin<=high[i]+.04)
                            or (dmax>=low[i]-.04 and dmax<=high[i]+.04)):
                        near=True; scan=False
                    elif dmax<low[i]-.04 or dmin>high[i]+.04:
                        scan=False
                if scan:
                    for y in range(y0,y1+1):
                        for x in range(x0,x1+1):
                            visited[i]+=1
                            if not valid[y,x]:
                                can_free=False; can_support=False
                            else:
                                value=depth[y,x]
                                if value<min_depth: min_depth=value
                                if value>=low[i]-.04 and value<=high[i]+.04: near=True
                            if can_support:
                                if not ground[y,x]: can_support=False
                                else:
                                    h=height[y,x]
                                    if h<min_height: min_height=h
                                    if h>max_height: max_height=h
                if can_support and (max_height-min_height>.006
                        or max(abs(query_height[i]-min_height),abs(query_height[i]-max_height))+radii[i]>.06):
                    can_support=False
                if near and not can_support:
                    done=True; break
            if done: break
        conflict[i]=near and not can_support
        supported[i]=can_support and not conflict[i]
        free[i]=can_free and min_depth-high[i]>=.04 and not conflict[i]
    return free,supported,conflict,visited,area


def warm_rectangle_kernel():
    """Compile before a control episode; never count cold compilation as latency."""
    floats=np.empty(0,np.float64); integers=np.empty(0,np.int64)
    arrays=[np.zeros((1,1),np.float32),np.zeros((1,1),np.bool_),np.zeros((1,1),np.float64),
            np.zeros((1,1),np.bool_),np.zeros((6,1,1),np.float64)]
    for array in arrays: array.flags.writeable=False
    reduce_rectangles(arrays[0],arrays[1],arrays[2],arrays[3],floats,floats,floats,floats,
        integers,integers,integers,integers,arrays[4])
