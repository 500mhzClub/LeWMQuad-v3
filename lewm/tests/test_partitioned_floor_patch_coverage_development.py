import numpy as np
import pytest
from lewm.partitioned_floor_patch_coverage_development import tile_squares,coverage


def test_closed_tiling_encloses_full_square_without_gaps():
    for centre in ([0.,0.],[.417,-.799],[-4.7,4.7]):
        for n in (1,2,4,8):
            tiles=tile_squares(centre,radius=.022,divisions=n)
            assert len(tiles)==n*n
            by_id={tuple(t['tile']):t for t in tiles}
            for (x,y),t in by_id.items():
                lo=np.array(t['lower_xy_m']);hi=np.array(t['upper_xy_m'])
                mid=np.array(t['centre_xy_m']);r=t['enclosing_radius_m']
                assert (mid-r<=lo).all() and (mid+r>=hi).all()
                if x+1<n:assert hi[0]==by_id[x+1,y]['lower_xy_m'][0]
                if y+1<n:assert hi[1]==by_id[x,y+1]['lower_xy_m'][1]
            assert (np.array(by_id[0,0]['lower_xy_m'])<=np.array(centre)-.022).all()
            assert (np.array(by_id[n-1,n-1]['upper_xy_m'])>=np.array(centre)+.022).all()


def test_partial_coverage_never_certifies_whole_foot():
    class SyntheticPatches:
        def __init__(self,missing):self.missing=missing
        def coverage(self,centres,radius):
            assert len(centres)==64 and radius>=.022/8
            return [dict(complete_nominal_foot_patch=i!=self.missing,
                coverage_witness=None if i==self.missing else dict(frame=i%2)) for i in range(64)]
    partial=coverage(SyntheticPatches(31),[0.,0.],divisions=8)
    assert partial['covered_tiles']==63 and not partial['entire_original_square_covered']
    complete=coverage(SyntheticPatches(None),[0.,0.],divisions=8)
    assert complete['covered_tiles']==64 and complete['entire_original_square_covered']
    assert not complete['ground_support_approved'] and not complete['navigation_qualified']
    for bad in (True,3,16):
        with pytest.raises(ValueError):tile_squares([0.,0.],radius=.022,divisions=bad)
