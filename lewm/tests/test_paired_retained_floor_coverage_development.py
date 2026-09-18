import numpy as np
import pytest
from lewm.tests.test_observed_geometry_refinement_development import floor
from lewm.tests.test_retained_floor_patch_development import memory
from lewm.paired_retained_floor_coverage_development import PairedRetainedFloorPatches
from lewm.partitioned_floor_patch_coverage_development import coverage


def paired(gap=False):
    d,v=floor();a=v.copy();b=v.copy();a[:,322:]=False;b[:,:316]=False
    if gap:a[:,318:322]=False;b[:,318:322]=False
    return memory(np.where(a,d,0.),a),memory(np.where(b,d,0.),b)


def test_two_partial_views_can_cover_every_closed_foot_tile_without_mutating_inputs():
    a,b=paired();joint=PairedRetainedFloorPatches(a,b)
    assert not a.coverage([[1.,0.]])[0]['complete_nominal_foot_patch']
    assert not b.coverage([[1.,0.]])[0]['complete_nominal_foot_patch']
    assert not joint.coverage([[1.,0.]])[0]['complete_nominal_foot_patch']
    result=coverage(joint,[1.,0.],divisions=8)
    assert result['entire_original_square_covered'] and result['covered_tiles']==64
    assert {t['coverage']['coverage_witness']['witness']['sensor_stream'] for t in result['tiles']}=={'primary','auxiliary'}
    assert all('sensor_stream' not in f['witness'] for m in (a,b) for f in m.frames)
    assert joint.frames[0]['prefix'] is a.frames[0]['prefix'] and not a.frames[0]['prefix'].flags.writeable


def test_shared_unobserved_strip_never_becomes_covered():
    a,b=paired(gap=True);result=coverage(PairedRetainedFloorPatches(a,b),[1.,0.],divisions=8)
    assert not result['entire_original_square_covered'] and result['covered_tiles']<64


def test_mismatched_pairing_rejected():
    a,b=paired();b.frames[0]['witness']['measured_ns']=1
    with pytest.raises(ValueError,match='epoch'):PairedRetainedFloorPatches(a,b)
