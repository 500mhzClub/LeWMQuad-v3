import numpy as np
import pytest

from lewm.uncertain_ray_memory_development import depth_evidence,query_envelopes
from lewm.tests.test_observed_turn_region_development import frame
from lewm.tests.test_observed_traversal_controller_development import Stream


def compare(e,points,radii,roles):
    reference=query_envelopes(e,points,radii,roles,backend='reference')
    compiled=query_envelopes(e,points,radii,roles,backend='compiled')
    for key in ('free','observed_ground_support','contradictory_or_near_surface','projected_window_pixels'):
        assert np.array_equal(reference[key],compiled[key]),key
    assert np.all(compiled['examined_pixels']<=reference['examined_pixels'])
    return reference,compiled


@pytest.mark.parametrize('scene',['wall','floor','clutter','missing'])
def test_compiled_matches_reference_on_random_queries_and_boundaries(scene):
    rng=np.random.default_rng(77531)
    if scene=='floor':
        _,d,_,_=frame(Stream(),0,floor=True); depth=d['depth_m']; valid=d['valid']
    else:
        depth=rng.uniform(.2,5.,(480,640)).astype(np.float32) if scene=='clutter' else np.full((480,640),2.,np.float32)
        valid=np.ones(depth.shape,bool)
        if scene=='missing': valid[rng.random(depth.shape)<.2]=False; depth[~valid]=0.
    e=depth_evidence(depth,valid,[0,0,1])
    points=rng.uniform([-1,-2,-.6],[5,2,1],size=(200,3)); roles=rng.random(200)<.5
    for radius in (0.,.001,.02,.08,.5,3.): compare(e,points,np.full(200,radius),roles)
    compare(e,np.empty((0,3)),np.empty(0),np.empty(0,bool))


def test_ground_support_and_exact_margin_neighbors_match():
    _,d,_,_=frame(Stream(),0,floor=True); e=depth_evidence(d['depth_m'],d['valid'],[0,0,1])
    points=np.array([[1.5,0,-.317],[1.5,0,-.377],[1.5,0,-.257]])
    for radius in (0.,np.nextafter(.06,0.),.06,np.nextafter(.06,1.)):
        compare(e,points,np.full(3,radius),np.ones(3,bool))
    e=depth_evidence(np.full((480,640),2.,np.float32),np.ones((480,640),bool),[0,0,1])
    points=np.array([[.326+z,0,.043] for z in (1.96,np.nextafter(1.96,0.),np.nextafter(1.96,3.),2.04)])
    compare(e,points,np.zeros(4),np.zeros(4,bool))


def test_rejection_short_circuit_does_not_claim_to_visit_entire_window():
    e=depth_evidence(np.ones((480,640),np.float32),np.ones((480,640),bool),[0,0,1])
    reference,compiled=compare(e,[[1.326,0,.043]],[.1],np.array([False]))
    assert compiled['contradictory_or_near_surface'][0]
    assert 0<compiled['examined_pixels'][0]<compiled['projected_window_pixels'][0]


def test_cached_evidence_cannot_be_accidentally_rewritten():
    depth=np.full((480,640),2.,np.float32); valid=np.ones(depth.shape,bool)
    e=depth_evidence(depth,valid,[0,0,1]); depth[:]=1.; valid[:]=False
    assert np.all(e['depth']==2.) and e['valid'].all()
    for field in ('depth','valid','height','ground','up','tiles'):
        with pytest.raises(ValueError): e[field].flat[0]=0
    with pytest.raises(TypeError): e['depth']=depth


def test_tile_extrema_straddling_conflict_interval_do_not_invent_an_intermediate_surface():
    depth=np.full((480,640),.5,np.float32); depth[:,::2]=1.5
    e=depth_evidence(depth,np.ones(depth.shape,bool),[0,0,1])
    _,compiled=compare(e,[[1.326,0,.043]],[.1],np.array([False]))
    assert not compiled['contradictory_or_near_surface'][0]
    assert not compiled['free'][0]
