import numpy as np
from lewm.tests.test_retained_floor_patch_development import memory
from lewm.tests.test_observed_geometry_refinement_development import floor
from lewm.retained_floor_visibility_diagnosis_development import explain


def test_visible_good_floor_and_out_of_view_are_distinguished():
    m=memory();good=explain(m,[1.,0.]);unseen=explain(m,[0.,0.])
    assert good['covered_tiles']==64 and all(t['passing_frames']==1 for t in good['tiles'])
    assert unseen['covered_tiles']==0
    assert all(t['failure_reason']=='NEVER_ENTIRELY_IN_FRUSTUM' for t in unseen['tiles'])


def test_visible_rejected_pixels_and_retained_good_view():
    d,v=floor();rectangle=memory(d,v).coverage([[1.,0.]])[0]['coverage_witness']['pixel_rectangle']
    (x0,y0),(x1,y1)=rectangle;d[y0:y1+2,x0:x1+2]=0;v[y0:y1+2,x0:x1+2]=False
    bad=memory(d,v);r=explain(bad,[1.,0.])
    assert r['covered_tiles']==0
    assert all(t['failure_reason']=='VISIBLE_BUT_FLOOR_QUADS_REJECTED' for t in r['tiles'])
    d,v=floor();bad.append(d,v,np.eye(3),np.zeros(3),-.32,dict(frame=1,measured_ns=100_000_000))
    r=explain(bad,[1.,0.])
    assert r['covered_tiles']==64 and all(t['passing_frames']==1 and t['visible_frames']==2 for t in r['tiles'])
