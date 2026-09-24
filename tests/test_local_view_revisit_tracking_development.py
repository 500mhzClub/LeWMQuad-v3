from types import SimpleNamespace
import numpy as np
from lewm.local_view_revisit_tracking_development import select_revisit,view_bin


def ref(frame,p=(0.,0.,0.),R=None):
    return SimpleNamespace(frame=frame,measured_ns=frame*100_000_000,
        position=np.asarray(p),gyro=np.eye(3) if R is None else R)


def test_revisit_requires_old_nearby_similar_view_and_skips_active_reference():
    old=ref(0);far=ref(1,(.5,0.,0.));recent=ref(95);active=ref(2)
    opposite=ref(3,R=np.diag([-1.,-1.,1.]))
    bank={i:(r,None) for i,r in enumerate((old,far,recent,active,opposite))}
    assert select_revisit(bank,[active],np.zeros(3),np.eye(3),10_000_000_000)[0] is old
    del bank[0]
    assert select_revisit(bank,[active],np.zeros(3),np.eye(3),10_000_000_000) is None


def test_view_bins_wrap_at_full_rotation():
    def R(a):
        c,s=np.cos(a),np.sin(a)
        return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    assert [view_bin(R(i*np.pi/4)) for i in range(8)]==list(range(8))
    assert view_bin(R(2*np.pi-.01))==view_bin(R(.01))==0
