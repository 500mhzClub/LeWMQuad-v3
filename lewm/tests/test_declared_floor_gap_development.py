from lewm.declared_floor_gap_development import DeclaredFloorGapRegistration,GAP_FRAMES
from lewm.floor_reacquisition_development import ReacquiringFloorRegistration,UNAVAILABLE


def test_withheld_pose_never_becomes_hidden_anchor_and_next_frame_can_resume(monkeypatch):
    registration=DeclaredFloorGapRegistration()
    old={'current_pose':{'frame':404}};reference={'initial':True}
    registration.anchor=old;registration.reference=reference;registration.frame=404
    def accept(self,p,d,a,raw,*,now_ns):
        assert raw['current_pose']['frame']==self.frame+1
        self.frame+=1;self.anchor={'current_pose':{'frame':self.frame}}
        return {'status':'CURRENT_FLOOR_REGISTERED_POSE','current_pose':raw['current_pose']}
    monkeypatch.setattr(ReacquiringFloorRegistration,'observe',accept)
    for frame in GAP_FRAMES:
        r=registration.observe(None,None,None,{'current_pose':{'frame':frame}},
            now_ns=1_500_000_000+frame*100_000_000)
        assert r['status']==UNAVAILABLE and r['current_pose'] is None
        assert registration.anchor is old and registration.reference is reference
    r=registration.observe(None,None,None,{'current_pose':{'frame':409}},now_ns=42_400_000_000)
    assert r['current_pose']['frame']==409 and registration.anchor['current_pose']['frame']==409
