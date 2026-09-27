from copy import deepcopy
from unittest.mock import patch
import numpy as np
from lewm.navigation_capability_completed_support_development import completed_corner_support,CompletedSupportRuntimeMixin,CompletedFramewiseRegistration
from lewm.navigation_capability_live_turn_binding_c1_development import LiveTurnMemoryRuntimeMixin
from lewm.sparse_corner_completion_runtime_development import strong_corner_support,StrongCornerFramewiseRegistration
from lewm.visual_recovery_dispatch_hold_development import PublishingSupportRegistration
from lewm.navigation_capability_exhausted_view_development import ExhaustibleSupportedView


def raw(strong=(35,24),actual=(125,150)):
    return dict(current_pose=dict(frame=3,measured_ns=300_000_000),last_accepted_feature_witness=dict(selected_features=actual[0],original_selected_count=strong[0]),auxiliary_feature_witness=dict(selected_features=actual[1],original_selected_count=strong[1]))


def test_counterfactual_count_substitution_keeps_original_evidence_and_pose():
    r=raw();original=deepcopy(r);old=strong_corner_support(r);new=completed_corner_support(r)
    assert old['selected_features']==[35,24] and new['selected_features']==[125,150]
    assert new['original_strong_selected_features']==old['selected_features']
    assert r==original and new['frame']==old['frame'] and new['measured_ns']==old['measured_ns']


def test_missing_pose_or_witness_remains_unavailable():
    r=raw();r['current_pose']=None;assert completed_corner_support(r)is None
    r=raw();r['auxiliary_feature_witness']=None;assert completed_corner_support(r)is None


def test_genuinely_sparse_selected_support_still_recovers():
    view=ExhaustibleSupportedView(maximum_view_age_ns=None);view.advance([150,150],np.zeros(3),np.eye(3),0,0)
    c,s=np.cos(.3),np.sin(.3);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    counts=completed_corner_support(raw(strong=(5,6),actual=(20,25)))['selected_features']
    assert view.advance(counts,np.zeros(3),R,100_000_000,0)is not None


def test_constructor_preserves_underlying_registration_recovery_and_publisher():
    original=object();events=[];views=ExhaustibleSupportedView(maximum_view_age_ns=None)
    def parent(self,*a,**k):
        inner=StrongCornerFramewiseRegistration(original);inner.views=views
        self.registration=PublishingSupportRegistration(inner,events.append)
    obj=object.__new__(CompletedSupportRuntimeMixin)
    with patch.object(LiveTurnMemoryRuntimeMixin,'__init__',parent):CompletedSupportRuntimeMixin.__init__(obj)
    assert type(obj.registration.original)is CompletedFramewiseRegistration
    assert obj.registration.original.original is original and obj.registration.original.views is views
    assert obj.registration.publish==events.append
