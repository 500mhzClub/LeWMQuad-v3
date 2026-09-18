import pytest
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView
from lewm.tests.test_geometry_progress_family_learning_view_development import windows
from lewm.tests.test_moving_action_switch_policy_stream_development import population
from scripts import augmented_family_switch_stream_development as mod


def stream(monkeypatch):
    class FakeFamily:
        def __init__(self):self.view=FamilyWindowView(windows());self.calls=[]
        def training_batch(self,indices):self.calls.append(('train',indices));return 'old_train'
        def inference_batch(self,indices,*,role):self.calls.append((role,indices));return 'old_inference'
    class FakeSwitch:
        def __init__(self):self.view=MovingActionSwitchView(population());self.calls=[]
        def training_batch(self,indices):self.calls.append(('train',indices));return 'new_train'
        def inference_batch(self,indices,*,role):self.calls.append((role,indices));return 'new_inference'
    monkeypatch.setattr(mod,'FamilyPolicyStream',FakeFamily);monkeypatch.setattr(mod,'MovingActionSwitchPolicyStream',FakeSwitch)
    return mod.AugmentedFamilySwitchStream(FakeFamily(),FakeSwitch())


def test_source_dispatch_preserves_exact_local_indices_and_role(monkeypatch):
    s=stream(monkeypatch);old=s.view.indices('train',source='family')[:6];new=s.view.indices('train',source='switch')[:6]
    assert s.training_batch(old)=='old_train' and s.training_batch(new)=='new_train'
    assert s.family.calls==[('train',old)]
    assert s.switch.calls==[('train',[i-s.view.switch_offset for i in new])]
    transfer=s.view.indices('geometry_transfer',source='switch')[:6]
    assert s.inference_batch(transfer,role='geometry_transfer')=='new_inference'
    assert s.switch.calls[-1]==('geometry_transfer',[i-s.view.switch_offset for i in transfer])


def test_mixed_source_or_transfer_training_is_rejected_before_reader_and_latches(monkeypatch):
    s=stream(monkeypatch)
    mixed=[s.view.indices('train',source=source)[0] for source in ('family','switch')]
    with pytest.raises(ValueError,match='homogeneous'):s.training_batch(mixed)
    assert s.family.calls==s.switch.calls==[] and s.failed
    with pytest.raises(ValueError,match='latched'):s.inference_batch(mixed[:1],role='train')
    s=stream(monkeypatch)
    with pytest.raises(ValueError,match='role'):s.training_batch(s.view.indices('geometry_transfer',source='switch')[:6])
    assert s.family.calls==s.switch.calls==[] and s.failed
