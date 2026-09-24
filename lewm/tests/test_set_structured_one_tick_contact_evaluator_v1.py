import torch

from lewm.safety import set_structured_one_tick_contact_evaluator_v1 as subject


def test_shared_contact_evaluator_contract():
    model = subject.SetStructuredOneTickContactEvaluator()
    depth = torch.zeros(2, 8, 48, 64)
    lidar = torch.zeros(2, 32, 180)
    embodied = torch.zeros(2, 5, 81)
    action = torch.zeros(2, 14, 9)
    output = model(depth, lidar, embodied, action)
    assert output.shape == (2, 14)
    assert subject.parameter_count(model) == 149_945
    assert subject.parameter_count(model) < 250_000


def test_action_and_state_change_contact_logits():
    torch.manual_seed(1)
    model = subject.SetStructuredOneTickContactEvaluator().eval()
    depth = torch.randn(1, 8, 48, 64)
    lidar = torch.randn(1, 32, 180)
    embodied = torch.randn(1, 5, 81)
    action = torch.randn(1, 14, 9)
    with torch.inference_mode():
        baseline = model(depth, lidar, embodied, action)
        changed_action = action.clone(); changed_action[:, 0, 0] += 0.2
        changed_state = embodied.clone(); changed_state[:, -1, 0] += 0.2
        assert not torch.equal(baseline, model(depth, lidar, embodied, changed_action))
        assert not torch.equal(baseline, model(depth, lidar, changed_state, action))
