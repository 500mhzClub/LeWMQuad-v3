import pytest
import torch
from lewm.neural_input_treatment_development import NeuralInputTreatmentMixin


class EchoModel(torch.nn.Module):
    def forward(self, *, observation_history):
        return observation_history['rgb'].clone()


class Base:
    def __init__(self, variant):
        self.model = EchoModel()
        self.variant = variant

    def _select_action(self, rgb):
        result = self.model(observation_history=dict(rgb=rgb))
        return dict(action='hold'), result


class Checked(NeuralInputTreatmentMixin, Base):
    pass


@pytest.mark.parametrize('variant', ['full', 'no_rgb'])
def test_actual_forward_receipt_preserves_outputs(variant):
    runtime = Checked(variant)
    rgb = torch.ones(2,3,4,4) if variant == 'full' else torch.zeros(2,3,4,4)
    before = rgb.clone()
    selected, result = runtime._select_action(rgb)
    assert torch.equal(result, before) and torch.equal(rgb, before)
    receipt = selected['model_input_treatment']
    assert receipt['input_variant'] == variant and receipt['checked_forward_calls'] == 1
    assert receipt['rgb_nonzero_values'] == int(torch.count_nonzero(rgb))


def test_nonzero_image_rejected_before_no_rgb_forward():
    runtime = Checked('no_rgb')
    with pytest.raises(ValueError, match='nonzero image'):
        runtime._select_action(torch.ones(1,3,4,4))
    assert runtime.model_input_calls == 0
