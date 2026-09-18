"""Record the RGB treatment at the actual neural forward call."""
import torch


class NeuralInputTreatmentMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input_calls = 0
        self.last_model_rgb_nonzero = None
        self.model_input_hook = self.model.register_forward_pre_hook(
            self._check_model_inputs, with_kwargs=True)

    def _check_model_inputs(self, module, args, kwargs):
        if args:
            raise ValueError('explicit model input keywords required')
        count = int(torch.count_nonzero(kwargs['observation_history']['rgb']).item())
        if self.variant == 'no_rgb' and count:
            raise ValueError('no-RGB model received nonzero image inputs')
        self.last_model_rgb_nonzero = count
        self.model_input_calls += 1

    def _select_action(self, *args, **kwargs):
        before = self.model_input_calls
        selected, correction = super()._select_action(*args, **kwargs)
        if self.model_input_calls != before + 1:
            raise ValueError('one checked neural call per selected plan required')
        selected['model_input_treatment'] = dict(input_variant=self.variant,
            checked_forward_calls=1, rgb_nonzero_values=self.last_model_rgb_nonzero,
            camera_based_tracking_and_mapping_retained=True)
        return selected, correction
