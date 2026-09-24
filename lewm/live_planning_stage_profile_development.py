"""Direct same-thread timers inside the unchanged planner's action selection."""
from functools import partial

from lewm.live_planning_profile_development import LivePlanningProfileRuntime


class LivePlanningStageProfileRuntime(LivePlanningProfileRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ('forward', 'encode_history', 'predict_latents', 'decode_rollout'):
            # Models share forward(), but need not expose the old recurrent
            # model's internal stages. Instrument only implemented methods.
            original = getattr(self.model, name, None)
            if original is not None:
                setattr(self.model, name, partial(self._measure, 'model_' + name, original))

    def _check_model_inputs(self, *args, **kwargs):
        return self._measure('model_input_check', super()._check_model_inputs, *args, **kwargs)

    def _select_clear_prediction(self, *args, **kwargs):
        return self._measure('predictive_clearance', super()._select_clear_prediction, *args, **kwargs)

    def _score(self, *args, **kwargs):
        return self._measure('prediction_score', super()._score, *args, **kwargs)
