"""Isolate neural yaw while retaining fitted XY and zero contact scoring."""
import numpy as np
from lewm.commanded_planar_motion_development import forecast

SOURCES = ('learned', 'command')


def choose_yaw(prediction, commanded, source):
    p = np.asarray(prediction); c = np.asarray(commanded)
    if (source not in SOURCES or p.shape != (6,8,5) or c.shape != (6,8,4)
            or not np.isfinite(p).all() or not np.isfinite(c).all()):
        raise ValueError('explicit yaw source and complete finite alternatives required')
    result = p.copy()
    if source == 'command': result[:,:,2:4] = c[:,:,2:4]
    return result


class YawSourceMixin:
    def __init__(self, *args, forecast_yaw_source, **kwargs):
        if forecast_yaw_source not in SOURCES: raise ValueError('fixed yaw source required')
        self.forecast_yaw_source = forecast_yaw_source
        super().__init__(*args, **kwargs)

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        upstream, receipt = super()._correct_prediction(prediction, packet, evidence, prefix)
        if receipt['forecast_xy_source'] != 'pose_command' or receipt['contact_score_mode'] != 'disabled':
            raise ValueError('same fitted XY and disabled contact in both yaw arms required')
        # The pulse wrapper adds its receipt only after selection returns.
        commanded = forecast(prefix, pulse=bool(self.planning_translation_pulse))
        selected = choose_yaw(upstream, commanded, self.forecast_yaw_source)
        neural = self.forecast_yaw_source == 'learned'
        return selected, receipt | dict(forecast_yaw_source=self.forecast_yaw_source,
            upstream_prediction_for_yaw_ablation=upstream.tolist(),
            applied_prediction_after_yaw_ablation=selected.tolist(),
            commanded_yaw_sin_cos=commanded[:,:,2:4].tolist(),
            both_yaw_alternatives_computed_in_both_arms=True,
            learned_yaw_retained=neural, learned_yaw_and_contact_retained=False,
            neural_outcomes_used_for_scoring=neural,
            xy_yaw_and_physical_guards_unchanged=neural,
            xy_contact_and_physical_guards_unchanged=True,
            pose_command_xy_remains_a_fitted_model=True,
            fully_model_free_controller=False)
