"""Compare learned corrected XY/yaw with fitted pose-command XY/integrated yaw."""
from lewm.commanded_planar_motion_development import forecast
from lewm.yaw_source_ablation_development import choose_yaw

SOURCES = ('learned', 'pose_command')


class CombinedMotionSourceMixin:
    def __init__(self, *args, motion_prediction_source, **kwargs):
        if motion_prediction_source not in SOURCES: raise ValueError('fixed motion source required')
        self.motion_prediction_source = motion_prediction_source
        super().__init__(*args, forecast_xy_source=motion_prediction_source,
            contact_score_mode='disabled', **kwargs)

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        upstream, receipt = super()._correct_prediction(prediction, packet, evidence, prefix)
        if receipt['forecast_xy_source'] != self.motion_prediction_source or receipt['contact_score_mode'] != 'disabled':
            raise ValueError('matched XY assignment and disabled contact required')
        commanded = forecast(prefix, pulse=bool(self.planning_translation_pulse))
        learned = self.motion_prediction_source == 'learned'
        yaw_source = 'learned' if learned else 'command'
        selected = choose_yaw(upstream, commanded, yaw_source)
        return selected, receipt | dict(motion_prediction_source=self.motion_prediction_source,
            forecast_yaw_source=yaw_source, upstream_prediction_for_yaw_ablation=upstream.tolist(),
            applied_prediction_after_yaw_ablation=selected.tolist(),
            commanded_yaw_sin_cos=commanded[:,:,2:4].tolist(),
            both_yaw_alternatives_computed_in_both_arms=True,
            learned_yaw_retained=learned, learned_yaw_and_contact_retained=False,
            neural_outcomes_used_for_scoring=learned,
            xy_yaw_and_physical_guards_unchanged=False,
            contact_and_physical_guards_unchanged=True,
            pose_command_xy_remains_a_fitted_model=True,
            learned_xy_includes_frozen_residual_correction=True,
            fully_model_free_controller=False)
