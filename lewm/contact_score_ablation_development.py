"""Remove only the learned contact score; preserve motion and physical guards."""
import numpy as np

MODES = ('learned', 'disabled')
DISABLED_LOGIT = -1000.0  # Finite; sigmoid underflows to exactly zero in the scorer.


def contact_score_prediction(prediction, mode):
    p = np.asarray(prediction)
    if mode not in MODES or p.shape != (6,8,5) or not np.isfinite(p).all():
        raise ValueError('complete finite forecast and explicit contact-score mode required')
    disabled = p.copy()
    disabled[:,:,4] = DISABLED_LOGIT
    return (p if mode == 'learned' else disabled).copy()


class ContactScoreAblationMixin:
    def __init__(self, *args, contact_score_mode, **kwargs):
        if contact_score_mode not in MODES: raise ValueError('fixed contact-score mode required')
        self.contact_score_mode = contact_score_mode
        super().__init__(*args, **kwargs)

    def _correct_prediction(self, *args, **kwargs):
        original, receipt = super()._correct_prediction(*args, **kwargs)
        selected = contact_score_prediction(original, self.contact_score_mode)
        enabled = self.contact_score_mode == 'learned'
        return selected, receipt | dict(contact_score_mode=self.contact_score_mode,
            upstream_prediction_for_contact_ablation=original.tolist(),
            applied_prediction_after_contact_ablation=selected.tolist(),
            learned_yaw_and_contact_retained=enabled, learned_yaw_retained=True,
            contact_predictions_used_for_scoring=enabled,
            contact_score_disabled_logit=None if enabled else DISABLED_LOGIT,
            zero_contact_score_is_not_a_contact_free_prediction=True,
            xy_yaw_and_physical_guards_unchanged=True,
            fully_model_free_controller=False)
