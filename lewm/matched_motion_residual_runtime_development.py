"""Use each fixed training condition's correction in the same pulse controller."""
import hashlib
import numpy as np

from lewm.arrival_entry_terminal_priority_development import ArrivalEntryTerminalPriorityRuntime
from scripts.navigation_artifact_root_development import BASE, validate_root

FITS = {
    'jepa': ('go2_closed_loop_motion_residual_study_v1_attempt_001',
        'ef7511b29afd0117291f600d7afad6adccdcd90199d0ea1f45519d7b81b01638'),
    'direct': ('go2_matched_motion_residual_direct_v1_attempt_001',
        '118fa93d4a420525c5685cf14b2aeec30cff827abf1d65029716aa7e69dd9e6a'),
    'supervised_rollout': ('go2_matched_motion_residual_supervised_rollout_v1_attempt_001',
        '413a3160c21dc43b4b8a1b87b4c092d59428158270d7b91179229d84b04fda1d'),
}


def load_fit(condition):
    if condition not in FITS:
        raise ValueError('fixed JEPA, direct or supervised-rollout condition required')
    name, identity = FITS[condition]
    path = validate_root(BASE/name)/'residual_fit.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != identity:
        raise ValueError('assigned frozen motion correction changed')
    with np.load(path, allow_pickle=False) as arrays:
        fit = {key:arrays[key].copy() for key in ('mean', 'scale', 'bias', 'coefficient')}
    return fit, dict(fit_sha256=identity, correction_root=name,
        correction_base_model=f'seed_2026091001_full_{condition}')


class MatchedMotionResidualRuntime(ArrivalEntryTerminalPriorityRuntime):
    def __init__(self, *args, condition, variant='full', _assigned_motion_fit=None, **kwargs):
        if _assigned_motion_fit is None:
            if variant != 'full':
                raise ValueError('no-RGB requires its explicitly assigned frozen correction')
            fit, binding = load_fit(condition)
        else:
            if variant != 'no_rgb':
                raise ValueError('explicit override is reserved for matched no-RGB assignments')
            fit, binding = _assigned_motion_fit
        super().__init__(*args, condition=condition, variant=variant, **kwargs)
        # Preserve the pulse-aware correction implementation and causal history.
        self.motion_residual.fit = fit
        self.matched_residual_binding = binding

    def _correct_prediction(self, *args, **kwargs):
        prediction, evidence = super()._correct_prediction(*args, **kwargs)
        return prediction, evidence | self.matched_residual_binding
