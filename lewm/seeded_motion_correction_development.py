"""Bind each frozen full-input model to its own fixed motion correction."""
import hashlib
import json
from pathlib import Path
import numpy as np

from lewm.pulse_timed_training_runner_development import state_digest
from scripts.navigation_artifact_root_development import BASE, validate_root

REGISTRY = Path('docs/go2_multiseed_navigation_models_2026-09-15.json')
REGISTRY_SHA256 = '2ed1cb433886b0f29c1f4add68b69ad3eba2ea0446272bf59b3c95575922ccff'
NO_RGB_REGISTRY = Path('docs/go2_no_rgb_navigation_models_2026-09-15.json')
NO_RGB_REGISTRY_SHA256 = '2fc9d21c99cba417ca45be2caf99f6b85ccd724ca71ad69d3b88057d10b1cda6'


def registry_identity(variant='full'):
    if variant not in ('full', 'no_rgb'):
        raise ValueError('fixed full/no-RGB input treatment required')
    return (REGISTRY, REGISTRY_SHA256) if variant == 'full' else (NO_RGB_REGISTRY, NO_RGB_REGISTRY_SHA256)


def registry(variant='full'):
    path, identity = registry_identity(variant)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != identity:
        raise ValueError('fixed nine-model registry changed')
    return json.loads(data)['models']


def assigned_fit(model, *, training_seed, condition, variant='full'):
    assignment = f'seed_{training_seed}_{variant}_{condition}'
    entry = registry(variant)[assignment]
    if state_digest(model.state_dict()) != entry['model_state_sha256']:
        raise ValueError('loaded model differs from the assigned correction base')
    path = validate_root(BASE / entry['root_name']) / 'residual_fit.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['fit_sha256']:
        raise ValueError('assigned correction changed')
    shapes = dict(mean=(8,42), scale=(8,42), bias=(8,2), coefficient=(8,42,2))
    with np.load(path, allow_pickle=False) as arrays:
        fit = {k: arrays[k].copy() for k in shapes}
    if (any(v.shape != shapes[k] or not np.isfinite(v).all() for k,v in fit.items())
            or not np.all(fit['scale'] > 0)):
        raise ValueError('finite complete frozen motion correction required')
    return fit, dict(fit_sha256=entry['fit_sha256'], correction_root=entry['root_name'],
        correction_base_model=assignment)


class SeededMotionCorrectionMixin:
    def __init__(self, model, *, training_seed, condition, variant='full', **kwargs):
        fit, binding = assigned_fit(model, training_seed=training_seed, condition=condition, variant=variant)
        extra = {} if variant == 'full' else dict(_assigned_motion_fit=(fit, binding))
        super().__init__(model, condition=condition, variant=variant, **extra, **kwargs)
        # No observations are submitted until construction completes. The
        # downstream corrector uses this same object, including its causal history.
        self.motion_residual.fit = fit
        self.matched_residual_binding = binding
