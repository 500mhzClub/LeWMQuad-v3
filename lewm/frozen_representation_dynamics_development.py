"""Fit history and latent transitions while observation representations stay fixed."""
import torch

MODULES = ('history', 'history_norm', 'transition')


def reset_predictor(model, initial):
    model.requires_grad_(False)
    for name in MODULES:
        module = getattr(model, name)
        module.load_state_dict(getattr(initial, name).state_dict())
        module.requires_grad_(True)
    return [parameter for name in MODULES for parameter in getattr(model, name).parameters()]


def predict_encoded(model, past, blocks, valid):
    _, hidden = model.history(past)
    z = model.history_norm(hidden[-1])
    return model.predict_latents(z, blocks, valid)


def per_context_loss(prediction, target, available):
    if prediction.shape != target.shape or available.shape != prediction.shape[:2]:
        raise ValueError('matching latent predictions and observed target mask required')
    squared = (prediction-target).square().mean(-1)
    counts = available.sum(-1)
    values = torch.where(available, squared, torch.zeros_like(squared)).sum(-1)/counts.clamp_min(1)
    return values, counts > 0


def predictor_state(model):
    return {name: {k:v.detach().clone() for k,v in getattr(model, name).state_dict().items()}
        for name in MODULES}


def install_predictor(model, state):
    if set(state) != set(MODULES):
        raise ValueError('only declared history and latent-transition modules may change')
    for name in MODULES:
        getattr(model, name).load_state_dict(state[name])
    return model.eval().requires_grad_(False)
