"""C3 motion-decoder input variants for development (Andrew, 30 Sep 2026).

The base decoder, `DenseVisualMotionReadout`, sees only the current and predicted-future pooled
features. The variants add inputs without disturbing its starting point: every new weight
starts at zero, so a variant initialised from the base behaves exactly like the base until
trained.

- `past_frames` (a): each token also gets [current - frame 0.5 s ago, current - frame 1 s ago],
  pooled from the same V-JEPA features. Its inputs stay visual.
- `history` (b): an embedding of the 1.5-s applied-command history (15 steps of forward and
  yaw, normalised as for C4) is added to the decoder's hidden layer.

Size (Andrew, 30 Sep evening: neither input variant closed most of the gap to C4, so try a larger
decoder; C4 has ~17M parameters): `proj` channels per token (base 32), `hidden` units (base
128) and `depth` residual blocks (base 0). The base weights are copied into the first
channels/units; every new unit has default-initialised incoming weights and zero outgoing
weights, and each residual block's last layer starts at zero. A larger decoder therefore still
starts exactly equal to the base, and every new parameter receives gradient.
"""
import copy

import torch
from torch import nn

BASE_PROJ, BASE_HIDDEN = 32, 128


class ReadoutVariant(nn.Module):
    def __init__(self, base, past_frames=False, history=False, proj=BASE_PROJ, hidden=BASE_HIDDEN, depth=0):
        super().__init__()
        self.past_frames, self.history = past_frames, history
        self.config = dict(proj=proj, hidden=hidden, depth=depth)
        base = copy.deepcopy(base)
        self.register_buffer('target_mean', base.target_mean.clone())
        self.register_buffer('target_scale', base.target_scale.clone())
        linear, base_hidden, base_out = base.project[0], base.decode[1], base.decode[3]
        assert (linear.out_features, base_hidden.out_features) == (BASE_PROJ, BASE_HIDDEN) and proj >= BASE_PROJ and hidden >= BASE_HIDDEN
        width = 4096 if past_frames else 2048
        tokens = base_hidden.in_features//BASE_PROJ
        self.project = nn.Sequential(nn.Linear(width, proj), nn.GELU())
        self.flatten, self.act = base.decode[0], base.decode[2]
        self.hidden = nn.Linear(tokens*proj, hidden)
        self.out = nn.Linear(hidden, 3)
        with torch.no_grad():
            self.project[0].weight[:BASE_PROJ].zero_()
            self.project[0].weight[:BASE_PROJ, :2048] = linear.weight
            self.project[0].bias[:BASE_PROJ] = linear.bias
            # Token-major flatten: input index t*proj+c. Base units read only the base channels.
            w = self.hidden.weight.view(hidden, tokens, proj)
            w[:BASE_HIDDEN].zero_()
            w[:BASE_HIDDEN, :, :BASE_PROJ] = base_hidden.weight.view(BASE_HIDDEN, tokens, BASE_PROJ)
            self.hidden.bias[:BASE_HIDDEN] = base_hidden.bias
            self.out.weight.zero_()
            self.out.weight[:, :BASE_HIDDEN] = base_out.weight
            self.out.bias.copy_(base_out.bias)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)) for _ in range(depth))
        with torch.no_grad():
            for block in self.blocks:
                block[2].weight.zero_()
                block[2].bias.zero_()
        if history:
            self.history_embedding = nn.Sequential(nn.Linear(30, 64), nn.GELU(), nn.Linear(64, hidden))
            with torch.no_grad():
                self.history_embedding[2].weight.zero_()
                self.history_embedding[2].bias.zero_()

    def normalized(self, current, future, past05=None, past10=None, history=None):
        parts = [current, future-current]
        if self.past_frames:
            parts += [current-past05, current-past10]
        h = self.hidden(self.flatten(self.project(torch.cat(parts, dim=-1))))
        if self.history:
            h = h+self.history_embedding(history.reshape(len(history), -1))
        a = self.act(h)
        for block in self.blocks:
            a = a+block(a)
        return self.out(a)

    def forward(self, current, future, past05=None, past10=None, history=None):
        return self.normalized(current, future, past05, past10, history)*self.target_scale+self.target_mean


class RuntimeVariantReadout(nn.Module):
    """Feed a `ReadoutVariant` its extra inputs inside the unchanged `DenseHorizonNavigationModel`.

    The model calls `readout(current, future)`. `install` makes the same call carry the pooled
    frames 0.5 s and 1 s ago (from the model's own three-frame context) and the normalised
    1.5-s applied-command history (from the same native context the predictor uses), both
    captured during the current forward pass.
    """

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.stash, self.generation = {}, 0

    def forward(self, current, future):
        stash = self.stash
        if stash.get('context_generation') != self.generation or stash.get('tokens_generation') != self.generation:
            raise ValueError('decoder extra inputs must come from the current prediction')
        n = len(current)
        past05 = stash['past05'].expand(n, -1, -1) if self.variant.past_frames else None
        past10 = stash['past10'].expand(n, -1, -1) if self.variant.past_frames else None
        history = stash['control'].expand(n, -1, -1, -1) if self.variant.history else None
        return self.variant(current, future, past05, past10, history)


class _StashingEncoder:
    def __init__(self, encoder, readout):
        self.encoder, self.readout = encoder, readout

    def tokens(self, pixels):
        from lewm.dense_horizon_navigation_development import pool_tokens
        tokens = self.encoder.tokens(pixels)
        context = torch.nn.functional.layer_norm(tokens.float(), (1024,))
        if context.shape[0] != 3:
            raise ValueError('three-frame context required')
        self.readout.stash.update(past10=pool_tokens(context[0:1]), past05=pool_tokens(context[1:2]),
                                  tokens_generation=self.readout.generation)
        return tokens

    def __getattr__(self, name):
        return getattr(self.encoder, name)


def install(model, variant):
    """Swap `model.readout` for a variant decoder and wire its extra inputs."""
    device = next(model.predictor.parameters()).device
    adapter = RuntimeVariantReadout(variant).to(device).eval().requires_grad_(False)
    model.readout = adapter
    model.encoder = _StashingEncoder(model.encoder, adapter)
    original = model.set_native_context

    def set_native_context(packets, *, observed_ns):
        original(packets, observed_ns=observed_ns)
        adapter.generation += 1
        control = model.pending_context['past_applied_commands'][:, [0, 2]].reshape(3, 5, 2).to(device)
        adapter.stash.update(control=((control-model.control_mean)/model.control_std)[None], context_generation=adapter.generation)
    model.set_native_context = set_native_context
    return model
