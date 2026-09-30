"""Check the runtime input path for C3 decoder variants against a direct call (development, CPU).

A stand-in model repeats the `DenseHorizonNavigationModel.forward` call pattern: the encoder
gives three frames of tokens, `set_native_context` precedes each prediction, and
`readout(current, future)` is called once per horizon. After `install`, the variant must see
the pooled 0.5-s and 1-s frames and the normalised command history, exactly as the fit feeds
them from the cache.
"""
import torch
from torch import nn
from torch.nn import functional as F

from lewm.dense_horizon_navigation_development import pool_tokens
from lewm.dev_readout_variants_development import ReadoutVariant, install
from scripts import train_go2_all_motion_horizon_readout_development as previous


class Encoder:
    def __init__(self):
        self.calls = 0

    def tokens(self, pixels):
        self.calls += 1
        return pixels


class StandIn(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.encoder, self.readout = Encoder(), readout
        self.predictor = nn.Linear(1, 1)
        self.register_buffer('control_mean', torch.randn(2))
        self.register_buffer('control_std', torch.rand(2)+.5)
        self.pending_context = None

    def set_native_context(self, packets, *, observed_ns):
        self.pending_context = packets

    def forward(self, future_tokens):
        native, self.pending_context = self.pending_context, None
        context = F.layer_norm(self.encoder.tokens(native['pixels']).float(), (1024,))[None]
        current = pool_tokens(context[:, -1])
        return [self.readout(current.expand(len(f), -1, -1), pool_tokens(F.layer_norm(f, (1024,)))) for f in future_tokens]


def check_large(base):
    """A larger decoder starts equal to the base and every parameter receives gradient."""
    x = [torch.randn(4, 192, 1024) for _ in range(4)]
    history = torch.randn(4, 3, 5, 2)
    reference = ReadoutVariant(base)(x[0], x[1]).detach()
    for kind in ('base', 'past_frames', 'history'):
        large = ReadoutVariant(base, past_frames=kind == 'past_frames', history=kind == 'history', proj=64, hidden=896, depth=1)
        out = large(x[0], x[1], x[2], x[3], history)
        assert torch.allclose(out, reference, atol=1e-5), (kind, float((out-reference).abs().max()))
        out.pow(2).sum().backward()
        dead = [n for n, p in large.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
        # Zero-initialised output layers pass no gradient back on the first step; their own gradients must be non-zero.
        assert not [n for n in dead if n.startswith(('out', 'blocks.0.2', 'history_embedding.2'))], dead
        count = sum(p.numel() for p in large.parameters())
        print(f'large {kind}: equals base at init, {count/1e6:.1f}M parameters')


def main():
    torch.manual_seed(0)
    base = previous.prior.previous.load('mixed_data').cpu()
    check_large(base)
    for kind, size in (('past_frames', {}), ('history', {}), ('past_frames', dict(proj=64, hidden=896, depth=1))):
        variant = ReadoutVariant(base, past_frames=kind == 'past_frames', history=kind == 'history', **size)
        with torch.no_grad():
            for p in variant.parameters():
                p.add_(.01*torch.randn_like(p))  # make the new inputs matter
        model = install(StandIn(base), variant.eval())
        for step in range(2):
            pixels = torch.randn(3, 768, 1024)
            past = torch.randn(15, 3)
            futures = [torch.randn(n, 768, 1024) for n in (1, 3)]
            model.set_native_context(dict(pixels=pixels, past_applied_commands=past), observed_ns=step)
            got = model(futures)
            context = F.layer_norm(pixels, (1024,))
            control = ((past[:, [0, 2]].reshape(3, 5, 2)-model.control_mean)/model.control_std)[None]
            for g, f in zip(got, futures):
                n = len(f)
                want = variant(pool_tokens(context[2:3]).expand(n, -1, -1), pool_tokens(F.layer_norm(f, (1024,))),
                               pool_tokens(context[1:2]).expand(n, -1, -1), pool_tokens(context[0:1]).expand(n, -1, -1),
                               control.expand(n, -1, -1, -1))
                assert torch.equal(g, want), (kind, float((g-want).abs().max()))
        try:
            model.readout(torch.zeros(1, 192, 1024), torch.zeros(1, 192, 1024))
            model.readout.generation += 1
            model.readout(torch.zeros(1, 192, 1024), torch.zeros(1, 192, 1024))
            raise AssertionError('stale extra inputs accepted')
        except ValueError:
            pass
        print(kind, 'runtime path matches the direct call; stale inputs refused')


if __name__ == '__main__':
    main()
