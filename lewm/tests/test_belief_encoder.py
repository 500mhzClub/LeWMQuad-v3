"""Unit tests for the BeliefEncoder (Stage 2 model).

Shape/normalization/variable-length/padding/gradient contracts. Runs under
pytest or standalone (no pytest dependency needed in the GPU venv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lewm.models.belief_encoder import BeliefEncoder  # noqa: E402


def test_output_shape_and_normalized():
    torch.manual_seed(0)
    enc = BeliefEncoder(latent_dim=32, embedding_dim=64).eval()
    z = torch.randn(5, 8, 32)
    out = enc(z)
    assert out.shape == (5, 64)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_variable_history_length():
    torch.manual_seed(0)
    enc = BeliefEncoder(latent_dim=16, embedding_dim=32, max_len=16).eval()
    for h in (1, 4, 8, 16):
        out = enc(torch.randn(3, h, 16))
        assert out.shape == (3, 32)


def test_rejects_overlong_history():
    enc = BeliefEncoder(latent_dim=8, max_len=8).eval()
    try:
        enc(torch.randn(2, 9, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError for history longer than max_len")


def test_padding_mask_changes_output():
    torch.manual_seed(1)
    enc = BeliefEncoder(latent_dim=16, embedding_dim=32).eval()
    z = torch.randn(2, 6, 16)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, -2:] = True  # ignore the last two steps
    out_full = enc(z)
    out_masked = enc(z, key_padding_mask=mask)
    assert not torch.allclose(out_full, out_masked, atol=1e-4)


def test_deterministic_in_eval():
    torch.manual_seed(2)
    enc = BeliefEncoder(latent_dim=16, embedding_dim=32).eval()
    z = torch.randn(4, 5, 16)
    assert torch.allclose(enc(z), enc(z))


def test_gradients_flow():
    enc = BeliefEncoder(latent_dim=16, embedding_dim=32)
    z = torch.randn(6, 4, 16)
    out = enc(z)
    loss = (out ** 2).sum()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} BeliefEncoder tests passed.")


if __name__ == "__main__":
    _run_all()
