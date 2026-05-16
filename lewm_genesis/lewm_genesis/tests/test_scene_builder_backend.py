"""Backend resolution tests for ``lewm_genesis.scene_builder``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lewm_genesis.scene_builder import _resolve_backend


def _fake_genesis(**symbols: int | None):
    defaults = {
        "cpu": 0,
        "gpu": None,
        "cuda": None,
        "amdgpu": None,
        "metal": None,
        "vulkan": None,
        "__version__": "test",
    }
    defaults.update(symbols)
    return SimpleNamespace(**defaults)


def test_explicit_vulkan_resolves_when_available(monkeypatch):
    monkeypatch.delenv("GS_BACKEND", raising=False)
    gs = _fake_genesis(vulkan=3)

    assert _resolve_backend(gs, "vulkan") == 3


def test_explicit_vulkan_unavailable_raises(monkeypatch):
    monkeypatch.delenv("GS_BACKEND", raising=False)
    gs = _fake_genesis()

    with pytest.raises(
        RuntimeError,
        match="Genesis backend 'vulkan' requested but unavailable",
    ):
        _resolve_backend(gs, "vulkan")


def test_gs_backend_overrides_auto(monkeypatch):
    monkeypatch.setenv("GS_BACKEND", "vulkan")
    gs = _fake_genesis(vulkan=3, gpu=1)

    assert _resolve_backend(gs, "auto") == 3


def test_gs_backend_override_fails_loudly_when_unavailable(monkeypatch):
    monkeypatch.setenv("GS_BACKEND", "vulkan")
    gs = _fake_genesis(gpu=1)

    with pytest.raises(
        RuntimeError,
        match="Genesis backend 'vulkan' requested but unavailable",
    ):
        _resolve_backend(gs, "auto")


def test_auto_falls_back_to_cpu_without_accelerated_symbols(monkeypatch):
    monkeypatch.delenv("GS_BACKEND", raising=False)
    gs = _fake_genesis()

    assert _resolve_backend(gs, "auto") == 0


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.delenv("GS_BACKEND", raising=False)
    gs = _fake_genesis()

    with pytest.raises(ValueError, match="Unknown Genesis backend"):
        _resolve_backend(gs, "warpdrive")
