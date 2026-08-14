from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from lewm.benchmarks import go2_matched_branch_successor_screen_v1 as reviewed
from scripts import dev_frozen_dense_representation_encoders_v1 as encoders


TIMM_MODULES = ("timm", "timm.models", "timm.models.layers")
_MISSING = object()


def _module_snapshot() -> dict[str, object]:
    return {name: sys.modules.get(name, _MISSING) for name in TIMM_MODULES}


def _assert_module_snapshot(snapshot: dict[str, object]) -> None:
    for name, prior in snapshot.items():
        if prior is _MISSING:
            assert name not in sys.modules
        else:
            assert sys.modules.get(name) is prior


@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("scale_by_keep", (False, True))
@pytest.mark.parametrize("drop_prob", (0.0, 0.35, 1.0))
def test_drop_path_matches_existing_reviewed_formula(
    training: bool,
    scale_by_keep: bool,
    drop_prob: float,
) -> None:
    value = torch.arange(64 * 2 * 3, dtype=torch.float32).reshape(64, 2, 3)
    torch.manual_seed(20260814)
    expected = reviewed.drop_path_compat_v1(
        value,
        drop_prob=drop_prob,
        training=training,
        scale_by_keep=scale_by_keep,
    )
    torch.manual_seed(20260814)
    actual = encoders.drop_path_compat_v1(
        value,
        drop_prob=drop_prob,
        training=training,
        scale_by_keep=scale_by_keep,
    )
    assert torch.equal(actual, expected)


def test_scoped_timm_drop_path_shim_restores_every_module() -> None:
    before = _module_snapshot()
    with encoders.scoped_timm_drop_path_shim_v1():
        from timm.models.layers import DropPath, drop_path

        assert drop_path is encoders.drop_path_compat_v1
        assert DropPath is encoders.DropPathCompatV1
    _assert_module_snapshot(before)


def test_vjepa_build_scopes_shim_to_import_and_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "never_opened.pt"
    arm = encoders.VJepa21Arm(checkpoint=checkpoint)
    monkeypatch.syspath_prepend(str(encoders.VJEPA_REPOSITORY))
    before = _module_snapshot()
    events: list[str] = []

    class FakeEncoder:
        def load_state_dict(self, state: object, *, strict: bool) -> None:
            _assert_module_snapshot(before)
            assert state == {"frozen": "encoder-state"}
            assert strict is True
            events.append("load_state_dict")

        def to(self, *, device: torch.device, dtype: torch.dtype) -> FakeEncoder:
            assert device == torch.device("cpu")
            assert dtype is torch.float32
            events.append("to")
            return self

        def eval(self) -> FakeEncoder:
            events.append("eval")
            return self

        def requires_grad_(self, value: bool) -> FakeEncoder:
            assert value is False
            events.append("requires_grad")
            return self

    encoder = FakeEncoder()
    predictor = object()

    def constructor(*, pretrained: bool) -> tuple[FakeEncoder, object]:
        from timm.models.layers import drop_path

        assert drop_path is encoders.drop_path_compat_v1
        assert pretrained is False
        events.append("constructor")
        return encoder, predictor

    def clean_backbone_key(state: object) -> object:
        _assert_module_snapshot(before)
        events.append("clean_state")
        return state

    backbones = SimpleNamespace(
        vjepa2_1_vit_large_384=constructor,
        _clean_backbone_key=clean_backbone_key,
    )
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "src.hub.backbones":
            from timm.models.layers import drop_path

            assert drop_path is encoders.drop_path_compat_v1
            events.append("import_backbones")
            return backbones
        return real_import_module(name, package)

    real_is_file = Path.is_file

    def fake_is_file(path: Path) -> bool:
        return path == checkpoint or real_is_file(path)

    def fake_torch_load(
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, object]:
        _assert_module_snapshot(before)
        assert path == checkpoint
        assert map_location == "cpu"
        assert weights_only is False
        events.append("load_checkpoint")
        return {"ema_encoder": {"frozen": "encoder-state"}}

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(torch, "load", fake_torch_load)

    assert arm.build(torch.device("cpu"), torch.float32) is encoder
    _assert_module_snapshot(before)
    assert events == [
        "import_backbones",
        "constructor",
        "load_checkpoint",
        "clean_state",
        "load_state_dict",
        "to",
        "eval",
        "requires_grad",
    ]


def test_v03_preprocessing_and_encoder_identity_surface_is_unchanged() -> None:
    arm = encoders.VJepa21CroppedV03Arm()
    assert arm.name == "vjepa2_1_vitl_384_v03crop"
    assert arm.constructor == "vjepa2_1_vit_large_384"
    assert arm.token_grid == (24, 32)
    assert arm.token_dim == 1024
    assert arm.input_hw == (384, 512)
    assert arm.preprocess is encoders.preprocess_vjepa_v03_crop
    assert encoders.preprocessing_hash(arm) == (
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
    )
