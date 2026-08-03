from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from lewm.benchmarks import go2_matched_branch_successor_screen_v1 as screen


@dataclass(frozen=True)
class _FakeBranch:
    action_id: int
    target_rgb_artifact_id: str

    @property
    def executed_command_tape(self) -> object:
        raise AssertionError("future executed tape was accessed")

    @property
    def labels(self) -> object:
        raise AssertionError("physical target labels were accessed")


class _FakeGroup:
    def __init__(self, role: str, index: int) -> None:
        self.role = role
        self.state_id = f"{role}-state-{index:03d}"
        self.family = f"family-{index % 8}"
        self.scene_id = f"{role}-scene-{index // 8:02d}"
        self.group_index = index + (0 if role == "train" else 128)
        self.state_index_in_scene = index % 8
        self.context_rgb_artifact_ids = tuple(
            f"{role}-{index:03d}-context-{slot}" for slot in range(3)
        )
        self.history_action_ids = (index % 9, (index + 1) % 9)
        self.branches = tuple(
            reversed(
                tuple(
                    _FakeBranch(
                        action_id=action_id,
                        target_rgb_artifact_id=(
                            f"{role}-{index:03d}-target-{action_id}"
                        ),
                    )
                    for action_id in range(9)
                )
            )
        )

    @property
    def history_executed_blocks(self) -> object:
        raise AssertionError("historical executed tape was accessed")


def _fake_bundle() -> SimpleNamespace:
    train = tuple(_FakeGroup("train", index) for index in range(128))
    evaluation = tuple(_FakeGroup("eval", index) for index in range(128))
    artifact_ids = {
        artifact_id
        for group in (*train, *evaluation)
        for artifact_id in (
            *group.context_rgb_artifact_ids,
            *(branch.target_rgb_artifact_id for branch in group.branches),
        )
    }
    return SimpleNamespace(
        groups_by_role={"train": train, "eval": evaluation},
        artifacts={artifact_id: object() for artifact_id in artifact_ids},
        access_audit={"rgb_leaf_open_count": 0},
    )


def _image_bytes(
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (224, 224),
    color: object = (0, 127, 255),
    image_format: str = "PNG",
) -> bytes:
    output = BytesIO()
    Image.new(mode, size, color=color).save(output, format=image_format)
    return output.getvalue()


def test_loader_uses_exact_preregistered_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    called: dict[str, object] = {}

    def fake_loader(root: object, **kwargs: object) -> object:
        called["root"] = root
        called.update(kwargs)
        return sentinel

    monkeypatch.setattr(screen.posthoc, "load_posthoc_bundle_v1", fake_loader)

    assert screen.load_bound_posthoc_bundle_v1() is sentinel
    assert called == {
        "root": screen.POSTHOC_ROOT,
        "expected_manifest_byte_count": 11_964,
        "expected_manifest_sha256": (
            "87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e"
        ),
        "expected_terminal_byte_count": 1_250,
        "expected_terminal_sha256": (
            "a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56"
        ),
        "terminal_review_path": screen.POSTHOC_TERMINAL_REVIEW_PATH,
        "expected_terminal_review_byte_count": 2_844,
        "expected_terminal_review_sha256": (
            "bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669"
        ),
    }


def test_train_plan_is_exact_role_separated_and_contains_no_executed_tapes() -> None:
    plan = screen.collect_train_feature_plan_v1(_fake_bundle())

    assert len(plan.states) == 128
    assert len(plan.artifact_ids) == 1_536
    assert len(plan.artifact_index_by_id) == 1_536
    assert all(artifact_id.startswith("train-") for artifact_id in plan.artifact_ids)
    assert plan.artifact_ids[:12] == (
        "train-000-context-0",
        "train-000-context-1",
        "train-000-context-2",
        *(f"train-000-target-{action_id}" for action_id in range(9)),
    )
    first = plan.states[0]
    assert first.screen_state_index == 0
    assert first.context_artifact_indices == (0, 1, 2)
    assert first.target_artifact_indices == tuple(range(3, 12))
    assert tuple(item.requested_action_id for item in first.candidate_inputs) == tuple(
        range(9)
    )
    assert all(
        item.context_rgb_artifact_ids
        == ("train-000-context-0", "train-000-context-1", "train-000-context-2")
        and item.history_action_ids == (0, 1)
        for item in first.candidate_inputs
    )
    assert set(vars(first.candidate_inputs[0])) == {
        "context_rgb_artifact_ids",
        "history_action_ids",
        "requested_action_id",
    }


def test_train_plan_rejects_eval_alias_and_opened_rgb_boundary() -> None:
    bundle = _fake_bundle()
    bundle.groups_by_role["train"][0].context_rgb_artifact_ids = (
        "eval-000-context-0",
        "train-000-context-1",
        "train-000-context-2",
    )
    with pytest.raises(screen.MatchedBranchSuccessorScreenError, match="eval-role"):
        screen.collect_train_feature_plan_v1(bundle)

    opened = _fake_bundle()
    opened.access_audit["rgb_leaf_open_count"] = 1
    with pytest.raises(screen.MatchedBranchSuccessorScreenError, match="metadata-only"):
        screen.collect_train_feature_plan_v1(opened)


def test_exact_rgb_preprocessing_for_dino_and_vjepa() -> None:
    raw = _image_bytes()
    dino = screen.preprocess_dinov2_png_bytes_v1(raw)
    vjepa = screen.preprocess_vjepa2_1_png_bytes_v1(raw)
    values = torch.tensor((0.0, 127.0 / 255.0, 1.0), dtype=torch.float32)
    expected = (values - torch.tensor(screen.IMAGENET_MEAN)) / torch.tensor(
        screen.IMAGENET_STD
    )

    assert dino.shape == (3, 224, 224)
    assert vjepa.shape == (3, 1, 384, 384)
    assert dino.dtype == torch.float32
    assert vjepa.dtype == torch.float32
    assert torch.equal(dino[:, 0, 0], expected)
    assert torch.equal(vjepa[:, 0, 0, 0], expected)
    assert torch.equal(dino[:, -1, -1], expected)
    assert torch.equal(vjepa[:, 0, -1, -1], expected)


@pytest.mark.parametrize(
    "raw",
    (
        _image_bytes(mode="L", color=127),
        _image_bytes(size=(223, 224)),
        _image_bytes(image_format="JPEG"),
        b"not a png",
    ),
)
def test_rgb_preprocessing_rejects_nonexact_inputs(raw: bytes) -> None:
    with pytest.raises(screen.MatchedBranchSuccessorScreenError):
        screen.preprocess_dinov2_png_bytes_v1(raw)
    with pytest.raises(screen.MatchedBranchSuccessorScreenError):
        screen.preprocess_vjepa2_1_png_bytes_v1(raw)


def test_dense_grid_conversion_has_fixed_shapes_and_unit_norms() -> None:
    torch.manual_seed(4)
    dino_source = torch.randn(2, 256, 384, dtype=torch.float16)
    vjepa_source = torch.randn(2, 576, 768, dtype=torch.float32)

    dino = screen.normalize_dense_token_grid_v1(dino_source)
    vjepa = screen.normalize_dense_token_grid_v1(vjepa_source)
    vjepa_area = F.interpolate(
        vjepa_source.transpose(1, 2).reshape(2, 768, 24, 24),
        size=(16, 16),
        mode="area",
    ).flatten(2).transpose(1, 2)

    assert dino.shape == (2, 256, 384)
    assert vjepa.shape == (2, 256, 768)
    assert dino.dtype == vjepa.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(dino, dim=-1), torch.ones(2, 256))
    assert torch.allclose(torch.linalg.vector_norm(vjepa, dim=-1), torch.ones(2, 256))
    assert torch.allclose(vjepa, F.normalize(vjepa_area, dim=-1))


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (torch.ones(1, 255, 384), ValueError),
        (torch.ones(1, 256, 384, dtype=torch.int64), TypeError),
        (torch.zeros(1, 256, 384), FloatingPointError),
        (torch.full((1, 256, 384), float("nan")), FloatingPointError),
    ),
)
def test_dense_grid_conversion_rejects_invalid_tensors(
    value: torch.Tensor, error: type[Exception]
) -> None:
    with pytest.raises(error):
        screen.normalize_dense_token_grid_v1(value)


def test_drop_path_eval_is_identity_and_seeded_training_is_deterministic() -> None:
    value = torch.ones(64, 2, 3)
    assert screen.drop_path_compat_v1(value, 0.5, training=False) is value

    torch.manual_seed(19)
    first = screen.drop_path_compat_v1(value, 0.5, training=True)
    torch.manual_seed(19)
    second = screen.drop_path_compat_v1(value, 0.5, training=True)
    assert torch.equal(first, second)
    assert set(first.unique().tolist()) == {0.0, 2.0}


@pytest.mark.parametrize("scale_by_keep", (False, True))
def test_drop_path_matches_the_timm_formula(scale_by_keep: bool) -> None:
    """Compare against timm's independent public ``drop_path`` formula."""

    value = torch.arange(64 * 2 * 3, dtype=torch.float32).reshape(64, 2, 3)
    keep_prob = 0.65
    shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    torch.manual_seed(23)
    random_tensor = value.new_empty(shape).bernoulli_(keep_prob)
    if scale_by_keep:
        random_tensor.div_(keep_prob)
    expected = value * random_tensor

    torch.manual_seed(23)
    actual = screen.drop_path_compat_v1(
        value,
        drop_prob=1.0 - keep_prob,
        training=True,
        scale_by_keep=scale_by_keep,
    )
    assert torch.equal(actual, expected)


def test_timm_drop_path_shim_is_scoped() -> None:
    names = ("timm", "timm.models", "timm.models.layers")
    before = {name: sys.modules.get(name) for name in names}
    with screen.scoped_timm_drop_path_shim_v1():
        from timm.models.layers import DropPath, drop_path

        assert drop_path is screen.drop_path_compat_v1
        assert DropPath is screen.DropPathCompatV1
    assert {name: sys.modules.get(name) for name in names} == before
