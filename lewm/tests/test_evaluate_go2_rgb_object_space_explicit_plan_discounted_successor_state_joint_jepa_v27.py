from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
import torch.nn as nn

from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as data
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    latent_energy_per_row,
)
from scripts import (
    evaluate_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27
    as evaluation,
)
from scripts import (
    run_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27
    as training,
)


def _row(
    role: str,
    index: int,
    plan: tuple[int, int, int, int],
) -> data.H6V2Row:
    family = data.FAMILIES[0]
    scene = f"{family}_{index + (0 if role == 'train' else 4096):012x}"
    environment = index % 48
    first_frame = environment + 48 * (100 + index * 50)
    leaves = tuple(
        f"{scene}/rgb/frame_{first_frame + 240 * horizon:06d}_env_"
        f"{environment:02d}.png"
        for horizon in range(7)
    )
    return data.H6V2Row(
        index=index,
        role=role,
        family=family,
        scene_id=scene,
        rgb=leaves,
        actions=(8, 8, *plan),
    )


def _leaf_path(root: Path, leaf: str) -> Path:
    return root / evaluation.RGB_ROOT_RELATIVE_PATH_V27 / leaf


def _write_images(root: Path, rows: tuple[data.H6V2Row, ...]) -> None:
    for row in rows:
        for horizon, leaf in enumerate(row.rgb[2:7]):
            path = _leaf_path(root, leaf)
            path.parent.mkdir(parents=True, exist_ok=True)
            value = 20 + row.index * 25 + horizon * 12
            Image.new(
                "RGB",
                (224, 224),
                color=(value, value + 3, value + 7),
            ).save(path, format="PNG")


class _TinyEvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.ema_target = nn.Linear(1, 1, bias=False)
        self.ema_target.requires_grad_(False)
        axis = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32)
        spatial = axis.view(1, 1, 1, 64) + axis.view(1, 1, 64, 1)
        channel = axis.view(1, 64, 1, 1)
        self.register_buffer("spatial", spatial + channel)
        self.register_buffer("plan_direction", channel.expand(1, 64, 64, 64))
        self.train(True)

    def train(self, mode: bool = True) -> _TinyEvaluationModel:
        super().train(mode)
        self.ema_target.eval()
        return self

    def target_modules(self) -> tuple[nn.Module, ...]:
        return (self.ema_target,)

    def _latent(self, rgb: torch.Tensor) -> torch.Tensor:
        signal = rgb.mean(dim=(1, 2, 3), keepdim=True)
        return self.spatial.expand(rgb.shape[0], -1, -1, -1) + signal

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._latent(rgb).detach()

    def encode_online_with_evidence(self, rgb: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent=self._latent(rgb) + self.anchor * 0.0)

    def predict_plan_successor(
        self,
        current: torch.Tensor,
        plans: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = torch.tensor(
            (0.005, 0.003, 0.002, 0.001),
            dtype=torch.float32,
            device=plans.device,
        )
        plan_value = (plans.float() * coefficients).sum(dim=1)
        return current + plan_value[:, None, None, None] * self.plan_direction


def test_safe_loader_builds_exact_four_row_training_batch(tmp_path: Path) -> None:
    rows = tuple(_row("train", index, (0, 1, 2, 3)) for index in range(4))
    _write_images(tmp_path, rows)
    loader = evaluation.SafeV27RGBLoader(tmp_path, rows)

    batch = evaluation.build_train_h6_microbatch(
        rows,
        row_start=0,
        loader=loader,
        device="cpu",
    )

    assert tuple(batch) == training.REQUIRED_H6_BATCH_KEYS_V27
    assert batch[training.H6_CURRENT_RGB_KEY_V27].shape == (4, 3, 112, 112)
    assert batch[training.H6_FUTURE_RGB_KEY_V27].shape == (4, 4, 3, 112, 112)
    assert batch[training.H6_FUTURE_ACTIONS_KEY_V27].tolist() == [
        [0, 1, 2, 3]
    ] * 4
    assert all(value.dtype == torch.float32 for value in batch.values() if value.ndim > 2)
    assert loader.access_snapshot()["rgb_open_success_count"] == 20
    assert loader.access_snapshot()["rgb_decode_success_count"] == 20
    assert loader.access_snapshot()["validation_cache_entry_count"] == 0
    assert loader.access_snapshot()["validation_cache_insert_count"] == 0

    loader.close()
    with pytest.raises(evaluation.V27EvaluationContractError, match="closed"):
        loader.load_current(rows[0])


def test_safe_loader_rejects_a_symlinked_rgb_leaf(tmp_path: Path) -> None:
    row = _row("val", 0, (0, 1, 2, 3))
    _write_images(tmp_path, (row,))
    current = _leaf_path(tmp_path, row.current_rgb)
    current.unlink()
    current.symlink_to(Path(row.future_rgb[0]).name)

    with evaluation.SafeV27RGBLoader(tmp_path, (row,)) as loader:
        with pytest.raises(evaluation.V27EvaluationContractError, match="no-follow"):
            loader.load_current(row)
        assert loader.access_snapshot()["rgb_open_attempt_count"] == 1
        assert loader.access_snapshot()["rgb_open_success_count"] == 0


def test_validation_stream_is_scalar_only_and_leave_scene_prior_is_exact(
    tmp_path: Path,
) -> None:
    plans = ((0, 1, 2, 3), (0, 1, 2, 3), (0, 4, 5, 6), (0, 4, 5, 6))
    rows = tuple(_row("val", index, plan) for index, plan in enumerate(plans))
    _write_images(tmp_path, rows)
    family = rows[0].family
    donors = data.DonorPanels(
        tail_donor_indices=(2, 2, 0, 0),
        wrong_plan_donor_indices=(2, 2, 0, 0),
        exact_plan_wrong_scene_donor_indices=(1, 0, 3, 2),
        exact_plan_eligible_indices=(0, 1, 2, 3),
        exact_plan_counts_by_family={family: 4},
        panel_sha256="0" * 64,
    )
    model = _TinyEvaluationModel()

    with evaluation.SafeV27RGBLoader(tmp_path, rows) as loader:
        vectors = evaluation.stream_validation_plan_energy_vectors(
            model,
            rows,
            donors,
            loader=loader,
            device="cpu",
        )
        assert model.training is True
        assert len(vectors.correct) == 4
        assert len(vectors.persistence) == 4
        assert set(vectors.wrong_scene) == {0, 1, 2, 3}
        assert vectors.access_receipt["rgb_tensor_request_count"] == 56
        assert vectors.access_receipt["rgb_open_success_count"] == 20
        assert vectors.access_receipt["rgb_decode_success_count"] == 20
        assert vectors.access_receipt["validation_cache_miss_count"] == 20
        assert vectors.access_receipt["validation_cache_hit_count"] == 36
        assert vectors.memory_receipt["retained_full_row_latent_count"] == 0
        assert vectors.memory_receipt["global_family_a0_sum_tensor_count"] == 1
        assert vectors.memory_receipt["complete_future_target_pass_count"] == 2
        assert vectors.memory_receipt["complete_ema_current_pass_count"] == 1
        assert vectors.memory_receipt["validation_rgb_cache_entry_count"] == 20
        assert vectors.integrity["passed"] is True

        current = torch.stack(tuple(loader.load_current(row) for row in rows))
        future = torch.stack(tuple(loader.load_future(row) for row in rows))
        with torch.no_grad():
            states = model.encode_target(future.reshape(16, 3, 112, 112))
            targets = data.discounted_successor_target(
                states.reshape(4, 4, 64, 64, 64)
            )
            online = model.encode_online_with_evidence(current).latent
            predictions = model.predict_plan_successor(
                online,
                torch.tensor(plans, dtype=torch.long),
            )
            target_sum = targets.sum(dim=0)
            priors = torch.stack(
                tuple((target_sum - targets[index]) / 3.0 for index in range(4))
            )
            expected = latent_energy_per_row(predictions, priors)
        torch.testing.assert_close(
            torch.tensor(vectors.mean_prior),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
