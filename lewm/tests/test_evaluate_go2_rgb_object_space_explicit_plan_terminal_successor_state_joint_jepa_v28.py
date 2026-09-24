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
    evaluate_go2_rgb_object_space_explicit_plan_terminal_successor_state_joint_jepa_v28
    as evaluation,
)
from scripts import (
    run_go2_rgb_object_space_explicit_plan_terminal_successor_state_joint_jepa_v28
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
    return root / evaluation.RGB_ROOT_RELATIVE_PATH_V28 / leaf


def _write_images(
    root: Path,
    rows: tuple[data.H6V2Row, ...],
    *,
    horizons: tuple[int, ...],
    delta: int = 0,
) -> None:
    for row in rows:
        for horizon in horizons:
            leaf = row.rgb[horizon]
            path = _leaf_path(root, leaf)
            path.parent.mkdir(parents=True, exist_ok=True)
            value = 20 + row.index * 25 + horizon * 12 + delta
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
        quadratic = axis.square() - axis.square().mean()
        self.register_buffer(
            "rgb_direction",
            quadratic.view(1, 64, 1, 1).expand(1, 64, 64, 64),
        )
        self.target_call_receipts: list[tuple[tuple[int, ...], bool]] = []
        self.train(True)

    def train(self, mode: bool = True) -> _TinyEvaluationModel:
        super().train(mode)
        self.ema_target.eval()
        return self

    def target_modules(self) -> tuple[nn.Module, ...]:
        return (self.ema_target,)

    def _latent(self, rgb: torch.Tensor) -> torch.Tensor:
        signal = rgb.mean(dim=(1, 2, 3), keepdim=True)
        return (
            self.spatial.expand(rgb.shape[0], -1, -1, -1)
            + signal * self.rgb_direction
        )

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        self.target_call_receipts.append((tuple(rgb.shape), torch.is_grad_enabled()))
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


def _batch_target_and_loss(
    root: Path,
    rows: tuple[data.H6V2Row, ...],
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
    list[tuple[tuple[int, ...], bool]],
]:
    model = _TinyEvaluationModel()
    with evaluation.SafeV28RGBLoader(root, rows) as loader:
        batch = evaluation.build_train_h6_microbatch(
            rows,
            row_start=0,
            loader=loader,
            device="cpu",
        )
        target = evaluation._target_batch(model, rows, loader, "cpu")
        online = model.encode_online_with_evidence(
            batch[training.H6_CURRENT_RGB_KEY_V28]
        ).latent
        prediction = model.predict_plan_successor(
            online,
            batch[training.H6_FUTURE_ACTIONS_KEY_V28],
        )
        loss = latent_energy_per_row(prediction, target).mean()
        access = loader.access_snapshot()
    return batch, target, loss, access, model.target_call_receipts


def test_endpoint_batch_and_target_ignore_every_forbidden_horizon(
    tmp_path: Path,
) -> None:
    rows = tuple(_row("train", index, (0, 1, 2, 3)) for index in range(4))
    first = tmp_path / "first"
    forbidden_changed = tmp_path / "forbidden_changed"
    terminal_changed = tmp_path / "terminal_changed"
    for root in (first, forbidden_changed, terminal_changed):
        _write_images(root, rows, horizons=(2,))
    _write_images(first, rows, horizons=(6,))
    _write_images(forbidden_changed, rows, horizons=(6,))
    _write_images(terminal_changed, rows, horizons=(6,), delta=40)
    _write_images(first, rows, horizons=(0, 1, 3, 4, 5))
    _write_images(
        forbidden_changed,
        rows,
        horizons=(0, 1, 3, 4, 5),
        delta=60,
    )

    baseline = _batch_target_and_loss(first, rows)
    forbidden = _batch_target_and_loss(forbidden_changed, rows)
    changed = _batch_target_and_loss(terminal_changed, rows)
    batch, target, loss, access, calls = baseline

    assert tuple(batch) == training.REQUIRED_H6_BATCH_KEYS_V28
    assert batch[training.H6_CURRENT_RGB_KEY_V28].shape == (4, 3, 112, 112)
    assert batch[training.H6_TERMINAL_RGB_KEY_V28].shape == (4, 3, 112, 112)
    assert batch[training.H6_FUTURE_ACTIONS_KEY_V28].shape == (4, 4)
    assert batch[training.H6_FUTURE_ACTIONS_KEY_V28].tolist() == [
        [0, 1, 2, 3]
    ] * 4
    assert all(value.dtype == torch.float32 for value in batch.values() if value.ndim > 2)
    assert access["rgb_tensor_request_count"] == 12
    assert access["rgb_open_success_count"] == 12
    assert access["rgb_decode_success_count"] == 12
    assert access["validation_cache_entry_count"] == 0
    assert access["validation_cache_insert_count"] == 0
    assert calls == [((4, 3, 112, 112), False)]

    forbidden_batch, forbidden_target, forbidden_loss, forbidden_access, forbidden_calls = forbidden
    assert tuple(forbidden_batch) == tuple(batch)
    for key in batch:
        torch.testing.assert_close(forbidden_batch[key], batch[key], rtol=0.0, atol=0.0)
    torch.testing.assert_close(forbidden_target, target, rtol=0.0, atol=0.0)
    torch.testing.assert_close(forbidden_loss, loss, rtol=0.0, atol=0.0)
    assert forbidden_access == access
    assert forbidden_calls == calls

    changed_batch, changed_target, changed_loss, changed_access, changed_calls = changed
    torch.testing.assert_close(
        changed_batch[training.H6_CURRENT_RGB_KEY_V28],
        batch[training.H6_CURRENT_RGB_KEY_V28],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        changed_batch[training.H6_TERMINAL_RGB_KEY_V28],
        batch[training.H6_TERMINAL_RGB_KEY_V28],
    )
    assert not torch.equal(changed_target, target)
    assert not torch.equal(changed_loss, loss)
    assert changed_access["rgb_tensor_request_count"] == 12
    assert changed_access["rgb_open_success_count"] == 12
    assert changed_calls == calls

    loader = evaluation.SafeV28RGBLoader(first, rows)
    loader.close()
    with pytest.raises(evaluation.V28EvaluationContractError, match="closed"):
        loader.load_current(rows[0])


def test_safe_loader_rejects_a_symlinked_rgb_leaf(tmp_path: Path) -> None:
    row = _row("val", 0, (0, 1, 2, 3))
    _write_images(tmp_path, (row,), horizons=(2, 6))
    current = _leaf_path(tmp_path, row.current_rgb)
    current.unlink()
    current.symlink_to(Path(row.rgb[6]).name)

    with evaluation.SafeV28RGBLoader(tmp_path, (row,)) as loader:
        with pytest.raises(evaluation.V28EvaluationContractError, match="no-follow"):
            loader.load_current(row)
        assert loader.access_snapshot()["rgb_open_attempt_count"] == 1
        assert loader.access_snapshot()["rgb_open_success_count"] == 0


def test_loader_rejects_forbidden_registered_horizon_before_open(
    tmp_path: Path,
) -> None:
    row = _row("val", 0, (0, 1, 2, 3))
    _write_images(tmp_path, (row,), horizons=(2, 3, 6))
    with evaluation.SafeV28RGBLoader(tmp_path, (row,)) as loader:
        with pytest.raises(
            evaluation.V28EvaluationContractError,
            match="cache allowlist",
        ):
            loader._decode_leaf(row, row.rgb[3])
        assert loader.access_snapshot()["rgb_tensor_request_count"] == 1
        assert loader.access_snapshot()["rgb_open_attempt_count"] == 0
        assert loader.access_snapshot()["rgb_open_success_count"] == 0


def test_validation_stream_matches_all_manual_endpoint_controls(
    tmp_path: Path,
) -> None:
    plans = ((0, 1, 2, 3), (0, 1, 2, 3), (0, 4, 5, 6), (0, 4, 5, 6))
    rows = tuple(_row("val", index, plan) for index, plan in enumerate(plans))
    _write_images(tmp_path, rows, horizons=(2, 6))
    family = rows[0].family
    donors = data.DonorPanels(
        tail_donor_indices=(2, 2, 0, 0),
        wrong_plan_donor_indices=(3, 2, 1, 0),
        exact_plan_wrong_scene_donor_indices=(1, 0, 3, 2),
        exact_plan_eligible_indices=(0, 1, 2, 3),
        exact_plan_counts_by_family={family: 4},
        panel_sha256="0" * 64,
    )
    model = _TinyEvaluationModel()

    with evaluation.SafeV28RGBLoader(tmp_path, rows) as loader:
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
        assert vectors.access_receipt["rgb_tensor_request_count"] == 20
        assert vectors.access_receipt["rgb_open_success_count"] == 8
        assert vectors.access_receipt["rgb_decode_success_count"] == 8
        assert vectors.access_receipt["validation_cache_miss_count"] == 8
        assert vectors.access_receipt["validation_cache_hit_count"] == 12
        assert vectors.access_receipt["validation_cache_insert_count"] == 8
        assert vectors.memory_receipt["retained_full_row_latent_count"] == 0
        assert vectors.memory_receipt["global_family_a0_sum_tensor_count"] == 1
        assert vectors.memory_receipt["complete_terminal_target_pass_count"] == 2
        assert vectors.memory_receipt["complete_ema_current_pass_count"] == 1
        assert (
            vectors.memory_receipt[
                "wrong_scene_donor_terminal_target_row_request_count"
            ]
            == 4
        )
        assert vectors.memory_receipt["expected_rgb_tensor_request_count"] == 20
        assert vectors.memory_receipt["expected_validation_cache_miss_count"] == 8
        assert vectors.memory_receipt["expected_validation_cache_hit_count"] == 12
        assert vectors.memory_receipt["validation_rgb_cache_entry_count"] == 8
        assert vectors.integrity["passed"] is True
        assert [batch for batch, _grad in model.target_call_receipts] == [
            (4, 3, 112, 112),
            (4, 3, 112, 112),
            (4, 3, 112, 112),
            (1, 3, 112, 112),
            (1, 3, 112, 112),
            (1, 3, 112, 112),
            (1, 3, 112, 112),
        ]
        assert all(not grad_enabled for _batch, grad_enabled in model.target_call_receipts)

        current = torch.stack(tuple(loader.load_current(row) for row in rows))
        terminal = torch.stack(tuple(loader.load_terminal(row) for row in rows))
        with torch.no_grad():
            targets = model.encode_target(terminal)
            persistence = model.encode_target(current)
            online = model.encode_online_with_evidence(current).latent
            correct_predictions = model.predict_plan_successor(
                online,
                torch.tensor(plans, dtype=torch.long),
            )
            wrong_plans = tuple(
                rows[index].plan for index in donors.wrong_plan_donor_indices
            )
            wrong_predictions = model.predict_plan_successor(
                online,
                torch.tensor(wrong_plans, dtype=torch.long),
            )
            tail_plans = tuple(
                rows[index].plan for index in donors.tail_donor_indices
            )
            tail_predictions = model.predict_plan_successor(
                online,
                torch.tensor(tail_plans, dtype=torch.long),
            )
            wrong_scene_targets = torch.stack(
                tuple(
                    targets[int(index)]
                    for index in donors.exact_plan_wrong_scene_donor_indices
                )
            )
            target_sum = targets.sum(dim=0)
            priors = torch.stack(
                tuple((target_sum - targets[index]) / 3.0 for index in range(4))
            )
            expected_vectors = {
                "correct": latent_energy_per_row(correct_predictions, targets),
                "persistence": latent_energy_per_row(persistence, targets),
                "wrong_plan": latent_energy_per_row(wrong_predictions, targets),
                "tail": latent_energy_per_row(tail_predictions, targets),
                "wrong_scene": latent_energy_per_row(
                    correct_predictions,
                    wrong_scene_targets,
                ),
                "mean_prior": latent_energy_per_row(correct_predictions, priors),
            }
        actual_vectors = {
            "correct": torch.tensor(vectors.correct),
            "persistence": torch.tensor(vectors.persistence),
            "wrong_plan": torch.tensor(vectors.wrong_plan),
            "tail": torch.tensor(vectors.tail),
            "wrong_scene": torch.tensor(
                [vectors.wrong_scene[index] for index in range(4)]
            ),
            "mean_prior": torch.tensor(vectors.mean_prior),
        }
        for name, expected in expected_vectors.items():
            torch.testing.assert_close(
                actual_vectors[name],
                expected,
                rtol=1e-5,
                atol=1e-6,
            )
