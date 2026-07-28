from __future__ import annotations

import importlib.util
import hashlib
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)


def _load_runner() -> Any:
    name = "_test_go2_projective_support_joint_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _ChannelVector(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(64, generator=generator) * 0.1)


class _PredictorVector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(43)
        self.weight = nn.Parameter(torch.randn(9, 64, generator=generator) * 0.05)


class _SyntheticJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ChannelVector(11)
        self.bev_lift = _ChannelVector(17)
        self.semantic_head = nn.Linear(64, 3)
        self.predictor = _PredictorVector()
        self.target_encoder = _ChannelVector(11)
        self.target_bev_lift = _ChannelVector(17)
        for module in (self.target_encoder, self.target_bev_lift):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("channel_template", torch.linspace(-1.0, 1.0, 64))
        self.predictor_forward_calls = 0
        self.target_encode_calls = 0

    def _latent(
        self,
        rgb: torch.Tensor,
        encoder: _ChannelVector,
        lift: _ChannelVector,
    ) -> torch.Tensor:
        observation = rgb.mean(dim=(1, 2, 3))[:, None]
        channel = (
            encoder.weight[None]
            + torch.tanh(lift.weight)[None]
            + observation * self.channel_template[None]
        )
        return channel[:, :, None, None].expand(-1, -1, 64, 64)

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._latent(rgb, self.encoder, self.bev_lift)

    @torch.no_grad()
    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        self.target_encode_calls += 1
        return self._latent(rgb, self.target_encoder, self.target_bev_lift).detach()

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return F.linear(
            latent.movedim(1, -1),
            self.semantic_head.weight,
            self.semantic_head.bias,
        ).movedim(-1, 1)

    def predict_all_actions(self, current_latent: torch.Tensor) -> torch.Tensor:
        self.predictor_forward_calls += 1
        return current_latent[:, None] + self.predictor.weight[
            None, :, :, None, None
        ]

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        for target, online in (
            (self.target_encoder.weight, self.encoder.weight),
            (self.target_bev_lift.weight, self.bev_lift.weight),
        ):
            target.mul_(0.996).add_(online, alpha=0.004)
        self.ema_update_count.add_(1)


def _semantic_labels() -> torch.Tensor:
    rows = torch.arange(64)[:, None]
    columns = torch.arange(64)[None, :]
    label = (rows + columns) % 3
    return label[None].expand(4, -1, -1).clone()


def _microbatches() -> list[dict[str, torch.Tensor]]:
    batches = []
    for microbatch in range(4):
        generator = torch.Generator().manual_seed(100 + microbatch)
        station_safe = torch.zeros((4, 9, 11), dtype=torch.float32)
        for row in range(4):
            for action in range(9):
                prefix = (3 * action + row + microbatch) % 12
                station_safe[row, action, :prefix] = 1.0
        batches.append({
            runner.CURRENT_RGB_KEY: torch.randn(
                4, 3, 2, 2, generator=generator
            ),
            runner.NEXT_RGB_KEY: torch.randn(
                4, 3, 2, 2, generator=generator
            ),
            runner.CURRENT_LABELS_KEY: _semantic_labels(),
            runner.NEXT_LABELS_KEY: _semantic_labels().roll(1, dims=1),
            runner.EXECUTED_ACTION_KEY: torch.tensor((0, 3, 6, 8)),
            runner.STATION_SAFE_KEY: station_safe,
        })
    return batches


def _label_rows(role: str = "train") -> tuple[dict[str, Any], ...]:
    rows = []
    for action_index, action in enumerate(
        (
            "arc_left",
            "arc_right",
            "backward",
            "forward_fast",
            "forward_medium",
            "forward_slow",
            "hold",
            "yaw_left",
            "yaw_right",
        )
    ):
        rows.append({
            "dataset_role": role,
            "role_state_index": 0,
            "pair_content_sha256": "a" * 64,
            "current_endpoint_sha256": "b" * 64,
            "scene_id": "scene-a",
            "family": "small_enclosed_maze",
            "action_index": action_index,
            "action": action,
            "station_safe": [action_index % 2 == 0] * 11,
            "immediate_primitive": {"feasible": True},
            "blind_bridge": {"feasible": True},
            "provenance": {"executed_pair_primitive": "arc_left"},
        })
    return tuple(rows)


def test_source_import_is_torch_and_numpy_free() -> None:
    source = f"""
import importlib.util, sys
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
spec=importlib.util.spec_from_file_location('_isolated_projective_runner', {str(RUNNER_PATH)!r})
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert module.MICROBATCHES_PER_UPDATE == 4
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-B", "-c", source),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_parameter_partition_and_frozen_optimizer() -> None:
    model = _SyntheticJointModel()
    partition = runner.partition_parameters_v1(model)
    assert partition.names == {
        "encoder": ("encoder.weight",),
        "lift_semantic": (
            "bev_lift.weight",
            "semantic_head.weight",
            "semantic_head.bias",
        ),
        "predictor": ("predictor.weight",),
        "target": ("target_encoder.weight", "target_bev_lift.weight"),
    }
    assert not set(map(id, partition.online)) & set(map(id, partition.target))
    optimizer = runner.build_frozen_optimizer_v1(partition)
    runner.validate_optimizer_v1(optimizer, partition)
    assert tuple(group["name"] for group in optimizer.param_groups) == (
        "encoder",
        "lift_semantic",
        "predictor",
    )
    assert tuple(group["lr"] for group in optimizer.param_groups) == (
        1e-4,
        3e-4,
        3e-4,
    )
    optimizer.param_groups[1]["lr"] = 9e-4
    with pytest.raises(RuntimeError, match="lift_semantic"):
        runner.validate_optimizer_v1(optimizer, partition)


def test_full_persistence_and_forward_cyclic_shuffled_score_helpers() -> None:
    model = _SyntheticJointModel()
    latent = model.encode_online(torch.randn(1, 3, 2, 2))
    full = runner.score_full_control_v1(model, latent)
    assert full.predicted_latents.shape == (1, 9, 64, 64, 64)
    assert full.semantic_logits.shape == (1, 9, 3, 64, 64)
    assert full.station_logits.shape == (1, 9, 11)
    assert full.prefix_utility.shape == (1, 9)
    assert bool(torch.isfinite(full.station_probabilities).all())

    shuffled = runner.score_shuffled_control_v1(model, full.predicted_latents)
    for action in range(9):
        source = (action + 1) % 9
        assert torch.equal(
            shuffled.predicted_latents[:, action],
            full.predicted_latents[:, source],
        )
        assert torch.equal(
            shuffled.semantic_logits[:, action],
            full.semantic_logits[:, source],
        )
    persistence = runner.score_persistence_control_v1(model, latent)
    assert persistence.predicted_latents is None
    assert persistence.semantic_logits.shape == (1, 3, 64, 64)
    assert persistence.station_logits.shape == (1, 9, 11)
    assert persistence.prefix_utility.shape == (1, 9)


@pytest.mark.parametrize(
    "field",
    (
        "predicted_latents",
        "semantic_logits",
        "free_log_odds",
        "station_logits",
        "station_probabilities",
        "prefix_utility",
    ),
)
def test_control_finiteness_rejects_nan_hidden_behind_finite_probabilities(
    field: str,
) -> None:
    finite = runner.BatchControlScoresV1(
        predicted_latents=torch.zeros((1, 9, 64, 1, 1)),
        semantic_logits=torch.zeros((1, 9, 3, 1, 1)),
        free_log_odds=torch.zeros((1, 9, 1, 1)),
        station_logits=torch.zeros((1, 9, 11)),
        station_probabilities=torch.full((1, 9, 11), 0.5),
        prefix_utility=torch.zeros((1, 9)),
    )
    corrupted = runner.dataclasses.replace(
        finite,
        **{field: torch.full_like(getattr(finite, field), torch.nan)},
    )

    assert runner._control_outputs_finite_v1(
        torch, finite, predicted_latents_expected=True
    )
    assert not runner._control_outputs_finite_v1(
        torch, corrupted, predicted_latents_expected=True
    )
    assert bool(torch.isfinite(corrupted.station_probabilities).all()) == (
        field != "station_probabilities"
    )


def test_role_population_propagates_hidden_control_nan_to_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = __import__("numpy")
    pair = {
        "current_endpoint_sha256": "endpoint-1",
        "next_endpoint_sha256": "endpoint-2",
        "scene_id": "scene-a",
        "family": "small_enclosed_maze",
    }
    labels = runner.FrozenRoleLabelsV1(
        role="checkpoint_selection",
        rows=(),
        state_groups=(),
        station_safe=np.zeros((1, 9, 11), dtype=bool),
        immediate_feasible=np.ones((1, 9), dtype=bool),
        blind_bridge_feasible=np.ones((1, 9), dtype=bool),
        scene_ids=("scene-a",),
        family_ids=("small_enclosed_maze",),
        endpoint_ids=("endpoint-1",),
    )

    class Model:
        training = True

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

        def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
            return rgb[:, :1, :1, :1]

        def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
            return torch.zeros((latent.shape[0], 3, 1, 1))

    class Loader:
        runtime = SimpleNamespace(torch=torch)

        def image(self, identity: str, **kwargs: Any) -> torch.Tensor:
            del kwargs
            value = 1.0 if identity == "endpoint-1" else 2.0
            return torch.full((3, 1, 1), value)

        def raster_label(self, identity: str, **kwargs: Any) -> torch.Tensor:
            del identity, kwargs
            return torch.zeros((1, 1), dtype=torch.long)

    def control_scores(
        batch_size: int, *, predicted: bool, hidden_nan: bool = False
    ) -> runner.BatchControlScoresV1:
        return runner.BatchControlScoresV1(
            predicted_latents=(
                torch.zeros((batch_size, 9, 1, 1, 1)) if predicted else None
            ),
            semantic_logits=torch.zeros(
                (batch_size, 9, 3, 1, 1)
                if predicted
                else (batch_size, 3, 1, 1)
            ),
            free_log_odds=torch.full(
                (batch_size, 9, 1, 1),
                torch.nan if hidden_nan else 0.0,
            ),
            station_logits=torch.zeros((batch_size, 9, 11)),
            station_probabilities=torch.full((batch_size, 9, 11), 0.5),
            prefix_utility=torch.zeros((batch_size, 9)),
        )

    full_calls = 0

    def full_score(model: Any, latent: torch.Tensor, **kwargs: Any) -> Any:
        nonlocal full_calls
        del model, kwargs
        full_calls += 1
        return control_scores(
            latent.shape[0], predicted=True, hidden_nan=full_calls == 1
        )

    monkeypatch.setattr(runner, "validate_pairs_against_labels_v1", lambda *args: None)
    monkeypatch.setattr(runner, "score_full_control_v1", full_score)
    monkeypatch.setattr(
        runner,
        "score_persistence_control_v1",
        lambda model, latent, **kwargs: control_scores(
            latent.shape[0], predicted=False
        ),
    )
    monkeypatch.setattr(
        runner,
        "score_shuffled_control_v1",
        lambda model, latent, **kwargs: control_scores(
            latent.shape[0], predicted=True
        ),
    )

    population = runner.score_role_population_v1(
        Model(),
        Loader(),
        (pair,),
        labels,
        {("checkpoint_selection", "scene-a", "endpoint-1"): "endpoint-2"},
        np.full((9, 11), 0.5),
        torch.device("cpu"),
        stage="checkpoint_selection",
        np=np,
    )

    assert np.isfinite(population.probabilities["full"]).all()
    assert not population.all_values_finite


def test_one_update_is_four_graphs_two_clips_one_optimizer_and_one_ema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SyntheticJointModel()
    partition = runner.partition_parameters_v1(model)
    optimizer = runner.build_frozen_optimizer_v1(partition)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer_steps = 0
    original_step = optimizer.step

    def counted_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    clip_calls: list[tuple[int, ...]] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def counted_clip(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        values = tuple(parameters)
        clip_calls.append(tuple(map(id, values)))
        return original_clip(values, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted_clip)
    result = runner.joint_training_update_v1(
        model,
        optimizer,
        _microbatches(),
    )
    assert optimizer_steps == 1
    assert model.predictor_forward_calls == 4
    assert model.target_encode_calls == 8
    assert int(model.ema_update_count) == 1
    assert clip_calls == [
        tuple(map(id, partition.representation)),
        tuple(map(id, partition.predictor)),
    ]
    assert result.accounting == runner.JointTrainingAccountingV1(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=4,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=4,
    )
    assert set(result.mean_losses) == {"S", "P", "Q", "R", "L"}
    assert all(math.isfinite(value) for value in result.mean_losses.values())
    assert all(value > 0.0 for value in result.gradient_l2.values())
    assert result.representation_clip_pre_l2 > 0.0
    assert result.predictor_clip_pre_l2 > 0.0
    assert result.ranking_eligible_rows > 0
    assert result.ranking_eligible_pairs > 0
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in partition.target)
    representation_post = torch.sqrt(sum(
        parameter.grad.detach().double().square().sum()
        for parameter in partition.representation
    ))
    predictor_post = torch.sqrt(sum(
        parameter.grad.detach().double().square().sum()
        for parameter in partition.predictor
    ))
    assert float(representation_post) <= 1.0 + 1e-6
    assert float(predictor_post) <= 1.0 + 1e-6


def test_batch_accounting_and_nonfinite_inputs_fail_before_an_update() -> None:
    with pytest.raises(RuntimeError, match="accounting"):
        runner.validate_accounting_v1(
            runner.JointTrainingAccountingV1(updates=1)
        )
    model = _SyntheticJointModel()
    optimizer = runner.build_frozen_optimizer_v1(model)
    with pytest.raises(ValueError, match="exactly four"):
        runner.joint_training_update_v1(model, optimizer, _microbatches()[:3])
    batches = _microbatches()
    batches[0][runner.CURRENT_RGB_KEY][0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        runner.joint_training_update_v1(model, optimizer, batches)
    assert int(model.ema_update_count) == 0


def test_role_freeze_pair_join_and_microbatch_never_request_fixed_negative() -> None:
    labels = runner.freeze_role_labels_v1(_label_rows(), role="train", np=__import__("numpy"))
    pair = {
        "dataset_role": "train",
        "content_sha256": "a" * 64,
        "current_endpoint_sha256": "b" * 64,
        "next_endpoint_sha256": "c" * 64,
        "scene_id": "scene-a",
        "family": "small_enclosed_maze",
        "primitive": "arc_left",
    }
    runner.validate_pairs_against_labels_v1((pair,), labels)

    class Loader:
        runtime = SimpleNamespace(torch=torch)

        def __init__(self) -> None:
            self.image_kinds: list[str] = []

        def image(self, identity: str, **kwargs: Any) -> torch.Tensor:
            del identity
            self.image_kinds.append(kwargs["kind"])
            return torch.zeros((3, 2, 2))

        def raster_label(self, identity: str, **kwargs: Any) -> torch.Tensor:
            del identity, kwargs
            return torch.zeros((64, 64), dtype=torch.uint8)

    loader = Loader()
    batch = runner.build_microbatch_v1(
        loader,
        (pair,),
        labels,
        (0, 0, 0, 0),
        torch.device("cpu"),
        stage="train_update_1",
        action_order=tuple(row["action"] for row in _label_rows()),
    )
    assert loader.image_kinds == ["current"] * 4 + ["next"] * 4
    assert "fixed_negative" not in loader.image_kinds
    assert set(batch) == set(runner.REQUIRED_BATCH_KEYS)
    assert batch[runner.STATION_SAFE_KEY].shape == (4, 9, 11)


def test_wrong_rgb_is_bound_and_never_the_paired_future_endpoint() -> None:
    rows = tuple(
        {
            **row,
            "role_state_index": state,
            "pair_content_sha256": ("a" if state == 0 else "d") * 64,
            "current_endpoint_sha256": ("b" if state == 0 else "e") * 64,
        }
        for state in range(2)
        for row in _label_rows()
    )
    labels = runner.freeze_role_labels_v1(rows, role="train", np=__import__("numpy"))
    from lewm.benchmarks import (
        go2_post_action_projective_support_metrics_v1 as metrics,
    )

    mapping = runner.build_wrong_rgb_mapping_v1(labels, metrics=metrics)
    pairs = (
        {
            "dataset_role": "train",
            "scene_id": "scene-a",
            "current_endpoint_sha256": "b" * 64,
            "next_endpoint_sha256": "c" * 64,
        },
        {
            "dataset_role": "train",
            "scene_id": "scene-a",
            "current_endpoint_sha256": "e" * 64,
            "next_endpoint_sha256": "f" * 64,
        },
    )
    binding = {
        "wrong_rgb_mapping": {
            "algorithm": "role_scene_local_lexicographic_cyclic_derangement_v1",
            "roles": [
                "train",
                "probability_calibration",
                "checkpoint_selection",
            ],
            "paired_next_collision_count": 0,
            "paired_next_collision_rows_sha256": hashlib.sha256(b"[]").hexdigest(),
            "mapped_endpoint_is_never_paired_next": True,
            "per_role": {
                "train": {
                    "row_count": len(mapping.rows),
                    "mapping_sha256": mapping.mapping_sha256,
                }
            },
        }
    }
    runner.validate_wrong_rgb_role_binding_v1(
        binding, mapping, pairs, role="train"
    )
    collided = ({**pairs[0], "next_endpoint_sha256": "e" * 64}, pairs[1])
    with pytest.raises(PermissionError, match="paired future endpoint"):
        runner.validate_wrong_rgb_role_binding_v1(
            binding, mapping, collided, role="train"
        )


def test_arm_proxy_relabels_every_legacy_access() -> None:
    calls: list[tuple[str, str]] = []

    class Target:
        marker = 7

        def read_rgb(self, *args: Any, **kwargs: Any) -> bytes:
            del args
            calls.append(("rgb", kwargs["arm"]))
            return b"rgb"

        def _shard(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args
            calls.append(("shard", kwargs["arm"]))
            return {}

        def _row_array(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            del args
            calls.append(("row", kwargs["arm"]))
            return torch.zeros(())

    proxy = runner.ExperimentArmRawInputsProxyV1(Target())
    assert proxy.marker == 7
    assert proxy.read_rgb("p", "h", role="train", arm="legacy", stage="s") == b"rgb"
    proxy._shard({}, arm="legacy", stage="s")
    proxy._row_array({}, {}, "raster_labels.u1", arm="legacy", stage="s")
    assert calls == [
        ("rgb", runner.EXPERIMENT_ARM),
        ("shard", runner.EXPERIMENT_ARM),
        ("row", runner.EXPERIMENT_ARM),
    ]


def test_fixed_training_driver_consumes_one_exact_schedule_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = runner.freeze_role_labels_v1(_label_rows(), role="train", np=__import__("numpy"))
    pairs = ({
        "dataset_role": "train",
        "content_sha256": "a" * 64,
        "current_endpoint_sha256": "b" * 64,
        "next_endpoint_sha256": "c" * 64,
        "scene_id": "scene-a",
        "family": "small_enclosed_maze",
        "primitive": "arc_left",
    },)
    built_indices: list[tuple[int, ...]] = []

    def fake_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        built_indices.append(tuple(args[3]))
        return {}

    def fake_update(
        model: Any,
        optimizer: Any,
        microbatches: Any,
        *,
        accounting: runner.JointTrainingAccountingV1,
        full_masks: Any,
    ) -> runner.JointUpdateResultV1:
        del model, optimizer, full_masks
        assert len(microbatches) == 4
        return runner.JointUpdateResultV1(
            accounting=runner._advanced_accounting(accounting),
            mean_losses={name: 1.0 for name in ("S", "P", "Q", "R", "L")},
            gradient_l2={name: 1.0 for name in ("encoder", "lift_semantic", "predictor")},
            representation_clip_pre_l2=1.0,
            predictor_clip_pre_l2=1.0,
            ranking_active_microbatches=2,
            ranking_eligible_rows=3,
            ranking_eligible_pairs=4,
        )

    monkeypatch.setattr(runner, "build_microbatch_v1", fake_build)
    monkeypatch.setattr(runner, "joint_training_update_v1", fake_update)
    accounting, trace, diagnostics = runner.run_fixed_training_v1(
        object(),
        object(),
        object(),
        pairs,
        labels,
        (0,) * 16_000,
        object(),
        action_order=tuple(row["action"] for row in _label_rows()),
    )
    assert accounting.updates == 1_000
    assert accounting.presentations == 16_000
    assert len(trace) == 1_000
    assert len(built_indices) == 4_000
    assert all(indices == (0, 0, 0, 0) for indices in built_indices)
    assert diagnostics["active_r_microbatch_count"] == 2_000
    with pytest.raises(PermissionError, match="cap"):
        runner.run_fixed_training_v1(
            object(), object(), object(), pairs, labels, (0,) * 16_000, object(),
            action_order=(), maximum_updates=999,
        )


def test_execution_envelope_freezes_caps_runtime_mapping_and_denials() -> None:
    denials = {"g2_authorized": False, "retry_authorized": False}
    contract = SimpleNamespace(
        EXECUTION_BINDING_SCHEMA="binding-v1",
        OUTPUT_ROOT_RELATIVE_PATH=".generated/attempt_v1",
        RUNTIME_INTERPRETER_PATH="/runtime/python",
        RUNTIME_SYS_PREFIX="/runtime",
        DOWNSTREAM_DENIALS=denials,
    )
    authority = SimpleNamespace(
        AUTHORIZATION_STATUS="AUTHORIZED",
        WRONG_RGB_MAPPING_ALGORITHM=(
            "role_scene_local_lexicographic_cyclic_derangement_v1"
        ),
    )
    binding = {
        "schema": "binding-v1",
        "status": "AUTHORIZED",
        "output_root": ".generated/attempt_v1",
        "caps": {
            "attempts": 1,
            "updates": 1_000,
            "presentations": 16_000,
            "microbatch_size": 4,
            "microbatches_per_update": 4,
            "effective_batch_size": 16,
            "target_ema_momentum": 0.996,
        },
        "seeds": {
            "initialization": 20260712,
            "schedule": 20260713,
            "experiment": 20260728,
            "bootstrap": 20260728,
        },
        "attempt": {
            "index": 1,
            "maximum_attempts": 1,
            "fresh": True,
            "retry": False,
            "resume": False,
        },
        "authority": {
            "one_exact_fresh_attempt_authorized": True,
            "retry_or_resume_authorized": False,
        },
        "runtime": {
            "interpreter_path": "/runtime/python",
            "sys_prefix": "/runtime",
        },
        "wrong_rgb_mapping": {
            "algorithm": authority.WRONG_RGB_MAPPING_ALGORITHM,
        },
        "downstream_denials": denials,
    }
    runner.validate_execution_envelope_v1(
        binding, contract=contract, authority=authority
    )
    binding["caps"] = {**binding["caps"], "updates": 999}
    with pytest.raises(PermissionError, match="envelope"):
        runner.validate_execution_envelope_v1(
            binding, contract=contract, authority=authority
        )


@pytest.mark.parametrize("injected_name", ("artifact.json", "result.json"))
def test_partial_terminal_publication_still_finishes_complete_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected_name: str,
) -> None:
    from lewm.benchmarks import (
        go2_post_action_projective_support_corridor_contract_v1 as contract,
    )
    from lewm.benchmarks import (
        go2_post_action_projective_support_metrics_v1 as metrics,
    )

    output_root = tmp_path / "attempt"
    output_root.mkdir()
    reservation = contract.with_content_sha256({"schema": "test_reservation_v1"})
    reservation_raw = contract.canonical_json_bytes(reservation) + b"\n"
    (output_root / "reservation.json").write_bytes(reservation_raw)
    gate = metrics.GateDecision(
        status="FAIL",
        passed=False,
        checks={"synthetic": False},
        failed_checks=("synthetic",),
        comparisons={},
        failed_calibration_arms=(),
    )
    science = {
        "gate": gate,
        "semantic": None,
        "integrity": None,
        "calibrations": {},
        "evaluations": {},
        "wrong_rgb_mapping": {},
        "immediate_support_regression": None,
    }
    progress = {
        "stage": "terminal_receipts",
        "role_transitions": [],
        "roles_opened": [],
        "accounting": {"updates": 1_000, "presentations": 16_000},
        "trace": (),
    }
    original_write = runner._write_exclusive_v1
    injected = False

    def fail_once(path: Path, raw: bytes) -> None:
        nonlocal injected
        if path.name == injected_name and not injected:
            injected = True
            raise OSError(f"injected failure at {injected_name}")
        original_write(path, raw)

    monkeypatch.setattr(runner, "_write_exclusive_v1", fail_once)
    with pytest.raises(OSError, match="injected failure") as caught:
        runner._publish_terminal_result_v1(
            contract=contract,
            output_root=output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
            science=science,
            progress=progress,
        )
    progress["stage"] = "terminal_failure:terminal_receipts"
    assert runner._publish_terminal_exception_v1(
        contract=contract,
        output_root=output_root,
        reservation=reservation,
        reservation_raw=reservation_raw,
        progress=progress,
        error=caught.value,
    ) == 2
    completed = contract.parse_canonical_json(
        (output_root / "completed.json").read_bytes(),
        name="completed",
    )
    result = contract.parse_canonical_json(
        (output_root / "result.json").read_bytes(),
        name="result",
    )
    assert completed["complete_failure_receipt"] is True
    assert completed["result"]["content_sha256"] == result["content_sha256"]
    assert (output_root / "access.json").is_file()
    assert (output_root / "failure.json").is_file()


def test_pass_completion_write_recovery_preserves_success_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.benchmarks import (
        go2_post_action_projective_support_corridor_contract_v1 as contract,
    )
    from lewm.benchmarks import (
        go2_post_action_projective_support_metrics_v1 as metrics,
    )

    output_root = tmp_path / "attempt"
    output_root.mkdir()
    reservation = contract.with_content_sha256({"schema": "test_reservation_v1"})
    reservation_raw = contract.canonical_json_bytes(reservation) + b"\n"
    (output_root / "reservation.json").write_bytes(reservation_raw)
    science = {
        "gate": metrics.GateDecision(
            status="PASS",
            passed=True,
            checks={"synthetic": True},
            failed_checks=(),
            comparisons={},
            failed_calibration_arms=(),
        ),
        "semantic": None,
        "integrity": None,
        "calibrations": {},
        "evaluations": {},
        "wrong_rgb_mapping": {},
        "immediate_support_regression": None,
    }
    progress = {
        "stage": "terminal_receipts",
        "role_transitions": [],
        "roles_opened": [],
        "accounting": {"updates": 1_000, "presentations": 16_000},
        "trace": (),
    }
    original_write = runner._write_exclusive_v1
    injected = False

    def fail_completion_once(path: Path, raw: bytes) -> None:
        nonlocal injected
        if path.name == "completed.json" and not injected:
            injected = True
            raise OSError("injected completion failure")
        original_write(path, raw)

    monkeypatch.setattr(runner, "_write_exclusive_v1", fail_completion_once)
    with pytest.raises(OSError, match="completion failure") as caught:
        runner._publish_terminal_result_v1(
            contract=contract,
            output_root=output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
            science=science,
            progress=progress,
        )
    assert runner._publish_terminal_exception_v1(
        contract=contract,
        output_root=output_root,
        reservation=reservation,
        reservation_raw=reservation_raw,
        progress=progress,
        error=caught.value,
    ) == 0
    completed = contract.parse_canonical_json(
        (output_root / "completed.json").read_bytes(), name="completed"
    )
    assert completed["mechanism_passed"] is True
    assert completed["complete_failure_receipt"] is False
    assert not (output_root / "failure.json").exists()
