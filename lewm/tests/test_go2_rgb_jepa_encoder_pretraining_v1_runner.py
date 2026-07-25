from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py"


def _load_runner(name: str = "_jepa_encoder_v1_runner_test"):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_source_only() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_source_only_runner", {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert module.PREFLIGHT_ENVIRONMENT_KEY == (
    "LEWM_RGB_PATCH_WHITENED_ACTION_RESIDUAL_JEPA_V1_PREFLIGHT_JSON"
)
print("PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def _tiny_model(torch):
    nn = torch.nn
    latent_dim = 192

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.75))
            self.seen: list[object] = []

        def forward_tokens(self, image):
            self.seen.append(image.detach().clone())
            pooled = image.mean(dim=(1, 2, 3))[:, None] * self.scale
            offsets = torch.linspace(
                -0.5,
                0.5,
                latent_dim,
                device=image.device,
                dtype=image.dtype,
            )
            base = pooled + offsets[None]
            cls = base[:, None, :]
            patches = torch.stack((base, base + 0.25), dim=1)
            return torch.cat((cls, patches), dim=1)

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state = nn.Linear(latent_dim, latent_dim, bias=False)
            self.action = nn.Linear(9, latent_dim, bias=False)

        def predict_step(self, state, action):
            return self.state(state) + self.action(action)[:, None, :]

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.online_geometry = nn.Identity()
            self.appearance_projector = nn.Linear(latent_dim, latent_dim)
            self.online_target_projector = nn.Linear(latent_dim, latent_dim)
            self.prediction_projector = nn.Linear(latent_dim, latent_dim)
            self.predictor = Predictor()
            self.target_encoder = Encoder()
            self.target_geometry_module = nn.Identity()
            self.target_projector = nn.Linear(latent_dim, latent_dim)
            for module in (
                self.target_encoder,
                self.target_geometry_module,
                self.target_projector,
            ):
                module.requires_grad_(False)

    return Tiny()


def test_residual_whitening_adapter_composes_loss_and_routes_gradients() -> None:
    torch = pytest.importorskip("torch")

    module = _load_runner("_jepa_encoder_v1_current_only")
    model = _tiny_model(torch)
    current = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(
        3, 3, 4, 4
    )
    next_rgb = (current + 1000.0).requires_grad_(True)
    action = torch.zeros((3, 9), dtype=torch.float32)
    action[torch.arange(3), torch.tensor([0, 6, 8])] = 1.0
    non_hold = torch.tensor([True, False, True])

    output = module._phase_a_current_only_loss(
        model,
        current,
        next_rgb,
        action,
        non_hold,
    )
    expected = (
        output["jepa_loss"]
        + output["wrong_action_loss"]
        + output["hold_action_loss"]
        + module.contract.WHITENING_VARIANCE_WEIGHT
        * (
            output["raw_whitening_variance_loss"]
            + output["projected_whitening_variance_loss"]
        )
        + module.contract.WHITENING_COVARIANCE_WEIGHT
        * (
            output["raw_whitening_covariance_loss"]
            + output["projected_whitening_covariance_loss"]
        )
    )
    assert torch.equal(output["loss"], expected)
    assert output["prediction"].shape == (3, 2, 192)
    assert output["control_predictions"].shape == (3, 8, 2, 192)
    assert output["control_indices"].shape == (3, 8)
    requested = action.argmax(dim=1)
    assert torch.all(output["control_indices"] != requested[:, None])
    assert set(output) == {
        "loss",
        "jepa_loss",
        "wrong_action_loss",
        "hold_action_loss",
        "raw_whitening_variance_loss",
        "raw_whitening_covariance_loss",
        "projected_whitening_variance_loss",
        "projected_whitening_covariance_loss",
        "prediction",
        "control_predictions",
        "control_indices",
        "online_state",
        "raw_target_next",
        "projected_target_next",
        "projected_target_current",
    }
    output["loss"].backward()

    assert len(model.encoder.seen) == 1
    assert torch.equal(model.encoder.seen[0], current)
    assert len(model.target_encoder.seen) == 2
    assert torch.equal(model.target_encoder.seen[0], current)
    assert torch.equal(model.target_encoder.seen[1], next_rgb.detach())
    assert next_rgb.grad is None
    assert model.encoder.scale.grad is not None
    assert torch.count_nonzero(model.encoder.scale.grad).item() > 0
    for parameter in (
        model.predictor.state.weight,
        model.predictor.action.weight,
        model.prediction_projector.weight,
        model.prediction_projector.bias,
        model.online_target_projector.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert all(
        parameter.grad is None
        for parameter in model.appearance_projector.parameters()
    )
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for target in (model.target_encoder, model.target_projector)
        for parameter in target.parameters()
    )


def test_phase_a_partition_freezes_and_excludes_appearance_exactly() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_partition")
    model = _tiny_model(torch)
    with pytest.raises(PermissionError, match="frozen parameter became trainable"):
        module._phase_a_parameter_partition(model)

    model.appearance_projector.requires_grad_(False)
    model.appearance_projector.eval()
    partition = module._phase_a_parameter_partition(model)
    assert partition["encoder"]
    assert partition["other"]
    assert partition["frozen"]
    assert all(not parameter.requires_grad for parameter in partition["frozen"])
    appearance_ids = {
        id(parameter) for parameter in model.appearance_projector.parameters()
    }
    assert appearance_ids.issubset({
        id(parameter) for parameter in partition["frozen"]
    })
    assert appearance_ids.isdisjoint({
        id(parameter)
        for parameter in (*partition["encoder"], *partition["other"])
    })
    assert partition["receipt"]["appearance_projector_frozen"] is True
    assert (
        partition["receipt"][
            "appearance_projector_excluded_from_optimizer_and_clip"
        ]
        is True
    )

    model.unregistered_science_delta = torch.nn.Parameter(torch.ones(()))
    with pytest.raises(PermissionError, match="partition changed"):
        module._phase_a_parameter_partition(model)


def _pair(
    *,
    scene: str,
    content: str,
    current: str,
    next_: str,
) -> dict[str, str]:
    return {
        "scene_id": scene,
        "content_sha256": content,
        "current_endpoint_sha256": current,
        "next_endpoint_sha256": next_,
    }


def test_scene_derangements_are_deterministic_local_and_identity_safe() -> None:
    module = _load_runner("_jepa_encoder_v1_derangement")
    pairs = [
        _pair(scene="s1", content="c", current="a", next_="n1"),
        _pair(scene="s1", content="a", current="b", next_="n2"),
        _pair(scene="s1", content="b", current="b", next_="n3"),
        _pair(scene="s2", content="e", current="d", next_="n5"),
        _pair(scene="s2", content="d", current="e", next_="n4"),
    ]
    first = module._scene_derangement(
        pairs, endpoint_key="current_endpoint_sha256"
    )
    second = module._scene_derangement(
        pairs, endpoint_key="current_endpoint_sha256"
    )
    assert first == second
    for index, candidate in enumerate(first):
        assert pairs[index]["scene_id"] == pairs[candidate]["scene_id"]
        assert (
            pairs[index]["current_endpoint_sha256"]
            != pairs[candidate]["current_endpoint_sha256"]
        )

    impossible = [
        _pair(scene="s", content="a", current="same", next_="n1"),
        _pair(scene="s", content="b", current="same", next_="n2"),
    ]
    with pytest.raises(PermissionError, match="cannot be constructed"):
        module._scene_derangement(
            impossible, endpoint_key="current_endpoint_sha256"
        )


def test_diagnostics_use_all_nine_cyclic_hardest_and_real_hold_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_diagnostics")
    latent_dim = 192
    patch_count = 256

    class TokenEncoder(torch.nn.Module):
        def forward_tokens(self, image):
            scalar = image[:, 0, 0, 0]
            patches = scalar[:, None, None].expand(
                -1, patch_count, latent_dim
            )
            cls = torch.zeros(
                image.shape[0],
                1,
                latent_dim,
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat((cls, patches), dim=1)

    class DiagnosticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = TokenEncoder()
            self.online_geometry = torch.nn.Identity()
            self.target_encoder = TokenEncoder()
            self.target_geometry_module = torch.nn.Identity()
            self.target_projector = torch.nn.Identity()
            self.predictor = torch.nn.Identity()
            self.prediction_projector = torch.nn.Identity()
            self.appearance_projector = torch.nn.Linear(
                latent_dim, latent_dim
            )

    class Loader:
        def batch(
            self,
            pairs,
            indices,
            _device,
            *,
            role,
            stage,
        ):
            assert role == "checkpoint_selection"
            assert stage == "phase_a_diagnostic_update_17"
            current = torch.tensor(
                [float(index + 1) for index in indices]
            ).reshape(-1, 1, 1, 1)
            next_rgb = torch.zeros_like(current)
            requested = torch.tensor(
                [int(pairs[index]["requested_index"]) for index in indices]
            )
            action = torch.nn.functional.one_hot(
                requested, num_classes=9
            ).to(torch.float32)
            return current, next_rgb, action, requested != 6

    pair_rows = [
        {
            "scene_id": "scene",
            "family": "all_actions",
            "content_sha256": f"content_{index}",
            "current_endpoint_sha256": f"current_{index}",
            "next_endpoint_sha256": f"next_{index}",
            "requested_index": index,
        }
        for index in range(9)
    ]
    monkeypatch.setitem(
        module.contract.SELECTION_ROLE_COUNTS, "pairs", len(pair_rows)
    )
    monkeypatch.setattr(
        module.contract, "SELECTION_NON_HOLD_PAIR_COUNT", 8
    )
    monkeypatch.setattr(
        module.contract, "SCENE_FAMILIES", ("all_actions",)
    )
    monkeypatch.setattr(
        module.contract, "MICROBATCH_SIZE", len(pair_rows)
    )
    monkeypatch.setattr(module, "_effective_rank", lambda *_args: 12.0)
    monkeypatch.setattr(module, "_state_sha", lambda *_args: "a" * 64)

    state_skip_pairs: list[tuple[object, object]] = []
    requested_seen: list[int] = []

    def predict_residuals(
        _predictor,
        _projector,
        state,
        requested_actions,
        ema_current,
    ):
        requested = requested_actions.argmax(dim=1)
        requested_seen.extend(requested.tolist())
        state_skip_pairs.append((
            state[:, 0, 0].detach().clone(),
            ema_current[:, 0, 0].detach().clone(),
        ))
        grid = torch.arange(9).expand(state.shape[0], -1)
        eligible = grid != requested[:, None]
        control_indices = grid[eligible].reshape(state.shape[0], 8)
        control_energy = control_indices.to(torch.float32) + 2.0
        controls = control_energy.sqrt()[:, :, None, None].expand(
            -1, -1, state.shape[1], state.shape[2]
        )
        return SimpleNamespace(
            true=torch.ones_like(state),
            controls=controls,
            layout=SimpleNamespace(
                control_indices=control_indices,
                non_hold_mask=requested != 6,
            ),
        )

    monkeypatch.setattr(
        module,
        "_phase_a_ops",
        lambda: SimpleNamespace(
            normalize_spatial_tokens=lambda tokens: tokens,
            predict_live_and_control_residuals=predict_residuals,
        ),
    )
    model = DiagnosticModel()
    model.train()
    model.appearance_projector.eval()
    observation = module._phase_a_diagnostics(
        SimpleNamespace(torch=torch),
        model,
        Loader(),
        pair_rows,
        torch.device("cpu"),
        update=17,
    )
    metric = observation["metric"]

    assert set(requested_seen) == set(range(9))
    assert len(requested_seen) == 18
    assert all(
        torch.equal(state_values, skip_values)
        for state_values, skip_values in state_skip_pairs
    )
    assert metric["pair_count"] == 9
    assert metric["cyclic_wrong_action_pair_count"] == 9
    assert metric["all_wrong_action_candidate_count"] == 72
    assert metric["non_hold_pair_count"] == 8
    assert metric["hold_action_pair_count"] == 8
    assert metric["hold_action_rows_match_non_hold_rows"] is True
    assert metric["true_pair_mse"] == pytest.approx(1.0)
    assert metric["cyclic_wrong_action_mse"] == pytest.approx(6.0)
    assert metric["hardest_wrong_action_mse"] == pytest.approx(19.0 / 9.0)
    assert metric["hold_action_mse"] == pytest.approx(8.0)
    family = metric["per_family"]["all_actions"]
    assert family["cyclic_wrong_action_minus_true_mse"] == pytest.approx(5.0)
    assert family["hardest_wrong_action_minus_true_mse"] == pytest.approx(
        10.0 / 9.0
    )
    assert family["hold_action_minus_non_hold_true_mse"] == pytest.approx(7.0)
    assert family["hold_action_rows_match_non_hold_rows"] is True
    assert observation["rng_state_preserved"] is True
    assert observation["state_mutation_count"] == 0
    assert model.training is True
    assert model.appearance_projector.training is False


def _passing_phase_a_metric(contract) -> dict[str, object]:
    per_family = {
        family: {
            "cyclic_wrong_action_minus_true_mse": 0.2,
            "hardest_wrong_action_minus_true_mse": -100.0,
            "hold_action_minus_non_hold_true_mse": 0.2,
            "hold_action_rows_match_non_hold_rows": True,
        }
        for family in contract.SCENE_FAMILIES
    }
    return {
        "all_values_finite": True,
        "ema_target_gradient_free": True,
        "pair_count": 495,
        "scene_family_count": 8,
        "cyclic_wrong_action_pair_count": 495,
        "all_wrong_action_candidate_count": 3_960,
        "non_hold_pair_count": contract.SELECTION_NON_HOLD_PAIR_COUNT,
        "hold_action_pair_count": contract.SELECTION_NON_HOLD_PAIR_COUNT,
        "hold_action_rows_match_non_hold_rows": True,
        "centered_raw_patch_effective_rank": 60.0,
        "centered_projected_target_effective_rank": 60.0,
        "raw_cross_sample_variance": 0.5,
        "content_residual_spatial_diversity": 0.5,
        "true_pair_mse": 0.8,
        "shuffled_next_mse": 1.0,
        "mean_target_mse": 1.0,
        "cyclic_wrong_action_mse": 1.0,
        "hardest_wrong_action_mse": 0.01,
        "non_hold_true_pair_mse": 0.8,
        "hold_action_mse": 1.0,
        "shuffled_current_mse": 1.0,
        "per_family": per_family,
    }


def test_exact_phase_gates_keep_strict_and_population_boundaries() -> None:
    module = _load_runner("_jepa_encoder_v1_gates")
    contract = module.contract
    metric = _passing_phase_a_metric(contract)
    update0 = {
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 1.0,
    }
    integrity = {
        "rng_state_preserved": True,
        "state_mutation_count": 0,
    }
    terminal = contract.evaluate_phase_a(metric, update0, integrity)
    assert terminal["passed"] is True
    assert (
        terminal["ratios"][
            "true_to_hardest_wrong_action_informational"
        ]
        == pytest.approx(80.0)
    )
    assert not any(
        "hardest" in name for name in terminal["conjuncts"]
    )
    assert contract.evaluate_phase_a_continuation(
        100, metric, update0, integrity
    )["passed"] is True
    assert contract.evaluate_phase_a_continuation(
        400, metric, update0, integrity
    )["passed"] is True

    metric["all_wrong_action_candidate_count"] = 3_959
    with pytest.raises(ValueError, match="control populations"):
        contract.evaluate_phase_a(metric, update0, integrity)
    metric["all_wrong_action_candidate_count"] = 3_960

    strict_boundary = _passing_phase_a_metric(contract)
    strict_boundary["cyclic_wrong_action_mse"] = (
        strict_boundary["true_pair_mse"] / 0.99
    )
    update100 = contract.evaluate_phase_a_continuation(
        100,
        strict_boundary,
        update0,
        integrity,
    )
    assert update100["passed"] is False
    assert update100["control"] == contract.CONTROL_PHASE_A_UPDATE_100_FAIL

    bad_integrity = {
        "rng_state_preserved": True,
        "state_mutation_count": 1,
    }
    update400 = contract.evaluate_phase_a_continuation(
        400,
        _passing_phase_a_metric(contract),
        update0,
        bad_integrity,
    )
    assert update400["passed"] is False
    assert update400["control"] == contract.CONTROL_PHASE_A_UPDATE_400_FAIL

    threshold = contract.PHASE_B_PASS_THRESHOLDS
    phase_b = {
        "complete_physical_scope_count": 1,
        "margin_count": 189,
        "passed_margin_count": 98,
        "total_shortfall":
            threshold["total_shortfall_strictly_less_than"] - 1e-8,
        "rough_motion": {
            "pixel_balanced_accuracy":
                threshold[
                    "rough_pixel_balanced_accuracy_strictly_greater_than"
                ] + 1e-8,
            "ground_balanced_accuracy":
                threshold[
                    "rough_ground_balanced_accuracy_strictly_greater_than"
                ] + 1e-8,
            "depth_p95_m":
                threshold["rough_depth_p95_m_strictly_less_than"] - 1e-8,
        },
    }
    assert contract.evaluate_phase_b(phase_b)["passed"] is True
    phase_b["total_shortfall"] = threshold[
        "total_shortfall_strictly_less_than"
    ]
    assert contract.evaluate_phase_b(phase_b)["passed"] is False


def test_update100_continuation_failure_stops_and_preserves_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_early_stop")

    class TrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1, bias=False)
            self.auxiliary = torch.nn.Linear(1, 1, bias=False)
            self.frozen = torch.nn.Linear(1, 1, bias=False)
            self.frozen.requires_grad_(False)
            self.ema_updates = 0

        def update_target_encoder(self) -> None:
            self.ema_updates += 1

    model = TrainModel()
    partition = {
        "encoder": list(model.encoder.parameters()),
        "other": list(model.auxiliary.parameters()),
        "frozen": list(model.frozen.parameters()),
        "receipt": {"status": "BOUND"},
    }
    initialization = {"status": "INITIALIZED"}
    monkeypatch.setattr(
        module,
        "_phase_a_model",
        lambda *_args: (model, partition, initialization),
    )
    monkeypatch.setattr(module, "_check_gpu_time", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.contract, "PHASE_A_MAXIMUM_UPDATE", 100)
    monkeypatch.setattr(module.contract, "CHECKPOINT_UPDATES", (100,))
    monkeypatch.setattr(module.contract, "MICROBATCH_SIZE", 1)
    monkeypatch.setattr(module.contract, "MICROBATCHES_PER_UPDATE", 1)
    monkeypatch.setattr(module.contract, "EFFECTIVE_BATCH_SIZE", 1)

    def loss_stub(*_args, **_kwargs):
        objective = (
            model.encoder.weight.square().mean()
            + model.auxiliary.weight.square().mean()
        )
        zero = objective * 0.0
        return {
            "loss": objective,
            "jepa_loss": objective,
            "wrong_action_loss": zero,
            "hold_action_loss": zero,
            "raw_whitening_variance_loss": zero,
            "raw_whitening_covariance_loss": zero,
            "projected_whitening_variance_loss": zero,
            "projected_whitening_covariance_loss": zero,
        }

    monkeypatch.setattr(module, "_phase_a_current_only_loss", loss_stub)

    diagnostic_updates: list[int] = []

    def diagnostic_stub(*_args, update, **_kwargs):
        diagnostic_updates.append(update)
        return {
            "update": update,
            "metric": {
                "raw_cross_sample_variance": 1.0,
                "content_residual_spatial_diversity": 1.0,
            },
            "rng_state_preserved": True,
            "state_mutation_count": 0,
        }

    monkeypatch.setattr(module, "_phase_a_diagnostics", diagnostic_stub)
    continuation_calls: list[tuple[object, ...]] = []

    def continuation_stub(*args):
        continuation_calls.append(args)
        return {
            "update": 100,
            "passed": False,
            "control": module.contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
            "conjuncts": {"deliberate_test_failure": False},
        }

    monkeypatch.setattr(
        module.contract,
        "evaluate_phase_a_continuation",
        continuation_stub,
    )
    snapshot_updates: list[int] = []

    def snapshot_stub(*_args, update, **_kwargs):
        snapshot_updates.append(update)
        return {
            "path": f"phase_a/checkpoints/update_{update}.pt",
            "file_sha256": "1" * 64,
            "content_sha256": "2" * 64,
            "byte_count": 1,
            "state_sha256": "3" * 64,
        }

    monkeypatch.setattr(module, "_snapshot_model", snapshot_stub)
    published: dict[str, dict[str, object]] = {}

    def publish_stub(path, core):
        value = module.contract.with_content_sha256(dict(core))
        raw = module.contract.canonical_json_bytes(value) + b"\n"
        published[path.name] = value
        return value, raw

    monkeypatch.setattr(module, "_publish_json", publish_stub)

    class Loader:
        supervision_array_open_count = 0
        general_frame_loader_call_count = 0

        @staticmethod
        def batch(*_args, **_kwargs):
            return (
                torch.zeros(1, 1),
                torch.zeros(1, 1),
                torch.zeros(1, 9),
                torch.ones(1, dtype=torch.bool),
            )

    runtime = SimpleNamespace(
        torch=torch,
        model_module=SimpleNamespace(
            tensor_state_dict_sha256=lambda _state: "4" * 64
        ),
    )
    progress: dict[str, object] = {}
    returned_model, artifact = module._phase_a_train(
        runtime,
        object(),
        object(),
        Loader(),
        [{}],
        [{}],
        [0] * 100,
        torch.device("cpu"),
        tmp_path,
        gpu_started=0.0,
        progress=progress,
    )

    expected_status = module.contract.CONTROL_PHASE_A_UPDATE_100_FAIL
    assert returned_model is model
    assert diagnostic_updates == [0, 100]
    assert len(continuation_calls) == 1
    assert continuation_calls[0][0] == 100
    assert continuation_calls[0][3] == {
        "rng_state_preserved": True,
        "state_mutation_count": 0,
    }
    assert snapshot_updates == [100]
    assert model.ema_updates == 100
    assert artifact["updates"] == 100
    assert artifact["presentations"] == 100
    assert artifact["ema_update_count"] == 100
    assert artifact["status"] == expected_status
    assert artifact["gate"]["control"] == expected_status
    assert artifact["training_trace"]["row_count"] == 100
    assert published["metrics.json"]["status"] == expected_status
    assert published["metrics.json"]["selection_evaluation_updates"] == [
        0,
        100,
    ]
    assert len(published["metrics.json"]["continuation_gates"]) == 1
    assert progress["phase_a_updates"] == 100
    assert progress["phase_a_presentations"] == 100


def test_rng_observation_wrapper_restores_cpu_and_gpu_streams() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_rng")
    runtime = SimpleNamespace(torch=torch)
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(54321)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()]

    observed = module._run_with_rng_preserved(
        runtime,
        lambda: (
            torch.rand(11),
            torch.rand(7, device="cuda") if torch.cuda.is_available() else None,
        ),
    )
    assert observed[0].shape == (11,)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert len(torch.cuda.get_rng_state_all()) == len(cuda_before)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            torch.cuda.get_rng_state_all(), cuda_before, strict=True
        )
    )


def test_schedule_adapter_is_reused_deterministically_and_mutation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_jepa_encoder_v1_schedule")
    indices = list(range(16))
    digest = module.contract.canonical_json_sha256(indices)
    monkeypatch.setattr(module.contract, "MAXIMUM_PRESENTATIONS", 16)
    monkeypatch.setattr(
        module.contract,
        "CHECKPOINT_SCHEDULE_PREFIX_SHA256",
        {100: digest, 400: digest, 1000: digest},
    )
    binding = {
        "path": "bound/schedule.json",
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "byte_count": 1,
    }
    authorization = {"runtime_inputs": {"schedule": binding}}
    train_pairs = [{"content_sha256": f"p{index}"} for index in range(16)]

    class Adapter:
        @staticmethod
        def validate_bound_schedule_phase_a(*, raw, binding):
            assert raw == b"x"
            return ("state", dict(binding))

        @staticmethod
        def finalize_train_identity(*, state, ordered_train_pair_ids):
            assert state[0] == "state"
            assert ordered_train_pair_ids == [
                f"p{index}" for index in range(16)
            ]
            return list(indices), dict(binding), {"status": "PASS"}

    monkeypatch.setattr(module, "_read_bound", lambda *_args, **_kwargs: b"x")
    first, _ = module._load_schedule(
        Adapter, authorization, train_pairs
    )
    second, _ = module._load_schedule(
        Adapter, authorization, train_pairs
    )
    assert first == second == indices

    indices[-1] = 0
    with pytest.raises(PermissionError, match="schedule changed"):
        module._load_schedule(Adapter, authorization, train_pairs)


def test_phase_b_training_mode_forces_every_frozen_module_to_eval() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_phase_b_mode")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in (
                "evidence_head",
                "encoder",
                "bev_decoder",
                "predictor",
                "occupancy_head",
                "target_encoder",
                "target_bev_decoder",
            ):
                setattr(self, name, torch.nn.Dropout())

    model = Tiny()
    module._set_phase_b_mode(model, training=True)
    assert model.evidence_head.training is True
    assert all(
        not getattr(model, name).training
        for name in (
            "encoder",
            "bev_decoder",
            "predictor",
            "occupancy_head",
            "target_encoder",
            "target_bev_decoder",
        )
    )


def test_phase_b_installs_warn_only_then_restores_strict_determinism() -> None:
    module = _load_runner("_jepa_encoder_v1_determinism")
    calls: list[tuple[bool, bool]] = []
    original_from_numpy_calls: list[object] = []

    class Scalar:
        pass

    class FakeNumpy:
        generic = Scalar

    class FakeTorch:
        @staticmethod
        def from_numpy(value):
            original_from_numpy_calls.append(value)
            return "array"

        @staticmethod
        def as_tensor(value):
            return ("scalar", value)

        @staticmethod
        def use_deterministic_algorithms(enabled, *, warn_only):
            calls.append((enabled, warn_only))

    expected = (
        "grid_sampler_2d_backward_cuda does not have a deterministic "
        "implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'."
    )

    def operation():
        assert FakeTorch.from_numpy(Scalar())[0] == "scalar"
        warnings.warn(expected, UserWarning)
        return "trained"

    result, receipt = module._run_phase_b_with_reviewed_determinism(
        SimpleNamespace(torch=FakeTorch, np=FakeNumpy),
        operation,
    )
    assert result == "trained"
    assert calls == [(True, True), (True, False)]
    assert receipt["strict_deterministic_algorithms_restored"] is True
    assert receipt["unexpected_warning_count"] == 0
    assert receipt["numpy_scalar_from_numpy_adaptation_count"] == 1
    assert receipt["torch_from_numpy_restored"] is True
    assert FakeTorch.from_numpy(Scalar()) == "array"
    assert len(original_from_numpy_calls) == 1


def test_tail_depth_adapter_replaces_frozen_runtime_without_mutation() -> None:
    module = _load_runner("_jepa_encoder_v1_frozen_runtime")
    original_adapter = object()
    preserved_runtime_field = object()

    @dataclass(frozen=True)
    class FrozenRuntime:
        preserved: object
        loss_adapter: object

    expected_loss = object()
    runtime = FrozenRuntime(
        preserved=preserved_runtime_field,
        loss_adapter=original_adapter,
    )
    replacement = module._runtime_with_tail_depth_loss_adapter(
        runtime,
        SimpleNamespace(
            observable_camera_ray_v4_tail_depth_loss_v4=expected_loss
        ),
    )

    assert replacement is not runtime
    assert runtime.loss_adapter is original_adapter
    assert runtime.preserved is preserved_runtime_field
    assert replacement.preserved is preserved_runtime_field
    assert (
        replacement.loss_adapter.observable_camera_ray_v4_loss_v4
        is expected_loss
    )
