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
assert module.PREFLIGHT_ENVIRONMENT_KEY == "LEWM_RGB_JEPA_ENCODER_PRETRAINING_V1_PREFLIGHT_JSON"
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

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.75))
            self.seen: list[object] = []

        def forward_tokens(self, image):
            self.seen.append(image.detach().clone())
            pooled = image.mean(dim=(1, 2, 3))[:, None] * self.scale
            base = torch.cat(
                [pooled + float(index) for index in range(4)], dim=1
            )
            cls = base[:, None, :]
            patches = torch.stack((base, base + 0.25), dim=1)
            return torch.cat((cls, patches), dim=1)

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state = nn.Linear(4, 4, bias=False)
            self.action = nn.Linear(9, 4, bias=False)

        def predict_step(self, state, action):
            return self.state(state) + self.action(action)[:, None, :]

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.online_geometry = nn.Identity()
            self.appearance_projector = nn.Linear(4, 4)
            self.online_target_projector = nn.Linear(4, 4)
            self.prediction_projector = nn.Linear(4, 4)
            self.predictor = Predictor()
            self.target_encoder = Encoder()
            self.target_geometry_module = nn.Identity()
            self.target_projector = nn.Linear(4, 4)
            for module in (
                self.target_encoder,
                self.target_geometry_module,
                self.target_projector,
            ):
                module.requires_grad_(False)
            self.action_margin_fraction = 0.10
            self.action_margin_floor = 1e-4
            self.sigreg_projections = 4
            self.sigreg_knots = 3
            self.spatial_target_std = 0.5
            self.action_identifiability_lambda = 1.0
            self.zero_action_lambda = 1.0
            self.appearance_sigreg_lambda = 0.09
            self.spatial_variance_lambda = 1.0

    return Tiny()


def test_current_only_adapter_isolates_next_rgb_and_target_gradients() -> None:
    torch = pytest.importorskip("torch")
    from lewm.models.phase2d_spatial_lewm import (
        action_identifiability_losses,
        normalize_spatial_tokens,
    )
    from lewm.models.sigreg import sigreg_stepwise
    from lewm.models.spatial_lewm import spatial_variance_floor_loss

    module = _load_runner("_jepa_encoder_v1_current_only")
    model = _tiny_model(torch)
    current = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(
        3, 3, 4, 4
    )
    next_rgb = (current + 1000.0).requires_grad_(True)
    action = torch.zeros((3, 9), dtype=torch.float32)
    action[torch.arange(3), torch.tensor([0, 4, 8])] = 1.0
    non_hold = torch.tensor([True, False, True])
    ops = SimpleNamespace(
        action_identifiability_losses=action_identifiability_losses,
        normalize_spatial_tokens=normalize_spatial_tokens,
        sigreg_stepwise=sigreg_stepwise,
        spatial_variance_floor_loss=spatial_variance_floor_loss,
    )

    output = module._phase_a_current_only_loss(
        model,
        current,
        next_rgb,
        action,
        non_hold,
        ops=ops,
    )
    output["loss"].backward()

    assert len(model.encoder.seen) == 1
    assert torch.equal(model.encoder.seen[0], current)
    assert len(model.target_encoder.seen) == 2
    assert torch.equal(model.target_encoder.seen[0], current)
    assert torch.equal(model.target_encoder.seen[1], next_rgb.detach())
    assert next_rgb.grad is None
    assert model.encoder.scale.grad is not None
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for target in (model.target_encoder, model.target_projector)
        for parameter in target.parameters()
    )
    expected_wrong = action.roll(shifts=1, dims=-1).argmax(dim=-1)
    assert torch.equal(expected_wrong, torch.tensor([1, 5, 0]))


def test_phase_a_partition_is_exact_and_rejects_an_extra_parameter() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_partition")
    model = _tiny_model(torch)
    partition = module._phase_a_parameter_partition(model)
    assert partition["encoder"]
    assert partition["other"]
    assert partition["target"]
    assert all(not parameter.requires_grad for parameter in partition["target"])

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


def _passing_phase_a_metric(contract) -> dict[str, object]:
    per_family = {
        family: {
            "wrong_action_minus_true_mse": 0.2,
            "zero_action_minus_non_hold_true_mse": 0.2,
            "zero_action_rows_match_non_hold_rows": True,
        }
        for family in contract.SCENE_FAMILIES
    }
    return {
        "all_values_finite": True,
        "ema_target_gradient_free": True,
        "pair_count": 495,
        "scene_family_count": 8,
        "wrong_action_pair_count": 495,
        "non_hold_pair_count": 400,
        "zero_action_pair_count": 400,
        "zero_action_rows_match_non_hold_rows": True,
        "centered_raw_patch_effective_rank": 60.0,
        "centered_projected_target_effective_rank": 60.0,
        "raw_cross_sample_variance": 0.5,
        "content_residual_spatial_diversity": 0.5,
        "true_pair_mse": 0.8,
        "shuffled_next_mse": 1.0,
        "mean_target_mse": 1.0,
        "wrong_action_mse": 1.0,
        "non_hold_true_pair_mse": 0.8,
        "zero_action_mse": 1.0,
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
    assert contract.evaluate_phase_a(metric, update0)["passed"] is True
    metric["zero_action_pair_count"] = 399
    with pytest.raises(ValueError, match="control populations"):
        contract.evaluate_phase_a(metric, update0)

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
