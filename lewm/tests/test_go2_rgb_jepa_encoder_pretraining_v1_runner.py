from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
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
    "LEWM_RGB_ACTION_CONDITIONED_LOCAL_CORRESPONDENCE_"
    "ALL_CANDIDATE_IDENTIFICATION_JEPA_V8_PREFLIGHT_JSON"
)
assert module.contract.PREREGISTRATION_COMMIT == (
    "2d5e3c01e363d4910f09597119393c57e7e8ca34"
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


def test_deferred_preflight_accepts_exact_unsigned_child_observation() -> None:
    module = _load_runner("_jepa_encoder_v1_preflight_observation")
    observation = {
        "preflight_child_process_id": 123,
        "visible_device_count": 1,
        "visible_device_index": 0,
        "visible_device_name": "AMD Radeon PRO R9700",
        "total_memory_bytes": 32_000_000_000,
        "torch_version": "test",
        "hip_version": "test",
        "tensor_allocation_count": 0,
        "payload_open_count": 0,
        "torch_device_api_call_count": 3,
    }
    stdout = (
        module.contract.canonical_json_bytes(observation).decode("ascii")
        + "\n"
    )
    assert module._parse_preflight_observation(stdout) == observation
    with pytest.raises(RuntimeError, match="not canonical"):
        module._parse_preflight_observation(stdout + "\n")


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
            spatial = torch.linspace(
                -0.5,
                0.5,
                256,
                device=image.device,
                dtype=image.dtype,
            )
            patches = base[:, None, :] + spatial[None, :, None]
            return torch.cat((cls, patches), dim=1)

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.num_spatial_tokens = 256
            self.spatial_pos_embed = nn.Parameter(
                torch.zeros(1, 256, latent_dim)
            )
            self.input_drop = nn.Identity()
            self.blocks = nn.ModuleList([Block()])
            self.norm = nn.LayerNorm(latent_dim)
            self.action_embed = nn.Linear(9, latent_dim, bias=False)

        def predict_step(self, state, action):
            raise AssertionError("V8 must bypass predictor.predict_step")

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(
                latent_dim,
                latent_dim,
                bias=False,
            )

        def forward(self, state, condition, *, causal):
            assert causal is False
            assert torch.count_nonzero(condition).item() == 0
            return state + self.projection(state)

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


def test_v8_adapter_adds_all_candidate_identification_and_routes_gradients() -> None:
    torch = pytest.importorskip("torch")

    module = _load_runner("_jepa_encoder_v1_current_only")
    model = _tiny_model(torch)
    shared_projector = model.prediction_projector
    model.prediction_projector = (
        module._phase_a_ops().ActionConditionedLocalCorrespondenceTransport(
            shared_projector
        )
    )
    current = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(
        3, 3, 4, 4
    )
    next_rgb = (current + 1000.0).requires_grad_(True)
    action = torch.zeros((3, 9), dtype=torch.float32)
    action[torch.arange(3), torch.tensor([0, 6, 8])] = 1.0

    output = module._phase_a_current_only_loss(
        model,
        current,
        next_rgb,
        action,
    )
    expected = (
        output["jepa_loss"]
        + module.contract.ACTION_INDEXED_ENERGY_NLL_WEIGHT
        * output["action_identification_loss"]
        + module.contract.LOCAL_CORRESPONDENCE_LOSS_WEIGHT
        * output["local_correspondence_loss"]
        + module.contract.CORRESPONDENCE_ACTION_IDENTIFICATION_LOSS_WEIGHT
        * output["correspondence_action_identification_loss"]
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
    assert module.contract.ACTION_INDEXED_ENERGY_NLL_WEIGHT == 1.0
    assert (
        module.contract.CORRESPONDENCE_ACTION_IDENTIFICATION_LOSS_WEIGHT
        == 1.0
    )
    assert float(output["action_identification_loss"].detach()) > 0.0
    assert float(output["correspondence_action_identification_loss"].detach()) > 0.0
    assert float(output["unscaled_correspondence_action_nll"].detach()) > 0.0
    assert torch.equal(output["loss"], expected)
    assert output["prediction"].shape == (3, 256, 192)
    assert output["all_action_predictions"].shape == (3, 9, 256, 192)
    assert output["control_predictions"].shape == (3, 8, 256, 192)
    assert output["control_indices"].shape == (3, 8)
    assert output["all_transport_logits"].shape == (3, 9, 256, 9)
    assert output["all_transport_probabilities"].shape == (3, 9, 256, 9)
    assert output["all_expected_offsets"].shape == (3, 9, 256, 2)
    assert output["all_transports"].shape == (3, 9, 256, 192)
    assert output["correspondence_action_nll_per_row"].shape == (3,)
    assert output["all_candidate_correspondence_costs"].shape == (3, 9)
    assert output["correspondence_action_scores"].shape == (3, 9)
    assert output["correspondence_action_probabilities"].shape == (3, 9)
    assert torch.equal(
        output["correspondence_action_scores"],
        -output["all_candidate_correspondence_costs"],
    )
    assert torch.equal(
        output["unscaled_correspondence_action_nll"],
        output["correspondence_action_nll_per_row"].mean(),
    )
    assert torch.allclose(
        output["correspondence_action_probabilities"].sum(dim=1),
        torch.ones(3),
        rtol=0.0,
        atol=1e-6,
    )
    assert output["local_correspondence_target"].probabilities.shape == (
        3,
        256,
        9,
    )
    requested = action.argmax(dim=1)
    assert torch.all(output["control_indices"] != requested[:, None])
    assert set(output) == {
        "loss",
        "jepa_loss",
        "action_identification_loss",
        "local_correspondence_loss",
        "local_correspondence_unscaled_cross_entropy",
        "local_correspondence_cross_entropy_per_row",
        "correspondence_action_identification_loss",
        "unscaled_correspondence_action_nll",
        "correspondence_action_nll_per_row",
        "all_candidate_correspondence_costs",
        "correspondence_action_scores",
        "correspondence_action_probabilities",
        "raw_whitening_variance_loss",
        "raw_whitening_covariance_loss",
        "projected_whitening_variance_loss",
        "projected_whitening_covariance_loss",
        "prediction",
        "all_action_predictions",
        "control_predictions",
        "control_indices",
        "all_transport_logits",
        "all_transport_probabilities",
        "all_expected_offsets",
        "all_transports",
        "local_correspondence_target",
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
        model.predictor.blocks[0].projection.weight,
        model.prediction_projector.shared_projector.weight,
        model.prediction_projector.shared_projector.bias,
        model.prediction_projector.transport_weight,
        model.online_target_projector.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert all(
        parameter.grad is None
        or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in model.predictor.action_embed.parameters()
    )
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


def test_phase_a_initialization_binds_zero_bias_free_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_operator_initialization")
    nn = torch.nn

    class GateBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(192, 6 * 192),
            )
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([GateBlock()])
            self.action_embed = nn.Linear(9, 192, bias=False)

    class InitializationModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.online_target_projector = nn.Linear(192, 192)
            self.prediction_projector = nn.Linear(192, 192)
            self.predictor = Predictor()
            self.appearance_projector = nn.Linear(192, 192)
            self.target_encoder = nn.Linear(2, 2)
            self.target_geometry_module = nn.Identity()
            self.target_projector = nn.Linear(192, 192)
            for frozen in (
                self.target_encoder,
                self.target_geometry_module,
                self.target_projector,
            ):
                frozen.requires_grad_(False)

    monkeypatch.setattr(module, "_state_sha", lambda *_args: "a" * 64)
    phase2d = SimpleNamespace(
        Phase2DSpatialLeWorldModel=lambda **_kwargs: InitializationModel()
    )
    fit = SimpleNamespace(encoder=nn.Linear(2, 2))
    model, partition, receipt = module._phase_a_model(
        SimpleNamespace(torch=torch),
        phase2d,
        fit,
        torch.device("cpu"),
    )

    initialization = receipt[
        "local_correspondence_transport_initialization"
    ]
    assert initialization["parameter_path"] == (
        "prediction_projector.transport_weight"
    )
    assert initialization["weight_shape"] == [8, 192]
    assert initialization["weight_scalar_count"] == 1_536
    assert initialization["exact_zero_weight_scalar_count"] == 1_536
    assert initialization["nonzero_weight_scalar_count"] == 0
    assert initialization["bias_parameter_count"] == 0
    assert initialization["bias"] is False
    assert initialization["action_embeddings_pairwise_distinct"] is True
    assert (
        initialization["all_eight_non_hold_relative_embeddings_nonzero"]
        is True
    )
    assert initialization["hold_relative_embedding_exactly_zero"] is True
    assert initialization["grid_shape"] == [16, 16]
    assert initialization["full_offset_order"] == [
        list(offset)
        for offset in module.contract.LOCAL_CORRESPONDENCE_FULL_OFFSETS
    ]
    assert initialization["center_offset_index"] == 4
    assert initialization["border_rule"] == "integer_index_clamp"
    assert initialization["neighbor_table_persistent"] is False
    assert (
        initialization[
            "centered_nine_logit_row_sum_exact_zero_by_construction"
        ]
        is True
    )
    assert (
        initialization[
            "uniform_student_identity_transport_at_initialization"
        ]
        is True
    )
    assert initialization["zero_initialized_without_rng_draw"] is True
    assert initialization["global_rng_state_preserved"] is True
    assert initialization["auxiliary_optimizer_learning_rate"] == 3e-4
    assert initialization["phase_b_copy_count"] == 0
    assert initialization["phase_b_optimizer_inclusion_count"] == 0
    assert model.prediction_projector.transport_weight.shape == (8, 192)
    assert (
        torch.count_nonzero(
            model.prediction_projector.transport_weight
        ).item()
        == 0
    )
    assert set(
        dict(
            model.prediction_projector.named_parameters(
                recurse=False
            )
        )
    ) == {"transport_weight"}
    assert any(
        parameter is model.prediction_projector.transport_weight
        for parameter in partition["other"]
    )


def test_update_zero_compares_all_36_action_pairs_across_495_rows() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_update_zero_symmetry")
    base = torch.arange(495 * 2 * 3, dtype=torch.float32).reshape(
        495, 1, 2, 3
    )
    predictions = base.expand(-1, 9, -1, -1).clone()
    row_count = 0
    comparison_count = None
    for batch in predictions.split(16):
        rows, pairs = module._verify_update_zero_action_symmetry_batch(
            torch,
            batch,
        )
        row_count += rows
        comparison_count = pairs if comparison_count is None else comparison_count
        assert pairs == comparison_count
    receipt = module._update_zero_action_symmetry_receipt(
        row_count=row_count,
        comparison_count=comparison_count,
    )
    assert receipt == {
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
    }

    predictions[494, 8, 1, 2] += 1.0
    with pytest.raises(PermissionError, match="not bitwise equal"):
        module._verify_update_zero_action_symmetry_batch(
            torch,
            predictions[-15:],
        )


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


def test_diagnostics_use_exact_row_correspondence_aggregations_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v1_diagnostics")
    latent_dim = 192
    patch_count = 256

    class OnlineTokenEncoder(torch.nn.Module):
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

    class TargetTokenEncoder(torch.nn.Module):
        def forward_tokens(self, image):
            scalar = image[:, 0, 0, 0]
            return scalar[:, None, None].expand(
                -1,
                patch_count + 1,
                latent_dim,
            )

    class DiagnosticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = OnlineTokenEncoder()
            self.online_geometry = torch.nn.Identity()
            self.target_encoder = TargetTokenEncoder()
            self.target_geometry_module = torch.nn.Identity()
            self.target_projector = torch.nn.Identity()
            self.predictor = torch.nn.Identity()
            self.prediction_projector = torch.nn.Module()
            self.prediction_projector.register_parameter(
                "transport_weight",
                torch.nn.Parameter(torch.ones(1)),
            )
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
            next_rgb = torch.tensor(
                [float(index + 101) for index in indices]
            ).reshape(-1, 1, 1, 1)
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

    def predict_transports(
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
        all_energy = grid.to(torch.float32) + 2.0
        all_predictions = all_energy.sqrt()[:, :, None, None].expand(
            -1, -1, state.shape[1], state.shape[2]
        ).clone()
        all_predictions[
            torch.arange(state.shape[0]),
            requested,
        ] = 1.0
        control_gather = control_indices[:, :, None, None].expand(
            -1, -1, state.shape[1], state.shape[2]
        )
        controls = all_predictions.gather(1, control_gather)
        logit_basis = torch.arange(
            9,
            dtype=state.dtype,
            device=state.device,
        ) - 4.0
        action_scale = (
            torch.arange(9, dtype=state.dtype, device=state.device)
            - 6.0
        )
        all_logits = (
            action_scale[None, :, None, None]
            * logit_basis[None, None, None, :]
            * 0.01
        ).expand(state.shape[0], -1, state.shape[1], -1).clone()
        probabilities = torch.softmax(all_logits, dim=-1)
        offsets = torch.tensor(
            module.contract.LOCAL_CORRESPONDENCE_FULL_OFFSETS,
            dtype=state.dtype,
            device=state.device,
        )
        expected_offsets = probabilities @ offsets
        expected_offsets[:, module.contract.HOLD_ACTION_INDEX].zero_()
        all_transports = ema_current[:, None].expand(
            -1, 9, -1, -1
        ).clone()
        for action_index in range(9):
            if action_index != module.contract.HOLD_ACTION_INDEX:
                all_transports[:, action_index].add_(0.1)
        return SimpleNamespace(
            executed_indices=requested,
            all_predictions=all_predictions,
            all_transport_logits=all_logits,
            all_transport_probabilities=probabilities,
            all_expected_offsets=expected_offsets,
            all_transports=all_transports,
            executed=torch.ones_like(state),
            controls=controls,
            control_indices=control_indices,
        )

    def targets(
        _projector,
        _ema_current,
        ema_next,
    ):
        basis = torch.arange(
            9,
            dtype=ema_next.dtype,
            device=ema_next.device,
        )
        logits = (
            ema_next[:, :1, :1] * basis[None, None, :] * 0.001
        ).expand(-1, patch_count, -1)
        probabilities = torch.softmax(logits, dim=-1)
        kl = (
            probabilities
            * (probabilities.log() + torch.log(torch.tensor(9.0)))
        ).sum(dim=-1).mean()
        return SimpleNamespace(
            logits=logits,
            probabilities=probabilities,
            mean_kl_to_uniform=kl,
        )

    def centered_ce(target_probabilities, student_logits):
        log_probability = torch.log_softmax(student_logits, dim=-1)
        center = log_probability[..., 4]
        return (
            -center
            - (
                target_probabilities
                * (log_probability - center[..., None])
            ).sum(dim=-1)
        )

    monkeypatch.setattr(
        module,
        "_phase_a_ops",
        lambda: SimpleNamespace(
            normalize_spatial_tokens=lambda tokens: tokens,
            predict_action_conditioned_local_transports=predict_transports,
            local_correspondence_targets=targets,
            centered_log_soft_cross_entropy=centered_ce,
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
    next_mapping = module._scene_derangement(
        pair_rows,
        endpoint_key="next_endpoint_sha256",
    )
    assert metric["pair_count"] == 9
    assert metric["cyclic_wrong_action_pair_count"] == 9
    assert metric["all_wrong_action_candidate_count"] == 72
    assert metric["non_hold_pair_count"] == 8
    assert metric["hold_action_pair_count"] == 8
    assert metric["hold_action_rows_match_non_hold_rows"] is True
    family = metric["per_family"]["all_actions"]
    assert family["hold_action_rows_match_non_hold_rows"] is True
    correspondence = metric["local_correspondence"]
    correct_next = torch.arange(101.0, 110.0)
    deranged_next = correct_next[list(next_mapping)]
    basis = torch.arange(9.0)
    correct_q = torch.softmax(
        correct_next[:, None] * basis[None] * 0.001,
        dim=-1,
    )
    deranged_q = torch.softmax(
        deranged_next[:, None] * basis[None] * 0.001,
        dim=-1,
    )
    action_scale = torch.arange(9.0) - 6.0
    all_logits = action_scale[:, None] * (basis[None] - 4.0) * 0.01
    executed_logits = all_logits[torch.arange(9)]
    expected_c = centered_ce(correct_q, executed_logits)
    expected_d = centered_ce(deranged_q, executed_logits)
    expanded_q = correct_q[:, None].expand(-1, 9, -1)
    expanded_logits = all_logits[None].expand(9, -1, -1)
    all_ce = centered_ce(expanded_q, expanded_logits)
    expected_scores = -all_ce
    expected_probabilities = torch.softmax(expected_scores, dim=1)
    expected_nll_rows = torch.nn.functional.cross_entropy(
        expected_scores,
        torch.arange(9),
        reduction="none",
    )
    expected_predictions = expected_scores.argmax(dim=1)
    expected_recalls = torch.tensor([
        float(expected_predictions[action_index] == action_index)
        for action_index in range(9)
    ])
    wrong_mask = ~torch.eye(9, dtype=torch.bool)
    expected_h = all_ce[wrong_mask].reshape(9, 8).min(dim=1).values
    assert correspondence["correct_centered_log_cross_entropy"] == (
        pytest.approx(float(expected_c.mean()))
    )
    assert correspondence["deranged_centered_log_cross_entropy"] == (
        pytest.approx(float(expected_d.mean()))
    )
    assert correspondence[
        "correct_to_deranged_cross_entropy_ratio"
    ] == pytest.approx(float(expected_c.mean() / expected_d.mean()))
    assert correspondence[
        "hardest_wrong_centered_log_cross_entropy"
    ] == pytest.approx(float(expected_h.mean()))
    assert correspondence[
        "executed_to_hardest_wrong_cross_entropy_ratio"
    ] == pytest.approx(float(expected_c.mean() / expected_h.mean()))
    assert correspondence[
        "per_family_deranged_minus_correct_cross_entropy"
    ]["all_actions"] == pytest.approx(
        float((expected_d - expected_c).mean()),
        abs=1e-7,
    )
    assert correspondence[
        "per_family_hardest_wrong_minus_executed_cross_entropy"
    ]["all_actions"] == pytest.approx(
        float((expected_h - expected_c).mean()),
        abs=1e-7,
    )
    assert correspondence["unscaled_correspondence_action_nll"] == (
        pytest.approx(float(expected_nll_rows.mean()))
    )
    assert correspondence[
        "correspondence_action_probabilities_all_values_finite"
    ] is True
    assert correspondence[
        "correspondence_action_probability_rows_normalized"
    ] is True
    assert correspondence["correspondence_action_top1_accuracy"] == (
        pytest.approx(
            float((expected_predictions == torch.arange(9)).float().mean())
        )
    )
    per_action = correspondence[
        "per_executed_action_correspondence_identification"
    ]
    assert tuple(per_action) == module.contract.ACTION_VOCABULARY
    for action_index, action_name in enumerate(
        module.contract.ACTION_VOCABULARY
    ):
        assert per_action[action_name] == {
            "row_count": 1,
            "mean_nll": pytest.approx(float(expected_nll_rows[action_index])),
            "recall": float(expected_recalls[action_index]),
        }
    assert correspondence[
        "correspondence_action_macro_balanced_accuracy"
    ] == pytest.approx(float(expected_recalls.mean()))
    assert torch.isfinite(expected_probabilities).all()
    assert correspondence[
        "all_candidate_correspondence_costs_bitwise_equal"
    ] is False
    assert correspondence[
        "all_candidate_correspondence_scores_bitwise_equal"
    ] is False
    assert correspondence[
        "correspondence_action_posterior_bitwise_equal_to_uniform"
    ] is False
    assert correspondence[
        "correspondence_action_nll_bitwise_equal_to_zero_logit_reference"
    ] is False
    assert correspondence[
        "non_hold_action_distribution_different_from_hold_count"
    ] == 8
    assert correspondence["hold_probabilities_bitwise_uniform"] is True
    assert correspondence["hold_expected_offset_exactly_zero"] is True
    assert correspondence["hold_transport_identity_exact"] is True
    assert correspondence["all_action_transports_identity_exact"] is False
    assert observation["rng_state_preserved"] is True
    assert observation["state_mutation_count"] == 0
    assert model.training is True
    assert model.appearance_projector.training is False


def _local_correspondence_metric(contract, *, update_zero: bool) -> dict[str, object]:
    baseline = 2.1972246170043945
    correct = baseline if update_zero else 1.0
    comparison = baseline if update_zero else 1.2
    action_counts = (55,) * 9
    per_executed_action = {
        action: {
            "row_count": action_counts[index],
            "mean_nll": baseline if update_zero else 1.0,
            "recall": (
                1.0
                if (not update_zero or index == 0)
                else 0.0
            ),
        }
        for index, action in enumerate(contract.ACTION_VOCABULARY)
    }
    return {
        "all_values_finite": True,
        "target_all_values_finite": True,
        "target_all_strictly_positive": True,
        "target_rows_normalized": True,
        "student_all_strictly_positive": True,
        "student_rows_normalized": True,
        "transport_weight_all_values_finite": True,
        "transport_weight_any_nonzero": not update_zero,
        "maximum_absolute_student_logit": 0.0 if update_zero else 0.5,
        "unscaled_correspondence_action_nll":
            baseline if update_zero else 1.0,
        "correspondence_action_probabilities_all_values_finite": True,
        "correspondence_action_probability_rows_normalized": True,
        "correspondence_action_top1_accuracy":
            action_counts[0] / 495.0 if update_zero else 1.0,
        "per_executed_action_correspondence_identification":
            per_executed_action,
        "correspondence_action_macro_balanced_accuracy":
            1.0 / 9.0 if update_zero else 1.0,
        "all_candidate_correspondence_costs_bitwise_equal": update_zero,
        "all_candidate_correspondence_scores_bitwise_equal": update_zero,
        "correspondence_action_posterior_bitwise_equal_to_uniform":
            update_zero,
        "correspondence_action_nll_bitwise_equal_to_zero_logit_reference":
            update_zero,
        "correct_centered_log_cross_entropy": correct,
        "deranged_centered_log_cross_entropy": comparison,
        "correct_to_deranged_cross_entropy_ratio": correct / comparison,
        "deranged_positive_family_margin_count": 0 if update_zero else 8,
        "per_family_deranged_minus_correct_cross_entropy": {
            family: 0.0 if update_zero else 0.2
            for family in contract.SCENE_FAMILIES
        },
        "per_action_correct_target_centered_log_cross_entropy": {
            action: correct for action in contract.ACTION_VOCABULARY
        },
        "hardest_wrong_centered_log_cross_entropy": comparison,
        "executed_to_hardest_wrong_cross_entropy_ratio":
            correct / comparison,
        "hardest_wrong_positive_family_margin_count":
            0 if update_zero else 8,
        "per_family_hardest_wrong_minus_executed_cross_entropy": {
            family: 0.0 if update_zero else 0.2
            for family in contract.SCENE_FAMILIES
        },
        "mean_target_kl_to_uniform": 0.1,
        "per_action_probability_rows_positive_and_normalized": {
            action: True for action in contract.ACTION_VOCABULARY
        },
        "non_hold_action_distribution_different_from_hold_count":
            0 if update_zero else 8,
        "per_action_distribution_different_from_hold": {
            action: False if update_zero else action != "hold"
            for action in contract.ACTION_VOCABULARY
        },
        "maximum_absolute_expected_offset_component":
            0.0 if update_zero else 0.5,
        "hold_probabilities_bitwise_uniform": True,
        "hold_expected_offset_exactly_zero": True,
        "hold_transport_identity_exact": True,
        "all_action_distributions_bitwise_equal_to_hold": update_zero,
        "all_action_distributions_bitwise_equal_to_uniform": update_zero,
        "correct_and_deranged_cross_entropy_bitwise_equal": update_zero,
        "all_action_transports_identity_exact": update_zero,
    }


def _passing_phase_a_metric(contract) -> dict[str, object]:
    per_family = {
        family: {
            "cyclic_wrong_action_minus_true_mse": 0.2,
            "hardest_wrong_action_minus_true_mse": 0.2,
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
        "hardest_wrong_action_mse": 1.0,
        "non_hold_true_pair_mse": 0.8,
        "hold_action_mse": 1.0,
        "shuffled_current_mse": 1.0,
        "per_family": per_family,
        "local_correspondence":
            _local_correspondence_metric(contract, update_zero=False),
    }


def test_exact_phase_gates_keep_strict_and_population_boundaries() -> None:
    module = _load_runner("_jepa_encoder_v1_gates")
    contract = module.contract
    metric = _passing_phase_a_metric(contract)
    update0 = {
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 1.0,
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
        "local_correspondence":
            _local_correspondence_metric(contract, update_zero=True),
    }
    integrity = {
        "rng_state_preserved": True,
        "state_mutation_count": 0,
    }
    terminal = contract.evaluate_phase_a(metric, update0, integrity)
    assert terminal["passed"] is True
    assert (
        terminal["ratios"][
            "true_to_hardest_wrong_action"
        ]
        == pytest.approx(0.8)
    )
    assert (
        terminal["conjuncts"][
            "true_at_most_point95_hardest_wrong_action"
        ]
        is True
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
            "action_identification_loss": zero,
            "local_correspondence_loss": zero,
            "local_correspondence_unscaled_cross_entropy": zero,
            "correspondence_action_identification_loss": zero,
            "unscaled_correspondence_action_nll": zero,
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
                "local_correspondence": _local_correspondence_metric(
                    module.contract,
                    update_zero=update == 0,
                ),
            },
            "action_indexed_symmetry": (
                {
                    "all_action_predictions_bitwise_equal": True,
                    "all_action_unordered_pair_count": 36,
                    "all_action_prediction_row_count": 495,
                }
                if update == 0
                else None
            ),
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
        def batch(*_args, include_non_hold=True, **_kwargs):
            core = (
                torch.zeros(1, 1),
                torch.zeros(1, 1),
                torch.zeros(1, 9),
            )
            if not include_non_hold:
                return core
            return (
                *core,
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
    trace_rows = [
        json.loads(row)
        for row in (
            tmp_path / "phase_a/training_trace.jsonl"
        ).read_text(encoding="ascii").splitlines()
    ]
    assert len(trace_rows) == 100
    assert {
        "correspondence_action_identification_loss",
        "unscaled_correspondence_action_nll",
    }.issubset(trace_rows[-1]["losses"])
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
    first_progress: dict[str, object] = {}
    second_progress: dict[str, object] = {}
    first, _ = module._load_schedule(
        Adapter,
        authorization,
        train_pairs,
        progress=first_progress,
    )
    second, _ = module._load_schedule(
        Adapter,
        authorization,
        train_pairs,
        progress=second_progress,
    )
    assert first == second == indices
    assert first_progress["schedule_open_attempted"] is True
    assert first_progress["schedule_open_succeeded"] is True
    assert first_progress["schedule_validated"] is True

    indices[-1] = 0
    with pytest.raises(PermissionError, match="schedule changed"):
        module._load_schedule(
            Adapter,
            authorization,
            train_pairs,
            progress={},
        )


def test_failure_custody_receipt_is_explicit_before_any_runtime_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_jepa_encoder_v1_failure_custody")
    monkeypatch.setattr(module, "sys", SimpleNamespace(modules={}))

    def binding(path: str, marker: str) -> dict[str, object]:
        return {
            "path": path,
            "file_sha256": marker * 64,
            "content_sha256": marker * 64,
            "byte_count": 1,
        }

    authorization = {
        "runtime_inputs": {
            "schedule": binding("runtime/schedule.json", "1"),
            "camera": {
                "gate": binding("runtime/gate.json", "2"),
                "checkpoint": binding("runtime/checkpoint.pt", "3"),
            },
        },
    }
    reservation = {"reviewed_sources": {}}
    monkeypatch.setattr(
        module.contract,
        "current_source_bindings",
        lambda _root: {},
    )
    receipt = module._failure_custody_attestation(
        authorization,
        reservation,
        {
            "phase_a_updates": 0,
            "phase_a_presentations": 0,
            "phase_b_updates": 0,
            "phase_b_presentations": 0,
            "phase_b_entered": False,
        },
    )

    assert receipt["consumed"]["status"] == "TRACKER_NOT_YET_CONSTRUCTED"
    assert receipt["roles_opened"] == []
    assert all(
        row["status"] == "NOT_OPENED"
        for row in receipt["fixed_runtime_input_rehash"].values()
    )
    assert receipt["schedule"]["phase_a_identity"]["presentations"] == 16_000
    assert receipt["schedule"]["phase_b_identity"]["presentations"] == 16_000
    assert receipt["determinism"]["torch_runtime_imported"] is False
    assert receipt["operation_counts"]["cumulative_optimizer_updates"] == 0
    assert set(receipt["forbidden_access_counts"].values()) == {0}
    assert receipt["reviewed_source_rehash"]["passed"] is True


def test_partial_raw_constructor_failure_retains_exact_read_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_runner("_jepa_encoder_v1_partial_raw_constructor")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    relative = "raw/manifest.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"manifest")
    digest = hashlib.sha256(b"manifest").hexdigest()
    matched = SimpleNamespace()

    def read_regular(path, *, expected_sha256=None):
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        return raw

    class RawInputs:
        def __init__(self, _runtime, _authorization):
            self.consumed = {}
            matched._read_regular(path, expected_sha256=digest)
            raise RuntimeError("injected after manifest read")

        def rehash_consumed(self):
            return {
                "unique_file_count": 0,
                "records": [],
                "all_consumed_files_rehashed": True,
            }

    matched._read_regular = read_regular
    matched.RawInputs = RawInputs
    progress: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="injected"):
        module._construct_raw_inputs_with_progress(
            matched,
            object(),
            {},
            progress,
        )
    assert matched._read_regular is read_regular
    assert progress["raw_inputs_constructed"] is not True
    assert progress["_raw_constructor_reads"] == {
        relative: {
            "path": relative,
            "expected_sha256": digest,
            "read_attempt_count": 1,
            "read_success_count": 1,
            "last_observed_file_sha256": digest,
            "last_observed_byte_count": len(b"manifest"),
        },
    }
    leaf = {
        "path": "unused",
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "byte_count": 1,
    }
    receipt = module._failure_custody_attestation(
        {
            "runtime_inputs": {
                "raw": {
                    "manifest": {
                        "path": relative,
                        "file_sha256": digest,
                        "content_sha256": "3" * 64,
                        "byte_count": len(b"manifest"),
                    },
                },
                "schedule": leaf,
                "camera": {"gate": leaf, "checkpoint": leaf},
            },
        },
        {"reviewed_sources": {}},
        progress,
    )
    raw_reads = receipt["raw_constructor_read_rehash"]
    assert raw_reads["record_count"] == 1
    assert raw_reads["all_attempted_reads_rehashed"] is True
    assert raw_reads["records"][0]["role"] == "authority"
    assert raw_reads["records"][0]["rehash_passed"] is True
    assert receipt["roles_opened"] == ["authority"]


def test_reservation_precomputation_failure_does_not_consume_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_runner("_jepa_encoder_v1_reservation_precompute")
    output_root = tmp_path / "attempt"
    monkeypatch.setattr(
        module.contract,
        "science_contract",
        lambda: (_ for _ in ()).throw(RuntimeError("injected science")),
    )
    with pytest.raises(RuntimeError, match="injected science"):
        module._reserve(
            output_root,
            review={"content_sha256": "1" * 64},
            review_raw=b"review",
            authorization={"content_sha256": "2" * 64},
            authorization_raw=b"authorization",
            sources={},
        )
    assert not output_root.exists()


def test_reservation_publication_failure_completes_and_seals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_runner("_jepa_encoder_v1_reservation_publication")
    output_root = tmp_path / "attempt"
    real_publish = module._publish_json

    def fail_reservation_only(path, core):
        if path.name == "reservation.json":
            raise RuntimeError("injected reservation publication")
        return real_publish(path, core)

    monkeypatch.setattr(module, "_publish_json", fail_reservation_only)
    with pytest.raises(RuntimeError, match="injected reservation"):
        module._reserve(
            output_root,
            review={"content_sha256": "1" * 64},
            review_raw=b"review",
            authorization={"content_sha256": "2" * 64},
            authorization_raw=b"authorization",
            sources={},
        )
    assert sorted(path.name for path in output_root.iterdir()) == [
        "completed.json",
        "failure.json",
    ]
    assert (output_root.stat().st_mode & 0o777) == 0o555
    assert all(
        (path.stat().st_mode & 0o777) == 0o444
        for path in output_root.iterdir()
    )


def test_exclusive_write_failure_removes_its_partial_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_runner("_jepa_encoder_v1_partial_write_cleanup")
    path = tmp_path / "partial.json"
    monkeypatch.setattr(
        module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(
            OSError("injected fsync failure")
        ),
    )
    with pytest.raises(OSError, match="injected fsync"):
        module._write_exclusive(path, b"partial")
    assert not path.exists()


def test_terminal_failure_replaces_unpublished_partial_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_runner("_jepa_encoder_v1_partial_completion_repair")
    output_root = tmp_path / "attempt"
    output_root.mkdir()
    reservation, reservation_raw = module._publish_json(
        output_root / "reservation.json",
        {
            "schema": "test_reservation",
            "attempt_identity": "1" * 64,
            "reviewed_sources": {},
        },
    )
    (output_root / "completed.json").write_bytes(b"partial")
    monkeypatch.setattr(
        module,
        "_failure_custody_attestation",
        lambda *_args, **_kwargs: {"status": "TEST"},
    )
    module._terminal_failure(
        output_root,
        reservation,
        reservation_raw,
        authorization={},
        error=RuntimeError("injected completion publication failure"),
        progress={"completion_published": False, "phase_b_entered": False},
    )
    assert (output_root / "failure.json").is_file()
    completed = module.contract.parse_canonical_json(
        (output_root / "completed.json").read_bytes(),
        name="repaired test completion",
    )
    assert completed["status"] == "TERMINAL_FAILURE"
    assert (output_root.stat().st_mode & 0o777) == 0o555


def test_terminal_sealing_repairs_one_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_jepa_encoder_v1_terminal_seal_repair")
    calls = 0

    def flaky_seal(_output_root):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected transient chmod failure")
        return {"all_files_mode": "0444", "all_directories_mode": "0555"}

    monkeypatch.setattr(module, "_seal_terminal", flaky_seal)
    receipt = module._seal_terminal_with_repair(Path("/unused"))
    assert calls == 2
    assert receipt["all_files_mode"] == "0444"


def test_failure_receipt_detects_partial_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_jepa_encoder_v1_partial_torch_import")
    fake_torch = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: False,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
    )
    monkeypatch.setitem(module.sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        module.contract,
        "current_source_bindings",
        lambda _root: {},
    )
    leaf = {
        "path": "unused",
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "byte_count": 1,
    }
    receipt = module._failure_custody_attestation(
        {
            "runtime_inputs": {
                "raw": {},
                "schedule": leaf,
                "camera": {"gate": leaf, "checkpoint": leaf},
            },
        },
        {"reviewed_sources": {}},
        {},
    )
    determinism = receipt["determinism"]
    assert determinism["torch_runtime_imported"] is True
    assert determinism["runtime_object_constructed"] is False


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


def test_phase_a_uses_strict_determinism_and_requires_zero_warnings() -> None:
    module = _load_runner("_jepa_encoder_v1_phase_a_determinism")
    calls: list[tuple[bool, bool]] = []

    class FakeTorch:
        @staticmethod
        def use_deterministic_algorithms(enabled, *, warn_only):
            calls.append((enabled, warn_only))

    def operation():
        return "trained"

    result, receipt = module._run_phase_a_with_reviewed_determinism(
        SimpleNamespace(torch=FakeTorch),
        operation,
    )
    assert result == "trained"
    assert calls == [(True, False), (True, False)]
    assert receipt == {
        "strict_deterministic_algorithms_restored": True,
        "warn_only_scope": "none",
        "expected_determinism_warning_count": 0,
        "warning_messages_sha256":
            module.contract.canonical_json_sha256([]),
        "unexpected_warning_count": 0,
    }

    with pytest.raises(RuntimeError, match="determinism warning"):
        module._run_phase_a_with_reviewed_determinism(
            SimpleNamespace(torch=FakeTorch),
            lambda: warnings.warn("unexpected", UserWarning),
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
