from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = "go2_direct_egocentric_bev_state_jepa_v8_learned_bev_query_prototype_decoder"
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
MODEL = ROOT / "lewm/models/direct_egocentric_bev_state_jepa_v8_learned_bev_query_prototype_decoder.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _accounting(contract: Any, update: int) -> dict[str, Any]:
    return {
        **contract.PHASE_ACCOUNTING[update],
        "v8_mechanism_receipt_ready": True,
        "active_phase_v6": "phase_one",
        "all_registered_values_finite": True,
        "state_nonconstant": True,
    }


def test_contract_freezes_exact_science_caps_schedule_and_governance() -> None:
    contract = _load(CONTRACT, "_direct_bev_v8_contract_test")
    assert contract.MAXIMUM_UPDATES == 250
    assert contract.MAXIMUM_PRESENTATIONS == 4_000
    assert contract.GPU_ACTIVE_TIME_CAP_MINUTES == 30
    assert contract.EXECUTION_AUTHORITY["maximum_updates"] == 250
    assert contract.EXECUTION_AUTHORITY["maximum_presentations"] == 4_000
    assert contract.EXECUTION_AUTHORITY["gpu_active_minutes_maximum"] == 30
    assert contract.EXECUTION_AUTHORITY[
        "science_identical_v7_runner_integrity_replacement_only"
    ] is False
    assert contract.EXECUTION_AUTHORITY[
        "coordinate_aware_film_unet_predictor_only"
    ] is False
    assert contract.EXECUTION_AUTHORITY[
        "phase_separated_frozen_state_prediction_only"
    ] is False
    assert contract.EXECUTION_AUTHORITY[
        "learned_bev_query_prototype_perception_only"
    ] is True
    assert contract.OBSERVATION_UPDATES == (0, 50, 100, 250)
    assert contract.CHECKPOINT_UPDATES == (50, 100, 250)
    assert contract.SCHEDULE_PREFIX_SHA256 == {
        50: "f7e06f741d96af1a3c7796096a38f616f40ee713b6258a217ffd5627afda0788",
        100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
        250: "ee3bc0dcf4c36c8cc66daa2ea8cda6653072fb18c8cf6d6fe1fe3bb50ab1218e",
    }
    assert contract.MODEL_PARAMETER_INVENTORY["decoder_state"] == {
        "parameter_count": 87_808,
        "tensor_count": 31,
        "ordered_parameter_name_sha256": (
            "3d59a484e25593e47bc5eb740618814bf71a15f5e1e41e302598b774c49a0dc8"
        ),
    }
    assert contract.MODEL_PARAMETER_INVENTORY["total"] == {
        "parameter_count": 5_987_763,
        "tensor_count": 297,
    }
    assert contract.build_schedule_identity()["presentations"] == 4_000
    assert contract.runtime_authorization_template()["schedule"] == (
        contract.build_schedule_identity()
    )
    science = contract.science_contract()
    assert science["scientific_delta"]["scientific_delta_count"] == 1
    assert science["model"]["state_head"]["separate_registered_module"] is True
    assert science["model"]["target"]["inventory"] == [
        "encoder",
        "bev_decoder",
        "state_head",
    ]
    assert science["lifecycle"]["predictor_phase_or_update"] is False
    assert science["lifecycle"]["gpu_active_minutes_maximum"] == 30
    assert "maximum_active_gpu_minutes" not in science["lifecycle"]
    assert science["phase_adapter"] == {
        "scope": "v8_learned_bev_query_prototype_perception_only",
        "updates": [1, 250],
        "presentations": [1, 4_000],
        "total": "G/log(2)",
        "trainable": "online_encoder_decoder_state",
        "frozen": "predictor_and_detached_target",
        "target_callback": "ema_0point996_once_after_every_update",
        "initial_online_to_target_hard_sync_count": 1,
        "optimizer": "one_v8_adamw_constructed_once_never_reset",
        "predictor_forward_objective_backward_or_update_count": 0,
        "second_phase_present": False,
    }
    assert "phase_successor" not in science
    assert "predictor_successor" not in science
    assert "frozen_v2_integrity_provenance" not in science
    active_science = dict(science)
    active_science.pop("frozen_v7_integrity_provenance")

    def walk(value: Any):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)
        else:
            yield value

    stale_scalars = {400, 401, 1_000, 6_400, 6_401, 16_000, 60}
    stale_text = ("ema_update_400", "j/log(2)+c", "phase_two_total")
    for scalar in walk(active_science):
        assert not (type(scalar) is int and scalar in stale_scalars)
        if isinstance(scalar, str):
            assert not any(token in scalar.casefold() for token in stale_text)
    governing = contract.validate_governing_documents()
    assert governing[contract.PREREGISTRATION_RELATIVE_PATH] == (
        contract.PREREGISTRATION_FILE_SHA256
    )
    assert governing[contract.V7_TERMINAL_AUDIT_RELATIVE_PATH] == (
        contract.V7_TERMINAL_AUDIT_FILE_SHA256
    )


def test_exact_v8_gate_boundaries_and_preliminary_mode() -> None:
    contract = _load(CONTRACT, "_direct_bev_v8_gate_test")
    preliminary = contract.evaluate_gate(50, {})
    assert preliminary["passed"] is True
    assert preliminary["gate_mode"].startswith("PRELIMINARY_")

    zero = {
        **_accounting(contract, 0),
        **{
            field: True
            for field in (
                "fresh_v8_model_and_optimizer_zero_prior_runtime_reuse",
                "n320_encoder_only_migration_exact",
                "registered_seed_draw_order_exact",
                "initial_model_state_matches_frozen_v8",
                "model_parameter_inventory_exact",
                "v8_decoder_parameter_inventory_exact",
                "learned_only_forbidden_geometry_absent",
                "two_residual_cross_attention_ffn_blocks_exact",
                "negative_squared_prototype_distance_formula_exact",
                "online_target_perception_bitwise_equal",
                "three_channel_state_exact",
                "all_logits_in_closed_interval_minus4_to0",
                "v8_intended_gradient_coverage_exact",
                "predictor_target_and_fixed_negative_gradients_absent",
                "no_hidden_auxiliary_bypass",
                "all_forbidden_access_counts_zero",
            )
        },
        "initial_online_to_target_hard_sync_count": 1,
        "G": 0.8,
        "aggregate_raster_balanced_accuracy": 0.3,
        "rough_raster_balanced_accuracy": 0.2,
        "rough_raster_occupied_recall": 0.1,
    }
    assert contract.evaluate_gate(0, zero)["passed"] is True
    bad_zero = dict(zero)
    bad_zero["all_logits_in_closed_interval_minus4_to0"] = False
    assert contract.evaluate_gate(0, bad_zero)["passed"] is False

    fifty = {
        **_accounting(contract, 50),
        "G": 0.7,
        "aggregate_raster_balanced_accuracy": 0.31,
        "aggregate_free_recall": 0.1,
        "aggregate_occupied_recall": 0.2,
        "rough_raster_balanced_accuracy": 0.1,
        "rough_raster_occupied_recall": 0.1,
        "correct_rgb_scene_win_count": 6,
    }
    assert contract.evaluate_gate(50, fifty, update_zero=zero)["passed"] is True

    hundred = {
        **_accounting(contract, 100),
        "G": 0.6,
        "aggregate_raster_balanced_accuracy": 0.70,
        "aggregate_free_recall": 0.50,
        "aggregate_occupied_recall": 0.80,
        "aggregate_raster_nll": 0.46,
        "rough_raster_balanced_accuracy": 0.21,
        "rough_raster_occupied_recall": 0.11,
        "correct_rgb_scene_win_count": 8,
    }
    assert contract.evaluate_gate(100, hundred, update_zero=zero)["passed"] is True

    terminal = {
        **_accounting(contract, 250),
        "G": 0.5,
        "aggregate_raster_balanced_accuracy": 0.80,
        "aggregate_free_recall": 0.68,
        "aggregate_occupied_recall": 0.88,
        "aggregate_raster_nll": 0.42,
        "rough_raster_balanced_accuracy": 0.7719525,
        "rough_raster_occupied_recall": 0.4319467,
        "correct_rgb_scene_win_count": 8,
    }
    assert contract.evaluate_gate(
        250,
        terminal,
        update_zero=zero,
        update_100={"aggregate_raster_nll": 0.41},
    )["control"] == contract.CONTROL_PASS
    bad_terminal = dict(terminal)
    bad_terminal["aggregate_free_recall"] = 0.679999
    assert contract.evaluate_gate(
        250,
        bad_terminal,
        update_zero=zero,
        update_100={"aggregate_raster_nll": 0.41},
    )["passed"] is False


def test_model_exact_learned_only_inventory_formula_and_predictor_exclusion() -> None:
    model_api = _load(MODEL, "_direct_bev_v8_model_test")
    torch = model_api.torch
    encoder = model_api._v6._v3._v1._construct_n320_encoder_without_rng_draw()
    for index, value in enumerate(encoder.state_dict().values()):
        if value.is_floating_point():
            value.copy_(
                torch.linspace(-0.01, 0.01, value.numel()).reshape_as(value)
            )
    caller_rng = torch.random.get_rng_state().clone()
    model = model_api.DirectEgocentricBevStateJepaV1(encoder.state_dict())
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    decoder_and_head = (*model.bev_decoder.parameters(), *model.state_head.parameters())
    assert sum(value.numel() for value in decoder_and_head) == 87_808
    assert len(decoder_and_head) == 31
    assert not tuple(model.bev_decoder.named_buffers())
    assert not any(
        isinstance(module, torch.nn.Conv2d)
        for module in model.bev_decoder.modules()
    )
    assert set(dict(model.state_head.named_parameters())) == {"prototypes"}
    assert model.state_head.out_channels == 3

    features = torch.randn(1, 64, 64, 64)
    logits = model.state_head(features)
    normalized_features = torch.nn.functional.normalize(
        features, dim=1, eps=1e-12
    )
    normalized_prototypes = torch.nn.functional.normalize(
        model.state_head.prototypes, dim=1, eps=1e-12
    )
    expected = -(
        normalized_features[:, None]
        - normalized_prototypes[None, :, :, None, None]
    ).square().sum(dim=2)
    assert torch.equal(logits, expected)
    assert float(logits.min().detach()) >= -4.000001
    assert float(logits.max().detach()) <= 0.000001

    model.arm_phase_schedule_v6()
    predictor_calls = {"count": 0}
    handles = [
        module.register_forward_pre_hook(
            lambda _module, _inputs: predictor_calls.__setitem__(
                "count", predictor_calls["count"] + 1
            )
        )
        for module in model.predictor.modules()
        if not tuple(module.children())
    ]
    torch.manual_seed(5)
    rgb = torch.rand(1, 3, 112, 112)
    actions = torch.zeros(1, 9)
    actions[:, 3] = 1.0
    labels = torch.randint(0, 3, (1, 64, 64))
    result = model.training_objective(
        current_rgb=rgb,
        next_rgb=rgb.roll(1, 3),
        fixed_negative_rgb=rgb.flip(3),
        action_one_hot=actions,
        non_hold_mask=torch.tensor([True]),
        current_labels=labels,
        next_labels=labels.roll(1, 2),
    )
    result.total.backward()
    for handle in handles:
        handle.remove()
    assert predictor_calls["count"] == 0
    assert result.total.item() == pytest.approx(
        (result.G / model_api.math.log(2.0)).item()
    )
    assert torch.equal(
        result.all_action_prediction_logits,
        result.current_state_logits[:, None].expand_as(
            result.all_action_prediction_logits
        ),
    )
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model._target_modules()[0].parameters())


def test_runner_import_is_source_only_and_preserves_all_v8_seams() -> None:
    program = f"""
import importlib.util, sys
from pathlib import Path
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('_v8_runner_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
module._assert_v8_seams()
args = module.parse_args(['--run', '--review-sha256', '0'*64, '--authorization-sha256', '1'*64])
assert args.review_sha256 == '0'*64
assert args.authorization_sha256 == '1'*64
module._assert_v8_seams()
print('PASS')
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


def test_runner_initializer_optimizer_and_schedule_prefix_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v8_runner_test")
    model_api = _load(MODEL, runner.V8_MODEL_RUNTIME_MODULE_NAME)
    shared = _load(
        ROOT / "lewm/models/shared_observable_camera_ray_jepa_v5.py",
        "_direct_bev_v8_shared_hash_test",
    )
    torch = model_api.torch
    encoder = model_api._v6._v3._v1._construct_n320_encoder_without_rng_draw()
    for value in encoder.state_dict().values():
        value.zero_()
    runtime = SimpleNamespace(torch=torch, model_module=shared)
    model, partition, receipt = runner._v8_initialize_model(
        runtime,
        model_api,
        SimpleNamespace(encoder=encoder),
        torch.device("cpu"),
    )
    optimizer, optimizer_receipt = runner._v8_build_optimizer(runtime, partition)
    assert receipt["v8_initial_decoder_state_sha256"] == (
        runner.contract.V8_INITIAL_DECODER_STATE_SHA256
    )
    assert optimizer_receipt["predictor_parameters_excluded"] is True
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert not optimizer_ids.intersection(
        id(parameter) for parameter in model.predictor.parameters()
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260713)
    full: list[int] = []
    while len(full) < 16_000:
        full.extend(torch.randperm(4_262, generator=generator).tolist())
    full = full[:16_000]
    binding = runner.contract.build_schedule_identity()["source"]

    class Adapter:
        @staticmethod
        def validate_bound_schedule_phase_a(*, raw, binding):
            assert raw == b"synthetic-bound-schedule"
            return object()

        @staticmethod
        def finalize_train_identity(*, state, ordered_train_pair_ids):
            assert len(ordered_train_pair_ids) == 4_262
            return list(full), dict(binding), {"synthetic": True}

    monkeypatch.setattr(
        runner._LEAF,
        "_read_bound",
        lambda path, observed: b"synthetic-bound-schedule",
    )
    progress: dict[str, Any] = {}
    used, schedule_receipt = runner._v8_load_schedule(
        Adapter,
        {"runtime_inputs": {"schedule": runner.contract.build_schedule_identity()}},
        [{"content_sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(4_262)],
        progress=progress,
    )
    assert used == full[:4_000]
    assert schedule_receipt["source_adapter_returned_presentations"] == 16_000
    assert schedule_receipt["used_presentation_count"] == 4_000
    assert progress["schedule_validated"] is True


def test_launcher_and_source_closure_use_only_six_additive_files() -> None:
    launcher = _load(LAUNCHER, "_direct_bev_v8_launcher_test")
    assert launcher.contract.RUNNER_RELATIVE_PATH == RUNNER.relative_to(ROOT).as_posix()
    assert launcher._LEAF._V11._BASE.RUNNER_PATH == RUNNER
    checker = _load(CHECKER, "_direct_bev_v8_checker_test")
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 122
    assert len(checker.contract.ADDITIVE_SOURCE_PATHS) == 6
    assert checker.contract.MODEL_RELATIVE_PATH in checker.contract.ADDITIVE_SOURCE_PATHS
    raw = checker.contract.canonical_json_bytes(manifest) + b"\n"
    assert checker.contract.validate_source_manifest(raw) == manifest
