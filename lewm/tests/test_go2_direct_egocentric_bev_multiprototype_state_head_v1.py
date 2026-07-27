from __future__ import annotations

import copy
import importlib.util
import inspect
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = "go2_direct_egocentric_bev_multiprototype_state_head_v1"
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
MODEL = ROOT / "lewm/models/direct_egocentric_bev_multiprototype_state_head_v1.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
PREREGISTRATION = (
    ROOT
    / "docs/lewm_go2_rgb_direct_egocentric_bev_"
    "multiprototype_state_head_v1_preregistration_2026-07-27.json"
)
V10_MODEL = (
    ROOT
    / "lewm/models/"
    "direct_egocentric_bev_state_jepa_v10_final_class_macro_grounding.py"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _walk(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)
    else:
        yield value


def _dict_nodes(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _dict_nodes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _dict_nodes(child)


def _assert_state_equal(left: Any, right: Any) -> None:
    assert tuple(left.state_dict()) == tuple(right.state_dict())
    for name, value in left.state_dict().items():
        assert value.shape == right.state_dict()[name].shape
        assert value.dtype == right.state_dict()[name].dtype
        assert value.device == right.state_dict()[name].device
        assert bool(value.equal(right.state_dict()[name])), name


def _construct_encoder(api: Any) -> Any:
    encoder = api._v10._v8._v6._v3._v1._construct_n320_encoder_without_rng_draw()
    for value in encoder.state_dict().values():
        if value.is_floating_point():
            value.zero_()
    return encoder


def _gate_common(contract: Any, update: int) -> dict[str, Any]:
    return {
        **contract.PERCEPTION_ACCOUNTING[update],
        "multiprototype_mechanism_receipt_ready": True,
        "active_training_scope_multiprototype_v1": "perception_only",
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "all_forbidden_access_counts_zero": True,
    }


def _gate_update_zero(contract: Any) -> dict[str, Any]:
    return {
        **_gate_common(contract, 0),
        **{
            field: True
            for field in contract.INTEGRITY_FIELDS
        },
        "initial_online_to_target_hard_sync_count": 1,
        "correct_rgb_scene_win_count": 8,
        "G": 1.0,
        "aggregate_raster_balanced_accuracy": 0.40,
        "aggregate_free_recall": 0.40,
        "aggregate_occupied_recall": 0.10,
        "aggregate_raster_nll": 0.90,
        "rough_raster_balanced_accuracy": 0.10,
        "rough_raster_occupied_recall": 0.10,
    }


@pytest.fixture
def model_pair() -> tuple[Any, Any, Any]:
    api = _load(MODEL, "_multiprototype_model_fixture")
    torch = api.torch
    encoder = _construct_encoder(api)
    caller_rng = torch.random.get_rng_state().clone()
    active = api.DirectEgocentricBevStateJepaV1(encoder.state_dict())
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    frozen = api._v10.DirectEgocentricBevStateJepaV1(encoder.state_dict())
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return api, active, frozen


@pytest.mark.parametrize("source", [CONTRACT, RUNNER, LAUNCHER, CHECKER])
def test_sources_import_without_tensor_or_image_runtime(source: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(source)!r})
spec = importlib.util.spec_from_file_location('_multiprototype_source_only', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
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


def test_contract_caps_denials_inventories_custody_and_149_source_closure() -> None:
    contract = _load(CONTRACT, "_multiprototype_contract_and_custody")
    checker = _load(CHECKER, "_multiprototype_source_closure")
    assert contract.MAXIMUM_ATTEMPTS == 1
    assert contract.ATTEMPT_INDEX == 1
    assert contract.MAXIMUM_UPDATES == 250
    assert contract.MAXIMUM_PRESENTATIONS == 4_000
    assert contract.GPU_ACTIVE_TIME_CAP_MINUTES == 30
    assert contract.OBSERVATION_UPDATES == (0, 50, 100, 250)
    assert contract.CHECKPOINT_UPDATES == (50, 100, 250)
    assert contract.SCHEDULE_PREFIX_SHA256 == {
        50: "f7e06f741d96af1a3c7796096a38f616f40ee713b6258a217ffd5627afda0788",
        100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
        250: "ee3bc0dcf4c36c8cc66daa2ea8cda6653072fb18c8cf6d6fe1fe3bb50ab1218e",
    }
    assert len(contract.REUSED_SOURCE_PATHS) == 143
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 6
    assert len(contract.SOURCE_PATHS) == 149
    assert set(contract.ADDITIVE_SOURCE_PATHS) == {
        contract.CONTRACT_RELATIVE_PATH,
        contract.MODEL_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        contract.TEST_RELATIVE_PATH,
    }
    assert set(contract.SOURCE_MANIFEST_ENTRYPOINTS) == {
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
    }
    science = contract.science_contract()
    state_head = science["model"]["state_head"]
    assert set(state_head) == {
        "prototype_shape",
        "prototype_parameter_count",
        "cell_and_prototype_l2_epsilon",
        "cell_feature_normalization_axis",
        "prototype_normalization_axis",
        "squared_distance_reduction_axis",
        "logsumexp_reduction_axis",
        "formula",
        "fixed_equal_component_weight",
        "learned_weight_temperature_bias_routing_or_auxiliary_loss",
        "output_classes",
        "output_logit_range_inclusive",
    }
    assert state_head["prototype_shape"] == [3, 4, 64]
    assert state_head["prototype_parameter_count"] == 768
    assert state_head["cell_and_prototype_l2_epsilon"] == 1e-12
    assert state_head["cell_feature_normalization_axis"] == (
        "feature_dimension_64_only"
    )
    assert state_head["prototype_normalization_axis"] == (
        "feature_dimension_64_only_per_row"
    )
    assert state_head["squared_distance_reduction_axis"] == (
        "feature_dimension_64_only"
    )
    assert state_head["logsumexp_reduction_axis"] == (
        "prototype_component_dimension_4_only"
    )
    assert state_head["formula"] == (
        "logsumexp_k(-sum_d((z_hat-p_hat[c,k])**2))-log(4)"
    )
    assert state_head["fixed_equal_component_weight"] == 0.25
    assert state_head[
        "learned_weight_temperature_bias_routing_or_auxiliary_loss"
    ] is False
    assert state_head["output_classes"] == ["UNKNOWN", "FREE", "OCCUPIED"]
    assert state_head["output_logit_range_inclusive"] == [-4.0, 0.0]
    assert science["model"]["bev_decoder"]["prototype_state_head"] == state_head
    assert science["model"]["bev_decoder"][
        "online_decoder_parameter_count"
    ] == 88_384
    assert science["model"]["bev_decoder"][
        "online_decoder_parameter_tensor_count"
    ] == 31
    assert not any(
        node.get("prototype_shape") == [3, 64]
        or node.get("prototype_parameter_shape") == [3, 64]
        for node in _dict_nodes(science["model"])
    )
    initialization = science["model"]["initialization"]
    assert initialization["fresh_v8_draw_order"] == list(
        contract.V8_FRESH_DRAW_ORDER
    )
    assert initialization["prior_v12_runtime_or_parameter_reuse"] is False
    initial_binding = initialization["initial_state_component_binding"]
    assert initial_binding["fresh_prototype_head_local_state_sha256"] == (
        contract.MULTIPROTOTYPE_INITIAL_HEAD_STATE_SHA256
    )
    assert initial_binding["prototype_parameter_shape"] == [3, 4, 64]
    assert initialization["initial_state_component_binding_sha256"] == (
        contract.canonical_json_sha256(initial_binding)
    )
    assert science["scientific_delta"] == {
        "trainable_mechanism_delta_count": 1,
        "mechanism": (
            "equal_weight_four_prototype_per_class_normalized_mixture_"
            "state_head"
        ),
        "old_prototype_shape": [3, 64],
        "new_prototype_shape": [3, 4, 64],
        "online_parameter_delta": 576,
        "decision_contract_is_new_family_specific": True,
        "decision_contract_changes": [
            "update_50_directional_health_gate",
            "update_100_decisive_thresholds_and_strict_v12_rough_comparators",
            "update_250_same_run_rough_occupied_nonregression",
            "new_family_controls_root_source_identity_and_no_successor_rules",
        ],
        "not_v13_timing_successor": True,
    }
    frozen_science = contract._V12.science_contract()
    for frozen_field in ("data", "loader", "objective", "optimizer", "schedule"):
        assert science[frozen_field] == frozen_science[frozen_field]
    inventory_pairs = {
        (
            node.get("parameter_count"),
            node.get("parameter_tensor_count", node.get("tensor_count")),
        )
        for node in _dict_nodes(contract.MODEL_PARAMETER_INVENTORY)
        if "parameter_count" in node
    }
    assert {
        (88_384, 31),
        (2_835_904, 109),
        (5_988_915, 297),
    }.issubset(inventory_pairs)

    authority = contract.EXECUTION_AUTHORITY
    for frozen_family in ("v8", "v10", "v12"):
        assert authority[f"science_identical_to_frozen_{frozen_family}"] is False
    for denied in (
        "predictor_training_or_evaluation_authorized",
        "g2_authorized",
        "navigation_authorized",
        "heldout_authorized",
        "sealed_authorized",
        "production_authorized",
        "promotion_authorized",
        "deployment_authorized",
    ):
        assert authority[denied] is False
    for key, value in authority.items():
        if key.startswith("v12_") and key.endswith("authorized"):
            assert value is False
    runtime_scope = contract.runtime_authorization_template()["experiment_scope"]
    assert runtime_scope["maximum_attempts"] == 1
    assert runtime_scope["maximum_updates"] == 250
    assert runtime_scope["maximum_presentations"] == 4_000
    assert runtime_scope["maximum_active_gpu_minutes"] == 30
    assert runtime_scope["fresh_initialization_required"] is True
    assert runtime_scope["perception_only"] is True
    assert runtime_scope["predictor_forward_or_training"] is False
    assert runtime_scope["prior_runtime_or_checkpoint_reuse"] is False
    assert runtime_scope["output_root_must_be_absent_before_reservation"] is True
    assert runtime_scope["reservation_consumes_the_sole_attempt"] is True
    assert runtime_scope["output_root"] == contract.OUTPUT_ROOT_RELATIVE_PATH

    audit = contract.v12_terminal_audit_binding()
    assert audit["status"] == contract.V12_TERMINAL_AUDIT_STATUS
    assert audit["classification"] == contract.V12_TERMINAL_AUDIT_CLASSIFICATION
    receipt_fields = ("metrics", "artifact", "result", "completion")
    for control in contract.FAILURE_CONTROLS:
        chain = dict.fromkeys(receipt_fields, control)
        assert contract.validate_failure_status_chain(chain) == chain
    with pytest.raises(ValueError, match="one exact"):
        contract.validate_failure_status_chain({
            field: contract.FAILURE_CONTROLS[index % 2]
            for index, field in enumerate(receipt_fields)
        })
    science_text = " ".join(str(item).casefold() for item in _walk(
        contract.science_contract()
    ))
    assert "predictor_training_or_evaluation_authorized true" not in science_text
    assert "heldout_authorized true" not in science_text
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 149
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0


def test_head_exact_formula_shape_range_and_forbidden_capacity() -> None:
    api = _load(MODEL, "_multiprototype_formula")
    torch = api.torch
    head = api.EqualWeightMultiprototypeStateHeadV1()
    assert tuple(head.prototypes.shape) == (3, 4, 64)
    assert head.prototypes.numel() == 768
    assert head.in_channels == 64
    assert head.out_channels == 3
    assert set(dict(head.named_parameters())) == {"prototypes"}
    assert not dict(head.named_buffers())
    assert not tuple(head.children())
    for forbidden in (
        "weights",
        "mixture_weights",
        "temperature",
        "logit_scale",
        "bias",
        "margin",
        "routing",
        "diversity_loss",
        "usage_loss",
        "auxiliary_loss",
    ):
        assert not hasattr(head, forbidden)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    features = torch.randn(2, 64, 64, 64, generator=generator)
    with torch.no_grad():
        head.prototypes.copy_(
            torch.linspace(-0.7, 0.9, 768).reshape(3, 4, 64)
        )
    component_logits = head.component_logits(features)
    observed = head(features)
    normalized_features = torch.nn.functional.normalize(
        features,
        p=2.0,
        dim=1,
        eps=1e-12,
    )
    normalized_prototypes = torch.nn.functional.normalize(
        head.prototypes,
        p=2.0,
        dim=2,
        eps=1e-12,
    )
    expected_components = -(
        normalized_features[:, None, None]
        - normalized_prototypes[None, :, :, :, None, None]
    ).square().sum(dim=3)
    expected = torch.logsumexp(expected_components, dim=2) - math.log(4.0)
    assert component_logits.shape == (2, 3, 4, 64, 64)
    assert observed.shape == (2, 3, 64, 64)
    assert torch.equal(component_logits, expected_components)
    assert torch.equal(observed, expected)
    assert float(observed.min().detach()) >= -4.0
    assert float(observed.max().detach()) <= 0.0


def test_within_class_permutation_invariance_and_duplicate_reduction() -> None:
    api = _load(MODEL, "_multiprototype_symmetries")
    torch = api.torch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(23)
    features = torch.randn(1, 64, 64, 64, generator=generator)
    head = api.EqualWeightMultiprototypeStateHeadV1()
    with torch.no_grad():
        head.prototypes.copy_(torch.randn(3, 4, 64, generator=generator))
    reference = head(features)
    permutations = torch.tensor(
        [[3, 1, 0, 2], [2, 0, 3, 1], [1, 3, 2, 0]],
        dtype=torch.long,
    )
    permuted = api.EqualWeightMultiprototypeStateHeadV1()
    with torch.no_grad():
        for state_class in range(3):
            permuted.prototypes[state_class].copy_(
                head.prototypes[state_class, permutations[state_class]]
            )
    torch.testing.assert_close(
        permuted(features), reference, rtol=2e-7, atol=2e-7
    )

    single = api._v10._v8.NormalizedPrototypeStateHeadV8()
    duplicated = api.EqualWeightMultiprototypeStateHeadV1()
    with torch.no_grad():
        single.prototypes.copy_(torch.randn(3, 64, generator=generator))
        duplicated.prototypes.copy_(single.prototypes[:, None].expand(-1, 4, -1))
    torch.testing.assert_close(
        duplicated(features), single(features), rtol=2e-7, atol=2e-7
    )


def test_all_twelve_rows_receive_gradients_without_parameter_mutation() -> None:
    api = _load(MODEL, "_multiprototype_row_gradients")
    torch = api.torch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    head = api.EqualWeightMultiprototypeStateHeadV1()
    with torch.no_grad():
        head.prototypes.copy_(torch.randn(3, 4, 64, generator=generator))
    before = head.prototypes.detach().clone()
    features = torch.randn(3, 64, 64, 64, generator=generator)
    labels = torch.arange(64 * 64).reshape(64, 64).remainder(3)
    labels = torch.stack((labels, labels.roll(1, 0), labels.roll(1, 1)))
    loss = torch.nn.functional.cross_entropy(head(features), labels)
    loss.backward()
    gradient = head.prototypes.grad
    assert gradient is not None
    assert gradient.shape == (3, 4, 64)
    assert bool(torch.isfinite(gradient).all())
    row_norms = gradient.square().sum(dim=2).sqrt()
    assert bool((row_norms > 0.0).all())
    assert torch.equal(head.prototypes.detach(), before)


def test_rng_frozen_modules_and_exact_parameter_inventories(
    model_pair: tuple[Any, Any, Any],
) -> None:
    api, model, frozen = model_pair
    assert api.DirectEgocentricBevStateJepaV1.training_objective is (
        api._v10.DirectEgocentricBevStateJepaV1.training_objective
    )
    assert api.DirectEgocentricBevStateJepaV1.wrong_rgb_grounding_control is (
        api._v10.DirectEgocentricBevStateJepaV1.wrong_rgb_grounding_control
    )
    _assert_state_equal(model.encoder, frozen.encoder)
    _assert_state_equal(model.bev_decoder, frozen.bev_decoder)
    _assert_state_equal(model.predictor, frozen.predictor)
    assert [name for name, _ in model.encoder.named_parameters()] == [
        name for name, _ in frozen.encoder.named_parameters()
    ]
    assert [name for name, _ in model.bev_decoder.named_parameters()] == [
        name for name, _ in frozen.bev_decoder.named_parameters()
    ]
    assert [name for name, _ in model.predictor.named_parameters()] == [
        name for name, _ in frozen.predictor.named_parameters()
    ]

    decoder_head = (*model.bev_decoder.parameters(), *model.state_head.parameters())
    target_stack = tuple(
        parameter
        for module in model._target_modules()
        for parameter in module.parameters()
    )
    assert (sum(value.numel() for value in decoder_head), len(decoder_head)) == (
        88_384,
        31,
    )
    assert (
        sum(value.numel() for value in model.encoder.parameters()),
        len(tuple(model.encoder.parameters())),
    ) == (2_747_520, 78)
    assert (
        sum(value.numel() for value in model.predictor.parameters()),
        len(tuple(model.predictor.parameters())),
    ) == (317_107, 79)
    online_stack = tuple(
        parameter
        for module in model._online_modules()
        for parameter in module.parameters()
    )
    assert (sum(value.numel() for value in online_stack), len(online_stack)) == (
        2_835_904,
        109,
    )
    assert (sum(value.numel() for value in target_stack), len(target_stack)) == (
        2_835_904,
        109,
    )
    parameters = tuple(model.parameters())
    assert (sum(value.numel() for value in parameters), len(parameters)) == (
        5_988_915,
        297,
    )
    assert tuple(model.state_head.prototypes.shape) == (
        api.PROTOTYPE_PARAMETER_SHAPE_MULTIPROTOTYPE_V1
    )
    rows = model.state_head.prototypes.detach().reshape(12, 64)
    assert all(
        not bool(rows[left].equal(rows[right]))
        for left in range(12)
        for right in range(left + 1, 12)
    )


def test_initial_target_sync_and_exact_ema_include_multiprototype_head(
    model_pair: tuple[Any, Any, Any],
) -> None:
    api, model, _frozen = model_pair
    torch = api.torch
    for online, target in zip(
        model._online_modules(), model._target_modules(), strict=True
    ):
        _assert_state_equal(online, target)
    assert all(
        not parameter.requires_grad
        for module in model._target_modules()
        for parameter in module.parameters()
    )
    assert int(model.ema_update_count) == 0

    model.arm_phase_schedule_v6()
    target_before = model.target_state_head.prototypes.detach().clone()
    with torch.no_grad():
        model.state_head.prototypes.add_(0.125)
    online_before = model.state_head.prototypes.detach().clone()
    model.update_target_ema_after_optimizer_step()
    expected = target_before * 0.996 + online_before * 0.004
    assert torch.equal(model.state_head.prototypes.detach(), online_before)
    torch.testing.assert_close(
        model.target_state_head.prototypes,
        expected,
        rtol=0.0,
        atol=torch.finfo(expected.dtype).eps,
    )
    assert int(model.ema_update_count) == 1
    assert model.phase_counters_v6()["target_update_callback_count"] == 1
    assert model.phase_counters_v6()["ema_arithmetic_update_count"] == 1
    assert all(
        not parameter.requires_grad
        for module in model._target_modules()
        for parameter in module.parameters()
    )


def test_preliminary_dispatch_truth_table_and_update_zero_integrity() -> None:
    contract = _load(CONTRACT, "_multiprototype_preliminary_and_zero")
    for update in contract.OBSERVATION_UPDATES:
        preliminary = contract.evaluate_gate(update, {})
        assert preliminary == contract._V12.evaluate_gate(update, {})
        assert preliminary["passed"] is True
        assert preliminary["scientific_gate_evidence"] is False
        assert preliminary["execution_training_checkpoint_terminal_pass_or_"
                           "downstream_authority"] is False

    marker = "multiprototype_mechanism_receipt_ready"
    scope = "active_training_scope_multiprototype_v1"
    for partial in (
        {marker: True},
        {scope: "perception_only"},
        {marker: False},
        {scope: "wrong"},
    ):
        with pytest.raises(ValueError, match="marker"):
            contract.evaluate_gate(0, partial)

    zero = _gate_update_zero(contract)
    result = contract.evaluate_gate(0, zero)
    assert result["passed"] is True
    assert result["control"] == contract.GATE_CONTROLS[0][1]
    for field in contract.INTEGRITY_FIELDS:
        failed = {**zero, field: False}
        assert contract.evaluate_gate(0, failed)["passed"] is False, field
    for mutation in (
        {"multiprototype_mechanism_receipt_ready": False},
        {"active_training_scope_multiprototype_v1": "wrong"},
        {"initial_online_to_target_hard_sync_count": 0},
        {"correct_rgb_scene_win_count": 7},
        {"all_forbidden_access_counts_zero": False},
        {"online_perception_optimizer_update_count": 1},
    ):
        assert contract.evaluate_gate(0, {**zero, **mutation})["passed"] is False


def test_exact_update_50_100_and_250_boundaries() -> None:
    contract = _load(CONTRACT, "_multiprototype_gate_boundaries")
    zero = _gate_update_zero(contract)
    fifty = {
        **_gate_common(contract, 50),
        "G": 0.90,
        "aggregate_raster_balanced_accuracy": 0.50,
        "aggregate_free_recall": 0.25,
        "aggregate_occupied_recall": 0.85,
        "aggregate_raster_nll": 0.80,
        "rough_raster_balanced_accuracy": 0.20,
        "rough_raster_occupied_recall": 0.20,
        "correct_rgb_scene_win_count": 8,
    }
    assert contract.evaluate_gate(
        50, fifty, update_zero=zero
    )["control"] == contract.GATE_CONTROLS[50][1]
    fifty_failures = (
        {"G": zero["G"]},
        {
            "aggregate_raster_balanced_accuracy": zero[
                "aggregate_raster_balanced_accuracy"
            ]
        },
        {"aggregate_free_recall": 0.249999},
        {"aggregate_occupied_recall": zero["aggregate_occupied_recall"]},
        {
            "aggregate_free_recall": 0.25,
            "aggregate_occupied_recall": 0.850001,
        },
        {"aggregate_raster_nll": zero["aggregate_raster_nll"]},
        {
            "rough_raster_balanced_accuracy": zero[
                "rough_raster_balanced_accuracy"
            ]
        },
        {
            "rough_raster_occupied_recall": zero[
                "rough_raster_occupied_recall"
            ]
        },
        {"correct_rgb_scene_win_count": 7},
        {"all_forbidden_access_counts_zero": False},
    )
    for mutation in fifty_failures:
        assert contract.evaluate_gate(
            50, {**fifty, **mutation}, update_zero=zero
        )["passed"] is False

    hundred = {
        **_gate_common(contract, 100),
        "G": 0.80,
        "aggregate_raster_balanced_accuracy": 0.72,
        "aggregate_free_recall": 0.68,
        "aggregate_occupied_recall": 0.88,
        "aggregate_raster_nll": 0.46,
        "rough_raster_balanced_accuracy": 0.733,
        "rough_raster_occupied_recall": 0.573,
        "correct_rgb_scene_win_count": 8,
    }
    assert contract.evaluate_gate(
        100, hundred, update_zero=zero
    )["control"] == contract.GATE_CONTROLS[100][1]
    hundred_failures = (
        {"G": zero["G"]},
        {"aggregate_raster_balanced_accuracy": 0.719999},
        {"aggregate_free_recall": 0.679999},
        {"aggregate_occupied_recall": 0.799999},
        {
            "aggregate_free_recall": 0.68,
            "aggregate_occupied_recall": 0.880001,
        },
        {"aggregate_raster_nll": 0.460001},
        {"rough_raster_balanced_accuracy": 0.732972219013282},
        {"rough_raster_occupied_recall": 0.5722940226171244},
        {"correct_rgb_scene_win_count": 7},
        {"all_forbidden_access_counts_zero": False},
    )
    for mutation in hundred_failures:
        assert contract.evaluate_gate(
            100, {**hundred, **mutation}, update_zero=zero
        )["passed"] is False

    terminal = {
        **_gate_common(contract, 250),
        "G": 0.70,
        "aggregate_raster_balanced_accuracy": 0.80,
        "aggregate_free_recall": 0.68,
        "aggregate_occupied_recall": 0.88,
        "aggregate_raster_nll": 0.42,
        "rough_raster_balanced_accuracy": 0.7719525,
        # Equality is deliberately a pass: this is same-run u100
        # nonregression, not the strict frozen-V12 comparator used at u100.
        "rough_raster_occupied_recall": hundred[
            "rough_raster_occupied_recall"
        ],
        "correct_rgb_scene_win_count": 8,
    }
    assert contract.evaluate_gate(
        250,
        terminal,
        update_zero=zero,
        update_100=hundred,
    )["control"] == contract.GATE_CONTROLS[250][1]
    terminal_failures = (
        {"aggregate_raster_balanced_accuracy": 0.799999},
        {"aggregate_free_recall": 0.679999},
        {"aggregate_occupied_recall": 0.879999},
        {
            "aggregate_free_recall": 0.68,
            "aggregate_occupied_recall": 0.930001,
        },
        {"aggregate_raster_nll": 0.420001},
        {"rough_raster_balanced_accuracy": 0.7719524},
        {
            "rough_raster_occupied_recall": (
                hundred["rough_raster_occupied_recall"] - 0.000001
            )
        },
        {"correct_rgb_scene_win_count": 7},
        {"all_forbidden_access_counts_zero": False},
    )
    for mutation in terminal_failures:
        assert contract.evaluate_gate(
            250,
            {**terminal, **mutation},
            update_zero=zero,
            update_100=hundred,
        )["passed"] is False
    relative_nll_failure = {
        **terminal,
        "aggregate_raster_nll": 0.410001,
    }
    assert contract.evaluate_gate(
        250,
        relative_nll_failure,
        update_zero=zero,
        update_100={**hundred, "aggregate_raster_nll": 0.40},
    )["passed"] is False
    assert contract.evaluate_gate(
        250,
        terminal,
        update_zero=zero,
        update_100=hundred,
        prior_gates_passed=False,
    )["passed"] is False
    no_u250_g_threshold = {**terminal, "G": zero["G"]}
    assert contract.evaluate_gate(
        250,
        no_u250_g_threshold,
        update_zero=zero,
        update_100=hundred,
    )["passed"] is True


def test_runner_has_exactly_two_custom_seams_and_preserves_seven() -> None:
    runner = _load(RUNNER, "_multiprototype_runner_seam_topology")
    launcher = _load(LAUNCHER, "_multiprototype_launcher_topology")
    runner._assert_multiprototype_bindings()
    launcher._assert_multiprototype_bindings()

    custom = dict(runner._MULTIPROTOTYPE_SEAM_TABLE)
    inherited_v9 = dict(runner._V9._V9_SEAM_TABLE)
    assert tuple(custom) == (
        "_gradient_integrity_probe",
        "_evaluate_observation_impl",
    )
    preserved: list[str] = []
    for name, expected_v8 in runner._V8._V8_SEAM_TABLE:
        expected = custom.get(name, inherited_v9.get(name, expected_v8))
        assert getattr(runner._LEAF, name) is expected
        if name not in custom:
            preserved.append(name)
    assert preserved == [
        "_initialize_model",
        "_build_optimizer",
        "_load_schedule",
        "_train_probe",
        "_write_training_trace",
        "_snapshot_model",
        "_terminal_failure",
    ]
    assert runner._LEAF._snapshot_model is runner._V9._v9_snapshot_model
    assert runner._LEAF._terminal_failure is runner._V9._v9_terminal_failure
    assert runner._LEAF._initialize_model is runner._V8._v8_initialize_model
    assert runner._LEAF._build_optimizer is runner._V8._v8_build_optimizer
    assert all(
        name == runner.MULTIPROTOTYPE_MODEL_RUNTIME_MODULE_NAME
        for name in runner._runtime_module_names()
    )

    # The V9-captured V8 observer remains an identity witness, but the active
    # leaf reaches the deeper frozen observer directly through the new seam.
    assert runner._V9._FROZEN_V8_EVALUATE_OBSERVATION_IMPL is (
        runner._V8._v8_evaluate_observation_impl
    )
    assert runner._LEAF._evaluate_observation_impl is (
        runner._multiprototype_evaluate_observation_impl
    )
    assert runner._LEAF._evaluate_observation_impl is not (
        runner._V9._v9_evaluate_observation_impl
    )
    core_source = inspect.getsource(runner._multiprototype_observation_core)
    assert "_V6._FROZEN_EVALUATE_OBSERVATION_IMPL(" in core_source
    assert "_FROZEN_V8_EVALUATE_OBSERVATION_IMPL(" not in core_source
    assert "_v8_evaluate_observation_impl(" not in core_source


def test_runner_gradient_hook_receipt_covers_rows_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER, "_multiprototype_runner_gradient_hook")
    torch = pytest.importorskip("torch")
    runtime = SimpleNamespace(torch=torch)
    state_head = torch.nn.Module()
    state_head.register_parameter(
        "prototypes",
        torch.nn.Parameter(torch.linspace(-1.0, 1.0, 768).reshape(3, 4, 64)),
    )
    model = SimpleNamespace(state_head=state_head)
    parameter = state_head.prototypes
    parameter.grad = torch.full_like(parameter, 7.0)
    parameter_before = parameter.detach().clone()
    gradient_before = parameter.grad.detach().clone()
    coefficients = torch.arange(1, 13, dtype=parameter.dtype).reshape(3, 4, 1)

    def frozen_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        previous = parameter.grad.detach().clone()
        parameter.grad = None
        try:
            (parameter * coefficients).sum().backward()
        finally:
            parameter.grad = previous
        return {"frozen_v8_receipt": True}

    monkeypatch.setattr(
        runner,
        "_FROZEN_V8_GRADIENT_INTEGRITY_PROBE",
        frozen_probe,
    )
    receipt = runner._multiprototype_gradient_integrity_probe(
        runtime,
        model,
        {},
        {},
    )
    rows = receipt["multiprototype_prototype_row_gradients"]
    assert rows["shape"] == [3, 4, 64]
    assert rows["capture_count"] == 1
    assert rows["all_twelve_rows_finite_nonzero"] is True
    assert receipt["all_twelve_prototype_row_gradients_finite_nonzero"] is True
    assert [
        row["component"]
        for class_name in ("UNKNOWN", "FREE", "OCCUPIED")
        for row in rows["classes"][class_name]
    ] == [0, 1, 2, 3] * 3
    assert all(
        row["finite_nonzero"] is True
        for class_name in ("UNKNOWN", "FREE", "OCCUPIED")
        for row in rows["classes"][class_name]
    )
    assert receipt["frozen_v8_receipt"] is True
    assert runner._GRADIENT_PROBE_ACTIVE is False
    assert torch.equal(parameter.detach(), parameter_before)
    assert torch.equal(parameter.grad, gradient_before)

    one_bad_row = coefficients.expand(3, 4, 64).clone()
    one_bad_row[2, 3].zero_()
    failed = runner._prototype_row_gradient_receipt(runtime, one_bad_row)
    assert failed["all_twelve_rows_finite_nonzero"] is False
    assert failed["classes"]["OCCUPIED"][3]["finite_nonzero"] is False


def test_utilization_receipt_exact_ties_zero_class_and_descriptive_only() -> None:
    runner = _load(RUNNER, "_multiprototype_utilization_math")
    torch = pytest.importorskip("torch")
    runtime = SimpleNamespace(torch=torch)
    accumulator = runner._MultiprototypeUtilizationAccumulator(runtime)
    component_logits = torch.zeros(1, 3, 4, 64, 64).expand(
        495, -1, -1, -1, -1
    )
    labels = torch.zeros(495, 64, 64, dtype=torch.long)
    labels[:, :, 32:] = 1
    accumulator.add(component_logits, labels)
    receipt = accumulator.receipt()

    assert receipt["descriptive_only"] is True
    assert receipt["population"] == {
        "role": "checkpoint_selection",
        "side": "current",
        "row_count": 495,
    }
    expected_count = 495 * 64 * 32
    for class_name in ("UNKNOWN", "FREE"):
        observed = receipt["classes"][class_name]
        assert observed["target_class_valid_cell_count"] == expected_count
        assert observed["per_component_posterior_responsibility_mean"] == (
            [0.25] * 4
        )
        assert observed["per_component_winner_share"] == [1.0, 0.0, 0.0, 0.0]
        assert observed["mean_responsibility_entropy_nats"] == pytest.approx(
            math.log(4.0), rel=1e-12, abs=1e-12
        )
        assert observed["effective_component_count"] == pytest.approx(
            4.0, rel=1e-12, abs=1e-12
        )
    assert receipt["classes"]["OCCUPIED"] == {
        "target_class_valid_cell_count": 0,
        "per_component_posterior_responsibility_mean": None,
        "per_component_winner_share": None,
        "mean_responsibility_entropy_nats": None,
        "effective_component_count": None,
    }
    assert runner.contract.MULTIPROTOTYPE_UTILIZATION["gating"] is False

    zero = _gate_update_zero(runner.contract)
    baseline = runner.contract.evaluate_gate(0, zero)
    with_diagnostic = runner.contract.evaluate_gate(
        0,
        {**zero, "multiprototype_utilization": receipt},
    )
    assert with_diagnostic == baseline


def test_completed_observation_uses_v9_persistence_receipts() -> None:
    runner = _load(RUNNER, "_multiprototype_v9_observation_persistence")
    receipts = runner._V9._V9_COMPLETED_OBSERVATION_RECEIPTS
    witnesses = runner._V9._V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES
    receipts_before = copy.deepcopy(receipts)
    witnesses_before = copy.deepcopy(witnesses)
    receipts.clear()
    witnesses.clear()
    loader = SimpleNamespace(progress={})
    result = {
        "update": 0,
        "gate": {"passed": True, "control": "synthetic_continue"},
        "metrics": {
            "score": 0.5,
            "multiprototype_utilization": {"descriptive_only": True},
        },
    }
    try:
        returned = runner._capture_completed_observation(
            SimpleNamespace(),
            loader,
            result,
            update=0,
        )
        assert returned is result
        assert len(receipts) == 1
        assert receipts[0]["update"] == 0
        assert receipts[0]["gate"] == result["gate"]
        assert receipts[0]["metrics"] == result["metrics"]
        assert runner.contract.is_sha256(receipts[0]["canonical_sha256"])
        assert loader.progress["completed_observation_receipts"] == receipts
        assert loader.progress["completed_observation_receipt_bindings"] == (
            runner._V9._observation_bindings(receipts)
        )
        assert loader.progress[
            "completed_observation_determinism_witnesses"
        ] == witnesses
        persisted = runner._V9._progress_observation_evidence(loader.progress)
        assert persisted == (
            loader.progress["completed_observation_receipts"],
            loader.progress["completed_observation_receipt_bindings"],
            loader.progress["completed_observation_determinism_witnesses"],
        )
        result["metrics"]["score"] = 99.0
        assert receipts[0]["metrics"]["score"] == 0.5
    finally:
        receipts[:] = receipts_before
        witnesses[:] = witnesses_before
