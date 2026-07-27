from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = "go2_direct_egocentric_bev_state_jepa_v10_final_class_macro_grounding"
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
MODEL = ROOT / "lewm/models/direct_egocentric_bev_state_jepa_v10_final_class_macro_grounding.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V10_"
    "FINAL_CLASS_MACRO_GROUNDING_PREFLIGHT_JSON"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bomb(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("an intermediate public entrypoint was called")


@pytest.mark.parametrize("source", [CONTRACT, RUNNER, LAUNCHER, CHECKER])
def test_contract_and_wrappers_import_source_only_without_tensor_runtime(
    source: Path,
) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(source)!r})
spec = importlib.util.spec_from_file_location('_v10_source_only', path)
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


def test_contract_binds_v9_and_freezes_every_nonobjective_science_leaf() -> None:
    contract = _load(CONTRACT, "_direct_bev_v10_contract_identity")
    assert contract.preregistration_binding() == {
        "path": contract.PREREGISTRATION_RELATIVE_PATH,
        "commit": "7c1886942a298964b083457f335329259e594593",
        "file_sha256": "3e901a8847c21d44d6d1f1a41e7a71deb8da221043dd6ed461926ced9e2fe4a6",
        "content_sha256": "341239f40885c5a55042cd88d49b4e6401a332fa066f3b7bb0d7062880105b83",
        "byte_count": 19_648,
        "status": contract.PREREGISTRATION_STATUS,
    }
    assert contract.frozen_v9_source_manifest_binding() == {
        "path": contract.FROZEN_V9_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": "2c349f8225e8525e691c77e2fb0fe5573d92cb89",
        "file_sha256": "1da8e6fe8babac775ae6a977e2b480ecb852f4cd2edbefca49d9e3105c9ab474",
        "content_sha256": "2003108a6a13a9532210f31839a632d44d7cd47e14965ad93a870d4633e5073a",
        "byte_count": 44_703,
        "status": "PASS_SOURCE_CLOSURE",
        "source_count": 127,
    }
    assert contract.frozen_v9_review_binding() == {
        "path": contract.FROZEN_V9_REVIEW_RELATIVE_PATH,
        "commit": "c0c7d4c6d26dc284b521cf696e7718bfabe73062",
        "file_sha256": "f7e45b7085696c20f472376b06020bec27623075e95acd7df8b1bb986708219f",
        "content_sha256": "65cb9734019cd7f7d78a7e9d9ae2d442bde59115c0e22b18dba9dda64efe853b",
        "byte_count": 70_484,
        "status": contract.FROZEN_V9_REVIEW_STATUS,
    }
    assert contract.frozen_v9_authorization_binding() == {
        "path": contract.FROZEN_V9_AUTHORIZATION_RELATIVE_PATH,
        "commit": "a4d31f688ccd05cea283db318fb73b988fbf087d",
        "file_sha256": "e61545300c9902cd9786f636049a143b945977e76f9fe5cc49abe7d84d02412e",
        "content_sha256": "f9f4c3acd33ff6de559c0815d980519e1f8e1d48f42e899747aca340715de498",
        "byte_count": 58_605,
        "status": contract.FROZEN_V9_AUTHORIZATION_STATUS,
    }
    assert contract.v9_terminal_audit_binding() == {
        "path": contract.V9_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": "7984c7749cc44c6444bf7229a809c6e9f01063bf",
        "file_sha256": "af5f82c809aae3f3954e64147b3a71af96286436e1790440fdd161bc86bd4c03",
        "content_sha256": "f43745ae1329775f36a0b9ad01b34588e662387bca98c34430324cf09a0cd69c",
        "byte_count": 18_951,
        "status": contract.V9_TERMINAL_AUDIT_STATUS,
        "classification": contract.V9_TERMINAL_AUDIT_CLASSIFICATION,
    }
    governing = contract.validate_governing_documents()
    for binding in (
        contract.frozen_v9_source_manifest_binding(),
        contract.frozen_v9_review_binding(),
        contract.frozen_v9_authorization_binding(),
        contract.v9_terminal_audit_binding(),
        contract.preregistration_binding(),
    ):
        assert governing[binding["path"]] == binding["file_sha256"]
    science = contract.science_contract()
    frozen = contract.frozen_v9_science_contract()
    identity = contract.science_identity_receipt()
    assert identity["scientific_delta_count"] == 1
    assert identity["frozen_v9_science_contract_sha256"] == (
        "bacb31b0eb2070821bbd37862e6f3b9a39d7ecb0ab14ed8d758894c36f06f728"
    )
    assert identity["v10_science_contract_sha256"] == (
        "bf839c0897d73f21b789b8e4c0d9277cba6c2c387e4ccbe347aa4cf91eadff43"
    )
    assert contract.normalize_v10_scientific_identity(science) == frozen
    for field in contract.PRESERVED_FROZEN_V9_SCIENCE_TOP_LEVEL_FIELDS:
        assert science[field] == frozen[field]
    assert contract.SCIENTIFIC_DELTA["scientific_delta_count"] == 1
    assert science["scientific_delta"] == contract.SCIENTIFIC_DELTA
    assert science["objective"]["J"] == frozen["objective"]["J"]
    assert science["objective"]["C"] == frozen["objective"]["C"]
    assert science["model"] == frozen["model"]
    assert science["data"] == frozen["data"]
    assert science["optimizer"] == frozen["optimizer"]
    assert science["schedule"] == frozen["schedule"]
    changed = copy.deepcopy(science)
    changed["optimizer"]["epsilon"] = 1e-7
    with pytest.raises(PermissionError, match="differs"):
        contract.normalize_v10_scientific_identity(changed)


def test_source_checker_is_exactly_six_additive_127_reused_133_total() -> None:
    checker = _load(CHECKER, "_direct_bev_v10_source_checker")
    contract = checker.contract
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 6
    assert len(contract.REUSED_SOURCE_PATHS) == 127
    assert len(contract.SOURCE_PATHS) == 133
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
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 133
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    raw = contract.canonical_json_bytes(manifest) + b"\n"
    assert contract.validate_source_manifest(raw) == manifest


def test_final_class_macro_nll_is_exact_and_omits_absent_classes() -> None:
    api = _load(MODEL, "_direct_bev_v10_macro_exact")
    torch = api.torch
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0, 0.5], [1.0, 0.0, -0.5]],
                [[-1.0, 2.0, 0.0], [0.0, 1.0, 0.5]],
                [[0.0, 0.0, 1.0], [-1.0, -1.0, 2.0]],
            ],
            [
                [[0.5, 1.5, -0.5], [0.0, 1.0, -1.0]],
                [[1.0, 0.0, 0.5], [1.5, -0.5, 0.0]],
                [[-0.5, -1.0, 1.0], [-1.0, 0.5, 1.5]],
            ],
        ],
        dtype=torch.float64,
    )
    # Raster zero omits OCCUPIED; raster one contains OCCUPIED only.
    labels = torch.tensor(
        [[[0, 0, 1], [1, 1, 0]], [[2, 2, 2], [2, 2, 2]]],
        dtype=torch.long,
    )
    observed = api._final_class_macro_nll_per_row_v10(logits, labels)
    log_prob = torch.log_softmax(logits, dim=1)
    expected = []
    for row in range(2):
        class_means = []
        for state_class in (0, 1, 2):
            mask = labels[row] == state_class
            if bool(mask.any()):
                class_means.append(
                    (-log_prob[row, state_class][mask]).mean()
                )
        expected.append(torch.stack(class_means).mean())
    assert torch.equal(observed, torch.stack(expected))
    assert torch.equal(observed[1], (-log_prob[1, 2]).mean())


def test_final_class_macro_nll_rejects_invalid_inputs() -> None:
    api = _load(MODEL, "_direct_bev_v10_macro_invalid")
    torch = api.torch
    logits = torch.zeros(1, 3, 2, 2)
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    invalid = [
        (logits[0], labels, ValueError, "shape"),
        (torch.zeros(1, 2, 2, 2), labels, ValueError, "shape"),
        (torch.zeros(1, 3, 2, 2, dtype=torch.long), labels, TypeError, "floating"),
        (logits, labels[:, :1], ValueError, "shape"),
        (logits, labels.float(), TypeError, "integer"),
        (logits, labels.bool(), TypeError, "integer"),
        (logits, torch.full_like(labels, 3), ValueError, "outside"),
        (
            torch.zeros(1, 3, 0, 1),
            torch.zeros(1, 0, 1, dtype=torch.long),
            ValueError,
            "at least one",
        ),
    ]
    for bad_logits, bad_labels, error, match in invalid:
        with pytest.raises(error, match=match):
            api._final_class_macro_nll_per_row_v10(bad_logits, bad_labels)
    meta_labels = torch.zeros(1, 2, 2, dtype=torch.long, device="meta")
    with pytest.raises(TypeError, match="share a device"):
        api._final_class_macro_nll_per_row_v10(logits, meta_labels)


def test_model_identity_objective_scaling_calls_isolation_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _load(MODEL, "_direct_bev_v10_model_objective")
    torch = api.torch
    encoder = api._v8._v6._v3._v1._construct_n320_encoder_without_rng_draw()
    for value in encoder.state_dict().values():
        if value.is_floating_point():
            value.zero_()
    caller_rng = torch.random.get_rng_state().clone()
    frozen = api._v8.DirectEgocentricBevStateJepaV1(encoder.state_dict())
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    model = api.DirectEgocentricBevStateJepaV1(encoder.state_dict())
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert tuple(model.state_dict()) == tuple(frozen.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(value, frozen.state_dict()[name])
    assert [name for name, _ in model.named_parameters()] == [
        name for name, _ in frozen.named_parameters()
    ]
    assert [name for name, _ in model.named_buffers()] == [
        name for name, _ in frozen.named_buffers()
    ]
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in frozen.parameters()
    )
    del frozen

    model.arm_phase_schedule_v6()
    calls = {"online": 0, "target": 0, "predictor": 0}
    frozen_online = model.online_state
    frozen_target = model.target_state

    def online(rgb: Any) -> Any:
        calls["online"] += 1
        return frozen_online(rgb)

    def target(rgb: Any) -> Any:
        calls["target"] += 1
        return frozen_target(rgb)

    monkeypatch.setattr(model, "online_state", online)
    monkeypatch.setattr(model, "target_state", target)
    handle = model.predictor.register_forward_pre_hook(
        lambda *_: calls.__setitem__("predictor", calls["predictor"] + 1)
    )
    torch.manual_seed(11)
    current = torch.rand(1, 3, 112, 112)
    next_rgb = current.roll(1, 3)
    fixed = current.flip(3).detach().requires_grad_(True)
    actions = torch.zeros(1, 9)
    actions[:, 3] = 1.0
    current_labels = torch.randint(0, 3, (1, 64, 64))
    next_labels = current_labels.roll(1, 2)
    result = model.training_objective(
        current_rgb=current,
        next_rgb=next_rgb,
        fixed_negative_rgb=fixed,
        action_one_hot=actions,
        non_hold_mask=torch.tensor([True]),
        current_labels=current_labels,
        next_labels=next_labels,
    )
    handle.remove()
    assert calls == {"online": 2, "target": 3, "predictor": 0}
    raw_current = api._final_class_macro_nll_per_row_v10(
        result.current_state_logits, current_labels
    ).mean()
    raw_next = api._final_class_macro_nll_per_row_v10(
        result.next_online_state_logits, next_labels
    ).mean()
    raw_macro = 0.5 * (raw_current + raw_next)
    assert api.GROUNDING_PUBLIC_SCALE_V10 == pytest.approx(
        math.log(2.0) / math.log(3.0), rel=0.0, abs=0.0
    )
    assert torch.equal(
        result.G_current, raw_current * api.GROUNDING_PUBLIC_SCALE_V10
    )
    assert torch.equal(
        result.G_next, raw_next * api.GROUNDING_PUBLIC_SCALE_V10
    )
    assert torch.equal(result.G, raw_macro * api.GROUNDING_PUBLIC_SCALE_V10)
    assert torch.equal(result.total, result.G / math.log(2.0))
    torch.testing.assert_close(
        result.total,
        raw_macro / math.log(3.0),
        rtol=2.0 * torch.finfo(result.total.dtype).eps,
        atol=0.0,
    )
    result.total.backward()
    for component in (model.encoder, model.bev_decoder, model.state_head):
        gradients = [
            parameter.grad
            for parameter in component.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    assert all(
        parameter.grad is None
        for module in model._target_modules()
        for parameter in module.parameters()
    )
    assert fixed.grad is None


def test_wrong_rgb_control_uses_raw_macro_loss_and_only_two_online_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _load(MODEL, "_direct_bev_v10_wrong_rgb")
    torch = api.torch
    encoder = api._v8._v6._v3._v1._construct_n320_encoder_without_rng_draw()
    for value in encoder.state_dict().values():
        if value.is_floating_point():
            value.zero_()
    model = api.DirectEgocentricBevStateJepaV1(encoder.state_dict())
    calls = {"online": 0, "predictor": 0}
    frozen_online = model.online_state

    def online(rgb: Any) -> Any:
        calls["online"] += 1
        return frozen_online(rgb)

    monkeypatch.setattr(model, "online_state", online)
    handle = model.predictor.register_forward_pre_hook(
        lambda *_: calls.__setitem__("predictor", calls["predictor"] + 1)
    )
    torch.manual_seed(13)
    next_rgb = torch.rand(2, 3, 112, 112)
    wrong_rgb = next_rgb.flip(3)
    labels = torch.randint(0, 3, (2, 64, 64))
    result = model.wrong_rgb_grounding_control(
        next_rgb=next_rgb,
        fixed_negative_rgb=wrong_rgb,
        next_labels=labels,
    )
    handle.remove()
    assert calls == {"online": 2, "predictor": 0}
    assert torch.equal(
        result.correct_next_loss_per_row,
        api._final_class_macro_nll_per_row_v10(
            result.correct_next_state_logits, labels
        ),
    )
    assert torch.equal(
        result.mapped_negative_loss_per_row,
        api._final_class_macro_nll_per_row_v10(
            result.mapped_negative_state_logits, labels
        ),
    )


def _gate_common(contract: Any, update: int) -> dict[str, Any]:
    return {
        **contract.perception_accounting(update),
        "v8_mechanism_receipt_ready": True,
        "active_training_scope_v8": "perception_only",
        "all_registered_values_finite": True,
        "state_nonconstant": True,
    }


def _update_zero_gate_metrics(contract: Any) -> dict[str, Any]:
    fields = (
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
    return {
        **_gate_common(contract, 0),
        **{field: True for field in fields},
        "initial_online_to_target_hard_sync_count": 1,
        "correct_rgb_scene_win_count": 8,
        "G": 1.0,
        "rough_raster_balanced_accuracy": 0.10,
        "rough_raster_occupied_recall": 0.10,
    }


def test_exact_u0_u50_u100_u250_gate_comparators_and_v9_counterexample() -> None:
    contract = _load(CONTRACT, "_direct_bev_v10_gate_contract")
    zero = _update_zero_gate_metrics(contract)
    assert contract.evaluate_gate(0, zero)["control"] == (
        contract.CONTROL_CONTINUE_UPDATE_ZERO
    )
    bad_zero = dict(zero)
    bad_zero["correct_rgb_scene_win_count"] = 7
    assert contract.evaluate_gate(0, bad_zero)["passed"] is False
    bad_zero = dict(zero)
    bad_zero["v8_intended_gradient_coverage_exact"] = False
    assert contract.evaluate_gate(0, bad_zero)["passed"] is False

    fifty = {
        **_gate_common(contract, 50),
        "G": 0.90,
        "aggregate_raster_balanced_accuracy": 0.60,
        "aggregate_free_recall": 0.25,
        "aggregate_occupied_recall": 0.75,
        "aggregate_raster_nll": 0.80,
        "rough_raster_balanced_accuracy": 0.01,
        "rough_raster_occupied_recall": 0.01,
        "correct_rgb_scene_win_count": 8,
    }
    assert contract.evaluate_gate(50, fifty, update_zero=zero)["control"] == (
        contract.CONTROL_CONTINUE_UPDATE_50
    )
    fifty_failures = (
        {"G": zero["G"]},
        {"aggregate_raster_balanced_accuracy": 0.599999},
        {"aggregate_free_recall": 0.249999},
        {"aggregate_occupied_recall": 0.749999},
        {"aggregate_free_recall": 0.25, "aggregate_occupied_recall": 0.850001},
        {"aggregate_raster_nll": 0.800001},
        {"rough_raster_balanced_accuracy": 0.0},
        {"rough_raster_occupied_recall": 0.0},
        {"correct_rgb_scene_win_count": 7},
    )
    for mutation in fifty_failures:
        candidate = {**fifty, **mutation}
        assert contract.evaluate_gate(
            50, candidate, update_zero=zero
        )["passed"] is False

    hundred = {
        **_gate_common(contract, 100),
        "G": 0.80,
        "aggregate_raster_balanced_accuracy": 0.70,
        "aggregate_free_recall": 0.50,
        "aggregate_occupied_recall": 0.80,
        "aggregate_raster_nll": 0.46,
        "rough_raster_balanced_accuracy": 0.100001,
        "rough_raster_occupied_recall": 0.100001,
        "correct_rgb_scene_win_count": 8,
    }
    assert contract.evaluate_gate(
        100, hundred, update_zero=zero
    )["control"] == contract.CONTROL_CONTINUE_UPDATE_100
    hundred_failures = (
        {"G": zero["G"]},
        {"aggregate_raster_balanced_accuracy": 0.699999},
        {"aggregate_free_recall": 0.499999},
        {"aggregate_occupied_recall": 0.799999},
        {"aggregate_free_recall": 0.50, "aggregate_occupied_recall": 0.850001},
        {"aggregate_raster_nll": 0.460001},
        {"rough_raster_balanced_accuracy": zero["rough_raster_balanced_accuracy"]},
        {"rough_raster_occupied_recall": zero["rough_raster_occupied_recall"]},
        {"correct_rgb_scene_win_count": 7},
    )
    for mutation in hundred_failures:
        candidate = {**hundred, **mutation}
        assert contract.evaluate_gate(
            100, candidate, update_zero=zero
        )["passed"] is False

    terminal = {
        **_gate_common(contract, 250),
        "G": 0.70,
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
    terminal_failures = (
        ({"aggregate_raster_balanced_accuracy": 0.799999}, {"aggregate_raster_nll": 0.41}),
        ({"aggregate_free_recall": 0.679999}, {"aggregate_raster_nll": 0.41}),
        ({"aggregate_occupied_recall": 0.879999}, {"aggregate_raster_nll": 0.41}),
        (
            {"aggregate_free_recall": 0.68, "aggregate_occupied_recall": 0.930001},
            {"aggregate_raster_nll": 0.41},
        ),
        ({"aggregate_raster_nll": 0.420001}, {"aggregate_raster_nll": 0.42}),
        ({"aggregate_raster_nll": 0.410001}, {"aggregate_raster_nll": 0.40}),
        ({"rough_raster_balanced_accuracy": 0.7719524}, {"aggregate_raster_nll": 0.41}),
        ({"rough_raster_occupied_recall": 0.4319466}, {"aggregate_raster_nll": 0.41}),
        ({"correct_rgb_scene_win_count": 7}, {"aggregate_raster_nll": 0.41}),
    )
    for mutation, baseline in terminal_failures:
        candidate = {**terminal, **mutation}
        assert contract.evaluate_gate(
            250,
            candidate,
            update_zero=zero,
            update_100=baseline,
        )["passed"] is False
    assert contract.evaluate_gate(
        250,
        terminal,
        update_zero=zero,
        update_100={"aggregate_raster_nll": 0.41},
        prior_gates_passed=False,
    )["passed"] is False

    v9_update_100 = {
        **_gate_common(contract, 100),
        "G": 0.2974271838713174,
        "aggregate_raster_balanced_accuracy": 0.6292200074101383,
        "aggregate_free_recall": 0.15401097508116818,
        "aggregate_occupied_recall": 0.8571428571428571,
        "aggregate_raster_nll": 0.5243744905949164,
        "rough_raster_balanced_accuracy": 0.100001,
        "rough_raster_occupied_recall": 0.100001,
        "correct_rgb_scene_win_count": 8,
    }
    v9_gate = contract.evaluate_gate(100, v9_update_100, update_zero=zero)
    assert v9_gate["passed"] is False
    for conjunct in (
        "aggregate_raster_balanced_accuracy_at_least_point70",
        "aggregate_free_recall_at_least_point50",
        "absolute_free_occupied_recall_gap_at_most_point35",
        "aggregate_raster_nll_at_most_point46",
    ):
        assert v9_gate["conjuncts"][conjunct] is False

    for control in contract.FAILURE_CONTROLS:
        chain = dict.fromkeys(("metrics", "artifact", "result", "completion"), control)
        assert contract.validate_failure_status_chain(chain) == chain
    with pytest.raises(ValueError, match="not one exact"):
        contract.validate_failure_status_chain({
            "metrics": contract.CONTROL_UPDATE_50_FAIL,
            "artifact": contract.CONTROL_UPDATE_50_FAIL,
            "result": contract.CONTROL_UPDATE_100_FAIL,
            "completion": contract.CONTROL_UPDATE_50_FAIL,
        })


def test_runner_preserves_v9_registry_failure_seams_and_delegates_to_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v10_runner_delegation")
    runner._assert_v10_bindings()
    expected_v9 = dict(runner._V9._V9_SEAM_TABLE)
    for name, expected_v8 in runner._V9._V8._V8_SEAM_TABLE:
        assert getattr(runner._LEAF, name) is expected_v9.get(name, expected_v8)
    snapshot = runner._LEAF._snapshot_model
    failure = runner._LEAF._terminal_failure
    observation = runner._LEAF._evaluate_observation_impl
    assert snapshot is runner._V9._v9_snapshot_model
    assert failure is runner._V9._v9_terminal_failure
    assert observation is runner._V9._v9_evaluate_observation_impl
    runner._rebind_inherited_runner()
    assert runner._LEAF._snapshot_model is snapshot
    assert runner._LEAF._terminal_failure is failure
    assert runner._LEAF._evaluate_observation_impl is observation
    for intermediate in (runner._V9, runner._V9._V8, runner._V9._V8._V7):
        monkeypatch.setattr(intermediate, "parse_args", _bomb)
        monkeypatch.setattr(intermediate, "run_parent", _bomb)
        monkeypatch.setattr(intermediate, "main", _bomb)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        runner._LEAF,
        "parse_args",
        lambda argv=None: calls.append(("parse", argv)) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner._LEAF,
        "run_parent",
        lambda **kwargs: calls.append(("run", kwargs)) or 31,
    )
    monkeypatch.setattr(
        runner._LEAF,
        "main",
        lambda argv=None: calls.append(("main", argv)) or 37,
    )
    assert isinstance(runner.parse_args(["synthetic"]), SimpleNamespace)
    assert runner.run_parent(
        review_file_sha256="1" * 64,
        authorization_file_sha256="2" * 64,
    ) == 31
    assert runner.main(["synthetic-main"]) == 37
    assert calls == [
        ("parse", ["synthetic"]),
        (
            "run",
            {
                "review_file_sha256": "1" * 64,
                "authorization_file_sha256": "2" * 64,
            },
        ),
        ("main", ["synthetic-main"]),
    ]


def test_launcher_rebinds_v10_authority_and_delegates_directly_to_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load(LAUNCHER, "_direct_bev_v10_launcher_delegation")
    launcher._assert_v10_bindings()
    for intermediate in (
        launcher._V9,
        launcher._V9._V8,
        launcher._V9._V8._V7,
    ):
        monkeypatch.setattr(intermediate, "parse_args", _bomb)
        monkeypatch.setattr(intermediate, "main", _bomb)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        launcher._LEAF,
        "parse_args",
        lambda argv=None: calls.append(("parse", argv)) or object(),
    )
    monkeypatch.setattr(
        launcher._LEAF,
        "main",
        lambda argv=None: calls.append(("main", argv)) or 41,
    )
    args = ["--review-sha256", "3" * 64]
    launcher.parse_args(args)
    assert launcher.main(args) == 41
    assert calls == [("parse", args), ("main", args)]
    assert launcher.contract.PREFLIGHT_ENVIRONMENT_KEY == PREFLIGHT_KEY
    assert launcher._LEAF._V11._BASE.RUNNER_PATH == RUNNER
    assert launcher.contract.MODEL_RELATIVE_PATH == MODEL.relative_to(ROOT).as_posix()
