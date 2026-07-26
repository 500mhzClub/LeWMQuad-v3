from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT / "scripts/run_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
LAUNCHER_PATH = (
    ROOT / "scripts/launch_go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)


def _load_runner(name: str = "_masked_pair_tubelet_v11_runner_test"):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_launcher(name: str = "_masked_pair_tubelet_v11_launcher_test"):
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_source_only_and_bound_to_v11() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_v11_source_only", {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert module.contract.PREREGISTRATION_COMMIT == "46de4c1b6a89dad43550b62a6e9327dec0a7b9da"
assert module.PREFLIGHT_ENVIRONMENT_KEY == "LEWM_RGB_MASKED_CURRENT_NEXT_PAIR_TUBELET_JEPA_V11_PREFLIGHT_JSON"
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


def test_launcher_is_source_only_and_delegates_to_exact_v11_runner() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_v11_launcher", {str(LAUNCHER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert module._BASE.RUNNER_PATH == module.ROOT / module.contract.RUNNER_RELATIVE_PATH
assert module._BASE.PREFLIGHT_ENVIRONMENT_KEY == module.PREFLIGHT_ENVIRONMENT_KEY
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


def test_executor_has_one_phase_and_never_enters_phase_b() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "_phase_b_train(" not in source
    assert 'progress["phase_b_entered"] = True' not in source
    assert '"phase_b": None' in source
    assert '"checkpoint_qualified": False' in source
    assert "_run_phase_a_with_strict_determinism" in source
    assert "warn_only=False" in source
    assert "contract.evaluate_phase_a_update_zero(" in source
    assert 'if early_failure is None\n        else ()' in source
    assert 'DOWNSTREAM_DENIALS["pass_next_step"]' not in source
    assert source.count('DOWNSTREAM_DENIALS["pass_authorizes"]') == 2
    assert 'required_model_facing_roles = {"checkpoint_selection"}' in source
    assert 'if phase_a["presentations"] > 0:' in source
    assert 'required_model_facing_roles.add("train")' in source


def test_runner_and_contract_gate_api_are_exactly_one_phase() -> None:
    module = _load_runner("_masked_pair_tubelet_v11_gate_api")
    assert tuple(
        inspect.signature(
            module.contract.evaluate_phase_a_continuation
        ).parameters
    ) == (
        "update",
        "metrics",
        "update0_metrics",
        "observation_integrity",
        "previous_metrics",
    )
    assert tuple(
        inspect.signature(module.contract.evaluate_phase_a).parameters
    ) == (
        "metrics",
        "update0_metrics",
        "observation_integrity",
        "previous_metrics",
    )
    assert module.contract.PHASE_A_MAXIMUM_UPDATE == 1_000
    assert module.contract.MAXIMUM_PRESENTATIONS == 16_000
    assert module.contract.PHASE_B_MAXIMUM_UPDATE == 0
    assert module.contract.CONTROL_PHASE_A_PASS == (
        "PASS_MASKED_PAIR_TUBELET_PROXY_SEPARATE_REQUALIFICATION_ONLY"
    )


def test_update_zero_compares_all_36_action_pairs_across_495_rows() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_masked_pair_tubelet_v11_symmetry")
    base = torch.arange(495 * 6, dtype=torch.float32).reshape(495, 1, 2, 3)
    predictions = base.expand(-1, 9, -1, -1).clone()
    row_count = 0
    comparison_count = None
    for batch in predictions.split(16):
        rows, pairs = module._verify_update_zero_action_symmetry_batch(
            torch, batch
        )
        row_count += rows
        comparison_count = pairs
    assert module._update_zero_action_symmetry_receipt(
        row_count=row_count,
        comparison_count=comparison_count,
    ) == {
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
    }
    predictions[-1, -1, -1, -1] += 1
    with pytest.raises(PermissionError, match="not bitwise equal"):
        module._verify_update_zero_action_symmetry_batch(
            torch, predictions[-15:]
        )


def test_gate_references_are_exact_float32_495x9_and_494x2() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_masked_pair_tubelet_v11_gate_refs")
    pairs = [
        {"primitive": module.contract.ACTION_VOCABULARY[index % 9]}
        for index in range(495)
    ]
    mapping = {"same_action_eligible": (True,) * 494 + (False,)}
    observed = module._phase_a_gate_references(
        SimpleNamespace(torch=torch),
        pairs,
        mapping,
        torch.device("cpu"),
    )
    assert observed["action_equal_logit_reference"] == pytest.approx(
        float(torch.log(torch.tensor(9.0, dtype=torch.float32)))
    )
    assert observed["two_target_equal_logit_reference"] == pytest.approx(
        float(torch.log(torch.tensor(2.0, dtype=torch.float32)))
    )


def test_phase_a_loss_keeps_future_tensors_out_of_online_autograd() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_masked_pair_tubelet_v11_loss_separation")

    class FakeModel:
        def __init__(self) -> None:
            self.online_inputs: list[object] = []
            self.target_inputs: list[tuple[object, object, object]] = []

        def predict_all_actions(self, current_rgb):
            self.online_inputs.append(current_rgb)
            base = current_rgb.mean(dim=(1, 2, 3))[:, None, None, None]
            action = torch.arange(9, dtype=current_rgb.dtype)[None, :, None, None]
            predictions = (base + 0.01 * action).expand(
                -1, -1, 256, 192
            )
            return SimpleNamespace(
                normalized_projected_future=predictions,
                action_indices=torch.arange(9, dtype=torch.long),
                shared_current_patch_tokens=base[:, 0].expand(-1, 256, 192),
            )

        def build_fixed_current_targets(
            self, current_rgb, correct_next_rgb, deranged_next_rgb
        ):
            self.target_inputs.append(
                (current_rgb, correct_next_rgb, deranged_next_rgb)
            )

            def target(value):
                scalar = value.detach().mean(dim=(1, 2, 3))[:, None, None]
                return scalar.expand(-1, 256, 192)

            return SimpleNamespace(
                correct_next=target(correct_next_rgb),
                deranged_next=target(deranged_next_rgb),
                no_change_current=target(current_rgb),
            )

    def objective(all_actions, targets, executed):
        predictions = all_actions.normalized_projected_future
        rows = torch.arange(predictions.shape[0])
        selected = predictions[rows, executed]
        action_energies = (
            predictions - targets.correct_next[:, None]
        ).square().mean(dim=(2, 3))
        target_candidates = torch.stack(
            (
                targets.correct_next,
                targets.deranged_next,
                targets.no_change_current,
            ),
            dim=1,
        )
        target_energies = (
            selected[:, None] - target_candidates
        ).square().mean(dim=(2, 3))
        action_logits = -action_energies
        target_logits = -target_energies
        action_nll = torch.nn.functional.cross_entropy(
            action_logits, executed, reduction="none"
        )
        target_nll = torch.nn.functional.cross_entropy(
            target_logits,
            torch.zeros(len(executed), dtype=torch.long),
            reduction="none",
        )
        jepa = (selected - targets.correct_next).square().mean()
        zero = jepa.new_zeros(())
        return SimpleNamespace(
            total=jepa + action_nll.mean() + target_nll.mean(),
            masked_future_jepa=jepa,
            action_retrieval=action_nll.mean(),
            target_retrieval=target_nll.mean(),
            whitening_variance=zero,
            whitening_covariance=zero,
            action_energies=action_energies,
            action_logits=action_logits,
            action_nll_per_row=action_nll,
            target_energies=target_energies,
            target_logits=target_logits,
            target_nll_per_row=target_nll,
            target_candidate_mask=torch.ones(
                len(executed), 3, dtype=torch.bool
            ),
        )

    model = FakeModel()
    current = torch.randn(2, 3, 4, 4, requires_grad=True)
    next_rgb = torch.randn(2, 3, 4, 4, requires_grad=True)
    deranged = torch.randn(2, 3, 4, 4, requires_grad=True)
    action = torch.zeros(2, 9)
    action[0, 0] = 1.0
    action[1, 6] = 1.0
    non_hold = torch.tensor((True, False))
    api = SimpleNamespace(masked_pair_tubelet_objective_v11=objective)
    first = module._phase_a_loss(
        SimpleNamespace(torch=torch),
        api,
        model,
        current,
        next_rgb,
        deranged,
        action,
        non_hold,
    )
    replacement = module._phase_a_loss(
        SimpleNamespace(torch=torch),
        api,
        model,
        current,
        next_rgb + 11.0,
        deranged - 7.0,
        action,
        non_hold,
    )
    assert torch.equal(
        first["all_action_predictions"],
        replacement["all_action_predictions"],
    )
    first["loss"].backward()
    assert current.grad is not None and torch.count_nonzero(current.grad)
    assert next_rgb.grad is None
    assert deranged.grad is None
    assert len(model.online_inputs) == 2
    assert len(model.target_inputs) == 2


def test_parameter_partition_rejects_any_unregistered_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_masked_pair_tubelet_v11_partition")
    monkeypatch.setattr(
        module.contract,
        "PHASE_A_PARAMETER_PARTITION",
        None,
        raising=False,
    )

    class Parameter:
        def __init__(self, *, trainable: bool) -> None:
            self.requires_grad = trainable

        def numel(self) -> int:
            return 1

    exact_frozen = next(iter(module.contract.PHASE_A_EXACT_FROZEN_PARAMETER_NAMES))
    encoder_prefix = module.contract.PHASE_A_ENCODER_PARAMETER_PREFIXES[0]
    other_prefix = module.contract.PHASE_A_AUXILIARY_PARAMETER_PREFIXES[0]
    frozen_prefix = module.contract.PHASE_A_FROZEN_PARAMETER_PREFIXES[0]
    rows = [
        (exact_frozen, Parameter(trainable=False)),
        (encoder_prefix + "synthetic", Parameter(trainable=True)),
        (other_prefix + "synthetic", Parameter(trainable=True)),
        (frozen_prefix + "synthetic", Parameter(trainable=False)),
    ]

    class Model:
        def named_parameters(self):
            return iter(rows)

    partition = module._phase_a_parameter_partition(Model())
    assert len(partition["encoder"]) == 1
    assert len(partition["other"]) == 1
    rows.append(("unregistered_science_delta", Parameter(trainable=True)))
    with pytest.raises(PermissionError, match="partition changed"):
        module._phase_a_parameter_partition(Model())


def test_runner_constructs_locked_model_from_n320_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from lewm.models import (  # deferred until the Torch-backed test
        rgb_masked_current_next_pair_tubelet_jepa_v11 as model_api,
    )
    from lewm.models.encoders import VisionEncoder

    module = _load_runner("_masked_pair_tubelet_v11_real_model")
    monkeypatch.setattr(module, "_state_sha", lambda *_args: "a" * 64)
    torch.manual_seed(701)
    fit = SimpleNamespace(
        encoder=VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
    )
    model, partition, receipt = module._phase_a_model(
        SimpleNamespace(torch=torch),
        model_api,
        fit,
        torch.device("cpu"),
    )
    assert receipt["n320_loaded_before_registered_new_parameter_draws"]
    assert receipt["target_ema_inventory_exactly_equal_at_update_zero"]
    assert receipt["target_ema_parameter_pair_count"] == len(
        module.contract.TARGET_EMA_PARAMETER_PAIRS
    )
    assert tuple(model.ema_inventory_exact()) == tuple(
        module.contract.TARGET_EMA_PARAMETER_PAIRS
    )
    assert int(model.ema_update_count) == 0
    assert torch.count_nonzero(model.online_action_embedding.weight) == 0
    assert partition["encoder"] and partition["other"] and partition["frozen"]
    assert not any(parameter.requires_grad for parameter in partition["frozen"])
