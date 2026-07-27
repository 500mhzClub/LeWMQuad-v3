from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)


def _load(name: str = "_geometry_anchored_joint_jepa_runner_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_source_only_under_isolated_python() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_runner", {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert module.RUNNER_PATH == module.ROOT / module.contract.RUNNER_RELATIVE_PATH
args = module.parse_args([
    "--review-sha256", "0" * 64,
    "--authorization-sha256", "1" * 64,
])
assert args.review_sha256 == "0" * 64
assert args.authorization_sha256 == "1" * 64
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


def test_semantic_objective_is_exact_equal_side_log3_normalization() -> None:
    runner = _load("_geometry_anchored_joint_jepa_semantic_test")

    class Rows:
        def __init__(self, value: float) -> None:
            self.value = value

        def mean(self) -> float:
            return self.value

    class Model:
        def encode_online(self, value: str) -> str:
            return f"latent:{value}"

        def semantic_logits_from_latent(self, value: str) -> str:
            return f"logits:{value}"

    class ModelApi:
        @staticmethod
        def final_class_macro_nll_per_row(logits: str, labels: str) -> Rows:
            assert logits == f"logits:latent:{labels.removesuffix('_labels')}_rgb"
            return Rows(2.0 if labels == "current_labels" else 4.0)

    result = runner._semantic_terms(
        ModelApi,
        Model(),
        {
            "current_rgb": "current_rgb",
            "next_rgb": "next_rgb",
            "current_labels": "current_labels",
            "next_labels": "next_labels",
        },
    )
    assert result["A"] == pytest.approx(3.0)
    assert result["S"] == pytest.approx(3.0 / math.log(3.0))
    assert result["current_latent"] == "latent:current_rgb"
    assert result["next_latent"] == "latent:next_rgb"


def test_update_401_phase_accounting_is_exact_and_conjunctive() -> None:
    runner = _load("_geometry_anchored_joint_jepa_phase_test")
    passing = {
        "optimizer_identity_unchanged": True,
        "optimizer_parameter_group_membership_unchanged": True,
        "joint_objective_formula_exact": True,
        "online_representation_gradient_finite_nonzero": True,
        "predictor_gradient_finite_nonzero": True,
        "target_gradients_absent": True,
        "shared_gradient_contribution_gate_passed": True,
        "online_optimizer_update_count": 401,
        "target_ema_update_count": 401,
        "predictor_optimizer_update_count": 1,
        "joint_optimizer_update_count": 1,
    }
    receipt = runner.contract.evaluate_update_401_phase_switch(passing)
    assert receipt["passed"] is True
    assert all(receipt["conjuncts"].values())
    assert receipt["control"] == runner.contract.PHASE_SWITCH_CONTROLS[1]

    failed = dict(passing, predictor_optimizer_update_count=0)
    receipt = runner.contract.evaluate_update_401_phase_switch(failed)
    assert receipt["passed"] is False
    assert receipt["control"] == runner.contract.PHASE_SWITCH_CONTROLS[0]
    assert receipt["conjuncts"][
        "first_predictor_optimizer_update_count_equals_1"
    ] is False


def test_integrity_scope_excludes_encoder_attention_and_artifact_schema_is_exact() -> None:
    runner = _load("_geometry_anchored_joint_jepa_scope_test")
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "for component in (model.bev_lift, model.predictor)" in source
    assert "for component in (model.encoder, model.bev_lift, model.predictor)" not in source
    assert 'output_root / "artifact.json"' in source
    assert '"schema": contract.ARTIFACT_SCHEMA' in source
    assert runner.contract.ARTIFACT_SCHEMA == (
        f"{runner.contract.SCHEMA_PREFIX}_artifact_v1"
    )


def test_shared_gradient_gate_measures_semantic_and_dynamics_directly() -> None:
    runner = _load("_geometry_anchored_joint_jepa_gradient_scope_test")
    source = inspect.getsource(runner._train_probe)
    assert source.count("torch.autograd.grad(") == 2
    assert (
        "S / contract.MICROBATCHES_PER_UPDATE,\n"
        "                    shared_parameters," in source
    )
    assert (
        'joint["D"] / contract.MICROBATCHES_PER_UPDATE,\n'
        "                    shared_parameters," in source
    )
    assert "total_gradient.detach() - semantic_gradient" not in source
    assert "semantic_gradient_accumulator" in source
    assert "dynamics_gradient_accumulator" in source


@pytest.mark.parametrize("scientific", (False, True))
def test_terminal_failure_writes_complete_no_retry_receipts(
    tmp_path: Path, scientific: bool
) -> None:
    runner = _load(
        f"_geometry_anchored_joint_jepa_failure_{int(scientific)}"
    )
    output_root = tmp_path / "attempt"
    output_root.mkdir(mode=0o700)
    reservation = runner.contract.with_content_sha256({
        "schema": f"{runner.contract.SCHEMA_PREFIX}_reservation_test_v1",
        "status": "RESERVED_SYNTHETIC_SOURCE_ONLY_TEST",
    })
    reservation_raw = runner.contract.canonical_json_bytes(reservation) + b"\n"
    error: BaseException
    if scientific:
        error = runner.ScientificGateFailure(
            "synthetic ratio failure",
            control=runner.contract.CONTROL_FAIL_JOINT_GRADIENT,
        )
        expected_control = runner.contract.CONTROL_FAIL_JOINT_GRADIENT
        expected_classification = "SCIENTIFIC_GATE_FAILURE"
    else:
        error = RuntimeError("synthetic operational failure")
        expected_control = runner.contract.CONTROL_FAIL_OPERATIONAL
        expected_classification = "OPERATIONAL_OR_INTEGRITY_FAILURE"
    try:
        runner._terminal_failure(
            output_root,
            reservation,
            reservation_raw,
            {"stage": "synthetic", "updates": 0, "presentations": 0},
            error,
        )
        failure = json.loads((output_root / "failure.json").read_bytes())
        metrics = json.loads((output_root / "metrics.json").read_bytes())
        artifact = json.loads((output_root / "artifact.json").read_bytes())
        access = json.loads((output_root / "access.json").read_bytes())
        result = json.loads((output_root / "result.json").read_bytes())
        completed = json.loads((output_root / "completed.json").read_bytes())
        assert failure["schema"] == runner.contract.FAILURE_SCHEMA
        assert failure["status"] == expected_control
        assert failure["classification"] == expected_classification
        assert failure["retry_resume_repair_or_replacement_authorized"] is False
        assert failure["checkpoint_qualified"] is False
        assert failure["g2_navigation_heldout_sealed_open_count"] == 0
        assert metrics["schema"] == runner.contract.METRICS_SCHEMA
        assert metrics["complete_failure_receipt"] is True
        assert metrics["failure"]["path"] == "failure.json"
        assert artifact["schema"] == runner.contract.ARTIFACT_SCHEMA
        assert artifact["complete_failure_receipt"] is True
        assert artifact["checkpoints"] == []
        assert artifact["training_trace"] is None
        assert artifact["checkpoint_read_count_after_write"] == 0
        assert artifact["training_trace_read_count_after_write"] == 0
        assert access["schema"] == runner.contract.ACCESS_SCHEMA
        assert access["complete_failure_receipt"] is True
        assert access["access_receipt_complete"] is False
        assert access["rejected_checkpoint_open_count"] == 0
        assert access["prior_runtime_output_open_count"] == 0
        assert access["written_checkpoint_read_count"] == 0
        assert access["training_trace_read_count"] == 0
        assert result["schema"] == runner.contract.RESULT_SCHEMA
        assert result["complete_failure_receipt"] is True
        assert result["mechanism_passed"] is False
        assert result["checkpoint_qualified"] is False
        assert result["retry_authorized"] is False
        assert {
            result[name]["path"] for name in (
                "metrics", "artifact", "access", "failure"
            )
        } == {"metrics.json", "artifact.json", "access.json", "failure.json"}
        assert completed["schema"] == runner.contract.COMPLETION_SCHEMA
        assert completed["status"] == expected_control
        assert completed["complete_failure_receipt"] is True
        assert completed["checkpoint_qualified"] is False
        assert completed["retry_authorized"] is False
        assert set(path.name for path in output_root.iterdir()) == {
            "metrics.json",
            "artifact.json",
            "access.json",
            "result.json",
            "failure.json",
            "completed.json",
        }
        assert all(
            stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o444
            for path in output_root.iterdir()
        )
    finally:
        for path in output_root.iterdir():
            os.chmod(path, 0o600, follow_symlinks=False)
        os.chmod(output_root, 0o700, follow_symlinks=False)
