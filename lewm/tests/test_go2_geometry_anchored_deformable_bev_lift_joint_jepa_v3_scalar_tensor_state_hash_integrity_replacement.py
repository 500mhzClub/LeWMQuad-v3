from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    GeometryAnchoredDeformableBevLiftJointJepaV1,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
RUNNER = (
    ROOT / "scripts/"
    "run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
LAUNCHER = (
    ROOT / "scripts/"
    "launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement.py"
)
CHECKER = (
    ROOT / "scripts/"
    "check_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v3_"
    "scalar_tensor_state_hash_integrity_replacement_source_closure.py"
)
FROZEN_V2_RUNNER = (
    ROOT / "scripts/"
    "run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _independent_tensor_state_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _synthetic_n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(9917)
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
        assert sum(parameter.numel() for parameter in encoder.parameters()) == 2_747_520
        return {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.mark.parametrize("path", [RUNNER, LAUNCHER])
def test_wrapper_import_is_source_only_under_isolation(path: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(path)!r})
spec = importlib.util.spec_from_file_location("_joint_jepa_v3_wrapper", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
assert module._V2.contract is module.contract
assert module._V2._V1.contract is module.contract
assert Path(module._V2.__file__).resolve() == path
assert Path(module._V2._V1.__file__).resolve() == path
if hasattr(module, "_tensor_state_sha256"):
    assert module._V2._V1._tensor_state_sha256 is module._tensor_state_sha256
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


def test_exact_v2_science_and_v3_root_delegate_through_both_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load("_joint_jepa_v3_scalar_contract_science", CONTRACT)
    v2 = contract._v2
    assert len(v2.SOURCE_PATHS) == 79
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 5
    assert len(contract.SOURCE_PATHS) == 84
    assert set(contract.SOURCE_PATHS) == {
        *v2.SOURCE_PATHS,
        *contract.ADDITIVE_SOURCE_PATHS,
    }
    assert contract.MODEL_RELATIVE_PATH == v2.MODEL_RELATIVE_PATH
    assert not any("models/" in path for path in contract.ADDITIVE_SOURCE_PATHS)
    assert contract.model_config() == v2.model_config()
    assert contract.objective_contract() == v2.objective_contract()
    assert contract.optimizer_contract() == v2.optimizer_contract()
    assert contract.build_schedule_identity() == v2.build_schedule_identity()
    assert contract.runtime_authorization_template() == (
        v2.runtime_authorization_template()
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v2.OUTPUT_ROOT_RELATIVE_PATH
    assert "v3_scalar_tensor_state_hash_integrity_replacement" in (
        contract.OUTPUT_ROOT_RELATIVE_PATH
    )
    assert contract.EXECUTION_AUTHORITY["maximum_updates"] == 1_000
    assert contract.EXECUTION_AUTHORITY["maximum_presentations"] == 16_000
    assert contract.EXECUTION_AUTHORITY["gpu_active_minutes_maximum"] == 30

    runner = _load("_joint_jepa_v3_scalar_runner_delegate", RUNNER)
    calls: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        assert runner._V2.contract is runner.contract
        assert runner._V2._V1.contract is runner.contract
        assert runner._V2._V1._load_post_reservation_stack is (
            runner._V2._load_post_reservation_stack
        )
        assert runner._V2._V1._tensor_state_sha256 is (
            runner._tensor_state_sha256
        )
        calls.append(argv)
        return 23

    monkeypatch.setattr(runner._V2, "main", fake_main)
    args = ["--review-sha256", "a" * 64, "--authorization-sha256", "b" * 64]
    assert runner.main(args) == 23
    assert calls == [args]

    launcher = _load("_joint_jepa_v3_scalar_launcher_delegate", LAUNCHER)
    parsed = launcher.parse_args(args)
    argv = launcher._V2._V1._runtime_argv(parsed)
    assert argv == [
        launcher.contract.RUNTIME_INTERPRETER_PATH,
        *launcher.contract.RUNTIME_INTERPRETER_ARGUMENTS,
        str(ROOT / launcher.contract.RUNNER_RELATIVE_PATH),
        "--review-sha256",
        "a" * 64,
        "--authorization-sha256",
        "b" * 64,
    ]
    assert launcher._V2._V1.OUTPUT_ROOT == (
        ROOT / launcher.contract.OUTPUT_ROOT_RELATIVE_PATH
    )


def test_scalar_float_integer_and_boolean_match_independent_raw_bytes() -> None:
    runner = _load("_joint_jepa_v3_scalar_runner_scalars", RUNNER)
    values = {
        "bool_scalar": torch.tensor(True, dtype=torch.bool),
        "float_scalar": torch.tensor(-3.25, dtype=torch.float32),
        "integer_scalar": torch.tensor(20260727, dtype=torch.int64),
    }
    assert all(value.ndim == 0 for value in values.values())
    assert runner._tensor_state_sha256(torch, values) == (
        _independent_tensor_state_sha256(values)
    )


def test_non_scalar_digest_is_exactly_frozen_v2_and_sensitive() -> None:
    runner = _load("_joint_jepa_v3_scalar_runner_nonscalar", RUNNER)
    frozen_v2 = _load("_joint_jepa_v3_fresh_frozen_v2_runner", FROZEN_V2_RUNNER)
    values = {
        "bools": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        "floats": torch.tensor([[1.0, -0.0, 2.5], [4.0, 8.0, 16.0]]),
        "integers": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    }
    v3_digest = runner._tensor_state_sha256(torch, values)
    assert v3_digest == frozen_v2._V1._tensor_state_sha256(torch, values)
    assert v3_digest == _independent_tensor_state_sha256(values)

    changed = {name: value.clone() for name, value in values.items()}
    changed["integers"][1, 1].add_(1)
    assert runner._tensor_state_sha256(torch, changed) != v3_digest


def test_complete_unchanged_v1_model_hashes_on_cpu_without_side_effects() -> None:
    runner = _load("_joint_jepa_v3_scalar_runner_full_model", RUNNER)
    torch.random.default_generator.manual_seed(421)
    caller_rng = torch.random.get_rng_state().clone()
    caller_path = list(sys.path)

    n320_state = _synthetic_n320_encoder_state()
    model = GeometryAnchoredDeformableBevLiftJointJepaV1(n320_state).cpu()
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    scalar_names = {
        "bev_lift.ground_z_m",
        "bev_lift.horizontal_fov_degrees",
        "bev_lift.vertical_fov_degrees",
        "bev_lift.camera_near_m",
        "target_bev_lift.ground_z_m",
        "target_bev_lift.horizontal_fov_degrees",
        "target_bev_lift.vertical_fov_degrees",
        "target_bev_lift.camera_near_m",
        "target_hard_sync_count",
        "ema_update_count",
    }
    assert scalar_names <= set(state_before)
    assert all(state_before[name].ndim == 0 for name in scalar_names)
    assert model.target_hard_sync_count.item() == 1
    assert model.ema_update_count.item() == 0

    module_states = {
        "predictor": model.predictor.state_dict(),
        "online_encoder": model.encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "online_bev_lift": model.bev_lift.state_dict(),
        "target_bev_lift": model.target_bev_lift.state_dict(),
        "full_model": model.state_dict(),
    }
    digests = {
        name: runner._tensor_state_sha256(torch, state)
        for name, state in module_states.items()
    }
    assert all(len(value) == 64 for value in digests.values())
    assert digests["online_encoder"] == digests["target_encoder"]
    assert digests["online_bev_lift"] == digests["target_bev_lift"]
    assert digests["full_model"] == _independent_tensor_state_sha256(
        model.state_dict()
    )
    changed_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    changed_state["ema_update_count"].add_(1)
    assert runner._tensor_state_sha256(torch, changed_state) != (
        digests["full_model"]
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert sys.path == caller_path
    assert state_before.keys() == model.state_dict().keys()
    assert all(
        torch.equal(state_before[name], model.state_dict()[name])
        for name in state_before
    )


def test_failure_receipts_keep_v2_schemas_but_bind_v3_terminal_status() -> None:
    contract = _load("_joint_jepa_v3_scalar_failure_contract", CONTRACT)
    runner = _load("_joint_jepa_v3_scalar_failure_runner", RUNNER)
    v2 = contract._v2
    for name in (
        "RESERVATION_SCHEMA",
        "METRICS_SCHEMA",
        "ARTIFACT_SCHEMA",
        "ACCESS_SCHEMA",
        "RESULT_SCHEMA",
        "COMPLETION_SCHEMA",
        "FAILURE_SCHEMA",
    ):
        assert getattr(contract, name) == getattr(v2, name)
    assert contract.NORMAL_RECEIPT_PATHS == v2.NORMAL_RECEIPT_PATHS
    assert contract.OPERATIONAL_FAILURE_RECEIPT_PATHS == (
        "metrics.json",
        "artifact.json",
        "access.json",
        "result.json",
        "failure.json",
        "completed.json",
    )
    assert contract.CONTROL_FAIL_OPERATIONAL == contract.OPERATIONAL_FAILURE_STATUS
    assert "V3_SCALAR_TENSOR_STATE_HASH_INTEGRITY_REPLACEMENT" in (
        contract.OPERATIONAL_FAILURE_STATUS
    )
    assert runner._V2._V1.contract.CONTROL_FAIL_OPERATIONAL == (
        contract.OPERATIONAL_FAILURE_STATUS
    )
    assert runner._V2._V1._tensor_state_sha256 is runner._tensor_state_sha256
    assert contract.EXECUTION_AUTHORITY["maximum_attempts"] == 1
    assert contract.EXECUTION_AUTHORITY[
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt_authorized"
    ] is False


def test_recursive_closure_hook_is_exactly_84_sources() -> None:
    checker = _load("_joint_jepa_v3_scalar_closure_test", CHECKER)
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 84
    assert manifest["source_paths"] == list(checker.contract.SOURCE_PATHS)
    assert set(manifest["source_paths"]) == {
        *checker.contract.REUSED_SOURCE_PATHS,
        *checker.contract.ADDITIVE_SOURCE_PATHS,
    }
    assert manifest["entrypoints"] == list(
        checker.contract.SOURCE_MANIFEST_ENTRYPOINTS
    )
