from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
)
RUNNER = (
    ROOT
    / "scripts/run_go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
)
LAUNCHER = (
    ROOT
    / "scripts/launch_go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement.py"
)
CHECKER = (
    ROOT
    / "scripts/check_go2_direct_egocentric_bev_state_jepa_v7_"
    "runner_integrity_replacement_source_closure.py"
)
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V7_"
    "RUNNER_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
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


def test_contract_normalizes_exactly_to_frozen_v6_and_binds_governance() -> None:
    contract = _load(CONTRACT, "_direct_bev_v7_contract_identity")
    identity = contract.science_identity_receipt()
    assert identity["scientific_delta_count"] == 0
    assert identity["normalized_exactly_equals_frozen_v6"] is True
    assert (
        identity["normalized_v7_science_contract_sha256"]
        == contract.FROZEN_V6_AUTHORIZATION_EXPERIMENT_SHA256
    )
    science = contract.science_contract()
    assert science["model"] == contract.frozen_v6_science_contract()["model"]
    assert science["objective"] == (
        contract.frozen_v6_science_contract()["objective"]
    )
    assert science["optimizer"] == (
        contract.frozen_v6_science_contract()["optimizer"]
    )
    assert science["schedule"] == (
        contract.frozen_v6_science_contract()["schedule"]
    )
    assert science["gates"] == contract.frozen_v6_science_contract()["gates"]
    assert science["lifecycle"]["maximum_updates"] == 1_000
    assert science["lifecycle"]["maximum_presentations"] == 16_000
    assert science["lifecycle"]["maximum_active_gpu_minutes"] == 60
    assert contract.normalize_v7_operational_identity(science) == (
        contract.frozen_v6_science_contract()
    )
    changed = json.loads(json.dumps(science))
    changed["schedule"]["presentations"] += 1
    with pytest.raises(PermissionError, match="differs"):
        contract.normalize_v7_operational_identity(changed)

    governing = contract.validate_governing_documents()
    assert governing[contract.V6_TERMINAL_AUDIT_RELATIVE_PATH] == (
        contract.V6_TERMINAL_AUDIT_FILE_SHA256
    )
    assert governing[contract.PREREGISTRATION_RELATIVE_PATH] == (
        contract.PREREGISTRATION_FILE_SHA256
    )


def test_source_closure_and_review_authorization_validators_are_exact() -> None:
    checker = _load(CHECKER, "_direct_bev_v7_closure_candidate")
    contract = checker.contract
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 116
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert set(contract.ADDITIVE_SOURCE_PATHS) == {
        contract.CONTRACT_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        contract.TEST_RELATIVE_PATH,
    }
    manifest_raw = contract.canonical_json_bytes(manifest) + b"\n"
    assert contract.validate_source_manifest(manifest_raw) == manifest
    manifest_binding = contract.artifact_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    )
    expected_sources = {
        row["path"]: row["file_sha256"]
        for row in manifest["source_bindings"]
    }
    expected_sources[contract.SOURCE_MANIFEST_RELATIVE_PATH] = (
        hashlib.sha256(manifest_raw).hexdigest()
    )
    expected_sources.update(contract.validate_governing_documents())

    review = contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": contract.REVIEW_STATUS,
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": "/root/v7_independent_source_review",
        "reviewed_sources": dict(expected_sources),
        "source_manifest": manifest_binding,
        "frozen_v6_source_manifest": (
            contract.frozen_v6_source_manifest_binding()
        ),
        "frozen_v6_source_review": contract.frozen_v6_review_binding(),
        "frozen_v6_execution_authorization": (
            contract.frozen_v6_authorization_binding()
        ),
        "v6_terminal_audit": contract.v6_terminal_audit_binding(),
        "v7_preregistration": contract.preregistration_binding(),
        "science_contract": contract.science_contract(),
        "science_identity": contract.science_identity_receipt(),
        "source_only_checks": {
            "stdlib_only_contract_import": True,
            "cpu_synthetic_torch_tests_permitted": True,
            "generated_inputs_opened": [],
            "checkpoints_tensors_traces_or_runtime_outputs_opened": [],
            "gpu_state_opened": [],
            "sealed_or_heldout_opened": [],
        },
        "integrity_checks": contract.INTEGRITY_REVIEW_CHECKS,
        "findings": [],
        "authority": contract.REVIEW_AUTHORITY,
    })
    assert contract.validate_review(
        review,
        expected_sources=expected_sources,
        source_manifest_binding=manifest_binding,
    ) == review

    review_raw = contract.canonical_json_bytes(review) + b"\n"
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "authorizer": "/root/v7_execution_authorizer",
        "independent_source_review": review_binding,
        "frozen_v6_source_manifest": (
            contract.frozen_v6_source_manifest_binding()
        ),
        "frozen_v6_source_review": contract.frozen_v6_review_binding(),
        "frozen_v6_execution_authorization": (
            contract.frozen_v6_authorization_binding()
        ),
        "v6_terminal_audit": contract.v6_terminal_audit_binding(),
        "v7_preregistration": contract.preregistration_binding(),
        "runtime_inputs": contract.runtime_authorization_template(),
        "experiment": contract.science_contract(),
        "science_identity": contract.science_identity_receipt(),
        "authority": contract.EXECUTION_AUTHORITY,
    })
    assert contract.validate_authorization(
        authorization,
        review_binding=review_binding,
        reviewer=review["reviewer"],
    ) == authorization
    changed = dict(authorization)
    changed["authority"] = dict(changed["authority"])
    changed["authority"]["g2_authorized"] = True
    with pytest.raises(PermissionError, match="authorization changed"):
        contract.validate_authorization(
            changed,
            review_binding=review_binding,
            reviewer=review["reviewer"],
        )


def test_runner_isolated_import_and_public_parse_preserve_all_v6_seams() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('_v7_runner_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
module._assert_full_v6_seam_table()
args = module.parse_args([
    '--run', '--review-sha256', '0' * 64,
    '--authorization-sha256', '1' * 64,
])
assert args.review_sha256 == '0' * 64
assert args.authorization_sha256 == '1' * 64
module._assert_full_v6_seam_table()
assert all(
    getattr(module._LEAF, leaf) is getattr(module._V6, owner)
    for leaf, owner in module._V6_SEAM_TABLE
)
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


def test_runner_delegates_directly_and_full_main_handoff_avoids_wrappers(
    monkeypatch,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v7_direct_delegation")
    for intermediate in (runner._V6, runner._V6._V5):
        monkeypatch.setattr(intermediate, "parse_args", _bomb)
        monkeypatch.setattr(intermediate, "run_parent", _bomb)
        monkeypatch.setattr(intermediate, "main", _bomb)

    calls: list[tuple[str, Any]] = []

    def leaf_parse(argv=None):
        calls.append(("parse", argv))
        return SimpleNamespace(
            review_sha256="2" * 64,
            authorization_sha256="3" * 64,
        )

    def leaf_run_parent(*, review_file_sha256, authorization_file_sha256):
        calls.append((
            "run_parent",
            (review_file_sha256, authorization_file_sha256),
        ))
        runner._assert_full_v6_seam_table()
        return 47

    monkeypatch.setattr(runner._LEAF, "parse_args", leaf_parse)
    monkeypatch.setattr(runner._LEAF, "run_parent", leaf_run_parent)
    argv = ["--synthetic-main-handoff"]
    assert runner.main(argv) == 47
    assert calls == [
        ("parse", argv),
        ("run_parent", ("2" * 64, "3" * 64)),
    ]


def test_initializer_to_optimizer_v6_model_witness_survives_public_parse(
    monkeypatch,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v7_model_witness")

    class Parameter:
        def __init__(self, requires_grad: bool) -> None:
            self.requires_grad = requires_grad

    class Model:
        def __init__(self) -> None:
            self.active_phase_v6 = "unarmed"
            self._v6_optimizer_for_integrity_probe = None

        def arm_phase_schedule_v6(self) -> None:
            self.active_phase_v6 = "phase_one"

    model = Model()
    partition = {
        "groups": {
            "encoder": [("encoder", Parameter(True))],
            "decoder_state": [("decoder", Parameter(True))],
            "predictor": [("predictor", Parameter(False))],
            "detached_target_encoder_decoder_state": [
                ("target", Parameter(False))
            ],
        }
    }
    initial_receipt = {
        "complete_initial_state_sha256": (
            runner.contract.FROZEN_V3_INITIAL_MODEL_STATE_SHA256
        ),
        "prior_runtime_parameter_reuse_count": 0,
    }
    optimizer = object()

    monkeypatch.setattr(
        runner._V6,
        "_FROZEN_INITIALIZE_MODEL",
        lambda runtime, model_api, fit, device: (
            model,
            partition,
            dict(initial_receipt),
        ),
    )
    monkeypatch.setattr(
        runner._V6,
        "_normalized_state_sha256",
        lambda runtime, candidate, prefixes: (
            "a" * 64
            if prefixes != runner._V6._PREDICTOR_PREFIXES
            else "b" * 64
        ),
    )

    def frozen_build(runtime, observed_partition):
        assert observed_partition["_v6_model"] is model
        return optimizer, {"schema": "synthetic_optimizer_receipt"}

    monkeypatch.setattr(runner._V6, "_FROZEN_BUILD_OPTIMIZER", frozen_build)

    runner.parse_args([
        "--run",
        "--review-sha256", "4" * 64,
        "--authorization-sha256", "5" * 64,
    ])
    initialized, observed_partition, receipt = runner._LEAF._initialize_model(
        object(), object(), object(), object()
    )
    built, optimizer_receipt = runner._LEAF._build_optimizer(
        object(), observed_partition
    )
    assert initialized is model
    assert observed_partition["_v6_model"] is model
    assert receipt["v6_phase_policy_armed_after_frozen_initialization"] is True
    assert built is optimizer
    assert model._v6_optimizer_for_integrity_probe is optimizer
    assert optimizer_receipt["single_optimizer_constructed_once"] is True


def test_launcher_delegates_directly_to_leaf_and_rebinds_fresh_runner_path(
    monkeypatch,
) -> None:
    launcher = _load(LAUNCHER, "_direct_bev_v7_launcher_delegation")
    for intermediate in (launcher._V6, launcher._V6._V5):
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
        lambda argv=None: calls.append(("main", argv)) or 53,
    )
    arguments = [
        "--review-sha256", "6" * 64,
        "--authorization-sha256", "7" * 64,
    ]
    launcher.parse_args(arguments)
    assert launcher.main(arguments) == 53
    assert calls == [("parse", arguments), ("main", arguments)]
    assert launcher.contract.PREFLIGHT_ENVIRONMENT_KEY == PREFLIGHT_KEY
    assert launcher._LEAF._V11._BASE.RUNNER_PATH == RUNNER
