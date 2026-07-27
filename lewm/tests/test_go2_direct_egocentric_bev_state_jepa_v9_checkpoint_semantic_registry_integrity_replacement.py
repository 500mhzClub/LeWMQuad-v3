from __future__ import annotations

import copy
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
STEM = (
    "go2_direct_egocentric_bev_state_jepa_v9_"
    "checkpoint_semantic_registry_integrity_replacement"
)
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V9_"
    "CHECKPOINT_SEMANTIC_REGISTRY_INTEGRITY_REPLACEMENT_PREFLIGHT_JSON"
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


def _checkpoint_metadata(runner: Any, update: int) -> dict[str, Any]:
    accounting = runner.contract.perception_accounting(update)
    return {
        "gate": {"passed": True, "control": f"synthetic_{update}"},
        "metrics": {"synthetic": float(update)},
        "optimizer_updates": update,
        "presentations": accounting["presentations"],
        "ema_updates": update,
    }


def test_contract_normalizes_exactly_to_frozen_v8_science() -> None:
    contract = _load(CONTRACT, "_direct_bev_v9_contract_identity")
    expected_science_sha256 = (
        "bacb31b0eb2070821bbd37862e6f3b9a39d7ecb0ab14ed8d758894c36f06f728"
    )
    science = contract.science_contract()
    normalized = contract.normalize_v9_operational_identity(science)
    identity = contract.science_identity_receipt()
    assert normalized == contract.frozen_v8_science_contract()
    assert contract.canonical_json_sha256(science) == expected_science_sha256
    assert contract.canonical_json_sha256(normalized) == expected_science_sha256
    assert identity["frozen_v8_science_contract_sha256"] == expected_science_sha256
    assert (
        identity["normalized_v9_science_contract_sha256"]
        == expected_science_sha256
    )
    assert identity["normalized_exactly_equals_frozen_v8"] is True
    assert identity["scientific_delta_count"] == 0
    lifecycle = science["lifecycle"]
    assert lifecycle["maximum_attempts"] == 1
    assert lifecycle["maximum_updates"] == 250
    assert lifecycle["maximum_presentations"] == 4_000
    assert lifecycle["gpu_active_minutes_maximum"] == 30
    expected_v9_root = (
        ".generated/go2_shared_observable_camera_ray_jepa_v9/"
        "rgb_direct_egocentric_bev_state_jepa_probe_v9_"
        "checkpoint_semantic_registry_integrity_replacement_v1"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == expected_v9_root
    assert contract.runtime_authorization_template()["experiment_scope"][
        "output_root"
    ] == expected_v9_root
    assert contract.EXECUTION_AUTHORITY["output_root"] == expected_v9_root
    assert lifecycle["predictor_phase_or_update"] is False
    assert science["authority"][
        "predictor_training_or_evaluation_authorized"
    ] is False

    changed = copy.deepcopy(science)
    changed["lifecycle"]["maximum_presentations"] += 1
    with pytest.raises(PermissionError, match="differs"):
        contract.normalize_v9_operational_identity(changed)


def test_candidate_manifest_review_and_authorization_validate_exactly() -> None:
    checker = _load(CHECKER, "_direct_bev_v9_authority_candidate")
    contract = checker.contract
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 127
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
        "reviewer": "/root/v9_independent_source_review",
        "reviewed_sources": dict(expected_sources),
        "source_manifest": manifest_binding,
        "frozen_v8_source_manifest": contract.frozen_v8_source_manifest_binding(),
        "frozen_v8_source_review": contract.frozen_v8_review_binding(),
        "frozen_v8_execution_authorization": (
            contract.frozen_v8_authorization_binding()
        ),
        "v8_terminal_audit": contract.v8_terminal_audit_binding(),
        "v9_preregistration": contract.preregistration_binding(),
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
        "authorizer": "/root/v9_execution_authorizer",
        "independent_source_review": review_binding,
        "frozen_v8_source_manifest": contract.frozen_v8_source_manifest_binding(),
        "frozen_v8_source_review": contract.frozen_v8_review_binding(),
        "frozen_v8_execution_authorization": (
            contract.frozen_v8_authorization_binding()
        ),
        "v8_terminal_audit": contract.v8_terminal_audit_binding(),
        "v9_preregistration": contract.preregistration_binding(),
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
    changed = json.loads(json.dumps(authorization))
    changed["authority"]["g2_authorized"] = True
    with pytest.raises(PermissionError, match="authorization changed"):
        contract.validate_authorization(
            changed,
            review_binding=review_binding,
            reviewer=review["reviewer"],
        )


def test_runner_import_is_source_only_and_installs_only_three_v9_seams() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('_v9_runner_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
module._assert_v9_seams()
assert module._LEAF._BASE.contract is module._LEAF._LEGACY_CONTRACT
assert module._V9_SEAM_NAMES == {{
    '_evaluate_observation_impl', '_snapshot_model', '_terminal_failure'
}}
v9 = dict(module._V9_SEAM_TABLE)
for name, expected_v8 in module._V8._V8_SEAM_TABLE:
    assert getattr(module._LEAF, name) is v9.get(name, expected_v8)
args = module.parse_args([
    '--run', '--review-sha256', '0' * 64,
    '--authorization-sha256', '1' * 64,
])
assert args.review_sha256 == '0' * 64
assert args.authorization_sha256 == '1' * 64
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


def test_snapshot_registry_accepts_50_100_250_and_restores_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v9_checkpoint_success")
    torch = pytest.importorskip("torch")
    assert tuple(runner.contract.CHECKPOINT_UPDATES) == (50, 100, 250)
    legacy = runner._LEAF._LEGACY_CONTRACT

    monkeypatch.setattr(
        runner._V8,
        "_v8_perception_accounting",
        lambda model, *, update: runner.contract.perception_accounting(update),
    )

    class SyntheticModelModule:
        @staticmethod
        def tensor_state_dict_sha256(state: dict[str, Any]) -> str:
            assert set(state) == {"synthetic.weight"}
            return "a" * 64

    class SyntheticModel:
        @staticmethod
        def state_dict() -> dict[str, Any]:
            return {"synthetic.weight": torch.tensor([1.0])}

    runtime = SimpleNamespace(torch=torch, model_module=SyntheticModelModule())
    runner._LEAF._BASE._reset_output_binding_registry(tmp_path)
    for update in (50, 100, 250):
        result = runner._v9_snapshot_model(
            runtime,
            SyntheticModel(),
            tmp_path,
            update=update,
            metadata=_checkpoint_metadata(runner, update),
        )
        assert result["update"] == update
        assert result["schedule_prefix_sha256"] == (
            runner.contract.SCHEDULE_PREFIX_SHA256[update]
        )
        assert result["path"] == f"checkpoints/update_{update}.pt"
        assert result["write_only"] is True
        assert result["read_count_after_write"] == 0
        assert runner._LEAF._BASE.contract is legacy
        assert runner._V9_SNAPSHOT_ATTEMPT_RECEIPT is None


@pytest.mark.parametrize("update", [50, 100, 250])
def test_snapshot_registry_restores_exact_object_after_baseexception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    update: int,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v9_checkpoint_baseexception")
    legacy = runner._LEAF._LEGACY_CONTRACT

    class SyntheticAbort(BaseException):
        pass

    monkeypatch.setattr(
        runner._V8,
        "_v8_perception_accounting",
        lambda model, *, update: runner.contract.perception_accounting(update),
    )

    def abort(*args: Any, **kwargs: Any) -> Any:
        assert runner._LEAF._BASE.contract is runner.contract
        raise SyntheticAbort("synthetic snapshot abort")

    monkeypatch.setattr(runner._V8._V6, "_FROZEN_SNAPSHOT_MODEL", abort)
    with pytest.raises(SyntheticAbort, match="snapshot abort"):
        runner._v9_snapshot_model(
            object(),
            object(),
            tmp_path,
            update=update,
            metadata=_checkpoint_metadata(runner, update),
        )
    assert runner._LEAF._BASE.contract is legacy
    marker = runner._V9_SNAPSHOT_ATTEMPT_RECEIPT
    assert marker is not None
    assert marker["stage"] == f"checkpoint_snapshot_update_{update}"
    assert marker["update"] == update
    assert marker["entered"] is True
    assert marker["completed"] is False
    assert marker["raised"] is True
    assert marker[
        "registry_contract_rebound_to_active_v9_science_identical_contract"
    ] is True
    assert marker["exact_prior_contract_restored"] is True
    assert marker["error"]["type"] == "SyntheticAbort"
    assert runner.contract.is_sha256(marker["error"]["message_sha256"])


def test_observation_capture_is_ordered_deep_copied_and_return_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v9_observation_capture")
    source_results = [
        {
            "schema": "synthetic_observation",
            "update": update,
            "gate": {"passed": True, "control": f"continue_{update}"},
            "metrics": {"score": update / 100.0, "nested": [update]},
            "untouched_scientific_field": {"value": update + 1},
        }
        for update in (0, 50)
    ]
    calls = iter(source_results)
    monkeypatch.setattr(
        runner,
        "_FROZEN_V8_EVALUATE_OBSERVATION_IMPL",
        lambda *args, **kwargs: next(calls),
    )

    observed: list[dict[str, Any]] = []
    loader = SimpleNamespace(progress={})
    for update, expected in zip((0, 50), source_results, strict=True):
        before = copy.deepcopy(expected)
        result = runner._v9_evaluate_observation_impl(
            SimpleNamespace(),
            object(),
            object(),
            {},
            loader,
            [],
            {},
            object(),
            update=update,
            update_zero=None,
            prior_gates_passed=True,
        )
        assert result is expected
        assert result == before
        observed.append(before)

    receipts = runner._V9_COMPLETED_OBSERVATION_RECEIPTS
    assert [item["update"] for item in receipts] == [0, 50]
    for source, receipt in zip(observed, receipts, strict=True):
        core = {
            "update": source["update"],
            "gate": source["gate"],
            "metrics": source["metrics"],
        }
        assert receipt == {
            **core,
            "canonical_sha256": runner.contract.canonical_json_sha256(core),
        }
        assert runner.contract.canonical_json_bytes({
            name: receipt[name] for name in ("update", "gate", "metrics")
        }) == runner.contract.canonical_json_bytes(core)

    source_results[0]["metrics"]["nested"].append(999)
    assert receipts[0]["metrics"]["nested"] == [0]
    assert loader.progress["completed_observation_receipts"] == receipts
    assert loader.progress["completed_observation_receipt_bindings"] == [
        {
            "update": item["update"],
            "canonical_sha256": item["canonical_sha256"],
        }
        for item in receipts
    ]


def test_terminal_failure_includes_completed_observation_receipts_and_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v9_failure_observations")
    cores = [
        {
            "update": update,
            "gate": {"passed": True, "control": f"continue_{update}"},
            "metrics": {"score": float(update)},
        }
        for update in (0, 50)
    ]
    runner._V9_COMPLETED_OBSERVATION_RECEIPTS.extend([
        {
            **copy.deepcopy(core),
            "canonical_sha256": runner.contract.canonical_json_sha256(core),
        }
        for core in cores
    ])
    runner._V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES.extend([
        {
            "update": core["update"],
            "state_after_completed_observation": None,
            "strict_determinism_exact": False,
        }
        for core in cores
    ])
    captured: dict[str, Any] = {}

    def publish(
        output_root: Path,
        reservation: dict[str, Any],
        reservation_raw: bytes,
        *,
        error: BaseException,
        progress: dict[str, Any],
    ) -> None:
        captured.update({
            "output_root": output_root,
            "reservation": reservation,
            "reservation_raw": reservation_raw,
            "error": error,
            "progress": progress,
        })

    monkeypatch.setattr(runner, "_FROZEN_V8_TERMINAL_FAILURE", publish)
    runner._V9_SNAPSHOT_ATTEMPT_RECEIPT = {
        "stage": "checkpoint_snapshot_update_50",
        "update": 50,
        "entered": True,
        "completed": False,
        "raised": True,
        "registry_contract_rebound_to_active_v9_science_identical_contract": (
            True
        ),
        "exact_prior_contract_restored": True,
        "error": {
            "module": "builtins",
            "type": "RuntimeError",
            "message": "RuntimeError: synthetic failure",
            "message_sha256": "b" * 64,
        },
    }
    # This is the real inherited stage during a snapshot call; V9's marker
    # provides the missing exact snapshot stage without rewriting it.
    progress = {
        "stage": "observation_update_50",
        "registered_observations": 2,
        "completed_observation_receipts": copy.deepcopy(
            runner._V9_COMPLETED_OBSERVATION_RECEIPTS
        ),
        "completed_observation_receipt_bindings": [
            {
                "update": item["update"],
                "canonical_sha256": item["canonical_sha256"],
            }
            for item in runner._V9_COMPLETED_OBSERVATION_RECEIPTS
        ],
        "completed_observation_determinism_witnesses": copy.deepcopy(
            runner._V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES
        ),
    }
    original = copy.deepcopy(progress)
    error = RuntimeError("synthetic failure")
    reservation = {
        "attempt_identity": "a" * 64,
        "independent_source_review": {"path": "review.json"},
        "execution_authorization": {"path": "authorization.json"},
        "reviewed_sources": {"synthetic.py": "c" * 64},
        "science_contract": runner.contract.science_contract(),
        "authority": dict(runner.contract.DOWNSTREAM_DENIALS),
    }
    runner._v9_terminal_failure(
        tmp_path,
        reservation,
        b"reservation",
        error=error,
        progress=progress,
    )
    assert progress == original
    assert captured["error"] is error
    assert captured["progress"]["stage"] == "checkpoint_snapshot_update_50"
    assert captured["progress"][
        "inherited_progress_stage_before_v9_snapshot_failure_marker"
    ] == "observation_update_50"
    receipts = captured["progress"]["completed_observation_receipts"]
    bindings = captured["progress"]["completed_observation_receipt_bindings"]
    assert [item["update"] for item in receipts] == [0, 50]
    assert bindings == [
        {
            "update": item["update"],
            "canonical_sha256": item["canonical_sha256"],
        }
        for item in receipts
    ]
    snapshot = captured["progress"]["checkpoint_snapshot_failure_receipt"]
    assert snapshot["stage"] == "checkpoint_snapshot_update_50"
    assert snapshot["exact_prior_contract_restored"] is True
    assert snapshot[
        "lexical_base_contract_restored_before_terminal_publication"
    ] is True
    context = captured["progress"]["v9_failure_context_binding"]
    assert context["attempt_identity"] == reservation["attempt_identity"]
    assert context["science_contract_sha256"] == (
        runner.contract.canonical_json_sha256(reservation["science_contract"])
    )
    assert set(context["inherited_input_and_access_evidence_pointers"]) == {
        "loader_access",
        "raw_constructor_reads",
        "consumed_inputs",
    }


def test_nonmechanical_v8_seams_and_direct_entrypoint_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER, "_direct_bev_v9_direct_delegation")
    wrapped = dict(runner._V9_SEAM_TABLE)
    for name, expected_v8 in runner._V8._V8_SEAM_TABLE:
        assert getattr(runner._LEAF, name) is wrapped.get(name, expected_v8)

    for intermediate in (runner._V8, runner._V8._V7, runner._V8._V6):
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
        lambda **kwargs: calls.append(("run_parent", kwargs)) or 41,
    )
    monkeypatch.setattr(
        runner._LEAF,
        "main",
        lambda argv=None: calls.append(("main", argv)) or 43,
    )
    assert isinstance(runner.parse_args(["synthetic"]), SimpleNamespace)
    assert runner.run_parent(
        review_file_sha256="2" * 64,
        authorization_file_sha256="3" * 64,
    ) == 41
    assert runner.main(["synthetic-main"]) == 43
    assert calls == [
        ("parse", ["synthetic"]),
        (
            "run_parent",
            {
                "review_file_sha256": "2" * 64,
                "authorization_file_sha256": "3" * 64,
            },
        ),
        ("main", ["synthetic-main"]),
    ]


def test_fresh_process_guard_rejects_every_stale_receipt_state() -> None:
    runner = _load(RUNNER, "_direct_bev_v9_fresh_process_guard")
    runner._assert_fresh_attempt_receipts()
    runner._V9_COMPLETED_OBSERVATION_RECEIPTS.append({"stale": True})
    with pytest.raises(RuntimeError, match="fresh process"):
        runner._assert_fresh_attempt_receipts()
    runner._V9_COMPLETED_OBSERVATION_RECEIPTS.clear()
    runner._V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES.append({
        "stale": True
    })
    with pytest.raises(RuntimeError, match="fresh process"):
        runner._assert_fresh_attempt_receipts()
    runner._V9_COMPLETED_OBSERVATION_DETERMINISM_WITNESSES.clear()
    runner._V9_SNAPSHOT_ATTEMPT_RECEIPT = {"stale": True}
    with pytest.raises(RuntimeError, match="fresh process"):
        runner._assert_fresh_attempt_receipts()


def test_launcher_rebinds_v9_authority_stack_and_delegates_to_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load(LAUNCHER, "_direct_bev_v9_launcher_delegation")
    for intermediate in (launcher._V8, launcher._V8._V7, launcher._V8._V6):
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
        lambda argv=None: calls.append(("main", argv)) or 47,
    )
    arguments = ["--review-sha256", "4" * 64]
    launcher.parse_args(arguments)
    assert launcher.main(arguments) == 47
    assert calls == [("parse", arguments), ("main", arguments)]
    assert launcher.contract.PREFLIGHT_ENVIRONMENT_KEY == PREFLIGHT_KEY
    assert launcher._LEAF._V11._BASE.RUNNER_PATH == RUNNER


def test_launcher_isolated_import_remains_no_tensor_source_only() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(LAUNCHER)!r})
spec = importlib.util.spec_from_file_location('_v9_launcher_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
assert module.contract.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert module._LEAF._V11._BASE.RUNNER_PATH == Path({str(RUNNER)!r})
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


def test_source_closure_is_exactly_five_additive_122_reused_127_total() -> None:
    checker = _load(CHECKER, "_direct_bev_v9_source_closure")
    contract = checker.contract
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 5
    assert len(contract.REUSED_SOURCE_PATHS) == 122
    assert len(contract.SOURCE_PATHS) == 127
    assert set(contract.ADDITIVE_SOURCE_PATHS) == {
        contract.CONTRACT_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        contract.TEST_RELATIVE_PATH,
    }
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 127
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
