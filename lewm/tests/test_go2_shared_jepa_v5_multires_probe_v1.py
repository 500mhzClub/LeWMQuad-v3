from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "_test_go2_multires_probe_contract",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py",
)
runner = _load(
    "_test_go2_multires_probe_runner",
    "scripts/run_go2_shared_jepa_v5_multires_probe_v1.py",
)
launcher = _load(
    "_test_go2_multires_probe_launcher",
    "scripts/launch_go2_shared_jepa_v5_multires_probe_v1.py",
)


def _evaluation(
    *,
    complete: int = 1,
    passed: int = 98,
    shortfall: float = 41.0,
    pixel: float = 0.82,
    ground: float = 0.648,
    depth: float = 0.977,
) -> dict[str, Any]:
    return {
        "complete_physical_scope_count": complete,
        "margin_count": 189,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": pixel,
            "ground_balanced_accuracy": ground,
            "depth_p95_m": depth,
        },
    }


def _physical_metrics() -> dict[str, Any]:
    return {
        **{name: threshold + 0.01
           for name, threshold in contract.PHYSICAL_LOWER_THRESHOLDS.items()},
        **{name: threshold - 0.01
           for name, threshold in contract.PHYSICAL_UPPER_THRESHOLDS.items()},
        "distance_group_balanced_accuracy": [0.93] * 5,
        "present_class_recall": {
            "free": 0.96,
            "occupied": 0.96,
            "unknown": 0.96,
            "outside": 0.96,
        },
    }


def _binding(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": contract.RUNTIME_FILE_SHA256[path],
        "content_sha256": contract.RUNTIME_CONTENT_SHA256[path],
        "byte_count": contract.RUNTIME_BYTE_COUNTS.get(path, 123),
    }


def _runtime_inputs() -> dict[str, Any]:
    return {
        "raw": {
            "root": contract.RAW_ROOT_RELATIVE_PATH,
            "manifest": _binding(contract.RAW_MANIFEST_RELATIVE_PATH),
            "audit": _binding(contract.RAW_AUDIT_RELATIVE_PATH),
            "role_counts": {
                "train": contract.TRAIN_ROLE_COUNTS,
                "checkpoint_selection": contract.SELECTION_ROLE_COUNTS,
            },
            "grant": {
                "allowed_roles": ["train", "checkpoint_selection"],
                "allowed_operations": [
                    "development_rgb_decode",
                    "multires_perception_training",
                    "physical_checkpoint_selection",
                ],
                "calibration_g2_navigation_heldout_or_production_use": False,
            },
        },
        "camera": {
            "root": contract.N320_ROOT_RELATIVE_PATH,
            "gate": _binding(contract.N320_GATE_RELATIVE_PATH),
            "checkpoint": _binding(contract.N320_CHECKPOINT_RELATIVE_PATH),
            "seed": 20260710,
            "fit_size": 320,
            "updates": 40_000,
            "gate_must_pass_all_checks": 26,
        },
        "schedule": _binding(contract.SCHEDULE_RELATIVE_PATH),
    }


def test_source_imports_do_not_import_torch_or_open_payloads() -> None:
    script = f"""
import importlib.util
import json
from pathlib import Path
import sys

root = Path({str(ROOT)!r})
opened = []
def audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        opened.append(str(args[0]))
sys.addaudithook(audit)
for index, relative in enumerate((
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py",
    "scripts/run_go2_shared_jepa_v5_multires_probe_v1.py",
    "scripts/launch_go2_shared_jepa_v5_multires_probe_v1.py",
)):
    path = root / relative
    spec = importlib.util.spec_from_file_location(f"_probe_import_{{index}}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
forbidden = [
    item for item in opened
    if item.startswith(str(root))
    and (
        "/.generated/" in item
        or "/config/" in item
        or "/configs/" in item
        or "/custody/" in item
        or "/data/" in item
        or "/datasets/" in item
        or "/checkpoints/" in item
        or "/sealed/" in item
        or "/sealed_" in item
        or item.endswith("/sealed_test.json")
    )
]
print(json.dumps({{"torch": "torch" in sys.modules, "forbidden": forbidden}}))
"""
    environment = dict(os.environ)
    for name in (
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed == {"torch": False, "forbidden": []}


def test_contract_binds_prereg_sources_and_deferred_runtime() -> None:
    assert contract.PREREGISTRATION_COMMIT == (
        "e3cc8f793758145a5865f27218250a7baaef35b8"
    )
    assert contract.PREREGISTRATION_FILE_SHA256 == (
        "13ee847c4651be12e46888fbac3db9d3ec86f57c4ccc0f2006784885c879ebd9"
    )
    assert contract.PREREGISTRATION_CONTENT_SHA256 == (
        "6fdf626175dbd262696546ac7f38267d54999cc3c0ea988edaeec06d9f4ad6c8"
    )
    assert contract.FROZEN_SOURCE_SHA256
    assert contract.validate_runtime_inputs(_runtime_inputs()) == _runtime_inputs()
    assert contract.lifecycle_contract()["source_review_may_open_generated_inputs"] is False
    assert all(value is False for value in contract.DOWNSTREAM_DENIALS.values())
    expected_direct_source_paths = (
        "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py",
        "scripts/run_go2_shared_jepa_v5_multires_probe_v1.py",
        "scripts/launch_go2_shared_jepa_v5_multires_probe_v1.py",
        "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v1.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py",
        "docs/lewm_go2_rgb_multiresolution_perception_v1_"
        "preregistration_2026-07-24.json",
        "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
        "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_"
        "protected_camera_adaptation_v4_tail_depth.py",
        "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
        "lewm/models/observable_camera_ray_evidence_v4.py",
        "lewm/models/observable_camera_ray_evidence_v4_training.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_"
        "full_training_v4_loss.py",
        "lewm/models/encoders.py",
        "lewm/models/egomotion_bev_jepa.py",
        "lewm/models/observable_camera_ray_evidence_v4_"
        "hierarchical_first_hit_v9.py",
        "lewm/models/observable_camera_ray_evidence_v4_"
        "gate_aligned_raster_nll_v12.py",
        "lewm/tests/test_shared_observable_camera_ray_jepa_v5_multires_v1.py",
    )
    expected_runtime_source_paths = (
        "lewm/__init__.py",
        "lewm/benchmarks/__init__.py",
        "lewm/benchmarks/counterfactual.py",
        "lewm/benchmarks/finalize_shared_observable_camera_ray_jepa_v5_g2.py",
        "lewm/benchmarks/finalize_shared_observable_camera_ray_jepa_v5_g3.py",
        "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
        "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
        "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py",
        "lewm/benchmarks/shared_observable_camera_ray_jepa_v5_"
        "finalizer_core.py",
        "lewm/benchmarks/shared_observable_camera_ray_jepa_v5_runner_policy.py",
        "lewm/models/__init__.py",
        "lewm/models/egomotion_bev_jepa.py",
        "lewm/models/encoders.py",
        "lewm/models/lewm.py",
        "lewm/models/observable_camera_ray_evidence_v4.py",
        "lewm/models/observable_camera_ray_evidence_v4_"
        "gate_aligned_raster_nll_v12.py",
        "lewm/models/observable_camera_ray_evidence_v4_"
        "hierarchical_first_hit_v9.py",
        "lewm/models/observable_camera_ray_evidence_v4_training.py",
        "lewm/models/phase2d_spatial_lewm.py",
        "lewm/models/predictor.py",
        "lewm/models/primitive_affordance.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_authority.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_"
        "full_training_v4_loss.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_"
        "protected_camera_adaptation_v4_tail_depth.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_registry_policy.py",
        "lewm/models/sigreg.py",
        "lewm/models/source_action_utility.py",
        "lewm/models/spatial_lewm.py",
        "lewm/models/spatial_predictor.py",
        "scripts/launch_go2_shared_jepa_v5_multires_probe_v1.py",
        "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
        "scripts/run_go2_shared_jepa_v5_multires_probe_v1.py",
    )
    manifest = contract.validate_source_manifest(
        (ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH).read_bytes()
    )
    assert tuple(manifest["source_paths"]) == expected_runtime_source_paths
    sources = contract.current_source_bindings(ROOT)
    assert contract.SOURCE_PATHS == expected_direct_source_paths
    expected_review_sources = tuple(dict.fromkeys((
        *expected_runtime_source_paths,
        *contract.SOURCE_REVIEW_ADDITIONAL_PATHS,
    )))
    assert tuple(sources) == expected_review_sources
    assert len(sources) == 41
    assert sources[contract.MODEL_RELATIVE_PATH] == contract.MODEL_FILE_SHA256


def test_recursive_local_import_closure_is_inside_exact_source_set() -> None:
    manifest = contract.validate_source_manifest(
        (ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH).read_bytes()
    )
    expected = set(manifest["source_paths"])
    implementation = [
        relative
        for relative in manifest["source_paths"]
        if relative.endswith(".py")
        and "/tests/" not in relative
    ]

    def local_module_path(parts: list[str]) -> str | None:
        if not parts or parts[0] != "lewm":
            return None
        candidate = "/".join(parts) + ".py"
        return candidate if (ROOT / candidate).is_file() else None

    observed: set[str] = set()
    for relative in implementation:
        tree = ast.parse((ROOT / relative).read_text("utf-8"), filename=relative)
        module_parts = list(Path(relative).with_suffix("").parts)
        package_parts = module_parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = local_module_path(alias.name.split("."))
                    if resolved is not None:
                        observed.add(resolved)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    remove = node.level - 1
                    base = (
                        package_parts[: len(package_parts) - remove]
                        if remove
                        else list(package_parts)
                    )
                else:
                    base = []
                if node.module:
                    base.extend(node.module.split("."))
                    resolved = local_module_path(base)
                    if resolved is not None:
                        observed.add(resolved)
                for alias in node.names:
                    resolved = local_module_path([*base, alias.name])
                    if resolved is not None:
                        observed.add(resolved)
    missing = sorted(observed - expected)
    assert not missing, missing


def test_exact_schedule_and_operation_caps() -> None:
    assert contract.CHECKPOINT_UPDATES == (100, 400, 1_000)
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert contract.validate_checkpoint_prefix((100, 400, 1_000)) == (
        100,
        400,
        1_000,
    )
    counts = contract.operation_counts(1_000, (100, 400, 1_000))
    assert counts == {
        "maximum_optimizer_updates": 1_000,
        "complete_optimizer_updates": 1_000,
        "maximum_pair_index_presentations": 16_000,
        "pair_index_presentations": 16_000,
        "microbatch_size": 4,
        "microbatches_per_update": 4,
        "camera_objective_count": 4_000,
        "backward_call_count": 4_000,
        "head_clip_invocation_count": 1_000,
        "encoder_clip_invocation_count": 1_000,
        "global_clip_invocation_count": 0,
        "optimizer_construction_count": 1,
        "checkpoint_selection_evaluation_count": 3,
        "checkpoint_selection_evaluation_updates": [100, 400, 1_000],
        "observer_evaluation_rerun_count": 0,
        "jepa_objective_count": 0,
        "jepa_backward_count": 0,
        "ema_update_count_after_initial_hard_sync": 0,
    }
    with pytest.raises(ValueError):
        contract.operation_counts(1_000, (100, 1_000))


def test_physical_evaluator_retains_nine_scopes_and_189_margins() -> None:
    scopes = {scope: _physical_metrics() for scope in contract.SCOPES}
    observed = contract.evaluate_physical_scopes(scopes)
    assert observed["complete_physical_scope_count"] == 9
    assert observed["margin_count"] == 189
    assert observed["passed_margin_count"] == 189
    assert observed["total_shortfall"] == 0.0
    assert len(observed["scope_evaluations"]) == 9


def test_updates_100_and_400_are_integrity_only() -> None:
    terrible = _evaluation(
        complete=0,
        passed=0,
        shortfall=10_000.0,
        pixel=0.0,
        ground=0.0,
        depth=100.0,
    )
    for update in (100, 400):
        decision = contract.checkpoint_control_decision(
            update=update,
            evaluation=terrible,
            integrity_pass=True,
        )
        assert decision["action"] == contract.CONTROL_CONTINUE
        assert decision["informational_only"] is True
        assert decision["terminal"] is False
    integrity = contract.checkpoint_control_decision(
        update=100,
        evaluation=terrible,
        integrity_pass=False,
    )
    assert integrity["action"] == contract.CONTROL_INTEGRITY_FAIL
    assert integrity["terminal"] is True


def test_update_1000_requires_every_strict_conjunct() -> None:
    passed = contract.checkpoint_control_decision(
        update=1_000,
        evaluation=_evaluation(),
        integrity_pass=True,
    )
    assert passed["action"] == contract.CONTROL_PASS
    assert all(passed["conjuncts"].values())

    equality_cases = (
        {"complete": 0},
        {"passed": 97},
        {"shortfall": 41.01776266878769},
        {"pixel": 0.8198594673963917},
        {"ground": 0.647134926562893},
        {"depth": 0.9777327477931971},
    )
    for changed in equality_cases:
        decision = contract.checkpoint_control_decision(
            update=1_000,
            evaluation=_evaluation(**changed),
            integrity_pass=True,
        )
        assert decision["action"] == contract.CONTROL_FAIL
        assert decision["qualifies_probe"] is False
        assert decision["retry_authorized"] is False


def test_review_and_authorization_are_exact_and_independent() -> None:
    sources = {"source.py": "0" * 64}
    review_core = {
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS_SOURCE_ONLY",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": "/root/independent_probe_reviewer",
        "reviewed_sources": sources,
        "preregistration": contract.preregistration_binding(),
        "frozen_source_bindings": {
            **contract.FROZEN_SOURCE_SHA256,
            contract.MODEL_RELATIVE_PATH: contract.MODEL_FILE_SHA256,
        },
        "science_contract": contract.science_contract(),
        "lifecycle_contract": contract.lifecycle_contract(),
        "source_only": True,
        "deferred_runtime_inputs_opened": [],
        "findings": [],
        "authority": contract.REVIEW_AUTHORITY,
    }
    review = contract.with_content_sha256(review_core)
    assert contract.validate_review(review, expected_sources=sources) == review
    bad_review = {**review, "source_only": False}
    with pytest.raises(PermissionError):
        contract.validate_review(bad_review, expected_sources=sources)

    review_raw = contract.canonical_json_bytes(review) + b"\n"
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization_core = {
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED_ONE_EXACT_BOUNDED_PROBE",
        "authorizer": "/root/independent_probe_authorizer",
        "independent_source_review": review_binding,
        "preregistration": contract.preregistration_binding(),
        "runtime_inputs": _runtime_inputs(),
        "hardware": contract.hardware_contract(),
        "experiment": contract.science_contract(),
        "lifecycle": contract.lifecycle_contract(),
        "authority": contract.EXECUTION_AUTHORITY,
    }
    authorization = contract.with_content_sha256(authorization_core)
    assert contract.validate_authorization(
        authorization,
        review_binding=review_binding,
        reviewer=review["reviewer"],
    ) == authorization
    same_person = {
        **authorization_core,
        "authorizer": review["reviewer"],
    }
    with pytest.raises(PermissionError):
        contract.validate_authorization(
            contract.with_content_sha256(same_person),
            review_binding=review_binding,
            reviewer=review["reviewer"],
        )


def test_runner_requires_complete_production_migration_receipt() -> None:
    class Module:
        def __init__(self, names: list[str]) -> None:
            self._state = {name: name for name in names}

        def state_dict(self) -> dict[str, str]:
            return dict(self._state)

    encoder = Module([f"encoder_state_{index:02d}" for index in range(78)])
    pixel = Module(["weight", "bias"])
    ground = Module(["0.weight", "0.bias", "2.weight", "2.bias"])
    decoder = Module(["decoder"])
    evidence = Module([
        *(f"pixel_head.{name}" for name in pixel.state_dict()),
        *(f"ground_head.{name}" for name in ground.state_dict()),
        "dense_decoder.decoder",
    ])
    model = SimpleNamespace(
        encoder=encoder,
        target_encoder=Module(list(encoder.state_dict())),
        evidence_head=SimpleNamespace(
            pixel_head=pixel,
            ground_head=ground,
            dense_decoder=decoder,
            state_dict=evidence.state_dict,
        ),
        _n320_initialization_complete=True,
    )
    fit = Module(["fit"])
    runtime = SimpleNamespace(
        torch=SimpleNamespace(__version__="test"),
        model_module=SimpleNamespace(
            tensor_state_dict_sha256=lambda state:
                contract.canonical_json_sha256(sorted(state))
        ),
    )
    multires = SimpleNamespace(
        INITIALIZATION_SCHEMA="initialization",
        MODEL_FAMILY="shared_observable_camera_ray_jepa_v5_multires_v1",
    )
    copied = sorted((
        *(f"encoder.{name}" for name in encoder.state_dict()),
        *(f"evidence_head.pixel_head.{name}" for name in pixel.state_dict()),
        *(f"evidence_head.ground_head.{name}" for name in ground.state_dict()),
    ))
    receipt = {
        "schema": "initialization",
        "model_family": multires.MODEL_FAMILY,
        "base_initialization_seed": 20260712,
        "decoder_initialization_seed": 20260724,
        "initialization_input_role": "n320_fit_initialization_only",
        "n320_checkpoint_file_sha256":
            contract.RUNTIME_FILE_SHA256[contract.N320_CHECKPOINT_RELATIVE_PATH],
        "n320_checkpoint_content_sha256":
            contract.RUNTIME_CONTENT_SHA256[
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
        "fit_model_state_sha256": runner._state_sha(runtime, fit),
        "shared_encoder_state_sha256": runner._state_sha(runtime, encoder),
        "pixel_head_state_sha256": runner._state_sha(runtime, pixel),
        "ground_head_state_sha256": runner._state_sha(runtime, ground),
        "decoder_state_sha256": runner._state_sha(runtime, decoder),
        "evidence_head_state_sha256": runner._state_sha(runtime, evidence),
        "copied_state_keys": copied,
        "copied_state_entry_count": 84,
        "copied_predecessor_dense_decoder_entry_count": 0,
        "canonical_ground_support_exact": True,
        "hard_sync_count": 1,
        "caller_cpu_rng_restored": True,
        "rejected_adaptation_checkpoint_open_count": 0,
        "torch_version": "test",
    }
    assert runner._validate_migration_receipt(
        runtime, multires, model, fit, receipt
    ) == receipt
    incomplete = {**receipt, "copied_state_entry_count": 83}
    with pytest.raises(PermissionError):
        runner._validate_migration_receipt(
            runtime, multires, model, fit, incomplete
        )


def test_run_parent_orders_authority_preflight_reservation_then_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    review = {"content_sha256": "1" * 64}
    authorization = {"content_sha256": "2" * 64}
    sources = {contract.LAUNCHER_RELATIVE_PATH: "3" * 64}
    reservation = {"attempt_identity": "4" * 64}

    def authority(*_args: Any) -> tuple[Any, ...]:
        events.append("authority")
        return review, b"review", authorization, b"authorization", sources

    def preflight(**_kwargs: Any) -> dict[str, Any]:
        events.append("preflight")
        return {"status": "pass"}

    def reserve(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        events.append("reserve")
        return reservation, b"reservation"

    def execute(**_kwargs: Any) -> int:
        events.append("execute")
        return 17

    monkeypatch.setattr(runner, "_load_authority_pre_reservation", authority)
    monkeypatch.setattr(runner, "_validate_preflight", preflight)
    monkeypatch.setattr(runner, "_reserve", reserve)
    monkeypatch.setattr(runner, "_execute_after_reservation", execute)
    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
        preflight_file_sha256="c" * 64,
    ) == 17
    assert events == ["authority", "preflight", "reserve", "execute"]


def test_reservation_is_mode_0700_and_consumes_attempt(tmp_path: Path) -> None:
    output = tmp_path / "attempt"
    review = {"content_sha256": "1" * 64}
    authorization = {"content_sha256": "2" * 64}
    reservation, raw = runner._reserve(
        output,
        review=review,
        review_raw=b"review",
        authorization=authorization,
        authorization_raw=b"authorization",
        sources={"x": "3" * 64},
        preflight={"status": "pass"},
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert reservation["reservation_consumes_attempt"] is True
    assert reservation["torch_imported_before_reservation"] is False
    assert reservation["runtime_input_opened_before_reservation"] is False
    assert (output / "reservation.json").read_bytes() == raw
    with pytest.raises(RuntimeError):
        runner._reserve(
            output,
            review=review,
            review_raw=b"review",
            authorization=authorization,
            authorization_raw=b"authorization",
            sources={"x": "3" * 64},
            preflight={"status": "pass"},
        )


def test_post_mkdir_reservation_failure_is_terminal_and_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed_attempt"
    original = runner._publish_json
    calls = 0

    def fail_reservation_once(path: Path, core: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic reservation publication failure")
        return original(path, core)

    monkeypatch.setattr(runner, "_publish_json", fail_reservation_once)
    with pytest.raises(
        OSError, match="synthetic reservation publication failure"
    ):
        runner._reserve(
            output,
            review={"content_sha256": "1" * 64},
            review_raw=b"review",
            authorization={"content_sha256": "2" * 64},
            authorization_raw=b"authorization",
            sources={"x": "3" * 64},
            preflight={"status": "pass"},
        )
    assert calls == 2
    assert not (output / "reservation.json").exists()
    assert (output / "reservation_failed.json").is_file()
    assert stat.S_IMODE((output / "reservation_failed.json").stat().st_mode) == 0o444
    assert stat.S_IMODE(output.stat().st_mode) == 0o555


def test_readonly_sidecar_and_terminal_sealing(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    nested = root / "checkpoints"
    nested.mkdir(parents=True, mode=0o700)
    sidecar = nested / "update_100.metrics.json"
    runner._publish_readonly_atomic(sidecar, b"immutable\n")
    assert sidecar.read_bytes() == b"immutable\n"
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    other = root / "result.json"
    other.write_bytes(b"result\n")
    receipt = runner._seal_terminal(root)
    assert receipt["files"] == [
        "checkpoints/update_100.metrics.json",
        "result.json",
    ]
    assert stat.S_IMODE(other.stat().st_mode) == 0o444
    assert stat.S_IMODE(nested.stat().st_mode) == 0o555
    assert stat.S_IMODE(root.stat().st_mode) == 0o555


def test_launcher_preflight_is_immediately_followed_by_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    observation = {
        "preflight_child_process_id": 123,
        "visible_device_count": 1,
        "visible_device_index": 0,
        "visible_device_name": "AMD Radeon AI PRO R9700",
        "total_memory_bytes": 32_000_000_000,
        "torch_version": "test",
        "hip_version": "test",
        "tensor_allocation_count": 0,
        "payload_open_count": 0,
        "torch_device_api_call_count": 3,
    }
    source_authority = {
        "source_binding_count": 17,
        "source_bindings_sha256": "f" * 64,
    }

    def authority(_args: Any) -> dict[str, Any]:
        events.append("authority")
        return source_authority

    def preflight(_environment: Any) -> dict[str, Any]:
        events.append("preflight")
        return observation

    def receipt(
        value: Any,
        authority_value: Any,
    ) -> tuple[dict[str, Any], bytes]:
        assert value is observation
        assert authority_value is source_authority
        events.append("receipt")
        return {"content_sha256": "0" * 64}, b"receipt\n"

    def execute(
        _args: Any,
        *,
        receipt: Any,
        receipt_raw: bytes,
        environment: Any,
    ) -> None:
        del receipt, receipt_raw, environment
        events.append("exec")
        raise RuntimeError("fake exec")

    monkeypatch.setattr(launcher, "_load_authority_before_hardware", authority)
    monkeypatch.setattr(launcher, "_run_no_tensor_preflight", preflight)
    monkeypatch.setattr(launcher, "_preflight_receipt", receipt)
    monkeypatch.setattr(launcher, "_exec_runner", execute)
    args = argparse.Namespace(
        review_sha256="a" * 64,
        authorization_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="fake exec"):
        launcher._launch(args, {})
    assert events == ["authority", "preflight", "receipt", "exec"]


def test_snapshot_evaluation_sidecar_control_source_order() -> None:
    source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text("utf-8")
    checkpoint_block = source[source.index("if update not in contract.CHECKPOINT_UPDATES"):]
    positions = [
        checkpoint_block.index("snapshot = _snapshot("),
        checkpoint_block.index("metric = _evaluate("),
        checkpoint_block.index("sidecar, control = _publish_metric_sidecar("),
        checkpoint_block.index("controls.append(control)"),
        checkpoint_block.index("if update in (100, 400):"),
    ]
    assert positions == sorted(positions)
    for version in range(1, 7):
        assert (
            f"run_go2_shared_jepa_v5_protected_camera_adaptation_v{version}"
            not in source
        )
    assert "run_go2_shared_jepa_v5_matched_training_v1.py" not in source
    assert "MATCHED_V1_RUNNER_RELATIVE_PATH" in source
