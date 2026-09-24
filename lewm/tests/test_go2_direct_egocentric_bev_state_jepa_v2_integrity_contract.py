from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
V1_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("_test_direct_bev_v2_integrity_contract", CONTRACT_PATH)
v1 = _load("_test_direct_bev_v2_integrity_frozen_v1_contract", V1_CONTRACT_PATH)


def _synthetic_manifest() -> tuple[dict[str, object], bytes]:
    paths = list(contract.SOURCE_PATHS)
    bindings = [
        {"path": path, "file_sha256": "1" * 64, "byte_count": 1}
        for path in paths
    ]
    core = {
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_CLOSURE",
        "entrypoints": list(contract.SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources": list(
            contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
        ),
        "excluded_runtime_categories": list(
            contract.PROHIBITED_RUNTIME_CATEGORIES
        ),
        "source_paths": paths,
        "source_bindings": bindings,
        "source_bindings_sha256": contract.canonical_json_sha256(bindings),
        "source_count": len(paths),
        "generated_input_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
        "authority": contract.SOURCE_ONLY_AUTHORITY,
    }
    value = contract.with_content_sha256(core)
    raw = contract.canonical_json_bytes(value) + b"\n"
    return value, raw


def test_isolated_import_is_source_only() -> None:
    program = f"""
import importlib.util, json, pathlib, sys
path = pathlib.Path({str(CONTRACT_PATH)!r})
spec = importlib.util.spec_from_file_location('_v2_contract_probe', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps({{
    'torch': 'torch' in sys.modules,
    'numpy': 'numpy' in sys.modules,
    'PIL': 'PIL' in sys.modules,
    'experiment': module.EXPERIMENT_ID,
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "PIL": False,
        "experiment": "go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity",
        "numpy": False,
        "torch": False,
    }


def test_amendment_and_terminal_audit_are_exactly_bound() -> None:
    observed = contract.validate_governing_documents(ROOT)
    assert observed[contract.INTEGRITY_AMENDMENT_RELATIVE_PATH] == (
        "ff06e8834a96cab616a8a8c5ed7589fb73de202166ad278253258a55ad688509"
    )
    assert observed[contract.FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH] == (
        "a2fc88e015e51f2d17263fc9f00cd26bb21964bc2fe1cb046f828b33805e07c7"
    )
    assert observed[contract.V1_TERMINAL_AUDIT_RELATIVE_PATH] == (
        "f928c11a2e52349145701b25a21f8b1b987ee80a365aaa2c3858d3cf650220c4"
    )
    assert contract.INTEGRITY_AMENDMENT_COMMIT == (
        "0221d4ddd5e266a9c715d8ccb788107c0671f6ee"
    )
    assert contract.V1_TERMINAL_AUDIT_COMMIT == (
        "ae94021d44711bf9ba5fbb1386b4f8caf2617dac"
    )


def test_frozen_v1_manifest_rehashes_every_bound_source() -> None:
    current = contract.validate_frozen_v1_source_closure(ROOT)
    assert contract.frozen_v1_source_manifest_binding() == {
        "path": contract.FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": "51ce1480ab2cfdcf9df7e984c7be6e58890811af",
        "file_sha256": (
            "a2fc88e015e51f2d17263fc9f00cd26bb21964bc2fe1cb046f828b33805e07c7"
        ),
        "content_sha256": (
            "e41e6cee37d5e2a69dafdb07e3219dbbf7cf484c16e5b99bd0e2975b8fba94d9"
        ),
        "byte_count": 19_505,
        "status": "PASS_SOURCE_CLOSURE",
    }
    for relative, binding in contract.FROZEN_V1_SOURCE_BINDINGS.items():
        assert current[relative] == binding["file_sha256"]


@pytest.mark.parametrize(
    "relative",
    (
        contract.FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH,
        contract.FROZEN_V1_LAUNCHER_RELATIVE_PATH,
    ),
)
def test_frozen_v1_closure_rejects_manifest_or_source_tamper(
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    original = contract._v1._read_regular_source
    target = ROOT / relative

    def tampered_read(path: Path) -> bytes:
        raw = original(path)
        return raw + b"\n" if Path(path) == target else raw

    monkeypatch.setattr(contract._v1, "_read_regular_source", tampered_read)
    with pytest.raises(PermissionError):
        contract.validate_frozen_v1_source_closure(ROOT)


def test_only_governance_lifecycle_and_rng_integrity_seam_change() -> None:
    v2_science = contract.science_contract()
    v1_science = v1.science_contract()
    for field in (
        "scientific_question",
        "repository_goal",
        "model",
        "data",
        "loader",
        "objective",
        "optimizer",
        "schedule",
        "gates",
        "access_policy",
        "authority",
    ):
        assert v2_science[field] == v1_science[field]
    for field, value in v1_science["lifecycle"].items():
        if field != "output_root":
            assert v2_science["lifecycle"][field] == value
    assert v2_science["lifecycle"]["output_root"] != (
        v1_science["lifecycle"]["output_root"]
    )
    assert v2_science["integrity_replacement"]["v1_retry"] is False if (
        "v1_retry" in v2_science["integrity_replacement"]
    ) else v2_science["lifecycle"]["v1_retry"] is False
    assert contract.INTEGRITY_DELTA == {
        "scope": "fresh_module_cpu_default_generator_seeding_only",
        "v1_call": "torch.random.manual_seed(20260712)",
        "v2_call": "torch.random.default_generator.manual_seed(20260712)",
        "caller_cpu_rng_preserved": True,
        "every_device_rng_preserved": True,
        "parameter_draw_order_changed": False,
        "initialized_parameter_bytes_changed": False,
        "architecture_data_objective_optimizer_schedule_gate_or_cap_changed": False,
    }


def test_all_science_component_hashes_match_frozen_v1() -> None:
    assert contract.canonical_json_sha256(v1.science_contract()) == (
        contract.FROZEN_V1_SCIENCE_CONTRACT_SHA256
    )
    observed = {
        "model": contract.canonical_json_sha256(contract.model_config()),
        "objective": contract.canonical_json_sha256(
            contract.objective_contract()
        ),
        "optimizer": contract.canonical_json_sha256(
            contract.optimizer_contract()
        ),
        "schedule": contract.canonical_json_sha256(
            contract.build_schedule_identity()
        ),
        "gate_thresholds": contract.canonical_json_sha256(
            contract.GATE_THRESHOLDS
        ),
    }
    assert observed == contract.FROZEN_V1_SCIENCE_COMPONENT_SHA256
    assert contract.runtime_authorization_template() == (
        v1.runtime_authorization_template()
    )


def test_seeds_schedule_gates_populations_and_caps_are_identical() -> None:
    for name in (
        "BASE_INITIALIZATION_SEED",
        "N320_FIT_SEED",
        "SCHEDULE_SEED",
        "TARGET_EMA_MOMENTUM",
        "MAXIMUM_ATTEMPTS",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "GPU_ACTIVE_TIME_CAP_MINUTES",
        "EFFECTIVE_BATCH_SIZE",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "CHECKPOINT_UPDATES",
        "OBSERVATION_UPDATES",
        "SCHEDULE_PREFIX_SHA256",
        "GATE_THRESHOLDS",
        "TRAIN_ROLE_COUNTS",
        "SELECTION_ROLE_COUNTS",
        "TARGET_MAPPING_BINDINGS",
        "AGGREGATE_RASTER_ENDPOINT_COUNT",
        "ROUGH_RASTER_ENDPOINT_COUNT",
    ):
        assert getattr(contract, name) == getattr(v1, name)
    assert contract.MAXIMUM_UPDATES == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000


def test_gate_evaluation_is_identical() -> None:
    metrics = {
        "three_logit_bottleneck_exact": True,
        "no_hidden_or_auxiliary_bypass": True,
        "prediction_is_exact_persistence": True,
        "all_nine_action_predictions_bitwise_equal": True,
        "target_parameters_gradient_free": True,
        "intended_online_path_gradient_nonzero": True,
        "six_call_graph_isolation_exact": True,
        "all_registered_values_finite": True,
        "action_nll": math.log(9.0),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
        "G": 1.0,
        "J": 1.0,
    }
    assert contract.evaluate_gate(0, metrics) == v1.evaluate_gate(0, metrics)


def test_v2_paths_are_distinct_and_closure_contains_all_v1_roots() -> None:
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v1.OUTPUT_ROOT_RELATIVE_PATH
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith("_v2_integrity")
    assert set(contract.ADDITIVE_SOURCE_PATHS).issubset(
        contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
    )
    assert set(v1.SOURCE_PATHS).issubset(
        contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
    )
    for path in (
        contract.FROZEN_V1_CONTRACT_RELATIVE_PATH,
        contract.FROZEN_V1_RUNNER_RELATIVE_PATH,
        contract.FROZEN_V1_LAUNCHER_RELATIVE_PATH,
        contract.FROZEN_V1_MODEL_RELATIVE_PATH,
        contract.FROZEN_V1_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    ):
        assert path in contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES


def test_source_only_authority_denies_execution_and_downstream_scope() -> None:
    assert contract.SOURCE_ONLY_AUTHORITY["execution_authorized"] is False
    assert contract.SOURCE_ONLY_AUTHORITY[
        "generated_input_access_authorized"
    ] is False
    assert contract.EXECUTION_AUTHORITY["maximum_updates"] == 1_000
    assert contract.EXECUTION_AUTHORITY["maximum_presentations"] == 16_000
    assert contract.EXECUTION_AUTHORITY["output_root"] == (
        contract.OUTPUT_ROOT_RELATIVE_PATH
    )
    assert contract.EXECUTION_AUTHORITY["v1_retry_authorized"] is False
    assert contract.DOWNSTREAM_DENIALS["heldout_authorized"] is False


def test_synthetic_manifest_validation_is_fail_closed() -> None:
    value, raw = _synthetic_manifest()
    assert contract.validate_source_manifest(raw) == value
    changed = dict(value)
    changed["generated_input_open_count"] = 1
    changed.pop("content_sha256")
    changed = contract.with_content_sha256(changed)
    changed_raw = contract.canonical_json_bytes(changed) + b"\n"
    try:
        contract.validate_source_manifest(changed_raw)
    except PermissionError:
        pass
    else:
        raise AssertionError("manifest accepted generated-input access")


def test_synthetic_review_and_authorization_validation() -> None:
    _manifest, manifest_raw = _synthetic_manifest()
    manifest_binding = contract.artifact_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=str(_manifest["content_sha256"]),
    )
    expected_sources = {
        path: "1" * 64 for path in contract.SOURCE_PATHS
    }
    expected_sources.update({
        contract.SOURCE_MANIFEST_RELATIVE_PATH: manifest_binding["file_sha256"],
        contract.FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH: (
            contract.FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256
        ),
        contract.INTEGRITY_AMENDMENT_RELATIVE_PATH: (
            contract.INTEGRITY_AMENDMENT_FILE_SHA256
        ),
        contract.V1_TERMINAL_AUDIT_RELATIVE_PATH: (
            contract.V1_TERMINAL_AUDIT_FILE_SHA256
        ),
    })
    review = contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS_SOURCE_AND_SCIENCE_IDENTICAL_INTEGRITY",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": "/root/independent_v2_reviewer",
        "reviewed_sources": expected_sources,
        "source_manifest": manifest_binding,
        "integrity_amendment": contract.integrity_amendment_binding(),
        "frozen_v1_source_manifest": (
            contract.frozen_v1_source_manifest_binding()
        ),
        "v1_terminal_audit": contract.v1_terminal_audit_binding(),
        "science_contract": contract.science_contract(),
        "source_only_checks": {
            "stdlib_only_contract_import": True,
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "sealed_or_heldout_opened": [],
        },
        "scientific_checks": contract.SCIENTIFIC_REVIEW_CHECKS,
        "findings": [],
        "authority": contract.REVIEW_AUTHORITY,
    })
    assert contract.validate_review(
        review,
        expected_sources=expected_sources,
        source_manifest_binding=manifest_binding,
    ) == review
    changed_sources = dict(expected_sources)
    changed_sources[contract.FROZEN_V1_SOURCE_MANIFEST_RELATIVE_PATH] = (
        "2" * 64
    )
    changed_review = dict(review)
    changed_review.pop("content_sha256")
    changed_review["reviewed_sources"] = changed_sources
    changed_review = contract.with_content_sha256(changed_review)
    with pytest.raises(PermissionError):
        contract.validate_review(
            changed_review,
            expected_sources=changed_sources,
            source_manifest_binding=manifest_binding,
        )
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization = contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "authorizer": "/root/independent_v2_authorizer",
        "independent_source_review": review_binding,
        "integrity_amendment": contract.integrity_amendment_binding(),
        "v1_terminal_audit": contract.v1_terminal_audit_binding(),
        "runtime_inputs": contract.runtime_authorization_template(),
        "experiment": contract.science_contract(),
        "authority": contract.EXECUTION_AUTHORITY,
    })
    assert contract.validate_authorization(
        authorization,
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    ) == authorization
