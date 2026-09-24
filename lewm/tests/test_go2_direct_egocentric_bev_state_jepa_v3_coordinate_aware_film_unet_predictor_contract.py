from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
V2_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("_test_direct_bev_v3_contract", CONTRACT_PATH)
v2 = _load("_test_direct_bev_v3_frozen_v2_contract", V2_CONTRACT_PATH)


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
        "forced_dynamic_sources": list(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES),
        "excluded_runtime_categories": list(contract.PROHIBITED_RUNTIME_CATEGORIES),
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
    return value, contract.canonical_json_bytes(value) + b"\n"


def _u100_metrics() -> dict[str, object]:
    return {
        "G": 0.50,
        "J": 0.60,
        "action_nll": 2.187,
        "action_macro_balanced_accuracy": 0.13,
        "correct_rgb_scene_win_count": 6,
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "hardest_wrong_positive_scene_count": 2,
        "aggregate_raster_balanced_accuracy": 0.65,
    }


def test_isolated_import_is_stdlib_and_source_only() -> None:
    program = f"""
import importlib.util, json, pathlib, sys
path = pathlib.Path({str(CONTRACT_PATH)!r})
spec = importlib.util.spec_from_file_location('_v3_contract_probe', path)
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
        "experiment": (
            "go2_rgb_direct_egocentric_bev_state_jepa_v3_"
            "coordinate_aware_film_unet_predictor"
        ),
        "numpy": False,
        "torch": False,
    }


def test_frozen_v2_closure_audit_and_preregistration_are_exact() -> None:
    current = contract.validate_frozen_v2_source_closure(ROOT)
    assert len(v2.SOURCE_PATHS) == 73
    for relative in v2.SOURCE_PATHS:
        assert current[relative]
    observed = contract.validate_governing_documents(ROOT)
    assert observed[contract.FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH] == (
        "a52b99f13cdbb3e8841e9c87e451d4ab5aa09db3c943acb8b14e67a49ec2e510"
    )
    assert observed[contract.V2_TERMINAL_AUDIT_RELATIVE_PATH] == (
        "93132058a0f94f652864e73e00cfb050c35f901e73d06277e13e3897825ef5a0"
    )
    assert observed[contract.PREREGISTRATION_RELATIVE_PATH] == (
        "be75f268816f422f1a40b7ee56dbf4bf544cd6893f9d3b296540ff4a98176c02"
    )
    assert contract.V2_TERMINAL_AUDIT_STATUS.endswith("CLOSES_V2_NO_RETRY")


@pytest.mark.parametrize(
    "relative",
    (
        contract.FROZEN_V2_SOURCE_MANIFEST_RELATIVE_PATH,
        contract.FROZEN_V2_CONTRACT_RELATIVE_PATH,
    ),
)
def test_frozen_v2_closure_rejects_manifest_or_source_tamper(
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    original = contract._v2._v1._read_regular_source
    target = ROOT / relative

    def tampered_read(path: Path) -> bytes:
        raw = original(path)
        return raw + b"\n" if Path(path) == target else raw

    monkeypatch.setattr(contract._v2._v1, "_read_regular_source", tampered_read)
    with pytest.raises(PermissionError):
        contract.validate_frozen_v2_source_closure(ROOT)


def test_only_predictor_model_and_v3_governance_change() -> None:
    v3_science = contract.science_contract()
    v2_science = v2.science_contract()
    for field in (
        "scientific_question",
        "repository_goal",
        "data",
        "loader",
        "objective",
        "optimizer",
        "schedule",
        "access_policy",
    ):
        assert v3_science[field] == v2_science[field]
    assert "integrity_replacement" not in v3_science
    assert v3_science["frozen_v2_integrity_provenance"] == {
        "scope": "historical_v1_to_v2_integrity_replacement_only",
        "not_a_v3_unchanged_architecture_claim": True,
        "v1_to_v2": v2_science["integrity_replacement"],
    }
    v3_model = contract.model_config()
    v2_model = v2.model_config()
    for field in v2_model:
        if field not in {"transition", "parameter_inventory"}:
            assert v3_model[field] == v2_model[field]
    for field in (
        "encoder", "decoder_state", "detached_target_encoder_decoder_state"
    ):
        assert v3_model["parameter_inventory"][field] == (
            v2_model["parameter_inventory"][field]
        )
    assert contract.objective_contract() == v2.objective_contract()
    assert contract.optimizer_contract() == v2.optimizer_contract()
    assert contract.build_schedule_identity() == v2.build_schedule_identity()


def test_predictor_topology_inventory_and_rng_are_exact() -> None:
    config = contract.PREDICTOR_CONFIG
    assert config["inputs_in_order"] == [
        "current_three_logit_state",
        "normalized_row_index",
        "normalized_column_index",
    ]
    assert config["coordinate_construction"] == {
        "row": "linspace(-1,1,H) expanded along columns",
        "column": "linspace(-1,1,W) expanded along rows",
        "dtype_and_device": "current_state_logits",
        "persistent_buffer": False,
        "metric_pose_or_geometry": False,
    }
    assert config["action_embedding"] == {"count": 9, "dim": 64}
    assert config["normalization"] == {
        "type": "GroupNorm", "groups": 4, "affine": True
    }
    assert config["film"]["channel_order"] == [64, 48, 32, 16]
    assert config["film"]["formula"] == "x*(1+gamma)+beta"
    assert config["all_actions"] == "encode_once_decode_nine_film_conditions"
    assert config["construction_draw_order"] == [
        "unchanged_v2_perception", "action_embedding", "enc64", "down32",
        "enc32", "down16", "enc16", "down8", "bottleneck", "film64",
        "dec16", "film48", "dec32", "film32", "dec64", "film16",
        "residual_head",
    ]
    assert config["rng"] == {
        "seed": 20260712,
        "seed_target": "torch.random.default_generator_only",
        "caller_cpu_rng_restored": True,
        "accelerator_seed_calls": 0,
    }
    inventory = contract.model_config()["parameter_inventory"]
    assert inventory["predictor"]["parameter_count"] == 317_107
    assert inventory["predictor"]["tensor_count"] == 79
    assert inventory["total"] == {
        "parameter_count": 6_552_249,
        "tensor_count": 277,
    }
    marker = inventory["predictor"]["ordered_parameter_name_sha256"]
    assert marker == (
        "0398031cb776c10a23b14c7935d2566f4a3087175213e87b49c2a05cadf6e1dd"
    )
    assert inventory["predictor"][
        "predictor_local_ordered_parameter_name_sha256"
    ] == (
        "ebbd0bb384b09862c867338b39b4ffcfa4072e43730451f0eee337be3167fad2"
    )
    assert inventory["predictor"]["ordered_parameter_inventory_sha256"] == (
        "5c8cac4bb77b3669894b04a7def61fe8f35ee2f7cb84bb2e38c0efdb8ab35665"
    )


def test_u100_strengthened_boundaries_are_inclusive_and_conjunctive() -> None:
    metrics = _u100_metrics()
    result = contract.evaluate_gate(
        100,
        metrics,
        update_zero={"G": 0.8, "J": 0.7},
    )
    assert result["passed"] is True
    assert result["control"] == "CONTINUE_AFTER_UPDATE_100_V3_PREDICTOR_GATE"
    expected = {
        "action_macro_balanced_accuracy": 0.129999,
        "action_nll": 2.187001,
        "hardest_wrong_positive_scene_count": 1,
        "aggregate_raster_balanced_accuracy": 0.649999,
        "J": 0.600001,
    }
    for field, failing in expected.items():
        changed = dict(metrics)
        changed[field] = failing
        failed = contract.evaluate_gate(
            100,
            changed,
            update_zero={"G": 0.8, "J": 0.7},
        )
        assert failed["passed"] is False, field
        assert failed["control"] == (
            "FAIL_UPDATE_100_V3_PREDICTOR_GATE_TERMINAL_NO_RETRY"
        )


def test_v2_base_u100_and_other_numerical_gates_are_preserved() -> None:
    assert contract.GATE_THRESHOLDS[400] == v2.GATE_THRESHOLDS[400]
    assert contract.GATE_THRESHOLDS[1000] == v2.GATE_THRESHOLDS[1000]
    assert contract.UPDATE_ZERO_ACTION_TOLERANCE == v2.UPDATE_ZERO_ACTION_TOLERANCE
    result = contract.evaluate_gate(
        100,
        _u100_metrics(),
        update_zero={"G": 0.8, "J": 0.7},
        prior_gates_passed=False,
    )
    assert result["passed"] is False
    for name in (
        "G_strictly_decreased",
        "J_strictly_decreased",
        "action_nll_below_log9",
        "action_macro_balanced_accuracy_above_one_ninth",
        "correct_rgb_wins_at_least_six_scenes",
        "registered_values_finite",
        "state_nonconstant",
    ):
        assert name in result["conjuncts"]


def test_v3_terminal_controls_are_exactly_runner_compatible() -> None:
    assert contract.FAILURE_CONTROLS == tuple(
        pair[0] for pair in contract.GATE_CONTROLS.values()
    )
    assert contract.GATE_CONTROLS[100][0] in contract.FAILURE_CONTROLS
    assert contract.GATE_CONTROLS[1000][1] == contract.CONTROL_PASS
    assert contract.CONTROL_PASS == (
        "PASS_DIRECT_BEV_V3_FILM_UNET_PERCEPTION_GATE_REQUALIFICATION_ONLY"
    )
    assert contract.CONTROL_UPDATE_100_FAIL == (
        "FAIL_UPDATE_100_V3_PREDICTOR_GATE_TERMINAL_NO_RETRY"
    )


def test_caps_distinct_output_and_present_denials_are_exact() -> None:
    assert contract.MAXIMUM_ATTEMPTS == 1
    assert contract.MAXIMUM_UPDATES == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert contract.GPU_ACTIVE_TIME_CAP_MINUTES == 60
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v2.OUTPUT_ROOT_RELATIVE_PATH
    assert contract.PRESENT_AUTHORITY["execution_authorized"] is False
    assert contract.PRESENT_AUTHORITY["generated_input_access_authorized"] is False
    science = contract.science_contract()
    assert science["authority"]["v3_execution_authorized_by_source_contract"] is False
    assert science["scientific_checks"] == contract.SCIENTIFIC_REVIEW_CHECKS
    assert science["lifecycle"][
        "retry_resume_repair_recovery_replacement_second_seed_or_v3"
    ] is False
    assert contract.EXECUTION_AUTHORITY["v2_retry_authorized"] is False
    assert contract.EXECUTION_AUTHORITY[
        "v2_checkpoint_or_runtime_output_reuse_authorized"
    ] is False
    for name in (
        "heldout_authorized", "sealed_authorized", "navigation_authorized",
        "g2_authorized", "production_authorized", "promotion_authorized",
        "deployment_authorized",
    ):
        assert contract.EXECUTION_AUTHORITY[name] is False


def test_v3_paths_include_all_v2_sources_and_are_distinct() -> None:
    assert len(contract.REUSED_SOURCE_PATHS) == 73
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 10
    assert len(contract.SOURCE_PATHS) == 83
    assert set(v2.SOURCE_PATHS).issubset(contract.SOURCE_PATHS)
    assert set(contract.ADDITIVE_SOURCE_PATHS).issubset(contract.SOURCE_PATHS)
    assert contract.CONTRACT_RELATIVE_PATH != v2.CONTRACT_RELATIVE_PATH
    assert contract.MODEL_RELATIVE_PATH != v2.MODEL_RELATIVE_PATH


def test_synthetic_manifest_is_source_only_and_fail_closed() -> None:
    value, raw = _synthetic_manifest()
    assert contract.validate_source_manifest(raw) == value
    changed = dict(value)
    changed.pop("content_sha256")
    changed["checkpoint_or_tensor_open_count"] = 1
    changed = contract.with_content_sha256(changed)
    with pytest.raises(PermissionError):
        contract.validate_source_manifest(
            contract.canonical_json_bytes(changed) + b"\n"
        )
