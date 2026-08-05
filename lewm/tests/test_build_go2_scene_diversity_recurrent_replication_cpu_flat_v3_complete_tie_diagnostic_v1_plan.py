from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from scripts import build_go2_scene_diversity_recurrent_replication_cpu_flat_v3_complete_tie_diagnostic_v1_plan as builder


def test_exact_predecessor_bindings_and_terminal_disposition():
    plan = builder.build_plan()
    predecessor = plan["predecessor"]
    assert predecessor["result_must_be_absent"] is True
    assert not Path(predecessor["result_path"]).exists()
    assert predecessor["scientific_plan_binding"] == {
        "path": str(builder.PREDECESSOR_PLAN.resolve()),
        "sha256": builder.PREDECESSOR_PLAN_SHA256,
        "byte_count": builder.PREDECESSOR_PLAN_BYTE_COUNT,
    }
    assert predecessor["terminal_binding"] == {
        "path": str(builder.PREDECESSOR_TERMINAL.resolve()),
        "sha256": builder.PREDECESSOR_TERMINAL_SHA256,
        "byte_count": builder.PREDECESSOR_TERMINAL_BYTE_COUNT,
    }
    assert predecessor["physics_result_binding"] == {
        "path": str(builder.PREDECESSOR_PHYSICS_RESULT.resolve()),
        "sha256": builder.PREDECESSOR_PHYSICS_RESULT_SHA256,
        "byte_count": builder.PREDECESSOR_PHYSICS_RESULT_BYTE_COUNT,
    }
    assert predecessor["checkpoint_binding"] == {
        "path": str(builder.PREDECESSOR_CHECKPOINT.resolve()),
        "sha256": builder.PREDECESSOR_CHECKPOINT_SHA256,
        "byte_count": builder.PREDECESSOR_CHECKPOINT_BYTE_COUNT,
    }
    assert predecessor["terminal_review_binding"] == {
        "path": str(builder.PREDECESSOR_TERMINAL_REVIEW.resolve()),
        "sha256": builder.PREDECESSOR_TERMINAL_REVIEW_SHA256,
        "byte_count": builder.PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT,
    }
    assert predecessor["evidence_disposition"] == {
        "predecessor_attempt_consumed": True,
        "predecessor_scientific_result_absent": True,
        "predecessor_scientific_decision_available": False,
        "physics_collection_complete": True,
        "checkpoint_integrity_review_passed": True,
        "complete_tie_eval_state_count": 4,
        "complete_tie_train_state_count": 0,
        "predecessor_artifact_reuse_authorized_by_review": False,
        "separate_preregistration_review_and_authority_required": True,
    }


def test_runner_interface_and_complete_tie_rule_are_exact():
    plan = builder.build_plan()
    assert plan["schema"] == builder.PLAN_SCHEMA
    assert plan["status"] == builder.PLAN_STATUS
    assert plan["development_only"] is True
    assert plan["post_hoc_nonconfirmatory"] is True
    assert plan["citable_as_scientific_evidence"] is False
    assert plan["fresh_root_required"] is True
    evaluation = plan["evaluation_contract"]
    assert evaluation["evaluation_only"] is True
    assert evaluation["training_authorized"] is False
    assert evaluation["rendering_authorized"] is False
    assert evaluation["collection_authorized"] is False
    assert evaluation["roles_reconstructed"] == ["train", "eval"]
    assert evaluation["train_role_use"] == (
        "live_task_action_only_control_metadata_only"
    )
    assert evaluation["train_context_rgb_open_count"] == 0
    assert evaluation["eval_context_rgb_open_count"] == 384
    assert evaluation["successor_rgb_open_count"] == 0
    assert evaluation["eval_state_count"] == 128
    assert evaluation["expected_eval_complete_tie_state_count"] == 4
    assert evaluation["eval_state_exclusion_authorized"] is False
    assert evaluation["complete_tie_rule"] == "all_actions_oracle_equivalent"
    assert evaluation["random_expected_denominator"] == (
        "max(1,max_dense_rank)"
    )
    assert evaluation["rank_tolerance_m"] == 0.01


def test_model_bootstrap_threshold_and_dino_science_remain_frozen():
    plan = builder.build_plan()
    evaluation = plan["evaluation_contract"]
    assert evaluation["frozen_recurrent_config"] == builder.benchmark.config_v1()
    assert evaluation["model_seeds"] == [
        2_026_080_411,
        2_026_080_412,
        2_026_080_413,
    ]
    assert evaluation["sampler_seed"] == 2_026_080_414
    assert evaluation["bootstrap_resamples"] == 10_000
    assert evaluation["bootstrap_seed"] == 2_026_080_407
    assert evaluation["frozen_thresholds"] == (
        builder.benchmark.config_v1()["frozen_recurrent_protocol"][
            "frozen_h1_thresholds"
        ]
    )
    assert evaluation["evaluation_repetitions"] == 2
    assert evaluation["repeat_evaluation_exact_required"] is True
    assert evaluation["compute_device"] == "cpu"
    assert plan["dino"] == builder.frozen_runner.expected_dino_v1()


def test_plan_is_exact_mutation_closed_and_does_not_create_attempt_root():
    assert not builder.ATTEMPT_ROOT.exists()
    plan = builder.build_plan()
    assert builder.validate_plan(plan) == plan
    changed = copy.deepcopy(plan)
    changed["evaluation_contract"]["expected_eval_complete_tie_state_count"] = 3
    with pytest.raises(builder.CompleteTieDiagnosticPlanError):
        builder.validate_plan(changed)
    assert not builder.ATTEMPT_ROOT.exists()


def test_frozen_plan_output_matches_builder_exactly():
    raw = builder.PLAN_OUTPUT.read_bytes()
    assert len(raw) == builder.PLAN_BYTE_COUNT
    assert hashlib.sha256(raw).hexdigest() == builder.PLAN_SHA256
    value = json.loads(raw)
    assert builder.validate_plan(value) == value
    assert value == builder.build_plan()


def test_builder_is_metadata_only_and_has_no_runtime_or_protected_route():
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not call_names & {"exec", "eval", "compile"}
    assert not attribute_calls & {
        "Popen",
        "run",
        "load_dino_trunk_v1",
        "fit_checkpoint_v1",
        "evaluate_checkpoint_v1",
        "mkdir",
    }
    assert "sealed" not in source.lower()
    assert "heldout" not in source.lower()
    assert "genesis.init" not in source


def test_main_writes_only_requested_plan_and_never_attempt_root():
    assert not builder.ATTEMPT_ROOT.exists()
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "plan.json"
        assert builder.main(["--plan-output", str(output)]) == 0
        raw = output.read_bytes()
        value = json.loads(raw)
        assert builder.validate_plan(value) == value
        assert hashlib.sha256(raw).hexdigest()
        assert len(raw) > 0
    assert not builder.ATTEMPT_ROOT.exists()
