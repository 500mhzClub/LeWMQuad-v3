"""Synthetic-only tests for the one-global exact small-completion model."""
from __future__ import annotations

import copy
import hashlib
import json
import warnings

import pytest

from lewm.oracle import go2_small_completion_global_exact_model_v1 as MODEL
from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOCATION
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as SELECTOR


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _variable(label: str, rotation: int = 0) -> dict:
    return {
        "key": label,
        "pair_identity": {
            "kind": "fixed_state",
            "state_identity_digest": _digest(label),
            "split_role": "fit",
            "candidate_rotation_index": rotation,
            "candidate_indices": list(MODEL.ROTATION_BLOCKS[rotation]),
        },
    }


def _constraint(name: str, terms: list[tuple[str, int]],
                lower: int | None, upper: int | None) -> dict:
    return {"name": name, "terms": [[key, value] for key, value in terms],
            "lower": lower, "upper": upper}


def _problem(name: str, variables: list[dict], constraints: list[dict]) -> dict:
    return {
        "schema": MODEL.GENERIC_PROBLEM_SCHEMA,
        "variables": variables,
        "constraints": constraints,
        "metadata": {"fixture": name},
    }


def _five_fixtures() -> list[dict]:
    return [
        _problem("one-hot", [_variable("a"), _variable("b"), _variable("c")], [
            _constraint("one", [("a", 1), ("b", 1), ("c", 1)], 1, 1),
        ]),
        _problem("forced", [_variable("d"), _variable("e", 1)], [
            _constraint("d-on", [("d", 1)], 1, 1),
            _constraint("e-off", [("e", 1)], 0, 0),
        ]),
        _problem("infeasible", [_variable("f")], [
            _constraint("f-off", [("f", 1)], 0, 0),
            _constraint("f-on", [("f", 1)], 1, 1),
        ]),
        _problem("dynamic-goal", [_variable("g"), _variable("h", 1)], [
            _constraint("one", [("g", 1), ("h", 1)], 1, 1),
            _constraint("minus1<=2A-n<=1", [("g", 1), ("h", -1)], -1, 1),
        ]),
        _problem("eligibility-and-cardinality", [
            _variable("i"), _variable("j", 1), _variable("k", 2),
            _variable("l", 3),
        ], [
            _constraint("choose-two", [(key, 1) for key in "ijkl"], 2, 2),
            _constraint("j-ineligible", [("j", 1)], 0, 0),
            _constraint("at-least-one-i-k", [("i", 1), ("k", 1)], 1, None),
        ]),
    ]


def _solve(model: dict) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return MODEL.solve_model(model)


def test_exact_authority_pair_preimage_coefficient_and_model_order():
    pair = _variable("known", 7)["pair_identity"]
    binding = MODEL.pair_objective_binding(pair)
    encoded = json.dumps(
        pair, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()
    expected = hashlib.sha256(
        MODEL.OBJECTIVE_DOMAIN.encode() + b"\x00" + encoded).hexdigest()
    assert expected == \
        "84443e811719f79e549787e5f8c23a2ae81986cbd427326638db969e1d7f2e16"
    assert binding["pair_digest"] == expected
    assert binding["objective_coefficient"] == 1 + int(expected[:10], 16)
    assert binding["objective_coefficient"] == 568_080_630_040
    assert MODEL.OBJECTIVE_CONTRACT_DIGEST == \
        MODEL.AUTHORITY.STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST

    problem = _problem("order", [_variable("z"), _variable("a", 3)], [
        _constraint("one", [("z", 1), ("a", 1)], 1, 1),
    ])
    model = MODEL.translate_binary_problem(problem)
    assert [(row["pair_digest"], row["canonical_pair_identity_json"])
            for row in model["variables"]] == sorted(
                (row["pair_digest"], row["canonical_pair_identity_json"])
                for row in model["variables"])
    assert model["objective_contract"] == MODEL.OBJECTIVE_CONTRACT
    assert model["objective_contract_digest"] == MODEL.OBJECTIVE_CONTRACT_DIGEST


@pytest.mark.parametrize("fixture", _five_fixtures(), ids=lambda row: row["metadata"]["fixture"])
def test_five_small_fixtures_match_exhaustive_control_and_repeat(fixture):
    model = MODEL.translate_binary_problem(fixture)
    first = _solve(model)
    second = _solve(model)
    control = MODEL.brute_force_model(model)
    assert first == second
    assert first["schema"] == control["schema"]
    if first["schema"] == MODEL.SOLUTION_SCHEMA:
        assert first["objective_value"] == control["objective_value"]
        assert first["selected_variable_keys"] == control["selected_variable_keys"]
        MODEL.validate_solution(model, first)
    else:
        MODEL.validate_infeasibility(model, first)


def _production_instance() -> dict:
    status = {"task_completed": False, "goal_claimed": False,
              "terminated": False, "truncated": False}
    evidence = SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=0, reachable=True, continuous_geodesic_m=0.1,
        bearing_body_rad=0.0, task_status=status,
        previous_applied_command=[0.0, 0.0, 0.0])
    eligibility = [row["eligible"] for row in evidence["rotations"]]
    fixed: list[dict] = []
    for family in MODEL.FAMILIES:
        for stratum in MODEL.STRATA:
            if (family, stratum) == (MODEL.SMALL_FAMILY,
                                     MODEL.COMPLETION_STRATUM):
                continue
            for ordinal in range(5):
                state_id = f"{family}/{stratum}/{ordinal}"
                identity = _digest(state_id)
                fixed.append({
                    "state_id": state_id,
                    "state_identity_digest": identity,
                    "scene_id": "fixed-scene/" + state_id,
                    "family": family,
                    "stratum": stratum,
                    "split_role": "calibration" if ordinal == 0 else "fit",
                    "goal_type": "goal-common",
                    "completion_rotation_eligibility_owner_digest": identity,
                    "completion_rotation_eligibility": (
                        eligibility if stratum == MODEL.COMPLETION_STRATUM
                        else [True] * 12),
                    "completion_rotation_evidence": (
                        evidence if stratum == MODEL.COMPLETION_STRATUM else None),
                })
    optional: list[dict] = []
    for index in range(17):
        scene_id = f"scene-{index:02d}"
        goal_cell = index + 20
        full_status = {
            **status,
            "production_claim_evidence": {
                "active_collector_visited_accessor_callable": True,
                "active_collector_claimed_cells": [],
                "designated_goal_cell": goal_cell,
            },
            "production_task_completion_reset_evidence": {
                "minimum_block_guard_pass": True,
                "scene_graph_available": True,
                "active_collector_route_like": True,
                "active_collector_non_revisit": True,
                "scene_landmark_cells_nonempty": True,
                "all_scene_landmark_cells_claimed": False,
            },
            "termination_flags": {
                "fall": False, "out_of_bounds": False,
                "tipped": False, "nan": False,
            },
        }
        distance = 1.2 if index == 0 else 0.1
        vector = SELECTOR.completion_rotation_eligibility_vector(
            graph_hops=0, reachable=True, continuous_geodesic_m=distance,
            bearing_body_rad=0.0, task_status=status,
            previous_applied_command=[0.0, 0.0, 0.0])
        raw_candidate = {
            "state_id": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
            "family": MODEL.SMALL_FAMILY,
            "scene_id": scene_id,
            "scene_dir": f"/synthetic/{scene_id}",
            "scene_manifest_sha256": _digest("manifest/" + scene_id),
            "scene_manifest_byte_count": 100 + index,
            "split": "synthetic", "drive_seed": index,
            "stratum": MODEL.COMPLETION_STRATUM,
            "split_role": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
            "warmup_blocks": index + 1, "source_step": 100 + index,
            "episode_id": index + 1,
            "episode_cluster_id": f"cluster-{index:02d}",
            "cell_id": index + 10,
            "boundary": {
                "command_block_tick": 0,
                "decimation_phase": 0,
                "observation_emission_phase_ns": 0,
                "reset": False,
                "terminated": False,
                "truncated": False,
                "source_step": 100 + index,
                "episode_step": 99 + index,
                "sim_time_ns": (
                    (999 + index) * MODEL.CANONICAL_BOUNDARY_STEP_NS),
                "boundary_digest": MODEL.CANONICAL_BOUNDARY_DIGEST,
            },
            "goal": {"landmark_id": f"goal-{index}",
                     "landmark_cell": goal_cell,
                     "material_id": "goal-common", "graph_edges": 0,
                     "start_geodesic_m": distance, "bearing_body_rad": 0.0,
                     "range_m": distance, "landmark_xy_m": [0.0, 0.0]},
            "goal_type": "goal-common", "body_clearance_m": 0.2,
            "clearance_m": 0.2,
            "completion_rotation_eligibility_vector": vector,
            "snapshot_task_status": full_status,
            "previous_applied_command": [0.0, 0.0, 0.0],
        }
        optional.append({
            "raw_candidate": raw_candidate,
            "completion_rotation_eligibility": [
                row["eligible"] for row in vector["rotations"]],
        })
    selection_digest = _digest("selection")
    scorer_digest = _digest("scorer")
    common = {key: _digest(key) for key in MODEL._PRE_ALLOCATION_COMMON_KEYS}
    common.update({
        "selection_digest": selection_digest,
        "scorer_contract_v1_2_digest": scorer_digest,
        "candidate_allocator_contract_digest":
            ALLOCATION.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            ALLOCATION.allocation_amendment_digest(),
        "boundary_digest": MODEL.CANONICAL_BOUNDARY_DIGEST,
        "source_repository_commit": "1" * 40,
        "genesis_backend": "cpu",
    })
    static = {
        "schema": "go2_branch_corpus_v1_2_pre_allocation_identity_manifest",
        "pool": "scorer_fit", "spec": copy.deepcopy(MODEL._SCORER_FIT_POOL_SPEC),
        **common,
    }
    return MODEL.build_production_instance(
        fixed_states=fixed, optional_candidates=optional,
        state_identity_lineage={
            "schema": MODEL.STATE_IDENTITY_LINEAGE_SCHEMA,
            "selection_digest": selection_digest,
            "scorer_contract_v1_2_digest": scorer_digest,
            "pool": "scorer_fit",
            "pre_allocation_identity_static": static,
        })


def test_production_compiler_is_exact_complete_and_deterministic_without_solving():
    instance = _production_instance()
    assert len(MODEL.validate_production_instance(instance)["fixed_states"]) == 115
    first = MODEL.build_production_model(instance)
    second = MODEL.build_production_model(copy.deepcopy(instance))
    assert first == second
    model = first["model"]
    assert len(model["variables"]) == 115 * 12 + 17 * 2 * 12
    assert model["metadata"]["active_completion_state_count"] == 40
    assert any("minus1<=2A-n<=1" in row["name"]
               and row["lower"] == -1 and row["upper"] == 1
               for row in model["constraints"])
    assert any(row["name"].startswith("eligibility/") and row["upper"] == 0
               for row in model["constraints"])
    assert model["solver_contract"]["threads"] == 1
    assert model["solver_contract"]["random_seed"] == 0
    MODEL.validate_production_model(instance, first)


def test_production_rejects_false_mask_attestation_and_non_string_state_id():
    instance = _production_instance()
    old = copy.deepcopy(instance)
    old.pop("scientific_masks_are_frozen_search_inputs")
    old["scientific_masks_accessed"] = False
    with pytest.raises(MODEL.GlobalExactModelError, match="surface"):
        MODEL.validate_production_instance(old)
    malformed = copy.deepcopy(instance)
    malformed["fixed_states"][0]["state_id"] = 9
    with pytest.raises(MODEL.GlobalExactModelError, match="identity"):
        MODEL.validate_production_instance(malformed)


def test_self_resigned_solution_tamper_fails_strict_validation():
    model = MODEL.translate_binary_problem(_five_fixtures()[0])
    solution = _solve(model)
    changed = copy.deepcopy(solution)
    changed["binary_values"][0] = 1 - changed["binary_values"][0]
    changed[MODEL.SOLUTION_DIGEST_KEY] = MODEL.canonical_digest({
        key: value for key, value in changed.items()
        if key != MODEL.SOLUTION_DIGEST_KEY
    })
    with pytest.raises(MODEL.GlobalExactModelError):
        MODEL.validate_solution(model, changed)


def test_self_resigned_model_tampering_is_recomputed_not_trusted():
    model = MODEL.translate_binary_problem(_five_fixtures()[0])
    changed_models = []
    for mutate in (
        lambda value: value["variables"][0].__setitem__(
            "objective_coefficient",
            value["variables"][0]["objective_coefficient"] + 1),
        lambda value: value["variables"][0].__setitem__("pair_digest", "0" * 64),
        lambda value: value["variables"][0].__setitem__("upper", 2),
        lambda value: value["variables"][0].__setitem__("integrality", 0),
        lambda value: value.__setitem__("objective_rule", "changed"),
        lambda value: value.__setitem__("variable_order", "changed"),
        lambda value: value["constraints"][0]["terms"][0].__setitem__(1, 7),
        lambda value: value["constraints"][0].__setitem__("upper", 2),
    ):
        changed = copy.deepcopy(model)
        mutate(changed)
        changed[MODEL.MODEL_DIGEST_KEY] = MODEL.canonical_digest({
            key: value for key, value in changed.items()
            if key != MODEL.MODEL_DIGEST_KEY
        })
        changed_models.append(changed)
    for changed in changed_models:
        with pytest.raises(MODEL.GlobalExactModelError):
            MODEL.validate_model(changed)


def test_closed_structural_candidate_rejects_outcome_injection_and_mask_tamper():
    instance = _production_instance()
    scene = instance["optional_scenes"][0]
    raw = {
        **scene["structural_scene_projection"]["raw_candidate"],
        "state_id": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
        "split_role": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
        "candidate_outcome": {"utility": 1.0},
    }
    with pytest.raises(MODEL.GlobalExactModelError, match="surface"):
        MODEL.structural_scene_identity_digest(
            raw,
            completion_rotation_eligibility=
                scene["completion_rotation_eligibility"])
    changed = copy.deepcopy(instance)
    changed["optional_scenes"][0]["completion_rotation_eligibility"][0] = False
    with pytest.raises(MODEL.GlobalExactModelError):
        MODEL.validate_production_instance(changed)


def test_structural_candidate_requires_complete_canonical_builder_boundary():
    instance = _production_instance()
    scene = instance["optional_scenes"][0]
    raw = {
        **scene["structural_scene_projection"]["raw_candidate"],
        "state_id": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
        "split_role": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
    }
    eligibility = scene["completion_rotation_eligibility"]
    expected_keys = {
        "command_block_tick", "decimation_phase",
        "observation_emission_phase_ns", "reset", "terminated", "truncated",
        "source_step", "episode_step", "sim_time_ns", "boundary_digest",
    }
    boundary = raw["boundary"]
    assert set(boundary) == expected_keys
    assert boundary["boundary_digest"] == MODEL.CANONICAL_BOUNDARY_DIGEST
    assert boundary["episode_step"] == raw["source_step"] - 1
    assert boundary["sim_time_ns"] >= 0
    assert boundary["sim_time_ns"] % MODEL.CANONICAL_BOUNDARY_STEP_NS == 0
    assert boundary["sim_time_ns"] != (
        boundary["episode_step"] * MODEL.CANONICAL_BOUNDARY_STEP_NS)
    projection = MODEL.structural_scene_projection(
        raw, completion_rotation_eligibility=eligibility)
    assert projection["raw_candidate"]["boundary"] == boundary
    assert instance["state_identity_lineage"][
        "pre_allocation_identity_static"]["boundary_digest"] == \
        MODEL.CANONICAL_BOUNDARY_DIGEST

    legacy_two_key = copy.deepcopy(raw)
    legacy_two_key["boundary"] = {
        "source_step": raw["source_step"],
        "boundary_digest": MODEL.CANONICAL_BOUNDARY_DIGEST,
    }
    with pytest.raises(
            MODEL.GlobalExactModelError,
            match="canonical boundary changed"):
        MODEL.structural_scene_projection(
            legacy_two_key, completion_rotation_eligibility=eligibility)


def test_identity_lineage_requires_same_canonical_boundary_digest() -> None:
    instance = _production_instance()
    changed = copy.deepcopy(instance)
    changed["state_identity_lineage"]["pre_allocation_identity_static"][
        "boundary_digest"] = "0" * 64
    with pytest.raises(MODEL.GlobalExactModelError, match="lineage changed"):
        MODEL.validate_production_instance(changed)


@pytest.mark.parametrize(("field", "value"), [
    ("command_block_tick", 1),
    ("decimation_phase", 1),
    ("observation_emission_phase_ns", 1),
    ("reset", True),
    ("terminated", True),
    ("truncated", True),
    ("source_step", 99),
    ("episode_step", 100),
    ("sim_time_ns", -MODEL.CANONICAL_BOUNDARY_STEP_NS),
    ("sim_time_ns", 1),
    ("boundary_digest", "0" * 64),
])
def test_structural_candidate_rejects_canonical_boundary_mutation(
        field: str, value: object) -> None:
    instance = _production_instance()
    scene = instance["optional_scenes"][0]
    raw = {
        **scene["structural_scene_projection"]["raw_candidate"],
        "state_id": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
        "split_role": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
    }
    raw["boundary"][field] = value
    with pytest.raises(
            MODEL.GlobalExactModelError,
            match="canonical boundary changed"):
        MODEL.structural_scene_projection(
            raw,
            completion_rotation_eligibility=
                scene["completion_rotation_eligibility"])


def test_mandatory_five_fixture_suite_uses_independent_semantic_control():
    receipt = MODEL.build_fixture_suite_result()
    assert receipt[MODEL.FIXTURE_SUITE_DIGEST_KEY] == \
        MODEL.FROZEN_FIXTURE_SUITE_RESULT_DIGEST
    assert [row["fixture_id"] for row in receipt["fixtures"]] == \
        MODEL.AUTHORITY.FIXTURE_VALIDATION_CONTRACT["required_fixture_ids"]
    assert all(row["solver_feasible"] == row["control_feasible"]
               for row in receipt["fixtures"])
    boundary = next(row for row in receipt["fixtures"]
                    if row["fixture_id"].startswith("MULTIPLE_FEASIBLE"))
    expected = copy.deepcopy(MODEL.AUTHORITY.FIXTURE_VALIDATION_CONTRACT[
        "mandatory_boundary_fixture"])
    expected.pop("fixture_id")
    assert boundary["boundary_predicates"] == expected
    assert MODEL.validate_fixture_suite_result(receipt) == receipt


def test_synthetic_production_one_shot_materializes_legacy_valid_manifest(
        monkeypatch):
    instance = _production_instance()
    plan = MODEL.build_execution_plan(instance)
    assert MODEL.validate_execution_plan(instance, plan) == plan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = MODEL.solve_once(instance, plan)
    assert result["status"] == MODEL.EXECUTION_PASS_STATUS
    assert len(result["selected_scene_indices"]) == 5
    materialized = result["materialized_allocation"]
    disposition = materialized["legacy_allocation_contract_disposition"]
    assert disposition == MODEL.legacy_allocation_contract_disposition()
    assert disposition["status"] == "STRUCTURAL_COMPATIBILITY_PROJECTION_ONLY"
    assert disposition["legacy_choice_rule_status"] == \
        MODEL.AUTHORITY.SUPERSEDED_CANONICAL_TIE_BREAK_STATUS
    assert disposition["standalone_legacy_canonicality_claim_accepted"] is False
    assert [row["assigned_split_role"]
            for row in materialized["selected_scene_rows"]] == [
                "calibration", "fit", "fit", "fit", "fit"]
    lineage = instance["state_identity_lineage"]
    for ordinal, selected_index in enumerate(
            materialized["selected_scene_indices"]):
        state = copy.deepcopy(instance["optional_scenes"][selected_index][
            "structural_scene_projection"]["raw_candidate"])
        state["state_id"] = (
            f"{lineage['pool']}-{MODEL.SMALL_FAMILY}-"
            f"{MODEL.COMPLETION_STRATUM}-{ordinal:02d}")
        state["split_role"] = "calibration" if ordinal == 0 else "fit"
        expected_identity = hashlib.sha256(json.dumps({
            "schema": "go2_branch_state_identity_v1_2",
            "selection_digest": lineage["selection_digest"],
            "scorer_contract_v1_2_digest":
                lineage["scorer_contract_v1_2_digest"],
            "state": state,
        }, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "utf-8")).hexdigest()
        assert materialized["selected_scene_rows"][ordinal][
            "state_identity_digest"] == expected_identity
    manifest = materialized["allocation_manifest"]
    legacy_source_digest = hashlib.sha256(json.dumps(
        materialized["source_identity_manifest_projection"],
        sort_keys=True, ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    assert manifest["source_identity_manifest_digest"] == legacy_source_digest
    assert manifest["source_identity_manifest_digest"] != MODEL.canonical_digest(
        materialized["source_identity_manifest_projection"])
    SELECTOR.validate_allocation_manifest_structure_solve_free(
        manifest,
        expected_source_identity_manifest_digest=
            manifest["source_identity_manifest_digest"])
    assert MODEL.validate_execution_result(instance, plan, result) == result
    production = MODEL.build_production_model(instance)
    forged_control_infeasibility = MODEL._execution_result(
        plan,
        MODEL._build_infeasibility(
            production["model"], solver="exhaustive_binary_control_v1"),
        materialized=None,
    )
    with pytest.raises(MODEL.GlobalExactModelError, match="frozen solver"):
        MODEL.validate_execution_result_solve_free(
            instance, plan, forged_control_infeasibility)
    assert MODEL.validate_solver_runtime_identity_record(
        MODEL.FROZEN_SOLVER_RUNTIME_IDENTITY
    ) == MODEL.FROZEN_SOLVER_RUNTIME_IDENTITY

    def forbidden_live_runtime():
        raise AssertionError("persisted validation imported the live solver")

    monkeypatch.setattr(MODEL, "solver_runtime_identity",
                        forbidden_live_runtime)
    assert MODEL.validate_execution_plan_solve_free(instance, plan) == plan
    assert MODEL.validate_execution_result_solve_free(
        instance, plan, result) == result
    with pytest.raises(AssertionError, match="live solver"):
        MODEL.validate_execution_plan(instance, plan)
    with pytest.raises(AssertionError, match="live solver"):
        MODEL.validate_execution_result(instance, plan, result)


def test_solve_free_plan_replay_does_not_import_local_solver(monkeypatch):
    instance = _production_instance()
    plan = MODEL._build_execution_plan_with_runtime(
        instance, MODEL.FROZEN_SOLVER_RUNTIME_IDENTITY)

    def forbidden_live_runtime():
        raise AssertionError("persisted validation imported the live solver")

    monkeypatch.setattr(MODEL, "solver_runtime_identity",
                        forbidden_live_runtime)
    assert MODEL.validate_execution_plan_solve_free(instance, plan) == plan

    tampered = copy.deepcopy(plan)
    tampered["solver_runtime_identity"]["scipy_version"] = "9.9.9"
    tampered[MODEL.EXECUTION_PLAN_DIGEST_KEY] = MODEL.canonical_digest({
        key: value for key, value in tampered.items()
        if key != MODEL.EXECUTION_PLAN_DIGEST_KEY
    })
    with pytest.raises(MODEL.GlobalExactModelError):
        MODEL.validate_execution_plan_solve_free(instance, tampered)
