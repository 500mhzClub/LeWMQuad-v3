"""Focused, outcome-free tests for the scorer-fit full-bank V2 builder."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from lewm.oracle import go2_scorer_fit_corpus_v2_design as AUTH
from scripts import build_go2_branch_corpus_v1_2 as B


_STATUS = {
    "task_completed": False,
    "goal_claimed": False,
    "terminated": False,
    "truncated": False,
}


def _boundary(source_step: int) -> dict:
    return {
        "command_block_tick": 0,
        "decimation_phase": 0,
        "observation_emission_phase_ns": 0,
        "reset": False,
        "terminated": False,
        "truncated": False,
        "source_step": source_step,
        "episode_step": source_step - 1,
        "sim_time_ns": source_step * 100_000_000,
        "boundary_digest": B.V1.BOUNDARY_DIGEST,
    }


def _goal(*, distance: float = 0.8, bearing: float = 0.0) -> dict:
    return {
        "landmark_id": "goal-0",
        "landmark_cell": 7,
        "material_id": "goal-material",
        "graph_edges": 2,
        "start_geodesic_m": distance,
        "bearing_body_rad": bearing,
        "range_m": distance,
        "landmark_xy_m": [1.0, 2.0],
    }


def _raw_completion(index: int, *, previous=None, status=None) -> dict:
    previous = [0.0, 0.0, 0.0] if previous is None else list(previous)
    status = dict(_STATUS if status is None else status)
    goal = _goal()
    vector = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=goal["graph_edges"],
        reachable=True,
        continuous_geodesic_m=goal["start_geodesic_m"],
        bearing_body_rad=goal["bearing_body_rad"],
        task_status=status,
        previous_applied_command=previous,
    )
    source_step = 41 + index
    return {
        "state_id": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
        "family": "small_enclosed_maze",
        "scene_id": f"small-scene-{index:02d}",
        "scene_dir": f"/synthetic/small-scene-{index:02d}",
        "scene_manifest_sha256": f"{index + 1:064x}",
        "scene_manifest_byte_count": 1000 + index,
        "split": "train",
        "drive_seed": 100 + index,
        "stratum": "completion_enriched",
        "split_role": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
        "warmup_blocks": 8,
        "source_step": source_step,
        "episode_id": index,
        "episode_cluster_id": f"cluster-{index:02d}",
        "cell_id": 10 + index,
        "boundary": _boundary(source_step),
        "goal": goal,
        "goal_type": goal["material_id"],
        "body_clearance_m": 1.0,
        "clearance_m": 1.0,
        "completion_rotation_eligibility_vector": vector,
        "snapshot_task_status": status,
        "previous_applied_command": previous,
    }


def _identity_bindings() -> dict:
    return {
        "selection_digest": "a" * 64,
        "scorer_contract_v1_2_digest": "b" * 64,
    }


def _candidate_checks(rows: list[dict], *, failures=()) -> dict:
    failed = set(failures)
    return {
        row["scene_id"]: {
            "pass": row["scene_id"] not in failed,
            "full_bank_revalidation_digest": hashlib.sha256(
                row["scene_id"].encode()).hexdigest(),
        }
        for row in rows
    }


def test_completion_order_matches_authority_preimage_and_binds_snapshot_inputs(
        ) -> None:
    raw = _raw_completion(0)
    structural = B._full_bank_v2_structural_state_identity(raw)
    goal = B._full_bank_v2_goal_identity(raw["goal"])
    order = B._full_bank_v2_candidate_order_material(
        raw, domain_separator=AUTH.COMPLETION_ORDER_DOMAIN,
        selector_digest=AUTH.ACTIVE_SELECTOR_DIGEST)
    expected_digest, expected_tie = AUTH.completion_order_key(
        structural, goal,
        active_selector_digest=AUTH.ACTIVE_SELECTOR_DIGEST)
    assert order["ordering_digest"] == expected_digest
    assert bytes.fromhex(
        order["structural_identity_tie_break_hex"]) == expected_tie
    assert structural["previous_applied_command"] == [0.0, 0.0, 0.0]
    assert structural["snapshot_task_status"] == _STATUS
    assert "completion_rotation_eligibility_vector" not in structural

    changed_previous = _raw_completion(0, previous=[0.1, 0.0, 0.0])
    changed_order = B._full_bank_v2_candidate_order_material(
        changed_previous, domain_separator=AUTH.COMPLETION_ORDER_DOMAIN,
        selector_digest=AUTH.ACTIVE_SELECTOR_DIGEST)
    assert changed_order["ordering_digest"] != order["ordering_digest"]

    changed_status = copy.deepcopy(raw)
    changed_status["snapshot_task_status"] = {
        **_STATUS, "task_completed": True}
    status_order = B._full_bank_v2_candidate_order_material(
        changed_status, domain_separator=AUTH.COMPLETION_ORDER_DOMAIN,
        selector_digest=AUTH.ACTIVE_SELECTOR_DIGEST)
    assert status_order["ordering_digest"] != order["ordering_digest"]

    retired_mask_only = copy.deepcopy(raw)
    retired_mask_only["completion_rotation_eligibility_vector"] = {
        **retired_mask_only["completion_rotation_eligibility_vector"],
        "eligible_rotation_indices": [],
    }
    same_order = B._full_bank_v2_candidate_order_material(
        retired_mask_only, domain_separator=AUTH.COMPLETION_ORDER_DOMAIN,
        selector_digest=AUTH.ACTIVE_SELECTOR_DIGEST)
    assert same_order == order


def test_structural_boundary_and_snapshot_status_are_closed() -> None:
    raw = _raw_completion(1)
    raw["boundary"]["unexpected"] = 1
    with pytest.raises(RuntimeError, match="canonical snapshot boundary"):
        B._full_bank_v2_structural_state_identity(raw)
    raw = _raw_completion(1)
    raw["snapshot_task_status"].pop("truncated")
    with pytest.raises(RuntimeError, match="snapshot input binding"):
        B._full_bank_v2_structural_state_identity(raw)


def test_full_bank_lmax_uses_all_twelve_without_branch_execution() -> None:
    raw = _raw_completion(2, previous=[0.1, 0.0, 0.2])
    evidence = B.full_bank_completion_reachability_evidence(raw)
    assert evidence["candidate_indices"] == list(range(12))
    assert evidence["candidate_count"] == 12
    assert len(evidence["candidate_path_lengths_m"]) == 12
    assert evidence["l_max_m"] == max(
        row["translational_path_length_m"]
        for row in evidence["candidate_path_lengths_m"])
    assert evidence["eligible"] is True
    assert evidence["branch_execution_used"] is False
    assert evidence["realised_outcome_used"] is False
    assert evidence["legacy_rotation_mask_used_as_active_gate"] is False


def test_deterministic_selection_is_solve_free_and_drops_rotation_vector(
        ) -> None:
    rows = [_raw_completion(index) for index in range(17)]
    kwargs = {
        "raw_candidates": rows,
        "candidate_revalidation": _candidate_checks(rows),
        "identity_bindings": _identity_bindings(),
        "domain_separator": AUTH.COMPLETION_ORDER_DOMAIN,
        "selector_digest": AUTH.ACTIVE_SELECTOR_DIGEST,
        "design_digest": "c" * 64,
        "mask_classification_digest": "d" * 64,
    }
    first = B.deterministic_full_bank_completion_selection(**kwargs)
    second = B.deterministic_full_bank_completion_selection(**kwargs)
    assert first == second
    assert first["optimisation_or_solver_used"] is False
    assert first["historical_rotation_evidence_accessed"] is True
    assert len(first["selected_scene_ids"]) == 5
    assert len(set(first["selected_scene_ids"])) == 5
    assert [row["split_role"] for row in first["selected_states"]].count(
        "calibration") == 1
    assert [row["split_role"] for row in first["selected_states"]].count(
        "fit") == 4
    assert all("completion_rotation_eligibility_vector" not in row
               for row in first["selected_states"])
    assert all(set(row["snapshot_task_status"]) == set(_STATUS)
               for row in first["selected_states"])


def test_deterministic_selection_reports_exact_preoutcome_feasibility_failure(
        ) -> None:
    rows = [_raw_completion(index) for index in range(17)]
    with pytest.raises(B.FullBankV2FeasibilityFailure) as exc_info:
        B.deterministic_full_bank_completion_selection(
            raw_candidates=rows,
            candidate_revalidation=_candidate_checks(
                rows, failures={row["scene_id"] for row in rows[4:]}),
            identity_bindings=_identity_bindings(),
            domain_separator=AUTH.COMPLETION_ORDER_DOMAIN,
            selector_digest=AUTH.ACTIVE_SELECTOR_DIGEST,
            design_digest="c" * 64,
            mask_classification_digest="d" * 64,
        )
    assert exc_info.value.calibration_count == 1
    assert exc_info.value.fit_count == 3
    assert len(exc_info.value.ordered_scene_ids) == 17


def _quota_states() -> list[dict]:
    rows = []
    ordinal = 0
    for family in B.STATE_SELECTOR.REQUIRED_FAMILIES:
        for stratum in B.STRATA:
            for within in range(5):
                rows.append({
                    "state_id": f"state-{ordinal:03d}",
                    "state_identity_digest": f"{ordinal + 1:064x}",
                    "scene_id": f"scene-{ordinal:03d}",
                    "family": family,
                    "stratum": stratum,
                    "split_role": "calibration" if within == 0 else "fit",
                    "goal_type": f"goal-type-{ordinal % 3}",
                })
                ordinal += 1
    return rows


def test_full_bank_assignment_manifest_validates_all_algebraic_counts() -> None:
    states = _quota_states()
    manifest = B.build_full_bank_v2_assignment_manifest(
        states=states, design_digest="c" * 64,
        identity_projection_digest="e" * 64,
        revalidation_digest="f" * 64)
    counts = manifest["algebraic_validation"]
    assert manifest["assignment_count"] == 1_440
    assert manifest["candidate_indices"] == list(range(12))
    assert set(counts["candidate_overall"].values()) == {120}
    assert all(value == {"fit": 96, "calibration": 24}
               for value in counts["candidate_by_split"].values())
    assert all(set(value.values()) == {40}
               for value in counts["candidate_by_stratum"].values())
    assert counts["unordered_candidate_pair_count"] == 66
    assert counts["pairwise_candidate_cooccurrence_exact"] == 120
    assert counts["all_candidate_goal_type_distributions_identical"] is True
    assert manifest["rotation_or_subset_decision_present"] is False


def test_builder_reads_selector_from_frozen_nested_order_contract(
        monkeypatch) -> None:
    monkeypatch.setattr(AUTH, "validate_design_amendment", lambda *_a, **_k: None)
    monkeypatch.setattr(
        AUTH, "validate_rotation_mask_classification",
        lambda *_a, **_k: None)
    design = {
        AUTH.DESIGN_SELF_KEY: "c" * 64,
        "count_contract": AUTH.FULL_BANK_COUNT_CONTRACT,
        "small_completion_selection": AUTH.COMPLETION_ORDERING_CONTRACT,
    }
    classification = {
        AUTH.MASK_CLASSIFICATION_SELF_KEY: "d" * 64,
        "counts": {
            "old_rotation_related_condition_count": 18,
            "partial_subset_allocation_only_count": 18,
            "true_branch_execution_requirement_count": 0,
        },
    }
    _design, _classification, selector, domain = \
        B._full_bank_v2_validate_design_payloads(design, classification)
    assert selector == AUTH.ACTIVE_SELECTOR_DIGEST
    assert domain == AUTH.COMPLETION_ORDER_DOMAIN


def test_active_projection_preserves_custody_identity_but_retires_mask() -> None:
    raw = _raw_completion(4)
    raw["state_identity_digest"] = "9" * 64
    raw.pop("previous_applied_command")
    raw.pop("snapshot_task_status")
    vector = raw.pop("completion_rotation_eligibility_vector")
    projected = B._full_bank_v2_active_state_projection(
        raw, preserved_vectors={"9" * 64: vector})
    assert projected["state_identity_digest"] == "9" * 64
    assert projected["previous_applied_command"] == [0.0, 0.0, 0.0]
    assert projected["snapshot_task_status"] == _STATUS
    assert "completion_rotation_eligibility_vector" not in projected


def test_v2_branch_paths_and_smoke_state_are_versioned_and_fit_only(
        tmp_path) -> None:
    manifest = {
        "schema": B.SCORER_FIT_V2_STATE_MANIFEST_SCHEMA,
        "pool": "scorer_fit_v2",
        "states": [
            {"state_id": "cal", "split_role": "calibration"},
            {"state_id": "fit", "split_role": "fit"},
        ],
    }
    identity = {
        "schema": B.SCORER_FIT_V2_BRANCH_IDENTITY_SCHEMA,
        "branch_identity_digest": "1" * 64,
    }
    assert B._branch_smoke_state(manifest)["state_id"] == "fit"
    assert B._row_path(tmp_path, identity).parent.name == "row_records_v2"
    assert B._compiled_output_paths(manifest, tmp_path) == (
        tmp_path / "branch_rows_v2.jsonl",
        tmp_path / "corpus_receipt_v2.json",
    )
    assert B._branch_frames_root(manifest, tmp_path).name == "frames_v2"
    assert B._branch_smoke_receipt_path(manifest, tmp_path).name == \
        "smoke_branch_receipt_v2.json"


def test_v2_redrive_requires_exact_frozen_previous_command_and_status() -> None:
    entry = _raw_completion(3)
    record = {
        key: copy.deepcopy(entry[key]) for key in (
            "boundary", "cell_id", "goal", "body_clearance_m",
            "clearance_m", "completion_rotation_eligibility_vector",
            "snapshot_task_status", "previous_applied_command",
        )
    }
    ctx = SimpleNamespace(runner=SimpleNamespace(episode_states=[
        SimpleNamespace(episode_id=entry["episode_id"]),
    ]))
    assert B._redrive_mismatch(
        entry, record, ctx, full_bank_v2=True) is None

    changed_previous = copy.deepcopy(record)
    changed_previous["previous_applied_command"][0] = 0.1
    previous_reason = B._redrive_mismatch(
        entry, changed_previous, ctx, full_bank_v2=True)
    assert previous_reason is not None
    assert "previous_applied_command" in previous_reason

    changed_status = copy.deepcopy(record)
    changed_status["snapshot_task_status"]["goal_claimed"] = True
    status_reason = B._redrive_mismatch(
        entry, changed_status, ctx, full_bank_v2=True)
    assert status_reason is not None
    assert "snapshot_task_status" in status_reason


def _v2_runtime_manifest(*, candidate_indices=(0, 1)) -> dict:
    lineage = {
        key: hashlib.sha256(key.encode()).hexdigest()
        for key in B._FULL_BANK_V2_BRANCH_LINEAGE_KEYS
    }
    identities = []
    for candidate_index in candidate_indices:
        identities.append({
            "schema": B.SCORER_FIT_V2_BRANCH_IDENTITY_SCHEMA,
            "candidate": B.V1.CANDIDATE_BANK[candidate_index][0],
            "candidate_index": candidate_index,
            "primitives": list(B.V1.CANDIDATE_BANK[candidate_index][1]),
            "assignment_identity_digest":
                hashlib.sha256(f"assignment-{candidate_index}".encode()).hexdigest(),
            "branch_identity_digest":
                hashlib.sha256(f"branch-{candidate_index}".encode()).hexdigest(),
        })
    return {
        "schema": B.SCORER_FIT_V2_STATE_MANIFEST_SCHEMA,
        "pool": "scorer_fit_v2",
        "state_manifest_digest": "8" * 64,
        "branch_identity_set_digest": B.canonical_digest(sorted(
            row["branch_identity_digest"] for row in identities)),
        "attempted_branch_count_registered": len(candidate_indices),
        **lineage,
        "states": [{
            "state_id": "fit-state",
            "state_identity_digest": "7" * 64,
            "split_role": "fit",
            "candidate_indices": list(candidate_indices),
            "branch_identities": identities,
        }],
    }


def test_v2_compilation_is_versioned_and_zero_new_byte_idempotent(
        tmp_path, monkeypatch) -> None:
    manifest = _v2_runtime_manifest()
    monkeypatch.setattr(B, "_validate_branch_row", lambda *_args: None)
    for candidate_index in (1, 0):
        identity = B._identity_for(
            manifest["states"][0], candidate_index)
        row = {
            "state_id": "fit-state",
            "candidate_index": candidate_index,
            "valid": True,
            "wall_time_s": 1.0,
            "context_frames": [],
            "horizon_frames": [],
            "branch_row_digest": hashlib.sha256(
                f"row-{candidate_index}".encode()).hexdigest(),
        }
        B._write_row(tmp_path, identity, row)
    receipt = B._compile_corpus(
        manifest, tmp_path, invocation_runtime_s=2.0)
    assert receipt["schema"] == B.SCORER_FIT_V2_CORPUS_RECEIPT_SCHEMA
    assert receipt["complete"] is True
    assert (tmp_path / "branch_rows_v2.jsonl").is_file()
    assert (tmp_path / "corpus_receipt_v2.json").is_file()
    assert not (tmp_path / "branch_rows.jsonl").exists()
    assert not (tmp_path / "corpus_receipt.json").exists()
    before_rows = (tmp_path / "branch_rows_v2.jsonl").read_bytes()
    before_receipt = (tmp_path / "corpus_receipt_v2.json").read_bytes()
    assert B._compile_corpus(
        manifest, tmp_path, invocation_runtime_s=999.0) == receipt
    assert (tmp_path / "branch_rows_v2.jsonl").read_bytes() == before_rows
    assert (tmp_path / "corpus_receipt_v2.json").read_bytes() == before_receipt


def test_v2_twelve_branch_smoke_receipt_is_exact_full_bank() -> None:
    manifest = _v2_runtime_manifest(candidate_indices=tuple(range(12)))
    rows = [{
        "candidate_index": index,
        "candidate": B.V1.CANDIDATE_BANK[index][0],
        "valid": True,
        "context_frames": [{}, {}, {}],
        "horizon_frames": [{}, {}, {}, {}],
        "branch_identity_digest": hashlib.sha256(
            f"branch-{index}".encode()).hexdigest(),
        "branch_row_digest": hashlib.sha256(
            f"row-{index}".encode()).hexdigest(),
    } for index in range(12)]
    replay = {
        "state_id": "fit-state",
        "candidate": rows[0]["candidate"],
        "snapshot_digest": "7" * 64,
        "exact_repeat": True,
        "separate_render_scene_physically_inert": True,
    }
    receipt = B._build_smoke_branch_receipt(
        manifest, rows, corpus_digest="6" * 64, replay_check=replay)
    assert receipt["schema"] == B.SCORER_FIT_V2_BRANCH_SMOKE_SCHEMA
    assert receipt["pass"] is True
    assert receipt["candidate_indices"] == list(range(12))
    assert receipt["branch_count"] == 12
    assert receipt["rendered_horizon_frame_count"] == 48
