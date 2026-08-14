"""Pure tests for the final scorer-fit horizon-reachability amendment."""
from __future__ import annotations

import inspect
import hashlib
import json
import math
import copy
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest

from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOCATION
from lewm.oracle import go2_scorer_contract_v1_2 as CONTRACT
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as S


FROZEN_BLOCK = ALLOCATION.ROTATION_BLOCKS[0]


def _status(**overrides):
    row = {
        "task_completed": False,
        "goal_claimed": False,
        "terminated": False,
        "truncated": False,
    }
    row.update(overrides)
    return row


def _full_status(*, goal_cell=7):
    return {
        **_status(),
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
            "fall": False,
            "out_of_bounds": False,
            "tipped": False,
            "nan": False,
        },
    }


def _eligibility(
    *,
    distance=1.0,
    bearing=0.0,
    reachable=True,
    status=None,
    previous=(0.0, 0.0, 0.0),
    candidate_indices=FROZEN_BLOCK,
):
    return S.completion_enriched_eligibility(
        graph_hops=0,
        reachable=reachable,
        continuous_geodesic_m=distance,
        bearing_body_rad=bearing,
        task_status=_status() if status is None else status,
        candidate_indices=candidate_indices,
        previous_applied_command=previous,
    )


def test_outside_radius_is_eligible_when_gap_is_within_exact_l_max():
    budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (0.0, 0.0, 0.0)
    )
    distance = S.COMPLETION_RADIUS_M + budget["l_max_m"]
    evidence = _eligibility(distance=distance)
    assert evidence["continuous_geodesic_m"] > S.COMPLETION_RADIUS_M
    assert evidence["continuous_geodesic_gap_m"] == budget["l_max_m"]
    assert evidence["eligible"] is True


def test_same_state_is_ineligible_when_gap_exceeds_exact_l_max():
    budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (0.0, 0.0, 0.0)
    )
    distance = math.nextafter(
        S.COMPLETION_RADIUS_M + budget["l_max_m"], math.inf
    )
    evidence = _eligibility(distance=distance)
    assert evidence["eligible"] is False
    assert (
        "completion_geodesic_gap_gt_allocated_subset_l_max"
        in evidence["rejection_reasons"]
    )


def test_unclaimed_state_inside_radius_remains_eligible():
    evidence = _eligibility(distance=math.nextafter(0.75, -math.inf))
    assert evidence["continuous_geodesic_gap_m"] == 0.0
    assert evidence["eligible"] is True


@pytest.mark.parametrize(
    "flag", ("task_completed", "goal_claimed", "terminated", "truncated")
)
def test_completed_claimed_or_inactive_state_is_rejected_regardless_of_distance(flag):
    evidence = _eligibility(distance=0.1, status=_status(**{flag: True}))
    assert evidence["eligible"] is False
    assert f"completion_snapshot_{flag}" in evidence["rejection_reasons"]


def test_unreachable_goal_is_rejected():
    evidence = _eligibility(distance=0.1, reachable=False)
    assert evidence["eligible"] is False
    assert "completion_unreachable" in evidence["rejection_reasons"]


def test_bearing_requirement_remains_exactly_75_degrees():
    assert S.COMPLETION_MAX_ABS_BEARING_DEG == 75.0
    assert _eligibility(
        distance=0.1, bearing=math.radians(75.0)
    )["eligible"] is True
    rejected = _eligibility(
        distance=0.1,
        bearing=math.nextafter(math.radians(75.0), math.inf),
    )
    assert rejected["eligible"] is False
    assert "completion_bearing_gt_75deg" in rejected["rejection_reasons"]


def test_l_max_uses_exact_post_slew_plans_from_actual_previous_command():
    standing = S.candidate_post_slew_plan(0, (0.0, 0.0, 0.0))
    reversing = S.candidate_post_slew_plan(0, (-0.2, 0.0, 0.0))
    assert len(standing) == len(reversing) == 20
    # forward_fast is limited from 0.0 to +0.25 at the first tick, but from an
    # actual previous -0.20 command it first reaches only +0.05.
    assert standing[0] == (0.25, 0.0, 0.0)
    assert reversing[0] == pytest.approx((0.05, 0.0, 0.0))
    standing_budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (0.0, 0.0, 0.0)
    )
    reversing_budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (-0.2, 0.0, 0.0)
    )
    assert standing_budget["l_max_m"] == pytest.approx(0.595)
    assert reversing_budget["l_max_m"] == pytest.approx(0.575)
    assert standing_budget["previous_applied_command"] == [0.0, 0.0, 0.0]
    assert reversing_budget["previous_applied_command"] == [-0.2, 0.0, 0.0]


class _OutcomeTripwireStatus(Mapping):
    """Mapping that fails if the selector iterates toward an outcome field."""

    def __getitem__(self, key):
        if key == "branch_outcome":
            raise AssertionError("selector read a branch outcome")
        if key in _status():
            return False
        raise KeyError(key)

    def __iter__(self):
        raise AssertionError("selector iterated over outcome-bearing mapping")

    def __len__(self):
        raise AssertionError("selector inspected outcome-bearing mapping length")


def test_no_branch_outcome_is_read_or_exposed_by_eligibility():
    evidence = _eligibility(distance=0.1, status=_OutcomeTripwireStatus())
    assert evidence["eligible"] is True
    S.validate_no_outcome_surface()
    parameters = inspect.signature(S.completion_enriched_eligibility).parameters
    assert not {
        "branch", "outcome", "collision", "progress", "completion_label",
        "future_frame", "prediction", "latent",
    }.intersection(parameters)


def test_full_snapshot_status_is_strictly_bound_through_four_flag_projection():
    full = _full_status(goal_cell=9)
    projected = _status()
    assert full != projected
    assert S.snapshot_task_status_projection(full) == projected
    assert S.validate_snapshot_task_status_binding(
        full, projected, designated_goal_cell=9
    ) == projected


@pytest.mark.parametrize(
    "flag", ("task_completed", "goal_claimed", "terminated", "truncated")
)
def test_full_snapshot_status_rejects_any_four_flag_projection_change(flag):
    changed = copy.deepcopy(_full_status())
    changed[flag] = True
    with pytest.raises(
        S.StateSelectorAmendmentError, match="selector projection"
    ):
        S.validate_snapshot_task_status_binding(
            changed, _status(), designated_goal_cell=7
        )


@pytest.mark.parametrize("surface", ("claim", "reset", "termination", "goal"))
def test_full_snapshot_status_rejects_production_evidence_tamper(surface):
    changed = copy.deepcopy(_full_status())
    expected_goal = 7
    if surface == "claim":
        changed["production_claim_evidence"][
            "active_collector_claimed_cells"] = [7]
    elif surface == "reset":
        changed["production_task_completion_reset_evidence"][
            "all_scene_landmark_cells_claimed"] = True
    elif surface == "termination":
        changed["termination_flags"]["fall"] = True
    else:
        expected_goal = 8
    with pytest.raises(S.StateSelectorAmendmentError):
        S.validate_snapshot_task_status_binding(
            changed, _status(), designated_goal_cell=expected_goal
        )


def test_candidate_bank_and_allocation_are_unchanged():
    assert S.candidate_bank_contract_digest() == S.CANDIDATE_BANK_DIGEST
    assert S.CANDIDATE_BANK_DIGEST == ALLOCATION.CANDIDATE_BANK_DIGEST
    assert (
        ALLOCATION.allocation_amendment_digest()
        == S.ALLOCATION_AMENDMENT_DIGEST
        == "4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc"
    )
    assert len(ALLOCATION.ROTATION_BLOCKS) == 12
    for rotation, block in enumerate(ALLOCATION.ROTATION_BLOCKS):
        assert S.candidate_rotation_index(block) == rotation
        assert len(block) == 6


def test_actual_selector_parameter_remains_exactly_point_75_without_tolerance():
    assert S.COMPLETION_RADIUS_M == 0.75
    assert S.COMPLETION_MAX_GEODESIC_M == 0.75
    assert S.completion_distance_gap_m(0.75) == 0.0
    assert S.completion_distance_gap_m(math.nextafter(0.75, math.inf)) > 0.0
    contract = S.state_selector_amendment_contract()
    separation = contract["preserved"]["completion_semantic_separation"]
    assert separation["not_interchangeable"] is True
    assert "graph cell" in separation["oracle_v1_2_label"]
    assert "range-envelope" in separation["snapshot_production_goal_claim"]
    assert "selector-only" in separation["r_complete_0_75m"]


def test_preidentity_vector_covers_all_rotations_but_is_not_an_assignment():
    vector = S.completion_rotation_eligibility_vector(
        graph_hops=0,
        reachable=True,
        continuous_geodesic_m=1.2,
        bearing_body_rad=0.0,
        task_status=_status(),
        previous_applied_command=(0.0, 0.0, 0.0),
    )
    assert vector["rotation_count"] == 12
    assert vector["is_candidate_assignment"] is False
    assert vector["pre_identity_fixture_used_as_assignment"] is False
    assert {row["candidate_rotation_index"] for row in vector["rotations"]} == set(
        range(12)
    )
    # The exact L_max is not invariant, proving why an arbitrary fixture mask
    # cannot satisfy the state-specific contract.
    assert len({row["l_max_m"] for row in vector["rotations"]}) > 1


def test_allocated_evidence_validator_recomputes_mask_and_fails_closed():
    evidence = _eligibility(distance=1.0)
    S.validate_allocated_completion_evidence(
        evidence,
        candidate_indices=FROZEN_BLOCK,
        previous_applied_command=(0.0, 0.0, 0.0),
    )
    tampered = dict(evidence)
    tampered["l_max_m"] += 0.01
    with pytest.raises(
        S.StateSelectorAmendmentError,
        match="exact allocated-subset arithmetic",
    ):
        S.validate_allocated_completion_evidence(
            tampered,
            candidate_indices=FROZEN_BLOCK,
            previous_applied_command=(0.0, 0.0, 0.0),
        )


def test_contract_binds_final_amendment_and_deterministic_circularity_resolution():
    contract = S.state_selector_amendment_contract()
    assert contract["superseded_start_distance_rule"]["status"] == (
        "SUPERSEDED_PRE_OUTCOME_START_RADIUS_NOT_HORIZON_REACHABILITY"
    )
    assert contract["freeze_policy"][
        "this_is_final_permitted_pre_outcome_selector_amendment"
    ] is True
    search = contract["allocation_circularity_resolution"][
        "deterministic_search"
    ]
    assert "lexicographic" in search["combination_order"]
    assert "unchanged canonical allocator" in search["per_combination_operation"]
    assert search["candidate_outcomes_consumed"] is False
    assert contract["census_reuse"][
        "actual_allocated_mask_check_required_before_manifest"
    ] is True
    assert contract["census_reuse"]["actual_allocated_mask_check_status"] == (
        "MANDATORY_DEFERRED_TO_JOINT_SEARCH_AND_PHASE2"
    )
    assert contract["lineage"]["frozen_failed_census_receipt"][
        "state_selector_feasibility_receipt_digest"
    ] == "2310c3d1b138b605fda483b39cbd4775479cbcc502a4e3707e7a8670457f54d7"
    assert contract["preserved"][
        "oracle_v1_2_completion_at_or_before_horizon"
    ] is True
    assert "actual_completion_at_or_before_horizon" not in contract["preserved"]
    assert contract["source_bindings"]["platform_command_envelope"][
        "sha256"
    ] == "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189"


@pytest.mark.parametrize("previous", [
    [0.300001, 0.0, 0.0], [-0.300001, 0.0, 0.0],
    [0.0, 0.0, 0.500001], [0.0, 0.0, -0.500001],
])
def test_previous_applied_command_must_fit_frozen_platform_envelope(previous):
    with pytest.raises(
        S.StateSelectorAmendmentError, match="frozen platform envelope"
    ):
        S.max_deterministic_translational_path_length_m(
            list(S.ALLOCATION.ROTATION_BLOCKS[0]), previous)


@pytest.mark.parametrize("observed", [
    S.PLATFORM_EXECUTED_VX_MIN_BINARY32_MPS,
    S.PLATFORM_EXECUTED_VX_MAX_BINARY32_MPS,
])
def test_exact_binary32_vx_endpoint_is_accepted_and_preserved(observed):
    assert abs(abs(observed) - 0.3) < 1e-6
    assert abs(observed) > 0.3
    budget = S.max_deterministic_translational_path_length_m(
        list(S.ALLOCATION.ROTATION_BLOCKS[0]), [observed, 0.0, 0.0])
    assert budget["previous_applied_command"] == [observed, 0.0, 0.0]
    evidence = S.completion_enriched_eligibility(
        graph_hops=1, reachable=True, continuous_geodesic_m=0.8,
        bearing_body_rad=0.0,
        task_status={
            "task_completed": False, "goal_claimed": False,
            "terminated": False, "truncated": False,
        },
        candidate_indices=list(S.ALLOCATION.ROTATION_BLOCKS[0]),
        previous_applied_command=[observed, 0.0, 0.0])
    assert evidence["previous_applied_command"] == [observed, 0.0, 0.0]
    assert evidence["l_max_m"] == budget["l_max_m"]
    candidate_index, delta = ((10, -0.25) if observed > 0.0 else (0, 0.25))
    first_post_slew = S.candidate_post_slew_plan(
        candidate_index, [observed, 0.0, 0.0])[0][0]
    assert first_post_slew == observed + delta


def _outward_binary32(value):
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    return float(struct.unpack("!f", struct.pack("!I", bits + 1))[0])


@pytest.mark.parametrize("endpoint", [
    S.PLATFORM_EXECUTED_VX_MIN_BINARY32_MPS,
    S.PLATFORM_EXECUTED_VX_MAX_BINARY32_MPS,
])
def test_next_outward_binary32_vx_value_is_rejected(endpoint):
    outward = _outward_binary32(endpoint)
    assert abs(outward) > abs(endpoint)
    with pytest.raises(
        S.StateSelectorAmendmentError, match="frozen platform envelope"
    ):
        S.max_deterministic_translational_path_length_m(
            list(S.ALLOCATION.ROTATION_BLOCKS[0]), [outward, 0.0, 0.0])


@pytest.mark.parametrize("endpoint,direction", [
    (S.PLATFORM_EXECUTED_VX_MAX_BINARY32_MPS, math.inf),
    (S.PLATFORM_EXECUTED_VX_MIN_BINARY32_MPS, -math.inf),
])
def test_value_one_float64_step_outside_binary32_vx_endpoint_is_rejected(
        endpoint, direction):
    outward = math.nextafter(endpoint, direction)
    with pytest.raises(
        S.StateSelectorAmendmentError, match="frozen platform envelope"
    ):
        S.max_deterministic_translational_path_length_m(
            list(S.ALLOCATION.ROTATION_BLOCKS[0]), [outward, 0.0, 0.0])


@pytest.mark.parametrize("yaw,direction", [(0.5, math.inf), (-0.5, -math.inf)])
def test_value_one_float64_step_outside_yaw_endpoint_is_rejected(
        yaw, direction):
    with pytest.raises(
        S.StateSelectorAmendmentError, match="frozen platform envelope"
    ):
        S.max_deterministic_translational_path_length_m(
            list(S.ALLOCATION.ROTATION_BLOCKS[0]),
            [0.0, 0.0, math.nextafter(yaw, direction)])


def test_binary32_input_representation_fix_does_not_amend_frozen_contract():
    assert S.PLATFORM_EXECUTED_VX_MIN_BINARY32_MPS == \
        -0.30000001192092896
    assert S.PLATFORM_EXECUTED_VX_MAX_BINARY32_MPS == \
        0.30000001192092896
    assert S.state_selector_amendment_digest() == (
        "8c1d9f5ff1430fda6d9d80512afdba3070c78301befa57604aafcad9cb5c880b"
    )
    assert "previous_command_execution_representation" not in (
        S.state_selector_amendment_contract()["single_replacement"][
            "l_max_calculation"])
    artifact = Path(S.AMENDMENT_ARTIFACT_PATH)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "e1ddafcf700009ef07865f7afb88d8ef8967c9b8d4ae3584135f2c9fb80ea9e5"
    )


def test_new_receipt_paths_cannot_overwrite_accepted_v1_failures():
    assert S.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME != (
        "state_selector_feasibility_receipt.json"
    )
    assert "reachability" in S.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME
    assert "reachability" in S.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME
    assert tuple(S.ACTIVE_SELECTOR_BINDING_KEYS) == (
        "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest",
        "preserved_state_revalidation_receipt_digest",
    )


def _managed_generated_fixture(tmp_path: Path):
    repository = tmp_path / "repository"
    generated_parent = repository / ".generated"
    generated_parent.mkdir(parents=True)
    storage_root = (
        tmp_path / "storage" / S.MANAGED_GENERATED_ROOT_RELATIVE.name
    )
    (storage_root / "scorer_fit").mkdir(parents=True)
    alias = generated_parent / S.MANAGED_GENERATED_ROOT_RELATIVE.name
    alias.symlink_to(storage_root, target_is_directory=True)
    return repository, alias, storage_root


def test_managed_generated_guard_allows_only_root_alias_and_pins_target(
        tmp_path):
    repository, alias, storage_root = _managed_generated_fixture(tmp_path)
    artifact = storage_root / "scorer_fit" / "receipt.json"
    artifact.write_text("original")
    guarded = S._managed_generated_artifact_path(
        repository / S.MANAGED_GENERATED_ROOT_RELATIVE / "scorer_fit/receipt.json",
        root=repository,
    )
    assert guarded == artifact

    alternate = tmp_path / "alternate" / S.MANAGED_GENERATED_ROOT_RELATIVE.name
    (alternate / "scorer_fit").mkdir(parents=True)
    (alternate / "scorer_fit/receipt.json").write_text("redirected")
    alias.unlink()
    alias.symlink_to(alternate, target_is_directory=True)
    # The caller reads the canonical object selected before the alias swap.
    assert guarded.read_text() == "original"


@pytest.mark.parametrize("kind", ("nested", "leaf"))
def test_managed_generated_guard_rejects_descendant_symlinks(tmp_path, kind):
    repository, _alias, storage_root = _managed_generated_fixture(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    if kind == "nested":
        (external / "receipt.json").write_text("{}")
        (storage_root / "scorer_fit").rmdir()
        (storage_root / "scorer_fit").symlink_to(
            external, target_is_directory=True)
    else:
        (external / "receipt.json").write_text("{}")
        (storage_root / "scorer_fit/receipt.json").symlink_to(
            external / "receipt.json"
        )
    with pytest.raises(
            S.StateSelectorAmendmentError, match="symlinked generated artifact"):
        S._managed_generated_artifact_path(
            repository / S.MANAGED_GENERATED_ROOT_RELATIVE
            / "scorer_fit/receipt.json",
            root=repository,
        )


def test_managed_generated_guard_rejects_custody_names_lexically_and_resolved(
        tmp_path):
    repository, alias, _storage_root = _managed_generated_fixture(tmp_path)
    with pytest.raises(
            S.StateSelectorAmendmentError, match="custody component"):
        S._managed_generated_artifact_path(
            repository / S.MANAGED_GENERATED_ROOT_RELATIVE
            / "sealed_payload/receipt.json",
            root=repository,
        )

    alias.unlink()
    sealed_target = (
        tmp_path / "sealed_storage" / S.MANAGED_GENERATED_ROOT_RELATIVE.name
    )
    sealed_target.mkdir(parents=True)
    alias.symlink_to(sealed_target, target_is_directory=True)
    with pytest.raises(
            S.StateSelectorAmendmentError, match="inaccessible"):
        S._managed_generated_artifact_path(
            repository / S.MANAGED_GENERATED_ROOT_RELATIVE
            / "scorer_fit/receipt.json",
            root=repository,
        )


def test_exact_frozen_json_loader_uses_managed_guard_for_leaf_symlink(tmp_path):
    repository, _alias, storage_root = _managed_generated_fixture(tmp_path)
    external = tmp_path / "outside.json"
    external.write_text("{}")
    leaf = storage_root / "scorer_fit/receipt.json"
    leaf.symlink_to(external)
    binding = {
        "path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/receipt.json"
        ),
        "byte_count": external.stat().st_size,
        "raw_sha256": S._file_sha256(external),
    }
    with pytest.raises(
            S.StateSelectorAmendmentError, match="symlinked generated artifact"):
        S._load_exact_frozen_json(binding, root=repository, label="synthetic")


def test_mixed_disposition_loader_rejects_leaf_symlink_before_json_validation(
        monkeypatch, tmp_path):
    repository, _alias, storage_root = _managed_generated_fixture(tmp_path)
    external = tmp_path / "outside.json"
    external.write_text("{}")
    relative = Path(
        S.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
    ).relative_to(S.MANAGED_GENERATED_ROOT_RELATIVE)
    leaf = storage_root / relative
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.symlink_to(external)
    validate = monkeypatch.setattr(
        S,
        "validate_preserved_state_mixed_precontract_disposition_receipt",
        lambda *_args, **_kwargs: pytest.fail(
            "JSON validator ran before custody rejection"
        ),
    )
    assert validate is None
    with pytest.raises(
            S.StateSelectorAmendmentError, match="symlinked generated artifact"):
        S.load_and_validate_preserved_state_mixed_precontract_disposition_receipt(
            root=repository
        )


def test_contract_issuer_uses_central_guarded_receipt_loaders(
        monkeypatch, tmp_path):
    source = {
        "source_repository_commit": "c" * 40,
        "bound_implementations_digest": "b" * 64,
    }
    feasibility = {
        "state_selector_feasibility_receipt_digest": "f" * 64,
    }
    monkeypatch.setattr(CONTRACT, "clean_source_binding", lambda: source)
    monkeypatch.setattr(
        CONTRACT, "_managed_scorer_contract_output_path",
        lambda _path: tmp_path / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME,
    )
    monkeypatch.setattr(S, "validate_authority_artifacts", lambda: None)
    monkeypatch.setattr(
        S, "validate_frozen_reachability_feasibility_pass",
        lambda **_kwargs: feasibility,
    )
    guarded = monkeypatch.setattr(
        S,
        "load_and_validate_preserved_state_mixed_precontract_disposition_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(
            S.StateSelectorAmendmentError("guarded leaf symlink")
        ),
    )
    assert guarded is None
    monkeypatch.setattr(
        CONTRACT,
        "_stream_file_sha256",
        lambda *_args, **_kwargs: pytest.fail(
            "checkpoint/source bytes opened before selector custody gate"
        ),
    )
    with pytest.raises(S.StateSelectorAmendmentError, match="guarded leaf symlink"):
        CONTRACT.issue_contract(tmp_path / "contract.json")


def _managed_scorer_package_fixture(tmp_path: Path):
    repository = tmp_path / "repository"
    generated = repository / ".generated"
    generated.mkdir(parents=True)
    storage = (
        tmp_path / "storage" / CONTRACT.SCORER_PACKAGE_ROOT_RELATIVE.name
    )
    storage.mkdir(parents=True)
    alias = generated / CONTRACT.SCORER_PACKAGE_ROOT_RELATIVE.name
    alias.symlink_to(storage, target_is_directory=True)
    return repository, alias, storage


def test_scorer_contract_output_guard_accepts_only_exact_root_alias_and_pins(
        tmp_path):
    repository, alias, storage = _managed_scorer_package_fixture(tmp_path)
    logical = (
        repository / CONTRACT.SCORER_PACKAGE_ROOT_RELATIVE
        / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME
    )
    pinned = CONTRACT._managed_scorer_contract_output_path(
        logical, root=repository
    )
    assert pinned == storage / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME

    alternate = (
        tmp_path / "alternate" / CONTRACT.SCORER_PACKAGE_ROOT_RELATIVE.name
    )
    alternate.mkdir(parents=True)
    alias.unlink()
    alias.symlink_to(alternate, target_is_directory=True)
    pinned.write_text("pinned")
    assert (storage / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME).read_text() == (
        "pinned"
    )
    assert not (alternate / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME).exists()


def test_scorer_contract_output_guard_rejects_wrong_root_and_escape(tmp_path):
    repository, _alias, _storage = _managed_scorer_package_fixture(tmp_path)
    for path in (
        repository / ".generated/other/scorer_contract_v1_2.json",
        repository / CONTRACT.SCORER_PACKAGE_ROOT_RELATIVE / "other.json",
    ):
        with pytest.raises(RuntimeError, match="exact managed package artifact"):
            CONTRACT._managed_scorer_contract_output_path(
                path, root=repository
            )


@pytest.mark.parametrize("kind", ("nested", "leaf"))
def test_scorer_contract_output_guard_rejects_descendant_symlink(
        tmp_path, kind):
    repository, _alias, storage = _managed_scorer_package_fixture(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    if kind == "nested":
        # A nested component cannot occur on the exact active filename, so
        # exercise the archive path that production supersession would use.
        (storage / "superseded_pre_run").symlink_to(
            external, target_is_directory=True
        )
        active = storage / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME
        active.write_text("{}")
        with pytest.raises(RuntimeError, match="symlinked scorer-package"):
            CONTRACT._prepare_contract_output(
                active,
                {"contract_artifact_digest": "bad"},
                managed_root=storage,
            )
    else:
        target = external / "contract.json"
        target.write_text("{}")
        (storage / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME).symlink_to(target)
        logical = (
            repository / CONTRACT.SCORER_PACKAGE_ROOT_RELATIVE
            / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME
        )
        with pytest.raises(RuntimeError, match="symlinked scorer-package"):
            CONTRACT._managed_scorer_contract_output_path(
                logical, root=repository
            )


def test_atomic_contract_write_rejects_predictable_temp_leaf_symlink(tmp_path):
    repository, _alias, storage = _managed_scorer_package_fixture(tmp_path)
    logical = (
        repository / CONTRACT.SCORER_PACKAGE_ROOT_RELATIVE
        / CONTRACT.SCORER_CONTRACT_ARTIFACT_NAME
    )
    output = CONTRACT._managed_scorer_contract_output_path(
        logical, root=repository
    )
    external = tmp_path / "external-contract.json"
    external.write_text("unchanged")
    temporary = output.with_name(f".{output.name}.tmp-{CONTRACT.os.getpid()}")
    temporary.symlink_to(external)
    payload = {"schema": "synthetic"}
    payload["contract_artifact_digest"] = CONTRACT._digest(payload)
    with pytest.raises(RuntimeError, match="symlinked scorer-package"):
        CONTRACT._atomic_write_contract_output(
            output, payload, managed_root=storage
        )
    assert external.read_text() == "unchanged"
    assert not output.exists()


def test_frozen_phase1_failure_derives_exact_mixed_disposition():
    failure = S.validate_frozen_preserved_precontract_failure()
    sets = S.mixed_precontract_disposition_sets()
    assert failure["status"] == "FAIL_PRECONTRACT_IDENTITY_REVALIDATION"
    assert failure["failure_count"] == 8
    assert len(sets["retained_predecessor_identities"]) == 37
    assert len(sets["rejected_predecessor_identities"]) == 8
    assert len(sets["replacement_slots"]) == 8
    assert all(row["stratum"] == "completion_enriched"
               for row in sets["rejected_predecessor_identities"])
    assert {
        row["state_identity_digest"]
        for row in sets["retained_predecessor_identities"]
    }.isdisjoint({
        row["state_identity_digest"]
        for row in sets["rejected_predecessor_identities"]
    })
    vectors = S._phase1_completion_vectors(root=S.ROOT)
    assert len(vectors) == 7
    assert set(vectors) == {
        row["state_identity_digest"]
        for row in sets["retained_predecessor_identities"]
        if row["stratum"] == "completion_enriched"
    }


def test_mixed_disposition_is_nonoverwriting_self_bound_and_outcome_free():
    failure = S.validate_frozen_preserved_precontract_failure()
    absence = failure["outcome_surface_absence_attestation"]
    receipt = S.build_preserved_state_mixed_precontract_disposition_receipt(
        source_repository_commit="c" * 40,
        clean_source_binding_digest="d" * 64,
        bound_implementations_digest="e" * 64,
        successor_selection_digest="f" * 64,
        outcome_surface_absence_attestation=absence,
    )
    S.validate_preserved_state_mixed_precontract_disposition_receipt(
        receipt,
        expected_source_commit="c" * 40,
        expected_clean_source_binding_digest="d" * 64,
        expected_bound_implementations_digest="e" * 64,
        expected_successor_selection_digest="f" * 64,
    )
    assert S.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH != (
        S.PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH
    )
    assert receipt["retained_predecessor_state_count"] == 37
    assert receipt["rejected_predecessor_state_count"] == 8
    assert receipt["replacement_slot_count"] == 8
    assert receipt["candidate_outcomes_loaded"] is False
    assert receipt["branches_attempted"] == 0

    changed = copy.deepcopy(receipt)
    changed["retained_predecessor_identities"][0]["scene_id"] = "substituted"
    changed["retained_predecessor_identity_set_digest"] = S._sha256(
        changed["retained_predecessor_identities"]
    )
    changed["mixed_precontract_disposition_receipt_digest"] = S._sha256({
        key: value for key, value in changed.items()
        if key != "mixed_precontract_disposition_receipt_digest"
    })
    with pytest.raises(S.StateSelectorAmendmentError, match="identities changed"):
        S.validate_preserved_state_mixed_precontract_disposition_receipt(changed)


def test_tracked_amendment_artifact_and_authority_chain_are_exact():
    artifact = json.loads(Path(S.AMENDMENT_ARTIFACT_PATH).read_text())
    S.validate_state_selector_amendment_artifact(artifact)
    assert artifact["state_selector_amendment_digest"] == (
        S.state_selector_amendment_digest()
    )
    S.validate_authority_artifacts()


def test_successor_scorer_contract_binds_v2_and_immediate_predecessor():
    selection = CONTRACT.CORPUS_SELECTION_CONTRACT
    assert selection["predecessor_selection_digest"] == (
        S.PREDECESSOR_SUCCESSOR_SELECTION_DIGEST
    )
    assert selection["state_selector_amendment_digest"] == (
        S.state_selector_amendment_digest()
    )
    assert "first lexicographically feasible" in selection["scorer_fit"]
    assert "retains 37 exact predecessor identities" in selection["scorer_fit"]
    assert "fills eight completion vacancies" in selection["scorer_fit"]
    assert "other four non-small successor families remain unchanged" in (
        selection["scorer_fit"]
    )
    assert "seven families retain" not in selection["scorer_fit"]
    assert selection["preserved_state_revalidation_receipt"][
        "expected_completion_enriched_state_count"
    ] == 40
    bindings = CONTRACT.source_bindings()
    assert bindings["state_selector_amendment_implementation"]["path"].endswith(
        "go2_scorer_state_selector_amendment_v2.py"
    )
    assert bindings["qualified_development_transfer_consumer"]["path"] == (
        "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py"
    )


def _synthetic_mixed_phase2(monkeypatch, tmp_path):
    """Build a source-only 120-state phase-2 fixture with eight replacements."""

    def state(index, *, state_id, scene_id, stratum, family, split_role,
              source_step=None):
        step = 1 + 5 * index if source_step is None else source_step
        row = {
            "state_identity_digest": f"{index + 1:064x}",
            "state_id": state_id,
            "scene_id": scene_id,
            "scene_dir": f"/synthetic/{scene_id}",
            "scene_manifest_sha256": f"{index + 1000:064x}",
            "scene_manifest_byte_count": 100 + index,
            "family": family,
            "stratum": stratum,
            "split_role": split_role,
            "split": "synthetic",
            "drive_seed": index,
            "warmup_blocks": index + 1,
            "source_step": step,
            "episode_id": index + 1,
            "episode_cluster_id": f"cluster-{index:03d}",
            "cell_id": index + 10,
            "boundary": {
                "source_step": step,
                "boundary_digest": "b" * 64,
            },
            "goal": {
                "landmark_id": f"goal-{index}",
                "landmark_cell": index + 20,
                "material_id": "landmark_red",
                "graph_edges": 0,
                "start_geodesic_m": 0.5,
                "bearing_body_rad": 0.0,
                "range_m": 0.5,
                "landmark_xy_m": [0.0, 0.0],
            },
            "goal_type": "landmark_red",
            "body_clearance_m": 0.2,
            "clearance_m": 0.2,
        }
        if stratum == "completion_enriched":
            full_status = _full_status(goal_cell=index + 20)
            vector = S.completion_rotation_eligibility_vector(
                graph_hops=0,
                reachable=True,
                continuous_geodesic_m=0.5,
                bearing_body_rad=0.0,
                task_status=full_status,
                previous_applied_command=[0.0, 0.0, 0.0],
            )
            row.update({
                "completion_rotation_eligibility_vector": vector,
                "snapshot_task_status": full_status,
                "previous_applied_command": [0.0, 0.0, 0.0],
            })
        return row

    retained = []
    retained_sources = []
    active = []
    next_index = 0
    for ordinal in range(37):
        stratum = "completion_enriched" if ordinal < 7 else "general"
        row = state(
            next_index,
            state_id=f"retained-{ordinal:02d}",
            scene_id=f"retained-scene-{ordinal:02d}",
            stratum=stratum,
            family=f"family-{ordinal % 8}",
            split_role="calibration" if ordinal % 5 == 0 else "fit",
        )
        next_index += 1
        retained_sources.append(copy.deepcopy(row))
        active.append(row)
        retained.append(S._mixed_identity_row(row))

    rejected_sources = []
    rejected = []
    slots = []
    replacements = []
    for ordinal in range(8):
        state_id = f"replacement-{ordinal:02d}"
        scene_id = f"rejected-scene-{ordinal:02d}"
        predecessor = state(
            500 + ordinal,
            state_id=state_id,
            scene_id=scene_id,
            stratum="completion_enriched",
            family=f"family-{ordinal}",
            split_role="calibration" if ordinal == 0 else "fit",
            source_step=101 + 5 * ordinal,
        )
        rejected_sources.append(predecessor)
        rejected.append(S._mixed_identity_row(
            predecessor,
            failure_reason=(
                "RuntimeError:amended classification failed: "
                "no_completion_enriched_goal"
            ),
        ))
        slots.append({
            "state_id": state_id,
            "family": predecessor["family"],
            "stratum": "completion_enriched",
            "split_role": predecessor["split_role"],
            "predecessor_state_identity_digest":
                predecessor["state_identity_digest"],
            "predecessor_scene_id": scene_id,
        })
        replacement = state(
            next_index,
            state_id=state_id,
            # The authorised policy permits the failed scene when the snapshot
            # itself differs.  The new source step/boundary prove that here.
            scene_id=scene_id,
            stratum="completion_enriched",
            family=predecessor["family"],
            split_role=predecessor["split_role"],
            source_step=102 + 5 * ordinal,
        )
        next_index += 1
        replacements.append(replacement)
        active.append(replacement)

    for ordinal in range(75):
        stratum = "completion_enriched" if ordinal < 25 else (
            "general" if ordinal < 50 else "safety_enriched"
        )
        row = state(
            next_index,
            state_id=f"new-{ordinal:02d}",
            scene_id=f"new-scene-{ordinal:02d}",
            stratum=stratum,
            family=f"family-{ordinal % 8}",
            split_role="calibration" if ordinal % 5 == 0 else "fit",
        )
        next_index += 1
        active.append(row)
    assert len(active) == 120
    assert sum(row["stratum"] == "completion_enriched" for row in active) == 40

    mixed = {
        "retained_predecessor_identities": retained,
        "rejected_predecessor_identities": rejected,
        "replacement_slots": slots,
        "rejected_predecessor_identity_set_digest": S._sha256(rejected),
        "mixed_precontract_disposition_receipt_digest": "d" * 64,
    }
    path = tmp_path / S.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(mixed))

    assignments = []
    for ordinal, row in enumerate(active):
        rotation = ordinal % ALLOCATION.CANDIDATE_COUNT
        assignments.append({
            "state_identity_digest": row["state_identity_digest"],
            "state_id": row["state_id"],
            "family": row["family"],
            "stratum": row["stratum"],
            "split_role": row["split_role"],
            "goal_type": row["goal_type"],
            "candidate_indices": list(ALLOCATION.ROTATION_BLOCKS[rotation]),
            "rotation_index": rotation,
        })
    allocation = {
        "assignments": assignments,
        "allocation_manifest_digest": "a" * 64,
        "source_identity_manifest_digest": "b" * 64,
        "post_identity_pre_outcome_validation": {
            "post_identity_validation_digest": "c" * 64,
        },
    }
    by_identity = {
        row["state_identity_digest"]: row for row in assignments
    }
    completion_rows = [
        S._completion_source_row_from_active_state(
            row,
            assignment=by_identity[row["state_identity_digest"]],
            preserved_vectors={},
        )
        for row in active if row["stratum"] == "completion_enriched"
    ]
    monkeypatch.setattr(S.ALLOCATION, "validate_allocation_manifest", lambda *_: None)
    monkeypatch.setattr(
        S, "validate_preserved_state_mixed_precontract_disposition_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(S, "_phase1_completion_vectors", lambda **_kwargs: {})
    monkeypatch.setattr(
        S,
        "load_preserved_state_shards",
        lambda *_args, **_kwargs: {
            "synthetic": {"states": retained_sources + rejected_sources},
        },
    )
    return active, replacements, rejected_sources, allocation, completion_rows


def _solve_free_certified_phase2(monkeypatch, tmp_path):
    """Upgrade the phase-2 fixture to one pure, fully valid allocation."""

    active, replacements, rejected, _allocation, _rows = \
        _synthetic_mixed_phase2(monkeypatch, tmp_path)
    strata = tuple(ALLOCATION.STRATA)
    slots = [
        (f"family-{family_index}", stratum, role, ordinal)
        for family_index in range(8)
        for stratum in strata
        for ordinal, role in (
            (0, "calibration"), (1, "fit"), (2, "fit"), (3, "fit"),
            (4, "fit"),
        )
    ]
    replacement_slots = {
        f"replacement-{family_index:02d}": (
            f"family-{family_index}",
            "completion_enriched",
            "calibration" if family_index == 0 else "fit",
            0 if family_index == 0 else 1,
        )
        for family_index in range(8)
    }
    remaining_slots = [
        slot for slot in slots if slot not in set(replacement_slots.values())
    ]
    assigned_slots: dict[str, tuple[str, str, str, int]] = {}
    for state in active:
        state_id = str(state["state_id"])
        slot = replacement_slots.get(state_id)
        if slot is None:
            slot = remaining_slots.pop(0)
        assigned_slots[state_id] = slot
        family, stratum, split_role, _ordinal = slot
        state["family"] = family
        state["stratum"] = stratum
        state["split_role"] = split_role
        state["goal_type"] = str(state["goal"]["material_id"])
        if (
            stratum == "completion_enriched"
            and "completion_rotation_eligibility_vector" not in state
        ):
            status = _full_status(
                goal_cell=int(state["goal"]["landmark_cell"])
            )
            state.update({
                "completion_rotation_eligibility_vector":
                    S.completion_rotation_eligibility_vector(
                        graph_hops=0,
                        reachable=True,
                        continuous_geodesic_m=0.5,
                        bearing_body_rad=0.0,
                        task_status=status,
                        previous_applied_command=[0.0, 0.0, 0.0],
                    ),
                "snapshot_task_status": status,
                "previous_applied_command": [0.0, 0.0, 0.0],
            })
    assert not remaining_slots

    retained_states = [
        copy.deepcopy(state) for state in active
        if str(state["state_id"]).startswith("retained-")
    ]
    replacement_by_id = {
        str(state["state_id"]): state for state in replacements
    }
    for predecessor in rejected:
        replacement = replacement_by_id[str(predecessor["state_id"])]
        for key in ("family", "stratum", "split_role", "goal_type"):
            predecessor[key] = replacement[key]
    rejected_rows = [
        S._mixed_identity_row(
            predecessor,
            failure_reason=(
                "RuntimeError:amended classification failed: "
                "no_completion_enriched_goal"
            ),
        )
        for predecessor in rejected
    ]
    mixed = {
        "retained_predecessor_identities": [
            S._mixed_identity_row(state) for state in retained_states
        ],
        "rejected_predecessor_identities": rejected_rows,
        "replacement_slots": [{
            "state_id": str(predecessor["state_id"]),
            "family": str(predecessor["family"]),
            "stratum": str(predecessor["stratum"]),
            "split_role": str(predecessor["split_role"]),
            "predecessor_state_identity_digest": str(
                predecessor["state_identity_digest"]
            ),
            "predecessor_scene_id": str(predecessor["scene_id"]),
        } for predecessor in rejected],
        "rejected_predecessor_identity_set_digest": S._sha256(rejected_rows),
        "mixed_precontract_disposition_receipt_digest": "d" * 64,
    }
    mixed_path = (
        tmp_path / S.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH
    )
    mixed_path.write_text(json.dumps(mixed))
    monkeypatch.setattr(
        S,
        "load_preserved_state_shards",
        lambda *_args, **_kwargs: {
            "synthetic": {"states": retained_states + rejected},
        },
    )

    calibration_base = (0, 6, 1, 7, 2, 8, 3, 9)
    calibration_offset = {
        "general": 0,
        "safety_enriched": 6,
        "completion_enriched": 2,
    }
    fit_rotations = (0, 6, 1, 7)
    assignments = []
    for state in active:
        family, stratum, split_role, ordinal = assigned_slots[str(
            state["state_id"]
        )]
        family_index = int(family.rsplit("-", 1)[1])
        rotation = (
            (calibration_base[family_index] + calibration_offset[stratum]) % 12
            if split_role == "calibration"
            else fit_rotations[ordinal - 1]
        )
        assignments.append({
            "state_id": str(state["state_id"]),
            "state_identity_digest": str(state["state_identity_digest"]),
            "family": family,
            "stratum": stratum,
            "split_role": split_role,
            "goal_type": str(state["goal_type"]),
            "rotation_index": rotation,
            "candidate_indices": list(ALLOCATION.candidate_block(rotation)),
        })
    assignments.sort(key=lambda row: (
        row["state_identity_digest"], row["state_id"]
    ))
    identity_rows = [{
        key: row[key] for key in (
            "state_id", "state_identity_digest", "family", "stratum",
            "split_role", "goal_type",
        )
    } for row in assignments]
    allocation = {
        "schema": ALLOCATION.SCHEMA,
        "status": ALLOCATION.STATUS,
        "source_identity_manifest_digest": "b" * 64,
        "pre_outcome_identity_digest":
            ALLOCATION.pre_outcome_identity_digest(identity_rows),
        "allocation_contract": ALLOCATION.algorithm_contract(),
        "allocation_contract_digest": ALLOCATION.allocation_contract_digest(),
        "allocation_amendment": ALLOCATION.allocation_amendment_contract(),
        "allocation_amendment_digest":
            ALLOCATION.allocation_amendment_digest(),
        "assignments": assignments,
        "contingency_tables": ALLOCATION._contingency_tables(assignments),
        "post_identity_pre_outcome_validation":
            ALLOCATION._post_identity_pre_outcome_validation(assignments),
    }
    allocation["allocation_manifest_digest"] = \
        ALLOCATION.allocation_manifest_digest(allocation)
    by_identity = {
        row["state_identity_digest"]: row for row in assignments
    }
    completion_rows = [
        S._completion_source_row_from_active_state(
            state,
            assignment=by_identity[str(state["state_identity_digest"])],
            preserved_vectors={},
        )
        for state in active if state["stratum"] == "completion_enriched"
    ]
    assert len(completion_rows) == 40
    return active, allocation, completion_rows


def _phase2_build_arguments(active, allocation, completion_rows, tmp_path):
    return {
        "allocation_manifest": allocation,
        "active_states": active,
        "completion_states": completion_rows,
        "source_repository_commit": "c" * 40,
        "successor_selection_digest": "e" * 64,
        "state_selector_feasibility_receipt_digest":
            S.FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"],
        "mixed_precontract_disposition_receipt_digest": "d" * 64,
        "root": tmp_path,
    }


def test_solve_free_certified_phase2_build_and_validate_are_byte_identical(
        monkeypatch, tmp_path):
    active, allocation, completion_rows = \
        _solve_free_certified_phase2(monkeypatch, tmp_path)
    arguments = _phase2_build_arguments(
        active, allocation, completion_rows, tmp_path
    )
    legacy = S.build_preserved_state_revalidation_receipt(**arguments)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("allocation solve/revalidation was reached")

    monkeypatch.setattr(
        S.ALLOCATION, "validate_allocation_manifest", forbidden
    )
    monkeypatch.setattr(S.ALLOCATION, "build_allocation_manifest", forbidden)
    monkeypatch.setattr(S.ALLOCATION, "_lexicographic_rotations", forbidden)
    monkeypatch.setattr(S.ALLOCATION, "_constraint_system", forbidden)
    import scipy.optimize as scipy_optimize
    monkeypatch.setattr(scipy_optimize, "milp", forbidden)
    certified_calls = []

    def certify(candidate):
        assert candidate == allocation
        certified_calls.append(candidate["allocation_manifest_digest"])
        return copy.deepcopy(allocation)

    observed = (
        S.build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            **arguments,
            certify_allocation_solve_free=certify,
        )
    )
    assert observed == legacy
    S.validate_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
        observed,
        allocation_manifest=allocation,
        active_states=active,
        certify_allocation_solve_free=certify,
        expected_source_commit="c" * 40,
        expected_successor_selection_digest="e" * 64,
        expected_feasibility_receipt_digest=
            S.FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"],
        expected_mixed_precontract_disposition_receipt_digest="d" * 64,
        root=tmp_path,
    )
    assert certified_calls == [
        allocation["allocation_manifest_digest"],
        allocation["allocation_manifest_digest"],
    ]


@pytest.mark.parametrize("failure", ("boolean", "different", "exception"))
def test_solve_free_allocation_certificate_callback_fails_closed(
        monkeypatch, tmp_path, failure):
    active, allocation, completion_rows = \
        _solve_free_certified_phase2(monkeypatch, tmp_path)
    arguments = _phase2_build_arguments(
        active, allocation, completion_rows, tmp_path
    )

    def certify(_candidate):
        if failure == "boolean":
            return True
        if failure == "exception":
            raise ValueError("certificate invalid")
        changed = copy.deepcopy(allocation)
        changed["source_identity_manifest_digest"] = "f" * 64
        return changed

    with pytest.raises(
            S.StateSelectorAmendmentError, match="solve-free allocation"):
        S.build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            **arguments,
            certify_allocation_solve_free=certify,
        )


def test_solve_free_certified_phase2_rejects_manifest_active_and_receipt_tamper(
        monkeypatch, tmp_path):
    active, allocation, completion_rows = \
        _solve_free_certified_phase2(monkeypatch, tmp_path)
    arguments = _phase2_build_arguments(
        active, allocation, completion_rows, tmp_path
    )
    certify = lambda _candidate: copy.deepcopy(allocation)
    receipt = (
        S.build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            **arguments,
            certify_allocation_solve_free=certify,
        )
    )

    changed_allocation = copy.deepcopy(allocation)
    changed_allocation["assignments"][0]["candidate_indices"][0] = 11
    with pytest.raises(
            S.StateSelectorAmendmentError, match="exact rotation"):
        S.build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            **{**arguments, "allocation_manifest": changed_allocation},
            certify_allocation_solve_free=certify,
        )

    changed_states = copy.deepcopy(active)
    changed_states[0]["goal_type"] = "different-material"
    with pytest.raises(
            S.StateSelectorAmendmentError, match="identity projection"):
        S.build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            **{**arguments, "active_states": changed_states},
            certify_allocation_solve_free=certify,
        )

    changed_masks = copy.deepcopy(active)
    changed_masks[0]["candidate_indices"] = [0, 1, 2, 3, 4, 5]
    with pytest.raises(
            S.StateSelectorAmendmentError, match="exact candidate mask"):
        S.build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            **{**arguments, "active_states": changed_masks},
            certify_allocation_solve_free=certify,
        )

    changed_receipt = copy.deepcopy(receipt)
    changed_receipt["retained_predecessor_candidate_masks"][0][
        "candidate_indices"
    ][0] = 11
    changed_receipt["preserved_state_revalidation_receipt_digest"] = S._sha256({
        key: value for key, value in changed_receipt.items()
        if key != "preserved_state_revalidation_receipt_digest"
    })
    with pytest.raises(
            S.StateSelectorAmendmentError, match="reconstruction"):
        S.validate_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
            changed_receipt,
            allocation_manifest=allocation,
            active_states=active,
            certify_allocation_solve_free=certify,
            root=tmp_path,
        )


def test_phase2_allows_same_failed_scene_only_for_a_distinct_snapshot(
        monkeypatch, tmp_path):
    active, replacements, rejected_sources, allocation, rows = \
        _synthetic_mixed_phase2(monkeypatch, tmp_path)
    receipt = S.build_preserved_state_revalidation_receipt(
        allocation_manifest=allocation,
        active_states=active,
        completion_states=rows,
        source_repository_commit="c" * 40,
        successor_selection_digest="e" * 64,
        state_selector_feasibility_receipt_digest=
            S.FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"],
        mixed_precontract_disposition_receipt_digest="d" * 64,
        root=tmp_path,
    )
    assert receipt["replacement_state_count"] == 8
    assert replacements[0]["scene_id"] == rejected_sources[0]["scene_id"]

    duplicate_snapshot = copy.deepcopy(active)
    replacement_index = duplicate_snapshot.index(next(
        row for row in duplicate_snapshot
        if row["state_id"] == replacements[0]["state_id"]
    ))
    replacement_identity = duplicate_snapshot[replacement_index][
        "state_identity_digest"
    ]
    duplicate_snapshot[replacement_index] = {
        **copy.deepcopy(rejected_sources[0]),
        "state_identity_digest": replacement_identity,
        "selector_successor_marker": "new-contract-only",
        "goal": {
            **copy.deepcopy(rejected_sources[0]["goal"]),
            "landmark_id": "alternate-goal-does-not-change-the-snapshot",
        },
    }
    with pytest.raises(
            S.StateSelectorAmendmentError,
            match="exact rejected predecessor snapshot"):
        S.build_preserved_state_revalidation_receipt(
            allocation_manifest=allocation,
            active_states=duplicate_snapshot,
            completion_states=rows,
            source_repository_commit="c" * 40,
            successor_selection_digest="e" * 64,
            state_selector_feasibility_receipt_digest=
                S.FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"],
            mixed_precontract_disposition_receipt_digest="d" * 64,
            root=tmp_path,
        )


def test_phase2_rejects_caller_completion_evidence_not_owned_by_active_state(
        monkeypatch, tmp_path):
    active, _replacements, _rejected, allocation, rows = \
        _synthetic_mixed_phase2(monkeypatch, tmp_path)
    changed = copy.deepcopy(rows)
    changed[0]["previous_applied_command"] = [0.1, 0.0, 0.0]
    with pytest.raises(
            S.StateSelectorAmendmentError,
            match="identity-owned evidence"):
        S.build_preserved_state_revalidation_receipt(
            allocation_manifest=allocation,
            active_states=active,
            completion_states=changed,
            source_repository_commit="c" * 40,
            successor_selection_digest="e" * 64,
            state_selector_feasibility_receipt_digest=
                S.FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"],
            mixed_precontract_disposition_receipt_digest="d" * 64,
            root=tmp_path,
        )


@pytest.mark.parametrize("surface", ("selector_flag", "claim", "reset"))
def test_phase2_rejects_full_snapshot_status_or_production_evidence_tamper(
        monkeypatch, tmp_path, surface):
    active, replacements, _rejected, allocation, rows = \
        _synthetic_mixed_phase2(monkeypatch, tmp_path)
    changed = copy.deepcopy(active)
    replacement = next(
        row for row in changed
        if row["state_id"] == replacements[0]["state_id"]
    )
    status = replacement["snapshot_task_status"]
    if surface == "selector_flag":
        status["goal_claimed"] = True
    elif surface == "claim":
        status["production_claim_evidence"][
            "active_collector_claimed_cells"] = [
                replacement["goal"]["landmark_cell"]]
    else:
        status["production_task_completion_reset_evidence"][
            "all_scene_landmark_cells_claimed"] = True
    with pytest.raises(
            S.StateSelectorAmendmentError, match="active task status"):
        S.build_preserved_state_revalidation_receipt(
            allocation_manifest=allocation,
            active_states=changed,
            completion_states=rows,
            source_repository_commit="c" * 40,
            successor_selection_digest="e" * 64,
            state_selector_feasibility_receipt_digest=
                S.FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"],
            mixed_precontract_disposition_receipt_digest="d" * 64,
            root=tmp_path,
        )


def test_phase2_rejects_resigned_retained_payload_with_changed_boundary_or_goal(
        monkeypatch, tmp_path):
    active, _replacements, _rejected, allocation, rows = \
        _synthetic_mixed_phase2(monkeypatch, tmp_path)
    changed = copy.deepcopy(active)
    retained_index = next(
        index for index, row in enumerate(changed)
        if row["state_id"].startswith("retained-")
        and row["stratum"] != "completion_enriched"
    )
    changed[retained_index]["boundary"] = {
        **changed[retained_index]["boundary"],
        "sim_time_ns": 123,
    }
    changed[retained_index]["goal"] = {
        **changed[retained_index]["goal"],
        "landmark_id": "resigned-alternate-goal",
    }
    # Keep the frozen predecessor digest deliberately: the central phase-2
    # validator must compare the full payload, not trust the declaration.
    with pytest.raises(
            S.StateSelectorAmendmentError,
            match="retained predecessor identity is absent or changed"):
        S.build_preserved_state_revalidation_receipt(
            allocation_manifest=allocation,
            active_states=changed,
            completion_states=rows,
            source_repository_commit="c" * 40,
            successor_selection_digest="e" * 64,
            state_selector_feasibility_receipt_digest=
                S.FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"],
            mixed_precontract_disposition_receipt_digest="d" * 64,
            root=tmp_path,
        )
