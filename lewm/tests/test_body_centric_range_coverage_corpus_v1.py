from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_STATE = "wide-cal-0-00"
CONTACT_FIXTURE_STATE = "wide-cal-0-02"


@pytest.fixture(scope="module")
def context() -> corpus.FrozenCorpus:
    return corpus.load_corpus_context(ROOT)


@pytest.fixture(scope="module")
def loaded_state(context: corpus.FrozenCorpus) -> corpus.LoadedFrozenState:
    return corpus.load_state(context, FIXTURE_STATE)


def test_frozen_bindings_roles_and_complete_row_custody(
    context: corpus.FrozenCorpus,
) -> None:
    receipt = context.binding_receipt
    assert receipt["source_lineage"] == corpus.SOURCE_LINEAGE
    assert receipt["completed_result_commit"] == corpus.COMPLETED_RESULT_COMMIT
    assert receipt["corpus_logical_digest"] == corpus.CORPUS_LOGICAL_DIGEST
    assert receipt["states"] == 176
    assert receipt["transitions"] == 29_470
    assert receipt["current_representatives"] == 1_594
    assert receipt["successor_representatives"] == 11_791
    assert receipt["sealed_or_g2_opened"] is False
    assert {role: len(context.state_ids_for_role(role)) for role in receipt["roles"]} == {
        "training": 128,
        "calibration": 24,
        "heldout": 24,
    }
    assert context.state_ids_for_role("internal_calibration") == context.state_ids_for_role(
        "calibration"
    )
    assert context.state_ids_for_role("development_heldout") == context.state_ids_for_role(
        "heldout"
    )
    assert len({*context.state_ids}) == 176


def test_state_exposes_scene_geometry_links_h3_and_all_transition_rows(
    loaded_state: corpus.LoadedFrozenState,
) -> None:
    state = loaded_state
    assert state.role == "training"
    assert state.family == "large_enclosed_maze"
    assert len(state.scene_boxes) == 125
    assert state.scene_boxes.centers_m.shape == (125, 3)
    assert state.scene_boxes.half_extents_m.shape == (125, 3)
    assert state.scene_boxes.yaw_rad.shape == (125,)
    assert len(state.geometry_contract) == 27
    assert state.protected_link_names == corpus.PROTECTED_LINK_NAMES
    assert state.body_region_by_link["base"] == "trunk"
    assert state.body_region_by_link["FL_hip"] == "front_limb"
    assert state.body_region_by_link["RL_thigh"] == "rear_limb"
    assert state.body_region_by_link["RR_calf"] == "calf"
    assert len(state.current_rows) == 14
    assert len(state.transition_rows) == 210
    assert [row["transition_index"] for row in state.transition_rows] == list(range(210))
    current = [row for row in state.transition_rows if row["level"] == "current"]
    assert len(current) == 14
    assert current[0]["h3_progress_m"] == pytest.approx(0.18208122396827342)
    assert current[0]["h3_heading_improvement_rad"] == pytest.approx(0.0)
    assert current[0]["decision_progress_m"] == current[0]["h3_progress_m"]
    assert current[12]["h3_progress_m"] is None
    assert current[12]["decision_progress_m"] == current[12]["immediate_progress_m"]


def test_applied_action_representatives_collapse_both_ply_prefixes(
    loaded_state: corpus.LoadedFrozenState,
) -> None:
    mapping = loaded_state.action_copy_map
    assert mapping.transition_count == 210
    assert mapping.current_representative_count == 10
    assert mapping.successor_representative_count == 95
    assert mapping.representative_count == 105
    # Current actions 1 and 9 execute the same route-controller command.
    assert mapping.current_action_representative[1] == 1
    assert mapping.current_action_representative[9] == 1
    assert mapping.representative_for(9) == mapping.representative_for(1) == 1
    assert sorted(row for rows in mapping.copies_by_representative.values() for row in rows) == list(
        range(210)
    )
    # Prefix 9 is physical copy of prefix 1, so none of its next rows is a rep.
    prefix_nine = [
        row
        for row in loaded_state.transition_rows
        if row["level"] == "successor" and row["current_action_index"] == 9
    ]
    assert len(prefix_nine) == 14
    assert not any(row["is_physical_representative"] for row in prefix_nine)


def test_shard_alignment_and_identical_representative_trajectories(
    loaded_state: corpus.LoadedFrozenState,
) -> None:
    shard = loaded_state.shard
    assert shard.arrays["qpos"].shape == (210, 50, 19)
    assert shard.arrays["link_transform"].shape == (210, 50, 13, 7)
    assert shard.arrays["geom_transform"].shape == (210, 50, 27, 7)
    assert shard.arrays["transition_level"].tolist()[:2] == ["current", "current"]
    validation = corpus.validate_representative_geometry(
        shard, loaded_state.action_copy_map
    )
    assert validation["pass"] is True
    assert validation["mismatches"] == []


def test_snapshot_boundaries_are_exact_arrays_and_match_successor_endpoint(
    context: corpus.FrozenCorpus, loaded_state: corpus.LoadedFrozenState
) -> None:
    state = loaded_state
    assert state.boundary_qpos.shape == (19,)
    assert state.boundary_geom_transform.shape == (27, 7)
    assert state.boundary_raw_geom_transform.shape == (27, 7)
    assert state.boundaries.current.link_transform.shape == (13, 7)
    assert state.boundaries.current.raw_geom_global_indices == tuple(range(126, 153))
    assert set(state.successor_boundary_qpos) == set(range(14))
    assert all(value.shape == (19,) for value in state.successor_boundary_qpos.values())
    assert all(
        value.shape == (27, 7)
        for value in state.successor_boundary_geom_transform.values()
    )

    check = corpus.validate_boundary_alignment(
        context,
        FIXTURE_STATE,
        0,
        shard=state.shard,
        boundaries=state.boundaries,
    )
    assert check["raw_boundary_geometry_is_last_27"] is True
    assert check["link_position_max_abs_m"] <= 2e-6
    assert check["link_quaternion_sign_invariant_max_abs"] <= 2e-6
    assert check["contract_geometry_position_max_abs_m"] <= 2e-6
    assert check["contract_geometry_quaternion_sign_invariant_max_abs"] <= 2e-6
    assert check["pass"] is True

    transition = state.shard.transition_index("current", -1, 0)
    np.testing.assert_allclose(
        state.successor_boundary_qpos[0],
        state.shard.arrays["qpos"][transition, -1],
        rtol=0,
        atol=2e-6,
    )


def test_exact_native_and_frozen_contact_attribution_are_independent(
    context: corpus.FrozenCorpus,
) -> None:
    shard = context.load_geometry_shard(CONTACT_FIXTURE_STATE)
    row = shard.transition_rows[0]
    assert row["frozen_contact"] is True
    attribution = corpus.contact_attribution(shard, 0)
    assert attribution["frozen"]["contact"] is True
    assert attribution["frozen"]["first_contact_step"] == 17
    assert attribution["native_replay"]["contact"] is True
    assert attribution["exact"]["contact"] is True
    assert attribution["native_replay"]["first_contact_step"] is not None
    assert attribution["exact"]["first_contact_step"] is not None
    assert attribution["exact"]["robot_link_name"] in corpus.PROTECTED_LINK_NAMES
    assert attribution["exact"]["body_region"] in {
        "trunk",
        "front_limb",
        "rear_limb",
        "calf",
    }
    assert attribution["exact"]["manifold_position_m"] is not None
    assert len(attribution["exact"]["manifold_position_m"]) == 3


def test_arrays_are_read_only_and_action_identity_is_controller_sensitive(
    loaded_state: corpus.LoadedFrozenState,
) -> None:
    assert loaded_state.boundary_qpos.flags.writeable is False
    assert loaded_state.scene_boxes.centers_m.flags.writeable is False
    assert loaded_state.shard.arrays["qpos"].flags.writeable is False
    route = {"controller": "route", "applied_action": [0.0, 0.2, 0.0]}
    lateral = {"controller": "lateral", "applied_action": [0.0, 0.2, 0.0]}
    assert corpus.applied_action_key(route) != corpus.applied_action_key(lateral)
