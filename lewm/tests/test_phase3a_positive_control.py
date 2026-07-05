from __future__ import annotations

from pathlib import Path

from lewm.benchmarks.phase3a_positive_control import (
    ACTION_NAMES,
    PHASE3A_ROW_SCHEMA,
    action_vector,
    generate_phase3a_rows,
    phase3a_action_only_prior,
    phase3a_dataset_audit,
    phase3a_source_oracles,
    read_jsonl,
    write_jsonl,
)


def test_action_vectors_are_stable_one_hot() -> None:
    assert action_vector("forward") == (1.0, 0.0, 0.0, 0.0)
    assert action_vector("hold") == (0.0, 0.0, 0.0, 1.0)


def test_phase3a_generation_is_deterministic_and_counterfactual() -> None:
    rows_a, audit_a = generate_phase3a_rows(
        split="train",
        scene_count=2,
        source_states_per_scene=3,
        seed=17,
    )
    rows_b, audit_b = generate_phase3a_rows(
        split="train",
        scene_count=2,
        source_states_per_scene=3,
        seed=17,
    )

    assert rows_a == rows_b
    assert audit_a == audit_b
    assert audit_a["source_states"] == 6
    assert audit_a["rows"] == 6 * len(ACTION_NAMES) ** 2
    assert audit_a["candidate_rows_per_source_histogram"] == {16: 6}

    first_source = [
        row for row in rows_a if row["scene_id"] == rows_a[0]["scene_id"] and row["source_index"] == 0
    ]
    assert len(first_source) == len(ACTION_NAMES) ** 2
    assert {
        tuple(row["primitive_sequence"]) for row in first_source
    } == {
        (first, second) for first in ACTION_NAMES for second in ACTION_NAMES
    }
    assert len({repr(row["start_observation_rgb"]) for row in first_source}) == 1


def test_phase3a_observations_have_channel_major_rgb_shape() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=1,
        seed=23,
        view_size=7,
    )
    observation = rows[0]["start_observation_rgb"]

    assert len(observation) == 3
    assert all(len(channel) == 7 for channel in observation)
    assert all(len(row) == 7 for channel in observation for row in channel)
    assert all(
        0.0 <= value <= 1.0
        for channel in observation
        for row in channel
        for value in row
    )
    beacon = tuple(channel[0][0] for channel in observation)
    assert any(value not in (0.05, 0.18, 0.72, 0.10) for value in beacon)


def test_phase3a_history_can_carry_beacon_when_current_frame_does_not() -> None:
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=1,
        seed=29,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=True,
    )

    row = rows[0]
    assert row["schema"] == PHASE3A_ROW_SCHEMA
    assert row["history_steps"] == 3
    assert row["current_goal_beacon"] is False
    assert row["history_goal_beacon"] is True
    assert len(row["history_observations_rgb"]) == 3
    assert len(row["history_actions"]) == 3
    assert len(row["history_primitive_sequence"]) == 3
    assert audit["history_step_counts"] == {3: len(rows)}
    assert audit["current_goal_beacon_counts"] == {"False": len(rows)}

    current_beacon = tuple(channel[0][0] for channel in row["start_observation_rgb"])
    history_beacon = tuple(
        channel[0][0] for channel in row["history_observations_rgb"][0]
    )
    assert current_beacon != history_beacon
    assert any(value not in (0.05, 0.18, 0.72, 0.10) for value in history_beacon)


def test_phase3a_goal_marker_can_be_hidden_from_current_and_future_views() -> None:
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=1,
        seed=37,
        history_steps=2,
        current_goal_beacon=False,
        history_goal_beacon=True,
        current_goal_marker=False,
        history_goal_marker=False,
        future_goal_marker=False,
    )

    row = rows[0]
    assert row["current_goal_marker"] is False
    assert row["future_goal_marker"] is False
    assert audit["current_goal_marker_counts"] == {"False": len(rows)}
    assert audit["future_goal_marker_counts"] == {"False": len(rows)}
    current_pixels = [
        tuple(row["start_observation_rgb"][channel][y][x] for channel in range(3))
        for y in range(9)
        for x in range(9)
    ]
    future_pixels = [
        tuple(frame["observation_rgb"][channel][y][x] for channel in range(3))
        for frame in row["future_observations"]
        for y in range(9)
        for x in range(9)
    ]
    goal_color = (0.10, 0.85, 0.18)
    assert goal_color not in current_pixels
    assert goal_color not in future_pixels


def test_phase3a_goal_variants_alias_current_view_but_not_history() -> None:
    goal_variants = 3
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=41,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=True,
        current_goal_marker=False,
        history_goal_marker=False,
        future_goal_marker=False,
        goal_variants_per_source=goal_variants,
    )

    assert audit["source_states"] == 2 * goal_variants
    assert audit["rows"] == 2 * goal_variants * len(ACTION_NAMES) ** 2
    assert audit["candidate_rows_per_source_histogram"] == {16: 2 * goal_variants}
    assert audit["goal_variants_per_source_counts"] == {goal_variants: len(rows)}
    assert audit["current_goal_beacon_counts"] == {"False": len(rows)}
    assert audit["history_goal_beacon_counts"] == {"True": len(rows)}

    first_candidate_aliases = [
        row
        for row in rows
        if row["base_source_index"] == 0 and row["candidate_index"] == 0
    ]
    assert len(first_candidate_aliases) == goal_variants
    assert len({row["goal_variant_index"] for row in first_candidate_aliases}) == goal_variants
    assert len({(row["goal"]["x"], row["goal"]["y"]) for row in first_candidate_aliases}) == goal_variants
    assert (
        len({repr(row["start_observation_rgb"]) for row in first_candidate_aliases})
        == 1
    )
    assert (
        len({repr(row["history_observations_rgb"]) for row in first_candidate_aliases})
        == goal_variants
    )


def test_phase3a_goal_variants_can_be_near_source_for_claim_contracts() -> None:
    goal_variants = 2
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=1,
        seed=73,
        history_steps=2,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        goal_variants_per_source=goal_variants,
        minimum_goal_variant_distance=1,
        maximum_goal_variant_distance=3,
    )

    assert audit["goal_variants_per_source_counts"] == {goal_variants: len(rows)}
    assert audit["minimum_goal_variant_distance"] == 1
    assert audit["maximum_goal_variant_distance"] == 3

    first_candidate_aliases = [
        row
        for row in rows
        if row["base_source_index"] == 0 and row["candidate_index"] == 0
    ]
    assert len(first_candidate_aliases) == goal_variants
    assert (
        len({repr(row["start_observation_rgb"]) for row in first_candidate_aliases})
        == 1
    )
    assert all(
        1
        <= row["consequence_labels"]["start_goal_distance_cells"]
        <= 3
        for row in first_candidate_aliases
    )


def test_phase3a_explore_then_claim_uses_visual_markers_without_beacons() -> None:
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=4,
        seed=59,
        history_steps=4,
        history_policy="explore",
        utility_mode="explore_then_claim",
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=True,
        history_goal_marker=True,
        future_goal_marker=True,
    )

    assert audit["current_goal_beacon_counts"] == {"False": len(rows)}
    assert audit["history_goal_beacon_counts"] == {"False": len(rows)}
    assert audit["history_policy_counts"] == {"explore": len(rows)}
    assert audit["utility_mode_counts"] == {"explore_then_claim": len(rows)}
    assert audit["history_goal_marker_seen_counts"] == {"False": 48, "True": 16}
    assert audit["current_goal_marker_seen_counts"] == {"False": 48, "True": 16}
    assert audit["future_goal_marker_seen_counts"] == {"False": 48, "True": 16}

    assert any("forward" in row["history_primitive_sequence"] for row in rows)
    assert all(row["current_goal_beacon"] is False for row in rows)
    assert all(row["history_goal_beacon"] is False for row in rows)
    assert {
        row["consequence_labels"]["utility_mode"] for row in rows
    } == {"explore_then_claim"}

    novelty_rows = [
        row
        for row in rows
        if not row["consequence_labels"]["goal_known_before_candidate"]
        and not row["consequence_labels"]["future_goal_marker_seen"]
        and row["consequence_labels"]["target_new_free_cells"] > 0
    ]
    assert novelty_rows
    novelty = novelty_rows[0]["consequence_labels"]
    expected = (
        0.35 * float(novelty["target_new_free_cells"])
        - 2.0 * float(novelty["collision_count"])
    )
    assert novelty["target_utility"] == expected
    assert novelty["target_utility"] != novelty["target_goal_progress_utility"]


def test_phase3a_novelty_then_claim_omits_hidden_discovery_bonus() -> None:
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=6,
        seed=50,
        view_size=7,
        history_steps=4,
        history_policy="explore",
        utility_mode="novelty_then_claim",
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=True,
        history_goal_marker=True,
        future_goal_marker=True,
    )

    assert audit["utility_mode_counts"] == {"novelty_then_claim": len(rows)}
    discovery_rows = [
        row
        for row in rows
        if not row["consequence_labels"]["goal_known_before_candidate"]
        and row["consequence_labels"]["future_goal_marker_seen"]
    ]
    assert discovery_rows
    labels = discovery_rows[0]["consequence_labels"]
    expected = (
        0.35 * float(labels["target_new_free_cells"])
        - 2.0 * float(labels["collision_count"])
    )

    assert labels["utility_mode"] == "novelty_then_claim"
    assert labels["target_utility"] == expected


def test_phase3a_can_sample_distant_sources_for_no_beacon_exploration() -> None:
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=3,
        seed=67,
        view_size=7,
        current_goal_beacon=False,
        history_goal_beacon=False,
        minimum_source_goal_distance=6,
    )

    assert audit["minimum_source_goal_distance"] == 6
    source_rows = [row for row in rows if row["candidate_index"] == 0]
    assert len(source_rows) == 3
    assert all(
        row["consequence_labels"]["start_goal_distance_cells"] >= 6
        for row in source_rows
    )


def test_phase3a_can_sample_near_goal_sources_for_claim_splits() -> None:
    rows, audit = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=3,
        seed=71,
        view_size=7,
        history_steps=0,
        current_goal_beacon=False,
        history_goal_beacon=False,
        minimum_source_goal_distance=1,
        maximum_source_goal_distance=2,
    )

    assert audit["minimum_source_goal_distance"] == 1
    assert audit["maximum_source_goal_distance"] == 2
    source_rows = [row for row in rows if row["candidate_index"] == 0]
    assert len(source_rows) == 3
    assert all(
        1
        <= row["consequence_labels"]["start_goal_distance_cells"]
        <= 2
        for row in source_rows
    )


def test_phase3a_oracles_and_action_prior_are_defined() -> None:
    train_rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=3,
        source_states_per_scene=4,
        seed=31,
    )
    validation_rows, _ = generate_phase3a_rows(
        split="validation",
        scene_count=2,
        source_states_per_scene=4,
        seed=10_031,
    )

    oracles = phase3a_source_oracles(validation_rows)
    prior = phase3a_action_only_prior(train_rows, validation_rows)

    assert len(oracles) == 8
    assert prior["schema"] == "jepa_phase3a_action_only_prior_v2"
    assert prior["selected_first_primitive"] in ACTION_NAMES
    assert tuple(prior["selected_primitive_sequence"]) in {
        (first, second) for first in ACTION_NAMES for second in ACTION_NAMES
    }
    assert 0.0 <= prior["primitive_match_rate"] <= 1.0
    assert prior["mean_target_utility_regret"] >= 0.0
    assert prior["mean_selected_sequence_target_utility_regret"] >= 0.0


def test_phase3a_jsonl_roundtrip_and_audit(tmp_path: Path) -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=47,
    )
    path = tmp_path / "phase3a.jsonl"
    write_jsonl(path, rows)

    loaded = read_jsonl(path)
    audit = phase3a_dataset_audit(loaded)

    assert len(loaded) == len(rows)
    assert loaded[0]["scene_id"] == rows[0]["scene_id"]
    assert loaded[0]["primitive_sequence"] == rows[0]["primitive_sequence"]
    assert audit["schemas"] == {PHASE3A_ROW_SCHEMA: len(rows)}
    assert audit["collision_rows"] > 0
