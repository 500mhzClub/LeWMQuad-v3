from __future__ import annotations

import io
import json
from copy import deepcopy

from lewm.benchmarks import go2_recurrent_jepa_main_pool_census as census


def _rows(
    *,
    blocks: int,
    reset_count: int = 0,
    frame_offset: int = 0,
    timestamp_offset_ns: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for zero_step in range(blocks * 5):
        block = zero_step // 5
        step = zero_step + 1
        rows.append(
            {
                "env_index": 0,
                "frame_index": frame_offset + zero_step,
                "timestamp_ns": timestamp_offset_ns + step * 100_000_000,
                "episode": {
                    "episode_id": f"episode-{reset_count}",
                    "episode_step": step,
                    "manifest_sha256": "a" * 64,
                    "reset_count": reset_count,
                    "scene_id": 17,
                    "split": "train",
                },
                "command_context": {
                    "sequence_id": block,
                    "primitive_name": census.PRIMITIVES[block % len(census.PRIMITIVES)],
                    "block_size": 5,
                    "command_dt_s": 0.10000000149011612,
                },
            }
        )
    return rows


def _scan(rows: list[dict[str, object]]) -> dict[str, object]:
    raw = b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows
    )
    return census.scan_binary_stream(
        io.BytesIO(raw), role="train", family="open_obstacle_field"
    )


def test_six_edge_windows_keep_boundary_row_and_pack_from_path_start() -> None:
    result = _scan(_rows(blocks=8))

    assert result["integrity"] == {}
    assert result["primitive_transitions"] == 7
    assert result["window_counts"] == {
        "h1": 7,
        "h2": 6,
        "h3": 5,
        "h4": 4,
        "h5": 3,
        "h6": 2,
    }
    assert result["packed_h6"] == 1
    assert result["maximal_path_histogram"] == {"7": 1}
    assert result["packed_leftover_histogram"] == {"1": 1}
    assert result["terminal_runs_without_endpoint"] == 1
    assert sum(result["action_position_counts"].values()) == 8


def test_reset_keys_never_join_into_a_false_h6() -> None:
    first = _rows(blocks=5)
    second = _rows(
        blocks=5,
        reset_count=1,
        frame_offset=len(first),
        timestamp_offset_ns=10_000_000_000,
    )

    result = _scan(first + second)

    assert result["integrity"] == {}
    assert result["stream_count"] == 2
    assert result["primitive_transitions"] == 8
    assert result["window_counts"].get("h6", 0) == 0
    assert result["packed_h6"] == 0


def test_mid_scene_reset_keeps_per_env_row_contract_without_fixed_stream_count() -> None:
    first = _rows(blocks=100)
    second = _rows(
        blocks=100,
        reset_count=1,
        frame_offset=len(first),
        timestamp_offset_ns=100_000_000_000,
    )

    result = _scan(first + second)

    assert result["integrity"] == {}
    assert result["stream_count"] == 2
    assert result["env_row_count_violation_count"] == 0
    assert result["primitive_transitions"] == 198
    assert result["window_counts"]["h6"] == 188
    assert result["packed_h6"] == 32


def test_per_environment_row_imbalance_is_not_hidden_by_file_total() -> None:
    rows = _rows(blocks=200)
    rows[-1]["env_index"] = 1
    rows[-1]["episode"]["episode_id"] = "second-env"

    result = _scan(rows)

    assert result["row_count"] == 1_000
    assert result["env_count"] == 2
    assert result["env_row_count_violation_count"] == 2


def test_irregular_tick_breaks_the_affected_transition() -> None:
    rows = _rows(blocks=8)
    rows[2]["timestamp_ns"] = int(rows[2]["timestamp_ns"]) + 50_000_000

    result = _scan(rows)

    assert result["integrity"]["run_irregular_tick_timing_count"] == 1
    assert result["primitive_transitions"] == 6
    assert result["window_counts"].get("h6", 0) == 1


def test_chunk_and_sequence_allowlist_is_narrow_and_includes_known_backfill() -> None:
    assert census._valid_chunk("open_obstacle_field", "chunk_0007")
    assert not census._valid_chunk("open_obstacle_field", "chunk_backfill")
    assert census._valid_chunk("rough_local_dynamics", "chunk_backfill")
    pattern = census._sequence_pattern("small_enclosed_maze")
    assert pattern.fullmatch("000123_small_enclosed_maze_012345abcdef")
    assert not pattern.fullmatch("000123_small_enclosed_maze_012345abcdef_extra")
    assert not pattern.fullmatch("000123_medium_enclosed_maze_012345abcdef")


def _synthetic_full_pool() -> tuple[list[census.SourceRef], list[dict[str, object]]]:
    sources: list[census.SourceRef] = []
    results: list[dict[str, object]] = []
    ordinal = 0
    cells = [
        f"p{position}:{primitive}"
        for position in range(2, 6)
        for primitive in census.PRIMITIVES
    ]
    for role in census.ROLES:
        for family in census.FAMILIES:
            for _ in range(census.EXPECTED_SCENES[role][family]):
                ordinal += 1
                sources.append(
                    census.SourceRef(
                        role=role,
                        family=family,
                        chunk="chunk_0000",
                        sequence=f"synthetic-{ordinal}",
                        byte_count=1,
                        ordinal=ordinal,
                    )
                )
                results.append(
                    {
                        "source_ordinal": ordinal,
                        "inventory_byte_count": 1,
                        "scan_fstat_byte_count": 1,
                        "byte_count": 1,
                        "row_count": census.ROWS_PER_SOURCE,
                        "env_count": census.ENVS_PER_SOURCE,
                        "env_row_count_violation_count": 0,
                        "stream_count": census.ENVS_PER_SOURCE + 1,
                        "first_global_frame_index": 0,
                        "last_global_frame_index": census.ROWS_PER_SOURCE - 1,
                        "primitive_transitions": 9_000,
                        "terminal_runs_without_endpoint": census.ENVS_PER_SOURCE + 1,
                        "accepted_transitions_covered_by_paths": 9_000,
                        "window_counts": {f"h{h}": 8_000 for h in range(1, 7)},
                        "packed_h6": 1_300,
                        "contributing_h6_stream_count": census.ENVS_PER_SOURCE,
                        "action_position_counts": {cell: 100 for cell in cells},
                        "action_position_presence": cells,
                        "maximal_path_histogram": {"99": census.ENVS_PER_SOURCE},
                        "packed_leftover_histogram": {"3": census.ENVS_PER_SOURCE},
                        "integrity": {},
                        "_scene_id_type": "int",
                        "_scene_id": ordinal,
                        "_manifest_sha256": f"{ordinal:064x}",
                        "_content_sha256": f"{ordinal + 10_000:064x}",
                    }
                )
    return sources, results


def test_aggregate_accepts_reset_breadth_and_rejects_duplicate_scene_identity() -> None:
    sources, results = _synthetic_full_pool()

    receipt = census.aggregate_results(sources, results, {})

    assert receipt["decision"] == "MAIN_POOL_H4_METADATA_FEASIBLE"
    assert receipt["failed_predicates"] == []
    assert receipt["identity"]["cross_role_manifest_identity_overlap_count"] == 0

    duplicated = deepcopy(results)
    duplicated[-1]["_scene_id"] = duplicated[-2]["_scene_id"]
    duplicated[-1]["_manifest_sha256"] = duplicated[-2]["_manifest_sha256"]
    receipt = census.aggregate_results(sources, duplicated, {})

    assert receipt["decision"] == "STOP_MAIN_POOL_H4_METADATA_INADEQUATE"
    assert "duplicate_scene_identity_within_role" in receipt["failed_predicates"]
    assert "duplicate_manifest_identity_within_role" in receipt["failed_predicates"]

    cross_role = deepcopy(results)
    cross_role[-1]["_scene_id"] = cross_role[0]["_scene_id"]
    cross_role[-1]["_manifest_sha256"] = cross_role[0]["_manifest_sha256"]
    receipt = census.aggregate_results(sources, cross_role, {})

    assert "train_val_scene_identity_overlap" in receipt["failed_predicates"]
    assert "train_val_manifest_identity_overlap" in receipt["failed_predicates"]
