from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import (
    ENVS_PER_SOURCE,
    FAMILIES,
    PRIMITIVES,
    SourceRef,
)
from lewm.datasets import go2_recurrent_h4_rgb_sequences as v1
from lewm.datasets import go2_recurrent_h4_rgb_sequences_v2 as v2
from lewm.datasets.go2_recurrent_h4_rgb_sequences_v2 import (
    ACTION_TO_INDEX,
    ALIGNMENT,
    SCHEMA,
    SEED,
    SequenceContractError,
    _FrameRecord,
    _adapt_legacy_window,
    _wanted_frame_indices,
    canonical_row_bytes,
)
from scripts import build_go2_recurrent_h4_rgb_index_v2 as builder


ENV_INDEX = 17
TICK_NS = 100_000_000
STREAM_KEY = (ENV_INDEX, "episode-0", 0)


def _source() -> SourceRef:
    family = FAMILIES[0]
    return SourceRef(
        role="train",
        family=family,
        chunk="chunk_0000",
        sequence=f"000000_{family}_0123456789ab",
        byte_count=1,
        ordinal=1,
    )


def _endpoint(step: int) -> v1.Endpoint:
    return v1.Endpoint(
        frame_index=ENV_INDEX + ENVS_PER_SOURCE * (step - 1),
        env_index=ENV_INDEX,
        episode_step=step,
        timestamp_ns=step * TICK_NS,
    )


def _legacy_window(command_start: int) -> v1.H6Window:
    source = _source()
    endpoints = tuple(
        _endpoint((command_start + offset) * 5 + 1) for offset in range(7)
    )
    return v1.H6Window(
        rank=v1._window_rank(source, STREAM_KEY, endpoints[0]),
        role=source.role,
        family=source.family,
        scene_id=source.sequence.split("_", 1)[1],
        endpoints=endpoints,
        actions=tuple(
            ACTION_TO_INDEX[PRIMITIVES[index % len(PRIMITIVES)]]
            for index in range(command_start, command_start + 6)
        ),
    )


def _record(
    step: int,
    *,
    command_index: int,
    key: tuple[int, str, int] = STREAM_KEY,
) -> _FrameRecord:
    primitive = PRIMITIVES[command_index % len(PRIMITIVES)]
    return _FrameRecord(
        key=key,
        endpoint=_endpoint(step),
        signature=(str(command_index), primitive),
        primitive=primitive,
        block_size=5,
        command_dt_s=0.1,
        request_timestamp_ns=(command_index * 5) * TICK_NS,
    )


def _records_for_second_group() -> dict[int, _FrameRecord]:
    records: dict[int, _FrameRecord] = {}
    predecessor = _record(30, command_index=5)
    records[predecessor.endpoint.frame_index] = predecessor
    for command_index in range(6, 12):
        first_step = command_index * 5 + 1
        for tick in range(5):
            record = _record(
                first_step + tick,
                command_index=command_index,
            )
            records[record.endpoint.frame_index] = record
    return records


def test_initial_group_is_discarded_without_shifting_v1_phase() -> None:
    initial = _legacy_window(0)
    second = _legacy_window(6)

    assert _adapt_legacy_window(initial, {}) is None
    adapted = _adapt_legacy_window(second, _records_for_second_group())
    assert adapted is not None
    assert adapted.actions == second.actions
    assert adapted.rank == second.rank
    assert adapted.actions[0] == ACTION_TO_INDEX[PRIMITIVES[6]]


def test_causal_endpoints_are_previous_final_then_target_block_finals() -> None:
    legacy = _legacy_window(6)
    adapted = _adapt_legacy_window(legacy, _records_for_second_group())
    assert adapted is not None

    assert [endpoint.episode_step for endpoint in adapted.endpoints] == [
        30,
        35,
        40,
        45,
        50,
        55,
        60,
    ]
    assert all(
        right.frame_index - left.frame_index == 5 * ENVS_PER_SOURCE
        and right.timestamp_ns - left.timestamp_ns == 5 * TICK_NS
        for left, right in zip(adapted.endpoints, adapted.endpoints[1:])
    )
    assert all(endpoint.episode_step != 61 for endpoint in adapted.endpoints)

    wanted = _wanted_frame_indices([_legacy_window(0), legacy])
    assert _endpoint(30).frame_index in wanted
    assert _endpoint(60).frame_index in wanted
    assert _endpoint(61).frame_index not in wanted


def test_reset_or_episode_change_is_rejected() -> None:
    legacy = _legacy_window(6)
    records = _records_for_second_group()
    changed = records[_endpoint(41).frame_index]
    records[_endpoint(41).frame_index] = replace(
        changed,
        key=(ENV_INDEX, "episode-1", 1),
    )

    with pytest.raises(SequenceContractError, match="reset-safe"):
        _adapt_legacy_window(legacy, records)


def test_five_tick_timing_and_requested_target_action_are_enforced() -> None:
    legacy = _legacy_window(6)
    records = _records_for_second_group()
    shifted = records[_endpoint(36).frame_index]
    records[_endpoint(36).frame_index] = replace(
        shifted,
        request_timestamp_ns=shifted.request_timestamp_ns + 1,
    )
    with pytest.raises(SequenceContractError, match="stable"):
        _adapt_legacy_window(legacy, records)

    records = _records_for_second_group()
    wrong = records[_endpoint(50).frame_index]
    wrong_primitive = PRIMITIVES[(PRIMITIVES.index(wrong.primitive) + 1) % len(PRIMITIVES)]
    records[_endpoint(50).frame_index] = replace(
        wrong,
        signature=(wrong.signature[0], wrong_primitive),
        primitive=wrong_primitive,
    )
    with pytest.raises(SequenceContractError, match="stable|primitive"):
        _adapt_legacy_window(legacy, records)

    records = _records_for_second_group()
    irregular = records[_endpoint(47).frame_index]
    records[_endpoint(47).frame_index] = replace(
        irregular,
        endpoint=replace(
            irregular.endpoint,
            timestamp_ns=irregular.endpoint.timestamp_ns + 10_000_000,
        ),
    )
    with pytest.raises(SequenceContractError, match="five regular ticks"):
        _adapt_legacy_window(legacy, records)


def test_v2_schema_and_output_root_are_distinct_and_explicit() -> None:
    adapted = _adapt_legacy_window(_legacy_window(6), _records_for_second_group())
    assert adapted is not None
    row = adapted.to_row()

    assert SEED == v1.SEED
    assert SCHEMA == "lewm_go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity"
    assert row["schema"] == SCHEMA
    assert set(row) == {"schema", "role", "family", "scene_id", "rgb", "actions"}
    assert len(row["rgb"]) == 7
    assert len(row["actions"]) == 6
    assert canonical_row_bytes(adapted).endswith(b"\n")
    assert ALIGNMENT == (
        "pre_request_final_previous_block -> fifth_tick_current_block; "
        "requested_target_block_action; no_next_block_tick"
    )
    assert builder.DEFAULT_OUTPUT == Path(
        ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity"
    )


def test_scan_filters_complete_v1_order_before_cap_and_backfills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = [
        replace(
            _legacy_window(0 if index < 2 else 6 * (index - 1)),
            rank=f"{index:064x}",
        )
        for index in range(5)
    ]

    def fake_v1_scan(_root: str, _source_ref: SourceRef, cap: int):
        assert cap == v2.ROWS_PER_SOURCE
        return {
            "candidate_count": len(legacy),
            "candidates": legacy,
            "scene_id": legacy[0].scene_id,
            "manifest_sha256": "a" * 64,
        }

    monkeypatch.setattr(v2._v1, "scan_source_candidates", fake_v1_scan)
    monkeypatch.setattr(
        v2,
        "_load_wanted_records",
        lambda *_args, **_kwargs: ({}, 0, "b" * 64),
    )

    def fake_adapt(window: v1.H6Window, _records):
        if window.rank in {legacy[0].rank, legacy[1].rank}:
            return None
        return v2.H6Window(
            rank=window.rank,
            role=window.role,
            family=window.family,
            scene_id=window.scene_id,
            endpoints=window.endpoints,
            actions=window.actions,
        )

    monkeypatch.setattr(v2, "_adapt_legacy_window", fake_adapt)
    result = v2.scan_source_candidates("/unused", _source(), 2)

    assert result["legacy_candidate_count"] == 5
    assert result["discarded_initial_group_count"] == 2
    assert result["candidate_count"] == 3
    assert result["emitted_causal_group_count"] == 3
    assert [window.rank for window in result["candidates"]] == [
        legacy[2].rank,
        legacy[3].rank,
    ]

    monkeypatch.setattr(
        v2._v1,
        "scan_source_candidates",
        lambda *_args: {
            "candidate_count": 6,
            "candidates": legacy,
            "scene_id": legacy[0].scene_id,
            "manifest_sha256": "a" * 64,
        },
    )
    with pytest.raises(SequenceContractError, match="full V1 candidate order"):
        v2.scan_source_candidates("/unused", _source(), 2)


def test_public_source_bindings_use_current_file_bytes(tmp_path: Path) -> None:
    payloads = {
        builder.PUBLIC_SOURCE_PATHS[0]: b"v1-source\n",
        builder.PUBLIC_SOURCE_PATHS[1]: b"v2-source\n",
        builder.PUBLIC_SOURCE_PATHS[2]: b"builder-source\n",
    }
    for relative, payload in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    binding = builder._public_source_bindings(tmp_path)
    assert binding["entry_count"] == 3
    assert [entry["path"] for entry in binding["entries"]] == [
        path.as_posix() for path in builder.PUBLIC_SOURCE_PATHS
    ]
    assert [entry["byte_count"] for entry in binding["entries"]] == [
        len(payloads[path]) for path in builder.PUBLIC_SOURCE_PATHS
    ]
    assert [entry["sha256"] for entry in binding["entries"]] == [
        hashlib.sha256(payloads[path]).hexdigest()
        for path in builder.PUBLIC_SOURCE_PATHS
    ]


def test_live_census_binding_uses_source_ordinal_and_fails_on_mismatch() -> None:
    first = SourceRef(
        role="train",
        family=FAMILIES[0],
        chunk="chunk_0000",
        sequence=f"000001_{FAMILIES[0]}_0123456789ab",
        byte_count=11,
        ordinal=1,
    )
    second = SourceRef(
        role="val",
        family=FAMILIES[1],
        chunk="chunk_0001",
        sequence=f"000002_{FAMILIES[1]}_abcdef012345",
        byte_count=22,
        ordinal=2,
    )
    results = [
        {
            "source_ordinal": 2,
            "role": second.role,
            "family": second.family,
            "source_content_byte_count": 22,
            "source_content_sha256": "2" * 64,
        },
        {
            "source_ordinal": 1,
            "role": first.role,
            "family": first.family,
            "source_content_byte_count": 11,
            "source_content_sha256": "1" * 64,
        },
    ]
    ordered_rows = [
        [
            first.role,
            first.family,
            first.chunk,
            first.sequence,
            11,
            "1" * 64,
        ],
        [
            second.role,
            second.family,
            second.chunk,
            second.sequence,
            22,
            "2" * 64,
        ],
    ]
    expected = hashlib.sha256(
        json.dumps(
            ordered_rows,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    observed, row_count = v2._require_census_source_binding(
        [second, first],
        results,
        expected_sha256=expected,
    )
    assert observed == expected
    assert row_count == 2

    reversed_order = hashlib.sha256(
        json.dumps(
            list(reversed(ordered_rows)),
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert reversed_order != expected
    with pytest.raises(SequenceContractError, match="frozen census"):
        v2._require_census_source_binding(
            [second, first],
            results,
            expected_sha256=reversed_order,
        )


def test_builder_validates_frozen_census_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_census(_repo_root: Path):
        calls.append("census")
        return {"ordered_source_content_binding_sha256": "0" * 64}

    def fake_sources(_repo_root: Path):
        calls.append("sources")
        return {"entries": [], "entry_count": 0, "ordered_binding_sha256": "0" * 64}

    def fake_build(_repo_root: Path, *, workers: int, progress):
        del workers, progress
        calls.append("build")
        return (
            {"train": [], "val": []},
            {
                "source": {
                    "discarded_initial_group_count": 0,
                    "emitted_causal_group_count": 0,
                }
            },
        )

    monkeypatch.setattr(builder, "_load_census_source_binding", fake_census)
    monkeypatch.setattr(builder, "_public_source_bindings", fake_sources)
    monkeypatch.setattr(builder, "build_index", fake_build)

    assert builder.main(
        [
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            "index-output",
            "--workers",
            "1",
        ]
    ) == 0
    assert calls == ["census", "sources", "build", "sources"]
