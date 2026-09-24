from __future__ import annotations

from pathlib import Path

from scripts import (
    collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v3
    as collector,
)


def _worker_kwargs() -> dict:
    return {
        "scene_index": 7,
        "plan_path": Path("/tmp/plan.json"),
        "expected_plan_byte_count": 123,
        "expected_plan_sha256": "a" * 64,
        "authority_path": Path("/tmp/authority.json"),
        "expected_authority_byte_count": 456,
        "expected_authority_sha256": "b" * 64,
        "reservation_binding": {
            "byte_count": 789,
            "file_sha256": "c" * 64,
        },
        "orchestrator_nonce": "d" * 64,
    }


def test_v3_is_identity_only_over_exact_v2_collector() -> None:
    assert collector.AUTHORITY_FIELDS is collector.predecessor.AUTHORITY_FIELDS
    assert collector.EXPECTED_CAPS is collector.predecessor.EXPECTED_CAPS
    assert collector.EXPECTED_COUNTS is collector.predecessor.EXPECTED_COUNTS
    assert collector.EXPECTED_PERMISSIONS is collector.predecessor.EXPECTED_PERMISSIONS
    assert (
        collector.PROCESS_RESET_EQUIVALENCE_AUDIT_V3
        is collector.predecessor.PROCESS_RESET_EQUIVALENCE_AUDIT_V2
    )
    assert collector.SCENE_COUNT == 64
    assert collector.AUTHORITY_STATUS.endswith("INTEGRITY_REPLACEMENT_V3")
    assert collector.ATTEMPT_ID.endswith("integrity-replacement-v3")


def test_worker_argv_changes_only_script_entry_point() -> None:
    expected = collector._ORIGINAL_WORKER_ARGV_V2(**_worker_kwargs())  # noqa: SLF001
    observed = collector._worker_argv_v3(**_worker_kwargs())  # noqa: SLF001

    assert observed[0] == expected[0]
    assert Path(observed[1]).resolve() == Path(collector.__file__).resolve()
    assert Path(expected[1]).resolve() == Path(
        collector.predecessor.__file__
    ).resolve()
    assert observed[2:] == expected[2:]


def test_collector_overlay_is_narrow_and_restored() -> None:
    overrides = collector._configuration_overrides_v3()  # noqa: SLF001
    original = {
        name: getattr(collector.predecessor, name) for name in overrides
    }

    with collector._configured_predecessor_collector_v3():  # noqa: SLF001
        assert all(
            getattr(collector.predecessor, name) is value
            for name, value in overrides.items()
        )

    assert all(
        getattr(collector.predecessor, name) is value
        for name, value in original.items()
    )


def test_collect_delegates_with_no_policy_or_argument_change(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_collect(**kwargs):
        observed.update(kwargs)
        observed["attempt_id"] = collector.predecessor.ATTEMPT_ID
        observed["schema"] = collector.predecessor.SCENE_RESULT_SCHEMA
        observed["worker_argv"] = collector.predecessor._worker_argv_v2
        return {"status": "TEST"}, Path("/tmp/result.json")

    monkeypatch.setattr(collector.predecessor, "collect_v2", fake_collect)
    kwargs = {
        "plan_path": Path("/tmp/plan.json"),
        "expected_plan_byte_count": 123,
        "expected_plan_sha256": "a" * 64,
        "authority_path": Path("/tmp/authority.json"),
        "expected_authority_byte_count": 456,
        "expected_authority_sha256": "b" * 64,
    }

    assert collector.collect_v3(**kwargs) == (
        {"status": "TEST"},
        Path("/tmp/result.json"),
    )
    assert all(observed[name] == value for name, value in kwargs.items())
    assert observed["attempt_id"] == collector.ATTEMPT_ID
    assert observed["schema"] == collector.SCENE_RESULT_SCHEMA
    assert observed["worker_argv"] is collector._worker_argv_v3  # noqa: SLF001


def test_validation_delegates_under_same_identity(monkeypatch) -> None:
    def fake_validator(*args, **kwargs):
        return {
            "args": args,
            "kwargs": kwargs,
            "attempt_id": collector.predecessor.ATTEMPT_ID,
        }

    monkeypatch.setattr(
        collector.predecessor, "validate_scene_process_evidence_v2", fake_validator
    )
    observed = collector.validate_scene_process_evidence_v3(
        {"value": 1}, authority_binding={"value": 2}
    )
    assert observed["attempt_id"] == collector.ATTEMPT_ID
    assert observed["args"] == ({"value": 1},)
    assert observed["kwargs"] == {"authority_binding": {"value": 2}}
