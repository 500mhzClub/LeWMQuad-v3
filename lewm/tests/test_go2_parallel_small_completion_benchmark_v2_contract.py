"""Source-only custody tests for the one-shot V2 benchmark contract."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lewm.oracle import \
    go2_parallel_small_completion_benchmark_v2_contract as CONTRACT


SOURCE_COMMIT = "a" * 40


def _scientific_inputs() -> dict:
    return {
        "schema": CONTRACT.SCIENTIFIC_INPUT_BINDINGS_SCHEMA,
        "provisional_search_plan_digest": "1" * 64,
        "benchmark_source_binding_digest": "2" * 64,
        "rank_zero_source_identity_manifest_digest": "3" * 64,
        "rank_zero_state_projection_digest": "4" * 64,
        "candidate_pool_scene_ids_digest": "5" * 64,
        "fixed_state_projection_digest": "6" * 64,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }


def _source_rows() -> list[dict]:
    return [{
        "path": path,
        "byte_count": index + 1,
        "sha256": hashlib.sha256(path.encode()).hexdigest(),
    } for index, path in enumerate(CONTRACT.EXPECTED_V2_SOURCE_PATHS)]


def _v1_binding() -> dict:
    return CONTRACT._expected_v1_failure_binding()


def _contract() -> dict:
    return CONTRACT.build_contract(
        source_repository_commit=SOURCE_COMMIT,
        source_bindings=_source_rows(),
        v1_failure_binding=_v1_binding(),
        predecessor_scientific_input_bindings=_scientific_inputs(),
    )


def _resign(payload: dict) -> None:
    payload[CONTRACT.SELF_DIGEST_KEY] = CONTRACT.canonical_digest({
        key: value for key, value in payload.items()
        if key != CONTRACT.SELF_DIGEST_KEY
    })


def test_authorised_v2_surface_is_narrow_and_exact():
    assert CONTRACT.V1_FAILURE_STATUS_DESCRIPTOR == \
        "IMMUTABLE_FAIL_COLD_START_INCLUDED_IN_FIRST_TIMED_WAVE"
    assert CONTRACT.SAMPLE_PREFIX_INDICES == (0, 1, 2)
    assert CONTRACT.WORKER_COUNT == 32
    assert CONTRACT.MAXIMUM_PARALLEL_FRACTION == 0.5
    assert CONTRACT.BENCHMARK_CONTRACT["sample_zero_retained"] is True
    assert CONTRACT.BENCHMARK_CONTRACT[
        "sample_substitution_permitted"] is False
    assert CONTRACT.BENCHMARK_CONTRACT["overall_gate"] == \
        "median_gate AND maximum_gate"
    assert CONTRACT.EAGER_READINESS_PROCEDURE[
        "startup_cost_recorded_separately"] is True
    assert CONTRACT.EAGER_READINESS_PROCEDURE[
        "startup_cost_excluded_from_samples"] == [0, 1, 2]
    assert CONTRACT.EAGER_READINESS_PROCEDURE[
        "candidate_outcomes_consumed"] is False
    assert CONTRACT.EAGER_READINESS_PROCEDURE[
        "scientific_masks_accessed"] is False
    assert "predecessor_scientific_input_bindings_digest" in \
        CONTRACT.EAGER_READINESS_PROCEDURE["readiness_return"]
    assert CONTRACT.LIVE_POOL_CONTINUITY_CONTRACT[
        "search_uses_benchmark_pool"] is True
    assert CONTRACT.LIVE_POOL_CONTINUITY_CONTRACT[
        "worker_restart_count_required"] == 0
    assert CONTRACT.ONE_SHOT_TERMINAL_POLICY["attempt_count"] == 1
    assert CONTRACT.ONE_SHOT_TERMINAL_POLICY[
        "v2_retry_permitted"] is False
    assert CONTRACT.ONE_SHOT_TERMINAL_POLICY[
        "automatic_v3_permitted"] is False


def test_issuance_absence_surface_registers_exact_v2_and_v1_outputs():
    contract = _contract()
    v2_rows = contract["runtime_outputs_absent_at_issue"]
    v1_rows = contract["v1_downstream_outputs_absent_at_issue"]
    assert len(v2_rows) == 8
    assert len(v1_rows) == 6
    assert contract["runtime_outputs_absent_at_issue_digest"] == \
        CONTRACT.canonical_digest(v2_rows)
    assert contract["v1_downstream_outputs_absent_at_issue_digest"] == \
        CONTRACT.canonical_digest(v1_rows)
    assert all(row["lineage"] == "V2" and row["artifact_absent"] is True
               and row["exists"] is False and row["symlink"] is False
               for row in v2_rows)
    assert all(row["lineage"] == "V1" and row["artifact_absent"] is True
               and row["exists"] is False and row["symlink"] is False
               for row in v1_rows)
    registered = {row["path"] for row in [*v2_rows, *v1_rows]}
    assert str(CONTRACT.CONTRACT_RELATIVE_PATH) not in registered
    assert {row["path"] for row in v2_rows} == {
        str(path) for _label, path, _kind
        in CONTRACT.V2_RUNTIME_OUTPUT_PATHS
    }
    assert {row["path"] for row in v1_rows} == {
        str(path) for _label, path, _kind
        in CONTRACT.V1_DOWNSTREAM_OUTPUT_PATHS
    }


def test_v1_failure_is_preserved_as_valid_fail_not_erased():
    binding = _v1_binding()
    assert binding["benchmark_receipt_digest"] == \
        CONTRACT.V1_FAILURE_RECEIPT_DIGEST
    assert binding["source_repository_commit"] == \
        CONTRACT.V1_FAILURE_SOURCE_REPOSITORY_COMMIT
    assert binding["status_descriptor"] == \
        CONTRACT.V1_FAILURE_STATUS_DESCRIPTOR
    assert binding["passes"] is False
    assert binding["median_gate_passes"] is True
    assert binding["maximum_gate_passes"] is False
    assert binding["overall_verdict"] == "FAIL"
    assert binding["disposition"] == \
        "preserve_complete_lineage_do_not_retry_or_overwrite"


def test_build_and_validate_are_deterministic_exact_schema_and_self_bound():
    first = _contract()
    second = _contract()
    assert first == second
    assert first[CONTRACT.SELF_DIGEST_KEY] == CONTRACT.canonical_digest({
        key: value for key, value in first.items()
        if key != CONTRACT.SELF_DIGEST_KEY
    })
    assert CONTRACT.validate_contract(
        first,
        expected_predecessor_scientific_input_bindings=_scientific_inputs(),
        expected_source_repository_commit=SOURCE_COMMIT,
        validate_live_authorities=False,
    ) == first
    assert first["source_binding_set_digest"] == \
        CONTRACT.canonical_digest(_source_rows())
    assert first["predecessor_scientific_input_bindings_digest"] == \
        CONTRACT.canonical_digest(_scientific_inputs())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda row: row["benchmark_contract"].update(
            {"sample_prefix_indices": [1, 2]}), "binding|exact"),
        (lambda row: row["eager_readiness_procedure"].update(
            {"worker_count": 31}), "binding|exact"),
        (lambda row: row["live_pool_continuity_contract"].update(
            {"search_uses_benchmark_pool": False}), "binding|exact"),
        (lambda row: row["immutable_v1_failure_receipt"].update(
            {"passes": True}), "V1|exact"),
        (lambda row: row["runtime_outputs_absent_at_issue"][0].update(
            {"exists": True}), "binding|exact"),
        (lambda row: row["v1_downstream_outputs_absent_at_issue"][0].update(
            {"artifact_absent": False}), "binding|exact"),
        (lambda row: row.update(
            {"candidate_outcomes_consumed_at_issue": True}), "binding|exact"),
        (lambda row: row.update({"unexpected": False}), "binding|exact"),
    ],
)
def test_self_resigned_contract_tamper_fails_closed(mutator, match):
    payload = copy.deepcopy(_contract())
    mutator(payload)
    _resign(payload)
    with pytest.raises(CONTRACT.BenchmarkV2ContractError, match=match):
        CONTRACT.validate_contract(
            payload,
            expected_predecessor_scientific_input_bindings=
                _scientific_inputs(),
            expected_source_repository_commit=SOURCE_COMMIT,
            validate_live_authorities=False,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"candidate_outcomes_consumed": True},
        {"scientific_masks_accessed": True},
        {"benchmark_source_binding_digest": "x" * 64},
        {"extra": False},
    ],
)
def test_predecessor_input_envelope_is_exact_and_outcome_blind(mutation):
    bindings = _scientific_inputs()
    bindings.update(mutation)
    with pytest.raises(CONTRACT.BenchmarkV2ContractError,
                       match="binding|digest|pre-outcome"):
        CONTRACT.validate_predecessor_scientific_input_bindings(bindings)


def test_source_binding_coverage_and_order_are_exact():
    missing = _source_rows()[:-1]
    with pytest.raises(CONTRACT.BenchmarkV2ContractError,
                       match="coverage"):
        CONTRACT.build_contract(
            source_repository_commit=SOURCE_COMMIT,
            source_bindings=missing,
            v1_failure_binding=_v1_binding(),
            predecessor_scientific_input_bindings=_scientific_inputs(),
        )
    reordered = list(reversed(_source_rows()))
    with pytest.raises(CONTRACT.BenchmarkV2ContractError,
                       match="source binding"):
        CONTRACT.build_contract(
            source_repository_commit=SOURCE_COMMIT,
            source_bindings=reordered,
            v1_failure_binding=_v1_binding(),
            predecessor_scientific_input_bindings=_scientific_inputs(),
        )


def _prepare_issue_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, list[dict]]:
    contract_path = tmp_path / CONTRACT.CONTRACT_RELATIVE_PATH
    contract_path.parent.mkdir(parents=True)
    rows = _source_rows()
    monkeypatch.setattr(
        CONTRACT, "_clean_source_commit", lambda **_options: SOURCE_COMMIT)
    monkeypatch.setattr(
        CONTRACT, "_read_source_bindings", lambda **_options: rows)
    monkeypatch.setattr(
        CONTRACT, "load_v1_failure_binding", lambda **_options: _v1_binding())
    return contract_path, rows


def test_issuer_writes_only_dedicated_path_and_is_exactly_idempotent(
        tmp_path, monkeypatch):
    path, _rows = _prepare_issue_fixture(tmp_path, monkeypatch)
    before_files = {row for row in tmp_path.rglob("*") if row.is_file()}
    assert before_files == set()

    first = CONTRACT.issue_contract(
        path,
        predecessor_scientific_input_bindings=_scientific_inputs(),
        source_repository_commit=SOURCE_COMMIT,
        root=tmp_path,
    )
    raw_before = path.read_bytes()
    second = CONTRACT.issue_contract(
        path,
        predecessor_scientific_input_bindings=_scientific_inputs(),
        source_repository_commit=SOURCE_COMMIT,
        root=tmp_path,
    )

    assert first == second
    assert path.read_bytes() == raw_before
    assert {row for row in tmp_path.rglob("*") if row.is_file()} == {path}
    assert json.loads(raw_before) == first


def test_issuer_never_overwrites_a_collision(tmp_path, monkeypatch):
    path, _rows = _prepare_issue_fixture(tmp_path, monkeypatch)
    collision = b'{"schema":"not-the-v2-contract"}\n'
    path.write_bytes(collision)

    with pytest.raises(CONTRACT.BenchmarkV2ContractError,
                       match="corrupt|binding|contract"):
        CONTRACT.issue_contract(
            path,
            predecessor_scientific_input_bindings=_scientific_inputs(),
            source_repository_commit=SOURCE_COMMIT,
            root=tmp_path,
        )
    assert path.read_bytes() == collision


def test_issuer_requires_contract_before_any_pool_or_search_state(
        tmp_path, monkeypatch):
    path, _rows = _prepare_issue_fixture(tmp_path, monkeypatch)
    receipt = CONTRACT.issue_contract(
        path,
        predecessor_scientific_input_bindings=_scientific_inputs(),
        root=tmp_path,
    )
    assert receipt["worker_pool_constructed_at_issue"] is False
    assert receipt["scientific_search_plan_issued_at_issue"] is False
    assert receipt["candidate_outcomes_consumed_at_issue"] is False
    assert receipt["scientific_masks_accessed_at_issue"] is False


def test_wrong_logical_contract_path_is_rejected_without_write(
        tmp_path, monkeypatch):
    _path, _rows = _prepare_issue_fixture(tmp_path, monkeypatch)
    wrong = tmp_path / ".generated/wrong-contract.json"
    with pytest.raises(CONTRACT.BenchmarkV2ContractError, match="path"):
        CONTRACT.issue_contract(
            wrong,
            predecessor_scientific_input_bindings=_scientific_inputs(),
            root=tmp_path,
        )
    assert not wrong.exists()


@pytest.mark.parametrize("lineage", ["V2", "V1"])
def test_preexisting_runtime_or_v1_downstream_output_blocks_install(
        tmp_path, monkeypatch, lineage):
    contract_path, _rows = _prepare_issue_fixture(tmp_path, monkeypatch)
    registry = (CONTRACT.V2_RUNTIME_OUTPUT_PATHS if lineage == "V2"
                else CONTRACT.V1_DOWNSTREAM_OUTPUT_PATHS)
    collision = tmp_path / registry[0][1]
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_text("preexisting\n")

    with pytest.raises(CONTRACT.BenchmarkV2ContractError,
                       match="predates|output"):
        CONTRACT.issue_contract(
            contract_path,
            predecessor_scientific_input_bindings=_scientific_inputs(),
            root=tmp_path,
        )
    assert not contract_path.exists()
    assert collision.read_text() == "preexisting\n"


def test_issuer_rechecks_absence_immediately_before_exclusive_install(
        tmp_path, monkeypatch):
    contract_path, _rows = _prepare_issue_fixture(tmp_path, monkeypatch)
    original = CONTRACT._audit_issuance_output_absence
    calls = 0
    late = tmp_path / CONTRACT.V2_RUNTIME_OUTPUT_PATHS[1][1]

    def audited(*, root):
        nonlocal calls
        calls += 1
        if calls == 2:
            late.write_text("late-runtime-output\n")
        return original(root=root)

    monkeypatch.setattr(CONTRACT, "_audit_issuance_output_absence", audited)
    with pytest.raises(CONTRACT.BenchmarkV2ContractError,
                       match="predates|output"):
        CONTRACT.issue_contract(
            contract_path,
            predecessor_scientific_input_bindings=_scientific_inputs(),
            root=tmp_path,
        )
    assert calls == 2
    assert not contract_path.exists()
    assert late.read_text() == "late-runtime-output\n"


def test_later_load_does_not_require_issuance_outputs_to_remain_absent(
        tmp_path, monkeypatch):
    contract_path, _rows = _prepare_issue_fixture(tmp_path, monkeypatch)
    issued = CONTRACT.issue_contract(
        contract_path,
        predecessor_scientific_input_bindings=_scientific_inputs(),
        root=tmp_path,
    )
    later = tmp_path / CONTRACT.V2_RUNTIME_OUTPUT_PATHS[0][1]
    later.write_text("later-readiness-record\n")
    later_v1 = tmp_path / CONTRACT.V1_DOWNSTREAM_OUTPUT_PATHS[0][1]
    later_v1.write_text("later-v1-lineage-witness\n")

    loaded = CONTRACT.load_contract(
        contract_path,
        expected_predecessor_scientific_input_bindings=_scientific_inputs(),
        expected_source_repository_commit=SOURCE_COMMIT,
        root=tmp_path,
    )
    assert loaded == issued
    assert later.is_file()
    assert later_v1.is_file()
