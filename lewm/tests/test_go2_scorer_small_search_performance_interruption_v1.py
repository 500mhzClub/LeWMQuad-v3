"""Pure tests for the 24-hour outcome-free search interruption lineage."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_small_search_performance_interruption_v1 as I
from scripts import build_go2_branch_corpus_v1_2 as BUILDER


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _bound(payload: dict, key: str) -> dict:
    out = copy.deepcopy(payload)
    out[key] = I._digest(out)
    return out


def _zero() -> dict:
    return {
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
    }


def _raw_binding(path: Path, *, root: Path, active: Path, archive: Path,
                 key: str, payload: dict, managed: Path) -> dict:
    raw = path.read_bytes()
    return {
        "managed_root": str(managed),
        "active_path": str(active),
        "archive_path": str(archive),
        "self_digest_key": key,
        "self_digest": payload[key],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _fixture(monkeypatch, tmp_path: Path) -> dict:
    corpus = tmp_path / I.CORPUS_ROOT_RELATIVE
    scorer = tmp_path / I.SCORER_ROOT_RELATIVE
    corpus.mkdir(parents=True)
    scorer.mkdir(parents=True)
    fit = corpus / "scorer_fit"

    authorities = {}
    authority_specs = [
        ("mixed_precontract_disposition", I.CORPUS_ROOT_RELATIVE,
         I.SCORER_FIT_RELATIVE / "mixed.json", "mixed_digest"),
        ("scorer_contract", I.SCORER_ROOT_RELATIVE,
         I.SCORER_ROOT_RELATIVE / "contract.json", "contract_digest"),
        ("clean_source_launch", I.CORPUS_ROOT_RELATIVE,
         I.SCORER_FIT_RELATIVE / "launch.json", "launch_digest"),
        ("projection_fix_interruption", I.CORPUS_ROOT_RELATIVE,
         I.SCORER_FIT_RELATIVE / "projection.json", "projection_digest"),
    ]
    for label, managed, active, key in authority_specs:
        payload = _bound({"schema": label, "label": label}, key)
        path = tmp_path / active
        _write(path, payload)
        archive = Path(managed) / "archive" / f"{label}.json"
        authorities[label] = _raw_binding(
            path, root=tmp_path, active=active, archive=archive, key=key,
            payload=payload, managed=Path(managed))

    fixed = []
    all_fixed_rows = []
    for family, kind in (("family_mixed", "mixed"),
                         ("family_ordinary", "ordinary")):
        provenance = []
        for index in range(2):
            prefix = ("mixed_replacement_scene" if kind == "mixed"
                      else "state_resolution_scene")
            request_key = f"{prefix}_request_digest"
            capture_key = f"{prefix}_capture_digest"
            request = _bound({
                "schema": "request", "family": family,
                "scene_ordinal": index, "scene": {"scene_id": f"{family}-{index}"},
                **_zero(),
            }, request_key)
            name = f"{request[request_key]}.json"
            request_root = ("mixed_preoutcome_replacement_scene_requests_v2"
                            if kind == "mixed" else
                            "state_resolution_scene_requests_v1")
            capture_root = ("mixed_preoutcome_replacement_scene_captures_v2"
                            if kind == "mixed" else
                            "state_resolution_scene_captures_v1")
            request_rel = I.SCORER_FIT_RELATIVE / request_root / family / name
            capture_rel = I.SCORER_FIT_RELATIVE / capture_root / family / name
            _write(tmp_path / request_rel, request)
            capture = _bound({
                "schema": "capture", "family": family,
                "scene_id": f"{family}-{index}", "request": request,
                f"{prefix}_request_digest": request[request_key],
                "chosen_state": {"state_id": f"{family}-{index}"},
                "worker_failure": None, **_zero(),
            }, capture_key)
            _write(tmp_path / capture_rel, capture)
            pair = {
                "request_path": str(request_rel),
                "request_raw_sha256": hashlib.sha256(
                    (tmp_path / request_rel).read_bytes()).hexdigest(),
                "request_byte_count": (tmp_path / request_rel).stat().st_size,
                "capture_path": str(capture_rel),
                "capture_raw_sha256": hashlib.sha256(
                    (tmp_path / capture_rel).read_bytes()).hexdigest(),
                "capture_byte_count": (tmp_path / capture_rel).stat().st_size,
            }
            provenance.append(pair)
        provenance_key = ("mixed_replacement_scene_capture_provenance"
                          if kind == "mixed" else
                          "state_resolution_scene_capture_provenance")
        shard = {
            "schema": "old-shard", "family": family, "kind": kind,
            "states": [{"state_id": f"{family}-state", "scientific": 7}],
            provenance_key: provenance,
            "source_repository_commit": I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
            "clean_source_binding_digest": "old-clean",
            "bound_implementations_digest": "old-impl",
            "scorer_contract_artifact_digest": "old-artifact",
            "scorer_contract_v1_2_digest": "old-contract",
            "clean_source_launch_receipt_digest": "old-launch",
            "mixed_precontract_disposition_receipt_digest": "old-mixed",
            **_zero(),
        }
        shard = _bound(shard, "state_shard_digest")
        name = f"state_shard_{family}.json"
        active = I.SCORER_FIT_RELATIVE / name
        archive = I.ARCHIVE_ROOT_RELATIVE / "fixed_state_shards" / name
        _write(tmp_path / active, shard)
        binding = _raw_binding(
            tmp_path / active, root=tmp_path, active=active,
            archive=archive, key="state_shard_digest", payload=shard,
            managed=I.CORPUS_ROOT_RELATIVE)
        binding.update({"family": family, "kind": kind})
        rows = []
        for pair in provenance:
            for transport_kind in ("request", "capture"):
                logical = pair[f"{transport_kind}_path"]
                transport_path = tmp_path / logical
                rows.append(I._transport_row(
                    family=family, kind=transport_kind,
                    path=transport_path, logical_path=logical,
                    payload=json.loads(transport_path.read_text()),
                    mixed=kind == "mixed"))
        binding.update({
            "transport_row_count": len(rows),
            "transport_byte_count": sum(row["byte_count"] for row in rows),
            "transport_row_set_digest": I._digest([
                {key: value for key, value in row.items()
                 if key != "archive_path"} for row in rows]),
        })
        fixed.append(binding)
        all_fixed_rows.extend(rows)

    # Exact ordinary small prefix: 12 requests/captures, 5G+5S selected and
    # two negative captures.  This preserves the production cardinalities.
    small_rows = []
    selected_states = []
    found = {
        "general": 0,
        "safety_enriched": 0,
        "completion_enriched": 0,
    }
    required = {
        "general": 5,
        "safety_enriched": 5,
        "completion_enriched": 0,
    }
    negative_indices = {2, 8}
    for index in range(12):
        requested = [name for name in required
                     if found[name] < required[name]]
        request = _bound({
            "schema": "small-request", "family": "small_enclosed_maze",
            "scene_ordinal": index,
            "scene": {"scene_id": f"small_enclosed_maze_{index:02d}"},
            "found_before_scene": dict(found),
            "required_counts": dict(required),
            "requested_strata_in_priority_order": requested,
            "stratum_priority": list(required),
            "state_shard_bindings": {
                **{key: f"old-{key}" for key in I.SUCCESSOR_LINEAGE_KEYS},
                "scientific_binding": "frozen-science",
            },
            **_zero(),
        }, "state_resolution_scene_request_digest")
        name = f"{request['state_resolution_scene_request_digest']}.json"
        request_rel = Path(I.SMALL_PREFIX_ROOTS["request"]) / name
        capture_rel = Path(I.SMALL_PREFIX_ROOTS["capture"]) / name
        _write(tmp_path / request_rel, request)
        chosen = None
        if index not in negative_indices:
            stratum = ("general" if found["general"] < 5
                       else "safety_enriched")
            ordinal = found[stratum]
            chosen = {
                "state_id": f"scorer_fit-small_enclosed_maze-{stratum}-{ordinal:02d}",
                "state_identity_digest": f"{index + 1:064x}",
                "scene_id": f"small_enclosed_maze_{index:02d}",
                "stratum": stratum,
                "split_role": "calibration" if ordinal == 0 else "fit",
            }
            selected_states.append(chosen)
            found[stratum] += 1
        capture = _bound({
            "schema": "small-capture", "family": "small_enclosed_maze",
            "scene_id": f"small_enclosed_maze_{index:02d}",
            "state_resolution_scene_request_digest": request[
                "state_resolution_scene_request_digest"],
            "request": request, "chosen_state": chosen,
            "worker_failure": None, **_zero(),
        }, "state_resolution_scene_capture_digest")
        _write(tmp_path / capture_rel, capture)
    small_rows = []
    for transport_kind in ("request", "capture"):
        directory = tmp_path / I.SMALL_PREFIX_ROOTS[transport_kind]
        small_rows.extend(
            I._small_row(transport_kind, path, root=tmp_path)
            for path in sorted(directory.glob("*.json"), key=lambda value: value.name))

    monkeypatch.setattr(I, "INTERRUPTED_AUTHORITIES", authorities)
    monkeypatch.setattr(I, "FIXED_STATE_SHARDS", tuple(fixed))
    monkeypatch.setattr(I, "FIXED_TRANSPORT_ROW_COUNT", len(all_fixed_rows))
    monkeypatch.setattr(I, "FIXED_TRANSPORT_BYTE_COUNT",
                        sum(row["byte_count"] for row in all_fixed_rows))
    monkeypatch.setattr(I, "FIXED_TRANSPORT_ROW_SET_DIGEST", I._digest([
        {key: value for key, value in row.items() if key != "archive_path"}
        for row in all_fixed_rows]))
    monkeypatch.setattr(I, "SMALL_PREFIX_BYTE_COUNT",
                        sum(row["byte_count"] for row in small_rows))
    monkeypatch.setattr(I, "SMALL_PREFIX_ROW_SET_DIGEST", I._digest([
        {key: value for key, value in row.items()
         if key not in {"path", "archive_path"}} for row in small_rows]))
    projection = [{key: state[key] for key in (
        "state_id", "state_identity_digest", "scene_id", "stratum", "split_role")}
        for state in sorted(selected_states, key=lambda value: value["state_id"])]
    monkeypatch.setattr(I, "SMALL_PREFIX_STATE_PROJECTION_DIGEST",
                        I._digest(projection))
    monkeypatch.setattr(I, "SMALL_PREFIX_CURSOR_SCENE_ID",
                        "small_enclosed_maze_11")
    projection_calls = []
    monkeypatch.setattr(I.PROJECTION, "issue_and_archive_interruption_receipt",
                        lambda **kwargs: projection_calls.append(kwargs) or {})
    return {
        "root": tmp_path, "authorities": authorities, "fixed": fixed,
        "fixed_rows": all_fixed_rows, "small_rows": small_rows,
        "projection_calls": projection_calls,
    }


def _attestation(root: Path) -> dict:
    return BUILDER._phase1_outcome_surface_absence_attestation(root=root)


def _issue(fixture: dict) -> dict:
    return I.issue_and_archive_performance_interruption_receipt(
        source_repository_commit="a" * 40,
        clean_source_binding_digest="b" * 64,
        bound_implementations_digest="c" * 64,
        outcome_surface_absent=lambda: _attestation(fixture["root"]),
        revalidate_small_prefix=_archived_prefix_replay,
        root=fixture["root"],
    )


def _successor_bindings() -> dict:
    return {
        "source_repository_commit": "a" * 40,
        "clean_source_binding_digest": "b" * 64,
        "bound_implementations_digest": "c" * 64,
        "scorer_contract_artifact_digest": "d" * 64,
        "scorer_contract_v1_2_digest": "e" * 64,
        "clean_source_launch_receipt_digest": "f" * 64,
        "mixed_precontract_disposition_receipt_digest": "0" * 64,
    }


def _archived_prefix_replay(pairs):
    assert len(pairs) == 12
    assert pairs[-1]["capture"]["chosen_state"] is not None
    return True


def _semantic_replay(predecessor, rows, bindings):
    assert predecessor["source_repository_commit"] == \
        I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT
    assert rows and bindings["source_repository_commit"] == "a" * 40
    return True


def _successor_prefix_replay(predecessors, successors, bindings):
    assert len(predecessors) == len(successors) == 12
    assert bindings == _successor_bindings()
    for old, new in zip(predecessors, successors, strict=True):
        assert old["capture"]["chosen_state"] == new["capture"]["chosen_state"]
        assert (I._small_request_semantic_projection(old["request"])
                == I._small_request_semantic_projection(new["request"]))
        assert (I._small_capture_semantic_projection(old["capture"])
                == I._small_capture_semantic_projection(new["capture"]))
    return True


def test_static_execution_and_exact_real_inventory_bindings():
    assert I.INTERRUPTED_SOURCE_REPOSITORY_COMMIT == \
        "a1b89521bb825a0673d4663d2a9bff3f8f976a7d"
    assert I.CUTOFF_ELAPSED == "1-00:00:57"
    assert I.CUTOFF_CPU == "1-07:46:21"
    assert I.INTERRUPTED_PID == 1_204_602
    assert I.INTERRUPTED_CALL_CHAIN[-1] == "scipy.optimize.milp"
    assert I.FIXED_TRANSPORT_ROW_COUNT == 290
    assert I.FIXED_TRANSPORT_BYTE_COUNT == 4_852_501
    assert I.FIXED_TRANSPORT_ROW_SET_DIGEST == \
        "9af407320ab46c329626e5d1effde671f9f39cbadaa810ae161b56f61707f26a"
    assert I.SMALL_PREFIX_CURSOR_SCENE_ID == "small_enclosed_maze_100b36b62f36"


def test_issuance_archives_exact_bytes_is_not_gate_and_reissues_projection(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _issue(fixture)
    assert receipt["attempt_complete"] is False
    assert receipt["scientific_gate_input"] is False
    assert receipt["may_satisfy_selector_gate"] is False
    assert receipt["execution"]["terminal_state_shard_issued"] is False
    assert receipt["small_family_candidate_allocation_search_complete"] is False
    assert len(fixture["projection_calls"]) == 1
    for binding in [*fixture["authorities"].values(), *fixture["fixed"]]:
        assert not (tmp_path / binding["active_path"]).exists()
        assert (tmp_path / binding["archive_path"]).is_file()
    prefix = I.validated_small_fixed_prefix(receipt, root=tmp_path)
    assert prefix["general_count"] == prefix["safety_enriched_count"] == 5
    assert prefix["completion_enriched_count"] == 0


def test_idempotence_and_successor_coexistence(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    first = _issue(fixture)
    second = _issue(fixture)
    assert second == first
    assert len(fixture["projection_calls"]) == 2
    outputs = I.reissue_fixed_state_shards(
        receipt=first, revalidate_predecessor=_semantic_replay,
        build_successor_bindings=_successor_bindings,
        outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    again = I.reissue_fixed_state_shards(
        receipt=first, revalidate_predecessor=_semantic_replay,
        build_successor_bindings=_successor_bindings,
        outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert outputs == again
    for family, wrapper in outputs.items():
        assert wrapper["schema"] == I.REISSUED_SHARD_SCHEMA
        successor = I.validate_reissued_fixed_state_shard(
            wrapper, receipt=first, revalidate_predecessor=_semantic_replay,
            root=tmp_path)
        assert successor["source_repository_commit"] == "a" * 40
        predecessor = next(row for row in fixture["fixed"]
                           if row["family"] == family)
        old = json.loads((tmp_path / predecessor["archive_path"]).read_text())
        assert successor["states"] == old["states"]


def test_partial_archive_prefix_recovers_without_overwrite(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    rows = [*fixture["fixed_rows"], *fixture["small_rows"]]
    for row in rows[:3]:
        active = tmp_path / row["path"]
        archive = tmp_path / row["archive_path"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active, archive)
    first_authority = next(iter(fixture["authorities"].values()))
    authority_archive = tmp_path / first_authority["archive_path"]
    authority_archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path / first_authority["active_path"], authority_archive)
    receipt = _issue(fixture)
    assert receipt["fixed_transport_row_count"] == len(fixture["fixed_rows"])


def test_tampered_transport_and_self_resigned_receipt_fail_closed(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _issue(fixture)
    first = fixture["fixed_rows"][0]
    path = tmp_path / first["archive_path"]
    original_bytes = path.read_bytes()
    path.write_bytes(original_bytes + b" ")
    with pytest.raises(I.PerformanceInterruptionError):
        I.load_and_validate_performance_interruption_receipt(
            expected_source_repository_commit="a" * 40,
            expected_clean_source_binding_digest="b" * 64,
            expected_bound_implementations_digest="c" * 64,
            root=tmp_path)

    # Restore exactly and prove the baseline passes before the independent
    # self-resigned narrative check.
    path.write_bytes(original_bytes)
    I.load_and_validate_performance_interruption_receipt(
        expected_source_repository_commit="a" * 40,
        expected_clean_source_binding_digest="b" * 64,
        expected_bound_implementations_digest="c" * 64,
        root=tmp_path)
    altered = copy.deepcopy(receipt)
    altered["execution"]["cutoff_elapsed"] = "shorter"
    key = "preoutcome_small_search_performance_interruption_receipt_digest"
    altered[key] = I._digest({name: value for name, value in altered.items()
                              if name != key})
    with pytest.raises(I.PerformanceInterruptionError):
        I.validate_performance_interruption_receipt(
            altered, expected_source_repository_commit="a" * 40,
            expected_clean_source_binding_digest="b" * 64,
            expected_bound_implementations_digest="c" * 64,
            root=tmp_path)


def test_reissue_rejects_scientific_mutation_and_failed_semantic_replay(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _issue(fixture)
    with pytest.raises(I.PerformanceInterruptionError, match="semantic replay"):
        I.reissue_fixed_state_shards(
            receipt=receipt,
            revalidate_predecessor=lambda *_args: False,
            build_successor_bindings=_successor_bindings,
            outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    predecessor = json.loads(
        (tmp_path / fixture["fixed"][0]["archive_path"]).read_text())
    successor = I._successor_payload(predecessor, _successor_bindings())
    successor["states"][0]["scientific"] = 8
    assert predecessor["states"][0]["scientific"] == 7


def test_managed_alias_is_pinned_and_nested_symlink_is_rejected(
        monkeypatch, tmp_path):
    physical = tmp_path / "physical" / I.CORPUS_ROOT_RELATIVE.name
    physical.mkdir(parents=True)
    managed_parent = tmp_path / I.CORPUS_ROOT_RELATIVE.parent
    managed_parent.mkdir(parents=True, exist_ok=True)
    managed = tmp_path / I.CORPUS_ROOT_RELATIVE
    managed.symlink_to(physical)
    target = physical / "scorer_fit" / "receipt.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    pinned = I._pin_managed(
        I.CORPUS_ROOT_RELATIVE / "scorer_fit/receipt.json", root=tmp_path)
    assert pinned == target
    outside = tmp_path / "outside"
    outside.mkdir()
    nested = physical / "scorer_fit" / "nested"
    nested.symlink_to(outside)
    with pytest.raises(I.PerformanceInterruptionError, match="symlink"):
        I._pin_managed(
            I.CORPUS_ROOT_RELATIVE / "scorer_fit/nested/file.json",
            root=tmp_path)


def test_outcome_surface_blocks_issuance_before_any_archive(monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    forbidden = tmp_path / ".generated/go2_branch_corpus_v1_2/scorer_fit/branch_rows.jsonl"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("not allowed")
    with pytest.raises(I.PerformanceInterruptionError, match="outcome-surface"):
        _issue(fixture)
    assert all((tmp_path / binding["active_path"]).is_file()
               for binding in fixture["authorities"].values())


def test_public_validation_is_read_only_on_same_inode_collision(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _issue(fixture)
    binding = next(iter(fixture["authorities"].values()))
    active = tmp_path / binding["active_path"]
    archive = tmp_path / binding["archive_path"]
    active.parent.mkdir(parents=True, exist_ok=True)
    os.link(archive, active)
    before = (active.stat().st_ino, archive.stat().st_ino)
    with pytest.raises(I.PerformanceInterruptionError, match="collision"):
        I.load_and_validate_performance_interruption_receipt(
            expected_source_repository_commit="a" * 40,
            expected_clean_source_binding_digest="b" * 64,
            expected_bound_implementations_digest="c" * 64,
            root=tmp_path)
    assert active.is_file() and archive.is_file()
    assert (active.stat().st_ino, archive.stat().st_ino) == before


def test_issuer_recovers_hardlink_then_unlink_crash_prefix(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    rows = [fixture["fixed_rows"][0], fixture["small_rows"][0]]
    for row in rows:
        active = tmp_path / row["path"]
        archive = tmp_path / row["archive_path"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.link(active, archive)
        assert os.path.samefile(active, archive)
    receipt = _issue(fixture)
    assert receipt["record_complete"] is True
    for row in rows:
        assert not (tmp_path / row["path"]).exists()
        assert (tmp_path / row["archive_path"]).is_file()


def test_projection_reissue_failure_recovers_after_receipt_install(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    calls = []

    def fail_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("injected projection reissue crash")
        return {}

    monkeypatch.setattr(
        I.PROJECTION, "issue_and_archive_interruption_receipt", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        _issue(fixture)
    receipt_path = tmp_path / I.RECEIPT_RELATIVE_PATH
    assert receipt_path.is_file()
    receipt = _issue(fixture)
    assert receipt["record_complete"] is True
    assert len(calls) == 2


def test_successor_nested_binding_projection_allows_exactly_seven_keys():
    predecessor = {
        **{key: f"historical-{key}" for key in I.SUCCESSOR_LINEAGE_KEYS},
        "candidate_allocator_contract_digest": "frozen-allocator",
        "state_selector_amendment_digest": "frozen-selector",
    }
    projected = I.project_successor_state_shard_bindings(
        predecessor, _successor_bindings())
    assert set(projected) == set(predecessor)
    assert projected["candidate_allocator_contract_digest"] == \
        "frozen-allocator"
    assert projected["state_selector_amendment_digest"] == "frozen-selector"
    for key in I.SUCCESSOR_LINEAGE_KEYS:
        assert projected[key] == _successor_bindings()[key]
    malformed = _successor_bindings()
    malformed["unregistered"] = "1" * 64
    with pytest.raises(I.PerformanceInterruptionError, match="key surface"):
        I.project_successor_state_shard_bindings(predecessor, malformed)
    missing = dict(predecessor)
    missing.pop("clean_source_launch_receipt_digest")
    with pytest.raises(I.PerformanceInterruptionError, match="lack"):
        I.project_successor_state_shard_bindings(
            missing, _successor_bindings())


def test_small_prefix_reissue_is_normal_schema_exact_and_idempotent(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    performance = _issue(fixture)
    first = I.reissue_small_fixed_prefix(
        performance_receipt=performance,
        build_successor_bindings=_successor_bindings,
        revalidate_prefix=_successor_prefix_replay,
        outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert first["schema"] == I.SMALL_PREFIX_REISSUE_SCHEMA
    assert first["active_identity_input"] is True
    assert first["changes_scientific_selection"] is False
    assert first["successor_transport_row_count"] == 24
    receipt_path = tmp_path / I.SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH
    first_bytes = receipt_path.read_bytes()
    binding = I.small_prefix_reissue_receipt_binding(first, root=tmp_path)
    assert binding["receipt_digest"] == first[I.SMALL_PREFIX_REISSUE_SELF_KEY]
    for mapping in first["mapping_rows"]:
        for kind in ("request", "capture"):
            active = tmp_path / mapping[f"successor_{kind}"]["path"]
            assert active.is_file() and not active.is_symlink()
            payload = json.loads(active.read_text())
            assert payload["schema"] in {"small-request", "small-capture"}
        assert (tmp_path / mapping["archived_request"]["path"]).is_file()
        assert (tmp_path / mapping["archived_capture"]["path"]).is_file()
    second = I.reissue_small_fixed_prefix(
        performance_receipt=performance,
        build_successor_bindings=_successor_bindings,
        revalidate_prefix=_successor_prefix_replay,
        outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert second == first
    assert receipt_path.read_bytes() == first_bytes


def test_small_prefix_reissue_callback_fails_before_first_write(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    performance = _issue(fixture)
    with pytest.raises(I.PerformanceInterruptionError, match="did not pass"):
        I.reissue_small_fixed_prefix(
            performance_receipt=performance,
            build_successor_bindings=_successor_bindings,
            revalidate_prefix=lambda *_args: False,
            outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert not (tmp_path / I.SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH).exists()
    assert not [
        path for kind in ("request", "capture")
        for path in (tmp_path / I.SMALL_PREFIX_ROOTS[kind]).glob("*.json")]


def test_small_prefix_reissue_recovers_partial_atomic_prefix(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    performance = _issue(fixture)
    original_atomic = I._atomic_write
    calls = 0

    def fail_fifth(path, payload):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected successor prefix crash")
        return original_atomic(path, payload)

    monkeypatch.setattr(I, "_atomic_write", fail_fifth)
    with pytest.raises(RuntimeError, match="injected"):
        I.reissue_small_fixed_prefix(
            performance_receipt=performance,
            build_successor_bindings=_successor_bindings,
            revalidate_prefix=_successor_prefix_replay,
            outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert calls == 5
    monkeypatch.setattr(I, "_atomic_write", original_atomic)
    recovered = I.reissue_small_fixed_prefix(
        performance_receipt=performance,
        build_successor_bindings=_successor_bindings,
        revalidate_prefix=_successor_prefix_replay,
        outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert recovered["successor_transport_row_count"] == 24


def test_small_prefix_reissue_never_overwrites_collision(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    performance = _issue(fixture)
    archived_rows = I._small_prefix_rows(root=tmp_path, archived=True)
    projection = I._small_prefix_projection(
        archived_rows, root=tmp_path, require_archived=True)
    pair = I._project_small_prefix_pair(
        projection["pairs"][0], _successor_bindings())
    collision = tmp_path / pair["request_row"]["path"]
    collision.parent.mkdir(parents=True, exist_ok=True)
    collision.write_text("collision")
    before = collision.read_bytes()
    with pytest.raises(I.PerformanceInterruptionError, match="collision"):
        I.reissue_small_fixed_prefix(
            performance_receipt=performance,
            build_successor_bindings=_successor_bindings,
            revalidate_prefix=_successor_prefix_replay,
            outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert collision.read_bytes() == before


def test_small_prefix_reissue_rejects_unregistered_active_inventory(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    performance = _issue(fixture)
    unexpected = tmp_path / I.SMALL_PREFIX_ROOTS["capture"] / "unexpected.json"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}")
    with pytest.raises(I.PerformanceInterruptionError, match="inventory"):
        I.reissue_small_fixed_prefix(
            performance_receipt=performance,
            build_successor_bindings=_successor_bindings,
            revalidate_prefix=_successor_prefix_replay,
            outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    assert unexpected.read_text() == "{}"


def test_small_prefix_and_performance_binding_helpers_reject_other_payload(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    performance = _issue(fixture)
    altered_performance = copy.deepcopy(performance)
    altered_performance["small_family_candidate_allocation_search_complete"] = True
    key = "preoutcome_small_search_performance_interruption_receipt_digest"
    altered_performance[key] = I._digest({
        name: value for name, value in altered_performance.items()
        if name != key})
    with pytest.raises(I.PerformanceInterruptionError, match="binding payload"):
        I.receipt_binding(altered_performance, root=tmp_path)

    reissue = I.reissue_small_fixed_prefix(
        performance_receipt=performance,
        build_successor_bindings=_successor_bindings,
        revalidate_prefix=_successor_prefix_replay,
        outcome_surface_absent=lambda: _attestation(tmp_path), root=tmp_path)
    altered_reissue = copy.deepcopy(reissue)
    altered_reissue["changes_scientific_selection"] = True
    altered_reissue[I.SMALL_PREFIX_REISSUE_SELF_KEY] = I._digest({
        name: value for name, value in altered_reissue.items()
        if name != I.SMALL_PREFIX_REISSUE_SELF_KEY})
    with pytest.raises(I.PerformanceInterruptionError, match="binding payload"):
        I.small_prefix_reissue_receipt_binding(
            altered_reissue, root=tmp_path)
