"""Pure custody/lineage tests for the outcome-free projection interruption."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_contract_v1_2 as CONTRACT
from lewm.oracle import go2_scorer_projection_fix_interruption_v1 as I
from scripts import build_go2_branch_corpus_v1_2 as BUILDER


ZERO = {
    "candidate_outcomes_loaded": False,
    "branch_identities_created": False,
    "branches_attempted": 0,
    "frames_rendered": 0,
    "target_latents_encoded": 0,
    "scorer_training_started": False,
    "predictor_checkpoints_opened": 0,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _self_bound(payload: dict, key: str) -> dict:
    payload = dict(payload)
    payload[key] = I._digest(payload)
    return payload


def _fixture(monkeypatch, tmp_path, *, request_count=4, capture_count=3):
    corpus = tmp_path / I.CORPUS_ROOT_RELATIVE
    scorer = tmp_path / I.SCORER_ROOT_RELATIVE
    corpus.mkdir(parents=True)
    scorer.mkdir(parents=True)
    roots = {
        "request": str(
            I.CORPUS_ROOT_RELATIVE / "scorer_fit/requests/large_enclosed_maze"),
        "capture": str(
            I.CORPUS_ROOT_RELATIVE / "scorer_fit/captures/large_enclosed_maze"),
    }
    requests = []
    for index in range(request_count):
        request = _self_bound({
            "schema": "synthetic-request",
            "binding_receipt": False,
            "scene_ordinal": index,
            "scene": {"scene_id": f"scene-{index:02d}"},
            **ZERO,
        }, "mixed_replacement_scene_request_digest")
        name = f"{request['mixed_replacement_scene_request_digest']}.json"
        _write_json(tmp_path / roots["request"] / name, request)
        requests.append((name, request))
    # Production capture filenames are the request digests.  Choose the first
    # lexical request names so the exact two-phase row join is deterministic.
    for name, request in sorted(requests, key=lambda row: row[0])[:capture_count]:
        capture = _self_bound({
            "schema": "synthetic-capture",
            "mixed_replacement_scene_request_digest":
                request["mixed_replacement_scene_request_digest"],
            "request": request,
            "scene_id": request["scene"]["scene_id"],
            "chosen_state": None,
            "worker_failure": None,
            **ZERO,
        }, "mixed_replacement_scene_capture_digest")
        _write_json(tmp_path / roots["capture"] / name, capture)

    mixed = _self_bound(
        {"schema": "mixed", "source_repository_commit": "o" * 40},
        "mixed_precontract_disposition_receipt_digest")
    contract = _self_bound({
        "schema": "contract",
        "scorer_contract_v1_2_digest": "s" * 64,
        "mixed_precontract_disposition_receipt_digest":
            mixed["mixed_precontract_disposition_receipt_digest"],
    }, "contract_artifact_digest")
    launch = _self_bound({
        "schema": "launch",
        "scorer_contract_artifact_digest": contract["contract_artifact_digest"],
        "mixed_precontract_disposition_receipt_digest":
            mixed["mixed_precontract_disposition_receipt_digest"],
    }, "clean_source_launch_receipt_digest")
    payloads = {
        "mixed_precontract_disposition": mixed,
        "scorer_contract": contract,
        "clean_source_launch": launch,
    }
    managed = {
        "mixed_precontract_disposition": I.CORPUS_ROOT_RELATIVE,
        "scorer_contract": I.SCORER_ROOT_RELATIVE,
        "clean_source_launch": I.CORPUS_ROOT_RELATIVE,
    }
    active = {
        "mixed_precontract_disposition":
            I.CORPUS_ROOT_RELATIVE / "scorer_fit/mixed.json",
        "scorer_contract": I.SCORER_ROOT_RELATIVE / "contract.json",
        "clean_source_launch": I.CORPUS_ROOT_RELATIVE / "scorer_fit/launch.json",
    }
    bindings = {}
    for label, payload in payloads.items():
        path = tmp_path / active[label]
        _write_json(path, payload)
        raw = path.read_bytes()
        key = {
            "mixed_precontract_disposition":
                "mixed_precontract_disposition_receipt_digest",
            "scorer_contract": "contract_artifact_digest",
            "clean_source_launch": "clean_source_launch_receipt_digest",
        }[label]
        row = {
            "managed_root": str(managed[label]),
            "active_path": str(active[label]),
            "archive_path": str(
                managed[label] / "superseded_projection" / f"{label}.json"),
            "self_digest_key": key,
            "self_digest": payload[key],
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
        if label == "scorer_contract":
            row["scorer_contract_v1_2_digest"] = "s" * 64
        bindings[label] = row
    monkeypatch.setattr(I, "INTERRUPTED_ARTIFACTS", bindings)
    monkeypatch.setattr(I, "ATTEMPT_ROOTS", roots)
    monkeypatch.setattr(I, "ATTEMPT_REQUEST_COUNT", request_count)
    monkeypatch.setattr(I, "ATTEMPT_CAPTURE_COUNT", capture_count)
    monkeypatch.setattr(I, "ATTEMPT_ROW_COUNT", request_count + capture_count)
    rows = []
    for kind in ("request", "capture"):
        directory = tmp_path / roots[kind]
        for path in sorted(directory.glob("*.json"), key=lambda value: value.name):
            rows.append(I._attempt_row(kind, path, json.loads(path.read_text())))
    monkeypatch.setattr(I, "ATTEMPT_ROW_BYTE_COUNT",
                        sum(row["byte_count"] for row in rows))
    monkeypatch.setattr(I, "ATTEMPT_ROW_SET_DIGEST", I._digest(rows))
    monkeypatch.setattr(I, "ATTEMPT_ROW_SET_COMPACT_COUNTERFACTUAL_DIGEST",
                        hashlib.sha256(json.dumps(
                            rows, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False).encode()).hexdigest())
    return {"root": tmp_path, "rows": rows, "bindings": bindings,
            "roots": roots, "payloads": payloads}


def _issue(fixture):
    return I.issue_and_archive_interruption_receipt(
        source_repository_commit="n" * 40,
        clean_source_binding_digest="c" * 64,
        bound_implementations_digest="b" * 64,
        root=fixture["root"],
    )


def test_exact_live_row_digest_uses_repo_default_sorted_json_not_compact():
    # The supplied 69040d binding uses the repository's default-spaced
    # canonical JSON.  The compact rendering is recorded only to prevent a
    # future well-intentioned canonicalizer change from rewriting lineage.
    assert I.ATTEMPT_ROW_SET_DIGEST == (
        "69040d869803606142db92c472b9c273b55da3ff6c863948633a91f373c7795a"
    )
    assert I.ATTEMPT_ROW_SET_COMPACT_COUNTERFACTUAL_DIGEST == (
        "5b2c31adca903b86c58026166b6bc560ae2f3d39fbba888dfa9e4752bc116171"
    )
    assert I.ATTEMPT_ROW_SET_CANONICALIZATION.endswith("default separators")


def test_interruption_issuance_archives_exact_authorities_and_is_not_a_gate(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _issue(fixture)
    assert receipt["record_complete"] is True
    assert receipt["attempt_complete"] is False
    assert receipt["binding_receipt"] is False
    assert receipt["scientific_gate_input"] is False
    assert receipt["may_satisfy_selector_gate"] is False
    assert receipt["cryptographically_bound_by_successor_contract"] is True
    assert receipt["positive_candidate_observed_transiently_in_worker_memory"] \
        is True
    assert receipt["durable_validated_selected_state_artifact_count"] == 0
    assert receipt["attempt_rows"] == fixture["rows"]
    for binding in fixture["bindings"].values():
        assert not (tmp_path / binding["active_path"]).exists()
        archive = tmp_path / binding["archive_path"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == \
            binding["raw_sha256"]


@pytest.mark.parametrize("archived_count", (1, 2, 3))
def test_every_partial_archive_prefix_is_recovered(
        monkeypatch, tmp_path, archived_count):
    fixture = _fixture(monkeypatch, tmp_path)
    for binding in list(fixture["bindings"].values())[:archived_count]:
        active = tmp_path / binding["active_path"]
        archive = tmp_path / binding["archive_path"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active, archive)
    _issue(fixture)
    assert all((tmp_path / binding["archive_path"]).is_file()
               for binding in fixture["bindings"].values())


def test_hardlink_crash_is_recovered_and_archive_collision_is_fail_closed(
        monkeypatch, tmp_path):
    hardlink_root = tmp_path / "hardlink-recovery"
    hardlink = _fixture(monkeypatch, hardlink_root)
    first = hardlink["bindings"]["mixed_precontract_disposition"]
    active = hardlink_root / first["active_path"]
    archive = hardlink_root / first["archive_path"]
    archive.parent.mkdir(parents=True)
    os.link(active, archive)
    _issue(hardlink)
    assert not active.exists()
    assert archive.is_file()

    collision_root = tmp_path / "collision"
    collision = _fixture(monkeypatch, collision_root)
    first = collision["bindings"]["mixed_precontract_disposition"]
    archive = collision_root / first["archive_path"]
    archive.parent.mkdir(parents=True)
    archive.write_text("collision")
    with pytest.raises(I.InterruptionLineageError, match="archive collision"):
        _issue(collision)
    assert (collision_root / first["active_path"]).is_file()


def test_existing_receipt_ignores_successor_extras_but_reopens_exact_old_rows(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _issue(fixture)
    for kind in ("request", "capture"):
        (tmp_path / fixture["roots"][kind] / "successor-extra.json").write_text(
            "{\"successor\": true}\n")
    assert _issue(fixture) == receipt

    key = "preoutcome_projection_fix_interruption_receipt_digest"
    narrative = copy.deepcopy(receipt)
    narrative["reason"] = "resigned narrative change"
    narrative[key] = I._digest({name: value for name, value in narrative.items()
                                if name != key})
    with pytest.raises(I.InterruptionLineageError,
                       match="exact reconstruction"):
        I.validate_interruption_receipt(
            narrative,
            expected_source_repository_commit="n" * 40,
            expected_clean_source_binding_digest="c" * 64,
            expected_bound_implementations_digest="b" * 64,
            root=tmp_path,
        )

    old = tmp_path / fixture["roots"]["request"] / fixture["rows"][0]["name"]
    old.write_bytes(old.read_bytes() + b" ")
    with pytest.raises(I.InterruptionLineageError,
                       match="interruption attempt bytes changed"):
        I.load_and_validate_interruption_receipt(
            expected_source_repository_commit="n" * 40,
            expected_clean_source_binding_digest="c" * 64,
            expected_bound_implementations_digest="b" * 64,
            root=tmp_path,
        )

def test_resigned_receipt_tamper_and_descendant_symlink_are_rejected(
        monkeypatch, tmp_path):
    fixture = _fixture(monkeypatch, tmp_path)
    receipt = _issue(fixture)
    changed = copy.deepcopy(receipt)
    changed["attempt_rows"][0]["scene_id"] = "resigned-scene"
    key = "preoutcome_projection_fix_interruption_receipt_digest"
    changed[key] = I._digest({name: value for name, value in changed.items()
                              if name != key})
    with pytest.raises(I.InterruptionLineageError,
                       match="interruption attempt bytes changed"):
        I.validate_interruption_receipt(
            changed,
            expected_source_repository_commit="n" * 40,
            expected_clean_source_binding_digest="c" * 64,
            expected_bound_implementations_digest="b" * 64,
            root=tmp_path,
        )

    symlink_root = tmp_path / "symlink"
    fixture = _fixture(monkeypatch, symlink_root)
    external = tmp_path / "external"
    external.mkdir()
    binding = fixture["bindings"]["mixed_precontract_disposition"]
    archive_parent = (symlink_root / binding["archive_path"]).parent
    archive_parent.parent.mkdir(parents=True, exist_ok=True)
    archive_parent.symlink_to(external, target_is_directory=True)
    with pytest.raises(I.InterruptionLineageError,
                       match="lineage path contains a symlink"):
        _issue(fixture)
    assert not list(external.iterdir())


def test_contract_and_launch_bind_the_single_interruption_receipt(monkeypatch):
    source = {
        "source_repository_commit": "n" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    feasibility = {"state_selector_feasibility_receipt_digest": "f" * 64}
    mixed = {"mixed_precontract_disposition_receipt_digest": "m" * 64}
    binding = {
        "path": str(I.RECEIPT_RELATIVE_PATH),
        "receipt_digest": "i" * 64,
        "raw_sha256": "r" * 64,
        "byte_count": 123,
        "status": I.STATUS,
    }
    performance_binding = {
        "path": str(CONTRACT.PERFORMANCE_INTERRUPTION.RECEIPT_RELATIVE_PATH),
        "receipt_digest": "j" * 64,
        "raw_sha256": "s" * 64,
        "byte_count": 456,
        "status": CONTRACT.PERFORMANCE_INTERRUPTION.STATUS,
    }
    monkeypatch.setattr(
        CONTRACT.STATE_SELECTOR, "validate_frozen_reachability_feasibility_pass",
        lambda **_kwargs: feasibility)
    monkeypatch.setattr(
        CONTRACT.STATE_SELECTOR,
        "validate_preserved_state_mixed_precontract_disposition_receipt",
        lambda *_args, **_kwargs: None)
    artifact = CONTRACT._contract_artifact_payload(
        source, feasibility, mixed, binding, performance_binding)
    assert artifact["preoutcome_projection_fix_interruption_verified"] is True
    assert artifact["preoutcome_projection_fix_interruption"] == binding
    assert artifact["preoutcome_small_search_performance_interruption"] == \
        performance_binding

    monkeypatch.setattr(BUILDER, "_issued_scorer_contract_path", lambda: Path("x"))
    monkeypatch.setattr(
        BUILDER, "_load_issued_scorer_contract_at_path", lambda _path: artifact)
    monkeypatch.setattr(BUILDER, "_load_state_selector_preconditions", lambda **_kw: {
        "state_selector_feasibility_receipt_digest": "f" * 64,
        "mixed_precontract_disposition_receipt_digest": "m" * 64,
    })
    monkeypatch.setattr(BUILDER, "file_sha256", lambda _path: "a" * 64)
    launch = BUILDER._build_clean_source_launch_receipt({
        "pre_identity_validation_digest": "p" * 64})
    assert launch["preoutcome_projection_fix_interruption"] == binding
    assert launch["preoutcome_small_search_performance_interruption"] == \
        performance_binding
    assert "preoutcome_projection_fix_interruption" not in \
        BUILDER.LAUNCH_BINDING_KEYS
