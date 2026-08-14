"""Non-Genesis durability tests for the v1.2 branch corpus pipeline."""
from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import build_go2_branch_corpus_v1_2 as B


def _archived_pair(
        *, kind: str, ordinal: int, scene_id: str,
        request_extra: dict, chosen_state: dict | None,
        rejection_reasons: dict[str, int] | None = None,
        provenance_extra: dict | None = None):
    request_key = (
        "mixed_replacement_scene_request_digest" if kind == "mixed" else
        "state_resolution_scene_request_digest")
    capture_key = (
        "mixed_replacement_scene_capture_digest" if kind == "mixed" else
        "state_resolution_scene_capture_digest")
    request_digest = f"{ordinal + 1:064x}"
    capture_digest = f"{ordinal + 101:064x}"
    request = {
        "scene_ordinal": ordinal,
        "scene": {"scene_id": scene_id},
        request_key: request_digest,
        **request_extra,
    }
    capture = {
        "scene_id": scene_id,
        "request": request,
        request_key: request_digest,
        capture_key: capture_digest,
        "chosen_state": chosen_state,
        "scene_rejection_reasons": dict(rejection_reasons or {}),
        "worker_failure": None,
    }
    request_path = f"requests/{request_digest}.json"
    capture_path = f"captures/{request_digest}.json"
    request_sha = f"{ordinal + 201:064x}"
    capture_sha = f"{ordinal + 301:064x}"
    provenance = {
        "scene_id": scene_id,
        request_key: request_digest,
        capture_key: capture_digest,
        "request_path": request_path,
        "request_raw_sha256": request_sha,
        "request_byte_count": 1000 + ordinal,
        "capture_path": capture_path,
        "capture_raw_sha256": capture_sha,
        "capture_byte_count": 2000 + ordinal,
        **dict(provenance_extra or {}),
    }
    projected = {
        "request": copy.deepcopy(request),
        "capture": copy.deepcopy(capture),
        "old_request": copy.deepcopy(request),
        "old_capture": copy.deepcopy(capture),
        "request_row": {
            "kind": "request", "path": request_path,
            "raw_sha256": request_sha, "byte_count": 1000 + ordinal,
        },
        "capture_row": {
            "kind": "capture", "path": capture_path,
            "raw_sha256": capture_sha, "byte_count": 2000 + ordinal,
        },
    }
    return provenance, projected


def _mock_interruption(monkeypatch):
    projection_binding = {
        "path": str(B.INTERRUPTION.RECEIPT_RELATIVE_PATH),
        "receipt_digest": "1" * 64,
        "raw_sha256": "2" * 64,
        "byte_count": 123,
        "status": B.INTERRUPTION.STATUS,
    }
    transition_binding = {
        "path": str(B.REISSUE_VALIDATION_INTERRUPTION.RECEIPT_RELATIVE_PATH),
        "receipt_digest": "3" * 64,
        "raw_sha256": "4" * 64,
        "byte_count": 456,
        "status": B.REISSUE_VALIDATION_INTERRUPTION.STATUS,
    }
    performance_binding = {
        "path": str(B.PERFORMANCE_INTERRUPTION.V2_RECEIPT_RELATIVE_PATH),
        "receipt_digest": "5" * 64,
        "raw_sha256": "6" * 64,
        "byte_count": 789,
        "status": B.PERFORMANCE_INTERRUPTION.V2_STATUS,
    }
    receipt = {"synthetic": "validated interruption receipt"}
    monkeypatch.setattr(
        B.INTERRUPTION, "load_and_validate_interruption_receipt",
        lambda **_kwargs: receipt)
    monkeypatch.setattr(
        B.INTERRUPTION, "receipt_binding",
        lambda _receipt, **_kwargs: dict(projection_binding))
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION,
        "load_and_validate_interruption_receipt",
        lambda **_kwargs: receipt)
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION, "receipt_binding",
        lambda _receipt, **_kwargs: dict(transition_binding))
    monkeypatch.setattr(
        B.PERFORMANCE_INTERRUPTION,
        "load_and_validate_performance_interruption_receipt_v2",
        lambda **_kwargs: receipt)
    monkeypatch.setattr(
        B.PERFORMANCE_INTERRUPTION,
        "performance_interruption_receipt_binding_v2",
        lambda _receipt, **_kwargs: dict(performance_binding))
    return {
        "projection": projection_binding,
        "transition": transition_binding,
        "performance": performance_binding,
    }


def test_pre_identity_validation_reopens_transition_certified_bytes_without_milp(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    pool_root = output_root / "scorer_fit"
    pool_root.mkdir(parents=True)
    artifact_path = pool_root / B.PRE_IDENTITY_VALIDATION_NAME
    raw = b'{"synthetic_version": 1}\n'
    artifact = {"synthetic_version": 1}
    transition = {"synthetic": "fully validated transition"}
    calls = []
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    monkeypatch.setattr(
        B, "_load_current_reissue_validation_interruption",
        lambda: transition)
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION,
        "validate_retained_preidentity_artifact",
        lambda receipt, **_kwargs: (
            calls.append(receipt) or dict(artifact)))
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION,
        "RETAINED_PREIDENTITY_ARTIFACT", {
            "byte_count": len(raw),
            "raw_sha256": B.hashlib.sha256(raw).hexdigest(),
        })
    monkeypatch.setattr(
        B.ALLOC, "validate_pre_identity_structural_validation",
        lambda *_args, **_kwargs: pytest.fail("MILP validator was reached"))
    monkeypatch.setattr(
        B.ALLOC, "build_pre_identity_structural_validation",
        lambda *_args, **_kwargs: pytest.fail("MILP builder was reached"))

    artifact_path.write_bytes(raw)
    first = B._load_pre_identity_allocation_validation()
    first["synthetic_version"] = 99
    second = B._load_pre_identity_allocation_validation()
    assert second == artifact
    assert calls == [transition, transition]

    artifact_path.write_bytes(b'{"synthetic_version": 2}\n')
    with pytest.raises(RuntimeError, match="differs from its transition proof"):
        B._load_pre_identity_allocation_validation()


def test_preflight_reuses_one_exact_validated_artifact_without_rebuilding(
        tmp_path, monkeypatch):
    out = tmp_path / "scorer_fit"
    out.mkdir()
    artifact = {
        "pre_identity_validation_digest": "a" * 64,
        "global": {"state_slot_count": 120, "candidate_slot_count": 720},
        "goal_type_validation": {
            "status": "NOT_EVALUABLE_BEFORE_STATE_IDENTITIES",
        },
    }
    artifact_path = out / B.PRE_IDENTITY_VALIDATION_NAME
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    launch = {
        "clean_source_launch_receipt_digest": "b" * 64,
        "source_repository_commit": "c" * 40,
        "pre_identity_allocation_validation_digest": "a" * 64,
    }
    monkeypatch.setattr(B, "_load_issued_scorer_contract", lambda: {})
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(
        B.ALLOC, "validate_allocation_amendment_artifact",
        lambda _payload: None)
    monkeypatch.setattr(
        B.ALLOC, "validate_pre_identity_structural_validation",
        lambda *_args, **_kwargs: pytest.fail("MILP validator was reached"))
    monkeypatch.setattr(
        B.ALLOC, "build_pre_identity_structural_validation",
        lambda: pytest.fail("valid retained preflight artifact was rebuilt"))
    monkeypatch.setattr(
        B, "_load_pre_identity_allocation_validation", lambda: dict(artifact))
    monkeypatch.setattr(
        B, "_build_clean_source_launch_receipt", lambda payload: (
            dict(launch) if payload == artifact else
            pytest.fail("launch received a different preflight artifact")))
    assert B.issue_pre_identity_allocation_validation(out) == 0
    assert B.issue_pre_identity_allocation_validation(out) == 0
    assert json.loads(artifact_path.read_text()) == artifact
    assert json.loads((out / B.LAUNCH_RECEIPT_NAME).read_text()) == launch


def test_fixed_reissue_interruption_stage_records_only_transition(
        monkeypatch):
    source = {
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    receipt = {
        "status": B.REISSUE_VALIDATION_INTERRUPTION.STATUS,
        "fixed_wrapper_count_issued": 0,
        "preidentity_exact_proof_reuse_only": True,
        "scientific_gate_input": False,
    }
    captured = {}
    monkeypatch.setattr(B, "clean_source_binding", lambda: dict(source))

    def issue(**kwargs):
        captured.update(kwargs)
        return dict(receipt)

    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION,
        "issue_and_archive_interruption_receipt", issue)
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION, "receipt_binding",
        lambda payload, **_kwargs: {"status": payload["status"]})
    assert B.stage_fixed_reissue_validation_interruption() == 0
    assert captured["source_repository_commit"] == "a" * 40
    assert captured["clean_source_binding_digest"] == B.canonical_digest(source)
    assert captured["bound_implementations_digest"] == "b" * 64
    assert callable(captured["outcome_surface_absent"])
    assert "execution_argv" not in captured
    assert "interpreter_versions" not in captured


def test_performance_interruption_stage_issues_transition_bound_v2(
        monkeypatch):
    source = {
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    transition = {"transition": "validated"}
    transition_binding = {"transition": "binding"}
    predecessor = {"performance": "archived-v1"}
    predecessor_binding = {"performance": "archived-v1-binding"}
    projection = {"projection": "current"}
    v2 = {
        "status": B.PERFORMANCE_INTERRUPTION.V2_STATUS,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
    }
    captured = {}
    monkeypatch.setattr(B, "clean_source_binding", lambda: dict(source))
    monkeypatch.setattr(
        B, "_load_current_reissue_validation_interruption",
        lambda: transition)
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION, "receipt_binding",
        lambda payload, **_kwargs: (
            transition_binding if payload is transition else
            pytest.fail("unexpected transition payload")))
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION,
        "load_archived_performance_receipt_v1",
        lambda payload, **_kwargs: (
            predecessor if payload is transition else
            pytest.fail("unexpected transition payload")))
    monkeypatch.setattr(
        B.REISSUE_VALIDATION_INTERRUPTION,
        "archived_performance_receipt_binding_v1",
        lambda payload, **_kwargs: (
            predecessor_binding if payload is transition else
            pytest.fail("unexpected transition payload")))
    monkeypatch.setattr(
        B.INTERRUPTION, "load_and_validate_interruption_receipt",
        lambda **_kwargs: projection)

    def issue_v2(**kwargs):
        captured.update(kwargs)
        return dict(v2)

    monkeypatch.setattr(
        B.PERFORMANCE_INTERRUPTION,
        "issue_performance_interruption_receipt_v2", issue_v2)
    monkeypatch.setattr(
        B.PERFORMANCE_INTERRUPTION,
        "performance_interruption_receipt_binding_v2",
        lambda payload, **_kwargs: {"status": payload["status"]})
    assert B.stage_small_search_performance_interruption() == 0
    assert captured["source_transition_receipt_binding"] is transition_binding
    assert captured["predecessor_v1_receipt"] is predecessor
    assert captured["predecessor_v1_receipt_binding"] is predecessor_binding
    assert captured["current_projection_receipt"] is projection
    assert captured["revalidate_small_prefix"] is \
        B._revalidate_performance_interrupted_small_prefix
    assert callable(captured["outcome_surface_absent"])


def test_mixed_request_validator_uses_supplied_state_shard_bindings(
        tmp_path, monkeypatch):
    family = str(B.STATE_SELECTOR.PRESERVED_STATE_SHARDS[0]["family"])
    scene_dir = tmp_path / "train" / family / "scene-a"
    scene_dir.mkdir(parents=True)
    (scene_dir / "manifest.json").write_text("{}\n")
    slot = {"state_id": "replacement-0"}
    interval = {
        "lower_scene_id_exclusive": None,
        "upper_scene_id_exclusive": None,
        "vacant_ordinals": [0],
        "replacement_slots": [slot],
    }
    plan = {
        "retained_scene_ids": [],
        "rejected_identity_digests": ["a" * 64],
        "interval_groups": [interval],
    }
    bindings = {"exact": "precomputed-state-shard-bindings"}
    args = B.argparse.Namespace(
        pool="scorer_fit", family=family, backend="cpu")
    monkeypatch.setattr(
        B, "_mixed_family_replacement_plan", lambda _family: plan)
    monkeypatch.setattr(
        B, "_state_shard_bindings",
        lambda *_args, **_kwargs: dict(bindings))
    monkeypatch.setattr(
        B, "_pin_generated_path", lambda raw, _expected, **_kwargs: raw)
    request = B._build_mixed_replacement_scene_request(
        args=args, out=tmp_path, scene_dir=scene_dir, scene_ordinal=0,
        interval=interval, slot=slot, accepted_scene_ids_before=[],
        exclusion={}, family_allow_list=[scene_dir.name], persist=False)

    monkeypatch.setattr(
        B, "_state_shard_bindings",
        lambda *_args, **_kwargs: pytest.fail(
            "supplied mixed-request bindings were recomputed"))
    B._validate_mixed_replacement_scene_request(
        request, args=args, out=tmp_path, pool={family: [scene_dir]},
        exclusion={}, expected_state_shard_bindings=bindings)


@pytest.mark.parametrize("kind", ("ordinary", "mixed"))
def test_fixed_shard_replay_threads_one_precomputed_binding_to_every_request(
        kind, tmp_path, monkeypatch):
    family = f"synthetic-{kind}"
    request_key = (
        "mixed_replacement_scene_request_digest" if kind == "mixed" else
        "state_resolution_scene_request_digest")
    capture_key = (
        "mixed_replacement_scene_capture_digest" if kind == "mixed" else
        "state_resolution_scene_capture_digest")
    provenance_key = (
        "mixed_replacement_scene_capture_provenance" if kind == "mixed" else
        "state_resolution_scene_capture_provenance")
    lineage_keys = tuple(B.PERFORMANCE_INTERRUPTION.SUCCESSOR_LINEAGE_KEYS)
    expected_bindings = {
        **{key: f"current-{index}" for index, key in enumerate(lineage_keys)},
        "unchanged_binding": "fixed",
    }
    old_bindings = {
        **{key: f"historical-{index}" for index, key in enumerate(lineage_keys)},
        "unchanged_binding": "fixed",
    }
    successor_bindings = {
        key: expected_bindings[key] for key in lineage_keys
    }
    request = {
        "state_shard_bindings": old_bindings,
        request_key: "1" * 64,
        "scientific_request_field": "unchanged",
    }
    capture = {
        "request": request,
        request_key: request[request_key],
        capture_key: "2" * 64,
        "scientific_capture_field": "unchanged",
        "worker_failure": None,
    }
    request_path = tmp_path / "archived-request.json"
    capture_path = tmp_path / "archived-capture.json"
    request_path.write_text(json.dumps(request))
    capture_path.write_text(json.dumps(capture))
    logical_request = "requests/archived-request.json"
    logical_capture = "captures/archived-capture.json"
    predecessor = {
        "family": family,
        provenance_key: [{
            "request_path": logical_request,
            "capture_path": logical_capture,
        }],
    }
    transport_rows = [
        {"path": logical_request, "archive_path": str(request_path)},
        {"path": logical_capture, "archive_path": str(capture_path)},
    ]
    binding_calls = []
    validator_bindings = []

    def state_shard_bindings(*_args, **_kwargs):
        binding_calls.append(True)
        return dict(expected_bindings)

    def validate_request(
            _request, *, expected_state_shard_bindings, **_kwargs):
        validator_bindings.append(dict(expected_state_shard_bindings))

    monkeypatch.setattr(
        B.PERFORMANCE_INTERRUPTION, "FIXED_STATE_SHARDS",
        ({"family": family, "kind": kind},))
    monkeypatch.setattr(B, "scene_pool", lambda _pool: ({family: [tmp_path]}, {}))
    monkeypatch.setattr(B, "_state_shard_bindings", state_shard_bindings)
    monkeypatch.setattr(
        B, "_validate_interrupted_state_identity_bindings",
        lambda bindings: bindings)
    monkeypatch.setattr(
        B.PERFORMANCE_INTERRUPTION, "_pin_managed",
        lambda path, **_kwargs: Path(path))
    if kind == "mixed":
        monkeypatch.setattr(
            B, "_validate_mixed_replacement_scene_request", validate_request)
        monkeypatch.setattr(
            B, "_validate_mixed_replacement_scene_capture",
            lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            B, "_replay_projected_mixed_fixed_shard",
            lambda **_kwargs: None)
    else:
        monkeypatch.setattr(
            B, "_validate_state_resolution_scene_request", validate_request)
        monkeypatch.setattr(
            B, "_validate_state_resolution_scene_capture",
            lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            B, "_replay_projected_ordinary_fixed_shard",
            lambda **_kwargs: None)

    assert B._revalidate_performance_interrupted_fixed_shard(
        predecessor, transport_rows, successor_bindings) is True
    assert len(binding_calls) == 1
    assert validator_bindings == [expected_bindings]


def test_archived_ordinary_wrapper_replays_dynamic_quota_and_first_cursor():
    required = dict(B.POOLS["scorer_fit"]["strata"])
    found = {name: 0 for name in required}
    provenance = []
    projected = []
    states = []
    rejections = {}
    ordinal = 0
    for stratum in B.STRATA:
        for index in range(required[stratum]):
            scene_id = f"scene-{ordinal:03d}"
            state = {
                "state_id": f"state-{stratum}-{index}",
                "scene_id": scene_id,
                "stratum": stratum,
                "state_identity_digest": f"{ordinal + 401:064x}",
            }
            requested = [name for name in B.STRATA
                         if found[name] < required[name]]
            row, pair = _archived_pair(
                kind="ordinary", ordinal=ordinal, scene_id=scene_id,
                request_extra={
                    "found_before_scene": dict(found),
                    "required_counts": dict(required),
                    "requested_strata_in_priority_order": requested,
                },
                chosen_state=state,
                rejection_reasons={"synthetic": ordinal},
            )
            provenance.append(row)
            projected.append(pair)
            states.append(state)
            rejections[scene_id] = {"synthetic": ordinal}
            found[stratum] += 1
            ordinal += 1
    states.sort(key=lambda state: (
        B.STRATA.index(state["stratum"]), state["scene_id"]))
    transport = {
        "schema": "go2_branch_corpus_v1_2_state_resolution_transport_v1",
        "one_scene_per_subprocess": True,
        "atomic_capture_write_before_native_cleanup": True,
        "return_code_ignored_only_after_valid_capture": True,
        "resume_scope": "MISSING_OR_INVALID_SCENE_CAPTURES_ONLY",
        "resolver_algorithm_digest":
            B.canonical_digest(B.STATE_RESOLUTION_REDUCER_CONTRACT),
        "resolver_cursor_scene_id": provenance[-1]["scene_id"],
        "scene_capture_count": len(provenance),
        "scene_capture_provenance_digest": B.canonical_digest(provenance),
        "candidate_outcomes_loaded": False,
    }
    predecessor = {
        "states": states,
        "scene_rejection_reasons": rejections,
        "state_resolution_subprocess_transport": transport,
        "state_resolution_scene_capture_provenance": provenance,
    }
    B._replay_projected_ordinary_fixed_shard(
        predecessor=predecessor, family="ordinary", projected_pairs=projected)

    tampered = copy.deepcopy(projected)
    tampered[6]["request"]["found_before_scene"]["general"] = 4
    with pytest.raises(RuntimeError, match="dynamic quota prefix"):
        B._replay_projected_ordinary_fixed_shard(
            predecessor=predecessor, family="ordinary",
            projected_pairs=tampered)


def test_archived_mixed_wrapper_replays_interval_slot_and_stop_prefix(
        tmp_path, monkeypatch):
    family = "mixed-family"
    retained = {
        "state_id": "retained-0", "scene_id": "retained",
        "stratum": "general", "state_identity_digest": "a" * 64,
        "split_role": "fit",
    }
    slot = {"state_id": "replacement-0"}
    interval = {
        "lower_scene_id_exclusive": None,
        "upper_scene_id_exclusive": None,
        "vacant_ordinals": [0],
        "replacement_slots": [slot],
    }
    plan = {
        "retained_states": [retained],
        "rejected_identity_digests": ["b" * 64],
        "interval_groups": [interval],
    }
    scenes = [tmp_path / "scene-a", tmp_path / "scene-b"]
    monkeypatch.setattr(
        B, "_mixed_family_replacement_plan", lambda _family: plan)
    monkeypatch.setattr(
        B, "scene_pool", lambda _pool: ({family: scenes}, {}))
    monkeypatch.setattr(
        B, "_mixed_disposition_sets",
        lambda: ({"retained": retained}, {"b" * 64: {}}, []))
    chosen = {
        "state_id": "replacement-0", "scene_id": "scene-a",
        "stratum": "completion_enriched",
        "state_identity_digest": "c" * 64, "split_role": "fit",
    }
    row, pair = _archived_pair(
        kind="mixed", ordinal=0, scene_id="scene-a",
        request_extra={
            "anchor_interval": {
                "lower_scene_id_exclusive": None,
                "upper_scene_id_exclusive": None,
                "vacant_ordinals": [0],
            },
            "replacement_slot": slot,
            "accepted_scene_ids_before": [],
        },
        chosen_state=chosen,
        rejection_reasons={"synthetic": 1},
        provenance_extra={
            "interval_index": 0,
            "replacement_slot_state_id": "replacement-0",
            "selected": True,
        },
    )
    provenance = [row]
    interval_rows = [{
        "interval_index": 0,
        "lower_scene_id_exclusive": None,
        "upper_scene_id_exclusive": None,
        "vacant_ordinals": [0],
        "replacement_slot_state_ids": ["replacement-0"],
        "candidate_scene_ids": ["scene-a", "scene-b"],
        "scanned_scene_ids": ["scene-a"],
        "selected_scene_ids": ["scene-a"],
        "stopped_at_first_complete_prefix": True,
    }]
    states = [retained, chosen]
    states.sort(key=lambda state: (
        B.STRATA.index(state["stratum"]), state["state_id"]))
    predecessor = {
        "states": states,
        "replacement_slot_fills": [{
            "state_id": chosen["state_id"],
            "state_identity_digest": chosen["state_identity_digest"],
            "scene_id": chosen["scene_id"],
            "split_role": chosen["split_role"],
        }],
        "retained_predecessor_identity_digests": ["a" * 64],
        "rejected_predecessor_identity_digests": ["b" * 64],
        "scene_rejection_reasons": {"scene-a": {"synthetic": 1}},
        "mixed_replacement_scene_capture_provenance": provenance,
        "mixed_replacement_subprocess_transport": {
            "schema": B.MIXED_REPLACEMENT_TRANSPORT_SCHEMA,
            "one_scene_per_subprocess": True,
            "atomic_capture_write_before_native_cleanup": True,
            "return_code_ignored_only_after_valid_capture": True,
            "resume_scope":
                "MISSING_OR_INVALID_REPLACEMENT_SCENE_CAPTURES_ONLY",
            "interval_rows": interval_rows,
            "scene_capture_count": 1,
            "scene_capture_provenance_digest": B.canonical_digest(provenance),
            "candidate_outcomes_loaded": False,
        },
    }
    B._replay_projected_mixed_fixed_shard(
        predecessor=predecessor, family=family, projected_pairs=[pair])

    extra_row, extra_pair = _archived_pair(
        kind="mixed", ordinal=1, scene_id="scene-b",
        request_extra=pair["request"] | {"scene_ordinal": 1},
        chosen_state=None,
        provenance_extra={
            "interval_index": 0,
            "replacement_slot_state_id": "replacement-0",
            "selected": False,
        },
    )
    post_quota = copy.deepcopy(predecessor)
    post_quota["mixed_replacement_scene_capture_provenance"].append(extra_row)
    post_quota["mixed_replacement_subprocess_transport"][
        "scene_capture_count"] = 2
    post_quota["mixed_replacement_subprocess_transport"][
        "scene_capture_provenance_digest"] = B.canonical_digest(
            post_quota["mixed_replacement_scene_capture_provenance"])
    with pytest.raises(RuntimeError, match="post-quota"):
        B._replay_projected_mixed_fixed_shard(
            predecessor=post_quota, family=family,
            projected_pairs=[pair, extra_pair])


def test_fixed_performance_family_states_entrypoint_is_read_only(
        tmp_path, monkeypatch):
    family = str(B.PERFORMANCE_INTERRUPTION.FIXED_STATE_SHARDS[0]["family"])
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    output_root.mkdir()
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    monkeypatch.setattr(sys, "argv", [
        "build_go2_branch_corpus_v1_2.py", "--pool", "scorer_fit",
        "--stage", "states", "--family", family,
    ])
    payload = {
        "family": family, "states": [{}], "state_shard_digest": "d" * 64,
    }
    evidence = {
        "active_path": f"active/{family}.json",
        "envelope_schema": B.PERFORMANCE_INTERRUPTION.REISSUED_SHARD_SCHEMA,
    }
    monkeypatch.setattr(
        B, "_load_active_state_shard_evidence",
        lambda *_args, **_kwargs: (tmp_path / "physical.json", payload, evidence))
    monkeypatch.setattr(
        B, "resolve_states",
        lambda *_args, **_kwargs: pytest.fail("fixed family was re-resolved"))
    monkeypatch.setattr(
        B, "resolve_mixed_active_family",
        lambda *_args, **_kwargs: pytest.fail("fixed family was re-resolved"))
    assert B.main() == 0

    def invalid_wrapper(*_args, **_kwargs):
        raise RuntimeError("tampered wrapper")

    monkeypatch.setattr(B, "_load_active_state_shard_evidence", invalid_wrapper)
    with pytest.raises(RuntimeError, match="restored only"):
        B.main()

    monkeypatch.setattr(
        B, "stage_state_resolution_scene_worker",
        lambda *_args, **_kwargs: pytest.fail("fixed family worker executed"))
    monkeypatch.setattr(sys, "argv", [
        "build_go2_branch_corpus_v1_2.py", "--pool", "scorer_fit",
        "--stage", "states", "--family", family,
        "--state-resolution-scene-request-digest", "a" * 64,
    ])
    with pytest.raises(SystemExit, match="cannot execute scene workers"):
        B.main()


def _manifest(candidate_indices=(0, 1)):
    state = {
        "state_id": "state-000",
        "scene_id": "scene-000",
        "episode_cluster_id": "scene-000:episode:0",
        "episode_id": 0,
        "source_step": 200,
        "goal": {
            "landmark_id": "landmark_red",
            "bearing_body_rad": 0.0,
            "range_m": 1.0,
        },
        "family": "family-a",
        "split": "train",
        "stratum": "general",
        "split_role": "fit",
    }
    state["state_identity_digest"] = B._state_identity_digest(state)
    state["state_index"] = 0
    bindings = {
        "pool": "synthetic_test",
        "candidate_allocation_manifest_digest": "allocation-digest",
        "candidate_allocator_contract_digest": B.ALLOC.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            B.ALLOC.allocation_amendment_digest(),
        "candidate_allocation_post_identity_validation_digest":
            "post-identity-validation-digest",
        "pre_identity_allocation_validation_digest":
            "pre-identity-validation-digest",
        "clean_source_launch_receipt_digest": "launch-receipt-digest",
        "source_repository_commit": "a" * 40,
        "clean_source_binding_digest": "clean-source-binding-digest",
        "bound_implementations_digest": "source-bindings-digest",
        "scorer_contract_artifact_digest": "contract-artifact-digest",
        "mixed_precontract_disposition_receipt_digest": "d" * 64,
        "invalid_scorer_identity_exclusion_digest":
            B.INVALID_IDS.invalid_identity_exclusion_digest(),
        "state_selector_amendment_digest":
            B.STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest": "f" * 64,
        "preserved_state_revalidation_receipt_digest": "e" * 64,
        "candidate_bank_digest": B.V1.bank_digest(),
        "progress_contract_digest": B.progress_digest(),
        "safety_contract_digest": B.safety_digest(),
        "oracle_v1_2_digest": B.v12_oracle_digest(),
        "scorer_contract_v1_2_digest": B.scorer_contract_digest(),
        "selection_digest": B.selection_digest(),
        "boundary_digest": B.V1.BOUNDARY_DIGEST,
        "render_contract_digest": B.render_contract_digest(),
        "textured_v03_renderer_contract_digest":
            B.textured_v03_renderer_contract_digest(),
        "preprocess_contract_digest": B.preprocess_contract_digest(),
        "preprocessing_digest":
            B.TARGET_ENCODER["preprocessing_identity_sha256"],
        "target_encoder_digest": B.target_encoder_digest(),
        "target_encoder_checkpoint_sha256":
            B.TARGET_ENCODER["checkpoint_sha256"],
    }
    identity_bindings = {"pool": bindings["pool"], **bindings}
    identities = [
        B._branch_identity(state, candidate_index, identity_bindings)
        for candidate_index in candidate_indices
    ]
    state["candidate_indices"] = list(candidate_indices)
    state["branch_identities"] = identities
    manifest = {
        "schema": "go2_branch_corpus_v1_2_state_manifest",
        **bindings,
        "states": [state],
        "attempted_branch_count_registered": len(candidate_indices),
        "branch_identity_set_digest": B.canonical_digest(sorted(
            identity["branch_identity_digest"] for identity in identities
        )),
        "exclusion_binding": {
            "invalid_scorer_identity_attempt":
                B.INVALID_IDS.load_invalid_identity_index().binding(),
        },
    }
    manifest["state_manifest_digest"] = B.canonical_digest(manifest)
    return manifest


def _frame_record(out: Path, name: str, *, kind: str, index: int):
    path = out / "frames" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"bound-frame:{name}".encode("ascii")
    path.write_bytes(payload)
    return {
        kind: index,
        "path": str(path.relative_to(out)),
        "sha256": B.file_sha256(path),
        "byte_count": len(payload),
        "shape": [224, 224, 3],
        "dtype": "uint8",
    }


def _row(manifest, candidate_index: int, *, out: Path | None = None,
         valid: bool = False):
    state = manifest["states"][0]
    identity = B._identity_for(state, candidate_index)
    context = []
    horizons = []
    if valid:
        assert out is not None
        context = [
            _frame_record(out, f"ctx-{slot}.png", kind="slot", index=slot)
            for slot in range(B.CONTEXT_SLOTS)
        ]
        horizons = [
            _frame_record(out, f"c{candidate_index}-h{horizon}.png",
                          kind="horizon", index=horizon)
            for horizon in range(1, B.HORIZONS + 1)
        ]
    previous = [0.0, 0.0, 0.0]
    candidate = B.V1.CANDIDATE_BANK[candidate_index]
    requested, post_slew_plan, action_blocks = B.candidate_planning_trajectory(
        candidate, previous
    )
    row = {
        "schema": "go2_branch_corpus_v1_2_branch_row",
        "record_complete": True,
        "pool": manifest["pool"],
        "state_id": state["state_id"],
        "state_index": state["state_index"],
        "state_identity_digest": state["state_identity_digest"],
        "branch_identity_digest": identity["branch_identity_digest"],
        "split_role": state["split_role"],
        "stratum": state["stratum"],
        "scene_id": state["scene_id"],
        "family": state["family"],
        "split": state["split"],
        "episode_cluster_id": state["episode_cluster_id"],
        "episode_id": state["episode_id"],
        "source_step": state["source_step"],
        "candidate": identity["candidate"],
        "candidate_index": candidate_index,
        "primitives": identity["primitives"],
        "state_manifest_digest": manifest["state_manifest_digest"],
        "goal": state["goal"],
        "goal_binding_input": [0.0, 1.0, 1.0],
        "requested": requested,
        "realised_requested_prefix": requested,
        "post_slew": post_slew_plan,
        "candidate_post_slew_plan": post_slew_plan,
        "action_blocks": action_blocks,
        "action_context_blocks": [[0.0] * B.SLEW.ACTION_DIM
                                  for _ in range(B.CONTEXT_SLOTS)],
        "previous_applied_command": previous,
        "context_frames": context,
        "horizon_frames": horizons,
        "context_paths": [frame["path"] for frame in context],
        "horizon_paths": [frame["path"] for frame in horizons],
        "proprio": [[0.0] * 30 for _ in range(B.PROPRIO_HISTORY)],
        "control": [[0.0] * 2 for _ in range(B.PROPRIO_HISTORY)],
        "valid": valid,
        "invalid_reason": None if valid else "synthetic-invalid",
        "snapshot_digest": "synthetic-snapshot-digest",
        "progress": 0.1 if valid else None,
        "safety": 0.0 if valid else None,
        "completion": 0.0 if valid else None,
        "utility": 0.1 if valid else None,
        "storage_bytes": sum(frame["byte_count"] for frame in context + horizons),
        "wall_time_s": 1.25,
        **B._row_bindings(manifest),
    }
    row["branch_row_digest"] = B.canonical_digest(row)
    return identity, row


def _write_manifest(out: Path, manifest):
    B.atomic_json(out / "state_manifest.json", manifest)


def _encoder_module():
    pytest.importorskip("torch")
    from scripts import encode_go2_branch_corpus_v1_2 as encoder
    return encoder


def _patch_encoder_launch(monkeypatch, encoder, manifest):
    """Inject the exact synthetic manifest launch binding for unit tests only."""

    expected = {key: manifest[key] for key in encoder.LAUNCH_BINDING_KEYS}
    expected.update({
        "launch_state_selector_feasibility_receipt_digest":
            manifest["state_selector_feasibility_receipt_digest"],
        "mixed_precontract_disposition_receipt_digest":
            manifest["mixed_precontract_disposition_receipt_digest"],
    })
    monkeypatch.setattr(
        encoder, "_load_clean_source_launch_receipt", lambda: dict(expected))
    selector = {key: manifest[key] for key in encoder.SELECTOR_BINDING_KEYS}
    monkeypatch.setattr(
        encoder, "_load_selector_successor_receipts",
        lambda *_args, **_kwargs: dict(selector))


def test_encoder_reopens_complete_preserved_identity_two_phase_chain(
        tmp_path, monkeypatch):
    encoder = _encoder_module()
    monkeypatch.setattr(encoder, "ROOT", tmp_path)
    pool_root = tmp_path / "pools"
    monkeypatch.setattr(encoder, "OUT_ROOT", pool_root)
    feasibility_path = (
        tmp_path / encoder.STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH)
    disposition_path = (
        tmp_path
        / encoder.STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH)
    revalidation_path = (
        tmp_path / encoder.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH)
    allocation_path = pool_root / "scorer_fit/candidate_allocation_manifest.json"
    for path in (feasibility_path, disposition_path, revalidation_path,
                 allocation_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    feasibility_path.write_text(json.dumps({
        "state_selector_feasibility_receipt_digest": "f" * 64,
    }))
    revalidation_path.write_text(json.dumps({
        "preserved_state_revalidation_receipt_digest": "r" * 64,
    }))
    allocation_path.write_text("{}")
    monkeypatch.setattr(
        encoder.STATE_SELECTOR, "validate_authority_artifacts", lambda: None)
    monkeypatch.setattr(
        encoder.STATE_SELECTOR, "validate_frozen_reachability_feasibility_pass",
        lambda **_kwargs: {"state_selector_feasibility_receipt_digest": "f" * 64})
    monkeypatch.setattr(
        encoder.STATE_SELECTOR,
        "validate_preserved_state_mixed_precontract_disposition_receipt",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        encoder.STATE_SELECTOR, "validate_preserved_state_revalidation_receipt",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        encoder.STATE_SELECTOR, "state_selector_amendment_digest",
        lambda: "a" * 64)

    with pytest.raises(RuntimeError, match="successor artifacts are missing"):
        encoder._load_selector_successor_receipts(
            source_commit="c" * 40,
            selection_digest="s" * 64,
            active_states=[],
            expected_feasibility_receipt_digest="f" * 64,
            expected_mixed_precontract_disposition_receipt_digest="p" * 64,
        )

    disposition_path.write_text(json.dumps({
        "mixed_precontract_disposition_receipt_digest": "p" * 64,
    }))
    assert encoder._load_selector_successor_receipts(
        source_commit="c" * 40,
        selection_digest="s" * 64,
        active_states=[],
        expected_feasibility_receipt_digest="f" * 64,
        expected_mixed_precontract_disposition_receipt_digest="p" * 64,
    ) == {
        "state_selector_amendment_digest": "a" * 64,
        "state_selector_feasibility_receipt_digest": "f" * 64,
        "preserved_state_revalidation_receipt_digest": "r" * 64,
    }


def test_pre_identity_allocation_preflight_is_deterministic_and_idempotent(
        tmp_path, monkeypatch):
    interruption = _mock_interruption(monkeypatch)
    contract_path = tmp_path / "issued_scorer_contract.json"
    clean_source = {
        "schema": "synthetic_clean_source_binding",
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    monkeypatch.setattr(B, "clean_source_binding", lambda: clean_source)
    selector_preconditions = {
        "state_selector_feasibility_receipt_digest": "c" * 64,
        "mixed_precontract_disposition_receipt_digest": "d" * 64,
    }
    monkeypatch.setattr(
        B, "_load_state_selector_preconditions",
        lambda **_kwargs: dict(selector_preconditions))
    contract_artifact = {
        "schema": "synthetic_current_clean_source_contract",
        "complete": True,
        "scorer_contract_v1_2_digest": B.scorer_contract_digest(),
        "source_repository_clean": True,
        "clean_source_binding": clean_source,
        "clean_source_binding_digest": B.canonical_digest(clean_source),
        "preoutcome_fixed_reissue_validation_interruption_verified": True,
        "preoutcome_fixed_reissue_validation_interruption":
            interruption["transition"],
        "preoutcome_projection_fix_interruption": interruption["projection"],
        "preoutcome_small_search_performance_interruption_verified": True,
        "preoutcome_small_search_performance_interruption":
            interruption["performance"],
        **selector_preconditions,
    }
    contract_artifact["contract_artifact_digest"] = B.canonical_digest(
        contract_artifact
    )
    B.atomic_json(contract_path, contract_artifact)
    monkeypatch.setattr(B, "SCORER_CONTRACT_ARTIFACT_PATH", contract_path)
    out = tmp_path / "scorer_fit"
    out.mkdir()
    artifact = {
        "pre_identity_validation_digest": "e" * 64,
        "global": {"state_slot_count": 120, "candidate_slot_count": 720},
        "goal_type_validation": {
            "status": "NOT_EVALUABLE_BEFORE_STATE_IDENTITIES",
        },
    }
    artifact_path = out / B.PRE_IDENTITY_VALIDATION_NAME
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(
        B, "_load_pre_identity_allocation_validation", lambda: dict(artifact))
    monkeypatch.setattr(
        B.ALLOC, "build_pre_identity_structural_validation",
        lambda: pytest.fail("transition-certified preflight was rebuilt"))
    monkeypatch.setattr(
        B.ALLOC, "validate_pre_identity_structural_validation",
        lambda *_args, **_kwargs: pytest.fail("MILP validator was reached"))
    assert B.issue_pre_identity_allocation_validation(out) == 0
    first = artifact_path.read_bytes()
    assert artifact["global"]["state_slot_count"] == 120
    assert artifact["global"]["candidate_slot_count"] == 720
    assert artifact["goal_type_validation"]["status"] == (
        "NOT_EVALUABLE_BEFORE_STATE_IDENTITIES"
    )
    assert B.issue_pre_identity_allocation_validation(out) == 0
    assert artifact_path.read_bytes() == first


def test_issued_scorer_contract_uses_exact_managed_utility_root(
        tmp_path, monkeypatch):
    interruption = _mock_interruption(monkeypatch)
    utility_root = tmp_path / "repo/.generated/go2_utility_scorer_v1_2"
    target_root = tmp_path / "managed/go2_utility_scorer_v1_2"
    contract_path = target_root / "scorer_contract.json"
    contract_path.parent.mkdir(parents=True)
    utility_root.parent.mkdir(parents=True)
    utility_root.symlink_to(target_root, target_is_directory=True)
    source = {
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    payload = {
        "complete": True,
        "scorer_contract_v1_2_digest": B.scorer_contract_digest(),
        "source_repository_clean": True,
        "clean_source_binding": source,
        "clean_source_binding_digest": B.canonical_digest(source),
        "preoutcome_fixed_reissue_validation_interruption_verified": True,
        "preoutcome_fixed_reissue_validation_interruption":
            interruption["transition"],
        "preoutcome_projection_fix_interruption": interruption["projection"],
        "preoutcome_small_search_performance_interruption_verified": True,
        "preoutcome_small_search_performance_interruption":
            interruption["performance"],
    }
    payload["contract_artifact_digest"] = B.canonical_digest(payload)
    B.atomic_json(contract_path, payload)
    lexical_contract = utility_root / "scorer_contract.json"
    monkeypatch.setattr(B, "SCORER_CONTRACT_ARTIFACT_PATH", lexical_contract)
    monkeypatch.setattr(B, "clean_source_binding", lambda: source)
    assert B._issued_scorer_contract_path() == contract_path
    assert B._load_issued_scorer_contract() == payload

    contract_path.unlink()
    other = target_root / "other.json"
    B.atomic_json(other, payload)
    contract_path.symlink_to(other)
    with pytest.raises(RuntimeError, match="symlinked corpus paths"):
        B._load_issued_scorer_contract()


def test_issued_scorer_contract_canonical_path_survives_root_alias_swap(
        tmp_path, monkeypatch):
    interruption = _mock_interruption(monkeypatch)
    lexical_root = tmp_path / "repo/.generated/go2_utility_scorer_v1_2"
    first_root = tmp_path / "first/go2_utility_scorer_v1_2"
    second_root = tmp_path / "second/go2_utility_scorer_v1_2"
    first_path = first_root / "scorer_contract.json"
    second_path = second_root / "scorer_contract.json"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    source = {
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }

    def artifact(marker):
        payload = {
            "marker": marker,
            "complete": True,
            "scorer_contract_v1_2_digest": B.scorer_contract_digest(),
            "source_repository_clean": True,
            "clean_source_binding": source,
            "clean_source_binding_digest": B.canonical_digest(source),
                "preoutcome_fixed_reissue_validation_interruption_verified":
                    True,
                "preoutcome_fixed_reissue_validation_interruption":
                    interruption["transition"],
                "preoutcome_projection_fix_interruption":
                    interruption["projection"],
                "preoutcome_small_search_performance_interruption_verified":
                    True,
                "preoutcome_small_search_performance_interruption":
                    interruption["performance"],
        }
        payload["contract_artifact_digest"] = B.canonical_digest(payload)
        return payload

    first = artifact("first")
    second = artifact("second")
    B.atomic_json(first_path, first)
    B.atomic_json(second_path, second)
    lexical_root.symlink_to(first_root, target_is_directory=True)
    monkeypatch.setattr(
        B, "SCORER_CONTRACT_ARTIFACT_PATH",
        lexical_root / "scorer_contract.json")
    monkeypatch.setattr(B, "clean_source_binding", lambda: source)

    pinned = B._issued_scorer_contract_path()
    lexical_root.unlink()
    lexical_root.symlink_to(second_root, target_is_directory=True)
    assert pinned == first_path
    assert B._load_issued_scorer_contract_at_path(pinned) == first
    assert B._load_issued_scorer_contract() == second


def test_parallel_artifacts_keep_lexical_identity_across_managed_root_alias(
        tmp_path, monkeypatch):
    """Resolved executor paths must never be reinterpreted as lexical paths."""

    lexical_root = tmp_path / "repo/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "managed/go2_branch_corpus_v1_2"
    target_root.mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    monkeypatch.setattr(B, "OUT_ROOT", lexical_root)
    out = lexical_root / "scorer_fit"

    raw_benchmark = out / B.PARALLEL_SMALL_BENCHMARK_NAME
    benchmark = {"schema": "synthetic-parallel-benchmark", "complete": True}
    B._write_or_require_exact_json(
        raw_benchmark, benchmark, label="synthetic parallel benchmark")
    assert json.loads(
        (target_root / "scorer_fit" /
         B.PARALLEL_SMALL_BENCHMARK_NAME).read_text()) == benchmark

    raw_checkpoint = out / B.PARALLEL_SMALL_CHECKPOINT_ROOT
    pinned_checkpoint = B._parallel_search_checkpoint_root(out)
    raw_rank = raw_checkpoint / "ranks/000000000000.json"
    pinned_rank = pinned_checkpoint / "ranks/000000000000.json"
    receipt = {"schema": "synthetic-rank"}
    receipt["rank_receipt_digest"] = B.PARALLEL_SEARCH.canonical_digest(
        receipt)
    B.atomic_json(pinned_rank, receipt)
    binding = B._parallel_certificate_binding(
        raw_rank, receipt, "rank_receipt_digest", pinned_path=pinned_rank)
    assert binding["path"] == str(raw_rank)
    assert binding["raw_sha256"] == B.file_sha256(pinned_rank)


def test_parallel_artifact_install_is_exclusive_under_concurrent_issuers(
        tmp_path, monkeypatch):
    output_root = tmp_path / ".generated/go2_branch_corpus_v1_2"
    output_root.mkdir(parents=True)
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    path = output_root / "scorer_fit/concurrent-receipt.json"
    path.parent.mkdir()
    payloads = [{"issuer": "first"}, {"issuer": "second"}]

    def issue(payload):
        try:
            B._write_or_require_exact_json(
                path, payload, label="synthetic concurrent receipt")
            return "installed"
        except RuntimeError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(issue, payloads))
    assert sorted(outcomes) == ["installed", "rejected"]
    assert json.loads(path.read_text()) in payloads


def test_parallel_exhaustion_must_certify_before_failure_receipt(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    out = output_root / "scorer_fit"
    out.mkdir(parents=True)
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    terminal_path = out / B.PARALLEL_SMALL_TERMINAL_RESULT_NAME
    B.atomic_json(terminal_path, {"status": "EXHAUSTED"})
    monkeypatch.setattr(B, "_parallel_small_search_inputs", lambda _out: {})
    monkeypatch.setattr(
        B, "_load_parallel_plan_and_benchmark",
        lambda _out, _inputs: ({"total_rank_count": 2}, {}, {}))
    monkeypatch.setattr(
        B, "_parallel_search_callbacks",
        lambda _inputs: (lambda *_args: {}, lambda *_args: False,
                         lambda *_args: True))
    monkeypatch.setattr(
        B, "_parallel_search_checkpoint_root", lambda _out: out / "checkpoints")

    def reject_partial(**_kwargs):
        raise B.PARALLEL_SEARCH.ParallelSearchError(
            "terminal scientific EXHAUSTED surface changed")

    monkeypatch.setattr(
        B.PARALLEL_SEARCH, "validate_exhausted_search_result", reject_partial)
    monkeypatch.setattr(
        B, "_parallel_failure_receipt",
        lambda **_kwargs: pytest.fail("uncertified failure receipt was built"))
    with pytest.raises(
            B.PARALLEL_SEARCH.ParallelSearchError,
            match="EXHAUSTED surface changed"):
        B.stage_parallel_small_completion_search()


def _synthetic_parallel_failure_plan_and_prepare():
    states = [{
        "state_id": f"state-{index:03d}",
        "state_identity_digest": f"{index + 1:064x}",
        "family": f"synthetic-family-{index // 15}",
        "stratum": B.ALLOC.STRATA[(index % 15) // 5],
        "split_role": "calibration" if index % 5 == 0 else "fit",
        "goal_type": "landmark",
    } for index in range(B.PARALLEL_SEARCH.PREFIX_STATE_COUNT)]
    plan = B.PARALLEL_SEARCH.build_search_plan(
        candidate_scene_ids=[f"scene-{index:03d}" for index in range(5)],
        combination_size=5, worker_count=1,
        source_repository_commit="a" * 40,
        clean_source_launch_receipt_digest="b" * 64,
        state_selector_amendment_digest="c" * 64,
        candidate_allocation_amendment_digest="d" * 64,
        fixed_state_projection_digest="e" * 64,
        resolver_cursor_scene_id="scene-before-pool",
        solver_identity={"name": "synthetic", "version": "1"},
        solver_options={"threads": 1, "mip_rel_gap": 0.0},
    )

    def prepare(_rank, _combination):
        return {
            "states": copy.deepcopy(states),
            "source_identity_manifest_digest": "f" * 64,
            "mask_context": {},
        }

    return plan, prepare, states


def _synthetic_parallel_wave(
        plan, projection, *, statuses, state_index=0, prefix=()):
    results = [{
        "rotation": rotation,
        "status": status,
        "message": f"synthetic {status.lower()}",
        "elapsed_s": 0.0,
        "solver_call_count": 1,
        "worker_pid": 1,
        "thread_environment": dict(B.PARALLEL_SEARCH.THREAD_ENVIRONMENT),
    } for rotation, status in enumerate(statuses)]
    wave_status, selected = B.PARALLEL_SEARCH._lexicographic_wave_decision(
        statuses, state_index=state_index)
    payload = {
        "schema": B.PARALLEL_SEARCH.WAVE_RECEIPT_SCHEMA,
        "search_plan_digest": plan["search_plan_digest"],
        "rank": 0,
        "state_index": state_index,
        "projection_digest": projection,
        "prefix_rotations_before": list(prefix),
        "rotation_results": results,
        "wave_status": wave_status,
        "selected_rotation": selected,
        "solver_call_count": B.PARALLEL_SEARCH.ROTATION_COUNT,
        "wave_elapsed_s": 0.0,
        "candidate_outcomes_consumed": False,
    }
    payload["wave_receipt_digest"] = \
        B.PARALLEL_SEARCH.canonical_digest(payload)
    return payload


def test_parallel_failure_inventory_rejects_noncanonical_checkpoint_file(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    out = output_root / "scorer_fit"
    rank_dir = (out / B.PARALLEL_SMALL_CHECKPOINT_ROOT /
                "waves/rank-000000000000")
    rank_dir.mkdir(parents=True)
    (rank_dir / "self-authored-fatal.json").write_text(
        json.dumps({"status": "FATAL"}))
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    plan, prepare, _states = _synthetic_parallel_failure_plan_and_prepare()
    with pytest.raises(RuntimeError, match="filename is noncanonical"):
        B._parallel_failure_evidence_inventory(
            out=out, plan=plan, prepare_rank=prepare)


def test_parallel_failure_inventory_requires_waves_for_every_rank_receipt(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    out = output_root / "scorer_fit"
    checkpoint = out / B.PARALLEL_SMALL_CHECKPOINT_ROOT
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    plan, prepare, states = _synthetic_parallel_failure_plan_and_prepare()
    projection = B.PARALLEL_SEARCH.canonical_digest(
        B.PARALLEL_SEARCH.project_allocator_identity_states(states))
    rank = B.PARALLEL_SEARCH._rank_payload(
        plan=plan, rank=0, projection_digest=projection,
        source_digest="f" * 64, classification="MASK_FAIL",
        rotations=[0] * B.PARALLEL_SEARCH.PREFIX_STATE_COUNT,
        allocation={"allocation_manifest_digest": "1" * 64},
        assignment_digest="2" * 64)
    B.PARALLEL_SEARCH.write_rank_receipt(
        checkpoint / "ranks/000000000000.json", rank,
        search_plan=plan)
    with pytest.raises(RuntimeError, match="lacks complete wave evidence"):
        B._parallel_failure_evidence_inventory(
            out=out, plan=plan, prepare_rank=prepare)


def test_parallel_failure_inventory_rejects_fatal_wave_for_infeasible_rank(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    out = output_root / "scorer_fit"
    checkpoint = out / B.PARALLEL_SMALL_CHECKPOINT_ROOT
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    plan, prepare, states = _synthetic_parallel_failure_plan_and_prepare()
    projection = B.PARALLEL_SEARCH.canonical_digest(
        B.PARALLEL_SEARCH.project_allocator_identity_states(states))
    rank = B.PARALLEL_SEARCH._rank_payload(
        plan=plan, rank=0, projection_digest=projection,
        source_digest="f" * 64, classification="ALLOCATOR_INFEASIBLE",
        rotations=[], allocation=None, assignment_digest=None)
    B.PARALLEL_SEARCH.write_rank_receipt(
        checkpoint / "ranks/000000000000.json", rank,
        search_plan=plan)
    statuses = ["FATAL"] + [
        "INFEASIBLE"] * (B.PARALLEL_SEARCH.ROTATION_COUNT - 1)
    B.atomic_json(
        checkpoint / "waves/rank-000000000000/prefix-000.json",
        _synthetic_parallel_wave(plan, projection, statuses=statuses))
    with pytest.raises(RuntimeError, match="infeasible rank wave evidence"):
        B._parallel_failure_evidence_inventory(
            out=out, plan=plan, prepare_rank=prepare)


def test_fatal_receipt_binds_nonfatal_speculative_checkpoint_bytes(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    out = output_root / "scorer_fit"
    checkpoint = out / B.PARALLEL_SMALL_CHECKPOINT_ROOT
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    plan, prepare, states = _synthetic_parallel_failure_plan_and_prepare()
    projection = B.PARALLEL_SEARCH.canonical_digest(
        B.PARALLEL_SEARCH.project_allocator_identity_states(states))
    statuses = ["FEASIBLE"] + [
        "INFEASIBLE"] * (B.PARALLEL_SEARCH.ROTATION_COUNT - 1)
    wave_path = checkpoint / "waves/rank-000000000000/prefix-000.json"
    B.atomic_json(
        wave_path,
        _synthetic_parallel_wave(plan, projection, statuses=statuses))
    monkeypatch.setattr(
        B, "_artifact_binding",
        lambda path, **_kwargs: {"path": str(path), "digest": "a" * 64})
    kwargs = {
        "out": out,
        "inputs": {"prefix": {"receipt_binding": {"digest": "b" * 64}}},
        "plan": plan,
        "benchmark": {},
        "status": "FATAL",
        "reason": "synthetic fatal outside rank evaluation",
        "terminal": None,
        "prepare_rank": prepare,
    }
    first = B._parallel_failure_receipt(**kwargs)
    assert [row["kind"] for row in first[
        "checkpoint_evidence_inventory"]] == ["prefix_wave_receipt"]
    wave_path.unlink()
    second = B._parallel_failure_receipt(**kwargs)
    assert second["checkpoint_evidence_inventory"] == []
    assert (first["parallel_small_completion_failure_receipt_digest"]
            != second["parallel_small_completion_failure_receipt_digest"])


def test_complete_ordinary_nonpass_frontier_cannot_be_fatal(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    out = output_root / "scorer_fit"
    out.mkdir(parents=True)
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    plan, prepare, _states = _synthetic_parallel_failure_plan_and_prepare()
    monkeypatch.setattr(
        B, "_parallel_failure_evidence_inventory",
        lambda **_kwargs: ({0: {"classification": "MASK_FAIL"}}, [], []))
    with pytest.raises(RuntimeError, match="requires EXHAUSTED evidence"):
        B._parallel_failure_receipt(
            out=out,
            inputs={"prefix": {"receipt_binding": {}}},
            plan=plan, benchmark={}, status="FATAL",
            reason="synthetic contradictory fatal", terminal=None,
            prepare_rank=prepare)


def test_parallel_terminal_failure_hard_stops_before_search(monkeypatch):
    monkeypatch.setattr(B, "_parallel_small_search_inputs", lambda _out: {})
    monkeypatch.setattr(
        B, "_load_parallel_plan_and_benchmark",
        lambda _out, _inputs: ({}, {}, {}))
    monkeypatch.setattr(
        B, "_parallel_search_callbacks",
        lambda _inputs: (lambda *_args: {}, lambda *_args: False,
                         lambda *_args: True))
    monkeypatch.setattr(
        B, "_load_existing_parallel_terminal_failure",
        lambda **_kwargs: {"status": "FATAL"})
    monkeypatch.setattr(
        B.PARALLEL_SEARCH, "run_scientific_parallel_search",
        lambda **_kwargs: pytest.fail("terminal FATAL retried the search"))
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        B.stage_parallel_small_completion_search()


def test_parallel_failure_binds_terminal_result_presence_or_absence(
        tmp_path, monkeypatch):
    output_root = tmp_path / "go2_branch_corpus_v1_2"
    out = output_root / "scorer_fit"
    out.mkdir(parents=True)
    monkeypatch.setattr(B, "OUT_ROOT", output_root)
    absent = B._parallel_terminal_result_disposition(
        out=out, status="FATAL", terminal=None)
    assert absent == {
        "status": "ABSENT",
        "path": str(out / B.PARALLEL_SMALL_TERMINAL_RESULT_NAME),
    }

    terminal = {
        "schema": B.PARALLEL_SEARCH.SEARCH_RESULT_SCHEMA,
        "status": "EXHAUSTED",
        "combination_attempt_count": 1,
        "allocator_infeasible_combination_count": 1,
        "search_plan_digest": "a" * 64,
        "candidate_outcomes_consumed": False,
    }
    path = out / B.PARALLEL_SMALL_TERMINAL_RESULT_NAME
    B.atomic_json(path, terminal)
    with pytest.raises(RuntimeError, match="contradicts"):
        B._parallel_terminal_result_disposition(
            out=out, status="FATAL", terminal=None)
    present = B._parallel_terminal_result_disposition(
        out=out, status="EXHAUSTED", terminal=terminal)
    assert present["status"] == "PRESENT_EXHAUSTED"
    assert present["terminal_result_digest"] == \
        B.PARALLEL_SEARCH.canonical_digest(terminal)
    with pytest.raises(RuntimeError, match="bytes changed"):
        B._parallel_terminal_result_disposition(
            out=out, status="EXHAUSTED", terminal={**terminal, "tamper": True})


def test_ordered_manifest_projection_requires_exact_shard_state_union():
    first = {
        "state_id": "state-a", "state_identity_digest": "a" * 64,
        "scene_id": "scene-a", "family": "family-a", "stratum": "general",
        "split_role": "fit",
        "goal_type": "landmark", "scientific_marker": {"exact": True},
    }
    second = {
        "state_id": "state-b", "state_identity_digest": "b" * 64,
        "scene_id": "scene-b", "family": "family-b",
        "stratum": "evaluation", "split_role": "fit",
        "goal_type": "landmark", "scientific_marker": {"exact": True},
    }
    expected = [first, second]
    merged = [{
        **state,
        "state_index": index,
        "candidate_indices": [0, 1, 2, 3, 4, 5],
        "candidate_rotation_index": index,
        "branch_identities": [{"branch_identity_digest": f"{index + 3:064x}"}],
    } for index, state in enumerate((first, second))]
    assert B._ordered_manifest_preallocation_state_projection(merged) == expected

    missing = merged[:1]
    extra = [*merged, {
        **first, "state_id": "state-c", "scene_id": "scene-c",
        "state_identity_digest": "c" * 64, "family": "family-z",
        "state_index": 2,
        "candidate_indices": [0, 1, 2, 3, 4, 5],
        "candidate_rotation_index": 2,
        "branch_identities": [{"branch_identity_digest": "f" * 64}],
    }]
    changed = copy.deepcopy(merged)
    changed[0]["scientific_marker"]["exact"] = False
    for nonexact in (missing, extra, changed):
        assert B._ordered_manifest_preallocation_state_projection(
            nonexact) != expected


def test_final_eval_manifest_state_forbids_candidate_rotation_index():
    state = {
        "state_id": "final-state",
        "state_index": 0,
        "candidate_indices": list(range(len(B.V1.CANDIDATE_BANK))),
        "branch_identities": [],
    }
    assert B._validate_manifest_pool_specific_state_fields(
        [state], pool="final_eval") is None
    with_rotation = {**state, "candidate_rotation_index": 0}
    with pytest.raises(RuntimeError):
        B._validate_manifest_pool_specific_state_fields(
            [with_rotation], pool="final_eval")


def test_manifest_common_bindings_match_every_shard_exactly():
    common = {
        key: f"synthetic-common-{index}"
        for index, key in enumerate(B.STATE_SHARD_COMMON_KEYS)
    }
    shards = [{**common, "family": f"family-{index}"}
              for index in range(2)]
    assert B._validate_manifest_common_bindings_against_shards(
        dict(common), shards) is None

    changed = copy.deepcopy(shards)
    changed[1]["scorer_fit_allocation_design_digest"] = "changed"
    with pytest.raises(RuntimeError):
        B._validate_manifest_common_bindings_against_shards(
            dict(common), changed)

    missing_manifest = dict(common)
    missing_manifest.pop("scorer_fit_allocation_design_digest")
    missing_shards = copy.deepcopy(shards)
    for shard in missing_shards:
        shard.pop("scorer_fit_allocation_design_digest")
    with pytest.raises(RuntimeError):
        B._validate_manifest_common_bindings_against_shards(
            missing_manifest, missing_shards)


def test_interrupted_identity_bindings_require_exact_selection_and_scorer():
    lineage = {
        "selection_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SELECTION_DIGEST,
        "scorer_contract_v1_2_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SCORER_CONTRACT_DIGEST,
    }
    assert B._validate_interrupted_state_identity_bindings(lineage) == lineage
    invalid = (
        {},
        {"selection_digest": lineage["selection_digest"]},
        {"scorer_contract_v1_2_digest":
            lineage["scorer_contract_v1_2_digest"]},
        {**lineage, "selection_digest": "0" * 64},
        {**lineage, "scorer_contract_v1_2_digest": "0" * 64},
    )
    for bindings in invalid:
        with pytest.raises(RuntimeError):
            B._validate_interrupted_state_identity_bindings(bindings)


def test_performance_lineage_requires_exact_registered_state_membership(
        monkeypatch):
    monkeypatch.setattr(B, "_preserved_states_by_digest", lambda: {})
    state = {
        "state_id": "historical-state",
        "family": "medium_enclosed_maze",
        "stratum": "general",
        "split_role": "fit",
        "goal_type": "landmark_red",
        "scientific_payload": {"exact": True},
    }
    lineage = {
        "selection_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SELECTION_DIGEST,
        "scorer_contract_v1_2_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SCORER_CONTRACT_DIGEST,
    }
    state["state_identity_digest"] = \
        B._state_identity_digest_for_bindings(state, lineage)
    assert B._state_identity_matches_active_or_preserved(state) is False
    assert B._state_identity_matches_active_or_preserved(
        state, exact_performance_lineage_states={
            state["state_identity_digest"]: copy.deepcopy(state),
        }) is True
    changed = copy.deepcopy(state)
    changed["scientific_payload"]["exact"] = False
    assert B._state_identity_matches_active_or_preserved(
        changed, exact_performance_lineage_states={
            state["state_identity_digest"]: copy.deepcopy(state),
        }) is False

    same_family_impostor = copy.deepcopy(state)
    same_family_impostor["state_id"] = "same-family-impostor"
    same_family_impostor["scientific_payload"] = {"exact": "invented"}
    same_family_impostor["state_identity_digest"] = \
        B._state_identity_digest_for_bindings(same_family_impostor, lineage)
    assert B._state_identity_matches_active_or_preserved(
        same_family_impostor, exact_performance_lineage_states={
            state["state_identity_digest"]: copy.deepcopy(state),
        }) is False


def test_parallel_small_lineage_requires_ten_prefix_and_five_current():
    def states(prefix_count, current_count):
        prefix = [{
            "state_id": f"historical-{index:02d}",
            "state_identity_digest": f"{index + 1:064x}",
            "scene_id": f"historical-scene-{index:02d}",
            "family": B.REACHABILITY_REDRIVE_FAMILY,
            "stratum": "general" if index < 5 else "safety_enriched",
            "split_role": "fit",
        } for index in range(prefix_count)]
        current = [{
            "state_id": f"current-{index:02d}",
            "scene_id": f"current-scene-{index:02d}",
            "family": B.REACHABILITY_REDRIVE_FAMILY,
            "stratum": "completion_enriched",
            "split_role": "fit",
        } for index in range(current_count)]
        for state in current:
            state["state_identity_digest"] = B._state_identity_digest(state)
        return [*prefix, *current], prefix

    exact, exact_prefix = states(10, 5)
    assert B._validate_parallel_small_state_identity_lineage(
        exact, exact_prefix) is None

    for prefix_count, current_count in ((11, 4), (9, 6)):
        nonexact, expected_prefix = states(prefix_count, current_count)
        with pytest.raises(RuntimeError):
            B._validate_parallel_small_state_identity_lineage(
                nonexact, expected_prefix)

    historical_current = copy.deepcopy(exact)
    lineage = {
        "selection_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SELECTION_DIGEST,
        "scorer_contract_v1_2_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SCORER_CONTRACT_DIGEST,
    }
    historical_current[-1]["state_identity_digest"] = \
        B._state_identity_digest_for_bindings(historical_current[-1], lineage)
    with pytest.raises(RuntimeError):
        B._validate_parallel_small_state_identity_lineage(
            historical_current, exact_prefix)


def test_v2_predecessor_envelope_api_is_exact_and_mask_free(monkeypatch):
    envelope = {
        "schema": B.PARALLEL_V2_PREDECESSOR_BINDINGS_SCHEMA,
        "provisional_search_plan_digest": "1" * 64,
        "benchmark_source_binding_digest": "2" * 64,
        "rank_zero_source_identity_manifest_digest": "3" * 64,
        "rank_zero_state_projection_digest": "4" * 64,
        "candidate_pool_scene_ids_digest": "5" * 64,
        "fixed_state_projection_digest": "6" * 64,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }
    material = {
        "inputs": {
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": False,
        },
        "benchmark": {"benchmark_receipt_digest": "7" * 64},
        "provisional_plan": {"search_plan_digest": "1" * 64},
        "benchmark_source_binding_digest": "2" * 64,
        "predecessor_scientific_input_bindings": envelope,
        "v1_failure_disposition":
            B.PARALLEL_V1_IMMUTABLE_FAILURE_DISPOSITION,
    }
    calls = []
    monkeypatch.setattr(
        B, "_v2_load_benchmark_material",
        lambda out: calls.append(out) or copy.deepcopy(material))
    monkeypatch.setattr(
        B, "_phase1_completion_rotation_vectors",
        lambda: pytest.fail("pre-gate preserved masks were opened"))

    assert B.build_v2_predecessor_scientific_input_bindings() == envelope
    loaded = B.load_v2_parallel_small_benchmark_inputs(
        predecessor_scientific_input_bindings=envelope)
    assert "preserved_vectors" not in loaded
    assert loaded["predecessor_v1_benchmark_source_binding_digest"] == "2" * 64
    assert loaded["v1_failure_disposition"] == \
        B.PARALLEL_V1_IMMUTABLE_FAILURE_DISPOSITION
    assert calls == [B.OUT_ROOT / "scorer_fit", B.OUT_ROOT / "scorer_fit"]

    changed = copy.deepcopy(envelope)
    changed["fixed_state_projection_digest"] = "9" * 64
    with pytest.raises(RuntimeError, match="exact d9d reconstruction"):
        B.load_v2_parallel_small_benchmark_inputs(
            predecessor_scientific_input_bindings=changed)


def test_v2_rank_identity_uses_predecessor_bindings_not_current_source(
        monkeypatch):
    bindings = {
        "selection_digest": "a" * 64,
        "scorer_contract_v1_2_digest": "b" * 64,
    }
    candidates = [{
        "state_id": "deferred",
        "scene_id": f"scene-{index:02d}",
        "family": B.REACHABILITY_REDRIVE_FAMILY,
        "stratum": "completion_enriched",
        "split_role": "deferred",
        "goal_type": "goal",
    } for index in range(5)]
    monkeypatch.setattr(
        B, "_state_identity_digest",
        lambda _state: pytest.fail("current-source identity digest was used"))
    selected = B._parallel_selected_completion_states(
        candidates, range(5), identity_bindings=bindings)
    assert [state["state_id"] for state in selected] == [
        f"scorer_fit-{B.REACHABILITY_REDRIVE_FAMILY}-"
        f"completion_enriched-{index:02d}" for index in range(5)]
    assert [state["state_identity_digest"] for state in selected] == [
        B._state_identity_digest_for_bindings(state, bindings)
        for state in selected]


def test_v2_plan_uses_predecessor_launch_and_only_pass_digest_changes(
        monkeypatch):
    fixed = [{
        "state_id": "fixed-0", "state_identity_digest": "a" * 64,
        "family": "family", "stratum": "general", "split_role": "fit",
        "goal_type": "goal",
    }]
    scenes = [f"scene-{index}" for index in range(5)]
    fixed_digest = B.canonical_digest(B._allocation_projection(fixed))
    candidate_digest = B.PARALLEL_SEARCH.canonical_digest(scenes)
    envelope = {
        "schema": B.PARALLEL_V2_PREDECESSOR_BINDINGS_SCHEMA,
        "provisional_search_plan_digest": "1" * 64,
        "benchmark_source_binding_digest": "2" * 64,
        "rank_zero_source_identity_manifest_digest": "3" * 64,
        "rank_zero_state_projection_digest": "4" * 64,
        "candidate_pool_scene_ids_digest": candidate_digest,
        "fixed_state_projection_digest": fixed_digest,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }
    launch = {
        "source_repository_commit":
            B.PARALLEL_V2_PREDECESSOR_SOURCE_COMMIT,
        "clean_source_launch_receipt_digest": "5" * 64,
        "state_selector_feasibility_receipt_digest": "6" * 64,
        "state_selector_amendment_digest":
            B.STATE_SELECTOR.state_selector_amendment_digest(),
        "candidate_allocation_amendment_digest":
            B.ALLOC.allocation_amendment_digest(),
    }
    inputs = {
        "fixed_states": fixed,
        "candidate_scene_ids": scenes,
        "resolver_cursor_scene_id": "cursor",
        "prefix": {
            "receipt_binding": {"receipt_digest": "7" * 64},
            "performance_receipt_binding": {"receipt_digest": "8" * 64},
        },
        "fixed_shard_evidence": [{"exact": True}],
        "predecessor_scientific_input_bindings": envelope,
        "predecessor_launch": launch,
        "candidate_outcomes_consumed": False,
    }
    monkeypatch.setattr(
        B, "_v2_load_d9d_authorities",
        lambda _out: {"clean_launch": dict(launch)})
    envelope_digest = B.PARALLEL_SEARCH.canonical_digest(envelope)
    provisional = B.build_v2_parallel_search_plan(
        inputs, source_repository_commit="c" * 40,
        benchmark_v2_contract_digest="d" * 64,
        predecessor_scientific_input_bindings_digest=envelope_digest,
        measured_benchmark_receipt_digest=None)
    final = B.build_v2_parallel_search_plan(
        inputs, source_repository_commit="c" * 40,
        benchmark_v2_contract_digest="d" * 64,
        predecessor_scientific_input_bindings_digest=envelope_digest,
        measured_benchmark_receipt_digest="e" * 64)
    assert provisional["source_repository_commit"] == "c" * 40
    assert provisional["clean_source_launch_receipt_digest"] == "5" * 64
    assert provisional["bindings"]["predecessor_source_repository_commit"] \
        == B.PARALLEL_V2_PREDECESSOR_SOURCE_COMMIT
    left = copy.deepcopy(provisional)
    right = copy.deepcopy(final)
    left.pop("search_plan_digest")
    right.pop("search_plan_digest")
    left["measured_benchmark_receipt_digest"] = "e" * 64
    assert left == right


def test_v2_small_prefix_exact_reducer_needs_no_source_validator(monkeypatch):
    pairs = []
    found = {
        "general": 0, "safety_enriched": 0, "completion_enriched": 0}
    selected = []
    trace = []
    for ordinal in range(12):
        requested = [name for name in B.STRATA if found[name] < {
            "general": 5, "safety_enriched": 5,
            "completion_enriched": 0}[name]]
        chosen = None
        if ordinal >= 2:
            stratum = "general" if ordinal < 7 else "safety_enriched"
            chosen = {
                "state_id": f"state-{ordinal:02d}",
                "state_identity_digest": f"{ordinal + 1:064x}",
                "scene_id": f"scene-{ordinal:02d}",
                "stratum": stratum,
                "split_role": "fit",
            }
        request = {
            "scene_ordinal": ordinal,
            "scene": {"scene_id": f"scene-{ordinal:02d}"},
            "required_counts": {
                "general": 5, "safety_enriched": 5,
                "completion_enriched": 0},
            "found_before_scene": dict(found),
            "requested_strata_in_priority_order": requested,
            "state_resolution_scene_request_digest": f"{ordinal + 101:064x}",
            "candidate_outcomes_loaded": False,
        }
        capture = {
            "request": request,
            "state_resolution_scene_request_digest": request[
                "state_resolution_scene_request_digest"],
            "state_resolution_scene_capture_digest": f"{ordinal + 201:064x}",
            "scene_id": f"scene-{ordinal:02d}",
            "chosen_state": chosen,
            "worker_failure": None,
            "candidate_outcomes_loaded": False,
        }
        chosen_stratum = None
        chosen_digest = None
        if chosen is not None:
            chosen_stratum = chosen["stratum"]
            chosen_digest = chosen["state_identity_digest"]
            found[chosen_stratum] += 1
            selected.append(chosen)
        trace.append({
            "scene_ordinal": ordinal,
            "scene_id": f"scene-{ordinal:02d}",
            "found_before_scene": request["found_before_scene"],
            "requested_strata_in_priority_order": requested,
            "chosen_stratum": chosen_stratum,
            "chosen_state_identity_digest": chosen_digest,
        })
        pairs.append({
            "scene_ordinal": ordinal,
            "scene_id": f"scene-{ordinal:02d}",
            "request": request,
            "capture": capture,
        })
    projection = [{key: state[key] for key in (
        "state_id", "state_identity_digest", "scene_id", "stratum",
        "split_role")}
        for state in sorted(selected, key=lambda state: state["state_id"])]
    receipt = {
        "selected_state_projection_digest": B.canonical_digest(projection),
        "reducer_trace_digest": B.canonical_digest(trace),
        "resolver_cursor_scene_id": "scene-11",
    }
    monkeypatch.setattr(
        B, "_validate_state_resolution_scene_request",
        lambda *_args, **_kwargs: pytest.fail("current request validator used"))
    monkeypatch.setattr(
        B, "_validate_state_resolution_scene_capture",
        lambda *_args, **_kwargs: pytest.fail("current capture validator used"))
    replay = B._v2_reduce_small_prefix(receipt=receipt, pairs=pairs)
    assert len(replay["states"]) == 10
    assert replay["resolver_cursor_scene_id"] == "scene-11"


def test_v2_mask_context_is_unreachable_before_validated_pass(monkeypatch):
    monkeypatch.setattr(
        B.STATE_SELECTOR, "validate_frozen_preserved_precontract_failure",
        lambda **_kwargs: pytest.fail("preserved masks opened before PASS"))
    with pytest.raises(Exception):
        B.attach_v2_parallel_search_mask_context(
            {
                "candidate_outcomes_consumed": False,
                "scientific_masks_accessed": False,
            },
            v2_pass_receipt={"passes": False},
        )


def test_v2_mask_context_attaches_only_after_pass_validator(
        monkeypatch):
    from lewm.oracle import go2_parallel_small_completion_search_v2 as SEARCH_V2

    envelope = {
        "schema": B.PARALLEL_V2_PREDECESSOR_BINDINGS_SCHEMA,
        "provisional_search_plan_digest": "1" * 64,
        "benchmark_source_binding_digest": "2" * 64,
        "rank_zero_source_identity_manifest_digest": "3" * 64,
        "rank_zero_state_projection_digest": "4" * 64,
        "candidate_pool_scene_ids_digest": "5" * 64,
        "fixed_state_projection_digest": "6" * 64,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }
    pass_receipt = {
        "benchmark_v2_contract_digest": "7" * 64,
        "source_binding_digest": "8" * 64,
        "passes": True,
        "median_gate_passes": True,
        "maximum_gate_passes": True,
        "worker_restart_count": 0,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
        "benchmark_receipt_digest": "9" * 64,
    }
    identities = [f"{index + 20:064x}" for index in range(7)]
    preserved = {"shards": [{"state_checks": [{
        "state_identity_digest": identity,
        "completion_rotation_eligibility": {"identity": identity},
    } for identity in identities]}]}
    disposition = {"retained_predecessor_identities": [{
        "state_identity_digest": identity,
        "stratum": "completion_enriched",
    } for identity in identities]}
    monkeypatch.setattr(
        SEARCH_V2, "validate_benchmark_receipt_v2",
        lambda *_args, **_kwargs: dict(pass_receipt))
    monkeypatch.setattr(
        B, "build_v2_predecessor_scientific_input_bindings",
        lambda: dict(envelope))
    monkeypatch.setattr(
        B.STATE_SELECTOR, "validate_frozen_preserved_precontract_failure",
        lambda **_kwargs: copy.deepcopy(preserved))
    monkeypatch.setattr(
        B, "_v2_load_d9d_authorities",
        lambda _out: {"mixed_disposition": copy.deepcopy(disposition)})
    attached = B.attach_v2_parallel_search_mask_context(
        {
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": False,
            "predecessor_scientific_input_bindings": envelope,
        },
        v2_pass_receipt=pass_receipt,
    )
    assert set(attached["preserved_vectors"]) == set(identities)
    assert attached["scientific_masks_accessed"] is True
    assert attached["mask_context_attached_after_v2_pass"] is True


def test_final_eval_allocation_is_reconstructed_exactly_from_states():
    states = [{
        "state_id": f"state-{index}",
        "state_identity_digest": f"{index + 1:064x}",
        "family": "family-a",
        "stratum": "evaluation",
        "scene_id": f"scene-{index}",
    } for index in range(2)]
    source_digest = "f" * 64
    allocation = B._build_final_eval_candidate_allocation(
        states, source_identity_manifest_digest=source_digest)
    expected = {
        "schema": "go2_final_eval_all_candidate_allocation_v1_2",
        "source_identity_manifest_digest": source_digest,
        "candidate_bank_digest": B.V1.bank_digest(),
        "assignments": [{
            "state_id": state["state_id"],
            "state_identity_digest": state["state_identity_digest"],
            "candidate_indices": list(range(len(B.V1.CANDIDATE_BANK))),
        } for state in states],
    }
    expected["allocation_manifest_digest"] = B.canonical_digest(expected)
    assert allocation == expected

    rebound = copy.deepcopy(allocation)
    rebound["assignments"][0]["state_identity_digest"] = "e" * 64
    rebound["allocation_manifest_digest"] = B.canonical_digest({
        key: value for key, value in rebound.items()
        if key != "allocation_manifest_digest"
    })
    assert rebound != B._build_final_eval_candidate_allocation(
        states, source_identity_manifest_digest=source_digest)


def test_launch_hashes_same_pinned_utility_contract_after_alias_swap(
        tmp_path, monkeypatch):
    interruption = _mock_interruption(monkeypatch)
    lexical_root = tmp_path / "repo/.generated/go2_utility_scorer_v1_2"
    first_root = tmp_path / "first/go2_utility_scorer_v1_2"
    second_root = tmp_path / "second/go2_utility_scorer_v1_2"
    first_path = first_root / "scorer_contract.json"
    second_path = second_root / "scorer_contract.json"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    source = {
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }
    selector = {
        "state_selector_feasibility_receipt_digest": "c" * 64,
        "mixed_precontract_disposition_receipt_digest": "d" * 64,
    }

    def artifact(marker):
        payload = {
            "marker": marker,
            "complete": True,
            "scorer_contract_v1_2_digest": B.scorer_contract_digest(),
            "source_repository_clean": True,
            "clean_source_binding": source,
            "clean_source_binding_digest": B.canonical_digest(source),
                "preoutcome_fixed_reissue_validation_interruption_verified":
                    True,
                "preoutcome_fixed_reissue_validation_interruption":
                    interruption["transition"],
                "preoutcome_projection_fix_interruption":
                    interruption["projection"],
                "preoutcome_small_search_performance_interruption_verified":
                    True,
                "preoutcome_small_search_performance_interruption":
                    interruption["performance"],
            **selector,
        }
        payload["contract_artifact_digest"] = B.canonical_digest(payload)
        return payload

    B.atomic_json(first_path, artifact("first"))
    B.atomic_json(second_path, artifact("second"))
    lexical_root.symlink_to(first_root, target_is_directory=True)
    monkeypatch.setattr(
        B, "SCORER_CONTRACT_ARTIFACT_PATH",
        lexical_root / "scorer_contract.json")
    monkeypatch.setattr(B, "clean_source_binding", lambda: source)
    monkeypatch.setattr(
        B, "_load_state_selector_preconditions",
        lambda **_kwargs: dict(selector))
    original_load = B._load_issued_scorer_contract_at_path

    def load_then_swap(path):
        payload = original_load(path)
        lexical_root.unlink()
        lexical_root.symlink_to(second_root, target_is_directory=True)
        return payload

    monkeypatch.setattr(
        B, "_load_issued_scorer_contract_at_path", load_then_swap)
    receipt = B._build_clean_source_launch_receipt({
        "pre_identity_validation_digest": "e" * 64,
    })
    assert receipt["scorer_contract_artifact_sha256"] == B.file_sha256(first_path)
    assert receipt["scorer_contract_artifact_sha256"] != B.file_sha256(second_path)


def test_state_and_branch_identities_are_canonical_and_pre_outcome():
    original = {
        "state_id": "state-x",
        "scene_id": "scene-x",
        "episode_cluster_id": "cluster-x",
        "source_step": 123,
        "goal": {"landmark_id": "red", "range_m": 1.0},
    }
    first = B._state_identity_digest(original)
    reordered = dict(reversed(list(original.items())))
    reordered.update({
        "state_identity_digest": "ignored",
        "state_index": 99,
        "candidate_indices": [11],
        "candidate_rotation_index": 7,
        "branch_identities": [{"post-outcome": "ignored"}],
    })
    assert B._state_identity_digest(reordered) == first
    changed = dict(original)
    changed["goal"] = {"landmark_id": "blue", "range_m": 1.0}
    assert B._state_identity_digest(changed) != first

    manifest = _manifest()
    state = manifest["states"][0]
    binding = {"pool": manifest["pool"], **manifest}
    identity_a = B._branch_identity(state, 0, binding)
    identity_b = B._branch_identity(dict(reversed(list(state.items()))), 0, binding)
    assert identity_a == identity_b
    assert B._branch_identity(state, 1, binding)["branch_identity_digest"] \
        != identity_a["branch_identity_digest"]


def test_public_active_manifest_validator_delegates_exactly_once(monkeypatch):
    calls = []
    manifest = {"schema": "synthetic-active-manifest"}
    monkeypatch.setattr(
        B, "_validate_state_manifest",
        lambda payload, pool: calls.append((payload, pool)))
    assert B.validate_active_state_manifest_for_consumption(manifest) is None
    assert calls == [(manifest, "scorer_fit")]
    with pytest.raises(RuntimeError, match="active scorer-fit"):
        B.validate_active_state_manifest_for_consumption(manifest, "final_eval")
    assert calls == [(manifest, "scorer_fit")]


def test_public_active_manifest_loader_pins_before_read_and_rejects_symlinks(
        tmp_path, monkeypatch):
    lexical_root = tmp_path / "repo/.generated/go2_branch_corpus_v1_2"
    first_root = tmp_path / "first/go2_branch_corpus_v1_2"
    second_root = tmp_path / "second/go2_branch_corpus_v1_2"
    first_path = first_root / "scorer_fit/state_manifest.json"
    second_path = second_root / "scorer_fit/state_manifest.json"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    first = {"marker": "first"}
    second = {"marker": "second"}
    first_path.write_text(json.dumps(first))
    second_path.write_text(json.dumps(second))
    lexical_root.symlink_to(first_root, target_is_directory=True)
    monkeypatch.setattr(B, "OUT_ROOT", lexical_root)
    calls = []

    def validate(manifest, pool="scorer_fit"):
        calls.append((manifest, pool))
        # A root-alias swap during validation cannot change the bytes already
        # read from the canonical pinned path.
        lexical_root.unlink()
        lexical_root.symlink_to(second_root, target_is_directory=True)

    monkeypatch.setattr(
        B, "validate_active_state_manifest_for_consumption", validate)
    loaded = B.load_active_state_manifest_for_consumption(
        lexical_root / "scorer_fit/state_manifest.json")
    assert loaded == first
    assert calls == [(first, "scorer_fit")]

    lexical_root.unlink()
    lexical_root.symlink_to(first_root, target_is_directory=True)
    redirected = first_root / "redirected.json"
    redirected.write_text(json.dumps(first))
    first_path.unlink()
    first_path.symlink_to(redirected)
    with pytest.raises(RuntimeError, match="symlinked corpus paths"):
        B.load_active_state_manifest_for_consumption(
            lexical_root / "scorer_fit/state_manifest.json")

    first_path.unlink()
    nested = first_root / "nested"
    nested.mkdir()
    (nested / "state_manifest.json").write_text(json.dumps(first))
    scorer_fit_dir = first_root / "scorer_fit"
    scorer_fit_dir.rmdir()
    scorer_fit_dir.symlink_to(nested, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked corpus paths"):
        B.load_and_validate_active_state_manifest_for_consumption(
            lexical_root / "scorer_fit/state_manifest.json")


def test_public_scorer_fit_artifact_pin_is_finite_and_exact(
        tmp_path, monkeypatch):
    lexical_root = tmp_path / "repo/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "managed/go2_branch_corpus_v1_2"
    artifact = target_root / "scorer_fit/candidate_allocation_manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    monkeypatch.setattr(B, "OUT_ROOT", lexical_root)
    raw = lexical_root / "scorer_fit/candidate_allocation_manifest.json"
    assert B.pin_active_scorer_fit_artifact_for_consumption(
        raw, "candidate_allocation_manifest.json") == artifact
    with pytest.raises(RuntimeError, match="registered scorer-fit"):
        B.pin_active_scorer_fit_artifact_for_consumption(
            raw, "row_records/arbitrary.json")
    with pytest.raises(RuntimeError, match="path identity changed"):
        B.pin_active_scorer_fit_artifact_for_consumption(
            lexical_root / "scorer_fit/state_manifest.json",
            "candidate_allocation_manifest.json")


def test_branch_row_validates_bound_frames_and_rejects_rehashed_binding_tamper(tmp_path):
    manifest = _manifest(candidate_indices=(0,))
    state = manifest["states"][0]
    identity, row = _row(manifest, 0, out=tmp_path, valid=True)
    B._validate_branch_row(row, state, identity, manifest, tmp_path)

    frame_path = tmp_path / row["horizon_frames"][0]["path"]
    frame_path.write_bytes(b"corrupted-frame")
    with pytest.raises(RuntimeError, match="frame receipt mismatch"):
        B._validate_branch_row(row, state, identity, manifest, tmp_path)

    # Restore the frame, then show that recomputing the row self-digest cannot
    # conceal a changed scientific binding.
    frame_path.write_bytes(b"bound-frame:c0-h1.png")
    tampered = dict(row)
    tampered["oracle_v1_2_digest"] = "other-oracle"
    tampered["branch_row_digest"] = B.canonical_digest(
        {key: value for key, value in tampered.items() if key != "branch_row_digest"}
    )
    with pytest.raises(RuntimeError, match="oracle_v1_2_digest mismatch"):
        B._validate_branch_row(tampered, state, identity, manifest, tmp_path)


def test_corrupted_row_is_preserved_then_exact_registered_row_resumes(tmp_path):
    manifest = _manifest(candidate_indices=(0,))
    identity, row = _row(manifest, 0)
    path = B._row_path(tmp_path, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    corrupted = b'{"schema":"broken-row"'
    path.write_bytes(corrupted)

    assert B._completed_rows(manifest, tmp_path) == {}
    assert not path.exists()
    preserved = list((tmp_path / "invalid_attempts").glob("*.invalid"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == corrupted

    B._write_row(tmp_path, identity, row)
    recovered = B._completed_rows(manifest, tmp_path)
    assert recovered[("state-000", 0)] == row
    receipt = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=2.5)
    assert receipt["complete"] is True
    assert receipt["attempted_branches"] == 1
    assert receipt["corpus_digest"] == B.canonical_digest(
        receipt["corpus_digest_payload"]
    )


@pytest.mark.parametrize(
    "artifact",
    ("branch_rows", "receipt", "latents", "row_record", "frame"),
)
def test_any_durable_outcome_artifact_seals_identity_replacement(
        tmp_path, artifact, monkeypatch):
    # Treat the pytest parent as the managed generated root so the production
    # helper pins ``tmp_path`` once as a pool directory before enumeration.
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path.parent)
    # Empty staging directories are not outcomes; the first durable file is.
    (tmp_path / "row_records").mkdir()
    (tmp_path / "frames" / "family-a").mkdir(parents=True)
    assert B._outcome_generation_started(tmp_path) is False
    if artifact == "branch_rows":
        path = tmp_path / "branch_rows.jsonl"
    elif artifact == "receipt":
        path = tmp_path / "corpus_receipt.json"
    elif artifact == "latents":
        path = tmp_path / "latents_index.json"
    elif artifact == "row_record":
        path = tmp_path / "row_records" / "registered.json"
    else:
        path = tmp_path / "frames" / "family-a" / "registered.png"
    path.write_bytes(b"durable-outcome")
    assert B._outcome_generation_started(tmp_path) is True


def test_compile_receipt_is_deterministic_and_independently_bound(tmp_path):
    manifest = _manifest(candidate_indices=(0, 1))
    # Write in reverse order; compilation must follow frozen manifest order.
    for candidate_index in (1, 0):
        identity, row = _row(manifest, candidate_index)
        B._write_row(tmp_path, identity, row)

    first = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=9.0)
    ledger = [json.loads(line) for line in
              (tmp_path / "branch_rows.jsonl").read_text().splitlines()]
    assert [row["candidate_index"] for row in ledger] == [0, 1]
    assert first["branch_rows_sha256"] == B.file_sha256(
        tmp_path / "branch_rows.jsonl"
    )
    assert first["corpus_digest"] == B.canonical_digest(first["corpus_digest_payload"])
    assert first["corpus_digest_payload"]["branch_row_digests"] == [
        row["branch_row_digest"] for row in ledger
    ]

    ledger_bytes = (tmp_path / "branch_rows.jsonl").read_bytes()
    receipt_bytes = (tmp_path / "corpus_receipt.json").read_bytes()
    second = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=99.0)
    assert second == first
    assert (tmp_path / "branch_rows.jsonl").read_bytes() == ledger_bytes
    assert (tmp_path / "corpus_receipt.json").read_bytes() == receipt_bytes


def test_compile_partial_to_full_preserves_ledger_and_then_is_byte_idempotent(
        tmp_path):
    manifest = _manifest(candidate_indices=(0, 1))
    identity_0, row_0 = _row(manifest, 0)
    B._write_row(tmp_path, identity_0, row_0)
    row_0_path = B._row_path(tmp_path, identity_0)
    row_0_bytes = row_0_path.read_bytes()

    partial = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=1.0)
    assert partial["complete"] is False
    partial_ledger = (tmp_path / "branch_rows.jsonl").read_bytes()
    partial_receipt = (tmp_path / "corpus_receipt.json").read_bytes()

    identity_1, row_1 = _row(manifest, 1)
    B._write_row(tmp_path, identity_1, row_1)
    complete = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=2.0)
    assert complete["complete"] is True
    assert complete["attempted_branches"] == 2
    assert row_0_path.read_bytes() == row_0_bytes
    preserved_ledgers = list((tmp_path / "invalid_attempts").glob(
        "branch_rows.jsonl.*.superseded-or-invalid-compilation.invalid"
    ))
    preserved_receipts = list((tmp_path / "invalid_attempts").glob(
        "corpus_receipt.json.*.superseded-or-invalid-compilation.invalid"
    ))
    assert len(preserved_ledgers) == 1
    assert len(preserved_receipts) == 1
    assert preserved_ledgers[0].read_bytes() == partial_ledger
    assert preserved_receipts[0].read_bytes() == partial_receipt

    full_ledger = (tmp_path / "branch_rows.jsonl").read_bytes()
    full_receipt = (tmp_path / "corpus_receipt.json").read_bytes()
    retained = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=99.0)
    assert retained == complete
    assert (tmp_path / "branch_rows.jsonl").read_bytes() == full_ledger
    assert (tmp_path / "corpus_receipt.json").read_bytes() == full_receipt
    assert len(list((tmp_path / "invalid_attempts").iterdir())) == 2


def test_zero_new_smoke_reuses_exact_replay_receipt_bytes(
        tmp_path, monkeypatch):
    manifest = _manifest(candidate_indices=tuple(range(6)))
    for candidate_index in range(6):
        identity, row = _row(
            manifest, candidate_index, out=tmp_path, valid=True)
        B._write_row(tmp_path, identity, row)
    corpus = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=3.0)
    rows = list(B._completed_rows(manifest, tmp_path).values())
    replay = {
        "state_id": manifest["states"][0]["state_id"],
        "candidate": rows[0]["candidate"],
        "snapshot_digest": rows[0]["snapshot_digest"],
        "exact_repeat": True,
        "separate_render_scene_physically_inert": True,
    }
    monkeypatch.setattr(B, "scorer_contract_digest", lambda: "f" * 64)
    smoke = B._build_smoke_branch_receipt(
        manifest, rows, corpus_digest=corpus["corpus_digest"],
        replay_check=replay)
    assert smoke["scorer_contract_v1_2_digest"] == manifest[
        "scorer_contract_v1_2_digest"]
    assert smoke["scorer_contract_v1_2_digest"] != "f" * 64
    B.atomic_json(tmp_path / "smoke_branch_receipt.json", smoke)
    before = (tmp_path / "smoke_branch_receipt.json").read_bytes()

    assert B._load_valid_smoke_branch_receipt(manifest, tmp_path, rows) == smoke
    assert B._compile_corpus(
        manifest, tmp_path, invocation_runtime_s=100.0) == corpus
    assert (tmp_path / "smoke_branch_receipt.json").read_bytes() == before

    tampered = dict(smoke)
    tampered["replay_check"] = {**replay, "exact_repeat": False}
    tampered["smoke_branch_receipt_digest"] = B.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "smoke_branch_receipt_digest"
    })
    B.atomic_json(tmp_path / "smoke_branch_receipt.json", tampered)
    with pytest.raises(RuntimeError, match="replay proof"):
        B._load_valid_smoke_branch_receipt(manifest, tmp_path, rows)


def test_encoder_accepts_valid_compiled_receipt_and_rejects_corruption(
        tmp_path, monkeypatch):
    encoder = _encoder_module()
    manifest = _manifest(candidate_indices=(0,))
    _patch_encoder_launch(monkeypatch, encoder, manifest)
    identity, row = _row(manifest, 0)
    B._write_row(tmp_path, identity, row)
    B._compile_corpus(manifest, tmp_path, invocation_runtime_s=1.0)
    _write_manifest(tmp_path, manifest)

    loaded_manifest, receipt, rows = encoder._load_inputs(
        tmp_path, allow_partial=False
    )
    assert loaded_manifest["state_manifest_digest"] == manifest["state_manifest_digest"]
    assert receipt["complete"] is True
    assert rows == [row]

    receipt_path = tmp_path / "corpus_receipt.json"
    corrupted_receipt = json.loads(receipt_path.read_text())
    corrupted_receipt["corpus_digest_payload"]["valid_branch_count"] = 999
    B.atomic_json(receipt_path, corrupted_receipt)
    with pytest.raises(RuntimeError, match="not independently reproducible"):
        encoder._load_inputs(tmp_path, allow_partial=False)

    # Restore a valid receipt, then alter only ledger bytes.  Parsing still
    # succeeds, so this specifically exercises the file-SHA binding.
    B._compile_corpus(manifest, tmp_path, invocation_runtime_s=1.0)
    with (tmp_path / "branch_rows.jsonl").open("a") as sink:
        sink.write("\n")
    with pytest.raises(RuntimeError, match="ledger digest"):
        encoder._load_inputs(tmp_path, allow_partial=False)


def test_encoder_refuses_partial_receipt_with_top_level_complete_tampered(
        tmp_path, monkeypatch):
    encoder = _encoder_module()
    manifest = _manifest(candidate_indices=(0, 1))
    _patch_encoder_launch(monkeypatch, encoder, manifest)
    identity, row = _row(manifest, 0)
    B._write_row(tmp_path, identity, row)
    receipt = B._compile_corpus(manifest, tmp_path, invocation_runtime_s=1.0)
    assert receipt["complete"] is False
    _write_manifest(tmp_path, manifest)

    receipt["complete"] = True
    B.atomic_json(tmp_path / "corpus_receipt.json", receipt)
    with pytest.raises(RuntimeError):
        encoder._load_inputs(tmp_path, allow_partial=False)


def test_encoder_binds_allocation_and_registered_identity_fields(
        tmp_path, monkeypatch):
    encoder = _encoder_module()
    manifest = _manifest(candidate_indices=(0, 1))
    _patch_encoder_launch(monkeypatch, encoder, manifest)
    rows = []
    for candidate_index in (0, 1):
        identity, row = _row(manifest, candidate_index)
        B._write_row(tmp_path, identity, row)
        rows.append(row)
    B._compile_corpus(manifest, tmp_path, invocation_runtime_s=1.0)
    _write_manifest(tmp_path, manifest)

    altered_binding = dict(rows[0])
    altered_binding["candidate_allocation_manifest_digest"] = "other-allocation"
    altered_binding["branch_row_digest"] = B.canonical_digest({
        key: value for key, value in altered_binding.items()
        if key != "branch_row_digest"
    })
    with pytest.raises(RuntimeError, match="candidate_allocation_manifest_digest"):
        encoder._validate_row(
            altered_binding, manifest, manifest["scorer_contract_v1_2_digest"]
        )

    # Keep a registered branch digest but relabel its candidate.  Rebind every
    # row-ledger and receipt digest so the independent identity mapping—not an
    # earlier checksum failure—is what refuses the corruption.
    relabelled = [dict(row) for row in rows]
    relabelled[0]["candidate"] = relabelled[1]["candidate"]
    relabelled[0]["candidate_index"] = relabelled[1]["candidate_index"]
    relabelled[0]["branch_row_digest"] = B.canonical_digest({
        key: value for key, value in relabelled[0].items()
        if key != "branch_row_digest"
    })
    ledger_text = "".join(
        json.dumps(B.V1._jsonable(row), sort_keys=True) + "\n" for row in relabelled
    )
    B.atomic_text(tmp_path / "branch_rows.jsonl", ledger_text)
    receipt_path = tmp_path / "corpus_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    ledger_sha = B.file_sha256(tmp_path / "branch_rows.jsonl")
    receipt["branch_rows_sha256"] = ledger_sha
    payload = receipt["corpus_digest_payload"]
    payload["branch_rows_sha256"] = ledger_sha
    payload["branch_row_digests"] = [
        row["branch_row_digest"] for row in relabelled
    ]
    receipt["corpus_digest"] = B.canonical_digest(payload)
    B.atomic_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="relabels a registered branch identity"):
        encoder._load_inputs(tmp_path, allow_partial=False)


def test_encoder_noop_index_write_preserves_exact_bytes_and_mtime(tmp_path):
    encoder = _encoder_module()
    path = tmp_path / "latents_index.json"
    payload = {"schema": "synthetic-index", "complete": True}
    payload["latents_index_digest"] = encoder.canonical_digest(payload)
    assert encoder._write_index_if_changed(path, payload, {}) is True
    first_bytes = path.read_bytes()
    first_mtime = path.stat().st_mtime_ns
    assert encoder._write_index_if_changed(path, payload, dict(payload)) is False
    assert path.read_bytes() == first_bytes
    assert path.stat().st_mtime_ns == first_mtime


def _synthetic_global_common() -> dict:
    common = {key: "1" * 64 for key in B.STATE_SHARD_COMMON_KEYS}
    common.update({
        "source_repository_commit": "2" * 40,
        "genesis_backend": "cpu",
    })
    return common


def _synthetic_global_execution_material() -> tuple[dict, dict, dict]:
    common = _synthetic_global_common()
    prefix_states = [
        {"scene_id": f"prefix-{index:02d}",
         "stratum": "general" if index < 5 else "safety_enriched"}
        for index in range(10)
    ]
    selected_states = [
        {"scene_id": f"selected-{index:02d}",
         "stratum": "completion_enriched"}
        for index in range(5)
    ]
    inputs = {
        "common": common,
        "resolver_cursor_scene_id": "prefix-cursor",
        "candidate_scene_ids": [f"candidate-{index:02d}" for index in range(17)],
        "fixed_shard_evidence": [{"family": f"family-{index}"}
                                 for index in range(7)],
        "prefix": {
            "states": prefix_states,
            "state_shard_bindings": {
                **common,
                "exclusion_binding": {"digest": "3" * 64},
                "family_allow_list_digest": "4" * 64,
            },
            "capture_provenance": [
                {"scene_id": f"prefix-{index:02d}"} for index in range(12)
            ],
            "scene_rejection_reasons": {},
            "receipt_binding": {"receipt_digest": "5" * 64},
            "performance_receipt_binding": {"receipt_digest": "6" * 64},
        },
    }
    material = {
        "inputs": inputs,
        "selected_states": selected_states,
        "plan": {
            "fixture_suite_digest": "7" * 64,
            "production_instance_digest": "8" * 64,
            "global_exact_model_plan_digest": "9" * 64,
            "model_execution_plan_digest": "a" * 64,
        },
        "terminal": {
            "model_execution_result_digest": "b" * 64,
            "global_exact_terminal_result_digest": "c" * 64,
        },
        "materialized": {
            B.GLOBAL_EXACT_MODEL.ALLOCATION_RESULT_DIGEST_KEY: "d" * 64,
            "selected_scene_ids": [row["scene_id"] for row in selected_states],
        },
        "allocation_contract_disposition":
            B.GLOBAL_EXACT_MODEL.legacy_allocation_contract_disposition(),
    }
    context = {
        "coupling_report": {B.GLOBAL_EXACT_AUTHORITY.REPORT_SELF_KEY: "e" * 64},
        "execution_amendment": {
            B.GLOBAL_EXACT_AUTHORITY.AMENDMENT_SELF_KEY: "f" * 64},
    }
    joint = {B.GLOBAL_EXACT_JOINT_RECEIPT_SELF_KEY: "0" * 64}
    return material, context, joint


def test_global_small_shard_uses_historical_bindings_and_new_certificate():
    material, context, joint = _synthetic_global_execution_material()
    shard = B._build_global_exact_small_terminal_shard(
        material, context, joint)
    assert shard["states"] == [
        *material["inputs"]["prefix"]["states"],
        *material["selected_states"],
    ]
    assert shard["source_repository_commit"] == "2" * 40
    assert shard["state_resolution_subprocess_transport"]["resume_scope"] == \
        B.GLOBAL_EXACT_SMALL_TRANSPORT_RESUME_SCOPE
    assert "small_completion_joint_allocation_search" not in shard
    assert shard["small_completion_global_exact_execution"][
        "global_exact_joint_receipt_digest"] == "0" * 64
    assert shard["state_shard_digest"] == B.canonical_digest({
        key: value for key, value in shard.items()
        if key != "state_shard_digest"
    })

    changed = copy.deepcopy(material)
    changed["inputs"]["prefix"]["state_shard_bindings"][
        "source_repository_commit"] = "3" * 40
    with pytest.raises(RuntimeError, match="small shard binding"):
        B._build_global_exact_small_terminal_shard(changed, context, joint)


def test_global_phase2_uses_historical_authority_and_exact_mapping_callback(
        monkeypatch):
    allocation = {"allocation_manifest_digest": "1" * 64}
    material = {
        "inputs": {"common": _synthetic_global_common()},
        "allocation": allocation,
        "states": [{"state_id": "state"}],
        "preserved_vectors": {},
    }
    monkeypatch.setattr(B, "_completion_states_for_phase2",
                        lambda **_kwargs: [{"completion": True}] * 40)
    observed = {}

    def build(**kwargs):
        observed.update(kwargs)
        assert kwargs["certify_allocation_solve_free"](allocation) == allocation
        with pytest.raises(RuntimeError, match="allocation changed"):
            kwargs["certify_allocation_solve_free"]({"different": True})
        return {"preserved_state_revalidation_receipt_digest": "2" * 64}

    monkeypatch.setattr(
        B.STATE_SELECTOR,
        "build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation",
        build)
    monkeypatch.setattr(
        B.STATE_SELECTOR,
        "validate_preserved_state_revalidation_receipt_from_solve_free_certified_allocation",
        lambda receipt, **_kwargs: receipt)
    receipt = B._build_global_exact_phase2_receipt(material)
    common = material["inputs"]["common"]
    assert receipt["preserved_state_revalidation_receipt_digest"] == "2" * 64
    assert observed["source_repository_commit"] == common[
        "source_repository_commit"]
    assert observed["successor_selection_digest"] == common[
        "selection_digest"]
    assert observed["state_selector_feasibility_receipt_digest"] == common[
        "state_selector_feasibility_receipt_digest"]
    assert observed["mixed_precontract_disposition_receipt_digest"] == common[
        "mixed_precontract_disposition_receipt_digest"]


def test_global_manifest_and_shard_route_before_legacy_validators(monkeypatch):
    manifest = {"small_completion_global_exact_execution": {}}
    shard = {
        "small_completion_global_exact_execution": {},
        "family": B.REACHABILITY_REDRIVE_FAMILY,
    }
    observed = []
    monkeypatch.setattr(
        B, "_validate_global_exact_state_manifest",
        lambda value: observed.append(("manifest", value)))
    monkeypatch.setattr(
        B, "_validate_global_exact_small_state_shard",
        lambda value, path: observed.append(("shard", value, path)))
    monkeypatch.setattr(
        B, "_load_clean_source_launch_receipt",
        lambda: (_ for _ in ()).throw(AssertionError("legacy launch opened")))
    monkeypatch.setattr(
        B, "_validate_small_completion_joint_search_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy search replayed")))
    B._validate_state_manifest(manifest, "scorer_fit")
    B._validate_state_shard(shard, Path("synthetic.json"), "scorer_fit")
    assert [row[0] for row in observed] == ["manifest", "shard"]


def test_global_allocation_consumer_requires_exact_supersession_certificate(
        monkeypatch, tmp_path):
    allocation = {
        "source_identity_manifest_digest": "1" * 64,
        "allocation_manifest_digest": "2" * 64,
    }
    manifest = {
        "small_completion_global_exact_execution": {
            "execution_amendment_digest": "3" * 64},
        "legacy_allocation_contract_disposition":
            B.GLOBAL_EXACT_MODEL.legacy_allocation_contract_disposition(),
        "pre_allocation_identity_manifest_digest": "1" * 64,
        "candidate_allocation_manifest_digest": "2" * 64,
        "state_selector_amendment_digest": "4" * 64,
        "state_selector_feasibility_receipt_digest": "5" * 64,
        "preserved_state_revalidation_receipt_digest": "6" * 64,
    }
    allocation_path = tmp_path / "candidate_allocation_manifest.json"
    allocation_path.write_text(json.dumps(allocation))
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path.parent)
    monkeypatch.setattr(B, "_pin_generated_path",
                        lambda _raw, _expected: allocation_path)
    observed = []
    monkeypatch.setattr(
        B, "_validate_global_exact_state_manifest",
        lambda supplied: observed.append(dict(supplied)))
    monkeypatch.setattr(
        B.STATE_SELECTOR, "validate_allocation_manifest_structure_solve_free",
        lambda supplied, **_kwargs: observed.append(dict(supplied)))
    certified = B.validate_global_exact_allocation_for_consumption(
        manifest, allocation)
    assert certified["allocation_manifest"] == allocation
    assert len(observed) == 2

    changed = copy.deepcopy(manifest)
    changed["legacy_allocation_contract_disposition"][
        "legacy_choice_rule_status"] = "PRESERVED_MANDATORY"
    disposition_key = (
        B.GLOBAL_EXACT_MODEL.ALLOCATION_CONTRACT_DISPOSITION_SELF_KEY)
    changed["legacy_allocation_contract_disposition"][
        disposition_key] = (
            B.GLOBAL_EXACT_MODEL.canonical_digest({
                key: value for key, value in changed[
                    "legacy_allocation_contract_disposition"].items()
                if key != disposition_key
            }))
    with pytest.raises(RuntimeError, match="supersession certificate"):
        B.validate_global_exact_allocation_for_consumption(
            changed, allocation)


def test_global_successor_contract_view_separates_current_scorer_digest(
        monkeypatch, tmp_path):
    payload = {
        "contract_body": {
            "current_scorer_contract_v1_2_digest": "1" * 64,
        },
        "clean_source_launch_receipt_digest": "2" * 64,
        "source_repository_commit": "3" * 40,
        "clean_source_binding_digest": "4" * 64,
        "bound_implementations_digest": "5" * 64,
        "scorer_contract_artifact_digest": "6" * 64,
        "operational_launch": {"synthetic": True},
        "launch_state_selector_feasibility_receipt_digest": "7" * 64,
        "mixed_precontract_disposition_receipt_digest": "8" * 64,
        "global_exact_execution_amendment_digest": "9" * 64,
        "global_exact_successor_scorer_contract_digest": "a" * 64,
        "scientific_predecessor_launch_bindings": {
            "scorer_contract_v1_2_digest": "b" * 64,
        },
    }
    path = tmp_path / "successor.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(B, "clean_source_binding", lambda: {})
    monkeypatch.setattr(
        B, "_global_exact_successor_contract_payload",
        lambda _manifest, *, source: copy.deepcopy(payload))
    monkeypatch.setattr(
        B, "GLOBAL_EXACT_SUCCESSOR_SCORER_CONTRACT_PATH", path)
    monkeypatch.setattr(B, "_pin_generated_path", lambda *_args, **_kwargs: path)
    view = B.load_global_exact_successor_scorer_contract_for_consumption({})
    assert view["current_scorer_contract_v1_2_digest"] == "1" * 64
    assert view["scientific_predecessor_launch_bindings"][
        "scorer_contract_v1_2_digest"] == "b" * 64


def test_encoding_smoke_requires_exact_global_scorer_lineage(monkeypatch):
    manifest = {
        "small_completion_global_exact_execution": {},
        "scorer_contract_v1_2_digest": "1" * 64,
    }
    successor = {
        "current_scorer_contract_v1_2_digest": "2" * 64,
        "global_exact_successor_scorer_contract_digest": "3" * 64,
    }
    monkeypatch.setattr(
        B, "load_global_exact_successor_scorer_contract_for_consumption",
        lambda supplied: successor if supplied is manifest else None)
    lineage = {
        "schema": "go2_utility_scorer_v1_2_global_exact_contract_lineage_v1",
        "scientific_predecessor_scorer_contract_v1_2_digest": "1" * 64,
        "current_scorer_contract_v1_2_digest": "2" * 64,
        "global_exact_successor_scorer_contract_digest": "3" * 64,
    }
    assert B._encoding_smoke_matches_global_exact_scorer_lineage(
        {"global_exact_scorer_contract_lineage": lineage}, manifest)
    changed = copy.deepcopy(lineage)
    changed["scientific_predecessor_scorer_contract_v1_2_digest"] = "4" * 64
    assert not B._encoding_smoke_matches_global_exact_scorer_lineage(
        {"global_exact_scorer_contract_lineage": changed}, manifest)
    assert B._encoding_smoke_matches_global_exact_scorer_lineage({}, {})


def test_global_finalizer_is_idempotent_ordered_and_never_uses_legacy_search(
        monkeypatch, tmp_path):
    material, context, _joint_seed = _synthetic_global_execution_material()
    material.update({
        "allocation": {"allocation_manifest_digest": "1" * 64},
        "states": [],
        "preserved_vectors": {},
    })
    joint = {
        B.GLOBAL_EXACT_JOINT_RECEIPT_SELF_KEY: "2" * 64,
    }
    shard = {"state_shard_digest": "3" * 64}
    phase2 = {"preserved_state_revalidation_receipt_digest": "4" * 64}
    manifest = {
        "global_exact_execution_amendment_digest": "5" * 64,
        "state_manifest_digest": "6" * 64,
    }
    writes = []
    monkeypatch.setattr(B, "ROOT", tmp_path)
    monkeypatch.setattr(
        B, "OUT_ROOT", tmp_path / ".generated/go2_branch_corpus_v1_2")
    monkeypatch.setattr(
        B, "_global_exact_validated_allocation_material",
        lambda **_kwargs: material)
    monkeypatch.setattr(
        B, "_build_global_exact_joint_receipt",
        lambda *_args: joint)
    monkeypatch.setattr(
        B, "_build_global_exact_small_terminal_shard",
        lambda *_args: shard)
    monkeypatch.setattr(
        B, "_build_global_exact_phase2_receipt",
        lambda *_args: phase2)
    monkeypatch.setattr(
        B, "_build_global_exact_state_manifest_payload",
        lambda **_kwargs: manifest)
    monkeypatch.setattr(
        B, "_write_or_require_exact_json",
        lambda path, payload, *, label: writes.append(
            (label, Path(path).name, dict(payload))) or dict(payload))
    monkeypatch.setattr(
        B, "issue_global_exact_successor_scorer_contract",
        lambda _manifest: {
            "global_exact_successor_scorer_contract_digest": "7" * 64})
    monkeypatch.setattr(B, "_validate_global_exact_state_manifest",
                        lambda _manifest: None)
    monkeypatch.setattr(
        B, "_validate_small_completion_joint_search_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy search replayed")))
    result = B.finalize_global_exact_feasible_allocation(
        execution_context={"candidate_outcomes_consumed": False},
        instance={}, execution_plan={}, execution_result={})
    assert [label for label, _path, _payload in writes] == [
        "global exact joint receipt",
        "global exact small terminal state shard",
        "global exact candidate allocation",
        "global exact preserved-state revalidation",
        "global exact state manifest",
    ]
    assert result["global_exact_successor_scorer_contract_digest"] == "7" * 64
    assert result["candidate_outcomes_consumed"] is False
