"""Non-Genesis durability tests for the v1.2 branch corpus pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_go2_branch_corpus_v1_2 as B


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
        **selector_preconditions,
    }
    contract_artifact["contract_artifact_digest"] = B.canonical_digest(
        contract_artifact
    )
    B.atomic_json(contract_path, contract_artifact)
    monkeypatch.setattr(B, "SCORER_CONTRACT_ARTIFACT_PATH", contract_path)
    out = tmp_path / "scorer_fit"
    assert B.issue_pre_identity_allocation_validation(out) == 0
    path = out / B.PRE_IDENTITY_VALIDATION_NAME
    first = path.read_bytes()
    artifact = json.loads(first)
    B.ALLOC.validate_pre_identity_structural_validation(artifact)
    assert artifact["global"]["state_slot_count"] == 120
    assert artifact["global"]["candidate_slot_count"] == 720
    assert artifact["goal_type_validation"]["status"] == (
        "NOT_EVALUABLE_BEFORE_STATE_IDENTITIES"
    )
    assert B.issue_pre_identity_allocation_validation(out) == 0
    assert path.read_bytes() == first


def test_issued_scorer_contract_uses_exact_managed_utility_root(
        tmp_path, monkeypatch):
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


def test_launch_hashes_same_pinned_utility_contract_after_alias_swap(
        tmp_path, monkeypatch):
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
def test_any_durable_outcome_artifact_seals_identity_replacement(tmp_path, artifact):
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


def test_zero_new_smoke_reuses_exact_replay_receipt_bytes(tmp_path):
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
    smoke = B._build_smoke_branch_receipt(
        manifest, rows, corpus_digest=corpus["corpus_digest"],
        replay_check=replay)
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
