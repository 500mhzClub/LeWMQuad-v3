from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from lewm.oracle.go2_textured_v03_renderer import BasePose
from scripts import build_go2_counterfactual_fidelity_stage_a_v1_2 as A
from scripts import encode_go2_counterfactual_fidelity_stage_a_v1_2 as E


def test_frozen_source_witness_and_identity_manifest_are_deterministic() -> None:
    source = A.load_source_evidence()
    assert len(source.manifest["states"]) == 20
    assert len(source.outcomes) == 240
    assert all(outcome.row["valid"] for outcome in source.outcomes.values())

    first = A.build_identity_manifest(source)
    second = A.build_identity_manifest(source)
    assert first == second
    A.validate_identity_manifest(first, source)
    assert first["state_count_registered"] == 20
    assert first["attempted_branch_count_registered"] == 240
    assert len({identity["branch_identity_digest"]
                for state in first["states"]
                for identity in state["branch_identities"]}) == 240
    assert A.canonical_digest({key: value for key, value in first.items()
                               if key != "stage_a_identity_manifest_digest"}) \
        == first["stage_a_identity_manifest_digest"]


def test_outcome_equality_projection_excludes_only_source_runtime() -> None:
    source = A.load_source_evidence()
    outcome = next(iter(source.outcomes.values())).row
    changed_runtime = copy.deepcopy(outcome)
    changed_runtime["wall_time_s"] = 123456.0
    assert A._outcome_projection(changed_runtime) == A._outcome_projection(outcome)

    changed_utility = copy.deepcopy(outcome)
    changed_utility["utility"] += 1e-9
    assert A.canonical_digest(A._outcome_projection(changed_utility)) \
        != A.canonical_digest(A._outcome_projection(outcome))
    assert set(outcome) - set(A.OUTCOME_FIELDS) == {"wall_time_s"}


def test_frame_receipt_binds_raw_pixels_pose_and_exact_camera(tmp_path: Path) -> None:
    pose = BasePose((1.0, 2.0, 0.4), (1.0, 0.0, 0.0, 0.0))
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    result = SimpleNamespace(
        image=image,
        camera_pose_world={
            "position": [1.0, 2.0, 3.0],
            "lookat": [2.0, 2.0, 3.0],
            "up": [0.0, 0.0, 1.0],
        },
        runtime_s=0.1,
    )
    renderer = SimpleNamespace(
        raw_manifest_digest="a" * 64,
        contract_digest=A.renderer_contract_digest(),
    )
    path = tmp_path / "frames/f.png"
    receipt = A._frame_receipt(
        result, pose, path, tmp_path, index_key="horizon", index_value=1,
        renderer=renderer)
    assert receipt["pixel_sha256"] == hashlib.sha256(image.tobytes()).hexdigest()
    assert receipt["base_pose"]["quaternion_order"] == "wxyz"
    A._validate_frame(tmp_path, receipt, index_key="horizon", index_value=1,
                      raw_digest="a" * 64)

    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(RuntimeError, match="frame receipt"):
        A._validate_frame(tmp_path, receipt, index_key="horizon", index_value=1,
                          raw_digest="a" * 64)


def test_corrupt_completed_row_is_preserved_not_overwritten(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = A.load_source_evidence()
    manifest = A.build_identity_manifest(source)
    state = manifest["states"][0]
    identity = state["branch_identities"][0]
    path = A._row_path(tmp_path, identity)
    A.atomic_json(path, {"schema": "corrupt-partial"})
    monkeypatch.setattr(A, "_validate_branch_row",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            RuntimeError("corrupt")))

    completed = A._completed_rows(manifest, source, tmp_path)
    assert completed == {}
    assert not path.exists()
    preserved = list((tmp_path / "invalid_attempts").glob("*.invalid"))
    assert len(preserved) == 1
    assert json.loads(preserved[0].read_text())["schema"] == "corrupt-partial"


def test_raw_encoder_tokens_are_rounded_before_consumer_normalisation() -> None:
    class Arm:
        @staticmethod
        def preprocess(_path: str) -> torch.Tensor:
            return torch.zeros(3, 2, 2)

    raw = torch.linspace(-3.0, 4.0, E.TOKENS * E.TOKEN_DIM).reshape(
        1, E.TOKENS, E.TOKEN_DIM)

    class Encoder:
        def __call__(self, _pixels: torch.Tensor) -> torch.Tensor:
            return raw

    encoded = E.encode_paths(
        Arm(), Encoder(), ["unused"], torch.device("cpu"), torch.float32)
    assert encoded.dtype == np.float16
    assert np.array_equal(encoded, raw.numpy().astype(np.float16))
    assert "consumers reload float16 as float32" in E.TARGET_NORMALISATION


def _sidecar_manifest() -> dict[str, str]:
    return {
        "stage_a_identity_manifest_digest": "1" * 64,
        "assay_spec_digest": "2" * 64,
        "candidate_bank_digest": "3" * 64,
        "oracle_v1_2_digest": "4" * 64,
        "render_contract_digest": "5" * 64,
        "textured_v03_renderer_contract_digest": "6" * 64,
        "preprocess_contract_digest": "7" * 64,
        "preprocessing_digest": "8" * 64,
        "target_encoder_digest": "9" * 64,
        "target_encoder_checkpoint_sha256": "a" * 64,
    }


def test_latent_shard_sidecar_recovers_without_an_index(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "OUT_ROOT", tmp_path)
    shape = (2, 3)
    shard = tmp_path / "latents/context/example.f16"
    digest, byte_count = E.atomic_f16(shard, np.arange(6, dtype=np.float16))
    identity = {
        "state_id": "state", "state_identity_digest": "b" * 64,
        "state_record_digest": "c" * 64,
    }
    frames = [{"sha256": "d" * 64, "slot": 0}]
    manifest = _sidecar_manifest()
    written = E._write_sidecar(
        shard, kind="context", identity=identity, frame_records=frames,
        shape=shape, sha256=digest, byte_count=byte_count, manifest=manifest)
    assert not (tmp_path / "latents_index.json").exists()
    recovered = E._load_sidecar(
        shard, expected_kind="context", expected_identity=identity,
        expected_frames_digest=A.canonical_digest(frames), expected_shape=shape,
        manifest=manifest)
    assert recovered == written
    assert recovered["latent_shard_receipt_digest"]


def test_preexecution_bindings_cover_stage_a_and_direct_contract_sources() -> None:
    bindings = A.source_bindings()
    for path in (
        "scripts/build_go2_counterfactual_fidelity_stage_a_v1_2.py",
        "scripts/encode_go2_counterfactual_fidelity_stage_a_v1_2.py",
        "lewm/oracle/go2_scorer_contract_v1_2.py",
        "scripts/build_dev_v03_proprio_action_manifest_v1.py",
        "scripts/dev_action_slew_reconstruction_v1.py",
        "scripts/dev_frozen_dense_representation_encoders_v1.py",
    ):
        assert path in bindings
        assert len(bindings[path]["sha256"]) == 64


def _synthetic_rows(manifest: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    pose = {
        "position_world_xyz": [1.0, 2.0, 0.4],
        "quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
        "quaternion_order": "wxyz",
    }
    rows = []
    flat = [(state, identity) for state in manifest["states"]
            for identity in state["branch_identities"]]
    for offset, (state, identity) in enumerate(flat[:count]):
        pixel_sha = f"{offset + 1000:064x}"
        row = {
            "state_id": state["state_id"],
            "candidate_index": int(identity["candidate_index"]),
            "candidate": identity["candidate"],
            "branch_identity_digest": identity["branch_identity_digest"],
            "branch_row_digest": f"{offset + 2000:064x}",
            "valid": True,
            "oracle_outcome_equal": True,
            "wall_time_s": 0.25,
            "horizon_base_poses": [copy.deepcopy(pose) for _ in range(4)],
            "horizon_frames": [
                {"pixel_sha256": pixel_sha} for _ in range(4)],
        }
        rows.append(row)
    return rows


def _write_corpus(tmp_path: Path, manifest: Mapping[str, Any],
                  rows: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(json.dumps(A.V1._jsonable(row), sort_keys=True) + "\n"
                   for row in rows)
    A.atomic_text(tmp_path / "branch_rows.jsonl", text)
    ledger_sha = hashlib.sha256(text.encode()).hexdigest()
    payload = A._corpus_identity_payload(manifest, rows, ledger_sha)
    receipt = {
        "schema": "go2_counterfactual_fidelity_stage_a_completion_receipt_v1_2",
        "status": A.STATUS,
        "complete": payload["complete"],
        "state_count": A.EXPECTED_STATES,
        "completed_state_count": payload["completed_state_count"],
        "expected_branch_count": A.EXPECTED_BRANCHES,
        "attempted_branch_count": len(rows),
        "valid_branch_count": len(rows),
        "oracle_equal_branch_count": len(rows),
        "invalid_branch_count": 0,
        "branch_rows_sha256": ledger_sha,
        "branch_row_digests": [row["branch_row_digest"] for row in rows],
        **A._manifest_bindings(manifest),
        "corpus_digest_payload": payload,
        "corpus_digest": A.canonical_digest(payload),
        "runtime_s_completed_rows": 0.0,
        "runtime_s_this_invocation": 0.0,
        "storage_bytes": len(text.encode()),
    }
    receipt["completion_receipt_digest"] = A.canonical_digest(receipt)
    A.atomic_json(tmp_path / "corpus_receipt.json", receipt)
    return receipt


def _raw_identity(state: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    pixel_sha = row["horizon_frames"][0]["pixel_sha256"]
    return {
        "state_id": state["state_id"],
        "candidate": row["candidate"],
        "branch_identity_digest": row["branch_identity_digest"],
        "horizon": 1,
        "captured_base_pose": row["horizon_base_poses"][0],
        "first_pixel_sha256": pixel_sha,
        "repeat_pixel_sha256": pixel_sha,
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "identical": True,
        "renderer_contract_digest": A.renderer_contract_digest(),
        "raw_manifest_digest": state["raw_manifest_digest"],
    }


def _write_smoke_gate(tmp_path: Path, manifest: Mapping[str, Any],
                      rows: list[dict[str, Any]]) -> None:
    smoke_rows = rows[:6]
    state = manifest["states"][0]
    raw = _raw_identity(state, smoke_rows[0])
    text = "".join(json.dumps(A.V1._jsonable(row), sort_keys=True) + "\n"
                   for row in smoke_rows)
    ledger_sha = hashlib.sha256(text.encode()).hexdigest()
    smoke_corpus_digest = A.canonical_digest(
        A._corpus_identity_payload(manifest, smoke_rows, ledger_sha))
    common = {
        "state_id": state["state_id"],
        "branch_identity_digests": [
            row["branch_identity_digest"] for row in smoke_rows],
        "branch_row_digests": [row["branch_row_digest"] for row in smoke_rows],
        "stage_a_identity_manifest_digest":
            manifest["stage_a_identity_manifest_digest"],
        "assay_spec_digest": manifest["assay_spec_digest"],
        "partial_corpus_digest": smoke_corpus_digest,
    }
    branch = {
        "schema": "go2_counterfactual_fidelity_stage_a_smoke_receipt_v1_2",
        "status": A.STATUS,
        "pass": True,
        "resume_only_verified": True,
        "state_identity_digest": state["state_identity_digest"],
        "new_rows_this_invocation": 0,
        "raw_frame_identity": raw,
        **common,
    }
    branch["smoke_receipt_digest"] = A.canonical_digest(branch)
    A.atomic_json(tmp_path / "smoke_receipt.json", branch)

    context_record = {
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
    }
    horizon_records = [{
        "branch_identity_digest": row["branch_identity_digest"],
        "branch_row_digest": row["branch_row_digest"],
    } for row in smoke_rows]
    index = {
        "schema": "go2_counterfactual_fidelity_stage_a_latents_index_v1_2",
        "status": A.STATUS,
        "complete": False,
        "state_count": 1,
        "branch_count": 6,
        "tokens": A.TARGET_ENCODER["tokens"],
        "token_dim": A.TARGET_ENCODER["token_dim"],
        "context_slots": A.CONTEXT_SLOTS,
        "horizons": A.HORIZONS,
        "dtype": "float16",
        "encoder_compute_dtype": "float32",
        "corpus_digest": smoke_corpus_digest,
        "branch_rows_sha256": ledger_sha,
        "context_shape": [1, 3, 768, 1024],
        "horizon_shape": [6, 4, 768, 1024],
        "context_records": [context_record],
        "horizon_records": horizon_records,
        **{key: manifest[key] for key in (
            "stage_a_identity_manifest_digest", "assay_spec_digest",
            "candidate_bank_digest", "oracle_v1_2_digest",
            "render_contract_digest", "textured_v03_renderer_contract_digest",
            "preprocess_contract_digest", "preprocessing_digest",
            "target_encoder_digest", "target_encoder_checkpoint_sha256",
            "source_state_manifest_digest", "source_pilot_branch_ledger_sha256",
        )},
    }
    index["latents_index_digest"] = A.canonical_digest(index)
    A.atomic_json(tmp_path / "latents_index.json", index)
    encoding = {
        "schema": "go2_counterfactual_fidelity_stage_a_smoke_encoding_receipt_v1_2",
        "status": A.STATUS,
        "pass": True,
        "resume_only_verified": True,
        "new_context_shards_this_invocation": 0,
        "new_horizon_shards_this_invocation": 0,
        "raw_frame_identity": raw,
        "latents_index_digest": index["latents_index_digest"],
        **common,
    }
    encoding["smoke_encoding_receipt_digest"] = A.canonical_digest(encoding)
    A.atomic_json(tmp_path / "smoke_encoding_receipt.json", encoding)


def test_branch_smoke_receipt_binds_assay_and_recovers_raw_without_branch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "stage_a_identity_manifest_digest": "1" * 64,
        "assay_spec_digest": "2" * 64,
        "states": [{
            "state_id": "state",
            "state_identity_digest": "3" * 64,
        }],
    }
    raw_identity = {"identical": True, "pixel_sha256": "4" * 64}

    completed = {
        ("state", index): {
            "valid": True,
            "oracle_outcome_equal": True,
                "horizon_frames": [{}, {}, {}, {}],
                "candidate": f"candidate-{index}",
                "branch_identity_digest": f"{index:064x}",
            "branch_row_digest": f"{index + 6:064x}",
        }
        for index in range(6)
    }
    monkeypatch.setattr(A, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(A, "_completed_rows", lambda *_args: completed)
    monkeypatch.setattr(A.V1, "_load_shared", lambda _backend: object())
    recovered = []
    monkeypatch.setattr(
        A, "_recover_raw_frame_identity",
        lambda *_args: recovered.append(True) or raw_identity,
    )
    monkeypatch.setattr(
        A, "execute_branch_with_pose_capture",
        lambda *_args, **_kwargs: pytest.fail("retained smoke reran a branch"),
    )
    monkeypatch.setattr(
        A, "_compile_corpus",
        lambda *_args: {"corpus_digest": "5" * 64},
    )
    monkeypatch.setattr(A, "_corpus_identity_payload", lambda *_args: {})

    assert A.stage_branches(
        manifest, SimpleNamespace(), smoke=True, state_offset=0, state_limit=1) == 0
    branch = json.loads((tmp_path / "smoke_receipt.json").read_text())
    assert branch["assay_spec_digest"] == manifest["assay_spec_digest"]
    assert branch["resume_only_verified"] is True
    assert recovered == [True]


def test_valid_state_record_is_retained_on_live_redrive_mismatch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"state_id": "state", "state_identity_digest": "3" * 64}
    manifest = {
        "stage_a_identity_manifest_digest": "1" * 64,
        "assay_spec_digest": "2" * 64,
    }
    path = A._state_record_path(tmp_path, state)
    record = {
        "state_record_digest": "4" * 64,
        "redrive_projection_digest": "5" * 64,
    }
    A.atomic_json(path, record)
    original = path.read_bytes()
    monkeypatch.setattr(A, "_validate_state_record", lambda *_args: None)

    with pytest.raises(RuntimeError, match="retained valid state record"):
        A._get_or_create_state_record(
            tmp_path, manifest, state, {"changed": True}, [], SimpleNamespace())
    assert path.read_bytes() == original
    evidence = list((tmp_path / "invalid_attempts/state_redrive_mismatches").glob("*.json"))
    assert len(evidence) == 1


def test_mismatched_prior_smoke_is_preserved(tmp_path: Path) -> None:
    manifest = A.build_identity_manifest(A.load_source_evidence())
    rows = _synthetic_rows(manifest, 6)
    state = manifest["states"][0]
    receipt = {
        "schema": "go2_counterfactual_fidelity_stage_a_smoke_receipt_v1_2",
        "status": A.STATUS,
        "pass": True,
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "branch_identity_digests": [row["branch_identity_digest"] for row in rows],
        "branch_row_digests": [row["branch_row_digest"] for row in rows],
        "stage_a_identity_manifest_digest": "0" * 64,
        "assay_spec_digest": manifest["assay_spec_digest"],
        "raw_frame_identity": _raw_identity(state, rows[0]),
    }
    receipt["smoke_receipt_digest"] = A.canonical_digest(receipt)
    path = tmp_path / "smoke_receipt.json"
    A.atomic_json(path, receipt)
    assert A._load_prior_smoke_receipt(
        path, tmp_path, manifest, state, rows) == {}
    assert not path.exists()
    assert len(list((tmp_path / "invalid_attempts").glob("*.invalid"))) == 1


def test_completed_corpus_receipt_is_byte_stable_on_noop(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = A.build_identity_manifest(A.load_source_evidence())
    rows = _synthetic_rows(manifest, A.EXPECTED_BRANCHES)
    completed = {(row["state_id"], row["candidate_index"]): row for row in rows}
    dummy = tmp_path / "row.json"
    dummy.write_text("{}")
    monkeypatch.setattr(A, "_completed_rows", lambda *_args: completed)
    monkeypatch.setattr(A, "_row_path", lambda *_args: dummy)
    first = A._compile_corpus(manifest, SimpleNamespace(), tmp_path, 1.0)
    first_bytes = (tmp_path / "corpus_receipt.json").read_bytes()
    second = A._compile_corpus(manifest, SimpleNamespace(), tmp_path, 99.0)
    assert first == second
    assert (tmp_path / "corpus_receipt.json").read_bytes() == first_bytes


def test_invalid_prior_ledger_and_receipt_are_preserved_before_rebuild(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = A.build_identity_manifest(A.load_source_evidence())
    rows = _synthetic_rows(manifest, 6)
    completed = {(row["state_id"], row["candidate_index"]): row for row in rows}
    dummy = tmp_path / "row.json"
    dummy.write_text("{}")
    monkeypatch.setattr(A, "_completed_rows", lambda *_args: completed)
    monkeypatch.setattr(A, "_row_path", lambda *_args: dummy)
    A._compile_corpus(manifest, SimpleNamespace(), tmp_path, 1.0)
    (tmp_path / "corpus_receipt.json").write_text("{bad receipt")
    (tmp_path / "branch_rows.jsonl").write_text("{bad ledger\n")

    rebuilt = A._compile_corpus(manifest, SimpleNamespace(), tmp_path, 2.0)
    assert rebuilt["attempted_branch_count"] == 6
    preserved = list((tmp_path / "invalid_attempts").glob("*.invalid"))
    assert any(path.name.startswith("corpus_receipt.json") for path in preserved)
    assert any(path.name.startswith("branch_rows.jsonl") for path in preserved)


@pytest.mark.parametrize("durable_count", [7, 24, A.EXPECTED_BRANCHES])
def test_progress_ahead_of_smoke_ledger_reconciles_and_gate_passes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        durable_count: int) -> None:
    manifest = A.build_identity_manifest(A.load_source_evidence())
    rows = _synthetic_rows(manifest, durable_count)
    state = {"rows": rows[:6]}
    completed = lambda values: {
        (row["state_id"], row["candidate_index"]): row for row in values}
    dummy = tmp_path / "row.json"
    dummy.write_text("{}")
    monkeypatch.setattr(A, "_completed_rows", lambda *_args: completed(state["rows"]))
    monkeypatch.setattr(A, "_row_path", lambda *_args: dummy)
    A._compile_corpus(manifest, SimpleNamespace(), tmp_path, 1.0)
    _write_smoke_gate(tmp_path, manifest, rows)

    state["rows"] = rows
    receipt = A._compile_corpus(manifest, SimpleNamespace(), tmp_path, 2.0)
    assert receipt["attempted_branch_count"] == durable_count
    A._validate_full_run_smoke_gate(manifest, tmp_path, completed(rows))


def test_valid_ledger_restores_missing_row_shard_without_branch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = A.build_identity_manifest(A.load_source_evidence())
    rows = _synthetic_rows(manifest, 6)
    _write_corpus(tmp_path, manifest, rows)
    monkeypatch.setattr(A, "_validate_branch_row", lambda *_args: None)
    for row, state in zip(rows[:5], [manifest["states"][0]] * 5):
        identity = A._identity_for(state, row["candidate_index"])
        A.atomic_json(A._row_path(tmp_path, identity), row)
    assert A._recover_row_records_from_ledger(
        manifest, SimpleNamespace(), tmp_path) == 1
    missing_identity = A._identity_for(manifest["states"][0], rows[5]["candidate_index"])
    restored = json.loads(A._row_path(tmp_path, missing_identity).read_text())
    assert restored == rows[5]


def test_encoder_complete_corpus_with_partial_index_fails_closed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "load_inputs", lambda *, smoke: pytest.fail(
        "partial index must not expand a --smoke invocation"))
    with pytest.raises(RuntimeError, match="requires full encoding"):
        E._resolve_encoding_scope(
            smoke=True, manifest={"id": 1}, receipt={"complete": True},
            states=[{"state_id": "s"}], rows=[{"candidate": "c"}],
            prior_index={"complete": False})


def test_encoder_complete_index_expands_postcomplete_smoke_scope(
        monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {"id": 1}
    receipt = {"complete": True}
    full_states = [{"state_id": str(index)} for index in range(20)]
    full_rows = [{"candidate": str(index)} for index in range(240)]
    monkeypatch.setattr(
        E, "load_inputs",
        lambda *, smoke: (manifest, receipt, full_states, full_rows),
    )
    resolved = E._resolve_encoding_scope(
        smoke=True, manifest=manifest, receipt=receipt,
        states=full_states[:1], rows=full_rows[:6],
        prior_index={"complete": True})
    assert resolved[2:] == (full_states, full_rows, True)


def test_midprogress_smoke_guards_leave_receipts_and_index_unchanged(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = A.build_identity_manifest(A.load_source_evidence())
    rows = _synthetic_rows(manifest, 7)
    receipt = _write_corpus(tmp_path, manifest, rows)
    _write_smoke_gate(tmp_path, manifest, rows)
    watched = [
        tmp_path / "corpus_receipt.json",
        tmp_path / "smoke_receipt.json",
        tmp_path / "smoke_encoding_receipt.json",
        tmp_path / "latents_index.json",
    ]
    before = {path.name: path.read_bytes() for path in watched}
    completed = {(row["state_id"], row["candidate_index"]): row for row in rows}
    monkeypatch.setattr(A, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(A, "_completed_rows", lambda *_args: completed)
    monkeypatch.setattr(
        A, "_recover_row_records_from_ledger",
        lambda *_args: pytest.fail("mid-progress smoke entered recovery"),
    )
    monkeypatch.setattr(
        A.V1, "_load_shared",
        lambda *_args: pytest.fail("mid-progress smoke loaded runtime"),
    )
    with pytest.raises(RuntimeError, match="resume --stage branches"):
        A.stage_branches(
            manifest, SimpleNamespace(), smoke=True, state_offset=0, state_limit=1)
    with pytest.raises(RuntimeError, match="resume full branches"):
        E._assert_smoke_encoding_scope(True, receipt)
    assert {path.name: path.read_bytes() for path in watched} == before
