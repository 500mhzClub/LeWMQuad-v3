from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v2_contract as C
from lewm.safety import physical_graph_edge_handoff_qualification_v2_metrics as M
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import run_physical_graph_edge_handoff_qualification_v2 as R


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(R.canonical_bytes(value))


def _snapshot_value() -> dict:
    numeric = {
        field: np.zeros(
            shape,
            dtype=(np.float32 if field == "previous_applied_command" else np.float64),
        )
        for field, shape in V1.SNAPSHOT_NUMERIC_FIELDS.items()
    }
    numeric["camera_world_transform"] = np.eye(4, dtype=np.float64)
    digest = V1.canonical_array_sha256
    return {
        "payload_bytes": b"serialized-fixture",
        **numeric,
        "serialized_solver_state_sha256": "1" * 64,
        "serialized_controller_state_sha256": "2" * 64,
        "serialized_rng_state_sha256": "3" * 64,
        "policy_last_action_sha256": digest(numeric["policy_last_action"]),
        "capture_timestamp_s": 0.0,
        "controller_observation_sha256": digest(numeric["controller_observation"]),
        "previous_policy_action_sha256": digest(numeric["previous_policy_action"]),
        "torch_cpu_rng_state_sha256": "4" * 64,
        "torch_device_rng_state_sha256s": [],
        "torch_device_count": 0,
        "previous_applied_command_sha256": digest(numeric["previous_applied_command"]),
        "command_history_sha256": digest(numeric["command_history"]),
        "control_history_sha256": digest(numeric["control_history"]),
        "low_level_policy_state_sha256": digest(numeric["low_level_policy_state"]),
        "solver_field_inventory": ["fixture.solver"],
        "controller_field_inventory": ["fixture.controller"],
        "rng_field_inventory": ["fixture.rng"],
    }


def test_v2_identity_and_scientific_projection_are_exact() -> None:
    assert R.EXPERIMENT_ID == "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2"
    assert len(R.ALL_OUTPUT_LEAVES) == 26
    assert set(C.REPRODUCTION_MISMATCH_LEAVES) == {
        "contract.json",
        "v1_custody_and_nonreuse.json",
        "scientific_invariance_receipt.json",
        "v1_v2_first_eight_reproduction.json",
    }
    assert C.scientific_invariance_projection(C.build_contract()) == C.V1_SCIENTIFIC_PROJECTION
    assert C.build_candidate_specs() == C.V1.build_candidate_specs()


def test_preregistration_discloses_bounded_authority_alignments() -> None:
    text = R._preregistration_text()
    assert C.PORT_HEADING_ALIGNMENT_DISPOSITION in text
    assert C.CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION in text
    assert "changes only `directed_port_world[2]`" in text
    assert (
        "changes only `port_progress_m`, `lateral_error_m`, and dependent "
        "`positive_port_progress`"
    ) in text
    assert "canonical actual-teacher-crossing port" in text
    assert "change no frozen gate, formula, threshold" in text
    assert "tuning choice, model, checkpoint, candidate bank" in text


def test_exact_persisted_array_digest_has_no_header_or_cast() -> None:
    array = np.asarray([[0.125, -0.25], [0.5, -0.75]], dtype=np.float64)
    expected = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    assert R.persisted_array_sha256(array) == expected
    assert R.persisted_array_sha256(array) == M.persisted_array_bytes_sha256(array)
    assert R.persisted_array_sha256(array.astype(np.float32)) != expected
    view = np.arange(12, dtype=np.int16).reshape(3, 4)[:, ::2]
    assert not view.flags.c_contiguous
    assert R.persisted_array_sha256(view) == hashlib.sha256(
        np.ascontiguousarray(view).tobytes(order="C")
    ).hexdigest()


def test_production_writer_reopens_and_rejects_manifest_tamper(tmp_path: Path) -> None:
    shard = tmp_path / "pool-000"
    arrays = {
        "float64": np.arange(6, dtype=np.float64).reshape(2, 3),
        "uint8": np.asarray([0, 1, 1], dtype=np.uint8),
    }
    R._write_material_shard_impl(
        shard,
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.teacher_pool_shard.v1",
            "experiment_id": C.V1_EXPERIMENT_ID,
        },
        arrays,
        root=tmp_path,
    )
    metadata = R._ordinary_json(shard / "metadata.json")
    assert metadata["schema"].startswith("physical_graph_edge_handoff_qualification_v2")
    reopened = R.validate_persisted_array_payload(shard, metadata, root=tmp_path)
    assert reopened["float64"].dtype.str == "<f8"
    assert np.array_equal(reopened["uint8"], arrays["uint8"])
    with zipfile.ZipFile(shard / "payload.npz") as archive:
        assert archive.comment.decode("utf-8") == C.NPZ_ARCHIVE_COMMENT
    tampered = copy.deepcopy(metadata["persisted_array_evidence"])
    tampered["arrays"][0]["dtype_str"] = "<f4"
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError):
        M.validate_persisted_array_evidence(tampered, reopened_arrays=reopened)


def test_snapshot_fix_changes_only_hash_bound_to_persisted_float64() -> None:
    value = _snapshot_value()
    before = copy.deepcopy({key: value[key] for key in V1.SNAPSHOT_METADATA_FIELDS})
    arrays, metadata = R._normalise_snapshot(value)
    expected = R.persisted_array_sha256(arrays["snapshot__previous_applied_command"])
    assert expected == C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"][
        "first_eight_expected_raw_bytes_sha256"
    ]
    assert arrays["snapshot__previous_applied_command"].dtype.str == "<f8"
    assert arrays["snapshot__previous_applied_command"].shape == (3,)
    assert before["previous_applied_command_sha256"] != expected
    assert metadata["previous_applied_command_sha256"] == expected
    for field in set(V1.SNAPSHOT_METADATA_FIELDS) - {
        "previous_applied_command_sha256"
    }:
        if field == "snapshot_payload_sha256":
            continue
        assert metadata[field] == before[field]


@pytest.mark.parametrize(
    ("pool_index", "inherited_yaw"),
    ((1, -0.16129238253231737), (5, -0.14203543522690865)),
)
def test_inherited_first_eight_port_heading_is_aligned_to_frozen_normal(
    pool_index: int, inherited_yaw: float,
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _fake_graph,
    )

    spec = copy.deepcopy(C.build_prospective_pool_specs()[pool_index])
    assert spec["state_id"] == f"pgehq-v1-state-00-0{pool_index}"
    edge = spec["geometry"]["selected_directed_edge"]
    opening = np.asarray(edge["opening_segment_world"], dtype=np.float64)
    normal = np.asarray(edge["opening_normal_world"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    midpoint = opening.mean(axis=0)
    xy = np.stack((midpoint - 0.01 * normal, midpoint + 0.01 * normal))
    pose = np.zeros((2, 7), dtype=np.float64)
    pose[:, :2] = xy
    pose[:, 2] = 0.35
    pose[:, 5] = math.sin(inherited_yaw / 2.0)
    pose[:, 6] = math.cos(inherited_yaw / 2.0)
    teacher = {
        "teacher_trace_id": f"teacher-{pool_index}",
        "first_crossing_sample_index": 1,
        "crossing_segment_fraction": 0.5,
        "crossing_velocity_world_xy": normal.tolist(),
        "crossing_velocity_heading_world_rad": float(math.atan2(normal[1], normal[0])),
    }
    original = R._ORIGINAL_V1_EDGE_PORT_RECORD(
        spec, teacher, {"base_pose_world": pose}, _fake_graph(spec)
    )
    corrected = R._edge_port_record_authority_alignment(
        spec, teacher, {"base_pose_world": pose}, _fake_graph(spec)
    )
    expected_heading = math.atan2(float(normal[1]), float(normal[0]))
    assert original["directed_port_world"][:2] == corrected["directed_port_world"][:2]
    assert original["directed_port_world"][2] == pytest.approx(inherited_yaw)
    assert corrected["directed_port_world"][2] == pytest.approx(expected_heading)
    assert corrected["teacher_crossing_velocity_world_xy"] == teacher[
        "crossing_velocity_world_xy"
    ]
    assert corrected["teacher_crossing_velocity_heading_world_rad"] == teacher[
        "crossing_velocity_heading_world_rad"
    ]
    corrected_with_inherited_heading = copy.deepcopy(corrected)
    corrected_with_inherited_heading["directed_port_world"][2] = original[
        "directed_port_world"
    ][2]
    assert corrected_with_inherited_heading == original


@pytest.mark.parametrize("pool_index", (1, 5))
def test_inherited_first_eight_candidate_metrics_use_actual_teacher_port(
    pool_index: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _fake_physics_trace,
    )

    metadata = V1.load_json(
        R.V1_MATERIAL_ROOT
        / "qualification"
        / f"pool-{pool_index:03d}"
        / "metadata.json"
    )
    spec = copy.deepcopy(metadata["candidate_spec"])
    assert spec["state_id"] == f"pgehq-v1-state-00-0{pool_index}"
    point = metadata["teacher"]["crossing"]["point_world"]
    normal = spec["geometry"]["selected_directed_edge"]["opening_normal_world"]
    canonical_port = [
        float(point[0]),
        float(point[1]),
        math.atan2(float(normal[1]), float(normal[0])),
    ]
    edge_index = V1.attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v2.edge_port_index.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "records": [
                {
                    "state_id": spec["state_id"],
                    "directed_port_world": canonical_port,
                }
            ],
        }
    )
    R._atomic_json(tmp_path / "edge_port_index.json", edge_index)
    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path)
    trace = _fake_physics_trace(spec, 0, correct=False)
    reset_pose = trace["base_pose_world"][0]
    inherited = R._ORIGINAL_V1_DERIVE_CANDIDATE_OUTCOME(
        spec, trace, reset_pose, 0
    )
    aligned = R._derive_candidate_outcome_authority_alignment(
        spec, trace, reset_pose, 0
    )
    expected = M.derive_candidate_port_metrics(
        reset_pose, trace["base_pose_world"], canonical_port
    )
    assert {field: aligned[field] for field in expected} == expected
    assert aligned["port_progress_m"] != inherited["port_progress_m"]
    assert aligned["lateral_error_m"] != inherited["lateral_error_m"]
    restored = copy.deepcopy(aligned)
    for field in expected:
        restored[field] = inherited[field]
    assert restored == inherited


def test_scientific_invariance_receipt_is_pure_exact_and_no_self_digest() -> None:
    receipt = R.build_scientific_invariance_receipt()
    assert "content_digest" not in receipt
    assert receipt["pass"] is True
    assert len(receipt["regression_results"]) == 10
    assert receipt["inherited_implementation_alignment_disposition"] == (
        C.PORT_HEADING_ALIGNMENT_DISPOSITION
    )
    assert receipt["port_heading_alignment_authority_content_digest"] == (
        C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"]
    )
    assert receipt["candidate_port_metric_alignment_disposition"] == (
        C.CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION
    )
    assert receipt["candidate_port_metric_alignment_authority_content_digest"] == (
        C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY[
            "content_digest"
        ]
    )
    assert M.validate_scientific_invariance_receipt(receipt) == receipt


def test_v1_namespace_patch_is_scoped_and_maps_only_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    original_contract = V1.CONTRACT
    original_root = V1.OUTPUT_ROOT
    original_writer = V1._write_material_shard
    original_edge_port = V1._edge_port_record
    original_candidate_outcome = V1.derive_candidate_outcome
    with R._v2_namespace():
        assert V1.CONTRACT is C
        assert V1.OUTPUT_ROOT == R.OUTPUT_ROOT
        assert V1._write_material_shard is R._write_material_shard
        assert V1._edge_port_record is R._edge_port_record_authority_alignment
        assert (
            V1.derive_candidate_outcome
            is R._derive_candidate_outcome_authority_alignment
        )
    assert V1.CONTRACT is original_contract
    assert V1.OUTPUT_ROOT == original_root
    assert V1._write_material_shard is original_writer
    assert V1._edge_port_record is original_edge_port
    assert V1.derive_candidate_outcome is original_candidate_outcome
    value = V1.attach_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v1.fixture.v1",
            "experiment_id": C.V1_EXPERIMENT_ID,
            "state_id": "pgehq-v1-state-000",
        }
    )
    mapped = R._v2_identity(value)
    C.validate_content_digest(mapped)
    assert mapped["schema"].startswith("physical_graph_edge_handoff_qualification_v2")
    assert mapped["experiment_id"] == C.EXPERIMENT_ID
    assert mapped["state_id"] == "pgehq-v1-state-000"


def test_publication_keeps_scientific_predecessor_not_v1_source_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = C.V1_SOURCE_FREEZE_COMMIT
    observed: list[str] = []
    monkeypatch.setattr(V1, "PARENT_COMMIT", source_parent)
    monkeypatch.setattr(
        R,
        "_ORIGINAL_V1_WRITE_PUBLICATION",
        lambda _metrics, _receipt: observed.append(V1.PARENT_COMMIT) or {"ok": True},
    )
    assert R._write_publication({}, {}) == {"ok": True}
    assert observed == [C.build_contract()["v2_context_binding"]["result_commit"]]
    assert V1.PARENT_COMMIT == source_parent


def test_initialize_validates_custody_before_creating_v2_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "official"
    material = tmp_path / "material"
    v1_official = tmp_path / "v1-official"
    v1_material = tmp_path / "v1-material"
    v1_official.mkdir()
    v1_material.mkdir()
    (v1_official / "contract.json").write_text("v1", encoding="utf-8")
    (v1_material / "inventory.txt").write_text("v1-material", encoding="utf-8")
    receipt_path = tmp_path / "v1-receipt.json"
    receipt_path.write_bytes(b"{}\n")
    binding = {
        "path": str(receipt_path),
        "bytes": receipt_path.stat().st_size,
        "sha256": R.sha256_file(receipt_path),
    }
    observed: list[str] = []

    monkeypatch.setattr(R, "OUTPUT_ROOT", output)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "V1_OFFICIAL_ROOT", v1_official)
    monkeypatch.setattr(R, "V1_MATERIAL_ROOT", v1_material)
    monkeypatch.setattr(R, "V1_CUSTODY_RECEIPT", receipt_path)
    monkeypatch.setattr(C, "V1_CUSTODY_RECEIPT_BINDING", binding)
    monkeypatch.setattr(
        R,
        "validate_v1_custody_before_creation",
        lambda **_kwargs: observed.append("custody") or {"external": "validated"},
    )
    monkeypatch.setattr(R, "build_scientific_invariance_receipt", lambda: {"pass": True})
    monkeypatch.setattr(R, "require_runtime_source_freeze", lambda: "a" * 40)
    monkeypatch.setattr(
        R,
        "_custody_nonreuse_receipt",
        lambda _freeze, _external: {"pass": True},
    )

    def fake_delegate(_name: str, **_kwargs):
        assert observed == ["custody"]
        assert not output.exists() and not material.exists()
        output.mkdir()
        material.mkdir()
        return {"initialized": True}

    monkeypatch.setattr(R, "_delegate", fake_delegate)
    evaluator = SimpleNamespace(
        validate_existing_v1_custody_receipt=lambda *_a, **_k: {
            "external": "validated"
        }
    )
    assert R.initialize_stage(fake_runtime=True, evaluator_module=evaluator) == {
        "initialized": True
    }
    assert (output / "v1_custody_and_nonreuse.json").exists()
    assert (output / "scientific_invariance_receipt.json").exists()


def test_delegated_initialize_and_first_pool_use_fresh_v2_schema_and_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakePhysicalBackend,
    )
    from scripts import evaluate_physical_graph_edge_handoff_qualification_v2 as E

    output = tmp_path / "physical_graph_edge_handoff_qualification_v2"
    material = tmp_path / "physical_graph_edge_handoff_qualification_v2_material"
    monkeypatch.setattr(R, "OUTPUT_ROOT", output)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    source_commit = "a" * 40
    monkeypatch.setattr(R, "require_runtime_source_freeze", lambda: source_commit)
    monkeypatch.setattr(V1, "_runtime_contract", lambda freeze: C.build_runtime_contract(freeze))
    monkeypatch.setattr(
        V1,
        "_scene_exclusion_audit",
        lambda _specs: {
            "authority_digest": hashlib.sha256(
                C.canonical_json_bytes(C.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
            ).hexdigest(),
            "checked_before_simulator_creation": True,
            "scene_overlap_count": 0,
            "scene_hash_overlap_count": 0,
            "state_or_episode_overlap_count": 0,
            "seed_overlap_count": 0,
            "path_or_geometry_overlap_count": 0,
            "structured_path_overlap_count": 0,
            "all_zero": True,
        },
    )
    pool = R.initialize_stage(fake_runtime=True, evaluator_module=E)
    assert pool["experiment_id"] == C.EXPERIMENT_ID
    assert pool["schema"].startswith("physical_graph_edge_handoff_qualification_v2")
    metadata = R.qualify_pool_state_stage(
        0, backend=_FakePhysicalBackend(), fake_runtime=True
    )
    assert metadata["experiment_id"] == C.EXPERIMENT_ID
    assert metadata["schema"].startswith("physical_graph_edge_handoff_qualification_v2")
    assert metadata["persisted_array_evidence"]["save_reopen_validation_passed"] is True
    assert metadata["snapshot"]["previous_applied_command_sha256"] == R.persisted_array_sha256(
        R._load_material_shard(material / "qualification" / "pool-000")[1][
            "snapshot__previous_applied_command"
        ]
    )
    previous_row = next(
        row
        for row in metadata["persisted_array_evidence"]["arrays"]
        if row["member"] == "snapshot__previous_applied_command"
    )
    assert previous_row["dtype_str"] == "<f8"
    assert previous_row["shape"] == [3]


def _qualification_arrays(index: int, *, mutate: bool = False) -> dict[str, np.ndarray]:
    pose = np.zeros((2, 7), dtype=np.float64)
    pose[:, 6] = 1.0
    if mutate:
        pose[-1, 0] = 0.125
    return {
        "snapshot_payload_bytes": np.frombuffer(f"snapshot-{index}".encode(), dtype=np.uint8).copy(),
        "snapshot__previous_applied_command": np.zeros(3, dtype=np.float64),
        "teacher__base_pose_world": pose,
        "teacher__requested_command": np.zeros((2, 3), dtype=np.float64),
        "teacher__physics_contact": np.asarray([0, int(index == 2)], dtype=np.uint8),
    }


def _qualification_metadata(index: int, arrays: dict[str, np.ndarray], *, v2: bool) -> dict:
    spec = C.build_prospective_pool_specs()[index]
    payload = arrays["snapshot_payload_bytes"].tobytes()
    corrected = R.persisted_array_sha256(arrays["snapshot__previous_applied_command"])
    value = {
        "schema": "physical_graph_edge_handoff_qualification_v1.teacher_pool_shard.v1",
        "experiment_id": C.V1_EXPERIMENT_ID,
        "pool_index": index,
        "candidate_spec": spec,
        "initial_decision_state_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot": {
            "snapshot_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "previous_applied_command_sha256": (
                corrected
                if v2
                else C.V1_CUSTODY_EXPECTED_PROJECTION["defect_evidence"]["metadata_sha256"]
            ),
        },
        "teacher": {"contact_free": index != 2},
        "qualified": index != 2,
        "rejection_reason": "QUALIFIED" if index != 2 else "TEACHER_PHYSICS_CONTACT",
        "stage_runtime": {"source": "identical_physical_runtime"},
    }
    return value


def _materialize_first_eight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mismatch_index: int | None = None,
) -> tuple[Path, Path, Path, dict]:
    v1_root = tmp_path / "v1-material"
    v2_root = tmp_path / "v2-material"
    output = tmp_path / "v2-official"
    for root in (v1_root, v2_root):
        (root / "qualification").mkdir(parents=True)
    output.mkdir()
    for name in (
        "contract.json",
        "v1_custody_and_nonreuse.json",
        "scientific_invariance_receipt.json",
    ):
        _json(output / name, {"fixture": name})
    monkeypatch.setattr(R, "V1_MATERIAL_ROOT", v1_root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", v2_root)
    monkeypatch.setattr(R, "OUTPUT_ROOT", output)
    monkeypatch.setattr(V1, "MATERIAL_ROOT", v1_root)
    for index in range(8):
        v1_arrays = _qualification_arrays(index)
        v2_arrays = _qualification_arrays(index, mutate=index == mismatch_index)
        V1._write_material_shard(
            v1_root / "qualification" / f"pool-{index:03d}",
            _qualification_metadata(index, v1_arrays, v2=False),
            v1_arrays,
        )
        R._write_material_shard_impl(
            v2_root / "qualification" / f"pool-{index:03d}",
            _qualification_metadata(index, v2_arrays, v2=True),
            v2_arrays,
            root=v2_root,
        )
    receipt = tmp_path / "v1-custody.json"
    receipt.write_bytes(b"{}\n")
    binding = {
        "path": str(receipt),
        "bytes": receipt.stat().st_size,
        "sha256": R.sha256_file(receipt),
    }
    monkeypatch.setattr(R, "V1_CUSTODY_RECEIPT", receipt)
    monkeypatch.setattr(C, "V1_CUSTODY_RECEIPT_BINDING", binding)
    evaluator = SimpleNamespace(validate_existing_v1_custody_receipt=lambda *_a, **_k: {})
    return v1_root, v2_root, output, {"binding": binding, "evaluator": evaluator}


def test_first_eight_pass_authorizes_pool_eight_and_preserves_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1_root, _v2_root, _output, context = _materialize_first_eight(tmp_path, monkeypatch)
    before = {
        str(path.relative_to(v1_root)): (path.stat().st_ino, R.sha256_file(path))
        for path in v1_root.rglob("*")
        if path.is_file()
    }
    receipt = R.compare_v1_first_eight_stage(evaluator_module=context["evaluator"])
    assert receipt["pass"] is True
    assert M.authorizes_full_v2_collection(receipt) is True
    assert R._require_reproduction_pass() == receipt
    after = {
        str(path.relative_to(v1_root)): (path.stat().st_ino, R.sha256_file(path))
        for path in v1_root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_first_eight_mismatch_is_four_leaf_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _v1, _v2, output, context = _materialize_first_eight(
        tmp_path, monkeypatch, mismatch_index=4
    )
    receipt = R.compare_v1_first_eight_stage(evaluator_module=context["evaluator"])
    assert receipt["status"] == C.REPRODUCTION_MISMATCH_DISPOSITION
    assert receipt["pass"] is False
    assert sorted(path.name for path in output.iterdir()) == sorted(
        C.REPRODUCTION_MISMATCH_LEAVES
    )
    with pytest.raises(R.ExperimentError, match=C.REPRODUCTION_MISMATCH_DISPOSITION):
        R._require_reproduction_pass()


def test_first_eight_missing_member_is_persisted_four_leaf_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _v1, v2, output, context = _materialize_first_eight(tmp_path, monkeypatch)
    directory = v2 / "qualification" / "pool-003"
    metadata, arrays = R._load_material_shard(directory)
    for field in ("content_digest", "payload", "persisted_array_evidence"):
        metadata.pop(field, None)
    arrays.pop("teacher__physics_contact")
    shutil.rmtree(directory)
    R._write_material_shard_impl(directory, metadata, arrays, root=v2)
    receipt = R.compare_v1_first_eight_stage(evaluator_module=context["evaluator"])
    assert receipt["status"] == C.REPRODUCTION_MISMATCH_DISPOSITION
    assert receipt["rows"][3]["teacher_trace_member_inventory_equal"] is False
    assert receipt["rows"][3]["contact_sequence_equal"] is False
    assert sorted(path.name for path in output.iterdir()) == sorted(
        C.REPRODUCTION_MISMATCH_LEAVES
    )


def test_pool_eight_is_blocked_before_reproduction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        R,
        "_require_reproduction_pass",
        lambda: (_ for _ in ()).throw(R.ExperimentError("blocked")),
    )
    with pytest.raises(R.ExperimentError, match="blocked"):
        R.qualify_pool_state_stage(8, backend=object(), fake_runtime=True)


def test_cli_exposes_first_eight_gate() -> None:
    parser = R.build_parser()
    assert parser.parse_args(["compare-v1-first-eight"]).stage == "compare-v1-first-eight"
    assert parser.parse_args(["qualify-pool-state", "--pool-index", "0"]).pool_index == 0


@pytest.mark.skipif(
    os.environ.get("RUN_FULL_PGEHQ_V2_FAKE_E2E") != "1",
    reason="full 256-state/960-trace synthetic integration is explicitly requested",
)
def test_complete_fake_v2_wrapper_flow_reaches_independent_reducer_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise all full-cardinality V2 handoffs without opening real outcomes.

    The only V1 payload reads below are a test-only reconstruction fixture for
    the registered first-eight comparison.  No production runner helper copies
    V1 arrays into V2, and every resulting V2 NPZ is independently written with
    its frozen V2 container provenance comment.
    """

    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakeEncoder,
        _FakePhysicalBackend,
        _FakeRanker,
    )
    from scripts import evaluate_physical_graph_edge_handoff_qualification_v2 as E

    root = tmp_path / "physical_graph_edge_handoff_qualification_v2"
    material = tmp_path / "physical_graph_edge_handoff_qualification_v2_material"
    final_receipt = tmp_path / "physical_graph_edge_handoff_qualification_v2_regeneration_receipt.json"
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", final_receipt)
    source_commit = "a" * 40
    monkeypatch.setattr(R, "require_runtime_source_freeze", lambda: source_commit)
    monkeypatch.setattr(V1, "_runtime_contract", lambda freeze: C.build_runtime_contract(freeze))
    monkeypatch.setattr(
        V1,
        "_scene_exclusion_audit",
        lambda _specs: {
            "authority_digest": hashlib.sha256(
                C.canonical_json_bytes(C.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
            ).hexdigest(),
            "checked_before_simulator_creation": True,
            "scene_overlap_count": 0,
            "scene_hash_overlap_count": 0,
            "state_or_episode_overlap_count": 0,
            "seed_overlap_count": 0,
            "path_or_geometry_overlap_count": 0,
            "structured_path_overlap_count": 0,
            "all_zero": True,
        },
    )

    # Test-only production-shaped physical runtime.  The production runner has
    # no fake-to-real adapter; this monkeypatch exists solely to let the
    # independent reducer exercise all physical joins with a fake backend.
    physical_authority = C.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    physical_core = {
        "stage_id": physical_authority["stage_id"],
        "python_executable": physical_authority["real_python_executable"],
        "python_version": physical_authority["python_version"],
        "torch_version": physical_authority["torch_version"],
        "torch_hip_version": physical_authority["torch_hip_version"],
        "genesis_version": physical_authority["genesis_version"],
        "quadrants_version": physical_authority["quadrants_version"],
        "visible_device_count": physical_authority["visible_device_count"],
        "device": physical_authority["device"],
        "backend": physical_authority["backend"],
        "deterministic_environment": dict(
            C.DIRECT_RUNTIME_POLICY["required_environment_before_simulator_creation"]
        ),
        "fake_runtime": False,
    }
    M.validate_physical_runtime_environment(
        {
            **physical_core,
            "runtime_core_sha256": M.runtime_environment_sha256(physical_core),
            "qualification_runtime_sha256s": [M.runtime_environment_sha256(physical_core)] * 256,
            "selected_snapshot_runtime_sha256s": [M.runtime_environment_sha256(physical_core)] * 64,
        }
    )
    original_require_runtime = V1.require_stage_runtime

    def test_runtime(kind: str, *, fake: bool = False, visual_role: str | None = None):
        if kind == "physical":
            return copy.deepcopy(physical_core)
        return original_require_runtime(kind, fake=fake, visual_role=visual_role)

    monkeypatch.setattr(V1, "require_stage_runtime", test_runtime)
    pool_index_by_candidate = {
        str(spec["candidate_spec_id"]): index
        for index, spec in enumerate(C.build_prospective_pool_specs())
    }

    class PhysicalBackend(_FakePhysicalBackend):
        runtime = {
            **_FakePhysicalBackend.runtime,
            "backend": physical_authority["backend"],
            "policy_device": physical_authority["device"],
        }

        def reset_fixture(self, spec: dict, payload: bytes) -> dict:
            value = super().reset_fixture(spec, payload)
            pool_index = pool_index_by_candidate[str(spec["candidate_spec_id"])]
            _metadata, qualification_arrays = R._load_material_shard(
                material / "qualification" / f"pool-{pool_index:03d}"
            )
            rgb_sha256 = V1.canonical_array_sha256(qualification_arrays["rgb"])
            for trial in value["reset_trials"]:
                trial["metadata"]["current_rgb_sha256"] = rgb_sha256
            return value

    backend = PhysicalBackend()
    pool = R.initialize_stage(fake_runtime=True, evaluator_module=E)
    assert len(pool["specs"]) == 256

    # Produce all first-eight shards through the actual V2 stage/writer, then
    # replace only their test-double logical evidence with immutable V1 logical
    # arrays so the registered exact-reproduction gate itself can be exercised.
    for index in range(8):
        fake_metadata = R.qualify_pool_state_stage(
            index, backend=backend, fake_runtime=True
        )
        v1_metadata, v1_arrays = V1._load_material_shard(
            R.V1_MATERIAL_ROOT / "qualification" / f"pool-{index:03d}"
        )
        metadata = copy.deepcopy(v1_metadata)
        metadata.pop("content_digest", None)
        metadata.pop("payload", None)
        metadata["stage_runtime"] = copy.deepcopy(fake_metadata["stage_runtime"])
        metadata["snapshot"]["previous_applied_command_sha256"] = R.persisted_array_sha256(
            v1_arrays["snapshot__previous_applied_command"]
        )
        directory = material / "qualification" / f"pool-{index:03d}"
        shutil.rmtree(directory)
        R._write_material_shard_impl(
            directory, metadata, v1_arrays, root=material
        )
    reproduction = R.compare_v1_first_eight_stage(evaluator_module=E)
    assert reproduction["pass"] is True

    for index in range(8, 256):
        R.qualify_pool_state_stage(index, backend=backend, fake_runtime=True)
    selection = R.select_teacher_pool_stage(fake_runtime=True)
    selected = [str(row["state_id"]) for row in selection["selected_specs"]]
    assert len(selected) == 64
    for state_id in selected:
        R.capture_selected_state_stage(
            state_id, backend=backend, fake_runtime=True
        )
    panel = R.freeze_panel_stage(fake_runtime=True)
    assert len(panel["states"]) == 64
    latent = R.encode_canonical_pixels_stage(
        encoder=_FakeEncoder(), fake_runtime=True
    )
    assert latent["records"]
    development = [row for row in panel["states"] if row["role"] == "DEVELOPMENT"]
    heldout = [row for row in panel["states"] if row["role"] == "DEVELOPMENT_HELDOUT"]
    assert (len(development), len(heldout)) == (48, 16)
    for state in development:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    target = R.development_target_selection_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    assert target["selection_frozen"] is True
    for state in heldout:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    assert len(
        R.heldout_ranker_scores_stage(ranker=_FakeRanker(), fake_runtime=True)
    ) == 64
    for state in heldout:
        R.repeat_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    assembled = R.assemble_row_evidence_stage(fake_runtime=True)
    assert assembled["candidate_trace_count"] == 960
    del pool, selection, selected, panel, latent, development, heldout, target, backend
    gc.collect()

    external_rows = []
    for index, row in enumerate(C.EXTERNAL_ARTIFACT_BINDINGS):
        path = tmp_path / f"external-{index}.bin"
        path.write_bytes(f"{row['role']}\n".encode())
        external_rows.append(
            {
                "role": row["role"],
                "kind": row["kind"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": R.sha256_file(path),
            }
        )
    monkeypatch.setattr(M, "external_artifact_bindings", lambda _contract: external_rows)
    source_observation = {
        "head_commit": source_commit,
        "parent_commit": C.SOURCE_PARENT_COMMIT,
        "freeze_subject": C.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "worktree_clean": True,
        "tracked_source_count": 15,
        "tracked_sources_sha256": "1" * 64,
        "source_closure_path": str(C.TRACKED_SOURCE_PATHS[4]),
        "source_closure_bytes": 1,
        "source_closure_sha256": "2" * 64,
        "source_closure_row_count": len(C.SOURCE_CLOSURE_PATHS),
        "source_closure_live_bytes_exact": True,
        "metrics_module": M.__name__,
    }

    # Encoder/ranker remain visibly fake until this test-only projection.  The
    # independent evaluator must reject the untouched evidence first.
    R._gated_delegate("recompute_and_persist_metrics_stage", fake_runtime=True)
    rejected = tmp_path / "fake-runtime-rejected.json"
    with pytest.raises(E.RegenerationError, match="marked fake"):
        E.verify_and_emit(
            root,
            rejected,
            metrics_module=M,
            source_freeze_observation=source_observation,
            material_root=material,
        )
    assert not rejected.exists()
    (root / "metrics.json").unlink()

    def visual_environment(role: str) -> dict:
        authority = C.RUNTIME_ENVIRONMENT_AUTHORITY[role]
        return M.validate_visual_runtime_environment(
            {
                field: (
                    False
                    if field == "fake_runtime"
                    else authority["real_python_executable"]
                    if field == "python_executable"
                    else authority[field]
                )
                for field in C.VISUAL_RUNTIME_ENVIRONMENT_FIELDS
            },
            runtime_role=role,
        )

    encoder_environment = visual_environment("encoder")
    ranker_environment = visual_environment("ranker")
    ranker_digest = M.runtime_environment_sha256(ranker_environment)

    def replace_document(path: Path, field: str, value: dict) -> None:
        document = R._ordinary_json(path)
        document.pop("content_digest")
        document[field] = value
        path.unlink()
        R._atomic_json(path, C.attach_content_digest(document))

    replace_document(root / "latent_index.json", "encoder_runtime_environment", encoder_environment)
    replace_document(
        root / "development_target_selection.json",
        "ranker_runtime_environment",
        ranker_environment,
    )
    score_path = root / "heldout_ranker_scores.jsonl"
    score_rows = V1.load_jsonl(score_path)
    for row in score_rows:
        row["ranker_runtime_environment_sha256"] = ranker_digest
    score_path.unlink()
    R._v2_atomic_jsonl(score_path, score_rows)

    class EvaluatorAdapter:
        @staticmethod
        def verify_and_emit(output_root, output, *, metrics_module):
            return E.verify_and_emit(
                output_root,
                output,
                metrics_module=metrics_module,
                source_freeze_observation=source_observation,
                material_root=material,
            )

        @staticmethod
        def validate_existing_regeneration_receipt(output_root, output, *, metrics_module):
            return E.validate_existing_regeneration_receipt(
                output_root,
                output,
                metrics_module=metrics_module,
                source_freeze_observation=source_observation,
                material_root=material,
            )

    result = R.report_stage(fake_runtime=True, evaluator_module=EvaluatorAdapter)
    assert result["models_trained"] == 0
    assert result["predecessor_result_commit"] == C.build_contract()["v2_context_binding"][
        "result_commit"
    ]
    assert final_receipt.is_file()
    assert set(path.name for path in root.iterdir()) == set(C.SUCCESS_OUTPUT_LEAVES)
    assert sum(1 for _ in material.rglob("payload.npz")) == 400
    E.validate_existing_regeneration_receipt(
        root,
        final_receipt,
        metrics_module=M,
        source_freeze_observation=source_observation,
        material_root=material,
    )
    assert "production_shaped_runtime" not in Path(R.__file__).read_text()
