from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.benchmarks import (
    go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1 as C,
)

RUNNER_PATH = C.ROOT / C.RUNNER_PATH
SPEC = importlib.util.spec_from_file_location("_physical_gate_oracle_runner_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def _json(core: dict) -> bytes:
    return C.canonical_bytes(C.content_value(core)) + b"\n"


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(C.canonical_bytes(C.content_value(row)) + b"\n" for row in rows)


def _write(path: Path, raw: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": path.name, "byte_count": len(raw), "file_sha256": hashlib.sha256(raw).hexdigest()}


def _attainable_endpoints() -> list[C.Endpoint]:
    distances = np.repeat(np.arange(6, dtype=np.float64) + 0.5, 2)
    clear0 = np.tile(np.array([False, True]), 6)
    hit0 = np.array([False, True, False, True, False, True])
    raster0 = np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8)
    result = []
    for family in C.FAMILIES:
        for index in range(2):
            result.append(
                C.Endpoint(
                    f"{family}:{index}", family,
                    hit0 if index == 0 else ~hit0,
                    np.full(hit0.shape, 0.1 if index == 0 else 1.1),
                    np.ones(distances.shape, dtype=bool),
                    clear0 if index == 0 else ~clear0,
                    distances,
                    raster0 if index == 0 else (raster0 + 1) % 3,
                )
            )
    return result


def test_registered_wrong_source_mapping_is_sorted_cyclic_and_adversarial() -> None:
    family = C.FAMILIES[0]
    template = _attainable_endpoints()[0]
    rows = [C.Endpoint(name, family, template.pixel_hit, template.pixel_depth,
                       template.ground_valid, template.ground_clear,
                       template.ground_distance, template.raster)
            for name in ("c", "a", "b")]
    assert C.wrong_mapping(rows) == [("a", "b"), ("b", "c"), ("c", "a")]
    with pytest.raises(ValueError):
        C.wrong_mapping(rows[:1])
    with pytest.raises(ValueError):
        C.wrong_mapping([rows[0], rows[0]])
    cross = C.Endpoint("z", C.FAMILIES[1], template.pixel_hit, template.pixel_depth,
                       template.ground_valid, template.ground_clear,
                       template.ground_distance, template.raster)
    with pytest.raises(ValueError):
        C.wrong_mapping([rows[0], cross])


def test_attainable_fixture_passes_exact_nine_scope_189_margin_gate() -> None:
    endpoints = _attainable_endpoints()
    result = C.evaluate(endpoints)
    assert result["all_nine_physical_pass"] is True
    assert result["physical_pass_count"] == 9
    assert list(result["physical_pass_by_scope"]) == list(C.SCOPES)
    assert all(result["physical_pass_by_scope"].values())
    margins = result["raw_margin_vector"]
    assert len(margins) == 189
    assert [row["index"] for row in margins] == list(range(189))
    assert [margins[index * 21]["scope"] for index in range(9)] == list(C.SCOPES)
    expected_names = (
        *C.CHECKPOINT_EVALUATOR.PHYSICAL_LOWER_THRESHOLDS,
        *C.CHECKPOINT_EVALUATOR.PHYSICAL_UPPER_THRESHOLDS,
        *(f"distance_group_balanced_accuracy[{index}]" for index in range(6)),
        "present_class_recall.free", "present_class_recall.occupied", "present_class_recall.unknown",
    )
    assert tuple(row["name"] for row in margins[:21]) == expected_names
    assert all(row["passes"] for row in margins)
    mapping = [
        {"family": family, "target": f"{family}:0", "source": f"{family}:1"}
        for family in C.FAMILIES
    ] + [
        {"family": family, "target": f"{family}:1", "source": f"{family}:0"}
        for family in C.FAMILIES
    ]
    mapping.sort(key=lambda row: (C.FAMILIES.index(row["family"]), row["target"]))
    assert result["wrong_source_fixed_point_count"] == 0
    assert result["wrong_source_mapping_count"] == 16
    assert result["wrong_source_mapping_sha256"] == C.canonical_sha(mapping)


def _synthetic_raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    raw_root = "raw"
    layout = (
        ("camera_origin_body_m.f4", "<f4", (3,)), ("camera_basis_body_fru.f4", "<f4", (3, 3)),
        ("ground_plane_z_body_m.f4", "<f4", ()), ("ground_support_in_frustum.u1", "|u1", (12,)),
        ("ground_support_clear_to_target.u1", "|u1", (12,)), ("pixel_hit_mask.u1", "|u1", (6,)),
        ("pixel_first_hit_distance_m.f4", "<f4", (6,)), ("raster_labels.u1", "|u1", (6,)),
    )
    monkeypatch.setattr(C, "RAW_ROOT", raw_root)
    monkeypatch.setattr(C, "RAW_MANIFEST_PATH", f"{raw_root}/manifest.json")
    monkeypatch.setattr(C, "RAW_AUDIT_PATH", "audit.json")
    monkeypatch.setattr(C, "ARRAYS", layout)
    monkeypatch.setattr(C, "PAIR_COUNT", 8)
    monkeypatch.setattr(C, "ENDPOINT_COUNT", 16)
    monkeypatch.setattr(C, "SCENE_COUNT", 8)
    distances = np.repeat(np.arange(6, dtype=np.float64) + 0.5, 2)
    monkeypatch.setattr(C, "ground_queries", lambda *_args, **_kwargs: SimpleNamespace(
        in_frustum=np.ones(12, dtype=bool), target_distance_m=distances))
    root_files, endpoints, pairs, array_paths = [], [], [], []
    for family_index, family in enumerate(C.FAMILIES):
        shard_path = f"shards/{family_index}/shard.json"
        local_records = [{"path": "index.jsonl", "byte_count": 0, "file_sha256": "0" * 64,
                          "dtype": "canonical_jsonl", "shape": [2]}]
        values = {
            "camera_origin_body_m.f4": np.zeros((2, 3), dtype="<f4"),
            "camera_basis_body_fru.f4": np.tile(np.eye(3, dtype="<f4"), (2, 1, 1)),
            "ground_plane_z_body_m.f4": np.zeros(2, dtype="<f4"),
            "ground_support_in_frustum.u1": np.ones((2, 12), dtype="u1"),
            "ground_support_clear_to_target.u1": np.array([np.tile([0, 1], 6), np.tile([1, 0], 6)], dtype="u1"),
            "pixel_hit_mask.u1": np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype="u1"),
            "pixel_first_hit_distance_m.f4": np.array([[0.1] * 6, [1.1] * 6], dtype="<f4"),
            "raster_labels.u1": np.array([[0, 1, 2, 0, 1, 2], [1, 2, 0, 1, 2, 0]], dtype="u1"),
        }
        for name, dtype, _trailing in layout:
            array = values[name].astype(dtype, copy=False)
            relative = f"shards/{family_index}/{name}"
            raw = array.tobytes(order="C")
            record = _write(tmp_path / raw_root / relative, raw)
            root_files.append({**record, "path": relative})
            local_records.append({**record, "path": name, "dtype": np.dtype(dtype).str, "shape": list(array.shape)})
            array_paths.append(tmp_path / raw_root / relative)
        shard_raw = _json({"schema": "synthetic_shard", "files": local_records})
        record = _write(tmp_path / raw_root / shard_path, shard_raw)
        root_files.append({**record, "path": shard_path})
        identities = [f"{family}:{index}" for index in range(2)]
        endpoints.extend({"dataset_role": "checkpoint_selection", "endpoint_identity_sha256": identity,
                          "family": family, "scene_shard": shard_path, "shard_row": index}
                         for index, identity in enumerate(identities))
        pairs.append({"dataset_role": "checkpoint_selection", "family": family,
                      "current_endpoint_sha256": identities[0], "next_endpoint_sha256": identities[1]})
    for name, raw in (("pairs.jsonl", _jsonl(pairs)), ("endpoints.jsonl", _jsonl(endpoints))):
        record = _write(tmp_path / raw_root / name, raw); root_files.append({**record, "path": name})
    manifest_raw = _json({"schema": "synthetic_manifest", "files": root_files})
    _write(tmp_path / raw_root / "manifest.json", manifest_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    audit_core = {"schema": "synthetic_audit", "verdict": "PASS", "dataset_manifest_file_sha256": manifest_sha}
    for name in ("rgb_decode_authorized", "dataset_use_authorized", "training_authorized", "selection_authorized",
                 "calibration_authorized", "g2_authorized", "heldout_authorized", "runtime_authorized", "navigation_authorized"):
        audit_core[name] = False
    audit_raw = _json(audit_core); _write(tmp_path / "audit.json", audit_raw)
    monkeypatch.setattr(C, "RAW_MANIFEST_FILE_SHA256", manifest_sha)
    monkeypatch.setattr(C, "RAW_MANIFEST_CONTENT_SHA256", C.parse_json(manifest_raw, "manifest")["content_sha256"])
    monkeypatch.setattr(C, "RAW_AUDIT_FILE_SHA256", hashlib.sha256(audit_raw).hexdigest())
    monkeypatch.setattr(C, "RAW_AUDIT_CONTENT_SHA256", C.parse_json(audit_raw, "audit")["content_sha256"])
    return tmp_path, array_paths


def test_synthetic_temp_loader_opens_only_registered_supervision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _array_paths = _synthetic_raw_root(tmp_path, monkeypatch)
    events: list[dict] = []
    endpoints, receipt = R.load_inputs(root, events)
    assert receipt["pair_count"] == 8 and receipt["unique_endpoint_count"] == 16
    assert C.evaluate(endpoints)["all_nine_physical_pass"] is True
    assert len(events) == 76
    assert {row["role"] for row in events} == {"raw_v13_audit", "raw_v13_manifest", "checkpoint_selection"}
    assert not any(token in row["path"].lower() for row in events for token in ("rgb", "image", "checkpoint", "heldout", "calibration"))
    assert all(row["operation"] in {"rehash_audit", "rehash_manifest", "open_pair_index", "open_endpoint_index",
                                    "open_shard_manifest", "open_supervision_array"} for row in events)


def test_synthetic_input_tamper_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, arrays = _synthetic_raw_root(tmp_path, monkeypatch)
    arrays[0].write_bytes(arrays[0].read_bytes() + b"x")
    with pytest.raises(PermissionError, match="hash changed"):
        R.load_inputs(root, [])


def test_bound_reader_rejects_escape_and_links_before_recording_access(tmp_path: Path) -> None:
    outside = tmp_path.parent / "oracle_outside.bin"; outside.write_bytes(b"outside")
    events: list[dict] = []
    with pytest.raises(PermissionError, match="forbidden"):
        R._read(tmp_path, "../oracle_outside.bin", hashlib.sha256(b"outside").hexdigest(), events, "test", "test")
    (tmp_path / "linked.bin").symlink_to(outside)
    with pytest.raises(PermissionError, match="forbidden"):
        R._read(tmp_path, "linked.bin", hashlib.sha256(b"outside").hexdigest(), events, "test", "test")
    assert events == []


def _terminal(root: Path) -> dict[str, dict]:
    return {name: R._publish(root, name, {"schema": f"test_{name}", "ordinal": index})
            for index, name in enumerate(C.SUCCESS_PATHS)}


def _review_execution(passed: int = 12) -> dict:
    return {"accelerators_hidden": list(C.ACCELERATOR_ENV), "bytecode_disabled": True,
        "compile": {"files": 3, "result": "PASS"},
        "focused_cpu_tests": {"command": "python3 -m pytest focused_oracle.py", "duration_s": 0.2,
                              "failed": 0, "passed": passed, "result": "PASS"},
        "hsa_override_absent": True, "plugin_autoload_disabled": True, "pytest_cache_disabled": True,
        "thread_environment": {name: "1" for name in C.THREAD_ENV}}


@pytest.mark.parametrize("mutation", ["tamper", "extra", "missing"])
def test_canonical_terminal_inventory_rejects_mutation(tmp_path: Path, mutation: str) -> None:
    expected = _terminal(tmp_path)
    R._inventory(tmp_path, expected)
    assert all((path.stat().st_mode & 0o777) == 0o444 for path in tmp_path.iterdir())
    if mutation == "tamper":
        path = tmp_path / "result.json"; path.chmod(0o644); path.write_bytes(_json({"schema": "tampered"})); path.chmod(0o444)
    elif mutation == "extra":
        (tmp_path / "extra.json").write_bytes(b"{}\n")
    else:
        path = tmp_path / "result.json"; path.chmod(0o644); path.unlink()
    with pytest.raises(PermissionError):
        R._inventory(tmp_path, expected)


def test_review_and_authorization_are_exact_and_separate() -> None:
    sources = [{"path": "candidate.py", "sha256": "a" * 64}]
    review = C.content_value({"schema": C.REVIEW_SCHEMA, "verdict": "PASS_SOURCE_ONLY_NO_EXECUTION_AUTHORITY",
        "reviewer": "/root/oracle_reviewer", "implementation_author": C.IMPLEMENTATION_AUTHOR,
        "candidate": sources, "experiment": C.experiment(), "findings": [],
        "test_execution": _review_execution(), "authority": C.REVIEW_AUTHORITY})
    review_raw = C.canonical_bytes(review) + b"\n"
    assert C.validate_review(review_raw, sources)["authority"]["execution_authorized"] is False
    blocked_core = dict(review); blocked_core.pop("content_sha256")
    blocked_core["findings"] = [{"code": "still_blocked", "severity": "blocking"}]
    with pytest.raises(PermissionError):
        C.validate_review(_json(blocked_core), sources)
    review_binding = {**C.binding(C.REVIEW_PATH, review_raw, review), "reviewer": review["reviewer"]}
    authorization = C.content_value({"schema": C.AUTHORIZATION_SCHEMA, "status": "authorized_one_exact_positive_control_attempt",
        "authorizer": "/root/oracle_authorizer", "implementation_author": C.IMPLEMENTATION_AUTHOR,
        "independent_review": review_binding, "candidate": sources, "raw": C.raw_bindings(),
        "experiment": C.experiment(), "authority": C.EXECUTION_AUTHORITY})
    raw = C.canonical_bytes(authorization) + b"\n"
    assert C.validate_authorization(raw, sources, review_binding)["authority"]["retry_authorized"] is False
    authorization["authority"]["retry_authorized"] = True
    with pytest.raises((ValueError, PermissionError)):
        C.validate_authorization(C.canonical_bytes(authorization) + b"\n", sources, review_binding)


def test_partial_completed_write_leaves_no_final_and_terminalizes_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _synthetic_raw_root(tmp_path, monkeypatch)
    monkeypatch.setattr(C, "OUTPUT_ROOT", "oracle_output")
    monkeypatch.setattr(C, "EXECUTION_AUTHORITY", {**C.EXECUTION_AUTHORITY, "mutation_scope": "oracle_output"})
    monkeypatch.setattr(R, "_environment", lambda: {"worker_count": 1, "synthetic_cpu_only": True})
    sources = C.source_bindings(C.ROOT)
    review = C.content_value({"schema": C.REVIEW_SCHEMA, "verdict": "PASS_SOURCE_ONLY_NO_EXECUTION_AUTHORITY",
        "reviewer": "/root/oracle_fault_reviewer", "implementation_author": C.IMPLEMENTATION_AUTHOR,
        "candidate": sources, "experiment": C.experiment(), "findings": [],
        "test_execution": _review_execution(), "authority": C.REVIEW_AUTHORITY})
    review_raw = C.canonical_bytes(review) + b"\n"; _write(tmp_path / C.REVIEW_PATH, review_raw)
    review_binding = {**C.binding(C.REVIEW_PATH, review_raw, review), "reviewer": review["reviewer"]}
    authorization = C.content_value({"schema": C.AUTHORIZATION_SCHEMA, "status": "authorized_one_exact_positive_control_attempt",
        "authorizer": "/root/oracle_fault_authorizer", "implementation_author": C.IMPLEMENTATION_AUTHOR,
        "independent_review": review_binding, "candidate": sources, "raw": C.raw_bindings(),
        "experiment": C.experiment(), "authority": C.EXECUTION_AUTHORITY})
    auth_raw = C.canonical_bytes(authorization) + b"\n"; _write(tmp_path / C.AUTHORIZATION_PATH, auth_raw)
    write, state = R.os.write, {"descriptor": None, "calls": 0, "failed": False}
    def fail_partial_completion(descriptor: int, data: object) -> int:
        payload = bytes(data)
        if state["descriptor"] is None and C.COMPLETION_SCHEMA.encode("ascii") in payload:
            state["descriptor"] = descriptor; state["calls"] += 1
            prefix = max(1, len(payload) // 3)
            return write(descriptor, payload[:prefix])
        if descriptor == state["descriptor"] and not state["failed"]:
            state["calls"] += 1; state["failed"] = True
            raise OSError("synthetic partial completed write failure")
        return write(descriptor, data)
    monkeypatch.setattr(R.os, "write", fail_partial_completion)
    with pytest.raises(OSError, match="synthetic partial completed"):
        R.execute(hashlib.sha256(auth_raw).hexdigest(), repository_root=C.ROOT, execution_root=tmp_path)
    output = tmp_path / "oracle_output"
    assert state["descriptor"] is not None and state["calls"] == 2 and state["failed"] is True
    assert sorted(path.name for path in output.iterdir()) == ["access.json", "failed.json", "reservation.json", "result.json"]
    assert not (output / "completed.json").exists()
    assert not any(path.name.startswith(".completed.json.") for path in output.iterdir())
    failed_raw = (output / "failed.json").read_bytes(); failed = C.parse_json(failed_raw, "post-result failure")
    assert failed["status"] == "terminal_post_result_failure_no_retry"
    assert failed["exact_paths"] == ["reservation.json", "access.json", "result.json", "failed.json"]
    assert [row["path"] for row in failed["partial_artifacts"]] == ["reservation.json", "access.json", "result.json"]
    bindings = {row["path"]: row for row in failed["partial_artifacts"]}
    bindings["failed.json"] = C.binding("failed.json", failed_raw, failed)
    R._inventory(output, bindings)
    assert all((path.stat().st_mode & 0o777) == 0o444 for path in output.iterdir())


def test_missing_review_stops_before_reservation_or_governed_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = False
    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("governed input opened")
    monkeypatch.setattr(R, "_environment", lambda: {"cpu": True})
    monkeypatch.setattr(C, "source_bindings", lambda _root: [])
    monkeypatch.setattr(R, "load_inputs", forbidden)
    with pytest.raises(FileNotFoundError):
        R.execute("0" * 64, execution_root=tmp_path)
    assert called is False
    assert not (tmp_path / ".generated").exists()


def test_sources_have_no_neural_image_or_process_pool_surface() -> None:
    runner_source = RUNNER_PATH.read_text(encoding="ascii")
    contract_source = Path(C.__file__).read_text(encoding="ascii")
    imports = {node.names[0].name.split(".")[0] for source in (runner_source, contract_source)
               for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom)) and node.names}
    assert not imports & {"torch", "PIL", "cv2", "multiprocessing", "concurrent"}
    loader_source = inspect.getsource(R.load_inputs)
    assert "image_path_metadata_only" not in loader_source
    assert "image_sha256_commitment_only" not in loader_source
    assert "ProcessPool" not in runner_source + contract_source
