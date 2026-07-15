#!/usr/bin/env python3
"""Run the one authorized CPU-only Camera physical-gate positive control."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lewm.benchmarks import go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1 as C  # noqa: E402


def _event(events: list[dict[str, Any]], role: str, operation: str, path: str, raw: bytes) -> None:
    core = {"sequence": len(events), "role": role, "operation": operation, "path": path,
            "file_sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw),
            "prior_event_sha256": events[-1]["event_sha256"] if events else "0" * 64}
    events.append({**core, "event_sha256": C.canonical_sha(core)})


def _read(root: Path, relative: str, digest: str, events: list[dict[str, Any]], role: str, operation: str) -> bytes:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PermissionError(f"forbidden oracle input: {relative}")
    path = root / relative_path
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or root.resolve() not in path.resolve().parents:
        raise PermissionError(f"forbidden oracle input: {relative}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"oracle input hash changed: {relative}")
    _event(events, role, operation, relative, raw)
    return raw


def _jsonl(raw: bytes, name: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n") or b"\n\n" in raw:
        raise ValueError(f"{name} is not canonical JSONL")
    rows = []
    for line in raw.splitlines():
        row = json.loads(line.decode("ascii")); core = dict(row); declared = core.pop("content_sha256", None)
        if type(row) is not dict or C.canonical_bytes(row) != line or C.canonical_sha(core) != declared:
            raise ValueError(f"{name} row changed")
        rows.append(row)
    return rows


def _record(manifest: Mapping[str, Any], path: str) -> dict[str, Any]:
    rows = [row for row in manifest["files"] if row.get("path") == path]
    if len(rows) != 1:
        raise PermissionError(f"manifest binding changed: {path}")
    return rows[0]


def _leaf(root: Path, manifest: Mapping[str, Any], path: str, events: list[dict[str, Any]], operation: str) -> bytes:
    record = _record(manifest, path)
    raw = _read(root, f"{C.RAW_ROOT}/{path}", record["file_sha256"], events, "checkpoint_selection", operation)
    if len(raw) != record["byte_count"]:
        raise PermissionError("manifest byte count changed")
    return raw


def load_inputs(root: Path, events: list[dict[str, Any]]) -> tuple[list[C.Endpoint], dict[str, Any]]:
    audit_raw = _read(root, C.RAW_AUDIT_PATH, C.RAW_AUDIT_FILE_SHA256, events, "raw_v13_audit", "rehash_audit")
    audit = C.parse_json(audit_raw, "raw V13 audit")
    denials = ("rgb_decode_authorized", "dataset_use_authorized", "training_authorized", "selection_authorized",
               "calibration_authorized", "g2_authorized", "heldout_authorized", "runtime_authorized", "navigation_authorized")
    if (audit["content_sha256"] != C.RAW_AUDIT_CONTENT_SHA256 or audit.get("verdict") != "PASS"
            or audit.get("dataset_manifest_file_sha256") != C.RAW_MANIFEST_FILE_SHA256
            or any(audit.get(name) is not False for name in denials)):
        raise PermissionError("raw V13 audit changed")
    manifest_raw = _read(root, C.RAW_MANIFEST_PATH, C.RAW_MANIFEST_FILE_SHA256, events, "raw_v13_manifest", "rehash_manifest")
    manifest = C.parse_json(manifest_raw, "raw manifest")
    if manifest["content_sha256"] != C.RAW_MANIFEST_CONTENT_SHA256:
        raise PermissionError("raw manifest changed")
    pairs = _jsonl(_leaf(root, manifest, "pairs.jsonl", events, "open_pair_index"), "pairs")
    endpoints = _jsonl(_leaf(root, manifest, "endpoints.jsonl", events, "open_endpoint_index"), "endpoints")
    pairs = [row for row in pairs if row.get("dataset_role") == "checkpoint_selection"]
    endpoints = [row for row in endpoints if row.get("dataset_role") == "checkpoint_selection"]
    by_id = {str(row.get("endpoint_identity_sha256")): row for row in endpoints}
    referenced = {str(row[f"{side}_endpoint_sha256"]) for row in pairs for side in ("current", "next")}
    shards = sorted({str(row.get("scene_shard")) for row in endpoints})
    if (len(pairs), len(endpoints), len(by_id), len(shards)) != (C.PAIR_COUNT, C.ENDPOINT_COUNT, C.ENDPOINT_COUNT, C.SCENE_COUNT):
        raise PermissionError("selection population changed")
    if referenced != set(by_id) or {row.get("family") for row in endpoints} != set(C.FAMILIES):
        raise PermissionError("selection joins changed")
    result = []
    for shard_path in shards:
        shard = C.parse_json(_leaf(root, manifest, shard_path, events, "open_shard_manifest"), "shard")
        records = {row["path"]: row for row in shard["files"]}
        if set(records) != {name for name, _dtype, _shape in C.ARRAYS} | {"index.jsonl"}:
            raise PermissionError("selection shard inventory changed")
        arrays = {}
        for name, dtype, trailing in C.ARRAYS:
            record, path = records[name], f"{Path(shard_path).parent.as_posix()}/{name}"
            raw = _leaf(root, manifest, path, events, "open_supervision_array")
            shape = tuple(record["shape"])
            if record["dtype"] != np.dtype(dtype).str or tuple(shape[1:]) != trailing or len(raw) != np.prod(shape) * np.dtype(dtype).itemsize:
                raise PermissionError("selection array contract changed")
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape)
        for row in (item for item in endpoints if item["scene_shard"] == shard_path):
            index = row["shard_row"]
            origin, basis = arrays["camera_origin_body_m.f4"][index], arrays["camera_basis_body_fru.f4"][index]
            queries = C.ground_queries(origin, basis, float(arrays["ground_plane_z_body_m.f4"][index]))
            valid = arrays["ground_support_in_frustum.u1"][index].astype(bool)
            if not np.array_equal(valid, queries.in_frustum):
                raise PermissionError("selection calibration changed")
            result.append(C.Endpoint(str(row["endpoint_identity_sha256"]), str(row["family"]),
                arrays["pixel_hit_mask.u1"][index].astype(bool), arrays["pixel_first_hit_distance_m.f4"][index], valid,
                arrays["ground_support_clear_to_target.u1"][index].astype(bool), queries.target_distance_m,
                arrays["raster_labels.u1"][index]))
    if len(result) != C.ENDPOINT_COUNT or len({row.identity for row in result}) != C.ENDPOINT_COUNT:
        raise PermissionError("selection supervision coverage changed")
    receipt = {"pair_count": len(pairs), "unique_endpoint_count": len(result), "scene_count": len(shards),
               "family_counts": dict(sorted(Counter(row.family for row in result).items())),
               "ordered_endpoint_identity_sha256": C.canonical_sha(sorted(row.identity for row in result))}
    return result, receipt


def _publish(root: Path, name: str, core: Mapping[str, Any]) -> dict[str, Any]:
    value = C.content_value(core)
    raw = C.canonical_bytes(value) + b"\n"
    directory = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.publishing"
    descriptor, temporary_exists = None, False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=directory)
        temporary_exists = True
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while publishing oracle JSON")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor); descriptor = None
        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            os.fsync(directory)
        os.close(directory)
    return C.binding(name, raw, value)


def _inventory(root: Path, expected: Mapping[str, Mapping[str, Any]]) -> None:
    if root.is_symlink() or sorted(row.name for row in root.iterdir()) != sorted(expected):
        raise PermissionError("terminal inventory changed")
    for name, binding in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise PermissionError("terminal artifact changed")
        raw = path.read_bytes(); value = C.parse_json(raw, name)
        if C.binding(name, raw, value) != dict(binding):
            raise PermissionError("terminal artifact binding changed")


def _environment() -> dict[str, Any]:
    if any(os.environ.get(name) != "1" for name in C.THREAD_ENV) or any(os.environ.get(name) != "" for name in C.ACCELERATOR_ENV):
        raise PermissionError("CPU-only environment changed")
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ or any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise PermissionError("forbidden accelerator or neural runtime state")
    return {"worker_count": 1, "threads": {name: "1" for name in C.THREAD_ENV}, "accelerators_hidden": list(C.ACCELERATOR_ENV)}


def _access(attempt: str, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"schema": C.ACCESS_SCHEMA, "attempt_identity": attempt, "events": list(events), "event_count": len(events),
            "role_counts": dict(sorted(Counter(row["role"] for row in events).items())), "rgb_open_count": 0,
            "train_or_calibration_leaf_open_count": 0, "checkpoint_model_g2_navigation_heldout_open_count": 0}


def _post_result_failure(output: Path, attempt: str, error: BaseException) -> None:
    partial = []
    for name in ("reservation.json", "access.json", "result.json", "completed.json"):
        path = output / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise PermissionError("post-result partial artifact changed") from error
        raw = path.read_bytes(); value = C.parse_json(raw, name)
        partial.append(C.binding(name, raw, value))
    if [row["path"] for row in partial[:3]] != ["reservation.json", "access.json", "result.json"]:
        raise PermissionError("post-result evidence prefix changed") from error
    failed = _publish(output, "failed.json", {"schema": C.FAILURE_SCHEMA, "status": "terminal_post_result_failure_no_retry",
        "stage": "completed_publication_or_terminal_inventory", "attempt_identity": attempt,
        "error_type": type(error).__name__, "error_message": str(error), "partial_artifacts": partial,
        "exact_paths": [row["path"] for row in partial] + ["failed.json"], "retry_authorized": False, "authority": C.DENIALS})
    _inventory(output, {row["path"]: row for row in (*partial, failed)})


def execute(authorization_sha256: str, repository_root: Path = ROOT, execution_root: Path | None = None) -> dict[str, Any]:
    execution_root = execution_root or repository_root
    environment, sources = _environment(), C.source_bindings(repository_root)
    review_raw = (execution_root / C.REVIEW_PATH).read_bytes(); review = C.validate_review(review_raw, sources)
    review_binding = {**C.binding(C.REVIEW_PATH, review_raw, review), "reviewer": review["reviewer"]}
    auth_raw = (execution_root / C.AUTHORIZATION_PATH).read_bytes()
    if hashlib.sha256(auth_raw).hexdigest() != authorization_sha256:
        raise PermissionError("authorization CLI hash changed")
    auth = C.validate_authorization(auth_raw, sources, review_binding)
    auth_binding = C.binding(C.AUTHORIZATION_PATH, auth_raw, auth)
    output = execution_root / C.OUTPUT_ROOT
    if output.exists() or output.is_symlink():
        raise FileExistsError("output root must be absent")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    attempt = C.canonical_sha({"authorization": auth_binding["file_sha256"], "candidate": sources, "experiment": C.experiment()})
    reservation = _publish(output, "reservation.json", {"schema": C.RESERVATION_SCHEMA,
        "status": "reserved_before_governed_input_open", "attempt_identity": attempt, "attempt_index": 1,
        "maximum_attempts": 1, "retry_authorized": False, "candidate": sources, "review": review_binding,
        "authorization": auth_binding, "raw": C.raw_bindings(), "experiment": C.experiment(), "environment": environment})
    events: list[dict[str, Any]] = []
    try:
        endpoints, population = load_inputs(execution_root, events); result = C.evaluate(endpoints)
        access = _publish(output, "access.json", _access(attempt, events))
        result_binding = _publish(output, "result.json", {"schema": C.RESULT_SCHEMA, "status": "completed_positive_control",
            "attempt_identity": attempt, "population": population, **result, "margin_count": len(result["raw_margin_vector"]),
            "decision": "PASS_GATE_ATTAINABLE_BY_ZERO_PARAMETER_ENDPOINT_IDENTITY_ORACLE" if result["all_nine_physical_pass"]
                else "BLOCK_GATE_NOT_ATTAINED_BY_PREREGISTERED_POSITIVE_CONTROL",
            "interpretation": "positive_control_only_not_learned_performance", "authority": C.DENIALS})
        completed = _publish(output, "completed.json", {"schema": C.COMPLETION_SCHEMA, "status": "complete_immutable_positive_control",
            "attempt_identity": attempt, "artifacts": {"reservation": reservation, "access": access, "result": result_binding},
            "exact_paths": list(C.SUCCESS_PATHS), "retry_authorized": False, "authority": C.DENIALS})
        _inventory(output, {row["path"]: row for row in (reservation, access, result_binding, completed)})
        return {"status": "complete", "all_nine_physical_pass": result["all_nine_physical_pass"], "attempt_identity": attempt}
    except BaseException as error:
        if (output / "result.json").exists():
            _post_result_failure(output, attempt, error)
        else:
            access = C.binding("access.json", (output / "access.json").read_bytes(), C.parse_json((output / "access.json").read_bytes(), "access")) if (output / "access.json").exists() else _publish(output, "access.json", _access(attempt, events))
            failed = _publish(output, "failed.json", {"schema": C.FAILURE_SCHEMA, "status": "terminal_failure_no_retry",
                "attempt_identity": attempt, "error_type": type(error).__name__, "error_message": str(error),
                "retry_authorized": False, "authority": C.DENIALS})
            _inventory(output, {row["path"]: row for row in (reservation, access, failed)})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--authorization-sha256", required=True)
    result = execute(parser.parse_args(argv).authorization_sha256); print(C.canonical_bytes(result).decode("ascii")); return 0


if __name__ == "__main__":
    raise SystemExit(main())
