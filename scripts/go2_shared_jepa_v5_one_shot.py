#!/usr/bin/env python3
"""Stdlib-only execution core for the staged Shared JEPA V5 lifecycle.

Each stage consumes a separate, immutable authority revision. Runner revisions
bind only inputs that already exist and paths that must not exist. Finalizer
revisions bind the exact runner authority, ledger, and outcomes. Publication
revisions bind finalized evidence for either a G2 candidate or full G2+G3
promotion. Production configuration is deliberately pending; synthetic test
artifacts are permanently production-ineligible. This core is executed only
from bytes captured by the fixed launcher; production authority identities live
in the three fixed wrappers so the core/source binding is non-circular.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


CANONICAL_REPOSITORY_ROOT = Path(
    "/home/andrewknowles/Workspace/LeWMQuad-v3"
).resolve()

AUTHORITY_SCHEMA = "lewm_go2_shared_jepa_v5_stage_authority_v3"
ROLE_MANIFEST_SCHEMA = "lewm_go2_shared_jepa_dataset_roles_v7"
RAW_SCENE_SCHEMA = "lewm_go2_shared_jepa_raw_scene_input_v1"
RAW_OUTCOME_SCHEMA = "lewm_go2_shared_jepa_raw_scene_outcome_v8"
LEDGER_SCHEMA = "lewm_go2_shared_jepa_runner_open_ledger_v8"
FINAL_REPORT_SCHEMA = "lewm_go2_shared_jepa_final_report_v9"
PUBLICATION_SCHEMA = "lewm_go2_shared_jepa_publication_v3"
EXECUTION_IDENTITY_SCHEMA = "lewm_go2_shared_jepa_v5_execution_identity_v1"

RUNNER_REVISIONS = {
    "g2": "runner_g2_inputs_v2",
    "g3": "runner_g3_inputs_v2",
}
FINALIZER_REVISIONS = {
    "g2": "finalizer_g2_evidence_v2",
    "g3": "finalizer_g3_evidence_v2",
}
PUBLISHER_REVISIONS = {
    "g2-candidate": "publisher_g2_candidate_v2",
    "full-promotion": "publisher_full_promotion_v2",
}

GATE_METRICS = {
    "g2": (
        "aggregate_physical_gate_pass_fraction",
        "per_family_physical_gate_pass_fraction",
        "jepa_health_gate_pass_fraction",
        "counterfactual_gate_pass_fraction",
    ),
    "g3": (
        "exact_morphology_equivalence_pass_fraction",
        "configuration_runtime_gate_pass_fraction",
        "safety_gate_pass_fraction",
        "task_gate_pass_fraction",
    ),
}
SYNTHETIC_AUTHORITY_ENV = "LEWM_V5_SYNTHETIC_AUTHORITY_PATH"
EXECUTION_SOURCE_PATHS = {
    "runner": "scripts/run_go2_shared_jepa_v5_gate.py",
    "finalizer": "scripts/finalize_go2_shared_jepa_v5_gate.py",
    "publisher": "scripts/publish_go2_shared_jepa_v5_checkpoint.py",
}
CAPTURED_LAUNCHER_PATH = "scripts/go2_shared_jepa_v5_launcher.py"
CAPTURED_CORE_PATH = "scripts/go2_shared_jepa_v5_one_shot.py"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _execution_identity(value: object, *, stage: str) -> dict[str, object]:
    if stage not in EXECUTION_SOURCE_PATHS or not isinstance(value, Mapping):
        raise ValueError("V5 execution stage or identity changed")
    if set(value) != {
        "schema",
        "entrypoint_wrapper",
        "captured_launcher",
        "captured_core",
    } or value.get("schema") != EXECUTION_IDENTITY_SCHEMA:
        raise ValueError("V5 execution identity fields changed")
    expected_paths = {
        "entrypoint_wrapper": EXECUTION_SOURCE_PATHS[stage],
        "captured_launcher": CAPTURED_LAUNCHER_PATH,
        "captured_core": CAPTURED_CORE_PATH,
    }
    normalized: dict[str, object] = {"schema": EXECUTION_IDENTITY_SCHEMA}
    for name, expected_path in expected_paths.items():
        spec = value.get(name)
        if not isinstance(spec, Mapping) or set(spec) != {"path", "file_sha256"}:
            raise ValueError(f"V5 {name} identity changed")
        path = _relative(spec.get("path"), name=f"V5 {name} path")
        if path != expected_path:
            raise PermissionError(f"V5 {name} source path changed")
        normalized[name] = {
            "path": path,
            "file_sha256": _sha256(
                spec.get("file_sha256"),
                name=f"V5 {name} source",
            ),
        }
    hashes = [
        normalized[name]["file_sha256"]
        for name in expected_paths
        if isinstance(normalized[name], Mapping)
    ]
    if len(set(hashes)) != len(hashes):
        raise PermissionError("V5 execution source identities overlap")
    return normalized


def _production_authority_inventory(
    value: object,
    *,
    stage: str,
) -> dict[str, tuple[Path, str | None]]:
    expected_revisions = {
        "runner": set(RUNNER_REVISIONS.values()),
        "finalizer": set(FINALIZER_REVISIONS.values()),
        "publisher": set(PUBLISHER_REVISIONS.values()),
    }[stage]
    if not isinstance(value, Mapping) or set(value) != expected_revisions:
        raise ValueError("V5 production authority inventory changed")
    normalized: dict[str, tuple[Path, str | None]] = {}
    for revision in sorted(expected_revisions):
        row = value[revision]
        if (
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], Path)
            or (row[1] is not None and type(row[1]) is not str)
        ):
            raise ValueError("V5 production authority binding changed")
        path, expected_hash = row
        if expected_hash is not None:
            expected_hash = _sha256(expected_hash, name="production authority")
        normalized[revision] = (path, expected_hash)
    return normalized


def _relative(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} must be a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name} must be a canonical relative path")
    return value


def _strict_object(encoded: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or encoded != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical newline-terminated JSON")
    claimed = _sha256(value.get("content_sha256"), name=f"{name} content")
    core = dict(value)
    del core["content_sha256"]
    if _canonical_sha256(core) != claimed:
        raise ValueError(f"{name} content hash changed")
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key}")
        value[key] = item
    return value


def _read_absolute(path: Path, *, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError(f"{name} is not a singly-linked regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"{name} changed while open")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _root_fd(root: Path) -> int:
    metadata = os.lstat(root)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise PermissionError("authority repository root is not a real directory")
    return os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )


def _walk_parent(root_fd: int, relative: str) -> tuple[int, str]:
    parts = PurePosixPath(_relative(relative, name="artifact path")).parts
    current = os.dup(root_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for part in parts[:-1]:
            following = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = following
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _read_at(
    root_fd: int,
    spec: Mapping[str, object],
    *,
    name: str,
) -> tuple[bytes, str, str]:
    if not isinstance(spec, Mapping) or set(spec) != {"path", "file_sha256"}:
        raise ValueError(f"{name} artifact binding changed")
    relative = _relative(spec.get("path"), name=f"{name} path")
    expected = _sha256(spec.get("file_sha256"), name=f"{name} file hash")
    parent, leaf = _walk_parent(root_fd, relative)
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise PermissionError(f"{name} is not a singly-linked regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(f"{name} changed while open")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    encoded = b"".join(chunks)
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != expected:
        raise PermissionError(f"{name} file hash changed")
    return encoded, relative, actual


def _write_exclusive(root_fd: int, relative: str, encoded: bytes) -> str:
    parent, leaf = _walk_parent(root_fd, relative)
    try:
        descriptor = os.open(
            leaf,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("exclusive artifact write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(encoded).hexdigest()


def _require_absent_at(root_fd: int, relative: str, *, name: str) -> None:
    parent, leaf = _walk_parent(root_fd, relative)
    try:
        try:
            os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"{name} already exists: {relative}")
    finally:
        os.close(parent)


def _authority_fields(revision: str) -> set[str]:
    common = {
        "schema",
        "lifecycle_revision",
        "synthetic_only",
        "repository_root",
        "protocol_generation",
        "content_sha256",
    }
    if revision in RUNNER_REVISIONS.values():
        fields = common | {
            "gate",
            "attempt_registry_path",
            "dataset_role_manifest",
            "evaluated_checkpoint",
            "runtime_modules",
            "inference_entry_module",
            "gate_spec",
        }
        if revision == RUNNER_REVISIONS["g3"]:
            fields.update(
                {
                    "g2_candidate_publisher_authority",
                    "g2_candidate_publication",
                    "g2_candidate_publisher_execution_identity",
                }
            )
        return fields
    if revision in FINALIZER_REVISIONS.values():
        return common | {
            "gate",
            "runner_authority",
            "runner_ledger",
            "outcome_files",
            "final_report_path",
            "runner_execution_identity",
        }
    if revision in PUBLISHER_REVISIONS.values():
        return common | {"finalized_gates", "publication_path"}
    raise ValueError("unknown V5 lifecycle revision")


def _validate_authority(
    value: Mapping[str, Any],
    *,
    revision: str,
    synthetic: bool,
    expected_root: Path | None = None,
    expected_protocol_generation: object | None = None,
) -> Path:
    if set(value) != _authority_fields(revision):
        raise ValueError("V5 stage authority fields changed")
    if (
        value.get("schema") != AUTHORITY_SCHEMA
        or value.get("lifecycle_revision") != revision
        or value.get("synthetic_only") is not synthetic
    ):
        raise PermissionError("V5 stage authority revision or mode changed")
    protocol_generation = value.get("protocol_generation")
    if type(protocol_generation) is not str or not protocol_generation:
        raise ValueError("authority protocol generation changed")
    if (
        expected_protocol_generation is not None
        and protocol_generation != expected_protocol_generation
    ):
        raise PermissionError("linked authority protocol generation changed")
    root_value = value.get("repository_root")
    if type(root_value) is not str:
        raise ValueError("authority repository root changed")
    root = Path(root_value).resolve(strict=True)
    if expected_root is not None and root != expected_root:
        raise PermissionError("linked authority repository root changed")
    if not synthetic and root != CANONICAL_REPOSITORY_ROOT:
        raise PermissionError("production authority uses a foreign repository root")
    if synthetic and root == CANONICAL_REPOSITORY_ROOT:
        raise PermissionError("synthetic authority cannot use the production repository root")

    gate = value.get("gate")
    if revision in RUNNER_REVISIONS.values():
        expected_gate = next(
            name for name, candidate in RUNNER_REVISIONS.items() if candidate == revision
        )
        if gate != expected_gate:
            raise PermissionError("runner authority gate changed")
    elif revision in FINALIZER_REVISIONS.values():
        expected_gate = next(
            name for name, candidate in FINALIZER_REVISIONS.items() if candidate == revision
        )
        if gate != expected_gate:
            raise PermissionError("finalizer authority gate changed")
    return root


def _authority(
    revision: str,
    production_authorities: Mapping[str, tuple[Path, str | None]],
) -> tuple[dict[str, Any], Path, bool, str]:
    if revision not in production_authorities:
        raise ValueError("unknown V5 lifecycle revision")
    synthetic_path = os.environ.get(SYNTHETIC_AUTHORITY_ENV)
    if synthetic_path is None:
        path, configured = production_authorities[revision]
        if path.resolve() != path or path.parent != CANONICAL_REPOSITORY_ROOT / "docs":
            raise PermissionError("production authority path is not canonical")
        if configured is None:
            raise PermissionError(
                "Shared JEPA V5 production lifecycle revision is pending: "
                + revision
            )
        expected = _sha256(configured, name="stage authority")
        synthetic = False
    else:
        path = Path(synthetic_path).resolve(strict=True)
        expected = None
        synthetic = True
    encoded = _read_absolute(path, name="V5 stage authority")
    actual = hashlib.sha256(encoded).hexdigest()
    if expected is not None and actual != expected:
        raise PermissionError("V5 stage authority file hash changed")
    value = _strict_object(encoded, name="V5 stage authority")
    root = _validate_authority(
        value,
        revision=revision,
        synthetic=synthetic,
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError("V5 stage authority is outside its repository root") from exc
    return value, root, synthetic, actual


def _linked_authority(
    root_fd: int,
    spec: object,
    *,
    revision: str,
    synthetic: bool,
    root: Path,
    protocol_generation: object,
    name: str,
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"{name} binding changed")
    encoded, path, file_hash = _read_at(root_fd, spec, name=name)
    value = _strict_object(encoded, name=name)
    _validate_authority(
        value,
        revision=revision,
        synthetic=synthetic,
        expected_root=root,
        expected_protocol_generation=protocol_generation,
    )
    return value, path, file_hash


class _CapturedLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, sources: Mapping[str, tuple[str, bytes, bool]]) -> None:
        self.sources = dict(sources)

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        row = self.sources.get(fullname)
        if row is None:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=row[2])

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        relative, encoded, is_package = self.sources[module.__name__]
        module.__file__ = str(CANONICAL_REPOSITORY_ROOT / relative)
        if is_package:
            module.__path__ = [str((CANONICAL_REPOSITORY_ROOT / relative).parent)]
        exec(compile(encoded, module.__file__, "exec"), module.__dict__)


class _BlockUncapturedProjectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if fullname == "lewm" or fullname.startswith("lewm."):
            raise ImportError(f"uncaptured project import is forbidden: {fullname}")
        return None


def _captured_runtime(
    root_fd: int,
    authority: Mapping[str, Any],
    events: list[dict[str, Any]],
) -> ModuleType:
    raw_modules = authority.get("runtime_modules")
    entry = authority.get("inference_entry_module")
    if not isinstance(raw_modules, Mapping) or type(entry) is not str or entry not in raw_modules:
        raise ValueError("captured runtime module inventory changed")
    sources: dict[str, tuple[str, bytes, bool]] = {}
    for name in sorted(raw_modules):
        spec = raw_modules[name]
        if type(name) is not str or not name or name in sys.modules:
            raise PermissionError(f"captured runtime module was already loaded: {name}")
        if not isinstance(spec, Mapping) or set(spec) != {"path", "file_sha256", "package"}:
            raise ValueError("captured runtime source binding changed")
        encoded, relative, actual = _read_at(
            root_fd,
            {"path": spec["path"], "file_sha256": spec["file_sha256"]},
            name=f"captured source {name}",
        )
        if type(spec["package"]) is not bool:
            raise ValueError("captured runtime package marker changed")
        sources[name] = (relative, encoded, spec["package"])
        events.append(
            {
                "sequence": len(events) + 1,
                "operation": "capture_runtime_source",
                "path": relative,
                "file_sha256": actual,
                "forbidden": False,
            }
        )
    loader = _CapturedLoader(sources)
    blocker = _BlockUncapturedProjectImports()
    sys.meta_path.insert(0, blocker)
    sys.meta_path.insert(0, loader)
    try:
        return importlib.import_module(entry)
    finally:
        sys.meta_path.remove(loader)
        sys.meta_path.remove(blocker)


def _content(value: Mapping[str, object]) -> dict[str, object]:
    core = dict(value)
    return {**core, "content_sha256": _canonical_sha256(core)}


def _json_output(value: Mapping[str, object]) -> bytes:
    return _canonical_bytes(_content(value)) + b"\n"


def _string_list(value: object, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not str or not item for item in value)
        or value != sorted(value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must be a sorted unique nonempty string list")
    return tuple(value)


def _roles(
    value: Mapping[str, Any],
    *,
    expected_protocol_generation: object,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    if set(value) != {"schema", "protocol_generation", "roles", "scene_families", "content_sha256"}:
        raise ValueError("dataset role manifest fields changed")
    if value.get("schema") != ROLE_MANIFEST_SCHEMA:
        raise ValueError("dataset role manifest schema changed")
    if value.get("protocol_generation") != expected_protocol_generation:
        raise PermissionError("dataset role manifest protocol generation changed")
    raw = value.get("roles")
    if not isinstance(raw, Mapping) or set(raw) != {"train", "g2", "g3"}:
        raise ValueError("dataset role inventory changed")
    roles = {name: _string_list(raw[name], name=f"{name} scenes") for name in ("train", "g2", "g3")}
    flat = [scene for rows in roles.values() for scene in rows]
    if len(flat) != len(set(flat)):
        raise PermissionError("dataset roles overlap")
    families = value.get("scene_families")
    if not isinstance(families, Mapping) or set(families) != set(flat) or any(
        type(item) is not str or not item for item in families.values()
    ):
        raise ValueError("scene-family map changed")
    return roles, dict(families)


def _role_manifest_identity(
    value: Mapping[str, Any],
    *,
    path: str,
    file_sha256: str,
    protocol_generation: object,
) -> dict[str, object]:
    return {
        "path": _relative(path, name="dataset role manifest path"),
        "file_sha256": _sha256(
            file_sha256,
            name="dataset role manifest file",
        ),
        "content_sha256": _sha256(
            value.get("content_sha256"),
            name="dataset role manifest content",
        ),
        "protocol_generation": protocol_generation,
    }


def _output_path(value: object, path: Sequence[str]) -> object:
    current = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError("inference output lacks a bound metric path")
        current = current[component]
    return current


def _metric_outcomes(
    output: object,
    targets: Mapping[str, object],
    rules: Mapping[str, object],
    metric_names: Sequence[str],
) -> dict[str, bool]:
    if set(rules) != set(metric_names):
        raise ValueError("metric-rule inventory changed")
    result: dict[str, bool] = {}
    for name in metric_names:
        rule = rules[name]
        if not isinstance(rule, Mapping) or set(rule) - {"output_path", "operator", "target", "value"}:
            raise ValueError(f"metric rule changed: {name}")
        path = rule.get("output_path")
        if not isinstance(path, list) or not path or any(type(item) is not str or not item for item in path):
            raise ValueError(f"metric output path changed: {name}")
        observed = _output_path(output, path)
        operator = rule.get("operator")
        expected = rule.get("value")
        if "target" in rule:
            target_name = rule["target"]
            if type(target_name) is not str or target_name not in targets:
                raise ValueError(f"metric target changed: {name}")
            expected = targets[target_name]
        if operator == "is_true":
            if type(observed) is not bool or "target" in rule or "value" in rule:
                raise ValueError(f"boolean metric rule changed: {name}")
            passed = observed
        elif operator in {"equal", "gte", "lte", "abs_error_lte"}:
            if operator == "equal":
                passed = observed == expected
            else:
                if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                    raise ValueError(f"numeric metric output changed: {name}")
                if isinstance(expected, bool) or not isinstance(expected, (int, float)):
                    raise ValueError(f"numeric metric target changed: {name}")
                if operator == "gte":
                    passed = float(observed) >= float(expected)
                elif operator == "lte":
                    passed = float(observed) <= float(expected)
                else:
                    tolerance = rule.get("value")
                    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or "target" not in rule:
                        raise ValueError(f"absolute-error tolerance changed: {name}")
                    passed = abs(float(observed) - float(expected)) <= float(tolerance)
        else:
            raise ValueError(f"unsupported metric operator: {operator}")
        result[name] = bool(passed)
    return result


def _runner_gate_spec(authority: Mapping[str, Any], gate: str) -> Mapping[str, Any]:
    value = authority.get("gate_spec")
    if authority.get("gate") != gate or not isinstance(value, Mapping) or set(value) != {
        "scene_inputs",
        "outcome_paths",
        "ledger_path",
        "metric_rules",
    }:
        raise ValueError(f"{gate} runner authority fields changed")
    return value


def _finalizer_evidence_spec(
    authority: Mapping[str, Any],
    gate: str,
) -> Mapping[str, Any]:
    if authority.get("gate") != gate:
        raise PermissionError("finalizer authority gate changed")
    for name in ("runner_authority", "runner_ledger"):
        value = authority.get(name)
        if not isinstance(value, Mapping) or set(value) != {"path", "file_sha256"}:
            raise ValueError(f"finalizer {name} binding changed")
    outcomes = authority.get("outcome_files")
    if not isinstance(outcomes, Mapping):
        raise ValueError("finalizer outcome-file binding changed")
    _execution_identity(
        authority.get("runner_execution_identity"),
        stage="runner",
    )
    _relative(authority.get("final_report_path"), name="final report path")
    return authority


def _publisher_gate_bindings(
    authority: Mapping[str, Any],
    mode: str,
) -> Mapping[str, Any]:
    value = authority.get("finalized_gates")
    expected = {"g2"} if mode == "g2-candidate" else {"g2", "g3"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PermissionError("publisher finalized-gate inventory changed")
    for gate in sorted(expected):
        binding = value[gate]
        if not isinstance(binding, Mapping) or set(binding) != {
            "finalizer_authority",
            "fixed_final_report",
            "finalizer_execution_identity",
        }:
            raise ValueError(f"publisher {gate} binding changed")
        for name in ("finalizer_authority", "fixed_final_report"):
            spec = binding[name]
            if not isinstance(spec, Mapping) or set(spec) != {"path", "file_sha256"}:
                raise ValueError(f"publisher {gate} {name} binding changed")
        _execution_identity(
            binding.get("finalizer_execution_identity"),
            stage="finalizer",
        )
    return value


def _validate_publication(
    value: Mapping[str, Any],
    *,
    mode: str,
    synthetic: bool,
) -> None:
    expected_gates = ["g2"] if mode == "g2-candidate" else ["g2", "g3"]
    expected_pending = ["g3"] if mode == "g2-candidate" else []
    expected_fields = {
        "schema",
        "publication_kind",
        "evaluated_checkpoint",
        "dataset_role_manifest",
        "final_reports",
        "satisfied_gates",
        "pending_gates",
        "g3_evaluation_eligible",
        "full_promotion_eligible",
        "publisher_authority_file_sha256",
        "publisher_execution_identity",
        "synthetic_only",
        "production_authority_eligible",
        "content_sha256",
    }
    reports = value.get("final_reports")
    if (
        set(value) != expected_fields
        or value.get("schema") != PUBLICATION_SCHEMA
        or value.get("publication_kind") != mode.replace("-", "_")
        or value.get("satisfied_gates") != expected_gates
        or value.get("pending_gates") != expected_pending
        or value.get("synthetic_only") is not synthetic
        or value.get("production_authority_eligible") is not (not synthetic)
        or value.get("g3_evaluation_eligible") is not (not synthetic)
        or value.get("full_promotion_eligible")
        is not (mode == "full-promotion" and not synthetic)
        or not isinstance(reports, Mapping)
        or set(reports) != set(expected_gates)
    ):
        raise PermissionError("publication lifecycle state changed")
    checkpoint = value.get("evaluated_checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "file_sha256",
    }:
        raise ValueError("publication checkpoint binding changed")
    _relative(checkpoint.get("path"), name="publication checkpoint path")
    _sha256(checkpoint.get("file_sha256"), name="publication checkpoint")
    _sha256(
        value.get("publisher_authority_file_sha256"),
        name="publisher authority",
    )
    manifest = value.get("dataset_role_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "path",
        "file_sha256",
        "content_sha256",
        "protocol_generation",
    }:
        raise ValueError("publication role-manifest binding changed")
    _relative(manifest.get("path"), name="publication role-manifest path")
    _sha256(manifest.get("file_sha256"), name="publication role-manifest file")
    _sha256(
        manifest.get("content_sha256"),
        name="publication role-manifest content",
    )
    if type(manifest.get("protocol_generation")) is not str or not manifest.get(
        "protocol_generation"
    ):
        raise ValueError("publication role-manifest protocol changed")
    _execution_identity(value.get("publisher_execution_identity"), stage="publisher")
    for gate in expected_gates:
        report = reports[gate]
        if not isinstance(report, Mapping) or set(report) != {
            "path",
            "file_sha256",
            "content_sha256",
            "finalizer_authority_file_sha256",
        }:
            raise ValueError(f"publication {gate} report binding changed")
        _relative(report.get("path"), name=f"publication {gate} report path")
        for name in (
            "file_sha256",
            "content_sha256",
            "finalizer_authority_file_sha256",
        ):
            _sha256(report.get(name), name=f"publication {gate} {name}")


def _reserve(root_fd: int, authority: Mapping[str, Any], gate: str) -> tuple[str, str]:
    registry = _relative(authority.get("attempt_registry_path"), name="attempt registry path")
    role_spec = authority.get("dataset_role_manifest")
    if not isinstance(role_spec, Mapping):
        raise ValueError("role-manifest authority changed")
    namespace = _canonical_sha256(
        {
            "schema": "lewm_go2_shared_jepa_role_global_namespace_v7",
            "gate": gate,
            "dataset_role_manifest_file_sha256": _sha256(role_spec.get("file_sha256"), name="role manifest"),
            "protocol_generation": authority.get("protocol_generation"),
        }
    )
    gate_parent, gate_leaf = _walk_parent(root_fd, f"{registry}/{gate}")
    gate_fd: int | None = None
    namespace_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        gate_fd = os.open(gate_leaf, flags, dir_fd=gate_parent)
        try:
            os.mkdir(namespace, mode=0o700, dir_fd=gate_fd)
        except FileExistsError as exc:
            raise PermissionError(f"{gate} role-global attempt was already consumed") from exc
        namespace_fd = os.open(namespace, flags, dir_fd=gate_fd)
        reservation = _json_output(
            {
                "schema": "lewm_go2_shared_jepa_runner_reservation_v7",
                "gate": gate,
                "namespace_sha256": namespace,
                "dataset_role_manifest_file_sha256": role_spec["file_sha256"],
                "protocol_generation": authority.get("protocol_generation"),
            }
        )
        descriptor = os.open(
            "reservation.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=namespace_fd,
        )
        try:
            view = memoryview(reservation)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("reservation write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(namespace_fd)
        os.fsync(gate_fd)
        return namespace, hashlib.sha256(reservation).hexdigest()
    finally:
        if namespace_fd is not None:
            os.close(namespace_fd)
        if gate_fd is not None:
            os.close(gate_fd)
        os.close(gate_parent)


def _run(
    gate: str,
    production_authorities: Mapping[str, tuple[Path, str | None]],
    execution_identity: Mapping[str, object],
) -> dict[str, object]:
    current_execution = _execution_identity(execution_identity, stage="runner")
    authority, root, synthetic, authority_file_sha256 = _authority(
        RUNNER_REVISIONS[gate],
        production_authorities,
    )
    root_fd = _root_fd(root)
    events: list[dict[str, Any]] = []
    try:
        gate_spec = _runner_gate_spec(authority, gate)
        scene_specs = gate_spec["scene_inputs"]
        outcome_paths = gate_spec["outcome_paths"]
        if (
            not isinstance(scene_specs, Mapping)
            or not isinstance(outcome_paths, Mapping)
            or not scene_specs
            or set(scene_specs) != set(outcome_paths)
            or any(type(scene_id) is not str or not scene_id for scene_id in scene_specs)
        ):
            raise PermissionError("authorized scene input/output inventory changed")
        normalized_outputs = {
            scene_id: _relative(outcome_paths[scene_id], name="raw outcome path")
            for scene_id in sorted(outcome_paths)
        }
        ledger_path = _relative(gate_spec["ledger_path"], name="runner ledger path")
        if len(set(normalized_outputs.values()) | {ledger_path}) != len(
            normalized_outputs
        ) + 1:
            raise PermissionError("runner output paths overlap")
        input_paths = {
            _relative(authority["dataset_role_manifest"]["path"], name="role path"),
            _relative(authority["evaluated_checkpoint"]["path"], name="checkpoint path"),
        }
        input_paths.update(
            _relative(spec["path"], name="runtime source path")
            for spec in authority["runtime_modules"].values()
        )
        input_paths.update(
            _relative(spec["path"], name="raw scene path")
            for spec in scene_specs.values()
        )
        if gate == "g3":
            input_paths.add(
                _relative(
                    authority["g2_candidate_publication"]["path"],
                    name="G2 candidate publication path",
                )
            )
            input_paths.add(
                _relative(
                    authority["g2_candidate_publisher_authority"]["path"],
                    name="G2 candidate publisher authority path",
                )
            )
            _execution_identity(
                authority.get("g2_candidate_publisher_execution_identity"),
                stage="publisher",
            )
        if (set(normalized_outputs.values()) | {ledger_path}) & input_paths:
            raise PermissionError("runner output path aliases a frozen input")
        for path in (*normalized_outputs.values(), ledger_path):
            _require_absent_at(root_fd, path, name="runner output")
        namespace, reservation_file_sha256 = _reserve(root_fd, authority, gate)
        candidate_predecessor: dict[str, object] | None = None
        if gate == "g3":
            candidate_predecessor, candidate_evidence = _reopen_g2_candidate(
                root_fd,
                authority,
                root=root,
                synthetic=synthetic,
            )
            for artifact, path, file_hash in candidate_evidence:
                events.append(
                    {
                        "sequence": len(events) + 1,
                        "operation": "verify_g2_candidate_evidence",
                        "artifact": artifact,
                        "path": path,
                        "file_sha256": file_hash,
                        "forbidden": False,
                    }
                )
        runtime = _captured_runtime(root_fd, authority, events)
        role_encoded, role_path, role_hash = _read_at(
            root_fd, authority["dataset_role_manifest"], name="dataset role manifest"
        )
        events.append({"sequence": len(events) + 1, "operation": "open_dataset_role_manifest", "path": role_path, "file_sha256": role_hash, "forbidden": False})
        role_value = _strict_object(role_encoded, name="dataset role manifest")
        roles, families = _roles(
            role_value,
            expected_protocol_generation=authority["protocol_generation"],
        )
        manifest_identity = _role_manifest_identity(
            role_value,
            path=role_path,
            file_sha256=role_hash,
            protocol_generation=authority["protocol_generation"],
        )
        if set(scene_specs) != set(roles[gate]):
            raise PermissionError("authorized scene input/output inventory changed")
        checkpoint, checkpoint_path, checkpoint_hash = _read_at(
            root_fd, authority["evaluated_checkpoint"], name="evaluated checkpoint"
        )
        events.append({"sequence": len(events) + 1, "operation": "open_evaluated_checkpoint", "path": checkpoint_path, "file_sha256": checkpoint_hash, "forbidden": False})
        load_checkpoint = getattr(runtime, "load_checkpoint", None)
        infer_one = getattr(runtime, "infer_one", None)
        if not callable(load_checkpoint) or not callable(infer_one):
            raise TypeError("captured inference entry lacks load_checkpoint/infer_one")
        model = load_checkpoint(checkpoint)
        inference_count = 0
        written: dict[str, dict[str, str]] = {}
        for scene_id in roles[gate]:
            encoded, scene_path, scene_hash = _read_at(root_fd, scene_specs[scene_id], name=f"raw {gate} scene {scene_id}")
            scene = _strict_object(encoded, name=f"raw scene {scene_id}")
            if set(scene) != {"schema", "scene_id", "family", "instances", "content_sha256"} or scene.get("schema") != RAW_SCENE_SCHEMA or scene.get("scene_id") != scene_id or scene.get("family") != families[scene_id]:
                raise PermissionError("raw scene identity changed")
            instances = scene.get("instances")
            if not isinstance(instances, list) or not instances:
                raise ValueError("raw scene has no inference instances")
            identifiers = [item.get("instance_id") if isinstance(item, Mapping) else None for item in instances]
            if any(type(item) is not str or not item for item in identifiers) or identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
                raise ValueError("raw scene instances are not sorted and unique")
            rows = []
            for instance in instances:
                if set(instance) != {"instance_id", "model_input", "targets"} or not isinstance(instance["targets"], Mapping):
                    raise ValueError("raw inference instance fields changed")
                output = infer_one(model, instance["model_input"])
                inference_count += 1
                output_bytes = _canonical_bytes(output)
                rows.append(
                    {
                        "instance_id": instance["instance_id"],
                        "inference_output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                        "metric_outcomes": _metric_outcomes(output, instance["targets"], gate_spec["metric_rules"], GATE_METRICS[gate]),
                    }
                )
            outcome = _json_output(
                {
                    "schema": RAW_OUTCOME_SCHEMA,
                    "gate": gate,
                    "scene_id": scene_id,
                    "family": families[scene_id],
                    "raw_scene_input_file_sha256": scene_hash,
                    "evaluated_checkpoint_file_sha256": checkpoint_hash,
                    "instances": rows,
                    "synthetic_only": synthetic,
                    "production_authority_eligible": not synthetic,
                }
            )
            outcome_path = normalized_outputs[scene_id]
            outcome_hash = _write_exclusive(root_fd, outcome_path, outcome)
            written[scene_id] = {"path": outcome_path, "file_sha256": outcome_hash}
            events.append(
                {
                    "sequence": len(events) + 1,
                    "operation": "open_raw_scene_and_run_each_instance",
                    "scene_id": scene_id,
                    "path": scene_path,
                    "file_sha256": scene_hash,
                    "instance_count": len(instances),
                    "inference_count": len(instances),
                    "forbidden": False,
                }
            )
        expected_count = sum(event.get("instance_count", 0) for event in events)
        if inference_count != expected_count:
            raise RuntimeError("model inference cardinality changed")
        ledger = _json_output(
            {
                "schema": LEDGER_SCHEMA,
                "gate": gate,
                "runner_authority_file_sha256": authority_file_sha256,
                "runner_execution_identity": current_execution,
                "namespace_sha256": namespace,
                "reservation_file_sha256": reservation_file_sha256,
                "dataset_role_manifest": manifest_identity,
                "evaluated_checkpoint_file_sha256": checkpoint_hash,
                "g2_candidate_predecessor": candidate_predecessor,
                "events": events,
                "scene_outcome_files": written,
                "total_instance_count": expected_count,
                "total_inference_count": inference_count,
                "synthetic_only": synthetic,
                "production_authority_eligible": not synthetic,
            }
        )
        ledger_hash = _write_exclusive(root_fd, ledger_path, ledger)
        return {
            "stage": "runner",
            "gate": gate,
            "runner_authority_file_sha256": authority_file_sha256,
            "runner_execution_identity": current_execution,
            "ledger_path": ledger_path,
            "ledger_file_sha256": ledger_hash,
            "total_inference_count": inference_count,
            "synthetic_only": synthetic,
            "production_authority_eligible": not synthetic,
        }
    finally:
        os.close(root_fd)


def _reconstruct_gate_report_core(
    root_fd: int,
    runner_authority: Mapping[str, Any],
    finalizer_authority: Mapping[str, Any],
    gate: str,
    synthetic: bool,
    *,
    root: Path,
    runner_authority_path: str,
    runner_authority_file_sha256: str,
    finalizer_authority_file_sha256: str,
    finalizer_execution_identity: Mapping[str, object],
    opened_artifacts: list[tuple[str, str, str]] | None = None,
) -> tuple[dict[str, object], str]:
    """Reopen fixed evidence and independently derive one complete gate report."""

    gate_spec = _runner_gate_spec(runner_authority, gate)
    evidence_spec = _finalizer_evidence_spec(finalizer_authority, gate)
    role_spec = runner_authority.get("dataset_role_manifest")
    if not isinstance(role_spec, Mapping):
        raise ValueError("dataset role-manifest authority changed")
    role_encoded, _, role_hash = _read_at(
        root_fd,
        role_spec,
        name="dataset role manifest",
    )
    if opened_artifacts is not None:
        opened_artifacts.append(
            (
                "dataset_role_manifest",
                _relative(role_spec.get("path"), name="role manifest path"),
                role_hash,
            )
        )
    role_value = _strict_object(role_encoded, name="dataset role manifest")
    roles, families = _roles(
        role_value,
        expected_protocol_generation=runner_authority["protocol_generation"],
    )
    manifest_identity = _role_manifest_identity(
        role_value,
        path=_relative(role_spec.get("path"), name="role manifest path"),
        file_sha256=role_hash,
        protocol_generation=runner_authority["protocol_generation"],
    )
    ledger_spec = evidence_spec["runner_ledger"]
    ledger_encoded, ledger_path, ledger_hash = _read_at(
        root_fd,
        ledger_spec,
        name=f"{gate} runner ledger",
    )
    if opened_artifacts is not None:
        opened_artifacts.append(("runner_ledger", ledger_path, ledger_hash))
    ledger = _strict_object(ledger_encoded, name=f"{gate} runner ledger")
    expected_ledger_fields = {
        "schema",
        "gate",
        "runner_authority_file_sha256",
        "runner_execution_identity",
        "namespace_sha256",
        "reservation_file_sha256",
        "dataset_role_manifest",
        "evaluated_checkpoint_file_sha256",
        "g2_candidate_predecessor",
        "events",
        "scene_outcome_files",
        "total_instance_count",
        "total_inference_count",
        "synthetic_only",
        "production_authority_eligible",
        "content_sha256",
    }
    if (
        set(ledger) != expected_ledger_fields
        or ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("gate") != gate
        or ledger.get("runner_authority_file_sha256")
        != runner_authority_file_sha256
        or ledger.get("runner_execution_identity")
        != finalizer_authority.get("runner_execution_identity")
        or ledger.get("dataset_role_manifest") != manifest_identity
        or ledger.get("synthetic_only") is not synthetic
        or ledger.get("production_authority_eligible") is synthetic
    ):
        raise PermissionError("runner ledger authority changed")
    _sha256(ledger.get("namespace_sha256"), name="runner namespace")
    _sha256(ledger.get("reservation_file_sha256"), name="runner reservation")
    bindings = ledger.get("scene_outcome_files")
    if not isinstance(bindings, Mapping) or set(bindings) != set(roles[gate]):
        raise PermissionError("runner outcome inventory changed")
    events = ledger.get("events")
    if not isinstance(events, list):
        raise PermissionError("runner open ledger events changed")

    runtime_modules = runner_authority.get("runtime_modules")
    if not isinstance(runtime_modules, Mapping) or not runtime_modules:
        raise ValueError("captured runtime module inventory changed")
    captured_sources: dict[str, dict[str, object]] = {}
    expected_events: list[dict[str, object]] = []
    expected_candidate_predecessor: dict[str, object] | None = None
    if gate == "g3":
        expected_candidate_predecessor, candidate_evidence = _reopen_g2_candidate(
            root_fd,
            runner_authority,
            root=root,
            synthetic=synthetic,
        )
        for artifact, path, file_hash in candidate_evidence:
            expected_events.append(
                {
                    "sequence": len(expected_events) + 1,
                    "operation": "verify_g2_candidate_evidence",
                    "artifact": artifact,
                    "path": path,
                    "file_sha256": file_hash,
                    "forbidden": False,
                }
            )
    if ledger.get("g2_candidate_predecessor") != expected_candidate_predecessor:
        raise PermissionError("runner ledger G2-candidate binding changed")
    for name in sorted(runtime_modules):
        source_spec = runtime_modules[name]
        if (
            type(name) is not str
            or not name
            or not isinstance(source_spec, Mapping)
            or set(source_spec) != {"path", "file_sha256", "package"}
            or type(source_spec.get("package")) is not bool
        ):
            raise ValueError("captured runtime source binding changed")
        normalized_source = {
            "path": _relative(source_spec.get("path"), name="captured source path"),
            "file_sha256": _sha256(
                source_spec.get("file_sha256"),
                name="captured source",
            ),
            "package": source_spec["package"],
        }
        captured_sources[name] = normalized_source
        expected_events.append(
            {
                "sequence": len(expected_events) + 1,
                "operation": "capture_runtime_source",
                "path": normalized_source["path"],
                "file_sha256": normalized_source["file_sha256"],
                "forbidden": False,
            }
        )
    checkpoint_spec = runner_authority.get("evaluated_checkpoint")
    if not isinstance(checkpoint_spec, Mapping) or set(checkpoint_spec) != {
        "path",
        "file_sha256",
    }:
        raise ValueError("evaluated-checkpoint authority changed")
    checkpoint_path = _relative(
        checkpoint_spec.get("path"),
        name="evaluated checkpoint path",
    )
    checkpoint_hash = _sha256(
        checkpoint_spec.get("file_sha256"),
        name="evaluated checkpoint",
    )
    if ledger.get("evaluated_checkpoint_file_sha256") != checkpoint_hash:
        raise PermissionError("runner ledger checkpoint binding changed")
    expected_events.extend(
        (
            {
                "sequence": len(expected_events) + 1,
                "operation": "open_dataset_role_manifest",
                "path": _relative(role_spec.get("path"), name="role manifest path"),
                "file_sha256": role_hash,
                "forbidden": False,
            },
            {
                "sequence": len(expected_events) + 2,
                "operation": "open_evaluated_checkpoint",
                "path": checkpoint_path,
                "file_sha256": checkpoint_hash,
                "forbidden": False,
            },
        )
    )

    scene_input_specs = gate_spec.get("scene_inputs")
    outcome_specs = evidence_spec.get("outcome_files")
    if (
        not isinstance(scene_input_specs, Mapping)
        or set(scene_input_specs) != set(roles[gate])
        or not isinstance(outcome_specs, Mapping)
        or set(outcome_specs) != set(roles[gate])
    ):
        raise ValueError("fixed scene evidence inventory changed")
    per_family: dict[str, dict[str, dict[str, int]]] = {}
    outcome_hashes: list[str] = []
    total_instances = 0
    for scene_id in roles[gate]:
        source = scene_input_specs[scene_id]
        expected = outcome_specs[scene_id]
        if (
            not isinstance(source, Mapping)
            or set(source) != {"path", "file_sha256"}
            or not isinstance(expected, Mapping)
            or set(expected) != {"path", "file_sha256"}
            or dict(expected) != dict(bindings[scene_id])
        ):
            raise PermissionError("fixed scene evidence binding changed")
        encoded, _, file_hash = _read_at(
            root_fd,
            expected,
            name=f"raw outcome {scene_id}",
        )
        if opened_artifacts is not None:
            opened_artifacts.append(
                (
                    f"raw_outcome:{scene_id}",
                    _relative(expected.get("path"), name="raw outcome path"),
                    file_hash,
                )
            )
        outcome = _strict_object(encoded, name=f"raw outcome {scene_id}")
        expected_outcome_fields = {
            "schema",
            "gate",
            "scene_id",
            "family",
            "raw_scene_input_file_sha256",
            "evaluated_checkpoint_file_sha256",
            "instances",
            "synthetic_only",
            "production_authority_eligible",
            "content_sha256",
        }
        if (
            set(outcome) != expected_outcome_fields
            or outcome.get("schema") != RAW_OUTCOME_SCHEMA
            or outcome.get("gate") != gate
            or outcome.get("scene_id") != scene_id
            or outcome.get("family") != families[scene_id]
            or outcome.get("synthetic_only") is not synthetic
            or outcome.get("production_authority_eligible") is synthetic
            or outcome.get("raw_scene_input_file_sha256")
            != source.get("file_sha256")
            or outcome.get("evaluated_checkpoint_file_sha256") != checkpoint_hash
        ):
            raise PermissionError("raw outcome authority changed")
        instances = outcome.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("raw outcome instances are empty")
        identifiers = []
        family = per_family.setdefault(families[scene_id], {})
        for instance in instances:
            if (
                not isinstance(instance, Mapping)
                or set(instance)
                != {"instance_id", "inference_output_sha256", "metric_outcomes"}
                or type(instance.get("instance_id")) is not str
                or not instance["instance_id"]
                or not isinstance(instance.get("metric_outcomes"), Mapping)
                or set(instance["metric_outcomes"]) != set(GATE_METRICS[gate])
            ):
                raise ValueError("raw outcome inference fields changed")
            identifiers.append(instance["instance_id"])
            _sha256(instance["inference_output_sha256"], name="inference output")
            for metric in GATE_METRICS[gate]:
                metric_passed = instance["metric_outcomes"][metric]
                if type(metric_passed) is not bool:
                    raise ValueError("raw metric outcome must be boolean")
                counts = family.setdefault(
                    metric,
                    {"numerator": 0, "denominator": 0},
                )
                counts["numerator"] += int(metric_passed)
                counts["denominator"] += 1
        if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("raw outcome instances are not sorted and unique")
        instance_count = len(instances)
        total_instances += instance_count
        expected_events.append(
            {
                "sequence": len(expected_events) + 1,
                "operation": "open_raw_scene_and_run_each_instance",
                "scene_id": scene_id,
                "path": _relative(source.get("path"), name="raw scene path"),
                "file_sha256": _sha256(
                    source.get("file_sha256"),
                    name="raw scene",
                ),
                "instance_count": instance_count,
                "inference_count": instance_count,
                "forbidden": False,
            }
        )
        outcome_hashes.append(file_hash)
    if events != expected_events:
        raise PermissionError("runner open ledger does not reproduce reopened evidence")
    if (
        type(ledger.get("total_instance_count")) is not int
        or type(ledger.get("total_inference_count")) is not int
        or ledger.get("total_instance_count") != total_instances
        or ledger.get("total_inference_count") != total_instances
    ):
        raise PermissionError("runner inference cardinality does not match reopened outcomes")

    metrics: dict[str, float] = {}
    for metric in GATE_METRICS[gate]:
        numerator = sum(row[metric]["numerator"] for row in per_family.values())
        denominator = sum(row[metric]["denominator"] for row in per_family.values())
        metrics[metric] = numerator / denominator
    passed = all(value >= 1.0 for value in metrics.values())
    core: dict[str, object] = {
        "schema": FINAL_REPORT_SCHEMA,
        "gate": gate,
        "passed": passed,
        "metrics": metrics,
        "per_family_counts": per_family,
        "dataset_role_manifest": manifest_identity,
        "evaluated_checkpoint_file_sha256": checkpoint_hash,
        "captured_runtime_sources": captured_sources,
        "runner_authority_path": runner_authority_path,
        "runner_authority_file_sha256": runner_authority_file_sha256,
        "runner_execution_identity": finalizer_authority["runner_execution_identity"],
        "finalizer_authority_file_sha256": finalizer_authority_file_sha256,
        "finalizer_execution_identity": _execution_identity(
            finalizer_execution_identity,
            stage="finalizer",
        ),
        "runner_ledger_path": ledger_path,
        "runner_ledger_file_sha256": ledger_hash,
        "raw_scene_outcome_file_sha256s": outcome_hashes,
        "total_instance_count": total_instances,
        "g2_candidate_predecessor": expected_candidate_predecessor,
        "synthetic_only": synthetic,
        "production_authority_eligible": not synthetic,
    }
    return core, _relative(
        evidence_spec["final_report_path"],
        name="final report path",
    )


def _finalize(
    gate: str,
    production_authorities: Mapping[str, tuple[Path, str | None]],
    execution_identity: Mapping[str, object],
) -> dict[str, object]:
    current_execution = _execution_identity(execution_identity, stage="finalizer")
    authority, root, synthetic, authority_file_sha256 = _authority(
        FINALIZER_REVISIONS[gate],
        production_authorities,
    )
    root_fd = _root_fd(root)
    try:
        evidence_spec = _finalizer_evidence_spec(authority, gate)
        report_path = _relative(
            evidence_spec["final_report_path"],
            name="final report path",
        )
        _require_absent_at(root_fd, report_path, name="final report")
        runner_authority, runner_authority_path, runner_authority_hash = (
            _linked_authority(
                root_fd,
                evidence_spec["runner_authority"],
                revision=RUNNER_REVISIONS[gate],
                synthetic=synthetic,
                root=root,
                protocol_generation=authority["protocol_generation"],
                name=f"{gate} runner authority",
            )
        )
        report_core, report_path = _reconstruct_gate_report_core(
            root_fd,
            runner_authority,
            authority,
            gate,
            synthetic,
            root=root,
            runner_authority_path=runner_authority_path,
            runner_authority_file_sha256=runner_authority_hash,
            finalizer_authority_file_sha256=authority_file_sha256,
            finalizer_execution_identity=current_execution,
        )
        frozen_inputs = {
            _relative(evidence_spec["runner_authority"]["path"], name="runner authority path"),
            _relative(evidence_spec["runner_ledger"]["path"], name="runner ledger path"),
        }
        frozen_inputs.update(
            _relative(spec["path"], name="outcome path")
            for spec in evidence_spec["outcome_files"].values()
        )
        if report_path in frozen_inputs:
            raise PermissionError("final report path aliases fixed runner evidence")
        report = _json_output(report_core)
        report_hash = _write_exclusive(root_fd, report_path, report)
        return {
            "stage": "finalizer",
            "gate": gate,
            "finalizer_authority_file_sha256": authority_file_sha256,
            "finalizer_execution_identity": current_execution,
            "passed": report_core["passed"],
            "final_report_path": report_path,
            "final_report_file_sha256": report_hash,
            "synthetic_only": synthetic,
            "production_authority_eligible": not synthetic,
        }
    finally:
        os.close(root_fd)


def _publication_core(
    root_fd: int,
    authority: Mapping[str, Any],
    *,
    root: Path,
    synthetic: bool,
    authority_file_sha256: str,
    mode: str,
    publisher_execution_identity: Mapping[str, object],
    opened_artifacts: list[tuple[str, str, str]] | None = None,
) -> tuple[dict[str, object], str, set[str]]:
    bindings = _publisher_gate_bindings(authority, mode)
    reports: dict[str, dict[str, str]] = {}
    checkpoint_spec: Mapping[str, object] | None = None
    checkpoint_path: str | None = None
    checkpoint_hash: str | None = None
    role_manifest_identity: Mapping[str, object] | None = None
    frozen_inputs: set[str] = set()
    for gate in sorted(bindings):
        binding = bindings[gate]
        finalizer_authority, finalizer_path, finalizer_hash = _linked_authority(
            root_fd,
            binding["finalizer_authority"],
            revision=FINALIZER_REVISIONS[gate],
            synthetic=synthetic,
            root=root,
            protocol_generation=authority["protocol_generation"],
            name=f"{gate} finalizer authority",
        )
        if opened_artifacts is not None:
            opened_artifacts.append(
                (f"{gate}_finalizer_authority", finalizer_path, finalizer_hash)
            )
        evidence_spec = _finalizer_evidence_spec(finalizer_authority, gate)
        runner_authority, runner_path, runner_hash = _linked_authority(
            root_fd,
            evidence_spec["runner_authority"],
            revision=RUNNER_REVISIONS[gate],
            synthetic=synthetic,
            root=root,
            protocol_generation=authority["protocol_generation"],
            name=f"{gate} runner authority",
        )
        if opened_artifacts is not None:
            opened_artifacts.append(
                (f"{gate}_runner_authority", runner_path, runner_hash)
            )
        reconstructed, expected_path = _reconstruct_gate_report_core(
            root_fd,
            runner_authority,
            finalizer_authority,
            gate,
            synthetic,
            root=root,
            runner_authority_path=runner_path,
            runner_authority_file_sha256=runner_hash,
            finalizer_authority_file_sha256=finalizer_hash,
            finalizer_execution_identity=binding["finalizer_execution_identity"],
            opened_artifacts=opened_artifacts,
        )
        current_manifest = reconstructed.get("dataset_role_manifest")
        if not isinstance(current_manifest, Mapping):
            raise PermissionError("final report role-manifest binding changed")
        if role_manifest_identity is None:
            role_manifest_identity = current_manifest
        elif dict(current_manifest) != dict(role_manifest_identity):
            raise PermissionError("G2 and G3 use different dataset role manifests")
        current_checkpoint = runner_authority["evaluated_checkpoint"]
        if checkpoint_spec is None:
            checkpoint_spec = current_checkpoint
            checkpoint, checkpoint_path, checkpoint_hash = _read_at(
                root_fd,
                current_checkpoint,
                name="evaluated checkpoint",
            )
            if opened_artifacts is not None:
                opened_artifacts.append(
                    ("evaluated_checkpoint", checkpoint_path, checkpoint_hash)
                )
            del checkpoint
        elif dict(current_checkpoint) != dict(checkpoint_spec):
            raise PermissionError("G2 and G3 use different evaluated checkpoints")
        if (
            reconstructed.get("passed") is not True
            or reconstructed.get("evaluated_checkpoint_file_sha256") != checkpoint_hash
        ):
            raise PermissionError(f"{gate} reconstructed evidence does not pass")
        fixed = binding["fixed_final_report"]
        encoded, path, file_hash = _read_at(
            root_fd,
            fixed,
            name=f"{gate} final report",
        )
        if opened_artifacts is not None:
            opened_artifacts.append((f"{gate}_final_report", path, file_hash))
        report = _strict_object(encoded, name=f"{gate} final report")
        expected_encoded = _json_output(reconstructed)
        if (
            path != expected_path
            or encoded != expected_encoded
            or report
            != _strict_object(
                expected_encoded,
                name=f"reconstructed {gate} final report",
            )
        ):
            raise PermissionError(
                f"{gate} final report does not reproduce its fixed evidence"
            )
        reports[gate] = {
            "path": path,
            "file_sha256": file_hash,
            "content_sha256": report["content_sha256"],
            "finalizer_authority_file_sha256": finalizer_hash,
        }
        if gate == "g3":
            predecessor = reconstructed.get("g2_candidate_predecessor")
            if (
                not isinstance(predecessor, Mapping)
                or predecessor.get("g2_final_report") != reports.get("g2")
            ):
                raise PermissionError(
                    "full promotion G2 report differs from G3 predecessor"
                )
        frozen_inputs.update((finalizer_path, runner_path, path))
    if (
        checkpoint_path is None
        or checkpoint_hash is None
        or role_manifest_identity is None
    ):
        raise RuntimeError("publisher has no finalized checkpoint evidence")
    frozen_inputs.add(checkpoint_path)
    publication_core: dict[str, object] = {
        "schema": PUBLICATION_SCHEMA,
        "publication_kind": mode.replace("-", "_"),
        "evaluated_checkpoint": {
            "path": checkpoint_path,
            "file_sha256": checkpoint_hash,
        },
        "dataset_role_manifest": dict(role_manifest_identity),
        "final_reports": reports,
        "satisfied_gates": sorted(reports),
        "pending_gates": ["g3"] if mode == "g2-candidate" else [],
        "g3_evaluation_eligible": not synthetic,
        "full_promotion_eligible": mode == "full-promotion" and not synthetic,
        "publisher_authority_file_sha256": authority_file_sha256,
        "publisher_execution_identity": _execution_identity(
            publisher_execution_identity,
            stage="publisher",
        ),
        "synthetic_only": synthetic,
        "production_authority_eligible": not synthetic,
    }
    publication_path = _relative(
        authority.get("publication_path"),
        name="publication path",
    )
    return publication_core, publication_path, frozen_inputs


def _reopen_g2_candidate(
    root_fd: int,
    runner_authority: Mapping[str, Any],
    *,
    root: Path,
    synthetic: bool,
) -> tuple[dict[str, object], list[tuple[str, str, str]]]:
    publisher_authority, publisher_path, publisher_hash = _linked_authority(
        root_fd,
        runner_authority.get("g2_candidate_publisher_authority"),
        revision=PUBLISHER_REVISIONS["g2-candidate"],
        synthetic=synthetic,
        root=root,
        protocol_generation=runner_authority["protocol_generation"],
        name="G2 candidate publisher authority",
    )
    opened_artifacts = [
        ("g2_candidate_publisher_authority", publisher_path, publisher_hash)
    ]
    expected_core, expected_path, _ = _publication_core(
        root_fd,
        publisher_authority,
        root=root,
        synthetic=synthetic,
        authority_file_sha256=publisher_hash,
        mode="g2-candidate",
        publisher_execution_identity=runner_authority[
            "g2_candidate_publisher_execution_identity"
        ],
        opened_artifacts=opened_artifacts,
    )
    encoded, path, file_hash = _read_at(
        root_fd,
        runner_authority.get("g2_candidate_publication"),
        name="G2 candidate publication",
    )
    publication = _strict_object(encoded, name="G2 candidate publication")
    opened_artifacts.append(("g2_candidate_publication", path, file_hash))
    _validate_publication(
        publication,
        mode="g2-candidate",
        synthetic=synthetic,
    )
    if (
        path != expected_path
        or encoded != _json_output(expected_core)
        or dict(publication["evaluated_checkpoint"])
        != dict(runner_authority["evaluated_checkpoint"])
        or publication["publisher_execution_identity"]
        != runner_authority["g2_candidate_publisher_execution_identity"]
        or publication["dataset_role_manifest"]["path"]
        != runner_authority["dataset_role_manifest"]["path"]
        or publication["dataset_role_manifest"]["file_sha256"]
        != runner_authority["dataset_role_manifest"]["file_sha256"]
        or publication["dataset_role_manifest"]["protocol_generation"]
        != runner_authority["protocol_generation"]
    ):
        raise PermissionError(
            "G2 candidate publication does not reproduce its fixed G2 evidence"
        )
    predecessor = {
        "publisher_authority": {
            "path": publisher_path,
            "file_sha256": publisher_hash,
        },
        "publisher_execution_identity": publication[
            "publisher_execution_identity"
        ],
        "publication": {
            "path": path,
            "file_sha256": file_hash,
            "content_sha256": publication["content_sha256"],
        },
        "dataset_role_manifest": publication["dataset_role_manifest"],
        "evaluated_checkpoint": publication["evaluated_checkpoint"],
        "g2_final_report": publication["final_reports"]["g2"],
    }
    return predecessor, opened_artifacts


def _publish(
    mode: str,
    production_authorities: Mapping[str, tuple[Path, str | None]],
    execution_identity: Mapping[str, object],
) -> dict[str, object]:
    current_execution = _execution_identity(execution_identity, stage="publisher")
    authority, root, synthetic, authority_file_sha256 = _authority(
        PUBLISHER_REVISIONS[mode],
        production_authorities,
    )
    root_fd = _root_fd(root)
    try:
        preflight_publication_path = _relative(
            authority.get("publication_path"),
            name="publication path",
        )
        _require_absent_at(
            root_fd,
            preflight_publication_path,
            name="publication",
        )
        publication_core, publication_path, frozen_inputs = _publication_core(
            root_fd,
            authority,
            root=root,
            synthetic=synthetic,
            authority_file_sha256=authority_file_sha256,
            mode=mode,
            publisher_execution_identity=current_execution,
        )
        publication_value = _content(publication_core)
        _validate_publication(
            publication_value,
            mode=mode,
            synthetic=synthetic,
        )
        publication = _canonical_bytes(publication_value) + b"\n"
        if publication_path in frozen_inputs:
            raise PermissionError("publication path aliases fixed evidence")
        publication_hash = _write_exclusive(root_fd, publication_path, publication)
        return {
            "stage": "publisher",
            "publication_kind": mode.replace("-", "_"),
            "publication_path": publication_path,
            "publication_file_sha256": publication_hash,
            "publisher_execution_identity": current_execution,
            "g3_evaluation_eligible": not synthetic,
            "full_promotion_eligible": mode == "full-promotion" and not synthetic,
            "synthetic_only": synthetic,
            "production_authority_eligible": not synthetic,
        }
    finally:
        os.close(root_fd)


def main_for_stage(
    stage: str,
    argv: Sequence[str] | None = None,
    *,
    production_authorities: object,
    execution_identity: object,
) -> int:
    if stage not in {"runner", "finalizer", "publisher"}:
        raise ValueError("unknown V5 one-shot stage")
    fixed_authorities = _production_authority_inventory(
        production_authorities,
        stage=stage,
    )
    fixed_execution = _execution_identity(execution_identity, stage=stage)
    parser = argparse.ArgumentParser(prog=f"go2-shared-jepa-v5-{stage}")
    if stage in {"runner", "finalizer"}:
        parser.add_argument("gate", choices=("g2", "g3"))
    elif stage == "publisher":
        parser.add_argument(
            "publication",
            choices=("g2-candidate", "full-promotion"),
        )
    args = parser.parse_args(argv)
    result = (
        _run(args.gate, fixed_authorities, fixed_execution)
        if stage == "runner"
        else _finalize(args.gate, fixed_authorities, fixed_execution)
        if stage == "finalizer"
        else _publish(args.publication, fixed_authorities, fixed_execution)
    )
    sys.stdout.buffer.write(_canonical_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit("invoke one of the three fixed captured V5 stage programs")
