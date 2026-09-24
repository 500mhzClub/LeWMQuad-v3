#!/usr/bin/env python3
"""Verify the recursive Python source closure of the RGB multires probe.

The checker parses imports without importing project modules.  Dynamic modules
loaded from reviewed paths are explicit roots.  It never opens generated
artifacts, configurations, datasets, custody material, or sealed benchmarks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_rgb_multiresolution_perception_v1_"
    "source_manifest_2026-07-24.json"
)
SCHEMA = "lewm_go2_rgb_multiresolution_perception_v1_source_manifest"

ENTRYPOINTS = (
    "scripts/launch_go2_shared_jepa_v5_multires_probe_v1.py",
    "scripts/run_go2_shared_jepa_v5_multires_probe_v1.py",
)

# importlib loads these exact reviewed files, so static import discovery cannot
# find the edges from their callers.
FORCED_DYNAMIC_SOURCES = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py",
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
)

PACKAGE_ROOTS = (
    ("lewm", ROOT / "lewm"),
    ("scripts", ROOT / "scripts"),
)

EXCLUDED_RUNTIME_CATEGORIES = (
    ".generated artifacts and attempt registries",
    "tensor checkpoints and metric sidecars",
    "raw RGB, scene shards, datasets, and role payloads",
    "configuration and custody roots",
    "sealed or held-out benchmark material",
    "review, authorization, result, and completion records",
)

FORBIDDEN_PATH_PARTS = {
    ".generated",
    "artifacts",
    "config",
    "configs",
    "custody",
    "data",
    "datasets",
    "checkpoints",
    "generated",
    "sealed",
}
FORBIDDEN_FILE_NAMES = {"sealed_test.json"}
FORBIDDEN_RUNNER_PREFIX = (
    "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read_regular_source(path: Path) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("O_NOFOLLOW is required for source custody")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"source is not regular: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise PermissionError(f"opened source is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not (
        fingerprint(before)
        == fingerprint(opened_before)
        == fingerprint(opened_after)
        == fingerprint(after)
    ):
        raise RuntimeError(f"source changed while read: {path}")
    return b"".join(chunks)


def _safe_source_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.name in FORBIDDEN_FILE_NAMES
        or any(part in FORBIDDEN_PATH_PARTS for part in path.parts)
        or any(part.startswith("sealed_") for part in path.parts)
        or relative.startswith(FORBIDDEN_RUNNER_PREFIX)
        or path.suffix != ".py"
    ):
        raise PermissionError(f"forbidden source-closure path: {relative}")


def _module_index() -> tuple[dict[str, Path], dict[Path, str]]:
    by_module: dict[str, Path] = {}
    by_path: dict[Path, str] = {}
    discovered = subprocess.run(
        [
            "rg",
            "--files",
            "--glob",
            "*.py",
            "--glob",
            "!**/sealed_test.json",
            "--glob",
            "!**/sealed/**",
            "--glob",
            "!**/sealed_*/**",
            "--glob",
            "!**/.generated/**",
            "--glob",
            "!**/artifacts/**",
            "--glob",
            "!**/checkpoints/**",
            "--glob",
            "!**/config/**",
            "--glob",
            "!**/configs/**",
            "--glob",
            "!**/custody/**",
            "--glob",
            "!**/data/**",
            "--glob",
            "!**/datasets/**",
            "--glob",
            "!**/generated/**",
            "--glob",
            "!scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v*.py",
            "lewm",
            "scripts",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if discovered.returncode != 0:
        raise RuntimeError(
            "ignore-honoring Python source discovery failed:\n"
            + discovered.stderr.strip()
        )
    relative_paths = [
        Path(line)
        for line in discovered.stdout.splitlines()
        if line
    ]
    if not relative_paths:
        raise RuntimeError("ignore-honoring Python source discovery was empty")
    for prefix, package_root in PACKAGE_ROOTS:
        if not package_root.is_dir():
            continue
        candidates = sorted(
            ROOT / relative
            for relative in relative_paths
            if (ROOT / relative).is_relative_to(package_root)
        )
        for path in candidates:
            relative_from_root = path.relative_to(ROOT).as_posix()
            _safe_source_path(relative_from_root)
            relative = path.relative_to(package_root)
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join((prefix, *parts)) if parts else prefix
            resolved = path.resolve()
            existing = by_module.get(module)
            if existing is not None and existing != resolved:
                raise RuntimeError(
                    f"duplicate local module {module}: {existing}, {path}"
                )
            by_module[module] = resolved
            by_path[resolved] = module
    return by_module, by_path


def _absolute_import_base(
    *,
    current_module: str,
    current_path: Path,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return node.module or ""
    package = (
        current_module
        if current_path.name == "__init__.py"
        else current_module.rpartition(".")[0]
    )
    parts = package.split(".") if package else []
    if node.level > len(parts):
        return ""
    kept = parts[: len(parts) - node.level + 1]
    if node.module:
        kept.extend(node.module.split("."))
    return ".".join(kept)


def _import_candidates(
    tree: ast.AST,
    *,
    current_module: str,
    current_path: Path,
) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_base(
                current_module=current_module,
                current_path=current_path,
                node=node,
            )
            if base:
                yield base
                for alias in node.names:
                    if alias.name != "*":
                        yield f"{base}.{alias.name}"


def _parent_package_paths(
    module: str,
    by_module: dict[str, Path],
) -> Iterable[Path]:
    parts = module.split(".")
    for length in range(1, len(parts)):
        candidate = by_module.get(".".join(parts[:length]))
        if candidate is not None and candidate.name == "__init__.py":
            yield candidate


def discover_source_closure() -> tuple[str, ...]:
    by_module, by_path = _module_index()
    queue = [
        (ROOT / relative).resolve()
        for relative in (*ENTRYPOINTS, *FORCED_DYNAMIC_SOURCES)
    ]
    visited: set[Path] = set()

    while queue:
        path = queue.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"closure source is absent: {_relative(path)}")
        try:
            current_module = by_path[path]
        except KeyError as error:
            raise RuntimeError(
                f"closure source is outside fixed package roots: {path}"
            ) from error
        relative = _relative(path)
        _safe_source_path(relative)
        visited.add(path)
        try:
            source = _read_regular_source(path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"closure source is not UTF-8: {relative}") from error
        tree = ast.parse(source, filename=str(path))
        for module in _import_candidates(
            tree,
            current_module=current_module,
            current_path=path,
        ):
            dependency = by_module.get(module)
            if dependency is None:
                continue
            queue.append(dependency)
            queue.extend(_parent_package_paths(module, by_module))

    sources = tuple(sorted(_relative(path) for path in visited))
    for relative in sources:
        _safe_source_path(relative)
    return sources


def build_manifest() -> dict[str, object]:
    sources = discover_source_closure()
    bindings = []
    for relative in sources:
        raw = _read_regular_source(ROOT / relative)
        bindings.append(
            {
                "path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
        )
    core: dict[str, object] = {
        "schema": SCHEMA,
        "date": "2026-07-24",
        "status": "SOURCE_ONLY_RECURSIVE_CLOSURE",
        "authority": (
            "source_closure_only_no_generated_input_checkpoint_training_gpu_"
            "qualification_g2_navigation_heldout_production_or_promotion_authority"
        ),
        "entrypoints": list(ENTRYPOINTS),
        "forced_dynamic_sources": list(FORCED_DYNAMIC_SOURCES),
        "source_count": len(sources),
        "source_paths": list(sources),
        "source_bindings": bindings,
        "source_bindings_sha256": hashlib.sha256(
            _canonical_bytes(bindings)
        ).hexdigest(),
        "excluded_runtime_categories": list(EXCLUDED_RUNTIME_CATEGORIES),
        "consumed_adaptation_runner_source_count": 0,
        "generated_input_open_count": 0,
        "tensor_checkpoint_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
    }
    return {
        **core,
        "content_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def _load_manifest() -> dict[str, object]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("multires source manifest must be a JSON object")
    return value


def _verify_tracked(paths: Iterable[str]) -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "closure contains files absent from the Git index:\n"
            + result.stderr.strip()
        )


def verify_manifest(*, require_tracked: bool) -> None:
    expected = _load_manifest()
    actual = build_manifest()
    if expected != actual:
        expected_paths = set(expected.get("source_paths", []))
        actual_paths = set(actual["source_paths"])
        details = {
            "missing_from_manifest": sorted(actual_paths - expected_paths),
            "stale_in_manifest": sorted(expected_paths - actual_paths),
            "expected_content_sha256": expected.get("content_sha256"),
            "actual_content_sha256": actual.get("content_sha256"),
        }
        raise RuntimeError(
            "multires source manifest does not match source bytes:\n"
            + json.dumps(details, sort_keys=True, indent=2)
        )
    if require_tracked:
        _verify_tracked(actual["source_paths"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit",
        action="store_true",
        help="print the canonical manifest candidate instead of validating",
    )
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help="also require every closure source to exist in the Git index",
    )
    args = parser.parse_args()
    if args.emit:
        print(json.dumps(build_manifest(), sort_keys=True, indent=2))
        return 0
    verify_manifest(require_tracked=args.require_tracked)
    print("Go2 RGB multiresolution probe source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
