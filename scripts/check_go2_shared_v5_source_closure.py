#!/usr/bin/env python3
"""Verify the fixed G2 and Shared V5 development-runner source closure.

This checker parses source imports without importing project modules. It never
walks configuration, data, artifact, generated, or custody directories.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "docs/lewm_go2_g2_runner_source_closure_v1_manifest_2026-07-24.json"
)
SCHEMA = "lewm_go2_g2_runner_source_closure_v1"

ENTRYPOINTS = (
    "scripts/run_go2_shared_v5_dev_maze.py",
    "scripts/run_go2_shared_jepa_v5_gate.py",
    "scripts/finalize_go2_shared_jepa_v5_gate.py",
    "scripts/publish_go2_shared_jepa_v5_checkpoint.py",
)

# These files are executed from captured bytes rather than imported normally.
FORCED_SOURCES = (
    "scripts/go2_shared_jepa_v5_launcher.py",
    "scripts/go2_shared_jepa_v5_one_shot.py",
)

PACKAGE_ROOTS = (
    ("lewm", ROOT / "lewm"),
    ("scripts", ROOT / "scripts"),
    ("lewm_genesis", ROOT / "lewm_genesis/lewm_genesis"),
    ("lewm_worlds", ROOT / "lewm_worlds/lewm_worlds"),
    ("lewm_go2_bringup", ROOT / "lewm_go2_bringup/lewm_go2_bringup"),
    ("lewm_go2_control", ROOT / "lewm_go2_control/lewm_go2_control"),
)

PENDING_GENERATED_AUTHORITIES = (
    "docs/lewm_go2_shared_jepa_v5_runner_g2_authority_v2.json",
    "docs/lewm_go2_shared_jepa_v5_runner_g3_authority_v2.json",
    "docs/lewm_go2_shared_jepa_v5_finalizer_g2_authority_v2.json",
    "docs/lewm_go2_shared_jepa_v5_finalizer_g3_authority_v2.json",
    "docs/lewm_go2_shared_jepa_v5_publisher_g2_candidate_authority_v2.json",
    "docs/lewm_go2_shared_jepa_v5_publisher_full_promotion_authority_v2.json",
)

EXCLUDED_RUNTIME_CATEGORIES = (
    ".generated artifacts and attempt registries",
    "candidate and auxiliary-head checkpoints",
    "dataset roles, raw scene inputs, and scene packs",
    "runner outcomes, ledgers, final reports, and publications",
    "target, physical, and G4 calibration artifacts",
    "logs, caches, videos, and metric sidecars",
)

AUTHORITY_BOUND_DYNAMIC_SOURCES = (
    "G2/G3 inference modules captured by each stage authority",
    "optional development observer bound by development authority",
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _module_index() -> tuple[dict[str, Path], dict[Path, str]]:
    by_module: dict[str, Path] = {}
    by_path: dict[Path, str] = {}
    for prefix, package_root in PACKAGE_ROOTS:
        if not package_root.is_dir():
            continue
        for path in sorted(package_root.rglob("*.py")):
            relative = path.relative_to(package_root)
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join((prefix, *parts)) if parts else prefix
            resolved = path.resolve()
            existing = by_module.get(module)
            if existing is not None and existing != resolved:
                raise RuntimeError(f"duplicate local module {module}: {existing}, {path}")
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
    package = current_module if current_path.name == "__init__.py" else current_module.rpartition(".")[0]
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


def _parent_package_paths(module: str, by_module: dict[str, Path]) -> Iterable[Path]:
    parts = module.split(".")
    for length in range(1, len(parts)):
        candidate = by_module.get(".".join(parts[:length]))
        if candidate is not None and candidate.name == "__init__.py":
            yield candidate


def discover_source_closure() -> tuple[str, ...]:
    by_module, by_path = _module_index()
    queue = [
        (ROOT / relative).resolve()
        for relative in (*ENTRYPOINTS, *FORCED_SOURCES)
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
        except KeyError as exc:
            raise RuntimeError(f"closure source is outside fixed package roots: {path}") from exc
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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

    return tuple(sorted(_relative(path) for path in visited))


def build_manifest() -> dict[str, object]:
    sources = discover_source_closure()
    bindings = [
        {
            "path": relative,
            "file_sha256": _sha256(ROOT / relative),
        }
        for relative in sources
    ]
    core: dict[str, object] = {
        "schema": SCHEMA,
        "date": "2026-07-24",
        "authority": (
            "source_closure_only_no_data_training_g2_navigation_runtime_"
            "production_or_heldout_authority"
        ),
        "entrypoints": list(ENTRYPOINTS),
        "forced_captured_sources": list(FORCED_SOURCES),
        "source_count": len(sources),
        "source_paths": list(sources),
        "source_bindings_sha256": hashlib.sha256(
            _canonical_bytes(bindings)
        ).hexdigest(),
        "pending_generated_authorities": list(PENDING_GENERATED_AUTHORITIES),
        "authority_bound_dynamic_sources": list(AUTHORITY_BOUND_DYNAMIC_SOURCES),
        "excluded_runtime_categories": list(EXCLUDED_RUNTIME_CATEGORIES),
        "production_authority_state": (
            "all_six_wrapper_authority_hashes_must_remain_none"
        ),
        "portability": (
            "production lifecycle remains intentionally bound to the canonical "
            "repository root; exported-tree certification is parse/hash/import "
            "only and grants no execution authority"
        ),
    }
    return {
        **core,
        "content_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def _load_manifest() -> dict[str, object]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("source-closure manifest must be a JSON object")
    return value


def _verify_tracked(paths: Iterable[str]) -> None:
    command = ["git", "ls-files", "--error-unmatch", *paths]
    result = subprocess.run(
        command,
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
            "source-closure manifest does not match source bytes:\n"
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
    print("Shared V5 G2/runner source closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
