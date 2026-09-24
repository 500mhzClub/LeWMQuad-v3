"""Independent contract checks for the frozen raw-supervision Builder V4."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v4.py"
FROZEN = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v4.py": (
        "e46f42db3b5ed50581ed916d459e05f2dd9b73dcbdd906ea5d1991b7b61893e0"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v4.py": (
        "db14bb159b39204e7576b71f3b93409e13b9f28c5cb0d2e87a627557471c0901"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4.py": (
        "80ca9d1d35b83fd29027ab297ac662c406dcdd15f68ac5aced9cc7419fef61c0"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_"
    "author_handoff_2026-07-13.md": (
        "575ae2a596901ba90253e57a9bd5f0e64dd5d07f6c5f8e4872cfefbf6fb93bdb"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def test_frozen_builder_v4_candidate_rehashes_exactly() -> None:
    assert {relative: _sha256(ROOT / relative) for relative in FROZEN} == FROZEN


def test_complete_second_validation_is_immediately_before_publication() -> None:
    """No manifest construction/write/fsync work may follow the second pass."""

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = _function(tree, "_build_exact_prepared_dataset_v4")
    calls = sorted(
        (
            (node.lineno, _call_name(node))
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        ),
        key=lambda item: item[0],
    )
    revalidations = [
        line
        for line, name in calls
        if name == "_revalidate_exact_before_publication"
    ]
    publications = [line for line, name in calls if name == "_libc_renameat2"]
    assert len(revalidations) == 1
    assert len(publications) == 1
    revalidation_line = revalidations[0]
    publication_line = publications[0]
    assert revalidation_line < publication_line

    forbidden_after_second_pass = {
        "_precommitted_audit_sample",
        "_with_content_sha256",
        "_validate_staging_inventory",
        "_write_json_exclusive",
        "_fsync_directory",
        "fsync",
    }
    intervening = [
        (line, name)
        for line, name in calls
        if revalidation_line < line < publication_line
        and name in forbidden_after_second_pass
    ]
    assert intervening == [], (
        "the complete second metadata/source validation is not immediately before "
        f"publication; intervening artifact work is {intervening}"
    )
