from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Callable

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v9 as builder
from lewm.tests import test_go2_shared_jepa_v5_raw_supervision_builder_v9 as author_tests


ROOT = Path(__file__).resolve().parents[2]


def _normalized_definitions(path: Path, *, v8_to_v9: bool) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    if v8_to_v9:
        source = source.replace("V8", "V9").replace("v8", "v9")
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert len({node.name for node in definitions}) == len(definitions)
    return {
        node.name: ast.dump(node, include_attributes=False) for node in definitions
    }


def test_independent_all_v9_definitions_match_v8_after_version_normalization() -> None:
    v8 = _normalized_definitions(
        ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v8.py",
        v8_to_v9=True,
    )
    v9 = _normalized_definitions(Path(builder.__file__), v8_to_v9=False)
    assert len(v8) == len(v9) == 80
    assert v9 == v8


def test_success_after_linearization_only_closes_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"reviewer-owned frozen source\n")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "publication" / "dataset"
    job, pair = author_tests._synthetic_job_and_pair()
    authorization_sha256 = "d" * 64
    context = builder.ExactPrepublicationContextV9(
        plan=None,  # type: ignore[arg-type]
        inventory=None,  # type: ignore[arg-type]
        source_records=(),
        authorization_sha256=authorization_sha256,
        workers=1,
    )

    monkeypatch.setattr(builder, "CANONICAL_OUTPUT", output)
    monkeypatch.setattr(builder, "_require_exact_authority", lambda _digest: None)
    monkeypatch.setattr(
        builder, "ProcessPoolExecutor", author_tests._InlineExecutor
    )
    monkeypatch.setattr(
        builder,
        "_precommitted_audit_sample",
        lambda _rows: {"records": [{"index": index} for index in range(24)]},
    )
    monkeypatch.setattr(
        builder,
        "_exact_publication_source_hashes",
        lambda _context: {source: source_sha256},
    )
    monkeypatch.setattr(
        builder, "_revalidate_exact_before_publication", lambda _context: None
    )

    linearized = False
    original_quiet = builder._ClosedPublicationTransaction.require_final_quiet

    def mark_linearized(transaction: builder._ClosedPublicationTransaction) -> None:
        nonlocal linearized
        original_quiet(transaction)
        linearized = True

    monkeypatch.setattr(
        builder._ClosedPublicationTransaction,
        "require_final_quiet",
        mark_linearized,
    )

    def forbid_after_linearization(function: Callable[..., Any]) -> Callable[..., Any]:
        def checked(*args: Any, **kwargs: Any) -> Any:
            if linearized:
                raise AssertionError(
                    f"filesystem helper ran after linearization: {function.__name__}"
                )
            return function(*args, **kwargs)

        return checked

    for name in (
        "_cleanup_owned_directory",
        "_fsync_directory",
        "_libc_renameat2",
        "_named_directory_identity",
        "_read_bound_regular_file",
        "_sha256_fd",
        "_sha256_file",
        "_validate_staging_inventory",
    ):
        monkeypatch.setattr(
            builder, name, forbid_after_linearization(getattr(builder, name))
        )
    monkeypatch.setattr(
        builder.os, "fsync", forbid_after_linearization(builder.os.fsync)
    )

    try:
        manifest = builder._build_exact_prepared_dataset_v9(
            (job,),
            (pair,),
            workers=1,
            input_provenance={"synthetic": True},
            access_ledger={"synthetic": True},
            prepublication_context=context,
        )
    finally:
        linearized = False

    assert manifest["status"] == "complete_pending_independent_audit"
    assert output.is_dir()
