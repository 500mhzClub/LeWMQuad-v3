"""Independent adversarial checks for the frozen raw-supervision Builder V5."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v5 as builder
from lewm.tests.test_go2_shared_jepa_v5_raw_supervision_builder_v5 import (
    _synthetic_job_and_pair,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v5.py": (
        "8d85635a85d5a6a3575602a89f37a01f97acf03bd0059a8ae452b21ed4cddce2"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v5.py": (
        "3116c2a5b429cf0fbed0674de91b0569d6ecf6e10c26cd6064a3bb0349e78019"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5.py": (
        "6b49d5d5847e22cea413a7b72da34d5fbf221f876b89bfdf899804024c9d05d6"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_"
    "author_handoff_2026-07-13.md": (
        "a8037613cca9c3879eb2dc8f9df847097a9053326ff973f01a79b3299aec9d26"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _InlineFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


class _InlineExecutor:
    def __init__(
        self,
        *,
        initializer: Any = None,
        initargs: tuple[Any, ...] = (),
        **_kwargs: Any,
    ) -> None:
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def submit(self, function: Any, *args: Any) -> _InlineFuture:
        return _InlineFuture(function(*args))


def test_frozen_builder_v5_candidate_rehashes_exactly() -> None:
    assert {relative: _sha256(ROOT / relative) for relative in FROZEN} == FROZEN


def test_staging_mutation_during_final_source_pass_cannot_be_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final source pass must not create an unchecked staging window."""

    output = tmp_path / "publication" / "dataset"
    authorization_sha256 = "a" * 64
    job, pair = _synthetic_job_and_pair()
    context = builder.ExactPrepublicationContextV5(
        plan=None,  # type: ignore[arg-type]
        inventory=None,  # type: ignore[arg-type]
        source_records=(),
        authorization_sha256=authorization_sha256,
        workers=1,
    )
    tampered_payload = b'{"tampered_during_second_source_pass":true}\n'
    observed_staging: list[Path] = []

    monkeypatch.setattr(builder, "CANONICAL_OUTPUT", output)
    monkeypatch.setattr(builder, "_require_exact_authority", lambda _digest: None)
    monkeypatch.setattr(builder, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        builder,
        "_precommitted_audit_sample",
        lambda _rows: {"records": [{} for _ in range(24)]},
    )

    def mutate_staging_during_source_pass(_context: Any) -> None:
        candidates = [
            path
            for path in output.parent.iterdir()
            if path.name.startswith(f".{output.name}.staging.")
        ]
        assert len(candidates) == 1
        staging = candidates[0]
        observed_staging.append(staging)
        (staging / "pairs.jsonl").write_bytes(tampered_payload)

    monkeypatch.setattr(
        builder,
        "_revalidate_exact_before_publication",
        mutate_staging_during_source_pass,
    )

    with pytest.raises(
        builder.RawSupervisionBuildError,
        match="staging.*changed|file.*changed|inventory.*changed",
    ):
        builder._build_exact_prepared_dataset_v5(
            (job,),
            (pair,),
            workers=1,
            input_provenance={},
            access_ledger={},
            prepublication_context=context,
        )

    assert observed_staging
    assert not output.exists()
