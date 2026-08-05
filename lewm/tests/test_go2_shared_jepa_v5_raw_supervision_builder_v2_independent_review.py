"""Different-agent review for the frozen raw-supervision builder V2.

The failing tests are the frozen BLOCK reproducers.  They exercise only
synthetic values and in-process fakes: no exact authorization, metadata,
development source, payload, accelerator, or canonical output is opened.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v1 as v1
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v2 as builder
from lewm.tests import test_go2_shared_jepa_v5_raw_supervision_builder_v2 as author_tests


ROOT = Path(__file__).resolve().parents[2]
FROZEN = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v2.py": (
        "0ae5ddd836802ced1fcf7524b67970247dccace6787fd0acc7268cbae4d3e71c"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v2.py": (
        "c11396874677c3cd3d0ef76353ea7de1449ef610d35f0b4256530a4f62b1d303"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2.py": (
        "6755044af535dc0c2de93f0f5bd79b01b140da33bc8ff2ec5b003ef592b50339"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_"
    "author_handoff_2026-07-13.md": (
        "7f278c5c24a8e9d89c6b0e3ecb9252acd0edec5729bd9fdde5d72231848bc04f"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = builder.canonical_json_sha256(result)
    return result


def test_builder_v2_frozen_candidate_rehashes() -> None:
    assert {relative: _sha(ROOT / relative) for relative in FROZEN} == FROZEN


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source_map"].pop(),
        lambda value: value["source_map"].append(
            {
                "role": "extra",
                "path": "review/extra.json",
                "sha256": "1" * 64,
            }
        ),
        lambda value: value["source_map"].__setitem__(
            0, {**value["source_map"][0], "role": "wrong"}
        ),
        lambda value: value["source_map"].__setitem__(
            0, {**value["source_map"][0], "path": "payload/selected.jsonl"}
        ),
        lambda value: value["source_map"].__setitem__(
            1, deepcopy(value["source_map"][0])
        ),
        lambda value: value["source_map"].__setitem__(
            slice(0, 2), [value["source_map"][1], value["source_map"][0]]
        ),
        lambda value: value["builder_review"]["candidate"][0].__setitem__(
            "sha256", "2" * 64
        ),
    ],
)
def test_invalid_phase_one_authority_reaches_zero_target_openers(
    mutate: Callable[[dict[str, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _raw_by_role, _digests = author_tests._valid_authorization()
    mutate(authority)
    authority = _rehash(authority)
    opened: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> bytes:
        opened.append("opened")
        raise AssertionError("an invalid phase-one authority reached a target opener")

    monkeypatch.setattr(builder, "_read_bound_regular_file", forbidden)
    monkeypatch.setattr(v1.plan_v5, "load_frozen_development_metadata", forbidden)
    monkeypatch.setattr(
        v1.plan_v5, "load_frozen_development_source_inventory", forbidden
    )
    monkeypatch.setattr(v1, "_read_exact_source", forbidden)
    with pytest.raises((PermissionError, builder.RawSupervisionBuildError)):
        builder._validate_authorization_phase_one(
            authority, authorization_file_sha256="3" * 64
        )
    assert opened == []


def test_valid_synthetic_phase_two_checks_exact_nine_fixed_targets(
    tmp_path: Path,
) -> None:
    authority, raw_by_role, digests = author_tests._valid_authorization()
    phase_one = builder._validate_authorization_phase_one(
        authority, authorization_file_sha256="4" * 64
    )
    role_by_path = {
        path: role for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    opened: list[str] = []

    def reader(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        relative = path.relative_to(repository_root).as_posix()
        role = role_by_path[relative]
        assert expected_sha256 == digests[role]
        opened.append(role)
        return raw_by_role[role]

    assert builder._validate_authorization_phase_two(
        phase_one,
        repository_root=tmp_path,
        reader=reader,
        rehash_frozen_parents=False,
    ) == authority
    assert opened == [role for role, _path in builder.AUTHORIZED_ROLE_PATHS]


def test_builder_v2_import_exposes_no_frozen_v1_exact_fallback() -> None:
    """BLOCK: importing V2 exposes the independently blocked V1 exact entry."""

    reachable_v1 = vars(builder).get("_v1")
    assert reachable_v1 is None or not callable(
        getattr(reachable_v1, "execute_exact_build_v1", None)
    ), "builder._v1.execute_exact_build_v1 remains reachable after importing V2"


def test_authorized_pool_rejects_caller_callback_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLOCK: the pool's authorization argument is unused before callback run."""

    called: list[str] = []

    def caller_callback() -> str:
        called.append("executed")
        return "executed"

    class ImmediateFuture:
        def __init__(self, value: Any) -> None:
            self.value = value

        def result(self) -> Any:
            return self.value

    class ImmediateExecutor:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> "ImmediateExecutor":
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def submit(self, function: Callable[..., Any], *arguments: Any) -> ImmediateFuture:
            return ImmediateFuture(function(*arguments))

    monkeypatch.setattr(builder, "ProcessPoolExecutor", ImmediateExecutor)
    with pytest.raises(PermissionError):
        builder._run_authorized_scene_pool(
            caller_callback,
            [()],
            workers=1,
            authorization_sha256="0" * 64,
        )
    assert called == [], "an invalid authorization reached a caller-supplied callback"


def test_v1_bridge_never_replaces_global_v1_authority_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLOCK: each bridge installs mutable process-global V1 authority."""

    original = v1._require_exact_authority
    observed: list[bool] = []
    synthetic_authority = {"content_sha256": "5" * 64}

    monkeypatch.setattr(
        builder,
        "_require_exact_authority",
        lambda _digest: deepcopy(synthetic_authority),
    )

    def probe(_digest: str) -> tuple[dict[str, Any], ...]:
        observed.append(v1._require_exact_authority is original)
        assert v1._require_exact_authority("5" * 64) == synthetic_authority
        return ()

    monkeypatch.setattr(v1, "_load_parent_contracts", probe)
    assert builder._call_v1_load_parent_contracts("5" * 64) == ()
    assert v1._require_exact_authority is original
    assert observed == [True], (
        "the V2 bridge replaced the global V1 authority validator with an "
        "accepting callback during the V1 call"
    )


def test_production_phase_two_exposes_no_caller_reader_or_parent_bypass() -> None:
    """BLOCK: a production validator exposes review/source reader test seams."""

    parameters = inspect.signature(builder._validate_authorization_phase_two).parameters
    assert "reader" not in parameters
    assert "repository_root" not in parameters
    assert "rehash_frozen_parents" not in parameters
