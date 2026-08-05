from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts import (
    collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2
    as collector,
)


def _set_exact_environment(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> dict[str, str]:
    expected = collector.plan_builder.rocm_execution_environment(role)
    keys = (
        set(collector.kernel._SANITIZED_SELECTOR_KEYS)  # noqa: SLF001
        | set(collector.ROCM_ADDITIONAL_SANITIZED_KEYS)
        | set(expected)
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    return expected


@pytest.fixture(autouse=True)
def exact_scientific_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_environment(monkeypatch, "scientific")


@pytest.fixture(scope="module")
def runtime_bindings() -> dict:
    return collector.plan_builder.build_rocm_runtime_bindings()


@pytest.fixture(scope="module")
def scientific_plan(runtime_bindings: dict) -> dict:
    frozen = copy.deepcopy(
        collector.plan_builder.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return collector.plan_builder.build_scientific_plan(
        frozen_plan=frozen, runtime_bindings=runtime_bindings
    )


@pytest.fixture(scope="module")
def qualification_plan(runtime_bindings: dict) -> dict:
    frozen = copy.deepcopy(
        collector.plan_builder.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
    return collector.plan_builder.build_qualification_plan(
        frozen_plan=frozen, runtime_bindings=runtime_bindings
    )


def test_collector_is_fresh_v2_identity_only() -> None:
    assert collector.ATTEMPT_ID == collector.plan_builder.DEFAULT_ATTEMPT_ID
    assert collector.EXPECTED_CAPS is collector.predecessor.EXPECTED_CAPS
    assert collector.EXPECTED_COUNTS is collector.predecessor.EXPECTED_COUNTS
    assert set(collector.AUTHORITY_FIELDS) == set(
        collector.predecessor.AUTHORITY_FIELDS
    ) | {"predecessor_v1_qualification_terminal_review_binding"}
    assert (
        collector.PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM[
            "v1_runtime_payload_reuse_authorized"
        ]
        is False
    )
    assert collector.CONTACT_FORCE_ROUTE_AUDIT["known_bad_api_reachable"] is False


def test_runtime_validator_owns_both_v2_roles(
    scientific_plan: dict, qualification_plan: dict
) -> None:
    assert collector._validate_rocm_plan_runtime(scientific_plan) == scientific_plan  # noqa: SLF001
    original = collector.ATTEMPT_ID
    collector.ATTEMPT_ID = collector.plan_builder.QUALIFICATION_ATTEMPT_ID
    try:
        assert collector._validate_rocm_plan_runtime(qualification_plan) == (  # noqa: SLF001
            qualification_plan
        )
    finally:
        collector.ATTEMPT_ID = original


def test_scoped_overlay_routes_nested_v1_hooks_to_v2_and_restores(
    scientific_plan: dict,
) -> None:
    overrides = collector._configuration_overrides_v2()  # noqa: SLF001
    originals = {
        name: getattr(collector.predecessor, name) for name in overrides
    }
    pilot_validate = collector.pilot.validate_plan
    pilot_environment = collector.pilot.EXECUTION_ENVIRONMENT

    with collector._configured_predecessor_collector_rocm():  # noqa: SLF001
        assert collector.predecessor.plan_builder is collector.plan_builder
        assert collector.predecessor.ATTEMPT_ID == collector.ATTEMPT_ID
        assert collector.predecessor._plan_role_for_identity() == "scientific"  # noqa: SLF001
        assert collector.pilot.validate_plan(scientific_plan) == scientific_plan
        assert collector.pilot.EXECUTION_ENVIRONMENT == (
            collector.plan_builder.rocm_execution_environment("scientific")
        )
        assert (
            collector.predecessor.predecessor._validate_authority_v2  # noqa: SLF001
            is collector._validate_authority_v2_review_bound  # noqa: SLF001
        )

    assert collector.pilot.validate_plan is pilot_validate
    assert collector.pilot.EXECUTION_ENVIRONMENT is pilot_environment
    assert all(
        getattr(collector.predecessor, name) is value
        for name, value in originals.items()
    )


def test_qualification_child_context_cannot_fall_back_to_v1(
    qualification_plan: dict,
) -> None:
    original = collector.ATTEMPT_ID
    collector.ATTEMPT_ID = collector.plan_builder.QUALIFICATION_ATTEMPT_ID
    try:
        with collector._configured_predecessor_collector_rocm():  # noqa: SLF001
            assert collector.predecessor._plan_role_for_identity() == (  # noqa: SLF001
                "qualification"
            )
            assert collector.pilot.validate_plan(qualification_plan) == (
                qualification_plan
            )
            assert collector.pilot.EXECUTION_ENVIRONMENT == (
                collector.plan_builder.rocm_execution_environment(
                    "qualification"
                )
            )
    finally:
        collector.ATTEMPT_ID = original


def test_worker_argv_is_the_exact_lexical_rocm_python_and_v2_module() -> None:
    kwargs = {
        "scene_index": 12,
        "plan_path": Path("/tmp/plan"),
        "expected_plan_byte_count": 1,
        "expected_plan_sha256": "a" * 64,
        "authority_path": Path("/tmp/authority"),
        "expected_authority_byte_count": 2,
        "expected_authority_sha256": "b" * 64,
        "reservation_binding": {"byte_count": 3, "file_sha256": "c" * 64},
        "orchestrator_nonce": "d" * 64,
    }
    inherited = collector._ORIGINAL_WORKER_ARGV(**kwargs)  # noqa: SLF001
    actual = collector._worker_argv_rocm(**kwargs)  # noqa: SLF001
    expected_python = str(collector.plan_builder.ROCM_PYTHON.absolute())
    assert collector.sys.executable == expected_python
    assert inherited[0] == expected_python
    assert actual[0] == expected_python
    assert actual[2:] == inherited[2:]
    assert actual[1] == str(Path(collector.__file__).resolve())
    assert Path(actual[1]).resolve() == Path(collector.__file__).resolve()
    assert Path(actual[1]).resolve() != Path(collector.predecessor.__file__).resolve()


def test_worker_argv_rejects_wrong_parent_python_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collector.sys, "executable", "/tmp/wrong-venv/bin/python")
    with pytest.raises(
        collector.SceneProcessCollectionError,
        match="exact lexical ROCm venv path",
    ):
        collector._worker_argv_rocm(  # noqa: SLF001
            scene_index=12,
            plan_path=Path("/tmp/plan"),
            expected_plan_byte_count=1,
            expected_plan_sha256="a" * 64,
            authority_path=Path("/tmp/authority"),
            expected_authority_byte_count=2,
            expected_authority_sha256="b" * 64,
            reservation_binding={"byte_count": 3, "file_sha256": "c" * 64},
            orchestrator_nonce="d" * 64,
        )


def test_collect_public_api_rejects_wrong_python_before_delegate_or_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_root = tmp_path / "scientific_collection"
    delegated: list[bool] = []
    monkeypatch.setattr(
        collector.plan_builder, "DEFAULT_OUTPUT_ROOT", collection_root
    )
    monkeypatch.setattr(
        collector.sys,
        "executable",
        "/tmp/wrong-collect-venv/bin/python",
    )
    monkeypatch.setattr(
        collector.predecessor.predecessor,
        "collect_v2",
        lambda **_kwargs: delegated.append(True),
    )

    with pytest.raises(
        collector.SceneProcessCollectionError,
        match="exact lexical ROCm venv path",
    ):
        collector.collect_rocm(
            plan_path=Path("/tmp/plan"),
            expected_plan_byte_count=1,
            expected_plan_sha256="a" * 64,
            authority_path=Path("/tmp/authority"),
            expected_authority_byte_count=2,
            expected_authority_sha256="b" * 64,
        )

    assert delegated == []
    assert not collection_root.exists()


@pytest.mark.parametrize("role", ["scientific", "qualification"])
def test_exact_sanitized_orchestrator_environment_passes(
    role: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _set_exact_environment(monkeypatch, role)
    assert collector.require_exact_orchestrator_environment(role) == expected


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("HSA_OVERRIDE_GFX_VERSION", "12.0.1"),
        ("LD_LIBRARY_PATH", "/tmp/injected"),
        ("HIP_VISIBLE_DEVICES", "1"),
        ("ROCR_VISIBLE_DEVICES", "1"),
        ("PYTHONHASHSEED", "1"),
    ],
)
def test_mutated_orchestrator_environment_rejects_before_collect_or_root(
    key: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_root = tmp_path / f"rejected-{key.lower()}"
    delegated: list[bool] = []
    monkeypatch.setattr(
        collector.plan_builder, "DEFAULT_OUTPUT_ROOT", collection_root
    )
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        collector.predecessor.predecessor,
        "collect_v2",
        lambda **_kwargs: delegated.append(True),
    )

    with pytest.raises(
        collector.SceneProcessCollectionError,
        match=key,
    ):
        collector.collect_rocm(
            plan_path=Path("/tmp/plan"),
            expected_plan_byte_count=1,
            expected_plan_sha256="a" * 64,
            authority_path=Path("/tmp/authority"),
            expected_authority_byte_count=2,
            expected_authority_sha256="b" * 64,
        )

    assert delegated == []
    assert not collection_root.exists()


def test_worker_authority_review_binding_is_semantically_exact() -> None:
    valid = {
        "predecessor_v1_qualification_terminal_review_binding": (
            collector._standard_v1_review_binding()  # noqa: SLF001
        )
    }
    collector._require_v1_review_binding(valid)  # noqa: SLF001
    changed = copy.deepcopy(valid)
    changed["predecessor_v1_qualification_terminal_review_binding"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(
        collector.SceneProcessCollectionError,
        match="terminal-review binding changed",
    ):
        collector._require_v1_review_binding(changed)  # noqa: SLF001
