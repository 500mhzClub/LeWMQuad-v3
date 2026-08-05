from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts import (
    collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1
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


def _worker_kwargs() -> dict:
    return {
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


def test_collector_is_fresh_replacement_layer_on_v3() -> None:
    assert collector.plan_builder.predecessor is collector.predecessor.plan_builder
    assert collector.ATTEMPT_ID == collector.plan_builder.DEFAULT_ATTEMPT_ID
    assert set(collector.AUTHORITY_FIELDS) == set(
        collector.predecessor.AUTHORITY_FIELDS
    ) | {"predecessor_v3_qualification_terminal_review_binding"}
    audit = collector.PROCESS_RESET_EQUIVALENCE_AUDIT_ROCM
    assert audit["v2_runtime_payload_reuse_authorized"] is False
    assert audit["v3_runtime_payload_reuse_authorized"] is False
    assert audit["v3_required_host_home"] == "/home/andrewknowles"


def test_exact_environment_requires_literal_home_and_absent_host_aliases() -> None:
    environment = collector.require_exact_orchestrator_environment("scientific")
    assert environment["HOME"] == collector.plan_builder.REQUIRED_HOST_HOME
    assert all(key not in environment for key in ("USER", "LOGNAME", "LANG"))


@pytest.mark.parametrize(
    ("key", "value", "remove"),
    [
        ("HOME", None, True),
        ("HOME", "/tmp/wrong-home", False),
        ("USER", "ambient-user", False),
        ("LOGNAME", "ambient-logname", False),
        ("LANG", "en_GB.UTF-8", False),
        ("HSA_OVERRIDE_GFX_VERSION", "12.0.1", False),
        ("LD_LIBRARY_PATH", "/tmp/injected", False),
    ],
)
def test_environment_mutation_rejects_before_collect_delegate_or_root(
    key: str,
    value: str | None,
    remove: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_root = tmp_path / "collection"
    delegated: list[bool] = []
    monkeypatch.setattr(
        collector.plan_builder, "DEFAULT_OUTPUT_ROOT", collection_root
    )
    if remove:
        monkeypatch.delenv(key, raising=False)
    else:
        assert value is not None
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        collector.predecessor,
        "collect_v2",
        lambda **_kwargs: delegated.append(True),
    )

    with pytest.raises(collector.SceneProcessCollectionError, match=key):
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


def test_worker_argv_preserves_lexical_python_and_owns_v3_child() -> None:
    inherited = collector._ORIGINAL_WORKER_ARGV(**_worker_kwargs())  # noqa: SLF001
    actual = collector._worker_argv_rocm(**_worker_kwargs())  # noqa: SLF001
    expected_python = str(collector.plan_builder.ROCM_PYTHON.absolute())
    assert inherited[0] == expected_python
    assert actual[0] == expected_python
    assert actual[1] == str(Path(collector.__file__).resolve())
    assert Path(inherited[1]).resolve() == Path(
        collector.predecessor.__file__
    ).resolve()
    assert actual[2:] == inherited[2:]


def test_v3_review_binding_is_semantically_exact() -> None:
    valid = {
        "predecessor_v3_qualification_terminal_review_binding": (
            collector._standard_v3_review_binding()  # noqa: SLF001
        )
    }
    collector._require_v3_review_binding(valid)  # noqa: SLF001
    changed = copy.deepcopy(valid)
    changed["predecessor_v3_qualification_terminal_review_binding"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(
        collector.SceneProcessCollectionError,
        match="V3 terminal-review binding changed",
    ):
        collector._require_v3_review_binding(changed)  # noqa: SLF001


def test_scoped_overlay_routes_replacement_and_restores_v3() -> None:
    overrides = collector._configuration_overrides_v3()  # noqa: SLF001
    originals = {name: getattr(collector.predecessor, name) for name in overrides}
    with collector._configured_predecessor_collector_rocm():  # noqa: SLF001
        assert collector.predecessor.plan_builder is collector.plan_builder
        assert collector.predecessor.ATTEMPT_ID == collector.ATTEMPT_ID
        assert collector.predecessor.ROCM_ADDITIONAL_SANITIZED_KEYS == (
            collector.ROCM_ADDITIONAL_SANITIZED_KEYS
        )
    assert all(
        getattr(collector.predecessor, name) is value
        for name, value in originals.items()
    )
