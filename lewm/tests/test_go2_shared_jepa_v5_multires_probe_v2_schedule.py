from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_matched_training_v1 as matched_v1
from lewm.benchmarks import go2_shared_jepa_v5_multires_probe_v1 as probe_v1
from lewm.benchmarks import (
    go2_shared_jepa_v5_multires_probe_v2_schedule as adapter,
)


ROOT = Path(__file__).resolve().parents[2]


def _binding(raw: bytes, content_sha256: str) -> dict[str, Any]:
    return {
        "path": "synthetic/matched_training_v4/schedule.json",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
    }


def _synthetic_fixture() -> tuple[
    adapter._SchedulePolicy,
    bytes,
    dict[str, Any],
    list[str],
    list[int],
]:
    pair_ids = [
        hashlib.sha256(f"synthetic-pair:{index}".encode("ascii")).hexdigest()
        for index in range(matched_v1.TRAIN_PAIR_COUNT)
    ]
    complete, remainder = divmod(
        matched_v1.PRESENTATION_COUNT,
        matched_v1.TRAIN_PAIR_COUNT,
    )
    indices = (
        list(range(matched_v1.TRAIN_PAIR_COUNT)) * complete
        + list(range(remainder))
    )
    v1 = matched_v1.with_content_sha256({
        **matched_v1.schedule_core(indices, pair_ids),
        "presentation_indices": indices,
    })
    v4_core = dict(v1)
    v4_core.pop("content_sha256")
    v4_core["schema"] = adapter.BOUND_V4_SCHEDULE_SCHEMA
    v4 = matched_v1.with_content_sha256(v4_core)
    raw = matched_v1.canonical_json_bytes(v4) + b"\n"
    binding = _binding(raw, v4["content_sha256"])
    policy = adapter._SchedulePolicy(
        path=binding["path"],
        file_sha256=binding["file_sha256"],
        content_sha256=binding["content_sha256"],
        byte_count=binding["byte_count"],
        bound_schema=adapter.BOUND_V4_SCHEDULE_SCHEMA,
        normalized_schema=matched_v1.SCHEDULE_SCHEMA,
        normalized_content_sha256=v1["content_sha256"],
        schedule_identity_items=tuple(sorted({
            key: v4[key]
            for key in adapter.FROZEN_SCHEDULE_IDENTITY
        }.items())),
        prefix_items=tuple(
            (
                presentations,
                matched_v1.canonical_json_sha256(indices[:presentations]),
            )
            for presentations in (1_600, 6_400, 16_000)
        ),
    )
    return policy, raw, binding, pair_ids, indices


def _install_synthetic_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[bytes, dict[str, Any], list[str], list[int]]:
    policy, raw, binding, pair_ids, indices = _synthetic_fixture()
    monkeypatch.setattr(adapter, "_FROZEN_POLICY", policy)
    return raw, binding, pair_ids, indices


def _canonical_mutation(
    value: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    core = dict(value)
    core.pop("content_sha256", None)
    mutated = matched_v1.with_content_sha256(core)
    raw = matched_v1.canonical_json_bytes(mutated) + b"\n"
    return raw, mutated


def test_two_phase_adapter_preserves_bytes_indices_and_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, binding, pair_ids, indices = _install_synthetic_policy(monkeypatch)
    before = bytes(raw)
    state = adapter.validate_bound_schedule_phase_a(
        raw=raw,
        binding=dict(binding),
    )
    returned, observed_binding, record = adapter.finalize_train_identity(
        state=state,
        ordered_train_pair_ids=list(pair_ids),
    )

    assert raw == before
    assert tuple(indices) == state.presentation_indices
    assert returned == indices[:16_000]
    assert observed_binding == binding
    assert observed_binding is not binding
    assert record["schedule_bytes_rewritten"] is False
    assert record["schedule_reopened_or_regenerated"] is False
    assert record["indices_mutated_reordered_filtered_or_reseeded"] is False
    assert record["phase_a_complete"] is True
    assert record["phase_b_train_identity_complete"] is True
    core = dict(record)
    declared = core.pop("content_sha256")
    assert matched_v1.canonical_json_sha256(core) == declared


def test_phase_a_rejects_wrong_owning_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, raw, _, _, _ = _synthetic_fixture()
    value = matched_v1.parse_canonical_json(raw, name="synthetic V4")
    value["schema"] = matched_v1.SCHEDULE_SCHEMA
    mutated_raw, mutated = _canonical_mutation(value)
    mutated_binding = _binding(mutated_raw, mutated["content_sha256"])
    monkeypatch.setattr(
        adapter,
        "_FROZEN_POLICY",
        replace(
            policy,
            file_sha256=mutated_binding["file_sha256"],
            content_sha256=mutated_binding["content_sha256"],
            byte_count=mutated_binding["byte_count"],
        ),
    )
    with pytest.raises(
        adapter.ScheduleAdapterIntegrityError,
        match="owning schema",
    ):
        adapter.validate_bound_schedule_phase_a(
            raw=mutated_raw,
            binding=mutated_binding,
        )


def test_phase_a_rejects_file_and_self_hash_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, binding, _, _ = _install_synthetic_policy(monkeypatch)
    with pytest.raises(
        adapter.ScheduleAdapterIntegrityError,
        match="file hash",
    ):
        adapter.validate_bound_schedule_phase_a(
            raw=raw[:-2] + b"0\n",
            binding=binding,
        )

    value = matched_v1.parse_canonical_json(raw, name="synthetic V4")
    value["content_sha256"] = "0" * 64
    malformed = matched_v1.canonical_json_bytes(value) + b"\n"
    malformed_binding = _binding(malformed, "0" * 64)
    policy = adapter._FROZEN_POLICY
    monkeypatch.setattr(
        adapter,
        "_FROZEN_POLICY",
        replace(
            policy,
            file_sha256=malformed_binding["file_sha256"],
            content_sha256=malformed_binding["content_sha256"],
            byte_count=malformed_binding["byte_count"],
        ),
    )
    with pytest.raises(
        adapter.ScheduleAdapterIntegrityError,
        match="canonical self-hashed",
    ):
        adapter.validate_bound_schedule_phase_a(
            raw=malformed,
            binding=malformed_binding,
        )


def test_phase_a_rejects_validly_rehashed_suffix_permutation_attack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, raw, _, _, _ = _synthetic_fixture()
    value = matched_v1.parse_canonical_json(raw, name="synthetic V4")
    attacked = list(value["presentation_indices"])
    attacked[-1] = attacked[-2]
    value["presentation_indices"] = attacked
    value["indices_sha256"] = matched_v1.canonical_json_sha256(attacked)
    attacked_raw, attacked_value = _canonical_mutation(value)
    attacked_binding = _binding(
        attacked_raw, attacked_value["content_sha256"]
    )
    identity = dict(policy.schedule_identity_items)
    identity["indices_sha256"] = attacked_value["indices_sha256"]
    monkeypatch.setattr(
        adapter,
        "_FROZEN_POLICY",
        replace(
            policy,
            file_sha256=attacked_binding["file_sha256"],
            content_sha256=attacked_binding["content_sha256"],
            byte_count=attacked_binding["byte_count"],
            schedule_identity_items=tuple(sorted(identity.items())),
            normalized_content_sha256="f" * 64,
        ),
    )
    with pytest.raises(
        adapter.ScheduleAdapterIntegrityError,
        match="integer or permutation",
    ):
        adapter.validate_bound_schedule_phase_a(
            raw=attacked_raw,
            binding=attacked_binding,
        )


def test_phase_a_rejects_prefix_or_normalized_hash_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, raw, binding, _, _ = _synthetic_fixture()
    monkeypatch.setattr(
        adapter,
        "_FROZEN_POLICY",
        replace(
            policy,
            prefix_items=(
                (1_600, "0" * 64),
                *policy.prefix_items[1:],
            ),
        ),
    )
    with pytest.raises(
        adapter.ScheduleAdapterIntegrityError,
        match="prefix changed",
    ):
        adapter.validate_bound_schedule_phase_a(raw=raw, binding=binding)

    monkeypatch.setattr(
        adapter,
        "_FROZEN_POLICY",
        replace(policy, normalized_content_sha256="0" * 64),
    )
    with pytest.raises(
        adapter.ScheduleAdapterIntegrityError,
        match="normalized content",
    ):
        adapter.validate_bound_schedule_phase_a(raw=raw, binding=binding)


def test_phase_b_rejects_actual_train_pair_order_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, binding, pair_ids, _ = _install_synthetic_policy(monkeypatch)
    state = adapter.validate_bound_schedule_phase_a(
        raw=raw,
        binding=binding,
    )
    with pytest.raises(
        adapter.ScheduleAdapterIntegrityError,
        match="actual ordered train-pair identity",
    ):
        adapter.finalize_train_identity(
            state=state,
            ordered_train_pair_ids=list(reversed(pair_ids)),
        )


def test_exact_production_bindings_and_science_identity_are_frozen() -> None:
    assert adapter._FROZEN_POLICY.binding() == {
        "path": adapter.BOUND_SCHEDULE_PATH,
        "file_sha256":
            "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270",
        "content_sha256":
            "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15",
        "byte_count": 607_373,
    }
    assert adapter.NORMALIZED_V1_SCHEDULE_CONTENT_SHA256 == (
        "893c48b2c2c591dbc90469e5a19a74e70bd54f96689b63881c216605255c0e5d"
    )
    assert adapter.FROZEN_PREFIX_SHA256 == (
        (
            1_600,
            "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
        ),
        (
            6_400,
            "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
        ),
        (
            16_000,
            "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
        ),
    )
    assert adapter.require_v1_science_contract_identity(
        probe_v1.science_contract()
    ) == (
        "e181381c00585fa5df41a71fff918b5599acc955d59283ce397ba6dd530dc23f"
    )


def test_adapter_source_has_no_file_runtime_tensor_or_torch_access() -> None:
    source_path = (
        ROOT
        / "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
    )
    tree = ast.parse(source_path.read_text("utf-8"), filename=str(source_path))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert all(
        not name.startswith(("torch", "numpy", "PIL", "cv2"))
        for name in imported
    )
    forbidden_calls = {
        "open",
        "read",
        "read_bytes",
        "read_text",
        "write",
        "write_bytes",
        "write_text",
        "load",
        "loads",
        "dump",
        "dumps",
    }
    observed_calls = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (observed_calls & forbidden_calls)
