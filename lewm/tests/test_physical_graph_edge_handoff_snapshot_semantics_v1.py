from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
import pickle
import subprocess
import textwrap

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_snapshot_semantics_v1 as S


REPO_ROOT = Path(__file__).resolve().parents[2]
GENESIS_PYTHON = (
    REPO_ROOT / ".generated/venvs/genesis_rocm_0_4_6_v1/bin/python"
)
RECOVERY_ROOT = Path("/home/andrewknowles/RecoveryStorage/LeWMQuad-v3")


def _dynamic_branch_snapshot(**overrides: object) -> object:
    namespace = {
        "__module__": "scripts.run_go2_oracle_branch_pilot_v1",
        "__slots__": S.STRUCTURED_FIELD_AUTHORITY[
            "scripts.run_go2_oracle_branch_pilot_v1.BranchSnapshot"
        ],
    }
    cls = type("BranchSnapshot", (), namespace)
    values: dict[str, object] = {
        "solver_state": {},
        "step_index": 0,
        "last_actions": np.zeros(1, dtype=np.float32),
        "harness": {},
        "rng": {},
        "counters": {},
        "goal": {},
        "identity": {},
        "boundary": {},
        "digest": None,
    }
    values.update(overrides)
    result = cls()
    for name in namespace["__slots__"]:
        setattr(result, name, values[name])
    return result


def _run_genesis(script: str) -> dict[str, object]:
    if not GENESIS_PYTHON.is_file():
        pytest.skip("bound Genesis Python is unavailable")
    output = subprocess.check_output(
        [str(GENESIS_PYTHON), "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        text=True,
    )
    return json.loads(output)


def test_01_typed_scalar_mapping_keys_are_sorted_and_bool_is_not_int() -> None:
    left = {"z": 1, b"x": 2, 4: 3, False: 4, None: 5, 2.5: 6}
    right = dict(reversed(list(left.items())))
    assert S.canonical_semantic_snapshot(left) == S.canonical_semantic_snapshot(right)
    assert S.semantic_snapshot_sha256({True: "x"}) != S.semantic_snapshot_sha256({1: "x"})
    assert S.semantic_snapshot_sha256({"a": 1}) != S.semantic_snapshot_sha256(
        OrderedDict({"a": 1})
    )
    with pytest.raises(S.SemanticSnapshotError, match="mapping key"):
        S.canonical_semantic_snapshot({(1, 2): "compound"})


def test_02_list_tuple_and_bytes_are_type_distinct() -> None:
    assert S.semantic_snapshot_sha256([1, 2]) != S.semantic_snapshot_sha256((1, 2))
    assert S.semantic_snapshot_sha256([1, 2]) != S.semantic_snapshot_sha256([2, 1])
    assert S.semantic_snapshot_sha256((1, 2)) != S.semantic_snapshot_sha256((2, 1))
    assert S.semantic_snapshot_sha256(b"abc") != S.semantic_snapshot_sha256("abc")
    unicode_value = {"μεταδεδομένα": "café ☃"}
    assert S.canonical_semantic_snapshot(unicode_value) == S.canonical_semantic_snapshot(
        json.loads(json.dumps(unicode_value, ensure_ascii=False))
    )
    with pytest.raises(S.SemanticSnapshotError, match="bytearray"):
        S.canonical_semantic_snapshot(bytearray(b"abc"))


def test_03_set_and_frozenset_are_sorted_distinct_and_scalar_only() -> None:
    assert S.canonical_semantic_snapshot({"b", "a"}) == S.canonical_semantic_snapshot(
        set(reversed(["b", "a"]))
    )
    assert S.semantic_snapshot_sha256({"a"}) != S.semantic_snapshot_sha256(
        frozenset({"a"})
    )
    with pytest.raises(S.SemanticSnapshotError, match="set member"):
        S.canonical_semantic_snapshot({(1, 2)})


def test_04_numpy_dtype_shape_stride_and_logical_bytes_are_semantic() -> None:
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    view = base[:, ::2]
    copied = np.ascontiguousarray(view)
    assert not view.flags.c_contiguous
    # Same logical values but different original strides are distinct.
    assert S.semantic_snapshot_sha256(view) != S.semantic_snapshot_sha256(copied)
    assert S.semantic_snapshot_sha256(copied) != S.semantic_snapshot_sha256(
        copied.astype(np.float32)
    )
    assert S.semantic_snapshot_sha256(copied) != S.semantic_snapshot_sha256(
        copied.reshape(2, 3)
    )
    mutated = copied.copy()
    mutated[0, 0] += 1.0
    assert S.semantic_snapshot_sha256(copied) != S.semantic_snapshot_sha256(mutated)
    lone_base = np.arange(6, dtype=np.int64)
    lone_slice = lone_base[2:4]
    assert lone_slice.strides == np.asarray([2, 3], dtype=np.int64).strides
    # The slice's true backing-storage offset remains semantic even when no
    # sibling view is present in the object graph.
    assert S.semantic_snapshot_sha256(lone_slice) != S.semantic_snapshot_sha256(
        np.asarray([2, 3], dtype=np.int64)
    )


def test_05_repeated_reference_uses_stable_def_ref_graph() -> None:
    child_a: list[object] = ["value"]
    child_b: list[object] = ["value"]
    aliased_a = [child_a, child_a]
    aliased_b = [child_b, child_b]
    copied = [["value"], ["value"]]
    first = S.semantic_snapshot_evidence(aliased_a)
    second = S.semantic_snapshot_evidence(aliased_b)
    assert first[S.SEMANTIC_DIGEST_NAME] == second[S.SEMANTIC_DIGEST_NAME]
    assert first["reference_alias_edge_count"] == 1
    assert first["reference_cycle_edge_count"] == 0
    assert first[S.SEMANTIC_DIGEST_NAME] != S.semantic_snapshot_sha256(copied)
    reloaded = pickle.loads(pickle.dumps(aliased_a, protocol=4))
    assert first[S.SEMANTIC_DIGEST_NAME] == S.semantic_snapshot_sha256(reloaded)
    receipt_a = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    receipt_b = json.dumps(
        S.semantic_snapshot_evidence(aliased_a),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert receipt_a == receipt_b


def test_06_reference_cycle_uses_stable_ref_without_pointer_identity() -> None:
    left: list[object] = []
    left.append(left)
    right: list[object] = []
    right.append(right)
    a = S.semantic_snapshot_evidence(left)
    b = S.semantic_snapshot_evidence(right)
    assert a[S.SEMANTIC_DIGEST_NAME] == b[S.SEMANTIC_DIGEST_NAME]
    assert a["reference_alias_edge_count"] == 1
    assert a["reference_cycle_edge_count"] == 1


def test_07_shared_numpy_storage_ids_and_offsets_are_allocation_independent() -> None:
    def fixture() -> list[np.ndarray]:
        base = np.arange(16, dtype=np.int64)
        return [base, base[2:10:2], base[5:]]

    first = S.semantic_snapshot_evidence(fixture())
    second = S.semantic_snapshot_evidence(fixture())
    assert first[S.SEMANTIC_DIGEST_NAME] == second[S.SEMANTIC_DIGEST_NAME]
    shared = [row for row in first["storage_manifest"] if len(row["member_object_ids"]) > 1]
    assert len(shared) == 1
    encoded = S.canonical_semantic_snapshot(fixture())
    # Neither decimal/hex addresses nor implementation storage keys are fields.
    assert b"data_ptr" not in encoded and b"_cdata" not in encoded


def test_08_shared_torch_storage_and_logical_stride_are_stable() -> None:
    torch = pytest.importorskip("torch")

    def fixture() -> list[object]:
        base = torch.arange(12, dtype=torch.float32)
        return [base, base[1:9:2]]

    first = S.semantic_snapshot_evidence(fixture())
    second = S.semantic_snapshot_evidence(fixture())
    assert first[S.SEMANTIC_DIGEST_NAME] == second[S.SEMANTIC_DIGEST_NAME]
    assert any(len(row["member_object_ids"]) == 2 for row in first["storage_manifest"])
    assert all(row["semantic_device_class"] == "cpu" for row in first["tensor_device_manifest"])
    raw = torch.arange(16, dtype=torch.uint8)
    cross_dtype = raw[4:12].view(torch.int16)
    cross_evidence = S.semantic_snapshot_evidence([raw, cross_dtype])
    assert any(
        len(row["member_object_ids"]) == 2
        for row in cross_evidence["storage_manifest"]
    )
    assert S.semantic_snapshot_sha256(cross_dtype) != S.semantic_snapshot_sha256(
        cross_dtype.clone()
    )


def test_09_declared_structured_field_order_is_encoded_and_enforced() -> None:
    snapshot = _dynamic_branch_snapshot(step_index=7)
    evidence = S.semantic_snapshot_evidence(snapshot)
    record = evidence["structured_type_inventory"][0]
    assert record == {
        "type": "scripts.run_go2_oracle_branch_pilot_v1.BranchSnapshot",
        "declared_fields": list(
            S.STRUCTURED_FIELD_AUTHORITY[
                "scripts.run_go2_oracle_branch_pilot_v1.BranchSnapshot"
            ]
        ),
    }
    Bad = type(
        "BranchSnapshot",
        (),
        {
            "__module__": "scripts.run_go2_oracle_branch_pilot_v1",
            "__slots__": tuple(reversed(record["declared_fields"])),
        },
    )
    bad = Bad()
    for name in record["declared_fields"]:
        setattr(bad, name, None)
    with pytest.raises(S.SemanticSnapshotError, match="field inventory"):
        S.canonical_semantic_snapshot(bad)


def test_10_exact_solver_sentinel_nonfinite_authority_is_preserved() -> None:
    values = np.zeros((18, 2), dtype=np.float32)
    values[:3] = math.inf
    values[3:6] = -math.inf
    key = "Scene._sim._coupler.rigid_solver.dofs_info.force_range"
    snapshot = _dynamic_branch_snapshot(solver_state={key: values})
    first = S.semantic_snapshot_sha256(snapshot)
    second = S.semantic_snapshot_sha256(_dynamic_branch_snapshot(solver_state={key: values.copy()}))
    assert first == second
    assert S.semantic_snapshot_evidence(snapshot)["nonfinite_sentinel_inventory"] == [
        {
            "path": "$root/solver_state/Scene._sim._coupler.rigid_solver.dofs_info.force_range",
            "dtype_str": "<f4",
            "shape": [18, 2],
            "strides_bytes": [8, 4],
            "positive_infinity_count": 6,
            "negative_infinity_count": 6,
            "nan_count": 0,
        }
    ]


def test_11_nonfinite_python_or_off_authority_array_is_rejected() -> None:
    with pytest.raises(S.SemanticSnapshotError, match="nonfinite Python"):
        S.canonical_semantic_snapshot(float("inf"))
    with pytest.raises(S.SemanticSnapshotError, match="outside authority"):
        S.canonical_semantic_snapshot(np.asarray([math.inf], dtype=np.float32))
    values = np.zeros((18, 2), dtype=np.float32)
    values[0, 0] = np.nan
    key = "Scene._sim._coupler.rigid_solver.dofs_info.limit"
    with pytest.raises(S.SemanticSnapshotError, match="authority drift"):
        S.canonical_semantic_snapshot(_dynamic_branch_snapshot(solver_state={key: values}))


def test_12_unknown_type_and_numpy_object_dtype_fail_closed() -> None:
    class Unknown:
        pass

    with pytest.raises(S.SemanticSnapshotError, match="unknown semantic type"):
        S.canonical_semantic_snapshot(Unknown())
    with pytest.raises(S.SemanticSnapshotError, match="dtype"):
        S.canonical_semantic_snapshot(np.asarray([Unknown()], dtype=object))


def test_13_true_two_process_legacy_torch_pickle_drift_is_semantically_equal() -> None:
    script = """
        import hashlib, json, pickle, torch
        from lewm.safety.physical_graph_edge_handoff_snapshot_semantics_v1 import semantic_snapshot_sha256
        value = torch.arange(5056, dtype=torch.uint8)
        raw = pickle.dumps(value, protocol=4)
        print(json.dumps({
            'artifact': hashlib.sha256(raw).hexdigest(),
            'semantic': semantic_snapshot_sha256(value),
        }, sort_keys=True))
    """
    left = _run_genesis(script)
    right = _run_genesis(script)
    assert left["artifact"] != right["artifact"]
    assert left["semantic"] == right["semantic"]


def test_14_production_first_eight_match_v1_v2_semantics_and_device_rule() -> None:
    v1 = RECOVERY_ROOT / "physical_graph_edge_handoff_qualification_v1_material"
    v2 = RECOVERY_ROOT / "physical_graph_edge_handoff_qualification_v2_material"
    if not v1.is_dir() or not v2.is_dir():
        pytest.skip("immutable production snapshot fixture is unavailable")
    script = f"""
        import hashlib, json, pickle
        from pathlib import Path
        import numpy as np
        from lewm.safety.physical_graph_edge_handoff_snapshot_semantics_v1 import semantic_snapshot_evidence
        roots = [Path({str(v1)!r}), Path({str(v2)!r})]
        rows = []
        for pool_index in range(8):
          pair = []
          for root in roots:
            path = root / 'qualification' / f'pool-{{pool_index:03d}}' / 'payload.npz'
            with np.load(path, allow_pickle=False) as archive:
                raw = archive['snapshot_payload_bytes'].tobytes()
            snapshot = pickle.loads(raw)
            evidence = semantic_snapshot_evidence(snapshot)
            roundtrip = pickle.loads(pickle.dumps(snapshot, protocol=4))
            pair.append({{
                'artifact': hashlib.sha256(raw).hexdigest(),
                'semantic': evidence['snapshot_semantic_digest_v1'],
                'evidence_sha256': hashlib.sha256(
                    json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest(),
                'roundtrip_semantic': semantic_snapshot_evidence(roundtrip)['snapshot_semantic_digest_v1'],
                'reference_alias_edge_count': evidence['reference_alias_edge_count'],
                'reference_cycle_edge_count': evidence['reference_cycle_edge_count'],
                'type_inventory': evidence['type_inventory'],
                'structured_type_inventory': evidence['structured_type_inventory'],
                'tensor_device_manifest': evidence['tensor_device_manifest'],
            }})
          rows.append(pair)
        print(json.dumps(rows, sort_keys=True))
    """
    rows = _run_genesis(script)
    assert len(rows) == 8
    for pair in rows:
        left, right = pair
        assert left["artifact"] != right["artifact"]
        assert left["semantic"] == right["semantic"]
        assert left["evidence_sha256"] == right["evidence_sha256"]
        assert left["roundtrip_semantic"] == left["semantic"]
        assert right["roundtrip_semantic"] == right["semantic"]
        assert left["reference_alias_edge_count"] == 0
        assert left["reference_cycle_edge_count"] == 0
        inventory = {row["type"]: row["count"] for row in left["type_inventory"]}
        assert inventory["numpy.ndarray"] == 506
        assert inventory["torch.Tensor"] == 3
        assert inventory["builtins.str"] == 563
        assert left["structured_type_inventory"] == right["structured_type_inventory"]
        assert left["tensor_device_manifest"] == right["tensor_device_manifest"]
        assert [row["path"] for row in left["tensor_device_manifest"]] == [
            "$root/rng/torch",
            "$root/rng/torch_devices/0",
            "$root/rng/torch_devices/1",
        ]
        assert all(
            row["semantic_device_class"] == "cpu"
            and row["source_device_type"] == "cpu"
            and row["source_device_index"] is None
            for row in left["tensor_device_manifest"]
        )
