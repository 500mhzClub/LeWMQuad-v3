from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_causal_belief_state_h4_chain_metadata_census_v1.py"
)
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1",
    CONTRACT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _rehash(row: dict[str, Any]) -> dict[str, Any]:
    core = dict(row)
    core.pop("content_sha256", None)
    return contract.with_content_sha256(core)


def _pair(
    index: int,
    *,
    current: str,
    next_: str,
    role: str = "train",
    family: str = contract.FAMILIES[0],
    primitive: str = contract.PRIMITIVES[0],
    context_id: str = "default",
) -> dict[str, Any]:
    return _rehash({
        "schema": contract.PAIR_SCHEMA,
        "dataset_role": role,
        "global_row": index,
        "scene_id": f"scene-{context_id}",
        "family": family,
        "episode_id": f"episode-{context_id}",
        "env_index": 0,
        "reset_count": 0,
        "source_split": "development",
        "frames_jsonl_sha256": _sha(f"frames-{context_id}"),
        "scene_manifest_sha256": _sha(f"manifest-{context_id}"),
        "primitive": primitive,
        "relative_se2_current_frame": [0.0, 0.0, 0.0],
        "current_endpoint_sha256": current,
        "next_endpoint_sha256": next_,
        "label_shard_path_metadata_only": f"labels/{context_id}.npz",
        "label_shard_sha256": _sha(f"labels-{context_id}"),
        "label_shard_row": index,
        "sidecar_row_identity_sha256": _sha(f"sidecar-{index}"),
    })


def _projection(row: dict[str, Any]) -> dict[str, Any]:
    raw = contract.canonical_json_bytes(row) + b"\n"
    return contract._validate_pair(row, raw, line_number=row["global_row"] + 1)


def _chain(
    length: int,
    *,
    chain_id: str,
    global_offset: int = 0,
    role: str = "train",
    family: str = contract.FAMILIES[0],
    primitives: list[str] | None = None,
) -> list[dict[str, Any]]:
    endpoints = [
        _sha(f"endpoint-{role}-{family}-{chain_id}-{position}")
        for position in range(length + 1)
    ]
    actions = primitives or [
        contract.PRIMITIVES[position % len(contract.PRIMITIVES)]
        for position in range(length)
    ]
    return [
        _projection(_pair(
            global_offset + position,
            current=endpoints[position],
            next_=endpoints[position + 1],
            role=role,
            family=family,
            primitive=actions[position],
            context_id=f"{role}-{family}-{chain_id}",
        ))
        for position in range(length)
    ]


def _adequacy_rows(
    *,
    omit_selection_family: bool = False,
    omit_p2_primitive: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_row = 0
    for chain_index in range(64):
        family = contract.FAMILIES[chain_index % len(contract.FAMILIES)]
        actions = [
            contract.PRIMITIVES[(chain_index + position) % len(contract.PRIMITIVES)]
            for position in range(6)
        ]
        if omit_p2_primitive is not None and actions[2] == omit_p2_primitive:
            actions[2] = contract.PRIMITIVES[0]
        rows.extend(_chain(
            6,
            chain_id=f"train-{chain_index}",
            global_offset=global_row,
            role="train",
            family=family,
            primitives=actions,
        ))
        global_row += 6
    for chain_index, expected_family in enumerate(contract.FAMILIES):
        family = (
            contract.FAMILIES[0]
            if omit_selection_family and chain_index == len(contract.FAMILIES) - 1
            else expected_family
        )
        rows.extend(_chain(
            6,
            chain_id=f"selection-{chain_index}",
            global_offset=global_row,
            role="checkpoint_selection",
            family=family,
        ))
        global_row += 6
    return rows


def test_canonical_pair_schema_and_self_hash_rejections() -> None:
    row = _pair(
        0,
        current=_sha("current"),
        next_=_sha("next"),
    )
    raw = contract.canonical_json_bytes(row) + b"\n"
    assert contract.parse_canonical_json_line(raw, name="synthetic pair") == row
    assert _projection(row) == {
        field: row[field] for field in contract.PROJECTED_FIELDS
    }

    mutations: list[dict[str, Any]] = []
    missing = dict(row)
    missing.pop("family")
    mutations.append(_rehash(missing))
    extra = dict(row)
    extra["unexpected"] = 1
    mutations.append(_rehash(extra))
    wrong_type = dict(row)
    wrong_type["global_row"] = True
    mutations.append(_rehash(wrong_type))
    wrong_schema = dict(row)
    wrong_schema["schema"] = "wrong"
    mutations.append(_rehash(wrong_schema))
    wrong_primitive = dict(row)
    wrong_primitive["primitive"] = "teleport"
    mutations.append(_rehash(wrong_primitive))
    bad_hash = dict(row)
    bad_hash["content_sha256"] = "f" * 64
    mutations.append(bad_hash)

    for mutation in mutations:
        mutation_raw = contract.canonical_json_bytes(mutation) + b"\n"
        with pytest.raises(contract.CensusValidationError):
            contract._validate_pair(mutation, mutation_raw, line_number=1)

    noncanonical = json.dumps(row, sort_keys=True).encode("ascii") + b"\n"
    with pytest.raises(contract.CensusValidationError, match="canonical"):
        contract.parse_canonical_json_line(noncanonical, name="synthetic pair")
    with pytest.raises(contract.CensusValidationError, match="one canonical JSON line"):
        contract.parse_canonical_json_line(raw + b"\n", name="synthetic pair")


@pytest.mark.parametrize(
    ("edge_count", "expected_windows", "packed", "leftovers"),
    [
        (6, [6, 5, 4, 3, 2, 1], 1, 0),
        (7, [7, 6, 5, 4, 3, 2], 1, 1),
        (12, [12, 11, 10, 9, 8, 7], 2, 0),
    ],
)
def test_sliding_windows_and_offset_zero_packing(
    edge_count: int,
    expected_windows: list[int],
    packed: int,
    leftovers: int,
) -> None:
    integrity, graph, _ = contract.census_rows(
        _chain(edge_count, chain_id=f"length-{edge_count}")
    )
    assert integrity["all_zero"] is True
    assert graph["sliding_candidate_counts"]["train"] == {
        f"H{horizon}": expected_windows[horizon - 1]
        for horizon in range(1, 7)
    }
    assert graph["row_disjoint_h6"]["train"] == {
        "count": packed,
        "by_family": {
            family: packed if family == contract.FAMILIES[0] else 0
            for family in contract.FAMILIES
        },
        "leftover_edge_count": leftovers,
    }


def test_future_actions_are_exactly_p2_through_p5_for_every_sliding_h6() -> None:
    actions = list(contract.PRIMITIVES[:7])
    _, graph, _ = contract.census_rows(_chain(
        7,
        chain_id="future-positions",
        primitives=actions,
    ))
    histograms = graph["train_future_primitive_histograms"]
    for position in range(2, 6):
        expected = {primitive: 0 for primitive in contract.PRIMITIVES}
        expected[actions[position]] += 1
        expected[actions[position + 1]] += 1
        assert histograms[f"p{position}"] == expected

    expected_tuples = sorted([actions[2:6], actions[3:7]])
    expected_hash = contract.canonical_json_sha256({
        "domain": "train_future_action_tuple_multiset_v1",
        "role": "train",
        "family": contract.FAMILIES[0],
        "tuples": expected_tuples,
    })
    assert graph[
        "train_future_primitive_tuple_multiset_sha256_by_family"
    ][contract.FAMILIES[0]] == expected_hash


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "cycle",
            {
                "cycle_component_count": 1,
                "cycle_edge_count": 3,
                "uncovered_edge_row_count": 3,
            },
        ),
        ("branch", {"duplicate_current_owner_endpoint_count": 1}),
        ("cross_role", {"cross_role_endpoint_count": 1}),
    ],
)
def test_graph_integrity_rejects_cycle_branch_and_cross_role_reuse(
    case: str,
    expected: dict[str, int],
) -> None:
    endpoints = [_sha(f"integrity-endpoint-{index}") for index in range(4)]
    if case == "cycle":
        specifications = [
            ("train", endpoints[0], endpoints[1]),
            ("train", endpoints[1], endpoints[2]),
            ("train", endpoints[2], endpoints[0]),
        ]
    elif case == "branch":
        specifications = [
            ("train", endpoints[0], endpoints[1]),
            ("train", endpoints[0], endpoints[2]),
        ]
    else:
        specifications = [
            ("train", endpoints[0], endpoints[1]),
            ("checkpoint_selection", endpoints[1], endpoints[2]),
        ]
    rows = [
        _projection(_pair(
            index,
            current=current,
            next_=next_,
            role=role,
            context_id="shared-integrity-context",
        ))
        for index, (role, current, next_) in enumerate(specifications)
    ]
    integrity, _, adequacy = contract.census_rows(rows)
    assert integrity["all_zero"] is False
    for counter, count in expected.items():
        assert integrity["counts"][counter] == count
    assert adequacy["predicates"]["integrity_all_zero"] is False


def test_synthetic_family_primitive_and_adequacy_gates() -> None:
    integrity, graph, adequacy = contract.census_rows(_adequacy_rows())
    assert integrity["all_zero"] is True
    assert graph["row_disjoint_h6"]["train"]["count"] == 64
    assert graph["row_disjoint_h6"]["checkpoint_selection"]["count"] == 8
    assert all(adequacy["predicates"].values())
    assert adequacy["failed_predicates"] == []

    _, missing_family_graph, missing_family = contract.census_rows(
        _adequacy_rows(omit_selection_family=True)
    )
    assert missing_family_graph[
        "row_disjoint_h6"
    ]["checkpoint_selection"]["count"] == 8
    assert missing_family["predicates"][
        "selection_row_disjoint_h6_at_least_8"
    ] is True
    assert missing_family["predicates"][
        "selection_each_family_row_disjoint_h6_at_least_1"
    ] is False

    _, _, missing_primitive = contract.census_rows(_adequacy_rows(
        omit_p2_primitive=contract.PRIMITIVES[-1]
    ))
    assert missing_primitive["predicates"][
        "train_all_primitives_each_future_position"
    ] is False


def test_receipt_is_self_hashed_and_contains_no_row_level_witnesses() -> None:
    rows = _chain(6, chain_id="receipt-leakage")
    row_secret = "ROW_LEVEL_SECRET"
    endpoint_secret = "ENDPOINT_LEVEL_SECRET"
    context_secret = "CONTEXT_LEVEL_SECRET"
    rows[0] = {
        **rows[0],
        "content_sha256": row_secret,
        "current_endpoint_sha256": endpoint_secret,
        "scene_id": context_secret,
        "label_shard_path_metadata_only": "ROW_DERIVED_PATH_SECRET",
        "relative_se2_current_frame": "POSE_VALUE_SECRET",
    }
    integrity, graph, adequacy = contract.census_rows(rows)
    census = {
        "input_bindings": {
            "manifest": {"path": contract.MANIFEST_PATH},
            "pairs": {"path": contract.PAIRS_PATH},
        },
        "populations": {
            role: {
                "pair_count": len(rows) if role == "train" else 0,
                "scene_count": 0,
                "unique_endpoint_count": 0,
            }
            for role in contract.ROLES
        },
        "integrity": integrity,
        "graph": graph,
        "adequacy": adequacy,
    }
    receipt = contract.build_receipt(
        census,
        access=contract.default_access_receipt(),
        work=contract.default_work_receipt(),
    )
    assert contract.validate_receipt(receipt) == receipt
    raw = contract.canonical_json_bytes(receipt) + b"\n"
    assert contract.parse_canonical_json_line(raw, name="synthetic receipt") == receipt
    for secret in (
        row_secret,
        endpoint_secret,
        context_secret,
        "ROW_DERIVED_PATH_SECRET",
        "POSE_VALUE_SECRET",
    ):
        assert secret.encode("ascii") not in raw

    mutated = deepcopy(receipt)
    mutated["decision"] = (
        "H4_METADATA_FEASIBLE"
        if receipt["decision"] != "H4_METADATA_FEASIBLE"
        else "STOP_H4_METADATA_INADEQUATE"
    )
    with pytest.raises(contract.CensusValidationError, match="identity changed"):
        contract.validate_receipt(mutated)


def _load_runner(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[name] = runner
    spec.loader.exec_module(runner)
    return runner


def test_runner_import_and_parse_refusal_are_source_only(
    tmp_path: Path,
) -> None:
    runner = _load_runner("_test_h4_census_runner_help")
    runner.ROOT = tmp_path
    with pytest.raises(SystemExit) as stopped:
        runner.parse_args(["--help"])
    assert stopped.value.code == 0
    with pytest.raises(SystemExit) as refused:
        runner.parse_args([])
    assert refused.value.code == 2
    assert not (tmp_path / ".generated").exists()


def test_runner_output_root_reservation_is_exclusive(tmp_path: Path) -> None:
    runner = _load_runner("_test_h4_census_runner_reservation")
    runner.ROOT = tmp_path
    output_root = (tmp_path / runner.contract.OUTPUT_PATH).parent
    output_root.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner._reserve_output_root()
    assert list(output_root.iterdir()) == []


def test_runner_synthetic_normal_stop_publishes_one_canonical_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_runner("_test_h4_census_runner_normal_stop")
    runner.ROOT = tmp_path
    (tmp_path / ".generated").mkdir()
    monkeypatch.setattr(runner, "_validate_preregistration", lambda: None)
    monkeypatch.setattr(runner, "_source_bindings", lambda: [])

    def read_exact(path: Path) -> bytes:
        relative = path.relative_to(tmp_path).as_posix()
        if relative == runner.contract.AUTHORIZATION_PATH:
            return b"authorization"
        if relative == runner.contract.MANIFEST_PATH:
            return b"manifest"
        if relative == runner.contract.PAIRS_PATH:
            return b"pairs"
        raise AssertionError(f"unexpected synthetic read: {relative}")

    monkeypatch.setattr(runner, "_read_regular", read_exact)
    monkeypatch.setattr(
        runner.contract,
        "validate_authorization",
        lambda *args, **kwargs: {},
    )

    census = {
        "input_bindings": {"manifest": {}, "pairs": {}},
        "populations": {role: {} for role in runner.contract.ROLES},
        "integrity": {},
        "graph": {},
        "adequacy": {
            "predicates": {"input_contract_valid": False},
            "failed_predicates": ["input_contract_valid"],
        },
    }
    monkeypatch.setattr(
        runner.contract,
        "census_from_raw",
        lambda *args: census,
    )

    assert runner.run_parent(authorization_file_sha256="0" * 64) == 2
    receipt_path = tmp_path / runner.contract.OUTPUT_PATH
    raw = receipt_path.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    receipt = runner.contract.parse_canonical_json_line(
        raw, name="synthetic runner receipt"
    )
    assert runner.contract.validate_receipt(receipt) == receipt
    assert receipt["decision"] == "STOP_H4_METADATA_INADEQUATE"
    assert list(receipt_path.parent.iterdir()) == [receipt_path]
