from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = (
    ROOT
    / "scripts/preflight_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _role_rows(module: Any, role: str, *, scene: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in range(2):
        for action_index, action in enumerate(module.contract.ACTION_VOCABULARY):
            rows.append(
                {
                    "dataset_role": role,
                    "role_state_index": state,
                    "scene_id": scene,
                    "family": "open_obstacle_field",
                    "current_endpoint_sha256": f"{state + 1:064x}",
                    "action_index": action_index,
                    "action": action,
                    "station_safe": [bool(state)] * 11,
                    "immediate_primitive": {"feasible": True},
                    "blind_bridge": {"feasible": True},
                }
            )
    return rows


def test_role_bridge_wrong_rgb_and_action_prior_are_label_only() -> None:
    module = _load("_test_projective_support_preflight_bridges")
    rows_by_role = {
        role: _role_rows(module, role, scene=f"scene-{role}")
        for role in module.contract.ROLE_ORDER
    }
    station, immediate, blind, scenes, families, endpoints = module._role_arrays(
        rows_by_role["train"]
    )
    assert station.shape == (2, 9, 11)
    assert immediate.all() and blind.all()
    assert scenes == ("scene-train", "scene-train")
    assert families == ("open_obstacle_field", "open_obstacle_field")
    assert len(set(endpoints)) == 2

    pairs_by_role = {
        role: tuple(
            {
                "dataset_role": role,
                "scene_id": f"scene-{role}",
                "current_endpoint_sha256": f"{state + 1:064x}",
                "next_endpoint_sha256": f"{state + 101:064x}",
            }
            for state in range(2)
        )
        for role in module.contract.ROLE_ORDER
    }
    mapping = module.wrong_rgb_mapping_binding_v1(
        rows_by_role,
        pairs_by_role,
        enforce_frozen_counts=False,
    )
    assert mapping["row_count"] == 6
    assert [mapping["per_role"][role]["row_count"] for role in module.contract.ROLE_ORDER] == [2, 2, 2]
    assert len(mapping["mapping_sha256"]) == 64
    assert mapping["paired_next_collision_count"] == 0
    assert mapping["mapped_endpoint_is_never_paired_next"] is True

    collided = {role: list(rows) for role, rows in pairs_by_role.items()}
    collided["train"][0] = {
        **collided["train"][0],
        "next_endpoint_sha256": f"{2:064x}",
    }
    with pytest.raises(PermissionError, match="paired future endpoint"):
        module.wrong_rgb_mapping_binding_v1(
            rows_by_role,
            collided,
            enforce_frozen_counts=False,
        )


def test_oracle_stop_writes_complete_bound_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load("_test_projective_support_preflight_failure")
    source_binding = {"path": "docs/source.json", "file_sha256": "a" * 64}
    review_binding = {"path": "docs/review.json", "file_sha256": "b" * 64}
    monkeypatch.setattr(
        module,
        "_source_custody",
        lambda root: (b"source", b"review", source_binding, review_binding),
    )
    binding_path = tmp_path / module.labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH
    binding_path.parent.mkdir(parents=True)
    binding_path.write_bytes(b"bound-builder\n")
    manifest_path = tmp_path / module.contract.LABEL_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"bound-manifest\n")
    manifest = {
        "content_sha256": "c" * 64,
        "files": [
            {
                "path": Path(relative).name,
                "file_sha256": f"{index + 1:x}" * 64,
                "byte_count": index + 1,
            }
            for index, relative in enumerate(module.authority.LABEL_FILE_PATHS)
        ],
    }
    monkeypatch.setattr(module, "build_from_binding", lambda *args, **kwargs: manifest)
    rows_by_role = {
        role: tuple(_role_rows(module, role, scene=f"scene-{role}"))
        for role in module.contract.ROLE_ORDER
    }
    monkeypatch.setattr(
        module.labels,
        "load_role_labels_v1",
        lambda path, *, role, expected_file_sha256: rows_by_role[role],
    )
    monkeypatch.setattr(
        module,
        "_preflight_inputs",
        lambda root: (
            {
                "raw_manifest": tmp_path / "raw-manifest",
                "raw_pairs": tmp_path / "raw-pairs",
                "raw_endpoints": tmp_path / "raw-endpoints",
            },
            {},
        ),
    )
    raw_pairs = tuple(
        {
            "dataset_role": role,
            "scene_id": f"scene-{role}",
            "current_endpoint_sha256": f"{state + 1:064x}",
            "next_endpoint_sha256": f"{state + 101:064x}",
        }
        for role in module.contract.ROLE_ORDER
        for state in range(2)
    )
    monkeypatch.setattr(
        module.labels,
        "load_and_validate_raw_indexes",
        lambda *args, **kwargs: SimpleNamespace(pairs=raw_pairs),
    )
    oracle = SimpleNamespace(
        passed=False,
        failed_checks=("synthetic_oracle",),
        checks={"synthetic_oracle": False},
    )
    monkeypatch.setattr(
        module.metrics,
        "oracle_metric_pipeline_preflight",
        lambda *args, **kwargs: oracle,
    )

    with pytest.raises(PermissionError, match="oracle metric-pipeline preflight STOP"):
        module.run_label_preflight_v1(repository_root=tmp_path)
    failure = module.contract.parse_canonical_json(
        (tmp_path / module.LABEL_PREFLIGHT_FAILURE_RELATIVE_PATH).read_bytes(),
        name="label preflight failure",
    )
    assert failure["phase"] == "oracle_metric_pipeline"
    assert failure["source_manifest"] == source_binding
    assert failure["independent_source_review"] == review_binding
    assert failure["label_builder_execution_binding"]["byte_count"] > 0
    assert failure["label_manifest"]["content_sha256"] == "c" * 64
    assert set(failure["label_files"]) == set(module.authority.LABEL_FILE_PATHS)
    assert failure["oracle_metric_pipeline"]["failed_checks"] == [
        "synthetic_oracle"
    ]
    assert failure["wrong_rgb_mapping"] is None
    assert failure["action_prior"] is None
    assert set(failure["protected_access_counts"].values()) == {0}
    assert failure["training_authorized"] is False

    prior = module.action_prior_binding_v1(
        rows_by_role["train"],
        enforce_frozen_count=False,
    )
    assert prior["shape"] == [9, 11]
    assert all(value == 0.5 for row in prior["probabilities"] for value in row)
    assert len(prior["probabilities_sha256"]) == 64


def test_oracle_mapping_is_exactly_json_native() -> None:
    module = _load("_test_projective_support_preflight_oracle")
    source = SimpleNamespace(
        passed=True,
        failed_checks=(),
        checks={"one": True, "two": True},
    )
    assert module.oracle_metric_mapping_v1(source) == {
        "status": "PASS",
        "passed": True,
        "failed_checks": [],
        "checks": {"one": True, "two": True},
    }


def test_source_inventory_binding_is_exactly_role_scene_purpose_ordered() -> None:
    module = _load("_test_projective_support_preflight_inventory")
    shards: dict[str, dict[str, str]] = {}
    inventory: list[dict[str, Any]] = []
    role_counts = {"train": 72, "probability_calibration": 8, "checkpoint_selection": 8}
    for role in module.contract.ROLE_ORDER:
        for index in range(role_counts[role]):
            scene = f"{role}-{index:02d}"
            shards[scene] = {
                "dataset_role": role,
                "family": "open_obstacle_field",
            }
            for purpose_index, purpose in enumerate(module.SOURCE_PURPOSES):
                inventory.append(
                    {
                        "path": f"sources/{scene}/{purpose}",
                        "byte_count": 10 + purpose_index,
                        "file_sha256": f"{purpose_index + 1:x}" * 64,
                        "purpose": purpose,
                        "scene_id": scene,
                    }
                )
    raw = SimpleNamespace(
        manifest={"input_provenance": {"source_payload_inventory": inventory}},
        shard_by_scene=shards,
    )
    records = module.source_records_from_raw_indexes_v1(raw)
    assert len(records) == 264
    assert [row["dataset_role"] for row in records[:3]] == ["train"] * 3
    assert [row["purpose"] for row in records[:3]] == list(module.SOURCE_PURPOSES)
    assert records[-1]["dataset_role"] == "checkpoint_selection"


def test_label_file_binding_uses_only_the_six_manifest_records() -> None:
    module = _load("_test_projective_support_preflight_label_files")
    records = []
    for index, relative in enumerate(module.authority.LABEL_FILE_PATHS, start=1):
        records.append(
            {
                "path": Path(relative).name,
                "file_sha256": f"{index:x}" * 64,
                "byte_count": index,
            }
        )
    manifest = {"files": records}
    bindings = module._label_file_bindings(manifest)
    assert set(bindings) == set(module.authority.LABEL_FILE_PATHS)
    assert all(value["path"] == key for key, value in bindings.items())
