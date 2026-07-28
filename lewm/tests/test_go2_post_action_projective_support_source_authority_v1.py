from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_PATH = (
    ROOT
    / "lewm/benchmarks/go2_post_action_projective_support_source_authority_v1.py"
)


def _load(name: str = "_post_action_source_authority_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, AUTHORITY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_import_is_tensor_and_data_free() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_authority", {str(AUTHORITY_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
print("PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def test_closure_is_exact_recursive_entrypoint_and_dynamic_source_closure() -> None:
    authority = _load("_post_action_source_authority_closure")
    assert tuple(authority._geometry.SOURCE_PATHS) == (
        authority.INHERITED_GEOMETRY_SOURCE_PATHS
    )
    discovered = authority.discover_recursive_source_closure_v1()
    assert discovered == authority.RECURSIVE_PYTHON_SOURCE_PATHS
    assert authority.SOURCE_PATHS == tuple(sorted({
        *discovered,
        authority.contract.PREREGISTRATION_RELATIVE_PATH,
        authority.contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
    }))
    assert len(discovered) == 109
    assert len(authority.SOURCE_PATHS) == len(discovered) + 2 == 111
    assert len(authority.INHERITED_GEOMETRY_SOURCE_PATHS) == 74
    assert len(authority.ADDITIVE_SOURCE_PATHS) == 19
    assert set(authority.IMPLEMENTATION_AUTHORS) == {
        "/root",
        "/root/counterfactual_label_mapping",
        "/root/joint_jepa_integration",
        "/root/probe_gate_review",
        "/root/label_boundary_fix",
        "/root/execution_authority_fix",
        "/root/attempt_runner",
        "/root/authority_source_review",
        "/root/authority_v2_adapter",
    }
    assert authority.SOURCE_MANIFEST_RELATIVE_PATH.endswith(
        "source_manifest_v2_2026-07-28.json"
    )
    assert authority.SOURCE_REVIEW_RELATIVE_PATH.endswith(
        "source_review_v2_2026-07-28.json"
    )
    assert authority.EXECUTION_BINDING_RELATIVE_PATH.endswith(
        "execution_binding_v2_2026-07-28.json"
    )
    assert "labels_v2_execution_binding" in (
        authority.LABEL_BUILDER_EXECUTION_BINDING_RELATIVE_PATH
    )
    assert authority.LABEL_PREFLIGHT_RECEIPT_RELATIVE_PATH.endswith(
        "labels_v2_preflight_receipt.json"
    )
    assert authority.contract.SOURCE_MANIFEST_SCHEMA.endswith(
        "_source_manifest_v1"
    )
    assert authority.contract.SOURCE_REVIEW_SCHEMA.endswith("_source_review_v1")
    assert authority.contract.EXECUTION_BINDING_SCHEMA.endswith(
        "_execution_binding_v1"
    )
    assert authority.LABEL_BUILDER_EXECUTION_BINDING_SCHEMA.endswith(
        "_execution_binding_v1"
    )
    assert set(authority.SOURCE_MANIFEST_ENTRYPOINTS) == {
        authority.LABEL_BUILDER_RELATIVE_PATH,
        authority.PREFLIGHT_RELATIVE_PATH,
        authority.EXECUTE_RELATIVE_PATH,
        authority.CORE_RUNNER_RELATIVE_PATH,
    }
    for path in (
        authority.AUTHORITY_RELATIVE_PATH,
        authority.contract.PREREGISTRATION_RELATIVE_PATH,
        authority.contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
        authority.PREFLIGHT_TEST_RELATIVE_PATH,
        authority.EXECUTE_TEST_RELATIVE_PATH,
    ):
        assert path in authority.ADDITIVE_SOURCE_PATHS
    for path in (
        "lewm/planning/__init__.py",
        "lewm/planning/geometry_contract.py",
        "lewm/planning/oriented_footprint.py",
        "lewm_worlds/lewm_worlds/__init__.py",
        "lewm_worlds/lewm_worlds/manifest.py",
        "lewm_worlds/lewm_worlds/scene_graph.py",
        "lewm_worlds/lewm_worlds/scene_validation.py",
    ):
        assert path in discovered

    omitted = tuple(
        path for path in discovered
        if path != "lewm/planning/oriented_footprint.py"
    )
    with pytest.raises(PermissionError, match="oriented_footprint.py"):
        authority.validate_recursive_source_paths_v1(omitted, discovered)


def _install_synthetic_source_layout(
    authority: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preregistration = "docs/synthetic_preregistration.md"
    amendment = "docs/synthetic_integrity_adapter_amendment.md"
    files = {
        "base.py": b"BASE = True\n",
        "entry.py": b"ENTRY = True\n",
        preregistration: b"# synthetic preregistration\n",
        amendment: b"# synthetic integrity-adapter amendment\n",
    }
    for relative, raw in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(authority._geometry, "SOURCE_PATHS", ("base.py",))
    monkeypatch.setattr(authority, "INHERITED_GEOMETRY_SOURCE_PATHS", ("base.py",))
    monkeypatch.setattr(
        authority,
        "ADDITIVE_SOURCE_PATHS",
        tuple(sorted(("entry.py", preregistration, amendment))),
    )
    monkeypatch.setattr(
        authority,
        "SOURCE_PATHS",
        tuple(sorted(("base.py", "entry.py", preregistration, amendment))),
    )
    monkeypatch.setattr(
        authority,
        "RECURSIVE_PYTHON_SOURCE_PATHS",
        ("base.py", "entry.py"),
    )
    monkeypatch.setattr(
        authority,
        "SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES",
        ("base.py", "entry.py"),
    )
    monkeypatch.setattr(authority, "SOURCE_MANIFEST_ENTRYPOINTS", ("entry.py",))
    monkeypatch.setattr(
        authority,
        "discover_recursive_source_closure_v1",
        lambda **_kwargs: ("base.py", "entry.py"),
    )
    monkeypatch.setattr(
        authority.contract, "PREREGISTRATION_RELATIVE_PATH", preregistration
    )
    monkeypatch.setattr(authority.contract, "PREREGISTRATION_COMMIT", "1" * 40)
    monkeypatch.setattr(
        authority.contract,
        "PREREGISTRATION_FILE_SHA256",
        hashlib.sha256(files[preregistration]).hexdigest(),
    )
    monkeypatch.setattr(
        authority.contract,
        "PREREGISTRATION_BYTE_COUNT",
        len(files[preregistration]),
    )
    monkeypatch.setattr(
        authority.contract,
        "INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH",
        amendment,
    )
    monkeypatch.setattr(
        authority.contract,
        "INTEGRITY_ADAPTER_AMENDMENT_FILE_SHA256",
        hashlib.sha256(files[amendment]).hexdigest(),
    )
    monkeypatch.setattr(
        authority.contract,
        "INTEGRITY_ADAPTER_AMENDMENT_BYTE_COUNT",
        len(files[amendment]),
    )


def _manifest_and_review(
    authority: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[bytes, bytes]:
    _install_synthetic_source_layout(authority, monkeypatch, tmp_path)
    manifest = authority.build_source_manifest(root=tmp_path)
    manifest_raw = authority.canonical_document_bytes(manifest)
    assert authority.validate_source_manifest(manifest_raw, root=tmp_path) == manifest
    review = authority.build_source_review_receipt(
        manifest_raw,
        reviewer="/root/independent_reviewer",
        source_freeze_commit="a" * 40,
        root=tmp_path,
    )
    review_raw = authority.canonical_document_bytes(review)
    assert authority.validate_source_review_receipt(
        review_raw, manifest_raw, root=tmp_path
    ) == review
    return manifest_raw, review_raw


def test_manifest_and_review_are_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _load("_post_action_source_authority_manifest")
    manifest_raw, review_raw = _manifest_and_review(
        authority, monkeypatch, tmp_path
    )
    manifest = authority.contract.parse_canonical_json(
        manifest_raw, name="synthetic manifest"
    )
    review = authority.contract.parse_canonical_json(
        review_raw, name="synthetic review"
    )
    assert manifest["source_count"] == 4
    assert manifest["integrity_adapter_amendment"] == (
        authority.contract.integrity_adapter_amendment_binding()
    )
    assert manifest["label_v1_terminal_predecessor_bindings"] == (
        authority.contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
    )
    assert not set(
        binding["path"]
        for binding in authority.contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS.values()
    ).intersection(manifest["source_paths"])
    assert review["integrity_adapter_amendment"] == manifest[
        "integrity_adapter_amendment"
    ]
    assert review["label_v1_terminal_predecessor_bindings"] == manifest[
        "label_v1_terminal_predecessor_bindings"
    ]
    assert review["science_contract"]["integrity_adapter_amendment"] == (
        manifest["integrity_adapter_amendment"]
    )
    assert review["science_contract"][
        "label_v1_terminal_predecessor_bindings"
    ] == manifest["label_v1_terminal_predecessor_bindings"]

    changed_manifest = copy.deepcopy(manifest)
    changed_manifest.pop("content_sha256")
    changed_manifest["label_v1_terminal_predecessor_bindings"]["failure"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(PermissionError, match="exact current closure"):
        authority.validate_source_manifest(
            authority.canonical_document_bytes(
                authority.contract.with_content_sha256(changed_manifest)
            ),
            root=tmp_path,
        )

    changed_review = copy.deepcopy(review)
    changed_review.pop("content_sha256")
    changed_review["integrity_adapter_amendment"]["file_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="source review receipt changed"):
        authority.validate_source_review_receipt(
            authority.canonical_document_bytes(
                authority.contract.with_content_sha256(changed_review)
            ),
            manifest_raw,
            root=tmp_path,
        )

    amendment_path = (
        tmp_path / authority.contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH
    )
    amendment_raw = amendment_path.read_bytes()
    amendment_path.write_bytes(b"# changed integrity-adapter amendment\n")
    with pytest.raises(PermissionError, match="amendment identity changed"):
        authority.build_source_manifest(root=tmp_path)
    amendment_path.write_bytes(amendment_raw)

    (tmp_path / "entry.py").write_bytes(b"ENTRY = False\n")
    with pytest.raises(PermissionError, match="exact current closure"):
        authority.validate_source_manifest(manifest_raw, root=tmp_path)
    with pytest.raises(PermissionError, match="not independent"):
        authority.build_source_review_receipt(
            authority.canonical_document_bytes(
                authority.build_source_manifest(root=tmp_path)
            ),
            reviewer="/root/joint_jepa_integration",
            source_freeze_commit="a" * 40,
            root=tmp_path,
        )
    with pytest.raises(PermissionError, match="not independent"):
        authority.build_source_review_receipt(
            authority.canonical_document_bytes(
                authority.build_source_manifest(root=tmp_path)
            ),
            reviewer="/root/authority_v2_adapter",
            source_freeze_commit="a" * 40,
            root=tmp_path,
        )
    with pytest.raises(PermissionError, match="not independent"):
        authority.build_source_review_receipt(
            authority.canonical_document_bytes(
                authority.build_source_manifest(root=tmp_path)
            ),
            reviewer="/root/authority_source_review",
            source_freeze_commit="a" * 40,
            root=tmp_path,
        )
    with pytest.raises(PermissionError, match="not independent"):
        authority.build_source_review_receipt(
            authority.canonical_document_bytes(
                authority.build_source_manifest(root=tmp_path)
            ),
            reviewer="/root/probe_gate_review",
            source_freeze_commit="a" * 40,
            root=tmp_path,
        )


def _label_fixture(
    authority: Any,
    label_builder_binding_raw: bytes | None = None,
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    if label_builder_binding_raw is None:
        label_builder_binding_raw = _label_builder_binding_fixture(authority)
    label_builder = authority.contract.parse_canonical_json(
        label_builder_binding_raw,
        name="synthetic label-builder binding",
    )
    bindings: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for index, role in enumerate(authority.contract.ROLE_ORDER, start=1):
        path = authority.contract.LABEL_ROLE_RELATIVE_PATHS[role]
        counts = authority.contract.ROLE_COUNTS[role]
        binding = {
            "path": path,
            "file_sha256": f"{index:x}" * 64,
            "byte_count": 100 + index,
        }
        bindings[path] = binding
        records.append({
            "path": Path(path).name,
            "file_sha256": binding["file_sha256"],
            "byte_count": binding["byte_count"],
            "schema": authority.contract.LABEL_ROW_SCHEMA,
            "dataset_role": role,
            "state_count": counts["states"],
            "action_row_count": counts["action_rows"],
        })
    for key in (
        "predicted_next_corridor_masks",
        "persistence_corridor_masks",
        "projective_support_mask",
    ):
        expected = authority.STATIC_MASK_EXPECTATIONS[f"{key}.u1"]
        path = authority.LABEL_MASK_RELATIVE_PATHS[key]
        bindings[path] = {
            "path": path,
            "file_sha256": expected["file_sha256"],
            "byte_count": expected["byte_count"],
        }
        records.append({
            "path": Path(path).name,
            "file_sha256": expected["file_sha256"],
            "byte_count": expected["byte_count"],
            "dtype": "|u1",
            "shape": expected["shape"],
            "set_cell_count": expected["set_cell_count"],
        })
    records.sort(key=lambda row: row["path"])
    non_hold = [
        action for action in authority.contract.ACTION_VOCABULARY if action != "hold"
    ]
    support = {
        population: {
            action: [
                {"safe": 1, "unsafe": 1}
                for _ in range(authority.contract.STATION_COUNT)
            ]
            for action in non_hold
        }
        for population in ("train", "calibration_plus_selection")
    }
    structural_preflight = {
        "exact_state_count": authority.contract.TOTAL_STATES,
        "exact_action_row_count": authority.contract.TOTAL_ACTION_ROWS,
        "exact_station_label_count": authority.contract.TOTAL_STATION_LABELS,
        "informative_state_counts": {
            "train": 512,
            "probability_calibration": 128,
            "checkpoint_selection": 128,
        },
        "train_action_ranking_participation_counts": {
            action: 1 for action in non_hold
        },
        "selection_family_informative_counts": {
            family: 8 for family in authority.contract.SCENE_FAMILIES
        },
        "role_scene_and_endpoint_disjoint": True,
        "role_scene_counts": {
            role: authority.contract.ROLE_COUNTS[role]["scenes"]
            for role in authority.contract.ROLE_ORDER
        },
        "minimum_states_per_role_scene": {
            role: 2 for role in authority.contract.ROLE_ORDER
        },
        "safe_unsafe_support": support,
        "every_non_hold_action_station_has_safe_and_unsafe_support": True,
        "frozen_schedule": {
            "presentation_count": authority.contract.MAXIMUM_PRESENTATIONS,
            "presentation_indices_sha256": (
                authority.contract.SCHEDULE_PREFIX_SHA256
            ),
            "informative_presentation_count": 512,
            "ranking_participation_presentations_by_action": {
                action: 32 for action in non_hold
            },
        },
    }
    reservation = authority.contract.with_content_sha256({
        "schema": authority.LABEL_RESERVATION_SCHEMA,
        "status": "RESERVED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT",
        "preregistration": authority.preregistration_binding(),
        "execution_binding_path": (
            authority.LABEL_BUILDER_EXECUTION_BINDING_RELATIVE_PATH
        ),
        "source_manifest": label_builder["source_manifest"],
        "independent_source_review": label_builder[
            "independent_source_review"
        ],
        "output_directory": authority.contract.LABEL_ROOT_RELATIVE_PATH,
        "attempt": {
            "index": 1,
            "maximum_attempts": 1,
            "retry_authorized": False,
            "resume_authorized": False,
            "second_invocation_authorized": False,
        },
        "access_ledger": dict(authority._LABEL_RESERVATION_ACCESS_LEDGER),
        "authority": dict(authority._LABEL_RESERVATION_AUTHORITY),
    })
    claim = authority.contract.with_content_sha256({
        "schema": authority.LABEL_BUILDER_CLAIM_SCHEMA,
        "status": "CLAIMED_ONE_EXACT_LABEL_BUILDER_INVOCATION",
        "reservation_content_sha256": reservation["content_sha256"],
        "execution_binding_content_sha256": label_builder["content_sha256"],
        "retry_authorized": False,
        "resume_authorized": False,
        "second_invocation_authorized": False,
    })
    manifest = authority.contract.with_content_sha256({
        "schema": authority.contract.LABEL_MANIFEST_SCHEMA,
        "status": "complete_pre_gpu_development_labels",
        "preregistration_commit": authority.contract.PREREGISTRATION_COMMIT,
        "roles": list(authority.contract.ROLE_ORDER),
        "action_order": list(authority.contract.ACTION_VOCABULARY),
        "state_count": authority.contract.TOTAL_STATES,
        "action_row_count": authority.contract.TOTAL_ACTION_ROWS,
        "station_label_count": authority.contract.TOTAL_STATION_LABELS,
        "files": records,
        "preflight": structural_preflight,
        "input_bindings": {
            "label_reservation": reservation,
            "label_builder_claim": claim,
            "integrity_adapter_amendment": label_builder[
                "integrity_adapter_amendment"
            ],
            "label_v1_terminal_predecessor_bindings": label_builder[
                "label_v1_terminal_predecessor_bindings"
            ],
            "source_manifest": label_builder["source_manifest"],
            "independent_source_review": label_builder[
                "independent_source_review"
            ],
            "execution_binding_content_sha256": label_builder[
                "content_sha256"
            ],
            "source_records_sha256": authority.contract.canonical_json_sha256(
                label_builder["source_records"]
            ),
            "schedule_prefix_sha256": (
                authority.contract.SCHEDULE_PREFIX_SHA256
            ),
        },
    })
    return authority.canonical_document_bytes(manifest), bindings


def _label_builder_binding_fixture(
    authority: Any,
    *,
    source_manifest: dict[str, Any] | None = None,
    source_review: dict[str, Any] | None = None,
) -> bytes:
    source_manifest = source_manifest or {
        "path": authority.SOURCE_MANIFEST_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "byte_count": 1,
    }
    source_review = source_review or {
        "path": authority.SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": "c" * 64,
        "content_sha256": "d" * 64,
        "byte_count": 1,
    }
    value = authority.contract.with_content_sha256({
        "schema": authority.LABEL_BUILDER_EXECUTION_BINDING_SCHEMA,
        "status": "AUTHORIZED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT",
        "preregistration_commit": authority.contract.PREREGISTRATION_COMMIT,
        "integrity_adapter_amendment": (
            authority.contract.integrity_adapter_amendment_binding()
        ),
        "label_v1_terminal_predecessor_bindings": copy.deepcopy(
            authority.contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
        ),
        "source_manifest": source_manifest,
        "independent_source_review": source_review,
        "output_directory": authority.contract.LABEL_ROOT_RELATIVE_PATH,
        "schedule_prefix_sha256": authority.contract.SCHEDULE_PREFIX_SHA256,
        "source_records": [{"index": index} for index in range(264)],
        "authority": {
            "development_label_preflight_authorized": True,
            "training_authorized": False,
            "heldout_authorized": False,
        },
    })
    return authority.canonical_document_bytes(value)


def _wrong_rgb_mapping_fixture(authority: Any) -> dict[str, Any]:
    return {
        "algorithm": authority.WRONG_RGB_MAPPING_ALGORITHM,
        "roles": list(authority.contract.ROLE_ORDER),
        "row_count": authority.contract.TOTAL_STATES,
        "mapping_sha256": "a" * 64,
        "per_role": {
            role: {
                "row_count": authority.contract.ROLE_COUNTS[role]["states"],
                "mapping_sha256": f"{index:x}" * 64,
            }
            for index, role in enumerate(authority.contract.ROLE_ORDER, start=1)
        },
        "paired_next_collision_count": 0,
        "paired_next_collision_rows_sha256": (
            authority.contract.canonical_json_sha256([])
        ),
        "mapped_endpoint_is_never_paired_next": True,
    }


@pytest.mark.parametrize(
    "field",
    ("integrity_adapter_amendment", "label_v1_terminal_predecessor_bindings"),
)
def test_label_builder_rejects_changed_adapter_governance(field: str) -> None:
    authority = _load(f"_post_action_source_authority_builder_{field}")
    value = authority.contract.parse_canonical_json(
        _label_builder_binding_fixture(authority),
        name="synthetic label-builder binding",
    )
    value = copy.deepcopy(value)
    value.pop("content_sha256")
    if field == "integrity_adapter_amendment":
        value[field]["file_sha256"] = "0" * 64
    else:
        value[field]["failure"]["content_sha256"] = "0" * 64
    changed_raw = authority.canonical_document_bytes(
        authority.contract.with_content_sha256(value)
    )
    with pytest.raises(
        PermissionError, match="label-builder execution binding changed"
    ):
        authority._label_builder_execution_binding(changed_raw)


def _action_prior_fixture(authority: Any) -> dict[str, Any]:
    probabilities = [
        [0.5 for _ in range(authority.contract.STATION_COUNT)]
        for _ in authority.contract.ACTION_VOCABULARY
    ]
    return {
        "source_role": "train",
        "source_roles": ["train"],
        "source_state_count": authority.contract.ROLE_COUNTS["train"]["states"],
        "action_order": list(authority.contract.ACTION_VOCABULARY),
        "station_count": authority.contract.STATION_COUNT,
        "shape": [
            len(authority.contract.ACTION_VOCABULARY),
            authority.contract.STATION_COUNT,
        ],
        "probabilities": probabilities,
        "probabilities_sha256": authority.contract.canonical_json_sha256(
            probabilities
        ),
    }


def _preflight_receipt_fixture(
    authority: Any,
    label_builder_binding_raw: bytes,
    label_manifest_raw: bytes,
    label_files: dict[str, dict[str, Any]],
) -> bytes:
    oracle = {
        "status": "PASS",
        "passed": True,
        "failed_checks": [],
        "checks": {
            name: True for name in authority.ORACLE_METRIC_PREFLIGHT_CHECKS
        },
    }
    receipt = authority.build_label_preflight_receipt(
        label_builder_binding_raw,
        label_manifest_raw,
        label_files,
        oracle_metric_pipeline=oracle,
        wrong_rgb_mapping=_wrong_rgb_mapping_fixture(authority),
        action_prior=_action_prior_fixture(authority),
    )
    return authority.canonical_document_bytes(receipt)


def test_label_preflight_receipt_is_exact_train_only_and_fail_closed() -> None:
    authority = _load("_post_action_source_authority_preflight")
    label_builder_binding_raw = _label_builder_binding_fixture(authority)
    label_manifest_raw, label_files = _label_fixture(
        authority, label_builder_binding_raw
    )
    receipt_raw = _preflight_receipt_fixture(
        authority,
        label_builder_binding_raw,
        label_manifest_raw,
        label_files,
    )
    receipt = authority.validate_label_preflight_receipt(
        receipt_raw,
        label_builder_binding_raw,
        label_manifest_raw,
        label_files,
    )
    assert receipt["action_prior"]["source_roles"] == ["train"]
    assert set(receipt["oracle_metric_pipeline"]["checks"]) == (
        authority.ORACLE_METRIC_PREFLIGHT_CHECKS
    )
    assert len(receipt["oracle_metric_pipeline"]["checks"]) == 29
    assert receipt["access_ledger"] == authority.LABEL_PREFLIGHT_ACCESS_LEDGER
    chain = receipt["label_materialization_chain"]
    assert chain["label_reservation"]["path"] == (
        authority.LABEL_RESERVATION_RELATIVE_PATH
    )
    assert chain["label_builder_claim"]["path"] == (
        authority.LABEL_BUILDER_CLAIM_RELATIVE_PATH
    )
    assert chain["label_builder_execution_binding_content_sha256"] == (
        authority.contract.parse_canonical_json(
            label_builder_binding_raw,
            name="synthetic builder",
        )["content_sha256"]
    )
    assert chain["integrity_adapter_amendment"] == (
        authority.contract.integrity_adapter_amendment_binding()
    )
    assert chain["label_v1_terminal_predecessor_bindings"] == (
        authority.contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
    )

    escaped_manifest = authority.contract.parse_canonical_json(
        label_manifest_raw,
        name="synthetic label manifest",
    )
    escaped_manifest = copy.deepcopy(escaped_manifest)
    escaped_manifest.pop("content_sha256")
    escaped_manifest["input_bindings"][
        "execution_binding_content_sha256"
    ] = "f" * 64
    escaped_manifest = authority.contract.with_content_sha256(escaped_manifest)
    with pytest.raises(PermissionError, match="builder provenance"):
        authority.build_label_preflight_receipt(
            label_builder_binding_raw,
            authority.canonical_document_bytes(escaped_manifest),
            label_files,
            oracle_metric_pipeline=receipt["oracle_metric_pipeline"],
            wrong_rgb_mapping=_wrong_rgb_mapping_fixture(authority),
            action_prior=_action_prior_fixture(authority),
        )

    escaped_governance = authority.contract.parse_canonical_json(
        label_manifest_raw,
        name="synthetic label manifest",
    )
    escaped_governance = copy.deepcopy(escaped_governance)
    escaped_governance.pop("content_sha256")
    escaped_governance["input_bindings"][
        "label_v1_terminal_predecessor_bindings"
    ]["reservation"]["file_sha256"] = "0" * 64
    escaped_governance = authority.contract.with_content_sha256(
        escaped_governance
    )
    with pytest.raises(PermissionError, match="builder provenance"):
        authority.build_label_preflight_receipt(
            label_builder_binding_raw,
            authority.canonical_document_bytes(escaped_governance),
            label_files,
            oracle_metric_pipeline=receipt["oracle_metric_pipeline"],
            wrong_rgb_mapping=_wrong_rgb_mapping_fixture(authority),
            action_prior=_action_prior_fixture(authority),
        )

    failed_oracle = copy.deepcopy(receipt)
    failed_oracle.pop("content_sha256")
    failed_oracle["oracle_metric_pipeline"]["checks"][
        "selection_utility_exact_one"
    ] = False
    failed_oracle = authority.contract.with_content_sha256(failed_oracle)
    with pytest.raises(PermissionError, match="did not pass exactly"):
        authority.validate_label_preflight_receipt(
            authority.canonical_document_bytes(failed_oracle),
            label_builder_binding_raw,
            label_manifest_raw,
            label_files,
        )

    broader_prior = _action_prior_fixture(authority)
    broader_prior["source_roles"] = ["train", "probability_calibration"]
    with pytest.raises(PermissionError, match="train-only"):
        authority.build_label_preflight_receipt(
            label_builder_binding_raw,
            label_manifest_raw,
            label_files,
            oracle_metric_pipeline={
                "status": "PASS",
                "passed": True,
                "failed_checks": [],
                "checks": {
                    name: True
                    for name in authority.ORACLE_METRIC_PREFLIGHT_CHECKS
                },
            },
            wrong_rgb_mapping=_wrong_rgb_mapping_fixture(authority),
            action_prior=broader_prior,
        )

    colliding_wrong_rgb = _wrong_rgb_mapping_fixture(authority)
    colliding_wrong_rgb["paired_next_collision_count"] = 1
    colliding_wrong_rgb["mapped_endpoint_is_never_paired_next"] = False
    with pytest.raises(PermissionError, match="wrong-RGB mapping summary"):
        authority.build_label_preflight_receipt(
            label_builder_binding_raw,
            label_manifest_raw,
            label_files,
            oracle_metric_pipeline=receipt["oracle_metric_pipeline"],
            wrong_rgb_mapping=colliding_wrong_rgb,
            action_prior=_action_prior_fixture(authority),
        )


def test_label_bundle_binds_exact_projective_support_mask() -> None:
    authority = _load("_post_action_source_authority_support_mask")
    label_manifest_raw, label_files = _label_fixture(authority)
    manifest = authority.contract.parse_canonical_json(
        label_manifest_raw, name="test label manifest"
    )
    changed = copy.deepcopy(manifest)
    changed.pop("content_sha256")
    support_name = "projective_support_mask.u1"
    support_path = authority.LABEL_MASK_RELATIVE_PATHS["projective_support_mask"]
    for record in changed["files"]:
        if record["path"] == support_name:
            record["file_sha256"] = "e" * 64
            break
    changed_files = copy.deepcopy(label_files)
    changed_files[support_path]["file_sha256"] = "e" * 64
    with pytest.raises(PermissionError, match="label mask record changed"):
        authority._label_bundle(
            authority.canonical_document_bytes(
                authority.contract.with_content_sha256(changed)
            ),
            changed_files,
        )
def test_execution_binding_binds_all_inputs_caps_and_denials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = _load("_post_action_source_authority_execution")
    source_manifest_raw, source_review_raw = _manifest_and_review(
        authority, monkeypatch, tmp_path
    )
    label_builder_binding_raw = _label_builder_binding_fixture(
        authority,
        source_manifest=authority.source_manifest_binding(source_manifest_raw),
        source_review=authority.source_review_binding(source_review_raw),
    )
    label_manifest_raw, label_files = _label_fixture(
        authority, label_builder_binding_raw
    )
    label_preflight_receipt_raw = _preflight_receipt_fixture(
        authority,
        label_builder_binding_raw,
        label_manifest_raw,
        label_files,
    )
    binding = authority.build_execution_binding(
        source_manifest_raw,
        source_review_raw,
        label_manifest_raw,
        label_files,
        label_builder_execution_binding_raw=label_builder_binding_raw,
        label_preflight_receipt_raw=label_preflight_receipt_raw,
        authorizer="/root/independent_authorizer",
        root=tmp_path,
    )
    raw = authority.canonical_document_bytes(binding)
    assert authority.validate_execution_binding(
        raw,
        source_manifest_raw,
        source_review_raw,
        label_manifest_raw,
        label_files,
        label_builder_execution_binding_raw=label_builder_binding_raw,
        label_preflight_receipt_raw=label_preflight_receipt_raw,
        root=tmp_path,
    ) == binding
    assert set(binding["label_bundle"]["files"]) == set(authority.LABEL_FILE_PATHS)
    assert set(binding["runtime_inputs"]) == {
        "raw_manifest",
        "raw_audit",
        "raw_pairs",
        "raw_endpoints",
        "n320_gate",
        "n320_encoder_checkpoint",
        "schedule",
    }
    assert set(binding["geometry_inputs"]) == set(authority.contract.GEOMETRY_BINDINGS)
    assert binding["caps"]["updates"] == 1_000
    assert binding["caps"]["presentations"] == 16_000
    assert binding["runtime"] == {
        "interpreter_path": authority.contract.RUNTIME_INTERPRETER_PATH,
        "sys_prefix": authority.contract.RUNTIME_SYS_PREFIX,
    }
    assert binding["wrong_rgb_mapping"] == (
        authority.contract.parse_canonical_json(
            label_preflight_receipt_raw, name="test receipt"
        )["wrong_rgb_mapping"]
    )
    assert binding["label_preflight_receipt"]["path"] == (
        authority.LABEL_PREFLIGHT_RECEIPT_RELATIVE_PATH
    )
    assert binding["integrity_adapter_amendment"] == (
        authority.contract.integrity_adapter_amendment_binding()
    )
    assert binding["label_v1_terminal_predecessor_bindings"] == (
        authority.contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
    )
    assert binding["attempt"] == {
        "index": 1,
        "maximum_attempts": 1,
        "fresh": True,
        "retry": False,
        "resume": False,
    }
    assert binding["downstream_denials"] == authority.contract.DOWNSTREAM_DENIALS

    altered = dict(binding)
    altered.pop("content_sha256")
    altered["output_root"] = f"{binding['output_root']}_other"
    altered = authority.contract.with_content_sha256(altered)
    with pytest.raises(PermissionError, match="execution binding changed"):
        authority.validate_execution_binding(
            authority.canonical_document_bytes(altered),
            source_manifest_raw,
            source_review_raw,
            label_manifest_raw,
            label_files,
            label_builder_execution_binding_raw=label_builder_binding_raw,
            label_preflight_receipt_raw=label_preflight_receipt_raw,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "path",
    (
        ".generated/runtime.py",
        "sealed/payload.py",
        "sealed_future/payload.py",
        "heldout/payload.py",
        "config/sealed_test.json",
        "../escape.py",
    ),
)
def test_source_path_guard_rejects_runtime_and_protected_paths(path: str) -> None:
    authority = _load(f"_post_action_source_authority_guard_{len(path)}")
    with pytest.raises(PermissionError):
        authority._safe_source_path(path)
