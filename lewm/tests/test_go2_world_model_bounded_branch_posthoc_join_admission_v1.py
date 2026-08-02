from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as consumer
from scripts import materialize_go2_world_model_bounded_branch_posthoc_join_admission_v1 as admission


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    REPO_ROOT
    / ".generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1"
)
PLAN = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_bounded_branch_integrity_replacement_v1_"
    "exact_plan_2026-08-02.json"
)
CALIBRATION = (
    REPO_ROOT
    / ".generated/dev/lewm-go2-wm-counterfactual-calibration-v3-textured-v03-"
    "posthoc-analysis-v1/calibration_receipt.json"
)


def _placeholder_binding(path: Path, token: str) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "file_sha256": token * 64,
        "byte_count": 1,
    }


def _source_bindings() -> list[dict[str, object]]:
    return [
        {"name": name, "binding": admission._file_binding(path)}
        for name, path in admission._expected_source_paths().items()
    ]


def _authority_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    source_commit = "a" * 40
    source_bindings = _source_bindings()
    review_path = tmp_path / "source_review.json"
    monkeypatch.setattr(admission, "SOURCE_REVIEW", review_path)
    review = {
        "schema": admission.SOURCE_REVIEW_SCHEMA,
        "status": admission.SOURCE_REVIEW_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "citable_as_scientific_evidence": False,
        "reviewer": {
            "identity": "independent-test-reviewer",
            "independence_basis": "did not author the fixture",
        },
        "reviewed_at": "2026-08-02T00:00:00Z",
        "source_commit": source_commit,
        "preregistration_binding": admission._file_binding(
            admission.PREREGISTRATION
        ),
        "failure_admissibility_audit_binding": admission._file_binding(
            admission.FAILURE_AUDIT
        ),
        "source_bindings": source_bindings,
        "checks": {name: True for name in admission.SOURCE_REVIEW_CHECKS},
        "findings": [],
        "protected_material_opened": False,
    }
    admission._write_json_exclusive(review_path, review)
    input_names = (
        "consumed_terminal",
        "physics_result",
        "physics_receipt_check",
        "collection_plan",
        "calibration_gate",
        "collection_source_review",
        "collection_execution_authority",
        "calibration_receipt",
    )
    return {
        "schema": admission.AUTHORITY_SCHEMA,
        "status": admission.AUTHORITY_STATUS,
        "authority_granted_by_this_document": True,
        "scientific_claim_authorized": False,
        "issued_at": "2026-08-02T00:00:01Z",
        "authorizer": "test-authorizer",
        "preregistration_binding": admission._file_binding(
            admission.PREREGISTRATION
        ),
        "failure_audit_binding": admission._file_binding(admission.FAILURE_AUDIT),
        "source_commit": source_commit,
        "source_review_binding": admission._file_binding(review_path),
        "source_bindings": source_bindings,
        "input_bindings": {
            name: _placeholder_binding(tmp_path / f"{name}.json", str(index + 1))
            for index, name in enumerate(input_names)
        },
        "attempt": {
            "id": "lewm-go2-wm-bounded-branch-posthoc-join-admission-v1",
            "root": str(admission.DEFAULT_OUTPUT_ROOT.resolve()),
            "maximum_attempts": 1,
            "root_creation_consumes_attempt": True,
            "must_be_absent": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
        },
        "permissions": {
            "source_receipt_reads": True,
            "decoded_pixel_verification": True,
            "metadata_only_derivation": True,
            "write_only_fresh_output_root": True,
            "collector_or_renderer": False,
            "physics_or_gpu": False,
            "training_or_checkpoint_access": False,
            "retry_resume_refill_or_overwrite": False,
            "protected_material": False,
            "scientific_verdict": False,
        },
        "expected_outputs": {
            name: {
                "name": value["name"],
                "byte_count": value["byte_count"],
                "sha256": value["sha256"],
            }
            for name, value in admission.EXPECTED_LEAVES.items()
        },
    }


def _bindings() -> dict[str, dict[str, object]]:
    return {
        "consumed_terminal": admission._file_binding(
            SOURCE_ROOT / "terminal_supervision.json"
        ),
        "physics_result": admission._file_binding(
            SOURCE_ROOT / "physics_result.json"
        ),
        "physics_receipt_check": admission._file_binding(
            SOURCE_ROOT / "physics_receipt_check.json"
        ),
        "collection_plan": admission._file_binding(PLAN),
        "calibration_receipt": admission._file_binding(CALIBRATION),
    }


@pytest.fixture(scope="module")
def real_derivation():
    if not SOURCE_ROOT.is_dir():
        pytest.skip("bounded posthoc source corpus is unavailable")
    bindings = _bindings()
    before = admission._source_inventory(SOURCE_ROOT)
    derived = admission.derive_documents_v1(
        terminal_binding=bindings["consumed_terminal"],
        physics_binding=bindings["physics_result"],
        physics_check_binding=bindings["physics_receipt_check"],
        plan_binding=bindings["collection_plan"],
        calibration_receipt_binding=bindings["calibration_receipt"],
        verify_textured_pixels=False,
    )
    assert admission._source_inventory(SOURCE_ROOT) == before
    return bindings, before, derived


def test_real_derivation_matches_registered_metadata(real_derivation) -> None:
    _bindings_value, _inventory, derived = real_derivation
    assert len(derived.rgb_manifest["artifacts"]) == 3_072
    assert len(derived.rows["train"]) == 128
    assert len(derived.rows["eval"]) == 128
    for name, expected in admission.EXPECTED_LEAVES.items():
        raw = derived.raw_by_leaf[name]
        assert len(raw) == expected["byte_count"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]


def test_binding_or_root_drift_fails_closed(real_derivation) -> None:
    bindings, _inventory, _derived = real_derivation
    changed = dict(bindings["physics_result"])
    changed["file_sha256"] = "0" * 64
    with pytest.raises(Exception):
        admission.derive_documents_v1(
            terminal_binding=bindings["consumed_terminal"],
            physics_binding=changed,
            physics_check_binding=bindings["physics_receipt_check"],
            plan_binding=bindings["collection_plan"],
            calibration_receipt_binding=bindings["calibration_receipt"],
            verify_textured_pixels=False,
        )


def test_authority_requires_exact_audit_review_commit_and_named_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "_validate_committed_sources_v1", lambda **_: None)
    authority = _authority_document(tmp_path, monkeypatch)
    normalized = admission.validate_authority_v1(authority)
    assert normalized["failure_audit_binding"] == admission._file_binding(
        admission.FAILURE_AUDIT
    )
    assert [row["name"] for row in normalized["source_bindings"]] == list(
        admission._expected_source_paths()
    )

    wrong_audit = copy.deepcopy(authority)
    wrong_audit["failure_audit_binding"]["file_sha256"] = "0" * 64
    with pytest.raises(admission.PosthocJoinAdmissionError):
        admission.validate_authority_v1(wrong_audit)

    wrong_source = copy.deepcopy(authority)
    wrong_source["source_bindings"][0]["binding"]["path"] = str(
        tmp_path / "different.py"
    )
    with pytest.raises(admission.PosthocJoinAdmissionError):
        admission.validate_authority_v1(wrong_source)

    wrong_review_commit = copy.deepcopy(authority)
    wrong_review_commit["source_commit"] = "b" * 40
    with pytest.raises(admission.PosthocJoinAdmissionError):
        admission.validate_authority_v1(wrong_review_commit)


def _patch_materialization_preamble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], dict[str, object]]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_root = tmp_path / "posthoc"
    monkeypatch.setattr(admission, "DEFAULT_OUTPUT_ROOT", output_root)
    inputs = {
        name: _placeholder_binding(tmp_path / f"{name}.json", str(index + 1))
        for index, name in enumerate(
            (
                "consumed_terminal",
                "physics_result",
                "physics_receipt_check",
                "collection_plan",
                "calibration_gate",
                "collection_source_review",
                "collection_execution_authority",
                "calibration_receipt",
            )
        )
    }
    inputs["physics_result"]["path"] = str(source_root / "physics_result.json")
    authority = {
        "attempt": {"root": str(output_root.resolve())},
        "input_bindings": inputs,
        "preregistration_binding": _placeholder_binding(
            tmp_path / "prereg.json", "a"
        ),
        "failure_audit_binding": _placeholder_binding(
            tmp_path / "failure.json", "b"
        ),
        "source_review_binding": _placeholder_binding(
            tmp_path / "source-review.json", "c"
        ),
        "source_bindings": [],
    }
    authority_binding = _placeholder_binding(tmp_path / "authority.json", "d")
    monkeypatch.setattr(
        admission.pilot,
        "read_bound_json",
        lambda *args, **kwargs: (authority, authority_binding),
    )
    monkeypatch.setattr(
        admission, "validate_authority_v1", lambda document: dict(document)
    )
    monkeypatch.setattr(admission, "_rehash_authority_inputs", lambda *_: None)
    monkeypatch.setattr(
        admission, "_validate_collection_lineage_inputs", lambda *_: None
    )
    return output_root, authority, authority_binding


def test_materialize_writes_exact_metadata_and_success_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, _authority, _authority_binding = _patch_materialization_preamble(
        tmp_path, monkeypatch
    )
    inventory = {"file_count": 3, "byte_count": 12, "inventory_sha256": "a" * 64}
    monkeypatch.setattr(admission, "_source_inventory", lambda *_: inventory)
    raw_by_leaf = {"rgb_manifest": b"{}", "train": b"train\n", "eval": b"eval\n"}
    expected_leaves = {
        name: {
            "name": f"{name}.jsonl" if name != "rgb_manifest" else "rgb_manifest.json",
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, raw in raw_by_leaf.items()
    }
    monkeypatch.setattr(admission, "EXPECTED_LEAVES", expected_leaves)
    derived = admission.DerivedDocumentsV1(
        collection={},
        calibration_receipt={},
        rgb_manifest={},
        rows={"train": (), "eval": ()},
        metadata={
            "render_profile": "test-profile",
            "visual_domain_parity_result_binding": {},
            "visual_domain_parity_terminal_binding": {},
            "visual_domain_parity_review_binding": {},
            "calibration_contract": {},
            "scene_ids": {"train": [], "eval": []},
            "action_catalog": [],
        },
        raw_by_leaf=raw_by_leaf,
    )
    monkeypatch.setattr(admission, "derive_documents_v1", lambda **_: derived)
    result = admission.materialize_v1(
        authority_path=tmp_path / "authority.json",
        expected_authority_sha256="d" * 64,
        expected_authority_byte_count=1,
    )
    assert Path(result["output_root"]) == output_root
    assert {path.name for path in output_root.iterdir()} == {
        "rgb_manifest.json",
        "train.jsonl",
        "eval.jsonl",
        "manifest.json",
        "terminal.json",
    }
    terminal = json.loads((output_root / "terminal.json").read_text())
    assert terminal["status"] == admission.TERMINAL_SUCCESS
    assert terminal["failure"] is None
    assert terminal["source_inventory_before"] == inventory
    assert terminal["source_inventory_after"] == inventory


def test_materialize_terminalizes_keyboard_interrupt_even_if_reinventory_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, _authority, _authority_binding = _patch_materialization_preamble(
        tmp_path, monkeypatch
    )
    inventory = {"file_count": 3, "byte_count": 12, "inventory_sha256": "a" * 64}
    calls = 0

    def inventory_once(_root):
        nonlocal calls
        calls += 1
        if calls == 1:
            return inventory
        raise RuntimeError("inventory unavailable")

    def interrupt(**_kwargs):
        raise KeyboardInterrupt("stop")

    monkeypatch.setattr(admission, "_source_inventory", inventory_once)
    monkeypatch.setattr(admission, "derive_documents_v1", interrupt)
    with pytest.raises(KeyboardInterrupt, match="stop"):
        admission.materialize_v1(
            authority_path=tmp_path / "authority.json",
            expected_authority_sha256="d" * 64,
            expected_authority_byte_count=1,
        )
    terminal = json.loads((output_root / "terminal.json").read_text())
    assert terminal["status"] == admission.TERMINAL_FAILURE
    assert terminal["source_inventory_after"] is None
    assert terminal["terminalization_inventory_failure"] == (
        "RuntimeError: inventory unavailable"
    )
    assert terminal["failure"] == "KeyboardInterrupt: stop"


def test_split_root_loader_reconstructs_normal_bundle(
    real_derivation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings, inventory, derived = real_derivation
    derived_root = tmp_path / "posthoc"
    derived_root.mkdir()
    monkeypatch.setattr(admission, "DEFAULT_OUTPUT_ROOT", derived_root)
    monkeypatch.setattr(admission, "derive_documents_v1", lambda **_: derived)
    monkeypatch.setattr(
        admission, "validate_authority_v1", lambda document: dict(document)
    )
    leaves = {}
    for name, expected in admission.EXPECTED_LEAVES.items():
        selected = derived_root / str(expected["name"])
        admission._write_exclusive(selected, derived.raw_by_leaf[name])
        leaves[name] = admission._file_binding(selected)
    input_bindings = {
        **bindings,
        "calibration_gate": _placeholder_binding(
            tmp_path / "calibration_gate.json", "a"
        ),
        "collection_source_review": _placeholder_binding(
            tmp_path / "collection_source_review.json", "b"
        ),
        "collection_execution_authority": _placeholder_binding(
            tmp_path / "collection_execution_authority.json", "c"
        ),
    }
    preregistration_binding = _placeholder_binding(tmp_path / "prereg.json", "d")
    failure_audit_binding = _placeholder_binding(tmp_path / "failure.json", "e")
    source_review_binding = _placeholder_binding(tmp_path / "source.json", "f")
    source_bindings: list[dict[str, object]] = []
    authority = {
        "preregistration_binding": preregistration_binding,
        "failure_audit_binding": failure_audit_binding,
        "source_review_binding": source_review_binding,
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
    }
    authority_path = tmp_path / "authority.json"
    admission._write_json_exclusive(authority_path, authority)
    authority_binding = admission._file_binding(authority_path)
    manifest = {
        "schema": admission.MANIFEST_SCHEMA,
        "status": admission.MANIFEST_STATUS,
        "citable_as_scientific_evidence": False,
        "original_attempt_completed_successfully": False,
        "authorizes_retry_or_resume": False,
        "source_receipt_root": str(SOURCE_ROOT.resolve()),
        "derived_output_root": str(derived_root.resolve()),
        "authority_binding": authority_binding,
        "preregistration_binding": preregistration_binding,
        "failure_audit_binding": failure_audit_binding,
        "source_review_binding": source_review_binding,
        "input_bindings": input_bindings,
        "source_bindings": source_bindings,
        "source_inventory_before": inventory,
        "source_inventory_after": inventory,
        "counts": admission.EXPECTED_COUNTS,
        "rgb_artifacts": 3_072,
        "role_scene_counts": {"train": 16, "eval": 16},
        "render_profile": derived.metadata["render_profile"],
        "visual_domain_parity_result_binding": derived.metadata[
            "visual_domain_parity_result_binding"
        ],
        "visual_domain_parity_terminal_binding": derived.metadata[
            "visual_domain_parity_terminal_binding"
        ],
        "visual_domain_parity_review_binding": derived.metadata[
            "visual_domain_parity_review_binding"
        ],
        "calibration_contract": derived.metadata["calibration_contract"],
        "scene_ids": derived.metadata["scene_ids"],
        "action_catalog": derived.metadata["action_catalog"],
        "derived_leaf_bindings": leaves,
        "derivation": "frozen_pixel_verifier_plus_pure_build_joined_documents_v1",
        "rgb_storage": "immutable_source_receipt_root_only",
        "consumer_compatibility_projection": (
            admission.CONSUMER_COMPATIBILITY_PROJECTION
        ),
    }
    manifest_path = derived_root / "manifest.json"
    admission._write_json_exclusive(manifest_path, manifest)
    manifest_binding = admission._file_binding(manifest_path)
    terminal = {
        "schema": admission.TERMINAL_SCHEMA,
        "status": admission.TERMINAL_SUCCESS,
        "citable_as_scientific_evidence": False,
        "scientific_claim_emitted": False,
        "authorizes_retry_or_resume": False,
        "original_terminal_remains_failure": True,
        "authority_binding": authority_binding,
        "manifest_binding": manifest_binding,
        "source_inventory_before": inventory,
        "source_inventory_after": inventory,
        "terminalization_inventory_failure": None,
        "generation_or_rendering_performed": False,
        "independent_review_required": True,
        "failure": None,
    }
    terminal_path = derived_root / "terminal.json"
    admission._write_json_exclusive(terminal_path, terminal)
    terminal_binding = admission._file_binding(terminal_path)
    terminal_review = {
        "schema": admission.TERMINAL_REVIEW_SCHEMA,
        "status": admission.TERMINAL_REVIEW_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "citable_as_scientific_evidence": False,
        "reviewer": {
            "identity": "independent-test-reviewer",
            "independence_basis": "did not author the fixture",
        },
        "reviewed_at": "2026-08-02T00:00:00Z",
        "terminal_binding": terminal_binding,
        "manifest_binding": manifest_binding,
        "authority_binding": authority_binding,
        "source_review_binding": source_review_binding,
        "preregistration_binding": preregistration_binding,
        "failure_admissibility_audit_binding": failure_audit_binding,
        "checks": {name: True for name in admission.TERMINAL_REVIEW_CHECKS},
        "findings": [],
        "protected_material_opened": False,
    }
    review_path = tmp_path / "terminal_review.json"
    admission._write_json_exclusive(review_path, terminal_review)
    review_binding = admission._file_binding(review_path)
    bundle = admission.load_posthoc_bundle_v1(
        derived_root,
        expected_manifest_byte_count=int(manifest_binding["byte_count"]),
        expected_manifest_sha256=str(manifest_binding["file_sha256"]),
        expected_terminal_byte_count=int(terminal_binding["byte_count"]),
        expected_terminal_sha256=str(terminal_binding["file_sha256"]),
        terminal_review_path=review_path,
        expected_terminal_review_byte_count=int(review_binding["byte_count"]),
        expected_terminal_review_sha256=str(review_binding["file_sha256"]),
    )
    assert bundle.root == SOURCE_ROOT.resolve()
    assert len(bundle.groups_by_role["train"]) == 128
    assert len(bundle.groups_by_role["eval"]) == 128
    assert len(bundle.artifacts) == 3_072
    assert bundle.access_audit["rgb_leaf_open_count"] == 0
    first_artifact = sorted(bundle.artifacts)[0]
    assert consumer.read_bound_rgb_bytes_v1(bundle, first_artifact)
    assert admission._source_inventory(SOURCE_ROOT) == inventory


def test_split_root_loader_rejects_incomplete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived_root = tmp_path / "posthoc"
    derived_root.mkdir()
    monkeypatch.setattr(admission, "DEFAULT_OUTPUT_ROOT", derived_root)
    admission._write_json_exclusive(
        derived_root / "manifest.json",
        {
            "schema": admission.MANIFEST_SCHEMA,
            "status": admission.MANIFEST_STATUS,
        },
    )
    manifest_binding = admission._file_binding(derived_root / "manifest.json")
    with pytest.raises(
        admission.PosthocJoinAdmissionError, match="posthoc manifest changed"
    ):
        admission.load_posthoc_bundle_v1(
            derived_root,
            expected_manifest_byte_count=int(manifest_binding["byte_count"]),
            expected_manifest_sha256=str(manifest_binding["file_sha256"]),
            expected_terminal_byte_count=1,
            expected_terminal_sha256="0" * 64,
            terminal_review_path=tmp_path / "missing-review.json",
            expected_terminal_review_byte_count=1,
            expected_terminal_review_sha256="0" * 64,
        )


def test_split_source_root_must_equal_physics_parent_and_frozen_plan(
    tmp_path: Path,
) -> None:
    if not SOURCE_ROOT.is_dir():
        pytest.skip("bounded posthoc source corpus is unavailable")
    bindings = _bindings()
    inputs = {
        "physics_result": bindings["physics_result"],
        "collection_plan": bindings["collection_plan"],
    }
    assert admission._validate_split_source_root_v1(
        manifest={"source_receipt_root": str(SOURCE_ROOT.resolve())},
        inputs=inputs,
    ) == SOURCE_ROOT.resolve()
    wrong_root = tmp_path / "wrong-source"
    wrong_root.mkdir()
    with pytest.raises(
        admission.PosthocJoinAdmissionError,
        match="not the bound physics and plan root",
    ):
        admission._validate_split_source_root_v1(
            manifest={"source_receipt_root": str(wrong_root.resolve())},
            inputs=inputs,
        )


def test_source_has_no_generation_entrypoint_or_legacy_join_call() -> None:
    source_path = Path(admission.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    assert not any(
        token in name
        for name in imported
        for token in ("collect", "render_replay", "genesis")
    )
    assert "join_pilot" not in calls


def test_consumed_terminal_is_not_relabelled_success(real_derivation) -> None:
    bindings, _inventory, _derived = real_derivation
    terminal = json.loads(Path(str(bindings["consumed_terminal"]["path"])).read_text())
    assert terminal["status"] == "CONSUMED_TERMINAL_FAILURE"
    assert terminal["joined_manifest_binding"] is None
    assert terminal["joined_receipt_check_binding"] is None
    assert admission.TERMINAL_SUCCESS != terminal["status"]
