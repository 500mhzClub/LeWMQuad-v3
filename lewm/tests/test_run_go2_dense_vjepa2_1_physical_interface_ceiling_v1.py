from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch


_EINOPS_WAS_LOADED = "einops" in sys.modules
from scripts import (  # noqa: E402
    run_go2_dense_vjepa2_1_physical_interface_ceiling_v1 as runner,
)
_EINOPS_LOADED_BY_RUNNER_IMPORT = not _EINOPS_WAS_LOADED and "einops" in sys.modules


def _binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def test_integrity_replacement_namespace_root_and_preregistration_binding() -> None:
    namespace = "integrity_replacement_v1"
    assert all(
        namespace in value.lower()
        for value in (
            runner.SCHEMA,
            runner.TERMINAL_SCHEMA,
            runner.RESERVATION_SCHEMA,
            runner.EVAL_CACHE_SCHEMA,
            runner.EVAL_CACHE_RECEIPT_SCHEMA,
            runner.REPLAY_SCHEMA,
            runner.REPLAY_STATUS,
            runner.AUTHORITY_SCHEMA,
            runner.AUTHORITY_STATUS,
            runner.SOURCE_REVIEW_SCHEMA,
            runner.SOURCE_REVIEW_STATUS,
        )
    )
    assert runner.PREREGISTRATION == runner.REPO_ROOT / (
        "docs/lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_"
        "integrity_replacement_v1_preregistration_2026-08-03.md"
    )
    assert runner.PREREGISTRATION_SHA256 == (
        "724c52e59696e22efcfdb9e3427cd5a622a536400359389476bff1c8d1fe3ce6"
    )
    assert runner.PREREGISTRATION_BYTE_COUNT == 11_704
    assert runner.SOURCE_REVIEW == runner.REPO_ROOT / (
        "docs/lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1_"
        "integrity_replacement_v1_source_review_2026-08-03.json"
    )
    assert runner.DEFAULT_OUTPUT_ROOT == runner.REPO_ROOT / (
        ".generated/dev/go2_dense_vjepa2_1_physical_interface_ceiling_v1/"
        "attempt_v2_integrity_replacement_v1"
    )


def test_frozen_inventories_classification_permissions_and_dependencies() -> None:
    inputs = runner._fixed_input_bindings_v1()  # noqa: SLF001
    assert len(inputs) == 30
    assert len(runner.SCIENTIFIC_INPUT_LABELS) == 20
    assert len(runner.LINEAGE_WITNESS_LABELS) == 10
    assert runner.SCIENTIFIC_INPUT_LABELS | runner.LINEAGE_WITNESS_LABELS == set(inputs)
    assert not (runner.SCIENTIFIC_INPUT_LABELS & runner.LINEAGE_WITNESS_LABELS)
    assert runner.config_v1()["scientific_input_file_count"] == 20
    assert runner.config_v1()["lineage_witness_file_count"] == 10
    assert runner.config_v1()["total_fixed_input_file_count"] == 30
    assert runner.OUTPUT_NAMES == (
        "reservation.json", "vjepa2_1_eval.pt", "vjepa2_1_eval.json",
        "ceiling_checkpoint.pt", "evaluation.json", "replay.json",
        "result.json", "terminal.json",
    )
    permissions = runner.permissions_v1()
    assert set(permissions) == runner.PERMISSION_FIELDS
    assert permissions["primary_eval_rgb_access"] is True
    assert permissions["primary_vjepa2_1_encoder_execution"] is True
    assert permissions["replay_rgb_access"] is False
    assert permissions["replay_encoder_execution"] is False
    assert permissions["train_rgb_access"] is False
    assert set(runner.VJEPA_TRANSITIVE_SOURCE_BINDINGS) == {
        "hubconf.py", "evals/hub/preprocessor.py", "src/hub/backbones.py",
        "app/vjepa_2_1/models/vision_transformer.py",
        "app/vjepa_2_1/models/predictor.py",
        "app/vjepa_2_1/models/utils/modules.py",
        "app/vjepa_2_1/models/utils/patch_embed.py",
        "src/masks/utils.py", "src/utils/tensors.py",
    }
    dependency = runner.einops_dependency_v1()
    assert dependency["version"] == "0.8.1"
    assert len(dependency["runtime_source_bindings"]) == 6
    assert len(dependency["distribution_metadata_bindings"]) == 4
    assert len(runner.SOURCE_PATHS) == 12
    assert _EINOPS_LOADED_BY_RUNNER_IMPORT is False


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/sealed_test.json", "/tmp/sealed/member", "/tmp/sealed_future/member",
        "/tmp/heldout/member", "/tmp/held_out/member", "/tmp/protected/member",
        "/tmp/protected_future/member",
    ],
)
def test_protected_names_rejected_without_open(path: str) -> None:
    with pytest.raises(runner.DenseVJEPACeilingRunnerError, match="protected material"):
        runner._reject_protected(Path(path), label="synthetic")  # noqa: SLF001


def test_bound_json_and_jsonl_are_each_single_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_path = tmp_path / "document.json"
    document_path.write_text('{"a":1}\n')
    document_binding = _binding(document_path)
    original_open = runner.os.open
    opened: list[Path] = []

    def tracked_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path) == document_path:
            opened.append(Path(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", tracked_open)
    document, observed = runner._read_bound_json(  # noqa: SLF001
        document_path,
        expected_sha256=document_binding["sha256"],
        expected_byte_count=document_binding["byte_count"],
        label="synthetic document",
    )
    assert document == {"a": 1}
    assert observed == document_binding
    assert opened == [document_path]

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('{"state_id":"a"}\n{"state_id":"b"}\n')
    rows_binding = _binding(rows_path)
    opened.clear()

    def tracked_rows_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path) == rows_path:
            opened.append(Path(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(runner.os, "open", tracked_rows_open)
    monkeypatch.setattr(runner, "ROLE_STATE_COUNT", 2)
    authority = {"input_bindings": {"posthoc_train_rows": rows_binding}}
    assert runner._read_jsonl_v1(authority, role="train") == [  # noqa: SLF001
        {"state_id": "a"}, {"state_id": "b"}
    ]
    assert opened == [rows_path]


def test_physics_state_receipt_helper_uses_contract_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physics_root = tmp_path / "physics"
    physics_root.mkdir()
    receipt_binding = {
        "path": "receipts/state-000.json",
        "sha256": "a" * 64,
        "byte_count": 1,
    }
    physics_result = tmp_path / "physics-result.json"
    physics_result.write_text(
        json.dumps(
            {
                "schema": runner.contract.PHYSICS_RESULT_SCHEMA,
                "status": "PHYSICS_COMPLETE",
                "failure": None,
                "state_receipt_bindings": [receipt_binding],
            }
        )
    )
    monkeypatch.setattr(runner, "STATE_RECEIPT_COUNT", 1)
    monkeypatch.setattr(runner, "PHYSICS_ROOT", physics_root)
    authority = {"input_bindings": {"physics_result": _binding(physics_result)}}

    assert runner._state_receipt_bindings_from_physics_v1(authority) == [  # noqa: SLF001
        {
            "path": str((physics_root / receipt_binding["path"]).resolve()),
            "sha256": receipt_binding["sha256"],
            "byte_count": receipt_binding["byte_count"],
        }
    ]


def test_rgb_manifest_loader_is_metadata_only_without_leaf_stat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "ROLE_ARTIFACT_COUNT", 1)
    artifacts = []
    for index in range(2):
        artifacts.append(
            {
                "artifact_id": f"rgb-{index}",
                "frame_identity": f"frame-{index}",
                "path": f"scene/rgb-{index}.png",
                "file_sha256": f"{index + 1:064x}",
                "pixel_sha256": f"{index + 3:064x}",
                "byte_count": 100 + index,
                "width": 224,
                "height": 224,
                "mode": "RGB",
                "format": "PNG",
                "camera_valid": True,
                "low_information": False,
                "low_info_reasons": [],
            }
        )
    monkeypatch.setattr(
        runner,
        "_bound_document_v1",
        lambda _authority, label: {
            "schema": runner.consumer.RGB_MANIFEST_SCHEMA,
            "artifacts": artifacts,
        }
        if label == "posthoc_rgb_manifest"
        else pytest.fail("unexpected document"),
    )
    monkeypatch.setattr(
        runner,
        "_safe_path",
        lambda *_args, **_kwargs: pytest.fail("RGB metadata must not touch a leaf"),
    )
    observed = runner._artifact_metadata_v1({}, Path("/synthetic/source"))  # noqa: SLF001
    assert set(observed) == {"rgb-0", "rgb-1"}
    assert observed["rgb-0"].relative_path == "scene/rgb-0.png"


def test_local_preprocess_normalizer_and_timm_shim_restore() -> None:
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    from io import BytesIO

    stream = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    prepared = runner.preprocess_vjepa2_1_png_bytes_v1(stream.getvalue())
    assert prepared.shape == (3, 1, 384, 384)
    assert prepared.dtype == torch.float32
    raw = torch.zeros(1, 576, 768)
    raw[:, :, 0] = 1.0
    normalized = runner.normalize_vjepa_token_grid_v1(raw)
    assert normalized.shape == (1, 256, 768)
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=-1), torch.ones(1, 256))
    prior = {name: sys.modules.get(name) for name in ("timm", "timm.models", "timm.models.layers")}
    with runner.scoped_timm_drop_path_shim_v1():
        from timm.models.layers import drop_path

        value = torch.ones(2, 3)
        assert torch.equal(drop_path(value, 0.0, False), value)
    for name, value in prior.items():
        if value is None:
            assert name not in sys.modules
        else:
            assert sys.modules[name] is value


class _FakeVJEPA(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(inputs.shape[0], 576, 768, dtype=torch.float32)
        result[:, :, 0] = 1.0
        return result


def test_eval_extraction_opens_exact_order_once_and_never_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROLE_ARTIFACT_COUNT", 2)
    monkeypatch.setattr(runner, "EVAL_CONTEXT_COUNT", 1)
    monkeypatch.setattr(runner, "EVAL_SUCCESSOR_COUNT", 1)
    monkeypatch.setattr(runner, "EVAL_BATCH_SIZE", 1)
    monkeypatch.setattr(runner.evaluator, "EXPECTED_EVAL_PLAN_IDENTITY", "eval-plan")
    root = Path("/synthetic/source")
    artifact_ids = ("eval-current", "eval-successor")
    artifacts = {
        artifact_id: runner.consumer.RGBArtifactV1(
            artifact_id=artifact_id,
            frame_identity=f"frame-{index}",
            relative_path=f"eval/{artifact_id}.png",
            byte_count=10,
            file_sha256=f"{index + 1:064x}",
            pixel_sha256=f"{index + 3:064x}",
            low_information=False,
            low_info_reasons=(),
        )
        for index, artifact_id in enumerate(artifact_ids)
    }
    state = SimpleNamespace(
        context_artifact_indices=(0,), target_artifact_indices=(1,)
    )
    plan = SimpleNamespace(
        artifact_ids=artifact_ids, states=(state,), identity_sha256="eval-plan"
    )
    bundle = SimpleNamespace(
        root=root, artifacts=artifacts, manifest_binding={"synthetic": True}
    )
    declared = [
        {
            "artifact_id": artifact_id,
            "path": str(root / artifacts[artifact_id].relative_path),
            "sha256": artifacts[artifact_id].file_sha256,
            "pixel_sha256": artifacts[artifact_id].pixel_sha256,
            "byte_count": artifacts[artifact_id].byte_count,
        }
        for artifact_id in artifact_ids
    ]
    authority = {
        "eval_rgb_bindings": declared,
        "encoder_source": {"synthetic": True},
    }
    opened: list[str] = []
    monkeypatch.setattr(
        runner,
        "read_bound_rgb_bytes_v1",
        lambda _bundle, artifact_id: opened.append(artifact_id) or b"synthetic",
    )
    monkeypatch.setattr(
        runner,
        "preprocess_vjepa2_1_png_bytes_v1",
        lambda _raw: torch.zeros(3, 1, 2, 2),
    )
    monkeypatch.setattr(
        runner, "_load_vjepa_encoder_v1", lambda *_args, **_kwargs: _FakeVJEPA()
    )
    receipt = runner.extract_eval_feature_cache_v1(
        authority,
        bundle,
        plan,
        device=torch.device("cpu"),
        output_path=tmp_path / "vjepa2_1_eval.pt",
    )
    assert opened == list(artifact_ids)
    assert receipt["eval_artifact_open_count"] == 2
    assert receipt["eval_context_open_count"] == 1
    assert receipt["eval_successor_open_count"] == 1
    assert receipt["train_artifact_open_count"] == 0
    assert receipt["encoded_frame_count"] == 2


def test_eval_cache_rejects_decoy_path_and_provenance_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROLE_ARTIFACT_COUNT", 1)
    monkeypatch.setattr(runner, "EVAL_CONTEXT_COUNT", 0)
    monkeypatch.setattr(runner, "EVAL_SUCCESSOR_COUNT", 1)
    monkeypatch.setattr(runner.evaluator, "EXPECTED_EVAL_PLAN_IDENTITY", "eval-plan")
    plan = SimpleNamespace(artifact_ids=("eval-0",), identity_sha256="eval-plan")
    authority = {"encoder_source": {"bound": True}, "eval_rgb_bindings": []}
    bundle = SimpleNamespace(manifest_binding={"manifest": True})
    decoy = tmp_path / "decoy.pt"
    decoy.write_bytes(b"not loaded")
    receipt = {
        "schema": runner.EVAL_CACHE_RECEIPT_SCHEMA,
        "encoder": "vjepa2_1",
        "role": "eval",
        "eval_plan_identity": "eval-plan",
        "artifact_order_sha256": hashlib.sha256(
            runner.canonical_bytes_v1(["eval-0"])
        ).hexdigest(),
        "artifact_count": 1,
        "eval_artifact_open_count": 1,
        "eval_context_open_count": 0,
        "eval_successor_open_count": 1,
        "train_artifact_open_count": 0,
        "decoded_pixel_verification_count": 1,
        "encoded_frame_count": 1,
        "shape": [1, 256, 768],
        "storage_dtype": "float16",
        "preprocessing": runner.feature_preprocessing_contract_v1(),
        "source_bundle_manifest": {"manifest": True},
        "encoder_source": {"bound": True},
        "authority_eval_rgb_binding_order_sha256": hashlib.sha256(
            runner.canonical_bytes_v1([])
        ).hexdigest(),
        "binding": _binding(decoy),
    }
    monkeypatch.setattr(
        runner,
        "_load_bound_torch_once_v1",
        lambda *_args, **_kwargs: pytest.fail("decoy cache must not be opened"),
    )
    with pytest.raises(runner.DenseVJEPACeilingRunnerError, match="cache path changed"):
        runner._load_eval_cache_v1(  # noqa: SLF001
            receipt, plan, authority=authority, bundle=bundle
        )
    receipt["binding"] = {
        **receipt["binding"],
        "path": str((runner.DEFAULT_OUTPUT_ROOT / "vjepa2_1_eval.pt").resolve()),
    }
    receipt["encoder_source"] = {"wrong": True}
    with pytest.raises(runner.DenseVJEPACeilingRunnerError, match="receipt changed"):
        runner._load_eval_cache_v1(  # noqa: SLF001
            receipt, plan, authority=authority, bundle=bundle
        )


def test_replay_recomputes_before_reference_loads_and_has_no_rgb_or_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "attempt"
    output_root.mkdir()
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", output_root)
    authority_binding_path = tmp_path / "authority.json"
    authority_binding_path.write_text("{}\n")
    authority_binding = _binding(authority_binding_path)
    authority = {"output_root": str(output_root), "source_bindings": {"ceiling_evaluator": {}}}
    reservation = {
        "schema": runner.RESERVATION_SCHEMA,
        "authority_binding": authority_binding,
        "attempt_root": str(output_root),
        "owner_pid": 1,
        "consumes_attempt": True,
        "reserved_before_cache_deserialization_or_rgb_decode": True,
    }
    runner._write_json_exclusive(output_root / "reservation.json", reservation)  # noqa: SLF001
    (output_root / "vjepa2_1_eval.pt").write_bytes(b"synthetic")
    runner._write_json_exclusive(output_root / "vjepa2_1_eval.json", {})  # noqa: SLF001
    eval_receipt_binding = runner.file_binding_v1(output_root / "vjepa2_1_eval.json")
    runner._save_torch_exclusive(output_root / "ceiling_checkpoint.pt", {"primary": torch.tensor(1)})  # noqa: SLF001
    checkpoint_binding = runner.file_binding_v1(output_root / "ceiling_checkpoint.pt")
    primary_evaluation = {"primary": True}
    runner._write_json_exclusive(output_root / "evaluation.json", primary_evaluation)  # noqa: SLF001
    evaluation_binding = runner.file_binding_v1(output_root / "evaluation.json")
    events: list[str] = []
    bundle = SimpleNamespace()
    plan = SimpleNamespace()
    monkeypatch.setattr(runner, "_load_narrow_bundle_v1", lambda _authority: (bundle, {}))
    monkeypatch.setattr(
        runner,
        "_feature_plans_v1",
        lambda _bundle: ((object(),), (object(),), plan, plan),
    )
    monkeypatch.setattr(
        runner,
        "_load_train_cache_v1",
        lambda *_args, **_kwargs: (events.append("train_cache") or torch.zeros(1), {}),
    )
    monkeypatch.setattr(
        runner,
        "_load_eval_cache_v1",
        lambda *_args, **_kwargs: events.append("eval_cache") or torch.zeros(1),
    )
    monkeypatch.setattr(runner, "_authorized_device_v1", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        runner.evaluator,
        "fit_primary_checkpoint_v1",
        lambda *_args, **_kwargs: events.append("recompute_checkpoint") or {"replay": torch.tensor(1)},
    )
    replay_evaluation = {"replay": True}
    monkeypatch.setattr(
        runner,
        "_evaluate_v1",
        lambda *_args, **_kwargs: events.append("recompute_evaluation") or replay_evaluation,
    )
    original_torch_loader = runner._load_bound_torch_once_v1  # noqa: SLF001
    original_json_loader = runner._read_bound_json  # noqa: SLF001

    def reference_loader(*args: Any, **kwargs: Any):
        events.append("load_primary_checkpoint")
        return original_torch_loader(*args, **kwargs)

    monkeypatch.setattr(runner, "_load_bound_torch_once_v1", reference_loader)

    def json_reference_loader(*args: Any, **kwargs: Any):
        if kwargs.get("label") == "primary evaluation":
            events.append("load_primary_evaluation")
        return original_json_loader(*args, **kwargs)

    monkeypatch.setattr(runner, "_read_bound_json", json_reference_loader)
    reproduction = {name: True for name in runner.REPRODUCTION_FIELDS}
    verdict = {"terminal_status": runner.STOP_STATUS}

    def reproduce(*_args: Any, **_kwargs: Any):
        events.append("compare_primary_evaluation")
        return reproduction, verdict

    monkeypatch.setattr(runner, "_reproduction_v1", reproduce)
    monkeypatch.setattr(
        runner,
        "read_bound_rgb_bytes_v1",
        lambda *_args, **_kwargs: pytest.fail("replay must not open RGB"),
    )
    monkeypatch.setattr(
        runner,
        "_load_vjepa_encoder_v1",
        lambda *_args, **_kwargs: pytest.fail("replay must not execute encoder"),
    )
    report = runner.run_replay_v1(
        authority,
        authority_binding=authority_binding,
        eval_cache_receipt_binding=eval_receipt_binding,
        checkpoint_binding=checkpoint_binding,
        evaluation_binding=evaluation_binding,
    )
    assert events.index("recompute_evaluation") < events.index("load_primary_checkpoint")
    assert events.index("recompute_evaluation") < events.index("load_primary_evaluation")
    assert events.index("recompute_evaluation") < events.index("compare_primary_evaluation")
    assert report["cache_only_feature_inputs"] is True
    assert report["comparison_reference_loads"] == {
        "primary_checkpoint": 1, "primary_evaluation": 1
    }
    assert report["comparison_references_loaded_after_recomputation"] is True
    assert report["rgb_access"] == {"train": 0, "eval": 0}
    assert report["encoder_execution"] == {"vjepa2_1": 0, "other": 0}


def test_reproduction_flags_reject_tensor_score_and_gate_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [
        {
            "true_training": {"optimizer_steps": runner.evaluator.OPTIMIZER_STEPS},
            "current_training": {"optimizer_steps": runner.evaluator.OPTIMIZER_STEPS},
            "state": torch.tensor([float(index)]),
        }
        for index in range(3)
    ]
    checkpoint = {
        "pca": {"values": torch.tensor([1.0])},
        "train_action_mean_innovation": {"values": torch.tensor([2.0])},
        "task_action_only": {"values": torch.tensor([3.0])},
        "members": members,
    }
    evaluation = {
        "prediction_diagnostics": {"seed": [1.0]},
        "score_evidence": {"ensemble": "a"},
        "arms": {
            "candidate": {
                "group_results": [{"selected_action_id": 1}],
                "summary": {"regret": 0.1},
            },
            "random_expected": {
                "group_results": [{"selected_action_id": "NOT_APPLICABLE"}],
                "summary": {"regret": 0.5},
            },
        },
        "paired_family_scene_cluster_comparisons": {"x": {"upper_95": -0.1}},
        "gates": {"x": {"passed": True}},
    }
    monkeypatch.setattr(
        runner.evaluator,
        "verdict_v1",
        lambda value, **_kwargs: {"gates": value["gates"], "terminal_status": "x"},
    )
    exact, _ = runner._reproduction_v1(  # noqa: SLF001
        checkpoint, deepcopy(checkpoint), evaluation, deepcopy(evaluation)
    )
    assert all(exact.values())
    assert runner._selected_actions_v1(evaluation) == {  # noqa: SLF001
        "candidate": [1],
        "random_expected": ["NOT_APPLICABLE"],
    }

    selection_drift = deepcopy(evaluation)
    selection_drift["arms"]["candidate"]["group_results"][0][
        "selected_action_id"
    ] = 2
    observed, _ = runner._reproduction_v1(  # noqa: SLF001
        checkpoint, deepcopy(checkpoint), evaluation, selection_drift
    )
    assert observed["selected_actions"] is False
    assert observed["complete_evaluation"] is False
    assert observed["exactly_reproduced"] is False

    tensor_drift = deepcopy(checkpoint)
    tensor_drift["members"][0]["state"][0] = 9.0
    observed, _ = runner._reproduction_v1(  # noqa: SLF001
        checkpoint, tensor_drift, evaluation, deepcopy(evaluation)
    )
    assert observed["model_state_identities_and_values"] is False
    assert observed["complete_checkpoint_tree"] is False
    assert observed["exactly_reproduced"] is False

    score_drift = deepcopy(evaluation)
    score_drift["prediction_diagnostics"]["seed"][0] = 2.0
    observed, _ = runner._reproduction_v1(  # noqa: SLF001
        checkpoint, deepcopy(checkpoint), evaluation, score_drift
    )
    assert observed["per_seed_scores"] is False
    assert observed["complete_evaluation"] is False
    assert observed["exactly_reproduced"] is False

    gate_drift = deepcopy(evaluation)
    gate_drift["gates"]["x"]["passed"] = False
    observed, _ = runner._reproduction_v1(  # noqa: SLF001
        checkpoint, deepcopy(checkpoint), evaluation, gate_drift
    )
    assert observed["gates"] is False
    assert observed["verdict"] is False
    assert observed["exactly_reproduced"] is False


@pytest.mark.parametrize(
    "row",
    [
        {"selected_action_id": True},
        {"selected_action_id": 1.0},
        {"selected_action_id": "1"},
        {"selected_action_id": None},
        {"selected_action_id": "OTHER"},
        {"selected_action_id": []},
        {"selected_action_id": {}},
        {"selected_action_id": -1},
        {"selected_action_id": 9},
        {},
        [],
    ],
)
def test_selected_actions_rejects_invalid_values_and_rows(row: object) -> None:
    evaluation = {"arms": {"candidate": {"group_results": [row]}}}
    with pytest.raises(runner.DenseVJEPACeilingRunnerError, match="selected action"):
        runner._selected_actions_v1(evaluation)  # noqa: SLF001


def test_execute_fits_before_extraction_and_writes_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "attempt"
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", output_root)
    events: list[str] = []
    authority = {
        "output_root": str(output_root),
        "source_bindings": {"ceiling_evaluator": {}},
        "preregistration_binding": {"synthetic": True},
    }
    authority_binding = {"path": "/synthetic/authority", "sha256": "a" * 64, "byte_count": 1}
    bundle = SimpleNamespace(
        manifest_binding={"synthetic": True},
        access_audit={"rgb_leaf_open_count": 0},
    )
    plan = SimpleNamespace()
    monkeypatch.setattr(runner, "_load_narrow_bundle_v1", lambda _authority: (bundle, {"ok": True}))
    monkeypatch.setattr(runner, "_feature_plans_v1", lambda _bundle: ((1,), (2,), plan, plan))
    monkeypatch.setattr(runner, "_eval_rgb_bindings_from_bundle_v1", lambda *_args: events.append("metadata"))
    monkeypatch.setattr(runner, "_load_train_cache_v1", lambda *_args: (events.append("train") or torch.zeros(1), {"train": True}))
    monkeypatch.setattr(runner, "_authorized_device_v1", lambda: torch.device("cpu"))
    monkeypatch.setattr(runner.evaluator, "fit_primary_checkpoint_v1", lambda *_args, **_kwargs: events.append("fit") or {"state": torch.tensor(1)})

    def extraction(_authority: Any, _bundle: Any, _plan: Any, *, device: Any, output_path: Path):
        events.append("extract")
        runner._save_torch_exclusive(output_path, {"eval": torch.tensor(1)})  # noqa: SLF001
        receipt = {"binding": runner.file_binding_v1(output_path)}
        runner._write_json_exclusive(output_path.with_suffix(".json"), receipt)  # noqa: SLF001
        return receipt

    monkeypatch.setattr(runner, "extract_eval_feature_cache_v1", extraction)
    monkeypatch.setattr(runner, "_load_eval_cache_v1", lambda *_args, **_kwargs: events.append("eval_load") or torch.zeros(1))
    evaluation = {"synthetic": True}
    monkeypatch.setattr(runner, "_evaluate_v1", lambda *_args, **_kwargs: events.append("evaluate") or evaluation)
    monkeypatch.setattr(runner, "_execution_bindings_unchanged_v1", lambda *_args: None)

    def replay_launch(**_kwargs: Any) -> None:
        events.append("replay")
        runner._write_json_exclusive(output_root / "replay.json", {"synthetic": True})  # noqa: SLF001

    monkeypatch.setattr(runner, "_launch_replay_v1", replay_launch)
    verdict = {"terminal_status": runner.STOP_STATUS}
    monkeypatch.setattr(runner, "_validate_replay_v1", lambda *_args, **_kwargs: verdict)
    monkeypatch.setattr(runner, "_verdict_status_v1", lambda *_args: runner.STOP_STATUS)
    report = runner.execute_v1(authority, authority_binding=authority_binding)
    assert events.index("fit") < events.index("extract")
    assert events.index("extract") < events.index("evaluate") < events.index("replay")
    assert report["status"] == runner.STOP_STATUS
    assert report["access_counts"]["replay"]["comparison_reference_loads"] == {
        "primary_checkpoint": 1,
        "primary_evaluation": 1,
    }
    assert set(path.name for path in output_root.iterdir()) == set(runner.OUTPUT_NAMES)
    assert not (output_root / "result.json").is_symlink()


def test_main_failure_terminal_has_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "attempt"
    authority = {"output_root": str(output_root)}
    binding = {"path": "/synthetic/authority", "sha256": "a" * 64, "byte_count": 1}
    monkeypatch.setattr(runner, "_read_authority", lambda *_args, **_kwargs: (authority, binding))

    def fail(_authority: Any, *, authority_binding: Any) -> dict[str, Any]:
        output_root.mkdir()
        runner._write_json_exclusive(output_root / "reservation.json", {"consumed": True})  # noqa: SLF001
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runner, "execute_v1", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.main(
            [
                "--authority", "/synthetic/authority",
                "--expected-authority-sha256", "a" * 64,
                "--expected-authority-byte-count", "1",
            ]
        )
    assert not (output_root / "result.json").exists()
    terminal = json.loads((output_root / "terminal.json").read_text())
    assert terminal["status"] == runner.FAIL_STATUS
    assert terminal["result_binding"] is None
    assert terminal["deterministic_replay_passed"] is False
