from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from lewm.benchmarks.experiment_manifest import sha256_file
from lewm.hierarchical_probability_calibration import CALIBRATION_METHOD
from scripts import rescore_go2_physical_development_checkpoint as rescore
from scripts.train_go2_egomotion_bev_jepa import (
    _canonical_json_sha256,
    _row_subset_record,
    _state_dict_sha256,
)


def _row(tmp_path: Path, global_row: int, role: str) -> dict:
    scene_id = f"{role}_scene"
    return {
        "global_row": global_row,
        "scene_id": scene_id,
        "dataset_role": role,
        "dataset_split": "train" if role == "train" else "validation",
        "family": "unit",
        "primitive": "hold",
        "label_shard_path": str(tmp_path / f"{role}.npz"),
        "label_shard_row": 0,
        "label_shard_sha256": f"{global_row + 1:x}" * 64,
        "current_image_path": str(tmp_path / f"{role}-current.png"),
        "current_image_sha256": "a" * 64,
        "next_image_path": str(tmp_path / f"{role}-next.png"),
        "next_image_sha256": "b" * 64,
    }


def _write_parent(tmp_path: Path, *, tamper: str | None = None) -> dict:
    roles = (
        "train",
        "checkpoint_selection",
        "probability_calibration",
        "g2_evaluation",
    )
    rows = [_row(tmp_path, index, role) for index, role in enumerate(roles)]
    index_path = tmp_path / "rows.jsonl"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = tmp_path / "dataset.json"
    manifest = {
        "schema": "lewm_go2_paired_navigation_dataset_v3",
        "label_semantics": {
            "target_occupancy_space": "observable_physical_occupancy",
            "per_frame_configuration_classes_supervised": False,
            "post_memory_configuration_derivation_is_evaluation_only": True,
        },
        "index": {"path": str(index_path), "sha256": sha256_file(index_path)},
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_sha = sha256_file(manifest_path)

    subsets = {
        "train": _row_subset_record((rows[0],), role="train"),
        "checkpoint_selection": _row_subset_record(
            (rows[1],), role="checkpoint_selection"
        ),
        "probability_calibration": _row_subset_record(
            (rows[2],), role="probability_calibration"
        ),
        "g2_evaluation": _row_subset_record((), role="g2_evaluation"),
    }
    if tamper == "subset":
        subset = subsets["probability_calibration"]
        subset["identities"][0]["current_image_sha256"] = "f" * 64
        subset["identity_sha256"] = _canonical_json_sha256(subset["identities"])

    role_provenance = {
        "train": {"roles": ["train"]},
        "checkpoint_selection": {"roles": ["checkpoint_selection"]},
        "probability_calibration": {"roles": ["probability_calibration"]},
        "g2_evaluation": None,
    }
    ledger = {
        "schema": "lewm_go2_dataset_access_ledger_v1",
        "scope": "trainer_process",
        "row_index_metadata": {
            "read": True,
            "row_count": 4,
            "all_role_metadata_read": True,
            "g2_row_metadata_count": 1,
        },
        "roles": {
            role: {
                "available_row_count": 1,
                "selected_row_count": 0 if role == "g2_evaluation" else 1,
                "row_subset_sha256": subsets[role]["identity_sha256"],
                "provenance_verification": role_provenance[role],
                "label_shard_files_hashed": 0 if role == "g2_evaluation" else 1,
                "image_files_hashed": 0 if role == "g2_evaluation" else 2,
                "model_output_rows": 0 if role == "g2_evaluation" else 1,
            }
            for role in roles
        },
        "g2_contact": {
            "row_metadata_read": True,
            "row_metadata_count": 1,
            "label_shard_byte_opens": 0,
            "image_byte_opens": 0,
            "model_output_rows": 0,
        },
    }
    if tamper == "g2":
        ledger["g2_contact"]["image_byte_opens"] = 1

    model_state = {"weight": torch.tensor([1.0])}
    state_sha = _state_dict_sha256(model_state)
    calibration_provenance = {
        "schema": "lewm_go2_probability_calibration_provenance_v1",
        "role": "probability_calibration",
        "dataset_manifest_sha256": manifest_sha,
        "selected_model_state_sha256": state_sha,
        "calibration_row_subset_sha256": subsets["probability_calibration"][
            "identity_sha256"
        ],
        "calibration_row_count": 1,
        "best_epoch": 2,
    }
    calibration = {
        "method": CALIBRATION_METHOD,
        "id": "go2-hier-cal-unit",
        "content_sha256": "c" * 64,
        "provenance": {"source": calibration_provenance},
    }
    training_core = {
        "schema": "lewm_go2_training_run_provenance_v1",
        "critical_inputs": {
            "trainer_source": {"path": "/old/trainer.py", "sha256": "1" * 64},
            "traversability_metrics_source": {
                "path": "/old/metrics.py",
                "sha256": "2" * 64,
            },
        },
        "row_subsets": subsets,
        "dataset_access_ledger": ledger,
        "checkpoint_artifact_included": False,
    }
    training_run = {
        **training_core,
        "content_sha256": _canonical_json_sha256(training_core),
    }
    dataset_provenance = {"development": {}, "g2_evaluation": None}
    checkpoint = {
        "schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v4",
        "model_state_dict": model_state,
        "model_config": {"image_size": 4, "action_dim": 1},
        "primitive_to_index": {"hold": 0},
        "nominal_primitive_delta_current": [[0.0, 0.0, 0.0]],
        "probability_calibration": calibration,
        "probability_calibration_id": calibration["id"],
        "probability_calibration_provenance": calibration_provenance,
        "occupancy_training_objective": {
            "mode": "hierarchical_equal_capacity_v1",
            "terms": {
                "unknown_vs_known": {"weights": [1.0, 1.0]},
                "free_vs_occupied_given_known": {"weights": [1.0, 1.0]},
            },
        },
        "occupancy_output_contract": {
            "target_occupancy_space": "observable_physical_occupancy"
        },
        "source_fov_rectification": {"center_crop_fraction_xy": [1.0, 1.0]},
        "traversability_thresholds": {
            "free_probability_min": 0.9,
            "occupied_probability_max": 0.1,
            "unknown_probability_max": 0.1,
            "occupied_detection_min": 0.5,
        },
        "best_epoch": 2,
        "dataset_manifest_sha256": manifest_sha,
        "dataset_provenance_verification": dataset_provenance,
        "dataset_role_provenance_verification": role_provenance,
        "dataset_access_ledger": ledger,
        "row_subsets": subsets,
        "training_run_provenance": training_run,
        "g2_evaluation": None,
        "g2_passes": False,
        "head_g2_evaluation": None,
        "head_g2_passes": False,
        "runtime_ready": False,
    }
    checkpoint_path = tmp_path / "parent.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    report = {
        "schema": "lewm_go2_egomotion_bev_jepa_training_report_v4",
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "dataset_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "dataset_provenance_verification": copy.deepcopy(dataset_provenance),
        "dataset_role_provenance_verification": copy.deepcopy(role_provenance),
        "dataset_access_ledger": copy.deepcopy(ledger),
        "row_subsets": copy.deepcopy(subsets),
        "probability_calibration": copy.deepcopy(calibration),
        "probability_calibration_provenance": copy.deepcopy(calibration_provenance),
        "training_run_provenance": copy.deepcopy(training_run),
        "best_epoch": 2,
        "label_semantics": copy.deepcopy(manifest["label_semantics"]),
        "final_g2_evaluation": None,
        "final_head_g2_evaluation": None,
        "promotion": {
            "head_g2_passes": False,
            "head_g2_evaluated": False,
            "runtime_ready": False,
        },
    }
    if tamper == "agreement":
        report["checkpoint"]["sha256"] = "0" * 64
    report_path = tmp_path / "parent.report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    return {
        "manifest": manifest_path,
        "checkpoint": checkpoint_path,
        "checkpoint_sha": checkpoint_sha,
        "report": report_path,
        "report_sha": sha256_file(report_path),
        "output": tmp_path / "rescore.json",
    }


def _install_lifecycle_mocks(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls = {"model": 0, "evaluations": []}

    monkeypatch.setattr(
        rescore,
        "resolve_dataset_scene_roles",
        lambda rows, _manifest, **_kwargs: {
            str(row["scene_id"]): str(row["dataset_role"]) for row in rows
        },
    )
    monkeypatch.setattr(
        rescore,
        "_verify_selected_artifacts",
        lambda rows_by_role: {
            role: {
                "selected_row_count": len(rows_by_role[role]),
                "label_shard_files_verified": 1,
                "image_files_verified": 2,
                "artifact_identity_sha256": "d" * 64,
            }
            for role in rescore.DEVELOPMENT_ROLES
        },
    )
    monkeypatch.setattr(
        rescore,
        "validate_hierarchical_probability_calibration",
        lambda _calibration: None,
    )
    monkeypatch.setattr(
        rescore,
        "_git_snapshot",
        lambda: {"head": "a" * 40, "status_short": ""},
    )

    class FakeModel(torch.nn.Module):
        def __init__(self, **_config):
            super().__init__()
            calls["model"] += 1
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(rescore, "EgomotionBevJepa", FakeModel)
    monkeypatch.setattr(
        rescore,
        "_loader",
        lambda rows, **_kwargs: {
            "role": str(rows[0]["dataset_role"]),
            "rows": len(rows),
        },
    )

    def fake_evaluate(_model, loader, *, thresholds, select_thresholds, **_kwargs):
        calls["evaluations"].append((loader["role"], select_thresholds))
        corrected = {
            "free_probability_min": 0.9,
            "occupied_probability_max": 0.01,
            "unknown_probability_max": 0.1,
            "occupied_detection_min": 0.02,
        }
        if select_thresholds:
            assert loader["role"] == "probability_calibration"
            assert thresholds is None
        else:
            assert loader["role"] == "checkpoint_selection"
            assert thresholds.occupied_detection_min == pytest.approx(0.02)
        return {
            "rows": loader["rows"],
            "thresholds": corrected,
            "threshold_selection": (
                {"candidate_count": 2016, "passing_candidate_count": 1}
                if select_thresholds
                else None
            ),
        }

    monkeypatch.setattr(rescore, "evaluate_model", fake_evaluate)
    return calls


def _argv(paths: dict, *, checkpoint_sha: str | None = None) -> list[str]:
    return [
        "--dataset-manifest",
        str(paths["manifest"]),
        "--parent-checkpoint",
        str(paths["checkpoint"]),
        "--expected-parent-checkpoint-sha256",
        checkpoint_sha or paths["checkpoint_sha"],
        "--parent-report",
        str(paths["report"]),
        "--expected-parent-report-sha256",
        paths["report_sha"],
        "--output",
        str(paths["output"]),
        "--device",
        "cpu",
    ]


def test_rescore_uses_exact_frozen_development_subsets_and_emits_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_parent(tmp_path)
    calls = _install_lifecycle_mocks(monkeypatch)

    assert rescore.main(_argv(paths)) == 0

    output = json.loads(paths["output"].read_text())
    core = dict(output)
    content_sha = core.pop("content_sha256")
    assert content_sha == _canonical_json_sha256(core)
    assert output["schema"] == rescore.OUTPUT_SCHEMA
    assert output["corrected_thresholds"]["occupied_detection_min"] == 0.02
    assert output["frozen_identity"]["model_state_unchanged"] is True
    assert output["frozen_identity"]["calibration_unchanged"] is True
    assert output["dataset_access_ledger"]["g2_contact"] == {
        "row_metadata_read": True,
        "row_metadata_count": 1,
        "label_shard_byte_opens": 0,
        "image_byte_opens": 0,
        "model_output_rows": 0,
    }
    assert output["eligibility"]["one_shot_promotion_eligible"] is False
    assert output["eligibility"]["runtime_promotion_eligible"] is False
    assert calls == {
        "model": 1,
        "evaluations": [
            ("probability_calibration", True),
            ("checkpoint_selection", False),
        ],
    }


@pytest.mark.parametrize("tamper", ("subset", "g2", "agreement"))
def test_rescore_fails_before_model_load_on_parent_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    paths = _write_parent(tmp_path, tamper=tamper)
    calls = _install_lifecycle_mocks(monkeypatch)

    with pytest.raises(ValueError):
        rescore.main(_argv(paths))

    assert calls["model"] == 0
    assert calls["evaluations"] == []
    assert not paths["output"].exists()


def test_rescore_rejects_expected_parent_hash_before_loading_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_parent(tmp_path)
    calls = _install_lifecycle_mocks(monkeypatch)
    monkeypatch.setattr(
        rescore,
        "_load_parent_checkpoint",
        lambda _path: pytest.fail("checkpoint must not load after expected-hash failure"),
    )

    with pytest.raises(ValueError, match="does not match expected"):
        rescore.main(_argv(paths, checkpoint_sha="f" * 64))

    assert calls["model"] == 0
    assert not paths["output"].exists()
