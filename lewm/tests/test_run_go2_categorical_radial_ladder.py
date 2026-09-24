from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from lewm.benchmarks.go2_categorical_radial_factorization import (
    audit_mapping_injectivity,
    geometry_metadata,
)
from lewm.benchmarks.go2_categorical_radial_micro_overfit import (
    canonical_json_sha256,
    frame_identity,
    select_ladder_frames,
)
from scripts import run_go2_categorical_radial_ladder as runner


FROZEN_LADDER_SHA256 = (
    "967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_manifest(tmp_path: Path) -> tuple[Path, str]:
    labels = np.zeros((16, 64, 64), dtype=np.uint8)
    labels[:, 21:42] = 1
    labels[:, 42:] = 2
    masks = np.ones_like(labels, dtype=bool)
    shard_path = tmp_path / "labels.npz"
    np.savez(
        shard_path,
        current_labels=labels,
        current_supervision_mask=masks,
    )
    shard_sha256 = _sha256_file(shard_path)

    records = []
    for index in range(16):
        image_path = tmp_path / f"image_{index:02d}.png"
        pixels = np.full((20, 24, 3), index * 8, dtype=np.uint8)
        Image.fromarray(pixels).save(image_path)
        records.append(
            {
                "scene_id": f"scene_{index:02d}",
                "family": f"family_{index % 5}",
                "global_row": index,
                "side": "current",
                "image_path": str(image_path.resolve()),
                "image_sha256": _sha256_file(image_path),
                "label_shard_path": str(shard_path.resolve()),
                "label_shard_sha256": shard_sha256,
                "label_shard_row": index,
            }
        )
    presence = {frame_identity(record): (True, True, True) for record in records}
    ladder = select_ladder_frames(records, class_presence=presence)

    panel_path = tmp_path / "parent_panel.json"
    panel_path.write_text('{"frozen":true}\n')
    panel_sha256 = _sha256_file(panel_path)
    current_sources = runner._source_hashes()
    roundtrip_report = {
        "frame_count": 320,
        "outside_support_known_count": 0,
        "roundtrip_mismatch_count": 0,
        "exact_roundtrip": True,
    }
    core = {
        "schema": runner.MANIFEST_SCHEMA,
        "created_at_utc": "2026-07-10T00:00:00+00:00",
        "invocation": ["synthetic-test"],
        "inputs": {
            "panel_manifest": {
                "path": str(panel_path.resolve()),
                "sha256": panel_sha256,
                "expected_sha256": panel_sha256,
                "content_sha256": "0" * 64,
                "pre_deserialization_hash_match": True,
            }
        },
        "factorization": geometry_metadata(),
        "mapping_audit": audit_mapping_injectivity(),
        "roundtrip_audit": {
            "all_960_frames_exact": True,
            "panels": {
                name: dict(roundtrip_report)
                for name in (
                    "fit",
                    "same_scene_holdout",
                    "cross_scene_holdout",
                )
            },
        },
        "ladder": ladder,
        "artifact_access_ledger": {
            "runner_input_contains_only_train_rows": True,
            "train_image_byte_opens": 0,
            "checkpoint_selection": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "probability_calibration": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "g2_evaluation": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
        },
        "source_hashes": {
            name: current_sources[name]
            for name in runner.MANIFEST_SOURCE_BINDINGS
        },
    }
    manifest = {**core, "content_sha256": canonical_json_sha256(core)}
    manifest_path = tmp_path / "ladder.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest_path, _sha256_file(manifest_path)


class _TinyCategoricalRadial(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 3, 64, 64))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(image.shape[0], -1, -1, -1)


def test_hierarchical_loss_matches_explicit_weighted_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(2, 3, 64, 64, generator=generator, requires_grad=True)
    labels = torch.arange(64).remainder(3)[None, :, None]
    labels = labels.expand(2, 64, 64).long()
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[:, ::5] = False

    loss = runner.hierarchical_occupancy_loss(logits, labels, mask)
    known_logit = torch.logsumexp(logits[:, 1:], dim=1)
    unknown_known_logits = torch.stack((logits[:, 0], known_logit), dim=1)
    unknown_known_labels = (labels != 0).long()
    unknown_known_weights = logits.new_tensor(
        runner.TRAINING_WEIGHTS["unknown_known"]
    )
    uk_per_cell = torch.nn.functional.cross_entropy(
        unknown_known_logits,
        unknown_known_labels,
        reduction="none",
    )
    uk_applied = unknown_known_weights[unknown_known_labels] * mask
    uk_loss = (uk_per_cell * uk_applied).sum() / uk_applied.sum()

    known_mask = mask & (labels != 0)
    free_occupied_labels = (labels - 1).clamp_min(0)
    free_occupied_weights = logits.new_tensor(
        runner.TRAINING_WEIGHTS["free_occupied"]
    )
    fo_per_cell = torch.nn.functional.cross_entropy(
        logits[:, 1:],
        free_occupied_labels,
        reduction="none",
    )
    fo_applied = free_occupied_weights[free_occupied_labels] * known_mask
    fo_loss = (fo_per_cell * fo_applied).sum() / fo_applied.sum()
    expected = 0.5 * uk_loss + 0.5 * fo_loss
    loss.backward()

    assert torch.allclose(loss, expected)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_cyclic_wrong_view_is_zero_match_and_rejects_duplicates() -> None:
    records = [
        {"scene_id": f"scene_{index}", "image_sha256": f"image_{index}"}
        for index in range(4)
    ]
    permutation, contract = runner.deterministic_cyclic_wrong_view(records)

    assert permutation == (1, 2, 3, 0)
    assert contract["same_scene_pairs"] == 0
    assert contract["same_image_pairs"] == 0
    duplicate = [dict(record) for record in records]
    duplicate[1]["scene_id"] = duplicate[0]["scene_id"]
    with pytest.raises(ValueError, match="scene- and image-disjoint"):
        runner.deterministic_cyclic_wrong_view(duplicate)


def test_determinism_warns_for_unsupported_rocm_kernels(monkeypatch) -> None:
    call = {}

    def use_deterministic(enabled: bool, *, warn_only: bool) -> None:
        call.update(enabled=enabled, warn_only=warn_only)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", use_deterministic)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = runner._configure_determinism(20260710)

    assert call == {"enabled": True, "warn_only": True}
    assert result["requested"] == "strict_deterministic_algorithms"
    assert result["effective"] == "strict_where_supported_warn_on_unsupported"
    assert result["warn_only"] is True


def test_cli_accepts_precommitted_real_ladder_hash(tmp_path: Path) -> None:
    args = runner._parse_args(
        [
            "--ladder-manifest",
            str(tmp_path / "ladder.json"),
            "--expected-ladder-sha256",
            FROZEN_LADDER_SHA256,
            "--output",
            str(tmp_path / "result.json"),
        ]
    )
    assert args.expected_ladder_sha256 == FROZEN_LADDER_SHA256


def test_manifest_validation_rejects_rehashed_source_tampering(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_sha256 = _write_synthetic_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert len(runner.validate_ladder_manifest(manifest)) == 16

    manifest["source_hashes"]["protocol"]["sha256"] = "0" * 64
    core = dict(manifest)
    core.pop("content_sha256")
    manifest["content_sha256"] = canonical_json_sha256(core)
    with pytest.raises(ValueError, match="different protocol"):
        runner.validate_ladder_manifest(manifest)


def test_smoke_wires_all_stages_and_never_reads_control_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest_sha256 = _write_synthetic_manifest(tmp_path)
    output_path = tmp_path / "smoke_result.json"
    monkeypatch.setattr(runner, "CategoricalRadialPerception", _TinyCategoricalRadial)

    status = runner.main(
        [
            "--ladder-manifest",
            str(manifest_path),
            "--expected-ladder-sha256",
            manifest_sha256,
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--seed",
            "20260710",
            "--non-authoritative-smoke",
        ]
    )
    result = json.loads(output_path.read_text())

    assert status == 0
    assert result["schema"] == runner.SMOKE_RESULT_SCHEMA
    assert result["authoritative"] is False
    assert result["promotion_eligible"] is False
    assert [stage["frame_count"] for stage in result["stages"]] == [1, 4, 16]
    assert all(stage["completed_updates"] == 3 for stage in result["stages"])
    assert all(stage["fixed_budget_consumed"] for stage in result["stages"])
    assert result["decision"]["stopped_on_first_failed_stage"] is False
    assert result["decision"]["smoke_exercised_all_stage_paths"] is True
    assert result["decision"]["n32_attempted"] is False
    assert result["model"]["stage_restart_initial_hashes_equal"] is True

    stages = {stage["frame_count"]: stage for stage in result["stages"]}
    assert stages[1]["access_ledger"]["target_requests"] == 6
    assert stages[4]["access_ledger"]["target_requests"] == 24
    assert stages[16]["access_ledger"]["target_requests"] == 60
    for size in (4, 16):
        for point in stages[size]["curve"]:
            control = point["evaluation"]["wrong_view_control"]
            assert control["same_scene_pairs"] == 0
            assert control["same_image_pairs"] == 0
    ledger = result["artifact_access_ledger"]
    assert ledger["selected_train_image_hash_byte_open_events"] == 32
    assert ledger["selected_train_label_shard_hash_byte_open_events"] == 2
    assert ledger["non_train_image_opens"] == 0
    assert ledger["non_train_label_shard_opens"] == 0


def test_authoritative_run_stops_immediately_after_failed_n1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest_sha256 = _write_synthetic_manifest(tmp_path)
    output_path = tmp_path / "authoritative_result.json"
    calls = []

    def failed_stage(records, **kwargs):
        calls.append(len(records))
        return {
            "frame_count": len(records),
            "initial_state_sha256": kwargs["initial_state_sha256"],
            "final_fit_gate_passes": False,
            "access_ledger": {},
        }

    monkeypatch.setattr(runner, "CategoricalRadialPerception", _TinyCategoricalRadial)
    monkeypatch.setattr(runner, "_train_stage", failed_stage)
    status = runner.main(
        [
            "--ladder-manifest",
            str(manifest_path),
            "--expected-ladder-sha256",
            manifest_sha256,
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--seed",
            "20260710",
        ]
    )
    result = json.loads(output_path.read_text())

    assert status == 0
    assert calls == [1]
    assert result["schema"] == runner.RESULT_SCHEMA
    assert result["decision"]["attempted_frame_counts"] == [1]
    assert result["decision"]["stopped_on_first_failed_stage"] is True
    assert result["decision"][
        "authoritative_first_failure_stop_policy_enforced"
    ] is True


def test_atomic_result_write_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "immutable.json"
    runner._atomic_write_json_exclusive(output, {"version": 1})
    with pytest.raises(FileExistsError, match="already exists"):
        runner._atomic_write_json_exclusive(output, {"version": 2})
    assert json.loads(output.read_text()) == {"version": 1}
