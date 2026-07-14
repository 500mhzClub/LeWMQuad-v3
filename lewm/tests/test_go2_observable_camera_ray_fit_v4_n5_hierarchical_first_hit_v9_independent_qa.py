"""Independent source-only review probes for Camera-ray N5 V9."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_ladder_gate as gate,
)
from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9 as policy,
)
from lewm.models.observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 import (
    hierarchical_first_hit_nll_breakdown_v9,
    hierarchical_first_hit_nll_from_log_probabilities_v9,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    ObservableCameraRayEvidenceV4Targets,
)
from lewm.tests.n5_hierarchical_first_hit_v9_synthetic_execution import (
    SyntheticExecutionV9,
)


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/raw_v8_builder_reviewer"
AMENDMENT = {
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_"
    "preimplementation_amendment_2026-07-13.md": (
        "ccc8097b4d3bd70aabf3c701226928e360fafb04a12a452c4fd406e9bba3db0a"
    )
}
SOURCES = {
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py": (
        "52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_"
    "hierarchical_first_hit_v9.py": (
        "00e0cbc796d83ce9137f95f853d6262cac4a464782540ecd05276927267c8be1"
    ),
    "scripts/train_go2_observable_camera_ray_fit_v4_n5_"
    "hierarchical_first_hit_v9.py": (
        "af8baa9a4aac7f0de19caa55f43e6120010e7d6765e0dceaa7cb18e95a88888f"
    ),
    "scripts/verify_go2_observable_camera_ray_fit_v4_n5_"
    "hierarchical_first_hit_v9.py": (
        "43142be57b105bacf90124223c67d93372482ae0eeb64f4e9a8658f5a951909e"
    ),
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_"
    "hierarchical_first_hit_v9.py": (
        "94cbe45f290f92a2a5ffaf7e87063e78e1aec17ba8d4fcae9e799e2235374246"
    ),
}
PROOFS = {
    "lewm/tests/n5_hierarchical_first_hit_v9_synthetic_execution.py": (
        "fd12a7dd1d877e507a0d332e4d96e684cc989fe0242fe1ee6ac61598d5702d3e"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_"
    "hierarchical_first_hit_v9.py": (
        "5bb9e1c31e26ef4d4490013b9d377db161fa5ecde7471d4fa9ca4eb44a6a227b"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_"
    "hierarchical_first_hit_v9_lifecycle.py": (
        "d7a7048d2242be98aec9f7e2d66d4121d0e5f67e65c9d51292c08b311e7053ee"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9_"
    "implementation_handoff_2026-07-13.md": (
        "50e22a56d2cb49e3b449aa760883c22dec1521abbd0d1b43fdbd0a69c5f374f2"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _targets(hit_bins: list[int | None]) -> ObservableCameraRayEvidenceV4Targets:
    hit_mask = torch.tensor(
        [value is not None for value in hit_bins], dtype=torch.bool
    ).reshape(1, 1, -1)
    bins = torch.tensor(
        [0 if value is None else value for value in hit_bins], dtype=torch.long
    ).reshape(1, 1, -1)
    return ObservableCameraRayEvidenceV4Targets(
        pixel_in_range_hit_mask=hit_mask,
        pixel_no_hit_mask=~hit_mask,
        pixel_hit_bin_index=bins,
        pixel_within_bin_offset_m=torch.zeros_like(hit_mask, dtype=torch.float64),
        ground_in_frustum=torch.zeros((1, 1, 1, 5), dtype=torch.bool),
        ground_clear_to_target=torch.zeros((1, 1, 1, 5), dtype=torch.bool),
    )


def _direct_loss(
    hit_probabilities: list[list[float]],
    no_hit_probabilities: list[float],
    hit_bins: list[int | None],
    *,
    requires_grad: bool = False,
) -> Any:
    assert len(hit_probabilities) == len(no_hit_probabilities) == len(hit_bins)
    assert all(
        sum(hit) + no_hit == pytest.approx(1.0)
        for hit, no_hit in zip(
            hit_probabilities, no_hit_probabilities, strict=True
        )
    )
    hit = torch.tensor(hit_probabilities, dtype=torch.float64).transpose(0, 1)
    hit_log = hit.reshape(1, hit.shape[0], 1, hit.shape[1]).log()
    no_hit_log = torch.tensor(
        no_hit_probabilities, dtype=torch.float64
    ).reshape(1, 1, -1).log()
    hit_log.requires_grad_(requires_grad)
    no_hit_log.requires_grad_(requires_grad)
    targets = _targets(hit_bins)
    return hierarchical_first_hit_nll_from_log_probabilities_v9(
        hit_log_probabilities=hit_log,
        no_hit_log_probability=no_hit_log,
        pixel_in_range_hit_mask=targets.pixel_in_range_hit_mask,
        pixel_no_hit_mask=targets.pixel_no_hit_mask,
        pixel_hit_bin_index=targets.pixel_hit_bin_index,
    )


def _definitions(path: Path, *, normalize_v8: bool = False) -> list[tuple[str, str]]:
    source = path.read_text(encoding="utf-8")
    if normalize_v8:
        source = source.replace(
            "go2_observable_camera_ray_fit_v4_n5_full_panel_v8",
            "go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9",
        )
        source = source.replace(
            "n5_full_panel_v8", "n5_hierarchical_first_hit_v9"
        )
        source = source.replace("V8", "V9").replace("v8", "v9")
    return [
        (node.name, ast.dump(node, include_attributes=False))
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]


def _named_definitions(path: Path, *, v1_to_v9: bool = False) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    if v1_to_v9:
        source = source.replace(
            "go2_observable_camera_ray_fit_v4_n5_full_panel_v1",
            "go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9",
        )
        source = source.replace(
            "n5_full_panel_v1", "n5_hierarchical_first_hit_v9"
        )
        source = source.replace("V1", "V9").replace("v1", "v9")
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _training_record() -> dict[str, Any]:
    components = {
        "hierarchical_first_hit_nll": 0.8,
        "target_bin_offset_smooth_l1": 0.02,
        "ground_clear_distance_state_balanced_bce": 0.04,
        "derived_raster_hierarchical_bce": 0.2,
    }
    trace = [
        {
            "step": step,
            "total": 0.25 * sum(components.values()),
            "components": dict(components),
            "gradient_norm_before_clip": 1.0,
        }
        for step in (1, *range(100, 4001, 100))
    ]
    return {
        "steps": 4000,
        "batch_size": 5,
        "evaluation_batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "precision": "float32",
        "autocast": False,
        "gradient_clip_norm": 1.0,
        "loss_weights": {name: 0.25 for name in policy.LOSS_COMPONENTS},
        "schedule_algorithm": policy.SCHEDULE_ALGORITHM,
        "schedule_sha256": policy.EXPECTED_SCHEDULE_SHA256,
        "checkpoint_selection": "final_update_only",
        "frame_exposures": 20000,
        "fresh_model_initialization": True,
        "diagnostic_updates": [row["step"] for row in trace],
        "initial": trace[0],
        "final": trace[-1],
        "trace": trace,
    }


def test_v9_frozen_source_and_proof_bytes_are_exact_ascii() -> None:
    frozen = {**AMENDMENT, **SOURCES, **PROOFS}
    assert {relative: _sha256(ROOT / relative) for relative in frozen} == frozen
    for relative in frozen:
        (ROOT / relative).read_bytes().decode("ascii")
    assert tuple(SOURCES) == policy.SUCCESSOR_SOURCE_PATHS
    assert tuple(PROOFS) == policy.SUCCESSOR_PROOF_PATHS


def test_v9_hierarchical_loss_matches_independent_state_and_bin_arithmetic() -> None:
    loss = _direct_loss(
        [
            [0.05, 0.03, 0.02],
            [0.10, 0.10, 0.20],
            [0.56, 0.14, 0.10],
            [0.05, 0.35, 0.10],
            [0.15, 0.15, 0.45],
        ],
        [0.90, 0.60, 0.20, 0.50, 0.25],
        [None, None, 0, 1, 2],
    )
    no_hit = (-math.log(0.90) - math.log(0.60)) / 2.0
    hit = (-math.log(0.80) - math.log(0.50) - math.log(0.75)) / 3.0
    presence = 0.5 * (no_hit + hit)
    conditional = (-math.log(0.70) - math.log(0.70) - math.log(0.60)) / 3.0
    assert loss.no_hit_presence_nll.item() == pytest.approx(no_hit)
    assert loss.hit_presence_nll.item() == pytest.approx(hit)
    assert loss.presence_nll.item() == pytest.approx(presence)
    assert loss.conditional_depth_nll.item() == pytest.approx(conditional)
    assert loss.total.item() == pytest.approx(0.5 * presence + 0.5 * conditional)
    assert loss.hit_distance_bin_counts == (1, 1, 1)
    old_group_weight = (no_hit - math.log(0.70) - math.log(0.70) - math.log(0.60)) / 4.0
    assert loss.presence_nll.item() != pytest.approx(old_group_weight)


def test_v9_loss_invariances_and_extreme_hazard_gradients_are_independent() -> None:
    first = _direct_loss(
        [[0.48, 0.24, 0.08], [0.08, 0.24, 0.48]],
        [0.20, 0.20],
        [0, 2],
    )
    redistributed = _direct_loss(
        [[0.64, 0.08, 0.08], [0.08, 0.08, 0.64]],
        [0.20, 0.20],
        [0, 2],
    )
    lower_presence = _direct_loss(
        [[0.24, 0.12, 0.04], [0.04, 0.12, 0.24]],
        [0.60, 0.60],
        [0, 2],
    )
    assert first.presence_nll.item() == pytest.approx(
        redistributed.presence_nll.item()
    )
    assert first.conditional_depth_nll.item() != pytest.approx(
        redistributed.conditional_depth_nll.item()
    )
    assert first.conditional_depth_nll.item() == pytest.approx(
        lower_presence.conditional_depth_nll.item()
    )
    assert first.presence_nll.item() != pytest.approx(
        lower_presence.presence_nll.item()
    )

    logits = torch.tensor(
        [
            [10000.0, -10000.0, 80.0, -80.0],
            [-10000.0, 10000.0, -80.0, 80.0],
            [80.0, -80.0, 10000.0, -10000.0],
        ],
        dtype=torch.float64,
    ).reshape(1, 3, 1, 4)
    logits.requires_grad_(True)
    extreme = hierarchical_first_hit_nll_breakdown_v9(
        logits, _targets([None, 0, 1, 2])
    )
    extreme.total.backward()
    assert torch.isfinite(extreme.total)
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert extreme.nonempty_presence_group_count == 2
    assert extreme.nonempty_conditional_depth_group_count == 3


def test_v9_schedule_is_independently_reconstructed_and_final_only() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260710)
    schedule = tuple(
        tuple(torch.randperm(5, generator=generator).tolist())
        for _update in range(4000)
    )
    assert len(schedule) == 4000
    assert sum(len(panel) for panel in schedule) == 20000
    assert all(len(panel) == 5 and set(panel) == set(range(5)) for panel in schedule)
    assert policy.canonical_json_sha256(schedule) == (
        "fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380"
    )
    assert policy._expected_diagnostic_updates() == (1, *range(100, 4001, 100))

    valid = _training_record()
    assert policy._validate_training_record(valid) == valid
    for field, changed in (
        ("steps", 3999),
        ("frame_exposures", 19995),
        ("checkpoint_selection", "best_loss"),
        ("fresh_model_initialization", False),
    ):
        mutated = deepcopy(valid)
        mutated[field] = changed
        with pytest.raises(PermissionError):
            policy._validate_training_record(mutated)


def test_v9_retains_model_panel_raster_and_exact_26_threshold_gate() -> None:
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )

    experiment = policy.experiment_contract()
    changed = {
        "optimizer_updates",
        "frame_exposures",
        "loss_weights",
        "schedule_sha256",
        "output_path",
    }
    assert {
        key: value for key, value in experiment.items() if key not in changed
    } == {
        key: value for key, value in retained.EXPERIMENT.items() if key not in changed
    }
    assert _sha256(ROOT / "lewm/models/observable_camera_ray_evidence_v4.py") == (
        policy.frozen_source_bindings()[
            "lewm/models/observable_camera_ray_evidence_v4.py"
        ]
    )
    assert _sha256(ROOT / "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py") == (
        policy.frozen_source_bindings()[
            "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py"
        ]
    )
    assert asdict(gate.FIT_THRESHOLDS[5]) == {
        "pixel_hit_balanced_accuracy_min": 0.99,
        "pixel_hit_depth_median_error_m_max": 0.06,
        "pixel_hit_depth_p95_error_m_max": 0.15,
        "ground_overall_balanced_accuracy_min": 0.99,
        "ground_distance_balanced_accuracy_min": 0.97,
        "ground_family_balanced_accuracy_min": 0.97,
        "raster_nll_max": 0.06,
        "raster_balanced_accuracy_min": 0.99,
        "raster_class_recall_min": 0.97,
        "wrong_pixel_balanced_accuracy_drop_min": 0.08,
        "wrong_depth_median_error_increase_m_min": 0.08,
        "wrong_depth_p95_error_increase_m_min": 0.12,
        "wrong_ground_balanced_accuracy_drop_min": 0.08,
        "wrong_raster_nll_increase_min": 0.08,
        "wrong_raster_balanced_accuracy_drop_min": 0.08,
    }
    matched = {
        "pixel_hit_no_hit": {"balanced_accuracy": 1.0},
        "pixel_hit_depth": {
            "median_absolute_error_m": 0.0,
            "p95_absolute_error_m": 0.0,
        },
        "ground_clear": {
            "overall": {"balanced_accuracy": 1.0},
            "by_distance_m": {
                f"distance_{index}": {"count": 1, "balanced_accuracy": 1.0}
                for index in range(6)
            },
            "by_family": {
                f"family_{index}": {"balanced_accuracy": 1.0}
                for index in range(5)
            },
        },
        "derived_raster": {
            "nll": 0.0,
            "balanced_accuracy": 1.0,
            "class_recalls": {name: 1.0 for name in ("unknown", "free", "blocked")},
        },
    }
    wrong = {
        "pixel_hit_no_hit": {"balanced_accuracy": 0.0},
        "pixel_hit_depth": {
            "median_absolute_error_m": 1.0,
            "p95_absolute_error_m": 1.0,
        },
        "ground_clear": {"overall": {"balanced_accuracy": 0.0}},
        "derived_raster": {"nll": 1.0, "balanced_accuracy": 0.0},
    }
    decision = gate._gate_stage({"fit_size": 5, "matched": matched, "wrong": wrong})
    assert decision["check_count"] == 26
    assert decision["failure_count"] == 0 and decision["passes"] is True


def test_v9_executor_and_synthetic_lifecycle_are_mechanical_v8_successors() -> None:
    previous = _definitions(
        ROOT / "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v8.py",
        normalize_v8=True,
    )
    successor = _definitions(ROOT / policy.EXECUTOR_RELATIVE_PATH)
    assert [name for name, _dump in previous] == [name for name, _dump in successor]
    changed = {
        name
        for (name, old), (_same_name, new) in zip(previous, successor, strict=True)
        if old != new
    }
    assert changed == {
        "_reservation_core",
        "_run_frozen_training",
        "_compute_verification_receipt_child",
        "run_cpu_contract_smoke",
    }
    previous_synthetic = _definitions(
        ROOT / "lewm/tests/n5_full_panel_v8_synthetic_execution.py",
        normalize_v8=True,
    )
    successor_synthetic = _definitions(ROOT / policy.SYNTHETIC_RELATIVE_PATH)
    assert previous_synthetic == successor_synthetic


def test_v9_trainer_and_verifier_deltas_are_only_the_reviewed_science() -> None:
    prior_trainer = _named_definitions(
        ROOT / "scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py",
        v1_to_v9=True,
    )
    trainer = _named_definitions(ROOT / policy.TRAINER_RELATIVE_PATH)
    assert set(prior_trainer) - set(trainer) == {"evaluate_full_panel_v9"}
    assert set(trainer) - set(prior_trainer) == {
        "build_checkpoint_metadata_v9",
        "compute_four_equal_v9_losses",
        "evaluate_hierarchical_first_hit_v9",
        "train_v9_fit",
    }
    assert {
        name
        for name in set(prior_trainer) & set(trainer)
        if prior_trainer[name] != trainer[name]
    } == {"_reservation_core", "_run_training", "run_cpu_contract_smoke"}

    prior_verifier = _named_definitions(
        ROOT / "scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py",
        v1_to_v9=True,
    )
    verifier = _named_definitions(ROOT / policy.VERIFIER_RELATIVE_PATH)
    assert set(prior_verifier) - set(verifier) == {"_isolated_child", "run"}
    assert set(verifier) - set(prior_verifier) == {
        "compute_four_equal_v9_losses_for_verification"
    }
    assert {
        name
        for name in set(prior_verifier) & set(verifier)
        if prior_verifier[name] != verifier[name]
    } == {"_compute_receipt", "main", "recompute_evaluation"}

    production_text = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            policy.LOSS_RELATIVE_PATH,
            policy.TRAINER_RELATIVE_PATH,
            policy.VERIFIER_RELATIVE_PATH,
            policy.EXECUTOR_RELATIVE_PATH,
        )
    )
    assert "ordered_first_hit_nll" not in production_text
    assert "n5_full_panel_recovery_v8" not in production_text
    trainer_text = (ROOT / policy.TRAINER_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "torch.load" not in trainer_text
    assert "load_state_dict" not in trainer_text
    assert trainer_text.count("base.ObservableCameraRayEvidenceV4Model()") == 1


def test_v9_terminal_v8_evidence_is_identity_only_without_opening_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []

    def rehash_only(relative: str, expected: str, *, name: str) -> bytes:
        del name
        assert policy.is_sha256(expected)
        observed.append((relative, expected))
        return b"identity-only"

    monkeypatch.setattr(policy, "_hash_file", rehash_only)
    binding = policy._validate_v8_terminal_result()
    expected_paths = {
        relative for relative, _digest in policy.RETAINED_V8_ARTIFACT_BINDINGS
    } | {
        policy.V8_RESULT_RELATIVE_PATH,
        policy.V8_METRIC_RELATIVE_PATH,
        policy.V8_GATE_RELATIVE_PATH,
    }
    assert {relative for relative, _digest in observed} == expected_paths
    assert binding["terminal"] is True
    assert binding["retry_authorized"] is False
    assert binding["checkpoint_input_authorized"] is False
    assert binding["numeric_payload_inspected"] is False
    assert binding["validation_mode"] == "exact_byte_rehash_only"


def test_v9_schema_rejects_old_loss_alias_and_checkpoint_selection() -> None:
    loss = {
        "hierarchical_first_hit_nll": 0.8,
        "target_bin_offset_smooth_l1": 0.02,
        "ground_clear_distance_state_balanced_bce": 0.04,
        "derived_raster_hierarchical_bce": 0.2,
        "total": 0.265,
    }
    assert policy._validate_loss_record(loss, name="review") == loss
    retired = dict(loss)
    retired["ordered_first_hit_nll"] = retired.pop("hierarchical_first_hit_nll")
    with pytest.raises(ValueError, match="loss fields changed"):
        policy._validate_loss_record(retired, name="review")

    from scripts import (
        train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9
        as trainer,
    )

    metadata = trainer.build_checkpoint_metadata_v9(
        source_review={"path": "review", "file_sha256": "a" * 64},
        inputs={"dataset": "frozen"},
        reservation_binding={"path": "reservation", "file_sha256": "b" * 64},
        training_schedule_sha256=policy.EXPECTED_SCHEDULE_SHA256,
    )
    assert metadata["checkpoint_selection"] == "final_update_only"
    assert metadata["loss_contract"] == {
        "version": "hierarchical_first_hit_v9",
        "components": list(policy.LOSS_COMPONENTS),
        "weights": {name: 0.25 for name in policy.LOSS_COMPONENTS},
    }


def test_v9_review_core_binds_all_sources_proofs_and_no_downstream_license() -> None:
    sources = {
        relative: {"path": relative, "file_sha256": digest}
        for relative, digest in SOURCES.items()
    }
    proofs = {
        relative: {"path": relative, "file_sha256": digest}
        for relative, digest in PROOFS.items()
    }
    core = policy.expected_source_review_core(
        reviewer=REVIEWER,
        successor_sources=sources,
        successor_proofs=proofs,
    )
    assert core["implementation_author"] == "/root/coordinator_v2_qa"
    assert core["reviewer"] == REVIEWER
    assert core["source_closure_approved"] is True
    assert core["exact_attempt_authorized"] is True
    assert core["scientific_successor_authorized"] is True
    assert core["scientific_retry_authorized"] is False
    assert core["v8_numeric_payload_inspected"] is False
    assert core["v8_checkpoint_inspected"] is False
    assert core["successor_sources"] == sources
    assert core["successor_proofs"] == proofs
    assert core["authority_boundary"] == {
        "source_construction_and_different_agent_review_authorized": True,
        "unreviewed_exact_execution_authorized": False,
        "passing_different_agent_closure_authorizes_exact_attempt": True,
        "scope": "one_exclusive_fresh_hierarchical_first_hit_v9_attempt",
        "retry_authorized": False,
        "scientific_retry_authorized": False,
        "v5_v6_v7_or_v8_numeric_state_authorized": False,
        "v8_checkpoint_input_authorized": False,
        "navigation_authorized": False,
        "production_or_promotion_authorized": False,
    }


def test_v9_reviewer_terminal_failure_is_owned_single_use_and_no_retry(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV9(tmp_path / "review-only-v9")
    reservation = operation.claim()
    try:
        for name in ("checkpoint.pt", "result.json", "completed.json"):
            operation.publish_claim_artifact(
                reservation, name, f"owned {name}".encode("ascii")
            )
        operation.publish_derived_artifact(
            reservation, "metric.json", b"owned metric"
        )
        failure = operation.terminate(
            reservation,
            RuntimeError("independent verifier failed"),
            stage="verification",
        )
        assert failure["retry_authorized"] is False
        assert {row["outcome"] for row in failure["artifact_cleanup"]} == {
            "removed_owned"
        }
        assert sorted(path.name for path in operation.attempt.iterdir()) == [
            "failed.json",
            "reservation.json",
        ]
        with pytest.raises(FileExistsError, match="already claimed"):
            SyntheticExecutionV9(operation.root).claim()
    finally:
        operation.close(reservation)
