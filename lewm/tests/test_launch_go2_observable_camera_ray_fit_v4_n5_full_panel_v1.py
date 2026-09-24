from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
)
from lewm.tests.n5_full_panel_v1_test_support import write_source_review
from scripts import launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as launcher


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_trigger_records_and_exact_experiment_contract() -> None:
    expected = {
        policy.PREREGISTRATION_RELATIVE_PATH: (
            "0ad13e3897c70f90df6705538f4d86262ec53d3e096618a69563acdf63567c01"
        ),
        policy.TRIGGER_AMENDMENT_RELATIVE_PATH: (
            "1e08aac0ace734d2cbcce9e965b10a7031a94764dd7b47114d38e33944990262"
        ),
        policy.TERMINAL_INVALIDATION_RELATIVE_PATH: (
            "1744a50badd6c9f5c1ef4c8c3cbd05f8c0fc8acff4fbbf066e40e1f7de24f560"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    static = policy.preflight_static_authority()
    assert static["authority_bindings"] == policy.AUTHORITY_BINDINGS
    assert policy.EXPERIMENT == {
        "seed": 20260710,
        "fit_size": 5,
        "fresh_model_initialization": True,
        "model_class": "ObservableCameraRayEvidenceV4Model",
        "optimizer": "AdamW",
        "optimizer_updates": 400,
        "training_batch_size": 5,
        "frame_exposures": 2000,
        "evaluation_batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "precision": "float32",
        "autocast": False,
        "gradient_clip_norm": 1.0,
        "loss_weights": policy.LOSS_WEIGHTS,
        "schedule_algorithm": policy.SCHEDULE_ALGORITHM,
        "schedule_sha256": policy.EXPECTED_SCHEDULE_SHA256,
        "checkpoint_selection": "final_update_only",
        "evaluation_controls": [
            "matched_rgb",
            "wrong_rgb_with_target_calibration",
        ],
        "device": "cuda:0",
        "device_name": "AMD Radeon AI PRO R9700",
        "raphael_igpu_forbidden": True,
        "rgb_worker_count_max": 5,
        "native_threads_per_process": 1,
        "attempt_count": 1,
        "output_path": str(policy.CANONICAL_ATTEMPT_PATH),
    }


def test_review_must_bind_every_source_and_be_by_another_agent(tmp_path: Path) -> None:
    valid = tmp_path / "review.json"
    digest = write_source_review(valid)
    review, _raw = policy.preflight_source_review(
        valid,
        digest,
        canonical_path=valid,
    )
    assert set(review["successor_sources"]) == set(policy.SUCCESSOR_SOURCE_PATHS)

    self_review = tmp_path / "self.json"
    self_digest = write_source_review(
        self_review,
        reviewer=policy.IMPLEMENTATION_AUTHOR,
    )
    with pytest.raises(PermissionError, match="different agent"):
        policy.preflight_source_review(
            self_review,
            self_digest,
            canonical_path=self_review,
        )

    corrupt = tmp_path / "corrupt.json"
    corrupt_digest = write_source_review(
        corrupt,
        corrupt_source=policy.TRAINER_RELATIVE_PATH,
    )
    with pytest.raises(PermissionError, match="source changed"):
        policy.preflight_source_review(
            corrupt,
            corrupt_digest,
            canonical_path=corrupt,
        )


def test_launcher_has_no_scientific_configuration_surface_and_binds_before_import() -> None:
    args = launcher.parse_args(
        [
            "--source-review",
            str(policy.CANONICAL_SOURCE_REVIEW_PATH),
            "--source-review-sha256",
            "a" * 64,
            "--rgb-workers",
            "5",
        ]
    )
    assert args.rgb_workers == 5
    with pytest.raises(SystemExit):
        launcher.parse_args(
            [
                "--source-review",
                str(policy.CANONICAL_SOURCE_REVIEW_PATH),
                "--source-review-sha256",
                "a" * 64,
                "--steps",
                "401",
            ]
        )
    source = (ROOT / policy.LAUNCHER_RELATIVE_PATH).read_text()
    run_body = source[source.index("def _run_authorized") : source.index("def main")]
    assert run_body.index("policy.verify_authority") < run_body.index(
        "from scripts import"
    )
    tree = ast.parse(source)
    top_imports = {
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not ({"torch", "numpy", "PIL"} & top_imports)


def test_successor_namespace_is_additive_and_disjoint_from_predecessor() -> None:
    predecessor = (
        ROOT / ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
    ).resolve()
    assert policy.CANONICAL_OUTPUT_ROOT != predecessor
    assert policy.CANONICAL_ATTEMPT_PATH == (
        policy.CANONICAL_OUTPUT_ROOT / "attempts/seed_20260710/n5"
    )
    assert policy.CANONICAL_METRIC_RECEIPT_PATH.parent == (
        policy.CANONICAL_OUTPUT_ROOT / "metric_verifications"
    )
    assert policy.CANONICAL_GATE_PATH.parent == policy.CANONICAL_OUTPUT_ROOT / "gates"
    assert policy.CANONICAL_SOURCE_REVIEW_PATH.parent == ROOT / "docs"
