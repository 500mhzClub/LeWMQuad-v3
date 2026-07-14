from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
)
from lewm.tests.n5_full_panel_v1_test_support import verified_test_authority
from scripts import train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as trainer


ROOT = Path(__file__).resolve().parents[2]


def test_cpu_smoke_is_exactly_400_full_panel_updates() -> None:
    smoke = trainer.run_cpu_contract_smoke()
    assert smoke == {
        "schedule_sha256": (
            "62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634"
        ),
        "update_count": 400,
        "frame_exposures": 2000,
        "every_update_is_full_panel": True,
        "losses": {
            "ordered_first_hit_nll": 0.8,
            "target_bin_offset_smooth_l1": 0.02,
            "ground_clear_distance_state_balanced_bce": 0.04,
            "derived_raster_hierarchical_bce": 0.2,
            "total": 0.265,
        },
    }


def test_selected_rgb_mutation_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    monkeypatch.setattr(base, "ROOT", tmp_path)

    frames = []
    for index in range(5):
        path = tmp_path / f"rgb_{index}.bin"
        payload = f"frame-{index}".encode("ascii")
        path.write_bytes(payload)
        frames.append(
            SimpleNamespace(
                rgb_path=path,
                image_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    assert trainer.revalidate_selected_rgb_before_publication(base, frames) == 5
    frames[3].rgb_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        trainer.revalidate_selected_rgb_before_publication(base, frames)

    source = (ROOT / policy.TRAINER_RELATIVE_PATH).read_text()
    run_body = source[source.index("def _run_training") : source.index("def run_exact")]
    assert run_body.index("revalidate_selected_rgb_before_publication") < run_body.index(
        "base._checkpoint_bytes"
    )


def test_reservation_claim_is_transactional_and_post_claim_failures_are_terminal(
    tmp_path: Path,
) -> None:
    authority = verified_test_authority(tmp_path / "review.json")

    before = tmp_path / "before/seed_20260710/n5"
    with pytest.raises(RuntimeError, match="before atomic"):
        trainer._reserve_attempt(
            authority,
            attempt_path=before,
            failure_injection="before_atomic_claim",
        )
    assert not before.exists()
    assert not (before.parent / ".n5.reservation-staging").exists()

    after = tmp_path / "after/seed_20260710/n5"
    with pytest.raises(RuntimeError, match="after atomic"):
        trainer._reserve_attempt(
            authority,
            attempt_path=after,
            failure_injection="after_atomic_claim",
        )
    assert sorted(path.name for path in after.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]
    failed = policy.parse_json((after / "failed.json").read_bytes(), name="failure")
    assert failed["status"] == "failed"
    assert failed["retry_authorized"] is False
    assert failed["partial_artifacts_removed"] is True


def test_reservation_license_schema_is_strict(tmp_path: Path) -> None:
    authority = verified_test_authority(tmp_path / "review.json")
    core = trainer._reservation_core(authority)
    reservation = {
        **core,
        "content_sha256": policy.canonical_json_sha256(core),
    }
    review = policy.source_review_binding(authority)
    policy.validate_reservation_structure(
        reservation,
        expected_source_review=review,
    )
    mutated = copy.deepcopy(reservation)
    mutated["licenses"]["retry_authorized"] = True
    mutated_core = dict(mutated)
    mutated_core.pop("content_sha256")
    mutated["content_sha256"] = policy.canonical_json_sha256(mutated_core)
    with pytest.raises(PermissionError, match="scope/licenses"):
        policy.validate_reservation_structure(
            mutated,
            expected_source_review=review,
        )


def test_trainer_is_import_safe_and_exact_run_requires_isolated_authority() -> None:
    source = (ROOT / policy.TRAINER_RELATIVE_PATH).read_text()
    tree = ast.parse(source)
    top_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not ({"torch", "numpy", "PIL"} & top_imports)
    with pytest.raises(PermissionError, match="requires isolated launcher"):
        trainer.run_exact(object(), rgb_workers=5)  # type: ignore[arg-type]
