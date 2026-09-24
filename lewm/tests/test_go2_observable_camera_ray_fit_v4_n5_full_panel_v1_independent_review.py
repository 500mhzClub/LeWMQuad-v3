"""Independent adversarial review probes for the frozen full-panel V1 source.

Failing tests are review findings. They define authority and durability
properties required before the exact train-only attempt can be licensed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
)
from lewm.tests.n5_full_panel_v1_test_support import verified_test_authority
from scripts import train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as trainer


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCES = {
    policy.POLICY_RELATIVE_PATH: (
        "875edc86efbe25d246b24c2ef2467cc7956b1b3bb90e6d8d1e03e4a9c5b11d88"
    ),
    policy.LAUNCHER_RELATIVE_PATH: (
        "3cb9ff782a15bc97dd3cca2cc25705e006d6af19a7dbef6d27dee893d9b570c8"
    ),
    policy.TRAINER_RELATIVE_PATH: (
        "48ac856c080906a8d73d5a9b97d1dcf7fe21f5bc99217cce669c43b9c091acca"
    ),
    policy.VERIFIER_RELATIVE_PATH: (
        "00c62cec39e1eb05bf23a96a9153aa8ff350235c2e5c6662f6148934ab9d85b0"
    ),
    policy.FINALIZER_RELATIVE_PATH: (
        "1d4471381a6c3b29f0b077e44e3126f956281ff105d4e38aa8e0f6ba18675b8b"
    ),
}


def test_frozen_source_schedule_and_prepublication_rehash_controls() -> None:
    for relative, expected in EXPECTED_SOURCES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    smoke = trainer.run_cpu_contract_smoke()
    assert smoke["schedule_sha256"] == policy.EXPECTED_SCHEDULE_SHA256
    assert smoke["update_count"] == 400
    assert smoke["frame_exposures"] == 2000
    assert smoke["every_update_is_full_panel"] is True

    source = (ROOT / policy.TRAINER_RELATIVE_PATH).read_text()
    run = source[source.index("def _run_training") : source.index("def run_exact")]
    assert run.index("revalidate_selected_rgb_before_publication") < run.index(
        "base._checkpoint_bytes"
    )
    assert run.index("base._checkpoint_bytes") < run.index("_publish_success")


def test_importable_global_marker_cannot_forge_verified_authority() -> None:
    forged = policy.VerifiedAuthority(
        static={},
        source_review={},
        source_review_file_sha256="0" * 64,
        source_review_content_sha256="1" * 64,
        _marker=policy._AUTHORITY_MARKER,
    )

    with pytest.raises(PermissionError, match="verified authority|forged"):
        policy.require_verified_authority(forged)


def test_verified_authority_clone_is_rejected(tmp_path: Path) -> None:
    authority = verified_test_authority(tmp_path / "review.json")
    cloned = replace(authority)

    with pytest.raises(PermissionError, match="exact live|copy|replay"):
        policy.require_verified_authority(cloned)


def test_verified_authority_cannot_reserve_more_than_one_attempt(
    tmp_path: Path,
) -> None:
    authority = verified_test_authority(tmp_path / "review.json")
    first = tmp_path / "first/seed_20260710/n5"
    second = tmp_path / "second/seed_20260710/n5"
    trainer._reserve_attempt(authority, attempt_path=first)
    with pytest.raises(PermissionError, match="consumed|replay|one attempt"):
        trainer._reserve_attempt(authority, attempt_path=second)


def test_stale_preclaim_staging_cannot_strand_the_sole_attempt(
    tmp_path: Path,
) -> None:
    authority = verified_test_authority(tmp_path / "review.json")
    attempt = tmp_path / "attempts/seed_20260710/n5"
    staging = attempt.parent / ".n5.reservation-staging"
    staging.mkdir(parents=True)
    (staging / "interrupted-write").write_bytes(b"pre-rename process death")

    try:
        trainer._reserve_attempt(authority, attempt_path=attempt)
    except FileExistsError:
        pass

    assert attempt.is_dir(), (
        "an unclaimed stale staging directory permanently stranded the sole "
        "attempt without a canonical reservation or terminal receipt"
    )


def test_post_rename_failure_durably_fsyncs_seed_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = verified_test_authority(tmp_path / "review.json")
    attempt = tmp_path / "attempts/seed_20260710/n5"
    seed_root = attempt.parent
    fsynced: list[Path] = []
    original = trainer._fsync_directory

    def recording_fsync(path: Path) -> None:
        fsynced.append(Path(path))
        original(path)

    monkeypatch.setattr(trainer, "_fsync_directory", recording_fsync)
    with pytest.raises(RuntimeError, match="after atomic"):
        trainer._reserve_attempt(
            authority,
            attempt_path=attempt,
            failure_injection="after_atomic_claim",
        )

    assert seed_root in fsynced, (
        "the renamed terminal attempt was not made durable in its parent directory"
    )
    assert sorted(path.name for path in attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]
