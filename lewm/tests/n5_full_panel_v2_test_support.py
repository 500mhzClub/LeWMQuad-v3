from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as policy,
)


ROOT = Path(__file__).resolve().parents[2]


def write_source_review(
    path: Path,
    *,
    reviewer: str = "/root/different_agent",
    corrupt_source: str | None = None,
) -> str:
    sources: dict[str, dict[str, str]] = {}
    for relative in policy.SUCCESSOR_SOURCE_PATHS:
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if relative == corrupt_source:
            digest = "0" * 64
        sources[relative] = {"path": relative, "file_sha256": digest}
    core: dict[str, Any] = policy.expected_source_review_core(
        reviewer=reviewer,
        successor_sources=sources,
    )
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def active_test_authority(
    root: Path,
    *,
    attempt_path: Path | None = None,
) -> tuple[policy.TestAuthorityCapabilityV2, policy.VerifiedAuthorityV2, Path]:
    root.mkdir(parents=True, exist_ok=True)
    attempt = (
        root / "attempts/seed_20260710/n5"
        if attempt_path is None
        else Path(attempt_path)
    ).resolve()
    review_path = root / "review.json"
    digest = write_source_review(review_path)
    capability = policy.create_test_authority_capability(root)
    authority = capability.issue(
        review_path,
        digest,
        target_path=attempt,
    )
    capability.transition(
        authority,
        target_path=attempt,
        from_states=("issued",),
        to_state="active",
    )
    return capability, authority, attempt
