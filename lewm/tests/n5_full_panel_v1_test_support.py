from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
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
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def verified_test_authority(path: Path) -> policy.VerifiedAuthority:
    digest = write_source_review(path)
    return policy.verify_authority(
        path,
        digest,
        canonical_review_path=path,
        require_unclaimed_output=False,
    )
