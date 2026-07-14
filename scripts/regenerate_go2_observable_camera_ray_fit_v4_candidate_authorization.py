#!/usr/bin/env python3
"""Regenerate the all-false V4 candidate source snapshot after remediation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json"
)
METRIC_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_authorization_2026-07-12.json"
)
REVIEW_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json"
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    authorization = json.loads(AUTHORIZATION_PATH.read_bytes())
    metric_authorization = json.loads(METRIC_AUTHORIZATION_PATH.read_bytes())
    review = json.loads(REVIEW_PATH.read_bytes())
    if (
        authorization.get("status") != "pending_independent_review"
        or any(value is not False for value in authorization["authorization"].values())
        or metric_authorization.get("status") != "pending_independent_review"
        or any(value is not False for value in metric_authorization["licenses"].values())
        or review.get("status") != "pending_second_independent_review"
        or review.get("decision") != "pending"
    ):
        raise PermissionError("V4 candidate regeneration requires all-false authority")

    entries = authorization["source_map"]["entries"]
    for entry in entries:
        entry["sha256"] = sha256((ROOT / entry["path"]).read_bytes())
    authorization["source_map"]["entry_count"] = len(entries)
    authorization["source_map"]["source_map_sha256"] = sha256(
        canonical_bytes(entries)
    )
    review_raw = canonical_bytes(review) + b"\n"
    authorization["review_record"]["file_sha256"] = sha256(review_raw)
    authorization["review_record"]["content_sha256"] = review["content_sha256"]
    core = dict(authorization)
    core.pop("content_sha256", None)
    authorization["content_sha256"] = sha256(canonical_bytes(core))
    payload = canonical_bytes(authorization) + b"\n"
    AUTHORIZATION_PATH.write_bytes(payload)
    print(
        json.dumps(
            {
                "authorization_file_sha256": sha256(payload),
                "authorization_content_sha256": authorization["content_sha256"],
                "source_map_sha256": authorization["source_map"][
                    "source_map_sha256"
                ],
                "entry_count": len(entries),
                "all_authorization_flags_false": True,
                "all_metric_licenses_false": True,
                "review_decision": "pending",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
