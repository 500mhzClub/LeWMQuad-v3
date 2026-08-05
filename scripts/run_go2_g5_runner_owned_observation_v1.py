#!/usr/bin/env python3
"""Fail-closed one-shot boundary for runner-owned G5 raw observations.

No production identity is frozen yet, so this process must stop before opening
G3, RGB, checkpoint, observation, or output files. The interface deliberately
contains raw runner inputs only; cell domains, localized target distributions,
and visibility probabilities are derived inside the future captured runner.
"""
from __future__ import annotations

from typing import NoReturn


_CANONICAL_ROOT_TEXT = "/home/andrewknowles/Workspace/LeWMQuad-v3"
_CANONICAL_G5_RAW_OBSERVATION_RELATIVE_PATH = (
    ".generated/go2_g5_runner_owned_observation_v1"
)
CANONICAL_G5_RUNNER_SOURCE_SHA256: str | None = None
CANONICAL_G3_RUNNER_OUTCOME_FILE_SHA256: str | None = None
CANONICAL_G5_RGB_FRAME_FILE_SHA256: str | None = None
CANONICAL_G5_OBSERVATION_HEAD_CHECKPOINT_FILE_SHA256: str | None = None
CANONICAL_G5_EPISODE_AUTHORITY_FILE_SHA256: str | None = None
CANONICAL_G5_PROMOTED_OUTPUT_FILE_SHA256: str | None = None

RAW_OBSERVATION_INTERFACE_SCHEMA = "lewm_go2_g5_runner_owned_raw_observation_v1"
RAW_OBSERVATION_INTERFACE_FIELDS = (
    "schema",
    "scene_id",
    "episode_id",
    "tick",
    "rgb_frame_file_sha256",
    "g3_runner_outcome_file_sha256",
    "pose_receipt_sha256",
    "camera_calibration_sha256",
    "observation_head_checkpoint_file_sha256",
    "runner_source_sha256",
)
FORBIDDEN_CALLER_AUTHORITY_FIELDS = (
    "candidate_domain",
    "localized_distribution",
    "unlocalized_probability",
    "visible_detection_probability",
    "physical_los_contract_sha256",
    "positive_evidence_producer_sha256",
    "negative_visibility_producer_sha256",
)


def require_frozen_production_identities() -> None:
    identities = {
        "runner_source": CANONICAL_G5_RUNNER_SOURCE_SHA256,
        "g3_runner_outcome": CANONICAL_G3_RUNNER_OUTCOME_FILE_SHA256,
        "rgb_frame": CANONICAL_G5_RGB_FRAME_FILE_SHA256,
        "observation_head_checkpoint": (
            CANONICAL_G5_OBSERVATION_HEAD_CHECKPOINT_FILE_SHA256
        ),
        "episode_authority": CANONICAL_G5_EPISODE_AUTHORITY_FILE_SHA256,
        "promoted_output": CANONICAL_G5_PROMOTED_OUTPUT_FILE_SHA256,
    }
    pending = sorted(name for name, value in identities.items() if value is None)
    if pending:
        raise PermissionError(
            "G5 runner-owned observation authority is pending reviewed identities: "
            + ", ".join(pending)
        )


def run_one_shot() -> NoReturn:
    """Fail before any canonical input or promoted-output file is opened."""

    require_frozen_production_identities()

    # Path construction and resolution belong inside the reviewed-identity
    # boundary. With the identities hard-unset, execution cannot reach them.
    from pathlib import Path

    canonical_observation_root = (
        Path(_CANONICAL_ROOT_TEXT).resolve()
        / _CANONICAL_G5_RAW_OBSERVATION_RELATIVE_PATH
    )
    raise PermissionError(
        "G5 production execution remains disabled pending captured-runner implementation: "
        f"{canonical_observation_root}"
    )


def main() -> int:
    run_one_shot()


if __name__ == "__main__":
    raise SystemExit(main())
