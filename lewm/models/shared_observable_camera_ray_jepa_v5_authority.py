"""Frozen repository authority for Shared JEPA V5 promotion.

Production promotion is intentionally disabled while any required file hash is
``None``.  Enabling it requires a reviewed source change that freezes the exact
repository authority and independently finalized G2/G3 report bytes.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath


# This is the reviewed installation anchor, not a path derived from the
# currently imported module. A copied tree must continue to point here and is
# rejected below because its authority source origin differs.
CANONICAL_REPOSITORY_ROOT = Path(
    "/home/andrewknowles/Workspace/LeWMQuad-v3"
).resolve()
CANONICAL_AUTHORITY_SOURCE_RELATIVE_PATH = PurePosixPath(
    "lewm/models/shared_observable_camera_ray_jepa_v5_authority.py"
)
CANONICAL_AUTHORITY_RELATIVE_PATH = PurePosixPath(
    "docs/lewm_go2_shared_jepa_v5_production_authority.json"
)
CANONICAL_ATTEMPT_REGISTRY_RELATIVE_PATH = PurePosixPath(
    ".generated/go2_shared_jepa_v5/role_global_attempt_registry"
)

# A reviewed authorization revision must replace every pending value together.
CANONICAL_AUTHORITY_FILE_SHA256: str | None = None
CANONICAL_G2_FINAL_REPORT_FILE_SHA256: str | None = None
CANONICAL_G3_FINAL_REPORT_FILE_SHA256: str | None = None
CANONICAL_DATASET_ROLE_MANIFEST_FILE_SHA256: str | None = None
CANONICAL_G2_RUNNER_LEDGER_FILE_SHA256: str | None = None
CANONICAL_G3_RUNNER_LEDGER_FILE_SHA256: str | None = None


def require_frozen_production_authority() -> dict[str, object]:
    expected_source = (
        CANONICAL_REPOSITORY_ROOT / CANONICAL_AUTHORITY_SOURCE_RELATIVE_PATH
    )
    if Path(__file__).resolve() != expected_source:
        raise PermissionError(
            "Shared JEPA V5 authority was imported from a copied or alternate root"
        )
    bindings = {
        "authority_file_sha256": CANONICAL_AUTHORITY_FILE_SHA256,
        "g2_final_report_file_sha256": CANONICAL_G2_FINAL_REPORT_FILE_SHA256,
        "g3_final_report_file_sha256": CANONICAL_G3_FINAL_REPORT_FILE_SHA256,
        "dataset_role_manifest_file_sha256": (
            CANONICAL_DATASET_ROLE_MANIFEST_FILE_SHA256
        ),
        "g2_runner_ledger_file_sha256": CANONICAL_G2_RUNNER_LEDGER_FILE_SHA256,
        "g3_runner_ledger_file_sha256": CANONICAL_G3_RUNNER_LEDGER_FILE_SHA256,
    }
    pending = sorted(name for name, value in bindings.items() if value is None)
    if pending:
        raise PermissionError(
            "Shared JEPA V5 production authority is pending independent artifacts: "
            + ", ".join(pending)
        )
    return {
        "repository_root": CANONICAL_REPOSITORY_ROOT,
        "authority_path": CANONICAL_REPOSITORY_ROOT
        / CANONICAL_AUTHORITY_RELATIVE_PATH,
        "attempt_registry_path": CANONICAL_REPOSITORY_ROOT
        / CANONICAL_ATTEMPT_REGISTRY_RELATIVE_PATH,
        **bindings,
    }


__all__ = [
    "CANONICAL_ATTEMPT_REGISTRY_RELATIVE_PATH",
    "CANONICAL_AUTHORITY_SOURCE_RELATIVE_PATH",
    "CANONICAL_AUTHORITY_FILE_SHA256",
    "CANONICAL_AUTHORITY_RELATIVE_PATH",
    "CANONICAL_DATASET_ROLE_MANIFEST_FILE_SHA256",
    "CANONICAL_G2_FINAL_REPORT_FILE_SHA256",
    "CANONICAL_G2_RUNNER_LEDGER_FILE_SHA256",
    "CANONICAL_G3_FINAL_REPORT_FILE_SHA256",
    "CANONICAL_G3_RUNNER_LEDGER_FILE_SHA256",
    "CANONICAL_REPOSITORY_ROOT",
    "require_frozen_production_authority",
]
