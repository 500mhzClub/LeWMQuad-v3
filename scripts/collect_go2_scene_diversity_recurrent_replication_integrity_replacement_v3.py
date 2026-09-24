#!/usr/bin/env python3
"""Run the V2 collector unchanged under a fresh V3 output identity.

V3 is an identity-only, final transient-infrastructure replacement.  This
module changes schemas, attempt identity and the worker script entry point so
fresh subprocesses receive that identity.  Collection order, worker lifetime,
policy, resource barriers, data, rendering, join and validation all remain the
reviewed V2 implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as predecessor  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_"
    "INTEGRITY_REPLACEMENT_V3"
)
ATTEMPT_ID = "go2-scene-diversity-recurrent-replication-integrity-replacement-v3"
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "collection_reservation_v1"
)
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "scene_physics_result_v1"
)
SCENE_EVIDENCE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
    "scene_process_evidence_v1"
)

# Exact aliases: V3 introduces no policy, cap, count, seed, audit or scientific
# behavior of its own.
AUTHORITY_FIELDS = predecessor.AUTHORITY_FIELDS
EXPECTED_CAPS = predecessor.EXPECTED_CAPS
EXPECTED_COUNTS = predecessor.EXPECTED_COUNTS
EXPECTED_HISTORY_PANEL = predecessor.EXPECTED_HISTORY_PANEL
EXPECTED_PERMISSIONS = predecessor.EXPECTED_PERMISSIONS
PLAN_FIRST_EFFECTIVE_GENESIS_SEED = predecessor.PLAN_FIRST_EFFECTIVE_GENESIS_SEED
PLAN_FIRST_PHYSICS_SEED = predecessor.PLAN_FIRST_PHYSICS_SEED
PROCESS_RESET_EQUIVALENCE_AUDIT_V2 = predecessor.PROCESS_RESET_EQUIVALENCE_AUDIT_V2
PROCESS_RESET_EQUIVALENCE_AUDIT_V3 = PROCESS_RESET_EQUIVALENCE_AUDIT_V2
ROLE_ORDER = predecessor.ROLE_ORDER
SCENE_COUNT = predecessor.SCENE_COUNT
SCENE_EVIDENCE_STATUS = predecessor.SCENE_EVIDENCE_STATUS
SceneProcessCollectionError = predecessor.SceneProcessCollectionError
pilot = predecessor.pilot
bounded = predecessor.bounded
_validate_scene_diversity_plan_v1 = predecessor._validate_scene_diversity_plan_v1

_ORIGINAL_WORKER_ARGV_V2 = predecessor._worker_argv_v2  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


def _worker_argv_v3(**kwargs: Any) -> list[str]:
    """Preserve the exact V2 argv while selecting this identity wrapper."""

    argv = _ORIGINAL_WORKER_ARGV_V2(**kwargs)
    if len(argv) < 2 or Path(argv[1]).resolve() != Path(predecessor.__file__).resolve():
        raise SceneProcessCollectionError("V2 worker entry point changed")
    argv[1] = str(Path(__file__).resolve())
    return argv


def _configuration_overrides_v3() -> dict[str, object]:
    return {
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": AUTHORITY_STATUS,
        "ATTEMPT_ID": ATTEMPT_ID,
        "RESERVATION_SCHEMA": RESERVATION_SCHEMA,
        "SCENE_RESULT_SCHEMA": SCENE_RESULT_SCHEMA,
        "SCENE_EVIDENCE_SCHEMA": SCENE_EVIDENCE_SCHEMA,
        "_worker_argv_v2": _worker_argv_v3,
    }


@contextmanager
def _configured_predecessor_collector_v3() -> Iterator[None]:
    """Apply and restore only V3 output identity and worker entry point."""

    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v3()
        originals = {name: getattr(predecessor, name) for name in overrides}
        try:
            for name, value in overrides.items():
                setattr(predecessor, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(predecessor, name, value)


def load_and_validate_v3(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_v3():
        return predecessor.load_and_validate_v2(*args, **kwargs)


load_and_validate_replacement_v3 = load_and_validate_v3
load_and_validate_replacement_v2 = load_and_validate_v3
load_and_validate_v2 = load_and_validate_v3


def validate_scene_process_evidence_v3(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_v3():
        return predecessor.validate_scene_process_evidence_v2(*args, **kwargs)


validate_scene_process_evidence_v2 = validate_scene_process_evidence_v3


def validate_scene_process_closure_v3(*args: Any, **kwargs: Any):
    with _configured_predecessor_collector_v3():
        return predecessor.validate_scene_process_closure_v2(*args, **kwargs)


validate_scene_process_closure_v2 = validate_scene_process_closure_v3


def collect_v3(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
):
    with _configured_predecessor_collector_v3():
        return predecessor.collect_v2(
            plan_path=plan_path,
            expected_plan_byte_count=expected_plan_byte_count,
            expected_plan_sha256=expected_plan_sha256,
            authority_path=authority_path,
            expected_authority_byte_count=expected_authority_byte_count,
            expected_authority_sha256=expected_authority_sha256,
        )


collect_v2 = collect_v3
collect_v1 = collect_v3


def build_parser():
    return predecessor.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_collector_v3():
        return predecessor.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_ID",
    "AUTHORITY_FIELDS",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "EXPECTED_CAPS",
    "EXPECTED_COUNTS",
    "EXPECTED_HISTORY_PANEL",
    "EXPECTED_PERMISSIONS",
    "PLAN_FIRST_EFFECTIVE_GENESIS_SEED",
    "PLAN_FIRST_PHYSICS_SEED",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_V2",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_V3",
    "ROLE_ORDER",
    "SCENE_COUNT",
    "SCENE_EVIDENCE_SCHEMA",
    "SCENE_EVIDENCE_STATUS",
    "SceneProcessCollectionError",
    "collect_v1",
    "collect_v2",
    "collect_v3",
    "load_and_validate_v2",
    "load_and_validate_v3",
    "validate_scene_process_closure_v2",
    "validate_scene_process_closure_v3",
    "validate_scene_process_evidence_v2",
    "validate_scene_process_evidence_v3",
]
