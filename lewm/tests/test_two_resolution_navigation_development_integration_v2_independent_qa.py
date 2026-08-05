"""Independent QA for the frozen standalone navigation integration V2.

The final test records the remaining late-commit atomicity defect.  It is
intentionally a failing review gate until a successor moves controller-record
construction inside the rollback boundary.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import pickle

import pytest

from lewm.benchmarks.go2_physical_claim_observer import (
    empty_evaluator_access_ledger,
)
from lewm.planning import (
    two_resolution_navigation_development_integration_v2 as module,
)
from lewm.planning.two_resolution_navigation_development_integration_v1 import (
    TwoResolutionDevelopmentNavigationIntegrationV1,
)
from lewm.planning.two_resolution_navigation_development_integration_v2 import (
    TwoResolutionNavigationIntegrationV2BindingError,
    TwoResolutionNavigationIntegrationV2ReplayError,
)
from lewm.tests.test_two_resolution_navigation_development_integration_v1 import (
    START,
    TARGET_ID,
)
from lewm.tests.test_two_resolution_navigation_development_integration_v2 import (
    _issue_v2,
    _next_red_outcome,
    _v2_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN_HASHES = {
    "lewm/planning/two_resolution_navigation_development_integration_v2.py": (
        "5a1379ee47b81a5f400b967abf092ca32431d0a19097d880916820d0cc8bd3de"
    ),
    "lewm/tests/test_two_resolution_navigation_development_integration_v2.py": (
        "a18434608a23ceaa58f975c208171258f9932af1afa058b63498448383497cca"
    ),
    "docs/lewm_go2_two_resolution_navigation_development_integration_v2_"
    "handoff_2026-07-13.md": (
        "9e188f5de337a6a821b2b27d879866769a629dffc16068a5a528228340f51008"
    ),
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_qa_freezes_exact_candidate_bytes() -> None:
    assert {
        relative: _file_sha256(ROOT / relative)
        for relative in FROZEN_HASHES
    } == FROZEN_HASHES


def test_independent_qa_exact_lifecycle_and_observer_is_one_way() -> None:
    bundle = _v2_bundle()
    assert not hasattr(bundle.integration_v2, "__dict__")
    for forbidden_alias in (
        "_v1",
        "_outcome_source",
        "_evidence_issuer",
        "_target_memory",
        "_target_router",
        "_waypoint_issuer",
        "_episode_issuer",
        "_issued",
        "_capability",
    ):
        assert not hasattr(bundle.integration_v2, forbidden_alias)

    artifact = _issue_v2(bundle)
    result = bundle.integration_v2.evaluate_observer_only(
        artifact,
        evaluator_access_ledger=empty_evaluator_access_ledger(),
    )

    assert artifact.raw_claim_trace["evaluator_feedback_to_controller"] == []
    assert result.to_dict()["controller_callback"] is None
    assert result.evaluator_access_ledger == empty_evaluator_access_ledger()
    assert artifact.production_promotion_authorized is False
    assert artifact.hardware_execution_authorized is False
    assert result.production_promotion_authorized is False
    assert result.hardware_execution_authorized is False

    for exact_object in (
        bundle.authority_issuer,
        bundle.authority,
        bundle.integration_v2,
        artifact,
        result,
    ):
        with pytest.raises(TypeError, match="non-copyable"):
            copy.copy(exact_object)
        with pytest.raises(TypeError, match="non-copyable"):
            copy.deepcopy(exact_object)
        with pytest.raises(TypeError, match="non-serializable"):
            pickle.dumps(exact_object)

    with pytest.raises(TwoResolutionNavigationIntegrationV2BindingError):
        replace(artifact).assert_integrity()
    with pytest.raises(TwoResolutionNavigationIntegrationV2BindingError):
        replace(result).assert_integrity()
    with pytest.raises(TwoResolutionNavigationIntegrationV2ReplayError):
        bundle.integration_v2.evaluate_observer_only(
            artifact,
            evaluator_access_ledger=empty_evaluator_access_ledger(),
        )
    bundle.integration_v2.consume_observer_evaluation(result)
    with pytest.raises(TwoResolutionNavigationIntegrationV2ReplayError):
        bundle.integration_v2.consume_observer_evaluation(result)


def test_independent_qa_separate_v1_cannot_mutate_shared_owners_unnoticed() -> None:
    bundle = _v2_bundle()
    retained = bundle.integration
    unbound_v1 = TwoResolutionDevelopmentNavigationIntegrationV1(
        projection=bundle.projection,
        planner=bundle.planner,
        physical_view_state_issuer=retained._view_issuer,
        frontier_viewpoint_planner=retained._frontier,
        target_evidence_issuer=retained._evidence_issuer,
        target_memory=bundle.target_memory,
        target_router=bundle.router,
        world_waypoint_issuer=bundle.waypoint_issuer,
        _synthetic_development_fixture=True,
    )
    with pytest.raises(ValueError):
        unbound_v1.issue_controller_claim_trace(
            snapshot=bundle.snapshot,
            component=bundle.component,
            outcome=bundle.outcome,
            start_configuration_cell=START,
            current_yaw_rad=-math.pi / 2.0,
            physical_manifest=bundle.manifest,
            trace_id="independent-qa-v1-trace",
            episode_id="independent-qa-v1-episode",
            event_id="independent-qa-v1-event",
            tick=25,
            event_index=0,
            robot_pose_world_xy_yaw=(16.75, -5.75, -math.pi / 2.0),
            task_object_id="not-a-manifest-task",
        )

    next_outcome = _next_red_outcome(bundle)
    with pytest.raises(
        TwoResolutionNavigationIntegrationV2BindingError,
        match="shared downstream owner state changed",
    ):
        _issue_v2(bundle, outcome=next_outcome)
    assert id(next_outcome) not in bundle.outcome_source._consumed


def test_independent_qa_late_controller_registration_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-transaction registry failure must not strand consumed state."""

    bundle = _v2_bundle()
    state = module._integration_state(bundle.integration_v2)
    before = module._transaction_owner_state_sha256(state)

    def fail_controller_record(**_values: object) -> object:
        raise RuntimeError("independent QA controller-registration fault")

    monkeypatch.setattr(module, "_ControllerRecordV2", fail_controller_record)
    with pytest.raises(
        RuntimeError,
        match="independent QA controller-registration fault",
    ):
        _issue_v2(bundle)

    after = module._transaction_owner_state_sha256(state)
    assert (
        after,
        id(bundle.outcome) in bundle.outcome_source._consumed,
        bundle.integration_v2.issued_controller_count,
        state.owner_state_sha256,
    ) == (before, False, 0, module._owner_state_sha256(state))
