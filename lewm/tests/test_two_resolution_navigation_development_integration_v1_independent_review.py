"""Independent adversarial review of the frozen development integration V1.

The failing tests state end-to-end authority and lifecycle properties that the
coordinator must provide before it can be treated as a complete development
navigation chain.  They are review evidence, not amendments to frozen V1.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from lewm.planning import two_resolution_navigation_development_integration_v1 as module
from lewm.planning.two_resolution_navigation_development_integration_v1 import (
    TwoResolutionNavigationIntegrationBindingError,
)
from lewm.tests.test_two_resolution_navigation_development_integration_v1 import (
    CALIBRATION,
    CHECKPOINT,
    RUNNER,
    START,
    TARGET_ID,
    _bundle,
    _hash,
    _issue,
    _manifest,
)


def _issue_with_manifest(bundle: object, manifest: object):
    return bundle.integration.issue_controller_claim_trace(
        snapshot=bundle.snapshot,
        component=bundle.component,
        outcome=bundle.outcome,
        start_configuration_cell=START,
        current_yaw_rad=-1.5707963267948966,
        physical_manifest=manifest,
        trace_id="independent-review-trace",
        episode_id="independent-review-episode",
        event_id="independent-review-claim",
        tick=25,
        event_index=0,
        robot_pose_world_xy_yaw=(16.75, -5.75, -1.5707963267948966),
        task_object_id=TARGET_ID,
    )


def _replace_outcome_target(bundle: object, target_id: str) -> None:
    original = bundle.outcome
    bundle.outcome = bundle.outcome_source.issue(
        snapshot=bundle.snapshot,
        outcome_kind="positive",
        target_id=target_id,
        pose_timestamp_ns=101,
        runner_execution_identity=RUNNER,
        checkpoint_file_sha256=CHECKPOINT,
        raw_outcome_file_sha256=_hash(f"independent-{target_id}-outcome"),
        camera_calibration_sha256=CALIBRATION,
        pose_provenance_sha256=_hash(f"independent-{target_id}-pose"),
        free_physical_cells=original.free_physical_cells,
        unknown_physical_cells=original.unknown_physical_cells,
        target_physical_cells=original.target_physical_cells,
        visible_physical_cells=original.visible_physical_cells,
        physical_probability=original.physical_probability,
        unlocalized_probability=original.unlocalized_probability,
        confidence=original.confidence,
    )


def _issue_with_invalid_task_object(bundle: object) -> None:
    bundle.integration.issue_controller_claim_trace(
        snapshot=bundle.snapshot,
        component=bundle.component,
        outcome=bundle.outcome,
        start_configuration_cell=START,
        current_yaw_rad=-1.5707963267948966,
        physical_manifest=bundle.manifest,
        trace_id="late-failure-trace",
        episode_id="late-failure-episode",
        event_id="late-failure-claim",
        tick=25,
        event_index=0,
        robot_pose_world_xy_yaw=(16.75, -5.75, -1.5707963267948966),
        task_object_id="not-a-manifest-task-object",
    )


def test_controller_trace_registry_rejects_clone_and_nested_rehash() -> None:
    bundle = _bundle()
    artifact = _issue(bundle)

    clone = replace(artifact)
    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="exact live",
    ):
        bundle.integration.assert_controller_claim_trace(clone)

    artifact.raw_claim_trace["trace_id"] = "post-issuance-substitution"
    object.__setattr__(
        artifact,
        "content_sha256",
        module._sha256(artifact.to_dict(False)),
    )
    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="original issuance",
    ):
        bundle.integration.assert_controller_claim_trace(artifact)


def test_g5_semantic_target_requires_exact_task_object_authority() -> None:
    bundle = _bundle()
    _replace_outcome_target(bundle, "blue")

    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="target.*object|object.*target",
    ):
        _issue(bundle)


def test_g3_scene_session_must_match_physical_claim_manifest() -> None:
    bundle = _bundle()
    foreign_manifest = _manifest("foreign-physical-claim-scene")

    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="scene|manifest|session",
    ):
        _issue_with_manifest(bundle, foreign_manifest)


def test_late_claim_validation_does_not_mutate_target_memory() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="canonical bound task set"):
        _issue_with_invalid_task_object(bundle)

    assert (
        bundle.outcome.raw_outcome_content_sha256
        not in bundle.target_memory._seen_raw_outcomes
    )
    with pytest.raises(KeyError):
        bundle.target_memory.snapshot(bundle.outcome.target_id)


def test_late_claim_validation_does_not_strand_single_use_outcome() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="canonical bound task set"):
        _issue_with_invalid_task_object(bundle)

    corrected = _issue(bundle)
    bundle.integration.assert_controller_claim_trace(corrected)


def test_observer_result_rejects_same_object_mutation_and_rehash() -> None:
    from lewm.benchmarks.go2_physical_claim_observer import (
        empty_evaluator_access_ledger,
    )

    bundle = _bundle()
    artifact = _issue(bundle)
    result = bundle.integration.evaluate_observer_only(
        artifact,
        evaluator_access_ledger=empty_evaluator_access_ledger(),
    )
    object.__setattr__(
        result,
        "evaluated_claim_trace",
        {
            "schema": "forged-independent-review-result",
            "physical_claim_summary": {"all_targets_claimed": False},
        },
    )
    object.__setattr__(
        result,
        "content_sha256",
        module._sha256(result.to_dict(False)),
    )

    with pytest.raises(
        TwoResolutionNavigationIntegrationBindingError,
        match="original|issued|exact live",
    ):
        result.assert_integrity()
