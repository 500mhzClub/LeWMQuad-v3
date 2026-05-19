"""Tests for :func:`lewm_worlds.splits.plan_corpus`.

Covers the validation/reroll path that keeps fragmented (route-unreachable)
scenes out of the materialized corpus. Audits showed up to ~26% of seeds
for some families produce a scene with no single free component containing
every beacon; we want the planner to reroll those slots so mass datagen
never lands on a scene the route teacher can't navigate.
"""

from __future__ import annotations

import pytest

from lewm_worlds.scene_validation import audit_scene_reachability
from lewm_worlds.splits import (
    PLAN_VERSION,
    SceneAssignment,
    _scene_seed,
    plan_corpus,
    smoke_corpus_plan,
)


def test_plan_corpus_is_deterministic_without_validation():
    totals = {"train": {"open_obstacle_field": 3}}
    a = plan_corpus(plan_seed=42, totals=totals)
    b = plan_corpus(plan_seed=42, totals=totals)
    assert a.assignments == b.assignments


def test_plan_corpus_salt_zero_matches_legacy_seed():
    # Salt=0 must reproduce the pre-validation seed exactly so existing
    # corpora keep mapping to the same scenes.
    seed_with_salt = _scene_seed(
        plan_seed=7, split="train", family="open_obstacle_field", index=0, salt=0
    )
    seed_legacy = _scene_seed(
        plan_seed=7, split="train", family="open_obstacle_field", index=0
    )
    assert seed_with_salt == seed_legacy


def test_plan_corpus_records_salt_default_zero():
    plan = plan_corpus(plan_seed=0, totals={"train": {"open_obstacle_field": 1}})
    assert plan.assignments[0].scene_seed_salt == 0


def test_plan_corpus_rerolls_when_validator_rejects():
    # First two attempts fail, third one passes — assignment should record
    # salt=2 and use the salt=2 seed.
    rejects: list[int] = []

    def validator(manifest):
        rejects.append(manifest.topology_seed)
        return len(rejects) >= 3  # reject the first two seeds, accept the third

    plan = plan_corpus(
        plan_seed=99,
        totals={"train": {"open_obstacle_field": 1}},
        validate=True,
        validator=validator,
    )
    assignment = plan.assignments[0]
    assert assignment.scene_seed_salt == 2
    expected_seed = _scene_seed(
        plan_seed=99,
        split="train",
        family="open_obstacle_field",
        index=0,
        salt=2,
    )
    assert assignment.scene_seed == expected_seed
    assert len(rejects) == 3


def test_plan_corpus_fails_loudly_when_validator_always_rejects():
    with pytest.raises(RuntimeError, match="reroll attempts"):
        plan_corpus(
            plan_seed=1,
            totals={"train": {"open_obstacle_field": 1}},
            validate=True,
            validator=lambda manifest: False,
        )


def test_smoke_corpus_plan_validated_scenes_are_all_reachable():
    # End-to-end: every scene in the validated smoke plan must pass the
    # reachability audit. This guards against future builder changes that
    # would re-introduce fragmented scenes.
    plan = smoke_corpus_plan(plan_seed=0, validate=True)
    from lewm_worlds.families import build_family_manifest

    for assignment in plan.assignments:
        manifest = build_family_manifest(
            scene_seed=assignment.scene_seed,
            family=assignment.family,
            split=assignment.split,
            difficulty_tier=None,
        )
        report = audit_scene_reachability(
            manifest,
            cell_size_m=0.05,
            inflation_m=0.20,
            standoff_m=0.85,
            spawn_clearance_floor_m=0.20,
        )
        assert report.is_valid, (
            f"validated plan still produced unreachable scene "
            f"{assignment.family}/{assignment.scene_id}: {report.failure_reason}"
        )


def test_navigable_standoff_criterion_flags_alcove_beacon():
    """A beacon whose only LOS standoff is in a sub-navigable corridor
    must fail the audit when ``min_navigable_standoffs_per_beacon`` is
    set, while staying valid under the default (grid-reachability-only)
    audit. This is the gate that keeps local_composite_motifs alcove
    beacons out of the materialized corpus.
    """

    from lewm_worlds.families import build_family_manifest

    # local_composite_motifs index 1 (plan_seed=1, train) was the
    # diagnosed alcove scene; rebuild the salt=0 seed directly.
    from lewm_worlds.splits import _scene_seed

    seed = _scene_seed(
        plan_seed=1, split="train", family="local_composite_motifs", index=1, salt=0
    )
    manifest = build_family_manifest(
        scene_seed=seed, family="local_composite_motifs", split="train", difficulty_tier=None
    )
    default = audit_scene_reachability(manifest)
    strict = audit_scene_reachability(manifest, min_navigable_standoffs_per_beacon=1)
    # The default audit only checks grid reachability; the alcove scene
    # passes it. The navigable criterion rejects it.
    assert default.is_valid
    assert not strict.is_valid
    assert strict.failure_reason == "beacon_lacks_navigable_standoff"
    assert strict.beacons_without_navigable_standoff


def test_default_validator_rerolls_to_navigable_scenes():
    """The default plan validator now enforces the navigable-standoff
    criterion, so a validated local_composite_motifs plan must contain
    only scenes where every beacon has a navigable LOS standoff.
    """

    from lewm_worlds.families import build_family_manifest

    plan = plan_corpus(
        plan_seed=1,
        totals={"train": {"local_composite_motifs": 8}},
        validate=True,
    )
    for assignment in plan.assignments:
        manifest = build_family_manifest(
            scene_seed=assignment.scene_seed,
            family=assignment.family,
            split=assignment.split,
            difficulty_tier=None,
        )
        report = audit_scene_reachability(
            manifest, min_navigable_standoffs_per_beacon=1
        )
        assert report.is_valid, (
            f"validated plan kept a non-navigable scene "
            f"{assignment.family}/{assignment.scene_id}: {report.failure_reason}"
        )


def test_plan_version_unchanged():
    # Plan version tracks the seed-derivation contract. Adding salt does
    # not change the legacy seed (salt=0), so version stays at "1".
    assert PLAN_VERSION == "1"
