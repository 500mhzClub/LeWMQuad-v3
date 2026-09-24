# Go2 G4 frontier/viewpoint source review

Date: 2026-07-13

Status: **PASS for the fail-closed G4 foundation; no G4 output authorized**

## Independent review

The initial candidate deterministically generated revision-bound viewpoint/yaw
options from the complete current configuration-FREE component, routed every
option through the current `ConfigurationPlanner`, conservatively stopped
camera-ground rays at missing, occupied, or first-UNKNOWN support, separated
coverage, entropy, and discovery terms, and rejected stale or mutated
states/candidates. Its focused 10-test suite passed.

The review found one production-authority defect: `record_view` accepted
caller-authored observed-cell sets and a hash-shaped observation identity. A
caller could therefore mark arbitrary registered cells as visually swept and
distort the coverage history and subsequent exploration ranking without a
qualified camera observation.

## Fail-closed remediation

`PhysicalViewStateIssuer.record_view` now rejects every call when the attached
physical memory is `promoted_runtime=True`. This is an intentional intermediate
boundary. Promoted visual history stays unavailable until the qualified learned
camera adapter can issue a current, content-bound view receipt; development-only
geometry and ranking tests remain usable.

Current SHA-256 values:

- `lewm/planning/frontier_viewpoint_information_gain.py`:
  `2ef20e8213a384e0f514705ca14c058eb7fbd81dcc4f6a53407414c1ba79e08e`;
- `lewm/tests/test_frontier_viewpoint_information_gain.py`:
  `02d5a0b0459f6fde43e046b2b9f86d13d21e7392119b57626f0a398ce4c5241e`.

With native numerical threads capped at one and `PYTHONPATH=.:lewm_worlds`, the
current G3 memory, G4 viewpoint, and G5 belief suites passed **86 tests in
28.22 seconds**, including the explicit G3 execution-block regression. The
current dependent test SHA-256 values are
`a60c6f21cb0e40966216428c938e82024eafcaded95a874c53b542befb9065d4`
for G3 and
`33740d7c19127dee18e33eff480b5b51e22016df887ad14d577ea6bc83e78c90`
for G5.

This does not pass G4. Remaining authority requires the qualified camera-view
receipt, reviewed G3 learned projection and exact equivalence, then the
preregistered viewpoint and learned-exploration comparisons. No RGB, model,
dataset, GPU, G2, held-out, sealed, or scene result was opened or created for
this review.
