# Terminal-event coverage implemented and verified on recorded evidence

The full scientific objective remains unachieved. This stage removes an
acquisition-accounting ambiguity before independent-layout learning; it does
not establish JEPA utility, reliable navigation or hardware performance.

## Completed coverage contract and raw compatibility check

Implemented the [terminal-event coverage contract](go2_terminal_event_coverage_contract_v1_2026-09-06.md)
in `lewm/terminal_event_coverage_development.py`, with an integration wrapper
that must run the frozen full raw sensor/contact/command/setup/stop auditor
first. New checks enforce native terminal samples, complete contacts and
attribution, exact requested commands, pre-terminal capture coverage, actual
action departure, and event/motion/image censoring. Population accounting keeps
setup failures, pre-departure stops, infrastructure truncation, invalid recordings
and unattempted cases separate and in the complete planned denominator.

New synthetic tests:31 passed in1.74s (61725). Combined focused tests:116 passed
in4.38s (2399). Full228-file regression:2,927 passed in221.18s (68433). These
test results are engineering evidence, not independent scenes or robot missions.

The recorded checker preflight (63865, exit0) verifies752 source bindings and
1,115 explicit predecessor artifact bindings. Checker59597 completes exit0,
recomputing all eight full raw audits and requiring exact report/prefix identity
with the frozen corrected pilot prechecks. Result:four SCHEDULE_RECORDED and
four PHYSICAL_TERMINAL_RECORDED, with no missing cases. All eight action outcomes
are recorded through their respective endpoints; only four command schedules
complete. Four observed contact events remain four events, not12 independent
collisions because they supply12 positive cumulative-horizon labels.

This post-hoc compatibility result does not reinterpret the original frozen
two-complete/two-partial repeat classification or its failed four-complete gate.
The two original strict junction depth failures remain failures. The pilot still
has no training eligibility or navigation qualification. All predecessor files
are read-only; only new metadata was written to the exclusive external child
`go2_terminal_event_coverage_check_v1_attempt_001`.

Coverage-check launch SHA-256:
`782ce67d34a38b7658b47f6195b1f782c9d73273a9851303c9b08209d4af5a93`.
Coverage-check result SHA-256:
`bf48e54c38e0fe902f2392a1199a87565bf0988c7c48958d077ee6e0c04d6eba`.

## Fixed native measurement assay completed

Implemented [nine fixed foreground-edge views](go2_native_raster_edge_crossing_v1_2026-09-06.md)
and a geometry-selected interior negative control for every view. Offsets cover
both sides at fractions1/4096,1/256,1/4 and1 pixel, plus the exact recorded
base pose. All sampled rays are evaluated; the known failed pixel is not the
only scored location. The negative control adds1cm error to a temporary analysis
copy, never to saved depth or policy observations. The original strict score,
boundary coverage and stable-interior score remain separate.

Focused65838 passes48 tests in2.47s, including15 new pose/analytic/error-control
tests and the frozen footprint/physical-first-surface tests. This focused suite
is additional to, not part of, the228-file full regression above. An initial
shell preflight command had an unmatched parenthesis before import or output;
the corrected preflight13659 completes exit0 with755 source bindings and9
fixed poses. Native assay80704 completes exit0 with one static scene,9 RGB and9
depth renders, zero physics steps and128MiB output cap/40GiB reserve.

All nine stable-interior metrics pass; all nine deliberate1cm interior errors
are rejected. No near-occlusion failure occurs. The base-pose depth image is
bit-identical to the committed pilot frame. The three strict failures at
offsets-1/4096,0,+1/4096 pixels remain explicit; each has one boundary-ambiguous
large residual and no stable-interior bad ray. At-1/256 pixel the tracked ray
and native measurement both select the foreground (~1.528m); at+1/256 they
both select the background (~2.680m). Around zero the centre ray and rasterizer
disagree about the foreground edge. Maximum stable-interior error across the
nine views is0.298775mm. This supports the finite-raster boundary mechanism,
not arbitrary boundary-value correctness.

Independent read-only67640 completes exit0:verifies all30 assay artifacts
(12,668,937 bytes excluding launch/result), recomputes all nine scores and
negative controls, confirms exact base-pose depth, and verifies the eight
coverage-check output bindings. Both frozen source closures verify unchanged.

Native-assay launch SHA-256:
`b19ab3392287f2eda9b3d51e043b14a4454cbd66d07a6cb2eff3863ae61ffb6b`.
Native-assay result SHA-256:
`e138738557fac7b41cd9672a2bd6d0d45842e39981ddbe99a296700d0b1ac263`.

## Next

The [modality-specific readiness decision](go2_independent_rgb_body_collection_readiness_2026-09-06.md)
now separates independent RGB/body prediction collection from depth-navigation
qualification. Source inspection shows depth and shadow tracking are absent
from learner tensors and the prescribed command selector. Preserve raw audits,
hard stable-interior/near-occlusion failures and all planned denominators, but
do not condition new RGB/body data eligibility on strict boundary-only depth
failures or eventual navigation success. Freeze these prospective rules before
the new collection; the current pilot remains ineligible.

The actual dataset materializer's new synthetic depth/shadow-separation test
passes:changing non-policy depth and tracking payloads leaves all learned inputs
and native target tensors identical. Combined12311 passes47 tests in2.33s
(1 dependency test,31 coverage tests and15 edge-assay tests). This is a source-
boundary check, not a new learned-model fit or qualification of corrupted data.

Analytic thin-obstacle/near-plane counterexamples and policy-side uncertainty
remain required for depth navigation, alongside dependable physical execution,
useful online rollout/memory, novel-maze evidence and deployment/hardware work.
