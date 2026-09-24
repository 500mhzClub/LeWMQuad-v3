# V4.1 amendments — frozen, approved, execution blocked

The [V4.1 JSON](go2_decision_headroom_protocol_v41_2026-09-23.json) is bound by SHA-256 `e7735ae9d98e6210d9e17f8301b5d6817826ce0b2a87d1971e5488cfa68fe190`. The [approval record](go2_decision_headroom_v41_approval_2026-09-23.json) preserves the user's complete approval text and binds this digest and the successful [configuration diff check](go2_decision_headroom_v41_diff_check_2026-09-23.json). Approval is recorded; it has not been converted into a simulation launch.

## Amended specification

- Continuous regret remains the unzeroed cost difference. The 0.10-s tolerance is for set membership only; the original mean-effect margin remains 0.02 s.
- The primary family is five regret quantities, six paired filter differences against R2, and two R2 levels: 13 quantities, confidence `1 − 0.05/13`. Candidate/state masks and weights are identical on both sides of each difference. Per-source and admitted-but-unsafe rates have a separate descriptive 95% panel.
- Memo conditions A and B and their priority are recorded verbatim in substance, with rate threshold 0.10. B's motion-binding restriction distinguishes observation-only exclusions. Inconclusive regret supports neither stopping nor continuing.
- Layout order is 00, 04, 01, 05, 02, 06, 03, 07, preserving assignment IDs and within-layout source order. Wall time is 72 hours and CPU is 1,152 core-hours. The separate GPU-owner allowance remains 48 hours; every other numerical cap is unchanged.
- The primary reference is unchanged. The labelled secondary reference adds the shortest non-inflated grid path to the nearest inflation-free cell, then uses the existing reference from that cell. Lexicographic ties, grid occupancy, heading convention, safety masks and missingness are explicit in the JSON. Both references report coverage by endpoint inflation stratum crossed with articulated safety.
- Localisation is evaluator-only: believed pose transformed through the captured map rotation and fixed initial physical anchor, compared with native true pose. Fixed position and yaw bins have descriptive 95% panels. The source input and reference target are never corrected using this covariate.
- Historical re-rendering has its separately approved unchanged caps and a separate owner, before Phase 2. Missing restoration inputs are unresolved and do not gate launch. No historical check has yet run.

All original bindings, layouts, assignments, sampling, models, controller implementations, physical definitions and reference weights remain unchanged. The 34 JSON differences are approval/version metadata, the specified amendments, and added implementation bindings. V4 files remain preserved. Implementation additions are not evidence that the complete audit owner passed an execution check.

## One confirmed launch blocker

The [failure record](go2_decision_headroom_v41_writer_blocker_2026-09-23.json) reproduces a frozen V4 output failure on the already-qualified `source_00/state_0132/hold_0` physical trace, using the original `ArticulatedSteps.evaluate` and original `physics.save`.

`ArticulatedSteps.evaluate` returns `complete` as `numpy.bool`. The original writer calls `json.dump` without a NumPy scalar conversion and raises:

```
TypeError: Object of type bool is not JSON serializable
```

The V4 branch owner writes `articulated_v4.json` before computing and retaining that state's filter/reference panel. This is a shared output-path defect, not a state-specific unresolved measurement. Launching as frozen would lose the intended audit panel after spending the branch budget. The numerical clearance computation itself has not been shown incorrect by this failure.

The narrow required correction is lossless conversion of this Boolean to a JSON Boolean at the audit output boundary. No such correction has been applied to the frozen writer or adapter. This is an additional implementation change beyond the requested amendment-only diff, so execution is stopped for a scoped disposition. No new qualification category, physics run, model recomputation, controller repair or trial retry is proposed.

The authoritative handoff's section 7 says: “If a protocol defect is found: stop affected work, document it, preserve the failed version and outputs, propose a successor protocol and affected reruns, and obtain approval for the correction before executing it.” Here no Phase 2 run exists to rerun. The user's V4.1 instruction also requires a diff limited to the listed amendments and preserves everything else in V4.

## Checks and work remaining

The six-state helper check passed localisation schema, paired identity contrasts, binding classification and reference-extension invariants where positional targets existed. The [helper check record](go2_decision_headroom_v41_component_checks_2026-09-23.json) records that earlier implementation snapshot; its source hashes are not a certification of subsequent edits. A broader reader/output check then failed on the NumPy Boolean. The direct writer reproduction isolates the cause. The initial reactive-versus-dense test-harness error is preserved separately. There were zero new physics steps and zero new scientific states; no comparative pilot results were retained.

Remaining work is finite: resolve the output-boundary defect within an explicitly identified successor; complete the permitted six-state output check; run the separate historical check; execute the fixed audit; deliver the audit report, versioned branch panel and single-step decision memo; stop. Approval of V4.1 has already been recorded and need not be requested again for unchanged scope. It does not silently authorize an unrecorded correction to the frozen writer.
