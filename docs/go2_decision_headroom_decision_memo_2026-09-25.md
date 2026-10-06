# Decision memo — V4.2, 2026-09-25

**Decision: inconclusive; apply the handoff's measurement/coverage/precision-limitation row. Stop the authorized audit at checkpoint (b).**

All 24 fixed source assignments are closed, with one preserved tracking failure. The audit produced 552 qualified states and 5,244 branch attempts. Source inputs, physical restoration and RGB comparisons passed for all audited states. This establishes a usable frozen-model diagnostic panel, not a JEPA advantage.

Primary 99.6154% regret intervals, in seconds:

| Quantity | Exposed (2 contributing layouts) | Initially unexamined (3 contributing layouts) |
| --- | --- | --- |
| G | 0.1050 [-0.5120, 0.7221]; n=2 | 0.1083 [-0.5987, 0.8153]; n=3 |
| H_scorer | 0.1057 [-0.5149, 0.7263]; n=2 | 0.1010 [-0.7825, 0.9845]; n=3 |
| H_motion | -0.0007 [-1.2383, 1.2370]; n=2 | 0.0073 [-0.2428, 0.2574]; n=3 |
| D_old | 0.1218 [-6.7738, 7.0174]; n=2 | 0.2357 [-0.2180, 0.6893]; n=3 |
| D_maze | 0.0960 [-4.9909, 5.1830]; n=2 | 0.1808 [-0.5604, 0.9220]; n=3 |

The practical-effect threshold is δ=0.02 s. None of these intervals supports material superiority, equivalence, or command-history near-optimality. Positive learned-minus-command-history point estimates are not a statistically established JEPA loss. Inconclusive regret supports neither stopping nor continuing visual ego-motion optimisation. Local costs cannot be summed into predicted mission-time savings without additional assumptions.

The filter audit shows substantial exclusion of safe movement even with true motion: R2 excluded-safe levels are 56.08% (exposed) and 40.35% (initially unexamined). Complete-exclusion levels are 40.82% and 22.22%. However, these include observation-only rules. Neither condition A (learned-motion-specific false exclusions) nor condition B (motion-gate conservatism independent of prediction quality) meets its frozen lower-bound test at τ=0.10 in either stratum. The restricted R2 motion-binding intervals are −38.42 to 69.67 percentage points and −125.33 to 147.55 percentage points. Do not choose a gate repair from the unrestricted totals.

The primary reference covers 337/417 positional states (80.82% unweighted); the secondary reference covers 375/417 (89.93%). Several source cells fail the frozen 90% quantity-coverage criterion. Two reactive source cells contain only target-free view-seeking states, and the failed reactive cell removes another complete layout cluster. The secondary reference recovers some coverage but does not rescue the primary inference. All tested bank candidates were safe over the short branch horizon, leaving unsafe-candidate discrimination untested. Degenerate zero-harm t intervals are not proof of zero risk.

**Single recommended next step, requiring separate approval:** prepare one bounded, preregistered revision of the decision-headroom study design that can meet positional-reference coverage and independent-layout precision requirements. Keep the current controller/model comparison fixed while specifying the measurement design; do not start another fitting or gate-tuning cycle.

The proposed preregistered evaluation should fix positional-objective/source quotas, a physically interpretable reference with a declared coverage gate, independent-layout sample size justified against δ=0.02 s, and a valid rare-harm uncertainty rule before new comparative data. Preserve the separation of physical safety, operating margin and controller eligibility. Fix maximum collection/compute/storage budgets and retain an inconclusive outcome when those budgets cannot support the precision target. Any new sample or metric constitutes a separately approved study, not an extension or replacement of this audit. This memo proposes that design work; it does not perform it or choose new quotas, thresholds, layouts or costs.

Historical training rendering remains unverified: the separate 16-example check could not rerender any sample through the qualified path because bound restoration packets were unavailable. L_readout/L_forecast localise arithmetic pathway differences only; they cannot isolate a representation defect from training-provenance or scorer interactions.

The full primary/secondary tables, source-cell denominators, paired masks, descriptive localisation and filter panels, Stage A holds, training-dependency qualifications, asset hashes and resource use are in [the result report](go2_decision_headroom_audit_result_2026-09-25.md). The versioned V4.2 branch panel is retained and bound by [the closeout JSON](go2_decision_headroom_audit_result_2026-09-25.json).

**No recommendation has been implemented. No further experiment is running or authorized by this closure.**
