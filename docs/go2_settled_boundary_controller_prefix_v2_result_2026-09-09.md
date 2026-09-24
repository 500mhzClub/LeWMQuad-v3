# Unchanged settling controller completes its full prospective prefix

Replay17684 completed with exit0 after1990.667471032s. All1867 observations
through1866 were processed from the ninth recorded public sensors. Every
requested command is exact; the complete decisions are exact outside the
declared settling/mission fields and the narrowly witnessed target-reset mode
effect. Model state, source and input checks passed. The original failed V1
attempt remains preserved; this is a fresh replay, not a repaired old result.

Output:go2_settled_boundary_controller_prefix_v2_attempt_001.
Result:146bcb82547e28e24aad784ed69f05cdf4596dcdee7ebf996fd3fe0a14c53a58.
Launch:e36fa13e0c1bfb58868e9d3e5d2d5e4bfac2b5cb9a8808fd45023f790488c7d5.
Compressed decisions:
f76b730368d565f5eac6a8c320458183e018f393696b8a25ddf1564605df9475.
1559 frozen sources; model state
4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.

The first quiet-counter difference is1857. The first mission behavior change
is1866: candidate remains OUTBOUND with eight quiet intervals, no arrival,
hold_required=True and exact zero requested command. Observed goal distance
is0.035493933874m. Both measured interval boundaries are quiet; current observed
interval speed is0.002101815342m/s. These interval averages do not bound
continuous native speed or calibrate pose uncertainty.

The original switches to RETURN and resets its target/planner mode at1866;
the candidate retains its preceding outbound target and WAYPOINT mode. The
declared comparison verifies this effect without permitting unrelated changes.
No later decision or alternative physical outcome was inferred. The unchanged
controller implementation is now eligible for the prepared V2 native preflight
using this actual result hash. A passing preflight admits a new simulation,
not arrival or return success. The full native2ms audit remains necessary.

The separate single-pass measured-bound query candidate was implemented while
waiting: lewm/single_pass_sample_bounds_development.py. It eliminates duplicate
broad-phase enumeration, batches Boolean box comparisons and retains original
scalar sphere-norm arithmetic.11 focused tests passed0.76s, including closed
boundaries, random evolving indices, witness copying, broad-only UNKNOWN and
invalid queries. It has not been timed, integrated or replay-qualified and is
excluded from this settling experiment. No performance or navigation claim
is made for that candidate.
