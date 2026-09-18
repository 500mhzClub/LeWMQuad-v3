# Measured-plane constrained visual registration component

The extended-budget return failed at the accumulated floor-height correction
gate. Its original selected primary pair reconstructs exactly and has only
0.22 mm local plane-offset disagreement. The preceding accumulated correction
was already -49.926 mm. This motivates incorporating the measured floor in the
pose fit before composing successive reference poses.

`lewm/measured_plane_rigid_fit_development.py` implements a separate pure
least-squares fit. Given paired points and validated joint floor planes, it
imposes `R n_current = n_reference` and
`n_reference · t = d_current - d_reference`. It solves the remaining yaw and
tangent translation from every supplied correspondence. Deterministic tangent
bases avoid the antiparallel-normal singularity. Unidentifiable paired yaw is
rejected. The original point-scatter and joint-plane validations remain in use.

The function does not relax the original global floor gate, reset a previous
pose, rewrite a reference/map history, integrate commands, or access native
pose. It is not installed in any controller. It assumes that the two measured
planes describe the same static floor; that assumption is not certified by
plane coherence alone. Raw packet binding, robust consensus, gyro checks,
temporal conflicts, reference ownership and whole-observer admission remain
the integrating caller's responsibility.

Focused tests: **12 passed in 0.21 s**. They cover exact recovery of known
motions, plane-offset sign, global optimality over the feasible yaw family,
coordinate-rotation invariance, determinism, no input mutation, invalid plane
evidence, degenerate geometry, and unidentifiable yaw.

One fixed, posthoc development pair was then evaluated: primary camera,
reference 3836 and terminal observation 3837. The raw images first reconstructed
the original features and original rigid fit exactly. The new fit used the
same 174 original inliers out of 177 lifted matches, with no point removed or
new robust consensus selected.

| Quantity | Original pair | Constrained pair |
| --- | ---: | ---: |
| RMS point residual | 1.335929 mm | 1.339500 mm |
| Original inliers passing original reprojection/residual checks | 174 | 174 |
| Relative gyro disagreement | 0.000278982 rad | 0.000385935 rad |
| Local plane-offset disagreement | -0.220000 mm | numerical zero |
| Local plane-normal disagreement | 0.000191508 rad | numerical zero |

All eight explicitly reported pair checks pass. The translation changes by
0.220634 mm and rotation by 0.000191841 rad. Satisfying plane constraints is a
property of this estimator, not independent evidence of physical accuracy.
This single pair does not repair the preceding accumulated drift or establish
navigation recovery, whole-history pose quality, timing, or generalization.

Probe receipt: `go2_extended_budget_plane_constrained_pair_provisional_2026-09-11.json`,
SHA-256 `3728852d5fca2acaebc27adfa93447b7ba8b6996987bb5207cf7cd1814086dda`.
It binds 2,146 sources, including the component and focused tests, and rechecks
the original raw input identities before and after. Probe session 37074 exited
zero. The original native completion remains pending; no queued experiment
or source-bound controller was modified.

The next implementation must integrate current and retained-reference measured
planes into an independently named observer, keep exact packet/clock identity,
preserve the original temporal conflict and geometric rejection rules, and
recompute its own causal reference history from frame zero. It needs complete
raw-history verification and a prospective closed-loop experiment. Applying
this fit only at the already-failed terminal frame cannot establish recovery.
