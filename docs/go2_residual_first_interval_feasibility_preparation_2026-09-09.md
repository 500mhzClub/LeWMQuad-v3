# First-interval residual feasibility fallback preparation

Status: source component and29 synthetic tests complete. No actual observation
prefix, model inference, native command execution or navigation improvement
has been demonstrated for this successor.

The existing controller applies the preceding eight observed one-step
prediction residuals to100ms waypoint scoring, while feasibility uses the
forecast before that online correction. Maze2 stopped while its current
position remained nominally clear, with all six original first-step forecasts
violating the0.45m nominal radius. The diagnosis and numerical receipts are in
`docs/go2_independent_maze02_feasibility_diagnosis_2026-09-09.md`.

The separate successor activates only when the original waypoint selector
has no action, the current position is still nominally clear, and a nonzero
already observed correction is available. It does not replace feasible
original choices, final-goal selection, view selection or existing reentry.
It subtracts the original score correction from only the first100ms XY point,
queries the existing persistent articulated-surface checker at that corrected
pose, and checks all eight nominal path segments. Correcting the first point
also changes the start of the second segment; points at200–800ms remain fixed.
Original surface vetoes and phase restrictions remain mandatory. Among
eligible actions it uses the original corrected100ms utility, not the incidental
800ms utility recomputed by the path helper.

The receipt validator checks clock alignment, distinct ordered past residual
ticks, availability, original prediction-minus-public-displacement arithmetic,
the exact mean, and exclusion of native outcomes, command-integrated pose or
changed model weights. Original raw path receipts must reconstruct exactly
against the current observed map. Original forecasts, yaw/contact predictions,
scores and veto records remain unchanged, with alternative checks recorded in
`residual_first_interval_feasibility`. In particular, residual memory continues
to remember the original selected prediction, avoiding correction feedback into
its own target. This is an explicit nominal-policy exception, not a calibrated
error bound or physical clearance certificate.

Source SHA-256:

- `lewm/residual_first_interval_feasibility_development.py`:
  `41b9e6801d5d9388680284412b133af4c76bcaf02f86df953982dc6cb7d340fb`.
- `lewm/residual_first_interval_controller_development.py`:
  `316f25c5db917519d07195d0a8cfcdc9a208352dc224fc57ac8385efa069f911`.
- `lewm/tests/test_residual_first_interval_feasibility_development.py`:
  `00b13b43802b5b39fe52f2b9d6befcb98f723fd2a0e335218fccb497f0399280`.

Validation:29 tests passed in1.85s using the existing Genesis environment,
one BLAS/OpenMP thread, deterministic hash seed, no bytecode/cache writes.
The tests use the real scoring, surface-filter orchestration and nominal
geometry helpers, with a synthetic surface query and observed pose. They cover
recovery, all-eight-segment rejection of later collisions, both original and
corrected surface vetoes, phase restrictions, score-horizon preservation,
raw residual targets, inactive policy branches, current-radius violation,
invalid/stale residual evidence and raw receipts, rejection of pre-episode
residual ticks, inactive empty history, and inherited failure
latching. They do not validate the physical footprint implementation anew or
establish learned forecast accuracy. All1658 sources of the completed baseline
cohort were checked unchanged after these additions.

Next: prepare a bound actual-RGBD/model prefix replay from episode start,
compare every prior requested command and retained raw state, stop at the
first changed prospective command, and freeze that evidence before a separate
fresh native attempt. Never infer its physical effect from the original
episode after the first command divergence. Preserve the already prepared
reactive and planning-memory comparison definitions and execution order.
