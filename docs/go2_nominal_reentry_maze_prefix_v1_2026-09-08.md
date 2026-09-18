# Explicit nominal-clearance reentry prefix V1

The first native maze failed its original nominal constraint after crossing two
valid maze edges. All measurement gates passed; no arrival occurred. Its exact
native result is `5874c1fea08b40d90e676b69acb7d3a103d96a6281e0911801bf6bd6e1ab1570`
and readout is `fe2b0fbdadba8f1ae29e67d8897d2ea3bdfe36ec5caa8e4a4a4200a349087bc4`.
Both remain unchanged and unsuccessful.

## Explicit prospective policy exception

`lewm/nominal_clearance_reentry_development.py` is a new experimental rule for
an already violated nominal radius. It activates only when the original checked
forecast bank has no feasible action, the view budget is not exhausted, and the
current observed point has positive clearance but fails the existing 0.45 m
nominal test. It cannot override an already feasible original selection.

Using the actual current mapper's full occupied-cell set and current public
observed pose, reconstruct all eight original segment checks exactly. A recovery
candidate must request a nonzero command, remain in the original phase allowance,
pass its unchanged first-step surface check, have every forecast segment at least
as far from occupied cells as the current starting clearance, and predict a
strictly positive first-100-ms clearance gain. The original 1e-12 numerical
allowance is used; there is no fitted margin or new global radius. Among eligible
candidates choose first-step clearance gain minus the original 1.2-weighted
800-ms contact score. Original raw forecasts, candidate utilities, phase, surface
and 0.45 m path-veto evidence remain intact; recovery scores and checks are stored
separately. Holds cannot reset the infeasibility wait through this exception.

The rule allows an experimental command despite the original nominal veto; it
does not relabel that veto as passed. It does not guarantee physical clearance,
monotonic actual motion, model accuracy, reentry or navigation. The learned
forecasts are still uncalibrated. No native pose/outcome or command-integrated
pose enters it. The `NominalReentryRoundTripController` changes only its selector
and identity; observation, history, residual update, mission/return state, global
3,000-tick budget, ten-command wait, arrival dwells and latched stops are inherited.

Eight tests passed: positive nonzero reentry with retained original evidence;
later-path worsening, surface, phase and zero-progress rejection; activation
only for a current nominal violation; changed-map/forged-check/native-input
rejection; and inherited mission/observer/failure behavior. These are source
checks, not executed recovery evidence.

## Recorded-prefix check

Run `scripts/replay_go2_nominal_reentry_maze_prefix_v1.py` once at the exclusive
`go2_nominal_reentry_maze_prefix_v1_attempt_001` root. Reconstruct the complete
actual primary/auxiliary public sensor prefix with one fresh unchanged JEPA
model, stopping at the first changed requested command or terminal. Require
exact equality of every original decision field except controller identity
before intervention. At the intervention observation, require identical causal
observer/map/mission evidence, raw forecasts and original constraint evidence,
and the exact new pure reentry transformation. No later recorded observation
may label the new unexecuted action's outcome.

Use one deterministic CPU replay worker. Admit at least 8 GiB available RAM and
256 MiB above the original 40 GiB reserve; record hardware before/after and all
source/input/output identities. Preserve any exclusive-root failure. The replay
does not execute physics, modify weights, retry the old attempt, certify native
recovery, select a new maze, or establish a matched-baseline advantage.
