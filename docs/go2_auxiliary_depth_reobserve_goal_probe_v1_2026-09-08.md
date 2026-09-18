# Bounded active reobservation native goal probe V1

Run the same fixed corrected seed-2026091001 JEPA/direct pair on reused
development layout `family_episode_039`, in unchanged seed-2026091501 order.
Require full correction admission, its completed readout, the previous paired-
camera native probe and readout, and two exact fresh bounded reobservation
replays per model with no controller failures and at least one proposed recovery.

Change only valid no-feasible-action termination: request zero and continue
observing both cameras, updating the original map and reevaluating the original
learned action bank for up to ten consecutive wait commands. A feasible action
resets the counter. The eleventh infeasible observation remains terminal. Never
resume a latched terminal state or sensor/model failure, and preserve view
exhaustion, arrival, mission budget and physical stops. Waiting consumes the
original 240 navigation ticks. Keep the separate original ten-command terminal
drain. A zero wait is an active phase-2 command, audited with the original
requested/applied/slew and phase checks. It is not a clearance certificate.

Keep both robot-visible camera streams, exact frame pairing, valid masks,
auxiliary all-return retention, observer, model input contract, learned weights,
training-only XY corrections, route/view logic, eight-step planning and every
surface and 0.45-m nominal constraint unchanged. The model receives no native
pose, segmentation, geometry labels or unexecuted target. Preserve the original
observed 0.04-m arrival and native 0.06-m one-second quiet/contact goal gate.

Freeze complete source and artifact identities before exclusive root
`go2_auxiliary_depth_reobserve_goal_probe_v1_attempt_001`. Use two independent
CPU scenes with one numerical thread each after rechecking hardware, 32-GiB
available RAM and 8-GiB storage allowance above the 40-GiB reserve. The preceding
paired-camera two-process run and passing four-process native benchmark establish
the workload basis. Monitor live resources and full iteration timing.

Require exact original first-four paired public frames and first-900 native
samples. Replay every decision with a fresh assigned corrected model and retain
the unchanged native, actuator, setup, stop and both-camera auditors. Require
all measurement gates for any counted goal success. Reverify model state and
all source, fit, correction and raw artifact bindings afterward. Preserve both
outcomes and every failure without retry or resume.

This is a prospective development integration probe, not an independent maze
evaluation. Recorded shadow recovery is not native recovery. Physics remains
paused during computation. Verified arrivals, independent novel mazes, actual
backtracking, matched reactive/nonpredictive and planning/memory comparisons,
realistic sensing/timing and bounded hardware evidence remain required.
