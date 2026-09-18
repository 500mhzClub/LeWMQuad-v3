# Auxiliary-only turns during primary depth blindness

Two failures in the fixed multi-seed comparison reached a wall close enough
that all primary optical depth fell below the unchanged 200-mm minimum.
The supervised-seed-1402 and reactive layout-1 runs remained contact-free
but never reached the goal. Their floor tracking and planning intermittently
resumed; the independent obstacle observer still returned unavailable because
it required nonempty depth clouds from both cameras. Neither run requested
any nonzero command after its first floor reacquisition.

The proposed development change supplies current auxiliary obstacle cells
when primary depth is empty and the original current floor fit is accepted.
It permits only hold or pure turns under the existing 45-cm nominal disk,
200-ms observation-age, command-window, prefix, floor-reacquisition and mission
gates. Translation still requires both cameras. A translation veto caused by
primary blindness requests the existing measured-view recovery. Invalid rays
remain unknown. This provides sampled obstacle evidence, without certifying
complete visibility, articulated-body clearance or hardware safety.

Implementation:

- `lewm/auxiliary_only_turn_recovery_development.py`
- `scripts/run_go2_auxiliary_turn_recovery_development.py`
- `scripts/evaluate_go2_auxiliary_turn_recovery_development.py`

Eight focused tests pass. They cover permitted turns, blocked translation,
stale/missing observations, the original disk veto, command expiry, unchanged
paired-camera dispatch, unavailable floor and the new view request. Runtime
method order preserves the existing outer commitment and mission handling.
Both launch annotations preserve the assigned neural model, correction,
footprint, budget and timing settings while identifying the new observer and
dispatch behavior. The observer initializer calls the original configuration
and warmup before replacing only the observer class.

## Saved-state evidence

The first three committed, on-time plans after first floor reacquisition were
probed in each failure. Each used the latest obstacle observation whose worker
had completed by the original dispatch time, with the original accepted plane
and exactly reconstructed noisy public depth. All six dispatches were on time.

All three reactive right turns would pass: auxiliary clouds had 10,918–11,151
sampled returns, 100-ms age and nearest observed cell distances 490–493 mm.
The first two supervised left turns would pass with 10,603/10,547 returns and
490-mm clearance; the third remains unavailable because its current plane
was rejected. Original requests were zero in all six cases. Reports are in
each failed root under `auxiliary_only_turn_guard_probe_v1/result.json`.
These are independent saved-state guard probes, not executed alternate
trajectories; they do not demonstrate physical recovery or navigation success.

A sequential observer replay over the first 820 frames of the supervised
failure completed in 56.448 seconds. All 820 floor/gyro receipts and all 567
originally available obstacle outputs match exactly. The new observer supplies
auxiliary-only evidence on 29 additional frames; both versions remain
unavailable on the other 224 frames. The replay includes the original blindness
and first floor reacquisition. Output: `auxiliary_turn_observer_replay_820_v1/`
in that failed root. No replay remains running.

## Fixed next experiment

Finish all 22 original assignments and their aggregate comparison first.
Then execute two new 4,800-tick layout-1 missions, in this order:

1. Reactive with auxiliary-only turn recovery.
2. Supervised rollout seed 2026091402 with auxiliary-only turn recovery.

Use the original layout-1 CPU group for both, sequentially; evaluate the first
before starting the second. No cache optimization, new fit, sensor threshold,
maze substitution or outcome-dependent replacement is included. The launcher
and evaluator above take `--arm reactive` or
`--arm seed_2026091402_full_supervised_rollout`; the launcher additionally
requires `--layout-index 1`. Every outcome remains separate from the original
22-run comparison and every failed recording remains full.

Report round-trip outcomes, contacts and actual recovery exposure together.
A successful run without auxiliary-only turns does not test the proposed
mechanism. Evidence of recovery requires executed degraded turns, restored
paired obstacle observations and resumed physical navigation. Further unseen
mazes, repeatability and hardware sensing/timing remain outside these two
development follow-ups.


## Both fixed follow-ups complete: neither recovered navigation

The reactive attempt completed its 4,800-tick budget without a goal, with
4,650 published poses, maximum position error 5.254 mm and zero contacts.
It executed 40 auxiliary-only nonzero turn requests; all were physically
applied and obeyed the original turn-only guards. The first occurred at
450.98 s. Paired obstacle observations subsequently appeared on 128 frames,
but no later translation was requested. The last translation had already
occurred at 398.18 s. The first-to-last degraded-request interval had net
physical yaw change -0.09597 rad and XY displacement 18.26 mm; that interval
also includes other requests. These are not isolated turn-response estimates.
There were 859 subsequent floor-reacquisition hold requests and 164 later
joint-plane rejections for inadequate two-axis extent. The run travelled
10.02 m but never came closer than 2.582 m to the goal. Transient paired
observations did not establish sustained physical navigation recovery.
Owner exit was 0 after 11:43.80, with no swaps.

The unchanged supervised-seed-1402 follow-up then failed with tracking queue
overflow after 468 captured frames and 434 published poses. It verified 108
learned XY/yaw plans, no arrivals, zero contacts and maximum pose error
3.003 mm. There were no auxiliary-only requests, so this run does not test
recovery behavior. Late tracking service medians were 109–112 ms against
the 100-ms camera interval; maximum completion age was 3.342 s. Clock-closed
tracking/registration exceptions arose during teardown after queue overflow.
Owner exit was 1 after 1:27.36, with no swaps.

Both attempts were run sequentially on CPUs 8–15,24–31 with the original
tracker. Neither failure replaces an original comparison assignment. All
recordings remain full. There is no recovery or native job still running
from this two-run experiment.

Each root contains `auxiliary_turn_recovery_evaluation_v1.json`, original
physical/treatment evaluations and full data. The reactive root additionally
contains `post_auxiliary_turn_physical_diagnostic_v1.json`; the supervised
root contains `terminal_tracking_queue_diagnostic_v1.json`.

The next priority is tracking throughput. Auxiliary-only turn permission
alone has not demonstrated recovery; the reactive case also requires
sustained acceptable floor support and actual translation resumption.

The four-panel original/follow-up comparison is saved and visually inspected:
`go2_auxiliary_only_turn_recovery_comparison_v1_attempt_001/` contains
`result.json` and `native_navigation_comparison.png/.svg`. Shared settings and
all common original runtime sources match (144 reactive; 159 supervised).
The four trajectories remain near the starting corridor; neither recovery
follow-up supplies goal-reaching evidence.
