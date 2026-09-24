# Supervised baseline: completed collection, negative physical outcome

The unchanged `seed_2026091001_full_supervised_rollout` controller exhausted
its 8,000-navigation-decision allowance on independent layout 0 without an
observed arrival. The closed physical trace also shows no goal-reaching or
maze-edge traversal. Collection completed normally; the existing full sensor
and command replay audit is running. This is a physical readout, not a completed
full-audit result.

The collection resource interval from `begin` to `completed` spans
8,912.97 seconds (148.55 minutes). This excludes the subsequent full audit.

- Terminal reason: `MISSION_TICK_BUDGET_EXHAUSTED`.
- 8,014 recorded decisions, including startup and terminal quiet steps;
  401,400 native physics samples at 2 ms intervals.
- Zero nonzero translation-request samples after initial settling.
- 9,350 physics samples with a nonzero yaw request: the controller sometimes
  requested turning, so the whole run was not a continuous zero-command hold.
- Maximum native displacement from the start: 0.009862 m.
- Closest native distance to the goal: 2.590427 m; final distance: 2.590958 m.
- No arrivals, no maze-edge crossings and zero recorded contact flags.
- No physical or acquisition stop. Terminal native quiet passes, while the
  native round-trip candidate fails.

The existing extended-budget independent physical evaluator produced these
results from the closed collection record and native trace. It does not feed
native geometry or poses back to the controller. Exact numerical results,
assignment, model identity and the two closed input hashes are recorded in
`go2_independent_layout00_supervised_physical_readout_2026-09-13.json`.

The absence of translation requests establishes that this trajectory did not
test whether sustained commanded forward motion could move the robot through
the maze. The recorded decision-200 and decision-3000 score diagnostics show
feasible translating actions losing to hold under the original score. The
already queued contact-horizon intervention remains unchanged and requires its
own prospective navigation result; retrospective rankings do not establish a
successful alternative trajectory.

On the same layout, the original JEPA controller completed its verified return
arrival at decision 3439 and terminal quiet by decision 3449. This is a
within-layout contrast with one training seed. The supervised full audit is
pending, and reliability or a general JEPA advantage cannot be inferred from
this single maze. The unchanged pair is queued on the next inventory layout.
All these simulations pause physics during controller computation; real-time
and hardware validation remain outstanding.

Artifact root:
`go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_supervised_rollout_v1_attempt_001`
under the existing RecoveryStorage navigation development artifact base.
The original evidence and negative outcome are preserved.
