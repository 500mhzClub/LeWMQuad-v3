# Replan interrupted camera viewpoints

The cached-connectivity trial lost tracking after repeatedly retrying a frontier
view that triggered weak-feature recovery. Exact sensor replay and the first
saved intervention opportunity are documented in
`go2_cached_fine_connectivity_2026-09-17.md`.

This experiment records a previously requested camera viewing position as
unsuccessful when measured visual recovery interrupts it within the existing
0.10-m viewpoint tolerance. It excludes that local viewing position using the
existing alternate-view search. It changes no floor/obstacle cell and does not
declare the requested patch observed. The measured recovery heading remains
active. A single recovery episode can retire its prior requested view once;
it cannot consume unexecuted alternatives while recovery continues.

Six focused tests passed in 1.71 seconds. They cover exclusion without map
mutation, distant approach, current-position exclusions, recovery-objective
preservation, repeated recovery and views without a prior heading request.
The recorded frame-344 probe found a different projected viewpoint after
excluding the unsuccessful one. This proves activation on a saved state, not
alternative navigation or future visibility.

Run one exposed-maze mission using the same model, six actions, sensors,
cached routing, coverage guard, tracking thresholds, timing, CPU group and
4800-tick budget. All movement checks stay fixed. Primary outcome is physical
goal-and-home arrival with no disallowed contact. Retain this attempt regardless
of its result; assess view exclusions/observations, tracking and deadlines after
owner exit and full archival. No competing simulation or heavy analysis runs
alongside the timed trial.

Launcher: `scripts/run_go2_interrupted_view_replan_development.py` (`--prepare`,
then no arguments, then `--evaluate` after exit).
Plan: `docs/go2_interrupted_view_replan_plan_2026-09-17.json`.
Output base: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1`.
Root: `go2_interrupted_view_replan_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Existing model/training inputs remain at their original locations.

Native launch acknowledged in session 72824, owner PID 4102492, CPUs
8–15,24–31. Owner exited zero; archival completed before evaluation.

**Verified round trip in 306.14 simulated seconds, with zero disallowed contacts.**
Goal arrival was frame 2043 and home arrival frame 3058. Across the respective
one-second quiet dwells, maximum physical distances were 21.13 mm and 14.35 mm;
maximum 100-ms speeds were 9.13 and 10.00 mm/s, and every requested command was
zero. The return physically reversed all eleven outbound corridor edges, with
no invalid graph transitions or samples outside the known evaluation grid.

Tracking supplied all 3060 registered poses; maximum position error was
10.77 mm. There were no pipeline faults. 674/747 plans were on time (90.2%).
Routing wall time was median 23.48 ms, p95 134.50 ms, maximum 641.61 ms; occasional
late plans remain. The measured simulation loop took 401.49 wall seconds, before
archival; this is not real-time qualification. Posthoc physical wall clearance
at planning samples stayed at least 0.49320 m, above the 0.45-m circular margin.
That margin is not articulated-body safety certification.

Seven in-place coverage visits completed through actual patch observation;
the translation filter rejected fourteen proposals. Frontier visits recorded
twenty observed-patch completions and one unavailable-route completion.
**The interrupted-view rule activated zero times.** Therefore this success
does not establish that the new rule prevents the preceding tracking failure.
The asynchronous trajectory differed before the old repeated-view episode.
This establishes one complete exposed-maze result for the combined controller,
not causal efficacy of each component or repeatability.

Readout: `scripts/read_go2_interrupted_view_replan_development.py`.
Receipts: `interrupted_view_replan_readout_v1.json`,
`continuous_native_arrival_evaluation.json`,
`physical_return_corridor_readout_v1.json`, and the coverage/forecast evaluations
in the root. Trajectory figure: `physical_round_trip_v1.png` / `.svg`, produced
by `scripts/plot_go2_interrupted_view_round_trip_development.py`.

Keep this full sensor recording as the current successful debugging reference;
preserve every preceding failure. Before a large fresh-layout comparison,
check repeatability and transfer back to the frozen JEPA model with a small,
fixed-order batch. Do not keep rerunning until a favorable result appears.
Any later comparison must retain failed repetitions and distinguish exposed
layout development from prospective independent layouts.

No independent-layout reliability, JEPA advantage, real-time or real-platform
validation follows from this exposed development experiment. Multi-seed,
contribution and realistic-sensing/timing work remains within the broader goal.
