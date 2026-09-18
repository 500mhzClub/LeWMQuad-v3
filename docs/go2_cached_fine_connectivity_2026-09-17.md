# Exact fine-connectivity cache experiment

The preceding in-place coverage-view trial resolved observed patches but stalled
on deadlines: all 710 plans from frame 1600 onward were late. That full failure
is retained and described in `go2_current_position_coverage_view_2026-09-17.md`.

Reconstructed 400 recorded mapping updates through frame 1596 using delivered
noisy depth and recorded estimator poses. Five sampled maps matched recorded
floor/fine-obstacle counts. The coarse route proposer alone took median 6.42 ms
at frame 1600; it did not explain the whole routing cost. At this point the goal
cell is observed, triggering a fine-clearance connectivity search that fails.
The existing exact edge-distance cache makes that search faster after its first
call, but still repeats the graph traversal and scans all floor cells for an
entry candidate on every plan.

`lewm/cached_fine_connectivity_development.py` caches the graph-search result,
including failure, keyed by exact floor, coarse and fine obstacles, footprint
radius, seed and target. It retains A-star edges, weights and tie-breaking.
Continuous start/goal clearance and the start connector are checked at their
exact current coordinates on every call. A floor spatial index conservatively
shortlists nearby candidates; original scalar distances then determine the exact
1.25-m membership and tuple ordering. No map cell is filled or removed.

Four focused tests passed in 1.90 seconds. They cover new floor reconnecting a
failed graph, obstacle invalidation, current connector rejection, tied routes,
new start/goal/radius, mutable returned paths and near-radius/tie ordering.
Across 120 calls on five reconstructed maps, all outputs matched the existing
implementation except elapsed-time fields. On the stalled frame-1600 graph,
warm median fine-route time fell from 32.91 ms to 2.88 ms in alternating-order
comparisons. Cold searches still need computation; this does not yet establish
live deadlines or navigation improvement. Earlier four maps did not activate
fine-goal search and are unchanged.

Replay/profile: `scripts/profile_go2_coverage_route_development.py` (initial
reconstruction, then `--refinements` and `--cached-refinements`). Comparison:
`scripts/compare_go2_cached_fine_connectivity_development.py`. Receipts live in
the preceding failure root, `routing_profile_v1/` and
`cached_fine_connectivity_comparison_v1.json`. No native state or wall geometry
was used for this replay or optimization. This profiles the routing proposer
and refinement, not the entire historical controller state.

One exposed native mission is assigned with the same supervised world model,
six candidates, viewing/coverage behavior, sensors, 4800-tick budget, CPU group,
300-ms deadline and additional 20-ms delay. The primary outcome remains physical
goal-and-home arrival without disallowed contact. Secondary outcomes include
deadline performance, routing time, queue failure and view resolutions. Preserve
the outcome irrespective of success.

Launcher: `scripts/run_go2_cached_fine_connectivity_development.py` (`--prepare`,
then no arguments, then `--evaluate` after complete archival and owner exit).
Plan: `docs/go2_cached_fine_connectivity_plan_2026-09-17.json`.
Output base: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1`.
Root: `go2_cached_fine_connectivity_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Existing model/training inputs stay on RecoveryStorage. No competing native
simulation or heavy analysis will run alongside this timed experiment.

Native launch acknowledged in session 88594, owner PID 4100060, using CPUs
8–15,24–31. Owner exited 1 and archival completed. Tracking failed after 682
acquired frames; 678 registered poses were retained. No arrivals, no contacts,
maximum registered-position error 3.82 mm. 165/169 plans were on time. The run
ended before the preceding large-map deadline workload, so it does not establish
the cache's live effectiveness. Routing wall time was median 7.57 ms, p95 19.67
ms in this shorter trajectory. Three in-place-view visits resolved their patches.

Readout: `scripts/read_go2_cached_fine_connectivity_development.py`, producing
`cached_fine_connectivity_readout_v1.json` alongside physical, forecast and
coverage receipts. Full failure depth and every other artifact remain retained.

`scripts/replay_go2_cached_connectivity_tracking_failure_development.py`
reproduced all 678 saved raw positions and rotations exactly, including selected
references/modes, then reproduced terminal tracking failure at frame 678. Both
delivered noisy-depth digests were checked at every replayed frame; no native
state entered tracking. The eight active references failed, and diagnostic-only
fits against the six stored views also found insufficient rigid-pose matches.
No failed observer was restarted. The direct previous-frame fallback had zero
qualified primary depth pairs and ten auxiliary pairs, below its unchanged
minimum of twelve. Receipts are in `tracking_failure_replay_v1/`.

The controller repeatedly returned to the same pending frontier view, started
at 16.3 s, unknown cell `[8,0]`, viewpoint `[.925,.025]`, requesting a heading
near pi. Ten weak-view interruptions of that pending visit appear in the saved
planning stream (first frame 344, final frame 668). It resumed that view after
each recovery. On the last episode, weak support triggered at 68.3 s, publication
occurred at 68.466 s, the previous left turn was vetoed at 68.48 s, and the reverse
right-turn request began at 69.0 s. Tracking failed at 69.3 s. These are the
recorded absolute sensor/simulation stamps, including the 1.5-s epoch. Publication,
commitment and actuation delays left only 0.3 s of requested reverse turning.
The final plan was on time; this is distinct from the preceding deadline stall.

Next intervention records a weak-support interruption as a failed viewing
position and searches for an alternative after the measured recovery completes.
It retains tracking acceptance, recovery thresholds, all movement checks and
the unknown status of the target. At the first saved opportunity, frame 344,
the prior frame-340 plan requested this same view. An exact recorded-map replay
matched floor/fine-obstacle counts; the proposed exclusion removed twelve cells
from viewpoint selection only, and found an alternative at `[.975,-.075]`.
Projection remains a hypothesis, not an observation or successful navigation.
Receipt: `interrupted_view_saved_activation_v1.json`; probe:
`scripts/probe_go2_interrupted_view_replan_development.py`.

This is an exposed development experiment, not independent replication,
evidence for JEPA superiority or real-platform validation.
