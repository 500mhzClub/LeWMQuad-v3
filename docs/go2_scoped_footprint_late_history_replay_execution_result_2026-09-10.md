# Paired late-history performance replay started

The prepared paired replay is running as PID 2754886, creation time
1789077365.71, tool session 5161. It reported
`SCOPED_FOOTPRINT_PAIRED_FULL_INPUT_ADMISSION_STARTED`. Initial input checking
precedes creation of its artifact root; an absent root during that stage is
not evidence that the process stopped. Check the exact process identity in
the execution record before interpreting a quiet tool session.

Before starting, the execution checks authenticated the completed predecessor
tracking replay, confirmed its process had ended, verified the 2,145 prepared
source bindings and the completed profiler's eight artifacts and 1,428
comparison rows, and passed the required available-memory, disk and physical
CPU checks. The completed late-history profile places approximately 79.5% of
its late-window inclusive time in footprint queries, supporting this fixed
comparison of exact query reuse against the frozen-footprint baseline.

The replay compares complete decisions and forecasts on 1,428 observations,
checks retained observed state at seven fixed frames, alternates controller
execution order, and times only controller observation calls. Full original
input admission is repeated after replay. Results must preserve regressions
and the original negative visibility outcome; no speedup or real-time
qualification has yet been established for this candidate.

[Execution record](go2_scoped_footprint_late_history_replay_execution_2026-09-10.json)
has SHA-256
`9e8098e1a28c55850b8ba2fd29b7633c0c129eb4dc83594a9ba09147fcf921a3`.

The current single simulator scene and the frontier, hold, contact and tracking
recovery queue continue under their existing owners. This CPU replay does not
start a simulator scene. Preserve any terminal failure; do not retry or resume
this fixed attempt.
