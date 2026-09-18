# Layout 0: contribution of accumulated routing cells

Run the existing `current_planning` mode on independent development layout 0
with `seed_2026091001_full_jepa`, after the queued layout-1 supervised case.
Use `scripts/run_go2_stop_conditioned_independent_case_v1.py` unchanged.
The frozen model state SHA-256 is
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.

The intervention restricts routing queries to floor and occupied cells from
the current paired camera observations. It retains tracking and floor anchors,
persistent contact/geometry evidence, temporal prediction history, observed
residual correction, selector scan state, and mission/settling state. This is
an accumulated-routing-map ablation, not a fully memoryless controller.
The learned predictor, original 100 ms progress / 800 ms contact score,
six-action bank, geometry constraints, sensors, gait, 8,000-decision allowance
and stop-conditioned physical evaluator retain their existing settings.
No retraining or new controller implementation is introduced.

Compare native goal-reaching, return, route crossings, contacts, termination
and decision count against the completed original JEPA reference on layout 0.
Also retain the original sensor/replay checks and timing results. A negative
outcome tests whether the current-view routing variant can complete this
case; it does not by itself establish a general memory benefit. Persistent
evidence retained elsewhere in the controller limits the scope of attribution.

At assignment, the reference JEPA case has a verified round trip; the reactive
case failed to finish and had one strict visibility failure; the supervised
case exhausted its navigation budget without translation, with full replay
audit still running. This is a development ablation chosen with those results
known, not a blind final benchmark. It adds no independent maze replicate.

The case waits for exact predecessor PID 3173601, creation time
1789272867.46, running the layout-1 original supervised queue. Operational
completion, including a negative scientific result, permits the next launch.
An operational failure stops this queue; no automatic retry, overwrite or
replacement is permitted. Native collection and replay audit remain serial.
The existing runner assesses resources at actual launch. Approximately 58 GiB
RAM and 473 GiB artifact disk were free during preparation, which does not
guarantee future resource admission. No extra preflight suite is added.

The queue owner and exact output identity are recorded in
`go2_independent_layout00_current_planning_queue_2026-09-13.json`.
The native case has not started at preparation. Continuous execution,
realistic sensing, broader replication and hardware evidence remain open;
this case still pauses physics during controller computation.
