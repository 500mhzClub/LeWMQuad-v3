# Task-directed acquisition: useful components, unchanged overall success

All28 trials completed and the full audit reproduced6,502 decisions. Every arm
achieves2/4 navigation-task successes. Early scan stopping avoids one collision
and shortens successful tasks substantially; stopped observation recovers one
missed arrival but exposes the next unsafe scan. Neither change, individually
or jointly, makes the local task reliable or separates JEPA from its controls.

## Fixed factorial and matched learned results

The [fixed protocol](go2_task_acquisition_continuation_development_v1_2026-09-05.md)
uses four reused fixtures. A2×2 acquisition-policy factorial shares fixed-forward
movement, followed by three learned methods under the joint policy. Initial
alignment, models, physical crossing/release criteria and all failures remain
unchanged. The new navigation-task metric permits an explicitly interrupted
scan with fresh branch evidence; the original full-scan metric is also retained.

| Controller / acquisition policy | First leg /4 | Task /4 | Contacts /4 | Strict full-scan task /4 |
|---|---:|---:|---:|---:|
| Fixed / baseline |3|2|1|2|
| Fixed / arrival only |4|2|2|2|
| Fixed / scan only |3|2|0|0|
| Fixed / both |4|2|1|0|
| Direct prediction / both |4|2|1|0|
| Supervised recurrent / both |4|2|1|0|
| JEPA recurrent / both |4|2|1|0|

The strict metric correctly remains false for interrupted scans, even on
successful navigation tasks. Across the panel there are15 explicit partial
scans, seven contact trials, five false second-leg arrival candidates and two
missed first-leg arrivals. No sensor-contract or body-stability failure occurs.

Successful fixed-forward tasks take45.4 s with full scanning and23.2 s with
task-directed scanning, a22.2 s reduction on the same two positive fixtures.
The three learned joint-policy arms each take23.3 s. This is a useful acquisition
efficiency result, not improved success probability or a JEPA advantage.

Each learned arm makes56 forward and two forward-right choices. Fixed-forward
makes50 choices under baseline/arrival-only and58 under scan-only/both. All
three learned arms have identical physics. There are only seven exact physical
trajectory groups, sized2/5/4/4/6/2/5. The four fresh baseline traces exactly
reproduce their predecessor's raw physics. Thus28 names do not represent28
independent mazes or random replications; no such confidence interval is valid.

## What the factorial establishes

### Negative corner: avoiding contact does not establish arrival

Baseline and arrival-only contact during the second scan quarter, as before.
Scan-only and both stop at the first acquired view, select its fresh side branch,
hold, align, reobserve and execute the second traversal without contact. All
five such runs reach COMPLETE_PROVISIONAL in the controller, but fail the
unchanged physical arrival criterion.

At the second arrival candidate the base is0.399406 m past the opening plane,
but the articulated body's rear support remains0.016183 m behind it. After
release it is0.022967 m behind. The required full-body margin is+0.020 m.
Actual second-leg progress is1.204381 m, versus command proxy1.193737 m and
runtime required span-plus-margin1.095420 m. A base-center crossing and apparent
visual change are therefore insufficient for the claimed whole-body transition.

This is an observation/arrival-model limitation. The required translation from
an arbitrary arrival pose depends on the opening's position relative to the
robot, not only its body span. There is no observed portal distance in the
current runtime criterion. Increasing a fixed margin to fit this fixture would
not resolve that missing state or establish generalization.

### Negative tee: stopped evidence fixes arrival, then reveals unsafe clearance

Baseline and scan-only stop at FAILED_FIRST_NO_VISUAL_CHANGE despite a physical
first arrival. Arrival-only and both instead brake at the same command-progress
cap, acquire enough fresh visual/quiet evidence and pass the original first-leg
crossing and release checks. This validates stopped observation as the cause
of the recovered arrival within these matched traces.

Those five recovered runs then contact the south wall before completing even
their first quarter-turn view. Early scan stopping has no acquired-view branch
to act on yet. Scan begins at(1.543600,-0.168971) m; maximum observed scan drift
is0.040149 m. Contact occurs at15.908 s global simulation time,14.408 s after
settling, with the native RR_calf rigid group at
(1.406392,-0.600042,0.207050) m and force magnitude78.136 N. Its group name
does not identify an exact primitive because fixed child shapes are merged.

The earlier lack of contact resulted from stopping before this maneuver, not
safe clearance. A successful arrival observation likewise does not prove that
the full articulated robot can turn from the resulting pose.

## Verified evidence and preservation

Collection95127: COMPLETE28, exit0. Full audit49649: PASS28/6,502 decisions,
exit0. Evidence covers350,010 physics/live-fast-gyro samples,34,999 ordinary
sensor samples and6,614 actual RGB packets. The audit reconstructs raw sensors,
native contacts, camera geometry, causal histories, command slew, exact controller/
model decisions, fresh partial-scan events, ledgers and both endpoint reductions.
Only the two declared learned timing fields are excluded from decision equality.

Twenty-three new tests passed, and the full focused suite passed916 tests across
81 files in34.92 s (session74202). Launch binds215 source/test/protocol paths,
176 inputs and two gait bindings. The eight new paths and all inherited evidence
remain frozen. Read-only diagnostic53468 completed exit0 and checked action
counts, actual geometry, repeated traces and exact baseline reproduction. It
did not write outcomes or rerun physics. No study or audit is now running.

Root: `.generated/go2_task_acquisition_continuation_development_v1_attempt_001`.

- Launch: `ed564ec9729d1f724f871438e15a3fdfddafc654afa5092771ef34ab8319eb06`.
- Result: `5ac8a8109ab0c4a3442955891f4a0365ff5a342175fd6bd33b0dbee808c0e135`.
- Full audit: `5b0ab7f848c0432eeb762f0694c65b58ee59323f158585bc3d4199f49e9c37ca`.

## Next priority: whole-task evidence, not another margin sweep

The executor now supports continuous actual observations, provisional arrival,
task-directed branch acquisition and bounded failures. Use these components to
build the [uncertain-memory/beacon-return prototype](go2_whole_task_hypothesis_memory_next_steps_2026-09-05.md),
while retaining the failed1.2 m fixtures as explicit limitations. Provisional
visits must not become certified places or trusted reverse edges. Measure true
return and false associations independently of controller-reported completion.

Arrival and clearance require better observed geometry/state: estimate opening
position and uncertainty relative to the articulated robot, or introduce a
separately declared deployment-realizable range/depth sensor arm. Do not treat
unknown near-field space as free, command distance as measured translation or
simulator wall coordinates as sensing. A first integrated whole-task development
domain may be declared separately, but cannot erase the present narrow-maze
failures or support independent-maze/hardware claims.

Final novel-maze exploration, actual hidden-beacon discovery and remembered
return, robust deployment-valid sensors, matched predictive-training/online-rollout
comparisons and bounded real-Go2 evidence remain unachieved. The scientific goal
is unchanged and remains active.
