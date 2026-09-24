# Independent layout integration checked; original native audit continues

This goal turn made progress: it implemented the new eight-layout native
initializer, collection and raw-audit integration, and numerical round-trip
evaluator without modifying the existing experiment sources. Tests and source
checks passed. No independent native scene was launched. The navigation goal
remains active with 30 completed audited episodes and zero verified round trips;
the 31st episode has completed collection but still awaits its original raw audit.

## Integration evidence

The new session retains the original acquisition, contact, native trace and
renderer-witness method order. Constructor tests reached a substituted scene
builder through the actual wrapper chain for all eight exact new specifications.
They did not construct a scene. Invalid predecessor/modified specifications and
the wrong backend were rejected before scene building. Structural comparisons
confirmed unchanged gait/physics initialization and unchanged collector/auditor
bodies apart from the declared session, layout, status and reporting substitutions.

Synthetic native-shaped traces tested all eight outbound/return routes and
arrival windows. Rejection cases covered contact, commands or speed during
dwell, missing drain, stops, incomplete arrivals, bad clocks or arrival identity,
closed edges, teleports and graphs evaluated under the wrong layout. The
numerical evaluator still returns only a candidate; actual raw sensor/command
replay and strict visibility remain required to verify a physical round trip.

Test session 74636 exited 0: 54 tests passed in 19.78s across
`test_independent_round_trip_native_integration_development.py`,
`test_independent_round_trip_evaluation_development.py` and
`test_independent_round_trip_layouts_development.py`. This is source and synthetic
test evidence, not physical execution.

Source verification 50642 exited 0, binding 1924 sources and reconstructing the
original completed inventory. All 1890 live-queue source bindings remained
unchanged before and after the work. Full identities are retained in
`go2_independent_round_trip_native_integration_source_verification_2026-09-10.json`,
SHA-256 `d00729a36fba6043988acdb3c5f8ebe4eeb72c73414546a2bd07f824d2adae5d`.
The original inventory result remains
`c8021010cd7d22ef7f5a0b057c2c1144dfdfb152dc95895a3066631de2221474`.

The new modules are:

- `scripts/independent_round_trip_session_development.py`
- `scripts/independent_residual_round_trip_episode_development.py`
- `scripts/independent_residual_round_trip_audit_development.py`
- `lewm/independent_round_trip_evaluation_development.py`

No population launcher or model assignment was added in this stage. The new
source protocol is `go2_independent_round_trip_session_evaluation_v1_2026-09-10.md`.
The direct model remains predictive; model RGB removal retains controller RGBD.
This integration alone does not isolate online planning or persistent memory.

## Original pilot observation, pending raw audit

Read-only scan 80407 exited 0 and verified unchanged decision-stream bytes before
and after reading all 2805 observations. Stream SHA-256:
`74ed36be5b09f40ec774cfb04dcf9ed2b8a614b610944df51de83d50afa319a7`.
The first terminal was observation 2794: `SENSOR_OR_MODEL_FAILURE`, with reason
`floor registration exceeds fixed development correction gates`. Its requested
command was zero, current pose was absent, and the last mission receipt was
outbound at frame 2793, observed goal distance 1.3400711274513757m, no arrivals.

Recorded selections were 2278 left turns, 406 left arcs, 89 right turns, 16 holds
and two right arcs. These counts describe the saved controller decisions; they
do not establish physical displacement or successful exploration.

The collection result reports 2804 completed command ticks, 140950 physics
samples, 2805 RGBD observations and 10 terminal zero-command drain ticks. Physical
and acquisition stops are null. Parent 2636286 (creation 1789017984.16) and worker
2637549 (creation 1789018454.4) remain live. Top-level result and failure files are
absent. Keep this episode outside the completed audited count until its original
worker finishes; do not launch a replacement or repeat its native collection.

Original waiter 14784/PID2641948 (creation 1789020925.89) remains live. Its exact
launch SHA-256 is
`6e96b6bc78f08f8dd7b5af3ad25e48b78ca915a5c3b5ede37efe0bc8a7be9b5e`.
The six-case expanded-model maze2 root is absent. This existing waiter owns the
next native launch, after authenticating the original pilot's completed result.

## Next work

1. Preserve and poll original native handle 25801 and waiter 14784. Authenticate
   the native terminal result when it exists, and let the original waiter launch
   its fixed six-case comparison. Retain every scientific failure.
2. Prepare the independent population protocol, exact layout/controller/model
   assignment and resource-admitted launcher using the new integration. Bind
   the completed inventory, source evidence and model admission. Do not consume
   a subset of the eight layouts through outcome-based replacement or tuning.
3. Obtain actual independent navigation and matched reactive/nonpredictive,
   JEPA/planning/memory evidence. Realistic sensing, real-time feasibility and
   bounded hardware evidence remain open. Storage is not the present blocker.
