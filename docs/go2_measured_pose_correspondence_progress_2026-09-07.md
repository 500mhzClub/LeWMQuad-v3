# Previous-measured-pose initializer: implemented and tested

## Terminal update

The [complete result and next decisions](go2_measured_pose_correspondence_result_2026-09-07.md)
now supersede the live preparation notes below. Replay10196 terminated exit0;
independent verification53017 passed all7,433 row joins,475 seed attempts and
27 artifact bindings. The candidate has473 available poses versus7,400 original
poses, with zero recoveries. It is rejected for adoption. Regression87857 passed
3,339 tests with a saved JUnit report. Neither replay nor regression is live.
The original collection supervisor and l03 child remain live, with54/120 saved
prechecks at the latest check; three earlier layouts are fully audited.

The preceding goal turn completed the rotation-only flow experiment and rejected
it for adoption:471/7,433 available poses versus7,400 for the original. This
turn implements the proposed initialization change, tests the previously missed
reference-displacement range, and prepares a fixed complete recorded comparison.
No frozen estimator, collection source, recorded result or acceptance gate is
modified. Reliable local physical execution and the full JEPA-navigation goal
remain unachieved.

## Implementation and synthetic evidence

The new `lewm/measured_pose_seeded_rgbd_development.py` forms each reference-frame
translation seed from its stored reference pose and the last accepted visual
position. The prior must be the immediately preceding100ms observation and
cannot precede its reference. Its time/vector are retained in diagnostics. It
is only an optical-flow initialization: current measured correspondences must
still support the rigid fit. It is not commanded displacement, native truth,
velocity extrapolation, a rigid-fit prior or a current-pose fallback.

The existing depth/flow, rigid-reprojection/fraction/grid/increment, reference
selection/promotion and terminal-failure checks remain. Compared with descriptor
association, correspondence checks still differ, as in the failed rotation-only
method; no equal-reliability or calibrated-uncertainty claim is made.

New multi-frame fixtures render static planes and a finite nearer panel with
nearest positive depth intersections, distinct textures and disocclusion. They
cover depths0.7/2/3.5m, positive/negative translation, combined rotation/translation
and movement through0.48m, including keyframe promotion. All seeds are checked
against previous *estimated* poses, not fixture truth. Other cases cover stale/
future/invalid priors, sensor/clock faults, blank imagery, incorrect prior
correction from new observations, zero-seed exact equivalence and periodic-image
ambiguity. The ambiguous case remains accepted under conditional gates; the
initializer does not solve that lack of information.

- Matcher/runtime82839 completes exit0:21 passed in7.06s.
- Three-method synthetic comparison66687 completes exit0 (1a6370): seven13-frame
  sequences, original91/91 available, rotation-only59/91, measured-pose91/91.
  Rotation-only fails at frames3/7/11/6/6 in five cases. Measured-pose maximum
  error across these sequences is0.214859mm. These are synthetic geometric
  observations, not91 physical navigation successes or independent maze trials.
- Focused13141 completes exit0:80 passed in23.50s, including21 new matcher/runtime
  and16 new replay tests plus43 frozen flow/replay tests. A final source review
  adds before/after SHA validation around reading the frozen comparison result
  and verifies its newly saved output before publishing completion. Focused48255
  then completes exit0:80 passed in23.30s on the final sources.

## Full recorded comparison

`scripts/replay_go2_measured_pose_correspondence_v1.py` runs original and measured-
pose observers on all six inner/intent trajectories,7,433 frames and14,866
frame/arm observations. It reproduces every original pose/selection/failure
against the completed original replay, retains all missing/drain frames, saves
all six sensor streams before native coordinates are parsed, and reports both
available-arm and paired common-frame position/orientation errors and complete
availability categories. Truth cannot enter either observer or its initializer.

The failed rotation-only run is not repeated. Its authenticated complete reports/
evaluations are retained as the third-method baseline after verifying identical
original whole-trajectory populations and first failures. Its old timings are
explicitly not contemporary measurements. There is no per-trajectory method
selection, hidden restart, native reanchor or control integration.

Read-only preflight16180 completes exit0 (c31e50): **696 sources**, complete
predecessor raw/native/output verification, a24,056,574-byte launch definition
under32MiB, and absent new output. The planned786-source matched-study definition
remains `3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.

Reviewed identities:

- Runtime: `9e999c46c9a7e679cfd03e0689a745162adf475519036813706ccb789fbfc565`
- Replay: `9affb7ce6e982a87d1dbd854d08251af9db90128099f4d61e66d5219c1916369`
- Runtime tests: `f203782e435c6e17c52c3a43f98bc18ec84c1e32fda8e32d65ad2861ab07246b`
- Replay tests: `d16b62b5a709ba32275fa087aacd0c3477e31d2b3b656202ca3b5049afe36216`
- Protocol: `a1d7ec90b08748116010b74ba389975f57ac755962a16d77c1072f5685c18b7e`

The earlier full245-file regression45052 no longer has an accessible process
handle, and its terminal result was not retained. It is not counted as passed.
The same explicit245-file inventory was reverified as session87857: terminal
exit0, **3,339 passed in292.87s** (f1888d), with a durable JUnit report at
`.generated/navigation-development-staging.m6MDz1/measured_pose_regression_reverification_v1.xml`.
The report independently parses as3,339 tests with0 errors/failures/skips;
SHA-256 `fc79834c10e487477fad41ee54cd5bb67e3b895971031e09f3c9c6ead79d50d1`.
This repeated software tests, not a scientific experiment. All five reviewed
source identities above remain unchanged. The single fixed replay is now
started as session10196, PID2082389, in the declared environment. Its preflight
was confirmed live before launch metadata was written; do not restart it on
quiet output. Launch metadata subsequently confirms696 sources, SHA-256
`a82c4e75a459896c70ec3f3486c26c00fba321dab55f078bd27ba7f2e9d4a8b6`.
Preserve all696 source bindings. Initial live output reaches inner-left frame1000
with1,001 original poses versus62 measured-pose poses, then starts inner-right.
The candidate first fails at frame62; this is partial negative evidence, not a
completed comparison. A sensor-only check of frames60–62 finds205 tracks at the
failure, after a qualified frame61 consensus fraction124/205 (0.604878) and8/8
grid coverage. The precise rejected consensus component is not recorded; do not
infer it from the composite error or lower a threshold to rescue this run.
Additional independent scene/sensor/error and full latency challenges remain
necessary before considering closed-loop adoption, even if this replay improves.

## Collection and remaining science

The original collector supervisor PID2063013 remains live. Its l02 child exited0
and the supervisor verified its receipt before starting l03, PID2079835.
Read-only verification94374 completed exit0: l00/l01/l02 each have120 eligible
departures, with21/22/22 contact-positive targets respectively. Their exact
launch/audit bindings verify. The l03 launch retains all765 original collection
sources, and the786-source matched-study definition remains unchanged.
Three of12 layouts are complete; these360 collection trials are not successful
navigation missions. Keep the771 supervisor and786 planned-study sources
unchanged. The36-fit matched JEPA/supervised/input-ablation study must wait for
all12 successful receipts. Online predictive benefit, reliable physical local
execution, memory/backtracking, unfamiliar-maze missions, realistic sensing,
real-time operation and bounded hardware evidence remain unfinished.
