# Fixed tracking interventions: prospective development preparation

These interventions are specified before collecting or exposing the new
eight-trial challenge. They do not change production observer sources, matcher
gates, the ten-frame bridge budget, or existing experiment results. This is a
sensor-stream component, not a native launcher or completed challenge.

The fixed onset is observation84, within the first commanded turn in the planned
442-command tape. Actual turning is not assumed: later native evaluation must
check motion coverage. An early stop or prior observer failure may prevent an
intervention from being exercised; report this explicitly, not as a stress pass.

All scenarios use fresh paired original/temporal observers and all recorded
frames, including frames after terminal failure. Both arms receive private
copies of identically intervened sensor packets. The original and transformed
packet bytes/metadata are separately hashed in each returned row. No native
truth, controller command integration or analytic scene queries enter the
observers. No final benchmark material is involved.

## Fixed scenarios

- Nominal: byte-identical packets, reference hook inactive.
- Retained-anchor absence: deny retained-reference candidate calls for1,10 or11
  consecutive frames beginning84. A distinct immediately previous observation
  is still available to the temporal observer. This is an explicit reference
  access intervention, **not simulated optical occlusion** or a recovery rate.
- Current RGB unavailable: frame84 image arrival is one nanosecond after the
  decision. It must not be used or replaced by an old/current fabricated image.
- Current depth unavailable: frame84 has no valid rays and zero unknown values.
- Current gyro unavailable: samples measured in the100ms interval starting at
  frame84 are invalid in both fast and slow histories, including later overlap.
- Repeated appearance: from84, replace RGB with a20-pixel black/white checker,
  shifting three columns per frame; retain recorded depth and rebind RGB identity.
  This is artificial image corruption, not a physically consistent new scene.
- Correlated depth drift: from84 add2mm per frame, capped at40mm; out-of-range
  rays become unknown, not clipped valid returns. Sensor-error counterexample,
  not a measured drift distribution or a registration-threshold search.
- Shared gyro bias: add0.02rad/s about body z to samples measured from frame84
  onward in both fast and slow channels. Timestamp-based transformation preserves
  shared samples across overlapping histories; no retroactive sample rewriting.
- Qualified incremental contradiction: on84 add3cm along initial-body y to a
  successfully measured previous-frame candidate, before the existing consistency
  decision. This deliberately tests fault rejection, not a physically induced
  correspondence failure. The original observer has no previous-frame branch;
  report actual injection events separately instead of claiming identical hooks.

The1/10/11 anchor cases exercise a single interruption and both sides of the
already frozen operational budget. They do not select a new budget. Fault
magnitudes are fixed engineering counterexamples, not measured hardware limits.
An available pose under a bias is not automatically a pass: later native scoring
must retain actual errors and full denominators. Likewise repeated appearance
does not have an assumed universal outcome. A missing sensor must not silently
produce a current pose, and conflict/budget failures must remain terminal.

## Remaining integration

The per-scenario paired callable does not yet persist/authenticate all eleven
scenarios across eight trials. Before native coordinate parsing, the future
cohort path must save and verify the complete base **and stress** populations,
counts, injection reach/application events, sensor identities and failure states.
The current base-only phase cannot stand in for that requirement. Add transformed
pose scoring and independently reconstructed stress summaries without treating
injected-reference events as native observations. Verify storage/runtime bounds
for the expanded streams; the earlier28GiB proposal is not yet proof of fit.
The maximum population is38,984 trial-scenario frame pairs and77,968 observer
calls (eight trials, eleven scenarios,443 frames). These are repeated paired
replays, not38,984 independent observations. With the current generic128KiB row
allowance, stress streams alone could exceed4GiB. The future writer needs an
explicit justified expanded resource contract or tighter schema-derived bounds;
it must not silently assume the existing headroom suffices.

No new native collection, RGB acquisition, model fitting, tracking adoption or
hardware execution is authorized by this source preparation. The original
12-layout collector and prepared36-fit study remain unchanged and take scheduling
priority. Novel-maze navigation, useful JEPA prediction, memory/backtracking and
deployment-valid sensing remain unachieved.

## Component verification

The first focused invocation87696 terminates with27 passing and8 failing tests:
the shared warmup fixture tried to deepcopy OpenCV keypoints, which cannot be
pickled. Each affected test now builds a fresh actual84-observation warmup;
there is no runtime state-copy/recovery mechanism. Invocation1537 terminates
exit0:35 tests pass in69.34s. Subsequent added tests cover full repeated-image/
shared-gyro rows and the distinction between expired outage samples and an
actually applied intervention. Final adjacent regression is recorded below only
after completion. These tests use synthetic texture/depth and ideal causal
histories, not new native scenes or deployment-calibrated sensors.

The runtime records `observer_update_attempted` separately from scheduled packet
interventions and actual reference-injection events. An observer already stopped
before onset is not a successfully exercised fault case. Gyro intervention
counts are explicitly history entries, including duplicated shared samples,
not independent measurements. Once all missing gyro samples age out, no packet
change is reported merely because the onset time has passed.

Read-only15065 verifies the unchanged786-source matched-study definition before
an inappropriate predecessor verifier rejects the replay launch's different
schema (`input_sha256` absent). No experiment ran. Corrected source-only guard
78008 terminates exit0 and verifies all701 completed-replay and771 live-collector
source hashes at their bound, nonprotected, nonsymlink paths. All three new
stress preparation paths are outside those frozen sets and the786-source study.

Final adjacent invocation70513 terminates exit0: **269 tests pass in239.41s**,
across eleven explicit files (38 new stress tests plus231 adjacent tests).
Durable JUnit:
`.generated/navigation-development-staging.m6MDz1/independent_tracking_stress_adjacent_v1.xml`.
This is not a full-repository regression, new native experiment or hardware
qualification. Runtime and test source hashes are preserved in the checkpoint;
no source under test changed during the final invocation.

The original collector is still live on l05, the sixth layout. Its final audit
is not present; only l00–l04 count as completed (600 eligible collection trials).
The all-layout terminal result and matched-study output remain absent. No
completed experiment was restarted and no competing native job was launched.
