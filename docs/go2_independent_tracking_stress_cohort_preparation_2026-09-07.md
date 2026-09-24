# Complete stress replay, phase admission and native scoring components

The new `scripts/independent_tracking_stress_cohort_development.py` integrates
all eleven fixed scenarios across all eight planned trials. It is callable
source preparation, **not a new native experiment, tracker adoption or maze
navigation result**. The original collector and pending matched learning study
are unchanged. A native launcher, source/runtime preflight and predecessor
scene/start/observational-prefix comparison are still required.

## Executable path

1. Construct `StressCohortStore` at a fresh validated attempt root. Collect and
   admit the eight episodes through the existing collector/receipt components.
2. Complete the unchanged base `replay_population` and its saved sensor phase.
3. Run `replay_stress_population`. Each trial/scenario gets a fresh paired
   observer, every recorded frame and an exclusive output stream. Zero-frame
   physical stops retain all eleven empty streams. Never infer success from an
   unrecorded onset, an already stopped observer or an empty error population.
4. `admit_complete_sensor_phase` authenticates all base and stress files and
   reconstructs every intervention from the original sensor packets. It compares
   original/transformed hashes, clocks, scenario definitions, exact row fields,
   observer attempts, terminal reasons, reference events, paired availability
   and continuity. It verifies the full population again after reconstruction.
5. Only then can `evaluate_complete_population` invoke the existing native raw
   auditor. One raw reconstruction per trial feeds the base scorer and all eleven
   transformed-stream scorers. All sensor inputs/outputs are authenticated again
   before the final result is written.

The base and stress sensor phases precede **every** native-coordinate callback,
not just the callback for a matching trial. Hashing recorded native artifact bytes
is allowed for integrity, but coordinates are not parsed during the sensor
phase. This is a source-enforced development boundary, not process isolation or
an independent final benchmark. The saved-row reconstruction reuses its explicit
accounting helper; it is not an independently reimplemented estimator.

The nominal scenario must reproduce the base poses, reference selections,
continuity and failures exactly; only observer wall time may differ. An available
pose alongside a failure, an update after a terminal latch, changing terminal
reasons, duplicate reference-injection events, injected contradictions accepted
as poses and current-measurement loss converted into a current pose are rejected.
Rows cannot carry undeclared privileged fields. Packet changes, actual update
attempts and reference-hook calls are separate denominators.

Each scored stream retains absolute and consecutive-estimate position/orientation
errors, common-frame denominators, bridge errors, missing estimates and observer
timings. The unchanged2cm/2° empirical allocation is not calibrated uncertainty.
Full availability alone cannot pass it. Empty/missing errors retain null maxima,
not zero error. Synthetic reference faults are still mechanism interventions,
not physically rendered occlusion. Repeated imagery is deliberately artificial
RGB corruption with the original depth; bias magnitudes are not calibrated
sensor distributions. All earlier negative results remain unchanged.

Any replay/admission/native-scoring exception preserves the partial files,
records terminal failure where storage permits and denies retry through the
cohort. A fresh store also rejects pre-existing base or stress metadata. There
is no runtime resume API. No global observer class or acceptance gate is patched.

## Corrected resource envelope

The previous28GiB proposal did not cover the expanded stress streams. This
component declares **36GiB total**, with the unchanged **40GiB free-space
reserve**, using the following maximum serialized-output envelope:

| Output | Bound |
| --- | ---: |
| Eight raw episode sets |24GiB |
|192 base/stress estimate/evaluation streams |443 rows each,128KiB per row |
|22 metadata files |32MiB each |
| Combined worst case |35.0703125GiB |
| Remaining total-cap headroom |0.9296875GiB |

The exact roster contains214 metadata/stream paths. Stream reads are bounded
before allocation and reject more than443 rows. Existing exclusive writers
account for the current open stream, serialized metadata and actual raw receipts.
The total cap remains enforced even if any individual output is unexpectedly
large. The36GiB budget is **not space already consumed or a completed preflight**.
The future launch definition must bind this resource contract and check actual
free space. Native external-write/deferred-history bounds, peak memory and
runtime scheduling still need final review; this is not an OS quota or a proved
memory bound. No new native process was started.

## Tests and limits

Initial focused invocation43839 terminates exit0:25 tests pass in43.35s. The
fixture uses actual frozen observers with synthetic causal packets. All native
files are explicitly non-NPZ sentinel bytes: reading coordinates in the sensor
phase would fail. For numerical scoring/phase-order tests, a mock raw auditor
supplies declared synthetic pose arrays; it does not qualify native reconstruction.

Each test receives only an explicit SHA-checked synthetic fixture roster in a
new temporary root. Fixture metadata restoration is test scaffolding, not a
production resume, clean source export or copying prior physical evidence. A
separate actual-observer synthetic87-frame sequence exercises the frame84 anchor
denial, one measured bridge and rejoin. That confirms the new exposure accounting
can distinguish an exercised fault from the short three-frame fixture whose
onset is never reached.

Corruption tests cover missing scenarios/trials, modified definitions/budgets,
source and transformed bindings, intervention details, pose shape, update/event
provenance, clocks, nominal disagreement and summary counts before any native
callback. Numerical tests retain a3cm native error despite complete availability.
Further cases add undeclared/incorrectly typed row fields, a partial replay
exception with no resume, and all-empty numerical populations. Final adjacent
regression is recorded after completion below.

## Next actions

1. Finish the native eight-trial launcher and frozen source/runtime/resource
   preflight, using this complete path rather than the old base-only evaluator.
2. Bind and compare actual predecessor geometry, starts and sensor prefixes;
   differences in filenames, labels or metadata alone do not establish novelty.
3. Review deferred native storage and memory limits and resource ownership. Do
   not start a competing native collection while the original supervisor runs
   or displace the pending36-fit learning comparison.
4. Complete and independently verify the actual challenge. Its final component
   result intentionally leaves `full_challenge_pass`, independent-observation,
   navigation, real-time and goal flags false until their missing evidence exists.
5. If adoption is supported, run a separately frozen fresh closed-loop turn/
   return assay. Latest actual simulated room return remains0/3. Complete all12
   learning-layout receipts, then the matched JEPA/supervised/ablation study.
   Useful prediction beyond simple baselines, online rollout, memory/backtracking,
   unseen-maze completion and deployment-valid hardware sensing remain required.

## Final preparation verification

Final invocation37689 terminates exit0: **297 tests passed in282.47s** across
twelve explicit files, including28 new integration tests and269 adjacent tests.
JUnit:
`.generated/navigation-development-staging.m6MDz1/independent_tracking_stress_cohort_adjacent_v1.xml`.
No runtime or test source changed during that invocation. This is not a new
full-repository regression, native challenge or deployment qualification.

Read-only11295 terminates exit0: the unchanged786-source matched-study definition
and701 completed-replay source hashes verify; the three new preparation paths
are outside both frozen sets. The matched definition verifies the original
collector launch as well. No inherited collector, tracker, replay or learning
source was edited.

The original supervisor2063013 and l05 child2097634 remain live. At b390f0,
l05 has92/120 prechecks and no terminal audit; five completed layouts still
account for600 eligible collection trials, not navigation successes. No competing
native collection, new model fit or tracker adoption was launched.
