# Eight-trial tracking replay and post-sensor raw evaluation

Implemented the full-population receipt admission, paired sensor-only replay and
post-sensor raw audit/scoring path for the independent tracking challenge.
**No new native tracking challenge, training fit or controller adoption has been
launched.** A frozen launcher, stress arms and predecessor-prefix comparison
remain required; this is executable component preparation, not a completed
scientific challenge or novel-maze result.

Final adjacent regression: **231 tests passed** across ten explicit files,
including35 new cohort/evaluation cases and the196 prior adjacent tests.

## Scientific boundaries enforced by the new path

The cohort requires all eight fixed trial receipts, in order. Every admitted
receipt must match its exact planned scene specification, saved result, artifact
roster, byte sizes and SHA256 bindings. Orphan/omitted frame files, failed internal
persistence, uninitialized/infrastructure-failed trials and inconsistent
capture/decision/dispatch counts are rejected. Physical stops remain population
members. A setup rejection with complete saved setup evidence is not silently
converted into an infrastructure failure. Empty observation populations remain
explicit and cannot pass the later pose-availability or motion-coverage checks.

Original multi-reference and frozen temporal-anchor observers receive private
copies of the same causal RGB/body/depth/gyro packets. No observer reads native
pose arrays. Both keep their complete recorded frame population after terminal
failure, with explicit missing-pose counts, paired/common availability and
continuity spans. A missing pose cannot be followed by an unreported reset.
Published pose/frame/clock/rotation validity and finite observer-only timing are
checked. The existing registration and bridge thresholds are unchanged.

Only after all eight paired streams are saved and authenticated is
`sensor_phase_complete.json` written. Native admission verifies the full
collection again, authenticates that marker and every paired stream, and
reconstructs saved frame counts, availability, first failures and continuity
summaries. It rejects a rebound but inconsistent marker/row population before
the native evaluator callback can execute. This is a source-enforced phase
boundary, not an OS-isolated final benchmark or proof of observational novelty.

After admission, the new native evaluation component combines:

- existing near-field raw sensor/contact reconstruction and native wall checks;
- actual initial pose versus the specified start;
- setup/support/stop recomputation and complete pre-decision friction clocks;
- exact command/timing audit and replay of the real causal-packet selector;
- core raster order/precision witnesses and unchanged strict-depth/footprint
  checks, retaining failures rather than repairing pixels;
- native measured turn/translation/stopping coverage and explicit prefix hashes;
- paired absolute and consecutive-estimate position/orientation errors, missing
  poses, common-frame denominators and bridge-only errors.

The empirical local allocation remains2cm/2°, with complete pose availability on
the scored tape. It is a task allocation, not calibrated uncertainty. Complete
availability is not sufficient: synthetic native3cm displacement with an unchanged
visual pose fails the allocation. Conversely, empty/missing error populations
have null maxima and cannot be reported as zero-error successes. Frames within a
trajectory remain correlated; they are not independent navigation trials.

The final base-audit result explicitly leaves `full_challenge_pass`,
`independent_observations_verified`, `navigation_qualified`, `real_time_qualified`
and `goal_achieved` false. This is intentional: the required stress-arm and
predecessor-prefix comparisons have not been implemented/completed, and scoring
recorded motion is not closed-loop execution. Observer timings exclude acquisition
and control and cannot qualify the full100ms control loop.

Exceptions during sensor or native phases retain already written streams/audits,
record a terminal phase failure where storage permits, and prevent retry through
the same cohort object. No existing experiment or frozen source is overwritten.

## Resource accounting correction

The earlier preparation proposed24GiB for the whole challenge while allowing
3GiB per trial. Eight maximum-size trials alone consume24GiB, leaving no room for
launch metadata, receipts, replay, audits or failure records. The new component
therefore separates **24GiB raw episode bytes** from **28GiB total cohort bytes**,
retaining the40GiB free-space reserve. The extra4GiB is headroom, not a claimed
measurement or an allocation already consumed. The future launcher must include
these exact limits in its frozen definition and check the actual free space.

Metadata files are bounded to32MiB and JSONL rows to128KiB. Streaming accounting
includes the currently open stream, not just previously closed files. The
single-episode external-renderer reservations are still admission/post-write
checks rather than OS quotas; deferred native contact/history and memory bounds
still need final validation before launch. The earlier in-memory mesh-size check
does not settle those remaining limits.

## Tests and what they do not establish

The new cohort tests construct complete bound synthetic artifact rosters whose
native files are deliberately non-NPZ sentinel bytes. A mock sensor reader supplies
valid retimed synthetic causal packets to the **actual frozen observers**. Thus
native bytes can be authenticated but cannot accidentally be parsed during the
sensor phase. These are not eight acquired physical scenes or independent inputs.

Scoring tests use the real sensor phase and error/coverage calculations, with an
injected raw-auditor fixture. They test ordering, numerical errors, missing-pose
denominators and failures, not real native reconstruction. Additional raw-audit
wiring tests retain real selector, command and coverage checks while mocking the
lower-level native reconstruction/setup/stop/footprint functions. Native geometry,
sensors and physics still need the actual challenge run and audit.

Initial cohort-only52726 terminates exit0:22 tests pass in57.37s. Combined cohort/
evaluation24841 terminates exit0:30 tests pass in89.34s before the five additional
raw-audit wiring cases. The final adjacent regression is recorded below after
completion; do not equate these tests with physical outcomes.

## Live independent-learning evidence

The original supervisor has completed and verified l00–l04: **five layouts and
600 eligible collection trials**, not600 navigation successes. Read-only11865
terminates exit0 and verifies l04's receipt/exit0, l05's exact preceding-receipt
link, and both children retaining the frozen765-source collection definition.

L04 has120 eligible departures, no excluded trials and21 contact-positive targets.
Three strict-visibility failures remain recorded:
`l04_open_passage_recent_forward_nominal_a3`,
`l04_open_passage_recent_forward_lower_friction_a3`, and
`l04_junction_recent_forward_lower_friction_a5`.

- Supervisor l04 verified-receipt SHA256:
  `028904304e893caf4a4b73869bbc44d346d5b4ba059d1adfbcef86f85fa105e4`.
- L04 audit SHA256:
  `0783b3dde0db8dc914dc972406d4c442526b4c5651f837e405ce811dac5f0571`.
- L05 launch SHA256:
  `e40c18b7d41b24d4757108712134803e0b8f548392eb4861e2c4f2c319f44729`.

Session25963 remains live. The old l04 child2090031 is terminal and must not be
restarted; supervisor2063013 has started l05 child2097634. This is normal sequence
advancement, not a replacement attempt. No competing native collection was started.

## Next actions toward the full goal

1. Complete the eight-trial native launcher and its frozen source/runtime/resource
   preflight, calling the existing single-episode collector then these cohort
   functions. Preserve the original collector and its pending36-fit matched study;
   require completed resource ownership before a new native challenge starts.
2. Specify and implement fixed stress arms before exposure: old-anchor absence
   versus current RGB-D loss, contradictory/repeated appearance and declared
   depth/gyro errors. Do not tune frozen bridge/registration thresholds to them.
3. Verify genuinely different actual native scene/start and sensor prefixes against
   the completed inner/intent recordings, using their exact bound witnesses.
   Hash differences due only to labels or metadata are not observational novelty.
4. Finish final resource/input/source review and complete the native challenge and
   independent result verification. A separately frozen fresh closed-loop turn/
   return assay follows only if the full challenge supports adoption.
5. Complete all12 learning-layout receipts and run the reviewed36-fit JEPA/
   supervised/input-ablation comparison. Establish useful prediction before
   matched online-rollout, memory/backtracking and independent-maze navigation
   claims. The latest physical-simulation room return remains0/3; deployment-valid
   sensors and bounded hardware evidence remain required for the full goal.

## Final preparation verification

Session60510 terminates exit0:231 tests pass in154.14s. Durable JUnit
`.generated/navigation-development-staging.m6MDz1/independent_tracking_cohort_preparation_v1.xml`
contains231 tests and zero errors/failures/skips, SHA256
`cdcbc9f5365d2080c3d669bf0600025082bd88f4114a49b816ed400d46389492`.
This is not a new full-repository regression or native challenge result.

Read-only38794 terminates exit0:701 completed tracking-replay source bindings and
the unchanged786-source matched-learning definition verify. New cohort/evaluation
sources and tests are outside both frozen sets. The matched-study definition
remains`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.
No original collector, observer, completed replay or matched-study source changed.
The final source/test hashes are preserved in the autonomous checkpoint.

The current l05 collector has17/120 prechecks at c3b83c and no terminal audit yet;
only l00–l04 count as completed layouts. All new test/verification handles from
this preparation turn are terminal. The original supervisor25963 continues.
