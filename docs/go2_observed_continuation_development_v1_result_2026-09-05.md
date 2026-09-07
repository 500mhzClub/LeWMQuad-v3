# Continuous observed branching: demonstrated in part, with scan-clearance failures

All16 physical trials completed. The corrected full audit passed the same raw
evidence, reproducing4,164 controller decisions. Every method—including the
matched non-learned fixed-forward control—completes2/4 two-leg tasks. The other
two fixtures cross and settle successfully, then contact a wall during the
intervening scan. This establishes a partial continuous observation-driven
branch/second-traversal demonstration, not reliable maze navigation or a
JEPA-specific advantage.

## What actually ran

The [fixed protocol](go2_observed_continuation_development_v1_2026-09-05.md) reuses
four local1.2 m corner/tee fixtures with coupled lateral/heading offsets. After
actual RGB-initiated traversal and a zero hold, the robot scans, selects an
observed non-forward side branch, aligns to its bearing, obtains a fresh forward
proposal and attempts a second traversal. One gyro reference and sensor stream
span the episode; there is no teleport, oracle route follower or injected place
identity. Three learned arms retain their frozen temporal ensembles. A fourth
arm uses fixed0.3 m/s forward actions under the same traversal/arrival wrapper.

| Method | First-leg successes /4 | Two-leg successes /4 | Native contact stops /4 | Selected traversal actions |
|---|---:|---:|---:|---|
| Fixed forward |4|2|2|50 forward|
| Direct-only prediction |4|2|2|50 forward|
| Supervised recurrent prediction |4|2|2|50 forward|
| JEPA recurrent prediction |4|2|2|50 forward|

All four arms have identical physical trajectories within each fixture. Across
the entire panel there are only three distinct raw-array trajectory groups:
four copies of each failed trajectory, and eight copies of one successful
trajectory across two rendered motifs. These are not16 independent mazes or
eight independent successful dynamics replications. No confidence interval
treating frames or repeated traces as independent is appropriate.

There are no sensor-contract or body-stability failures. All eight native stops
occur during SCAN, before its first completed quarter-turn view. All eight
successful tasks complete scan, observed side selection, gyro alignment, fresh
forward reobservation, second arrival and final release. No trusted map edge,
place identity, beacon detection or return is established by their two ledgers.

## The physical failure and what it means

For the negative lateral/heading condition, scan begins with the base at
(1.230,-0.207) m in the corner or (1.556,-0.254) m in the tee. Both then contact
the south wall, whose inner face is y=-0.600 m, with the native RR_calf rigid
group. The first stop is9.022 s after initial settling in the corner and9.774 s
in the tee. Reported scan drift through the last observed decision is0.0399 and
0.0347 m respectively. The rigid group includes fixed child shapes; a link
name alone does not identify a particular collision primitive.

The positive condition begins scanning at(1.235,+0.154) m and completes the
continuous task in43.0 s after initial settling. Its scan drift reaches0.0921 m.
The first traversal progresses1.235 m and the second1.221 m along their respective
evaluation directions. The scan-selected ray is16.7 s old when selected, but
the controller requires a new forward proposal after alignment before translating.
That does not calibrate the unmodeled translation error in stored scan rays.

The first crossing/zero-release criteria genuinely pass in every fixture; a
later scan contact must not erase that earlier event. Conversely, those criteria
do not establish a pose from which turning is feasible. The current learned
rankers choose forward despite an observed nonzero exit bearing in the failed
initial condition. Their half-second cost scores contact and translation toward
a cue, not arrival orientation or the feasibility of the subsequent scan.
Heading and lateral offset are coupled in this panel, so their separate causal
effects have not been identified.

This also exposes an important limit of the scientific comparison: scanning and
alignment are hand-designed, not controlled by JEPA predictions. The frozen
learned action bank contains stop, forward, forward-left, forward-right and
reverse, but no pure in-place turn. A scan-contact failure is therefore not an
isolated test of JEPA's ability to predict or select that maneuver. Adding model
capacity alone would not make this experiment test that question.

The matched fixed-forward arm removes the prior panel's speed/cadence ambiguity:
on these fixtures, learned action ranking is unnecessary for the observed
successes. This is a useful local negative result, not a claim that learning
cannot help in unfamiliar or more demanding mazes.

## Audit correction and evidence

Collection65024 completed16, exit0. Original audit86158 passed four short contact
episodes, then failed before reading the first long completed episode: the
legacy route loader allowed341 frames, whereas the continuation protocol and
trial auditor already allowed806. Successful episodes contain431 frames.

The [reader-only correction](go2_observed_continuation_reader_correction_2026-09-05.md)
adds a separate806-frame loader. Source-comparison tests verify that its function
body differs only in name and population bound, and that the corrected trial
auditor differs only in loader calls. The old loader and failed audit remain
unchanged. No physics, model, sensor packet, criterion or recorded outcome was
edited or rerun. Corrected audit54267 passes16 trials and all4,164 decisions,
exit0, including150 learned ensemble choices and50 fixed-forward choices.

Full evidence covers221,592 physics samples,221,592 live fast-gyro measurements,
22,156 ordinary sensor samples and4,212 actual RGB packets. It includes native
contact reconstruction, causal histories, camera transforms, paired initial
physics/body histories, command slew, full controller/ledger replay and both
per-leg crossing/release windows. Inference timing exclusions remain the two
explicit nested learned timing fields only. Current sensing and emergency stops
remain ideal/privileged simulation mechanisms, not a hardware safety claim.

The188 original source/test/protocol paths,165 inputs and two gait bindings are
unchanged. The correction additionally binds six source/test/document/fixture
paths and exact predecessor identities. Before collection845 tests passed
across77 files; with the18 correction tests the full focused suite passes863
across78 files. No study or audit is now running.

Root: `.generated/go2_observed_continuation_development_v1_attempt_001`.

- Launch: `cda6d07ddc8238f2a3b540924fab153829a031e24e4a7db8f046a0f4b7f162d7`.
- Physical result: `003147bede55befe019362335faefe5173ae6e98c4b1bbb429fb22ffa44ff6bf`.
- Preserved original FAIL: `df3ed99d49c66e4161e6054358f4f4dd05309b041f95e54f8d29ee3e02dccd2c`.
- Correction binding: `72b06a87062929232872d1b76c30f420ecfd382e182037b8650d96746bd38aa9`.
- Corrected full PASS: `ad4f7d0e517041566b36b88a94b39a5d0527a93fc65790fade1726cc01ddfd3d`.

## Next intervention toward the real task

1. Test observation-driven alignment/centering before translation, retaining
   fixed-forward as a matched control. In the failed corner's initial observation
   the proposed bearing is+0.0873 rad, yet every arm translates without turning.
   A coarse0.08 rad turn tolerance is nearly that entire correction; it should
   not be assumed to produce useful corridor alignment. Specify any finer
   alignment criterion from the intended motion envelope before new execution,
   preserve this panel, and measure actual arrival pose and scan contact rather
   than merely reporting smaller estimated heading error. This remains a
   hypothesis: floor-extension bearing is not a certified corridor centerline.
2. Keep the intervention in a continuous task. Do not return to centered-only
   turn assays or accept first-crossing success as proof of useful arrival.
   If heading alignment is insufficient, test measured visual centering,
   repositioning or a separately declared additional-view sensor configuration.
3. Extend the navigation memory to explicitly uncertain place/branch hypotheses
   that can guide bounded exploration and return. Evaluate false associations
   and recovery; do not wait for perfect certification or convert evaluation cell
   labels into runtime observations. Add physically rendered, actually observed
   beacon acquisition as part of the same task.
4. To test prediction over the whole local task, supply training/evaluation
   support for the scan/alignment/repositioning command families and compare
   matched predictive and nonpredictive choices at those real decision points.
   Retain the strong fixed/visual-feedback baselines. Do not claim a multi-step
   planning benefit from the current one-transition action head.

Reliable novel-maze exploration, beacon discovery/directed return, independent
layout and seed comparisons, robustness and bounded real-Go2 evidence remain
unachieved. This result advances continuous execution and identifies its next
failure mechanism; it does not shrink the ultimate objective.
