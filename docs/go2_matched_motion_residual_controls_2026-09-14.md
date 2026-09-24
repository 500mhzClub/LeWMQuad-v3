# Matched motion-correction fits for training-method controls

The twelve-trial training-method comparison is complete. JEPA achieved 3/4
verified goals and 2/4 round trips; direct achieved 3/4 and 1/4; supervised rollout
achieved 2/4 and 1/4. All twelve runs had zero disallowed contacts. Each method
used serial/original capture on layouts 0 and 1 and compact/parallel capture on
layouts 2 and 3. All four within-layout comparisons verified equal non-treatment
settings/source identities and correct model/correction bindings. These results
do not establish reliability or a statistical training-method advantage.

Combined evidence:
`go2_matched_training_cohort_layout00_03_v1_attempt_001/result.json`.
The next prospective survey-repositioning experiment has been launched on
layout 0, using the original serial capture setup and the same frozen JEPA fit.

The deployed development correction was fitted specifically to the primary JEPA
model. Reusing it unchanged with another model would confound a training-method
comparison. `scripts/fit_matched_closed_loop_motion_residual_development.py`
prepares separate corrections for the fixed seed-2026091001 full-input direct
and supervised-rollout models. Neural weights remain frozen.

Use the original four training recordings, validation recording, registered
visual-pose labels, command-prefix masks, stationary thinning, ridge penalty and
features from `fit_closed_loop_motion_residual_development.py`. Every collected
window's target, mask and grouping must equal the saved original JEPA population.
The direct condition uses its trained direct head; supervised rollout uses its
trained rollout head. Freeze coefficients before collecting validation windows.
No fresh-maze outcomes or native pose labels enter fitting.

`scripts/public_policy_replay_development.py` reads retained RGB/body inputs
without requiring retired depth. Four retained policy packets matched the full
reader exactly; four packets from a depth-retired training recording validated.
The full sensor reader still rejects retired depth, and the policy-only reader
does not expose full sensor replay. Both fitting modules imported successfully
and the original collector's private binding was checked.

Both fits completed with owner exit 0 between native trials, in 40.58 seconds
(direct) and 38.60 seconds (supervised rollout). Each reproduced all 1,207 training
and 448 validation windows, targets, masks and grouping records exactly. Neural
state identities were unchanged. No depth regeneration was needed.

At the 700-ms horizon, 119 validation windows had matched moving commands:

| Condition | Base XY RMSE | Corrected XY RMSE |
| --- | ---: | ---: |
| JEPA (existing fit) | 20.85 mm | 8.03 mm |
| Supervised rollout | 18.62 mm | 7.96 mm |
| Direct | 22.99 mm | 8.47 mm |

These similar development errors establish neither statistical equivalence nor
a JEPA navigation advantage. The correction labels are registered visual poses,
not native truth. Prospective navigation with these controls remains outstanding.

Results and frozen coefficients are retained under RecoveryStorage:

- `go2_matched_motion_residual_direct_v1_attempt_001`, fit SHA-256
  `118fa93d4a420525c5685cf14b2aeec30cff827abf1d65029716aa7e69dd9e6a`.
- `go2_matched_motion_residual_supervised_rollout_v1_attempt_001`, fit SHA-256
  `413a3160c21dc43b4b8a1b87b4c092d59428158270d7b91179229d84b04fda1d`.

Bind each correction to its corresponding fixed model for the subsequent
training-method control trials. Do not alter the ongoing four-layout
fixed-controller transfer cohort. Correction validation is development evidence;
only subsequent matched native navigation can establish whether training method
affects goal-reaching.

`lewm/matched_motion_residual_runtime_development.py` now loads the exact fitted
correction for each of the three fixed conditions into the existing pulse-aware
runtime and records that correction's actual identity. It leaves the action
selector, causal pose history, pulse handling and neural prediction heads intact.
This class is not used by the ongoing fixed-controller fresh-maze cohort.

A lightweight integration check loaded all three real frozen models, constructed
and cleanly stopped all three runtimes, and exercised six correction calls
(full-command and pulse contexts for each condition). All six used the expected
fit identity and produced finite XY corrections while preserving yaw/contact
channels. The check used synthetic causal pose history and forecasts; it is
wiring evidence, not another navigation trial or prediction-accuracy result.

The prospective launcher is now
`scripts/run_go2_matched_training_navigation_development.py`, with required
`--layout-index 0..3` and `--condition jepa|direct|supervised_rollout`. It applies
the same recovery-release controller to every condition and records the actual
assigned correction after all inherited configuration writers. It uses distinct
`go2_matched_training_<condition>_heading_release_native_layoutXX_4800_v1_attempt_001`
roots. The existing recovery-release cohort is unchanged.

A lightweight launcher check intercepted execution before native scene creation
and exercised all three conditions. The complete metadata writer chain preserved
the assigned model/correction, runtime arguments retained the common budget and
arrival radius, and a mismatched model condition was rejected. Method resolution
selects the existing matched-correction constructor and correction function plus
the shared heading-release selection function. An initial closure incompatibility
in the writer was corrected; the three-condition rerun passed. This check used a
runtime spy, not new model inference or navigation evidence.

After the four recovery-release trials, run the training controls on all four
layouts in index order, using JEPA, direct and supervised rollout in that order
on each layout. Include a new JEPA arm through this shared launcher; do not
compare controls against a differently configured controller or select layouts
from successful outcomes. Keep this controller revision and the three frozen
models/corrections unchanged across that comparison. This is a first-seed
development comparison, not an established replication result or final benchmark.

After completing and physically evaluating all four recovery-release trials
(4/4 goal arrivals, 3/4 round trips), the first shared-launcher JEPA trial was
launched on layout 0:
`go2_matched_training_jepa_heading_release_native_layout00_4800_v1_attempt_001`.
It uses the same frozen original JEPA correction through the matched-correction
runtime. Its completed result is below; direct and supervised rollout on layout 0
follow after owner exit, archiving and physical evaluation.

## First JEPA trial: no arrival; earlier layout-0 goal success did not repeat

The owner exited 0 after retaining all 4,805 camera pairs. Independent physical
evaluation confirmed no arrivals and zero disallowed contacts. Median/maximum
position error was 21.83/48.03 mm; 1,180/1,200 plans were on time. Closest/final
physical goal distance was 1.253/2.036 m; path length was 10.774 m. Timed wall
duration was 481.895 s.

All 1,200 recorded corrections matched the assigned original JEPA fit and model.
Common controller source hashes were unchanged from the preceding recovery trial.
The run selected 1,067 pure turns (88.92% of plans), only 29 translating actions,
and 104 holds. Recovery release activated 187 times, 182 on time, yet the robot
remained on a frontier route at the end. Its saved
`matched_training_runtime_diagnostic_v1.json` records the model/correction binding
and action sequence. The precise cause of divergence from the earlier run is
unresolved; do not attribute it to training method, launcher changes, timing or
pose drift without further evidence.

This failure remains in the new matched cohort and weakens any inference of
repeatability from the earlier 4/4 outbound recovery cohort. Neither arrival
limits nor controller behavior were changed after observing it.

Further saved-record analysis found 173/187 releases followed within five plans
by a new active latch in the opposite direction; 45 relatched on the immediately
next plan. There were 381 adjacent pure-turn direction switches. The middle
frames 2,000–4,000 consisted of 501 survey plans, so the waypoint-heading
diagnostic had no applicable windows. A separate survey diagnostic matched the
actual requested prefix through 700 ms for 491 windows and recovered the saved
predicted heading gain from scan utility and contact penalty, without model
reruns. All 222 matched left turns predicted and achieved improved survey
alignment; all 215 matched right turns predicted and achieved worse alignment.
The other 54 windows were holds: 29 predicted improvement but actually worsened
heading, with median actual worsening only 0.00077 rad and maximum 0.01449 rad.
The dominant repeated-turning failure therefore involved selecting opposing
turns despite correct predicted direction of their alignment effect; hold-yaw
prediction errors also exist. These diagnostics do not establish the outcome
of a changed recovery or survey rule. Evidence is retained in
`recovery_relatch_diagnostic_v1.json`,
`saved_waypoint_alignment_2000_4000_v1.json` and
`saved_scan_heading_alignment_2000_4000_v1.json` in the JEPA run root.

The direct-trained model was then launched on the same layout:
`go2_matched_training_direct_heading_release_native_layout00_4800_v1_attempt_001`.
The prior owner was absent; CPU utilization was 0.5%, available RAM was 77.15 GiB
and RecoveryStorage had 86.66 GiB free. Both GPUs were idle. Its assigned
correction is the frozen direct-head fit, while the navigation controller,
sensor configuration and budget remain common. Supervised rollout follows.

## Direct layout 0: verified goal; return incomplete

The direct owner exited 0 after retaining all 4,805 camera pairs. Independent
physical evaluation verified outbound arrival at frame 4,492. The one-second
dwell stayed 11.28–15.88 mm from the goal, with maximum 100-ms speed 0.00865 m/s
and all requested commands zero. There were zero disallowed contacts. Return
did not complete before the unchanged budget; final physical home distance was
2.381 m. Count this as a failed round trip.

Median/maximum position error was 11.24/15.39 mm; 1,179/1,197 plans were on time.
Path length was 21.977 m and timed wall duration was 481.894 s. All 1,197 saved
corrections used the assigned direct-model fit. The recovery-release rule did not
activate. Selected actions included 494 pure turns, 280 translating actions and
423 holds. This single direct-model goal arrival versus JEPA's no-arrival outcome
does not establish a training-method advantage.

The supervised-rollout model was then launched on layout 0:
`go2_matched_training_supervised_rollout_heading_release_native_layout00_4800_v1_attempt_001`.
The prior owner was absent, CPU utilization was 0.3%, available RAM was 77.07 GiB,
and RecoveryStorage had 83.25 GiB free. Both GPUs were idle. Its frozen correction
is the separately fitted supervised-rollout correction; all common controller
settings remain fixed.

`scripts/compare_matched_training_navigation_development.py --layout-index N`
is prepared to summarize each completed three-condition set. It checks equal
non-treatment launch fields including recorded source identities, verifies each
runtime correction against its assigned model/fit, and retains all independently
evaluated outcomes. Its first execution completed successfully on layout 0.
Continue all three methods on layouts 1–3 after this set.

## Supervised rollout layout 0: verified goal; first three-way set complete

The supervised-rollout owner exited 0 after retaining all 4,805 camera pairs.
Independent physical evaluation verified outbound arrival at frame 4,556. The
one-second dwell stayed 12.10–17.09 mm from the goal, with maximum 100-ms speed
0.01715 m/s and all requested commands zero. There were zero disallowed contacts.
Return did not complete; final physical home distance was 1.802 m. Count this as
a failed round trip.

Median/maximum position error was 5.32/9.76 mm; 1,170/1,191 plans were on time.
Path length was 21.985 m and timed wall duration was 481.896 s. All 1,191 saved
corrections used the assigned supervised-rollout fit and model. Recovery release
activated 28 times; selected actions included 547 pure turns, 305 translating
actions and 339 holds.

The three-way comparison confirmed equal non-treatment settings and recorded
source identities, plus correct assigned model/correction binding for every
recorded plan. Result:
`go2_matched_training_comparison_layout00_v1_attempt_001/result.json`.

| Model | Verified goal frame | Verified round trip | Disallowed contacts |
| --- | --- | --- | --- |
| JEPA | None | No | 0 |
| Direct | 4,492 | No | 0 |
| Supervised rollout | 4,556 | No | 0 |

The first JEPA run on layout 1 was launched next:
`go2_matched_training_jepa_heading_release_native_layout01_4800_v1_attempt_001`.
The prior owner was absent; CPU utilization was 3.4% during the small comparison
summary, available RAM was 76.75 GiB and RecoveryStorage had 79.94 GiB free. The
comparison process completed before native launch; both GPUs were idle. Continue
the prespecified three-method order with the same controller and budget.

## Proposed parallel execution: pending recording-memory assessment

Following the user's concurrency question, a live resource check found 16
physical cores / 32 logical CPUs, about 7.5% total CPU utilization and 68.89 GiB
available RAM. The simulator and gait policy use the CPU backend; both GPU busy
counters were zero. CPU capacity supports two native owners with disjoint CPU
allocations, but the later memory measurement below prevents starting this pair
yet.

Finish the already-started layout-1 comparison serially, matching its JEPA run.
If recording-memory headroom is established, run layouts 2 and 3 together for each method, in JEPA/direct/supervised-rollout
order. Use `taskset -c 0-7,16-23` for layout 2 and
`taskset -c 8-15,24-31` for layout 3. CPU topology confirms these are disjoint
groups of eight physical cores, including both hardware threads of each core.
The existing launcher and controller sources remain unchanged; subprocesses
inherit the affinity, including later archive workers.

Record a `parallel_execution_profile.json` in each parallel run root with its
actual CPU affinity, paired layout, and two-owner execution mode. Keep the same
allocation and mode for all three models on each layout. Check inherited affinity,
sampled memory use and the existing on-time-plan results during the first pair
before increasing concurrency; start with two only. Wait for both owners to
finish saving before the next pair. Layouts 0 and 1 remain serial evidence;
layouts 2 and 3 will be parallel evidence, with model comparisons made within
each matching layout/profile. Do not claim identical host contention or strict
real-time operation. No parallel native pair has yet been launched.

The mid-run JEPA layout-1 process tree reached 30.45 GiB aggregate RSS, with
48.45 GiB system RAM available. Total RAM is 91.96 GiB. Capture retains both
640x480 RGB images, native float32 depths, derived float32 depths and Boolean
valid masks for every frame: 33.0 GiB of these arrays alone at 4,805 frames,
before simulator/controller and other recording overhead. This is a source-based
buffer calculation, not a measured end-of-run process peak. Consequently two
full-length owners are not yet shown to fit safely. Smaller recording buffers or
measured headroom are needed before executing the proposed full-length pair.
The next serial run will report maximum RSS through `/usr/bin/time -v`, without
changing the controller or capture source. Shorter simulations and lightweight
post-run analysis have more scope for overlap.

## JEPA layout 1 complete

The owner exited 0 and retained 4,805 camera pairs. Independent evaluation verified
the goal at frame 3,788; the one-second quiet dwell remained 12.37–16.13 mm from
the physical goal, with maximum 100-ms speed 0.01616 m/s and all requests zero.
Return did not complete; final home distance was 3.578 m. Disallowed contacts
were zero. Median/maximum position error was 5.79/11.69 mm, 1,169/1,197 plans were
on time, and timed execution took 481.910 seconds. This remains a failed round
trip. The direct condition on layout 1 is next.

## Saved survey recovery diagnosis while direct layout 1 runs

`scripts/diagnose_saved_scan_recovery_development.py` reads completed planning
records only. Applied over the same frames 2,000–4,000:

| Run | Survey plans | Opposite preferred turn selected | New alternative-turn latches | Full-reserve translation excluded from survey scoring |
| --- | ---: | ---: | ---: | ---: |
| JEPA layout 0 | 501 | 217 | 101 | 342 |
| Direct layout 0 | 92 | 0 | 0 | 92 |
| Supervised rollout layout 0 | 162 | 32 | 20 | 95 |
| JEPA layout 1 | 205 | 0 | 0 | 199 |

All 101 JEPA-layout-0 latch triggers retained nominal footprint clearance for
the preferred turn, but missed its additional reserve by a median 2.91 mm
(range 0.032–11.42 mm). This explains the reserve trigger; it does not establish
that the reserve can be removed. The corresponding 20 supervised-layout-0
triggers also retained nominal footprint clearance, with median deficit 1.76 mm.

In 341 of the 501 JEPA-layout-0 survey plans, at least one translating forecast
passed full reserve and had minimum path clearance no worse than hold. Choosing
the largest endpoint-clearance gain among those candidates gives 18.70–49.88 mm
gain over hold (median 41.58 mm). These actions were not assigned survey utility.
The six trained actions already include these translations; no novel reverse or
strafe command is required to investigate repositioning. Saved report:
`go2_matched_training_jepa_heading_release_native_layout00_4800_v1_attempt_001/saved_scan_reposition_opportunity_v1.json`.

This supports a prospective survey-repositioning hypothesis after the frozen
training-method comparisons: allow an observation-grounded, full-reserve
translation to escape repeated turn recovery, then continue the unfinished view.
It is not a claim of executed safety, information gain, or counterfactual
navigation success. Survey translations can already be selected by downstream
recovery/terminal overrides in some records; the counts above describe exclusion
from the initial scan score specifically. No current controller source changed.

## First divergence between the two same-controller JEPA layout-0 runs

The earlier recovery-release run reached the goal; the matched-training JEPA
run did not. Their main source identity and all 83 shared extra-source hashes
match. The first 656 requested command intervals were identical, with no missing
request timestamps before their first difference. Native base poses were bitwise
identical up to that request as well.

At simulator time 14.62 s (13.12 s after the recording origin), the earlier run
requested zero with `CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE`; the later run
continued `[0, 0, 0.45]` with `CURRENT_NOMINAL_OBSTACLE_TEST_PASSED`. Their latest
observation timestamps were 14.40 and 14.50 s respectively. First native-pose
divergence was sample 7,360, and first captured pixel-digest divergence was frame
133. Selected plan actions first differed at frame 160; on-time status and
committed prefixes first differed at frame 132. Whole-record request timestamp
sets differ only starting at 481.96 s, after this initial divergence.

This is direct evidence that observation delivery timing produced different
executed commands before policy choices diverged. It does not establish that
this one event caused the eventual failure. Repeatability must therefore include
the asynchronous execution path; identical layouts, seeds and controller sources
alone do not establish identical trials. Saved evidence:
`go2_matched_training_jepa_heading_release_native_layout00_4800_v1_attempt_001/same_controller_first_divergence_v1.json`.

## Recording-memory reduction prepared, not yet used in native trials

Direct layout 1 reached 41.80 GiB aggregate process-tree RSS near frame 4,700,
with 37.63 GiB system RAM available. This does not leave demonstrated headroom
for another full run using the current buffer.

`scripts/compact_depth_retention_session_development.py` adds a separate mixin
which computes digests of the actual captured primary/auxiliary depth packets,
then replaces only their recording references with those digests. Live returned
packets remain intact. Original RGB and native depth arrays remain retained.
The existing archive writer is privately bound to read the stored digests;
archive pixels and packet identities are unchanged. Hashing occurs during
capture and is explicitly added to acquisition wall-time accounting, rather than
hidden outside the measured run.

This releases 2 x 640 x 480 x (4 + 1) bytes of duplicate derived-depth/valid-mask
arrays per frame, or 13.75 GiB at 4,805 frames. Two focused tests passed in
1.89 s: original live packets and hashes remain intact, saved PNG bytes and native
NPY payload bytes match, returned archive metadata matches, and a request to
archive discarded derived arrays is rejected. This is component evidence only;
there has been no compact-buffer native run or parallel pair. Existing launcher
and controller sources remain unchanged. Complete serial layout 1 before using
the compact buffer consistently across all three methods on layouts 2 and 3;
record that execution change and measure its timing/memory effects.

## Direct layout 1 complete; supervised rollout launched

Direct exited 0 after 4,805 camera pairs. Independent evaluation confirmed no
arrival and zero disallowed contact samples. Closest physical goal distance was
1.292 m, final goal distance 3.029 m, and path length 17.125 m. Median/maximum
position error was 2.81/9.72 mm; 1,177/1,200 plans were on time. Timed simulation
took 481.902 seconds. The full process including setup and archiving took 647.99
seconds, with GNU time maximum RSS 39,880,096 KiB (38.03 GiB) and zero swaps.
That maximum is not a simultaneous aggregate of all child-process RSS; the
separate live process-tree sample was 41.80 GiB. Resource evidence is saved as
`process_resource_usage.json` and `observed_memory_samples.jsonl` in the run root.

After its archive and independent evaluation completed, launched
`go2_matched_training_supervised_rollout_heading_release_native_layout01_4800_v1_attempt_001`
with the existing serial launcher and unmodified recording/controller sources.
Complete this third condition before the layout-1 comparison and before changing
the capture profile for layouts 2 and 3.

The separate compact launcher is now prepared:
`scripts/run_go2_compact_matched_training_navigation_development.py`.
Use the existing condition/layout flags, with `taskset -c 0-7,16-23` for layout 2
and `taskset -c 8-15,24-31` for layout 3. It privately substitutes the compact
camera-session class into the existing matched launcher. No controller, model,
layout, or base launcher source was changed. Launch metadata and
`parallel_execution_profile.json` record actual CPU affinity, paired layout,
planned maximum of two owners, and capture-time packet hashing. The existing
three-way comparator already requires these launch fields and source identities
to match within each layout; it needs no additional framework.

All six layout/condition writer combinations and compact-session method dispatch
were checked without launching physics. Three focused compact-recording tests
now cover exact archive evidence, unchanged returned live packet objects,
hash-time accounting and rejecting unavailable derived-array output. First native
use will be the scheduled JEPA layouts 2 and 3, after serial layout 1 completes;
retain both results and measure memory/deadlines during that pair. No additional
pilot simulation is required before these development trials.

A 20-pair synthetic hashing measurement gave median 1.234 ms and maximum
1.498 ms per primary/auxiliary pair. These measurements do not qualify native
timing. The latest three-test run passed in 2.06 s.

The same saved survey diagnostic on direct layout 1 found 457 survey plans in
frames 2,000–4,000, with 64 opposite-preferred turns and 31 newly latched
alternative turns. All 31 preferred turns still had nominal footprint clearance;
their median extra-reserve deficit was 2.73 mm. Full-reserve translations were
excluded from scan scoring in 399 of those plans. This extends the survey-recovery
concern beyond JEPA, without claiming that repositioning has succeeded.

## Layout 1 complete; first compact parallel pair launched

Supervised rollout exited 0 with all 4,805 camera pairs. Independent evaluation
confirmed no arrival and zero disallowed contacts. Closest/final goal distances
were 1.291/1.768 m, and path length was 19.638 m. Median/maximum position error was
6.17/12.95 mm; 1,162/1,200 plans were on time. Timed simulation took 481.926 s;
full process duration was 646.41 s, GNU time maximum RSS 39,879,904 KiB and swaps
zero. Summary and resource evidence are retained in its run root.

`go2_matched_training_comparison_layout01_v1_attempt_001/result.json` verifies
equal non-treatment settings/source identities and assigned model/correction
bindings. JEPA alone reached the goal on this layout; none returned. Across the
two complete layouts each model reached one of two goals, with zero of two round
trips. This is development evidence, not a reliability or method-advantage result.

After the comparison process exited, launched the scheduled compact JEPA pair:

- Layout 2: `go2_matched_training_jepa_heading_release_native_layout02_4800_v1_attempt_001`,
  PID 3400566, CPU group `0-7,16-23`; launch log 17:56:10.
- Layout 3: `go2_matched_training_jepa_heading_release_native_layout03_4800_v1_attempt_001`,
  PID 3400601, CPU group `8-15,24-31`; launch log 17:56:12.

Both owner processes were confirmed live with the assigned disjoint affinity.
Available RAM during startup was 67.69 GiB. This is the first native compact-buffer
use and first parallel pair. Measure memory/deadlines as it progresses and wait
for both archives before the direct-model pair. Keep the same capture/CPU profile
for every method within each remaining layout. The separate survey-repositioning
variant is not active in this cohort.

At frame 600 both owners were advancing in parallel. Process-tree RSS was
9.57 GiB (layout 2) and 9.42 GiB (layout 3), with 62.19 GiB system RAM available.
All observed child affinities matched the respective owner. Each run stores these
live measurements in `observed_memory_samples.jsonl`. End-of-run memory,
deadline rates and compact archive completion remain to be assessed.

Three-condition trajectory PNG/SVG figures for both completed layouts are saved
as `native_navigation_comparison` in their comparison roots and were visually
inspected. Layout 0 shows JEPA stalled at the first passage while the other two
models reached the goal; layout 1 shows JEPA reaching the goal while the controls
spent their remaining budget along the upper passage. No method has yet completed
a round trip in these first two layouts.

## First compact parallel result: JEPA layout 2 verified round trip

Layout 2 exited 0 after 2,393 camera pairs. Independent physical evaluation
verified outbound arrival at frame 1,568 and return at 2,391. Outbound dwell
distance was 21.08–21.96 mm, return dwell 4.04–5.93 mm; maximum 100-ms speeds
were 0.00914 and 0.01263 m/s, with all requested intervals zero. There were no
disallowed contacts. Median/maximum position error was 7.11/12.38 mm, and
580/590 plans were on time. Timed duration was 240.073 s; total process duration
including setup/archive was 345.46 s. Path length was 16.996 m and final physical
home distance 4.10 mm. GNU time maximum RSS was 13,852,544 KiB; swaps were zero.

The unchanged public reader reconstructed frames 0, 1,196 and 2,392 and verified
both RGB hashes, native-depth hashes and derived-packet hashes against the actual
capture digests. This confirms sampled native compact-archive compatibility,
not a full sensor audit. Evidence is saved as `compact_archive_replay_sample.json`.
Layout 3 continues its return phase. Wait for its completion/archive before the
next two-owner direct-model batch.

## Executed forecast error on completed layouts 0 and 1

`scripts/evaluate_saved_executed_motion_forecasts_development.py` compares saved
raw/corrected XY forecasts with evaluator-only physical motion. It accepts a
window only when all 35 actual requested 20-ms intervals match the known prefix
and selected candidate through 700 ms, including terminal translation pulses.
It does not evaluate unexecuted actions or the unmatched 800-ms stopping horizon.

| Layout / model | Matched windows | Raw endpoint XY RMSE (mm) | Corrected endpoint XY RMSE (mm) | Maximum corrected XY error through 700 ms (mm) |
| --- | ---: | ---: | ---: | ---: |
| 0 / JEPA | 1,175 | 10.44 | 5.48 | 22.51 |
| 0 / Direct | 1,176 | 24.32 | 9.33 | 38.89 |
| 0 / Supervised rollout | 1,169 | 18.30 | 8.68 | 38.94 |
| 1 / JEPA | 1,166 | 18.27 | 7.61 | 25.58 |
| 1 / Direct | 1,173 | 20.62 | 8.29 | 36.96 |
| 1 / Supervised rollout | 1,163 | 16.12 | 7.86 | 35.36 |

Saved reports include action-group breakdowns. These aggregates compare different
executed trajectories/action mixtures, not matched prediction examples, and the
overlapping windows are not independent. They establish neither a training-method
advantage nor a calibrated safety reserve. Direct/supervised errors sometimes
exceed 30 mm even within this shorter horizon, so the existing reserve should not
be described as a proven error bound. JEPA layout 0 failed despite modest executed
XY errors, reinforcing the need to test action selection and survey recovery.

## JEPA layout 3 verified round trip; direct parallel pair launched

Layout 3 exited 0 with 3,554 camera pairs. Independent evaluation verified goal
arrival at frame 2,336 and return at 3,552. The one-second dwells stayed
16.10–19.19 mm from the goal and 22.80–25.34 mm from home. Maximum 100-ms speeds
were 0.00726 and 0.01613 m/s, with all requested intervals zero. Contacts were
zero. Median/maximum position error was 3.40/8.24 mm; 857/881 plans were on time.
Timed execution took 356.415 s; full process time was 520.31 s, maximum RSS
19,410,912 KiB, and swaps zero. Path length was 28.948 m; final home distance
25.21 mm. The unchanged reader verified compact archive frames 0, 1,777 and 3,553.

Both first parallel trials therefore completed verified round trips and saved
readable compact recordings. Their on-time-plan rates were 98.3% and 97.3%; this
does not establish strict real-time qualification or timing equivalence to serial
execution. Both use the same physical checks as the serial trials. Full-length
eight-minute peak memory will remain observable if subsequent owners use their
whole budget. JEPA's four-layout results are 3/4 goal arrivals and 2/4 round trips;
the first two runs were serial and the last two compact/parallel, so retain that
execution distinction when interpreting the aggregate.

After both archives, evaluations and small replay samples completed, launched
the direct-model pair on layouts 2 and 3 with the same compact launcher and
respective CPU groups. Both roots retain the standard
`go2_matched_training_direct_heading_release_native_layoutXX_4800_v1_attempt_001`
names. Wait for both completions and independent evaluations before the final
supervised-rollout pair. The survey-repositioning hypothesis is still untested
in native execution and is not active in these comparisons.

## Direct layout 2 verified round trip

The direct owner exited 0 after 2,970 camera pairs. Independent evaluation
verified goal arrival at frame 1,304 and home arrival at 2,968. Dwell distances
were 5.71–13.81 mm at the goal and 14.98–21.58 mm at home; maximum 100-ms speeds
were 0.02204 and 0.02012 m/s, with all requested intervals zero. There were zero
disallowed contacts. Median/maximum position error was 3.56/5.69 mm; 715/725 plans
were on time. Path length was 16.484 m and final home distance 21.73 mm. Selected
actions included 298 holds, 144 pure turns and 283 translations; heading release
did not activate. Timed execution was 297.875 s, total process time 421.59 s,
maximum RSS 16,638,996 KiB and swaps zero.

Direct reached the goal earlier than JEPA on this layout (130.4 versus 156.8 s),
but completed the round trip later (296.8 versus 239.1 s). These are individual
development trials, not an established timing advantage. Direct layout 3 remains
live; its result and archive precede the final supervised-rollout pair.

## Direct layout 3 verified goal; final supervised pair launched

Direct layout 3 exited 0 after 4,805 camera pairs. Independent evaluation verified
goal arrival at frame 4,256, with a one-second dwell 18.58–21.84 mm from the goal,
maximum 100-ms speed 0.01552 m/s and all requested intervals zero. Return did not
complete; final physical home distance was 3.833 m. There were zero disallowed
contacts. Median/maximum position error was 4.75/8.80 mm; 1,145/1,187 plans were
on time. Path length was 25.718 m, timed execution 481.879 s and full process
duration 690.68 s. Maximum RSS was 25,341,684 KiB (24.17 GiB), with zero swaps.
The completed eight-minute compact recording therefore used substantially less
peak memory than the earlier approximately 38-GiB full serial recordings, in
line with dropping 13.75 GiB of duplicate arrays. These were different trajectories,
not a controlled peak-memory microbenchmark.

The complete direct set has 3/4 goal arrivals and 1/4 round trips; JEPA has 3/4
and 2/4. All eight runs had zero disallowed contacts. Each set contains two serial
and two compact/parallel trials. The sample is small and does not establish a
training-method advantage.

After both direct owners finished and physical evaluations/summaries were saved,
launched the final supervised-rollout pair using the same compact launcher and
CPU allocations on layouts 2 and 3. Complete these, run the existing three-way
comparisons for layouts 2 and 3, and summarize all four matched layouts before
the prospective survey-repositioning experiment.

## Supervised layout 2 verified round trip; three-way layout 2 complete

Supervised rollout exited 0 with 3,082 camera pairs. Physical evaluation verified
goal arrival at frame 1,933 and return at 3,080. Goal dwell was 4.31–7.55 mm and
home dwell 10.27–17.04 mm; maximum 100-ms speeds were 0.01548 and 0.01708 m/s,
with all requested intervals zero. Contacts were zero. Median/maximum position
error was 3.52/8.17 mm; 737/746 plans were on time. Path length was 21.200 m and
final home distance 17.20 mm. Timed execution took 309.065 s; full process duration
was 436.67 s, maximum RSS 17,110,872 KiB and swaps zero.

The existing three-way comparator completed successfully on layout 2, checking
model/correction assignments and equal non-treatment launch fields, including
source identities and the compact/parallel execution profile. All three models
have verified round trips on this layout. The PNG/SVG comparison was generated
and visually inspected: supervised rollout explored the northern dead end before
using the southern route, then omitted that detour on return. This is observed
backtracking/route-use evidence, not a causal memory-ablation result. Layout 3's
supervised run finished its timed budget with no observed arrival; its archive
and physical evaluation are still pending.

Its saved decisions identify a different deadlock: every one of the 501 plans
in frames 2,000–4,000 selected hold, all in `FRONTIER_STANDOFF_REQUIRES_VIEW`,
while the preferred action was right turn. There were no active alternative-turn
latches and no full-reserve translating candidates. At frame 3,000, hold had
minimum forecast clearance 0.45211 m; left/right turns had 0.45292/0.45162 m,
below their 0.48-m reserve requirement, while all three translations forecast
encroachment within the nominal 0.45-m footprint. The prepared repositioning rule
does not cover this case. Preserve it as a separate failure rather than claiming
that one recovery change addresses all observed stalls. Reports:
`middle_window_decision_diagnostic_v1.json` and
`saved_scan_recovery_2000_4000_v1.json` in the supervised layout-3 root.

## Twelve-trial comparison complete

Supervised layout 3 exited 0 after 4,805 camera pairs. Independent evaluation
confirmed no arrival and zero disallowed contacts. Closest/final goal distances
were 1.270/2.309 m; path length was 5.889 m. Median/maximum position error was
4.75/9.16 mm, and 1,168/1,200 plans were on time. Timed duration was 481.914 s,
full process duration 699.02 s, maximum RSS 25,239,968 KiB, and swaps zero.

The layout-3 three-way comparison passed, completing all four matched comparisons.
Their results are combined in
`go2_matched_training_cohort_layout00_03_v1_attempt_001/result.json`, retaining
all twelve outcomes and the serial/parallel execution distinction.

| Method | Verified goals | Verified round trips | Disallowed contact samples |
| --- | ---: | ---: | ---: |
| JEPA | 3/4 | 2/4 | 0 |
| Direct | 3/4 | 1/4 | 0 |
| Supervised rollout | 2/4 | 1/4 | 0 |

All four three-condition trajectory figures were generated and visually inspected.
Failures include prolonged turn recovery, hold deadlock and late goals leaving
insufficient return time. The experiment demonstrates learned-model navigation
and some physical backtracking, but not reliable completion, a statistical JEPA
advantage, causal memory benefit, realistic sensing or hardware validation.

After both final owners and the summaries/figures completed, launched
`go2_survey_reposition_matched_jepa_native_layout00_4800_v1_attempt_001` using
`scripts/run_go2_survey_reposition_native_development.py --layout-index 0`.
Both prior owners were absent, available RAM was 76.79 GiB, RecoveryStorage free
space was 55.13 GiB and CPU affinity included all 32 logical CPUs. The original
serial capture profile is retained for this first controller experiment against
the diagnosed serial JEPA layout-0 reference. Its outcome is not yet known.
