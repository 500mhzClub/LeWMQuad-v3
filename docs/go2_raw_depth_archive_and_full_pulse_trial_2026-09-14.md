# Raw depth archive and full-maze pulse trial

The three local terminal-control recordings left 10.77 GiB free, below the
approximately 16-GiB size of recent full navigation recordings. The new
explicit `in_memory_paired_camera_development.v2_raw_depth` format stores
native float32 optical depth without duplicate derived depth/validity arrays.
It retains RGB, body and gyro recordings, execution/physics records, capture
identities, raw pixel hashes and a typed digest of each original complete
public depth packet. `PublicReplay` reconstructs depth and checks its complete
packet digest plus native/RGB pixel hashes. V1 reading remains supported.
The new `RawDepthPairedCameraSession` inherits the same timed acquisition;
the storage change only occurs after native execution.

Verification used all 907 frames from the 650-mm local pulse recording.
The actual writer produced temporary V2 archives and the actual replay reader
reconstructed all public packets. Every value, type, array dtype, shape and
array byte matched V1. Deliberately altered packet witnesses and raw pixels
were rejected. Original depth storage was 3,082,133,770 bytes; V2 used
1,539,649,917 bytes, saving 50.046%. Four workers completed validation in
29.49 seconds. Temporary regenerated copies were removed; originals remain.
Result: the source root's `raw_depth_archive_verification_v1/result.json`.

ZIP member metadata from all 4,805 frames of the preceding full native run
estimates an 8.430-GiB V2 total versus its 16.356-GiB original flat artifact
size, before the small added packet-witness metadata. This is a measurement
of the existing recording's compressible native depth, not a bound for a
different trajectory. The source root contains
`future_raw_depth_storage_estimate_v1.json`.

The prospective full trial is
`go2_raw_depth_terminal_pulse_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001`.
It uses the 4,800-interval layout-6 round-trip mission, orthonormal gyro
perception, fine-goal routing, predictive hold, and the new 100-ms terminal
translation treatment. The local comparison proved that one pulse executed
as forecast and achieved an outbound arrival, but showed no advantage over
standard commands. This run tests the treatment on the harder maze approach.
Arrival criteria, learned weights, fixed residual weights, delays, planning
cadence and turn durations are unchanged; translation candidates near the
exact goal use the declared pulse/settling sequence.

Before launch: 16 physical/32 logical cores, affinity 32, CPU 3.1%, 77.00 GiB
available RAM, 10.77 GiB free disk, both GPUs idle, no experiment owner.
One native owner runs, followed by four archive writers. After owner exit
and final archive completion, independently evaluate arrivals and verify V2
public-packet reconstruction. No old recording or scientific result was
deleted, no sealed material was accessed, and no transfer or hardware
qualification is claimed by this development trial.

## Completed full run

The owner exited successfully after 4,805 camera pairs and 4,805 admitted
registered poses. Outbound arrival at frame 1,499 passed the independent
physical check: distance 11.60–13.73 mm throughout the one-second dwell,
maximum 100-ms speed 0.01624 m/s, and all requested commands zero. There was
no return arrival before the 480-second budget expired. Final physical home
distance was 1.353 m. No disallowed contact occurred; 1,115/1,192 plans were
on time. This is an outbound success and round-trip failure, with no causal
claim of benefit from the terminal pulse.

The pulse treatment was enabled on 12 plans and selected on 11, all on time.
Nine executed exactly five 20-ms command steps followed by fifteen zero
steps; the other two were suppressed by mission settling. All eleven
settling tails were zero. The run root's
`terminal_pulse_dispatch_diagnostic_v1.json` records each plan.

All 4,805 V2 public packets reconstructed successfully in 27.48 seconds with
four readers. Every complete derived depth-packet digest, native depth/RGB
pixel hash, and public schema/clock check passed. The exact result is
`raw_depth_archive_readback_verification_v1.json` inside the run root.

The return stalled inside the additional clearance reserve: at frame 4,800,
all six candidates cleared the nominal 0.45-m footprint, but every moving
candidate failed the 0.48-m reserve or existing recovery conditions. The
last 100 plans contained 98 holds and two left turns. A left arc had a
forecast minimum clearance of 0.46765 m and final clearance of 0.47135 m;
the existing translation recovery requires restoring the entire reserve.

Registered position error was median 14.32 mm and maximum 91.00 mm. The
`return_pose_drift_diagnostic_v1.json` separates axes and raw/registered
estimates: horizontal drift accumulated during the prolonged stall, from
approximately 1 cm at outbound arrival to 9 cm near the end; height error
remained submillimetre. References were repeatedly promoted for half-feature
overlap, often selecting the immediately preceding frame. Before changing
control clearance, a public-recording experiment disables only that overlap
promotion from frame 1,500 onward. Motion, low-support and qualified fallback
promotion remain available, as do all image/depth/continuity acceptance
checks. The unchanged prefix is compared against saved raw poses; physical
truth is loaded only after estimation. This is a diagnostic replay, not new
closed-loop navigation evidence.

The replay completed with a negative result: the first 1,500 raw poses were
exactly unchanged, but disabling overlap-only promotion lost tracking at
frame 1,542 because the bounded measured bridge exhausted its retained-anchor
support. This treatment is not selected for native execution. Result:
`retained_reference_from1500_replay_v1/result.json`. Frequent replacement
cannot simply be removed without maintaining usable correspondence support.

The controller diagnostic found 181 held plans with an otherwise rejected
motion whose minimum forecast clearance was at least that of hold, whose
entire forecast cleared the nominal footprint, and whose endpoint improved
over hold by at least max(1 mm, 10% of the remaining reserve deficit). The
first was frame 1,600, before substantial pose drift. This is retrospective
eligibility, not evidence that the motion would physically succeed.

The next prospective attempt is
`go2_hold_relative_recovery_pulse_round_trip_native_layout06_4800_v1_attempt_001`
under the workspace volume's `.generated/navigation_development_artifacts_v1`.
It keeps the current perception, pulse controller, learned weights and
mission settings. Only when hold is selected and every moving candidate is
rejected does the new rule consider these improving recovery forecasts,
ranking their minimum clearance first and endpoint clearance second. Each
recovery reselects on the next plan rather than preserving the blocked turn
latch. Current-depth dispatch vetoes remain active. The comparison is against
hold's forecast, not a requirement to avoid any dip below prefix clearance;
no calibrated safety bound is claimed.

Five focused behavioral tests passed: gradual recovery, minimum-clearance
regression rejection, invalid/nominally encroaching path rejection, preserving
already eligible choices, and insufficient-gain rejection. The actual new
selection function changed exactly the 181 saved plans identified by the
diagnostic, with its first change at frame 1,600. The perception experiment's
new default-preserving hook reproduced every prefix raw pose exactly.

Before this native launch: 16 physical/32 logical cores, affinity 32, CPU
0.5%, available RAM 77.07 GiB, workspace free 19.75 GiB, RecoveryStorage free
2.66 GiB, both GPUs idle, and no competing scientific Python job. One native
owner uses its existing tracking/registration/mapping workers; four archive
writers follow execution. Output guards are explicitly bound to the ordinary
workspace artifact directory; model inputs remain in RecoveryStorage.

## Verified layout-6 round trip

The workspace attempt exited 0 after 4,062 camera pairs and registered poses.
Both arrivals passed independent physical evaluation, with no disallowed
contacts. Outbound arrival at frame 1,552 stayed 15.19–18.51 mm from its target
during the quiet dwell (maximum 100-ms speed 0.00904 m/s). Return arrival at
frame 4,060 stayed 3.64–5.81 mm from home (maximum speed 0.01952 m/s). Both
one-second dwells had all requested commands zero. The owner stopped at
406.44 simulated seconds; 873/1,004 plans were on time. Median/maximum
registered position error was 12.35/14.73 mm. Horizontal travel was 21.46 m.

Crucially, the new hold-relative recovery rule activated on **zero plans**.
This is positive closed-loop evidence for the existing gyro/pulse system on
layout 6, not evidence that the proposed recovery rule improved navigation.
Measured asynchronous execution varies between attempts; the preceding
attempt with the existing controller failed to return. One successful repeat
does not establish reliability or causal treatment benefit. Physical arrival
results and `treatment_activation_summary_v1.json` are preserved in the
workspace attempt. Realistic sensor uncertainty, host real-time qualification,
training/planning/memory ablations and hardware validation remain outstanding.

The unchanged controller is next tested prospectively on development layout
4, where the earlier learned and reactive arms failed. The layout-4 attempt
uses the same workspace prefix and settings, with `layout04` in its basename.
This is a transfer test to an existing development layout, not a newly sealed
evaluation. Before launch: CPU 0.6%, 76.80 GiB available RAM, 13.07 GiB free
workspace disk, both GPUs idle and no scientific job competing. Concurrency
remains one native owner with the existing worker arrangement. No recording
was deleted. The CLI's layout parameter is the only launcher change.

An independent follow-up perception experiment has been prepared for recorded
sensor replay. It keeps one preferred stable reference alongside seven recent
references, continues all low-overlap feature refreshes, and advances the
stable reference only after an accepted fallback or the existing motion
threshold. The inherited reference conflict, incremental continuity, camera,
floor and bridge rules still run. It changes the preference among eight
retained references rather than adding an unbounded map or freezing pose.
It is not selected by the live layout-4 run. Its first diagnostic will use the
same frame-1,500 activation on the earlier drifting layout-6 recording, with
an exact unchanged-prefix comparison and physical scoring after estimation.

## Layout-4 transfer failure

The layout-4 owner exited 1 after 3,546 camera pairs and 3,545 registered
poses, with a `Full()` exception in the registration stage while publishing
downstream work. No arrival occurred; no disallowed contact occurred. Median
and maximum pose error were 20.31 and 22.39 mm. Of 835 completed candidate
plans, 430 were on time. The recovery rule activated once, at frame 1,460
(right arc, on time), so it did execute in this different layout.

This stall was caused by missed planning deadlines, not rejection of all
motions: every completed plan from frame 2,000 onward selected left arc but
was late. All six final candidate paths exceeded their full clearance
requirements. Median planning execution rose from 60 ms before frame 1,500
to 446–448 ms after frame 2,000, with maxima around one second. The planner
was still working on frame 3,532 when the final mapping update reached
3,544; this points to planning queue backpressure rather than loss of
visual tracking. Exact event and activation summaries are in
`planning_deadline_failure_diagnostic_v1.json`.

The next diagnostic reconstructs the actual recorded mapping updates at
three planning frames and profiles the fine-goal connectivity fallback.
The separate stable-reference replay is deferred until the current execution
bottleneck is understood. No queue enlargement or deadline relaxation has
been made.

Profiling confirmed the fine-goal fallback as the bottleneck. At frame 1,600
the goal cell was unobserved and the fallback returned immediately. At 2,100
and 3,500 it was observed but disconnected under the clearance constraints;
each fallback made 3,000 exact clearance queries and returned the original
frontier route. Both maps contained the same 6,703 floor cells and 4,284 fine
obstacle cells. Profiles and the reconstructed maps are in
`fine_goal_search_profile_v1/` inside the workspace layout-4 root.

`CachedFineGoalRecoveryRuntime` caches exact segment endpoint queries for up
to 16,384 segments on each of two complete obstacle-cell sets. Endpoints are
not rounded. Changed obstacle sets cannot reuse old geometric results. The
original A* function, edge predicates, costs, routes and acceptance decisions
remain in use. On the saved maps, ordinary searches took about 192 ms;
warm cached searches took 26.6–27.2 ms with the entire returned route equal.
The cold cached call took 274 ms, so first-use latency remains a limitation.
Two focused tests passed, including successful-route equality, obstacle
invalidation and distinct close segment coordinates.

The next native attempt is
`go2_cached_fine_goal_lzma_hold_relative_recovery_pulse_round_trip_native_layout04_4800_v1_attempt_001`
on the workspace volume. Existing perception, action selection, delays and
deadline checks remain active. Before launch: 16 physical/32 logical cores,
affinity 32, CPU 0.8%, available RAM 76.97 GiB, workspace free 7.23 GiB, both
GPUs idle and no scientific job competing.

To fit another full recording without deletion, this attempt uses standard
ZIP_LZMA compression for the same raw-depth NPZ arrays. On 32 archives sampled
uniformly from the full layout-6 recording, LZMA used 34.76% of the original
DEFLATE-level-1 bytes; every decompressed NPY member was byte-identical.
DEFLATE level 9 saved only 3.79%. The actual revised NPZ writer and NumPy
reader also reproduced all bytes of six full native arrays across both
cameras and three frames. The V2 pixel/packet witnesses and live acquisition
are unchanged. Twelve archive workers run only after native execution;
the timed controller worker arrangement is unchanged. Sample and actual-writer
results are preserved in the RecoveryStorage full raw-depth layout-6 root.
The sample predicts roughly 3 GiB for a full recording, not a trajectory size
bound. No old archive or scientific result was removed.

A current reactive comparison is prepared in
`run_go2_matched_pulse_reactive_native_development.py`. It uses the same
orthonormal gyro perception, observed memory, clearance-preferred routing,
cached fine-goal connectivity, 300-ms dispatch delay, 400-ms planning cadence,
and 100-ms translations within the same exact-goal approach region. It selects
the six primitives from current waypoint feedback without a world model or
motion residual. Predictive hold and forecast-based clearance recovery are
absent. Two focused tests confirm model-free selection and matched near/far/
survey command durations. This comparison concerns the complete predictive
selector, not JEPA-specific training benefit; it has not yet been executed.

## Cached layout-4 outcome

The cached attempt exited 0 and saved all 4,814 camera pairs and registered
poses. It exhausted the mission budget without either arrival and had zero
disallowed contacts. Median/maximum position error was 14.81/17.57 mm; final
physical outbound-goal distance was 1.573 m. Travel was 10.50 m. Only
458/1,198 completed plans were on time. This is another navigation failure,
despite completing the run without the previous queue exception.

Late-run planning computation was now approximately 98–100 ms, rather than
the preceding attempt's 446–448 ms median. However, tracking processing grew
to a 100-ms median at the 100-ms camera cadence, with occasional much slower
frames. Its median completion age rose from 116 ms before frame 1,800 to
1,859 ms after frame 4,000. Planning then completed at a median age of
2,020 ms. Every plan after frame 4,000 was late. Detailed per-stage durations
and observation ages are in `stage_timing_diagnostic_v1.json`. The two live
runs took different trajectories, so their global success/timing differences
are not a controlled estimate of cache benefit; exact-map replay established
the cache's numerical equivalence and warm-search speedup separately.

The LZMA archive completed successfully, using approximately 3.56 GiB of
workspace space including all other run artifacts. PublicReplay verified the
complete packet digests, native/RGB pixels and clock/schema checks at frames
0, 1,000, 2,000, 3,000, 4,000 and 4,813. This sampled readback is recorded in
`lzma_public_replay_sample_verification_v1.json`; it is not a full raw-sensor
audit. Workspace free space is now approximately 3.67 GiB.

Two independent CPU-only replays are running after owner exit: an unchanged
tracker replay profiling frames 3,200–3,219 of this recording, and the stable
reference treatment from frame 1,500 on the earlier drifting layout-6
recording. The former compares every prefix raw pose with saved output; the
latter scores physical error only after estimation. Before these jobs, CPU
was 0.5%, available RAM 76.78 GiB, affinity 32 on 16 physical cores, and no
scientific process remained. Each replay uses one OpenCV/BLAS thread and
writes only small diagnostics. The prepared reactive trial is deferred until
these perception findings and storage allowance are resolved.

## Perception replay findings and next experiment

The unchanged tracker replay completed all 3,220 requested frames with every
raw pose and reference choice exactly equal to the saved native recording.
Profiling frames 3,200–3,219 took 1.830 s including profiler overhead. Floor
candidate extraction alone accounted for 0.529 s across 40 camera calls.
The input packet serialization microbenchmark measured a 4,927,977-byte
packet at median 0.411-ms encoding and 0.173-ms decoding; this excludes pipe
transfer and scheduling and is not a full IPC measurement.

The stable-reference replay also completed: all 4,805 frames were admitted,
with the first 1,500 raw poses exactly unchanged. Maximum registered position
error fell from 91.00 mm to 13.47 mm, and final error from 89.63 mm to 8.87 mm.
Post-activation median error fell from 26.61 mm to 9.39 mm. The median selected
reference age was 78 frames and maximum 621 frames. All low-overlap refreshes
continued. This supports a perception improvement on a fixed recording; it
does not establish closed-loop navigation benefit. The stable-reference
treatment remains absent from native launchers.

A first sparse NumPy floor extraction produced identical candidate/mask bytes
on 28 real camera observations but reduced median cost only from 12.83 ms to
12.10 ms; it remains inactive. A subsequent Numba kernel compiles the existing
quad predicates with `fastmath=False`, preserving all original acceptance
thresholds and dense fallback near numerical boundaries. It avoids repeated
array allocations and retains the original final candidate-point arithmetic.
On the same 28 recorded camera cases, all candidate and mask bytes matched,
with median cost reduced from 12.01 ms to 1.73 ms. Both height/normal-boundary
and missing-neighbour tests passed. Float32/float64 compilation took 0.63 s
and runs before timed acquisition. A complete tracker equivalence/profile
replay through frame 3,219 is running; native use awaits that result.

Both learned and matched-reactive launchers now accept the explicit
`--jit-floor` option. Neither has run with it yet. The current scientific
sequence is to establish the kernel's complete tracker equivalence, then
test native timing and navigation without simultaneously adopting stable
reference selection. The separate stable-reference treatment can follow.

The user has been asked about the exact lossless recompression proposal in
`go2_recent_depth_archive_recompression_proposal_2026-09-14.json`. It covers
only depth NPZ files in three recent named recordings, estimates 12.77 GiB
reclaimed across the two volumes, and preserves every array/member and all
other files while changing compressed container hashes. Approval is pending;
no existing archive has been replaced. Earlier approval was specific to GSD
cache retirement. Replays and source work continue independently.

The compiled-floor tracker replay has now completed all 3,220 requested
frames with zero raw position/rotation differences, no reference-choice
differences and no failure. The same profiled 20-frame window decreased from
1.830 s to 1.315 s (about 28% less profiled execution time). These separate
profiling runs are not a paired live timing experiment. The controlled
per-kernel comparison and full replay equality support prospective native
use; no native run with this kernel has yet occurred. The comparison is
saved as `jit_floor_tracking_comparison_v1.json` in the cached layout-4 root.

## Stable-reference replay on layout 4

The stable-reference treatment, active from frame zero with compiled floor
candidate extraction, completed all 4,814 frames of the cached layout-4
recording without a tracking failure. Physical truth was loaded only after
estimation. Results are in `stable_reference_from0_compiled_floor_replay_v1/`
under that recording's workspace root.

| Registered position error | Recorded tracker | Stable references with compiled floor extraction |
| --- | ---: | ---: |
| Median | 14.81 mm | 8.70 mm |
| Maximum | 17.57 mm | 10.22 mm |
| Final | 15.05 mm | 8.74 mm |

Tracking plus registration took a median 68.72 ms, p95 85.71 ms and maximum
250.80 ms. Total replay elapsed time was 539.40 s, including archive reading
and final scoring. The selected reference age had median 70 frames and
maximum 1,626 frames. Low-overlap and support refreshes remained enabled.
The frame-zero activation has no unchanged prefix; the report's true prefix
equality field is vacuous for this run.

Together with the earlier layout-6 replay, this supports stable reference
selection on two different development recordings. The layout-6 treatment
started at frame 1,500 without the compiled kernel; layout 4 started at zero
with it, so these are not identical protocols. Kernel equivalence was checked
separately through frame 3,219. Neither replay establishes improved live
navigation or concurrent processing deadlines.

The next full native experiment remains compiled-floor layout 4 with the
original reference policy, followed by its matched reactive comparison.
Keeping reference selection unchanged in that experiment isolates the timing
intervention. Stable-reference native evaluation can follow. No simulation is
currently running. The exact archive recompression proposal remains pending;
no existing recording has been replaced or deleted.

The exact proposed recompression is now implemented in
`scripts/recompress_recent_depth_archives_development.py`. It binds the named
proposal by hash, compresses bounded batches with four workers, verifies NPY
member bytes before atomic replacement, records old/new container hashes, and
verifies each reconstructed packet against its original capture witnesses.
Workspace recordings are processed first. An interrupted attempt leaves a
journal and is not automatically restarted. Three temporary-file tests passed,
including preservation of NaN payload bits and signed zero, and refusal to
replace an original or prepared output that changed after preparation.

This command is prepared but has **not** been executed; it requires the pending
user approval:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lewm_genesis:lewm_worlds .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/recompress_recent_depth_archives_development.py --apply-approved-proposal
```

Current free space remains approximately 3.67 GiB on workspace and 2.65 GiB
on RecoveryStorage. No recording was replaced during implementation or tests.

## Storage resolved and compiled-floor native trial started

The user subsequently authorized bounded retention and deletion of unnecessary
development depth. The completed cleanup removed 103,096 depth NPZ files from
28 superseded runs, recovering 143.97 GiB while preserving result/failure records,
all other file types, active replay recordings and selected comparison cohorts.
See `go2_development_artifact_retention_2026-09-14.md` and
`go2_depth_retirement_result_2026-09-14.json`. The earlier recompression proposal
is superseded and was never executed. The uncached workspace layout-4 recording
now has intentionally retired depth; its diagnoses and outcomes remain.

The prepared native trial started with `--layout-index 4 --cached-fine-goal
--lzma-archive --jit-floor` in
`go2_jit_floor_cached_fine_goal_lzma_hold_relative_recovery_pulse_round_trip_native_layout04_4800_v1_attempt_001`.
It keeps the original reference-selection policy. Prelaunch resources were
0.3% CPU utilization, 16 physical/32 available logical cores, 76.78 GiB available
RAM, 9.02 GiB free workspace storage and 141.21 GiB free RecoveryStorage, with
no other scientific Python owner. One native scene uses its existing perception
and planning workers; the matched reactive trial follows owner completion.
Launch handle 10625, PID 3354960. Outcomes remain pending.

The compiled-floor learned layout-4 owner exited 0 and saved all 4,805 camera
pairs and registered poses. It had 1,134/1,187 on-time plans (95.53%), versus
458/1,198 (38.23%) in the preceding cached run. After frame 4,000, median tracking
duration/completion age was 64/100 ms, versus 100/1,859 ms previously; median
planning completion age was 208 ms versus 2,020 ms. Different closed-loop
trajectories prevent treating these ratios as a controlled cost estimate, but
the new run did not develop the old sustained backlog.

Navigation still failed. The controller reported outbound arrival at frame 4,100,
but evaluator-only physical distance during the quiet dwell was 62.93–69.50 mm,
outside the 40-mm requirement. Requests were all zero and maximum 100-ms speed
was 0.01510 m/s. Median/maximum position error was 38.45/52.80 mm. No disallowed
contacts occurred. The mission exhausted its budget during return, with final
physical home distance 2.138 m and path length 22.444 m. This is a false outbound
arrival and no verified round trip. All physics checks and timing windows are
saved in the new root. Timed execution was 481.89 wall seconds; raw archive
persistence completed afterward.

The matched reactive trial then started with `--layout-index 4 --jit-floor`,
unchanged original reference selection, cached fine-goal routing and LZMA raw
archives. Root:
`go2_jit_floor_cached_fine_goal_lzma_pulse_reactive_round_trip_native_layout04_4800_v1_attempt_001`.
Prelaunch CPU was 0.5%, available RAM 76.86 GiB and workspace free space 5.62 GiB;
the learned owner was absent. Handle 29740. The stable-reference treatment is
still excluded from this matched pair and remains a subsequent experiment.

The reactive owner exited 0 with 4,805 camera pairs and poses, no arrivals and
zero disallowed contacts. Position median/maximum error was 20.06/20.43 mm.
Its closest physical goal distance was 1.250 m and path length 5.991 m, compared
with 0.06156 m and 22.444 m for the learned arm. Neither arm achieved a verified
arrival. Reactive plans were on time for 1,188/1,199 decisions (99.08%). From
frame 1,600 onward, all 801 reactive decisions selected hold; final current
stored clearance was 0.3970 m against the nominal 0.45-m disk requirement.
This was a stored-clearance deadlock, with no sustained tracking backlog.
Its `reactive_stall_diagnostic_v1.json` preserves the selection evidence.

The complete paired summary is in workspace root
`go2_jit_floor_current_pulse_matched_comparison_layout04_v1_attempt_001/result.json`.
The intended mission, timing, perception, routing and pulse settings match, and
all common recorded source hashes match. This compares the complete predictive
selector with reactive feedback; it does not isolate JEPA training, prove
repeatability or establish successful navigation.

`scripts/run_go2_stable_reference_native_development.py` now provides the next
native experiment for either arm, using the already replayed stable-reference
model with compiled floor extraction from frame zero. It retains the same
reference population bound and acceptance checks, runtime classes, mission,
timing, pulse durations and raw-only LZMA archive format. Output is on
RecoveryStorage, which has space recovered by the approved cleanup. The CLI
import/help check passed. The comparison helper now distinguishes stable
reference settings as well.

The learned layout-4 stable-reference trial was launched after both matched
owners had exited. Its root is
`go2_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout04_4800_v1_attempt_001`.
No stable-reference native result is yet claimed.

That stable-reference trial has now completed: outbound arrival physically
verified at frame 2,436, maximum pose error 15.67 mm, zero contacts, but no home
arrival before the budget ended. The final approach repeatedly mixed short
forward pulses with full turns. Its executed-forecast calibration and a concrete
terminal-priority heading diagnosis are recorded in
`go2_stable_reference_and_terminal_heading_result_2026-09-14.md`, which defines
the next arrival-entry-gated terminal-priority intervention.
