# Raw height-cluster floor candidates

Live local-floor mapping trials 0/3 lost near-wall floor candidates and later
failed registration. Exact packet replays reproduced both failures and all
2,379/1,121 admitted raw/registered pose-field rows. At failure the detector
had only 66/51 auxiliary candidates. The transported anchors were 81.0/58.4
seconds old; one candidate in each case exceeded the fixed 3-mm limit, with
maxima 3.0633/3.0520 mm and mean signed residuals -2.0027/-2.2269 mm.
These original failures remain unchanged.

The layout-0 raw-height probe found thousands of points in a coherent narrow
height band at the same observations. The new development selector replaces
pixel-scale mesh-normal filtering with a paired raw-depth height cluster.
It preserves the original stride-four sample grid and nine-valid-pixel
neighborhood, requires points below -0.15 m in the body up direction, selects
the densest 6-mm height interval, and requires at least 25% of the below-body
pool. It fits a common plane and removes residual outliers above 3 mm for at
most eight iterations. Selected points retain original raw coordinates.
The fraction and bounded refinement are new selection rules; this is not
the original untrimmed raw-pool estimator. All exclusions are recorded.
The downstream joint-plane count, extent, alignment and 3-mm residual gates
remain unchanged, as do image/pose acceptance and raw obstacle extraction.
Small coherent populations remain visible to the downstream count/conflict
checks rather than being silently replaced by empty observations.

This is a flat-floor development hypothesis, not a general floor-identity
classifier. A dominant horizontal object surface can still be ambiguous.
The concentration threshold rejects diffuse wall-height populations but does
not establish hardware calibration or support safety.

`lewm/robust_height_floor_candidates_development.py` implements the paired
selector; `lewm/robust_height_floor_tracking_development.py` applies it to
tracker floor estimation, registration and independent obstacle-floor sensing.
The existing local inverse-depth correspondence endpoints are unchanged.
Three focused synthetic tests passed: noisy paired floor with invalid rays,
diffuse vertical wall rejection, and preserving a 49-point population for the
downstream count gate. Five actual layout-0 probes passed the original joint
plane gate, retaining 23,666/4,392/3,953/3,559/3,615 points.

## Failed preceding-pose initialization and current-gyro revision

The first complete-recording attempt, `robust_height_floor_replay_v1` under
mapping layout 0, failed tracking at frame 15 after 15 accepted frames. All
15 independent floor/obstacle observations were available. Its floor estimate
conflicted with retained image fits during tilt. The original tracker uses the
preceding admitted visual attitude for candidate alignment; using that stale
attitude to select a narrow *height interval* is a different and biased operation.

The saved frame-15 component comparison held the pixels and downstream alignment
reference fixed. Previous-pose selection retained 6,968 points and estimated
offset 0.313295 m. Current public gyro selection retained 24,453 points and
estimated 0.318397 m, close to the original live local detector's 0.318372 m.
`frame15_height_seed_diagnostic_v1.json` preserves both outcomes. No rejection
threshold was relaxed and the failed attempt was not relabeled.

`lewm/gyro_seeded_height_floor_tracking_development.py` therefore initializes
tracker candidate selection with the current public gyro attitude. The original
downstream plane alignment reference remains unchanged. Registration already
uses the current image pose, and the independent observer already uses the
current public gyro; neither needs that tracker-specific timing correction.

Two full-recording tests launched with
`scripts/replay_go2_robust_height_floor_development.py --layout-index I --gyro-seeded`
on the retained noisy mapping layout-0/3 trajectories (2,381/1,124 frames).
Outputs are `gyro_seeded_height_floor_replay_v1` in their respective roots.
Sessions 86927/85850 run on separate CPU groups. All actual noisy packet digests
are verified; physics is loaded only after estimation stops for error scoring.
No counterfactual commands are executed. Completed results follow below. Do not claim
native navigation success from these recorded-path tests.

## Completed recorded-path results and fixed live experiment

Both gyro-seeded replays exited 0 and accepted their entire recordings.
Layout 0: 2,381 poses, independent floor/obstacle evidence on 2,378 frames,
median/max/final position error 4.906/6.812/6.812 mm, elapsed 349.33 s.
Layout 3: 1,124 poses, floor/obstacle evidence on every frame,
median/max/final error 4.488/8.036/7.381 mm, elapsed 152.69 s.
Neither tracking nor registration failed. The three missing independent
observations on layout 0 retain bounded-refinement failure rather than
inventing an obstacle observation. The original live failures remain in the
comparison; these recorded-path results do not execute new decisions.

The candidate now proceeds to one prospective noisy run on each existing
development layout 0–3 using
`scripts/run_go2_live_gyro_height_floor_noise_development.py --layout-index I`.
Run pairs 0/1 then 2/3 on the established disjoint CPU groups, waiting for both
archives and owner exits between pairs. Output roots are
`go2_live_gyro_height_floor_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
Fix all four assignments before launch; preserve every failure and do not tune
between them or replace failed assignments.

Only shared floor-candidate selection changes in tracking, registration and
independent obstacle sensing, including current-gyro initialization of tracker
height selection. Keep local inverse-depth image correspondence endpoints,
current local-floor routing geometry, persistent memory, learned weights,
actions, full-3-D 0.3-m/s stop, footprint, arrival checks, 4,800-tick budget and
fixed 2-mm noise recipe unchanged. Preserve and measure additional processing
cost. The primary comparator is the completed four-run local-mapping study,
which had no goals or returns. This is a development revisit, not a fresh
independent maze cohort or hardware qualification.

The full launch writer chain passed: it identifies all three new observers,
explicit raw-pool exclusion and concentration/refinement settings, unchanged
mapping and the original noise assignment. No further full replay is required
before this prospective development test. Live results are pending.

The first live pair started on September 15 around 01:55 local log time:
layout 0 PID 3544585 / session 46157 on CPUs 0–7,16–23;
layout 1 PID 3544682 / session 40656 on CPUs 8–15,24–31.
Both owners and their launch records were confirmed live, with the intended
tracker, registration, independent observer and unchanged mapping treatment.
Before launch the reviewed superseded independent-floor depth retirement
reclaimed 5.563 GiB; idle available RAM was 76 GiB. Wait for both archives and
owner exits, evaluate both outcomes, then run layouts 2/3 without tuning.

Layout 0 exited 1 after 167.97 s wall time with `queue.Full` from tracking
submission (1,084 acquisitions, 1,050 accepted poses). Its sole recorded
pipeline fault is the subsequent tracking clock-close during shutdown, not a
geometric rejection. Independent evaluation found no arrivals or contacts,
median/max position errors 1.881/4.582 mm. The saved stage timing diagnostic
shows tracking start lag growing to 3.244 s; tracking median/p95 duration was
62/142 ms against the 100-ms camera period. This is a live throughput failure,
preserved as assigned, not a successful sensor replay or a replacement run.
Maximum RSS was 7,915,128 KiB, zero swaps. Layout 1 remains live.

During this first pair, a 25.09-s recorded-map diagnostic on CPU 15 examined
the previous mapping layout-2 frontier failure. This briefly shared layout 1's
CPU allocation; host timing remains shared-machine development evidence.

The accepted layout-0 tracker prefix was replayed for profiling on otherwise
idle CPU 0 while layout 1 continued. All 1,050 raw pose witnesses reproduced
exactly (seven identity/pose fields). No physics or counterfactual commands
were used. `tracking_backlog_profile_v1` contains the profile and frame times.
Tracking mean measured duration was 57–71 ms across the earlier frame windows,
then 118.65 ms at frames 900–1050; the backlog grew in this late section.
Profiling frames 900–1049 took 18.656 s including profiler overhead. It found
2,773 retained candidate attempts and 2,354 `local_depth` calls consuming
5.813 s cumulatively, within repeated correspondence depth lifting. The
existing cache ends at every observation, so immutable retained references
are re-estimated at subsequent observations. Bounded reuse across observations
is a concrete next performance hypothesis, requiring identical recorded pose
outputs and a fresh live test; it is not implemented in the fixed live roster.

Layout 1 completed and exited 0 after 698.70 s including recording save.
All 4,805 poses were accepted; independent floor evidence was available on
4,804/4,805 frames. No pipeline faults, arrivals or contacts occurred. The
mission exhausted its tick budget after a sampled horizontal path of 27.583 m;
minimum outbound goal distance was 1.758 m. Median/max pose errors were
5.148/13.307 mm. Of 1,200 selected plans, 1,174 were on time; route statuses
included 601 frontier-route and 526 standoff-view plans. Maximum RSS was
25,477,768 KiB and zero swaps. Both first-pair outcomes and evaluations are
preserved. After both owners exited, layouts 2/3 launched unchanged on their
assigned CPU groups; available artifact space was 9.3 GiB.

Second-pair owners confirmed live: layout 2 PID 3547281 / session 93866
(02:07:46 local initialization log), layout 3 PID 3547397 / session 61545
(02:08:04). Both launch records identify the fixed treatment. Do not evaluate
until each recording finishes and its owner exits; do not restart a timeout.

While the unchanged second pair runs, a separate prospective implementation
in `lewm/retained_depth_cache_tracking_development.py` replaces the tracker’s
per-observation local-depth cache with a 32-entry LRU cache. It holds exact
owned input-array identities alive, makes cached owned arrays read-only and
leaves non-owning views uncached. It reuses only the original local-depth
calculation; no pose estimates, candidate ordering, estimator thresholds,
registration, mapping, planning or commands are changed. Two focused tests
passed, covering exact lift outputs, unchanged pixel values, writable-input
rejection, eviction and non-owning-input updates. This implementation is not
used by any member of the current live roster. The prepared profile command
is `scripts/profile_go2_height_floor_backlog_development.py --retained-depth-cache`;
run it after a native owner exits to avoid competing for its CPU allocation.
Exact recorded-pose equivalence and live throughput remain unverified.

## Complete live roster and cache replay result

All four runs are terminal and independently evaluated: zero goals, zero
round trips and zero contacts. Floor evidence was available on 15,497 of
15,499 receipts. Layout 0 failed from tracking backlog; layouts 1–3 exhausted
their navigation budgets. Layout 2 accepted all 4,806 poses, floor evidence
on all frames, median/max error 5.934/7.603 mm, path 3.069 m; its 1,005
view-budget-exhausted decisions reproduce the separate exploration stall.
Layout 3 accepted all 4,805 poses and floor observations, median/max error
5.824/16.364 mm, path 26.243 m, with no pipeline faults. Their wall times
including archive were 658.22/712.17 s, max RSS 25,419,508/25,506,288 KiB,
zero swaps. All four treatment source sets match; common predecessor source
hashes and non-treatment settings match. The combined result is in
`go2_live_gyro_height_floor_noise_four_layout_summary_v1_attempt_001/result.json`.

The bounded-cache replay completed: all 1,050 recorded raw pose witnesses
matched exactly. Frames 900–1049 took 13.561 s under profiling versus 18.656 s
previously (27.3% lower); this is a recorded computation comparison including
profiler overhead, not live timing qualification. The cache ended at its
32-entry bound, 98,304,000 array bytes, with 13,865 hits and 1,523 misses.
Its complete result is `retained_depth_cache_profile_v1` under gyro-floor
layout 0. No native trials were changed during execution.

Proceed with one targeted prospective cached live layout-0 trial, using the
same 2-mm noise, model, estimator, routing, actions, timing and 4,800-tick
budget. Change only retained local-depth reuse. Preserve the uncached queue
failure and report whether the new trial completes its budget or mission,
its tracking backlog, sensor failures and navigation outcome. This single
trial tests the observed throughput failure; it is not a four-layout
reliability result. No further full replay is needed before this experiment.

The targeted live trial launched using
`scripts/run_go2_retained_depth_cache_noise_development.py` on CPUs 0–7,16–23.
Root: `go2_retained_depth_cache_noise_2mm_native_layout00_4800_v1_attempt_001`.
PID 3549829 / session 61431 confirmed live; initialization log 02:23:59 local.
The actual launch writer chain identifies the cached tracker and unchanged
noise, registration, independent observer and mapping relative to gyro-floor
layout 0. A preceding standalone writer probe lacked the required CPU affinity
and stopped at that check; the correctly affinitized live writer completed.
Only this one prospective cache assignment is scheduled. Preserve its full
outcome and evaluate after archive/owner completion. Cache replay and the
four-run gyro-height experiment are complete; cache live results are pending.

The cached live navigation phase completed its 4,800-tick budget: 4,805
camera frames, no physical stop, no reported arrivals or contacts, and no
tracking-queue termination. It selected 812 plans, 810 on time and two late.
Measured navigation wall time was 481.80 s for 480.96 s simulation time;
these remain shared-host development timings. The owner is still saving its
recording, so final exit, independent arrival/pose evaluation and backlog
statistics remain pending. Do not treat `PACED_NATIVE_PREFIX_COMPLETE` as
archive completion or a navigation success.

The cached owner subsequently exited 0 after 662.48 s including archive,
maximum RSS 25,422,280 KiB, zero swaps. Independent evaluation accepted all
4,805 recorded poses; floor evidence was available on every frame, with no
pipeline faults, contacts or arrivals. Median/max position error was
8.286/11.781 mm. The robot travelled 15.819 m and exhausted its budget; 388
decisions reported view-budget exhaustion. Tracking median/p95 measured
duration was 56/64 ms; maximum start lag 736 ms, final lag 44 ms. Common source
hashes and non-treatment settings match the uncached predecessor. The live
trajectories differ, so this is not a fixed-path live timing counterfactual.
It supports using the bounded cache in subsequent development studies while
retaining exploration failure as unresolved. Final summary and diagnostics
are in the cached root. No cached native owner remains active.
