# Individual floor-cloud statistics: replay and native follow-up

Repeated paired-floor fitting contributes to the tracking queue overflows.
The new replay-only prototype caches each retained raw cloud's mean,
covariance and second eigenvalue, using owned read-only arrays and at most
32 entries. It still validates each pair's gyro, up vectors, pool counts and
extent requirement, and recomputes all pair-dependent normals, residuals and
pruned-subset statistics. No acceptance thresholds or reference policy change.

Source: `lewm/cached_floor_moments_development.py`. Six focused tests passed,
including the two original floor-constraint tests. They cover tilted patches,
outlier pruning, reversed pairs, different clouds, changed acceptance inputs,
read-only ownership, bounded eviction and subset recomputation.

## Recorded direct-seed-1402 failure replay

All 460 raw/registered poses, paired-floor constraints and revisit receipts
match both the original replay and a fresh baseline replay exactly. All 426
originally published native raw poses match. The cache recorded 3,684 hits,
478 misses and 3,940 pruned-subset reductions.

All three replays used CPUs 8–15,24–31 sequentially. The fresh baseline was
faster than the older baseline, so the fresh comparison is the more useful
estimate of the improvement:

| Frame range | Fresh baseline median | Cached median | Baseline / cached calls >100 ms |
| --- | ---: | ---: | ---: |
| 0–299 | 44.97 ms | 44.96 ms | 2 / 1 |
| 300–349 | 95.85 ms | 89.15 ms | 9 / 3 |
| 350–399 | 95.13 ms | 88.21 ms | 11 / 0 |
| 400–459 | 95.73 ms | 88.00 ms | 14 / 3 |

Total replay time was 56.969 s baseline versus 56.672 s cached. The older
baseline took 59.608 s and had late medians near 101 ms. The evidence supports
roughly 7 ms lower late motion-call medians in this fresh sequential pair,
not a substantial overall replay speedup or hard real-time qualification.

Artifacts are within the retained original direct-seed-1402 layout-1 failure:
`gyro_coherent_floor_cached_floor_moments_replay_v1/`, including
`exact_match_and_cost_comparison_v1.json`, and
`gyro_coherent_floor_batched_consensus_replay_v1_moments_control/`.

## Direct native follow-up: verified round trip

The same direct-seed-1402 layout-1 assignment completed with only the
floor-statistics tracker change, on CPUs 8–15,24–31. Original sensor settings,
model and correction, controller, floor acceptance and mission budget remain.
The auxiliary-only recovery change and deferred registration-copy optimization
are not included. This is an exposed-maze follow-up; it does not replace the
original queue failure or constitute new independent-maze evidence.

Launcher: `scripts/run_go2_cached_floor_moments_followup_development.py`.
Evaluator: `scripts/evaluate_go2_cached_floor_moments_followup_development.py`.
Native root:
`go2_cached_floor_moments_seed_2026091402_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`.
Session 17044 completed and archived, exit 0 after 4:09.69, maximum RSS
10,334,200 KiB, no swaps. Physical evaluation verified goal frame 1126 and
home frame 1617, with one-second quiet dwell at both. Maximum physical target
distance during dwell was 19.552 mm at the goal and 14.399 mm at home, within
the original 40-mm requirement. There were 1,619 published poses, zero
contacts, maximum pose error 7.558 mm, and 387/397 on-time plans. The mission
took 162.12 simulated seconds. Actual learned XY/yaw and the assigned model
and correction were verified.

The original run had 3.340 s maximum tracking completion age and a queue
overflow. The follow-up had 558 ms maximum completion age, no pipeline faults
and typical stage-service medians of 56–62 ms. The native trajectories and
scheduling differed: these large native service differences cannot all be
attributed to the cache's isolated 7-ms replay improvement. This is one
successful exposed-maze follow-up, not a hard real-time or reliability result.

All shared settings and 159 common original runtime source hashes match.
`go2_cached_floor_moments_direct_followup_comparison_v1_attempt_001/` contains
the comparison, tracking statistics and inspected PNG/SVG trajectories.
The original failure and new full recording are retained.

## Separate JEPA queue failure: exact replay and native test

The original JEPA-seed-1401 layout-1 failure was also replayed with the
baseline and cached tracker, sequentially on CPUs 0–7,16–23 while the direct
native run/archival used the other CPU group. All 494 raw/registered poses,
paired-floor constraints and revisit receipts match exactly, including all
460 original published raw poses. Late frame medians fell from 100.685 to
92.513 ms (350–399) and 99.543 to 90.391 ms (400–493). Calls exceeding 100 ms
fell from 71 to six across those late frames. Total replay time was 62.371 s
baseline versus 60.509 s cached. Cache totals: 3,847 hits, 515 misses, 4,276
pruned reductions, 32 retained entries. These timing observations include
different phases of activity on the other CPU group.

Both replay directories and `exact_match_and_cost_comparison_v1.json` are
inside the original retained JEPA failure root.

A fixed native JEPA-seed-1401 layout-1 follow-up completed with the same
cache and unchanged original model, controller and sensing settings. It uses
CPUs 8–15,24–31 and preserves the original failure. Launcher:
`scripts/run_go2_cached_floor_moments_jepa_followup_development.py`; evaluate
the completed archived attempt with that script's `--evaluate` option.
Root: `go2_cached_floor_moments_seed_2026091401_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Session 76558 completed and archived, exit 0 after 4:11.92, maximum RSS
10,353,528 KiB, no swaps. No additional tracker change or auxiliary recovery
was included. Physical evaluation verified a round trip: goal frame 1156,
home frame 1629, one-second quiet dwell and maximum target distances of
7.959/12.436 mm. There were 1,631 published poses, zero contacts, maximum
pose error 10.813 mm and 389/399 on-time plans. Simulated duration was 163.22 s.
Actual JEPA model/correction binding and learned XY/yaw use were verified.

Both cache follow-ups succeeded, while both original assignments remain queue
failures in the original 22-run comparison. The four-panel comparison is saved
under `go2_cached_floor_moments_two_failures_comparison_v1_attempt_001/`.

## Fixed original-tracker controls under lighter host load

The original failures ran alongside another native simulation. The direct
cache follow-up overlapped short estimator replays on the other CPU group;
the JEPA follow-up ran without another native simulation. Timing and native
trajectories differ. The cache's isolated estimator improvement is supported
by exact replays, but these two native successes do not isolate it from
host-load and trajectory effects.

To test that ambiguity, run these two original-tracker controls once each,
sequentially on CPUs 8–15,24–31 without another native simulation or estimator
replay. Keep the original BatchedConsensusMotion, model/correction, layout,
noise, floor/obstacle acceptance, planner and budget:

1. Direct seed 2026091402, layout 1.
2. JEPA seed 2026091401, layout 1, regardless of the first control's outcome.

Launcher/evaluator: `scripts/run_go2_serial_tracking_controls_development.py`
with `--layout-index 1 --arm <arm>`; add `--evaluate` after archival completes.
Roots: `go2_serial_original_tracker_{arm}_noise_2mm_native_layout01_4800_v1_attempt_001`.
Both fixed controls completed and passed physical evaluation:

| Model | Original tracker, lighter workload | Cached tracker | Contacts, both |
| --- | ---: | ---: | ---: |
| Direct seed 1402 | Round trip, 157.64 sim s | Round trip, 162.12 sim s | 0 |
| JEPA seed 1401 | Round trip, 146.50 sim s | Round trip, 163.22 sim s | 0 |

The original-tracker direct control had 1,574 poses, goal/home frames
1056/1572, maximum pose error 5.425 mm and 373/387 on-time plans. Both quiet
dwells passed, with maximum target distances 12.759/19.046 mm. Owner exit 0
after 4:03.38, maximum RSS 10,086,904 KiB, no swaps. The JEPA control had
1,463 poses, goal/home frames 972/1461, maximum pose error 8.702 mm and
350/359 on-time plans. Dwell maxima were 17.133/21.456 mm. Owner exit 0
after 3:47.92, maximum RSS 9,594,348 KiB, no swaps.

Both original-tracker controls had median tracking service 62 ms, no pipeline
faults and maximum tracking completion ages 500/490 ms. All shared settings
and 159 common original runtime sources match across original, cached and
serial-control runs within each model. The complete comparison is in
`go2_serial_original_vs_cached_tracking_controls_v1_attempt_001/`.

The cache's estimator cost improvement is supported by exact recorded replays.
There is **no demonstrated native success or mission-speed advantage** over
these original-tracker controls. Host workload and trajectory variation remain
possible explanations for the original failures; these controls do not
separately identify those effects. No original failure is replaced, and all
four follow-up recordings remain full. No native job from this study remains
running. Close-wall floor/perception recovery is the next development focus.

These are diagnostic controls on an exposed maze, not replacements for any
prior outcome, independent-maze evidence or proof of hardware readiness.

Retention update: their completed four-way diagnosis releases full depth from
the four successful follow-ups under
`docs/go2_development_artifact_retention_2026-09-14.md`. All original failures,
exact replays, non-depth records and comparisons remain. The newer combined
tracker and signed-reactive successes remain full recordings.
