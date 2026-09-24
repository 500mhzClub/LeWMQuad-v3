# Combined tracking cost reductions for the supervised recovery test

The supervised seed-1402 test with transported-normal floor candidates failed
with a tracking queue overflow at 463 captured frames, before close-wall
recovery exposure. The original tracker needed 112–113 ms per late frame in
that native run against a 100-ms camera period. The fixed reactive test then
finished and was evaluated; neither fixed floor-consumer test reached a goal.

The next single supervised follow-up combines two separately tested cost
reductions: reuse individual retained floor-cloud statistics and avoid one
duplicate deep copy during consensus refitting. The reference policy,
estimator equations, acceptance limits and final independent evidence copies
remain. Source: `lewm/cached_moments_deferred_copy_tracking_development.py`.
Ten focused predecessor tests passed, and the combined class was exercised
in the complete recorded-failure replay below.

## Exact 463-frame comparison

The baseline and combined tracker were replayed sequentially on CPUs
8–15,24–31 after the reactive native owner finished archival. All 463 raw
poses, registered positions, paired-floor constraints and revisit receipts
match exactly. Both match all 429 originally published raw poses. The cache
recorded 4,561 hits, 493 misses and 4,290 pruned reductions, with 32 entries.

| Frames | Baseline median | Combined median | Calls >100 ms, baseline / combined |
| --- | ---: | ---: | ---: |
| 0–199 | 44.44 ms | 44.01 ms | 2 / 1 |
| 200–299 | 89.49 ms | 81.46 ms | 20 / 3 |
| 300–399 | 96.48 ms | 88.05 ms | 18 / 1 |
| 400–462 | 98.66 ms | 88.59 ms | 20 / 1 |

Total replay times were 60.495 s and 58.394 s. These are estimator-only
timings, not hard real-time or native mission-reliability evidence. The
previous cache-only native successes also occurred with an unmodified tracker
under lighter workload, so they do not establish a native cache advantage.

The new comparison lives inside the retained supervised transported-floor
failure, under `gyro_coherent_floor_cached_moments_deferred_copy_replay_v1/`,
including `exact_match_and_cost_comparison_v1.json`. Its baseline is the sibling
`gyro_coherent_floor_batched_consensus_replay_v1/`.

## One fixed native follow-up completed

Run supervised seed 2026091402 on the same exposed layout 1, with the combined
tracker and unchanged transported-normal registration, gyro-conditioned
auxiliary obstacle observer, model/correction, sensor noise, controller,
command guards and 4,800-tick budget. Run alone on CPUs 8–15,24–31.
The original failure remains intact. Report any tracking failure, actual
degraded-turn exposure, translation resumption, arrivals and contacts together.

Launcher: `scripts/run_go2_combined_tracking_floor_recovery_development.py`.
Evaluator: `scripts/evaluate_go2_combined_tracking_floor_recovery_development.py`.
Root: `go2_combined_tracking_floor_recovery_seed_2026091402_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
The owner completed archival and exited successfully after 4:07.05 wall-clock
time, with no swapping. The physical and treatment evaluator passed: 1,598
poses, verified outbound arrival at frame 1,124 and return arrival at frame
1,596, zero contacts, maximum pose error 5.605 mm, and 389 selected learned
plans. Both arrivals passed the physical radius and one-second quiet-dwell
checks. No terminal failure was recorded.

There were zero auxiliary-only guard requests, zero auxiliary-only nonzero
turn requests and zero primary-blind translation vetoes. This is a successful
navigation follow-up on the exposed maze, but it did not exercise degraded
perception recovery. It does not isolate a native causal benefit of the
tracking optimization or replace the original comparison failures. No
independent-new-maze or hardware claim is made.

The next unresolved local-control question is the reactive planner's repeated
right-arc veto followed by a left-turn view recovery. Test recovery in the
vetoed arc's turn direction while retaining the existing view angle and
command guards, as a separate prospective development experiment.
