# Independent learned floor-transport maze cohort result

The fixed development cohort completed all three layouts in order, with **zero
verified outbound arrivals and zero verified round trips**. These are independent
layout executions with fresh processes, model instances and controller memory.
They are development evidence, not a sealed final evaluation.

Root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_floor_transport_mazes_v1_attempt_001`.
Session11941 closed normally with exit0. Final result SHA-256:
`a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`.
Launch SHA-256:
`3053ca602d8e45700550188a3da12e69c3b83314af5a74f32bc616c5425b91c9`.
Final result binds 1658 sources and 6200 artifacts; wall time3070.694777289871s.
The runner completed input/artifact verification, and all1658 source hashes
were independently rechecked unchanged after completion. No model training.

| Layout | Paired observations | Completed commands | Stop | Verified arrival / round trip |
|---|---:|---:|---|---|
| 1 | 225 | 224 | Current visual evidence unavailable at observation214 | 0 / 0 |
| 2 | 514 | 513 | No action satisfying surface and nominal constraints at503 | 0 / 0 |
| 3 | 275 | 274 | Current visual evidence unavailable at observation264 | 0 / 0 |

Each includes ten zero-command drain ticks. All three pass raw sensor
reconstruction, model/command replay, command audit, model-state preservation
and strict physical visibility checks. Hard measurement failures are empty.
Physical and acquisition stops are absent. These valid negative outcomes remain
preserved; the cohort was neither retried nor changed after launch.

Maze1 diagnosis is recorded in
`docs/go2_independent_maze01_correspondence_diagnosis_result_2026-09-09.md`
and `docs/go2_independent_maze01_match_stage_diagnosis_result_2026-09-09.md`.
Maze2 scoring/feasibility diagnosis is recorded in
`docs/go2_independent_maze02_feasibility_diagnosis_2026-09-09.md`.
Maze3 has not yet received an equivalent correspondence-stage diagnosis.
At observation263 it requested left_turn `[0,0,0.45]`; observation264 returned
`SENSOR_OR_MODEL_FAILURE`, `same-episode current visual evidence required`,
and a zero request. The failure receipt retains tick263, so264 here denotes
the stream observation index. Native evaluation reports no cell crossing,
only the start cell `[-1,0]`, and no arrival window.

Maze3 identities:

- Collection `full_jepa_novel_maze_03/result.json`:
  `67d6a2f9e99b592497712ec53d5dff85a5626abb67e9a6cab2f451bb6ff48420`.
- Raw audit `full_jepa_novel_maze_03_audit.json`:
  `47bc53f3141c65a480fcd2d8e1e01e477c0461920e4f6ba1e8f415401f7c08dd`.
- Worker terminal `full_jepa_novel_maze_03_worker_terminal.json`:
  `ca6553e0441dfffddc91430444c04591d4c0c506c81a28377e4b9fa7f0005311`.
- Progress `cohort_progress_after_03.json`:
  `b308991e374a0df7f4146e284b78b1a5709abd1447d0be05a023cb3e0f9771b8`.
- Decision stream `full_jepa_novel_maze_03/context_decisions.jsonl.gz`:
  `c1b72a7da677efa4cf37d4a39d46a815d38fc9aa37296c0a12ac5b3aae5518fa`,
  rehashed unchanged around inspection.

Maze3 contains14450 physics samples; worker wall833.068659078097s,
peakRSS3,491,966,976bytes. Its model was unchanged. No navigation, real-time,
hardware, learned-planning or memory advantage is established.

Across the current pilot sequence,15 episodes are now complete and audited:
12 executions of the reused development maze plus these3 independent layouts.
The three earlier verified settled outbound arrivals share the same maze0
prefix; they are not three independent-layout successes. Zero verified round
trips remain. Next execute the already prepared reactive maze0 comparison,
its paired readout and independent reactive cohort; retain the planning-memory
comparison and separately validate tracking/feasibility successors.
