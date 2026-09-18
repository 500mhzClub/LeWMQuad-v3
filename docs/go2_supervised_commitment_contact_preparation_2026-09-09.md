# Supervised commitment-contact replay preparation

All41 focused tests passed (2.37s), covering the new scoring, full-decision causal
comparison, replay reader boundary, verification wrapper and original executed
waypoint behavior. During pre-freeze development, the controller inheritance test
rejected an assumed ExecutedWaypointSelector: the actual original selector is
ViewReentrySelector. The new class now extends that exact original class, keeping
its translating view recovery and inherited final-goal handling. No experiment
ran with the rejected development constructor.

Frozen source identities before read-only preflight43825:

| Path | SHA-256 |
| --- | --- |
| lewm/commitment_contact_score_development.py | cb51e94a31fc748592212811a7a7085356fd9aab3c4bae6653b41a18dda562b1 |
| lewm/commitment_contact_controller_development.py | f549494e10b947e44ceddc20dbcdeb7808d51e381e054280db3c27d583c03a93 |
| lewm/commitment_contact_prefix_development.py | 2074f7fa732784dfaf63e85f31d66172a08e45d19a9f51c89b06e815ecafe06d |
| scripts/replay_go2_supervised_commitment_contact_prefix_v1.py | 518e7eecf8b29647747c02b1a2b3965876d84818bbb8ec676766fd73ea10fc59 |
| lewm/tests/test_commitment_contact_development.py | 9d409c01a77e11d82c49891027c319edd2a01907287df4f62b6d6bb061a37bfb |
| lewm/tests/test_commitment_contact_replay_development.py | 1de8a23352875169f8022d9ac92458457084474ed46019741b5c9cca84a48002 |
| docs/go2_supervised_commitment_contact_prefix_v1_2026-09-09.md | 3302669aa015d260f4558e63edaa42dc0f6f0253d73863331c18f16962131459 |

Read-only preflight uses the original Genesis interpreter, complete explicit
PYTHONPATH, PYTHONDONTWRITEBYTECODE=1, PYTHONHASHSEED=0 and one OMP/MKL/OpenBLAS
thread. No output directory or native scene is created by preflight. Current
original supervised cohort19047 remains on maze2/worker2563586; scheduler37343
waits for that complete cohort before its original three ordered pilots.

Latest native hardware monitor before preflight:73,378,967,552 bytes available
RAM,696,138,268,672 artifact bytes free,21,357,879,296 workspace bytes free,
5.4% CPU busy with all32 logical cores available. Card1 GPU0% and1,667,854,336
of34,208,743,424 VRAM bytes used. New runner measures its own hardware both before
verification and immediately before execution; one8GiB CPU replay allowance
plus32GiB native allowance,4GiB output plus40GiB reserve. This is resource
admission, not an enforced memory limit.

The completed first supervised episode and all-selection diagnostic motivate the
new horizon; no new command has yet been physically executed. Aggregate remains
24 completed raw-audited native episodes and zero verified round trips.

Read-only43825 exited0. All1721 source bindings and original verification
conditions passed. Scoped counters:944,482 requests,806,618 guarded cache hits,
137,864 unique files,64,214,606,916 bytes hashed initially and again finally,
23 isolated verification functions, no imported globals changed and no retained
cache. No output was created. Final preflight available RAM72,561,979,392 bytes,
artifact free695,725,502,464 bytes, workspace free21,357,916,160 bytes,3.4% CPU
busy, full32-core affinity and card1 GPU0%. The exact frozen runner is now
submitted without --preflight-only; it will reauthenticate before exclusive
launch and after the causal replay. A completed replay is not yet assumed.

Actual93799/PID2568108 passed the same input verification and launched with SHA
`a412b59f4f865daf8920bd1a3f894a4fa3f685985c4d1130c1ea38cc19407e18`.
Narrow read-only16907 exited0 after observing a closed four-row decision stream,
SHA`deedbdd7ae5840c632c72495d0797011bad851506e58a05780bd56946dff8735`,
unchanged before/after inspection. At frame3, original right turn[0,0,-.45]
becomes right arc[.16,0,-.45], with no candidate terminal or failure. The saved
complete comparison passes and stops; no fifth observation is present. This is
an intermediate observation while93799 performs final source/input verification,
not yet an authenticated completed result or any physical-execution claim.

At the boundary, new utilities in metres: hold.003967728656195596,
forward.011695974077733862, left arc.0064026730621350204,
right arc.013119958975441482, left turn.0002799640512121186,
right turn.007216563819274523. Right arc's100ms contact score is
.0001236472229186693, versus its retained800ms score.006997195675243411.
Forward's corresponding scores are.0001424375594345399 and.021508392177601945.
Neither contact score is a calibrated probability.

The comparison establishes equality of complete saved observed-state and residual
receipts through the boundary. Internal pending next-step forecast bookkeeping
necessarily refers to the newly selected action; no identity of that unobserved
future label, or of state after executing a different action, is claimed.

Replay93799 subsequently exited0 with result SHA
`37b29828635e88fab77f81447f6a05b890911426fc3478d8aac451426a229de0`.
Independent49502 authenticated all sources/output bindings and reexecuted all
four saved comparisons. The completed result supersedes the pending observations
above; see docs/go2_supervised_commitment_contact_prefix_result_2026-09-09.md.
