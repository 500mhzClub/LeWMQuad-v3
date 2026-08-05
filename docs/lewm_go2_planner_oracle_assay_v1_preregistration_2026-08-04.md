# Go2 planner-oracle assay V1 preregistration

Date: 2026-08-04

Status: preregistered before the 24-scene result was generated. This is a
development-only mechanism assay, not held-out evaluation or a promotion run.

## Question

Can the existing nine-primitive, receding-horizon execution seam turn a correct
candidate ranking into better local navigation? This assay isolates that
question before another visual representation is trained.

The positive-control scorer uses the privileged scene grid, current planar pose,
and target coordinate to rank the nominal kinematic endpoint of every candidate.
`oracle_shuffled` applies the same score multiset to a deterministic permutation
of the unchanged candidate rows. `hold` and `random` are negative controls.
`bearing` is a privileged, historically saturated ceiling and is reported but is
not a superiority gate.

## Scope and non-claims

- Only `.generated/scene_corpus/go2_generalization_v4/development` is eligible.
  V4 is permanently development-only. Its materialized scene manifests retain
  the legacy label `candidate`; that label grants no held-out status.
- The assay uses kinematic execution. Here the endpoint scorer is an exact
  positive control for the nominal transition and occupancy veto. It is not an
  oracle for physical locomotion dynamics.
- Kinematic fall rate is structurally uninformative and is not a safety gate.
- The task is local control to an already visible beacon from about 1.2 m. It
  does not establish global navigation, persistent memory, or planning utility
  of a learned representation.
- The checkpoint is loaded because the benchmark currently requires it, but
  none of the five registered arms consults the checkpoint or rendered image.

## Fixed materials and configuration

- Repository base commit: `4adcbdad16baa81c93112e9f0f4a6aa643008fad`
- Checkpoint:
  `models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9_b050000.pt`
- Checkpoint SHA-256:
  `862c99311ec271d4484c93d54369b9e8ff7ea4fbdef8c8888a6c78eac9a0f66b`
- Corpus family: `go2_deployment_medium_maze`
- Scene selection: sorted development scenes, offset 0, limit 24, one trial per
  scene. All 24 were checked as physically eligible/reachable and matched the
  semantic hashes in `config/go2_generalization_v4/development.json` before the
  smoke run.
- Task/mode: `visible-beacon`, kinematic CPU execution.
- Seed: 7.
- Horizon: 1.
- Budget: 12 primitive blocks; goal radius 0.35 m; standoff 0.85 m; approach
  distance 1.2 m; start-yaw jitter 0.7 rad.
- Ordered primitive vocabulary: `hold`, `forward_slow`, `forward_medium`,
  `forward_fast`, `arc_left`, `arc_right`, `yaw_left`, `yaw_right`, `backward`.
- Arms, in reporting order: `oracle_mpc`, `oracle_shuffled`, `bearing`, `hold`,
  `random`.
- Candidate limit: unset. At horizon 1 this yields exactly nine ordered rows.
- Occupancy grid: 0.05 m cells and 0.20 m inflation.
- Tie rule: the first candidate in registered order is selected among costs
  equal within the implementation's fixed tolerance. Action disagreement is
  defined by positive first-action regret above that tolerance, not by differing
  from the first tied row. Tie multiplicity and optimal first-action set are
  recorded.
- Shuffle: the benchmark's named/versioned deterministic score permutation,
  seeded by 7 and block index. Candidate rows themselves never move.

The exact benchmark, analysis, and scorer-seam source hashes are bound in the
freeze record below after review fixes and before the claim-bearing command.

## Outcomes and analysis

The analysis unit is a complete scene: all five arms must be present for that
scene. Any skipped scene or incomplete pairing invalidates this run; it is not
silently dropped.

Primary paired contrast: `oracle_mpc - oracle_shuffled`.

- Final distance in metres (lower is better).
- Progress in metres (higher is better).

Secondary outcomes are success rate, path efficiency, and contrasts against
`hold` and `random`. `bearing` is shown only as a ceiling/reference. Per-decision
first-action regret and regret-positive rate diagnose whether the intervention
actually changed useful choices.

For each scene-level paired mean difference, use 10,000 whole-scene bootstrap
resamples with replacement, RNG seed 2026080401, and the 2.5/97.5 percentiles.
No per-tick pseudo-replication is permitted.

## Fixed gate

Horizon 1 establishes usable planner headroom only if all of the following hold:

1. There are 24 complete paired scenes and no skipped scenes.
2. Every `oracle_mpc` decision has first-action regret at or below the fixed
   numerical tolerance.
3. The shuffled intervention is non-identity, has a regret-positive rate of at
   least 0.25, and mean first-action regret of at least 0.02 m.
4. `oracle_mpc` improves mean progress over `oracle_shuffled` by at least 0.15 m,
   and the 95% whole-scene bootstrap interval for that improvement excludes 0.
   (The paired final-distance contrast is the sign-reversed consistency check
   because all arms share a start.)
5. `oracle_mpc` has greater mean progress than both `hold` and `random`.

Equality with or failure to beat `bearing` is not a failure: bearing already
saturates this kind of short visible-target task. Success rate alone is not the
primary gate because the 12-block budget and 0.35 m threshold discretize an
otherwise continuous control result.

## Conditional horizon-2 rule

Horizon 2 is not an automatic second try. It may be run once, with every other
setting unchanged, only if horizon 1 fails the outcome gate while its plumbing
invariants pass and the recorded action bank shows a specific one-step inability
to reorient then translate (or to route around an occupancy veto). The diagnosis
and the horizon-2 command must be written before viewing its result. A generic
weak effect, implementation failure, or noisy result does not authorize horizon
2.

## Decision rule after this assay

- Pass: implement one frozen dense-DINOv2, temporal, action-conditioned scorer
  and test whether its candidate ranking preserves enough of the demonstrated
  oracle headroom. Do not start a model sweep.
- Fail with the registered horizon mismatch: exercise the conditional horizon-2
  assay once.
- Otherwise: stop model work and revise the action/target/benchmark seam. More
  data or a larger encoder is not a remedy for a failed positive control.

## Claim-bearing command

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/benchmark_lewm_closed_loop_mpc.py \
  --checkpoint models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9_b050000.pt \
  --scene-corpus .generated/scene_corpus/go2_generalization_v4 \
  --split development \
  --family go2_deployment_medium_maze \
  --scene-offset 0 \
  --scene-limit 24 \
  --trials-per-scene 1 \
  --task visible-beacon \
  --mode kinematic \
  --backend cpu \
  --model-device cpu \
  --horizon 1 \
  --max-blocks 12 \
  --goal-radius-m 0.35 \
  --goal-standoff-m 0.85 \
  --beacon-approach-distance-m 1.2 \
  --beacon-start-yaw-jitter-rad 0.7 \
  --seed 7 \
  --primitive-names hold,forward_slow,forward_medium,forward_fast,arc_left,arc_right,yaw_left,yaw_right,backward \
  --policies oracle_mpc,oracle_shuffled,bearing,hold,random \
  --output .generated/oracle_mpc_assay_v1/full_development_24scene_h1_seed7.json
```

## Freeze record

Completed before executing the claim-bearing command:

- benchmark source SHA-256:
  `596ed8bae689573da1d1ca74c915fa365d9775e6701e6b34f4e46ff454e29867`
- analysis source SHA-256:
  `4f979e4ad8bbd20b7e05e0461624741f2884ea3ffce27312a83748751b054cd3`
- LocalMPC scorer source SHA-256:
  `ef796470f438bc8088f348dddad875e214bbc1391b8ad747a16ccd79b47e7eff`
- focused tests: 33 passed in 0.84 s (`test_closed_loop_mpc_oracle_assay.py`,
  `test_local_mpc_candidate_scorer.py`, `test_planning_refactor.py`, and
  `test_analyze_go2_planner_oracle_assay_v1.py`); both source entry points
  compiled and `git diff --check` was clean.
- reviewed two-scene replay smoke: all structural/intervention checks passed;
  all scientific criteria passed except the intentionally unmet 24-scene-count
  criterion.
