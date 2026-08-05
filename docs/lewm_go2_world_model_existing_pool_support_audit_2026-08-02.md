# Go2 world-model existing-pool support audit — 2026-08-02

## Executive conclusion

The existing-pool diagnosis is now narrower and better supported:

- every bound pre-action H6 history is exactly unique, but **local action
  overlap is not absent**;
- the current 16,000-row training pack contains substantial cross-scene local
  overlap for the common actions, while overlap is sharply weaker for
  `forward_fast`, `forward_slow`, `hold`, and `backward` after conditioning on
  scene family and the two requested history actions;
- the full corpus contains hundreds of thousands of the rare forward-speed
  actions, so the 16,000-row result does not establish that the existing
  corpus has exhausted its useful support;
- the physical outcomes are strongly dynamic. Even `hold` exceeds the
  descriptive 2.5 cm-or-0.05 rad motion threshold in 71.1% of selected train
  rows, showing that incoming velocity and controller/body lag materially
  affect the half-second successor;
- exact requested-versus-executed/clipped command differences cannot be
  measured from `frames.jsonl`. It contains requested command context and
  observed body twist, not an executed-command tape or clipping flag.

Therefore, “one action per exact state” is true but is not enough to conclude
that factual learning lacks local intervention support. The evidence supports
an evaluation-first counterfactual set and a larger, stratified existing-pool
comparison before any bulk counterfactual training-data job.

## Scope and custody

The audit opened only:

- the exact SHA-256-bound corrected-H6 V2 train and validation indices;
- the frozen main-pool census receipt; and
- the 1,150 allowlisted public-development train/validation `frames.jsonl`
  leaves identified by that census.

It opened no RGB, raw-message, label, checkpoint, tensor, test, held-out,
sealed, navigation, network, or GPU payload.

Access accounting:

| Item | Count |
|---|---:|
| `frames.jsonl` sources | 1,150 |
| metadata bytes streamed and SHA-256-bound | 138,549,246,020 |
| selected metadata rows parsed | 144,384 |
| bound H6 train rows | 16,000 |
| bound H6 validation rows | 2,048 |
| train / validation scenes | 1,000 / 150 |
| cross-role scene overlap | 0 |
| RGB/raw/label/checkpoint/protected opens | 0 |

The recomputed ordered source-content binding was
`0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696`,
exactly matching the frozen census.

This is development evidence only. It is not blind qualification, promotion,
or deployed-planning evidence.

## Command and artifacts

Executed command:

```text
python3 scripts/audit_go2_world_model_existing_pool_support_v1.py \
  --workers 8 \
  --knn-k 16 \
  --output .generated/dev/world_model_existing_pool_support_audit_v1/result_v2.json
```

Result:

- path:
  `.generated/dev/world_model_existing_pool_support_audit_v1/result_v2.json`;
- byte count: `122,912`;
- SHA-256:
  `2cda59a6ed5effdd22db5a77fca0ab9a98717185efc524c45957b5d317b60b6d`;
- wall time: approximately 29 seconds;
- status: `COMPLETE_DEVELOPMENT_AUDIT`.

Implementation and focused tests:

- `scripts/audit_go2_world_model_existing_pool_support_v1.py`;
- `lewm/tests/test_audit_go2_world_model_existing_pool_support_v1.py`;
- `python3 -m pytest -q
  lewm/tests/test_audit_go2_world_model_existing_pool_support_v1.py`:
  **10 passed**.

The earlier
`docs/lewm_go2_main_pool_action_frame_alignment_audit_2026-07-28.md`
was reused rather than rerun for its already-bound conclusions: action IDs
match requested primitive context, corrected boundary validation action
separability is `0.452270` balanced accuracy, and `frames.jsonl` lacks
executed/clipped command fields.

## Full-corpus scale and discrete support

The frozen census covers 55.2 million metadata rows, 10.96 million primitive
transitions, 10.61 million sliding H6 windows, and 1,807,552 row-disjoint
packed H6 windows. The corrected V2 construction later found approximately
1.74 million causal-valid groups before the per-scene cap.

Full-census train `p2` requested-action counts are:

| Action | Count | Fraction |
|---|---:|---:|
| `arc_left` | 1,600,146 | 17.34% |
| `arc_right` | 664,517 | 7.20% |
| `backward` | 614,813 | 6.66% |
| `forward_fast` | 317,797 | 3.44% |
| `forward_medium` | 2,495,751 | 27.04% |
| `forward_slow` | 258,713 | 2.80% |
| `hold` | 438,137 | 4.75% |
| `yaw_left` | 1,776,604 | 19.25% |
| `yaw_right` | 1,063,360 | 11.52% |

These are full-corpus sliding-H6 census counts, not a recomputed physical
neighborhood audit over all 1.74 million corrected candidates. They establish
large marginal populations for every action, not conditional local overlap.

The 16,000-row pack is very close to those marginal proportions. Its smallest
classes are `forward_slow=447` and `forward_fast=545`, while
`forward_medium=4,303`. Scaling the same uniform selection would therefore
add many rare-class rows, but would preserve the approximately 10:1 class
imbalance.

Only 40/1,000 train scenes and 3/150 validation scenes contain all nine actions
in the selected pack. Scene-macro action entropy is `2.298` bits for train and
`2.229` bits for validation, versus a nine-action maximum of `3.170` bits.

## Local pre-action support

### Feature and neighborhood definition

The pre-action feature contains 39 physical-history values available before
candidate action `a2` begins:

- two egocentric endpoint displacements, each `(forward, lateral, yaw)`;
- current base `z`, roll, and pitch;
- current six-axis body twist;
- twelve joint positions; and
- twelve joint velocities.

Each feature is standardized from train rows. The audit selects the 16 nearest
references and excludes every same-scene reference for train leave-one-out.
It reports four progressively stricter reference groups: all rows, same scene
family, same two-action requested history, and both family plus history.

This is a physical-state positivity diagnostic. It does not include RGB
appearance, and its distances are not a learned sufficient-state metric.

### Aggregate findings

| Neighbor restriction | Train eligible | Mean local actions / 16 | Entropy (bits) | Zero factual-action support among eligible | kNN action balanced accuracy |
|---|---:|---:|---:|---:|---:|
| all, cross-scene | 100.0% | 5.02 | 1.798 | 12.0% | 0.396 |
| same family, cross-scene | 100.0% | 5.25 | 1.920 | 10.9% | 0.359 |
| same history, cross-scene | 100.0% | 4.33 | 1.516 | 9.4% | 0.451 |
| same family + history, cross-scene | 78.0% | 3.54 | 1.221 | 7.6% | 0.556 |

Validation queries against train are nearly identical. Under the strict
family-plus-history grouping, 79.7% have 16 train references; eligible rows
average 3.52 local actions and `1.211` bits of entropy, with 8.0% lacking their
own factual action among the 16 neighbors. Strict `k=16` plus nonzero factual
support covers 72.1% of train queries and 73.3% of validation queries.

Two conclusions coexist:

1. There is real, scene-disjoint local overlap. Exact-state uniqueness did not
   eliminate all factual action contrast.
2. The behavior curriculum remains strongly confounded. A simple physical
   kNN predicts the action at `0.556` balanced accuracy under the strict
   grouping, far above `1/9`, and over one quarter of queries fail the combined
   strict-neighborhood/factual-support diagnostic.

### Per-action strict support

The fraction below requires both 16 cross-scene references with the same
family/history and at least one neighbor carrying the query's factual action.

| Action | Train | Validation |
|---|---:|---:|
| `arc_left` | 84.9% | 87.1% |
| `arc_right` | 53.2% | 60.1% |
| `backward` | 46.2% | 38.5% |
| `forward_fast` | **18.2%** | **18.9%** |
| `forward_medium` | 90.6% | 89.4% |
| `forward_slow` | **6.5%** | **5.9%** |
| `hold` | 40.2% | 44.2% |
| `yaw_left` | 82.3% | 84.1% |
| `yaw_right` | 64.9% | 66.7% |

This action-specific imbalance is a more plausible contributor to the failed
hardest-action gate than a corpus-wide absence of action signal. It also warns
against treating all nominally wrong actions as equally distinct.

## Requested commands and realized motion

Each action has exactly one requested five-tick command tape in the selected
population:

| Action | Mean requested `(vx, vy, yaw rate)` |
|---|---|
| `arc_left` | `(0.20, 0.00, +0.45)` |
| `arc_right` | `(0.20, 0.00, -0.45)` |
| `backward` | `(-0.20, 0.00, 0.00)` |
| `forward_fast` | `(0.30, 0.00, 0.00)` |
| `forward_medium` | `(0.25, 0.00, 0.00)` |
| `forward_slow` | `(0.20, 0.00, 0.00)` |
| `hold` | `(0.00, 0.00, 0.00)` |
| `yaw_left` | `(0.00, 0.00, +0.45)` |
| `yaw_right` | `(0.00, 0.00, -0.45)` |

Requested `vx` versus mean observed body-twist `x` correlation is `0.644` on
train and `0.622` on validation. Requested yaw rate versus observed body yaw
rate is `0.670` on train and `0.654` on validation. These measurements confirm
physical action signal, but the residual includes controller dynamics,
contacts, inertia, terrain, and estimation noise.

The selected metadata contains zero keys naming execution or clipping. Exact
requested-versus-executed command comparison is therefore unavailable. A raw
message join would require a separately scoped hundreds-of-gigabytes scan and
was intentionally not performed.

## Motion density and partial observability

The descriptive “meaningful motion” threshold is planar translation at least
2.5 cm or absolute yaw at least 0.05 rad over the half-second candidate edge.
It is not a preregistered pass gate.

| Action | Train rows | Meaningful-motion fraction | Median forward displacement | Median yaw |
|---|---:|---:|---:|---:|
| `arc_left` | 2,959 | 86.7% | +0.076 m | +0.092 rad |
| `arc_right` | 1,197 | 89.1% | +0.070 m | -0.178 rad |
| `backward` | 1,075 | 91.6% | -0.048 m | +0.022 rad |
| `forward_fast` | 545 | 83.7% | +0.056 m | -0.031 rad |
| `forward_medium` | 4,303 | 89.0% | +0.101 m | -0.042 rad |
| `forward_slow` | 447 | 80.3% | +0.033 m | -0.031 rad |
| `hold` | 767 | 71.1% | +0.007 m | +0.006 rad |
| `yaw_left` | 2,893 | 87.1% | +0.013 m | +0.160 rad |
| `yaw_right` | 1,814 | 88.3% | +0.010 m | -0.196 rad |

The surprising `hold` motion is not evidence that HOLD causes movement. It
shows that the observation is not a Markov state unless incoming velocity,
attitude, contact/controller state, and action history are represented. This
also explains why persistence can be a strong baseline and why nominal action
identity need not imply a unique half-second outcome.

The forward-speed classes overlap heavily, and the selected rows reproduce the
earlier inversion in which median/mean `forward_medium` progress exceeds
`forward_fast`. A nine-way strict label ordering is therefore a poor substitute
for physical successor accuracy or planning regret.

## Implications for the experiment plan

1. **Do not claim that observational training has no local contrast.** The
   exact-duplicate argument is withdrawn by measurement, not merely theory.

2. **Do not assume the 3 TB corpus has solved positivity either.** The strict
   selected-pack audit remains weak for the rare forward-speed actions and
   action is highly predictable from state/history.

3. **Use the existing corpus before bulk branch generation.** Construct a
   larger family/history/action-stratified pack from the existing causal-valid
   pool, deliberately increasing `forward_fast`, `forward_slow`, `hold`, and
   `backward`. Compare it with an equal-size naturally sampled pack. This tests
   whether the failure is finite-sample imbalance or structural policy
   confounding.

4. **Add proprioception or a learned belief state to the dynamics input.** The
   physical-history audit shows large incoming-motion effects that the current
   RGB/action-only interface must infer indirectly.

5. **Keep counterfactual branches evaluation-first.** They remain the cleanest
   way to measure untaken-action successor fidelity, but these results do not
   justify using them as training data before the existing-pool comparison.

6. **Score physical outcome/regret, not nominal action identity alone.** The
   forward-speed overlap, HOLD inertia, and requested/realized residuals make
   strict all-nine label separation scientifically misaligned.

7. **Preserve requested/executed semantics.** Requested actions are valid model
   inputs. Any future branch collector should record the post-controller
   executed tape as provenance and an outcome/audit target, without leaking it
   into the pre-action input.

## Limitations

- The local-neighborhood result covers the exact 16,000/2,048 bound pack, not
  every corrected candidate in the 3 TB corpus.
- The full-corpus action counts are marginal sliding-H6 census counts; they do
  not establish local conditional overlap.
- Physical features omit RGB appearance and may join visually different
  states or separate visually similar ones.
- The distance metric and `k=16` are descriptive choices. A multi-k/fixed-radius
  sensitivity audit would be appropriate before turning them into a gate.
- There is no direct collision/contact flag, and joint effort is absent, so
  “dynamic event” coverage is limited to pose, attitude, twist, joint, and
  motion proxies.
- Nothing here establishes world-model generalization, counterfactual validity,
  rollout quality, planner utility, or navigation success.
