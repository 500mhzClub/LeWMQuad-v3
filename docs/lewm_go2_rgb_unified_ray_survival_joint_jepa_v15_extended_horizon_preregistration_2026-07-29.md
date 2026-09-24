# RGB Unified Ray-Survival Joint JEPA V15 extended-horizon preregistration

Date: 2026-07-29

Status: preregistered design only; no V15 reservation, training, checkpoint,
qualification, calibration, G2, navigation, or held-out access has occurred.

## Question and prior evidence

- V14 completed its exact cap of 1,000 joint updates and 16,000
  presentations. Its terminal scientific result is frozen in commit
  `d54dfea445dc9bc80cee6421c1b0aea2639463f1` at
  `docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_scientific_result_2026-07-29.json`,
  with SHA-256
  `290cde5ef5dd2bf4fc93fd15b5fc1fd107fd857291abf29d4d57351d843f5263`
  and byte count `9806`.
- V14 legitimately passed its matched update-400 falsification: `72/189`
  margins, shortfall `68.96964862816927`, rough depth p95
  `1.8582415819168085` m, all twelve causal controls, and all registered
  integrity checks.
- V14 remained measurably improving from update 400 to 1,000: margins rose
  `72 -> 89`, shortfall fell `68.96964862816927 -> 41.41604892978589`,
  rough depth p95 fell `1.8582415819168085 -> 1.5208376884460448` m,
  rough ground balanced accuracy rose
  `0.6388506985089961 -> 0.6861817193868919`, and inherited V12 checks rose
  `21/24 -> 23/24`.
- V14 nevertheless failed the final gate with zero complete scopes, margins
  `89 < 112`, shortfall `41.41604892978589 >= 33.05143763708337`, rough
  depth p95 `1.5208376884460448 >= 0.9777327477931971` m, and failed
  `semantic_free_recall`. It published no checkpoint and cannot be resumed.

The falsifiable V15 question is whether V14 was still undertrained at its
1,000-update cap. V15 is one fresh, longer attempt with a feasibility stop;
it is not a seed, coefficient, data, loss, or architecture sweep.

## Sole material scientific change

- Keep the exact V14 model and mechanism: RGB encoder, dense decoder, unified
  64-bin first-hit hazard and ground-survival field, within-bin offsets,
  registered fixed camera geometry, FREE/OCCUPIED evidence lift, 64-channel
  sole JEPA state, semantic decoder, action-conditioned predictor, EMA target,
  optimizer, losses, loss weights, parameter initialization, seeds, labels,
  development roles, evaluation, causal controls, and final thresholds.
- Increase the maximum horizon from 1,000 to 2,000 joint updates and from
  16,000 to 32,000 presentations.
- Load the same frozen, authority-bound 16,000-presentation schedule exactly
  once after reservation, validate its original update-100, update-400, and
  update-1,000 identities, and form the V15 schedule in memory as
  `base_schedule + base_schedule`. The first and second 16,000-presentation
  halves must be element-for-element identical. No schedule file is generated,
  edited, extended, shuffled, or written.
- Add one update-1,400 feasibility gate and move the unchanged final
  development gate and any eligible checkpoint from update 1,000 to update
  2,000. Update 1,000 is observation-only.
- The trajectory is continuous from update 1 through update 2,000. Wrapping
  the data-index schedule at presentation 16,000 must not reconstruct the
  model, optimizer, Adam moments, EMA target, accounting object, loader,
  stochastic state, or RNG; must not hard-sync the target; and must not reset
  or reseed anything. Only the next schedule index returns to the first bound
  base-schedule entry.

No V14 state is reused because V14 correctly published no failed checkpoint.
V15 starts fresh from the same accepted N320 initialization. The reviewed
ROCm runtime retains the same deterministic-algorithm `warn_only=True`
setting; identical seeds and schedule do not assert bitwise equality through
kernels that report no deterministic implementation.

This remains one jointly optimized JEPA. The encoder, learned physical
evidence, latent state, semantic decoder, and action-conditioned predictor are
trained in the same optimizer updates; no predictor is trained separately.

## Frozen model, optimization, and schedule identity

- Preserve constructor seed `20260712`, schedule seed `20260713`, stochastic
  execution seed `20260728`, projection seed `20260729`, and bootstrap seed
  `20260728`.
- Preserve the exact base-schedule prefix SHA-256 values: update 100
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`,
  update 400
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`,
  and update 1,000
  `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.
  The repeated half identity is checked directly rather than represented by a
  newly generated schedule artifact.
- Preserve float32 AdamW, encoder learning rate `1e-4`, all other online
  learning rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay
  `1e-4`, route-wise norm-one clipping, four `B=4` microbatches per update,
  one optimizer step, and one EMA step per update.
- Preserve V14 parameter counts: shared `3,102,824`, representation `22,020`,
  predictor `259,073`, total online `3,383,917`, target bottleneck
  `3,106,216`, and role projections `3,392`.
- The V15 attempt root must initially be absent and is
  `.generated/go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon/attempt_v1`.
  There is exactly one fresh attempt, no retry, and no resume.

Maximum update-2,000 accounting is 2,000 updates, 32,000 presentations,
8,000 microbatch graphs, 16,000 backward calls, 8,000 Camera-route gradient
calls, 8,000 joint-route gradient calls, 64,000 Camera-frame objectives,
2,000 optimizer steps, 2,000 EMA steps, 8,000 predictor forwards, and 8,000
predictor objectives.

Immutable observations are update 0, 100, 400, 1,000, 1,400, and, only if
earned, 2,000.

## Stopping rules

- Preserve V14's update-400 gate exactly. Continue only if all inherited
  directional and twelve causal-control checks pass and the result strictly
  beats matched V13 update 400 on all three registered residuals: margins
  greater than `71`, shortfall less than `71.67935936391197`, and rough depth
  p95 less than `1.936374711990354` m. Equality fails.
- Update 1,000 is observation-only. It grants no checkpoint, promotion,
  calibration, G2, navigation, held-out access, retry, or resume.
- At update 1,400, continue only if every one of these checks passes:
  structural integrity; inherited V12 `24/24`; all twelve causal-control
  checks; at least `99/189` nonnegative physical margins; total shortfall
  strictly below `38.1`; rough depth p95 strictly below `1.304` m; rough
  pixel balanced accuracy strictly above `0.8198594673963917`; and rough
  ground balanced accuracy strictly above `0.647134926562893`. Equality fails
  for every strict continuous threshold. This gate consumes exactly 1,400
  updates and 22,400 presentations if it stops.
- The update-1,400 causal controls must be freshly evaluated from the
  update-1,400 model using the same four V14 comparison arms and the same
  three Boolean checks per arm: positive equal-scene delta, positive
  bootstrap lower-95 bound, and positive-family count. The immutable
  update-1,400 observation and gate receipt must carry all twelve values;
  reusing the update-400 control Booleans is forbidden.
- At update 2,000, apply the unchanged V14/V13 final gate: structural
  integrity; inherited V12 `24/24`; at least `112/189` nonnegative margins;
  total shortfall strictly below `33.05143763708337`; at least one complete
  physical scope; rough pixel balanced accuracy strictly above
  `0.8198594673963917`; rough ground balanced accuracy strictly above
  `0.647134926562893`; and rough depth p95 strictly below
  `0.9777327477931971` m. Equality fails.
- Only a complete update-2,000 pass may publish a perception checkpoint. Any
  update-400, update-1,400, update-2,000, integrity, accounting, custody, or
  exception failure is terminal and publishes no checkpoint.

## Interpretation boundary

- Passing update 1,400 only establishes that the improving V14 mechanism is
  on a preregistered feasible trajectory; it is not a perception success.
- Passing update 2,000 supports the undertraining hypothesis and earns only a
  jointly trained development checkpoint plus physical-adapter
  preregistration eligibility.
- Failure at update 1,400 or 2,000 closes longer training of this exact
  mechanism. The next materially different candidate may tie the semantic
  occupancy raster to the unified learned ray geometry, but V15 does not make
  that architectural change.

## Authority boundary

Implementation, focused tests, recursive source closure, independent source
review, clean-export certification, and one-shot execution authority must be
separately frozen before reservation. Until a complete final development
pass, probability calibration, G2, navigation, held-out, sealed, production,
promotion, deployment, retry, and resume remain forbidden.
