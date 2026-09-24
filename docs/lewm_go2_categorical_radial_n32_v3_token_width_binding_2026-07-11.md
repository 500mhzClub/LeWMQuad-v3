# Go2 categorical-radial N32 V3 token-width execution binding

Date: 2026-07-11

Status: frozen before any N32 V3 dataset-backed model output.

## License and question

The exposure-matched N32 V2 run failed the immutable all-family fit rule after
500 epochs. Its result report is
`docs/lewm_go2_categorical_radial_n32_v2_result_2026-07-11.md` (SHA-256
`d5e5748db8177d925990b5c31e23c45d43e16c62e0aac4f389ab47b1fa6547e0`).
The authoritative result file/content SHA-256 values are
`0a5f8a822d7fec8287a30103125fca1a4927f0413e2f0906db431cef54ec2265` /
`e070cc96d69b76e1f85f533fa1d94221225963a2b66a491f0c2a867c008b97ef`.
That frozen failure licenses one representation/capacity intervention at N32.

V2 passed every aggregate fit threshold and three of five scene families. The
substantive residual was rough-terrain UNKNOWN/known discrimination. The
earliest irreversible task bottleneck is the learned `192 -> 24` token
projection before registered projective sampling. Later context layers cannot
recover visual cues discarded there. V3 asks only whether retaining eight more
features per image token removes that residual.

## Sole intervention

The only architecture change is:

- `token_feature_dim`: `24 -> 32`;
- `context_dim`: remains `64`;
- encoder: remains image 112, patch 7, hidden dimension 192, depth 6, heads 6;
- projective anchors, camera geometry, 64 x 256 polar lattice, Cartesian
  factorization, angular block, six full-ray dilations `(1,2,4,8,16,32)`, class
  order, loss, supervision mask, and output support: unchanged.

The registered candidate parameter count is 2,891,171, exactly 4,104 above V2.
Only these state tensors change shape:

- `token_projection.weight`: `[24,192,1,1] -> [32,192,1,1]`;
- `token_projection.bias`: `[24] -> [32]`;
- `context_stem.0.weight`: `[64,154,1,1] -> [64,194,1,1]`.

All other parameter and buffer keys, shapes, dtypes, and values must match the
seed-corresponding V2 initialization bit for bit. To construct the candidate,
the runner must save the CPU RNG state immediately after the registered
determinism setup, instantiate the V2 width-24 model, restore that RNG state,
instantiate the V3 width-32 model, and copy every same-key/same-shape V2 state
entry into V3. The three shape-changed tensors retain their deterministic V3
initialization. No trained V2 weight may be loaded.

The candidate must live in a new versioned model module. Frozen V1/V2/V3
ladder and N32 V1/V2 source files must not be modified.

## Data and controls

The N32 V1 panel, exact 320 fit frames, five scene families, same-scene
holdout, cross-scene holdout, artifact commitments, immutable failed patch-7
comparator, causal wrong-view permutations, label geometry, and supervision
masks remain unchanged. The candidate may open only fit-role artifacts until
the exact terminal fit gate passes.

The fit panel retains both controls:

- same-scene wrong-view RGB;
- role-global cross-scene shuffled RGB.

Neither control may share the correct image, transition, or forbidden scene
identity defined by the original contract. No checkpoint-selection,
probability-calibration, non-train, G2, or sealed payload/model output may be
opened by this diagnostic.

## Optimizer and schedule

V3 transfers the V2 optimizer and exposure exactly:

- seed 20260710 first; seed 20260711 remains conditional;
- direct FP32 batch of 80 frames, no microbatching or accumulation;
- 320 frames, four updates per epoch;
- 2,000 optimizer updates, 500 effective epochs, 160,000 frame presentations;
- AdamW, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, no AMSGrad;
- global gradient clipping at `1.0`, once per optimizer update;
- one-indexed V3 cosine learning rate, `2e-4` at update 1 to `1e-5` at update
  2,000, assigned immediately before each optimizer step;
- no warmup, AMP, autocast, compilation, quantization, EMA, or augmentation;
- evaluation every 100 updates, with the final three evaluations fixed at
  updates 1,800, 1,900, and 2,000;
- the exact V2 seed-specific minibatch schedules and control permutations.

A dataset-free seed-1 feasibility check was completed before this binding. One
direct batch-80 forward/backward was finite, used 2,891,171 parameters, and
peaked at 15,409,975,296 allocated bytes on the registered ROCm device. It
produced no dataset-backed research artifact and is not gate evidence.

## Gates and access order

Fit gates are unchanged. At each of updates 1,800, 1,900, and 2,000, the
aggregate and every one of the five family reports must simultaneously pass
all original thresholds, including:

- hierarchical balanced NLL `<= 0.03`;
- UNKNOWN/known balanced accuracy `>= 0.99`;
- FREE/OCCUPIED balanced accuracy `>= 0.99`;
- UNKNOWN recall `>= 0.98`, FREE recall `>= 0.98`, OCCUPIED recall `>= 0.98`;
- FREE recall `>= 0.95` in each registered distance band from 1 m onward;
- each wrong-view-minus-correct NLL delta `>= 0.25`.

If any terminal fit evaluation fails, the runner must record zero same-scene
and cross-scene holdout byte opens/model outputs, mark seed 20260710
unfavorable, and forbid seed 20260711.

Only an exact terminal fit pass authorizes opening the two train-role holdouts.
Their reports and comparator-based checks remain exactly those of the N32 V1
contract. One fully favorable seed authorizes only seed 20260711. The second
seed must receive the primary result path and precommitted file SHA-256, and a
strict finalizer must validate the primary before any device or panel access.

Only two independently favorable seeds may license construction of the
shared-JEPA full-training candidate. V3 cannot pass G2, authorize calibration,
set `runtime_ready`, license G3, or authorize any sealed evaluation.

## Required implementation evidence

Before the authoritative seed-20260710 command, freeze and record:

- the new model, pure decision module, runner, finalizer, and test hashes;
- both expected seed-specific initial-state hashes;
- both exact minibatch-schedule hashes;
- proof that the only state-shape changes are the three registered tensors and
  that every other initial state entry is bit-identical to V2;
- exact parameter count, output/token/sample shapes, full-ray reachability,
  causal controls, source map, and artifact commitments;
- strict smoke access reconciliation and adversarial finalizer tests.

The authoritative result paths are:

- `.generated/go2_categorical_radial_n32/v3/seed_20260710_result.json`;
- `.generated/go2_categorical_radial_n32/v3/seed_20260711_result.json`.

Results are immutable and must be written with exclusive atomic creation.
