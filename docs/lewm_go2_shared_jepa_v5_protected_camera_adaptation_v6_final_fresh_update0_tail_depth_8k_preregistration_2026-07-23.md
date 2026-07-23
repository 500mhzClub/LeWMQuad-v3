# Protected Camera Adaptation V6 final fresh-update-zero tail-depth 8k preregistration — 2026-07-23

## Boundary

This source-free document preregisters exactly one final possible Camera-only successor rooted at `.generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v6_final_fresh_update0_tail_depth_8k`. It grants no implementation, source-mutation, test, review, execution, GPU, data-access, training, checkpoint-selection, promotion, JEPA, G2, navigation, runtime, or held-out authority.

Before the root may be reserved, a future implementation must have exact source closure, an independent source review, a separate explicit one-attempt execution authorization, a passing no-tensor visibility preflight, and proof that the root is absent. Reservation consumes the one attempt. A reserved root is immutable evidence whether the process succeeds or fails; it may not be deleted, reused, resumed, or recovered.

V6 is a fresh update-zero experiment, not a V4 or V5 retry. It may not load any V1–V5 Camera checkpoint or optimizer state. It must reconstruct Shared-V5 initial state SHA-256 `e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87` directly from the qualified N320 checkpoint with file/content SHA-256 `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0` / `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.

## Committed evidence and rationale

The immediately preceding terminal evidence is committed at `6053ab7150706aa021a1cd9a1a80951076c1c4c5`:

- `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_environment_recovery_v1_terminal_audit_2026-07-23.json`
- file/content SHA-256 `4284014d283a94d4a45decb9aee5164a45f35a93c36afd2e31a93685564ad5de` / `582325e20fa4622c9f9be1c46ae67011c4df1fc43ad2cdcc396fb5a1df6c671f`

V5 was healthy and closely reproduced the V3 trajectory, but its exact-float reproduction floor stopped it at update 1000. It met `P=106`, while `S=49.13255561472496` missed `49.09939462151839` by `0.033160993206564626` and `W=-7.945521640777587` missed `-7.944758415222166` by `0.0007632255554206324`. No checkpoint qualified, and there was zero G2, navigation, held-out, promotion, or retry authority. That result supports removing brittle cross-run float reproduction floors; it does not establish bitwise determinism or support lowering the physical qualification gate.

The already-executed V4 tail-depth evidence remains the sole basis for the loss choice:

| Evidence | Binding |
|---|---|
| V4 tail-depth preregistration, commit `0fdf1b163394aefa1a0a3731f9609ba4fa314f77` | `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_tail_depth_successor_preregistration_2026-07-15.md`, file SHA-256 `cada72599abfec257583986a8fb08254f9d16b8644b4062e17323da3004c81c8` |
| V4 independent review | `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_independent_review_2026-07-15.json`, file/content SHA-256 `c8ef0dc4ab2f415bc757fde963094eb163315b217cc0baf2770c5421bfdf8d93` / `52f7b233ffbf03abdc6743954b2529f89aae5054e7034b9cbea497bb36ea8f12` |
| V4 execution authorization | `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_execution_authorization_2026-07-15.json`, file/content SHA-256 `749ab396723422b16f919bd6b8838d9dba1ce160cc2cbaa315da04bf01c80502` / `f0d1aaf0226977a6865ea86c3fc91a3f6bc3644712671234cfab2ab850f5e5a6` |
| Executed V4 tail-depth loss source | `lewm/models/shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth.py`, file SHA-256 `6fc0a114386ee2fb0ae98704a970d38a7194db192283b904138015498fb02384` |
| V4 terminal audit, commit `e6698bccea632334c9156f1a382f2f00e724dc3d` | file/content SHA-256 `5d0d4a1cf966e5f612e15da9cacbc705ace4f629183038c6743f0e2fac1b355f` / `246e50b986316f7dc8c806960e8661cf83417fd34c0baa269d83b221cf98d5e2` |

V4 reduced total shortfall and increased its passing-margin count before its old cross-run cutoff:

| update | P | S | W |
|---:|---:|---:|---:|
| 100 | 61 | 112.38092829435729 | -8.20910987854004 |
| 400 | 84 | 63.3565430408583 | -5.343397927284242 |
| 1000 | 97 | 41.00174362036205 | -5.476026201248172 |

This is development evidence only. At update 1000, V4 still had 92/189 negative margins, passed 0/9 scopes, and its worst margin had slightly regressed from update 400. The values are not V6 continuation thresholds, qualification thresholds, or evidence that V6 is likely to pass. V6 is the one bounded combination not yet executed: the existing V4 tail-depth objective with the already-published full 8,000-update schedule. It adds no model, data, coefficient, optimizer, seed, or schedule design.

## Exact experiment

Relative to V5, the only training-science substitution is exact reuse of the previously implemented and reviewed V4 tail-depth objective slot. Relative to V4, the only training boundary difference is an 8,000-update maximum over all 128,000 existing presentations instead of V4's 4,000-update maximum over the first 64,000. No new scientific component is introduced by this pairing.

The implementation must exact-import or exact-reuse the V4 tail-depth loss at file SHA-256 `6fc0a114386ee2fb0ae98704a970d38a7194db192283b904138015498fb02384`; it may not copy, approximate, retune, or reinterpret it. For each separately computed real B=4 current or next frame and each represented in-range target-hit ray `r`, retain the V4 definition:

- condition the unchanged first-hit mass `q[r,b]` across the 64 finite-hit bins;
- use predicted depth `d[r,b]` equal to the unchanged bin centre plus the existing per-bin offset;
- compute `e[r] = sum_b q[r,b] * abs(d[r,b] - y[r]) / 0.25m`;
- replace only the old target-bin offset slot by the mean of the largest `ceil(0.05*N)` represented-ray values of `e`;
- retain the slot's objective coefficient `0.25`.

The other four Camera terms, every coefficient, current/next weighting, and four-microbatch reduction remain exactly V4. The five terms are `hierarchical_first_hit_nll`, `tail_depth_p95_cvar`, `ground_clear_distance_state_balanced_bce`, `derived_raster_hierarchical_bce`, and `derived_raster_cell_nll`. The backward scalar remains `observable_camera_ray_v4.total`.

Only `encoder.` (78 tensors, 2,747,520 parameters) and `evidence_head.` (14 tensors, 357,993 parameters) are trainable. `bev_decoder.`, `predictor.`, `occupancy_head.`, `target_encoder.`, and `target_bev_decoder.` remain frozen. JEPA objective, JEPA backward, EMA update, calibration, G2, navigation, and held-out counts are all exactly zero.

Retain exact AdamW settings: float32 without autocast, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`, `amsgrad=false`, separate `evidence_head` then `encoder` parameter groups, the existing V1 learning-rate function for both groups, independent clip norm `1.0` per group, four real B=4 microbatches per optimizer update, and effective batch size 16. There is no optimizer warm start.

Consume all 128,000 pair presentations from `.generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v4/schedule.json`, file/content SHA-256 `08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270` / `274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15`, with seed `20260713` and exactly 8,000 optimizer updates. Its terminal audit binding is `docs/lewm_go2_shared_jepa_v5_matched_training_v4_terminal_numeric_failure_audit_2026-07-15.json`, file/content SHA-256 `70371a2cd09e912e05ba0b5efdf75ee2de38cc89347e8111fff303e2a55c485b` / `ae86d1479fc3016eb96302304e079b7bf9647e26b24b3d860e7d32013bf9c6f4`.

Retain the exact existing data split: 4,262 train pairs from 72 scenes with 7,777 unique endpoints, and 495 checkpoint-selection pairs from eight scenes with 924 unique endpoints. Retain the physical evaluator, nine scopes, ordered 189 margins, thresholds, and cyclic-plus-one-within-family wrong-RGB mapping. No file, label, image, scene, manifest, split, sampling rule, or calibration may be added, removed, regenerated, or refined.

The checkpoint schedule and canonical presentation prefixes remain:

| update | canonical presentation-index prefix SHA-256 |
|---:|---|
| 100 | `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51` |
| 400 | `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92` |
| 1000 | `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528` |
| 4000 | `14e83952c758c2ee4118d38c116625feb351813bc24b017d7b47f53426df47ab` |
| 6000 | `5ba218ed5335c357b60d5f8c2f2d0a3f9e1171631cc299e5d0747ae858e92c50` |
| 8000 | `a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663` |

## Simple checkpoint control

For the exact ordered 189-margin vector, define `P=count(m>=0)`, `S=sum(max(0,-m))`, and `W=min(m)`. Integrity failure always has first precedence. At every fixed checkpoint, all state and reported metrics must be finite, the frozen-state hash must remain unchanged, state mutation during evaluation must be zero, the inline evaluator must return exactly 189 finite margins, and the metric sidecar must be published immutably before its control branch.

The earliest fixed checkpoint with exact 9/9 physical scopes and 189/189 nonnegative margins qualifies. This unchanged all-nine condition is the only qualification rule.

The only non-integrity early-stop rules are two coarse, same-run health checks:

- Update 100 is a functionality spotcheck. It additionally requires all 92 trainable gradients present and finite through the unchanged clips and trainable-state movement from update zero. Absent 9/9, it continues.
- Update 400 compares only with this run's immutable update-100 sidecar. Absent 9/9, it continues if either `P400 >= P100 + 5` or `S400 <= 0.90 * S100`. It stops as a clear early plateau only when both conditions are false.
- Update 1000 compares only with this run's immutable update-400 sidecar. Absent 9/9, it continues if either `P1000 >= P400 + 5` or `S1000 <= 0.90 * S400`. It stops as a clear early plateau only when both conditions are false.
- Updates 4000 and 6000 are informational spotchecks. Absent 9/9 and with valid integrity, they continue without a numeric cutoff.
- Update 8000 qualifies only on exact 9/9 and 189/189. Otherwise the attempt stops unqualified.

Equality continues. `W` and the Camera loss are reported for diagnosis and checked for finiteness, but neither controls continuation. There is no exact cross-run float baseline, Pareto ladder, or extrapolation. The prospective same-run `+5 P` / `10% S` health floors do not reclassify any prior checkpoint or relax exact 9/9 qualification. Once those two early health checks pass, the experiment is allowed to answer the intended question over the full existing schedule.

At each fixed checkpoint the one trainer process must complete the optimizer update, publish the CPU-weight snapshot, run exactly one inline nonmutating checkpoint-selection evaluation, atomically publish a mode-0444 canonical metric sidecar, and only then apply the declared branch. A read-only observer may inspect only the completed sidecar. Checkpoint `.pt` existence is not readiness; observers may not load checkpoints or rerun evaluation.

## Operational preflight

Immediately before the one authorized launch, a no-tensor Python `-I -B` probe under `HIP_VISIBLE_DEVICES=0` and all six native thread variables set to `1` must establish one visible discrete R9700, absence of `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION`, absence of another `.generated` mutator or KFD training process, absence of competing GPU work, and absence of the V6 root. The probe may not open a model, checkpoint, dataset, RGB, selection, navigation, or held-out artifact. No GPU-management query or competing GPU workload may intervene between the final preflight and launch.

If this preflight fails, the root must not be reserved and the trainer must not launch. Once the root is reserved, any environment, integrity, numeric, or physical failure consumes the sole attempt. No recovery attempt is preregistered.

## Outcome and explicit denials

If a checkpoint qualifies, that exact earliest checkpoint may proceed only to immutable audit and a separately preregistered, independently reviewed, explicitly authorized frozen-camera JEPA stage. This document itself grants no downstream authority.

If update 8000 does not qualify, every V6 checkpoint is rejected and the Camera-training branch stops. There is no automatic V7 or further Camera loss/data refinement. Any different direction requires a new user-directed architecture-level decision.

No retry, resume, recovery, warm start, checkpoint continuation, optimizer reconstruction, extension beyond update 8000, second seed, second attempt, loss blend, coefficient change, architecture change, data/refinement change, sampling change, evaluator change, threshold relaxation, soft/closest promotion, probability calibration, JEPA or predictor training, G2, navigation, runtime use, held-out access, held-out tuning, production, deployment, or automatic successor is preregistered or authorized here.
