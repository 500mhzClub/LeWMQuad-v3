# Structured Spatial Safety State JEPA Development V2

Date: 2026-08-20  
Source commit: `f72fe00c8426a973fbb56c521e5a89a563a9373f`  
Final classification: `TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO`

## Scope and custody

This experiment preserves `SAFETY_AUXILIARY_JEPA_DEVELOPMENT_NO_SIGNAL`.
Stage A trained and evaluated one structured stuck/blocked-motion component on
true H1–H3 representations. It failed the frozen true-future gate, so Stage B
was not opened: no JEPA predictor was trained, no historical predictor
checkpoint was opened for inference, and no dynamic gradient balancing or
predictive non-regression run was applicable.

The frozen route corpus was reused unchanged: 48 states and 576 branches, split
into 32/8/8 states and 384/96/96 branches for fit/calibration/held-out. The
target-latent index matched
`df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874`.

The route ledger retained poses, quaternions, scene identities, RGB receipts,
and metric clearance but had not materialized its 64×64 occupancy rasters. The
rasters were therefore deterministically derived once from the frozen stored
poses and scene manifests using the same pure V4 ray-evidence rasterizer as the
qualified occupancy assay. This did not execute Genesis, render RGB, or encode
latents. The resulting `[576,3,64,64]` uint8 array is bound by SHA-256
`b065c32811edb5b2a3c02acc6e3304e89b5201f6a87cd9cc46ab328fcd92d314`.

## Evaluator-first fixture

The fixture passed clear path, footprint intersection, marginal clearance,
stuck without collision, collision without stuck, all-safe, all-unsafe,
reject-all, and deterministic tie cases. It serialized and reloaded
deterministically.

Fixture SHA-256: `fe07be967079a0e345875f12236cb5c372dc5ca7f1d272a8653ec5e9488eb04c`.

## Stage-A component contracts

### Frozen occupancy and deterministic clearance

- Frozen occupancy package:
  `b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686`.
- Frozen probe weights:
  `95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322`.
- Probe inference: affine-free per-token LayerNorm, unchanged
  `SharedTokenToBev(1024)`, and fixed three-class decision.
- Body footprint radius: `0.47 m`.
- Frozen clearance threshold: `0.15 m`.
- Clearance estimate: nearest predicted occupied cell centre minus the frozen
  body radius. The continuous footprint risk is the maximum occupied
  probability within radius `0.47 + 0.15 m`.
- No clearance head was trained because the frozen BEV grid already carries a
  metric 0.10 m cell contract.

### STUCK_BLOCKED_MOTION_HEAD_V1

The only trained component has 34,102 parameters. It uses a shared
`1024→16` token projection, true-current/future/difference mean and max pooled
features over H1–H3, a 32-D action/control/nominal-motion MLP, and a 64-D fusion
layer. Its six outputs are cumulative stuck logits at H1–H3 and signed
realised-minus-nominal displacement residuals at H1–H3. It has no aggregate
unsafe, collision, progress, utility, completion, place, or goal output.

Training used seed `2026082005`, AdamW, learning rate `1e-3`, weight decay
`1e-4`, 60 epochs, and final epoch only. Fit loss decreased from `1.241885` to
`0.036653`. H1–H3 stuck positive weights were `2.45946`, `1.34146`, and
`0.882353`.

Checkpoint:
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/structured_spatial_safety_state_jepa_dev_v2/stuck_blocked_motion_head_seed_2026082005_epoch60.pt`  
Checkpoint SHA-256:
`e715f913858820bc195cfbe1e7bdcb73a3d9b4fde93e1c709e9f7e50aff6528b`.

## Calibration

The eight calibration states were opened only after training. The stuck
temperature was `7.112426`; the stuck threshold was `0.110166`. Combined with
the fixed clearance rule, this achieved calibration unsafe recall `0.9861` and
safe retention `0.1667`. The occupancy probe itself was not recalibrated.

## True-future spatial and stuck results

| Metric | H1 | H2 | H3 | Gate |
|---|---:|---:|---:|---|
| Pooled occupied IoU | 0 | .000551 | .001152 | H3 ≥.35: fail |
| Clearance MAE | 4.2588 m | 4.6917 m | 4.0814 m | diagnostic |
| Clearance Spearman | -.1551 | .0209 | .1119 | H3 ≥.60: fail |
| Signed displacement-shortfall MAE | .0392 m | .0581 m | .0691 m | diagnostic |

There were no H3 low-clearance-positive rows in the held-out split, so
low-clearance recall was undefined and the required ≥.90 gate could not pass.
The frozen occupancy interface did not transfer to this purpose-built maze
panel under the registered label construction.

H3 stuck metrics:

| Positive rows | AUC | AP | Recall | FNR | ECE |
|---:|---:|---:|---:|---:|---:|
| 44/96 | .7950 | .7605 | 1.0000 | 0 | .0860 |

Stuck recall and FNR passed, but AUC failed the required `.85` floor.

The clearance rule alone recalled `0/24` collision/contact-positive held-out
rows. Thus collision risk was not traceably recovered from predicted occupancy
and footprint clearance.

## Per-family results

| Family | Structured unsafe AUC | Recall | FNR | Safe retention | Stuck AUC | Clearance rho | Mean defined-row H3 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| large_enclosed_maze | .5625 | .9500 | .0500 | 0 | .6786 | .0856 | 0 |
| medium_enclosed_maze | .8593 | 1.0000 | 0 | .0667 | .8203 | -.0487 | .000706 |
| small_enclosed_maze | .8429 | .9286 | .0714 | .1000 | .8112 | .4324 | .002377 |
| loop_alias_stress | .9407 | 1.0000 | 0 | .1111 | .9407 | .7259 | .000733 |

The large-maze family collapsed in both spatial and stuck discrimination. The
other families did not provide enough safe retention for candidate planning.

## Structured filtering and kinematic planning

Overall held-out structured safety metrics were:

| AUC | AP | Unsafe recall | FNR | Safe retention | Admitted |
|---:|---:|---:|---:|---:|---:|
| .7750 | .8425 | .9655 | .0345 | .0789 | 5/96 |

The high-recall branch requirements passed, but only 3 of 38 safe candidates
were admitted. Three of eight states retained a safe candidate; two states had
only unsafe candidates admitted.

| State | Family | Admitted safe/unsafe | Selected | Safe | Distance progress |
|---|---|---:|---:|---|---:|
| purpose-10 | large_enclosed_maze | 0/1 | 11 | no | -.0364 m |
| purpose-11 | large_enclosed_maze | 0/0 | abstain | NA | NA |
| purpose-22 | medium_enclosed_maze | 0/0 | abstain | NA | NA |
| purpose-23 | medium_enclosed_maze | 1/0 | 11 | yes | .0609 m |
| purpose-34 | small_enclosed_maze | 0/1 | 11 | no | .0115 m |
| purpose-35 | small_enclosed_maze | 1/0 | 11 | yes | .0761 m |
| purpose-46 | loop_alias_stress | 1/0 | 11 | yes | .0198 m |
| purpose-47 | loop_alias_stress | 0/0 | abstain | NA | NA |

The filtered kinematic planner selected unsafe branches in 2/5 non-abstaining
states (`selected_unsafe_rate = .40`), selected mean distance progress only
`.0264 m`, had normalized safe-progress regret `.6667`, best-safe top-3 `.125`,
and three false abstentions. The oracle-safety kinematic upper bound selected
mean progress `.2184 m`, zero unsafe branches, zero regret, and best-safe top-3
`1.0`.

## Gate and interpretation

Passed: aggregate recall/FNR, stuck recall/FNR, and the vacuous no-safe-state
abstention condition. Failed: occupancy IoU, clearance correlation,
low-clearance recall availability, stuck AUC, safe retention, state retention,
only-safe admission, selected safety, route progress, regret, best-safe top-3,
and false-abstention limits.

Therefore Stage A terminates as
`TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO`. Under the frozen protocol this
means the available true RGB-future representation plus occupancy/clearance and
stuck labels did not support a qualified factorised safety interface. Stage B,
dynamic gradient balancing, predicted-future safety evaluation, and predictive
non-regression were prohibited and not run. This result does not overwrite the
historical predictor or its successful fidelity/action-sensitivity evidence.

## Runtime, storage, and artefacts

- Occupancy-label derivation: `36.84 s`.
- Stuck-head training: `105.13 s`.
- Complete Stage-A run: `146.16 s`.
- Checkpoint: `143,931` bytes.
- Derived occupancy array: `7,078,016` bytes.
- Total cache directory: `7,222,529` bytes.

| Artefact | SHA-256 |
|---|---|
| Stage-A source | `d05299f6c97f228b89c12855fb5e9de7dc43b27e546c5c96f747f55e110c9604` |
| Stuck-head checkpoint | `e715f913858820bc195cfbe1e7bdcb73a3d9b4fde93e1c709e9f7e50aff6528b` |
| Occupancy array | `b065c32811edb5b2a3c02acc6e3304e89b5201f6a87cd9cc46ab328fcd92d314` |
| Occupancy index file | `7d30660846eda7fbfd5d807a16a8790d191b3cb1774811892d7872f295df3696` |
| Evaluator fixture | `fe07be967079a0e345875f12236cb5c372dc5ca7f1d272a8653ec5e9488eb04c` |
| Machine result | `832528bcdcf179aa0f628654d876233f426cf0914c19df0c5d11af9a82caca2d` |

Exactly one new component seed (`2026082005`) and zero new predictor seeds were
trained. No simulation, RGB rendering, latent encoding, branch generation,
matched one-step training, memory, novelty, beacon capture, or closed-loop
navigation occurred. Nothing remained running at handoff.
