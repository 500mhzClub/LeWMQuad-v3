# Action/time structure is predictable; current JEPA fitting misses it

The completed empirical control scores **0.859 cm planar error** on the same
917 targets where zero motion scores 2.233 cm and the three JEPA recursive
heads score 4.103–4.932 cm. It uses the same 72 scheduled training draws
(62 distinct windows), with duplicates retaining their weights. All 30 exact
action/time cells have training support; no interpolation, fallback or target
exclusion was needed. Yaw error is 0.00795 rad versus zero yaw's 0.12860 rad.

This establishes action/time structure in this development corpus that the
short neural pilot did not capture. It does not establish RGB utility, a useful
learned controller, independent-maze generalization or JEPA benefit. The control
was designed after seeing the neural result; all 185 scored windows belong to
the same train-role room, including overlapping unsampled windows.

## The aggregate hides a support-dependent failure

| Recorded condition | Empirical planar error |
| --- | ---: |
| Nominal left | 0.719 cm |
| Nominal right | 0.511 cm |
| Lower-friction left | 4.073 cm |

On lower friction, zero motion scores 2.994 cm: the empirical control is worse.
The baseline pools conditions without privileged friction input. Its aggregate
success cannot justify installing another fixed action table in the controller.
State/support-dependent prediction and balanced collection remain necessary.
Only two distinct short-forward examples exist. Every contact label is negative;
perfect contact accuracy and the clipped baseline Brier score are not risk-learning
or probability-calibration evidence.

## Objective-scale hypothesis now has measured support, not causal proof

The second diagnostic examines all nine fitted models at reconstructed initial
and saved final weights: 18 measurements on the identical first six-row training
batch. It performs no optimizer step or EMA update. The weighted decomposition
matches the frozen objective and every active-parameter gradient; maximum observed
gradient difference is zero. Model tensor hashes and empty parameter gradients
are unchanged afterward.

At JEPA initialization, combined direct/recursive position losses account for
0.042–0.376% of total loss. Latent-term parameter-gradient norms are 12.94–16.78,
versus 0.106–0.721 for recursive position; latent-gradient cosine with the total
is 0.943–0.964. At saved final weights, the combined position-loss share remains
0.170–0.236%; recursive-position gradient norms are 0.0558–0.0945 versus latent
0.646–0.953. Direct/recursive contact and angle gradients are also reported.

The loss mixes raw metre displacements with unit sin/cos, contact BCE and latent
loss. A single-coordinate 1 cm error contributes only 0.0000125 to the four-channel
SmoothL1 mean. These measurements motivate a task-scaled outcome comparison.
They do not prove that reweighting improves learning, that JEPA is inherently
unsuitable, or that small gradient norms alone determine Adam's updates. Norms
are not additive contributions, and this single shared batch does not describe
the full training trajectory. Twelve updates remain a very short budget.

## Reproducibility and preserved boundaries

Baseline output: `.generated/go2_pulse_action_time_baseline_v1_attempt_001`.
Terminal handle 32126, exit 0. Result SHA256:
`a03459e6b59f8bae3ac1ca1acd2c3964e6abb1484bd3949a60438e9888389d63`.

Loss-scale output: `.generated/go2_pulse_loss_scale_diagnostic_v1_attempt_001`.
Terminal handle 51913, exit 0. Result SHA256:
`2d39a27cc83ff461e08424a8092a75545b2bdf501474002627ec90afc5978a71`.

Independent verification handle 62055 exits 0. It reconstructs all 917 empirical
predictions with scalar accumulation from scheduled float32 training labels,
independently checks planar/yaw/condition scores and cell counts, verifies all
25 output identities and both inherited source/input closures. Full identity
bindings and gradient summary are in the
[integrity record](go2_pulse_baseline_and_loss_scale_integrity_2026-09-06.json).
No original source, dataset, checkpoint, failure or experiment output was edited.
No sealed access, source export, new physics, GPU fitting or deployment occurred.

The full explicitly enumerated 210-file regression finishes with **2,638 passed
in 203.01 s**, handle 20412, exit 0. This includes the 14 new empirical-baseline
tests and four new loss-decomposition tests. Focused checks also passed before
launch (36 baseline/dataset/trainer tests; four decomposition tests). These are
implementation checks, not additional maze or hardware evidence.

## Next bounded work

1. Make one prospectively fixed scale/budget diagnostic: retain architecture,
   seed pairing, sensor exposure and data; compare original versus dimensionless
   task-scaled outcome losses at matched short and longer budgets. Derive and
   freeze physical scales before fitting, not from whichever scores look best.
   Fresh initialization only, deterministic repeated schedule, final checkpoints
   and predeclared learning-curve snapshots, no best-snapshot selection. Report
   against the empirical control and by condition, not just zero and total loss.
   Do not simultaneously introduce baseline residualization or architecture
   changes, which would confound this diagnostic.
2. Regardless of that result, move beyond room resubstitution: collect independent
   connected layouts with fixed roles, multiple starts/body histories, all six
   supported action-duration cells across observable support states, and obstacle/
   contact outcomes with correct interruption censoring. Avoid extending same-room
   fitting indefinitely or claiming repeated draws fix absent information.
3. Establish useful dynamics and RGB/body/history/action dependence on independent
   development layouts before matched online candidate selection. Keep predictive
   training, online rollout and persistent-memory effects separate.
4. In parallel scientific staging, repair tracking qualification and state-dependent
   local execution, then demonstrate observed branches and physically executed
   backtracking. The latest room-return result remains 0/3. Realistic sensing,
   uninterrupted deadlines, full body sweep and bounded hardware remain required.

The ultimate goal remains active and unachieved.
