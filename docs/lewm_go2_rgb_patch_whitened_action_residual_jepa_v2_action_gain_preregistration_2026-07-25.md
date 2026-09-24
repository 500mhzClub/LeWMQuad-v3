# RGB Patch-Whitened Action-Residual JEPA V2 Action-Gain preregistration

Date: 2026-07-25

## Decision

Authorize source preparation and one independent combined source/science
review for exactly one V2 Action-Gain successor probe.

Execution is not authorized by this document. It requires a fresh exact source
manifest, a passing combined review, and a distinct one-attempt execution
authorization.

## Bound V1 evidence

V1 is a valid terminal update-100 scientific failure. Its terminal audit is:

- path:
  `docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v1_terminal_audit_2026-07-25.json`;
- commit: `5c1ebb2b5f07f7be9ee152ea75b409358fb41477`;
- file SHA-256:
  `a87d1a706b912e8774a8e13b858e568ae91fbc1529ea4744adb189f0569457c7`;
- content SHA-256:
  `ad6a97738c7143f6649d43a85376507f82c3522d79de667406a9a73ecffb5a8c`;
- byte count: `13,309`.

V1 stopped exactly at 100 optimizer/EMA updates and 1,600 presentations.
Phase B was not entered. No checkpoint or training-trace payload was opened by
the auditor, and V1 may not be retried, resumed, extended, or reopened.

The result split cleanly:

- the whitening mechanism worked: raw effective rank rose from
  `27.717458724975586` to `53.53275680541992`, and projected effective rank
  rose from `17.426651000976562` to `41.314571380615234`;
- raw variance and spatial-diversity health ratios remained `0.33841648` and
  `0.40679615`;
- content discrimination was already useful: true/shuffled-next,
  true/shuffled-current, and true/mean-target ratios were `0.76270988`,
  `0.76299225`, and `0.89741433`;
- real-hold separation was directionally correct in `8/8` scene families but
  its ratio was only `0.9986068160133832`, versus the unchanged strict
  update-100 requirement `<0.99`;
- cyclic wrong-action separation remained effectively tied at
  `1.0000037031351818`, with only `4/8` positive families.

The observed real-hold relative separation is
`1/0.9986068160133832 - 1 = 0.0013951276561265935`. The update-100 boundary
requires more than `1/0.99 - 1 = 0.010101010101010166`, a `7.2402x` magnitude
gap. This is evidence for one action-gradient gain correction, not a change to
the successful representation mechanism.

## Exactly one scientific change

Define one new scalar:

`ACTION_DISCRIMINATION_WEIGHT = 10.0`.

Replace only the V1 total-loss coefficients:

`L_V1 = L_jepa + L_wrong + L_hold`
`       + 0.50*(V_raw + V_projected)`
`       + 0.02*(K_raw + K_projected)`

with:

`L_V2 = L_jepa + 10.0*(L_wrong + L_hold)`
`       + 0.50*(V_raw + V_projected)`
`       + 0.02*(K_raw + K_projected)`.

`L_wrong`, `L_hold`, their masks, their detached true-energy thresholds, the
detached control state, all nine real one-hot primitives, row-first candidate
averaging, and the real-hold treatment are byte-for-science identical to V1.
The hinge remains self-limiting once its registered inequality is satisfied.

The `10.0` value is fixed before execution. A first-order extrapolation from
the V1 hold signal gives a ratio near
`1/(1 + 10*0.0013951276561265935) = 0.9862406834689527`; this motivates the
single capped falsification but is not a claim that the gate will pass.
Cyclic separation remains the principal falsification risk.

Do not increase residual alpha, change AdaLN initialization, add a cyclic-only
loss, change candidate aggregation, or relax any gate. A cyclic-specific
training term would train against the acceptance sentinel and is forbidden.

## Everything else remains exact

Preserve V1 exactly:

- Raw V13 train/checkpoint-selection roles and counts;
- qualified N320 encoder initialization only;
- base seed `20260712`, schedule seed `20260713`, and the same first 16,000
  presentations;
- RGB-only current/next inputs and the nine-action vocabulary/order;
- ViT/predictor/projector architecture, float32 execution, no autocast, and
  EMA momentum `0.996`;
- residual alpha `0.1/sqrt(192) = 0.007216878364870322`;
- isolated AdaLN gate initialization, including its generator, draw order,
  weight standard deviation, and bias;
- frozen appearance projector and optimizer/clip exclusion;
- both exact patch-whitening branches, epsilon, formulas, and weights;
- optimizer, learning rates, batch geometry, gradient clip, checkpoint
  observation updates, 60-minute Phase-A cap, 1,000-update/16,000-presentation
  Phase-A maximum, and 120-minute cumulative cap;
- cyclic acceptance, hardest-wrong informational metric, real-hold metric,
  shuffled-current mapping, population counts, and mutation/RNG checks;
- every update-100, update-400, final Phase-A, and conditional Phase-B
  threshold and comparison operator;
- unchanged physical-readout Phase B, entered only after an exact final
  Phase-A pass;
- all downstream denials for G2, navigation, held-out, sealed, production,
  promotion, and deployment.

At update 100 and 400, use the same fail-fast controls as V1:

- `FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL`;
- `FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL`.

Missing any continuation conjunct terminates V2 without Phase B or retry.

## Fresh custody

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_patch_whitened_action_residual_jepa_probe_v2_action_gain`

It must be absent before reservation. V1 and every earlier attempt root or
checkpoint are historical evidence only and may not be runtime inputs.

The exact schema prefix is:

`lewm_go2_rgb_patch_whitened_action_residual_jepa_v2_action_gain`.

Use one fresh source manifest, one combined independent source/science review,
one distinct one-attempt authorization, and one terminal audit. There is no
resume, second seed, threshold edit, schedule extension, observer rerun, or
automatic V3.
