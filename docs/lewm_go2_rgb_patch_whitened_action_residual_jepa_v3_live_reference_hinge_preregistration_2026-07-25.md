# RGB Patch-Whitened Action-Residual JEPA V3 Live-Reference Hinge preregistration

Date: 2026-07-25

## Decision

Authorize source preparation and one independent combined source/science
review for exactly one V3 Live-Reference Hinge successor probe.

Execution is not authorized by this document. It requires a fresh exact source
manifest, a passing independent review, and a distinct one-attempt execution
authorization.

## Bound V2 evidence

V2 Action-Gain is a valid terminal update-100 scientific failure. Its audit is:

- path:
  `docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v2_action_gain_terminal_audit_2026-07-25.json`;
- commit: `e7670b82bd4d31cba2b6d9b76fb8c11c04e1f18d`;
- file SHA-256:
  `cb0d0f789bfd6d0ec861b19c597a9c203d9d93eb1f0f2c89c04876579eb2b405`;
- content SHA-256:
  `1deef9dd068ade6556dd3eecb87f1ee7896acc0394e8eb9dab943d03749d2c87`;
- byte count: `14,618`;
- status:
  `PASS_EXACT_VALID_SCIENTIFIC_FAILURE_TERMINAL_NO_RETRY`.

V2 stopped at 100 optimizer/EMA updates and 1,600 presentations. Phase B was
not entered. The auditor opened only the six JSON receipts and did not open,
hash, parse, copy, or deserialize the checkpoint or training trace.

The tenfold coefficient did not amplify usable action signal:

- raw and projected effective rank remained essentially identical to V1 at
  `53.53243637084961` and `41.314334869384766`;
- the real-hold ratio worsened from `0.9986068160133832` to
  `0.9999506233699988`, and its absolute margin retained only
  `0.041930847916138304` of V1;
- the cyclic ratio moved only to `0.9999968834906031`, with `5/8` positive
  families, versus the unchanged requirements `<0.99` and at least `6/8`;
- true/shuffled-next, true/shuffled-current, and true/mean-target weakened to
  `0.8012107172357854`, `0.8015246138767856`, and
  `0.9297111996093846`;
- all integrity, population, whitening-health, rank, finite-value, RNG, and
  mutation checks passed.

V2 may not be retried, resumed, extended, or scalar-retuned.

## Structural diagnosis

The V2 action hinge used:

`threshold = stop_gradient(E_true) / 0.95`

and minimized:

`relu(threshold - E_control)`.

Therefore its action-discrimination gradient could repel a control prediction
but could not reduce the registered true-action energy. The control state was
also detached, as intended for stability. Multiplying this one-sided gradient
by ten preserved whitening but increased true-pair error by `18.82%` and
collapsed the already-small real-hold margin.

The ordinary JEPA term still attracted the true branch with weight one, but it
was not the matched live side of the tenfold relative comparison. This is a
specific gradient-topology defect, not evidence for another coefficient or
residual-alpha sweep.

## Exactly one scientific change

Replace only:

`threshold_V2 = stop_gradient(E_true) / 0.95`

with:

`threshold_V3 = E_true / 0.95`.

Keep:

`ACTION_DISCRIMINATION_WEIGHT = 10.0`.

The total remains:

`L = L_jepa + 10.0*(L_wrong + L_hold)`
`    + 0.50*(V_raw + V_projected)`
`    + 0.02*(K_raw + K_projected)`.

This makes the existing exhaustive relative hinge bidirectional: active
comparisons pull the requested-action prediction toward the observed next
latent while pushing the same-row counterfactual actions away. All control
states remain detached. The EMA next target remains detached. Every row still
uses all eligible real wrong actions, and the cyclic action remains an
acceptance sentinel only.

Do not:

- change the `10.0` coefficient;
- change residual alpha;
- change AdaLN initialization;
- add a cyclic-specific objective;
- add an inverse-dynamics head;
- add an action-indexed output head;
- relax a gate or alter a comparison operator.

## Everything else remains exact

Preserve V2 exactly:

- Raw V13 train and checkpoint-selection roles, counts, bytes, and hashes;
- qualified N320 online and EMA encoder initialization only;
- base seed `20260712`, schedule seed `20260713`, and the same first 16,000
  presentations;
- RGB-only current/next inputs and the exact nine-action vocabulary/order;
- ViT, predictor, projector, and evidence-head architecture;
- float32, no autocast, EMA momentum `0.996`, AdamW groups and learning rates,
  weight decay, epsilon, and global clip `1.0`;
- residual alpha `0.1/sqrt(192)`;
- isolated AdaLN generator, draw order, gate-row initialization, standard
  deviation, and bias;
- frozen appearance projector and optimizer/clip exclusion;
- both exact patch-whitening branches and their formulas, epsilon, and weights;
- detached control state, all-action row-first averaging, real-hold population,
  and empty-hold behavior;
- update observations at 100, 400, and 1,000;
- Phase-A cap of 1,000 updates, 16,000 presentations, and 60 active GPU
  minutes;
- cumulative cap of 2,000 updates, 32,000 presentations, and 120 active GPU
  minutes;
- every update-100, update-400, final Phase-A, and conditional Phase-B gate;
- conditional Phase B, entered only after an exact final Phase-A pass;
- all denials for G2, navigation, held-out, sealed, production, promotion, and
  deployment.

The unchanged update-100 gate requires all of:

- raw effective rank strictly greater than `27.717458724975586`;
- projected effective rank strictly greater than `17.426651000976562`;
- true/cyclic-wrong ratio strictly less than `0.99`;
- non-hold-true/real-hold ratio strictly less than `0.99`;
- positive cyclic and hold margins in at least `6/8` families;
- exact populations, finiteness, EMA-gradient freedom, RNG preservation, and
  zero model-state mutation during observation.

Failure of any conjunct must publish
`FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL` and stop without Phase B,
retry, resume, or observer rerun.

## Fresh custody

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_patch_whitened_action_residual_jepa_probe_v3_live_reference_hinge`

It must be absent before reservation. V1, V2, and every earlier runtime root,
checkpoint, and trace are historical evidence only and may not be runtime
inputs.

The exact schema prefix is:

`lewm_go2_rgb_patch_whitened_action_residual_jepa_v3_live_reference_hinge`.

Use one fresh source manifest, one independent combined source/science review,
one distinct one-attempt authorization, and one terminal audit. There is no
retry, second seed, threshold edit, schedule extension, observer rerun, or
automatic V4.

If V3 fails any update-100 or update-400 continuation gate, the final Phase-A
gate, or conditional Phase B, terminate this live-reference hinge plus
small-open AdaLN residual family. Any later candidate requires a separately
preregistered explicit action-indexed residual mechanism; it is not authorized
here.
