# Direct RGB→BEV JEPA V5: all-actions state-delta contrast

Date: 2026-07-26

Status: preregistered, source-only, no execution authority

Experiment ID: `go2_rgb_direct_egocentric_bev_state_jepa_v5_all_actions_state_delta_contrast`

## Decision

V4 is permanently closed. Its valid update-100 result learned useful RGB→BEV state but failed the action-mechanism gate: raster balanced accuracy reached `0.6635896329705784`, while action macro balanced accuracy was only `0.12183883000980973`, action NLL was `2.1964272181193034`, `J` was `0.6246209078364902`, and all eight scene-level hardest-wrong margins were negative. A later stale lexical failure-control validator prevented normal result publication but did not cause or invalidate the scientific failure.

V5 will not rerun V4, relax its gate, extend the same loss to update 400, change the encoder, or reuse any V4 runtime output. It tests one small, materially action-focused loss change on the otherwise exact V3/V4 learned model.

Governing V4 audit:

- Commit: `dcd509d9ded153d07c6a4513da328c92398d1b7c`
- Path: `docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v4_residual_head_hook_integrity_terminal_audit_2026-07-26.json`
- Raw SHA-256: `94d1a2f15e43d8d04f7f1e6941ae5ce5da4499f7452c297a7b5badadc673fcb2`
- Content SHA-256: `c4e04e181c713c16c14a4fcf259ede41d160bf1ba5b56e31705ba2eaff88d5ed`
- Bytes: `12,147`

## Hypothesis

V4 compared complete predicted and target maps. Most cells are static, so accurate perception and an action-independent persistence solution can dominate that energy even when the commanded action is not represented well. V5 removes that static baseline only for one auxiliary contrast: it asks which of the nine predicted *changes* best matches the learned EMA target change.

If static-map domination is the bottleneck, the frozen V3 predictor should cross the unchanged V3 update-100 action thresholds when trained with this auxiliary. If it does not, this loss-only mechanism is falsified and stops at the first failed gate.

## Exact scientific delta

Let `S` be the online current three-logit state, `P_a` the full predicted next logits for action `a`, and `T_current` and `T_next` the detached EMA target states. Let `p(x)` be softmax over the three state channels.

For batch row `b` and action `a`:

```text
predicted_delta[b,a] = p(P[b,a]) - p(S[b])
target_delta[b]      = stop_gradient(p(T_next[b]) - p(T_current[b]))
D[b,a]               = mean_c,h,w((predicted_delta[b,a] - target_delta[b])^2)
scale[b]             = stop_gradient(mean_a(D[b,a])).clamp_min(1e-4)
delta_logits[b,a]    = -D[b,a] / scale[b]
A                     = mean(cross_entropy(delta_logits, executed_action)) / log(9)
```

The frozen V3 objective is augmented with weight exactly `1`:

```text
C_v5     = C_v3 + A
total_v5 = G/log(2) + J/log(2) + C_v5
```

The online `p(S)` term is not detached. At exact persistence its gradient through the current-state identity path cancels the matching path through `p(P)`, while the residual head still receives the action-conditioned auxiliary gradient. Both EMA target terms are explicitly detached.

No raster label, pose, depth, odometry, camera geometry, ray, warp, privileged state, or navigation signal enters `A`. It is computed only from learned RGB state tensors and the already-authorized executed action.

## Frozen science

Everything except `A` is frozen V3/V4-exact:

- N320 RGB encoder initialization and encoder-only migration.
- RGB inputs, raster grounding labels, row roles, target mappings, action vocabulary, pair order, schedule bytes, seed, and draw order.
- Global learned BEV decoder, three-logit state bottleneck, coordinate-aware FiLM U-Net predictor, zero residual-head initialization, and EMA target stack.
- `G`, `J`, and the original `C_v3` computation are frozen, as are the optimizer, learning rates, weight decay, clipping, EMA momentum, precision, observations, and existing perception/retrieval metrics. The reported and traced `C` scalar alone becomes the preregistered `C_v5 = C_v3 + A`.
- Parameter inventory: predictor `317,107` parameters / `79` tensors; total `6,552,249` parameters / `277` tensors.
- Fresh initial model state SHA-256: `84748bc66f0639b9dae1c81880f5c0fa756f4c4d9e75d0ffddac1310c7d05d0a`.

V5 adds no parameter, buffer, module, state call, target call, output head, data field, or supervision channel.

## Integrity and gates

Update zero must preserve every V4 integrity conjunct, including exact persistence, bitwise-equal nine-action predictions, chance action retrieval, `3/1/3` online/predictor/target call counts, nonzero intended gradients, target isolation, fixed-negative isolation, and the exact fresh-state hash. In addition:

- `C_v5` must be finite and lie in `[1.99, 2.01]`, proving that the normalized delta contrast is active at its chance value of approximately `1`.
- The model parameter/tensor inventory must remain exact V3.
- Target delta tensors must be detached.

The update-100 V3 gate is unchanged:

- action macro balanced accuracy `>= 0.13`;
- action NLL `<= 2.187`;
- hardest-wrong-positive scenes `>= 2`;
- aggregate raster balanced accuracy `>= 0.65`;
- `J <= 0.60`;
- all inherited finite, directional, state-nonconstant, correct-RGB, and prior-gate conjuncts.

The update-400 and update-1000 V3/V4 gates and final perception qualification thresholds are unchanged. Existing absolute next-state retrieval metrics, not the new auxiliary logits, decide promotion. Stop at the first failed conjunct.

## Caps and lifecycle

- One fresh attempt only.
- Maximum `1,000` updates and `16,000` presentations.
- Observations at updates `0`, `100`, `400`, and `1,000` only.
- Maximum `60` active GPU minutes.
- No retry, resume, repair, recovery, second seed, or checkpoint/runtime-output reuse.
- V3 and V4 checkpoints, tensors, traces, and runtime outputs are forbidden inputs.
- A distinct output root must be absent before reservation:
  `.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_direct_egocentric_bev_state_jepa_probe_v5_all_actions_state_delta_contrast`

Passing establishes only that the V5 checkpoint met this development perception/world-model probe. It may become a candidate for a separately reviewed and separately authorized G2 step; it does not authorize checkpoint read/use, G2, navigation, held-out, sealed, production, deployment, or promotion access.

## Receipt adapter closure

Before execution, V5 must define or bind a version-local `validate_failure_status_chain` against V5's own `FAILURE_CONTROLS`. A behavioral test must accept every V5 failure control and reject mismatched chains and pass controls. This is an operational receipt fix only; it cannot change training, metrics, gates, or scientific status.

## Falsification rule

Scale only through the already-preregistered checkpoints while every gate passes. A weak above-chance movement that still misses the unchanged update-100 gate is a failure, not permission to relax the threshold or repeat the mechanism.
