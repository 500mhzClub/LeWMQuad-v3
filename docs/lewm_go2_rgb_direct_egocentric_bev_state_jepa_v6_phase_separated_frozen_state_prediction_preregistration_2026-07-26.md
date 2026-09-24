# Direct RGB→BEV JEPA V6: phase-separated frozen-state prediction

Date: 2026-07-26

Status: preregistered, source-only, no execution authority

Experiment ID: `go2_rgb_direct_egocentric_bev_state_jepa_v6_phase_separated_frozen_state_prediction`

## Decision

V5 is permanently closed. It passed update-zero integrity and delivered about `49.2×` the V4 predictor gradient, but its valid update-100 result made both the shared state and action discrimination worse: raster balanced accuracy was `0.6076338204838655`, free recall was `0.06446429009543904`, `J` was `0.7173812877048146`, action macro balanced accuracy was `0.10911680911680913`, action NLL was `2.1971453213932537`, and all eight scene-level hardest-wrong margins were negative. It stopped after exactly `100` updates and `1,600` presentations.

This falsifies the V5 all-actions state-delta contrast mechanism. V6 will not retry it, carry its auxiliary `A` forward, relax its gates, reuse its checkpoint, or extend its run. V4 had already shown that the frozen V3 architecture can learn materially useful RGB→BEV perception while joint predictor learning remains weak. Together, V4 and V5 support one materially different test: remove shared-gradient and moving-representation interference by learning perception first, then learning dynamics against a frozen state.

Governing V5 audit:

- Commit: `458f590605178f1460d043a48ed629c181f593a4`
- Path: `docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v5_all_actions_state_delta_contrast_terminal_audit_2026-07-26.json`
- Raw SHA-256: `e4c9a329322e641b9c096ae3bc163876991e4d90c1bb24dc48146a2dd30acd20`
- Content SHA-256: `b89afbcfaff1a703bb924f5cc028613bd927c316db5a0c1066bccae3e567526e`
- Bytes: `11,272`

## Hypothesis

The predictor has so far been trained while the online RGB→BEV state and its EMA target change on every update. A predictor gradient therefore chases a moving coordinate system and also pushes on the same perception stack that defines that coordinate system. V5 proved that simply making this gradient much larger does not solve the problem and can damage the learned state.

V6 tests whether the existing V3 JEPA predictor can learn action-conditioned dynamics when its input and target representations are stable. Phase one learns only RGB→BEV grounding. If perception reaches the already-registered final perception standard, V6 performs one exact online-to-target synchronization and freezes both stacks. Phase two then learns only the predictor with the original V3 JEPA and contrastive losses.

This is a full JEPA training test, not a new encoder test. The encoder, decoder, three-state bottleneck, and predictor architecture are unchanged. The scientific delta is only the optimization schedule and phase boundary.

## Frozen model and data

V6 starts fresh from the exact V3/V4/V5 initialization; it does not read any prior runtime output.

- N320 RGB encoder initialization and encoder-only migration are unchanged.
- Fresh initialization seed, module construction order, and complete initial state SHA-256 remain `84748bc66f0639b9dae1c81880f5c0fa756f4c4d9e75d0ffddac1310c7d05d0a`.
- The global learned BEV decoder, three-logit state bottleneck, coordinate-aware FiLM U-Net predictor, and exact-zero residual-head initialization are unchanged.
- Parameter inventory remains predictor `317,107` parameters / `79` tensors and total `6,552,249` parameters / `277` tensors.
- RGB inputs, raster grounding labels, row roles, mappings, action vocabulary, pair order, schedule bytes, seed, microbatches, effective batch size, precision, learning rates, weight decay, clipping, and observation rows remain frozen.
- No depth, pose, odometry, camera geometry, ray, warp, privileged simulator state, navigation label, or held-out signal is added.
- V5's auxiliary state-delta contrast `A` is absent. V6 uses only the original V3 `G`, `J`, and `C` definitions.

One frozen V3 AdamW optimizer is constructed once and is never rebuilt or reset. It retains the exact two parameter groups and hyperparameters. At each phase, inactive parameters have `requires_grad=False` and `grad=None`; therefore AdamW cannot update or weight-decay them. Bitwise state hashes at the phase gates must prove this isolation.

## Exact two-phase objective

Let `G`, `J`, and `C` have their exact V3 meanings. The phase is selected by the global update before its optimizer step.

### Phase one: perception grounding

Updates `1..400`, presentations `1..6,400`:

```text
L_phase1 = G / log(2)
trainable = online RGB encoder + BEV decoder + three-logit state head
frozen    = predictor + detached target stack
```

`J` and `C` may be evaluated for unchanged diagnostics, but neither contributes to the backward scalar. Predictor parameters must receive no gradient and no optimizer state update. The target stack remains detached and receives the frozen EMA update with momentum `0.996` after each of the 400 phase-one optimizer steps.

Immediately after the 400th EMA update, and before the update-400 observation, the target encoder, target BEV decoder, and target state head are copied exactly once from their online counterparts. This boundary synchronization must not reset the global-update counter.

### Phase two: frozen-state JEPA prediction

Updates `401..1,000`, presentations `6,401..16,000`:

```text
L_phase2 = J / log(2) + C
trainable = coordinate-aware FiLM U-Net predictor only
frozen    = online RGB encoder + BEV decoder + state head + target stack
```

`G` may be evaluated for unchanged diagnostics, but it does not contribute to the backward scalar. Online state tensors and all target state tensors are frozen with respect to optimization. No target EMA arithmetic or target copy occurs in phase two. The inherited update callback may advance its global call counter to preserve the runner's exact update accounting, but it must be a target-state no-op for all 600 phase-two steps. Receipts must distinguish exactly `400` EMA arithmetic updates, `1` boundary hard synchronization, and `600` target no-ops.

For every phase-two forward, the online encoder, online BEV decoder, online state head, and complete detached target stack must be in deterministic evaluation mode; only the predictor is in training mode. This policy is reapplied after every observation because the inherited observation wrapper temporarily changes module modes. The online and target perception states must be bitwise identical at the boundary and remain bitwise unchanged through update 1,000. A fixed all-zero RGB integrity witness, used only as a non-training diagnostic, must produce bitwise-identical repeated online states and bitwise-identical repeated target states at updates 400 and 1,000. Together these checks make the predictor's input and target coordinate system stationary in parameters, buffers, module mode, and repeated execution.

## Update-zero integrity

Before training, all inherited structural and initialization checks must pass:

- exact initial model-state SHA-256 and exact parameter/tensor inventory;
- exact three-logit bottleneck and no hidden or auxiliary state bypass;
- exact-persistence prediction, bitwise-equal predictions for all nine actions, action NLL equal to `log(9)`, and action macro balanced accuracy equal to `1/9` within the frozen tolerance;
- finite, nonconstant registered state and detached target tensors;
- no prior runtime, checkpoint, trace, navigation, held-out, or sealed input.

A non-mutating dual gradient probe must additionally prove both registered phase paths:

- Phase-one scalar `G/log(2)` gives finite nonzero gradients to the encoder and decoder/state groups, with predictor and target gradients exactly absent.
- Phase-two scalar `J/log(2)+C`, evaluated with perception frozen, gives a finite nonzero predictor gradient, with online perception and target gradients exactly absent.
- Each probe preserves the registered online/predictor/target call boundary, next-RGB grounding isolation, fixed-negative optimizer isolation, prior gradients, parameter trainability flags, module modes, RNG state, optimizer state, and model state.

Failure of any update-zero conjunct is terminal and grants no repair or retry.

## Update-100 directional perception gate

Update 100 is phase one and must satisfy every conjunct below:

- `G < 0.90 × G_update0`;
- aggregate raster balanced accuracy `>= 0.65`;
- aggregate raster NLL is strictly lower than update zero;
- rough raster balanced accuracy is strictly higher than update zero;
- correct-RGB scene wins `>= 6/8`;
- all registered values are finite and the state is nonconstant;
- predictor parameter/state SHA-256 is exactly its initialization SHA-256;
- predictor residual head remains exact zero;
- all nine action predictions remain bitwise equal exact persistence;
- action NLL remains equal to `log(9)` and action macro balanced accuracy remains equal to `1/9` within the frozen tolerance;
- global target-update callback count is `100`, with exactly `100` EMA arithmetic updates and no boundary sync.

There is deliberately no action-improvement or `J`-improvement requirement at update 100 because the predictor has received zero training updates.

## Update-400 perception qualification and boundary gate

After phase-one optimizer step 400, EMA update 400, and the one boundary hard sync, perception will never change again. It must therefore meet the existing final perception qualification thresholds at update 400:

- aggregate raster balanced accuracy `> 0.9009460724448773`;
- aggregate free recall `> 0.91637020862468`;
- aggregate occupied recall `> 0.8059679976935274`;
- aggregate raster NLL `< 0.18704089070408247`;
- rough raster balanced accuracy `> 0.7719525130620232`;
- rough raster occupied recall `> 0.4319466882067851`;
- correct-RGB scene wins `8/8`;
- all registered values are finite and the state is nonconstant.

The phase-boundary integrity conjuncts are also mandatory:

- predictor parameter/state SHA-256 still exactly equals initialization, its residual head is exact zero, all action predictions are exact persistence, and action retrieval remains exact chance;
- online and target perception stacks are byte-identical after synchronization;
- the online and target perception stacks are both in evaluation mode, the predictor alone is in training mode, and the fixed all-zero RGB witness is bitwise deterministic for each stack across two repeated calls;
- exactly `400` perception optimizer updates, `0` predictor optimizer updates, `400` EMA arithmetic updates, and `1` boundary hard sync occurred;
- the online perception state SHA-256, target perception state SHA-256, complete predictor SHA-256, and all perception metrics are recorded as immutable phase-two baselines;
- a non-mutating boundary probe of `J/log(2)+C` proves that only predictor parameters receive gradients.

Any miss stops V6 at update 400. A merely improving but unqualified state does not authorize predictor training within this attempt.

## Update-1,000 predictor qualification gate

After exactly 600 predictor-only updates, every conjunct below is required:

- online and target perception SHA-256 values are exactly unchanged from update 400 and remain equal to each other;
- phase-two module modes remain exact and the fixed all-zero RGB witness remains bitwise deterministic for both perception stacks;
- every registered perception metric is bitwise or exact-serialized equal to its update-400 value and remains above the update-400 qualification thresholds;
- `J <= 0.90 × J_update400_boundary`;
- `C < C_update400_boundary`;
- action NLL `< 0.95 × log(9) = 2.0873633484694083`;
- action macro balanced accuracy `> 2/9`;
- hardest-wrong-positive scene count `>= 6/8`;
- same-action target NLL `< 0.95 × log(2) = 0.658489821531948`;
- same-action target strict-win rate `>= 0.65`;
- target-positive scene count `>= 6/8`;
- correct-RGB scene wins remain `8/8`;
- all registered values are finite and the state is nonconstant;
- exactly `400` perception optimizer updates, `600` predictor optimizer updates, `400` EMA arithmetic updates, `1` boundary hard sync, `600` phase-two target no-ops, `1,000` optimizer steps, and `16,000` presentations occurred.

Existing absolute next-state and action-retrieval metrics decide success. No auxiliary classifier, linear probe, oracle state, or newly invented score may substitute for these gates.

## Caps, custody, and lifecycle

- One fresh V6 attempt only.
- Maximum `1,000` updates and `16,000` presentations.
- Observations only at updates `0`, `100`, `400`, and `1,000`.
- Maximum `60` active GPU minutes.
- Stop at the first failed gate.
- No retry, resume, repair, recovery, second seed, threshold relaxation, or checkpoint/runtime-output reuse within V6.
- V3, V4, and V5 checkpoints, snapshots, tensors, traces, and runtime outputs are forbidden inputs.
- The V4 and V5 rejected checkpoints remain permanently unopened.
- The distinct V6 output root must be absent before reservation:
  `.generated/go2_shared_observable_camera_ray_jepa_v6/rgb_direct_egocentric_bev_state_jepa_probe_v6_phase_separated_frozen_state_prediction`
- Every written snapshot is write-only until separately qualified and authorized; a failed-gate snapshot is permanently rejected and must never be opened.

Passing establishes only that the V6 development perception/world-model probe met its registered gates. It may become a candidate for a separately reviewed and separately authorized G2 step. It does not authorize checkpoint read/use, G2, navigation, held-out, sealed, production, deployment, or promotion access.

## Falsification and next decision

V6 tests one mechanism once: stable phase-separated representation learning. It is not permission for repeated phase-boundary, loss-weight, or threshold tuning.

- Failure before update 400 means this grounding schedule did not qualify the learned perception state under the frozen data and architecture. The receipt may motivate one separately preregistered material perception change if progress is informative; it does not authorize a V6 retry.
- Passing update 400 but failing update 1,000 means a stable static three-state representation is insufficient for the existing action predictor. The next mechanism should add learned temporal context or memory inside the perception-only JEPA stack, not another scalar loss weight or another retry of the static predictor.
- Passing update 1,000 permits only the already-ordered, separately authorized G2→G8 promotion path. The sealed V4 30-scene benchmark remains unopened until its governing gates explicitly permit access.
