# Go2 RGB fixed-teacher latent-momentum causal innovation-filter trajectory H4 JEPA V1 source review — 2026-07-28

## Review decision

- Status: `CLEAR_FOR_ONE_BOUND_PREFLIGHT_AND_ONE_ATTEMPT_IF_PREFLIGHT_PASSES`.
- Preregistration commit:
  `67121f034f13df0126c43aa1673bbd2a78e72d33`.
- Model, runner, and proof-test commit:
  `3a40aabada4b914d0cc5dba02c3680bc57a771e2`.
- This review authorizes one non-training bound preflight and, only if that
  passes, one fresh 1,000-update / 16,000-presentation attempt. It authorizes
  no retry, resume, extension, predecessor-checkpoint read, navigation,
  held-out/sealed access, promotion, or deployment.

## Frozen primary bindings

| Source | SHA-256 | Bytes |
|---|---|---:|
| Preregistration | `9ac4ab029c59f5ab54b88f9574a2903ea87fa59e163c82b1dbe2969f17afa3c1` | 13,369 |
| Latent-momentum model | `46fe5f22ff7b2416f9f6bdc4feb362d895b183ba6266b9db96ff98e4eaf9eb3e` | 16,417 |
| Bound runner | `398bd99a9fdb9f9e6d7b6ef54089f96440fadf7544d0eedabf68344e0fec4e59` | 19,659 |
| Model proof tests | `668f3a1e73fd9ac7f0b47ee5e9a46a77474a753e98efd57ba28731fa75877307` | 16,955 |
| Runner proof tests | `0f84dea0fa40c69f8611e4cf478de1a847ce0be6404edcfa3390238aa4ea428a` | 15,760 |

- The runner's external self-binding is exactly:
  - `LEWM_LATENT_MOMENTUM_CAUSAL_INNOVATION_FILTER_TRAJECTORY_H4_V1_WRAPPER_SHA256=398bd99a9fdb9f9e6d7b6ef54089f96440fadf7544d0eedabf68344e0fec4e59`
  - `LEWM_LATENT_MOMENTUM_CAUSAL_INNOVATION_FILTER_TRAJECTORY_H4_V1_WRAPPER_BYTES=19659`
- Source-only closure verification resolved all 16 frozen sources, including
  the exact factorized predecessor wrapper/model and every inherited V2
  schedule, factual evaluator, trajectory model, shared runner, and encoder
  dependency.

## Scientific review

- The predictive state is exactly four equal-mass `(q,v)` atoms: normalized
  feature-lattice content and tangent latent momentum. It has no physical
  units, pose, geometry, or hand-written motion semantics.
- The same prior transition is called on all six edges. It receives only
  `(q,v,current categorical action)` plus learned centered modes and shared
  parameters. The complete nine-action learned tower is mean-centered before
  selection, and one shared bias-free zero-initialized head produces the
  action-conditioned acceleration.
- The two history edges predict and emit their priors before the same observer
  assimilates `z_(t+1)-q^-_(t+1)`. After the second observation, the belief
  contains only packed `(q2,v2)`. The four future edges recurse state-only;
  future RGB remains confined to the fixed no-grad target encoder.
- Update zero is persistence: `q0=z0`, `v0=0`, unit observer gain, zero
  momentum correction, and zero acceleration. Learned mode rows only permute
  equal-mass atoms and leave the centroid invariant.
- Observed local predictions are scored as `q^-_(t+1)-z_t`; future local
  predictions are recursively realized `q^-_(t+1)-q_t`. The cumulative future
  trajectory is the same recursively produced `q` sequence.
- The exact three-term jointly trained JEPA objective is retained: weight-one
  online/fixed-teacher history alignment, half all-six local proper energy
  score, and half cumulative H4 proper energy score. One summed backward and
  optimizer step jointly train the online encoder, modes, observer, state
  context, action tower, and acceleration head. The accepted N320 target is
  fixed, no-grad, and receives no EMA update.
- There is no separately trained predictor, raw `z2`/incoming-delta bypass,
  anchor, factual overwrite inside the predictor, horizon query, future RGB
  input, action classifier/ranking loss, reconstruction, navigation loss,
  flow, warp, BEV, pose, or geometry target.

## Test and inventory evidence

- Reviewed Torch runtime:
  `/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python`.
- Focused plus inherited compatibility suite: 117 passed, zero failed and zero
  warnings in 5.69 seconds. It covers the new model/runner and the exact
  factorized, factual V1, V2 schedule-integrity, local-innovation, and
  trajectory-distribution ancestors.
- Full-size source-only parameter inventory:
  - encoder: 2,747,520 scalars / 78 tensors;
  - history/filter: 679,104 scalars / 19 tensors;
  - predictor: 594,624 scalars / 20 tensors;
  - total trainable: 4,021,248 scalars / 117 tensors;
  - fixed target: 2,747,520 non-trainable scalars / 78 tensors.
- Parameter groups are disjoint, cover every trainable tensor, and exclude the
  fixed target. Tests prove q/v geometry, action/mode centering, tangent
  projection, update-zero persistence, six-prior/two-observer order, no future
  leakage, q/v-only belief, history/action sensitivity after head opening,
  exact realized-increment and loss arithmetic, target detachment, gradient
  staging and opened gradients through every learned route, optimizer coverage,
  and K4 mode-permutation invariance.

## Runner, receipts, and custody review

- The exact causal V2 indexes, evaluator, selection rule, bootstrap procedure,
  thresholds, and all 32 gates are inherited without relaxation or addition.
  Seed `20260727`, batch `16`, observations `0/250/500/750/1000`, 1,000
  updates, 16,000 train presentations, 10,240 validation presentations,
  183,680 expected RGB opens, 1,000 bootstrap replicates, and 5,400 active-GPU
  seconds are frozen.
- Runner tests reject every train/validation/model binding substitution and
  every retry, resume, seed, update, presentation, batch, GPU-cap, and
  arbitrary-checkpoint override surface.
- Configuration and terminal metrics use the exact new objective wording.
  The adapter verifies and truthfully relabels both training-loss buckets and
  removes predecessor mechanism claims before adding the `(q,v)` filter
  contract.
- The exact inherited complete terminal handler remains installed. It derives
  forbidden counters from registered access state and writes cross-bound
  normal or caught-failure JSON receipt chains without granting scientific or
  checkpoint authority.
- Exclusive root:
  `.generated/go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_trajectory_h4_jepa_v1/probe_v1`.
  Preflight is zero-RGB and zero-reservation. Runtime checkpoint/trace files
  are write-only; terminal review is restricted to the six exact canonical
  JSON receipts and may never list, stat, hash, or open a runtime checkpoint.

## Independent review history

- Independent model/science review: CLEAR on the frozen model and model-proof
  tests after checking the full preregistration.
- Independent runner/custody review found and closed two real pre-freeze gaps:
  stale predecessor objective wording and incomplete override/accounting test
  coverage. A second pass then found and closed an unreachable mocked
  objective state that would have caused a post-training terminal failure.
- Final independent runner/custody re-review: CLEAR. No review accessed runtime
  outputs, checkpoints, RGB, indexes, metadata, navigation, held-out, or sealed
  material.
