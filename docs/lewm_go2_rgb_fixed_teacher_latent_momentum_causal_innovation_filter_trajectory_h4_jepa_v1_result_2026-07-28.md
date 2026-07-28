# Go2 RGB fixed-teacher latent-momentum causal innovation-filter trajectory H4 JEPA V1 result — 2026-07-28

## Terminal outcome

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_LATENT_MOMENTUM_CAUSAL_INNOVATION_FILTER_TRAJECTORY_H4_JEPA_V1`.
- The sole reserved probe completed normally at the exact cap: 1,000 optimizer
  updates, 16,000 ordered training presentations, 10,240 validation
  presentations, and `793.893503608997` active-GPU seconds.
- Update 1,000 / presentation 16,000 was selected by the frozen minimum
  combined-energy rule. The result passed 29 of 32 gates and failed only the
  three ordered-history gates.
- This is a scientific STOP, not an execution, receipt, collapse, action,
  HOLD, target-integrity, or custody failure. The exact mechanism is closed
  without retry, resume, extension, second seed, checkpoint opening, or
  post-hoc checkpoint choice.
- Operator-observed event outside the canonical receipt chain: a preliminary
  launcher command omitted the mandatory `--execute` flag and was rejected by
  argument parsing before reservation. The exclusive root was then observed
  absent. The canonical receipts begin with the subsequent sole fresh
  reservation and record zero retry/resume.

## What was tested

- Four equal-mass latent states each contained only feature-lattice content
  `q` and tangent latent momentum `v`. These have no physical units and are
  not pose, metric velocity, geometry, or hand-written navigation state.
- One shared prior was used on all six edges. A complete nine-action learned
  categorical tower was centered before selection, and one shared bias-free
  zero-initialized head produced action-conditioned latent acceleration.
- On `p0` and `p1`, the prior was emitted and scored before the same observer
  assimilated the newly available RGB feature innovation. After `e2`, only
  packed `(q2,v2)` remained; `p2:p5` recursed state-only with no future RGB,
  raw `z2`, explicit incoming difference, anchor, or horizon query.
- The online encoder, modes, observer, state context, action tower, and
  acceleration head were trained jointly in one JEPA graph and one summed
  backward. The accepted N320 RGB encoder was the only fixed target.
- The unchanged objective was weight-one online/fixed-teacher history
  alignment, half all-six local proper energy score, and half cumulative H4
  proper energy score. All action, HOLD, history, persistence, and collapse
  controls remained evaluation-only.

## Selected aggregate result

| Measure | Update 1,000 value |
|---|---:|
| Combined normalized energy score | 0.758406567107 |
| Joint-trajectory normalized energy score | 0.755083531641 |
| Future `p2:p5` local combined score | 0.834626731604 |
| Pre-observation `p0:p1` local-prior combined score | 0.791719023489 |
| Pre-observation `p0:p1` persistence gap | +0.208280976145 |
| H4 marginal normalized energy score | 0.844133202913 |
| H4 persistence gap | +0.155866797108 |
| H4 cyclic-action gap | +0.067419811407 |
| H4 cyclic-action bootstrap lower 95% | +0.059211505646 |
| H4 ordered-history gap | -0.039452624975 |
| H4 ordered-history bootstrap lower 95% | -0.057519764818 |
| H4 all-HOLD gap | +0.111124921749 |
| Combined distribution-value gap | +0.263655679872 |
| H4 normalized pairwise spread | 1.456827871544 |
| H4 best-atom normalized squared error | 2.321136442925 |
| H4 centroid normalized squared error | 3.399911368788 |

- Requested-action sensitivity remained valid: H4 action value exceeded the
  `0.05` gate, its bootstrap lower bound was positive, and all eight families
  were positive.
- The filter solved the predecessor's HOLD-breadth weakness. H4 HOLD value was
  strongly positive, all eight families were positive, and the minimum family
  gap was `+0.075916814605`.
- Generic prediction and the pre-observation priors beat persistence with
  positive bootstrap and eight-family breadth. The learned four-atom
  distribution added value in all eight families and remained noncollapsed.
- Correct ordered history was counterproductive in every family. Its aggregate
  H4 gap and bootstrap lower bound were negative, and zero of eight families
  were positive. These are the only failed gates.
- The distribution is not planner-quality. Best-atom and centroid H4 squared
  errors remain well above persistence-normalized `1.0`, and the centroid is
  particularly poor despite a good proper energy score.

## Learning trajectory

| Update | Presentations | Combined | H4 score | H4 action | H4 history | H4 HOLD | Action-positive families | History-positive families | HOLD-positive families |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.000000 | 1.000000 | 0.000000 | -0.000000 | 0.000000 | 0 | 0 | 0 |
| 250 | 4,000 | 0.799004 | 0.920441 | +0.053590 | -0.063121 | +0.076726 | 8 | 0 | 8 |
| 500 | 8,000 | 0.801678 | 0.930879 | +0.140228 | -0.063700 | +0.150294 | 8 | 0 | 8 |
| 750 | 12,000 | 0.765481 | 0.860189 | +0.123046 | -0.035238 | +0.084773 | 8 | 0 | 8 |
| 1,000 selected | 16,000 | 0.758407 | 0.844133 | +0.067420 | -0.039453 | +0.111125 | 8 | 0 | 8 |

- Action and HOLD support appeared by update 250 and remained broad through
  the cap. Generic combined prediction improved substantially after update
  500 and was still improving at update 1,000, correctly selecting the cap.
- Ordered-history value was negative at every trained observation. It became
  less harmful between updates 500 and 750, then regressed slightly. More
  updates are not justified: the sign never changed, all eight families agree,
  and the exact cap and mechanism are closed.
- The best atom improved from `4.286141` at update 250 to `2.321136` at update
  1,000, but never approached the planner-relevant persistence baseline.

## Comparison with factorized conditional-increment V1

- Combined and joint scores improved by `0.018809` and `0.019016` respectively
  (lower is better), and future-local and `p0:p1` scores also improved.
- HOLD gap improved from `+0.005951` with three positive families to
  `+0.111125` with eight positive families. This is the clear success of the
  second-order filter.
- Action remained passing but weakened from `+0.108100` to `+0.067420`.
- Ordered history worsened from `-0.012314` to `-0.039453`; both mechanisms
  had zero positive families. Adding latent momentum and post-prior innovation
  assimilation therefore did not create useful temporal state.
- H4 marginal score, best-atom error, and centroid error worsened. Particle
  spread increased from `1.400538124091` to `1.456827871544`; this is a
  lower-bound diversity diagnostic, not a monotone error measure. The lower
  overall proper score must not be mistaken for a planner-ready representation.

## Receipt and custody audit

- All six exact canonical JSON receipts are present and independently
  self-bound and cross-bound. Completion file SHA-256:
  `4f5f5be0e44edf6684ff021643f9efaccee8ba3f3197da92f479ff726f6dc3a0`;
  completion content SHA-256:
  `ee3aa5a22b237ad2ec4ba2d13fdcfd5cbea34f1879791d9667e1df0594197655`.
- The fixed target's initial and final state hashes are both
  `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`;
  target EMA updates are zero.
- Access is complete: 1,000 optimizer updates, 16,000 training sequences,
  10,240 validation sequences, and exactly 183,680 successful RGB opens.
  Test/held-out, sealed, label, navigation, retry/resume, arbitrary
  initialization, predecessor-predictor checkpoint, and target-EMA counters
  are all zero.
- Runtime checkpoints and traces remain unopened write-only artifacts. This
  audit read only the six exact canonical JSON receipts and did not discover,
  list, stat, hash, or open any runtime checkpoint.

## Scientific conclusion and next boundary

- The useful idea to retain is the centered categorical action mechanism plus
  causal state recursion: it gives broad action, HOLD, persistence, and
  distribution value.
- The idea to close is token-local latent momentum with same-lattice
  innovation assimilation. It stores ordered history, but that state changes
  forecasts in the wrong direction across every family.
- A successor must change what history represents, rather than tune this
  filter, repeat the seed, extend training, alter thresholds, or revisit the
  schedules. Earlier local-correspondence and dense-history branches are
  already closed. The leading boundary is an action-attributed causal
  system-identification memory: use observed prior errors and their requested
  past actions to infer a compact sequence-specific dynamics code, then let
  that code modulate—but never bypass—the successful centered-action
  transition. It must remain RGB-only, causal, perception-only, and jointly
  trained as one JEPA on the same unopened held-out progression.
- This STOP grants no checkpoint access, navigation, G2, held-out/sealed
  access, promotion, production, or deployment authority.
