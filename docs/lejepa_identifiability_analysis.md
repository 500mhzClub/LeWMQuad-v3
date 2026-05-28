# LeJEPA Identifiability — Implications for v3 Data Collection

Analysis of Klindt, LeCun, Balestriero, *"When Does LeJEPA Learn a World Model?"*
(arXiv:2605.26379v1, 25 May 2026), against the v3 collection policy in
[fresh_retrain_data_spec.md §13](fresh_retrain_data_spec.md) and
[collection_policy.md](collection_policy.md).

This is an interpretive doc, not a spec change. Concrete spec changes that fall
out of it are tracked as next-step items at the bottom.

## 1. What the paper proves

LeJEPA (alignment loss + SIGReg Gaussian regularization on embeddings, the
exact recipe scaled by LeWorldModel [paper ref 9]) **linearly identifies the
world's latent variables, up to an orthogonal rotation, iff**:

1. The world's latents are Gaussian, with stationary additive-noise
   (Ornstein–Uhlenbeck) transitions between positive pairs.
2. The observed embedding marginal approximates that Gaussian (this is the
   SIGReg objective doing its job).

Surrounding results:

- **Thm. 2 (converse).** Among the class of worlds with independent stationary
  additive-noise latents, *only* the Gaussian latent distribution yields linear
  identifiability. Any other distribution provably breaks the recovery.
- **Thm. 3 (graceful degradation).** Recovery error is bounded by the
  whitening error `ε = ‖Cov(h(z)) − I‖_F` and the alignment gap
  `δ = L(h) − 2(1−ρ)·tr(Cov(h(z)))`. Identifiability degrades smoothly, not
  off a cliff, when the preconditions are only approximately met.
- **Thm. 4 (planning equivalence).** Any rotation-invariant planning cost
  transfers without modification from the true latent to the learned one when
  identifiability holds.

The empirical result that matters most for us is **Table 2 / §6.4**: the *same
physical system* (DMC Reacher) is linearly identified when sampled with OU
(Gaussian random-walk) positive pairs — `R² ≈ 0.95` — and **collapses to
`R² ≤ 0.5` when sampled from goal-directed RL trajectories**, because the policy
concentrates the marginal onto a low-entropy region of latent space. The paper's
explicit prescription (§8 Discussion):

> *"for self-supervised pretraining, exploration approximating an isotropic
> random walk keeps the data in the regime our theory covers."*

## 2. What translates onto v3, and what doesn't

### Translates cleanly

- **LeWM is the relevant target.** LeWM is the action-conditioned scaling of
  the LeJEPA recipe (the paper cites LeWorldModel as ref [9]). The encoder we
  are about to train is directly governed by Thm. 1's preconditions.
- **The SIGReg-vs-VICReg split in
  [v3_hjepa_plan.md §3.4](v3_hjepa_plan.md) is correct.** SIGReg for the
  predictive LeWM latents; VICReg + supervised contrastive for the
  retrieval-side BeliefEncoder. The paper validates this — SIGReg is the most
  robust of the three Gaussianity enforcers (Tab. 1, Fig. 4b), but it is only
  appropriate where Gaussian-shaped *targets* help the predictor.
- **The Reacher RL-trajectory failure mode is the exact mechanism that
  threatens the LeWM stage of v3.** A goal-directed teacher biases the marginal
  toward "facing down a corridor toward the goal," produces anisotropic
  temporal correlation across heading vs. translation, and induces joint-limit-
  like wrapping at walls — all three of the conditions Table 2 calls out as
  drivers of the collapse from 0.95 to ≤ 0.5.

### Does not translate identically

- **Gaussian-latent assumption will not hold in a maze.** Real latents over a
  maze scene (pose × surroundings × gait × lighting × texture) are categorically
  non-Gaussian — wall-vs-corridor is bimodal, textures categorical, gait phase
  cyclic. Thm. 2 says we cannot reach the strict optimum here regardless of
  sampling. The relevant question is whether Thm. 3's graceful degradation buys
  a usable representation, not whether we satisfy Thm. 1.
- **Planning equivalence (Thm. 4) does not ride directly on the v3 planner.**
  v3 plans for navigation; cells, walls, and occupancy are not rotation-
  invariant in the latent geometry. The maze-solver does not factor through
  Thm. 4 even if the encoder were perfectly identifiable.
- **v3 trains across hundreds of scenes × spawn-randomized poses.** That
  broadens the marginal substantially compared to the paper's single-scene
  single-policy setup. It is a meaningful mitigation the paper does not capture.
- **v3's action space is a discrete primitive bank** (forward / arc / yaw /
  back / hold). `ou_noise` already snaps to the nearest primitive. The OU
  collector therefore explores a discretized action space, not a continuous
  one — closer to a categorical random walk than a literal OU process.

## 3. Risk to the current §13 mix

From [fresh_retrain_data_spec.md §13](fresh_retrain_data_spec.md) and the
implementation in [collection_policy.md](collection_policy.md):

| Collector | Share | Effect on LeWM marginal |
| --- | ---: | --- |
| `route_teacher` | 30 % | Concentrates: corridor-center, forward-biased |
| `frontier` | 20 % | Concentrates (less): exploration toward novel cells |
| `loop_revisit` | 10 % | Same as `route_teacher` |
| `primitive_curriculum` | 20 % | Near-uniform over action primitives |
| `ou_noise` | 10 % | Smooth random walk in command space — closest to the paper's prescription |
| `recovery` | 10 % | Strongly non-isotropic (wall-contact-conditioned) |

By the paper's standard, only ~30 % of the corpus
(`primitive_curriculum + ou_noise`) is exploratory in the sense Thm. 1
prefers; ~60 % is teacher-conditioned and shaped like the Reacher
trajectory condition that broke identifiability.

This is not evidence the §13 mix is wrong. It is sized to give the
**downstream heads** (Reachability, BeliefEncoder, GoalAdapter) the route /
landmark / loop-closure pairs they need to train — none of which `ou_noise`
can produce. The conclusion is narrower:

> **The §13 mix is optimized for privileged-label-consuming downstream heads,
> not for the JEPA encoder underneath them. The paper is direct theoretical
> and empirical evidence that this design choice hurts the encoder more than
> is currently accounted for.**

## 4. Mitigations already in place

These already reduce the risk and should be preserved:

- **Spawn-pose randomization on every reset.**
  `RolloutConfig.randomize_spawn_pose=True` (default) converts "policy
  occupancy on a fixed goal trajectory" into "policy occupancy averaged over
  random starts × random goals," which broadens the per-cell heading marginal
  even under heavy teacher share. This is the single biggest existing
  mitigation against the Table 2 failure mode and **must stay on for any LeWM
  training shard.**
- **Per-episode collector resampling.** `EpisodeScheduler.on_episode_reset`
  redraws the collector per env per reset, so the §13 mix holds across the
  whole run rather than only at episode 0.
- **`command_source` stamped on every `CommandBlock`** (see
  [collection_policy.md](collection_policy.md) "Privileged-label outputs" and
  `lewm_genesis/lewm_genesis/collectors/base.py:53`). This makes any
  per-collector reweighting a dataloader-side filter, not a re-collection job.
- **`collector_mix_realized` in per-scene `summary.json`**
  (`lewm_genesis/lewm_genesis/rollout.py:890`) lets us audit realized shares
  per scene without rerunning anything.
- **SIGReg-vs-VICReg split** ([v3_hjepa_plan.md §3.4](v3_hjepa_plan.md)) is
  the right choice and matches the paper's analysis of where each estimator
  works.

## 5. Next steps and changes

Ordered by leverage. None of these require re-collecting data.

### Priority 1 — Cheap, paper-aligned, no risk

- [ ] **Split the data budget by training stage.** Keep the full §13 mix as the
      *full-corpus* set for downstream-head training. Train the LeWM encoder
      on a *subset* weighted toward `ou_noise` + `primitive_curriculum`. A
      reasonable starting target is ~50 % isotropic-equivalent
      (`ou_noise + primitive_curriculum`) / ~30 % curriculum
      (`primitive_curriculum` overlap accepted) / ~20 % teacher
      (`route_teacher + frontier + loop_revisit`), filtered at the dataloader
      by `command_source`. Treat this as a hyperparameter to sweep, not a
      number to commit to.
  - The contract is documented in §7 below: every dataloader row already
    carries `command_source`, so this is a dataloader-side filter, no
    re-collection needed. The LeWM trainer lives outside this repo;
    implement the filter where the trainer reads `messages.jsonl`.
- [x] **Per-scene marginal-isotropy audit, per collector.** Each
      per-scene `summary.json` now carries `marginal_isotropy[source]` with
      `blocks`, `distinct_cells`, `distinct_cell_yaw_bins`,
      `mean_per_cell_yaw_entropy_nats`,
      `max_per_cell_yaw_entropy_nats`, `primitive_entropy_nats`, and the
      per-source `primitive_counts` histogram. The corpus-level
      aggregation is in
      `scripts/audit_jepa_corpus.py::marginal_isotropy_by_source`.
      Implementation:
      [`rollout.py::_summarize_marginal_isotropy`](../lewm_genesis/lewm_genesis/rollout.py),
      tests in
      [`tests/test_marginal_isotropy.py`](../lewm_genesis/lewm_genesis/tests/test_marginal_isotropy.py).
- [x] **Cross-reference this analysis from
      [collection_policy.md](collection_policy.md).**

### Priority 2 — Worth doing before the next major LeWM training run

- [ ] **Pilot-shard ablation.** On the existing pilot corpus, train two LeWM
      checkpoints: one on the full §13 mix, one on the
      `ou_noise + primitive_curriculum`-only subset. Compare on:
      - Whitening error `ε = ‖Cov(h(z)) − I‖_F`.
      - Alignment gap `δ = L(h) − 2(1−ρ)·tr(Cov(h(z)))`.
      - The downstream proxy that matters most for the maze task: latent-to-
        graph-distance Spearman ρ on a held-out scene set (this is already
        the diagnostic [v3_hjepa_plan.md §4.5](v3_hjepa_plan.md) uses).
      If the isotropic-subset encoder is worse on the downstream proxy, the
      paper's prescription is the wrong tradeoff for the maze task and the
      full mix wins. If it is comparable or better, lock in the subset filter.
- [ ] **Add a `lewm_training_subset` field to the data spec.** Document the
      filter as a contract between data collection and LeWM training, so the
      ablation result is reproducible.

### Priority 3 — Lower leverage, do only if Phase A4 ambiguous

- [ ] **Per-collector eigenvalue / Hermite-mode probes** on a small held-out
      scene set, mirroring Fig. 4a of the paper. Useful only if the Phase A4
      reachability probe in [v3_hjepa_plan.md §4.5](v3_hjepa_plan.md) lands in
      the ambiguous regime — gives a paper-faithful diagnostic for *why* the
      latents are or are not preserving geometry.

### Explicitly not doing

- **Drop teacher collectors.** Downstream heads need them; Thm. 3 says
  moderate goal-directed share is fine if the encoder shard itself is
  exploratory-dominated. The decision is which collectors land in which
  training shard, not whether to generate them.
- **Try to satisfy the strict Gaussian-latent assumption.** It will not hold
  in a maze. The paper itself notes finite-sample, off-optimum behavior is
  governed by Thm. 3, not Thm. 1. Treat the paper as direction-of-travel
  guidance, not a spec.
- **Change BeliefEncoder training.** VICReg + supervised contrastive on
  privileged cell labels sidesteps the Gaussian-marginal issue entirely
  ([v3_hjepa_plan.md §3.4](v3_hjepa_plan.md) already makes this argument).

## 6. LeWM training subset filter contract

The LeWM trainer lives outside this repo. This is the dataloader-side
contract it should honour to consume the encoder-favourable subset.

**Inputs already in the corpus.** Every emitted `CommandBlock` carries
`command_source ∈ {route_teacher, frontier, primitive_curriculum,
ou_noise, recovery, loop_revisit}`
(see [collection_policy.md](collection_policy.md) "Privileged-label
outputs"). The same field lands in compact `messages.jsonl` rows and is
preserved through the render-replay path
(`lewm_genesis/lewm_genesis/render_replay.py:197`). No re-collection is
required.

**Recommended starting filter** for the LeWM training stage:

```python
LEWM_ISOTROPIC_SOURCES = {"ou_noise", "primitive_curriculum"}
LEWM_PERMITTED_SOURCES = LEWM_ISOTROPIC_SOURCES | {
    "route_teacher", "frontier", "loop_revisit",
}

# Per row r in messages.jsonl:
def keep_for_lewm(r) -> bool:
    return r.get("command_source") in LEWM_PERMITTED_SOURCES

# Then per-source weights when building the training index:
LEWM_SOURCE_WEIGHTS = {
    "ou_noise": 0.30,
    "primitive_curriculum": 0.20,
    "route_teacher": 0.20,
    "frontier": 0.20,
    "loop_revisit": 0.10,
}
```

Notes:

- `recovery` is excluded by default. Wall-contact-conditioned trajectories
  are the *most* concentrated marginal in the corpus and dominate the OU
  signal if let in. Re-include only if a planned downstream head wants
  contact frames.
- Downstream-head trainers (Reachability, BeliefEncoder, GoalAdapter) keep
  the full §13 mix — no filter — because they consume privileged labels
  that only the teachers produce.
- Treat the weights as a hyperparameter to sweep (see Priority-2
  ablation), not a number to commit to.

## 7. Commands for the next build

Run after `scripts/build_go2_sim.sh`. The two stages — per-scene audit
emission and corpus-level aggregation — are independent; you can run the
aggregator against any existing run that has the updated `summary.json`
schema.

### 7.1 Verify the unit tests pass

```bash
# From the repo root, with lewm_worlds + lewm_genesis on PYTHONPATH (the
# colcon overlay does this automatically; the explicit PYTHONPATH below
# only matters on the dev box without an overlay).
PYTHONPATH=lewm_genesis:lewm_worlds \
  python -m pytest \
  lewm_genesis/lewm_genesis/tests/test_marginal_isotropy.py -v
```

Expected: 3 passed.

### 7.2 Confirm the per-scene audit lands in the next rollout

```bash
# Smoke a single-scene pilot under the existing P1 ramp recipe.
scripts/genesis_bulk_rollout.sh \
  --scene-corpus .generated/scene_corpus/acceptance \
  --split train \
  --scene-limit 1 \
  --n-envs 512 \
  --n-blocks 20 \
  --backend cpu \
  --no-rgb \
  --out .generated/genesis_bulk_rollouts/lejepa_audit_pilot

# Inspect the new field. Expect one entry per command source the
# scheduler drew, with `blocks > 0`.
python -c "import json, pathlib; \
  p=next(pathlib.Path('.generated/genesis_bulk_rollouts/lejepa_audit_pilot').glob('*/summary.json')); \
  print(json.dumps(json.loads(p.read_text())['stats']['marginal_isotropy'], indent=2))"
```

### 7.3 Aggregate across a full corpus

After a mass-datagen run that produced `rollout/<scene>/summary.json` for
every scene:

```bash
scripts/audit_jepa_corpus.py <mass-datagen-root> \
  --out .generated/audits/jepa_corpus_audit.json

# Pretty-print the per-collector marginal-isotropy roll-up.
python -c "import json; \
  a=json.load(open('.generated/audits/jepa_corpus_audit.json')); \
  print(json.dumps(a['marginal_isotropy_by_source'], indent=2))"
```

What to look for, by collector:

- `ou_noise` and `primitive_curriculum`: `mean_per_cell_yaw_entropy_nats`
  close to `max_per_cell_yaw_entropy_nats` (≈ 2.485 for the default 12
  yaw bins) and high `primitive_entropy_nats` (≥ 1.5 nats over 5+
  trainable primitives ≈ uniform).
- `route_teacher`, `frontier`, `loop_revisit`: expect lower
  `mean_per_cell_yaw_entropy_nats` and lower `primitive_entropy_nats`
  (these collectors concentrate toward `forward_medium`); this is the
  exact signal the paper warns about.
- `recovery`: expect the lowest entropies and a strong bias to
  `backward` + `yaw_*` primitives — confirms wall-contact concentration.
- `distinct_cell_yaw_bins_sum`: a coverage number; bigger is better, and
  the gap between `ou_noise` and the teachers is the directly auditable
  version of "the encoder shard is more isotropic than the head shard."

### 7.4 Pilot ablation (Priority 2, when you're ready)

Once a corpus is in hand, run two LeWM training shards: one on the full
§13 mix, one filtered per §6. Compare on the diagnostics listed in §5
Priority 2 of this doc. There is nothing to wire in this repo for that —
the filter is in the trainer's dataloader.

## 8. Bottom line

The strongest takeaway from the paper for v3 is narrow and actionable: **the
goal-directed share of the §13 mix is a quiet tax on LeWM identifiability**,
and the cheapest response is a dataloader-side filter that trains LeWM on an
exploratory-weighted subset of the same corpus, while leaving the downstream
heads on the full §13 mix. Everything else the paper says — Gaussian latents,
planning equivalence — degrades from theorem to heuristic in the maze setting
and should not drive design decisions on its own.
