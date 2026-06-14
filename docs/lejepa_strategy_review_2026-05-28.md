# LeJEPA Paper Impact Review - 2026-05-28

## Scope

This review covers the changes pulled into `datagen-nan-fix-rebalance` by merge
`6419006`, specifically:

- `docs/lejepa_identifiability_analysis.md`
- `docs/collection_policy.md`
- `lewm_genesis/lewm_genesis/rollout.py`
- `lewm_genesis/lewm_genesis/tests/test_marginal_isotropy.py`
- `scripts/audit_jepa_corpus.py`

It also checks whether the new paper changes the v3 LeWM/H-JEPA strategy. The
paper source checked was Klindt, LeCun, and Balestriero, ["When Does LeJEPA
Learn a World Model?"](https://arxiv.org/abs/2605.26379), submitted May 25,
2026, plus the authors' [project page](https://klindtlab.github.io/lejepa-identifiability/).

The current local `scripts/train_lewm.py` was also inspected because it is the
place where the proposed collector filtering would have to become operational.
It was untracked at review time, so it is treated here as strategy context, not
as a pulled-code finding.

## Executive Judgment

The paper should change the data and training strategy, but it does not require
abandoning LeWM or JEPA-style representation learning.

The main correction is that the v3 corpus mix cannot be treated as a single
optimal distribution for every training stage. Goal-directed and recovery-heavy
data is valuable for downstream labels, but it can be actively bad for LeWM
encoder pretraining if it dominates the marginal state distribution. SIGReg
helps shape the embedding distribution, but it cannot recover state-space
coverage that the data never contains.

The right response is to keep generating the full corpus, but split how it is
used:

- Train the LeWM encoder/predictor on an exploratory-balanced subset or weighted
  sampler, biased toward `ou_noise` and `primitive_curriculum`.
- Train reachability, belief, goal-adapter, and other supervised heads on the
  full label-rich mix.
- Add explicit ablations and diagnostics before making the subset policy a hard
  contract.

## Code Findings

### High: The New Corpus Audit Does Not Support the Current Full-Corpus Layout

`scripts/audit_jepa_corpus.py` expects each root to contain:

```text
<root>/rollout/<scene>/summary.json
<root>/labels/<scene>/labels.jsonl
```

The resumable full-corpus runner writes chunk roots instead:

```text
<root>/rollout/<split>/<family>/chunk_XXXX/rollout/<scene>/summary.json
<root>/rollout/<split>/<family>/chunk_XXXX/labels/<scene>/labels.jsonl
```

Evidence:

- `scripts/audit_jepa_corpus.py:91-103` checks only immediate
  `root/rollout/<scene>` children and requires `root/labels`.
- `scripts/datagen_rollout_resumable.sh:85-99` creates per-chunk roots under
  `rollout/<split>/<family>/chunk_XXXX` and runs `run_mass_datagen.sh --out`
  inside those chunk roots.
- The documented command in `docs/lejepa_identifiability_analysis.md:301-309`
  tells users to run `scripts/audit_jepa_corpus.py <mass-datagen-root>`.

Observed result on the current corpus:

```text
python3 scripts/audit_jepa_corpus.py .generated/datagen_full --out /tmp/jepa_audit_test.json
scene_count: 0
label_row_count: 0
effective_sequence_count: 0
schema_violations: [".generated/datagen_full: missing rollout/ or labels/"]
```

Impact: the new marginal-isotropy aggregate cannot validate the actual full
corpus as documented. It will either fail fast or produce an empty audit, which
is worse than having no metric because it looks like a completed run.

Recommended fix: make the audit script accept both layouts. It should discover
all chunk roots matching `rollout/*/*/chunk_*/` and aggregate their
`rollout/<scene>` plus `labels/<scene>` contents. The docs should distinguish
single `run_mass_datagen.sh` roots from resumable full-corpus roots.

### Medium: The Per-Scene Verification Command Reads the Wrong Summary Path

The doc says to inspect:

```python
json.loads(p.read_text())["stats"]["marginal_isotropy"]
```

But the rollout runner returns marginal-isotropy inside the rollout stats, and
the writer stores those under:

```text
summary["extra"]["rollout_stats"]["marginal_isotropy"]
```

Evidence:

- `lewm_genesis/lewm_genesis/rollout.py:950-964` returns
  `marginal_isotropy` in the rollout stats dict.
- `scripts/genesis_bulk_rollout.py:329-336` writes that dict as
  `extra.rollout_stats`.
- `lewm_genesis/lewm_genesis/mcap_writer.py:211-225` uses `stats` for writer
  message counts and puts caller metadata under `extra`.
- `docs/lejepa_identifiability_analysis.md:294-298` reads `stats`.

Impact: the smoke command in the docs will fail or inspect the writer message
stats instead of the new isotropy field.

Recommended fix: change the command to:

```python
print(json.dumps(json.loads(p.read_text())["extra"]["rollout_stats"]["marginal_isotropy"], indent=2))
```

### Medium: The Isotropy Metric Is a Useful Proxy, Not an Identifiability Test

The new rollout metric is directionally useful. It records collector-specific
coverage over visited cells, yaw bins, and primitive names. That is worth
keeping.

However, it should not be interpreted as evidence that LeJEPA's assumptions are
satisfied.

Limitations:

- It samples yaw at block-request cadence, not every rendered frame consumed by
  the encoder (`rollout.py:1133-1155`).
- It measures cell/yaw and primitive entropy, not latent covariance, Gaussianity,
  temporal correlation spectra, linear recoverability, or planning quality.
- It averages yaw entropy per visited cell (`rollout.py:70-80`) and then
  averages per-scene means in the aggregate audit
  (`scripts/audit_jepa_corpus.py:195-205`). That is a scene-level proxy, not a
  corpus-wide marginal.

Impact: the metric can catch obvious collector concentration, but it cannot
prove that the LeWM encoder will learn a linearly identifiable world state.

Recommended fix: keep the metric as a collection-health signal, but pair it with
training-time probes:

- embedding covariance and whitening error
- source-stratified prediction loss
- action sensitivity
- latent-to-graph-distance Spearman correlation on held-out scenes
- full-mix vs exploratory-subset ablation

### Low: The Paper Quote Should Be Verified or Rewritten as a Paraphrase

`docs/lejepa_identifiability_analysis.md:40-43` includes an exact quote about
self-supervised pretraining and isotropic random walks. The substance matches
the paper's direction, but the exact sentence was not verified from the arXiv
abstract, project page, or search results during this review.

Impact: low technical risk, but high credibility cost in a research note.

Recommended fix: either cite the exact PDF location or rewrite the sentence as a
paraphrase:

```text
The practical implication is that self-supervised pretraining should be biased
toward broad, approximately isotropic exploration rather than narrow
goal-directed trajectory marginals.
```

## Strategy Re-Evaluation

### What the Paper Actually Says

The paper proves that LeJEPA can linearly recover latent variables, up to an
orthogonal rotation, under a specific set of assumptions:

- latent variables are Gaussian
- transitions are stationary with additive noise
- the learner enforces a Gaussian embedding distribution
- planning guarantees apply to rotation-invariant costs

The authors also emphasize that Gaussianity is the unique latent distribution
for the strict identifiability guarantee, and that approximation quality
degrades as the assumptions fail.

This is highly relevant to v3, but it is not a proof that our maze setting will
work. Maze navigation has categorical topology, wall/contact discontinuities,
cyclic gait phase, visual aliasing, and partial observability. Those are not
Gaussian latent variables with clean stationary additive-noise dynamics.

### What Our Strategy Got Wrong

The main mistake was treating "more useful navigation data" and "better LeWM
pretraining data" as the same objective.

They are not the same. A route teacher produces valuable supervision and
successful trajectories, but it also concentrates the state marginal around
task-directed corridors, goal-facing headings, and recovery cases. That is
exactly the kind of narrow marginal that weakens the LeJEPA identifiability
story.

The second mistake was assuming that SIGReg alone solves this. SIGReg can
regularize the embedding distribution, but it cannot invent state-space support
missing from the collection policy.

The third mistake was leaving the new subset strategy at the documentation
level. The proposed filter in `docs/lejepa_identifiability_analysis.md:212-258`
is not enough unless the trainer or dataloader actually consumes
`command_source`. The current local `scripts/train_lewm.py` parses
`ExecutedCommandBlock` messages into actions and resets, builds a uniform
sequence index, and returns image/action sequences. It does not read or weight
`command_source`.

> **Implementation update, 2026-06-04:** P1 is now operational in
> `scripts/train_lewm.py`. The trainer joins requested `CommandBlock.command_source`
> to executed blocks by `(env_index, sequence_id)`, records per-window source
> tags, and exposes `--source-allow`, `--source-cap`, and `--source-weight`.
> The scaled ablation plan is documented in
> `docs/lewm_scaled_ablation_decisions_2026-06-04.md`.

### What the Strategy Still Gets Right

The overall architecture is still defensible:

- LeWM remains a reasonable base representation and dynamics learner.
- SIGReg remains the right regularizer for the LeWM side, given the current
  JEPA/LeWM framing.
- BeliefEncoder should remain a separate component; the paper does not solve
  partial observability or topological aliasing.
- Teacher, frontier, loop-revisit, and recovery collectors should not be
  removed. They are needed for labels, edge cases, and downstream heads.

The change is not "drop teacher data." The change is "do not let teacher data
define the LeWM pretraining marginal by default."

## Recommended Plan

### P0: Fix the Audit So It Measures the Real Corpus

- Update `scripts/audit_jepa_corpus.py` to discover both flat
  `run_mass_datagen.sh` roots and chunked resumable roots.
- Update the docs to use the correct `extra.rollout_stats.marginal_isotropy`
  summary path.
- Re-run the audit on `.generated/datagen_full` after a fresh rollout that
  includes the new field.

### P1: Make the LeWM Subset Operational

- Preserve `command_source` alongside each parsed action sequence in the LeWM
  dataloader.
- Add either:
  - a hard source filter, or
  - a weighted sampler over sequence source composition.
- Exclude `recovery` from the default LeWM pretraining subset unless a specific
  ablation shows that contact-heavy frames help more than they hurt.

### P2: Run the Critical Ablation

Train at least three small LeWM checkpoints:

1. full corpus mix
2. exploratory-only: `ou_noise + primitive_curriculum`
3. weighted mix: exploratory-biased, with some route/frontier/loop-revisit

Compare:

- prediction loss by command source
- whitening error
- action sensitivity
- latent-to-graph-distance Spearman correlation
- downstream reachability/belief proxy performance
- qualitative nearest-neighbor latent interpolation on held-out scenes

If the exploratory subset loses on the downstream proxy, keep the full mix for
LeWM. If it is comparable or better, lock in the source filter before the next
large training run.

### P3: Tighten the Claim Language

Use the paper as guidance and a source of diagnostics, not as a guarantee that
the maze system learns a world model. The strict theorem does not apply cleanly
to our setting.

Recommended wording:

```text
The LeJEPA identifiability result makes broad, exploratory coverage more
important for LeWM pretraining than our original collection policy assumed. It
does not prove that the v3 maze latent will be identifiable, but it gives us a
clear failure mode to test: goal-directed collection can concentrate the
training marginal enough to hurt representation geometry.
```

## Verification Performed

Unit test for the new helper:

```text
PYTHONPATH=lewm_genesis:lewm_worlds python3 -m pytest lewm_genesis/lewm_genesis/tests/test_marginal_isotropy.py -q
3 passed in 0.10s
```

Audit command against the current full corpus:

```text
python3 scripts/audit_jepa_corpus.py .generated/datagen_full --out /tmp/jepa_audit_test.json
scene_count: 0
label_row_count: 0
effective_sequence_count: 0
schema_violations: [".generated/datagen_full: missing rollout/ or labels/"]
```

## Bottom Line

The pulled analysis is directionally right: the new LeJEPA paper is a serious
warning against training the base LeWM encoder on a narrow goal-directed
trajectory marginal.

The implementation is not yet sufficient: the audit does not work on the real
full-corpus layout, the doc smoke command reads the wrong summary field, and the
LeWM source filter is not yet wired into training.

The next move should be pragmatic: fix the audit, make source-aware LeWM
sampling real, and run the full-mix vs exploratory-subset ablation before
committing the next expensive training run.
