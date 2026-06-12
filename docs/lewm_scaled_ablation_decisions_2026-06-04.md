# LeWM Scaled SIGReg / Source-Sampling Ablation Decisions (2026-06-04)

## Purpose

Prepare the next experiment without committing to another full 10-epoch main
run. The diagnostic result is clear enough to test a source-level fix:

- A2: projected latent distance has almost no graph-distance structure
  (`rho_proj` median about 0.03 on the full e9 diagnostic).
- A3: place identity is present (`retrieval@1` about 0.42 vs about 0.02 chance),
  but cross-scene reachability from latent pairs stays near baseline.
- Training metrics: held-out `eval_std` rises from 0.65 to about 0.75 while
  train std is about 1.0, and eval prediction loss plateaus early.
- A2 trajectory artifacts already exist for e0/e2/e4/e6/e8/e9_b060000; the
  projected rho median stays low rather than improving with training:

| checkpoint | scenes | rho_proj median | rho_proj yaw-matched median | rho_raw median |
|---|---:|---:|---:|---:|
| e0 | 16 | 0.055 | 0.184 | 0.074 |
| e2 | 16 | 0.059 | 0.261 | 0.102 |
| e4 | 16 | 0.062 | 0.261 | 0.108 |
| e6 | 16 | 0.055 | 0.250 | 0.117 |
| e8 | 16 | 0.054 | 0.261 | 0.129 |
| e9_b060000 | 16 | 0.048 | 0.250 | 0.126 |

This makes the ablation a mechanism test, not a fishing expedition: lower the
isotropy pressure and/or change the pretraining marginal, then re-run the same
A2/A3 readouts.

## Decisions

**1. Run scaled ablations first, not a full retrain.**

Full seq4 training has been running at roughly 12 hours per epoch. A 10-epoch
full retrain is therefore another main run, not an ablation. The ablation should
use a fixed subset (`--max-sessions`, default runner value 300) and 2-3 epochs
to answer whether the metric/readout direction moves.

**2. Use a lambda dose response, not one low-lambda point.**

Default sweep cells include `lambda = 0.09, 0.03, 0.01` under the uniform source
mix. `0.09` is the scaled control. `0.03` and `0.01` test whether relaxing
SIGReg monotonically improves A2 rho / A3 retrieval. We avoid `lambda = 0`
because removing anti-collapse entirely adds a collapse risk that would obscure
the mechanism.

**3. Wire source-aware sampling now.**

`command_source` is present on requested `CommandBlock` messages and joins to
executed blocks by `(env_index, sequence_id)`. The trainer now carries this into
per-window source tags. A smoke test on two sessions produced `unknown_fraction
= 0.0`, confirming the join works on current data.

**4. Use hard filter + size cap as the primary sampling arm.**

The paper-grade exploratory arm is `ou_noise,primitive_curriculum`. Filtering
alone would reduce the dataset size and confound source mix with sample budget,
so the runner supports `--source-cap auto`: inspect the scaled dataset, count
allowed exploratory windows, then cap the uniform control to that same number.
The optional `--source-weight` path remains available for exploratory weighting,
but it is not the primary paper comparison because it is less directly
interpretable and cannot resume mid-epoch deterministically.

**5. Staged factorial, not all claims at once.**

Default runner cells:

- `uniform:0.09` scaled control
- `uniform:0.03`, `uniform:0.01` lambda-only dose response
- `exploratory:0.09` sampling-only arm
- `exploratory:0.03`, `exploratory:0.01` combined arms

For a paper, promote the best binary low-lambda setting into a seeded 2x2
factorial: `{lambda high, lambda low} x {uniform, exploratory}` with at least
three seeds. The dose-response sweep explains the mechanism; the 2x2 factorial
separates lambda effect, source-mix effect, and interaction.

**6. Screen every cell with cheap A2 + A3 retrieval before expensive probes.**

The new probe wrapper runs A2 and A3 retrieval-only by default. That gives the
two most relevant fast signals for this ablation:

- A2 projected rho and bucket flatness: did metric geometry recover?
- A3 retrieval@1/lift: did place identity improve or degrade?

Run full A3 reachability/history only after a cell is worth promoting. Full A3
is still the registered gate, but it is too slow to run after every short
training cell if the cheap metrics show no movement.

**7. Benchmark fixes are configuration-first.**

The closed-loop benchmark code already has yaw jitter (`--beacon-start-yaw-
jitter-rad`) and `--max-blocks`; the old zero-jitter beacon result is
oracle-trivial, and the old stress run is under-budgeted because the bearing
oracle also scores 0/6. The discriminating benchmark should be re-run after the
ablation shortlist, not inside every training cell:

- local beacon: add yaw jitter, e.g. `--beacon-start-yaw-jitter-rad 0.7`
- stress: raise budget to about `--max-blocks 40` or shorten starts before using
  it as evidence

**8. Keep longer-horizon prediction as a follow-up objective ablation.**

The final-checkpoint rollout gate showed a specific temporal failure: rollout
beats persistence at horizons 1-2 but loses to persistence from horizon 3
onward. Direct multi-horizon supervision is therefore a plausible way to make
the local planner useful over multiple seconds.

Do not replace the representation ablation with this objective yet. A2/A3 show
that the latent space is not metric, so better long-horizon prediction may still
produce a poor planning cost if "near" and "far" are not encoded correctly.

Preferred follow-up design after the scaled lambda/source shortlist:

- keep adjacent prediction (`h=1`) to preserve immediate action sensitivity;
- add a small temporal-pyramid rollout loss, first `h={1,2,3,5}`;
- only then test longer targets such as `h={1,2,3,5,8,10}`;
- compare against the same rollout gate, receding-MPC proxy, A2/A3, and
  yaw-jitter beacon benchmark.

Lay explanation: longer prediction could help the robot plan several seconds
ahead, but only after the internal map is less distorted. Otherwise the model
may predict farther into the future on a map where distance itself is still
wrong.

## Implementation

Trainer:

- `scripts/train_lewm.py` now parses requested `CommandBlock.command_source`,
  joins it onto executed actions by sequence id, and tags each valid window by
  anchor-block source.
- New CLI: `--source-allow`, `--source-cap`, `--source-weight`.
- Checkpoints and metrics JSONL now record source config, sampler kind, training
  sample count, and train source mix.
- Partial resume validates source config and true training sample count.
- Weighted sampling rejects intra-epoch checkpoints because it cannot resume by
  deterministic batch offset.
- The material-render guard now permits `visuals='textured_v03'` instead of
  rejecting it only because it shares the v03 schema string.

Probe tooling:

- `scripts/probe_lewm_reachability_a3.py --retrieval-only` skips train banks,
  reachability-head training, localization, and history.
- `scripts/eval_lewm_ablation_probes.sh` runs the A2 + A3 screening suite.
- `scripts/inspect_lewm_source_mix.py` reports scaled source counts and the
  auto-cap value.
- `scripts/run_lewm_scaled_factorial_ablation.sh` launches the staged cells and
  optionally runs screening probes after each completed checkpoint.

## Launch Command

Default staged run:

```bash
bash scripts/run_lewm_scaled_factorial_ablation.sh \
  --data-root .generated/datagen_full \
  --render-root .generated/datagen_full/render_textured_v03 \
  --checkpoint-root models/checkpoints_textured_v03_scaled_ablation_20260604 \
  --max-sessions 300 \
  --epochs 3 \
  --source-cap auto
```

For a quick lambda-only start:

```bash
bash scripts/run_lewm_scaled_factorial_ablation.sh \
  --data-root .generated/datagen_full \
  --render-root .generated/datagen_full/render_textured_v03 \
  --checkpoint-root models/checkpoints_textured_v03_scaled_ablation_20260604 \
  --max-sessions 300 \
  --epochs 3 \
  --cells uniform:0.09,uniform:0.03,uniform:0.01 \
  --source-cap auto
```

Use `LEWM_PYTHON=/path/to/python` to force a specific training/probe venv. The
runner otherwise prefers `GENESIS_ROCM_PYTHON`, then the local ROCm venv if it
exists, then the local `genesis_render_vulkan` venv.
