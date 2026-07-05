# JEPA Phase 2D Stage 2 Corrected Model Implementation

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

Registration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

Prior foundation:
`docs/lewm_jepa_phase2d_stage0_stage1_implementation_2026-06-14.md`

Trainer/statistics continuation:
`docs/lewm_jepa_phase2d_stage3_trainer_statistics_2026-06-14.md`

## Scope And Evidence Status

This increment implements and verifies the preregistered Stage 2 corrected
model and diagnostic contract. It does not train C0, C1, or C2 on research
data, inspect confirmatory validation results, or provide evidence that the
model learns action-conditioned navigation dynamics.

The synthetic smoke result is interface and gradient-routing evidence only.
Its losses and representation statistics have no model-performance
interpretation.

## Implementation

### Corrected Spatial JEPA

`lewm/models/phase2d_spatial_lewm.py` adds
`Phase2DSpatialLeWorldModel` with the registered fixed-capacity defaults:

- latent dimension `48`;
- encoder depth `2`, heads `3`, and MLP ratio `2`;
- predictor depth `2`, heads `4`, head dimension `12`, and MLP dimension `96`;
- appearance SIGReg weight `0.09`;
- normalized spatial variance-floor weight `1.0`;
- per-token spatial target standard deviation `1 / sqrt(latent_dim)`.

The corrected spatial path has three distinct projection roles:

1. `online_target_projector`, which supplies the online normalized target and
   the source for EMA target updates;
2. `target_projector`, which is a frozen EMA copy for C1 and C2;
3. `prediction_projector`, which independently projects predictor outputs.

All three spatial projectors are a single linear map without BatchNorm or any
batch-dependent state. The separate appearance projector retains the existing
BatchNorm path for SIGReg.

The model supports the registered cells without changing architecture
capacity:

| Cell | `target_ema_momentum` | action loss weights |
| --- | ---: | ---: |
| C0 | `None` | action `0`, zero `0` |
| C1 | `0.99` | action `0`, zero `0` |
| C2 | `0.99` | action `1`, zero `1` |

C2 configuration fails closed: a positive action-identifiability weight
requires explicit wrong actions and their mask, and a positive zero-action
weight requires an explicit non-hold mask. The model cannot silently optimize
a zero-valued C2 contrastive term because controls were omitted.

### Registered Loss Contract

`action_identifiability_losses` computes:

- valid-transition prediction MSE;
- exhaustive wrong-action hinge loss over every `wrong_pair_mask=True` pair;
- zero-action hinge loss only where both transition-valid and non-hold;
- per-transition target-change MSE;
- dynamic margin `0.10 * max(target_change_mse, 1e-4)`.

Distances are MSE over independently L2-normalized spatial tokens. Empty
eligible sets produce a differentiable zero instead of a division error, while
C2 configuration still requires that the relevant masks are supplied.

The forward result emits separate scalar metrics and detailed tensors. In
particular, `mean_target_change_mse` is the aggregate scalar and
`target_change_mse` is the per-transition tensor. This naming prevents detailed
diagnostic output from silently overwriting the scalar logging schema.

### EMA And Gradient Contract

For C1 and C2:

- target encoder and target projector parameters have
  `requires_grad=False`;
- target modules remain in evaluation mode when the online model trains;
- target parameters update only through explicit EMA;
- prediction and target projectors are never shared;
- no spatial BatchNorm buffers exist to copy or drift.

For C0, the normalized online target remains differentiable as registered for
the corrected online historical control.

### Stability Diagnostics

`lewm/benchmarks/rollout_diagnostics.py` now adds:

- covariance effective rank and effective-rank fraction;
- mean feature-coordinate standard deviation;
- pre-normalization token norm mean, median, p05, and p95;
- normalized token norm mean;
- pairwise state discrimination MSE;
- target-change MSE relative to feature variance;
- collapse, low-rank, and near-static warnings.

These metrics complement, rather than replace, the persistence, zero-action,
and shuffled-action rollout controls. A non-collapsed representation is not
treated as an action-identifiable representation.

## Stage 2 Gate Verification

### Focused Unit Tests

Command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2d_spatial_lewm.py \
  lewm/tests/test_rollout_diagnostics.py -q
```

Result: `13 passed`.

The focused tests verify:

- per-token unit normalization;
- exhaustive wrong-pair masking and dynamic margins;
- zero-action eligibility masking;
- independent non-BatchNorm spatial projection paths;
- C2 failure when required controls are absent;
- emitted scalar metrics, detailed masks, and control tensors;
- online encoder, predictor, and projector gradient flow;
- frozen EMA targets and exact EMA parameter updates;
- normalized free-running rollout output;
- effective-rank and stability warnings on full-rank and collapsed targets.

### Deterministic Synthetic Smoke

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/smoke_phase2d_spatial_model.py \
  --output .generated/jepa_counterfactual/phase2d_stage2_model_smoke.json
```

The smoke run used seed `20260614` and verified:

| Contract | Observed |
| --- | ---: |
| valid transitions | `3` |
| eligible wrong-action transitions | `3` |
| eligible wrong-action pairs | `4` |
| eligible zero-action transitions | `2` |
| normalized target token norm mean | `1.0` |
| online encoder receives gradient | yes |
| predictor receives gradient | yes |
| prediction projector receives gradient | yes |
| EMA target receives gradient | no |

The smoke emitted all registered Stage 2 losses, mask counts, prediction
shapes, and stability diagnostics. Its content-addressed report hash is:

`56f404440b970882278665d93d47c0b47a5c1d9a0c88b531d47ddaef8b54639b`

The verified manifest is:

`.generated/jepa_counterfactual/phase2d_stage2_model_smoke_manifest.json`

Manifest hash:

`3abf86e19e90fa879a583aee11fa602cc0f84c91ca8f739348366f385813d652`

The manifest records the exact command, seed, environment, Git state, input
hashes, and artifact hash. Manifest verification passed.

### Repository Regression

Command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `111 passed`, `3 subtests passed`, with six existing
`belief_encoder.py` nested-tensor warnings.

## Gate Decision

The preregistered Stage 2 implementation gate passes:

- corrected model requirements are implemented;
- focused unit tests pass;
- synthetic smoke confirms every required loss, metric, mask, and gradient
  route is emitted.

This does not authorize confirmatory training. Stage 1's confirmatory-data gate
still fails, and the corrected trainer, evaluation estimands, power check, and
frozen C0/C1/C2 manifests are not yet complete.

## Limitations And Open Risks

- No state-only or action-only diagnostic control is implemented yet.
- No research-data trainer consumes the source-grouped batches and exhaustive
  hard-negative index yet.
- No measured result establishes that normalized image-aligned patches are
  sufficiently Markov-visible or motion-equivariant.
- The normalized variance floor guarantees neither high effective rank nor
  action semantics; both remain measured gates.
- The synthetic smoke uses random images and actions and cannot detect
  optimization failure, shortcut learning, or generalization failure.
- C0's differentiable online target is intentionally retained as the registered
  historical control. C1 and C2 are the relevant stop-gradient EMA cells.

## Next Implementation Sequence

1. Add the trainer's confirmatory per-source-state evaluation table.
2. Integrate hierarchical bootstrap confidence intervals and the registered
   checkpoint rule into the final validation report.
3. Generate and audit full-81-candidate confirmatory data and unopened test
   splits.
4. Freeze immutable C0/C1/C2 manifests before any confirmatory result is
   inspected.
