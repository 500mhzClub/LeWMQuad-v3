# Go2 recurrent H4 joint-JEPA V1 result — 2026-07-27

## Outcome

- Terminal decision: `STOP_MAIN_POOL_RECURRENT_H4_JOINT_JEPA_V1_PROBE`.
- This was a clean scientific falsification, not an execution failure.
- The run completed exactly 1,000 optimizer updates and 16,000 training-sequence presentations in 509.927 active seconds.
- The selected checkpoint was update 750 / 12,000 presentations. It is diagnostic only and is not eligible for reuse or promotion.
- Nine of 18 registered gates failed. No held-out, test, sealed, label, navigation, or deployment input was opened.

## What was tested

- Frozen source commit: `e162d93` (`add capped recurrent H4 joint JEPA probe`).
- RGB-only joint training of:
  - the accepted N320 visual encoder;
  - a causal three-observation/two-past-action recurrent belief;
  - one shared predictor unrolled over four future actions;
  - a stop-gradient EMA target encoder.
- Training used the fixed main-pool schedule:
  - 16,000 train sequences, exactly 2,000 per scene family;
  - 2,048 fixed validation sequences, exactly 256 per family;
  - seven RGB observations and six reset-safe primitive transitions per sequence;
  - train and validation scene identities were disjoint.
- Objective: feature-summed joint-JEPA prediction loss, normalized spatial variance floor, and a cyclic wrong-action margin.
- Registered observations: updates 0, 250, 500, 750, and 1,000.

## Input identities

- Train index SHA-256: `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`.
- Validation index SHA-256: `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`.
- Index manifest SHA-256: `b7adcf59a7531d39f1f3f4151746299d361c832dba86ae076390a196761903b0`.
- Main-pool ordered source-content binding: `0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696`.
- N320 checkpoint file SHA-256: `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`.
- N320 content SHA-256: `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.
- Only the 78 `encoder.*` tensors were copied; the recurrent history and predictor were freshly initialized.

## Main metrics

| Update | H4 error / persistence | H4 wrong-action gap | H4 history gap | H4 persistence gap | Target / online rank |
|---:|---:|---:|---:|---:|---:|
| 0 | 21.4949 | 0.0018 | -0.0898 | -20.4949 | 0.1750 / 0.2075 |
| 250 | 4.6229 | 0.1581 | -0.3516 | -3.6229 | 0.1373 / 0.1295 |
| 500 | 3.5016 | 0.5235 | -0.2709 | -2.5016 | 0.1313 / 0.1389 |
| 750 | **2.2802** | **0.5255** | **-0.1335** | **-1.2802** | 0.1379 / 0.1495 |
| 1,000 | 2.7644 | 0.6084 | -0.2019 | -1.7644 | 0.1480 / 0.1623 |

- Prediction improved sharply: selected H4 normalized error fell 89.39% from initialization.
- Action conditioning succeeded for the first time in this line of work:
  - selected H4 wrong-action gap was 0.5255, above the 0.05 gate;
  - the scene-bootstrap lower bound was 0.3653, above zero;
  - all eight scene families had a positive H4 action gap;
  - H1–H3 action gaps were also non-negative.
- The representation remained non-collapsed at every observation.
- The model nevertheless remained worse than copying the current latent:
  - selected H4 error was 2.2802 times persistence error;
  - all eight families had a negative H4 persistence gap.
- Ordered history was harmful rather than useful:
  - selected H4 history gap was -0.1335;
  - no family had a positive H4 history gap;
  - reset/reordered-history controls predicted better than the real ordered history.
- Update 1,000 strengthened action identity but worsened generic H4 prediction, so update 750 was selected by the preregistered mean H1–H4 error rule.

## Interpretation

- V1 disproves the earlier concern that the main-pool encoder/predictor cannot learn action-sensitive latent dynamics at this budget. The action mechanism is now clearly real and general across families.
- V1 does not yet produce a useful recurrent world state. Its absolute predictor throws away the strong current-frame latent and then relearns a coarse future, so even its best state loses to the zero-dynamics persistence baseline.
- The training objective rewarded action identity but did not reward using ordered history. The recurrent history path therefore became a nuisance transformation instead of a belief-state improvement.
- This is a localized mechanism failure, not evidence to return to data refinement, Camera-V6-style encoder probing, or another seed of the same architecture.

## Registered next step

- Do not resume or reuse any V1 checkpoint.
- Test one fresh V2 on the identical fixed schedule and cap, initialized only from the accepted N320 encoder.
- Change exactly the failed mechanisms:
  - anchor every rollout to the current spatial latent so a zero residual equals persistence;
  - predict action-conditioned latent deltas rather than an unconstrained absolute future;
  - initialize the residual path near zero;
  - add a real-history-versus-reset/reordered-history contrast so ordered memory must earn predictive value.
- Preserve the V1 action counterfactual and all existing validation controls. Scale beyond 16,000 presentations only if V2 beats persistence, keeps the action gate, and makes the history gap positive across families.

## Terminal bindings

- Completion SHA-256: `3c624232d728987fd9ee1d1115bffeb6b4f0b92b6c705e8a48c62c9e53332465`.
- Result SHA-256: `542ca7d388a694f0eec9e859d3b93034cd9d88de3567aa92b07c4a083bdab829`.
- Metrics SHA-256: `a5353456e057745c8e3d590e0db9300599347d4d61d18c133446ead65754683b`.
- Artifact SHA-256: `b5b4efba433ebe49fafe986e71ffe7e360574001579402486014a48925ab5c2c`.
- Access SHA-256: `81a49c9a5032ff8df8b09638dddff1a53ec8b296704c935cae46e6aa347dea95`.
