# Go2 categorical radial ladder v2 result

Date: 2026-07-10

Status: N=1 and N=4 passed; N=16 terminal gate failed; N32 not licensed

This is a train-role-only implementation and capacity diagnostic. It did not
evaluate G2, select a checkpoint, fit calibration, or license a perception or
navigation promotion.

## Immutable artifacts

- frozen ladder manifest:
  `.generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json`;
- ladder manifest file SHA-256:
  `967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12`;
- preregistered optimizer amendment:
  `docs/lewm_go2_categorical_radial_ladder_v2_optimizer_amendment_2026-07-10.md`;
- amendment file SHA-256:
  `58f994a639c8e5a733d92c6da1fad63fa654e1f57aa7be0a8373e3eaa47b3f46`;
- V2 result:
  `.generated/go2_categorical_radial_micro_overfit/v2/seed_20260710_ladder_result.json`;
- V2 result file SHA-256:
  `06517e2c6641495a6262aa9f8a5cb45648912c575f1c3663df899c50a2867daa`;
- V2 result content SHA-256:
  `8528ae02d6faaf25eb666d591e15180e82f74c9cf4d798c8322f9d5c50c910bc`.

The result content hash recomputes exactly. It binds the immutable V1 result,
the final V1 sources, the V2 runner, the optimizer amendment, the model, and
the frozen panel and ladder. All checkpoint-selection, probability-calibration,
G2, and other non-train access counters are zero. The production ROCm runtime
reproduced the V1 seed-20260710 initialization hash exactly. The artifact
discloses the warn-only nondeterministic ROCm `grid_sample` backward kernel.

## Fixed-terminal results

| Frames | Balanced NLL | UNKNOWN recall | FREE recall | OCCUPIED recall | Wrong-view NLL delta | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.00013225 | 1.00000 | 1.00000 | 1.00000 | n/a | PASS |
| 4 | 0.00125846 | 0.99909 | 1.00000 | 0.99194 | 4.58099 | PASS |
| 16 | 0.01151191 | 0.98747 | 0.99828 | 0.97007 | 3.53510 | FAIL |

All stages consumed their complete registered budgets and used `1e-5` on the
final update. N=1 passed every evaluation from step 400 onward. N=4 passed
every evaluation from step 400 onward and ended without the V1 tail excursion.
Its terminal OCCUPIED result was 123/124 cells, exactly the one-error margin
permitted by the 0.99 gate; this confirms the optimizer intervention but is not
generalization evidence.

N=16 never passed the complete gate. Its curve improved smoothly through the
terminal step rather than crashing:

- NLL fell from 0.28293 at step 100 to 0.01151 at step 2,000;
- UNKNOWN recall rose to 0.98747 and OCCUPIED recall to 0.97007;
- FREE recall reached 0.99828, including 0.99731 at 3 m and beyond;
- correct-RGB versus wrong-view separation grew to 3.53510 NLL;
- the terminal joint confusion contained 12 truly OCCUPIED cells predicted
  UNKNOWN and 736 truly UNKNOWN cells predicted known.

The residual is specifically UNKNOWN-versus-known evidence, not conditional
FREE-versus-OCCUPIED discrimination. The terminal conditional confusion is
`[[6390, 11], [0, 401]]`: all 401 OCCUPIED cells beat FREE once admitted as
known. UNKNOWN/known weighted NLL is `0.01902187` versus `0.00400196` for
FREE/occupied, contributing 82.62% of the hierarchical loss. With the latter
component fixed, the UNKNOWN/known term needs a 15.90% reduction for aggregate
NLL to pass. This localizes the next hypothesis to cross-range occlusion and
known-boundary context.

The strong wrong-view control proves the model still uses the correct image.
The stable late curve rules out the specific V1 constant-rate terminal
excursion. Under the preregistered interpretation, passing N=1/N=4 but failing
N=16 is a capacity or architectural-structure failure at the fixed budget.
Neither more updates, an alternate schedule, a best intermediate checkpoint,
nor a second seed is licensed by this result.

## Decision

N32 and both holdout panels remain blocked. The next experiment must be one
dated, architecture-only intervention that preserves the frozen frames,
factorization, labels, loss, V2 learning-rate schedule, budgets, controls, and
gates. It must restart N=1/4/16 from the seed-specific initialization and stop
on its first failed terminal stage. No non-train role may be opened.
