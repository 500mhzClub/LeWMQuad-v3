# Go2 categorical-radial N32 V1 result

Date: 2026-07-11

Status: authoritative seed 20260710 failed the fit gate; seed 20260711 forbidden

## Bound execution

The run used the immutable contract in
`docs/lewm_go2_categorical_radial_n32_execution_binding_2026-07-10.md` without
changing the model, panel, controls, optimizers, budgets, batching, or gates.
The canonical result is
`.generated/go2_categorical_radial_n32/v1/seed_20260710_result.json`.

- file SHA-256:
  `2f079925000ebbcd06843c413f4dcfd07fce93358482dd05512735af69cbc946`
- content SHA-256:
  `ef023faff0e49888ca673cfab5fca0c1110852e49312ce339ecb7f03ab3a8d5b`
- elapsed wall time: 46 minutes 52 seconds
- model: `CategoricalRadialPerceptionFullRay`, 2,887,067 parameters
- initial state SHA-256:
  `8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278`

The hardened finalizer independently reloaded the bound evidence, regenerated
the seed-specific wrong-view controls, validated every stored schedule and
access count, recomputed both terminal gates, and reproduced
`classification=fit_gate_failed`.

## Result

The production-faithful branch consumed all 2,000 updates and failed. The
ceiling branch restarted from the exact same initial state, consumed all 5,000
updates, and also failed. Neither branch passed a single complete aggregate
plus five-family fit evaluation, so no qualifying optimizer stage exists.

At the best aggregate ceiling evaluation, step 4,900:

| Metric | Value | Gate |
| --- | ---: | ---: |
| hierarchical balanced NLL | 0.04954 | <= 0.03 |
| UNKNOWN/KNOWN balanced accuracy | 0.97265 | >= 0.99 |
| FREE/OCCUPIED balanced accuracy | 0.98850 | >= 0.99 |
| UNKNOWN recall | 0.95222 | >= 0.98 |
| FREE recall | 0.99017 | >= 0.98 |
| OCCUPIED recall | 0.93674 | >= 0.98 |
| far-FREE recall | 0.99000 | >= 0.95 |

Wrong-view separation was already large, so the head was using RGB rather than
collapsing to label priors. At the final ceiling evaluation, the cross-scene
and same-scene wrong-view NLL deltas were 3.16751 and 1.85377, both well above
the fixed 0.25 gate. The remaining failures were correct-view UNKNOWN/KNOWN
and OCCUPIED accuracy, concentrated differently by family: open fields and
rough dynamics were weakest on UNKNOWN/OCCUPIED separation, while enclosed
mazes retained lower FREE/OCCUPIED and far-FREE recall.

## Isolation

The conditional access contract worked:

- fit images decoded: 320; fit label shards opened: 20;
- same-scene holdout image/shard/model-output access: 0/0/0;
- cross-scene holdout image/shard/model-output access: 0/0/0;
- checkpoint-selection, calibration, non-train, and G2 byte/model access: zero;
- no seed-20260711 run is authorized;
- no full-training, calibration, G2, runtime, or promotion license exists.

## Diagnosis and next license

The preceding N=16 ladder stage contained 16 frames, batch size 4, and 2,000
updates: 500 effective passes over each frame. N32 contains 320 fit frames.
Its faithful and ceiling branches therefore supplied only 25 and 62.5
effective passes per frame. The ladder-to-N32 jump was 20x in unique frames but
only 2.5x in maximum updates, an eightfold reduction in per-frame exposure.

The ceiling curve improved smoothly from NLL 0.22104 at step 500 to 0.04954 at
step 4,900, while UNKNOWN/KNOWN accuracy rose from 0.92961 to 0.97265. There is
no optimizer excursion, control collapse, factorization failure, or observed
plateau proving an architectural capacity limit. The highest-information next
experiment is therefore a preregistered exposure-matched N32 retry that keeps
the frozen model, panel, controls, loss, fit/holdout gates, and conditional
access rules. It must be bound before output and must preserve the two-seed
authorization rule. This result licenses construction of that train-role-only
retry, and nothing beyond it.

