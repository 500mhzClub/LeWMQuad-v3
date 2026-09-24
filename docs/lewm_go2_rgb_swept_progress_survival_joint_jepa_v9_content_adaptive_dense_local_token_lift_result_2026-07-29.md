# RGB Swept-Progress Survival Joint-JEPA V9 — Dense Local-Token Lift Result

- Terminal status: **VALID COMPLETE PASS — FULL DEVELOPMENT ARM**.
- V9 passed all 24 unchanged development checks and is staged for exactly one
  separately frozen physical-evidence calibration. It is not yet physically
  qualified and no G2, navigation, held-out, sealed, promotion, or deployment
  operation is open.
- Preregistration / amendment / source / execution-binding commits:
  `47043472466e7a258ad0f0be854c05393e233db8` /
  `04db6b26d46875297e3aa515fdf1d688bee2b755` /
  `5c70884c108fe8c6b4051249cb614a31c442f0fd` /
  `29cff2ba5c88321fc1bc98bf1075f5de1526a233`.
- The sole authorized process exited `0` with
  `PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION`. No retry, resume, alternate
  seed, schedule extension, calibration, or benchmark run occurred.

## Integrity and execution

- Result: `69,002` bytes; file SHA-256
  `698acce34e9221e1660d243133937b621abc6742a5436a859c91b7ffbf55c7e5`;
  self-verifying content SHA-256
  `344d10db882314fa3f227597dba4fc7e96747e3fdbe3f6d134e6c7f28c5c2c28`.
- Training trace: `903,741` bytes; file SHA-256
  `8c13bb04f9dfaf44d4336e3ffa6e17b3352c8a2707bf128d7421a266facff225`;
  self-verifying content SHA-256
  `6d7fd806ff9e4316f41cf7447694dad3f6d419ea78b3ce6ab9b734e93de9ee3b`.
  The result's embedded trace binding matches all three values.
- The result embeds a development-checkpoint receipt of `25,427,815` bytes
  and SHA-256
  `5456dc94136503543439e4bf691b8120c63c45a04e692f640c9c246f243c5ffd`.
  This line records only the result receipt; the checkpoint was not opened,
  listed, statted, or independently hashed during result audit.
- Accounting is exact: 1,000 updates, optimizer steps, and EMA steps; 4,000
  microbatch graphs, backward calls, predictor forwards, and predictor
  objectives; 16,000 presentations; and 1,000 ordered trace rows with
  `presentations = 16 * update`.
- `L=S+P+U+R+O` throughout with maximum absolute floating error `5.52e-7`.
  Ranking was active in all 4,000 microbatches, with 284,795 eligible pairs
  and 1,318,068 supervised survival decisions.
- All seven new Q/K/V/O attention tensors, totaling 16,576 parameters, received
  finite nonzero gradients from update 1 through update 1,000. Target-gradient
  tensor count remained zero. The predictor, encoder, lift, and semantic route
  were trained jointly from update one; no head or predictor was trained
  separately.
- The initial receipt confirms exact inherited V4 state, exact initial online
  and target copies, one hard sync, 25 row-major local supports, per-head valid
  weights summing to one, zero invalid-support weights, and inherited null
  evidence for all-invalid cells.
- Forbidden input count, every forbidden semantic counter, and G2/final
  evaluation opens were zero. Held-out and sealed material remained unopened.

## Training behavior

| Loss mean | Updates 1–100 | Updates 801–900 | Updates 901–1000 |
|---|---:|---:|---:|
| Total `L` | `7.767034` | `5.724620` | `5.926284` |
| Semantic `S` | `2.112402` | `1.894127` | `1.971677` |
| JEPA persistence `P` | `2.366728` | `1.347222` | `1.371285` |
| Survival `U` | `0.667851` | `0.335124` | `0.348355` |
| Ranking `R` | `0.846656` | `0.513725` | `0.525086` |
| Half-weight occupied auxiliary `O` | `1.773396` | `1.634422` | `1.709881` |

- All objectives improved substantially relative to the first 100 updates.
  The modest last-window rebound does not alter the fixed terminal evaluation
  or authorize schedule extension.

## Unchanged development gate

| Selection metric | V4 | V8 | V9 | Gate | V9 |
|---|---:|---:|---:|---:|---|
| Balanced accuracy | `0.850286` | `0.849307` | `0.846930` | `>=0.80` | PASS |
| Free recall | `0.857970` | `0.849104` | `0.859012` | `>=0.85` | PASS |
| Occupied recall | `0.744512` | `0.749337` | `0.734002` | `>=0.70` | PASS |
| Rough occupied recall | `0.703615` | `0.754943` | `0.675055` | `>=0.65` | PASS |
| Unknown recall | `0.948376` | `0.949480` | `0.947775` | `>=0.90` | PASS |
| Informative action utility | `0.906910` | `0.906094` | `0.902861` | `>=0.85` | PASS |
| Selected zero-prefix rate | `0.035088` | `0.032581` | `0.030075` | `<=0.05` | PASS |
| Unequal-pair concordance | `0.868433` | `0.862805` | `0.858114` | `>=0.75` | PASS |

- V9 recovered the FREE-recall failure that closed V6 and V8 and improved the
  selected zero-prefix rate relative to V4 and V8. Against V4 it traded away
  occupied recall, rough occupied recall, utility, and concordance while still
  clearing every fixed floor. It therefore establishes viability, not broad
  dominance of the dense lift.
- All family utility, zero-prefix, and concordance checks passed. All twelve
  causal-control checks passed. Equal-scene delta / bootstrap lower 95% /
  positive families were: persistence `+0.144147 / +0.084947 / 8`, shuffled
  action `+0.317187 / +0.258760 / 8`, wrong RGB
  `+0.098213 / +0.053544 / 7`, and train-action prior
  `+0.071000 / +0.032902 / 7`.

## Decision and next step

- The content-adaptive 5x5 lift solved the immediate complete-development-gate
  problem without changing data, schedule, losses, encoder family, predictor,
  initialization, or thresholds. The result also confirms that the new lift
  actually learned inside the joint JEPA rather than acting as dead plumbing.
- Because rough occupied recall is weaker than prior candidates, the mandatory
  next discriminator is the existing conservative physical-evidence protocol:
  one calibration fit on `probability_calibration`, one frozen threshold
  selection over the registered 2,016 tuples, and one score on
  `checkpoint_selection`.
- A separately committed preregistration and execution binding must precede
  checkpoint access. A calibration pass may open preparation of G2; a
  scientific calibration failure closes V9 without threshold relaxation or a
  repeated calibration variant.
