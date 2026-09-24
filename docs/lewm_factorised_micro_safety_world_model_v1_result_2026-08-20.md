# Factorised micro-safety world model V1

Date: 2026-08-20

Starting source: `4bb63ff19a3972aa594fa9d14ea39f55a1401ccb`

Preserved terminal: `SPECIALIST_SCORE_FRONTIER_NO_GO`

## Classification

`FACTORISED_MICRO_SAFETY_TRUE_FUTURE_NO_SIGNAL`

The prospectively designed, mechanism-separated contact and stuck specialists did not produce a useful safety–mobility operating point on the fresh scene-disjoint panel. The joint rule rejected every unsafe branch, but it retained only one of 107 safe held-out branches, falsely abstained in 21 states, and collapsed completely in three families. This is not a successful safety result: a filter must preserve enough safe actions for the task to proceed.

No candidate-conditioned predictor was opened. Under the frozen decision rule, this result closes post-hoc true-future filtering for the current candidate bank and enhanced embodied sensor contract. The next decision is an explicitly changed environmental sensor contract (for example depth/LiDAR) or a safety claim narrowed to observable failure modes.

Machine-readable result SHA-256: `38e7301bc5b9af6cc17ad0b99ea058e6ce7709c48553c8c87575aaa093c3e42d`.

## Bindings and fresh panel

The 48-state/576-branch predecessor panel became development fit data in full:

- enhanced sensor index: `d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d`;
- specialist ledger content digest: `e4e7ae1b494b171dd8a623a5368045a07f315e4ff05a85921b7e004c7d55e9de`;
- specialist ledger SHA-256: `a28be7a1254a77b553730c3024fb6ef24ed914a64ebf8bae3458142e3b0f8a08`.

The prospective panel was frozen before candidate execution:

- manifest content digest: `b5ee0e3a0814bb25058ea8a2cba77c108cce3a2761c61f74b10c9ccff197ebce`;
- manifest SHA-256: `74ee7e3bc63888d9ac96543f6d7b626c8ec1f01911d2c8068efcbd0681769410`;
- fresh sensor-index content digest: `aa9ba33349683b59edb5b23a4929a9c732224d24bbfa161d595c2e09f701dccc`;
- 24 calibration and 24 held-out states, six per family per split;
- 288 calibration and 288 held-out branches;
- 48 distinct scenes and state clusters;
- overlap with the predecessor panel: zero scenes;
- overlap with the bound predictor training/selection manifest: zero scenes.

Calibration scenes, in family order, were:

- large: `7e2aa44b4d66`, `fdada5570a66`, `2729a7617bc7`, `047e61730db3`, `3b5933ca2500`, `fe99f301c4e6`;
- medium: `efd8701e00b4`, `415c0a8663d3`, `fca64441f28e`, `6bd3ca82314d`, `4d4be6cd81fb`, `69b6772f365d`;
- small: `20d2b35b76f0`, `39dcf9e05de3`, `6d0b740337dc`, `51eb8cadc0cc`, `94abe93c521f`, `67c4389e5cb3`;
- loop alias: `c0a5973b549d`, `18e59310957b`, `3d25ba7623f2`, `6f352017a9d6`, `729e0984f4a4`, `fa1f20095916`.

Held-out scenes were:

- large: `e2ac8bcf8f49`, `5695763f07a8`, `152a2fdbeaf3`, `fe0c81bd12dd`, `f7da20d7a290`, `fdf7e1819b35`;
- medium: `159020a9456d`, `5a25e2e79f71`, `3a445c99735b`, `b05c73f530b6`, `fb9493819208`, `4553360e630e`;
- small: `0722badd3d11`, `5e61ce176922`, `5350355198d3`, `918c2c37c23f`, `44c52d505897`, `dc9d2c4be8c7`;
- loop alias: `2410ee35f838`, `16271f065364`, `17a8e40b8b5e`, `edc18de1ee98`, `0675b5a93974`, `8a6dfbf89389`.

Every branch receipt contains 15 ticks, exact registered requested/post-slew action identity, finite 73-channel enhanced embodied state, six action/control channels, and matching pose/safety trace cardinality. No RGB was rendered or encoded.

## Fresh-panel adequacy

The frozen adequacy gate passed before training.

| Split | Safe | Unsafe | Contact positive | Stuck positive | Contact/stuck overlap | Contact event ticks | Stuck event ticks | States with no safe candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration | 98 | 190 | 93 | 152 | 55 | 200 | 402 | 4/24 |
| Held-out | 107 | 181 | 97 | 147 | 63 | 210 | 347 | 2/24 |

Every family in both splits contained contact and stuck examples. Calibration had at least four safe-retaining states in each family (5 large, 4 medium, 6 small, 5 loop); held-out had 5, 5, 6, and 6 respectively.

## Evaluator and model contract

All twelve evaluator fixtures passed: transient contact, persistent contact, delayed stuck, safe branch, all unsafe, one safe, separate component decisions, deterministic OR, strict threshold tie, abstention, deterministic kinematic selection, and deterministic row-ledger serialization.

The single model object had no shared trainable parameters or normalization statistics:

| Specialist | Inputs | Temporal model | Parameters |
|---|---|---|---:|
| Contact/impact | acceleration, gyro, joint acceleration, torque, calf force, previous action, command/candidate | causal residual TCN, width 96, dilations 1/2/4 | 97,346 |
| Stuck/motion shortfall | action/command/history, previous action, joint position/velocity, projected gravity, gyro | one-layer GRU, width 128 | 107,906 |
| Total | — | — | 205,252 |

Both specialists used per-tick and cumulative balanced BCE plus the frozen H3 within-state ranking weight `0.25`. Seed `2026082010`, AdamW `1e-3`, weight decay `1e-4`, and 60 final-epoch-only epochs were used exactly once.

| Epoch | Contact total / BCE / rank | Stuck total / BCE / rank |
|---|---:|---:|
| 1 | 1.28473 / 1.14661 / 0.55249 | 1.26798 / 1.09552 / 0.68984 |
| 60 | 0.00248 / 0.00248 / 0.00000 | 0.15841 / 0.15633 / 0.00832 |

Final checkpoint SHA-256: `93f919238ff7b757b77f5281f45c59818c9f2b33fa5fbd96a2554b7aea14776e`.

## Fresh calibration

Temperatures and thresholds were fit on the 24 fresh calibration states only:

- contact temperature: `10.214079476441897`;
- stuck temperature: `4.769742081402547`;
- contact threshold: `0.24261455237865448`;
- stuck threshold: `0.13684044778347015`;
- equality rejects;
- complete frontier: 84,100 threshold pairs, 2,173 feasible under the frozen safety eligibility rule;
- frontier SHA-256: `af8e78bf77931228bc967491bf2de0a0e51caa636c12c020f45b213d0d1d768d`;
- frontier content digest: `ee9b1861682738aaa66d62bbad9b70f13e288dc9355768323849cbc6fb730a8d`.

At the selected calibration point, aggregate recall/FNR were `1.0/0.0`, contact recall `0.9140`, stuck recall `0.9934`, and zero unsafe candidates were admitted. The rule retained 4/98 safe candidates across only 3/24 states, already demonstrating severe conservatism; it was nevertheless selected by the frozen lexicographic rule because no more usable safety-feasible pair existed.

## Fresh held-out component metrics

| Component | AUC | AP | Recall / FNR | Specificity | ECE | Brier | Event-tick recall | Median delay | Missed transient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Contact | 0.7660 | 0.7309 | 0.9381 / 0.0619 | 0.1623 | 0.2204 | 0.2209 | 0.8286 | 0 ticks | 0.1190 |
| Stuck | 0.8068 | 0.7817 | 1.0000 / 0.0000 | 0.0355 | 0.0675 | 0.1811 | 0.9827 | 0 ticks | 0.0179 |

The component gates failed contact AUC (`0.7660 < 0.80`) and stuck AUC (`0.8068 < 0.85`). Event detection remained substantially better than candidate-level negative discrimination.

## Combined filtering and kinematic planning

| Metric | Held-out value | Frozen requirement |
|---|---:|---:|
| Aggregate unsafe recall / FNR | 1.0000 / 0.0000 | ≥0.95 / ≤0.05 |
| Safe retention | **0.00935 (1/107)** | ≥0.40 |
| Admitted unsafe | 0/181 | 0 selected; no unsafe-only state |
| States retaining a safe candidate | **1/24** | ≥18/24 |
| States with no admitted candidate | 23/24 | — |
| False abstentions | **21** | ≤3 |
| Selected unsafe rate | 0 | 0 |
| Selected progress | 0.0973 m | — |
| Oracle-safety kinematic progress | 0.2067 m | — |
| Oracle-progress fraction | **0.4705** | ≥0.80 |
| Normalized safe-progress regret | **0.7283** | ≤0.20 |
| Best-safe top-1 / top-3 | **0 / 0** | top-3 ≥0.75 |

Only `micro-held-2-04` admitted a candidate: candidate 11, safe, with `0.09727 m` distance progress and `0.09138 rad` heading improvement. Every other state abstained. The progress ratio uses the existing evaluator convention (means over non-abstaining selections), so it is not a mission-level retention guarantee.

### Per-family held-out result

| Family | Contact AUC | Stuck AUC | Unsafe recall | Safe retention | Safe-retaining states | False abstentions | Selected progress | Regret | Top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Large enclosed | 0.7498 | 0.8621 | 1.0000 | 0 | 0/6 | 5 | 0 | NA | 0 |
| Medium enclosed | 0.7643 | 0.8891 | 1.0000 | 0 | 0/6 | 5 | 0 | NA | 0 |
| Small enclosed | 0.8323 | 0.8351 | 1.0000 | 0.0323 | 1/6 | 5 | 0.0973 m | 0.7283 | 0 |
| Loop alias stress | 0.7884 | 0.7548 | 1.0000 | 0 | 0/6 | 6 | 0 | NA | 0 |

Three families collapsed completely. All frozen safety, mobility, ranking, and family gates were evaluated; the complete machine result records each Boolean failure.

## Row-level evidence

The held-out ledger persists identities, family/split, candidate action/control inputs, raw per-tick contact and stuck logits, calibrated H3 probabilities, labels, threshold decisions, kinematic and realised route quantities, admitted set, and selected candidate:

- rows/states: 288/24;
- SHA-256: `5775578c067d3efc0b4088cbf104c17b5b51552b3665bc335e9f631572caceaf`;
- decoded content digest: `0a49d31e86ae5fd87a0ba609e28e3833a8640c736203db7862dbd96fa1db00c5`;
- bytes: 71,765.

This satisfies `ROW_LEVEL_EVIDENCE_PERSISTENCE` and permits all aggregate, component, admission, and state-level planning reductions without rerunning inference.

## Runtime, storage, and custody

- candidate-blind eligibility scan: 92 scene attempts; 2,626.3 aggregate process-seconds;
- branch generation: 414.5 s wall, 1,602.4 aggregate process-seconds;
- fresh sensor shards: 2,302,232 bytes;
- training: 19.69 s; peak VRAM 73,708,032 bytes;
- training, calibration, and evaluation process: 42.58 s;
- checkpoint: 835,245 bytes;
- generated experiment directory: 5,741,952 bytes;
- high-capacity cache directory including logs/frontier/ledger: 7,162,640 bytes;
- focused tests: 12 passed across the new and predecessor frontier suites.

Exactly one factorised seed (`2026082010`) was trained. No JEPA predictor was opened or trained. No RGB was rendered or encoded. No global memory, novelty, routing, beacon, or navigation model was opened or trained. Candidate generation ended before model training, and no process remains running.
