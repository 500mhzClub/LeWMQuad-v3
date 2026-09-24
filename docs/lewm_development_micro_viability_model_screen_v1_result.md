# DEVELOPMENT_MICRO_VIABILITY_MODEL_SCREEN_V1

Status: completed

Source baseline: `53a011f0b55b816294c49ced88bd3a6a55c4adec`

Model classification: `DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL`

Compute classification: `MICRO_VIABILITY_COMPUTE_SIGNAL`

Interface status: `REPLANNING_INTERFACE_UNRESOLVED`

## Claims boundary

This was a development-only, non-claim-bearing architecture screen for
`SIMULATED_ONE_TICK_CONTACT_AND_SUCCESSOR_VIABILITY`. The predecessor terminal
`FRESH_MICRO_VIABILITY_PANEL_INADEQUATE` remains valid: that panel was not an
untouched claim-bearing panel, and no prior model result is retroactively
created. Correct abstention in an oracle-nonviable state is desirable, but
repeated abstention still fails mission-progress requirements.

No learned closed-loop safety, material-impact safety, physical Go2 safety, or
JEPA route-planning claim is supported. Platform-equivalent emergency stopping
remains unresolved. Memory, novelty, topology, beacon discovery, and global
navigation remain later layers.

## Frozen evidence and split

The predecessor 2,464-row ledger was verified at SHA-256
`0a273a3f464f770ccf8d28a1c6c3d9ddad63efdb767c1a63175ddcb479a18eea`.
No state, oracle branch, identity, candidate, or label was generated or
changed.

Before model initialization, the 128 original fit states were ordered by a
domain-separated SHA-256 and exhaustively label-stratified within each family.
Six states per family formed `DEVELOPMENT_INTERNAL_CALIBRATION_V1`. The
remaining 104 original fit states plus all 24 states from the old calibration
panel formed development training. The existing 24 held-out states remained
development held-out.

| Role | States | Rows | Contact + | Nonviable successors | Viable / nonviable states |
|---|---:|---:|---:|---:|---:|
| Development training | 128 | 1,792 | 313 | 38 | 116 / 12 |
| Internal calibration | 24 | 336 | 135 | 44 | 19 / 5 |
| Development held-out | 24 | 336 | 86 | 20 | 20 / 4 |

Each role contains 32/6/6 states per family, respectively. All three scene sets
are mutually disjoint.

The internal calibration identities are:

- large: `wide-cal-0-00`, `wide-cal-0-03`, `viability-fit-0-14`,
  `wide-cal-0-05`, `viability-fit-0-11`, `viability-fit-0-02`;
- medium: `viability-fit-1-06`, `viability-fit-1-18`, `wide-cal-1-03`,
  `wide-held-1-00`, `viability-fit-1-09`, `wide-held-1-02`;
- small: `wide-cal-2-04`, `viability-fit-2-18`, `wide-held-2-00`,
  `viability-fit-2-04`, `viability-fit-2-09`, `viability-fit-2-00`;
- loop: `wide-cal-3-02`, `viability-fit-3-18`, `viability-fit-3-17`,
  `wide-cal-3-04`, `viability-fit-3-19`, `wide-held-3-03`.

The development-held-out identities are `viability-held-{0,1,2,3}-{00..05}`.
The exact ordered 128-state training list and all 24+24 evaluation identities
are frozen in `development_internal_calibration_v1.json`, SHA-256
`d4148595ae1b3336eb7b5b597e78f83303c79af41bdc2e3210cd9c39b1c72db2`.

## Model and training

The architecture remained exactly 167,550 parameters: 64-D depth and LiDAR
encoders, a 96-D embodied/controller GRU, shared 160-D state encoding, 48-D
candidate encoding, batched 14-candidate fusion, and the frozen six outputs.
It received planning-boundary sensor/controller histories and candidate
contracts only; no future sensor, label, map, scene identity, route utility,
JEPA latent, or progress target was an input/output.

The evaluation fixture passed all 11 cases. The corrected smoke passed input
allow-list, candidate/successor alignment, leakage, contact/nonviability
gradients, candidate and temporal sensitivity, finite gradients, save/reload,
determinism, and evaluation-split isolation.

One seed, `2026082016`, ran for 60 epochs with AdamW, learning rate `1e-3`,
weight decay `1e-4`, complete-state batch size eight, and final epoch only.
No sweep, second seed, or best-checkpoint selection occurred.

| Epoch | Total | Contact BCE | Nonviability BCE | Ordinal BCE | Count Huber |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.37425 | 1.14732 | 1.40526 | 0.05812 | 3.17044 |
| 60 | 0.42278 | 0.16045 | 0.22054 | 0.01652 | 0.13409 |

Final checkpoint SHA-256:
`d30036746d0226eb36e9582c89122ea828aa2f509e18135135198639e51f846a`
(684,145 bytes).

## Development calibration

The independent scalar temperatures are:

- immediate contact: `6.0275520718`;
- successor nonviability: `12.1886439836`.

The complete constrained frontier contained 235 pairs, of which 100 met the
frozen calibration safety clauses. The lexicographically selected thresholds
are:

- contact: `0.0773603916`;
- successor nonviability: `0.1341051757`.

The selected calibration rule abstained on all 24 states: it correctly
abstained on all five oracle-nonviable states but falsely abstained on all 19
oracle-viable states. Thus even the calibration operating point already showed
the safety/mobility failure.

## Development held-out prediction

| Metric | Immediate contact | Successor nonviability |
|---|---:|---:|
| AUC | 0.48572 | 0.49261 |
| Average precision | 0.24063 | 0.08152 |
| Recall | 1.00000 | 1.00000 |
| FNR | 0.00000 | 0.00000 |
| Negative/viable retention | 0.03200 | 0.03478 |
| ECE | 0.13837 | 0.22658 |
| Brier | 0.22849 | 0.13489 |

The high recall is achieved by near-total rejection, not useful
discrimination. Per-family AUC/AP values were:

| Family | Contact AUC/AP | Contact retention | Nonviability AUC/AP | Viable retention |
|---|---:|---:|---:|---:|
| Large | 0.34259 / 0.11186 | 0.11111 | 0.67164 / 0.11409 | 0.11940 |
| Medium | 0.74286 / 0.29377 | 0.00000 | 0.43537 / 0.10165 | 0.00000 |
| Small | 0.89603 / 0.71321 | 0.00000 | 0.58730 / 0.18096 | 0.00000 |
| Loop alias | 0.12628 / 0.21126 | 0.00000 | 0.29091 / 0.02500 | 0.00000 |

All family recalls were 1.0 and FNRs zero under the conservative thresholds.
ECE/Brier were contact `0.23439/0.19396`, `0.23915/0.18546`,
`0.30372/0.19183`, `0.46645/0.34271`; nonviability
`0.19944/0.10566`, `0.28175/0.18208`, `0.12997/0.13667`,
`0.29955/0.11185` for large, medium, small, and loop, respectively.

## State decisions

On the 20 oracle-viable states:

- states retaining an admissible action: 1/20;
- selected contacts/nonviable successors: 0/0;
- false abstentions: 19;
- selected/oracle H3 progress: `0.15641 / 4.43258 m` (3.53%);
- normalized regret: `0.96471`;
- best-admissible top-1/top-3: `0.05 / 0.05`.

Large retained 1/6 viable states; medium, small, and loop retained 0/5, 0/4,
and 0/5. This is complete family collapse in three families.

On the four oracle-nonviable states, the model correctly abstained 4/4 and
made zero unsafe movement decisions. These are correct abstentions and were
excluded from false-abstention and movement-progress denominators.

The architecture misses many gates, so
`DEVELOPMENT_MICRO_VIABILITY_POSITIVE_TENDENCY` is not supported.

## Compute benchmark

The CPU float32 path loaded and normalized a production-format sensor row,
constructed tensors, encoded state once, scored all 14 candidates in one
batch, calibrated probabilities, thresholded, selected deterministically, and
serialized the command. It used 30 warm-ups and 1,000 timed iterations.

| P50 | P90 | P95 | P99 | Maximum |
|---:|---:|---:|---:|---:|
| 0.857 ms | 0.896 ms | 0.904 ms | 0.920 ms | 36.845 ms |

Deadline misses at 50/80/100 ms were 0/0/0. Peak RSS was 539,025,408 bytes,
RSS growth was 835,584 bytes, peak VRAM was zero, and memory was stable. CPU
use was 1547% of one core equivalent because the production PyTorch CPU path
used multiple worker threads; no GPU was available.

This passes `MICRO_VIABILITY_COMPUTE_SIGNAL`. It does not qualify per-tick
observation delivery, command replacement/acknowledgement, or physical Go2
latency, so `REPLANNING_INTERFACE_UNRESOLVED` remains.

## Decision

The primary development result is `DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL`.
The frozen architecture does not learn useful contact/nonviability
discrimination or a safety-mobility operating point from the existing data.
Compute is not the bottleneck.

Do not collect `FRESH_MICRO_VIABILITY_PANEL_V2` for this architecture. Close
`LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_V1` before spending further simulation
on claim-bearing evaluation. Any successor requires a separately authorized
architecture/target decision; no automatic second seed, architecture, panel,
or JEPA integration is justified.

The two-rate concept remains architectural context only: a fast micro
contact/viability layer and an approximately 200 ms macro JEPA route loop. No
JEPA predictor was opened in this pass, and lateral actions remain outside its
historical contract.

## Persistence and runtime

Training took `6.399 s`; the 1,000-iteration benchmark took `0.922 s`.
Experiment output and external row/timing/frontier evidence occupy 2,984,410
bytes.

The 2,464-row model ledger is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/development_micro_viability_model_screen_v1/row_level_model_evidence_v1.jsonl`,
1,700,696 bytes, SHA-256
`555ba6d2678e543cf78d6a53977eceeaa5bddf60a6c16c2510ee028db9f7cba2`.
It persists raw logits, calibrated probabilities, labels, thresholds,
admission, selections, and route quantities for training, calibration, and
development-held-out rows.

Exactly one model seed was trained. No new state/panel or oracle branch was
generated. No JEPA predictor, utility model, learned closed loop, memory,
novelty, routing, beacon capture, or navigation system was opened, trained, or
executed.
