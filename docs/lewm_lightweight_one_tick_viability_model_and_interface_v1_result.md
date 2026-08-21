# LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_AND_INTERFACE_V1

Status: completed at the preregistered pre-training stop

Source baseline: `90dda7ecde62a6edfb1c837a0b456e4950b31f7d`

Primary classification: `FRESH_MICRO_VIABILITY_PANEL_INADEQUATE`

## Result

The fresh-panel adequacy gate failed before training. Calibration retained an
oracle viability-admissible action in 22/24 states, but held-out retained one
in only 20/24. The calibration `small_enclosed_maze` family contained no
contact-positive candidate, and calibration `small_enclosed_maze` and
`loop_alias_stress` contained no nonviable successor. Consequently, the
required per-family contact/nonviability coverage also failed.

Section 9 required stopping without training and prohibited replacing a state
after its outcomes were known. No checkpoint, calibration temperatures,
decision thresholds, fresh held-out model metrics, interface timing result, or
learned closed-loop result therefore exists. This is a panel-design terminal,
not a negative result about the specified model architecture.

## Claims boundary and preserved results

The intended target was
`SIMULATED_ONE_TICK_CONTACT_AND_SUCCESSOR_VIABILITY`. It is a simulated
contact/viability target, not evidence about material impact, injury, property
damage, physical Go2 safety, a qualified emergency stop, JEPA route utility,
or global navigation. Ideal LiDAR and depth remain an explicit changed
deployment sensor contract.

The result preserves:

- `LATERAL_AUGMENTED_STATE_ELIGIBILITY_SIGNAL`;
- `LATERAL_RECOVERY_CONTROLLER_QUALIFIED`;
- `LATERAL_CONTROLLER_SIGNAL_VIABILITY_NO_GO`;
- `ONE_TICK_FULL_JEPA_COMPUTE_NO_GO`;
- `TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED`;
- `REPLANNING_INTERFACE_UNRESOLVED`;
- `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`.

It also preserves the central oracle finding: with earlier intervention, the
14-action micro bank and unchanged H3 ranker selected zero contacts and zero
nonviable successors, kept at least two safe next actions in 96.49% of cycles,
and caused no matched-control route-progress regression. `wide-held-2-04`
remains a state-eligibility violation because it was already in disallowed
contact at its registered boundary. Lateral recovery is simulation-qualified,
not physically Go2-qualified, and the historical JEPA predictor remains
unqualified for `vy` actions.

## Frozen panel and inputs

The manifest SHA-256 is
`520fa89b894c7cac84ea15b6b53c1d3ac969bf5e4282387c286ff80a23c1c04f`;
its content digest is
`34f7eb6609dd28053eba066fb56f18f3b68451809cd22eab1378bb95b1d02695`.
All 176 state and scene identities are distinct; fresh evaluation overlap with
fit is zero, and new-corpus overlap with historical or predictor scenes is
zero. Identities were frozen before candidate execution.

| Role | States | Scenes | Candidate rows | Source |
|---|---:|---:|---:|---|
| Fit | 128 | 128 | 1,792 | 48 compatible historical roots plus 80 new roots |
| Calibration | 24 | 24 | 336 | fresh, six per family |
| Held-out | 24 | 24 | 336 | fresh, six per family |

Fit contains 32 states per family. Calibration and held-out contain six states
per family. The pre-action stratification used clearance bands at
`0.27324061 m` and `0.33389097 m`, command-magnitude median `0.25`, and a
turning rule of absolute applied yaw rate at least `0.10 rad/s`. The clearance
quantity was an analytic base-point distance to frozen collision boxes for
stratification only; it was not represented as a Genesis articulated
positive-distance claim.

Planning inputs persisted per state are three current/history depth frames and
validity masks, three LiDAR frames and validity masks, five enhanced embodied
and controller-history samples, active/previous controller state, and all 14
candidate contracts. No future sensor, global pose/yaw, scene/family identity,
map, exact clearance, label, or successor count is a model input.

## Oracle-tree materialisation

Every new state used the frozen 14-action micro bank. Every contact-free prefix
was expanded through all 14 successor actions. Historical root outcomes were
reused only for the 48 compatible fit states.

| Population | New states | Generated branches |
|---|---:|---:|
| New fit | 80 | 13,104 |
| Fresh calibration | 24 | 4,200 |
| Fresh held-out | 24 | 3,836 |
| Reused historical fit | 48 | 0 |
| **Total** | **176 records** | **21,140** |

No evaluation state was replaced after outcomes. The fresh panel was not used
for fitting.

## Prevalence and adequacy

| Split | Contact + / total | Nonviable / contact-free | States with >=1 admissible | States with >=2 admissible |
|---|---:|---:|---:|---:|
| Fit | 388/1,792 (21.65%) | 79/1,404 (5.63%) | 113/128 | 106/128 |
| Calibration | 60/336 (17.86%) | 3/276 (1.09%) | 22/24 | 21/24 |
| Held-out | 86/336 (25.60%) | 20/250 (8.00%) | 20/24 | 20/24 |

Fresh per-family values are:

| Split/family | Contact + | Nonviable successors | States with >=1 admissible |
|---|---:|---:|---:|
| Calibration / large | 23 | 2 | 5/6 |
| Calibration / medium | 23 | 1 | 6/6 |
| Calibration / small | 0 | 0 | 6/6 |
| Calibration / loop alias | 14 | 0 | 5/6 |
| Held-out / large | 12 | 5 | 6/6 |
| Held-out / medium | 14 | 7 | 5/6 |
| Held-out / small | 32 | 7 | 4/6 |
| Held-out / loop alias | 28 | 1 | 5/6 |

Both aggregate contact classes and both aggregate viable/nonviable successor
classes occur in each split. The panel fails the two remaining frozen clauses:

- held-out does not reach 22/24 oracle-viable states;
- every family does not contain both contact and nonviability examples.

## Evaluator and unexecuted model contract

The evaluation-first fixture passed all 11 cases: viable prefix, nonviable
prefix, immediate contact, one/two safe successors, lateral-required recovery,
route/lateral transition, all rejected, threshold tie, deterministic
selection, and serialization. Four focused source tests pass.

The implemented but untrained architecture has 167,550 parameters: 64-D depth
and LiDAR encoders, a 96-D embodied/controller GRU, a shared 160-D state
embedding, a 48-D candidate encoder, a batched candidate fusion head, and the
six preregistered contact/nonviability/ordinal/count outputs. Because the panel
gate failed, seed `2026082016` was not launched. There are no training metrics
or checkpoint SHA-256, and no calibration temperatures or thresholds were
selected.

The simulation-side 100 ms interface source and conditional closed-loop
evaluator were implemented but not executed. Without a prospectively qualified
offline model, their scientific gates could not be reached.

## Exact next decision

Do not train on this panel and do not relabel or recycle its calibration or
held-out states. Specify a fresh `FRESH_MICRO_VIABILITY_PANEL_V2` before another
model attempt. Its pre-action selection contract must be frozen prospectively
and must target:

- at least 22/24 oracle-viable states in each evaluation split;
- contact and successor-nonviability examples in every family;
- no outcome-based state replacement;
- complete scene/state disjointness from this now-observed panel.

The dominant panel-design corrections are to include pre-action states closer
to the learned eligibility-envelope boundary in calibration small/loop scenes,
while excluding already unrecoverable held-out starts using only prospective
pre-action eligibility variables. This is a panel-design decision, not
authorization to tune against these observed outcomes.

Only after a new panel passes its preregistered adequacy gate may the same
single-seed model experiment be reconsidered. JEPA access, utility learning,
memory, beacon discovery, and global navigation remain out of scope. Physical
deployment remains blocked by sensor-contract, command-latency, lateral-control,
and platform stopping-mode qualification.

## Runtime and persistence

The parent four-worker oracle-tree collector reported `1,328.601 s` (22 min
8.601 s). Up to eight additional disjoint-index helpers ran concurrently, for
a maximum of 12 state processes; the parent hash-validated and skipped their
completed shards.
The experiment-specific result tree, both ordinary scene corpora, and external
cache occupy 109,479,214 bytes (about 104.41 MiB).

The 2,464-row ledger is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/lightweight_one_tick_viability_model_and_interface_v1/row_level_evidence_v1.jsonl`,
2,239,577 bytes, SHA-256
`0a273a3f464f770ccf8d28a1c6c3d9ddad63efdb767c1a63175ddcb479a18eea`.
Its rows bind every candidate label and decision quantity to the SHA-256 of the
state input shard. Aggregates reproduce without simulation or model inference.

No model seed was trained because the mandatory pre-training gate failed. No
JEPA predictor, utility model, memory, or global navigation system was opened,
trained, or executed. No interface benchmark or learned closed-loop evaluation
ran.
