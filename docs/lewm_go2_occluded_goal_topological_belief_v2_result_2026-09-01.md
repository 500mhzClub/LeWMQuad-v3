# OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2

Outcome-observed corrected development replacement on the unchanged constructed alias benchmark.

> V1 Stage A was observed before the batch-slot-dependent duplicate-image encoding defect was detected; V1 latent-derived evidence is invalid and unused.

> This is a constructed perceptual-aliasing challenge set, not an estimate of natural alias prevalence.

## Canonical encoding gate

The 375 template rows resolve to 157 unique exact-pixel rows and 33,384 occurrences. Each unique pixel was encoded alone, in sorted hash order, through two fresh encoder loads. Preprocessed tensors, raw tokens, spatial descriptors, and the canonical cache digest agreed exactly; tolerance was zero.

## Decision

Primary classification: `LOCAL_EXECUTION_INTERFACE_NO_GO`.
Secondary classifications: `TOPOLOGICAL_MAP_SUFFICIENT`.
Exact next experiment: `LOCAL_EXECUTION_INTERFACE_DIAGNOSTIC_V1`.

## Stage A

| Condition | Top-1 | Top-3 | MRR | NLL | Brier | ECE-10 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| CURRENT_FRAME_NEAREST_NODE | 0.164062 | 0.164062 | 0.177675 | 23.097807 | 1.671875 | 0.835938 | 0.000000 |
| FIXED_WINDOW_SEQUENCE | 0.492188 | 0.492188 | 0.501315 | 14.031378 | 1.015625 | 0.507812 | 0.000000 |
| MAP_FILTER | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| TOP_K_BELIEF | 1.000000 | 1.000000 | 1.000000 | 0.001319 | 0.000003 | 0.001318 | 0.001980 |
| FULL_BELIEF | 1.000000 | 1.000000 | 1.000000 | 0.002778 | 0.000010 | 0.002774 | 0.005050 |
| ORACLE_PLACE_IDENTITY | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| NO_ACTION_CONSISTENCY | 1.000000 | 1.000000 | 1.000000 | 0.002779 | 0.000010 | 0.002774 | 0.005050 |
| SHUFFLED_ACTION_HISTORY | 1.000000 | 1.000000 | 1.000000 | 0.002779 | 0.000010 | 0.002774 | 0.005050 |
| NO_OBSERVATION_LIKELIHOOD | 0.000000 | 0.000000 | 0.019156 | 5.143124 | 0.994446 | 0.012867 | 0.990299 |

| Condition | Edge acc. | Regret | Route top-3 | Wrong turn | False-conf. | False merge | False loop | Abstain | Correct abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CURRENT_FRAME_NEAREST_NODE | 0.312500 | 0.687500 | 1.000000 | 0.687500 | 0.835938 | 0.195312 | 0.171875 | 0.000000 | 0.000000 |
| FIXED_WINDOW_SEQUENCE | 0.492188 | 0.507812 | 1.000000 | 0.507812 | 0.507812 | 0.117188 | 0.000000 | 0.000000 | 0.000000 |
| MAP_FILTER | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | n/a |
| TOP_K_BELIEF | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | n/a |
| FULL_BELIEF | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | n/a |
| ORACLE_PLACE_IDENTITY | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | n/a |
| NO_ACTION_CONSISTENCY | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| SHUFFLED_ACTION_HISTORY | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| NO_OBSERVATION_LIKELIHOOD | 0.000000 | 1.000000 | 0.492188 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |

Stage-A classification: `TOPOLOGICAL_MAP_SUFFICIENT`. Strongest memory condition: `MAP_FILTER`. Stage B authorized: `True`.

Fresh calibration selected grid index `0` with parameters `{"action_compatible_edge_probability":0.7,"normalized_entropy_abstention_threshold":0.25,"observation_softmax_temperature":0.01,"transition_noise_probability":0.01}`.

FULL versus MAP gate: values `{"correct_next_edge_accuracy_gain":0.0,"false_confident_localisation_rate_reduction":0.0,"localisation_top3_gain":0.0,"normalized_graph_distance_regret_reduction":0.0}`, checks `{"edge":false,"false_confident":false,"regret":false,"top3":false}`, pass `False`.

Incremental-over-current dispositions for FIXED_WINDOW_SEQUENCE, MAP_FILTER, TOP_K_BELIEF, and FULL_BELIEF: `{"FIXED_WINDOW_SEQUENCE":{"checks":{"edge":false,"top3_or_regret":true},"pass":false,"values":{"correct_next_edge_accuracy_gain":0.1796875,"localisation_top3_gain":0.328125,"normalized_graph_distance_regret_reduction":0.1796875}},"FULL_BELIEF":{"checks":{"edge":true,"top3_or_regret":true},"pass":true,"values":{"correct_next_edge_accuracy_gain":0.6875,"localisation_top3_gain":0.8359375,"normalized_graph_distance_regret_reduction":0.6875}},"MAP_FILTER":{"checks":{"edge":true,"top3_or_regret":true},"pass":true,"values":{"correct_next_edge_accuracy_gain":0.6875,"localisation_top3_gain":0.8359375,"normalized_graph_distance_regret_reduction":0.6875}},"TOP_K_BELIEF":{"checks":{"edge":true,"top3_or_regret":true},"pass":true,"values":{"correct_next_edge_accuracy_gain":0.6875,"localisation_top3_gain":0.8359375,"normalized_graph_distance_regret_reduction":0.6875}}}`.

Fixed window matches strongest persistent condition: `False`; strongest absolute-gate-passing persistent condition: `MAP_FILTER`.

## Conditional Stage B

Stage B ran conditionally with the frozen prior current-visual ranker and no future predictor.

| Condition | Goals | Edge accuracy | Path efficiency | Wrong turns | False-confident wrong turns |
|---|---:|---:|---:|---:|---:|
| CURRENT_FRAME_NEAREST_NODE | 0/16 | 0.902344 | 0.000000 | 0.949219 | 0.097656 |
| STRONGEST_STAGE_A_MEMORY | 0/16 | 1.000000 | 0.000000 | 0.949219 | 0.000000 |
| ORACLE_PLACE_IDENTITY | 0/16 | 1.000000 | 0.000000 | 0.949219 | 0.000000 |

## Claims and safety boundary

Positive wording is limited to: **JEPA place belief under oracle topology on a corrected development cache**

This result does not establish deployment safety, learned contact avoidance, physical Go2 safety, online map construction, hidden beacon discovery, novelty, or complete maze navigation. The separate safety workstream remains `REQUIREMENTS_ACQUISITION_REQUIRED`.

All prohibited-action counters are zero. V1 latent-derived science was not reused; no predictor, ranker, or safety model was trained; and no custom audit/startup/forensic framework was introduced.

Prepublication runtime/storage observation: `2116.315549` seconds, `21` official leaves, `898120105` bytes (before result/report/hash-manifest publication).

Independent reducer receipt SHA-256: `12216a4146f148ba9c61b236c4f1937aee8387f08f317c0abae5dc22d63bd574`.
