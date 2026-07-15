# Protected-Camera physical-gate oracle V1 preregistration

Status: `BLOCK_UNPROVEN`. This document grants no execution, dataset, generated-mutation, training, checkpoint, model, RGB-decoding, calibration, G2, navigation, heldout, retry, promotion, or deployment authority.

## Question

Before spending another training run, determine whether the unchanged protected-Camera gate is attainable on the frozen V13 `checkpoint_selection` population: 495 pairs, 924 unique endpoints, eight families, nine declared scopes, and exactly 21 physical margins per scope (189 total).

## Fixed positive control

The diagnostic has zero learned parameters. For a matched observation it predicts the exact immutable raw supervision of that endpoint. For the wrong-observation counterfactual it predicts the raw supervision of the cyclic-plus-one source after sorting endpoint identities within the same family. There are no fixed points, cross-family sources, fitted values, thresholds, or calibration.

NumPy accumulates the existing raw binary/raster confusions, linear `n-1` depth quantiles, six fixed ground-distance groups, and float32-epsilon-clamped one-hot raster NLL. The frozen `evaluate_checkpoint_scope` reached through protected-Camera V3 computes each 21-element margin row; the frozen protected-Camera V3 `evaluate_physical_scopes` computes the nine-scope decision. Their margin rows must agree exactly.

## Frozen boundary

Any future execution must first receive an independent exact-source review and a separate exact authorization. It must reserve one previously absent output root before opening governed raw inputs, run CPU-only with one worker and six native thread controls set to one, and publish only immutable canonical `reservation.json`, `access.json`, `result.json`, and `completed.json`, or the three-file failure inventory. Maximum attempts are one and retry is false.

The only governed inputs are the exact V13 audit and manifest, the global pair/endpoint indexes needed to identify the registered role, and the eight checkpoint-selection shard manifests and supervision arrays bound by that manifest. Train/calibration leaves, RGB bytes, checkpoints, models, neural runtimes, G2, navigation, and heldout material are forbidden.

## Interpretation

An all-nine pass establishes only that the literal gate is attainable by this preregistered supervision-backed positive control. It does not qualify a learned checkpoint or justify promotion. A miss keeps `BLOCK_UNPROVEN` and forbids threshold relaxation, data changes, retry, or a successor without a new review and authorization.

This source-only preregistration explicitly provides no execution authority.

## Source-review successor

The immutable V1 review at `docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_independent_review_2026-07-15.json` remains the predecessor BLOCK record. Revised sources require a fresh `independent_review_v2` record containing explicit `findings` and `test_execution`; PASS is invalid while any finding has blocking severity. The review still grants no execution authority.

If result publication succeeds but completion publication or terminal inventory fails, the attempt must publish one immutable no-retry failure record binding every already-published canonical artifact and declaring the exact partial terminal inventory. No artifact may be deleted, rewritten, retried, or promoted.
