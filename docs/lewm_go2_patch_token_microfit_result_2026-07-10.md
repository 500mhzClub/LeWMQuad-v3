# Go2 patch/token micro-fit result

Date: 2026-07-10

Status: authoritative negative; patch-7 full training is not licensed

This note records the completed train-role-only diagnostic without modifying
the source-hashed protocol or execution-contract files that were bound into
the immutable panel and result artifacts. It is not a G2 evaluation.

## Immutable artifacts

- panel:
  `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`
- panel file SHA-256:
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`
- panel content SHA-256:
  `f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f`
- seed-20260710 result:
  `.generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json`
- result file SHA-256:
  `6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c`
- result content SHA-256:
  `32d848d3df68e670ddb4cc24436981f62a1aa5562b89e6d6719ecb113f66b749`

The hardened finalizer's complete single-result validator accepts the result
and independently recomputes its per-family fit gates, learning-curve flags,
consecutive-pass counts, milestones, terminal decision, support thresholds,
artifact counts, access reconciliation, and cross-arm classification.

The authoritative run used batch size 4, 2,000 faithful updates at learning
rate `2e-4`, and a mandatory 3,000-update ceiling restart at learning rate
`1e-3` because both faithful arms failed. Evaluation occurred every 100
updates. The panel contained 160 transitions and 320 frames in each of fit,
same-scene holdout, and cross-scene holdout. The 480 rows and 960 endpoint
images were globally disjoint.

## Decision

The recomputed classification is
`both_arms_fail_patch7_tokenization_bundle_insufficient`.

- patch14/8x8 never passed the aggregate fit gate or any terminal all-family
  gate;
- patch7/16x16 never passed the aggregate fit gate or any terminal all-family
  gate;
- neither arm was near the precommitted gate;
- a second seed is not required by the protocol;
- `patch7_full_train_candidate_licensed` is false.

No checkpoint-selection, probability-calibration, or G2 image/label bytes were
opened and no model output was produced for those roles. The result records
960 train images and 45 train label shards, with all non-train access counters
at zero.

## Aggregate fit evidence

| Stage and arm | Balanced NLL | UK / FO balanced accuracy | U / F / O recall | FREE recall 1-2 / 2-3 / 3+ m | Cross / same wrong-view NLL delta |
|---|---:|---:|---:|---:|---:|
| faithful patch14 | 0.223 | 0.934 / 0.868 | 0.903 / 0.915 / 0.627 | 0.906 / 0.904 / 0.921 | +1.416 / +0.544 |
| faithful patch7 | 0.252 | 0.929 / 0.864 | 0.886 / 0.921 / 0.672 | 0.904 / 0.904 / 0.932 | +1.208 / +0.420 |
| ceiling patch14 | 0.706 | 0.789 / 0.734 | 0.916 / 0.617 / 0.268 | 0.492 / 0.593 / 0.664 | +0.594 / +0.001 |
| ceiling patch7 | 0.323 | 0.890 / 0.819 | 0.829 / 0.878 / 0.519 | 0.772 / 0.834 / 0.925 | +0.630 / +0.232 |

The gate required NLL at most 0.03, both balanced accuracies at least 0.99,
every class recall at least 0.98, FREE recall at least 0.95 in every gated
distance bin, and both wrong-view deltas at least 0.25. The faithful arms pass
only the RGB-control terms. The ceiling optimizer is unstable and worse.

The family signature localizes the defect:

- enclosed mazes retain UNKNOWN but miss visible free corridors, with faithful
  far-FREE recall only 0.239-0.434;
- open and rough scenes recover FREE strongly but miss occupied surfaces and
  visibility boundaries;
- correct RGB beats both cross-scene and same-scene wrong views by large
  margins, so the model is image- and view-grounded;
- the best faithful NLL values, 0.200 at step 1,700 for patch14 and 0.246 at
  step 1,800 for patch7, remain far from the gate.

This rejects an RGB-insensitive head, gross axis misalignment, simple
undertraining, and token count alone as explanations. The remaining failure is
the decoder's representation of precise first-surface depth, occupied
boundaries, and unknown-behind-occluder structure. Both arms used the same
effective projective blur: one 14-pixel token for patch14 and two 7-pixel
tokens for patch7.

## Next admissible experiment

The next intervention is a geometry-structured categorical radial head. It
keeps the patch7 image encoder, physical labels, frozen train-only panels, and
fit gates, but preserves bearing/range evidence and performs a deterministic
polar-to-Cartesian lift. Before GPU training it must pass an exact geometry
round trip and a one-frame overfit ladder. A full G2 candidate remains blocked
until this or a later train-only architecture passes the aggregate and all five
family gates without non-train contact.

The current candidates are supervised perception diagnostics: their JEPA,
equivariance, action-contrast, and variance weights are zero. They are not
evidence that the predictive JEPA component works; predictive objectives are
restored and gated only after traversability perception is expressively sound.
