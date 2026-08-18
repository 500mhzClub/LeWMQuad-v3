# Claims matrix

| Claim | Status | Evidence / estimator | Effect and interval | Scope and limitation | Permitted wording | Prohibited stronger wording |
|---|---|---|---|---|---|---|
| Two-step rollout improves direct fidelity at H1–H4 | ESTABLISHED | Eight paired seeds; equal-family primary | H2 cosine `+0.0080767` [0.0064285, 0.0097249]; H3/H4 source-audited in number audit | 20 fixed counterfactual states, eight families, 240 branches | “Improved counterfactual latent fidelity under the frozen assay.” | “Generalizes to arbitrary environments.” |
| Two-step improves action-specific retrieval | ESTABLISHED | Branch retrieval, equal-family seed replication | H4 effects are positive in the frozen result; broader retrieval is clearest H3–H4 | Candidate bank and state snapshots are fixed | “Improved selected action-specific retrieval outcomes.” | “Improved planning or reward.” |
| Proprioception amplifies rollout supervision | SUPPORTED_NULL | Eight quadruplets; equal-family interaction | H2 interaction `+0.0006129` [-0.0018563, 0.0030820] | Deployment-valid proprioceptive subset only | “No material proprioception-by-rollout interaction was detected.” | “Proprioception is never useful.” |
| Four-step improves H4 fidelity | POSITIVE_TENDENCY | Eight paired historical controls; equal-family | Cosine `+0.0141863` [0.0110915, 0.0172812]; normalized-error `+0.0247302` [0.0192803, 0.0301801] | Historical one/two-step controls are not sample-matched | “Four-step improves H4 fidelity in this ablation.” | “Four-step is the superior model.” |
| Four-step improves action differentiation | NOT_TESTED | Retrieval effects are exploratory/diagnostic | H4 top-1 `+0.0058594` [-0.0173367, 0.0290554]; pairwise `+0.0037090` [-0.0094083, 0.0168262] | Intervals include zero; H1 regression is present | “No additional action-specific improvement was established.” | “Four-step improves action selection.” |
| Spatial/occupancy retention is preserved | NOT_TESTED | Frozen occupancy probe is a co-outcome | Primary intervals include zero; no numerical non-inferiority margin was frozen | H1 unavailable; H2–H4 probe only | “Occupancy is reported as an inconclusive co-outcome.” | “Reliable occupancy preservation.” |
| Fixed-pooling ViT-L utility scorer qualifies | VALID_NEGATIVE_RESULT | Frozen scorer gates | Safety AUC and latent pairwise gain failed their prespecified thresholds | Utility line is separate from predictive dynamics | “The fixed-pooling scorer failed qualification.” | “The predictor cannot support utility.” |
| ViT-g provides useful scaling | VALID_NEGATIVE_RESULT | Frozen scale ablation | Safety AUC and latent incremental value worsened | Exploratory scale comparison | “No scaling signal was observed.” | “ViT-g is universally worse.” |
| Attentive utility scorer result | TECHNICAL_NON_RESULT | Lawful terminal absent | No scientific metric result | Must not be interpreted scientifically | “Technical non-result; no evidence either way.” | “Attentive architecture failed scientifically.” |
| Non-learned ViT-L shadow/native memory interface | VALID_NEGATIVE_RESULT | True-target upper-bound assays | Both failed their registered gates | Place and predictive spaces are distinct | “The tested non-learned interfaces were not viable.” | “A learned bridge cannot work.” |
| Self-supervised place head | UNRESOLVED | Training receipt only | Seed `2026081801`; panel unavailable | No held-out retrieval was performed | “Trained but unevaluated because the frozen panel was unavailable.” | “The place head works or fails.” |

All primary estimates use equal-family aggregation; corpus-weighted values are
secondary. Seed intervals represent training-seed uncertainty only and do not
remove fixed-state environmental uncertainty. Infrastructure and technical
non-results are not scientific evidence.
