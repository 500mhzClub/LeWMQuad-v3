# Results draft

## Direct counterfactual fidelity

The equal-family eight-seed analysis compares RGB one-step and RGB two-step
rollout models at epoch 21 on the frozen counterfactual design. Rollout
supervision improves changed-token cosine at every evaluated horizon H1–H4;
the H2 mean effect is 0.0080767 (95% t interval 0.0064285–0.0097249). The
same analysis reports lower normalized prediction error at H1–H4. These are
the primary predictive endpoints; corpus-weighted estimates are secondary.
The complete seed vectors and authoritative paths are in the number audit.

## Action specificity

The branch-retrieval analysis uses the same twelve candidates from each frozen
state. Top-1 retrieval improves at H2–H4, while MRR, top-3, pairwise
discrimination, and margin diagnostics are most consistently favorable at
H3–H4. The result is evidence that rollout training retains more action
identity than a one-step objective, not evidence of better planning.

## Proprioception interaction

Adding the deployment-valid proprioceptive subset does not materially amplify
the rollout effect. The equal-family H2 interaction is 0.0006129 with a
95% interval of −0.0018563 to 0.0030820. This is a supported null within the
registered subset, not a claim that proprioception is generally unhelpful.

## Spatial retention

Predicted latents retain some decodable occupancy information at H2–H4, but
the primary equal-family occupancy rollout intervals include zero and no
non-inferiority margin was frozen. Occupancy is therefore an inconclusive
co-outcome, not a spatial-preservation claim.

## Four-step trade-off

The four-step arm improves H4 changed-token cosine by 0.0141863 and normalized
error reduction by 0.0247302 relative to two-step training. However, H1
fidelity materially regresses, and H4 top-1 and pairwise effects have intervals
that include zero. Four-step training is consequently retained as a diagnostic
ablation rather than the selected primary model.

## Heterogeneity and limits

Family-level values are reported in the source tables. Unequal family counts,
the fixed twenty-state environment panel, and historical controls for the
four-step comparison limit generalization. The fixed-pooling ViT-L scorer
failed its safety-AUC and incremental-pairwise gates; ViT-g showed no scaling
signal. The attentive scorer was a technical non-result. Non-learned ViT-L
and native-LeWM memory interfaces failed true-target gates. The self-supervised
place head was trained once but could not be evaluated because its required
panel was unavailable.

## Conclusion

The supported result is predictive: two-step autoregressive supervision
improves counterfactual fidelity and selected action specificity. No result
supports utility scoring, selected-candidate reward, planning, navigation, or
physical transfer.
