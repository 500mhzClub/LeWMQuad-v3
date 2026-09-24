# Paper outline

## Working titles

1. *Action-Specific Counterfactual Futures from Autoregressive JEPA Rollouts*
2. *Rollout Supervision for Action-Conditioned Quadruped World Models*
3. *Predictive Fidelity and Action Specificity in Counterfactual Latent Dynamics*

## Abstract

Embodied world models must predict not only plausible futures but futures that
distinguish competing actions from the same state. We study an
action-conditioned JEPA predictor for quadruped navigation using deterministic
counterfactual branches: twelve post-slew action sequences are applied from a
shared simulator snapshot and compared with frozen future visual targets.
Across eight paired training seeds, two-step autoregressive supervision
improves direct future-latent fidelity from H1 through H4 and improves
action-specific branch retrieval, with the clearest retrieval gains at H3–H4.
At H2, the equal-family rollout-minus-one-step cosine effect is 0.00808
(95% t interval 0.00643–0.00972). A deployment-valid proprioceptive subset
does not materially amplify the rollout effect: the H2 interaction is 0.00061
(−0.00186–0.00308). A four-step ablation improves H4 fidelity relative to
two-step training, but regresses H1 fidelity and does not establish additional
action differentiation. Occupancy decoding is reported only as an
inconclusive co-outcome. Fixed-pooling and ViT-g utility-scoring lines do not
qualify, and tested non-learned memory interfaces fail their true-target gates.
The trained place head could not be evaluated because its frozen held-out panel
was unavailable. Thus the evidence supports predictive representation gains,
not planning utility, selected-action reward, or closed-loop navigation.

## Introduction

Motivate the distinction between low prediction error and action-specific
counterfactual fidelity; explain why matched branches from one state control
for scene and history; state the prediction–planning gap; list contributions:
the deterministic branch assay, horizon-resolved fidelity/retrieval analysis,
proprioception interaction test, and explicit separation of predictive, place,
and utility representations.

## Related work

Cover (with literature placeholders): JEPA/self-supervised visual prediction;
latent video/world models; action-conditioned and rollout training; model-based
control and counterfactual evaluation; topological visual memory and place
recognition; contrastive representation learning; and uncertainty/occupancy
prediction. Citation selection remains a writing task, not a new experiment.

## Method

Describe the simulator/robot, frozen V-JEPA ViT-L target encoder, RGB and
control-history inputs, one-step and two-step objectives, deployment-valid
proprioception, four-step ablation, deterministic snapshots and twelve-action
candidate bank, true-future encoding, fidelity/retrieval metrics, occupancy
probe, and equal-family/corpus-weighted seed estimators.

## Experiments and results

1. Loss-scaling attribution check.
2. Eight-seed one-step versus two-step factorial.
3. H1–H4 factual rollout evaluation.
4. Counterfactual branch qualification.
5. Proprioception interaction.
6. Four-step horizon-depth ablation.
7. Spatial co-outcome.
8. Utility-scorer and memory-interface limitations.

Organize results by scientific question rather than infrastructure chronology.

## Discussion

Discuss horizon-growing gains, fidelity versus action specificity, possible
predictive smoothing, weak/inconclusive occupancy retention, absent
proprioception interaction, and the separation of predictive, place, and
utility representations.

## Limitations

The assay covers twenty fixed counterfactual states with unequal family
counts; environment generalization is unresolved; H1/H2 candidate-prefix
degeneracy limits interpretation; four-step controls are historical and not
sample-matched; the occupancy probe has no frozen non-inferiority margin; no
planning, closed-loop, or physical Go2 endpoint was measured; and the place
head remains unevaluated.

## Conclusion

Two-step action-conditioned rollout supervision is supported as a predictive
representation improvement. The evidence does not establish planning ability;
future work must qualify a place representation and evaluator independently.
