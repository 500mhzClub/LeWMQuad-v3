# Matched-budget cross-trajectory goal metric

Status: **FIT, FIXED EVALUATION AND FOUR CONTROL TRIALS COMPLETE**. Training session 10259,
PID 56080, exited 0 after 24 epochs and 4,560 updates in 862.80 seconds.
Encoding took 835.50 seconds; final training log-distance MSE is 0.011974.
The fixed final checkpoint SHA-256 is
`3d040c24de9481ea85ed6ca6eae2b2fca7081790268aedcdd6e9eb6615032f66`.
The prospective local pilot achieved **0/4 final arrivals**, one transient
visit and one contact. See `go2_cross_trajectory_goal_pilot_2026-09-17.md`.
Training loss is not directly comparable to the old loss because the
supervised pairs differ.

The original goal metric fits within-trajectory turn distances but severely
underestimates separation between opposing turns, including on training images.
This intervention changes pair supervision while freezing the encoder and
action-conditioned predictor. It does not repeat an encoder screen or train
a new world model.

Exactly half of the original 24,294 pair positions retain their original pair.
The other 12,147 retain the same first endpoint and draw a second endpoint
from another matching training recording. Every first-endpoint exposure and
the complete 5,552-path image list remain unchanged. Second-endpoint frequencies
change as part of the treatment. Cross pairs are unique and sampled without
looking at target distance or transfer performance, using seed 2026091807.

Matching uses geometry including spawn/world coordinates, procedural seed,
appearance arm/seed, friction, render near plane, visual surface contract and
effective actuator gains. All 138 source recordings have training roles; no
post-contact frame is admitted. There are eight matching environment groups:
six contain six recordings each and two contain 51 each. The existing
72 switch and 18 pulse recordings concentrate in those latter groups; the
intervention preserves the original first-endpoint distribution rather than
claiming balanced environment exposure.

The labels remain squared world-planar XY separation / 3 cm plus squared
wrapped world-yaw separation / 5 degrees, summed. Recomputed old labels match
the saved labels. New target quantiles (0/25/50/75/90/100%) are
0, 0.583, 22.423, 109.529, 270.820 and 1718.246. Cross pairs add relationships
between existing observations; they introduce extra pair labels, not images
or training updates.

Reuse the original fit loop, architecture (426,016 parameters), initialization
seed 2026091707, image order, 24 epochs, batch 128, 4,560 updates, AdamW lr .001,
weight decay .0001 and gradient clipping at 1. The loss is MSE between log1p
predicted and physical squared costs. The final epoch is fixed before fitting;
no selection or tuning uses transfer outcomes. The original within-only model
and all failed navigation recordings remain retained.

Before launch: 73 GiB RAM available, GPU utilization 5%, no competing experiment,
about 682 MiB free on the root volume. One GPU process uses CPU cores 8-11.
Batch-eight frozen encoding shares duplicate images, leaving 3,979 unique
images; approximately 2.03 GiB pooled FP16 features stay in RAM. Only compact
labels, one resumable checkpoint and metrics are written. Expected duration
is about 15 minutes, dominated by encoding; follow measured progress.

The fixed evaluation completed in 15.15 seconds using:
`scripts/evaluate_go2_cross_trajectory_goal_metric_development.py`. It measures
the same training turn-separation probes and three exposed decision states:
the first right-opening failure plus both earlier left near-goal departures.
It compares old/new costs on actual and causally predicted successor images,
physical action rankings and regret. Exact mixed-fit pair membership is
reported for the training probes. Neither in-sample fit nor retrospective
ranking substitutes for prospective control on the full task.

On the two right-opening training layouts, estimated opposing-turn squared
distance increased from 3.3/3.9% to 97.3/99.7% of physical cost. The exact
probe pairs were not sampled into the mixed fit, but their images were
training inputs. This demonstrates correction of a training-representation
distance defect, not independent environment generalisation.

On the exposed right-opening start, actual successor-image costs now prefer
the physically best right arc. Predicted successor costs prefer forward;
the old metric preferred left turn with both actual and predicted images.
Both exposed near-goal diagnostics now prefer hold under prediction.
The four subsequent online control trials nevertheless all fail the fixed
final-arrival criterion. Do not promote this head as reliable navigation.

Plan: `go2_cross_trajectory_goal_metric_plan_2026-09-17.json`.
Training: `scripts/train_go2_cross_trajectory_goal_metric_development.py`.
Output: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_cross_trajectory_goal_metric_v1_attempt_001`.
