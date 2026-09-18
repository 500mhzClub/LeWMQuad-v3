# Direct visual-goal readout baseline component

Status: **FIT AND FIXED EVALUATION COMPLETE**. Training session 46814
(PID 46421) exited 0 after 24 epochs and 4,560 updates in 862.36 seconds.
Encoding took 836.15 seconds; final training normalized MSE is 0.027407.
Checkpoint SHA-256: `c270cffac40b0e917f05a4f83edf105d132b970586290ef610a0afc8077ef74c`.
Evaluation session 9093 exited 0 in 8.76 seconds. On the 20 current-to-goal
pairs, mean position-vector error improved from 7.661 to 1.999 cm and mean
heading error from 9.044 to 1.177 degrees. Arrival classification detected
2 of 9 within-goal states, missed 7, and produced no false arrivals on the
11 outside states. Better pose accuracy does not establish reliable detection.
On reversed pairs, errors were 1.902 cm / 1.373 degrees, with the same
confusion counts; the previous head gave 8.817 cm / 16.974 degrees and three
false arrivals. Results: `go2_direct_visual_goal_readout_evaluation_2026-09-17.json`.

The readout has subsequently been used for observed-image arrival recognition
with the existing world-model planner: both exposed local tasks ended within
tolerance and remained there after detection. See
`go2_dense_visual_arrival_pilot_2026-09-17.md`. A subsequent fixed direct
feedback controller completed 1/2 arrivals versus the planner's retained 2/2:
`go2_direct_visual_feedback_pilot_2026-09-17.md`. Its failed trajectory exposes
a false arrival estimate (1.63 cm estimated versus 5.27 cm actual), despite
zero false positives in the earlier small diagnostic. Neither comparison
isolates a JEPA-training benefit or establishes general arrival reliability.

The earlier 500-ms motion readout is inadequate as an off-the-shelf direct
visual-servo component: its training support ends at 10.52 cm, whereas the
local goals start 27.7 cm away and can lie behind the robot after approach.
This successor uses the existing goal-metric training pairs to cover longer
and reversed displacements without using exposed diagnostic images.

The dataset contains exactly the same 24,294 pairs, 138 training recordings
and 5,552 RGB paths as the learned goal cost, at offsets 100, 200, 500, 1,000,
2,000 and 3,000 ms. Both directional labels are retained, but each base pair
is used only once per epoch: a seeded balanced choice supplies forward versus
reverse ordering. Thus 24 epochs, batch 128 and 4,560 updates match the goal
cost's example/update budget. No extra camera observations are introduced.

Labels are signed planar body-frame goal XY and wrapped world-yaw difference.
Native poses supply targets only. Reversing a pair recomputes translation in
the reversed current frame; it does not merely negate XY. Composition of the
forward and reverse transforms was checked on all pairs, and the squared
normalized label magnitude matches the original scalar goal-cost label in
both directions. Neither check uses transfer data.

Training support across both directions is forward displacement
[-0.5670, +0.5673] m, lateral displacement [-0.3129, +0.2710] m and heading
[-1.3905, +1.3905] rad (about +/-79.67 degrees).

The frozen V-JEPA encoder and 2x2 spatial pooling are unchanged. A shared
1024-to-32 GELU projection feeds concatenated current features and projected
goal-minus-current features into a 32-unit GELU head with three outputs.
Subtracting its identical-pair output enforces zero displacement for identical
images during fitting and inference. This is part of the model definition,
not post-hoc bias correction of the older readout. The model has 426,144
parameters, 128 more than the goal metric (about 0.03%).

The planned loss is MSE in the same 3-cm / 5-degree units as the physical goal
cost. AdamW lr .001, weight decay .0001, gradient clip 1; seed 2026091709;
fixed final epoch only. The data and update budget are matched to the cost
head, but target form differs (signed vector versus scalar distance), and
compute/model size are not matched to the entire world-model planner.
This distinction must remain explicit in scientific claims.

A CPU implementation check confirmed exactly zero output for identical images
and finite, nonzero training gradients on distinct synthetic inputs. It saved
no model and supplies no evidence of prediction or navigation performance.

Fitting used batch-eight encoding, approximately 2.03 GiB of pooled
features in RAM and one small latest resumable checkpoint. Hardware
and competing jobs were checked before launch: GPU idle, 73 GiB available
RAM, about 1004 MiB free output storage; the output is a small readout
checkpoint and metrics, with features in RAM. It ran alone on GPU with
four CPU threads on cores 8-11. The measured duration was 14.37 minutes,
dominated by encoding. The fixed readout alone is not a strong reactive controller; the
component alone does not demonstrate closed-loop performance.

Plan: `go2_direct_visual_goal_readout_plan_2026-09-17.json`.
Model: `lewm/direct_visual_goal_readout_development.py`.
Training: `scripts/train_go2_direct_visual_goal_readout_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_direct_visual_goal_readout_v1_attempt_001`.

The fixed evaluator is now implemented and syntax-checked:
`scripts/evaluate_go2_direct_visual_goal_readout_development.py`. It compares
the new and old readouts using the same GPU float32 features on the same
20 observed current/goal pairs as the preceding diagnostic, plus all reversed
pairs. It records signed displacement/heading errors and goal-detection
confusion counts under common planar targets. Predictions precede target
loading. It also reports the old CPU-versus-GPU prediction discrepancy.
This remains a compound baseline-component change (coverage, direction,
architecture and loss), not an isolated ablation of any one ingredient or
a navigation result. Evaluation session 9093 completed successfully; no fit
or evaluator process remains live.
