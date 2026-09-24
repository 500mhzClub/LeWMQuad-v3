# Task-relevant dense predictor continuation

Status: **FIT, FIXED EVALUATION AND LIVE COMPARISON COMPLETE; NO PROMOTION**. Training tool session
70471 (PID 40768) exited 0 after all eight epochs and 1,760 updates per arm,
in 2,696.8 seconds (44.95 minutes). Encoding took 838.1 seconds and used
8.13 GiB host RAM; peak training GPU allocation was 6.86 GiB. The final
checkpoints and result JSON are saved. Evaluation session 66935 exited 0 in 42.96 seconds. It does not support
promoting the auxiliary objective: dense transfer error worsens, goal-space
retrieval falls, and all four diagnosed states select right turn. Dense
continuation improves the late hold-versus-turn comparisons but one full
six-action choice is right arc. All four intended prospective task cells are complete, including two fresh
executions after preserved storage interruptions. Neither continuation improves
on the parent: final XY means 8.112 cm dense, 3.888 cm auxiliary, versus
3.160 cm parent; all have 0/2 final arrivals. See
`go2_dense_task_goal_pilot_2026-09-17.md` and
`go2_dense_task_predictor_evaluation_2026-09-17.md`.

| Final epoch training mean | Dense action | Geometry-aware action | Dense blind | Geometry-aware blind |
|---|---:|---:|---:|---:|
| Dense L1 | 0.209610 | 0.378579 | 0.255342 | 0.406215 |
| Log goal-embedding MSE | 0.282613 | 0.211626 | 0.467405 | 0.379305 |

The auxiliary action arm trades 80.6% higher dense L1 for 25.1% lower
training goal-embedding error. These are online minibatch averages, not
final-checkpoint transfer scores. No training benefit is yet a navigation claim.

The learned goal cost improved final position error from 118.71 cm to 3.16 cm,
but both trials later made an unnecessary turn. Matched hold counterfactuals
showed that actual image costs correctly prefer holding; imperfect predicted
features reverse that ordering. This motivates a task-relevant predictor loss
rather than fitting a stopping threshold on those two exposed states.

## Fixed experiment

Four arms continue the existing action and no-future-action predictors:
`dense_action`, `metric_action`, `dense_no_future_action`, and
`metric_no_future_action`. Each dense/metric pair restores the same parent
weights and AdamW state. All arms use the same seeded minibatch order and
8 additional epochs (1,760 updates per arm), batch 16, lr 0.0003, weight decay
0.01 and gradient clip 1. The frozen V-JEPA encoder and frozen learned goal
metric are unchanged. Future action inputs alone are zeroed for blind arms;
past visual observations and applied controls are unchanged.

The control retains dense normalized-token L1. The intervention adds
`lambda * mean(log1p(mean((goal_embed(pred)-goal_embed(target))**2)))`.
The coefficient is calculated once from 128 seeded training examples as the
initial action model's dense-loss/auxiliary-loss ratio, then shared and fixed
for both auxiliary arms. The resulting fixed coefficient is 0.773472095, from initial dense L1
0.225157171 and auxiliary loss 0.291099282. There is no transfer-based
coefficient or checkpoint selection. The final eighth extra epoch is the sole evaluated checkpoint.
The goal metric contributes previously learned physical supervision; this is
not an isolated JEPA representation-training comparison.

Inputs are exactly the prior 3,518 admitted training sequences from 138
recordings and 5,552 image paths. No live-pilot, counterfactual or transfer
images enter fitting. Features use batch-eight float32 encoder inference,
normalized FP16 storage in host RAM and identical-image reuse. The first image
is checked against batch-one inference; this changes extraction throughput,
not the specified representation. No dense feature files are written.

## Resources and verification

The GPU was free; 72 GiB RAM was available, with 1.9 GiB free on the output
volume and 1.2 GiB on the repository volume. Four interleaved arms share one
GPU process and four CPU cores (8-11). A real training example repeated to
batch 16 passed the new forward/backward path in 0.55 seconds with 6.12 GB
peak GPU allocation. Predictor gradients were finite and the goal metric
received no parameter gradients. This is an implementation check, not an
experimental result. No parameters from that check were saved or used.

The plan checks capacity for four latest resumable checkpoints plus one
replacement file and a 512 MiB reserve. Checkpoints replace only their own
latest version, preserving the original model families and failed experiments.
Encoding throughput observed so far is about 4.75 unique images/s; full fit
was estimated at 45-50 minutes, subject to measured epoch time. Neither a
background process nor a stale state file alone establishes progress: poll
session 70471 and inspect the live PID/log.

## Evaluation protocol (now executed)

`scripts/evaluate_go2_dense_task_predictor_development.py` compares all four
new models with both unchanged parents and persistence. It evaluates the
fixed 36 native pulse branches (18 training, 18 exposed transfer), recording
dense and goal-embedding fidelity, action retrieval and centered action
response. Four original near-goal/late-turn states are evaluated separately;
late-state physical regrets cover only hold versus the recorded turn. All
causal forecasts precede loading future scoring images. Blind forecasts are
computed once and broadcast, preserving exact ties. Dense caches remain in RAM.

`scripts/run_go2_dense_task_goal_pilot_development.py` prepares four prospective
action-conditioned trials after fitting/evaluation complete: dense versus
metric continuation on the same two local tasks, with counterbalanced order.
The fixed encoder, learned cost, supplied goals, candidate commands, 500-ms
execution blocks, budget, seed and native stops are retained. No stopping rule
is added. The existing blind trajectories can be reused explicitly as reference:
all candidates have exactly the same blind forecast, so uniform tie breaking
is independent of the fitted predictor weights. They are not new replicates.
The initial storage estimate proved insufficient. Two interrupted attempts
were preserved and reexecuted separately after retiring unused diagnostic
depth; all four task cells are now complete. See the live-result note.

The prospective reader is `scripts/read_go2_dense_task_goal_pilot_development.py`.
Evaluation and live execution completed. The reader verified identical
initial RGB and exact native prefixes for the two storage replacements.
Training and live figures are `go2_dense_task_predictor_training_2026-09-17.png`
and `go2_dense_task_goal_pilot_2026-09-17.png`.

Full independent-maze navigation, a strong reactive/non-predictive baseline,
JEPA representation-training attribution, dense-model integration with
exploration/memory/backtracking, real-time sensing and hardware validation
remain incomplete. The active goal has not been narrowed or marked complete.

Plan: `go2_dense_task_predictor_plan_2026-09-17.json`.
Training source: `scripts/train_go2_dense_task_predictor_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_dense_task_predictor_v1_attempt_001`.

An independent CPU-only baseline-component check completed while encoding
continued: `go2_visual_servo_readout_diagnostic_2026-09-17.md`. The existing
500-ms motion probe substantially underestimates distant/reversed goal
displacement; its training motion support ends at 10.52 cm. It is not promoted
as a strong reactive baseline. No running experiment was modified.

First continuation epoch completed in 231.5 seconds, with 220 updates per arm
and all four resumable checkpoints saved. Dense-only action L1 averaged
0.223978. Geometry-aware action L1/log-embedding error averaged
0.426188 / 1.175961, worse than its initial calibration values; these are
training averages across changing weights, not endpoint or transfer results.
The fixed eight-epoch run completed without outcome-dependent changes.
A direct visual-goal baseline dataset/model was prepared on spare CPU capacity
while this fit ran; its separate fit is now running as session 46814. See
`go2_direct_visual_goal_readout_2026-09-17.md`.
