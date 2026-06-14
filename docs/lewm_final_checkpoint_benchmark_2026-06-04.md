# LeWM Final Checkpoint Benchmark Rerun - 2026-06-04

This note reruns the final LeWM checkpoint through the benchmark/probe suite
that motivated the lower-SIGReg/source-sampling ablations.

Checkpoint:

- `models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9.pt`
- saved: 2026-06-04 16:21 local time
- size: 207 MB
- final training row: epoch 9 complete

## Executive Conclusion

The final checkpoint does **not** remove the reason for the ablations. It
strengthens the same diagnosis seen at `e9_b050000`:

- The model has a useful place-recognition signal.
- The latent space is still poor as a metric map.
- A learned reachability head is still near majority baseline cross-scene.
- Rollout prediction is useful for 1-2 macro-steps, then loses to persistence
  from horizon 3 onward.
- The yaw-jitter closed-loop local beacon benchmark is now discriminating and
  the final checkpoint does not pass it.
- The stress benchmark remains oracle-limited even after the block-budget fix,
  so it should not carry the main conclusion.

Lay summary: the model often knows "I have seen this kind of view/place before,"
but the embedding does not reliably say "this place is close to that place" or
"this action sequence moves me toward the goal." That is exactly the failure
mode the `lambda` and source-sampling ablations are meant to test.

## Why Rerun The Final Checkpoint

Earlier decisions used the partial epoch-9 checkpoint
`lewm_seq4_e9_b050000.pt`. The full epoch-9 checkpoint finished shortly after,
so we reran the decision suite to check whether the final training tail changed
the diagnosis.

Lay summary: before spending GPU time on ablations, check whether the finished
model already fixed the issue.

## Final Training Metrics

From the final `metrics.jsonl` row:

| metric | value |
|---|---:|
| epoch_complete | true |
| train_loss | 0.0940 |
| train_pred | 0.0274 |
| train_sig | 0.7402 |
| train_std | 1.0030 |
| eval_loss | 3.3231 |
| eval_pred | 0.0428 |
| eval_rollout_pred | 0.1632 |
| eval_sig | 36.4570 |
| eval_std | 0.7565 |
| eval_action_shuffle_delta | 0.0377 |
| eval_action_zero_delta | 0.0451 |

Interpretation: the model remains action-sensitive, but eval embeddings are
under-dispersed (`eval_std` 0.7565 vs target near 1.0) and rollout error remains
large relative to one-step prediction.

Lay summary: training finished cleanly and the model reacts to actions, but its
held-out representation is still compressed and its longer predictions are
still shaky.

## Artifacts

| check | artifact | status |
|---|---|---|
| A2 latent aliasing | `models/checkpoints_textured_v03_full_20260531/sweep_seq4/latent_aliasing_e9_final_testid.json` | complete |
| A3 reachability/history | `models/checkpoints_textured_v03_full_20260531/sweep_seq4/reachability_a3_e9_final_testid.json` | complete |
| rollout horizons | `models/checkpoints_textured_v03_full_20260531/sweep_seq4/planning_gate_lewm_seq4_e9_final.json` | complete |
| receding-MPC proxy | `models/checkpoints_textured_v03_full_20260531/sweep_seq4/receding_mpc_proxy_lewm_seq4_e9_final.json` | complete |
| yaw-jitter visible beacon | `models/checkpoints_textured_v03_full_20260531/sweep_seq4/closed_loop_mpc_visible_beacon_e9_final_yawjitter0p7_testid_open_obstacle_field.json` | complete |
| fixed-budget stress | `models/checkpoints_textured_v03_full_20260531/sweep_seq4/closed_loop_mpc_stress_e9_final_budget40_testid_visual_sensor_stress.json` | complete |

## A2 Latent Aliasing

Configuration: `test_id`, 32 held-out scenes, final checkpoint.

| measurement | value |
|---|---:|
| projected Spearman rho, median | 0.0288 |
| projected Spearman rho, mean | 0.0311 |
| yaw-matched projected Spearman rho, median | 0.1938 |
| raw Spearman rho, median | 0.0684 |
| nearest-10%-latent but graph-far, median | 0.2205 |

Projected latent distance by graph-distance bucket:

| graph distance | norm. latent median |
|---|---:|
| 0-1 same/adjacent | 0.942 |
| 2-3 | 1.005 |
| 4-7 | 1.008 |
| 8-15 | 1.001 |
| 16+ | 0.990 |

Verdict: insufficient (`rho < 0.40`). The result is effectively unchanged from
the partial e9 checkpoint.

Lay summary: points that are next to each other and points that are many cells
apart are almost the same distance in latent space. The embedding is good at
being evenly spread out, but not at behaving like a map.

## A3 Reachability And History

Configuration: train heads on `train`, evaluate on `test_id`, final checkpoint.

Place recognition:

| measurement | projected | raw |
|---|---:|---:|
| retrieval@1 median | 0.3958 | 0.4333 |
| retrieval@5 median | 0.6188 | 0.6333 |
| lift@1 median | 20.97x | 22.72x |
| recognition localization R2 median | 0.2283 | 0.2254 |
| metric localization R2 median | -0.0887 | -0.1959 |

Reachability bucket head:

| head | train top1 | eval top1 | majority baseline | eval gain |
|---|---:|---:|---:|---:|
| linear abs-diff | 0.3596 | 0.2573 | 0.2464 | +0.0109 |
| MLP concat | 0.9414 | 0.2647 | 0.2464 | +0.0183 |

History disambiguation:

| horizon | single-frame AUC median | history-window AUC median |
|---|---:|---:|
| H4 | 0.7607 | 0.7887 |
| H8 | 0.7698 | 0.7802 |

Verdict: insufficient; cross-scene reachability is only slightly above the
majority baseline and far below the planned +15pp bar.

Lay summary: the model can recognize a place much better than chance, but even a
small trained classifier cannot reliably say whether two places are near or far
in a new scene. More history helps a little, but it does not turn the embedding
into a map.

## Rollout Horizon Gate

Configuration: held-out eval sessions, uncapped population, 816 sessions,
144,106 valid windows, 256 evaluated samples, CPU, single-process dataloader.
The first attempt used `--num-workers 2` and failed in the sandbox's
multiprocessing socket setup; the completed rerun used `--num-workers 0` with
the same checkpoint, data roots, holdout, horizons, and scoring logic.

| horizon | seconds | rollout point MSE | persistence point MSE | rollout / persistence | shuffled - rollout | zero - rollout |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.0375 | 0.0707 | 0.530 | +0.0558 | +0.0312 |
| 2 | 1.0 | 0.1228 | 0.1639 | 0.749 | +0.0800 | +0.0577 |
| 3 | 1.5 | 0.2740 | 0.2500 | 1.096 | +0.0833 | +0.0566 |
| 5 | 2.5 | 0.4865 | 0.4758 | 1.023 | +0.0835 | +0.0605 |
| 8 | 4.0 | 0.7550 | 0.7106 | 1.063 | +0.0642 | +0.0333 |
| 10 | 5.0 | 0.8665 | 0.7913 | 1.095 | +0.0657 | +0.0167 |
| 16 | 8.0 | 1.1761 | 0.9932 | 1.184 | +0.0274 | +0.0245 |
| 20 | 10.0 | 1.2656 | 1.0690 | 1.184 | +0.0140 | +0.0301 |

Interpretation: the final checkpoint has a useful short-horizon dynamics signal
at 1-2 macro-steps, but it loses to persistence by horizon 3 and stays worse
than persistence through 20 macro-steps. Positive shuffled/zero margins show the
model still uses the action inputs, but the multi-step rollout is not stable
enough to serve as the main planning cost.

Lay summary: for the next half-second to one second, the model imagines the
future better than simply saying "nothing changes." After about 1.5 seconds,
the cheap "nothing changes" guess is closer to reality, even though the model
still reacts to actions. That is a planning-horizon problem, not just a missing
action-sensitivity problem.

## Receding-MPC Proxy

Configuration: held-out eval sessions, horizons 1/2/3, 4 replan steps,
64 sampled sequences, 768 evaluated decisions, CPU.

| horizon | recorded MSE | persistence MSE | recorded / persistence | top1 among recorded/zero/shuffled | win vs zero | win vs shuffled |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.1781 | 0.0880 | 2.024 | 0.652 | 0.730 | 0.723 |
| 2 | 0.2822 | 0.2018 | 1.398 | 0.621 | 0.750 | 0.727 |
| 3 | 0.3777 | 0.3086 | 1.224 | 0.527 | 0.680 | 0.633 |

Interpretation: recorded actions beat zero/shuffled actions most of the time,
so the predictor is action-conditional. But persistence still has lower terminal
MSE at every horizon, so the model's predicted motion is not yet better than
"assume the latent stays where it is" for local planning cost.

Lay summary: when comparing possible action sequences, the model can often tell
the real action sequence from fake ones. But the predicted future still drifts
enough that simply assuming "no change" is closer to the true future latent.

## Closed-Loop Visible Beacon With Yaw Jitter

Configuration: `test_id/open_obstacle_field`, 10 scenes, one trial per scene,
visible beacon, `0.7 rad` yaw jitter, horizon 2, max 12 blocks.

| policy | success | mean progress m | mean final dist m | path efficiency |
|---|---:|---:|---:|---:|
| bearing | 10/10 | 0.902 | 0.298 | 0.890 |
| lewm | 0/10 | 0.347 | 0.853 | 0.497 |
| random | 0/10 | 0.381 | 0.819 | 0.472 |
| hold | 0/10 | 0.000 | 1.200 | 0.000 |

LeWM primitive counts: `hold 32`, `yaw_right 22`, `forward_fast 19`,
`yaw_left 17`, `arc_left 14`, `arc_right 11`, `forward_medium 5`.

Verdict: this fixed beacon benchmark is discriminating. The privileged bearing
baseline solves it; the final LeWM checkpoint does not.

Lay summary: when the robot starts pointed the wrong way, the simple bearing
controller turns and reaches the beacon. LeWM moves somewhat, but not in a
reliable enough direction; it often ends up turning or stopping instead of
closing the last meter.

## Fixed-Budget Stress Benchmark

Configuration: `test_id/visual_sensor_stress`, 7 scenes, one trial per scene,
landmark task, horizon 2, max 40 blocks.

| policy | success | mean progress m | mean final dist m | path efficiency |
|---|---:|---:|---:|---:|
| bearing | 0/7 | 0.477 | 3.840 | 0.616 |
| random | 0/7 | 0.034 | 4.283 | -0.235 |
| hold | 0/7 | 0.000 | 4.317 | 0.000 |
| lewm | 0/7 | -0.054 | 4.371 | -0.131 |

LeWM primitive counts: `hold 160`, `arc_left 40`, `backward 35`,
`yaw_left 23`, `yaw_right 21`, `arc_right 1`.

Verdict: still not a valid primary discriminator because the privileged bearing
baseline remains 0/7. It is useful as a stress symptom only: LeWM is mostly
stuck, but the benchmark still needs redesign before it can support a clean
claim.

Lay summary: this course is still too hard or misconfigured for the current
baselines. Since even the controller that knows the goal direction cannot finish,
we should not use this result alone to judge LeWM. It does show the same
"stall/hold" behavior.

## Longer Time-Span Prediction Question

Question: would LeWM benefit from predicting longer time spans directly, e.g.
training/evaluating targets around later frames such as 10/15/20 rather than
only adjacent `stride=5` macro-steps?

Short answer: probably yes as a follow-up objective ablation, but it should not
replace the current lower-SIGReg/source-sampling plan.

Current setup:

- Training uses `max_seq_len=4` and `stride=5`, which corresponds to about
  `0.5s` per macro-step.
- The training loss directly supervises adjacent macro-step predictions inside
  that short window.
- Long-horizon planning uses autoregressive rollout: the predictor feeds its
  own previous prediction back in repeatedly.
- The final checkpoint beats persistence at horizons 1-2, but loses to
  persistence from horizon 3 onward.

Interpretation: the current model has a short-horizon dynamics signal, but
rollout errors compound after about 1.5 seconds. Direct multi-horizon
supervision could reduce this compounding error and make a short-term planner
more useful over multiple seconds.

The safer version is not to train only on sparse long jumps. Use a temporal
pyramid / multi-step loss that keeps the near-term target and adds longer
targets:

- keep `h=1` so the model does not lose immediate control sensitivity;
- add `h=2,3,5` first because the current failure starts at horizon 3;
- optionally add `h=8,10` after the short multi-step objective is stable;
- downweight longer horizons so noisy far-future targets do not dominate the
  local-control signal.

Why not do this first: A2/A3 show that the latent space is not metric. If latent
distances do not behave like a map, better long-horizon prediction alone may
still give the planner a poor cost surface. The lambda/source ablations target
that representation problem directly. A temporal-objective ablation should come
after the scaled representation ablation shortlist, or be run as a small
follow-up cell on the best low-lambda/source setting.

Lay summary: yes, teaching the model to predict "where will I be in a few
seconds?" could help planning. But if the model's internal map still cannot tell
near from far, a longer prediction is like forecasting movement on a distorted
map. First fix the map geometry, then add longer-horizon forecasting to make the
planner useful over several seconds.

## Ablation Decision

The ablation remains warranted.

Recommended order:

1. Scaled baseline at current `lambda=0.09`, uniform sampling.
2. Scaled lambda sweep (`0.03`, `0.01`, optionally `0.003`).
3. Sampling-only arm using source-aware filtering/weighting.
4. Combined low-lambda plus source-aware sampling.

Lay summary: first prove the small experiment reproduces the problem, then turn
one knob at a time, then try both knobs together. That gives a clean paper-grade
answer instead of a confusing "changed everything at once" result.
