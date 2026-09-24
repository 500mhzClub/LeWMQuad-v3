# Temporal visual history in the anchored predictor

**Complete:** both fits, the fixed branch assay and all 2,404 recorded visual
forecast windows finished with process exit zero. Initial parameters,
normalization and persistence training errors exactly match the preceding
mixed-history experiment. The recorded evaluation took 59.12 seconds.

## Results

All errors below use the same frozen visual target space, not physical units.

| Predictor | Transfer branch 800-ms MSE | Recorded 700-ms MSE |
| --- | ---: | ---: |
| Visual history, with future actions | 0.047586 | 0.026704 |
| Visual history, no future actions | 0.082411 | 0.026547 |
| Constant visual velocity | 0.190285 | 0.051132 |
| Persistence | 0.003251 | 0.027610 |
| Prior mixed history, with future actions | 0.036537 | 0.030777 |
| Prior mixed history, no future actions | 0.047465 | 0.031301 |
| Original visual-target JEPA | 0.092360 | 0.089297 |

Visual history improves recorded prediction relative to mixed history, and the
action model beats persistence by 3.3% in pooled 700-ms error. However, its
no-action counterpart is slightly better still. Both visual-history models
beat persistence in recordings 2 and 3 and lose in recordings 1 and 4. This is
not a consistent across-recording advantage or evidence that action-conditioned
planning has become useful.

On the fixed transfer branches, action retrieval improves from 6/18 for the
mixed-history action model to 8/18. Centered action-effect error is 0.850 times
the no-effect baseline (previously 0.838), and full error worsens. The visual
history no-action and constant-velocity baselines retrieve 6/18. Persistence
still decisively wins full branch prediction. Do not elevate the small
retrieval increase over those contrary results.

The new action model's weighted training MSE is 0.032538, versus 0.036785 without
future actions and 0.051314 for persistence. The preceding mixed-history models
fit training better (0.025136/0.026202). The differing training, short-pulse and
recorded-navigation results point to substantial motion-distribution effects.
All horizons/actions and both failed returns remain in the recorded result.
Constant-velocity extrapolation beats persistence at 100–300 ms in the recorded
population, but overshoots badly at longer horizons. No damping was fitted.

Branch result: `docs/go2_visual_history_dynamics_result_2026-09-17.json`.
Recorded result: `docs/go2_visual_history_navigation_result_2026-09-17.json`.

## Training coverage and next experiment

A post-hoc count of the unchanged training schedule found 3,831 draws with
four nonzero requested commands after the first three ticks. Sustained action
data exists. However, the explicitly verified identical-history branch
departures contribute only 90 of 7,200 draws (1.25%), and every one of those
branches executes just one nonzero 100-ms action tick. This distinction matters:
ordinary sustained trajectories are not matched alternative futures from the
same causal history. It does not prove that sparse branching caused the failure.

Before another architecture change, collect the matched four-tick (400-ms)
branches used by the ordinary navigation planner, preserving the same three
committed zero ticks. Keep the existing geometry roles and common pre-branch
histories for a controlled action-duration comparison. The existing collector
can retain its 25 command ticks by replacing one pulse plus eight drain ticks
with four action ticks plus five drain ticks. This collection has not been
implemented or launched here. Preserve the short-pulse results as a separate
regime; do not replace them with the easier/newer population.

Coverage record: `docs/go2_visual_dynamics_training_command_coverage_2026-09-17.json`.
The coverage aggregation's initial read-only schema error and correction are
recorded there; no fit or evaluation was restarted or changed.

## Controlled comparison

The preceding anchored predictor reduced error but still lost to visual
persistence. Its current state was visual, while temporal context came from
mixed online RGB/body/control embeddings. This comparison changes only that
history source: encode each of the four past RGB observations with the same
frozen EMA visual encoder used for current and future visual states. The new
predictors do not consume past body/control values. The original visual-JEPA
model remains unchanged as a separately evaluated baseline.

Keep the common frozen target, current visual anchor, 32-dimensional history
GRU, 256-unit transition hidden layer, initialization, normalization, loss,
optimizer and exact training schedule. Fit both action-conditioned and
no-future-action predictors for 1,200 updates on the same 4,694 training
contexts/7,200 draws. Both initially predict exact persistence. Target and
training normalization are unchanged, so their initial weights and weighted
persistence training errors must match the preceding experiment exactly.

Evaluate the same fixed branch population and all 2,404 recorded navigation
windows, including failed returns. Add a prespecified unfitted baseline:
current visual state plus horizon times the difference of the last two visual
states, without clipping or fitted damping. The four observations are spaced
100 ms apart. This constant-visual-velocity baseline uses the same frozen
representation and needs no future commands or sensor observations.

Primary diagnostics remain 800-ms branch action specificity and 700-ms visual
error on recorded executions. Compare all predictions in the same frozen
visual target space. This tests temporal information and action conditioning,
not a JEPA-versus-supervised representation-training advantage. These are
exposed development populations and one training seed; there is no model
promotion, new motion readout or new navigation execution in this experiment.

The existing small training/evaluation functions are reused with an explicit
visual-history view; prior source, weights and result artifacts remain intact.
One shared encoding pass feeds both small fits. Before launch: 72 GiB RAM and
4.4 GiB artifact storage available, both GPUs idle, no competing training
process. Use one CPU thread on core 8 and retain compact weights/results only.

Runner: `scripts/run_go2_visual_history_dynamics_development.py`.
Plan: `docs/go2_visual_history_dynamics_plan_2026-09-17.json`.
Artifact: `go2_visual_history_dynamics_v1_attempt_001` on the artifact volume.
