# Matched maze-view readout intervention in closed-loop development navigation

**Current authority:** the [final decision-headroom handoff](go2_decision_headroom_agent_handoff_final_2026-09-23.md)
preserves these four Stage A assignments and supersedes the historical predictor-fit
suggestions later in this record. No follow-up fitting is authorised.

The two-maze predictor/readout diagnostic completed all 240 windows. At 700 ms,
the maze-data head decodes action predictions with translation XY RMSE
46.62/52.15 mm, versus the initial head's 83.63/90.92 mm. No-future-action
predictions decoded by the maze head have errors 78.65/81.22 mm, actual-future
features 27.02/27.17 mm, and command history 9.92/9.68 mm. All-window latent
MSE also favours action conditioning at both horizons on both mazes. These
results support useful predictive information but do not establish a JEPA
objective effect or a navigation improvement. Full tables are in
`go2_maze_view_prospective_transfer_2026-09-23.md`.

The next experiment changes only the frozen motion readout in the existing
action-conditioned navigation controller. Both readouts completed the same
440-update budget, with identical initialization, architecture, normalization
and optimizer. The old-data control received existing training examples;
the maze-data treatment replaced half of each batch with the four new training
mazes. The encoder and horizon-conditioned action predictor are unchanged.

| Fixed execution order | Existing cohort layout | Readout |
|---|---:|---|
| 1 | 00 | Existing-data control |
| 2 | 00 | Maze-data treatment |
| 3 | 02 | Maze-data treatment |
| 4 | 02 | Existing-data control |

Layout 00 is the first prior action-conditioned failure and has already been
examined in readout diagnostics. Layout 02 is the sole prior action-conditioned
success. Their selection is explicitly informed by previous results. Neither
layout supplied training examples, but both are exposed development evaluation
layouts. This is not a new independent prospective cohort. The order and all
four assignments were fixed before the follow-up began, with no retries or
outcome-driven replacement. Previous failures and successes remain unchanged.

The primary outcome is physically verified goal and return arrivals without
disallowed contact. Secondary outcomes are simulated duration, hold counts and
same-executed-window motion errors. Six candidate actions, eight forecast
horizons, the 4800-tick navigation budget, mapping, memory, control, sensing and
arrival criteria remain unchanged. Physics waits for high-level computation;
this is untimed simulation with the existing paired RGB/depth input and ideal
body gyro. It is not a hardware or real-time result. A positive outcome would
still require new prospective navigation and a separate JEPA-objective ablation.

Only one native/GPU job runs at a time, on CPU cores 8--15 and 24--31. Preparation
found 61.6 GiB available RAM and 40.8 GiB free on the output filesystem. The
run retains RGB, commands, physics, perception receipts, outcomes and failures;
unused raw depth arrays are not written. No training features are persisted.

Study root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_maze_view_readout_navigation_v1_attempt_001`.

The coordinator is `scripts/run_go2_maze_view_readout_navigation_development.py`.
Preparation completed successfully. The detached coordinator was launched as
PID 250817; `plan.json` fixes source/checkpoint identities, assignments and
comparisons, and `coordinator.log` records progress. Each native owner is
followed by the existing physical reader after it exits. All four assignments
and physical readers completed by 19:22 BST on 23 September. Runs use fresh sibling output roots containing `maze_view_old_data`
or `maze_view_maze_data`; no preceding artifact is overwritten. This explicit
development treatment does not replace the default navigation readout.

The loader verifies each checkpoint against its completed fit, fixed update
count and training plan. The runtime records the selected checkpoint and its
100--800-ms training horizons. The navigation launch and physical report label
these repeated layouts as exposed, avoiding a second prospective claim.
All four changed Python sources passed parsing before launch; preparation
verified completed inputs and available resources. The native trials themselves
provide the integration and physical-execution evidence.

Initial execution check: coordinator PID 250817 and native owner PID 250853
were confirmed live. The first trial reached camera frame 20 with three
world-model calls and nonzero turn commands. Its launch records the expected
old-data checkpoint hash, all eight trained horizons, selected CPU affinity,
and `new_independent_development_layout=false`. No arrival outcome is available.

First completed trial, 23 September: layout 00 with the matched existing-data
readout exhausted its navigation budget without a goal or home arrival. The
native owner exited successfully and the physical reader completed before the
coordinator launched layout 00 with the maze-data readout as PID 267135.

| Layout | Readout | Goal / home arrivals | Simulated duration | Disallowed contacts | Pipeline faults | Status |
|---|---|---|---:|---:|---:|---|
| 00 | Existing data | 0 / 0 | 480.32 s | 0 | 0 | Complete: budget exhausted |
| 00 | Maze data | 0 / 0 | 480.32 s | 0 | 0 | Complete: budget exhausted |
| 02 | Maze data | 0 / 0 | 480.32 s | 0 | 0 | Complete: budget exhausted |
| 02 | Existing data | 0 / 0 | 480.32 s | 0 | 0 | Complete: budget exhausted |

The completed control made 1,198 model calls and selected 987 holds, 21 forward
commands, 39 arcs and 151 turns. Its longest continuous zero-request interval
was 358.82 simulated seconds. In 899 hold decisions, a moving candidate had
higher recorded utility, but only hold passed the full predicted clearance
reserve. Thus the recorded clearance selection explains those holds; it does
not establish that executing a rejected candidate would have been safe or
successful. The last decision still had an observed-floor route to a frontier.

Reading the saved candidate clearances narrows this diagnosis further. All 899
of those decisions rejected every moving candidate under the full reserve
already within the shared first 300 ms, before their candidate commands
diverged. Their shared-prefix minimum clearances ranged from 0.45991 to
0.47570 m (median 0.47062 m), below the 0.48 m moving-action requirement.
Every moving candidate nevertheless passed the 0.45 m nominal footprint test
over its predicted path in all 899 decisions. No moving candidate passed either
recorded reserve-recovery test in 898 of the 899 decisions. These decisions
span frames 1196--4800; they are overlapping observations of one trajectory.
At the final frame, moving-path minimum clearances were 0.46523--0.46725 m;
hold required only 0.45 m and remained eligible.

Consequently, improving only the candidate-dependent future endpoint cannot
remove the full-reserve rejection in these saved states: the shared prefix
already fails it. A better forecast might still permit the existing recovery
rule, or lead the controller into a different state earlier in the run. This
is a controller/model interaction hypothesis, not evidence that reducing the
reserve is safe, nor a reason to change the controller during the paired study.
The source is the completed control's `planning.json`, using its selected hold
rows with a higher-utility moving candidate and their
`selection.memory_forecast_candidates` records. No sensor replay, changed
candidate bank or counterfactual physical execution was performed.

A subsequent CPU-only diagnostic measures motion error during that shared
prefix on the physical reader's same executed-window population. It uses the
saved native trajectory only for evaluation and verifies that all candidate
XY predictions share the first 300 ms. Both original 700-ms aggregate errors
are reproduced within 1e-8 mm. No new simulation, training or sensor replay
is involved. The 700-ms matching requirement leaves 898 of the 899
reserve-limited holds; 896 also have zero requested commands throughout the
prefix. The final plan lacks a complete future window and is excluded.

| Post-hoc population | Windows | Horizon | Learned XY RMSE | Command-history XY RMSE | Zero-motion XY RMSE |
|---|---:|---:|---:|---:|---:|
| Translation | 60 | 100 ms | 7.45 mm | 2.85 mm | 10.60 mm |
| Translation | 60 | 200 ms | 15.09 mm | 4.93 mm | 21.36 mm |
| Translation | 60 | 300 ms | 25.18 mm | 6.11 mm | 32.50 mm |
| Translation | 60 | 700 ms | 66.94 mm | 13.59 mm | 85.09 mm |
| Reserve-limited hold, zero requested prefix | 896 | 100 ms | 3.16 mm | 0.61 mm | 0.60 mm |
| Reserve-limited hold, zero requested prefix | 896 | 200 ms | 3.27 mm | 1.17 mm | 1.16 mm |
| Reserve-limited hold, zero requested prefix | 896 | 300 ms | 3.47 mm | 1.70 mm | 1.70 mm |
| Reserve-limited hold, zero requested prefix | 896 | 700 ms | 4.34 mm | 3.53 mm | 3.51 mm |

The learned prefix has residual error during the holds, but much larger errors
occur on moving windows. This distinguishes the motion-prediction limitation
from the recorded shared-prefix clearance rejection; it does not isolate their
causal contributions to navigation. In particular, physical XY error is not
the same quantity as mapped obstacle-clearance error, and this table cannot
certify a smaller reserve. These are overlapping windows on one exposed run,
not independent replications or a treatment-effect estimate.
The [diagnostic result](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_maze_view_old_data_v1_attempt_001/prefix_motion_diagnostic_v2.json)
retains all rows and additional hold/turn/all-window summaries. Its evaluator
is [evaluate_go2_maze_view_navigation_prefix_development.py](../scripts/evaluate_go2_maze_view_navigation_prefix_development.py).

At 700 ms, same-executed-window translation XY RMSE was 66.94 mm for the learned
forecast versus 13.59 mm for command history across 60 overlapping translation
windows. Across all 1,197 windows, dominated by holds, these errors were 19.11
and 5.37 mm respectively. The translation-specific result is more informative
about motion quality than the aggregate. These retrospective references do not
measure alternative navigation outcomes. Model-call median/p95 latency was
2.750/2.761 seconds; this remains untimed simulation. Native execution took
4,256.43 wall seconds, excluding the later physical reader.

Evidence: [native result](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_maze_view_old_data_v1_attempt_001/result.json),
[physical and decision readout](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_maze_view_old_data_v1_attempt_001/dense_navigation_readout.json),
and [coordinator result row](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_maze_view_readout_navigation_v1_attempt_001/layout00_old_data_result.json).
Both layout-00 failures are retained and do not change the fixed remaining
assignments.

The maze-data treatment has now completed its native run and physical reader.
It also exhausted the 480.32-s budget without either arrival, with zero recorded
disallowed contacts and no pipeline faults. The coordinator then launched the
fixed layout-02 maze-data trial as PID 282475. On this exposed layout, the
readout intervention changes behaviour but does not improve the binary
navigation outcome. It does not establish equivalence or a general failure of
the training intervention: this is one run per treatment on one layout, with
the second layout still pending.

| Layout-00 metric | Existing-data control | Maze-data treatment |
|---|---:|---:|
| Goal / home arrivals | 0 / 0 | 0 / 0 |
| Selected hold plans / all plans | 987 / 1198 | 979 / 1198 |
| Longest continuous zero request | 358.82 s | 274.40 s |
| Translation selections (forward + arcs) | 60 | 95 |
| Turn selections | 151 | 124 |
| Stopping-projection selection changes | 4 | 91 |
| 700-ms translation XY RMSE, learned / command history | 66.94 / 13.59 mm | 79.65 / 9.01 mm |
| 700-ms all-window XY RMSE, learned / command history | 19.11 / 5.37 mm | 24.96 / 4.71 mm |
| 700-ms all-window yaw RMSE, learned / command history | 4.36 / 0.51 degrees | 3.06 / 0.52 degrees |
| Model-call median / p95 | 2.750 / 2.761 s | 2.738 / 2.751 s |
| Native wall duration, excluding physical reader | 4256.43 s | 4217.26 s |

Each error comparison against command history uses the same executed windows
within that run. The two treatments followed different trajectories and have
different motion populations, so cross-column RMSE changes do not isolate the
effect of readout training on prediction accuracy. Fixed-tape component
improvements have not translated into goal-reaching on this layout.

Of the treatment's 960 holds with a higher-scoring moving candidate, 871 had
only hold pass the full predicted reserve, 79 had no full-reserve candidate,
and ten had at least one full-reserve moving candidate. The stopping projection
changed 84 of these 960 decisions. The final decision was
`ADDITIONAL_VIEW_REQUIRED` with no candidate passing even the nominal predicted
footprint check, unlike the control's final nominally clear frontier route.
Thus the two failures should not be described as the same single clearance
mechanism merely because both contain long holds.

The matching prefix diagnostic also completed for the treatment. At 300 ms,
learned/command-history XY RMSE is 34.78/4.41 mm on its 95 translation windows,
and 5.17/1.48 mm on 867 holds that had a higher-utility moving candidate, only
hold passing full reserve, and a zero requested prefix. Both 700-ms aggregate
metrics reproduce the physical reader's result. A diagnostic correction was
needed: V1 always ranked base candidate utilities, whereas view-seeking choices
use `scan_utilities`. V2 follows the same score selection as the physical
reader. V1 outputs are preserved, the control's reported metrics are unchanged,
and the treatment's zero-prefix subgroup increases from 567 to 867 windows.
The V1 treatment subgroup values are superseded; all-window and translation
metrics are unchanged. This correction changes only the post-hoc analysis.

Treatment evidence: [native result](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_maze_view_maze_data_v1_attempt_001/result.json),
[physical and decision readout](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_maze_view_maze_data_v1_attempt_001/dense_navigation_readout.json),
and [corrected prefix diagnostic](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_maze_view_maze_data_v1_attempt_001/prefix_motion_diagnostic_v2.json).

While the layout-02 pair runs, a bounded input check confirmed that the existing
recordings from four training mazes can support an action-conditioned predictor fit,
not only a motion-readout fit. All 2,448 departure contexts have the required
RGB history at -1000/-500/0 ms, available past applied commands, and all eight
100--800-ms future images: 19,584 overlapping training windows. Future commands
computed from the known requested tape and current limiter state match the
recording exactly. Future measured commands were checked for alignment only;
they are not proposed model inputs. All 2,736 required RGB images are present.
Dense FP16 features for these images would occupy approximately 4.01 GiB RAM,
before old-data features, models and training batches. No additional collection
or depth replay is needed to use this particular population.

This makes a matched existing-data versus maze-supplemented predictor fit a
concrete next option if the completed navigation comparison still points to
forecast transfer error. Encoder and motion readout would need to remain fixed
to distinguish predictor training from the readout intervention. The present
navigation assignments remain unchanged; no new fit or encoding has started.
These recordings have only four independent training geometries and a fixed
motion tape, so their sample count does not imply broad coverage or same-state
counterfactual supervision. The [input-support record](go2_maze_view_predictor_training_support_2026-09-23.json)
records the counts and exact command-alignment result; it is not a prediction
or navigation result.

A subsequent CPU-only coverage analysis makes the fixed-tape limitation more
specific. The 2,448 new-maze contexts contain only **153 distinct combined
command sequences** when comparing the 15 past applied forward/yaw pairs and
seven future pairs at 700 ms. Only 16 contexts have an entirely zero past,
and none has both an entirely zero past and a zero seven-step future. The same
17-second tape repeats across all 16 recordings; its one-second hold blocks
do not supply sustained stationary contexts with stationary futures.

| Command coverage against the new-maze population | Layout-00 control | Layout-00 treatment |
|---|---:|---:|
| Matched executed windows | 1197 | 1197 |
| Entire past and future commands zero | 951 | 938 |
| Past history exactly represented | 1025 | 1017 |
| Future sequence exactly represented | 1109 | 1120 |
| Combined past and future exactly represented | 6 | 8 |
| Translation windows with represented combined sequence | 1 / 60 | 0 / 95 |

Exact matching uses a 1e-6 component tolerance. Separate past/future matches
can refer to different training examples; their high marginal coverage does
not imply coverage of the combined predictor input. Median nearest combined
sequence RMS, with forward/yaw scaled by 0.20 m/s and 0.45 rad/s, is 0.499 on
the control's translating windows and 0.530 on the treatment's. These are
descriptive command distances, not calibrated uncertainty or a failure
threshold. The future commands here are executed, post-hoc comparison data,
not model inputs or unexecuted candidate outcomes.

This does **not** mean the existing predictor has never seen stationary
sequences. Its original 700-ms training pool contains 3,326 samples, including
480 all-zero combined sequences: 216 from family recordings, 256 from switch
recordings and eight from balanced starts. That count reconstructs physical
past commands using the original saved normalization and uses the first seven
stored future commands. The original pool has 665 distinct combined sequences
after rounding to six decimals. Thus the new data expands maze imagery but
does not cover the full command/context distribution of autonomous navigation.
Retaining the old-data half of a proposed continuation is material; replacing
it wholesale with the new tape would discard useful command coverage.

The next predictor experiment should distinguish improvement on the matching
fixed tape from transfer to autonomous histories, reporting holds, translation
and turns separately. Exact novelty alone does not show that a model cannot
generalize, and this analysis does not isolate image or physical-state
coverage. No evaluation recordings were added to training, no new fit was
started, and the fixed navigation comparison remains unchanged.

Evidence: [new-maze command-coverage result and per-window rows](go2_maze_view_command_coverage_2026-09-23.json),
[CPU evaluator](../scripts/evaluate_go2_maze_view_command_coverage_development.py),
[original predictor training samples](/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001/samples.json),
and [original command normalization](/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/proprio_norm_stats.json).
The original-sample SHA-256 is
`19097c2b611bc01b54d864e32318ab857ea5e29fa3bfc51da2e94775cae4dc6e`;
the normalization SHA-256 is
`9380b4c6d9b59099e43bba9898e1417c273f88075d1ed122401cbb3272e18f94`.

## Third assignment complete: layout 02, maze-data head

The native owner and physical reader completed by 18:12 BST on 23 September.
The original coordinator then started the fourth assignment, layout 02 with
the existing-data head. No assignment was duplicated, restarted or extended.

| Recorded outcome | Layout 02, maze-data head |
|---|---:|
| Goal / home arrivals | 0 / 0 |
| Simulated duration / native wall duration | 480.32 / 4241.26 s |
| Disallowed contact samples / pipeline faults | 0 / 0 |
| Selected plans / holds | 1198 / 1140 |
| Selected forward / arcs / turns | 0 / 4 / 54 |
| Longest continuous zero-request interval | 452.82 s |
| Holds with higher recorded non-hold utility | 1137 |
| Decisions reporting no clear candidate | 1132 |
| Stopping-projection changes | 0 |
| 700-ms matched executed windows | 1195 |
| All-window XY RMSE, learned / command history | 8.53 / 3.72 mm |
| Hold XY RMSE, learned / command history; windows | 6.04 / 3.47 mm; 1139 |
| Translation XY RMSE, learned / command history; windows | 86.70 / 9.34 mm; 4 |
| Turn XY RMSE, learned / command history; windows | 17.10 / 6.85 mm; 52 |
| All-window yaw RMSE, learned / command history | 2.00 / 0.35 degrees |
| Model-call median / p95 | 2.745 / 2.753 s |

This trial failed to reproduce the retained initial-head success on layout 02.
It is one development run on an exposed layout, not an isolated causal estimate
of a readout effect. The low aggregate XY error is dominated by holding and
does not establish good motion forecasts or good decisions: only four matched
windows involved translation. Recorded utility and clearance gates do not
establish the safety or benefit of unexecuted candidates. No new counterfactual
comparisons were made while consolidating this assignment.

Evidence: [native result](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout02_action_maze_view_maze_data_v1_attempt_001/result.json)
and [physical reader with executed-window errors](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout02_action_maze_view_maze_data_v1_attempt_001/dense_navigation_readout.json).

## Retained initial-head comparators for Phase 0

The initial-head rows below are reused observations from the completed dense
cohort, not extra Stage A trials or independent replications. Their frozen
readout was `go2_full_heading_readout_v1_attempt_001/mixed_data_final.pt`,
SHA-256 `bbbb05fd2e2984ac4d818abc986ec0401e24e9d91f53ee4e5b49443b832bdf85`.
The two matched heads' identities remain those in the Stage A plan.

| Layout | Readout / evidence role | Goal / home | Simulated seconds | Holds / plans | All-window 700-ms XY RMSE: learned / command (mm) | Translation windows | Translation XY RMSE: learned / command (mm) |
|---|---|---|---:|---:|---:|---:|---:|
| 00 | Initial, retained predecessor | 0 / 0 | 480.32 | 504 / 1198 | 26.19 / 6.06 | 116 | 68.48 / 10.55 |
| 00 | Matched existing-data control | 0 / 0 | 480.32 | 987 / 1198 | 19.11 / 5.37 | 60 | 66.94 / 13.59 |
| 00 | Matched maze-data treatment | 0 / 0 | 480.32 | 979 / 1198 | 24.96 / 4.71 | 95 | 79.65 / 9.01 |
| 02 | Initial, retained predecessor | 1 / 1, physically verified | 359.62 | 117 / 888 | 52.72 / 10.78 | 346 | 73.70 / 12.96 |
| 02 | Matched maze-data treatment | 0 / 0 | 480.32 | 1140 / 1198 | 8.53 / 3.72 | 4 | 86.70 / 9.34 |
| 02 | Matched existing-data control | 0 / 0 | 480.32 | 957 / 1198 | 25.77 / 6.14 | 102 | 68.76 / 13.41 |

All six completed rows have zero disallowed contact samples and no pipeline
faults. The initial layout-02 trial additionally has eight short translation-
pulse windows, with learned/command-history XY RMSE 23.55/7.82 mm; these are
kept separate from the 346 ordinary translation windows in the table.
Matched-window counts for the initial layout-00/layout-02 rows are 1197/858,
with all-window yaw RMSE 9.13/0.82 and 7.58/1.06 degrees, respectively.
Their longest continuous zero-request intervals are 29.20 and 9.60 seconds.

The successful initial layout-02 trajectory has substantially higher aggregate
forecast error than the failed maze-head trajectory. This is a concrete warning
against treating aggregate motion RMSE as navigation quality: the latter
population is almost entirely holding. Each learned-versus-command error pair
uses identical executed windows within its own run, but the populations differ
between runs. Neither this contrast nor the prior success isolates a causal
readout effect. The forthcoming audit asks about selections on common restored
states; these historical trajectories cannot answer that question themselves.

Retained evidence: [layout-00 initial reader](/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_mixed_data_v1_attempt_001/dense_navigation_readout.json)
and [layout-02 initial reader](/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout02_action_mixed_data_v1_attempt_001/dense_navigation_readout.json).

## Stage A closed: fourth assignment and complete interpretation

The final layout-02 existing-data run and its physical reader completed before
19:22 BST. The coordinator's final result is `COMPLETE`, covering exactly the
four original assignments, with coordinator wall duration 17105.12 s. Both
original owners exited. No trial was duplicated, restarted or extended.

| Final assignment metric | Layout 02, existing-data control |
|---|---:|
| Goal / home arrivals | 0 / 0 |
| Simulated duration / native wall duration | 480.32 / 4164.82 s |
| Disallowed contact samples / pipeline faults | 0 / 0 |
| Selected plans / holds | 1198 / 957 |
| Selected forward / arcs / turns | 28 / 74 / 139 |
| Longest continuous zero-request interval | 172.82 s |
| Holds with higher recorded non-hold utility | 860 |
| Decisions reporting no clear candidate | 565 |
| Stopping-projection changes | 15 |
| Matched 700-ms executed windows | 1196 |
| All-window XY RMSE, learned / command history | 25.77 / 6.14 mm |
| Hold XY RMSE, learned / command history; windows | 14.52 / 3.63 mm; 956 |
| Translation XY RMSE, learned / command history; windows | 68.76 / 13.41 mm; 102 |
| Turn XY RMSE, learned / command history; windows | 28.32 / 10.12 mm; 138 |
| All-window yaw RMSE, learned / command history | 3.98 / 0.56 degrees |
| Model-call median / p95 | 2.744 / 2.752 s |

Neither matched head completed either exposed layout: existing-data 0/2 and
maze-data 0/2. All four runs exhausted their fixed navigation budgets without
either arrival. The retained initial-head comparison remains 0/1 on layout 00
and 1/1 on layout 02. Thus neither matched head reproduced the initial-head
layout-02 success, despite the maze head's lower component errors on the
separate fixed-tape assay. These outcomes do not establish equivalence, a
causal readout effect or generalisation to independent layouts: there is one
run per cell on two deliberately selected, exposed development layouts.

The layout-02 treatment held in 1140/1198 decisions versus 957/1198 for the
matched control; its longest zero-request interval was 452.82 versus 172.82 s.
Its much lower all-window learned error (8.53 versus 25.77 mm) came from a
different trajectory with only four matched translation windows, versus 102.
On their respective translation windows, command-history motion estimates
outperformed learned estimates in all four Stage A trials. None of these
executed-window comparisons measures the quality of unchosen actions.

Stage A therefore supplies the completed development intervention required by
Phase 0. It motivates the already authorised decision audit, not additional
fitting, controller repair or a repeated navigation cohort.

Final evidence: [coordinator result](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_maze_view_readout_navigation_v1_attempt_001/result.json),
[fourth native result](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout02_action_maze_view_old_data_v1_attempt_001/result.json),
and [fourth physical reader](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout02_action_maze_view_old_data_v1_attempt_001/dense_navigation_readout.json).
