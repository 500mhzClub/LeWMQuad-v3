# Prepared dense-predictor comparison on saved navigation recordings

Status: **COMPLETE**, process/session 70961 exited 0. Native adaptation
and the corrected branch comparison completed before launch. All 384 selected
windows were evaluated in 463.4 seconds. No new navigation or fitting.

The adapted action model improves pooled feature error by **17.2% against
persistence** and **11.7% against the matched no-future-action predictor**.
It beats the no-action model in every recording and every action category.
It beats persistence in three of four recordings and all five moving-action
categories, but loses substantially on hold. These are future visual-feature
errors; physical motion accuracy and navigation benefit remain untested.

| Predictor | Pooled 500-ms feature MSE | MSE / persistence |
| --- | ---: | ---: |
| Adapted, future actions | 0.486834 | 0.8275 |
| Adapted, no future actions | 0.551460 | 0.9374 |
| Unchanged historical rollout | 0.734123 | 1.2479 |
| Persistence | 0.588290 | 1.0000 |

| Action | Windows | Adapted action MSE | No-action MSE | Persistence MSE |
| --- | ---: | ---: | ---: | ---: |
| Hold | 59 | 0.497005 | 0.538083 | 0.227087 |
| Forward | 91 | 0.488350 | 0.542052 | 0.507625 |
| Left arc | 52 | 0.478129 | 0.538081 | 0.693880 |
| Right arc | 43 | 0.474742 | 0.549255 | 0.710225 |
| Left turn | 86 | 0.489464 | 0.565817 | 0.709058 |
| Right turn | 53 | 0.486990 | 0.574121 | 0.730392 |

Per-recording action-model MSE/persistence ratios: **0.7721, 1.1332, 0.7796,
0.7282**. Run 2 includes 43 hold windows in its 96-window sample. This explains
why action-conditional improvement against no-action does not imply an overall
win over persistence on every mission. No post-hoc hold substitution was applied.

Complete numerical results: `go2_dense_native_recordings_result_2026-09-17.json`.
Per-window results remain under `recorded_mission_evaluation/run_01.json` through
`run_04.json` in the adaptation attempt. All four recordings and both failed
returns remain in the result; none was excluded based on prediction errors.

The pulse panel measures action discrimination over small differences. This
complementary diagnostic covers recorded mission motion using 96 evenly spaced
eligible windows from each of the four return-memory runs, including both failed
returns. Of the original 2,404 matched execution windows, eight lack the required
one-second visual history. The fixed selection contains 384 of the remaining
2,396 windows and includes all six actions in each recording. Sampling was fixed
before model evaluation, without using future prediction errors.

Compare adapted action, adapted no-future-action, unchanged historical rollout,
and current-feature persistence at 500 ms in the same frozen V-JEPA token space.
Use context at -1000/-500/0 ms and the actual observation-time candidate plan,
converted through the platform limiter from the causal last applied command.
The 384 histories, image times, file availability and planned-versus-executed
command sequences were checked successfully on CPU. No model or RGB encoder
was run for that check. Execution must still verify future command agreement
after each forecast and load target RGB only after that forecast.

Report per-recording, per-action and pooled MSE/persistence ratios. Equal sample
counts weight the four recordings equally; the pooled result is not the natural
frequency of all mission windows. These are dependent retrospective windows,
not independent navigation replications or unexecuted-action outcomes.

Source: `scripts/evaluate_go2_dense_native_recordings_development.py`.
Fixed inputs: `go2_dense_native_recordings_plan_2026-09-17.json`.
Output: the active adaptation attempt's `recorded_mission_evaluation/` directory.
The run retains compact metrics and a bounded RAM feature cache, no dense disk
cache. Before launch: 72 GiB available RAM, 4.0 GiB output-volume free space,
about 1.84 GB of 34.21 GB GPU VRAM occupied, and CPU 99% idle. Training and branch
evaluation were terminal. One GPU process uses four CPU threads on CPUs 8-11;
the prior measured batch-one encoder path was retained. First 96 windows took
118.7 seconds; the complete evaluation took 463.4 seconds. Do not duplicate it.
