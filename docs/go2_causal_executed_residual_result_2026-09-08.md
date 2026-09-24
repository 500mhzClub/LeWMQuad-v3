# Causal observed residual diagnostic: prediction evidence only

The JSON-admission integrity replacement completed all four fixed input cases.
It used the preceding eight ticks' mean predicted-minus-observed XY residual,
with the current target revealed only after the current correction. Native
outcomes were excluded from the correction helper and used only afterward for
evaluation. No model, controller or native trajectory changed.

| Recorded approach / condition | Eligible steps | Original mean XY error | Corrected mean XY error | Original max | Corrected max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact target / JEPA, all | 239 | 11.7241 mm | 8.4302 mm | 32.4656 mm | 25.8987 mm |
| Exact target / JEPA, final goal | 71 | 9.1484 mm | 6.4188 mm | 30.7892 mm | 20.8731 mm |
| Executed horizon / JEPA, all | 240 | 11.3120 mm | 8.4831 mm | 32.4656 mm | 25.8987 mm |
| Executed horizon / JEPA, final goal | 72 | 7.8103 mm | 6.6230 mm | 24.8890 mm | 24.6322 mm |
| Direct, same trajectory in both inputs | 43 each | 7.3986 mm | 4.3764 mm | 13.0820 mm | 13.5489 mm |

The two direct trajectories are duplicates, not independent replications.
JEPA also shares its first 168 forecast steps between approaches. Final-goal
groups use the existing controller's exact-target flag, not a tuned distance
threshold. Native goal outcomes remain unchanged: zero fully verified arrivals.

Correction reduced error on 53/71 original JEPA final-goal steps and 49/72
executed-horizon final-goal steps; it worsened the remaining 18 and 23. Across
all eligible steps, improvements/worsenings were 169/69 and 165/74; each first
step was unchanged because no preceding residual was available. Direct had
32 improvements, 10 worsenings and one unchanged step, including a slightly
larger maximum error after correction. Do not turn mean improvement into an
error bound or claim that every corrected forecast is better.

Observed-vs-native 100-ms displacement disagreement averaged about 0.19 mm
overall, with maximum 1.0566 mm. The new JEPA final-goal subgroup had maximum
0.4805 mm. These are retrospective comparisons on valid measured trajectories,
not sensor calibration or prospective uncertainty guarantees. Each direct
tick-46 terminal hold had native labels but no next public pose; those targets
remain recorded separately and were not imputed into the causal correction.
All JEPA matched steps had a public endpoint. Every stored multi-step native
label stops when its complete actual command prefix stops matching the plan.

Six causal/prefix tests passed in 0.22 s. After preserving the original
JSON-list/tuple adapter failure, six adapter tests passed in 1.80 s and all
582 non-null recorded poses passed original admission through the new adapter.
Analysis and summary function ASTs stayed identical. The successful diagnostic
took 25.468856108142063 s after launch, with 1,357 bound source paths. Terminal
available RAM was 82,754,543,616 B and artifact free space 60,542,676,992 B.

Under `go2_causal_executed_residual_diagnosis_integrity_v1_attempt_001`:

- Launch: `c37334aaa931e3dd0298188dc2116e0e75e8508807d8ff84aa16245a94a59e63`
- Motion: `beee296bb7ac860cdbc00e1ebbb4d44e575d2e0d8b60ff4a752625faa5ec5dbb`
- Result: `c07af36277147654ee0c9b651ab46c280833b128d4b657adee1225bfa2cf4dae`

The failed original root, launch/failure and every original source remain
immutable. No native task is left running from these experiments.

## Next implementation supported by this evidence

Implement a separately named causal final-goal score correction. Store only
the actually requested previous command's original 100-ms prediction and
admitted starting pose. At the next admitted public pose, compute its observed
residual; retain the same eight-tick history. Use original model predictions
for residual updates, so the estimator does not learn its own already-corrected
error. Update only after the executed step is observed; never use native labels,
future frames or an unexecuted candidate's outcome. Keep model weights frozen.

Apply the shared XY offset only to final-goal utility scoring, where the
executed-horizon scorer already operates. Preserve original forecasts and all
original surface/eight-step nominal checks, contact score/horizon, yaw,
intermediate/view utility, arrival thresholds and mission budget. Record the
separate scoring pose and full residual provenance; do not relabel the raw
prediction or claim its checked path is a corrected-path certificate. Verify
causal timing, terminal/drain behavior, immutable forecasts/constraints and two
fresh prefix replays before a separately declared prospective native pair.

The four new maze source candidates still need variable-goal/return control,
longer bounded sessions and physical return evaluation. Reactive/nonpredictive,
JEPA/RGB/planning/memory comparisons, realistic timing, independent navigation
and bounded hardware evidence remain required for the overall goal.
