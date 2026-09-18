# Expanded corrected prediction readout completed and independently checked

Result434dd34eaf9e07d672624d0ec1d9d0a910b747d8ae65b34d75415be311660ce2,
root go2_all_phase_corrected_prediction_readout_v1_attempt_001.
Original handle45959/PID2643209 closed0. Launch
0701ae2cd1eedb705267a97c348ce4d31b893a3c17a209a518e281fe070c876e.
1169 source bindings,75 output bindings. All18 models,30 trained heads and both
roles were evaluated:36 corrected array files,60 head/role before-after score
pairs,1080 per-model metric rows,216 descriptive groups and540 fixed matched
comparisons. Reported readout phase wall109.78195782913826s excludes prerequisite
full correction/model admission before output launch.

All coefficients were reconstructed from the original training-only correction
before readout. All corrected arrays were saved before any corrected scoring.
Every original primary score reproduced. Yaw/contact components, clocks, masks,
padding, target populations, censoring and non-position scores/counts remained
unchanged. No neural forward pass, weight update, coefficient refit, model
selection or native execution occurred in the scoring phase.

Independent43021 closed0 after reconstructing all36 corrected arrays and all60
head/role score pairs directly from bound original arrays and targets. It also
reproduced all1080 per-model rows,216 descriptive groups and540 comparisons,
and checked all original primary scores. Source, readout, fit and correction
artifact bindings were verified before and after. The original runner had also
authenticated the original input collections again after scoring. No narrower
component test is being substituted for these actual result checks.

## Fixed primary model: effect of its training-only correction

Values are target-count-weighted mean XY position errors in millimetres, using
all valid motion targets across the eight100ms horizons. They are not RMSE.
The model is the already fixed seed_2026091001_full_jepa.

| Population | Raw | Corrected | Valid motion targets |
| --- | ---: | ---: | ---: |
| Training family | 19.0130 | 7.5764 | 11856 |
| Training switch | 18.3198 | 8.0967 | 16588 |
| Development transfer family | 19.8390 | 8.8236 | 2592 |
| Development transfer switch | 23.7388 | 16.6392 | 572 |

At the first100ms horizon, corrected development transfer error is7.8042mm on
family data and13.9375mm on switch data. At500ms it is7.8506mm and16.9308mm.
These are completed prediction results on the existing development population;
they do not demonstrate successful online navigation or stopping stability.

## All fixed primary heads across three optimization seeds

Mean XY error in millimetres after correction; mean plus/minus sample standard
deviation across the three fixed optimization seeds. These are not independent
maze confidence intervals. Every source retains its original available transfer
contexts:348 family and72 switch. Related windows/horizons are not independent
replications. The complete source/stratum and auxiliary-head results remain in
the bound metrics and score artifacts.

| Model input | Training condition / primary head | Family transfer | Switch transfer |
| --- | --- | ---: | ---: |
| Full | JEPA / rollout | 13.6776 ± 4.9757 | 18.2943 ± 1.6795 |
| Full | Supervised / rollout | 8.6825 ± 1.7538 | 14.1000 ± 3.4089 |
| Full | Direct / direct | 11.1801 ± 0.8813 | 18.1855 ± 2.0822 |
| No model RGB | JEPA / rollout | 11.4864 ± 8.0463 | 15.8787 ± 7.0442 |
| No model RGB | Supervised / rollout | 7.6974 ± 3.6618 | 15.1882 ± 5.6558 |
| No model RGB | Direct / direct | 10.9750 ± 2.3026 | 17.0213 ± 5.6939 |

Full-input supervised models have lower descriptive mean error than full-input
JEPA on both development sources. Removing model RGB does not consistently
worsen these prediction scores. Neither observation establishes navigation,
planning or memory advantage, and neither changes the queued cases. The direct
condition remains a predictive model control. Model-RGB ablation leaves the
native controller's RGBD mapping and localization active.

## Execution identities and next work

Completed full-fit input:
44c4cd65812b021b29cfb0aff33e2058cfaaa71dcc627d36eced30dfac17ea35.
Completed training-only correction:
1b36dc77ca51d342e45d73da027ebdbdbd1263ab5be4142766948fd8258dd460.
Source/protocol/test hashes and preflight evidence are retained in
docs/go2_all_phase_corrected_readout_progress_2026-09-10.md. Its earlier pending
status is superseded by this completed result; do not relaunch readout45959.

The native assignments remain the six seed2026091001 full/no_rgb JEPA,
supervised and direct models fixed before score inspection. Original native
25801/PID2636286 with worker2637549 is still live at the latest process check;
waiter14784/PID2641948 remains live and owns the next six-case launch. Preserve
both owners and their frozen sources. No expanded-model native case has yet
been claimed completed. The goal remains active with30 completed audited native
episodes and zero verified round trips at this observation.

Next evidence must come from the existing original pilot's terminal raw audit,
then the six assigned fresh physical executions and full raw audits. Independent
maze results, matched planning/memory controls, realistic sensing, real-time
feasibility and bounded hardware evidence remain open. Training resubstitution
or improved development prediction scores do not complete those requirements.
