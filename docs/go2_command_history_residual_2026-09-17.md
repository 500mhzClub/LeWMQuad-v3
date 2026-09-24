# Command-history reference plus learned RGB/body correction

The preceding fixed 6000-update experiment did not improve overall position
prediction. Before another native mission, test a change to the motion reference:
replace ideal command integration with the existing frozen command-history fit,
and train the same RGB/body/action model to correct it. Preserve the latent JEPA
objective and the matched supervised condition. This is a hybrid reference plus
learned residual; the reference alone is an essential comparison.

## Diagnosis motivating the change

`scripts/probe_go2_forecast_observation_dependence_development.py` completed at
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_forecast_observation_dependence_v1_attempt_001/`.
Its fixed every-tenth-window sample contains 376 windows from all six recent
missions, selected before errors. Replacing past RGB frames with copies of the
current RGB changed original JEPA position predictions by only 0.0068 mm RMS;
the position error was essentially unchanged. The 6000-update JEPA changed by
1.47 mm under that intervention and 6.29 mm under half brightness, without an
overall position-error improvement. These inference perturbations are not
trained no-RGB ablations or proof of a causal navigation effect.

All 3726 executed windows' first three commands appeared in training. Full
seven-command sequences appeared for 3720/3726 windows, including every left
and right turn. This rules out missing turn-command sequences as the immediate
explanation, but does not establish coverage of observed states or histories.
The coverage and perturbation receipts are `command_coverage.json` and
`result.json` in that diagnostic directory.

## Fixed experiment

Plan: `docs/go2_command_history_residual_plan_2026-09-17.json`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_command_history_residual_matched_fits_v1_attempt_001/`.

- Two fresh fits, JEPA and supervised rollout; same seed, 4694 original training
  contexts, 7200 scheduled draws and 1200 optimizer updates as the original fits.
- The frozen command-history reference was fitted on that same training
  population and draw weights. It consumes only four past public command
  histories and the causal prefix of prospective commands. No estimated poses,
  native state or recent navigation targets enter the model or its training.
- RGB, body and command inputs, latent architecture, loss weights and optimizer
  remain the same. Reference coefficients are frozen model buffers; only the
  learned correction is optimized.
- Compare reference-only, both corrected models and both original 1200-update
  models on identical causal inputs for all 3726 matched executed windows from
  the six recent recordings. Report prefix, action-increment and whole-window
  XY errors plus heading error, including per-run and per-action results.
- Saved unexecuted turn alternatives can diagnose clearance effects but cannot
  establish their safety or navigation outcome. This is exposed development
  evaluation, not independent generalization.

The three focused checks passed in 2.32 s: NumPy-reference parity and partial
causal plans, and training/inference composition with frozen reference buffers
for each training condition. Available memory was 64 GiB and the output volume
had 34 GiB free. Both CPU fits launched concurrently with one thread each,
JEPA on core 8 (session 88038) and supervised on core 0 (session 85019), with no
native simulation running.

Both fits completed at 1200 updates and exited zero. JEPA took 167.06 s and
supervised 143.95 s, including about 67 s preparing the unchanged training
inputs. Peak RSS was 10.10/10.09 GB respectively. Their initial model/reference
state was identical (`779b6bf38acf74fe8bab85d92c2dd7b446264dc1020025ef5c0aa80083a1b1d5`).
Both final snapshots reloaded with the reference buffers unchanged. Final model
states are JEPA `b47e0182a7cffb8e36d7d5c52ceb57fdcc8a37427381c4ab19adc4ae599bc274`
and supervised `83cdbc77e9d01089cd92fe1f194c59ca9b3ff54bc3196a1ce2ee287f433a6bba`.
Paired prediction evaluation launched in session 3371 after both fits exited.

Sources: `lewm/command_history_residual_learning_development.py`,
`scripts/command_history_residual_snapshot_development.py`,
`scripts/train_go2_command_history_residual_development.py`, and
`scripts/evaluate_go2_command_history_residual_development.py`.

The overall navigation goal remains incomplete. Navigation benefit requires a
subsequent prospective execution; forecast improvement alone cannot establish it.

## Complete prediction result

Evaluation session 3371 exited zero in 154.64 s, including every one of the 3726
preselected windows from all six recordings. All original neural forecasts
reproduced array-exactly. The largest difference between the reference computed
from float32 model inputs and the recorded double-precision command-history
forecast was 8.65e-9 (metres or radians, depending on component).

| Model | Prefix XY RMSE (mm) | Whole 700-ms XY RMSE (mm) | Action-increment XY RMSE (mm) | Yaw RMSE (degrees) |
| --- | ---: | ---: | ---: | ---: |
| Command-history reference only | 3.51 | 7.14 | 4.89 | 0.64 |
| Reference + JEPA correction | 12.10 | 13.38 | 4.94 | 3.35 |
| Reference + supervised correction | 11.23 | 14.86 | 6.37 | 0.94 |
| Original JEPA | 6.12 | 9.76 | 7.50 | 2.77 |
| Original supervised | 5.39 | 8.66 | 6.56 | 1.19 |

Both learned corrections worsen whole-window position error relative to the
reference in every one of the six recordings. The reference also has lower
whole-window position error than both original models in each recording.
This is matched-input prediction evidence; it does not show that using the
reference as a navigation controller would complete these missions.

The new JEPA has a prefix mean signed error of [+8.64, -4.70] mm and a whole-window
mean signed error of [+8.37, -4.25] mm. Its action-increment error is close to
the reference because much of that offset appears at both endpoints. On hold
windows its prefix error is 13.41 mm versus reference 2.08 mm. Both left- and
right-turn prefix errors are also worse. Do not treat the smaller action-increment
error than original JEPA as a successful intervention while ignoring the large
prefix bias. Neither new model is promoted; no native mission uses these weights.

Results and individual matched-window errors are in `prediction_evaluation/`.
The next diagnosis evaluates the frozen models on all original training inputs,
with original draw weights and labels used only for scoring. This distinguishes
failure to fit the original motion task from transfer failure before another
architecture or optimization change. It makes no optimizer updates and does not
load target images. Script:
`scripts/diagnose_go2_command_residual_training_error_development.py`;
session 83041; output `training_prediction_diagnosis/`.

That diagnosis completed and exited zero. All 4694 training contexts and 7200
draw weights were retained; motion metrics use available labels at each horizon.
At 300 ms, reference/JEPA/supervised XY RMSE was 2.85/10.93/10.54 mm. At 700 ms
it was 5.68/11.63/13.19 mm, over 3938 labelled contexts representing 5956 draws.
At 800 ms it was 6.27/11.86/13.65 mm. The JEPA mean signed error was approximately
[+8.0,-4.0] mm on training inputs as well as navigation inputs. The reference's
weighted training mean error was effectively zero, as expected from its fit.
Thus the new models degrade the reference on their own training distribution;
transfer shift alone cannot explain the failure. This does not yet identify
which optimization or multi-objective interaction caused the bias.

Next test one explicit optimization change: learning rate 1e-4 instead of 1e-3,
retaining the same 1200 updates, initialization, schedule, architecture, losses,
reference and evaluation populations. Train both JEPA and supervised conditions
fresh; no checkpoint selection, post-hoc residual damping or navigation promotion
based on a few favorable frames. Keep the failed 1e-3 results unchanged.

The fixed lower-rate plan is
`docs/go2_command_residual_lower_rate_plan_2026-09-17.json`; its output is
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_command_residual_lower_rate_matched_fits_v1_attempt_001/`.
`scripts/run_go2_command_residual_lower_rate_development.py` reuses the same
trainer, snapshot format, evaluator and training-input diagnostic. Fresh JEPA
session 52799 runs on core 8; supervised session 21443 on core 0. No native
simulation runs alongside them.

Both lower-rate fits completed and exited zero at 1200 updates (JEPA 165.9 s,
supervised 138.8 s). Their saved configurations record learning rate 1e-4.
Navigation-recording prediction evaluation launched in session 88434 on core 8;
the training-input diagnostic launched in session 22587 on core 0. These are
fixed-weight readouts on separate cores, with no native simulation running.

## Lower-rate result: improved optimization, still no reference gain

Both readouts completed and exited zero. Navigation prediction evaluation took
154.38 s; training-input diagnosis took 45.55 s. All populations and original
prediction-reproduction checks passed unchanged. Initial model/reference state
was exactly the same as in the 1e-3 fits. Final lower-rate states are JEPA
`2e08818bfd4169109938a6d6846ba63dca147dba59bded1c0e5555c83ab6328e` and supervised
`79993a3763f99a19ece62af7bab29a1d607a9a5c67673c48328a4c158ee2a8e9`.

| Model | Navigation prefix XY (mm) | Navigation whole XY (mm) | Navigation increment XY (mm) | Navigation yaw (degrees) | Training whole XY (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference only | 3.51 | 7.14 | 4.89 | 0.64 | 5.68 |
| Reference + JEPA, 1e-4 | 6.11 | 8.55 | 4.88 | 0.81 | 6.97 |
| Reference + supervised, 1e-4 | 4.27 | 7.88 | 5.01 | 0.74 | 5.99 |

All errors are RMSE; whole-window errors use the 700-ms horizon. Training metrics
use the original sampling weights and all available motion labels at that
horizon. Navigation windows remain overlapping and selected by the recorded
policies; the six runs are exposed development recordings.

Lower rate reduces both models' position errors substantially relative to their
1e-3 counterparts. It does not eliminate degradation of the reference: both
remain worse in whole-window position error in every one of the six recordings.
Both turn directions' prefix and whole-window errors are worse than reference.
There are small action-specific gains (for example straight-forward whole-window
error), and JEPA's pooled action-increment error is 0.016 mm lower than reference,
but these do not offset the full prediction errors. Original supervised remains
inferior to the new supervised model on each recording, while original JEPA is
better than the new JEPA on two recordings. None of this proves navigation gain.

At 300 ms on training data, reference/JEPA/supervised RMSE is 2.85/5.20/3.47 mm;
at 700 ms it is 5.68/6.97/5.99 mm. Thus optimization step size contributes to the
problem, but the correction still does not improve the reference even on its
training inputs. Neither new pair is promoted into navigation. All four new
checkpoints and both complete negative comparisons are retained; no simulator
run was started for them.

Next scientific step: freeze the learned latent dynamics and fit the motion
readout separately on the same training-only labels, with the reference-alone
and matched supervised comparisons retained. A fixed untrained-feature control
would distinguish learned representation information from a generic nonlinear
readout. This tests whether the latent representation contains useful motion
information that the jointly optimized output head failed to extract, before
another architecture change or large navigation batch. This readout experiment
has not been implemented or launched yet. Reliable navigation, independent
fresh-layout evaluation and deployment evidence remain outstanding.
