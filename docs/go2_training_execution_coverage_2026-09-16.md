# Training versus execution: action coverage and motion inputs

The completed pilot exposes a concrete training mismatch. None of the 4,010
available training contexts contains two future command changes. Online,
67,267/98,604 scored candidate tapes contain two or more changes. All 346
matched executed short-translation windows have seven-tick command tapes
absent from the original motion-labeled training population.

This is a development diagnosis on the already exposed two-maze pilot, not
new generalization evidence. Exact tape membership is not a requirement for
a model to generalize. It identifies a distribution difference worth testing;
it does not establish the cause of prediction error.

## What the model actually received

Training uses four causal RGB/body/control observations and eight prospective
100-ms command intervals. Body measurements include gyro, specific force and
joint histories, with validity and age. There is no explicit registered visual
pose or velocity input to the neural encoder. RGB history can implicitly
contain motion evidence; the model is not devoid of motion information.

The separate fitted control and neural XY residual correction use four causal
registered visual poses. They also were fitted on different, closed-loop
recordings. Therefore their better prediction accuracy is not an isolated
architecture comparison or proof that an added pose input alone fixes learning.

All-phase expansion retained 120 original recordings: 48 family and 72 moving
switch episodes. The switch contexts begin at the switch itself, frame 13,
so its command change appears at the past/future boundary, rather than inside
the future command sequence. Later windows come from long fixed suffixes.
Future changes are limited to turning arcs becoming straight and movement
becoming zero. Training has 3,284 zero-transition and 726 one-transition
contexts, and no visible zero/nonzero/zero single-tick pulse.

Runtime instead supplies three committed command ticks, then a four-tick
candidate, then zero. Terminal translations use one tick followed by zero.
An unchanged model must therefore compose future switches and braking events
that were never jointly represented in one training target window.

## Completed coverage measurement

Command equality is rounded to six decimals to ignore float32 serialization
noise. Original training has 43 distinct motion-labeled 700-ms tapes and 49
at 800 ms. Known-plan support and motion-labeled support coincide at both
horizons, although their numbers of available contexts differ.

| Matched executed group | Windows | Outside original 700-ms tape support |
|---|---:|---:|
| Hold | 669 | 13 |
| Turn | 6,396 | 1,372 |
| Translation | 8,390 | 3,812 |
| Short translation | 346 | 346 |
| All | 15,801 | 5,543 |

On supported/unsupported actual windows, raw-neural endpoint RMSE is
19.172/24.052 mm, corrected-neural 6.107/8.836 mm, and fitted-control
5.412/8.138 mm. These groups differ in action and state distributions;
the difference is not a causal estimate. Windows overlap. Unexecuted
candidate outcomes are not inferred, and actual command matching extends
through 700 ms only.

Reproducer: `scripts/read_go2_training_execution_coverage_development.py`.
Result under the configured artifact base:
`go2_training_execution_coverage_v1_attempt_001/result.json`.

## Training targets recovered without new simulation

Recovered frames 6 through 12 from each of the existing 72 training switch
recordings. These contexts precede the known switch by 700 through 100 ms.
All **504 contexts are available**, with 504 valid first-horizon motion
targets and 498 valid 800-ms motion targets. Six long-horizon motion targets
remain censored; no missing motion was filled from commands.

The original frame-13 targets were reproduced exactly for all 72 recordings.
Actual requested commands are checked against the recorded 2-ms trace by the
existing target derivation. Past and future image indices are checked against
recorded timestamps. Native pose is target-only. No geometry-transfer
recordings were read, no original files changed, and no additional independent
episodes were created. Future RGB tensors have not yet been materialized.

Adding these contexts increases distinct motion-labeled tapes to 205 at
700 ms and 238 at 800 ms. On the exposed pilot:

| Population | Count | Original 700-ms support | Augmented 700-ms support |
|---|---:|---:|---:|
| All scored candidates | 98,604 | 26,999 | 95,643 |
| Selected plans | 16,434 | 10,504 | 16,027 |
| Matched execution windows | 15,801 | 10,258 | 15,456 |

This covers about 97% of scored 700-ms candidate tapes, versus 27% originally.
It does **not** solve the full 800-ms task: only 31,337/98,604 candidate tapes
are covered there, because the final planned brake adds another transition.
No short-pulse executions were collected. Coverage alone establishes neither
better predictions nor better navigation.

Reproducer: `scripts/derive_go2_pre_switch_training_targets_development.py`.
Targets/result:
`go2_pre_switch_training_targets_v1_attempt_001/{windows,result}.json`.

## Next bounded learning experiment

First isolate the recovered prospective-switch training data, keeping the
neural inputs, architecture, objective, command interface and controller fixed.
Use full-input JEPA, direct and supervised-rollout models at the original
first seed, 2026091001, with the original 1,200-update, six-sample budget.
Preserve the original family/switch balance and per-trial draw totals; expand
each switch trial's shuffled context cycle to include its seven recovered
contexts. The original completed fits are the predecessor comparison. Freeze
the schedule before fitting; do not choose contexts from prediction errors.

Read out all three methods on the same development transfer situations,
including departures before switches, with separate stable-motion, switch
and braking results at 300, 700 and 800 ms. Compare against command-based
motion controls on the same targets. This tests a data change, not JEPA
superiority or a matched-data comparison against the older fitted control.
Use the final fixed-budget checkpoint, without best-checkpoint selection.

If prediction improves, test actual online choices and round trips against
the unchanged predecessor and simpler control. Independent new mazes remain
necessary; these two exposed pilot mazes cannot establish generalization.
In parallel with interpretation, keep short pulses, multiple transitions,
obstacle interaction and explicitly observed motion as the remaining data/model
questions. Do not launch another three-seed unchanged-task sweep or tune more
terminal controller rules to hide the prediction weakness.

The schedule is now materialized at
`go2_pre_switch_training_schedule_v1_attempt_001/schedule.json`, produced by
`scripts/prepare_go2_pre_switch_training_schedule_development.py`.
It covers every one of the 4,514 available contexts in 7,200 draws, preserves
all 600 original family batches exactly and preserves every original trial's
total draw count. All three methods share it. All three fixed-budget fits and
the same-target transfer comparison are now complete; see
`docs/go2_pre_switch_learning_result_2026-09-16.md`. The result is mixed, with
braking regressions in all three methods and command integration still better
overall. No successor navigation trial has been run.

The coverage reconstruction was checked against the production candidate
builder for 216 prefix/action/pulse combinations. All 4,010 original available
training tapes match their source action schedules. These are focused checks
of the scientific counts, not a rerun of historical infrastructure audits.
