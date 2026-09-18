# Causal executed-step residual diagnosis V1

Authenticate the completed exact-target and executed-horizon native pairs,
their complete raw audits and the executed-horizon readout. For each recorded
selection, label only the action whose complete known command prefix actually
executed. Keep every available prefix horizon; use the first 100 ms for this
diagnosis. No outcome is assigned to an unexecuted action or incomplete step.

Reconstruct the observed body displacement from consecutive admitted public
joint poses. A tick-t prediction residual is available only at tick t+1. Before
each forecast, subtract the mean observed XY residual from the preceding eight
ticks, pooling executed actions. Use zero if no residual is available; gaps
reduce sample count, and older residuals expire. Eight ticks are the existing
0.8-second plan horizon. This single fixed window is not a hyperparameter sweep.
The current and future observations cannot affect the current correction.
Keep yaw and contact predictions unchanged; no model or controller is modified.

Only after computing that causal diagnostic, compare original/corrected XY
predictions with evaluator-only native outcomes. Report all cases and action
groups, and the existing exact-final-target vs intermediate/view phases, with
sample counts, original/corrected mean and maximum errors, mean residuals,
and observed-vs-native displacement agreement. Preserve deterioration as well
as improvement. This cannot establish unexecuted actions' accuracy, calibrated
uncertainty, closed-loop benefit or navigation success.

Exclusive `go2_causal_executed_residual_diagnosis_v1_attempt_001`, one CPU
thread/process, 8-GiB RAM and 256-MiB output allowance above the 40-GiB reserve.
Freeze/reverify complete source, native and readout bindings. No fitting,
checkpoint selection, native execution, runtime correction, retry or resume.
