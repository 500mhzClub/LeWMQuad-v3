# Auxiliary native executed-motion diagnosis V1

Read both fully authenticated auxiliary native missions and their completed
readout. No new model call, training, controller modification or scene execution
is performed. Evaluate only forecast horizons whose entire prospective command
prefix equals the complete recorded executed command prefix. End the match at
the first different, missing or incomplete command. Derive targets from native
poses strictly downstream of the original controller, in the forecasting body's
coordinate frame. The rejected hold forecast at the terminal decision can be
compared with the actual zero-command drain when commands match.

For each matched first step, retain predicted and actual motion and XY/yaw
errors. Use the original observed map rotation and native initial anchor to
compare the sampled native centre curve with that forecast's nearest observed
occupied cell. This is a diagnostic for one already observed cell, not a new
all-obstacle audit, continuous-path or articulated safety certificate. Preserve
all original nominal checks and surface checks. Report small predicted margins
separately from actual measured centre clearance and prediction residuals.

Bind the source closure and all native/readout artifacts before the exclusive
`go2_auxiliary_depth_executed_motion_diagnosis_v1_attempt_001` output, and reverify
afterward. Use a single CPU process for this bounded two-case array analysis,
with 8 GiB available RAM and 256 MiB output allowance above the 40-GiB reserve.
Report hardware before/after. Infer no unexecuted action outcome or navigation,
independent-maze, real-time, hardware or overall-goal qualification.
