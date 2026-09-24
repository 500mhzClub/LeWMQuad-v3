# Sustained hold-reorientation maze 02 native pilot V1

This is one fresh development execution of the sustained-turn controller on
reused maze 02, with the original full-JEPA model and seed. It follows the
completed 407-observation paired raw replay and the entire original five-stage
diagnostic queue. Neither source preparation nor a replay prediction is a
physical outcome.

The only controller successor is SustainedHoldReorientationController. Its
bounded recovery continues a freshly admissible turn toward the observed
heading target for at most eight commands. All original sensing, model state,
mission, action bank, surface and 800 ms nominal-path vetoes remain in force.
The original 3000-navigation-tick budget, 100 ms command interface, deterministic
renderer, robot geometry, terminal drain and joint physical/visibility success
criteria are unchanged. No queued diagnostic policy change is adopted.

Admission requires the exact completed raw replay and final extended-budget
waiter result hashes. It reconstructs the original raw-input admission and
authenticates the completed five-stage queue, including full original budget
input admission and all scientific failures. Live original owners, incomplete
results, changed source or artifact bytes, mismatched model or predecessor
identities, and incomplete physical intervention evidence reject execution.

One spawned process collects the full raw episode, reconstructs its commands
and model decisions through the raw auditor, and compares the actual native
prefix with the prospective raw replay. All 407 public observations and
candidate decisions must match. All 21,050 physical samples before changed
command 406 must match the original hold run. The first changed command must
actually complete, requiring at least 21,100 physical samples; subsequent
physical outcomes are measured by the new run and are not inferred from the
old trajectory. The comparison includes 404 model forecasts.

The original assigned-model loader verifies the exact correction/checkpoint
bytes and final model state. The native scene runs alone, with one worker,
one task per process, single-thread OpenCV/BLAS, and physics paused during
computation. Resource admission and periodic monitoring remain enabled.

Source preflight verifies the recursive source closure and resource envelope
without admitting runtime inputs, constructing a model or creating output.
Full preflight additionally admits all inputs and the actual assigned model,
but creates no output. Execution exclusively creates
`go2_sustained_hold_reorientation_maze02_pilot_v1_attempt_001`; failures are
preserved and there is no retry or resume. No independent study policy,
navigation qualification, real-time qualification, hardware qualification,
training, or goal-completion claim follows from this pilot.
