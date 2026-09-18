# Recorded controller timing diagnosis V1

Replay only the first 101 observations of the completed seventh maze-0 attempt
with its unchanged ConfirmedFloorRoundTripController and assigned model. Require
every complete controller decision to equal its saved raw-audited decision.
Profile controller.observe at fixed frames 20, 60 and 100 with cProfile; report
the other steps separately because profiling adds overhead. No native scene,
new command outcome, training, controller modification or timing qualification.

Exclusive root: go2_controller_timing_diagnosis_v1_attempt_001. Bind the completed
native result, all its recorded artifacts and source closure before and after
the diagnostic. Bind this source and protocol before execution. Stop on any
mismatch; preserve failures. Output consists of timings and profiler summaries,
not duplicate observations or trajectories.

One CPU process, one numerical thread, at least 4 GiB available RAM and 128 MiB
additional output allowance above the 40 GiB reserve. It may run beside the one
native experiment: its input is an immutable completed predecessor and it owns
a separate output root. Record current hardware and competition before/after.
This short diagnostic does not alter the native experiment's frozen settings.
