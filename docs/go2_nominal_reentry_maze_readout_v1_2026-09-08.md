# Completed nominal-reentry maze readout V1

Read only the completed and raw-audited recovery attempt using
`scripts/read_go2_nominal_reentry_maze_pilot_v1.py --native-result-sha256 HASH`.
Output is exclusive `go2_nominal_reentry_maze_readout_v1_attempt_001`.
Authenticate the native result, its artifact bindings, frozen source/native
dependencies and launch before and after readout. Require the completed native
comparison to establish identical physical and public observations through the
first changed command. Preserve the original navigation outcome and identify
this as reuse of development maze 0, with zero new independent layout executions.

Retain the first-maze readout's full decision trace, native goal-distance and
path summaries, actual mission arrivals, first and last infeasibility evidence,
raw measurement gates and all recorded acquisition/control/receipt timing.
Additionally identify every actual nonzero nominal-reentry request, requiring
the selected eligible candidate and original command tape to agree. Compare
that candidate's original corrected-model first 100 ms XY prediction with the
native displacement over exactly the executed 50 physics steps, rotated into
the native body frame at interval start. Native poses are evaluator-only.
Incomplete intervals are censored; no unexecuted endpoint is inferred.

Track whether a recovery sequence is subsequently followed by an ordinary
selection passing the original radius 0.45 m nominal path gate. Its first
segment includes the current observed point, so the segment minimum provides
only a lower bound on current clearance in that observed map. This is not
physical clearance certification, predicted recovery success, goal arrival,
or an independently measured memory/planning advantage. Preserve unresolved
recovery starts and every failed navigation outcome.

Require 8 GiB available RAM and 128 MiB above the unchanged 40 GiB storage
reserve. One CPU readout process is appropriate: one completed case has one
ordered decision stream; there is no training or independent native job to
parallelize. Record hardware admission. No GPU or native scene is created.
Preserve paused-physics and ideal-sensor limitations, and retain any readout
failure separately from the immutable native attempt.
