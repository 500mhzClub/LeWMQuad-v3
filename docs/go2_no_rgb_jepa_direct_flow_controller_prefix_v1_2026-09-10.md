# No-RGB JEPA direct-flow full-controller prefix V1

The existing direct-flow observer recovered the previously failed frame 859
after exactly reproducing frames 0–858. It selected the auxiliary camera with
31 inliers and measured bridge count seven of the unchanged ten-frame budget.
Test whether the current anchored controller can consume that measured pose.

Use a distinct DirectFlowResidualAnchoredController. It inherits the original
ResidualAnchoredContinuationController, replacing only its visual observer and
adding explicit controller identity metadata. Preserve the learned no-RGB JEPA
model, calibrated translation correction, floor registration, mapper, memory,
mission, residual update, action selection, all geometry checks and budgets.
Do not introduce recent-anchor retention, partial-height rules, or performance
optimizations in this experiment.

Replay the actual paired sensors and fast gyro from frame zero through 859
inclusive. Every public packet and candidate raw visual receipt must match the
completed observer replay. Every complete controller decision through frame
858 must equal the original recording after only the two declared controller
metadata changes. Bind and compare all actual preceding command endpoints.
At 859, preserve the original failure evidence and validate live (unserialized)
raw visual and floor-registered evidence. A recovered controller requires a
complete forecast selection and a command consistent with its selected action.
Otherwise preserve its terminal reason and zero command. Stop at 859 in either
case; consume no later observation and execute no new command.

Authenticate the completed observer result, its entire source closure and all
original episode inputs before and after. Reexecute the original full model,
training-input and native-predecessor admission before and after this replay.
The model must remain at its assigned state with no gradients. Run one CPU
controller replay only after the existing late-history profile owner ends and
before starting the pending paired timing replay. Require 48 GiB available RAM
(16 GiB replay plus 32 GiB concurrent native allowance), four physical CPUs,
40 GiB artifact reserve plus 1 GiB output allowance. OpenCV, Torch and BLAS use
one thread. The existing native queue remains unchanged.

Use an exclusive attempt with terminal failure preservation and no retry or
resume. Source-only preflight creates no output and loads no model. This is
prospective controller replay, not evidence of navigation beyond the original
failed observation. A successful boundary requires a subsequent fresh native
episode to test continuity, anchor recovery and physical navigation.
