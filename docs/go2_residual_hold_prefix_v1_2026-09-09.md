# Hold reconsideration: bounded prospective controller replay

Use only the completed, raw-audited residual-first-interval maze2 pilot in
go2_residual_first_interval_maze_pilot_v1_attempt_001. Require its actual final
result SHA-256, complete artifact/source/environment verification, unchanged
assigned JEPA model, and the admitted physical prefix against original maze2.
A collection result alone is insufficient. Negative scientific outcomes are
retained and do not bar development analysis.

Instantiate ResidualHoldFeasibilityController from the original initial state.
Preserve the existing no-action recovery. Reconsider a selected feasible
waypoint hold only when another action has strictly higher original utility and
passes the causal corrected-first-interval geometry, all eight path segments,
original and corrected surface gates, and phase restrictions. No hold timeout,
forced translation, changed clearance radius, score weight or forecast target.
The first XY correction remains the existing last-eight-observed-residual mean;
later points, yaw, contact predictions and model weights are unchanged. This
does not supply a calibrated model-error or physical-clearance bound.

Replay sequential paired public packets from observation zero, at most 3004
observations, ending at the first changed command or original terminal. Compare
every preceding complete decision, raw forecast, actual command, observed map,
mission and residual receipt exactly. At a changed hold, only the declared
selection choice, its feasibility receipt, requested command and selected action
may differ. Do not consume the next recorded decision or packet. Check public
arrays and model state before/after, and absent gradients. A replay with no
intervention is a valid negative finding. Never infer changed-command outcomes.

One CPU replay, single OpenCV/BLAS thread, 8GiB RAM admission and 256MiB output
allowance above the existing 40GiB reserve. Refresh measured hardware before
launch. It may overlap one separately owned native scene with measured memory
headroom. No new native scene, training, GPU requirement, sealed access, source
export or real-robot action. Preserve exclusive attempt output and any failure.

Output: go2_residual_hold_prefix_v1_attempt_001. Fresh physical continuation and
independent-layout navigation remain necessary regardless of replay outcome.
