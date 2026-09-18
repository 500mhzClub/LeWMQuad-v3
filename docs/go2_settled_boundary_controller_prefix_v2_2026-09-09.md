# Unchanged settling controller with explicit target-reset comparison

The V1 replay failed at frame1866 because its comparison omitted one consequence
of the declared mission change: MissionTargetWaypointSelector.set_goal resets
planner_mode to NEW on changing the goal. The original reaches RETURN and sets
home; the candidate remains OUTBOUND with the prior WAYPOINT mode. Both hold
zero command without a new selection. The original failure remains terminal:
launch cf320b97b50fd5e9d6369608d690147cdb1cce9bb15a19d54cfb89bc126f9f1e,
failure f7dedb15dabba46667624a63434b4ba0bcf01d7b294c501dc19a095bf5f138ef.
No complete model/source/input postcheck was reached in that failed attempt.

Run a fresh complete public-sensor/controller replay from frame0 through the
first mission behavior change1866 in the exclusive root
go2_settled_boundary_controller_prefix_v2_attempt_001. Controller, mission,
observer, maps, forecasts, contact checks and numeric/model settings are
unchanged from the failed V1. No checkpoint/resume or simulator is used.
Retain the original failure and bind its launch, failure, mismatch and partial
stream; inherit and validate its frozen source identities.

The new comparison delegates to the frozen mission comparison after checking
the sole additional field, planner_mode. A difference is allowed only on the
original OUTBOUND_TO_RETURN transition with confirmed original arrival and no
candidate arrival. The candidate must retain the preceding candidate's outbound
goal and mode, the original target must become home with NEW mode, and both
current decisions plus the immediately preceding candidate must be held zero
commands without selections. Require consecutive frame/timestamps. All other
complete fields remain exact outside the original declared mission fields.
Stop before any subsequent observation/decision. Persist the full candidate
decision and comparison, explicitly recording the target-reset mode difference.
Do not call this equality outside mission fields alone.

Use the same1867 ninth public RGB-D/auxiliary packet population, command tape,
model admission and saved mission comparison as V1. Verify complete source,
input and model identities before and after. The ninth native audit/readout are
now completed; any prospective native run must bind them and use this revised
comparison and actual completed V2 result. V1 cannot supply a success result.

One CPU process and one OpenCV/Torch/BLAS thread alongside the independent
packed-owned equivalence replay. Recheck topology/affinity, CPU/GPU/VRAM, RAM,
competition and storage. Require8GiB available RAM and1GiB output above40GiB
reserve; these are capacity admissions, not enforced OS limits. No parallel
native scene, training, altered sensing, gate tolerance change or optimization
adoption. Full-loop timing, strict visibility, arrivals, returns, independent
layouts, matched comparisons and hardware remain unqualified.
