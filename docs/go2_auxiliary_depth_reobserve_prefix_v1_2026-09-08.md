# Bounded active reobservation prefix replay V1

Use both fixed corrected seed-2026091001 models on their authenticated auxiliary
native mission recordings. Change only the treatment of a valid no-feasible-action
selection: request zero, keep acquiring and processing both cameras, and reconsider
at the next observation. Permit at most ten consecutive zero wait commands inside
the unchanged 240-tick mission limit. Reset the counter only when the original
selector admits an action. The eleventh consecutive infeasible observation retains
the original terminal failure. Do not resume any already latched terminal state,
sensor/model failure, view exhaustion, mission limit or arrival candidate.

Keep the original observer, auxiliary map, model, training-only coefficients,
selector and every surface/nominal constraint. Zero waiting is no clearance or
stopping certificate. The direct predecessor's actual zero drain crossed the
nominal boundary and remains a recorded failure.

Replay each new controller twice from a fresh model and state. Require exact
fresh replay and unchanged model state. Require all original decision fields,
except the controller name, to match before the predecessor terminal decision.
Record every proposed wait/recovery. Stop the replay immediately after the first
command that differs from the executed tape. Distinguish the earlier terminal-
policy intervention from the later command difference. The recorded zero-command
continuation is shadow evidence after the intervention; infer no prospective
recovery or goal arrival. If the tape ends first, record that explicit bound.

Authenticate native/readout/diagnostic artifacts and source closures before the
exclusive `go2_auxiliary_depth_reobserve_prefix_v1_attempt_001` root, and reverify
afterward. Use one CPU replay process and one numerical thread with an 8-GiB RAM
allowance and 256-MiB output allowance above the 40-GiB storage reserve. Record
hardware before/after. No native execution, training, independent-maze or hardware
qualification is performed.
