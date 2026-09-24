# Whole-task RGB marker search and tentative return: fixed development V1

This protocol fixes four fresh continuous physical runs before execution: two
connected layouts, each paired across episodic-route memory and local-only
navigation. The local method is fixed_forward in every run. This tests whole-task
integration and a memory ablation, not a JEPA advantage, learned-policy comparison,
independent final benchmark or hardware transfer. Protected material stays closed.

## Scene population and observations

`whole_task_scene_development.py` defines the two exact cell-connection trees:
north_dogleg and south_branch. Their pitch is 1.44 m, wall thickness .08 m and
wall height 1.4 m; corresponding room clear width is 1.36 m. Both have a branch
choice and a terminal physical ordered red/blue marker, with its centers occluded
from the nominal initial camera. Runtime initial RGB non-detection and actual
initial camera-to-marker center occlusion are evaluated separately and both are
required for this task's success. A violation remains a failure, not an invitation
to move the marker or discard the case. Seeds 2026100300 and 2026100301 are paired
within layout. Initial pose is (0,0,0); the actual post-settle position defines home.

These are development layouts, not independent held-out evidence. The wider domain
does not erase preceding 1.2 m false-arrival or scan-clearance failures. The
constructor receives only wall boxes/spawn for physics; no teacher route is run.
The controller receives only actual policy RGB/body packets, causal fast gyro,
robot collision geometry, its declared arm and local-method identity. No maze
map, cell, beacon location, target pose or destination image enters decisions.

## Continuous runtime and matched ablation

Both arms use the same current marker detector, .02-rad persistent-feedback
alignment, stopped-observation local traversal, camera/body histories and gait
limits. Initial observation uses four actual frames, followed by observed bearing
alignment and fresh local-executor warmup. Between local attempts, observe scan
views and choose left, forward, right, then reverse relative to the latest
departure direction. An acquired current left branch may interrupt a scan;
otherwise full-scan evidence may suggest alignment, but translation still requires
an independently acquired current exit agreeing within .35 rad.

The marker observer runs on every camera tick. Discovery switches the task to
return. An unstarted outward selection is canceled; an already executing outward
leg finishes its bounded local observation before the return decision. No selected
outward action is retroactively relabeled as a return.

Episodic memory records all exploration movements, including reversals and visits
to dead ends, as a chronological unverified route. It does not identify cells or
merge revisits. Return attempts retrace that history using opposite recorded
bearings, each with fresh exit evidence. Provisional return arrivals pop only the
route hypothesis; emptying it yields HOME_CANDIDATE_ROUTE_HYPOTHESIS, not a verified
home fix. This simple backtracking can redundantly revisit explored detours.

The local-only arm has no persistent visit/route history. It retains the current
scan, current local executor, most recent incoming bearing, task/discovery state,
bounded counters and the actually acquired starting RGB reference. Its first
return selection reverses the latest incoming bearing; subsequent selections use
the same left/forward/right/reverse local ordering. Audit ledgers are retained
but never consulted for its route selection. This is a declared local baseline,
not a claim to match the strongest classical mapping or navigation system.

Both arms also have the same provisional home-image cue: 4x4 block-mean RGB L1
distance <=.02 to the actual initial image, initial gyro-frame heading error
<=.2 rad, and at least .3 s of consecutive quiet-body observations while outside
an active traversal. This uncalibrated cue can be wrong; false claims remain
measurable. Memory additionally provides route-empty stopping. Thus this ablation
includes route-based stopping as well as return direction selection, not only
place recognition. Memory and cue outputs cannot assert a trusted map edge.

## Budgets, stops and physical outcome

Each run settles for 1.5 s (750 native 2-ms rows), then uses 100-ms decisions for
at most 360 s/3601 decisions and at most 36 allocated local-executor instances.
The camera population cap is 3606, including the final .5 s zero-command release.
Local alignment, scanning and traversal retain their own inherited deadlines.
All native contact/body stops terminate physics immediately. Sensor faults latch
and request zero; never reset within an episode or restart a completed/failed root.

Primary physical task success requires all of:

- No marker detection in the first four actual stationary decision frames, and
  both physical marker centers occluded from the actual initial camera by a wall.
- Actual three-frame marker discovery after the robot has moved at least .70 m
  from its post-settle home position. No geometric/proximity detection substitute.
- Final actual position within .35 m of home, stable body height/roll/pitch for
  .5 s after discovery, actual zero requests over that interval, and planar speed
  <=.10 m/s and yaw rate <=.25 rad/s over its last .2 s.
- An actual 250-row zero-command release after the last decision, with no native
  stop, contact or sensor-contract failure anywhere in the run.

Physical success does not require the controller to recognize home. Report its
home claim, false home claim, actual return without a claim, first stable physical
return time, final/max home distance, base path length, elapsed time and every
failed/incomplete local attempt. An initial-marker-visible early home claim is
not a hidden-beacon task success. No threshold will be fitted to these outcomes.
Provisional visit IDs are not certified cell transitions; retain the known local
arrival limitations rather than relabeling arbitrary event IDs as recognized places.

## Evidence and audit

Freeze source/protocol/tests and all inherited marker/memory/local-execution
bindings before launch. Retain raw physics and native contacts, ordinary and
fast sensors/histories, actual RGB/camera frames, native static-box identities,
actuator readback, exact controller/marker/memory outputs, command tape, terminal
ledgers/memory and process logs. Pair the settling physical/history prefix within
each layout before any arm-specific action. Do not edit bound ancestors.

Full audit must replay every decision and marker/memory transition, verify native
sensor/contact/command/camera accounting and independently reduce physical return.
Static scene-pack coordinates remain exactly bound; native position readback is
compared to their float32 representation at 1e-7 m absolute tolerance. Native
box records require three dimensions followed by exactly four zero padding values.
Only declared model timing fields may differ if a later explicitly scoped learned
arm is added; no such arm is in this four-run population. No follow-up trajectories,
retries, geometry changes or rescoring to make this V1 pass.

Regardless of this pilot's result, the final objective still requires reliable
observed local geometry, robust sensing, matched predictive-training and online-
rollout tests, broader independent novel mazes and bounded real-platform evidence.
