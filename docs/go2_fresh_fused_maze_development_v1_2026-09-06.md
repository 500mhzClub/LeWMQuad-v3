# Fresh sensor-controlled complete-maze attempt V1

One development attempt, not a final benchmark, hardware run or JEPA test. Its
purpose is to execute the now-integrated sensor-based controller on a fresh
complete discovery/return task and retain the first causal failure if it cannot
finish. Earlier whole-maze0/2, blank-scene fusion failure and all interface
diagnostics remain unchanged. This attempt does not qualify future gait safety.

## Frozen conditions

Exact source-declared specification:
`lewm/fresh_fused_maze_scene_development.py:specification` and `pack`. Eight cells,
seven connections, a degree4junction, dead-end branches, hidden south marker;
pitch1.8m. Start(0,0,.375), heading0. Physics/topology seed2026090610 and independent
appearance seed2026090611. One distinctive grayscale-texture appearance, episodic
route hypotheses. This is one new development layout, not multiple independent
test units or evidence of generalization.

The marker is the unchanged ordered red/blue panel definition. A separate static
camera at(3.6,-3.6,.43), facing negativeY, must render a detectable pair before any
physics steps. Save that RGB and camera/clock witness separately; never send it
to the controller or label it body-mounted sensing. Actual first body-mounted RGB
must have no marker detection, and evaluator geometry must show marker-centre
occlusion. Subsequent camera poses follow only the actual robot mount. Primitive
collisions, dimensions, floor, friction/solver, Go2URDF and learned gait/gains
retain their reviewed identities. The new visual adapter changes marker colors
only and verifies serialized meshes/native geometry.

## Controller and simulation supervision

Fifteen actual zero settling ticks establish the1.5s sensor anchor. Verify native
feet/four support groups, initial velocity within the supplied0±.02m/s ball and
an initial-body±.8m empty nonfloor prism. That prism expires at2.5s and is used
only to check setup, not as future clearance or a navigation map. Only the
explicit velocity prior and sensor packets enter DepthProposalNavigation.

Use one persistent controller, RGBDInertialRayMemory, gyro owner and episodic
memory throughout. The full initial/approach/scan/align/marker/return logic and
measured-depth proposals remain frozen. Raw depth rank/nulls are preserved.
Point hypotheses remain2mm/10mm and the uncalibrated position-scale budget80mm;
no resetting/relaxing it if it prevents the mission. A terminal/faulted controller
never resumes. No learned high-level navigation or JEPA selection is claimed.

Each requested command must be finite, vx in[0,.2]m/s, vy=0 and |yaw|<=.35rad/s.
Retain actual clip/slew and command acknowledgements. At every2ms physics sample,
preserve native disallowed-contact/fall/stability termination and additional
nonfoot-ground contact and body-speed>.3m/s guards. Guards use evaluator state,
never robot-policy inputs. They stop the attempt, not steer or repair it.

This is **externally supervised simulation development**. The inherited approach
controller's future body/foot/braking envelope remains unvalidated. Sampled floor
extensions and nominal turn checks do not prove traversability. Do not describe
this run as a fail-proof physical controller or hardware safety evaluation.
The outstanding prospective-envelope work remains required; empirical contacts,
execution error and first causal failure here can inform it. There is no hardware
actuation and no authorization to remove native stop conditions.

## Execution and outcomes

Bound the mission to the unchanged360s/36leg rules. Execute actual current sensor-
selected commands, not the old stimulus tape. At controller terminal/fault, issue
five actual zero ticks if native physics has not already stopped. Capture every
tail RGBD/body/gyro frame; ingest them only into a still-valid sensor owner without
restarting mission logic. A failed estimator is not reinvoked; log those tail
observations without pretending they were accepted. A native guard stop permits
no further physics, even to complete a tail.

Record all controller rows, proposed/executed commands, partial command ticks,
raw500Hz physics/contacts/gyro,50Hzbody/joints,10HzRGBD, task ledgers/memory, native
geometry/gain witnesses and outer capture/controller/command timing. Active-loop
timing is a full sequential cycle, not isolated deployment timing. Include actual
zero-tail timings separately. No forced arrival, known-map route command or
unrecorded observation reset.

Use the unchanged evaluation-only whole-task reducer: initially hidden marker,
actual discovery after >=.70m departure, return within.35m, stable/quiet measured
zero release, no contacts/native stops/sensor faults. Keep controller home
candidates separate from physically verified success; tail sensor failure vetoes
success. An incomplete/failed mission is a retained result. Independent replay
must verify sensor/contact reconstruction, decisions, actual command execution,
native guards, marker visibility, depth alignment, stopping and physical metrics.

## Custody and outputs

Fresh exclusive root:
`.generated/go2_fresh_fused_maze_development_v1_attempt_001`.
Bind the completed depth-proposal interface launch/result/audit and its artifacts,
all inherited source/input/native/OpenCV identities, plus this protocol, runner,
17focused fresh-maze tests and recursively discovered new sources before launch.
Verify again after acquisition. No sealed access, whole-tree export, old-source
edit, runtime-output overwrite, failed-controller restart, threshold search,
training or hardware. Preserve any infrastructure failure without silently retrying.
The goal remains full novel-maze science with matched JEPA/multistep/memory and
independent-layout/seed/robustness evidence; this single attempt cannot establish it.
