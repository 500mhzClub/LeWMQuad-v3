# Startup cameras: rendered, but no sensing configuration qualified

The fixed six-view study completed 12 actual RGB/depth renders in two zero-step
scenes at the recorded fitting startup posture. The robot-visible scene uses
all 17 URDF visual instances, assembled from seven exact COLLADA assets
(384,727 vertices, 398,632 triangles). The control has identical environment
mesh bytes and camera poses but no robot. No physics, gait, training, validation
trial or navigation was run. The earlier mesh material-conversion preflight
failure was fixed before launch; no launched source or outcome was replaced.

## Nominal design measurements

| Camera | Full-frustum shapes /27 | Entire enclosing rectangle floor-observed /27 | Observed footprint samples /675 |
| --- | ---: | ---: | ---: |
| Forward | 0 | 0 | 0 |
| Front down60 | 0 | 0 | 4 |
| Front down90 | 10 | 0 | 4 |
| Overhead down90 | 21 | 0 | 0 |
| Left outboard down90 | 11 | 2 | 283 |
| Right outboard down90 | 11 | 2 | 206 |

The outboard pair's sampled union is 420/675, with 255 missing samples. This is
not continuous coverage, calibrated terrain evidence, or permission to walk.
Visual inspection confirms body/leg occlusion in the saved images. An overhead
camera sees the top of the robot, not floor through it. Outboard housings,
installation, changed width/mass and hardware availability are unmodeled.

## Independent ray audit — negative result retained

An independent trimesh intersection algorithm checked a fixed 20x15 pixel grid
per view against the saved robot triangles and analytical floor. Of 1,511 rays
whose background is in-range floor, 32 differ from the two-sided triangle
prediction by more than the unchanged 1-mm tolerance: 0/1/29/1/0/1 across the
six views above. The maximum disagreement is 316.435 mm in front-down90.
All 300 grid rays from EACH of the three front-mount views intersect robot
geometry before the native .05-m near clip. No such near intersection occurred
in the other three grids. Consequently, the front camera/URDF/aperture model
does not yet justify a physically unobstructed optical path. Do not claim the
existing forward camera is self-occlusion calibrated or that clip-discarded
geometry is absent in reality.

A separately bound POST-HOC diagnostic removes back-facing triangle hits to
test the renderer's culling convention. It explains the 30 front-down
disagreements, leaving two unresolved: overhead pixel (528,336), 19.134-mm
depth discrepancy, and right-outboard pixel (144,208), 1.332-mm discrepancy.
This does not revise the original audit criterion or qualify visibility.
Raster sampling/edge behavior and actual optical aperture geometry need an
explicit new verification; do not enlarge tolerance or silently move the camera.
The audit is sparse and shares saved visual geometry, not a hardware proof.

## Scientific interpretation and next implementation

Do not turn this into an endless search for a camera that sees every point
under the robot. The 27 floor-footprint rectangles are a conservative diagnostic,
not a theorem that navigation requires complete underbody floor imagery.
Conversely, simply deleting that guard would make missing terrain look safe.

The next controller contract must distinguish three obligations:

1. Current support: measured foot loading plus joint/body state supports a local
   contact hypothesis, subject to slip, calibration, timing and support-height
   checks. It does not establish a continuous floor between feet.
2. New support: proposed foot landings and their uncertainty need observed
   traversable terrain or a separately declared, justified terrain assumption.
3. Body/leg sweep and stopping: observed non-floor geometry and validated
   action-conditioned motion/braking must support the actual swept volume.

Implement the planned separately typed support-sensor diagnostic next. Use
only identified foot channels and all their loads, not evaluator ground/object
labels. Raw hardware fields remain uncalibrated counts; ideal simulated loads
must be labeled as such. Preserve missingness, stale values and invalid
contacts; do not install this offline diagnostic as a closed-loop result.
In parallel task planning (not concurrent agent work), retain a bounded camera
aperture/raster verification prerequisite before any new view enters a policy.
No new maze run should be launched with the currently unqualified views.

Then validate short sensor-supported start/forward/turn/brake execution,
persistent branch/place memory and full exploration/backtracking/home return.
The required matched geometric/supervised/JEPA, predictive-training, genuine
multistep-rollout, memory, independent-layout/seed, timing and later hardware
evidence remains missing. The scientific goal is active, not achieved.

## Verification and identities

Focused geometry/camera tests: 19 passed (15 new tests and four existing optical
readback tests). Full regression: 2,207 passed across 177 explicit test files in
180.09 s. Unit tests do not clear the ray failures.

Output: `.generated/go2_startup_self_visible_camera_development_v1_attempt_001`.
Launch binds 543 source paths, 11,306 inputs, prior native/OpenCV identities and
seven separately verified visual assets. This is not a whole-tree export or a
new claim of complete third-party dependency closure.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 6a3c2fbdddb7ffd2db550e7e659db18ef9643c53fdaee09af467106e3ec46ec9 |
| result.json | 5df4a458504ef413b92e3a232c663ddccbda318636cd76b69b58fcd8f74ddc15 |
| sparse_ray_audit.json | bc1df0b190107e246a21ceda56d09d0e59c49bf1b1bb86bd0ce6e6063fc6df42 |
| ray_disagreement.json | 71a4bfe1b6be79e45a4dc9fffb1a98bf253dadb73568ca879b8eac250b019e3a |
