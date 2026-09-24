# Fixed startup camera design V1 — development, render only

Before rendering, freeze six 640x480 candidate optical poses: original forward,
same front mount pitched down 60 and 90 degrees, overhead (0,0,.20) down90,
and left/right outboard (0,+/-.30,.20) down90, in metres relative to body.
Intrinsics remain the original focal length and principal point; each mount is
a NEW calibration identity. Native clipping .05..200 m, modeled depth .2..5 m.
Do not adapt candidates, ranges or 1-mm floor-comparison tolerance to outcomes.

Use only fitting trial initial sample 749 from the already recorded longer
motion tape. Native pose and floor identity are scene-generation/evaluator
information, not navigation inputs. Render two independently built zero-step
scenes: actual articulated URDF visual meshes present, and identical environment
without robot. Bind the seven exact COLLADA assets; no external texture fetch.
Convert each untextured material's constant RGBA to vertex colors before merging
its geometry, retaining scene graph and URDF visual origin transforms. This is
actual visual geometry with simplified diffuse appearance, not hardware imagery.
No physics step, controller, training, checkpoint or held-out access is needed.

Persist mesh, camera-pose/intrinsics, RGB and raw optical-depth witnesses. Verify
native camera transforms and raster encoding. Classify floor only where the
background and visible render agree with analytical floor depth within 1 mm.
Closer robot returns occlude floor even when below the modeled sensor minimum.
Missing/out-of-range readings remain unavailable; background walls are not floor.
Clip-induced invisible surfaces closer than .05 m are not certified absent.

Score the 27 instantaneous collision-shape world-axis floor-footprint rectangles
separately from visual meshes: full frustum, conservative enclosing pixel
rectangle entirely floor-observed, plus fixed 5x5 samples per shape. Report the
outboard-pair sampled union, explicitly NOT a continuous coverage certificate.
This local rectangular projection assumes the known simulated z=0 floor only
for sensor-design scoring; it supplies no unseen-floor prior to a controller.

No housing/mount geometry, changed width/mass, feasible installation, sensor
noise, calibration, contacts, future gait or terrain generalization is proved.
One posture is not a deployment configuration qualification. Preserve negatives
and terminal failures; no retry in this output. Next work must return to a
sensor-supported startup and prospective forward/turn/brake validation before
full maze memory and matched geometric/supervised/JEPA scientific comparisons.
