# Next: controlled visual information, then complete missions

Update: the [controlled appearance assay is complete](go2_appearance_information_result_2026-09-06.md).
Neutral0/13, repeated13/13 and distinctive13/13 with unchanged observer rules,
bit-identical matched depth and independently reproduced sensor predictions.
This resolves the rendering-information question, not independent motion or
navigation. Implement the [fusion and fresh-motion work sequence](go2_rgbd_fusion_and_fresh_motion_next_steps_2026-09-06.md)
next; retain the original diagnostic below as history, not a request to rerun it.

The [V1 recorded diagnostic](go2_rgbd_correspondence_motion_diagnostic_v1_result_2026-09-06.md)
found no usable RGB point correspondence in A/B. B's thirteen depth-weak pairs
have zero detected keypoints. Source explicitly disables textures. This updates
the post-B plan: further tuning of match acceptance on those images is not the
next experiment.

## One controlled rendering question

Build a new appearance-only development path. Retain collision planes/boxes,
their transforms, friction/materials, robot geometry and gains. Do not enable the
existing texture branch that replaces collision boxes with meshes. Use separate
visual-only surfaces or a verified material/UV path whose visible geometry
coincides with the original physical surfaces. Read back actual native geometry,
roles and materials rather than assuming `collision=False` proves the whole
interface is unchanged. Avoid overlapping/z-fighting surfaces or depth offsets.

Compare three preregistered appearances with the same physical geometry and
camera poses: neutral textureless, repeated-pattern ambiguity control, and
nonsemantic independently seeded distinctive surface appearance. Randomize
appearance independently of maze topology, robot start, marker location and
training/evaluation role. No coordinate codes, goal arrows, unique location IDs
or prior map input. Random procedural surface color/texture is test data, not
evidence of realistic hardware texture robustness.

Use new output identities and actual rendered RGB/depth. Do not edit old pixels
or splice new RGB into A/B and report a changed A/B controller outcome. A
render-only matched-pose experiment may use evaluator poses to place a camera,
but those transforms must never enter the correspondence estimator. It would
isolate appearance, not be independent motion or navigation validation. Verify
depth/surface agreement and visible content, not just requested material flags.

Keep the frozen V1 feature/tracking/3-D rules. Compare accepted/rejected pairs,
original plane-depth rank, point support, correspondence error and latency for
all conditions, including low/repeated texture negatives. A synthetic positive
already exists; the next evidence must use the actual renderer and sensor
calibration. One controlled diagnostic should resolve this interface question;
do not replace the whole mission with repeated appearance tuning.

## Integrate and challenge the full task

With independently supported visual constraints, implement an explicitly sourced
RGB-D/inertial fusion update that can constrain a plane-depth weak direction
without changing the reported original depth rank. Preserve missing-data states,
accumulated global uncertainty, epoch/availability clocks and correspondence
provenance. Reject dynamic-object/occlusion/aliasing failures rather than calling
every image match a static-world motion measurement. Calibrate or clearly label
any new error model; no automatic return to zero uncertainty on a good frame.

Test on a fresh motion/layout, then connect to the existing complete-task
controller with one persistent observer/memory, uncertainty-reserve observation
actions, explicit starting/support assumptions, factored non-floor/ground/unknown
evidence, junction decisions, backtracking, marker detection, verified return and
actual stop tails. This must leave the tiny calibration arena and terminate on
the complete mission's criteria. Keep all original0/2 failures unchanged.

Use the frozen local response and both persistence baselines honestly; neither
its modest body prediction improvement nor image tracking is a JEPA navigation
policy. Compare matched geometry/supervised/JEPA training and genuine no/one/
multistep online action rollout, plus memory-disabled/enabled conditions on
independently generated layouts/seeds. Include textureless and repeated texture
as robustness conditions, not just a favorable textured scene. Preserve equal
physics, sensor, controller and compute conditions across arms. Measure complete
mission success, collisions/falls, false home declarations, interventions,
uncertainty stops, path/time/effort proxies and full-loop latency.

Profile shared preparation and eliminate redundant queries with recorded
reference equivalence, or jointly rederive a slower sensor/control cadence.
Observer-only49ms does not meet the full10Hz control contract. Hardware evidence
still requires calibrated real sensors, resolved gait/model warnings and bounded
operator-supervised access. Final benchmarks remain outside the model-facing
checkout. The full scientific goal remains active and unachieved.
