# Conditional visible-surface interval diagnosis V1

This evaluator-only development step constructs separate physical surface depth
ranges over an explicitly supplied pixel region. It does not validate that
region as a renderer/sensor error bound, change a policy, or relax any old gate.

Project front-facing opaque box faces and the ground into the requested pixel
rectangle. Optical inverse depth on each plane is affine in pixel coordinates.
For every face, subtract the convex projected region where another face has
greater inverse depth (is nearer). Split the remaining polygon into convex
pieces, and evaluate inverse-depth extrema at their vertices. Preserve separate
depth intervals instead of filling an empty gap between foreground/background.
Coincident faces may duplicate reported area; areas are not a union-coverage
certificate. Zero-area boundary ties and floating-point error bounds remain
uncertified. A 1e-9 m positive projection clip is used; native near clipping
never removes physical occluders from this calculation.

Evaluate the fixed synthetic thin-post and near-plane fixtures at supplied
pixel radii 1/256 and 1/2. The small region must reject a missed post; the larger
region can see both post and background, exposing the importance of validating
the assumed angular bound. Both must reject fabricated 1.5 m empty-gap depth.
The near-plane occluder must reject background at both radii.

Bind completed native result
0f40eb01e5d5feaf004d0c0e98a9b6d712791dcd676b6013ac965fbb1603ffb8
and completed edge diagnosis
73d6077b81430ebe78518634568a2fe4f5295966d551ca258c0c53f0a1f0b9a1.
Reconstruct the original strict score at frame 909 from raw depth and geometry,
requiring exact agreement with the completed diagnosis and preserving its false
result. Evaluate pixel [260,428] under the same two assumed regions. A supported
native depth is conditional geometric evidence only, not a passing strict score
or a complete-frame sensor audit. No native geometry enters the policy.

Eleven tests pass, covering polygon subtraction, actual occlusion, two separate
surfaces, impossible gap values, near-plane retention, ordering, invalid radii,
and 392 independently traced interior rays across eight rotated/overlapping
scenes. A broad rotated-occluder test initially exposed a numerical clipping
sliver that left a 1.14e-13 pixel² background fragment; clipping now explicitly
solves one coordinate on its boundary plane. This fix preceded source freeze.
Tests are examples, not a proven floating-point enclosure.

One CPU diagnosis may accompany the existing native scene; require 8 GiB RAM
and 128 MiB above the unchanged 40 GiB reserve. Bind sources and completed
artifacts before/after execution. Exclusive output:
go2_visible_surface_intervals_v1_attempt_001. This does not implement public
uncertainty handling, certify full visibility or establish navigation success.
