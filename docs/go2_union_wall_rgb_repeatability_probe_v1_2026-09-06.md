# Fixed native bench for overlapping-wall RGB interference

The live l00 collector remains frozen and unchanged. Its first24completed
episodes pass individual depth/raw checks, but0/20sibling full-packet prefixes
match exactly. First-pair native/contact traces and non-RGB packet fields agree;
RGB frame0 differs at3,718pixels (max channel difference114). Analytic box-ray
intersection locates3,560changed pixels on exactly coincident front surfaces of
two adjacent-wall pairs. The other158changed pixels are not yet explained by
that centre-ray test. Images show thin strips of coplanar-surface interference.
Do not relax the exact-prefix test or promote this to an exact RGB-matched set.

The distinct union-wall visual constructor removes internal/duplicate-area faces
of the union of the exact equal-height, axis-aligned native boxes. It does not
change physical walls, camera near/intrinsics/mount or public depth validity.
Boundary construction uses exact integer-micrometre coordinates; off-grid,
rotated, floating-base or unequal-height geometry is rejected, not approximated.
Stable surface-based appearance seeds make input box ordering immaterial. The
surface colours change from the predecessor: no science-identical-image claim.

## Frozen native experiment

Use exactly the first l00 scene specification and its recorded camera poses at
frames0and8, bound from the already committed first episode. Those are bench
camera poses only; never supply native state to an inference controller. Build
two independent CPU/offscreen scene instances with identical physical boxes,
physics/appearance seeds and new union visuals. Render two RGB/native-depth
pairs per instance, four captures total, with zero physics steps and no gait.
The camera is explicitly posed; no simulated mounted-motion claim is made.

Check all native collision/visual identities, true intrinsics/clip planes,
optical poses, same RGB/depth epoch and single-sample depth framebuffer. Both
poses must pass the already frozen physical first-surface1mm depth check. Each
RGB and depth array must be bit-identical across the two independently built
instances at the same pose. Preserve any failure; no repeat until lucky, changed
pose, threshold relaxation or automatic successor. Native depth is not filled
from analytic geometry. No model fitting or navigation run is included.

Exclusive output: `go2_union_wall_rgb_repeatability_probe_v1_attempt_001` in the
owned external navigation-development root. Keep40GiB free and allow256MiB;
preflight must count serialized launch metadata before creating the output.
Freeze source/native inputs and the first episode's exact commit/file bindings
before execution. Save all native/RGB/identity evidence and reverify at terminal.
This bench does not alter the running batch, grant a replacement collection,
establish dynamic RGB repeatability or prove hardware/JEPA/navigation success.
