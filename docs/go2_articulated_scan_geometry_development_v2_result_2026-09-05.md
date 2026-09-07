# Articulated geometry explains clearance missed by a torso outline

The corrected diagnostic is **COMPLETE** for24 recorded scan specimens and
4,504 actual control decisions, including six contact trials and eight native
contact rows. The new body-geometry component accepts current ordered joint
sensing, not environment geometry or world pose. Its outputs remain nominal
instantaneous extents, not a future swept-volume or safe-turn certificate.

## Main finding

At every recorded wall-contact row, a torso-only outline has0.207–0.217 m of
separation from the contacted wall plane. Full articulated support instead
overlaps that plane by0.000059–0.000502 m. The primary nominal calf cylinder
is the limiting primitive in all eight rows. Actual contact-position plane
residuals are0.000012–0.000119 m.

The old scan did not use a torso collision check: this shows why adding only
one would be insufficient. Small base drift, good yaw tracking and a visible
forward opening do not guarantee that the articulated robot fits through a turn.

These plane calculations use true pose, identified wall and native contact
location ONLY for evaluation. Infinite-plane overlap is not a complete finite-wall
collision test, a recovered native geometry-index identity, or a reproduction of
contact forces. Contact-selected postures do not calibrate predictive safety.

## Complete runtime posture population

The kernel transforms all27 box/cylinder/sphere collision primitives in the
bound URDF through their full fixed/revolute chains and collision origins.
Exact primitive support functions give the union's bounds along body axes.
Those bounds also enclose its convex hull; gaps between limbs are not represented.

| Body-axis extent |Minimum observed|Maximum observed|Torso primitive alone|
|---|---:|---:|---:|
|X length|0.722774 m|0.735083 m|0.376200 m|
|Y width|0.325957 m|0.418148 m|0.093500 m|
|Z height|0.378812 m|0.391651 m|0.114000 m|

Across all4,504 observations, minimum body-X support reaches−0.395083 m and
maximum reaches0.340000 m. These are sampled current-posture extrema, not bounds
on arbitrary gait, intervening2-ms states, future joint uncertainty or hardware.
Foot sphere centers agree with independent closed-form kinematics to1.67e−16 m
at every frame. That checks kinematics, not environmental perception.

## Native grouping correction

V1 failed before completing its first trajectory: it assumed URDF `dont_collapse`
hints described actual native-link retention. The audited simulator retains only
base and twelve hip/thigh/calf links; fixed heads and feet are also merged. Its
empty FAIL and all source/input bindings remain preserved.

V2 resolves each shape through fixed joints to the nearest member of the actual
ROBOT link roster, rejecting missing movable links and nonrobot names. Each calf
group therefore includes its main cylinder, two fixed child cylinders and foot
sphere; head shapes resolve to base. A calf-link contact alone cannot identify
which primitive collided. In these particular plane calculations the primary
calf cylinder supplies the largest support.

The original kernel's `urdf_rigid_group` field remains only a nominal URDF
retention interpretation. Consumers of native contact identity must use the
separate resolver and `native_shape_groups` mapping. No primitive transform,
support equation, physical trajectory, sensor value or task result changed.

Another existing limitation is now explicit: the legacy support-link rule allows
calf-group/ground contacts because feet are merged into those groups. This does
not prove absence of non-foot ground contact. All robot-wall contacts remain
disallowed, so the reported wall-contact counts remain valid. Future foot-only
support/body-clearance claims require geometry-level contact identity or a
separately justified contact-position/shape check, not link names alone.

## Verification and scope

Twenty-one new geometry/grouping/accounting tests pass; the combined suite
passes803 tests across71 explicit files. V2's preflight validates all24 actual
topologies and all eight contact rows. The analysis checks source/input bindings
before and after, uses previously raw-audited records, and compares independent
foot kinematics at every frame. This is tested geometry analysis, not an
independently implemented second physics auditor. Repeated deterministic
trajectories, frames and contact rows are not independent navigation trials.
No physics, model fitting, inflation-margin selection or controller execution
was performed by this diagnostic.

Exact root: `.generated/go2_articulated_scan_geometry_development_v2_attempt_001`.

- Corrected launch: `f00109aaff563c9417b695979ea46bf79b94c9418ac3df373765741a7baf2723`.
- Corrected result: `1544b05b6232fc37fd32b89ca17b09d3ec907318786ff495f6a942e5ccbe58cc`.
- Preserved V1 launch: `a104c36315d341d4d92d37f50dd9e497d0337d5bced068cb249e3f6a83a595b0`.
- Preserved V1 FAIL: `30f1e766bdc5dcbe76a814857ce91b6904617251c63c740e27f9b97dd8c72c58`.

## Next action

Use instantaneous body shape as an explicit input—not a safe-action label—in
the next actual observation-to-traversal-to-arrival prototype. Keep future gait
sweep, unobserved near-field space, sensor uncertainty and place association
provisional. A declared wider prototype can test runtime integration while
narrow-wall failure stays unresolved. Avoid another training sweep before
observations can initiate and terminate a physical traversal and build usable
online memory. Full-maze, JEPA-utility and real-platform evidence remain incomplete.
