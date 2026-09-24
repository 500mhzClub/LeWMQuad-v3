# Multi-junction route development V1

Specified before collection. Eight fresh cases: left-right, right-left, hairpin
and dead-end-return route motifs, each at clear widths 0.9 and 1.2 m. Cells have
pitch width+0.08 m and explicit graph connections; a wall separates adjacent
cells unless a connection exists. Interior nodes have additional dead-end spurs,
so traversals include actual branch junctions, not just repeated boundaries in
one straight corridor. All walls share the same physical/render scene.

Narrow cases start at lateral offset +0.08 m and yaw +0.12 rad; wide cases use
-0.08 m and -0.12 rad. Width and spawn perturbation are intentionally confounded
in this small coverage panel; do not estimate their separate effects. Seeds are
2026091000+i, with fresh identities. No old snapshots or protected inputs.

Use the corrected checkpoint actuator gains with readback, unchanged baseline
pure-pursuit controller, 1.5-s settling and at most 85 command ticks per edge.
Each route has three edges, except dead-end return with four, including an
explicit reverse-direction traversal. The existing baseline turns to drive
forward rather than introducing a new reversing controller. Incoming route
segments pass through the current cell centre at turns; a direct reversal uses
only the outgoing segment to avoid a self-overlapping pursuit path.

No reset/teleport occurs between edges. Continue after an instantaneous arrival
proxy miss if the required sustained directed crossing occurred. Stop the route
at a missed crossing, physical contact or stability stop; retain all partial
evidence. Infrastructure/integrity failures stop the study. No retries or
outcome-dependent parameter changes. Every case remains in the denominator.

Primary task endpoint: every planned directed crossing without contact plus
the final usable instantaneous arrival under the unchanged thresholds. Report
all intermediate arrival checks, sustained final motion, route time, contacts
and missed crossings separately. This does not require every intervention or
case to succeed; a failure constrains the next controller/action-coverage step.

Capture causal native RGB and ideal simulated body history before every executed
command, plus terminal boundaries, using the verified acquisition path. Keep
oracle route/pose/geometry and outcomes in separate labels. A versioned route
observation manifest permits up to 341 frames (4*85+1); it must not weaken sensor
or path validation. The controller still uses oracle pose/route, not the images.
No JEPA, visual navigation, independent-maze or hardware claim is made.

Audit actual walls/graph construction, source/gait/gain bindings, raw contacts,
causal decisions and commands, all sensor histories/images and actual edge
continuity. This panel supplies broader physical evidence and synchronized
route data; it is not a final benchmark or a learned-policy training study.
