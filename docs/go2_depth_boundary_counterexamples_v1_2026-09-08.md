# Synthetic depth-boundary counterexamples V1

Exercise the unchanged physical first-surface evaluator and pixel-footprint
diagnostic using explicit synthetic geometry and depth arrays. This reads no
native runtime artifacts and creates no public policy packet, scene or model.
Preserve the old evaluator sources. No existing outcome is relabelled.

Three fixed cases share a camera 0.6 m above the floor, a background wall with
front optical depth 1.96 m, and sampled pixel [244,324]. First, a 1 mm wide post
with front at 0.96 m is entirely absent from the synthetic depth image. Second,
the same image additionally reports a fabricated 1.5 m value at that pixel,
in the empty gap between foreground and background. Third, a tiny opaque box
with front at 0.004 m occludes the pixel but the synthetic depth again sees the
background, beyond the native 0.005 m near plane and public 0.2 m lower bound.

Report actual expected/returned depths, sampled foreground population and
exclusions, original strict and footprint scores. The expected mechanism is
that the first two incorrect arrays pass both interior metrics because every
thin-post hit lies within 2 cm of an edge. This disproves promoting those metrics
alone to general visibility certification. The near-plane case must still fail
the original clipping and false-public-validity checks, independently of the
interior mask. These are general geometric counterexamples; they do not claim
that the current 8 cm-wall maze contains the synthetic 1 mm post.

Three focused tests pass, including shape-order invariance and rejection by the
unchanged near-plane guard. Bind the explicit recursive source closure and
record versions before computation, then reverify sources. One bounded CPU
diagnostic may run alongside the separately owned native scene; require 8 GiB
available RAM and 32 MiB artifact headroom above the unchanged 40 GiB reserve.
Exclusive output: `go2_depth_boundary_counterexamples_v1_attempt_001`.
No sensor uncertainty bound, new policy, navigation success or hardware evidence
is established. The prospective visibility treatment must handle these cases.
