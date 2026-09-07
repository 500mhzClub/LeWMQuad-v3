# RGB visible-floor evidence: fixed development baseline

The current renderer's plane uses material key `floor`, whose default palette
is RGBA(.45,.47,.42,1). The old neutral-floor override uses `NEUTRAL_FLOOR`, not
that key. Neutral walls use equal RGB components. This source-level appearance
cue explains the greenish floor in a viewed route image; it must not be mistaken
for learned geometry or transfer-ready semantic perception.

Implement a deliberately narrow RGB-only baseline: native8-bit pixel evidence
requires G-R>=2, G-B>=4 and R-B>=1, computed in signed arithmetic. A column's
envelope uses contiguous positive pixels from the image bottom, with at least12
rows. Holes end the envelope; do not fill unknown space. These fixed margins are
not fitted on validation images. Negative evidence means unknown, not an
obstacle. Do not turn a pixel row into metric distance without attitude/height
information. Runtime accepts only the existing causal RGB/body packet and reads
no pose, camera-world transform, geometry, target image or map.

Evaluate all818 unique current context frames underlying the completed914
subtrajectory windows. Initial canonical siblings are deduplicated; later
contexts use their own source frame. Keep the original24 layout roles16 train
and eight development-validation. No fitting or independent final evaluation
occurs, and818 frames do not become818 independent environments.

Evaluation alone casts native pixel-centre rays at an8-pixel stride using the
already-audited optical pose and fixed640x480,78.323-degree horizontal FOV.
Visible floor is an intersection with the ground plane before any wall box,
within the existing.05–200 m optical-depth clip range. Handle yawed boxes with
independent slab intersections. Rays whose near plane is inside a wall are
explicitly ambiguous. Report every valid sampled ray and a separately fixed
one-grid-cell interior mask, not an error-dependent exclusion. Save reference
labels separately from runtime observations; they are not a navigable map.

Compare original RGB to two fixed digital appearance negative controls:
grayscale and swapped red/blue channels, without altering the physical label.
Retain per-frame confusion counts and per-layout precision/recall. Precision
with zero predicted positives is undefined, not perfect; recall can be zero.
This tests palette dependence, not physical appearance randomization or hardware
robustness. Precision of floor pixels is not robot-footprint clearance: a wall
above visible ground, narrow opening, blind near field or unobserved direction
still prevents a traversability claim. No place/exit identity is emitted.

Fixed fresh output:
`.generated/go2_rgb_floor_evidence_development_v1_attempt_001`.
Pre-launch check rejected a transcribed predecessor-audit SHA before creating
this output directory or processing any image. The exact value was checked
against both the current artifact and the already completed derivation's launch
binding and corrected before this study's source freeze. No prior artifact,
threshold or experimental outcome was changed.
Bind source/protocol/tests and completed corpus/window/audit/camera identities
before execution; verify afterward. Preserve terminal failures without retry or
post-hoc thresholds. Results will inform a separate learned semantic observation
head and genuine observed-exit acquisition, not authorize palette-only full-maze
or real-robot navigation. The running moving-prefix collection remains untouched.
