# Read-only physical visibility audit of the completed twelve-episode pilot

Audit all390existing frames of the twelve frozen pilot cases, not only the
discovered bad frame. Bind the original result and its explicitly enumerated
specifications, camera/depth metadata, native-depth and public-depth files before
reading their payloads. Retain the original dataset and audits unchanged.

Use the distinct physical first-surface reference frozen in the completed
near-field camera bench. Recompute each frame at stride8 using its actual
recorded optical pose and its own native near plane0.05m. Report all sampled
clipped opaque rays, falsely public-valid near rays and metric errors. Independently
check the public depth validity mask against the unchanged0.2–5m conversion.
Count every case and frame; no replacement render, physics, model fit or pixel
repair. Failed views remain failures even when the old renderer-matching audit
passed. Passing this sampled check does not establish real-sensor validity,
robot self-occlusion or complete-image physical correctness.

Write only a distinct exclusive audit root
`go2_independent_pulse_context_physical_visibility_audit_v1_attempt_001` in the
owned external navigation-development artifact root. Preserve40GiB free and
allow8MiB of new metadata. Freeze source/input bindings first and reverify at
completion. Any exception leaves a terminal failure; no retry/overwrite/resume.
This audit bounds the defect's observed extent in this pilot; it must not turn
passing frames into independent evaluation data or silently salvage an episode
for training. Training eligibility requires a separately explicit decision.
