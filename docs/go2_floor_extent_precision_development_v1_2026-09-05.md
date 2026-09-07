# Floor extent / native depth conversion: fixed development comparison

Question: does the native1000-m two-triangle visual produce greater optical-depth
error than a32-m visual covering the same evaluated rays? Separately, how much
error is due to float32 conversion of the same24-bit depth buffer?

This is a new rendering experiment on eight new development camera viewpoints,
not a repeat or acceptance replacement for the failed aligned-floor/contact V1.
No sphere, Go2, contacts, controller, learned model or training. No physics steps.
Preserve all previous failures and the0/2 maze result. Do not change installed
renderer code, policy depth, runtime error allowances or camera near/far.

Two fresh CPU scenes: native collision-only plane at zero and native visual-only
plane at+5mm, extent1000m versus32m. Default otherwise-identical materials and
one-metre texture tiles. Both actual visual world meshes must align to physical
z=0 within1nm readback error. Only tangential visual extent differs. Physics
collision plane remains infinite in both; every ray assessed must intersect
inside BOTH finite visual squares with at least1cm edge margin. No statement
about finite-floor coverage beyond this explicitly checked domain is permitted.

Scene seed2026090502, dt2ms, no stepping. Eight fixed(x,y,z,yaw,pitch) tuples in
`floor_extent_precision_development.py`: translated positions within1.3m of
the origin, z0.30–0.42m, yaw0.17..3.0 and negative yaw, pitch-0.21..+0.04.
Same eight viewpoints in both conditions, none identical to the earlier assay.
Camera640x480, original78.323° horizontal field of view, near0.05/far200m.
Separate RGB/depth calls at the same native pose and physics step0; verify
single-sample framebuffer. Save unmodified native RGB, native optical-depth
float32, normalized GL_DEPTH_COMPONENT float32 readback, and actual camera/plane
identity. The raw normalized buffer must come from that depth-only framebuffer,
not an RGB multisample target. Save and restore read framebuffer binding.
Reconstruct every native pixel exactly from the raw buffer using the installed
float32 conversion expression in an independent no-GL compiled function. This
checks raw-buffer/native-output pairing before interpreting conversion effects.

Analysis uses ALL pixel-centre rays with physical-plane optical distance0.2..5m,
not a subsample. Compare native output and independent float64 conversion of
the same raw buffer against analytic physical-plane rays. For each camera,
report maximum/mean native error, maximum decode64 error and native-decode64
difference. Fixed assessment: all eight32m native images within1mm is a bounded
render check, not a calibrated sensor bound. Report control errors regardless
of pass/fail, and paired changes; no requirement that every control image fail.
No more extents, clip values, camera variants, retries or parameter searches
within this one-shot experiment.

One-shot directory `.generated/go2_floor_extent_precision_development_v1_attempt_001`.
Bind all338 predecessor sources, inherited inputs/artifacts, the predecessor
launch/result identities, new exact source/test/protocol paths and12 installed
native sources. Before/after binding checks; preserve raw artifact hashes and
an exclusive read-only-recomputed audit result. An existing output directory
blocks another launch. Exceptions are terminal; no retry/resume in this attempt.

Even a positive result requires explicit integration into a new Go2 scene/session
with domain coverage and contact identities. Dynamic contact, perception under
noise/latency, ground hypothesis provenance, prospective motion and full timing
remain unresolved. Complete missions and matched supervised/JEPA, memory and
genuine multistep rollout comparisons on independent layouts are still required.
