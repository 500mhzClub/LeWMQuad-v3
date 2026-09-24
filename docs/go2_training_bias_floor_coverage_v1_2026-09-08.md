# Corrected JEPA terminal floor coverage diagnosis V1

Authenticate the complete corrected two-case native probe and its readout, then
replay the fixed seed-2026091001 full-JEPA case without changing any controller,
model, coefficients, sensing or thresholds. Verify every recorded decision and
complete model state. At its no-candidate terminal observation, inspect every
candidate foot whose measured-floor intersection lacks complete coverage and
whose non-floor/unknown partition contains no intersecting sample.

Reconstruct each nominal foot centre from the same current measured posture,
observed map pose and first predicted displacement/yaw. Require the original
single-frame coverage witness to reproduce exactly. Query the original 44-mm
enclosing foot square at fixed subdivisions 1, 2, 4 and 8 per side. Tiles share
closed edges and their query squares round outward; every tile must pass the
original retained measured-pixel coverage function. Different tiles may use
different causally retained frames. Retain all successes, failures and pixel
witnesses at every level, including uncovered tiles; choose no level for a
native controller in this diagnostic.

This assesses a sufficient nominal geometric coverage condition. It shrinks no
foot, extrapolates no plane, infers no unseen support, and supplies no pose,
terrain, motion or physical-support uncertainty certificate. It changes no
recorded decision and implies no counterfactual executed outcome.

Freeze this script, helper, focused tests, previous result report and complete
source closure before the exclusive
`go2_training_bias_floor_coverage_v1_attempt_001` output. Use one CPU process,
one numerical thread, at most 256 recorded observations, 8 GiB available RAM
and a 256-MiB output allowance above the standing 40-GiB reserve. This bounded
serial causal replay provides no useful multi-process scene workload; no native
scene or optimizer is launched. Reverify inputs, source, coefficients and
model artifacts afterward and preserve any terminal diagnostic failure.
