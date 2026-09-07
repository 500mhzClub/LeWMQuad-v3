# Measured-depth proposal consumer diagnostic V1

The previous fused navigation interface V2 stopped at frame3/1.8s in all three
appearances: FAILED_INITIAL_NO_EXIT. Fusion exactly matched the original shadow
observations but the palette-specific floor detector found zero floor pixels.
That negative remains unchanged. This is a scientific successor replacing the
exit-proposal evidence source, not a retry of the same candidate.

Use the same 54-frame matched physical tapes solely for already-seen sensor
interface development. The new controller retains RGB for correspondence,
marker discovery and appearance memory. Floor-extension proposals now use
measured optical-depth surface points, local four-pixel planar support and the
current observed depth-floor plane. Preserve invalid rays, reject vertical or
different-height surfaces, and keep the same angular/radial proposal support
criteria. An extension proposal is not a free-body volume, traversable path,
identified junction or verified route. No missing surface is filled with an
extrapolated plane intersection. Existing turn-volume/arrival/timeout rules
and the same uncalibrated fusion/error hypotheses remain unchanged.

Two copied controller methods differ only in the explicit proposal-provider
call; source AST-equivalence tests enforce that restriction. No global monkey
patch, RGB rewriting, prior reset, command-derived pose or depth-rank relabel.

Protocol otherwise follows
`docs/go2_rgbd_fused_navigation_interface_development_v2_2026-09-06.md` and V1:
local-only route memory, each original anchor/prior, causal frames until a
terminal decision or fault or tape end, no terminal reinvocation. Compare
admitted fusion and raw depth exactly in JSON representation. Record proposals
and all failures. Recorded applied commands remain those of the old physical
tape; proposed commands are never executed and cannot establish navigation.

Output exclusively:
`.generated/go2_depth_proposal_navigation_interface_development_v1_attempt_001`.
Verify frozen predecessor launch/result/arm hashes and full inherited
source/input/native/OpenCV closure before and after. Add this protocol, its
script and the12focused depth-proposal/controller tests and recursive source
dependencies to the frozen launch. Preserve all earlier failures and results.
A subsequent independently sourced auditor may replay these exact saved
controller interfaces, but grants no new physical execution or navigation claim.

Next required experiment is a separately frozen fresh complete-maze mission,
with prospective forward-body/foot/braking checks and actual sensor-controlled
execution. The old arena tape cannot supply that evidence. Long-duration error
transport, calibrated/noisy sensors, matched JEPA/rollout/memory contributions,
real-time performance and hardware remain outstanding.
