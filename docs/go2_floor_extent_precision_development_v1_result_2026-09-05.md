# Floor extent precision V1: paired rendering improvement verified

The previous goal turn made progress by exposing actual contact/range failures.
This turn isolates a scene-construction contribution to the optical-depth error.
A new32-m native visual square reduces maximum depth error in all eight paired
development views compared with a1000-m visual square. No camera clips, installed
renderer, policy depth, runtime uncertainty allowance, old scene or result changed.

## New paired evidence

Acquisition57821 completed16 native RGBD frames in two fresh CPU scenes, no
physics steps, sphere, Go2, learned model or training. The eight camera poses are
new translated/yawed/pitched development viewpoints, distinct from the earlier
aligned-floor assay; they are not independent navigation layouts or a sealed
test set. Both native collision planes remain infinite and at zero. Actual
visual surfaces align to them within1.118e-10m in both conditions.

Raw audit18730 completed successfully. It verifies identical native camera poses,
original intrinsics/clips, single-sample framebuffer readback, source/artifact
bindings, actual visual sizes/roles, and that every assessed ray falls inside
BOTH finite visual squares. All4,915,200 native image pixels reconstruct exactly
from the saved normalized depth buffer with an independent implementation of
the installed float32 conversion expression. Thus the buffer comparison uses
the same acquired data, not another render or a substituted ideal image.

All pixel-centre physical-plane rays at optical distances0.2..5m are assessed:
1,299,840 rays per condition. These pixels are correlated measurements within
eight paired views, not1.3million independent experiments.

| View | 1000m visual: native max error | 32m visual: native max error |
| --- | ---: | ---: |
| 0 | 0.935602 mm | 0.088099 mm |
| 1 | 0.532192 mm | 0.062375 mm |
| 2 | 0.135387 mm | 0.059044 mm |
| 3 | 1.479955 mm | 0.095515 mm |
| 4 | 1.121024 mm | 0.030843 mm |
| 5 | 0.038050 mm | 0.029197 mm |
| 6 | 1.274573 mm | 0.052942 mm |
| 7 | 0.968638 mm | 0.034339 mm |

The bounded visual meets the fixed1-mm comparison in8/8 views; the control does
so in5/8. The smaller visual improves every paired maximum, with worst-case
error0.095515mm versus1.479955mm (about15.5x reduction in the maximum across
these views). Some control views already have very small errors; do not claim
a uniform factor or universal failure of a large plane.

## Extent versus depth conversion

The installed renderer uses a24-bit normalized depth attachment, then float32
arithmetic to recover metric optical depth. Independent float64 conversion of
the same buffer changes native depths by at most0.016470mm on the assessed rays.
For the three failing control views, float64 maxima remain1.488332/1.104826/
1.288283mm. Changing conversion precision alone does not address the millimetre
errors. The32m float64 maximum over these views is0.094239mm.

With the same physical plane, clips, native camera matrices and evaluated rays,
the only deliberate scene difference is tangential visual extent (including its
corresponding native texture coordinates; texture does not determine geometric
depth). This is causal evidence for an extent-dependent rasterization effect
in this installed pipeline. It does not isolate a particular vertex, clipping
or interpolation arithmetic instruction. It supports using a bounded visual
mesh in a distinctly named future scene rather than changing native depth
conversion or silently enlarging the previous range allowance.

## Verification and identities

Focused57825 passed37 tests before exact-buffer reconstruction was added.
Final full59841 passed1,524 tests across129 explicit files in95.38s, including
the final17 new extent/conversion tests. No concurrent edits to the launched or
tested sources. All343 source bindings (338 predecessor plus5 new), inherited
input/artifact bindings,36 new artifact files and12 installed native source
bindings are verified by the audit. All execution/test/audit handles are terminal.

Output: `.generated/go2_floor_extent_precision_development_v1_attempt_001`.

- Launch SHA-256: `aad642ac9576cb3075bf132e97b01d3227c50767b9f7c56c02cce3b749b91f58`.
- Acquisition result: `4dc9a5915fa9503efa4a66f58d98fd2bdbf3fb73997dc38da8eb946115e7d27b`.
- Raw artifact audit: `4832196c35509b1e594e0a7806b31a9d307a6b45250daabe904f6349d64b7e15`.

This comparison does NOT overturn the preceding aligned-floor V1 failure:
its3.69-mm sphere impact penetration and its original depth errors are preserved.
The new assay deliberately has no contact dynamics and cannot repair that model.

## Next work toward the actual scientific goal

1. Integrate the aligned, bounded visual through an explicitly new Go2 builder/
   session and native contact-identity auditor, preserving all non-floor physics,
   materials, camera settings and learned gait. Do not monkeypatch global native
   classes or edit old source-bound builders/sessions. The old single-plane
   identity schema is intentionally incompatible with this separate pair.
2. Specify and verify the finite visual's domain: camera rays needed throughout
   the new maze/trajectory must remain inside the visual support. The32m square
   in this assay is NOT automatically adequate for an arbitrary maze. Unknown
   or out-of-domain depth must never become free space. Include walls/occlusion,
   motion and fresh camera poses in the integrated interface validation before
   relying on a whole-pipeline range bound; this plane-only experiment is not it.
3. Establish a dynamics-aware contact model under realistic Go2 foot loads and
   approach velocities, with explicit timestep/compliance assumptions and
   non-foot negatives. A dropped light sphere's penetration is not a calibrated
   allowance for a quadruped. Keep evaluator contact/world-pose data outside the
   onboard sensor packet and distinguish observed versus modelled contact.
4. Resolve per-frame measured surface hypotheses/provenance, prospective motion,
   fused speed, observation actions and actual acquisition-to-command timing in
   the fresh controller. Then execute complete discovery-and-return missions.
5. Test matched supervised/JEPA predictive training, memory and genuine multistep
   online rollouts across independent layouts/seeds/robustness conditions, and
   collect bounded platform evidence when hardware access permits.

Whole-maze success remains0/2. No JEPA-contribution, learned maze policy, memory
benefit, independent-maze generalization or hardware result is added by these
renders. The complete scientific goal remains active.
