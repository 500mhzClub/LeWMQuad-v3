# Aligned floor V1: geometry aligned, interface acceptance failed

The preceding user turn was a status restatement (no progress). This goal turn
implemented and executed a distinctly named scene-interface assay. It produces
new evidence that changes the next action: both the contact model and the 1-mm
optical-range assumption need work before a new Go2 navigation controller.
No old recording, controller, clearance radius, threshold or result changed.

## Actual new execution

The new builder adds a collision-only native plane at zero and a visual-only
native plane translated +5 mm. Actual local visual vertices remain at -5 mm;
native world-space readback confirms coincidence with the collision surface to
1.118e-10 m. This removes the construction offset in this NEW scene; it is not a
correction to recorded depth or an evaluator input passed to a policy.

Acquisition session53329 completed one CPU scene, 500 physics steps at2ms,
one22-mm sphere dropped from100mm centre height, and four640x480 RGBD views.
No Go2 or learned checkpoint was loaded. The five new source/test/protocol paths
were bound at launch and are now immutable for this assay. All333 predecessor
source bindings and inherited inputs/artifacts, plus12 installed native source
bindings, were checked before and after. The one-shot directory cannot be
relaunched or resumed.

Output: `.generated/go2_aligned_floor_interface_development_v1_attempt_001`.
Launch SHA-256: `3112197fe089b3fceacfa39385ab3fe12b4f527ba8a47831cd98996a648cabc9`.
Acquisition result SHA-256:
`bc8d6ed7cc07e9ceb8ede789a4bd38359fed16dc85f0f9e31dd9d511a8f27e50`.
The acquisition status means only that raw data collection completed, not PASS.

## Strict audit failure and contact evidence

Audit session64276 exited1 with `ValueError: sphere collision-surface settling`.
The frozen auditor checks both the last100 samples within1mm of the sphere
radius AND no whole-trace gap below-1mm. The prose protocol explicitly described
the settled check, but did not separately describe this stricter whole-impact
condition. That documentation/source mismatch must be made explicit in any
successor; neither the frozen condition nor its failure is changed here.

Raw trace decomposition:

- Maximum penetration:3.694713mm, step67, with17 samples below-1mm.
- Maximum absolute settled gap, last100 samples:0.107761mm.
- All100 settled samples have nonzero native sphere/plane contact forces.
- First native contact is step64; last native upward sphere force is0.262529N.

The static surfaces coincide but simulated dynamic contact does not enforce a
zero-penetration hard constraint at every integration step. This does NOT prove
a3.7-mm Go2 foot tolerance: sphere loading, inertia and impact differ from an
articulated robot. The next contact model must explicitly distinguish geometric
surface identity, measured pose uncertainty and modelled dynamic contact.
Do not relabel every calf contact as allowed or tune the old clearance gate.

## Separate saved-depth diagnostic: two views exceed1mm

Because the frozen auditor stopped at the earlier contact condition, a new
read-only diagnostic decomposed the already saved artifacts. It did not rerun
physics, change thresholds, replace the audit, or award a new acceptance result.
Session63037 completed with all bindings preserved.

| Camera yaw / pitch | Interior rays | Maximum optical-depth error to physical plane |
| --- | ---: | ---: |
| 0 / 0 | 2,160 | 0.575472 mm |
| 0 / -0.15 | 2,720 | 0.191659 mm |
| +0.4 / -0.15 | 2,720 | 1.760453 mm |
| -0.4 / -0.15 | 2,720 | 1.346618 mm |

All four recorded intrinsics and single-sample framebuffer readbacks match the
requested contract; physics remains at step500 across each RGB/depth pair.
Using the saved native camera transform instead of the requested double-precision
pose changes predicted rays by at most4.15e-7m. Errors in the two yawed views
remain1.760039mm and1.347032mm. Camera-pose readback rounding therefore does not
explain the observed millimetre errors. The old -5-mm plane reference disagrees
by60.6–69.5mm at these oblique rays, further separating the old construction
offset from the smaller remaining renderer error.

The native visual uses two triangles spanning a1000-m square. Large-triangle
floating-point rasterization is a plausible next hypothesis, not yet a verified
cause. Test native depth precision/geometry extent in a separately specified
rendering investigation; do not silently enlarge the present1-mm allowance or
reuse these diagnostic poses as independent validation. Hardware RGBD range,
extrinsic, timing and surface properties remain separate calibration questions.

## Tests and custody

Focused63557 passed33 tests (20new,13unchanged). Full1012 passed1,507 across128
explicit files in94.70s. No concurrent source edits occurred in the tested and
launched paths. These tests establish schema/geometry rejection and regression
properties, not physical-interface acceptance. The fresh real-render/contact
negative takes precedence over any interpretation of the green unit suite.

The post-failure diagnostic is
`scripts/diagnose_go2_aligned_floor_interface_development_v1.py`, SHA-256
`788954e894eb2ce115e2d692edc75ae541d1cfe95fc6f7e2ede6222c7df7f21b`.
It is not part of the frozen acquisition source and gives no replacement
execution authority. All listed process handles are terminal.

## Revised next actions toward the full goal

1. Diagnose the yaw-dependent depth error against actual native camera/raster
   geometry. Test a justified geometry/precision correction in a fresh distinct
   scene interface; validate new views before adopting a range-error model.
2. Specify contact admissibility with dynamics and observed state. Include
   realistic Go2 foot loads, approach velocities, timestep/compliance dependence
   and non-foot contact negatives; use evaluator contacts only for validation,
   never as undeclared onboard sensing. Geometric alignment is not this model.
3. Adopt the new floor pair explicitly in a separately named Go2 builder/session
   and native contact-identity auditor. The old session's one-plane schema is
   intentionally incompatible; no global monkeypatch or old-source mutation.
4. Resolve query-dependent missing floor seeds with per-observation measured
   surface hypotheses and provenance, preserving unknown/obstacle contradictions.
   Finish actual successive-tick and full-loop timing, prospective motion,
   fused-speed and observation/repositioning integration.
5. Run complete discovery/return missions, then matched supervised/JEPA
   predictive-training, memory and genuine multistep online-rollout comparisons
   on independent layouts/seeds and robustness conditions; obtain bounded
   hardware evidence when access permits.

Whole-mission success remains0/2. No new JEPA-contribution, learned-navigation,
novel-layout or hardware result was produced. The full goal remains active.
