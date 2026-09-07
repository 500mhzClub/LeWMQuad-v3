# Compiled primitive beams and shared floor-family evidence

The previous goal turn made progress on observation-bound obstacle semantics.
This turn reduces computation while preserving the recorded decisions and all
view-level work counts. Five source/test/probe paths are unlaunched. Frozen
controllers, physical results, transport radii and contact permissions are
unchanged. No new navigation or learning experiment was executed.

## Implementation

`primitive_beam_kernel_development.py` implements the pixel-cell/optical-box
linear inequalities and four-corner return reductions in a CPU compiled kernel.
It retains the same outward arithmetic allowance, incomplete-view early-out,
validity requirements, range uncertainty, floor classification, near-return
veto, boundary convention and complete-view requirement as the reference. It
uses neither fast-math nor parallelism, GPU execution or persistent JIT caching.
Construction explicitly warms the compiled backend before observation collection.

The existing NumPy two-triangle plane-family predicate builds a shared mask; it
has not been replaced by an approximate classifier. Each immutable prepared
frame retains at most two exact-parameter mask/index entries. Keys include exact
anchor/normal bytes and normal/offset allowances. Masks and summed-invalid-cell
indexes are immutable, and eviction is bounded. The summed index lets complete
primitive floor footprints reuse the same classification as the non-floor beam
checks, including missing-cell negatives.

Floor queries now return the exact observed-frame plane identity. The consumer
uses it directly rather than rotating it to the body frame and back before
looking up the same family. Both reference and compiled paths use this identity;
their outputs and the previous recorded primitive decision flags agree.

The compiled backend is explicitly selected. The reference remains available
for differential checks; selecting compiled computation does not qualify contact,
future gait, sensing assumptions or the navigation controller.

## Range arithmetic correction

The earlier NumPy reference could subtract a Python range-error scalar in
float32 but a NumPy float64 scalar in float64. That made boundary decisions depend
on the scalar's type despite its equal numerical value. Minimum/maximum return
depths and the range allowance now use explicit float64 interval arithmetic.
An adversarial boundary fixture distinguishes the two previous results and
requires identical corrected reference/compiled decisions for both scalar types.

This is an explicit numerical correction, not a tolerance increase or a claim
that the declared 1-mm range bound is calibrated. The final recorded primitive
decisions still agree with the preceding development diagnostic.

## Recorded comparison

Both probes reconstruct the first 181 original north packets using the saved
nominal observer records, then compare reference, compiled, and repeated compiled
queries at ticks80/140/180. They use the same current posture, observations and
unchanged error hypotheses. Comparison includes every result field and view-level
cell count, not merely aggregate success.

Initial probe29415 completed with exact agreement. Its compiled new-mask queries
took357/622/716ms; identical-observation repeats took132/253/303ms. After sharing
the floor-family index and exact observed plane, final probe57945 reported:

| Tick | Views | Reference | Compiled, new masks | Compiled, same-observation repeat |
| --- | ---: | ---: | ---: | ---: |
| 80 | 21 | 1,715 ms | 313 ms | 80 ms |
| 140 | 29 | 2,657 ms | 493 ms | 124 ms |
| 180 | 30 | 3,158 ms | 559 ms | 137 ms |

All reference/compiled/repeat fields match exactly. The primitive decision lists
also match prior probe12459 from the previous goal turn: conditional-clear counts
10/19/21; foot-candidate counts0/2/2; near-return veto counts4/6/4; no floor-
penetration or all-clear pass. At180 the two candidates remain rear feet, and
FL_calflower, FL_calflower1, FL_foot and FR_foot remain vetoed by seedless views.
No contact permission or physical mission result changed.

The final construction/warmup took1.395s before observations. The median
observation subset was54.09ms, excluding live nominal depth registration,
acquisition, gait and remaining control. The regression suite overlapped these
probes. Same-observation repeats reuse EXACT plane identities; they do not prove
cache hits on a changed next-tick seed. Even warm queries at140/180 exceed100ms
alone, and the complete loop remains unqualified. Scanned-cell counts remain
4,666,825/7,570,544/8,441,233; this optimization changes implementation, not coverage.

## Verification

Initial focused32136 passed47 tests, and full89148 passed1,483 tests across127
explicit files in95.67s. After the shared index extension, focused9270 passed61
tests, and final full20519 passed1,487 across127 files in96.70s, with no concurrent
source edits. All test/probe handles are terminal.

The 23 new tests cover exact dictionary/array agreement on geometric negatives,
image edges, missing rays, camera-plane crossings, zero error sets, 60 randomized
boxes with and without floor hypotheses, scalar-type boundaries, bounded cache
reuse/immutability, direct-versus-cached complete primitive floor checks, and
consumer-level view/work-count agreement. They complement rather than replace
the existing physical-semantic and custody tests.

Both completed probes verify all333 predecessor source bindings and bound inputs
and artifacts before and after processing. No experiment directory was created.

## Scientific limits that the next controller must address

The source recheck confirms the PREVIOUSLY RECORDED visual/collision-floor
mismatch: `check_floor_identity` verifies visual vertices at−5mm and the collision
plane at0. The native `floor_identity` record is explicitly evaluation-only.
The current1-mm plane-offset hypothesis therefore does not contain that known
physical surface discrepancy. Agreement with measured depth is not by itself
agreement with the collision surface; current physical-gap outputs remain
conditional on their supplied plane model, not physical-clearance qualification.

Do not inject evaluator world poses or silently add5mm to old policy observations,
rewrite the frozen identity check, or rescore the old missions. Before a contact-
aware physical successor, either build and independently verify aligned visual/
collision surfaces in a distinctly named acquisition/simulation setup, or specify
and validate an explicit surface-to-contact calibration/error model. Preserve all
old data and outcomes. This issue is separate from query-dependent missing floor
seeds and the ambiguous foot-contact states.

Next actions:

1. Establish a query-independent observed ground hypothesis with sensor/posture
   provenance and explicit contact/surface assumptions. It must classify actual
   returns without extrapolating unseen support, and retain wall, overhang,
   missing-floor and seedless-view negatives until resolved by evidence. Account
   explicitly for visual/collision alignment before contact admissibility.
2. Reuse validated per-frame hypotheses across changing current queries, measure
   actual successive-tick cache behavior, and finish the acquisition-to-command
   timing budget. A warm repeat of one observation is not a real-time result.
3. Integrate modelled foot-contact admissibility, prospective commanded motion,
   fused speed and observation/repositioning actions in a fresh development
   controller. State any planning/low-level-control rate split and latency model
   explicitly; do not hide computation time by treating stale evidence as current.
4. Execute complete discovery/return missions, then matched supervised/JEPA
   predictive-training, memory and genuine multistep online-rollout comparisons
   on independent layouts/seeds. Keep hardware qualification separate until
   bounded platform testing is possible.

Whole-task success remains0/2. No new learned-navigation, JEPA-contribution,
independent-maze or hardware evidence was produced. The full goal remains active.
