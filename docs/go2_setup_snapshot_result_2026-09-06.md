# Starting-region and velocity checks pass on the recorded setup

The proposed finite startup priors are consistent with the first sensor epoch
of the existing bounded-floor Go2 tape. This is a separate evaluator check of
one recorded setup, not a newly successful physical run or a validated future
gait. Historical no-prior and prior-budget failures and whole-maze 0/2 remain.

## What was implemented

`lewm/setup_snapshot_evaluation_development.py` checks the entire initial-body
prism against the native oriented boxes, using all 15 separating-axis families
with conservative treatment of touching and near-degenerate arithmetic. A
positive reported projection gap is a distance lower bound, not exact distance.
The checker rejects incomplete/duplicate/moving/disabled/multi-geometry object
inventories, invalid native BOX dimensions/reserved fields, malformed rotations,
wrong epochs/episodes and mismatched setup provenance. It also checks the
initial velocity ball and containment of all 27 URDF primitives plus 4-cm padding.

The object list must match an independently enumerated physical non-floor
roster. The recorded driver obtains that roster from the bound native contact
topology, excludes only the verified physical ground, and checks that visual-only
links are absent. Completeness therefore relies on that recorded native
enumeration and its bound acquisition source, not on an arbitrary hand-picked
subset of walls. The verified native collision plane remains z-up at z=0.

A separate helper reports native support-group forces and nominal non-foot
gaps at the setup instant. It does not identify individual foot collision geoms
within the merged calf groups, infer deployment contact, certify compliance or
friction, or grant future contact permission. Evaluator pose/velocity, native
object geometry and contacts are never placed in sensor packets or passed to
the runtime prior/plane-memory consumers.

## Fixed recorded check and evidence

Protocol: `go2_setup_snapshot_recorded_check_2026-09-06.md`.
Execution 24061 completed the read-only check at sample 749, time 1.5 s,
episode (0,0,0). Bounds were fixed before executing the checker and not resized:
initial-body prism [-1,-0.75,-0.5] to [1,0.75,0.6] m, expiration 3.5 s;
zero-mean initial velocity ball of radius 0.02 m/s. This proposal follows the
recording and is not an independent preregistered physical sample.

- Reference initial velocity error: 0.001700376 m/s, within the proposed ball.
- All 27 instantaneous URDF primitives with 4-cm padding fit in the prism.
- Entire prism is separated from all four physical walls. Lower bounds are
  1.490793 m front, 1.489438 m back, 1.718059 m left and 1.747153 m right.
- Positive upward native forces are 38.756668 N FL, 50.443150 N FR,
  43.615471 N RL and 14.489084 N RR support groups. No other loaded contact
  row is present at this instant.
- All nominal non-foot primitives are above the verified physical plane;
  the minimum gap is 28.199193 mm.

The native support witness is present, but support/contact-model qualification
and navigation flags remain false. The checker does not create a floor patch in
camera memory, reinterpret the previously unknown own-body depth footprint,
or re-enable historical memory after its recorded budget stop.

All 350 inherited launched-source bindings, bound inputs/artifacts/reader
identities and 18 explicit development source/test/protocol snapshots were
verified before and after. No recorded artifact, frozen source or old result was
changed. No sealed material, physics execution or training was involved.

## Tests and next controller step

Focused 26728 passed 30 tests. Additional strict contact identity/force checks
were then added; focused 69310 passed all 33 tests in 1.26 s. Forty random OBB
pairs are compared against independent bounded linear-feasibility problems,
alongside touching and common-transform cases. Other tests preserve invalid
starts, missing objects, mismatched identity/epoch, oversized body footprints,
lost/reversed support loads, unexpected body/object contacts and absent forces.
Expanded regression 13169 passed 1,635 tests across 135 explicit files in
96.64 s. No tested source was edited concurrently. All diagnostic and test
handles from this turn are terminal; no simulation or training job was launched.

Next connect a setup-validation result to an explicitly new development startup
controller, not to an old recorded experiment. Capture the native per-geometry
robot identities needed to distinguish actual foot spheres from merged calf
groups. Define the ground/contact and prospective motion-envelope assumptions
before a fresh bounded startup observation-turn experiment. Start that observation
action promptly after setup rather than using the existing non-informative
zero/forward command prefix; test whether it supplies full-rank depth motion
before the unchanged uncertainty budget is exhausted. Preserve failure if it
does not, and do not infer successful contact or clearance from a command alone.

Any startup prior must be identical across geometry, supervised and JEPA arms,
with invalid-start/no-prior controls. Full sensing-to-command timing, complete
discovery/return missions, predictive-training and genuinely multistep planning
comparisons, independent layouts/seeds/robustness and hardware evidence remain
required. This setup check is not a substitute for those outcomes.
