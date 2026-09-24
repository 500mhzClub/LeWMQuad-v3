# Genesis Narrowphase and Candidate Feasibility V1

## Result boundary

This is the no-training result for
`GENESIS_NARROWPHASE_AND_CANDIDATE_FEASIBILITY_V1`, run from source commit
`0d490eb7651254c15ace65582cef06be6d007617`.

The target is `H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`: any disallowed
robot/environment contact during the committed five-tick, 0.5 s H1 block.
Ordinary calf/foot support against the ground and robot self-contact remain
excluded. This is a simulated contact/separation proxy. It is not evidence of
material safety, injury prevention, property protection, human safety, or
fragile-infrastructure safety. Historical 10 Hz results remain unchanged.

The predecessor terminal is preserved with its precise scope:

> `PHYSICS_RATE_FULL_GEOMETRY_SCORE_NO_GO` means that the earlier approximate
> analytical/SAT/sampled-capsule reducer failed its frozen gate. It did not
> establish that contact dynamics are required.

## Bound evidence and custody

- Physics-rate ledger SHA-256:
  `3e5de8b6b4007f9ac066bb981e23f9fc59b28459caa23d93c9c222431b18b8ee`
- 24 calibration states and 24 held-out states; 12 candidates per state.
- 250 physics steps per branch, 144,000 total comparisons.
- No state or candidate identity was created or replaced.
- 576 deterministic replays were required because pre-step configurations,
  native manifolds, and responsible pairs were absent from the prior cache.
- The native replay reproduced all 144,000 frozen physics-step labels and all
  576 branch outcomes exactly.

## Exact Genesis collision-query contract

The adapter binds Genesis 0.3.14 and the same packaged Go2 URDF, 27 robot
collision shapes, scene collision primitives, shape transforms, broadphase,
pair filters, and numerical implementation as the frozen simulator. The
narrowphase is MPR with GJK fallback, multi-contact is enabled, the historical
box-box special path is disabled, and no positive collision margin was found
for this rigid configuration.

For each registered branch the replay records the articulated `qpos` entering
every 2 ms physics step and then captures the native post-step manifold before
any collider reset. Only after every native branch for a state is complete is
`RigidEntity.detect_collision()` called at each saved pre-step configuration.
This ordering prevents a history-free query from perturbing the registered
trajectory. The reconstructed forward kinematics agrees with the persisted
post-step positions to machine precision; the maximum quaternion angular
receipt is below numerical precision after normalization.

The historical target additionally requires solver contact force greater than
1 mN. The static exact query therefore uses the same geometric verdict and
pair exclusions but cannot reproduce a constraint-solver force cutoff by
construction. That distinction is retained in the residual inventory.

## Exact reproduction

| Reduction | TP | FP | FN | TN | Sensitivity | Specificity | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native replay vs frozen, physics step | 5,852 | 0 | 0 | 138,148 | 1.000000 | 1.000000 | 1.000000 |
| History-free exact narrowphase vs frozen, physics step | 5,852 | 269 | 0 | 137,879 | 1.000000 | 0.998053 | 0.998132 |
| Earlier approximate query at zero clearance | 5,383 | 50,905 | 469 | 87,243 | 0.919856 | 0.631518 | 0.643236 |

At branch level, exact narrowphase has 231 true positives, 345 true
negatives, no false positives, and no false negatives: 576/576 agreement.
First-contact-step median and maximum absolute errors are both zero. The
responsible link/object pair agrees on 98.94% of jointly positive physics
steps. Exact event-span agreement is 489/576 branches; the difference is due
to 269 exact-overlap steps that do not cross the frozen 1 mN solver-force
cutoff.

The residual ledger contains 269 exact-query-positive/frozen-negative steps
across 87 branches. Four are within 10 µm and are classified
`NUMERICAL_TOLERANCE`; 265 contain a deeper exact manifold and are classified
`DYNAMIC_OR_CONSTRAINT_SOLVER_DEPENDENCE`. There are no exact-query false
negatives, and none of the residuals changes a branch-level contact outcome.
This does not justify a learned articulated-dynamics model: the exact static
query already reproduces the branch target and meets the frozen step-level
sensitivity/specificity requirements.

## Candidate feasibility

The feasibility ontology is now applied before mobility metrics.

| Split | Safe-candidate-available | No-safe-candidate | Safe-state fraction |
|---|---:|---:|---:|
| Calibration | 20/24 | 4/24 | 0.8333 |
| Held-out | 17/24 | 7/24 | 0.7083 |

The seven held-out no-safe states are therefore correct-abstention states, not
false abstentions. All 12 registered candidates contact during H1 in each.

| State | Prospective no-safe class | Causal attribution | First contact range (2 ms steps) | Hold/reverse |
|---|---|---|---:|---|
| `wide-held-0-05` | `PRE_EXISTING_OR_IMMEDIATE_UNAVOIDABLE_CONTACT` | commitment latency | 0–0 | both contact |
| `wide-held-1-02` | `CANDIDATE_BANK_SAFETY_COVERAGE_FAILURE` | slew-limiter authority | 29–59 | both contact |
| `wide-held-1-03` | `CANDIDATE_BANK_SAFETY_COVERAGE_FAILURE` | candidate-bank coverage | 65–149 | both contact |
| `wide-held-2-00` | `UNRESOLVED_NO_SAFE_CANDIDATE` | unresolved physics | 22–24 | both contact |
| `wide-held-2-04` | `UNRESOLVED_NO_SAFE_CANDIDATE` | slew-limiter authority | 4–4 | both contact |
| `wide-held-3-03` | `CANDIDATE_BANK_SAFETY_COVERAGE_FAILURE` | slew-limiter authority | 28–66 | both contact |
| `wide-held-3-04` | `PRE_EXISTING_OR_IMMEDIATE_UNAVOIDABLE_CONTACT` | commitment latency | 8–8 | both contact |

Across calibration and held-out together, the prospective classes are six
candidate-bank failures, three immediate/unavoidable contacts, and two
unresolved states. The direct causal accounting is five slew-authority, three
commitment-latency, two candidate-bank-coverage, and one unresolved case.
None begins with a native disallowed-contact label at the exact branch
boundary, but the immediate cases contact before candidate trajectories can
materially diverge. Candidate commands differ at the first policy tick; base
or link trajectories require between 1 and 12 physics steps to diverge by the
frozen 1 mm/1 mrad tolerance.

Because fewer than 90% of states have a safe registered action in both splits,
the independent classification is:

`CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO`

The smallest prospective correction supported by the causal distribution is
a true emergency-brake primitive with braking authority distinct from the
current slew-limited hold. It must be qualified under oracle physics-rate
contact before learned prediction resumes.

## Feasibility-aware exact-geometry result

On the 17 held-out states where movement is physically available, the exact
binary filter retains all 147 contact-negative candidates and all 17 states,
admits and selects zero contact-positive candidates, has zero false
abstentions, and retains 100% of the oracle-contact kinematic progress
(0.17040 m mean). Normalized route-progress regret is 0.09010. All seven
no-safe states correctly abstain.

Per family, retained safe states are 5/5 large, 4/4 medium, 4/4 small, and 4/4
loop-alias; all have zero selected contact and 100% oracle-contact progress.
Best-safe top-3 is 1.0 in large, medium, and small, but 0.5 in loop-alias.
Overall it is 15/17 = 0.88235, narrowly below the frozen 0.90 requirement.

Consequently, the exact contact-query tests all pass, as do safety,
availability, abstention, progress, and regret, but the complete supplied
upper-bound gate does not pass because of the independent kinematic top-3
criterion. The supplied primary taxonomy has no terminal for “exact query
resolved; independent route ranker misses.” The conservative recorded primary
terminal is therefore:

`GENESIS_EXACT_GEOMETRY_QUERY_UNRESOLVED`

This terminal must be read with the machine-recorded protocol note: the query
itself is resolved, no Genesis API or collider state is missing, and no
dynamics-model inference is supported. The remaining complete-gate failure is
`KINEMATIC_RANKING_LIMITATION`, not collision-query uncertainty.

Because the full composite upper-bound gate did not formally qualify, the
conditional depth, LiDAR, and fused sensor-geometry recalibration was not run.
This follows the preregistered condition and prevents sensor conclusions from
being drawn after an upstream planner-gate miss.

## Decision

Secondary classifications are:

- `CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO`
- `KINEMATIC_RANKING_LIMITATION`
- `CONTROL_RESPONSE_LATENCY_FAILURE`
- `SLEW_LIMITED_BRAKING_FAILURE`

`ARTICULATED_CONTACT_DYNAMICS_STATE_V1` is not justified. The exact next
experiment is `H1_SAFE_ACTION_SET_SUCCESSOR_V1`: prospectively add and qualify
a true emergency brake under oracle physics-rate contact, without learned
safety prediction. Only after the corrected action set is shown to provide a
safe response envelope should geometric prediction or learned contact
filtering resume. The frozen route ranker must also be reported against the
corrected bank; it is not modified by this result.

## Persistence, runtime, and prohibited work

- Narrowphase index digest:
  `9c660bab3f2201b1c11aa997fceae5b3ea2a332f34398a369a2d7bb43009e053`
- Row ledger SHA-256:
  `5524480467405fcf282a4532cb167d9d28d14a3f707f9509a17084e6e6c1c57e`
- New evidence storage: 10,267,809 bytes.
- Replay compute: 1,528.92 s; four-worker wall time: 421.27 s.
- Read-only reduction: 0.62 s.

No model was trained or executed. No learned checkpoint or JEPA artefact was
opened. No state, candidate, label, ranker, memory, novelty, routing, beacon,
or navigation system was created or changed.
