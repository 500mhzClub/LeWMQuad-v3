# Augmented-model native goal probe result

Both fixed first-seed full-JEPA and full-direct cases completed collection and
independent raw audit. Neither reached the goal. Both stopped at decision 33
with `NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`, followed
by the unchanged ten-command zero drain. There were no physical or acquisition
stops, all strict visibility gates passed, and there were no hard measurement
failures. Both models remained unchanged and every decision was replayed.

Both executed three right-turn commitments, one hold and two right-arc
commitments. Their complete 44-frame physical, public-history, RGB and
observer/memory prefixes are exact matches. The final actual initial-body XY
displacement was [0.108432591, -0.120375423] m and minimum/terminal goal distance
1.098184707 m. Each recorded 43 commands and 2,900 physics samples. Maximum
observed-pose XY error was 2.107 mm. There are zero verified arrivals.

Compared with the original full-direct run, commands first differ at tick 28:
the original turns right, while both augmented models choose another right
arc. All raw physics, policy/gyro histories, RGB and observer/memory evidence
match through the 29th observation, before that differing command executes.
The subsequent paths are separately executed outcomes, not counterfactuals.

At tick 33, all six surface checks pass, but all nominal checks fail. The
measured start clearance is 0.448523715 m against the unchanged 0.45-m radius.
At tick 28 the selected forecast clearances were 0.462129090 m (direct) and
0.460918206 m (JEPA). This is a predictive-constraint failure to diagnose;
the 0.45-m requirement and all previous outcomes remain unchanged.

The two-process phase took 79.376 seconds after launch, with maximum worker
RSS 2,227,597,312 bytes. Preflight saw 82,381,488,128 bytes available RAM and
70,952,132,608 bytes artifact space. Complete-iteration medians were 486.745 ms
(JEPA) and 487.677 ms (direct); all 43 iterations per case exceeded 100 ms.
These measurements include concurrent execution, with physics paused during
compute. They do not establish real-time or hardware qualification.

Exact identities under the established navigation development artifact root:

- `go2_augmented_family_switch_goal_probe_v1_attempt_001/launch.json`:
  `39ef9f34cb7548c378f619d87d48fe02b5d5a1647fa382fa3e846f46d9cad112`.
- Its `result.json`:
  `f2e0a45a77f27b7a23ca3f621d40de68ad9fcb10c40515abcefb11e7c6e85dab`.
- `go2_augmented_family_switch_goal_readout_v1_attempt_001/launch.json`:
  `c78018dd933c3dc5d9bbaf50358f5ddc176fff597a8fc8ecbf34b21a0f92c415`.
- Its `result.json`:
  `8d3fd2d78d016fe250a295b17d451c5392be62720999fa13292036c7e4ea49e2`.

The probe binds 1,080 source paths and 418 artifacts; readout binds 1,083
source paths. Three focused scope tests and two readout-boundary tests passed.
This reused layout contributes zero independent mazes. No navigation, JEPA
advantage, physical-backtracking or deployment claim follows from this result.
