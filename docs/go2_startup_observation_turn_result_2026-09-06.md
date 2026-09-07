# Startup observation turn: one verified local simulation success

The fresh setup-conditioned turn completed and its raw audit passed. This
establishes a local active-observation maneuver under the declared simulator
conditions, not maze discovery/return, a learned navigation policy, or JEPA benefit.
The earlier full-mission result remains **0/2**.

## Executed evidence

Output: `.generated/go2_startup_observation_turn_development_v1_attempt_001`.
The unchanged gait/checkpoint, four-wall arena geometry and aligned native floor
were instantiated with the new declared scene ID and seed. The launch binds
385 explicit source paths, inherited inputs and 12+4 native implementation files.
All 74 expected acquisition artifacts are present. Bindings passed before/after
execution and raw replay; launched sources, protocol and recordings remain frozen.

After 1.5 s settling, the live evaluator verified the new initial-body [-1,1]^3 m
cube, zero-mean 0.02-m/s initial-velocity ball, physical/visual floor identity,
four loaded support groups and four individual native foot spheres. Initial
velocity was 0.00170038 m/s. The minimum native wall separation lower bound for
the setup cube was 1.46806 m. Native foot-centre/URDF coordinate disagreement was
at most 3.898e-8 m. These evaluator checks were not passed as sensor measurements.

The controller requested seven positive-yaw ticks at 0.35 rad/s. Depth motion
became rank 3 at **2.2 s**, 0.7 s after admission, at an inferred heading of
0.21820 rad. It then requested zero. Three consecutive quiet rank-3 observations
were reached at **2.6 s**, with a 0.0324154-m combined position scale, below the
unchanged 0.08-m budget. Four controller braking ticks and three additional
zero-tail ticks were physically executed. Acquisition ended at 2.9 s, with
1,450 physics samples, 15 RGB-D observations and 12 controller decisions.

Raw replay exactly reconstructed body sensors, fast gyro, depth packets, relative
motion, contact classifications, command cadence, setup checks and controller
decisions. No native disallowed contact, loaded non-foot ground contact, body
stability stop or declared speed-cap violation occurred. Maximum active base
speed was 0.0671771 m/s. The padded body remained inside the supplied cube; all
sampled prospective position envelopes contained actual motion. Eleven decision
horizons were fully recorded; the final terminal horizon was truncated at the
planned end of recording and is not a claim about unrecorded motion.

The entire final 100 ms was quiet: maximum actual linear speed 0.0138916 m/s and
angular speed 0.00622443 rad/s. The extra three-tick tail displaced the base
4.66449 mm. All 42,381 evaluated interior floor/wall depth rays passed the 1-mm
check; maximum error was 0.0529279 mm. Nominal non-foot ground gap stayed at
least 25.5156 mm. Active nominal foot penetration reached 1.33694 mm; this is a
recorded diagnostic, **not** a newly allowed penetration bound. Native terminal
foot identity, friction, solver parameters and actuator gains matched admission.

## Limits that change the next action

The full collector took 25.305 s wall time. Acquisition plus controller alone
exceeded 100 ms at all 12 decisions: 104.2–132.8 ms after the first frame, before
roughly 30 ms/tick of simulator execution and other overhead. No real-time claim
is justified. Genesis also emitted inherited model COM/joint-limit warnings;
this run does not qualify those dynamics for hardware.

A separate read-only diagnostic replayed the exact controller-terminal state,
without postterminal observations, restart or outcome changes. Its five retained
views have **zero own-body floor-covered primitives**, zero positive conditional
clearances and zero foot-contact candidates. There are no observed conflicts,
but absence of conflict is not observed clearance. Motion observability has
improved; general navigation handoff has not yet been established. The setup
region expires at 3.5 s and must not be silently extended or promoted to a map.

Next work is specified in
[the continuous-navigation handoff plan](go2_post_startup_continuous_navigation_plan_2026-09-06.md).
It must address current-body blind space and action-conditioned motion, not
repeat successful startup turns as a substitute for full missions.

## Verification record

- Focused session 75304: 56 tests passed in 4.56 s.
- Expanded session 76084: 1,692 tests across 138 explicit files passed in 98.06 s.
- Physical session 20929: acquisition complete, exit 0.
- Raw audit session 46062: local observation-turn success, exit 0.
- Handoff diagnostic session 32157: terminal replay complete, exit 0.
- Summary/binding sessions 62187 and 94980: terminal, exit 0. No live jobs remain.

The handoff diagnostic is a separately bound read-only script, not part of the
launched source or the preceding 138-file regression. Its actual replay completed
with before/after source/input/artifact verification.

Identity witnesses:

| Artifact | SHA-256 |
| --- | --- |
| launch.json | `81d4caf731b0b607c55f967601e309b252a651cb95fb76b1559e4c17cbf12f0d` |
| result.json | `cc299c1a1aea1bad7162da8283929ea41daa3669fc8a35e47124ac85cf21feb5` |
| raw_artifact_audit.json | `2f4fa9785cf1aaf7cad848a080fa1754f581df25508b97ad40089f721f4f20fc` |
| handoff diagnostic source | `299c59ecc49b72b1bd03742b2edfb271f79f436bc0b6a0e5019f08e057f6b1ea` |
