# CONTACT_HAZARD_ONTOLOGY_AND_INSTRUMENTATION_V1 result

Date: 2026-08-21
Source commit: `c4790b85ebd0f58846c1cb73772be7d030896d95`
Classification: **`CONTACT_HAZARD_ONTOLOGY_OR_INSTRUMENTATION_INSUFFICIENT`**

## Outcome

The physics instrumentation qualified, but the hazard ontology did not. All 48 states and 576 branches reproduced exactly; 33,495 disallowed-contact point records reduced to 3,565 deterministic events. Link/body region, force/velocity/impulse evidence, duration, and object class are now available. However, the assets provide no mechanical material, mass, fragility, safety-critical, damage, permitted-contact, or human-consequence evidence and the project has no calibrated material-hazard threshold. Consequently all 3,565 contact events remain `SEVERITY_UNRESOLVED`.

The prior binary-contact results remain valid under their original definition. No historical row was relabelled.

## Replay and fixture

| Check | Result |
|---|---:|
| Development states / branches | 48 / 576 |
| Physics steps per branch | 750 |
| Passing states | 48 |
| Mismatched states / branches | 0 / 0 |
| Action, H3 pose, contact-tick, stuck and aggregate verification | Exact |
| New identities | 0 |
| Instrumentation fixture | PASS |

## Event inventory

| Quantity | Result |
|---|---:|
| Raw disallowed-contact points | 33,495 |
| Contact events | 3,565 |
| Contact-positive branches | 393/576 (0.6823) |
| No-contact branches | 183 |
| Material-hazard events / branches | 0 / 0 |
| Recoverable events / branches | 0 / 0 |
| Unresolved events / branches | 3,565 / 393 |
| Unresolved event rate | 1.0000 |
| Contact-followed-by-stuck events | 2,528 |
| Contact-with-progress-loss events | 1,652 |
| Stability hazards / recorded damage | 0 / 0 |

| Distribution | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Duration (s) | 0.002 | 0.014 | 0.032 | 0.186 |
| Integrated normal impulse (N·s) | 0.0008 | 1.5488 | 4.7815 | 23.2010 |
| Integrated tangential impulse (N·s) | 0.0008 | 1.1145 | 3.1171 | 11.2533 |
| Relative normal speed (m/s) | 0.0040 | 0.1881 | 0.8318 | 2.1785 |
| Penetration (m) | 0 | 0.00050 | 0.00233 | 0.00568 |

These are Genesis solver quantities and simulation event distributions, not validated real-world damage thresholds.

## Body region and object distributions

| Side | Events | Side | Events |
|---|---:|---|---:|
| front | 892 | rear | 620 |
| front-left | 348 | rear-left | 62 |
| front-right | 561 | rear-right | 168 |
| left | 11 | right | 10 |
| underside | 874 | unresolved | 19 |

Robot links: base 892; FL hip/calf/thigh 348/20/1; FR hip/calf/thigh 560/816/1; RL hip/calf/thigh 56/224/2; RR hip/calf/thigh 124/449/72. Environment classes are fixed walls (3,404 events) and fixed landmarks (161). No people, fragile objects, movable objects, or safety-critical assets are encoded.

| Family | Events | Contact branches | Wall / landmark events |
|---|---:|---:|---:|
| large enclosed | 904 | 106 | 904 / 0 |
| medium enclosed | 692 | 88 | 662 / 30 |
| small enclosed | 689 | 91 | 689 / 0 |
| loop alias stress | 1,280 | 108 | 1,149 / 131 |

## Measurement and observability audit

Body region is resolved for 99.47% of events, duration and object class for 100%, and solver impulse plus pre-contact relative velocity for 100%. Object-consequence metadata is complete for 0%.

All 3,565 events have aligned front-depth, LiDAR, enhanced-embodied, and depth-plus-embodied evidence. The exact low leg/body contact points are outside the frozen front camera and four-channel LiDAR vertical ray set under the point-in-FOV test, although nearby geometry is observable: front depth reports a ≤0.35 m return before or at 2,803 events, and LiDAR does so for 3,559. This is a physical observability description, not a classifier result and not proof that the specific contact point was visible.

## Readiness gate

Passed: body-region availability, force/velocity substitute availability, complete duration, object-class availability, deterministic classification, and documented screen rationale.

Failed:

- resolved fraction ≥90% (actual 0%);
- ≥24 material-hazard-positive branches (actual 0);
- ≥24 recoverable-positive branches (actual 0);
- either class in at least three families.

The blocker is not the Genesis contact API. It is missing safety-requirement and environment-consequence evidence: calibrated robot/object damage limits, human/fragile thresholds, mechanical material and mass, asset criticality, damage/stability consequence, and a defensible unacceptable-separation criterion.

## Historical selected contacts

These are `POST_HOC_DESCRIPTIVE_CASE_STUDY` records and did not validate or tune the ontology.

### `scale-held-0-02:00`

Exact replay produced 58 raw points grouped into ten contacts with `wall_outer_1` from physics steps 40–232: seven FR-calf, two RR-calf and one RR-thigh events; nine underside and one rear-right. Durations were 0.002–0.024 s; peak solver normal force 228.52 N; summed normal/tangential impulse 8.088/6.168 N·s; peak pre-contact relative normal speed 1.480 m/s; maximum penetration 0.00392 m. No stuck or progress-loss consequence was recorded. Front-depth and LiDAR proximity evidence existed by the recorded event ticks. Prospective class: `SEVERITY_UNRESOLVED` because object consequence and calibrated damage limits remain absent.

### `scale-held-0-03:06`

Exact replay produced ten points in one RL-calf/rear contact with `wall_outer_1`, physics steps 144–153 (0.020 s). Peak solver normal force was 162.49 N, normal/tangential impulse 1.476/1.119 N·s, relative normal speed 0.320 m/s, and penetration 0.000882 m. It was followed by stuck and route-progress loss. The exact contact point was outside the frozen vertical ray sets and neither depth nor LiDAR had a ≤0.35 m return by the event tick. Prospective class: `SEVERITY_UNRESOLVED`; stuck/progress consequences are separately annotated.

## Evidence and runtime

- Raw index: SHA-256 `73e4644c77d5f9c82349311c7a8a7d084d7a6fa0d370069a8ee2ff1ccf67e8bd`, content digest `70cb2c4e029f5c8cc5d169ea876b8cd3b9b0a1e479e2b4a44db8116846512e59`.
- Event ledger: 3,565 rows, SHA-256 `99925291fa1bc2d859161da5f4dab9f70869dcdfc9cce0046411a7377bf834a4`, content digest `f2c710c67c2ab83f859af60c370173e4dac6ff88e6c33adfb0f1108a66248699`.
- Branch ledger: 576 rows, SHA-256 `c003e23641985a4ee8f07cae6c08cff73cc3a385a79b7e061fefec619b5f279b`, content digest `f07b15f9e7be0989a0224e7d97370815d83dd1f1743a44880290ef29e321e47a`.
- Raw compute runtime: 1,682.16 s; four-process wall interval: 479.49 s; case-study compute: 51.96 s; reducer: 2.15 s.
- New raw, ledger, and case evidence: 36,241,146 bytes. The committed ledgers account for 26,486,104 bytes.

No model was trained or executed, no frame was rendered or encoded, no fresh scientific panel was generated, and no JEPA predictor, navigation, memory, novelty, or beacon component was opened or run.
