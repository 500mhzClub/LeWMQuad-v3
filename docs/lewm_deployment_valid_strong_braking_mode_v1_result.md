# Deployment-valid strong braking mode V1 result

## Terminal

`DEPLOYMENT_VALID_BRAKING_MODE_UNAVAILABLE`

This is a pre-fixture implementation terminal, not a physical braking no-go.
No local mode passed the platform-equivalence and behavioural-implementation
gate, so the preregistered Section 5 rule stopped the experiment before any
physics fixture or frozen-state branch was executed.

The predecessor findings remain unchanged:

- `H1_SAFE_ACTION_SET_SUCCESSOR_NO_GO`
- `EMERGENCY_BRAKE_INSUFFICIENT`
- `CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO`
- `KINEMATIC_ROUTE_RANKING_LIMITATION`

The target remains `H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT`, a simulated
contact/separation proxy. Nothing here establishes material-impact, injury,
property-damage, human, fragile-infrastructure, learned-safety, mission-safety,
or closed-loop assurance.

## Official/local mode mapping

The official Go2 SDK2 SportClient exposes `StopMove`, `BalanceStand`, `Damp`,
`StandDown`, and `RecoveryStand`. The public client API establishes that these
are distinct service requests; it does not include the robot-side controller
implementation, transition gains, or stopping-envelope behavior.

| Candidate | Official concept | Local binding | Active balance / control | Acknowledgement and latency | Fixture eligible |
|---|---|---|---|---|---:|
| `ACTIVE_STOP` | `StopMove` | `mode_manager` sets `stop` and publishes one zero `Twist` | no distinct active-stop controller | local service flag only; platform transition absent and latency unmeasured | no |
| `BALANCE_STAND_TRANSITION` | `BalanceStand` | `stand`/`hold` publish one zero `Twist` | no distinct stationary-balance transition | local mode string only; latency unmeasured | no |
| `DAMPING_MODE` | `Damp` | no runtime command path; passive URDF damping is not a mode | no validated joint velocity/torque damping controller | none | no |
| `STAND_DOWN` | `StandDown` | absent | not established as a moving emergency stop | none | no |
| `RECOVERY_STAND` | `RecoveryStand` | explicitly a zero-velocity CHAMP stance alias | local status says no sport-mode recovery primitive | alias only; latency unmeasured | no |

The Genesis runner always supplies the velocity command to the frozen
locomotion policy and applies returned joint-position targets. Its only direct
joint-velocity write occurs during spawn reset. The ROS/CHAMP tree contains a
velocity smoother and a generic effort trajectory controller, but neither
implements or acknowledges a Go2 stopping mode. Neither `unitree_sdk2` nor
`unitree_sdk2py` is installed locally.

Accordingly, substituting zero `Twist` would repeat the already failed
zero-command experiment. Inventing damping gains would be a new custom
low-level controller, not a platform-equivalent `Damp` binding. Both were
excluded before outcomes.

Official interface references:

- [Unitree Go2 SportClient header](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/sport/sport_client.hpp)
- [Unitree Go2 sport client example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_sport_client.cpp)
- [Unitree Python Go2 SportClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/sport/sport_client.py)

## Fixture and scientific execution result

The deterministic source-binding fixture passed byte-identical regeneration.
It bound all five official/local concepts and returned an empty eligible set.
Consequently:

- primary stopping mode: none;
- physics fixture modes/runs: 0 / 0;
- selected-mode acknowledgement latency: not measured;
- new 48-state branches: 0;
- predecessor branches: 0;
- stopping-time and stopping-distance distributions: not evaluated;
- contact, fall, and stability outcomes: not evaluated.

These are unavailable measurements, not zero-event physical outcomes. The 48
frozen identities were bound but not restored or simulated. The preserved
action inventory therefore remains 37 states with a historical safe route and
11 with neither a safe route nor a qualified fallback.

## Classification and control implication

The failure is attributable to both `SIMULATOR_CONTROLLER_LIMITATION` and
`MISSING_PLATFORM_MODE_IMPLEMENTATION`. It is not evidence that official Go2
`StopMove`, `BalanceStand`, or `Damp` is physically inadequate.

The candidate-bank terminal remains unchanged because no fallback was
qualified. Learned contact prediction remains blocked: a predictor cannot
provide a safe response that the action interface cannot execute.

## Prospective stopping envelope

`ONE_CYCLE_STOPPING_ENVELOPE_GUARD_V1` is specified, but its numeric envelope
is deliberately undefined until a stopping mode qualifies. For a qualified
mode it is indexed by current planar speed, absolute yaw rate, current command,
candidate command, stopping mode, validated stopping distance/time, and an
uncertainty margin:

`required_clearance = validated_stop_distance(mode, state, command) + uncertainty_margin`

The guard rejects a route action if it can enter a next planning state where
neither a contact-negative route response nor a qualified stop fits inside the
available clearance. The envelope must include request-to-acknowledgement
latency, be conservatively monotone in speed and absolute yaw rate, and cover
controller/plant variability. With no qualified mode, the guard cannot
authorize reliance on a stopping fallback.

## Exact next experiment

The next required implementation is
`GENESIS_GO2_SPORT_MODE_ADAPTER_V1`, followed by a fresh authorization of this
qualification. It must:

1. obtain documented behavior or black-box physical Go2 traces for exactly one
   of `StopMove` or `BalanceStand` (with `Damp` retained only as last resort);
2. implement an explicit request/acknowledgement mode state machine in Genesis;
3. drive physically simulated joint position, velocity, or torque interfaces
   using validated platform-equivalent parameters;
4. preserve momentum, gravity, collision, actuator limits, and stability;
5. qualify the training-only fixture matrix before any 48-state run.

No learned perception or safety experiment should resume until this produces a
bounded, stable stopping envelope. A contact-free stop would still not satisfy
search/inspection progress: repeated stopping or abstention is not successful
mission behavior.

## Evidence and custody

The machine result is written to
`.generated/deployment_valid_strong_braking_mode_v1/result.json`. The immutable
row ledger contains source bindings and zero scientific rows; it is written to
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/deployment_valid_strong_braking_mode_v1/row_level_evidence_v1.json`.
It occupies 6,968 bytes, has SHA-256
`00da83c7d7f39ef41b535abc4cb8ec2e788af327b0c0a31225b908786f48f454`,
and has canonical content digest
`39d8e23ffeeec7cec2cf407404a365f86b8cd6c232718e7e2a53bca039ca1a64`.
The binding/evaluator runtime was 0.002 seconds; simulation, training, and
learned-inference runtime and new raw-physics storage were all zero.

No model training or inference, JEPA access, simulation, new scientific panel,
memory, novelty, routing, beacon, or navigation work occurred. Nothing was
left running.
