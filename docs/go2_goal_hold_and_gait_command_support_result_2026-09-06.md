# Goal-hold V1: two failures, and a gait command-support mismatch

Implemented bounded final-yaw corrections and executed the two fixed new-start
trials. Neither completed the full local sequence. The strongest new finding is
not another pose-estimation failure: the high-level servo requests commands that
are absent from the gait checkpoint's saved discrete training command bank.
This changes the next action to command-support/response identification.

## Implementation and actual outcomes

The [frozen protocol](go2_goal_hold_visual_servo_v1_2026-09-06.md) preserves the
0.4m / +0.3rad task, 0.06m / 0.05rad final tolerances, 35s command budget and
native stops. At most three corrective turns are permitted, triggered beyond
0.04rad final-braking yaw error, with a 0.015rad corrective arrival target.
Each final hold needs ten new zero-command intervals with quiet motion and both
visual endpoints inside the final pose region. An independent evaluator also
scores all 501 physics poses in that second, including intersample excursions.
No real trial reached that verification; its success path has synthetic tests
only. This is not continuous-time assurance or a JEPA controller.

Both trials used a genuinely changed actual spawn (-0.60,-0.10,0.375)m,
yaw +0.06rad, nominal/lower friction 1.0/0.15 on robot and floor. Each recorded
351 decisions, 360 command ticks including failure drain, 18,750 physics samples,
361 RGB-D captures and 37.5s total simulated time. Neither had a native stop.

| Result | Nominal | Lower friction |
| --- | --- | --- |
| Complete task | Failed | Failed |
| Terminal stage/reason | Corrective turn / time limit | Forward brake / time limit |
| Corrections attempted | 1 | 0 |
| Final native planar error | 0.03020m | 0.01173m |
| Final native yaw error | 0.03542rad | 0.78415rad |
| Maximum visual position error at decisions | 0.00334m | 0.00626m |
| Final 200ms maximum native speed | 0.00223m/s | 0.00071m/s |

Nominal: forward braking at tick 57, turn at 67, final braking at 94, corrective
turn at 99. It then stalled short of the 0.015rad internal corrective target.
At timeout, visual yaw error was 0.01678rad, already INSIDE the actual final
0.05rad task tolerance. The subsequent failure drain also ended inside both
native pose tolerances. This is not proof that the task is physically infeasible:
the chosen internal correction target prevented a new hold check. Preserve the
declared trial failure, but do not elevate that internal margin into a scientific
requirement. Future corrections should be bounded in duration and return to
actual-goal verification, not wait indefinitely for a tighter intermediate pose.

During 20–35s, requested yaw ranged 0.02559–0.03011rad/s; mean applied yaw command
was 0.027784rad/s. Native yaw changed only 0.003043rad over 15s, mean
0.00020284rad/s, and maximum native body speed was 0.000590m/s. The weak response
is physical in this trace, not merely visual pose drift. This is an observed
operating regime, not a globally identified hard deadband or its cause.

Lower friction reached forward arrival only at tick 342, leaving insufficient
time to finish braking and turning. No goal-hold correction was reached. A new
start is not an independent causal ablation of controller benefit.

## Command-support finding

After SHA-256 checking the exact gait configuration named in the platform
manifest, inspected its command configuration. It declares:

- Discrete bank: hold; forward 0.20, 0.25, 0.30m/s; backward -0.20m/s;
  yaw ±0.45rad/s; and arcs combining 0.20m/s with ±0.45rad/s.
- Sampler `lewm_trainable_velocity_primitives`, command jitter standard deviation
  zero. The current training source samples bank indices; nominal numeric ranges
  are not evidence of continuous command training.
- The PPO adapter scales command channels directly into its observation. The
  inspected adapter does not implement a small-command cutoff.

All 334 nominal and 342 lower-friction NONZERO requested ticks in this new
trial lie outside that saved discrete bank (componentwise matching at 1e-7).
The servo's entire nonzero command domain is below the first forward primitive
and below the trained turn magnitude. Interpolation may sometimes work, but
neither an interval-valued platform limit nor successful model loading proves
reliable tracking there. This mismatch was missed in the recent servo designs.
Saved configuration plus current source is not a complete reconstruction of
historical training; it supports the declared training-support diagnosis, not
a causal claim that all failures arise from the sampler.

Also checked the two prior servo recordings for executed zero-command runs
with future visual labels: heading-servo nominal has one run of 17 intervals;
course-servo nominal has runs of 12 and 10; neither low-friction trajectory has
a run of ten. The 12 overlapping one-second windows represent only THREE
distinct nominal events, not independent samples or low-friction stopping data.
Other older fixed-schedule recordings exist; this count is specifically the
servo data used for the response fit and its immediate successor, not the repo.

## Verification and preserved limitations

Raw audit PASS: 702 exact sensor/interface/controller decision replays, 722
RGB-D depth checks within the existing 1mm diagnostic, actual new spawn and
different 750-sample prefix versus old fitting and course-servo traces. Audited
raw contacts, body/gyro clocks, requested/applied commands, gain/friction
identities and native stops. Full regression: 2,327 passed across 186 explicit
files in 189.77s; focused 41 passed in 1.85s. Tests overlapped collection.

All launched sources, protocols and artifacts retained unchanged. No protected
benchmark access, predecessor retry, gait retraining or hardware execution.
Ideal robot-hidden camera, controlled continuous floor and paused-physics
computation remain. No useful online-memory/maze/JEPA/multistep or deployment
result is established. Follow the
[command-supported execution plan](go2_command_supported_execution_next_steps_2026-09-06.md).

## Exact identities

Output: `.generated/go2_goal_hold_visual_servo_v1_attempt_001`; launch binds
602 source paths / 15,552 inputs plus inherited native identities.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 86740c721e427575a124cc6f2eb201daf916ac7819b3dd9299ed7437c2565952 |
| result.json | f0eda03ceb27477345446679e22f3cdfd5d48919cad06ebb5454c5b712f4af83 |
| raw_servo_audit_launch.json | cec0a8f8288f35f45c53689e663875ee7dfa3037e08c8b491102d18871f66941 |
| raw_servo_audit.json | 17dbc39d0e471707c25b545d9f0b902d1a0bd310d9b445defef1116a5204d5ad |
| nominal_servo_evaluation.json | b0e8cf5da2f662af77b3c12b47be5ca63f9e18bbef6c987a4a4734556dcb4a64 |
| lower_friction_servo_evaluation.json | c126c90f0231dc46a3ae72c3a99f1f583dda9b6cce692858659e37cbccd7b9ba |
| models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl | bc3e68c18252475199e57b30c8ac49d813e3c784a3983e0e8b1a762490dde24f |
| scripts/train_genesis_go2_locomotion_contract.py | e786eff51054c012fd81e91a6199c50d0d184ccd323a221050a1a1e78a6dca04 |
| config/go2_primitive_registry.yaml | cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8 |
| lewm_genesis/lewm_genesis/rollout.py | 06501bbbdd1e071a3a91e765d77bd19da5f2c311c35d75df4631c452beea034a |
