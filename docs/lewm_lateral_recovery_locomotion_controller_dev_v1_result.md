# LATERAL_RECOVERY_LOCOMOTION_CONTROLLER_DEV_V1

Status: stopped at the frozen controller-qualification gate

Source baseline: `690bd1ffbf0a59ba806fb62d4d5fe521f296bd3f`

Primary classification: `LATERAL_CONTROLLER_QUALIFICATION_NO_GO`

Controller classification: `LATERAL_TRACKING_AUTHORITY_NO_GO`

## Claim boundary

This was a simulation-only, single-seed low-level-controller development run.
It does not qualify physical Go2 lateral control, stopping, human safety,
material-impact safety, learned planner safety, or navigation.

All predecessor terminals remain unchanged, including
`SUPPORTED_VX_YAW_CONTROL_AUTHORITY_NO_GO`,
`CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO`,
`LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO`, and
`GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING`. The older lateral terminal still
means only that the frozen route controller has `vy=0`; lateral retreat had
not been evaluated through that controller.

## Original controller binding

The frozen route controller remained unchanged:

- checkpoint: `models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt`;
- SHA-256: `e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff`;
- configuration SHA-256:
  `bc3e68c18252475199e57b30c8ac49d813e3c784a3983e0e8b1a762490dde24f`;
- architecture: 45-input actor, MLP widths 512/256/128, 12 joint actions;
- PPO iteration: 500 from a 501-iteration frozen run;
- command ranges: `vx=[-0.20,0.30] m/s`, `vy=0`, yaw `±0.45 rad/s`;
- policy period: 20 ms; command period: 100 ms.

The existing linear-velocity reward already sums squared body-frame velocity
error over x and y. It was reused unchanged. No broad reward redesign was
made.

## Frozen continuation contract

Exactly one complete controller seed, `2026082014`, was trained from the
frozen actor, critic, and optimiser state. The budget was floor(25% of 501),
or 125 PPO updates, below the 1,000-update cap. Training used 4,096 parallel
environments and the original PPO hyperparameters and domain contract.

The environment-index-stable command mixture was:

| Category | Fraction | Contract |
|---|---:|---|
| Historical route | 50% | existing route-command bank, `vy=0` |
| Pure lateral | 25% | `vx=wz=0`, mirrored `|vy|∈[0.05,0.20] m/s` |
| Route-to-lateral | 25% | historical command, then mirrored pure lateral after 1.0 s |

Left/right signs alternated deterministically and were exactly balanced. The
contract digest is
`e1e7f3742f2aad0a25646bce2d58ebdd344805e060977c829e3d4ea7d54142c5`.
No distribution, update count, seed, or checkpoint-selection rule changed
after evaluation.

The training-only smoke passed nonzero-`vy` observation and reward wiring,
balanced sampling, finite observations and updated parameters, checkpoint
write/reload, and deterministic fixed-state inference. It opened no frozen
scientific state.

## Training result

The sole continuation ran from iteration 500 through final iteration 624 in
161.917 seconds. Final checkpoint:

- path:
  `.generated/lateral_recovery_locomotion_controller_dev_v1/seed_2026082014/model_624.pt`;
- bytes: 4,547,691;
- SHA-256: `9199cfde3d26b421fc50bc5a7a94f69b23eb7befc7509175174eb3059f35d18b`.

Selected first/final TensorBoard metrics were:

| Metric | Iteration 500 | Iteration 624 |
|---|---:|---:|
| Mean reward | 0.3220 | 32.1422 |
| Linear tracking reward | 0.01155 | 0.93009 |
| Angular tracking reward | 0.00880 | 0.74330 |
| Value loss | 0.00914 | 0.000159 |
| Surrogate loss | 0.00694 | -0.001275 |
| Mean action std | 0.25280 | 0.40279 |

All logged values remained finite. Final-update selection only was used.

## Route non-regression

The frozen and successor controllers were evaluated on the same nine-command
obstacle-free `vx`/yaw suite, with two repetitions per command.

| Metric | Frozen route | Lateral successor | Frozen allowance |
|---|---:|---:|---:|
| Mean `vx` absolute error | 0.00690 m/s | 0.00544 m/s | ≤0.02690 |
| Mean yaw absolute error | 0.01134 rad/s | 0.03391 rad/s | ≤0.06134 |
| Unintended `vy` | 0.00890 m/s | 0.00901 m/s | descriptive |
| Energy proxy | 43.115 | 45.462 | descriptive |
| Action smoothness | 0.26488 | 0.26639 | descriptive |
| Peak tilt proxy | 0.07733 | 0.10200 | no material instability |
| Contact/fall | 0/0 | 0/0 | zero |
| Joint/torque-limit violations | 0/0 | 0/0 | zero |

`LATERAL_CONTROLLER_ROUTE_NON_REGRESSION_FAILURE` is therefore not supported.
This does not promote the new controller to the macro route plant.

## Lateral tracking

Ninety-six rows covered `vy=±0.05, ±0.10, ±0.15, ±0.20 m/s` from rest,
forward, reverse, both yaw directions, and an asymmetric gait phase, with two
repetitions. Measurements were persisted at 0.1, 0.2, 0.5, and 1.0 seconds.

- correct displacement sign at 0.2 s: 83/96 rows;
- measurable displacement (≥1 mm) at 0.2 s: 88/96 rows;
- median absolute displacement: 0.00377 m at 0.2 s, 0.01414 m at 0.5 s,
  and 0.02243 m at 1.0 s;
- for requested `|vy|=0.20 m/s`, median achieved velocity at 0.5 s was
  23.71% of request (about 0.0474 m/s), below the required 50%;
- 47/48 mirrored condition/magnitude repeat pairs differed after 1e-4
  reduction;
- contacts, falls, unsafe terminations, and joint/torque violations: zero.

The response was real but weak, sometimes had the wrong early sign, decayed
by 1.0 s, and was not deterministically reproduced across paired lanes. The
frozen lateral gate therefore failed.

## Controller transitions

Twenty-four route→lateral→route rows covered rest, forward, reverse, both yaw
directions, and asymmetric phases with both lateral directions and repeats.
There were zero contacts, falls, or joint/torque-limit violations. Frozen-route
tracking resumed in every row: maximum return errors were 0.0771 m/s in `vx`
and 0.0481 rad/s in yaw, within the prospectively frozen 0.15/0.20 limits.

Peak observed transition values were 1.9568 normalized joint-action units,
9.840 m/s² base acceleration, and 94.053 rad/s² angular acceleration. All 12
repeat pairs differed at 1e-4 in the frozen discontinuity reduction, so the
complete deterministic transition gate did not pass. Because lateral tracking
had already failed, the single Section-8 controller classification remains
`LATERAL_TRACKING_AUTHORITY_NO_GO`.

## Scientific viability stage and decision

Per the frozen protocol, controller qualification failure stopped the pass
before any frozen scientific state was restored. Branch counts are therefore:

| Scientific branch category | Count |
|---|---:|
| Current lateral branches | 0 |
| Successor augmentation | 0 |
| Multi-cycle branches | 0 |

Residual and matched-control viability were not re-evaluated. Full-panel
availability remains historically 40/48 before augmentation; there is no
valid after value because the micro bank was not augmented.
`CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO` is preserved.

The primary classification is `LATERAL_CONTROLLER_QUALIFICATION_NO_GO`. The
exact blocker is inadequate and nondeterministic mirrored lateral tracking in
the one frozen continuation attempt. No second seed, longer budget, reward
tuning, or alternative controller is automatically authorised. A further
controller-design decision is required before viability or learned planning
can resume.

Historical route actions still use the frozen controller and remain within
the historical JEPA contract. The JEPA was not opened. No lateral prediction
or predictor non-regression claim is made. The 100 ms viability interface,
memory, novelty, routing, beacon capture, and navigation remain unimplemented.
`GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING` is unaffected.

## Persistence and runtime

The accepted smoke, training, and qualification runtimes were 26.850 s,
161.917 s, and 28.282 s respectively: 217.049 s total. Generated artifacts
occupy 18,717,160 bytes; the 156-row qualification ledger occupies 147,132
bytes.

Ledger:
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/lateral_recovery_locomotion_controller_dev_v1/controller_qualification_rows_v1.jsonl`

- SHA-256: `8f1df48ce9dec6dbd179d73a9f8525c3b95b5ceed6c6aa40c9531c5aa8598cd8`.

One complete controller seed was trained. No JEPA predictor, utility, safety,
progress, motion, or place model was opened, trained, or executed. No memory
or navigation system was implemented or executed.
