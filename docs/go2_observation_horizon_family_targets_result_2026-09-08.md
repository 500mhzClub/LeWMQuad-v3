# Observation-horizon target derivation result

The complete derivation passed across all 240 recorded source episodes.
It preserves 912 context slots, 828 available contexts and all 84 unavailable
contexts with unchanged roles and reasons. Every shared 500-ms native motion
and contact target exactly matches its prior label. There was no new native
execution, model fitting, RGB-image access or tensor materialization.

| Source and role | Available contexts | Motion targets | Contact targets | Contact positives | First-100-ms motion targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original family train | 336 | 2506 | 2586 | 80 | 336 |
| Original family transfer | 348 | 2592 | 2670 | 78 | 344 |
| Moving branch train | 72 | 566 | 576 | 10 | 72 |
| Moving branch transfer | 72 | 572 | 576 | 4 | 72 |

The original family contributes 48 unavailable training and 36 unavailable
transfer contexts. Both branch roles retain all 72 cells. Future-observation
availability equals motion validity in this new contact-censored target
definition. All missing/inactive slots remain explicit. Four original-family
transfer contexts have no valid first-100-ms motion target; they are retained.

The run took 64.819 seconds after launch and binds 1,104 source paths.
Three focused target/clock/censoring tests passed. Exact artifact identities
under `go2_observation_horizon_family_targets_v1_attempt_001` in the established
navigation development artifact root:

- `launch.json`: `b7ba49b2c354600108aa9ba0e0e35ba5290f9acf94069b13ab5cea633fd8bddd`.
- `windows.json`: `c293e454ac391da377a282c61dc30dd6abbdd5274d253c963e4837a78e8f7811`.
- `training_schedules.json`: `85f6d9d91558a660ec3988b0cbaf7620a93210ce7f63b5add5201f8ff3fdb0d1`.
- `result.json`: `fe3ab252e6da0ebadba13927c0dad7410d2084145c3b50439f6848ea5b65a775`.

These are new target-side products. Neither old target population changes,
and neither predictive accuracy nor navigation is established by derivation.
