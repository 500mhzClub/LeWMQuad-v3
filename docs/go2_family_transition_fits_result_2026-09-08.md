# Family-transition six-fit result — 2026-09-08

All six preregistered transition fits and the paired raw-score readout completed.
The results are mixed: full-RGB JEPA reduced transfer contact Brier error relative
to full-RGB supervised rollout, while its position and yaw predictions were worse.
These development results do not establish a general JEPA or RGB benefit, calibrated
contact probabilities, or successful navigation.

The input scope is the separately admitted transition-prediction task in
[the input result](go2_family_transition_bootstrap_inputs_result_2026-09-08.md).
The original family navigation/task-design failure remains unchanged. All 336
training and 348 geometry-transfer windows were retained, including the recorded
missing-context accounting and contact-censored motion targets. Transfer inputs
were past-only. No outcome-driven exclusions or checkpoint selection occurred.

Seed 2026091001, 1,200 updates, batch size six, latent dimension 32, learning rate
0.001 and EMA 0.99 were fixed before fitting. The six combinations were full/no-RGB
inputs and direct/supervised-rollout/JEPA objectives. All started with the same
model identity and used the same episode-balanced schedule. The 7,200 durable
update rows, sample indices, final model identities and all saved raw predictions
were verified. The complete fits took 343.650 seconds using four fresh workers
followed by two, as selected by the prior exact-equivalence
[concurrency benchmark](go2_family_transition_fit_benchmark_result_2026-09-08.md).

Transfer scores below are unweighted means of the two geometry-cluster scores,
using each method's preregistered primary head. Position is Euclidean displacement
error; yaw is wrapped angular error. Brier uses the uncalibrated contact score.

| Inputs / objective | Position error (mm) | Yaw error (rad) | Contact Brier |
|---|---:|---:|---:|
| Full / direct | 17.380 | 0.06426 | 0.05349 |
| Full / supervised rollout | 17.248 | 0.08408 | 0.04701 |
| Full / JEPA | 48.683 | 0.14312 | 0.03469 |
| No RGB / direct | 21.504 | 0.03819 | 0.03039 |
| No RGB / supervised rollout | 21.149 | 0.03743 | 0.03798 |
| No RGB / JEPA | 22.922 | 0.11082 | 0.03997 |

The readout also retains all train, initial-context and moving-context scores,
36 method/role/scope rows with two cluster cells each, and 42 fixed contrasts.
There is one optimization seed, two transfer geometry clusters and dependent
mirrors/windows; no confidence intervals, significance or independent-maze claim
are made. Motion and contact populations differ because contact-censored motion
is not an available target.

Artifacts are under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`.
The scientific fit root is `go2_family_transition_fits_v1_attempt_001` (856 frozen
sources; 62 bound artifacts, 55,889,309 bytes); the score root is
`go2_family_transition_fit_readout_v1_attempt_001` (860 frozen sources).

| Identity | SHA-256 |
|---|---|
| Fit launch | `8ab16b173ee5a7e2ffa8ded32b0d05075d18b8b66d4cfe0caf28dc48ab581683` |
| Fit result | `ed3e2f6385991439fd390ffc64e647f6763fb3576b35f7c767fab19e4a29398c` |
| Readout launch | `60081a4e28a0f99b6267414f946437a2d449c368b46ef0f1f5a06a40973c2fe8` |
| Readout result | `ba1abbf794f1ba2a0135e4c7984419836ae3f19816d9ecb800c297bf67c33103` |
| Schedule file | `6d242e257b2151e00b042b3a005988ca0efb279e0abf1f33c0e2eb06fe1236e9` |
| Full-JEPA snapshot | `bcb8874e2adf89053463206267a4ccb90380909c324e303734a59b038f5b1821` |
| Full-JEPA model state | `10d95af9a8c950d5d04d6451a0c845bfda5c9d3237babe4e944a138f63aeb7e8` |

The final full-JEPA snapshot was designated for the separate native goal probe
before observing fit scores. It was not substituted with a better-scoring model.
The [native result](go2_family_transition_goal_probe_result_2026-09-08.md) records
that downstream test independently. Four focused fitting tests, two readout tests
and two native integration tests passed; full raw fit and native replay checks
provide the data-specific validation. No old frozen source was edited, and no
checkpoint was resumed or trained further.
