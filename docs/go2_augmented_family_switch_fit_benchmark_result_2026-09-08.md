# Augmented family/switch fitting benchmark result

All eight separately seeded 20-update full-JEPA benchmark fits completed.
Every serial/four-worker pair had exactly equal update ledgers and final model
identities. These short fits consumed the fixed mixed schedule: ten old-family
and ten new-switch batches per model. Their weights remain excluded from
scientific fitting.

Serial phase time was 472.6024623788 seconds; four-worker phase time was
128.5390551561 seconds, a 3.6767227035 speedup. Maximum observed worker RSS was
2,051,256,320 bytes, below the 8-GiB worker allowance. Both equality and memory
gates passed, so the frozen rule selects four scientific workers.

The benchmark binds 886 source paths and 44 artifacts, with all 160 benchmark
updates durably accounted for. The exclusive root is
`go2_augmented_family_switch_fit_benchmark_v1_attempt_001` under the owned
navigation development artifact base.

| Identity | SHA-256 |
|---|---|
| `launch.json` | `ac48ca1034d01fba552dba4705f918aa434d39e40cf4aa432557d46fcaba446a` |
| `result.json` | `1553f25beb279bcc239030236626ff34385325bcb0e5018ee1678920dbc377a8` |
| Canonical old/new input pair | `4cf1b5bea97aa7075b7077fb438d47050d45696bca04d65635ede78b8a841b3c` |
| Mixed schedule, seed 2026091001 | `73592f86381cd0195c9596732e7226502fe1c72005436820b7b577a4b175b0bb` |
| Mixed schedule, seed 2026091401 | `3cc01906634fce15537df579bb8eaba792cce93064fad7bc24a38b7ddb478fec` |
| Mixed schedule, seed 2026091402 | `0a5df879aed29508ede1dbb94beee2f5a159f353ed99c778c6604a7a08768c05` |

The eighteen-fit study must use these exact sources, settings, input pair and
schedules, with fresh models and 1,200 updates each. No scientific final model,
navigation command, verified arrival or deployment qualification was produced
by this infrastructure benchmark.
