# Protected Camera Adaptation V4 tail-depth successor preregistration — 2026-07-15

## Boundary

This source-free amendment preregisters exactly one possible future attempt rooted at `.generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v4`. It is not implementation, review, execution, training, test, mutation, selection, or promotion authority. Exact source closure, independent review, and a separate authorization are required before reserving a previously absent root.

V4 is a fresh deterministic N320-to-Shared-V5 update-zero migration, never a V1/V2/V3 checkpoint or optimizer continuation. It must reproduce initial state SHA-256 `e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87` from N320 checkpoint file/content `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0` / `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`, whose gate file/content is `4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6` / `76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b`.

## Committed evidence bindings

Every JSON row gives `file_sha256 / content_sha256`.

| Evidence | Hashes |
|---|---|
| update-zero diagnostic terminal audit | `52d6ac4a7287b9cb9bd33fbdd3eadbb939f9368643953fe61866777f620914bf` / `86276a89a6cc637aefdee798c916eeaafe1d30ca2c69038b6debdaa8332f8fe0` |
| matched-training V4 terminal audit | `70371a2cd09e912e05ba0b5efdf75ee2de38cc89347e8111fff303e2a55c485b` / `ae86d1479fc3016eb96302304e079b7bf9647e26b24b3d860e7d32013bf9c6f4` |
| Camera V1 terminal audit | `c52bd5e58be3b76389d6f992675f6518ab5e062a8bbf84736123fe415476feb7` / `42108f767ce648a4b2e99f6303f922e5372f981922dd3a969a2d255795e03447` |
| Camera V2 terminal audit | `568941cedb1b9e127e9c12f625022f5d5937c49158510cfbf39fd5a9b8940bc8` / `9d4d9552d43e8782e46f0b48bbd61bd3e65972d23d6a6ed50025b682ca0f5285` |
| Camera V3 terminal audit | `3eb77a83ede536680e03363521f73f41205ac17d845a0e28251a40dcf82f77ab` / `a5a86d5260c519003f7a5efeb1d21c535afeb65ef7596a627174a41c633be2ac` |
| V3 warm-start science BLOCK | `b37829a2c311533240f6191c099d79411d453adbde43cd0304f1e5c74bd676d7` / `f317c80e527706faf267ba0be3ab8a19187aeeabba49896f4f7d0722aac98168` |
| physical-gate oracle PASS review V2 | `1b4345911f51bcb60e472a366b1b3b68858e9d82673a357a0a75cd81d72c41d6` / `e7a87b43a02fffbb59a891dfab1be133c8d72f02d9abb05358f171269dede99d` |
| physical-gate oracle authorization | `5a5b9e5bb04e1218614ce84d54ca286989d111a090f2f0d1f634b4c112fd6246` / `c38d401d00dfb3d34bc3d78fac0fc7d88142c1b5031c52e13e533151ac83da27` |
| physical-gate oracle terminal audit | `a899c199cb03be09c795eac0203747e6e1e507cca6d2e4f5a5b9db3b41435dee` / `29f406b6bfa251819dff7e56c69adc8dbe2244037e24594d422156780f14d617` |

The executed oracle source hashes are `d34d9475eb79e228f3d7d3b1511e93c2c31c9900a16d2b792a910874766be773` for `lewm/benchmarks/go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1.py` and `1873df123b5d4b48fc5bdb0e24a05b596b96537c9b58f0b6e4fd5a1ae2ac0084` for its runner. Its immutable `result.json` file/content hashes are `ce23d00fab6b5be3b222b837cc70635ebdc5955ca82bf738ba7ef0d9731e24f8` / `86be0cbcede35ba3373d8261ac4cd18ead9598d8bcdf4e0bf2f8f435562db5fd`; it proves only that the unchanged physical gate is attainable, not that a learned checkpoint qualifies.

V3 immutable metric-sidecar baselines, relative to its audited output root, are:

| Update/path | `file_sha256 / content_sha256` |
|---|---|
| 100 `checkpoints/update_100.metrics.json` | `33104dcfa12bd90cc3db0366059a06b5adf84b6b440deb6181b0a618221d930d` / `8eeacf3d833fc2401b05dd8c8d5709acb11eec0ae9299f454bec5a4a8aa25b62` |
| 400 `checkpoints/update_400.metrics.json` | `c53711248f70482ed790484591503741255c7a5a9d2429d165d7c9c42f0be31a` / `2b3ca28c40f6ff67c975ff5cfe7f8c43a3030415407ca06d21b547add4842a30` |
| 1000 `checkpoints/update_1000.metrics.json` | `26f5e06d141b974b335d7f056b5392bd308342082bf832acfbd83f70b451e926` / `26e369149c30afdaf676a6ad111f0914bc410896ec6c0ac145db2a221a7e394a` |
| 2000 `checkpoints/update_2000.metrics.json` | `28fb55ed2c679d8af84ecdff4159e52832a2337c95fc8c10db60a683567f4b7a` / `93b493da6816e8d6365b6dbdb66966a0d0c5456c754006fbc4b3e0dcf3ae070f` |
| 4000 `checkpoints/update_4000.metrics.json` | `5b83a880d13983c398083525fb05d939673cad2a86ec38596a7f279670cf1a05` / `55dda1394ecb201c37ade773c76e1b30c3238e3c33d5d68fcc09d90266141f1a` |

## Sole scientific delta

Retain the exact V3 architecture and 92 trainable tensors: `encoder.` 78 and `evidence_head.` 14. Keep `bev_decoder.`, `predictor.`, `occupancy_head.`, `target_encoder.`, and `target_bev_decoder.` frozen. Keep exact V3 AdamW settings, independent head/encoder group clipping, learning rates, 4,000-update schedule, frozen train and checkpoint-selection data, presentation sampling, physical evaluator, 189 margins and nine-scope thresholds, cyclic-plus-one-within-family wrong-RGB mapping, and checkpoints 100/400/1000/2000/4000.

Replace only the `target_bin_offset_smooth_l1` slot. For each separately computed real B=4 current or next frame and each represented target-hit ray `r`, let `q[r,b]` be the unchanged 64-bin first-hit mass conditioned across all finite-hit bins, `d[r,b]` the unchanged bin centre plus that bin's predicted offset, and `y[r]` the metric target depth. Define `e[r] = sum_b q[r,b] * abs(d[r,b] - y[r]) / 0.25m`; with `N` represented target-hit rays, the frame slot is the mean of the largest `ceil(0.05*N)` values of `e`. Preserve the current/next reduction, use `frame_slot` as the replacement component at the unchanged objective coefficient `0.25`, and leave every other objective term and weight unchanged.

## Fixed-checkpoint controls and spot checks

Let `P=count(m>=0)`, `S=sum(max(0,-m))`, and `W=min(m)` over the exact ordered 189-margin vector. Integrity failure has first precedence; the earliest fixed checkpoint passing all nine physical scopes qualifies.

- At update 100, require finite state/metrics, all 92 trainable gradients present and finite through the unchanged clips, unchanged frozen-state hash, trainable-state movement from update zero, and exactly 189 finite margins; otherwise stop terminally.
- Update 400 has no new numeric cutoff: qualify on 9/9 or continue.
- At update 1000, absent 9/9, continue only if `(P,S,W)` Pareto-dominates V3 `(106,49.09939462151839,-7.944758415222166)`: `P>=106`, `S<=49.09939462151839`, `W>=-7.944758415222166`, with at least one strict inequality; otherwise stop terminally.
- At update 2000, absent 9/9, continue only if the same rule holds against V3 `(121,30.06221418748834,-5.833248805999755)`; otherwise stop terminally.
- At update 4000, qualify only on 9/9; otherwise stop unqualified.

The trainer pauses only after each fixed checkpoint is complete: snapshot, one inline nonmutating evaluation, immutable externally readable sidecar publication, then the declared control decision. An external observer may read that sidecar only; it never reads the checkpoint, invokes the evaluator, or touches the live process. No separate monitor or evaluator framework is introduced.

## Explicit denials

No retry, resume, warm start, extension, second attempt, soft/closest promotion, threshold relaxation, architecture change, data/refinement change, sampling change, calibration change, JEPA or predictor change/training, G2, navigation, runtime, heldout access, or heldout tuning is preregistered or authorized here.
