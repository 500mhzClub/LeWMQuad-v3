# Protected Camera Adaptation V5 native-schedule completion preregistration — 2026-07-16

## Boundary

This source-free document preregisters exactly one possible fresh Camera-only attempt rooted at `.generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v5_native_schedule_completion`. It is not implementation, review, execution, training, selection, promotion, or downstream authority. Exact source closure, an independent source review, and a separate execution authorization are required before reserving a previously absent output root.

V5 is not a V3/V4 retry, resume, warm start, or checkpoint continuation. It must reconstruct the exact Shared-V5 update-zero state SHA-256 `e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87` from the already-qualified N320 checkpoint file/content SHA-256 `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0` / `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.

## Committed evidence

The terminal Camera V4 audit is `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_terminal_audit_2026-07-16.json`, file/content SHA-256 `5d0d4a1cf966e5f612e15da9cacbc705ace4f629183038c6743f0e2fac1b355f` / `246e50b986316f7dc8c806960e8661cf83417fd34c0baa269d83b221cf98d5e2`. V4 stopped correctly at update 1000: `(P,S,W)=(97,41.00174362036205,-5.476026201248172)` improved severity but missed the frozen V3 pass-count floor `P=106`. No V4 checkpoint qualified.

The terminal Camera V3 audit is `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_terminal_audit_2026-07-15.json`, file/content SHA-256 `3eb77a83ede536680e03363521f73f41205ac17d845a0e28251a40dcf82f77ab` / `a5a86d5260c519003f7a5efeb1d21c535afeb65ef7596a627174a41c633be2ac`. Its immutable progress was monotonic through the native schedule's first half:

| update | P | S | W |
|---:|---:|---:|---:|
| 1000 | 106 | 49.09939462151839 | -7.944758415222166 |
| 4000 | 134 | 19.869159033399846 | -4.920835733413693 |

The V3 warm-start review is `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_warmstart_science_review_2026-07-15.json`, file/content SHA-256 `b37829a2c311533240f6191c099d79411d453adbde43cd0304f1e5c74bd676d7` / `f317c80e527706faf267ba0be3ab8a19187aeeabba49896f4f7d0722aac98168`. It remains binding: optimizer state was not published, so resetting AdamW from a V3 weight snapshot would be a confounded new attempt. V5 therefore starts only from update zero.

The successful standalone N320 run used 200,000 frame exposures and passed its 26/26 gate. Its result file/content SHA-256 is `9fb603566002cd57797895fe27cb2ccabf0e39484c2a8e705c99982933aa3a44` / `8be838e6b558b396d926f24432d95e1ba9f691d12752cde088e061f13d97d768`; its gate file/content SHA-256 is `4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6` / `76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b`. This is compute-scale context only, not evidence that V5 will pass.

## Training-science delta and control-policy delta

Retain the exact V3 architecture, five-term Camera loss, 92 trainable tensors, frozen tensor partition, AdamW parameters, independent head/encoder clipping, learning-rate function, current/next reduction, four real B=4 microbatches per update, train and checkpoint-selection inputs, physical evaluator, nine scopes, ordered 189 margins, physical-gate thresholds, and cyclic-plus-one-within-family wrong-RGB mapping.

The sole training-science delta is to change the terminal schedule boundary from update 4000 to update 8000. Consume all 128,000 pair presentations from the already-published native schedule `.generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v4/schedule.json`, file/content SHA-256 `08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270` / `274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15`, with seed `20260713`, 4,262 train pairs, effective pair batch 16, and exactly 8,000 updates. The terminal matched-training V4 audit that binds this schedule is `docs/lewm_go2_shared_jepa_v5_matched_training_v4_terminal_numeric_failure_audit_2026-07-15.json`, file/content SHA-256 `70371a2cd09e912e05ba0b5efdf75ee2de38cc89347e8111fff303e2a55c485b` / `ae86d1479fc3016eb96302304e079b7bf9647e26b24b3d860e7d32013bf9c6f4`.

The nonmutating checkpoint/control policy also changes explicitly from V3: update 2000 is omitted, update 1000 uses the frozen V3 reproduction floor without loss, update 4000 becomes a reproduction continuation gate, update 6000 adds a within-run Pareto gate, and update 8000 becomes the exact terminal physical gate. These controls do not change any optimizer update that occurs before their declared branch; they are governance and early-stop deltas, not additional training-objective, data, sampling, or architecture deltas.

V5 remains Camera-only: JEPA objective, JEPA backward, EMA update, calibration, G2, navigation, and held-out counts are all zero. The failed matched-training V4 numeric outcome is not reused; only its independently bound immutable schedule is consumed.

The exact schedule-prefix SHA-256 values are:

| update | canonical presentation-index prefix SHA-256 |
|---:|---|
| 100 | `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51` |
| 400 | `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92` |
| 1000 | `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528` |
| 4000 | `14e83952c758c2ee4118d38c116625feb351813bc24b017d7b47f53426df47ab` |
| 6000 | `5ba218ed5335c357b60d5f8c2f2d0a3f9e1171631cc299e5d0747ae858e92c50` |
| 8000 | `a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663` |

## Fixed spotchecks and controls

For the exact ordered 189-margin vector, define `P=count(m>=0)`, `S=sum(max(0,-m))`, and `W=min(m)`. Loss is never used for continuation. Integrity failure has first precedence, and the earliest fixed checkpoint passing all nine physical scopes qualifies.

- Update 100: require finite state and metrics, all 92 trainable gradients present and finite through the unchanged clips, unchanged frozen-state hash, trainable-state movement from update zero, one inline nonmutating physical evaluation, and exactly 189 finite margins. Otherwise stop terminally.
- Update 400: informational spotcheck; absent 9/9, continue if integrity remains valid.
- Update 1000: absent 9/9, continue only if `P>=106`, `S<=49.09939462151839`, and `W>=-7.944758415222166`. Strict improvement is not required because this is a fresh reproduction of the frozen V3 prefix.
- Update 4000: absent 9/9, continue only if `P>=134`, `S<=19.869159033399846`, and `W>=-4.920835733413693`. Strict improvement is not required.
- Update 6000: absent 9/9, continue only if its `(P,S,W)` weakly Pareto-dominates this same V5 run's immutable update-4000 sidecar, with at least one strict improvement.
- Update 8000: qualify only on exact 9/9 physical scopes and 189/189 nonnegative margins; otherwise stop unqualified.

At each fixed checkpoint the one trainer process performs: complete update, CPU-weight snapshot, one inline nonmutating evaluation, immutable mode-0444 metric-sidecar publication, then the declared control branch. An external observer reads only the completed sidecar and never loads a checkpoint or reruns the evaluator.

## Explicit denials

No retry, resume, warm start, optimizer reconstruction, schedule extension beyond update 8000, second seed, loss blend, loss coefficient, architecture change, data/refinement change, sampling change, threshold relaxation, soft/closest promotion, calibration, JEPA/predictor training, G2, navigation, runtime, held-out access, held-out tuning, or automatic successor is preregistered or authorized here. If update 8000 does not qualify, this attempt stops before JEPA and requires a genuinely new user-directed scientific decision.
