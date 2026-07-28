# Go2 RGB fixed-teacher factual shared-transition trajectory-H4 JEPA V2 schedule-integrity source review — 2026-07-28

## Status

- Independent review decision: **CLEAR FOR THE ONE EXACT V2 PREFLIGHT AND,
  ONLY IF IT PASSES, THE ONE EXACT CAPPED PROBE**.
- Corrected preregistration commit:
  `8ae2c14255cc5a7e3bb9e83398e784c622bd8761`.
- Frozen adapter implementation commit:
  `50aa0cf10d6cdb3285f9ac8255319e01d54d6fa3`.
- Frozen index-result commit:
  `ba19e0803326d9418961714c07d3e6d6ae75ef09`.
- Frozen runner implementation commit:
  `4d870486c40880f788b8c82d21169a4f98fec1ed`.
- This review preserves the user's authority for exactly one science-identical
  schedule-integrity replacement. It grants no retry, resume, alternate seed,
  threshold change, data scale-up, checkpoint inspection, navigation,
  held-out, sealed, promotion, production, or deployment authority.

## New V2 source bindings

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Preregistration | `docs/lewm_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity_preregistration_2026-07-28.md` | 16,355 | `0e586cb59fade4eab69a9fd12b7949f267ed4350e0f9209cbe0a59f97c8f3e3e` |
| Index result | `docs/lewm_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity_index_result_2026-07-28.md` | 6,482 | `62f5570ba578a0b5f0c43283a3109d297fd4ba48d0f6f7df02abf06f94587a16` |
| Endpoint adapter | `lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py` | 21,001 | `3d49e710304ad685f9d161a84586229a6036b652f84df877772afe5b827c51ea` |
| Index builder | `scripts/build_go2_recurrent_h4_rgb_index_v2.py` | 7,995 | `6d4dc0ad8626e53ab36d170d8b5d5d33af0a0c30cf68ad11ed34e6eb23831ce4` |
| Adapter tests | `lewm/tests/test_go2_recurrent_h4_rgb_sequences_v2.py` | 13,320 | `0c4eed119bd2398d4d3dff89f321d0f3f9a79a7ae60c0cacf19e16b31f9e6dec` |
| Thin runner | `scripts/run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity.py` | 10,129 | `b8d4f861b8a465da6530dd7997a27875dbc431a875bb214badc87c8bb798b14e` |
| Runner tests | `lewm/tests/test_run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity.py` | 13,778 | `f9140ce2c477aec271785bc585325a0952823b9ca186d5ddbda092b572b36885` |

## Frozen scientific witnesses

- The runner source is still exactly V1's executable science:
  `scripts/run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py`,
  34,730 bytes, SHA-256
  `693cbea45b2a49f0f3edfb7cabce347b852a67af78df1ecf5462c65be48cd977`.
- The model source remains byte-identical:
  `lewm/models/go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py`,
  21,734 bytes, SHA-256
  `38e264f8e18ffa3c3da4775fdd7d4a38549e8544f99cd863bfd2534999cd5b36`.
- The thin wrapper's 12-entry live source closure passed exact SHA-256 and byte
  validation. It includes the frozen V1 runner/model, trajectory, local,
  dense, base recurrent, encoder, shared-runner dependencies, and the V2
  adapter/builder.
- The accepted N320 initialization remains exactly 13,777,100 bytes, file
  SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
  content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.
  Preflight and execution may copy only its 78 reviewed `encoder.*` tensors.

## Frozen schedule bindings

| Role | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| Manifest | 1 | 26,926 | `d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e` |
| Train | 16,000 | 10,328,000 | `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77` |
| Validation | 2,048 | 1,317,888 | `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6` |

- Two restricted independent audits returned **CLEAR**. They verified all
  18,048 rows, all 108,288 unique causal transitions, exact `+240` endpoint
  deltas, all action-position cells, quotas, source/census bindings,
  train/validation disjointness, and zero protected paths.

## Semantic-diff result

- The V2 wrapper first installs V1 unchanged, then assigns only train/val
  paths, hashes and byte counts; index row schema; output/schema/PASS/STOP
  identities; source bindings; and an additive schedule-integrity receipt.
- The evaluator and run handler are the exact V1 function objects. The
  decision wrapper calls the exact V1 decision and preserves its 32 gates,
  failed-gate list, diagnostics, and selection result, replacing only the
  terminal decision and authority text.
- Model/module/source, parameter construction and RNG order, initialization,
  forward path, fixed teacher, K=4 particles, losses and weights, optimizer,
  learning rates, betas, epsilon, weight decay, gradient clipping, seed,
  batch size, 1,000-update / 16,000-presentation cap, validation cadence,
  5,400-second active-GPU cap, checkpoint selection, thresholds, and every
  gate are unchanged.
- CLI binding checks reject V1 or arbitrary schedule substitutions. There is
  no seed, update-count, resume, or checkpoint-input override surface. Fresh
  exclusive output reservation provides the one-shot boundary.
- Synthetic terminal-failure review produced exactly `failure.json`,
  `failure_access.json`, and `completed.json`, with V2 schema identities,
  cross-bindings, completed counters, truthful incomplete-access status, and
  zero retry/resume authority.

## Verification

- Focused V2 source-only suite: 19 passed in 0.08 seconds.
- V2 adapter/runner plus frozen V1 model/runner suite under the reviewed ROCm
  runtime: 36 passed in 5.04 seconds.
- Independent semantic review: **CLEAR**, with no edits.
- The exact output root was absent after review:
  `.generated/go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity/probe_v1`.

## Exact preflight and conditional execution

- Interpreter:
  `/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python`.
- Working directory:
  `/home/andrewknowles/Workspace/LeWMQuad-v3`.
- Required environment:
  `LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_V2_SCHEDULE_INTEGRITY_WRAPPER_SHA256=b8d4f861b8a465da6530dd7997a27875dbc431a875bb214badc87c8bb798b14e`
  and
  `LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_V2_SCHEDULE_INTEGRITY_WRAPPER_BYTES=10129`.
- Preflight argument: `--preflight-only`. It must return
  `PREFLIGHT_PASS_NO_OUTPUT_RESERVED_NO_RGB_OPENED`, exact 16,000/2,048 row
  counts, the unchanged parameter inventory, zero RGB opens, zero output
  reservations, and zero training updates.
- Only if preflight passes and the output root is still absent may the same
  interpreter, environment, wrapper, and working directory run `--execute`.
  Reservation, scientific PASS/STOP, or complete operational-failure receipts
  consume the sole attempt. There is no retry or resume.

## Custody

- Runtime checkpoints and traces written by this attempt remain inaccessible.
  Terminal audit may open only the JSON receipts named by the runner; it may
  not list, stat, hash, or open any generated `.pt`.
- Stopped/rejected checkpoints, prior predictors, test role, held-out, sealed,
  legacy V4 sealed material, labels, raw messages, navigation, G2--G8,
  benchmarks, promotion, and deployment remain outside this authority.
