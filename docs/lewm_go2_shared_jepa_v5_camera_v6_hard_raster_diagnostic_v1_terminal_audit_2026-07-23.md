# Shared V5 Camera V6 hard-raster diagnostic V1 terminal audit

Date audited: 2026-07-23
Auditor: `/root/hard_raster_terminal_audit`
Execution source commit: `9cb39a72a35da2038521e55db29886f5a30bbf78`

## Verdict

**Integrity: PASS. Scientific result: `FAIL_HYPOTHESIS_REJECTED`.**

The single authorized forward-only attempt completed cleanly, reproduced the
immutable soft and direct physical metrics exactly at zero tolerance, and made
no training or model-state change. The fixed hard adapter did not meet the
preregistered materiality gate: it passed zero of the required six
balanced-accuracy gain scopes, missed both aggregate recall-gain requirements,
and passed only the wrong-RGB sensitivity guard.

This rejects the narrow hypothesis that the existing soft raster stage is
discarding a large amount of Camera V6 evidence in a way repaired by this
fixed hard decoder. Camera V6 remains rejected and unqualified; no checkpoint
is promoted. No retry, threshold tuning, alternate decoder, successor
implementation or training, G2, navigation, runtime, production, or held-out
action is authorized by this result.

## Sealed outputs

Only `result.json`, `access.json`, and `completed.json` were opened. Their
canonical content hashes were independently recomputed after removing
`content_sha256` and all matched.

| File | Bytes | File SHA-256 | Canonical content SHA-256 |
|---|---:|---|---|
| `result.json` | 21,599 | `a86a8f11c15c3d4b0dcafdf466847fdaeeddda48b20caa7bece77e11f802b49a` | `84f2088e49007a0a373765ec9622694dd5010074b8f0d801258d6d316f188c43` |
| `access.json` | 642,083 | `6326cc1b5e4f48224589fe7d48f93605e627d549af8e6e9963b4b63cea4cba52` | `1263dcffbb1b0580f5b1b80c6776aa491c60ff88400044ba9abba7779f44c7e1` |
| `completed.json` | 1,143 | `e5f499bb1e3434f4b50366272982c5d6e234e6efc918be7c50de5019f0347fc0` | `76b4f20b6b2ba2067376233ef73f4beda74356eb4ae993b2cc80d4f73a54d420` |

Filesystem metadata confirms the exact terminal inventory
`access.json`, `completed.json`, `reservation.json`, and `result.json`, with
all four regular files mode `0444`; the root is mode `0700`.
`reservation.json` was not opened. Its metadata is 5,579 bytes and mode
`0444`; `completed.json` binds its file SHA-256 as
`72945744d7326464108aaa0c14652f8eaaf88c6228595b8976e6250c4854cb29`
and content SHA-256 as
`6c536039a8d99cc1c4f18ba7b755428b63500b64fd271638e37fefc70e896f5e`.
The attempt is consumed, and root reuse, repair, resume, or retry is denied.

The final preregistration has 12,248 bytes and file SHA-256
`accb965e4134c2b91395679994a307ee5d19136b44542b2b6ffdbb3b9b3c4d14`.
The committed independent review and authorization retained valid canonical
hashes, and all 16 reviewed source files matched their bound byte counts and
file hashes.

## Integrity closeout

- `access.json` reports `PASS_exact_permitted_reads_only`: 1,003 completed,
  unique top-level reads covering 1,000 unique input files plus the bound V6
  terminal audit, metric sidecar, and checkpoint. The 1,000 delegated-read
  paths exactly equal the 1,000 access-record paths.
- All `train`, `probability_calibration`, G2, navigation,
  runtime/production, and held-out open counters are zero.
- The checkpoint had exactly one filesystem read and one deserialization.
  There was one model construction, 924 matched frame presentations, and 924
  cyclic wrong-RGB frame presentations over 495 selection pairs and 924
  unique endpoints.
- Optimizer construction/steps, backward, gradients, clipping, EMA, autocast,
  checkpoint writes, and parameter/buffer mutation are all zero. Model state
  SHA-256 is unchanged at
  `960854245db49a048e3a99e91b08d6746795f8c1abd52a267f592900259eee22`.
- The result and access operation ledgers agree exactly.
  `training_or_state_mutation_count` and `heldout_open_count` are zero.
  Hard-raster NLL is correctly excluded as non-comparable.

## Preregistered scientific decision

The confusion matrices independently reproduce every published class recall,
balanced accuracy, hard-minus-soft gain, and matched-minus-wrong difference.

| Non-rough scope | Hard minus soft BA | Hard matched minus wrong BA | Gain `>= 0.05` | Wrong guard `>= 0.12` |
|---|---:|---:|---|---|
| `aggregate` | -0.029343341917019927 | 0.25000366795037854 | no | yes |
| `large_enclosed_maze` | 0.006115111575428767 | 0.39333653117338613 | no | yes |
| `local_composite_motifs` | 0.012641122514260417 | 0.43200317952394557 | no | yes |
| `loop_alias_stress` | -0.00699866932509996 | 0.3844120111576673 | no | yes |
| `medium_enclosed_maze` | -0.00016977267314588484 | 0.4110672632523194 | no | yes |
| `open_obstacle_field` | -0.04841841705919969 | 0.25682403587896063 | no | yes |
| `small_enclosed_maze` | 0.028214097483173273 | 0.44758419678651223 | no | yes |
| `visual_sensor_stress` | -0.0013029399150203957 | 0.4574398336161651 | no | yes |

The balanced-accuracy gain count is `0/8`, versus at least `6/8` required.
Aggregate free recall is `0.951577156783369`, a gain of
`0.03520694815868897`, below the required `0.05`. Aggregate occupied recall
is `0.6784272740377685`, a gain of `-0.12754072365575897`, also below the
required `0.05`. The wrong-RGB guard passes `8/8`, showing that the output
remains image-dependent, but that guard cannot compensate for the three
failed materiality requirements. The exact preregistered conjunction therefore
evaluates to `FAIL_HYPOTHESIS_REJECTED`.

The bounded next action is closeout, not another diagnostic or training run.
Any different architecture direction requires a separate user decision and
new preregistration.
