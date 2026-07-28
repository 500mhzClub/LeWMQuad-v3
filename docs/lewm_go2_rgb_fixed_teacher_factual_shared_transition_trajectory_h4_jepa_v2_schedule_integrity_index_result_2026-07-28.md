# Go2 RGB fixed-teacher factual shared-transition trajectory-H4 JEPA V2 schedule-integrity index result — 2026-07-28

## Outcome

- The one-time V2 schedule build completed successfully from frozen commit
  `50aa0cf10d6cdb3285f9ac8255319e01d54d6fa3` in `190.06` seconds.
- Decision: **PASS_INDEX_INTEGRITY**. The generated train and validation
  schedules implement the preregistered causal edge
  `F(i-1,5) --p_i--> F(i,5)` and are frozen for the sole V2 probe.
- This is an index-build and integrity result, not training authority or a
  scientific model result. No training, optimizer update, GPU probe,
  checkpoint output access, navigation, test, held-out, or sealed evaluation
  occurred.

## Frozen generated artifacts

| Artifact | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/manifest.json` | 1 | 26,926 | `d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e` |
| `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/train.jsonl` | 16,000 | 10,328,000 | `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77` |
| `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/val.jsonl` | 2,048 | 1,317,888 | `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6` |

- Row schema:
  `lewm_go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity`.
- The seed remains exactly
  `go2_recurrent_h4_rgb_sequence_index_v1_20260727`.
- The 16,000-row training schedule contains exactly 2,000 rows from each of
  the eight development maze families and 1,000 train scenes.
- The 2,048-row validation schedule contains exactly 256 rows from each family
  and 150 validation scenes.
- Train and validation have zero scene overlap and zero RGB-path overlap.

## Frozen implementation and corpus bindings

| Source | Bytes | SHA-256 |
|---|---:|---|
| `lewm/datasets/go2_recurrent_h4_rgb_sequences.py` | 27,386 | `3f8c2a89af2934e8225dd98447b952d9e5ce8bedac99a7f834118263957652e6` |
| `lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py` | 21,001 | `3d49e710304ad685f9d161a84586229a6036b652f84df877772afe5b827c51ea` |
| `scripts/build_go2_recurrent_h4_rgb_index_v2.py` | 7,995 | `6d4dc0ad8626e53ab36d170d8b5d5d33af0a0c30cf68ad11ed34e6eb23831ce4` |
| `lewm/tests/test_go2_recurrent_h4_rgb_sequences_v2.py` | 13,320 | `0c4eed119bd2398d4d3dff89f321d0f3f9a79a7ae60c0cacf19e16b31f9e6dec` |

- The manifest's ordered public-source binding is
  `10b0c26f6c33327800b3d72478074a489aed9b6a5ce8d11e4bba33f325a1aaac`.
- The frozen census receipt is 54,695 bytes with SHA-256
  `aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408`.
- Its ordered 1,150-source content binding and the build's independently
  recomputed live binding are both exactly
  `0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696`.
- The builder made 2,300 complete metadata passes over 52,259,089 actual
  metadata rows and validated 126,117 unique selected RGB leaves totalling
  4,783,410,691 bytes.

## Causal selection audit

- The V1 enumeration contained 1,807,552 logical six-block candidates. V2
  rejected exactly 67,821 episode-initial groups that have no real
  same-episode predecessor boundary, retaining 1,739,731 causal candidates
  before the inherited caps and quotas.
- Every one of the 108,288 serialized action transitions has the same scene
  and environment at both endpoints and an exact `+240` frame-index delta:
  five real `0.1`-second ticks with 48 interleaved environments.
- There are zero repeated action transitions. Adjacent packed H6 groups remain
  transition-disjoint but may share their one boundary RGB; the audit found
  219 such shared boundaries, each with maximum multiplicity two.
- All 288 family/action-position cells in each split are populated. There are
  no missing future-action cells, forbidden paths, malformed roles, wrong
  schemas, cross-environment edges, cross-scene edges, or protected-path
  references.
- Of the V2 rows, 15,380 train and 1,973 validation rows are exact causal
  endpoint shifts of rows in the selected V1 schedules. The remaining 620
  train and 75 validation rows are deterministic V1-rank backfills needed after
  causal filtering and quota restoration; they are not mined by outcome or RGB
  content.

## Frozen hold-control support

Future `p2:p5` all-hold rows, recorded before execution as required:

| Family | Train | Validation |
|---|---:|---:|
| `large_enclosed_maze` | 1 | 0 |
| `local_composite_motifs` | 129 | 16 |
| `loop_alias_stress` | 21 | 3 |
| `medium_enclosed_maze` | 0 | 0 |
| `open_obstacle_field` | 36 | 4 |
| `rough_local_dynamics` | 19 | 4 |
| `small_enclosed_maze` | 54 | 3 |
| `visual_sensor_stress` | 12 | 2 |
| **Total** | **272** | **32** |

- Complete six-action all-hold support is 263 train rows and 31 validation
  rows. The preregistered control, thresholds, breadth requirement, family
  floor, and checkpoint-selection rule remain unchanged despite sparse support
  in the large and medium maze families.

## Review and custody

- The focused V2 adapter suite passed: 9 tests. The combined V1/V2 source-only
  suite passed: 14 tests.
- A restricted independent audit recomputed all three artifact hashes, byte
  counts, row counts, schemas, canonical JSON, quotas, action coverage, causal
  filename deltas, source bindings, disjointness, duplicate-transition counts,
  RGB-path multiplicities, and protected-path checks and returned **CLEAR**.
- The independent audit opened only the exact manifest, train/validation
  schedules, and three bound public source files. It did not open corpus
  metadata, RGB bytes, any checkpoint, any runtime output, or any test,
  held-out, or sealed material.
- The manifest contains a descriptive `elapsed_seconds` field, so a hypothetical
  rebuild's manifest receipt would not be byte-identical even though the
  scientific JSONL schedules are deterministic. The schedules were generated
  once and are now hash-frozen; this non-scientific receipt field is not a
  reason to rebuild or revise them.

## Consequence

- The next permitted step is to freeze and independently semantic-diff the
  thin V2 runner against the V1 executable science, then run its bound
  preflight.
- Only after that source review and preflight clear may the sole fresh V2
  attempt reserve its absent output root and execute exactly 1,000 optimizer
  updates / 16,000 ordered training presentations. There is no retry or
  resume.
