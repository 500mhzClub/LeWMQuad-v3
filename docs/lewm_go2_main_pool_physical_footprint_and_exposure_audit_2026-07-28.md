# Go2 main-pool physical-footprint and exposure audit — 2026-07-28

## Scope and custody

- This was a read-only audit of the active non-held-out `datagen_full`
  train/validation pool used by the recurrent-H4 RGB probes.
- Raw-rollout sizing traversed only the literal roots
  `.generated/datagen_full/rollout/train` and
  `.generated/datagen_full/rollout/val`.
- Render sizing did **not** list or recursively traverse the mixed
  `render_textured_v03` root. The 1,150 scene names were taken only from the
  two frozen train/validation indices, checked against the eight-family plus
  12-hex scene-name allowlist, deduplicated, and passed individually to
  `du -s -B1` beneath
  `.generated/datagen_full/render_textured_v03/<allowed-scene>`.
- Selected-RGB sizing used the same method for the 126,116 unique, regex-bound
  RGB leaves named by the indices. `*.pt`, test, held-out, and sealed names
  were excluded. No checkpoint was listed, statted, hashed, or opened.
- The only file contents read were non-protected source/configuration, the
  frozen census/index JSON metadata, run JSON receipts, and one train-scene
  render summary. No protected role, RGB pixels, label payload, benchmark, or
  navigation data was opened.

## Physical footprint

The following are allocated filesystem bytes reported by `du -B1`, not
logical payload bytes:

| Role | Raw rollout | Textured RGB | Combined |
|---|---:|---:|---:|
| Train | 547,026,419,712 | 1,981,099,188,224 | 2,528,125,607,936 |
| Validation | 81,988,288,512 | 286,359,408,640 | 368,347,697,152 |
| **Total** | **629,014,708,224** | **2,267,458,596,864** | **2,896,473,305,088** |

- The active non-held-out pool is therefore **2.896 TB decimal / 2.634 TiB
  allocated**. The informal “approximately 3 TB” description is accurate for
  these two active roots.
- This total excludes every role outside the exact train/validation allowlist
  and excludes unrelated generated artifacts. It should not be interpreted as
  the size of the entire workspace or mixed render root.
- Allocated size includes filesystem block rounding. Summing per-scene values
  could double-count cross-scene hard links if any existed; the render
  pipeline creates independent scene outputs, but this audit did not perform a
  55.2-million-file inode-equivalence census.
- The frozen metadata census independently records 55,200,000 frame rows and
  138,549,246,020 logical bytes in the `frames.jsonl` leaves. That metadata is
  only one component of the 629 GB allocated raw-rollout footprint.

## What the capped schedule actually uses

- Frozen train index: 16,000 H6 sequences, 1,000 scenes, 111,814 unique RGB
  leaves, and 4,493,778,944 allocated RGB bytes.
- Frozen validation index: 2,048 H6 sequences, 150 disjoint scenes, 14,302
  unique RGB leaves, and 553,984,000 allocated RGB bytes.
- Combined: 18,048 sequences, 126,116 unique RGB leaves, and
  5,047,762,944 allocated RGB bytes. The index manifest reports
  4,788,786,226 logical file bytes for the same unique leaves.
- The combined index covers **0.998477%** of the 1,807,552 row-disjoint packed
  H6 candidates, **0.228471%** of the 55.2 million source-frame rows, and
  **0.222618%** of the train/validation textured-RGB allocation.
- Training is one deterministic pass: 1,000 updates times batch 16 equals
  16,000 sequence presentations and 112,000 RGB references. It does not
  randomly stream from, or make an epoch over, the approximately 3 TB pool.
- The latest dual-domain run performed 183,680 successful RGB opens and read
  6,900,398,764 physical bytes: 112,000 train views plus 14,336 validation
  views repeated at updates 0, 250, 500, 750, and 1,000. Those repeated opens
  are not additional unique training data.

## Scene, action, hold, and history coverage

- Every train scene and every validation scene appears. Each of the eight
  families contributes exactly 2,000 train and 256 validation sequences.
  Family balancing therefore produces unequal per-scene exposure: about 8 to
  40 train rows per scene depending on family size.
- Each row contains seven same-stream RGB endpoints and six reset-safe actions:
  three history observations joined by two past actions, followed by four
  factual future actions and endpoints. The full census contains 10,614,345
  sliding and 1,807,552 row-disjoint reset-safe H6 windows.
- All nine primitives occur at every one of the six positions in the selected
  train and validation schedules, and every family has every primitive at
  every position. Training contains 5,430 distinct six-action strings, 3,144
  distinct four-action future strings, and all 81 adjacent ordered action
  pairs.
- The train history pair at positions 0 and 1 covers all 81 ordered pairs
  globally and in seven families. `small_enclosed_maze` covers 80/81, missing
  only `backward -> forward_fast`. Validation covers all 81 globally, but only
  42–60 pairs per family because each family has 256 rows.
- Train action positions are imbalanced in a way close to the full-pool
  marginals: `forward_medium` is about 27%, `arc_left` and `yaw_left` about 18%
  each, and the rare forward speeds about 3% each. The selected future hold
  rate is 4.750%, versus 4.779% in the full train census.
- Training has 4,618 hold occurrences, 2,485 rows containing at least one
  hold, and 252 all-hold H6 rows. However, all-hold rows occur in only six of
  eight train families: none were selected for `large_enclosed_maze` or
  `medium_enclosed_maze`. Factual hold still occurs at every position in those
  families. This is a real composition-level caveat for interpreting an
  all-hold-per-family gate, not evidence that the pool lacks hold transitions.
- The rarest train family/action/position cell has 33 rows. The rarest
  validation family/action/position cell has one row, so family-level rare
  action estimates are substantially noisier than aggregate metrics.

## Integrity and remaining uncertainty

- The census covers 1,000 train and 150 validation identities with zero
  within-role duplicate identity, zero cross-role scene or manifest overlap,
  zero source-shape failures, and no failed feasibility predicate.
- Train and validation index byte counts and SHA-256 hashes match the frozen
  runner bindings. Their manifests report no missing action-position cell;
  the index builder checked the 126,116 selected leaves as regular files with
  PNG signatures. Recent runs decoded every scheduled access without an RGB
  open failure.
- The census establishes metadata continuity, reset safety, and action
  support. It does not establish corpus-wide pixel informativeness, blur,
  corruption, or visual diversity because all 55.2 million PNG payloads were
  not decoded. Likewise, it does not enumerate full six-action-string counts
  across all 1.8 million packed candidates.

## Science decision

- The repository really does have an approximately 3 TB active
  train/validation pool, but the recent probes deliberately use only about
  0.22% of its textured RGB allocation in a broad, one-pass falsification.
- Lack of scene, family, marginal action, hold-transition, or reset-safe
  history coverage is not a credible primary explanation for the repeated
  all-family ordered-history failure. The two-family all-hold composition gap
  is a narrower schedule confound and should be recorded when interpreting
  the all-hold breadth gate.
- Do not scale a failed score formulation merely because most bytes remain
  unseen. First require a materially different shared, action-conditioned
  temporal mechanism to improve prediction, action dependence, and ordered
  history on the cap. If that mechanism passes, the existing pool has ample
  unused reset-safe windows for scale-up; no new data collection is currently
  justified.
