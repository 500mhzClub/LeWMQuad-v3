# Go2 main-pool recurrent H4 training-exposure audit — 2026-07-28

## Question

- Did the recent 16,000-presentation JEPA probes meaningfully consume the
  approximately 3 TB main pool, and could their failures simply reflect bad or
  narrow data?
- This was a read-only audit of the committed main-pool metadata census and the
  exact authorized train/validation sequence indices. It did not open RGB
  pixels, checkpoint files, or any protected role.

## Authoritative evidence

- Main-pool census receipt:
  `.generated/go2_recurrent_jepa_main_pool_census_v2/receipt.json`, SHA-256
  `aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408`.
- Exact train index: 16,000 rows, SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`.
- Exact validation index: 2,048 rows, SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`.
- The census covers 1,150 scenes, 55.2 million frame-metadata rows,
  10,614,345 reset-safe sliding H6 windows, and 1,807,552 row-disjoint packed
  H6 windows. Its metadata payload alone is 138,549,246,020 bytes.

## What the probes actually consumed

- One probe trains on 16,000 sequences and validates on 2,048 sequences. That
  combined schedule is approximately `0.9985%` of packed H6 candidates and
  `0.1700%` of sliding H6 candidates.
- The two indices reference 111,814 unique train RGB paths and 14,302 unique
  validation RGB paths: 126,116 unique frame paths, approximately `0.2285%`
  of the 55.2 million source frame rows.
- A normal capped run reports 183,680 physical RGB opens because it reads
  112,000 training views and repeats the 14,336 validation views at five
  observations. That access count must not be mistaken for unique data.
- Therefore the recent probes did **not** train over approximately 3 TB. They
  deliberately used a small, broad falsification schedule. The metadata audit
  cannot convert the unique-path count into an exact physical-byte fraction or
  certify the stated total disk size because it did not open the full RGB
  corpus.

## Breadth and integrity of the sampled schedule

- All 1,000 train scenes and 150 validation scenes appear. Each of the eight
  families contributes exactly 2,000 train and 256 validation sequences.
- Every one of nine action primitives appears at all six sequence positions in
  every family. All 81 adjacent ordered action pairs appear.
- There are 3,144 distinct future four-action strings and 5,430 distinct full
  six-action strings in training.
- No malformed index row, duplicate complete sequence, duplicate seven-frame
  tuple, train/validation scene overlap, frame-path overlap, or full-identity
  overlap was found.
- Only 186 train and 34 validation frame paths repeat. Each is a legitimate
  shared boundary between adjacent packed sequences, occurs exactly twice,
  and appears once as position 6 and once as position 0.
- Every sequence contains seven same-environment frames at a constant
  frame-index stride of 240 and six in-vocabulary actions. Role, family, scene,
  and path identities are internally consistent.

## Caveats that matter for later scaling

- Family balancing deliberately oversamples smaller families by scene:
  train exposure ranges from eight rows per scene in the largest family to 40
  rows per scene in the smallest families.
- Actions are imbalanced. `forward_medium` is about 27% of action positions;
  `arc_left` and `yaw_left` are about 18% each; the rare forward speeds are
  about 3%. About 20.5% of train sequences repeat one primitive for all six
  transitions, including 252 all-hold sequences.
- The validation schedule has complete action-position coverage, but its
  rarest family/action cells contain one row from one scene. Aggregate and
  bootstrap results are more reliable than conclusions about those cells.
- The full census established metadata continuity, reset safety, scene and
  manifest separation, and action support. It did not inspect every PNG's
  decoded pixels, so it does not certify corpus-wide blur, corruption, texture
  diversity, or visual informativeness. The indexed subset has repeatedly
  decoded without runtime failures, and the index builder validated its PNG
  signatures and render summaries.

## Decision

- Main-pool breadth is not the primary explanation for the failed mechanisms.
  The current schedule already spans every train scene, family, and action,
  while the failures are structured: coordinate collapse in WDPS-D8 and
  marginally whitened but sample-misaligned states in full-whitened D8, with
  ordered-history evidence consistently negative.
- Small exposure can still limit final accuracy and rare-action precision. It
  becomes the correct scaling lever after a mechanism first demonstrates
  noncollapsed state, positive action and ordered-history dependence, and a
  persistence improvement on the capped schedule.
- The next probe should therefore remain capped and change the predictive
  target/learning signal. If it passes or shows coherent improvement across
  those task-relevant metrics, scale it with fresh packed windows from this
  existing pool. No new dataset collection or further data-format refinement
  is presently justified.
