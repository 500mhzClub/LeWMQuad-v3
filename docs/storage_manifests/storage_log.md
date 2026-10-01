# RecoveryStorage deletion log

Each entry records a deletion approved by Andrew, with its manifest and the free space measured before and after. Manifests stay in this directory after the data is gone.

## 2026-09-30 08:51 BST: `go2_supervised_rollout_mazes_v1_attempt_001`

- **Approval.** Andrew, 30 September 2026, for E1 storage headroom: "Storage: approved. Delete go2_supervised_rollout_mazes_v1_attempt_001, keep the committed manifest, confirm the free space afterwards, and record the deletion in the storage log."
- **Path.** `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_supervised_rollout_mazes_v1_attempt_001`
- **Contents.** The closed 9 September supervised-rollout evaluation cohort: 3 reused development layouts, 0/3 round trips, no model training. See the [summary](go2_supervised_rollout_mazes_v1_attempt_001.md).
- **Provenance.** Not derived into any current training set (C3/C4 v1, v2, the C3-v3 round) or the transfer set. The trace is `e1_storage/provenance_go2_supervised_rollout_mazes_v1_attempt_001.json`, from commit 5ce9a693.
- **Manifest.** [`go2_supervised_rollout_mazes_v1_attempt_001.manifest.tsv.gz`](go2_supervised_rollout_mazes_v1_attempt_001.manifest.tsv.gz), sha256 `d167384f928388ec0874a077b66f882d10254d2459fbc0d79e7b2af5cc5033ee`.
  - Before deletion, the directory matched the manifest exactly: 54,372 files and 28,699,375,622 bytes, with an identical file list and sizes.
  - File contents were hashed when the manifest was written on 30 September, about 01:26 BST.
- **Deleted** with `rm -rf` at 2026-09-30T08:51:23+01:00. Afterwards the path does not exist.

**Free space on RecoveryStorage (`/`):**

| | Bytes | GiB |
|---|---:|---:|
| Before | 102,909,198,336 | 95.84 |
| After | 131,731,193,856 | 122.68 |
| Freed | 28,821,995,520 | 26.84 |

- **Free space had already risen before the deletion.** The storage survey (29 September, about 21:10) measured 81.1 GiB free. By 08:51 on 30 September it was 95.8 GiB, about 14.7 GiB more, from activity outside this session's jobs.
- **E1 headroom now.** After E1's projected 64.9 GiB, about 57.8 GiB would remain free, which is 45.8 GiB above the 12-GiB reserve. The requirement is at least 15 GiB.

## 1 October 2026: development feature cache shrunk (feature arrays deleted)

- **Approval.** Andrew, 30 September ("Delete or shrink the cache before that run") and 1 October ("Delete the feature cache once the decoder is chosen, as authorised"). The decoder was chosen at 02:56 on 1 October: the large past-frames decoder `dev_decoder_fits/p3_large_past_frames_s2026093011.pt`, with its matched C4 in the same file.
- **Path.** `<capability root>/dev_c3_cache_v1`, built by `scripts/build_go2_dev_c3_feature_cache_development.py` (commit df8e7a18) and rebuildable with it.
- **Deleted** (`rm`, 2026-10-01T02:57:13+01:00): the two derived feature arrays.
  - `frames.f16.npy`, 25,945,178,240 bytes: pooled V-JEPA features of 65,982 frames.
  - `pred.f16.npy`, 24,214,634,624 bytes: predicted-future features for 61,581 context and horizon pairs.
- **Kept** (metadata that documents the cache), with sha256:
  - `result.json`: `c4b542e86016d32e6100226753b35d50d605cc06a00d501de45ef57499b7d653`
  - `items.json`: `f83dae1ae6bd41a7379cb2b7ddce7c38438d33747448d7f3e133c9d051a62032`
  - `pairs.json`: `263e3281c76edd93d43f1b3910a60a6b5a9d15c740dc0f18889edf0c5d52b61f`
  - `frame_rows.json`: `af14d0701c8d38aa32c2b26357e3cfbff983f015a0edb0313623c088f9b85309`
  - `progress.jsonl`

**Free space on RecoveryStorage (`/`):**

| | Bytes | GiB |
|---|---:|---:|
| Before | 63,133,396,992 | 58.80 |
| After | 113,293,266,944 | 105.51 |
| Freed | 50,159,869,952 | 46.72 |

The preliminary run's estimated output, about 330 missions at roughly 140 MB each (about 46 GB), leaves about 59 GB free, well above the 12-GiB reserve.
