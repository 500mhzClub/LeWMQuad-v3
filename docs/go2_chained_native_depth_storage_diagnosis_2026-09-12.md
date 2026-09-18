# Depth storage diagnosis for future native collection

The completed chained native run's 24,124 bound artifacts occupy
12,402,149,312 bytes (11.5504 GiB). A new metadata breakdown identifies the
main storage cost precisely: primary/auxiliary depth archives account for
91.4079% of those bytes, while both RGB image streams together account for
1.84135%. The earlier total is unchanged.

| Bound file group | Files | Bytes |
| --- | ---: | ---: |
| Auxiliary raw and derived depth, masks and segmentation | 4,014 | 5,563,227,000 |
| Primary derived depth and masks | 4,014 | 2,888,138,169 |
| Primary raw depth | 4,014 | 2,885,178,815 |
| Complete compressed decision stream | 1 | 609,591,253 |
| Primary RGB | 4,014 | 125,341,555 |
| Auxiliary RGB | 4,014 | 103,025,172 |

The original root result was authenticated before and after inspection at
SHA-256 `163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849`.
Only paths explicitly bound by that result were visited. Relative and resolved
paths were checked against custody restrictions and the exact artifact root.
No recursive discovery, source export, artifact mutation or deletion occurred.
The overall file-size breakdown is metadata-only; it does not newly rehash
non-depth artifact contents.

## Complete depth-archive inspection

The subsequent read-only command, session 26837, exited 0 after authenticating
all 12,042 depth archives against their bound file hashes before and after
reading their ZIP directory metadata. It required the exact expected NumPy
member set and DEFLATE compression in every archive. Array payloads were not
decoded, and no extraction or new archive was performed.

The compressed `depth_m.npy` and `valid.npy` members occupy:

- Primary derived archives: 2,887,118,613 bytes.
- Auxiliary archives: 2,772,333,549 bytes.
- Combined: **5,659,452,162 bytes, or 45.6328% of total recorded storage**.

The auxiliary diagnostic segmentation payload is only 17,949,774 compressed
bytes across the full episode. Removing diagnostic segmentation would therefore
miss the dominant storage cost and would discard evidence needed by the audit.
Equal compressed member sizes are not treated as proof of equal array values.

The original bound source explains the derivation: `RGBDSession.capture_observation`
constructs public depth and validity from the retained raw primary depth;
auxiliary capture stores raw depth together with its range-masked depth and
validity arrays. The existing raw sensor audit checks reconstruction. The five
inspected capture/conversion source files were rehashed against the original
native result and remained unchanged. This establishes a reason to investigate
a future raw-depth-only storage representation; it does not prove byte-identical
regeneration of every current NPZ archive or permit removing bound originals.

Full group/member byte counts and the inspection scope are retained in
`go2_chained_native_depth_storage_diagnosis_2026-09-12.json`.

## Consequences for the next population

Current storage remains sufficient for the prepared single longer native
attempt. Its collection and whole-worker allowances are unchanged. However,
the older 32-case study's 392 GiB admission must not be reused blindly for
longer trials. Applying the currently prepared 44 GiB whole-worker growth
allowance to 32 cases plus a 40 GiB reserve requires **1,448 GiB**. Even simple
doubling of this episode across 32 cases gives about **739.2 GiB before reserve**,
above the approximately 529 GiB available during inspection. Neither scaling
calculation is a measured independent-population requirement.

As an arithmetic illustration only, subtracting the identified derived payloads
while holding every other byte fixed would leave about 6.2796 GiB per current
episode. Doubling that amount over 32 cases plus a 40 GiB reserve gives about
441.9 GiB. This is not a tested format, a bound, a reservation or admission for
any population: image content, duration, decision history, compression and
metadata differ between cases.

The concrete future optimization is to preserve raw primary/auxiliary depth,
RGB, segmentation, capture witnesses and complete physical/command records,
while reconstructing derived depth/masks under a separately reviewed reader
and format. Before adoption it needs complete actual packet equality, retained
raw-audit compatibility, measured capture/reconstruction cost and storage, and
a prospective resource definition. The current prefix and prepared 8,000-step
native experiment retain their existing formats and identities. No storage
savings or sensor-latency improvement is claimed as achieved by this diagnosis.
