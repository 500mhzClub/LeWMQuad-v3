# Go2 N32 dynamic cell-square geometry result

Date: 2026-07-11

Status: independently finalized pass. This licenses an attitude-aware
Cartesian projective traversability head; it does not itself pass learned G2.

## Frozen Identities

- execution binding SHA-256:
  `211043ee3c3200d1fc93febbae73059341aea19560c83f53f3b3bb231bf06e66`
- human implementation manifest SHA-256:
  `be6decf28456bd16e16c49963a104d334a0064778598998cd20b262c39939fe6`
- machine implementation manifest file/content SHA-256:
  `14cd38b5ec025b9fc41aefa1a2564901ba1d36f235d2a0a40abfe2455306f48f` /
  `eb7d5dfc30afb4cf7801e268374b0c652fa28eb7c706829d9628b409698363d8`
- runner candidate file/content SHA-256:
  `5a10effe26dec2e45d2d8f28270e5cf4c50f3badad45f2fb89cdaec788d920bc` /
  `fd3828937268e012a03eaf8a84cbcdf0737d616abce6ae9b00ab13e5ae546a77`
- finalized result file/content SHA-256:
  `ace9b39c4be31fad84eb7bc2aa65c584acec04febb638672fbcead0db4b6b4fe` /
  `923b401d062819578ee65130007daffeee658044b9ebceccab3f70c1df830567`
- independently equal scientific-core SHA-256:
  `9ec13c60b8e8d9596fe17062ba7e8dd2869cd133f8de250c61cc2c597be27482`

## Full-Label Result

| Quantity | Result |
|---|---:|
| registered frames | 320 |
| categorical cells scored | 1,310,720 |
| UNKNOWN | 1,181,699 |
| FREE | 118,793 |
| OCCUPIED | 10,228 |
| known label occurrences | 129,021 |
| level-center known misses | 373 |
| static cell-square known misses | 4 OCCUPIED cells in 4 frames |
| dynamic full-attitude cell-square known misses | **0** |
| dynamic supported known labels | **129,021/129,021** |

Every gate is true, including binding/source integrity, predecessor authority,
label byte/count reconciliation, exhaustive static and dynamic scoring, zero
dynamic unsupported known cells, access reconciliation, and independent
recomputation. Candidate and finalizer scientific cores are exactly equal.

All five registered families have complete dynamic FREE and OCCUPIED support.
The four static misses occur in `rough_local_dynamics` (one) and
`medium_enclosed_maze` (three); retaining measured base roll/pitch through the
full quaternion resolves them without changing labels, thresholds, roles, or
the 64 x 64 Cartesian grid.

## Access Evidence

Preparation performed 20 metadata-only shard stats and zero shard byte opens.
Runner and finalizer independently performed 20 pre-hash reads, 20 post-hash
reads, 20 NPZ parses, decompressed exactly 20 current and 20 next label arrays,
and selected exactly 320 registered rows. Both retained and scored zero
unselected rows. All three ledgers reconcile exactly with no denial or
forbidden-role count.

No RGB, source geometry, model output, selection, calibration, G2, runtime,
held-out, or sealed payload was opened. Learning was false. The audit ran on
CPU; neither GPU was used.

## Consequence

The failed N32 V4 center/static geometry was an observation-lift mismatch, not
an impossible target. The successor model must project each frame using the
deployment-equivalent base quaternion plus stored yaw, the fixed camera mount,
five cell-square horizontal samples, and five vertical anchors. A static
attitude prior is no longer eligible for promotion. Learned precision/recall,
calibration, scene-disjoint G2, memory fusion, and navigation gates remain to
be demonstrated.
